#!/usr/bin/env python3
"""Local model benchmark for the outline rectifier and validator slots (issue #14).

Plan: docs/benchmarks/model_benchmark_plan.md. Config: configs/benchmark.yaml.

    python scripts/benchmark_models.py build-evalset   # fixed subjects + label set
    python scripts/benchmark_models.py smoke           # 1 sample per model: loads? latency? memory?
    python scripts/benchmark_models.py rectifiers      # every rectifier x every eval subject
    python scripts/benchmark_models.py validators      # every validator x label set (+ labels if present)
    python scripts/benchmark_models.py referee --model qwen3-vl-8b   # judge rectifier outputs
    python scripts/benchmark_models.py report

Full generator + judge setups through the production retry loop:
scripts/benchmark_setups.py.

Models run strictly one at a time: each rectifier in its own subprocess (so
its memory is fully returned and peak RSS is attributable), each Ollama
validator unloaded before the next. FLUX + Ollama together has OOM'd this box.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import resource
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib import request

import yaml
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from wildtrace.config import load_runtime_config  # noqa: E402
from wildtrace.diagram import (  # noqa: E402
    build_outline_rectifier,
    build_outline_validator,
    validate_with_opencv,
)
from wildtrace.gold_stage import _build_trajectory_payload  # noqa: E402

BENCH_DIR = REPO_ROOT / "benchmarks"
EVALSET = BENCH_DIR / "evalset_v1.json"
LABELSET = BENCH_DIR / "labelset_v1.json"
LABELS = BENCH_DIR / "labels_v1.ndjson"
SHARED_SILHOUETTE_KEYS = ("min_component_area", "subject_threshold", "silhouette_close_kernel")


# ── config / provenance ────────────────────────────────────────────────────

def load_bench() -> dict[str, Any]:
    return yaml.safe_load((REPO_ROOT / "configs" / "benchmark.yaml").read_text())


def run_dir(bench: dict[str, Any]) -> Path:
    path = REPO_ROOT / "outputs" / "benchmarks" / str(bench["run_id"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def provenance() -> dict[str, str]:
    digest = hashlib.sha256()
    for name in ("benchmark.yaml", "models.yaml", "export.yaml"):
        digest.update((REPO_ROOT / "configs" / name).read_bytes())
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()
    return {"git_sha": f"{sha}-dirty" if dirty else sha, "config_hash": digest.hexdigest()[:12]}


def read_ndjson(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_ndjson(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def rectifier_settings(entry: dict[str, Any], models: dict[str, Any]) -> dict[str, Any]:
    production = models["outline_rectifier"]
    if entry["backend"] == production.get("backend"):
        base = dict(production)
    else:
        # Don't leak FLUX-specific keys (gguf_repo, ...) into other backends.
        base = {key: production[key] for key in SHARED_SILHOUETTE_KEYS if key in production}
        base.update({key: value for key, value in production.items() if key.startswith("omnigen2_")})
    return {**base, **{k: v for k, v in entry.items() if k not in ("name", "reference")}}


def validator_settings(entry: dict[str, Any], bench: dict[str, Any], models: dict[str, Any]) -> dict[str, Any]:
    return {**models["outline_validator"], **bench["validator_defaults"], **{k: v for k, v in entry.items() if k != "name"}}


def entry_by_name(entries: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for entry in entries:
        if entry["name"] == name:
            return entry
    raise SystemExit(f"No model named {name!r} in configs/benchmark.yaml")


# ── eval set ───────────────────────────────────────────────────────────────

def cmd_build_evalset(args: argparse.Namespace) -> None:
    bench = load_bench()
    rng = random.Random(bench["seed"])
    checkpoints = REPO_ROOT / "outputs" / "silver" / "checkpoints"

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for subject in read_ndjson(checkpoints / "silver_subjects.ndjson"):
        if all((REPO_ROOT / subject[key]).exists() for key in ("crop_path", "mask_path")):
            by_category[subject["category"]].append(subject)
    subjects = []
    for category in sorted(by_category):
        pool = sorted(by_category[category], key=lambda s: s["sample_id"])
        subjects += rng.sample(pool, min(bench["subjects_per_category"], len(pool)))

    diagrams = [d for d in read_ndjson(checkpoints / "validated_diagrams.ndjson") if (REPO_ROOT / d["diagram_path"]).exists()]
    label_items = []
    for status, quota in (("accepted", bench["label_set"]["accepted"]), ("rejected", bench["label_set"]["rejected"])):
        pool = defaultdict(list)
        for diagram in sorted(diagrams, key=lambda d: d["sample_id"]):
            if diagram["diagram_validation_status"] == status:
                pool[diagram["category"]].append(diagram)
        for pool_list in pool.values():
            rng.shuffle(pool_list)
        # Round-robin across categories so every category is represented.
        picked: list[dict[str, Any]] = []
        while len(picked) < quota and any(pool.values()):
            for category in sorted(pool):
                if pool[category] and len(picked) < quota:
                    picked.append(pool[category].pop())
        label_items += [
            {
                "item_id": f"{d['category'].lower()}_{d['sample_id'][:8]}",
                "category": d["category"],
                "sample_id": d["sample_id"],
                "diagram_path": d["diagram_path"],
                "pipeline_status": d["diagram_validation_status"],
            }
            for d in picked
        ]

    BENCH_DIR.mkdir(exist_ok=True)
    EVALSET.write_text(json.dumps({"seed": bench["seed"], "subjects": subjects}, indent=2))
    LABELSET.write_text(json.dumps({"seed": bench["seed"], "items": label_items}, indent=2))
    counts = defaultdict(int)
    for subject in subjects:
        counts[subject["category"]] += 1
    print(f"eval subjects: {len(subjects)} {dict(counts)} -> {EVALSET.relative_to(REPO_ROOT)}")
    print(f"label set: {len(label_items)} ({bench['label_set']}) -> {LABELSET.relative_to(REPO_ROOT)}")


# ── rectifiers ─────────────────────────────────────────────────────────────

def trajectory_stats(image_path: Path, export: dict[str, Any]) -> dict[str, Any]:
    payload = _build_trajectory_payload(Image.open(image_path), export)["payload"]
    return {
        "stroke_count": payload["stroke_count"],
        "point_count": payload["point_count"],
        "trajectory_ok": 1 <= payload["stroke_count"] <= int(export["max_strokes"]),
    }


def cmd_rectify_one(args: argparse.Namespace) -> None:
    """Child process: run one rectifier over the eval subjects (or --limit)."""
    bench = load_bench()
    runtime = load_runtime_config(REPO_ROOT)
    entry = entry_by_name(bench["rectifiers"], args.model)
    out_dir = run_dir(bench) / ("smoke" if args.smoke else "rectifiers") / entry["name"]
    results_path = run_dir(bench) / ("smoke_rectifiers.ndjson" if args.smoke else "rectifiers.ndjson")
    subjects = json.loads(EVALSET.read_text())["subjects"][: args.limit or None]
    prov = provenance()

    import torch

    rectifier = build_outline_rectifier(rectifier_settings(entry, runtime["models"]))
    for index, subject in enumerate(subjects):
        crop = Image.open(REPO_ROOT / subject["crop_path"])
        isolated_path = REPO_ROOT / subject["isolated_path"] if subject.get("isolated_path") else None
        isolated = Image.open(isolated_path) if isolated_path and isolated_path.exists() else None
        mask = Image.open(REPO_ROOT / subject["mask_path"])
        destination = out_dir / f"{subject['category']}_{subject['sample_id']}.png"
        row: dict[str, Any] = {
            "model": entry["name"], "backend": entry["backend"], "reference": bool(entry.get("reference")),
            "sample_id": subject["sample_id"], "category": subject["category"], **prov,
        }
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            conditioning = rectifier.prepare_conditioning_image(crop, isolated)
            rectifier.run(sample=subject, subject_image=conditioning, subject_mask=mask,
                          generated_path=None, destination=destination, params={})
            row["latency_s"] = round(time.perf_counter() - started, 3)
            # The first call includes model load; report it separately.
            row["includes_model_load"] = index == 0
            opencv = validate_with_opencv(destination, runtime["models"]["opencv_prescreen"])
            row.update({
                "output_path": str(destination.relative_to(REPO_ROOT)),
                "opencv_passed": opencv.passed, "opencv_score": opencv.score, "opencv_flags": opencv.flags,
                **trajectory_stats(destination, runtime["export"]),
            })
        except Exception as exc:  # record and continue: one bad sample must not sink the sweep
            row.update({"error": f"{type(exc).__name__}: {exc}"[:500], "latency_s": round(time.perf_counter() - started, 3)})
        if torch.cuda.is_available():
            row["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        row["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
        append_ndjson(results_path, row)
        print(json.dumps({k: row.get(k) for k in ("model", "sample_id", "latency_s", "opencv_passed", "stroke_count", "peak_vram_gb", "error")}), flush=True)
    rectifier.unload()


def cmd_rectifiers(args: argparse.Namespace, smoke: bool = False) -> None:
    bench = load_bench()
    names = args.models or [entry["name"] for entry in bench["rectifiers"]]
    for name in names:
        print(f"== rectifier {name}", flush=True)
        command = [sys.executable, __file__, "_rectify-one", "--model", name]
        if smoke:
            command += ["--smoke", "--limit", "1"]
        elif args.limit:
            command += ["--limit", str(args.limit)]
        code = subprocess.run(command, cwd=REPO_ROOT).returncode
        if code != 0:
            print(f"!! {name} exited with {code}", flush=True)


# ── validators ─────────────────────────────────────────────────────────────

def ollama(path: str, body: dict[str, Any], host: str) -> dict[str, Any]:
    req = request.Request(f"{host.rstrip('/')}{path}", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with request.urlopen(req, timeout=600) as response:
        return json.loads(response.read().decode() or "{}")


def ollama_resident(host: str) -> list[dict[str, Any]]:
    with request.urlopen(f"{host.rstrip('/')}/api/ps", timeout=30) as response:
        return json.loads(response.read().decode()).get("models", [])


def unload_all(host: str) -> None:
    for model in ollama_resident(host):
        ollama("/api/generate", {"model": model["name"], "keep_alive": 0}, host)


def judge(validator: Any, item: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    path = REPO_ROOT / item["diagram_path"]
    opencv = validate_with_opencv(path, runtime["models"]["opencv_prescreen"])
    result = validator.validate(item, path, opencv)
    return {
        "passed": result.passed, "score": result.score, "reason": result.reason[:300],
        "opencv_passed": opencv.passed, "opencv_score": opencv.score,
        **{k: result.metadata.get(k) for k in ("latency_s", "prompt_tokens", "output_tokens", "raw_verdict", "parse_failed")},
    }


def run_validator_over(entry: dict[str, Any], items: list[dict[str, Any]], results_path: Path, bench: dict[str, Any], extra: dict[str, Any]) -> None:
    runtime = load_runtime_config(REPO_ROOT)
    settings = validator_settings(entry, bench, runtime["models"])
    host = settings.get("host") or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    unload_all(host)
    validator = build_outline_validator(settings)
    prov = provenance()
    # Warm-up call (not recorded) so the first row doesn't carry model load.
    if items:
        judge(validator, items[0], runtime)
    vram = sum(m.get("size_vram", 0) for m in ollama_resident(host)) / 1e9
    total = sum(m.get("size", 0) for m in ollama_resident(host)) / 1e9
    for item in items:
        row = {"model": entry["name"], "ollama_model": settings["ollama_model_name"], "item_id": item.get("item_id") or item.get("sample_id"),
               **extra, **prov, "resident_gb": round(total, 2), "vram_gb": round(vram, 2)}
        try:
            row.update(judge(validator, item, runtime))
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"[:500]
        append_ndjson(results_path, row)
        print(json.dumps({k: row.get(k) for k in ("model", "item_id", "passed", "score", "latency_s", "parse_failed", "error")}), flush=True)
    unload_all(host)


def cmd_validators(args: argparse.Namespace, smoke: bool = False) -> None:
    bench = load_bench()
    items = json.loads(LABELSET.read_text())["items"]
    if smoke:
        items = items[:2]  # 1 warm-up + 1 recorded
    results_path = run_dir(bench) / ("smoke_validators.ndjson" if smoke else "validators.ndjson")
    names = args.models or [entry["name"] for entry in bench["validators"]]
    for name in names:
        print(f"== validator {name}", flush=True)
        run_validator_over(entry_by_name(bench["validators"], name), items, results_path, bench, {"task": "labelset"})


def cmd_referee(args: argparse.Namespace) -> None:
    bench = load_bench()
    rows = [r for r in read_ndjson(run_dir(bench) / "rectifiers.ndjson") if r.get("output_path")]
    items = [{**r, "diagram_path": r["output_path"], "item_id": f"{r['model']}/{r['sample_id']}"} for r in rows]
    run_validator_over(entry_by_name(bench["validators"], args.model), items, run_dir(bench) / "referee.ndjson", bench, {"task": "referee"})


def cmd_smoke(args: argparse.Namespace) -> None:
    if not args.only or args.only == "rectifiers":
        cmd_rectifiers(args, smoke=True)
    if not args.only or args.only == "validators":
        cmd_validators(args, smoke=True)


# ── memory watch (shared with benchmark_setups.py) ─────────────────────────

def _system_memory_gb() -> tuple[float, float]:
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        info[key] = int(value.split()[0]) / 1e6
    return info["MemTotal"] - info["MemAvailable"], info["SwapTotal"] - info["SwapFree"]


def _gpu_used_gb() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        return round(float(out.split()[0]) / 1024, 2)
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


# ── report ─────────────────────────────────────────────────────────────────

def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 2)


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 3) if values else None


def summarize_validators(rows: list[dict[str, Any]], labels: dict[str, bool]) -> list[dict[str, Any]]:
    out = []
    for model in dict.fromkeys(r["model"] for r in rows):
        mine = [r for r in rows if r["model"] == model and "error" not in r]
        lat = [r["latency_s"] for r in mine if r.get("latency_s") is not None]
        summary: dict[str, Any] = {
            "model": model, "n": len(mine),
            "errors": sum(1 for r in rows if r["model"] == model and "error" in r),
            "parse_fail_rate": round(sum(1 for r in mine if r.get("parse_failed")) / max(len(mine), 1), 3),
            "pass_rate": round(sum(1 for r in mine if r["passed"]) / max(len(mine), 1), 3),
            "score_mean": _mean([r["score"] for r in mine]),
            "score_stdev": round(statistics.pstdev([r["score"] for r in mine]), 3) if mine else None,
            "latency_p50_s": _pct(lat, 0.5), "latency_p95_s": _pct(lat, 0.95),
            "vram_gb": mine[0].get("vram_gb") if mine else None, "resident_gb": mine[0].get("resident_gb") if mine else None,
        }
        judged = [(r["passed"], labels[r["item_id"]]) for r in mine if r["item_id"] in labels]
        if judged:
            tp = sum(1 for p, t in judged if p and t)
            fp = sum(1 for p, t in judged if p and not t)
            fn = sum(1 for p, t in judged if not p and t)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            summary.update({
                "labeled_n": len(judged),
                "accuracy": round(sum(1 for p, t in judged if p == t) / len(judged), 3),
                "precision": round(precision, 3), "recall": round(recall, 3),
                "f1": round(2 * precision * recall / (precision + recall), 3) if precision + recall else 0.0,
            })
        out.append(summary)
    return sorted(out, key=lambda s: (-(s.get("f1") or 0), s["latency_p50_s"] or 1e9))


def summarize_rectifiers(rows: list[dict[str, Any]], referee: list[dict[str, Any]]) -> list[dict[str, Any]]:
    verdicts = {r["item_id"]: r for r in referee if "error" not in r}
    out = []
    for model in dict.fromkeys(r["model"] for r in rows):
        mine = [r for r in rows if r["model"] == model]
        ok = [r for r in mine if "error" not in r]
        warm = [r["latency_s"] for r in ok if not r.get("includes_model_load")]
        judged = [verdicts[f"{model}/{r['sample_id']}"] for r in ok if f"{model}/{r['sample_id']}" in verdicts]
        out.append({
            "model": model, "reference": mine[0].get("reference", False), "n": len(mine), "errors": len(mine) - len(ok),
            "opencv_pass_rate": round(sum(1 for r in ok if r["opencv_passed"]) / max(len(ok), 1), 3),
            "opencv_score_mean": _mean([r["opencv_score"] for r in ok]),
            "trajectory_ok_rate": round(sum(1 for r in ok if r["trajectory_ok"]) / max(len(ok), 1), 3),
            "strokes_mean": _mean([r["stroke_count"] for r in ok]),
            "referee_pass_rate": round(sum(1 for j in judged if j["passed"]) / len(judged), 3) if judged else None,
            "referee_score_mean": _mean([j["score"] for j in judged]),
            "sec_per_img_p50": _pct(warm, 0.5),
            "first_call_s": next((r["latency_s"] for r in ok if r.get("includes_model_load")), None),
            "peak_vram_gb": max((r.get("peak_vram_gb") or 0 for r in mine), default=None),
            "peak_rss_gb": max((r.get("peak_rss_gb") or 0 for r in mine), default=None),
        })
    return sorted(out, key=lambda s: (s["reference"], -(s["referee_pass_rate"] or 0), -s["opencv_pass_rate"]))


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join("" if row.get(c) is None else str(row.get(c)) for c in columns) + " |")
    return "\n".join(lines)


def contact_sheet(rows: list[dict[str, Any]], destination: Path, cell: int = 160) -> None:
    models = list(dict.fromkeys(r["model"] for r in rows if r.get("output_path")))
    samples = list(dict.fromkeys(r["sample_id"] for r in rows if r.get("output_path")))
    lookup = {(r["model"], r["sample_id"]): r["output_path"] for r in rows if r.get("output_path")}
    label_w = 190
    sheet = Image.new("RGB", (label_w + cell * len(samples), cell * len(models)), "white")
    draw = ImageDraw.Draw(sheet)
    for y, model in enumerate(models):
        draw.text((6, y * cell + cell // 2), model, fill="black")
        for x, sample in enumerate(samples):
            if (model, sample) in lookup:
                thumb = Image.open(REPO_ROOT / lookup[(model, sample)]).convert("RGB")
                thumb.thumbnail((cell - 6, cell - 6))
                sheet.paste(thumb, (label_w + x * cell + 3, y * cell + 3))
    sheet.save(destination)


def cmd_report(args: argparse.Namespace) -> None:
    bench = load_bench()
    out = run_dir(bench)
    labels = {r["item_id"]: bool(r["label"]) for r in read_ndjson(LABELS)}
    validators = summarize_validators(read_ndjson(out / "validators.ndjson"), labels)
    rect_rows = read_ndjson(out / "rectifiers.ndjson")
    rectifiers = summarize_rectifiers(rect_rows, read_ndjson(out / "referee.ndjson"))
    if rect_rows:
        contact_sheet(rect_rows, out / "contact_rectifiers.png")
    summary = {"run_id": bench["run_id"], **provenance(), "labels": len(labels), "validators": validators, "rectifiers": rectifiers}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    md = [
        f"# Model benchmark `{bench['run_id']}`",
        f"git `{summary['git_sha']}`, config hash `{summary['config_hash']}`, human labels: {len(labels)}.",
        "Plan and method: `docs/benchmarks/model_benchmark_plan.md`.",
        "\n## Validators (label set)",
        md_table(validators, ["model", "labeled_n", "accuracy", "precision", "recall", "f1", "pass_rate", "parse_fail_rate",
                              "score_mean", "score_stdev", "latency_p50_s", "latency_p95_s", "vram_gb", "resident_gb", "errors"]),
        "\n## Rectifiers (eval subjects, one attempt each)",
        md_table(rectifiers, ["model", "reference", "n", "referee_pass_rate", "referee_score_mean", "opencv_pass_rate",
                              "opencv_score_mean", "trajectory_ok_rate", "strokes_mean", "sec_per_img_p50", "first_call_s",
                              "peak_vram_gb", "peak_rss_gb", "errors"]),
        "\nContact sheet: `outputs/benchmarks/" + str(bench["run_id"]) + "/contact_rectifiers.png`.",
    ]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    docs = REPO_ROOT / "docs" / "benchmarks" / f"{bench['run_id']}.md"
    docs.write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build-evalset")
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--only", choices=["rectifiers", "validators"])
    smoke.add_argument("--models", nargs="*")
    for name in ("rectifiers", "validators"):
        p = sub.add_parser(name)
        p.add_argument("--models", nargs="*", help="Subset of model names from configs/benchmark.yaml.")
        p.add_argument("--limit", type=int, default=None)
    referee = sub.add_parser("referee")
    referee.add_argument("--model", required=True)
    sub.add_parser("report")
    one = sub.add_parser("_rectify-one")
    one.add_argument("--model", required=True)
    one.add_argument("--limit", type=int, default=None)
    one.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    handlers = {
        "build-evalset": cmd_build_evalset, "smoke": cmd_smoke, "rectifiers": cmd_rectifiers,
        "validators": cmd_validators, "referee": cmd_referee, "report": cmd_report, "_rectify-one": cmd_rectify_one,
    }
    if args.command in ("smoke",):
        args.limit = None
    handlers[args.command](args)


if __name__ == "__main__":
    main()
