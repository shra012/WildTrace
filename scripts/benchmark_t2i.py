#!/usr/bin/env python3
"""Text-to-image generator comparison (FA2-31): SANA vs FLUX-schnell on FAL.

Both models get the production line-drawing prompt (build_line_drawing_prompt)
for each r2 evalset subject, with the same seed, and every output goes
through the production binarization. Metrics reuse wildtrace/eval_metrics.py
and the r2 judges; shape/pose fidelity is skipped because neither model sees
the source photo. Config: configs/benchmark.yaml `t2i_comparison`.

    python scripts/benchmark_t2i.py generate [--limit 2]   # FAL; needs FAL_KEY
    python scripts/benchmark_t2i.py evaluate               # metrics + OpenRouter judges
    python scripts/benchmark_t2i.py report                 # docs/benchmarks/t2i_r1/

Steps are resumable: rows already in the ndjson logs are skipped.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error, request

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_models import REPO_ROOT, append_ndjson, load_bench, md_table, provenance, read_ndjson  # noqa: E402

from wildtrace import eval_metrics as em  # noqa: E402
from wildtrace.config import load_runtime_config  # noqa: E402
from wildtrace.diagram import (  # noqa: E402
    _binarize_to_size, build_line_drawing_prompt, build_outline_validator, compute_opencv_metrics, validate_with_opencv,
)


def cfg() -> dict[str, Any]:
    return load_bench()["t2i_comparison"]


def out_dir() -> Path:
    path = REPO_ROOT / "outputs" / "benchmarks" / cfg()["run_id"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def docs_dir() -> Path:
    path = REPO_ROOT / "docs" / "benchmarks" / cfg()["run_id"].replace("-", "_")
    path.mkdir(parents=True, exist_ok=True)
    return path


def subjects() -> list[dict[str, Any]]:
    return json.loads((REPO_ROOT / cfg()["evalset"]).read_text())["subjects"]


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


# ── generate ───────────────────────────────────────────────────────────────

def _fal_generate(endpoint: str, body: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]:
    req = request.Request(
        f"https://fal.run/{endpoint}",
        data=json.dumps(body).encode("utf-8"),
        headers={"authorization": f"Key {os.environ['FAL_KEY']}", "content-type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=300) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        raise RuntimeError(f"FAL {endpoint} failed ({exc.code}): {exc.read().decode('utf-8', 'ignore')[:300]}") from exc
    url = payload["images"][0]["url"]
    with request.urlopen(url, timeout=120) as response:
        image = Image.open(io.BytesIO(response.read()))
        image.load()
    return image, payload


def _cost_usd(generator: dict[str, Any], size: tuple[int, int]) -> float:
    megapixels = size[0] * size[1] / 1e6
    if generator.get("round_up_mp"):
        megapixels = math.ceil(megapixels)
    return round(megapixels * float(generator["usd_per_mp"]), 5)


def cmd_generate(args: argparse.Namespace) -> None:
    load_runtime_config(REPO_ROOT)  # loads .env
    if not os.getenv("FAL_KEY"):
        raise SystemExit("FAL_KEY is not set (add it to .env).")
    c = cfg()
    log = out_dir() / "generations.ndjson"
    done = {(r["generator"], r["sample_id"]) for r in read_ndjson(log) if "error" not in r}
    todo = subjects()[: args.limit] if args.limit else subjects()
    for index, subject in enumerate(todo):
        prompt = build_line_drawing_prompt(subject)
        seed = int(c["seed"]) + index
        for generator in c["generators"]:
            if (generator["name"], subject["sample_id"]) in done:
                continue
            row: dict[str, Any] = {"generator": generator["name"], "endpoint": generator["endpoint"],
                                   "sample_id": subject["sample_id"], "category": subject["category"],
                                   "seed": seed, "prompt": prompt}
            body = {"prompt": prompt, "seed": seed, "num_images": 1, "output_format": "png",
                    "enable_safety_checker": False,
                    "image_size": {"width": int(c["size"]), "height": int(c["size"])}, **generator.get("params", {})}
            start = time.perf_counter()
            try:
                image, payload = _fal_generate(generator["endpoint"], body)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"[:500]
                append_ndjson(log, row)
                print(json.dumps(row), flush=True)
                continue
            raw = out_dir() / generator["name"] / "raw" / f"{subject['sample_id']}.png"
            final = out_dir() / generator["name"] / f"{subject['sample_id']}.png"
            raw.parent.mkdir(parents=True, exist_ok=True)
            image.save(raw)
            _binarize_to_size(image, image.size).save(final)  # production post-processing
            row.update({"raw_path": rel(raw), "output_path": rel(final), "size": list(image.size),
                        "gen_latency_s": round(time.perf_counter() - start, 2),
                        "fal_inference_s": (payload.get("timings") or {}).get("inference"),
                        "gen_cost_usd": _cost_usd(generator, image.size)})
            append_ndjson(log, row)
            print(json.dumps({k: row[k] for k in ("generator", "sample_id", "gen_latency_s", "output_path")}), flush=True)


# ── evaluate ───────────────────────────────────────────────────────────────

def cmd_evaluate(args: argparse.Namespace) -> None:
    runtime = load_runtime_config(REPO_ROOT)
    models = runtime["models"]
    by_id = {s["sample_id"]: s for s in subjects()}
    gens = [r for r in read_ndjson(out_dir() / "generations.ndjson") if "error" not in r]

    metrics_log = out_dir() / "metrics.ndjson"
    scored = {(r["generator"], r["sample_id"]) for r in read_ndjson(metrics_log)}
    pending = [g for g in gens if (g["generator"], g["sample_id"]) not in scored]
    if pending:
        categories = [c["name"] for c in runtime["datasets"]["categories"]]
        draw_model = em.calibrate_draw_time(REPO_ROOT / "isaac-sim")
        scorer = em.BioClipScorer({**models["enrichment"], "device": os.getenv("BIOCLIP_DEVICE", "cpu")}, categories)
        for g in pending:
            path = REPO_ROOT / g["output_path"]
            drawing = Image.open(path).convert("L")
            ink = em.ink_mask(drawing)
            opencv = validate_with_opencv(path, models["opencv_prescreen"])
            out = {
                "generator": g["generator"], "sample_id": g["sample_id"], "category": g["category"],
                "opencv_passed": opencv.passed, "opencv_score": round(opencv.score, 4),
                **{k: round(v, 4) for k, v in compute_opencv_metrics(drawing).items()},
                **em.structure_metrics(ink),
                **em.trajectory_metrics(drawing, ink, runtime["export"], draw_model),
                **scorer.score(drawing, None, g["category"]),
            }
            append_ndjson(metrics_log, out)
            print(json.dumps({k: out.get(k) for k in ("generator", "sample_id", "opencv_passed", "blank", "clip_top1",
                                                       "trajectory_ok")}), flush=True)

    judge_log = out_dir() / "judges.ndjson"
    judged = {(r["judge"], r["generator"], r["sample_id"]) for r in read_ndjson(judge_log) if "error" not in r}
    for entry in cfg()["judges"]:
        settings = {**models["outline_validator"], "backend": "openrouter_semantic_validator", "force_semantic": True,
                    "openrouter_model_name": entry["openrouter_model_name"]}
        judge = build_outline_validator(settings)
        for g in gens:
            if (entry["name"], g["generator"], g["sample_id"]) in judged:
                continue
            path = REPO_ROOT / g["output_path"]
            opencv = validate_with_opencv(path, models["opencv_prescreen"])
            out = {"judge": entry["name"], "strong": bool(entry.get("strong")), "generator": g["generator"],
                   "sample_id": g["sample_id"]}
            try:
                result = judge.validate(by_id[g["sample_id"]], path, opencv)
                out.update({"passed": result.passed, "score": result.score, "reason": result.reason[:300],
                            "latency_s": result.metadata.get("latency_s"), "cost_usd": result.metadata.get("cost_usd"),
                            "parse_failed": bool(result.metadata.get("parse_failed"))})
            except Exception as exc:
                out["error"] = f"{type(exc).__name__}: {exc}"[:500]
            append_ndjson(judge_log, out)
            print(json.dumps({k: out.get(k) for k in ("judge", "generator", "sample_id", "passed", "score", "error")}),
                  flush=True)


# ── report ─────────────────────────────────────────────────────────────────

REPORT_KEYS = [
    ("B. Drawing structure", ["opencv_passed", "opencv_score", "foreground_ratio", "component_count",
                              "small_contour_ratio", "body_bottom_reach", "fill_ratio", "stroke_width_px", "blank"]),
    ("E. Semantic recognizability", ["clip_top1", "clip_p_true"]),
    ("G. Robot drawability", ["stroke_count", "trajectory_ok", "coverage", "path_mm", "pen_up_mm"]),
    ("H. Cost and speed", ["gen_latency_s", "gen_cost_usd"]),
]
RATES = {"opencv_passed", "blank", "clip_top1", "trajectory_ok", "within_budget", "passed", "parse_failed"}


def _fmt(key: str, mean: float | None, ci: tuple[float | None, float | None]) -> str:
    if mean is None:
        return "n/a"
    if key in RATES:
        text = f"{100 * mean:.0f}%"
        return text + (f" ({100 * ci[0]:.0f}%–{100 * ci[1]:.0f}%)" if ci[0] is not None else "")
    if key == "gen_cost_usd":
        return f"${mean:.4f}"
    digits = 0 if abs(mean) >= 100 else 1 if abs(mean) >= 10 else 2
    text = f"{mean:.{digits}f}"
    return text + (f" ({ci[0]:.{digits}f}–{ci[1]:.{digits}f})" if ci[0] is not None else "")


def _stat(values: list[Any]) -> tuple[float | None, tuple[float | None, float | None]]:
    data = [float(v) for v in values if v is not None]
    if not data:
        return None, (None, None)
    return round(float(np.mean(data)), 4), em.bootstrap_ci(data)


def _better(key: str, means: dict[str, float | None], cis: dict[str, tuple]) -> str:
    registry = {m.key: m for m in em.METRICS}
    direction = registry[key].direction if key in registry else ("lower" if key == "gen_cost_usd" else "info")
    names = [n for n in means if means[n] is not None]
    if direction == "info" or key == "stroke_count" or len(names) != 2:
        return ""
    a, b = names
    if means[a] == means[b]:
        return "tie"
    if direction == "target":
        target = 0.16
        best = min(names, key=lambda n: abs(means[n] - target))
    else:
        best = max(names, key=lambda n: means[n]) if direction == "higher" else min(names, key=lambda n: means[n])
    other = b if best == a else a
    lo_b, hi_b = cis[best]
    lo_o, hi_o = cis[other]
    overlap = None in (lo_b, hi_b, lo_o, hi_o) or not (hi_b < lo_o or hi_o < lo_b)
    return f"{best} (CIs overlap)" if overlap else f"**{best}**"


def contact_sheet(gens: list[dict[str, Any]], path: Path, names: list[str], tile: int = 192) -> None:
    ids = list(dict.fromkeys(g["sample_id"] for g in gens))
    lookup = {(g["generator"], g["sample_id"]): g for g in gens}
    sheet = Image.new("RGB", (tile * len(ids), (tile + 18) * len(names)), "white")
    draw = ImageDraw.Draw(sheet)
    for col, sample_id in enumerate(ids):
        for row, name in enumerate(names):
            g = lookup.get((name, sample_id))
            x, y = col * tile, row * (tile + 18)
            if g:
                image = Image.open(REPO_ROOT / g["output_path"]).convert("RGB")
                image.thumbnail((tile - 4, tile - 4))
                sheet.paste(image, (x + 2, y + 18))
            draw.text((x + 4, y + 2), f"{name}: {(g or {}).get('category', '')}", fill="black")
    sheet.save(path)


def cmd_report(args: argparse.Namespace) -> None:
    c = cfg()
    names = [g["name"] for g in c["generators"]]
    gens = [r for r in read_ndjson(out_dir() / "generations.ndjson") if "error" not in r]
    errors = [r for r in read_ndjson(out_dir() / "generations.ndjson") if "error" in r]
    metrics = {(r["generator"], r["sample_id"]): r for r in read_ndjson(out_dir() / "metrics.ndjson")}
    judges = [r for r in read_ndjson(out_dir() / "judges.ndjson") if "error" not in r]
    gen_by = {(g["generator"], g["sample_id"]): g for g in gens}

    # Paired: only subjects every generator produced.
    paired = sorted(set.intersection(*[{g["sample_id"] for g in gens if g["generator"] == n} for n in names]))
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"run_id": c["run_id"], **provenance(), "n_subjects": len(paired),
                               "generation_errors": len(errors), "metrics": {}}
    for title, keys in REPORT_KEYS:
        rows.append({"metric": f"**{title}**"})
        for key in keys:
            means, cis = {}, {}
            for n in names:
                source = gen_by if key.startswith("gen_") else metrics
                means[n], cis[n] = _stat([source.get((n, s), {}).get(key) for s in paired])
            summary["metrics"][key] = {n: {"mean": means[n], "ci95": cis[n]} for n in names}
            registry = {m.key: m for m in em.METRICS}
            label = registry[key].name if key in registry else {"gen_latency_s": "Generation time per image (s)",
                                                                  "gen_cost_usd": "Generation cost per image (USD)"}[key]
            arrow = {"higher": "↑", "lower": "↓", "target": "◎ 0.16"}.get(registry[key].direction, "") if key in registry \
                else "↓"
            if key == "stroke_count":
                arrow = "1–6 exportable"
            rows.append({"metric": label, "better": arrow, **{n: _fmt(key, means[n], cis[n]) for n in names},
                         "winner": _better(key, means, cis)})

    # Latency tail, as in the Molmo report (mean / p50 / p95).
    tails = {}
    for n in names:
        latencies = [float(gen_by[(n, s)]["gen_latency_s"]) for s in paired]
        tails[n] = (round(float(np.percentile(latencies, 50)), 2), round(float(np.percentile(latencies, 95)), 2))
    summary["metrics"]["gen_latency_p50_p95_s"] = tails
    rows.append({"metric": "Generation time p50 / p95 (s)", "better": "↓",
                 **{n: f"{tails[n][0]:.2f} / {tails[n][1]:.2f}" for n in names}, "winner": ""})

    rows.append({"metric": "**F. Judges (OpenRouter)**"})
    judge_pass: dict[str, dict[str, dict[str, bool]]] = {}
    for entry in c["judges"]:
        jn = entry["name"]
        judge_pass[jn] = {n: {r["sample_id"]: bool(r["passed"]) for r in judges
                              if r["judge"] == jn and r["generator"] == n and r["sample_id"] in paired}
                          for n in names}
        means, cis = {}, {}
        for n in names:
            means[n], cis[n] = _stat(list(judge_pass[jn][n].values()))
        summary["metrics"][f"judge_pass:{jn}"] = {n: {"mean": means[n], "ci95": cis[n]} for n in names}
        label = f"Judge pass: `{entry['openrouter_model_name']}`" + (" (strong)" if entry.get("strong") else "")
        rows.append({"metric": label, "better": "↑", **{n: _fmt("passed", means[n], cis[n]) for n in names},
                     "winner": _better("clip_top1", means, cis)})
        latency = {n: [float(r["latency_s"]) for r in judges if r["judge"] == jn and r["generator"] == n
                       and r["sample_id"] in paired and r.get("latency_s") is not None] for n in names}
        rows.append({"metric": f"Judge latency p50 (s): `{jn}`", "better": "↓",
                     **{n: f"{np.percentile(latency[n], 50):.2f}" if latency[n] else "n/a" for n in names},
                     "winner": ""})

    strong = next(e["name"] for e in c["judges"] if e.get("strong"))
    # Agreement of each cheaper judge with the strong one, as r2 reports for its own judge.
    for entry in c["judges"]:
        if entry.get("strong"):
            continue
        jn, kappas = entry["name"], {}
        for n in names:
            ids = [s for s in paired if s in judge_pass[jn][n] and s in judge_pass[strong][n]]
            kappas[n] = em.cohen_kappa([judge_pass[jn][n][s] for s in ids], [judge_pass[strong][n][s] for s in ids])
        summary["metrics"][f"kappa:{jn}_vs_{strong}"] = kappas
        rows.append({"metric": f"`{jn}` vs strong judge (Cohen's κ)", "better": "↑",
                     **{n: "n/a" if kappas[n] is None else f"{kappas[n]:.2f}" for n in names}, "winner": ""})

    a, b = names
    wins = sum(judge_pass[strong][a].get(s, False) and not judge_pass[strong][b].get(s, False) for s in paired)
    losses = sum(judge_pass[strong][b].get(s, False) and not judge_pass[strong][a].get(s, False) for s in paired)
    summary["strong_judge_sign_test"] = {"wins": {a: wins, b: losses}, "p": em.sign_test_p(wins, losses)}
    judge_cost = sum(float(r.get("cost_usd") or 0) for r in judges)
    gen_cost = {n: round(sum(float(gen_by[(n, s)]["gen_cost_usd"]) for s in paired), 4) for n in names}
    summary["total_cost_usd"] = {"generation": gen_cost, "judges": round(judge_cost, 4)}

    (docs_dir() / "summary.json").write_text(json.dumps(summary, indent=2))
    for name in ("generations.ndjson", "metrics.ndjson", "judges.ndjson"):
        (docs_dir() / name).write_text((out_dir() / name).read_text())
    contact_sheet([g for g in gens if g["sample_id"] in paired], docs_dir() / "contact_sheet.png", names)

    per_category = []
    for category in sorted({gen_by[(a, s)]["category"] for s in paired}):
        ids = [s for s in paired if gen_by[(a, s)]["category"] == category]
        per_category.append({"category": f"{category} (n={len(ids)})",
                             **{n: f"{sum(judge_pass[strong][n].get(s, False) for s in ids)}/{len(ids)}" for n in names}})

    md = [
        f"# Text-to-image comparison `{c['run_id']}`: SANA vs FLUX-schnell",
        f"git `{summary['git_sha']}`, config hash `{summary['config_hash']}`. Config: `configs/benchmark.yaml` "
        "`t2i_comparison`; script: `scripts/benchmark_t2i.py`.",
        "",
        f"Both models run on FAL (`{c['generators'][0]['endpoint']}`, `{c['generators'][1]['endpoint']}`) at "
        f"{c['size']}×{c['size']}, one image per subject, prompted with the production line-drawing prompt for each "
        f"of the {len(paired)} r2 evalset subjects (`{c['evalset']}`), same seed per subject, production "
        "binarization. Neither model sees the source photo, so shape and pose fidelity (r2 groups C, D) are not "
        "measured. Estimated draw time is not reported: its model calibrates from Isaac Sim run logs that were not "
        "available on the machine that ran this. Values are means or rates with bootstrap 95% CI. **Bold** = better "
        "and CIs don't overlap.",
        "",
        "## Results",
        md_table(rows, ["metric", "better", *names, "winner"]),
        "",
        f"Strong judge ({strong}) paired sign test: {a} passes where {b} fails on {wins} subjects, the reverse on "
        f"{losses}; p = {summary['strong_judge_sign_test']['p']}.",
        "",
        f"Cost for this run: generation {', '.join(f'{n} ${v:.4f}' for n, v in gen_cost.items())}; "
        f"OpenRouter judges ${judge_cost:.4f}. Generation errors: {len(errors)}.",
        "",
        f"## Strong-judge pass by category ({strong})",
        md_table(per_category, ["category", *names]),
        "",
        f"Contact sheet: `{rel(docs_dir() / 'contact_sheet.png')}`.",
    ]
    findings = docs_dir() / "findings.md"
    if findings.exists():
        md[5:5] = ["## Key findings", findings.read_text().strip(), ""]
    text = "\n".join(md) + "\n"
    (docs_dir() / "README.md").write_text(text)
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--limit", type=int, default=None, help="Only the first N evalset subjects.")
    sub.add_parser("evaluate")
    sub.add_parser("report")
    args = parser.parse_args()
    {"generate": cmd_generate, "evaluate": cmd_evaluate, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    main()
