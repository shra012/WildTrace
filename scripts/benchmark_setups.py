#!/usr/bin/env python3
"""End-to-end comparison of full generator + judge setups (issue #14, run r2).

Each setup is a rectifier plus its own validator, run through the production
generate -> OpenCV -> judge -> feedback -> retry loop. Config:
configs/benchmark.yaml `comparison`. Metric definitions: wildtrace/eval_metrics.py
(METRICS) and docs/benchmarks/eval_suite_r2.docx.

    python scripts/benchmark_setups.py build-evalset
    python scripts/benchmark_setups.py pipeline --setup omnigen2-openrouter [--limit 1]
    python scripts/benchmark_setups.py pipeline --setup flux-ollama
    python scripts/benchmark_setups.py evaluate     # per-attempt metrics (BioCLIP on GPU)
    python scripts/benchmark_setups.py referee      # every final drawing x every referee
    python scripts/benchmark_setups.py report       # summary.json, docs/benchmarks/r2.md, contact sheet

Every generate/judge/feedback call is logged to attempts.ndjson by recording
proxies, so the production loop itself stays untouched.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import re
import resource
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib import error, request

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_models import (  # noqa: E402
    REPO_ROOT, _gpu_used_gb, _mean, _pct, _system_memory_gb, append_ndjson, entry_by_name,
    load_bench, md_table, provenance, read_ndjson,
)

from wildtrace.config import load_runtime_config  # noqa: E402
from wildtrace.diagram import (  # noqa: E402
    OutlineRectifier, SemanticValidator, _cropped_mask_image, build_outline_rectifier,
    build_outline_validator, compute_opencv_metrics, run_langgraph_validation_loop, validate_with_opencv,
)
from wildtrace import eval_metrics as em  # noqa: E402

ATTEMPT_RE = re.compile(r"_attempt(\d+)\.png$")


def comparison() -> dict[str, Any]:
    return load_bench()["comparison"]


def out_dir() -> Path:
    path = REPO_ROOT / "outputs" / "benchmarks" / str(comparison()["run_id"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def evalset_path() -> Path:
    return REPO_ROOT / comparison()["evalset"]["path"]


def subjects_by_id() -> dict[str, dict[str, Any]]:
    return {s["sample_id"]: s for s in json.loads(evalset_path().read_text())["subjects"]}


def setup_by_name(name: str) -> dict[str, Any]:
    return entry_by_name(comparison()["setups"], name)


def rel(path: Path | str) -> str:
    return str(Path(path).resolve().relative_to(REPO_ROOT))


# ── eval set ───────────────────────────────────────────────────────────────

def cmd_build_evalset(args: argparse.Namespace) -> None:
    cfg = comparison()["evalset"]
    rng = random.Random(cfg["seed"])
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for subject in read_ndjson(REPO_ROOT / "outputs" / "silver" / "checkpoints" / "silver_subjects.ndjson"):
        if all(subject.get(k) and (REPO_ROOT / subject[k]).exists() for k in ("crop_path", "mask_path")):
            by_category[subject["category"]].append(subject)
    configured = [c["name"] for c in load_runtime_config(REPO_ROOT)["datasets"]["categories"]]
    subjects, missing = [], []
    for category in configured:
        pool = sorted(by_category.get(category, []), key=lambda s: s["sample_id"])
        want = int(cfg["per_category"]) + int(cfg.get("extra", {}).get(category, 0))
        if not pool:
            missing.append(category)
            continue
        subjects += rng.sample(pool, min(want, len(pool)))
    path = evalset_path()
    path.write_text(json.dumps({"seed": cfg["seed"], "excluded_categories": missing, "subjects": subjects}, indent=2))
    counts = defaultdict(int)
    for subject in subjects:
        counts[subject["category"]] += 1
    print(f"{len(subjects)} subjects {dict(counts)}; no data for {missing} -> {rel(path)}")


# ── recording proxies ──────────────────────────────────────────────────────

class RecordingRectifier(OutlineRectifier):
    """Delegates to the real rectifier and logs every run() to attempts.ndjson."""

    def __init__(self, inner: OutlineRectifier, log: Any, unload_after_run: bool = False) -> None:
        super().__init__(inner.settings)
        self.inner, self.log, self.calls = inner, log, 0
        self.unload_after_run = unload_after_run

    def prepare_conditioning_image(self, crop_image: Image.Image, isolated_image: Image.Image | None = None) -> Image.Image:
        return self.inner.prepare_conditioning_image(crop_image, isolated_image)

    def run(self, sample, subject_image, subject_mask, generated_path, destination, params):  # noqa: ANN001, ANN201
        self.calls += 1
        match = ATTEMPT_RE.search(str(destination))
        row = {"kind": "generate", "sample_id": sample["sample_id"], "category": sample["category"],
               "attempt": int(match.group(1)) if match else None,
               "includes_model_load": self.calls == 1 or self.unload_after_run,
               "params": {k: v for k, v in params.items() if k in ("strength", "prompt", "image_guidance_scale")}}
        started = time.perf_counter()
        try:
            result = self.inner.run(sample, subject_image, subject_mask, generated_path, destination, params)
        except Exception as exc:
            self.log({**row, "latency_s": round(time.perf_counter() - started, 3), "error": f"{type(exc).__name__}: {exc}"[:500]})
            raise
        self.log({**row, "latency_s": round(time.perf_counter() - started, 3), "output_path": rel(result.diagram_path),
                  "prompt_used": result.metadata.get("prompt"),
                  "generator": {k: v for k, v in result.metadata.items() if k != "prompt"}})
        if self.unload_after_run:
            release_rectifier(self.inner)
        return result

    def unload(self) -> None:
        self.inner.unload()


def release_rectifier(rectifier: OutlineRectifier) -> None:
    """Drop the loaded pipeline and hand freed heap back to the OS, so a
    local judge can load next. FLUX's CPU offload holds ~20GB of host RAM and
    loading Ollama's model beside it OOM-kills this 24GB box; gc alone leaves
    the freed arenas mapped."""
    import ctypes

    rectifier.unload()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:  # pragma: no cover - non-glibc platforms
        pass


class RecordingValidator(SemanticValidator):
    """Delegates to the real validator and logs validate()/suggest_params()."""

    def __init__(self, inner: SemanticValidator, log: Any) -> None:
        super().__init__(inner.settings)
        self.inner, self.log = inner, log

    def validate(self, sample, diagram_path, opencv_result):  # noqa: ANN001, ANN201
        started = time.perf_counter()
        base = {"kind": "judge", "sample_id": sample["sample_id"], "category": sample["category"],
                "diagram_path": rel(diagram_path), "opencv_passed": opencv_result.passed,
                "opencv_score": round(opencv_result.score, 4), "opencv_flags": opencv_result.flags}
        try:
            result = self.inner.validate(sample, diagram_path, opencv_result)
        except Exception as exc:
            self.log({**base, "latency_s": round(time.perf_counter() - started, 3), "error": f"{type(exc).__name__}: {exc}"[:500]})
            raise
        meta = result.metadata
        self.log({**base, "passed": result.passed, "score": result.score, "reason": result.reason[:300],
                  "latency_s": meta.get("latency_s"), "skipped": bool(meta.get("skipped")),
                  "parse_failed": bool(meta.get("parse_failed")), "raw_verdict": meta.get("raw_verdict"),
                  "prompt_tokens": meta.get("prompt_tokens"), "output_tokens": meta.get("output_tokens"),
                  "cost_usd": meta.get("cost_usd"), "model": meta.get("model_name")})
        return result

    def suggest_params(self, sample, subject_path, diagram_path, opencv_result, current_params):  # noqa: ANN001, ANN201
        started = time.perf_counter()
        suggestion = self.inner.suggest_params(sample, subject_path, diagram_path, opencv_result, current_params)
        self.log({"kind": "feedback", "sample_id": sample["sample_id"], "category": sample["category"],
                  "diagram_path": rel(diagram_path), "latency_s": round(time.perf_counter() - started, 3),
                  "suggestion": suggestion})
        return suggestion


# ── pipeline ───────────────────────────────────────────────────────────────

def setup_settings(setup: dict[str, Any], models: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rectifier and validator settings for a setup. The outline_rectifier block
    holds namespaced keys for every backend, so it is a safe base for either;
    the setup, not .env, decides backend and model."""
    entry = entry_by_name(load_bench()["rectifiers"], setup["rectifier"])
    rectifier = {**models["outline_rectifier"], **{k: v for k, v in entry.items() if k not in ("name", "reference")},
                 **setup.get("rectifier_overrides", {})}
    validator = {**models["outline_validator"], **setup["validator"]}
    return rectifier, validator


def cmd_pipeline_one(args: argparse.Namespace) -> None:
    """Child process: run the setup over the eval subjects (or --limit)."""
    setup = setup_by_name(args.setup)
    runtime = load_runtime_config(REPO_ROOT)
    models = runtime["models"]
    rectifier_cfg, validator_cfg = setup_settings(setup, models)
    subjects = list(subjects_by_id().values())[: args.limit or None]
    if args.resume:
        done = {r["sample_id"] for r in read_ndjson(out_dir() / "pipeline.ndjson") if r["setup"] == setup["name"]}
        subjects = [s for s in subjects if s["sample_id"] not in done]
    prov = provenance()
    attempts_path = out_dir() / "attempts.ndjson"
    current: dict[str, Any] = {}

    def log(row: dict[str, Any]) -> None:
        append_ndjson(attempts_path, {"setup": setup["name"], **row, **prov, "logged_at": time.time()})

    import torch

    validator = RecordingValidator(build_outline_validator(validator_cfg), log)
    rectifier = RecordingRectifier(build_outline_rectifier(rectifier_cfg), log,
                                   unload_after_run=bool(setup.get("unload_rectifier_after_run")))
    for subject in subjects:
        current.clear()
        row: dict[str, Any] = {"setup": setup["name"], "rectifier": setup["rectifier"],
                               "validator": validator_cfg["backend"], "sample_id": subject["sample_id"],
                               "category": subject["category"], **prov}
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            record = run_langgraph_validation_loop(
                sample=subject,
                subject_path=REPO_ROOT / subject["crop_path"],
                conditioning_path=REPO_ROOT / subject["isolated_path"] if subject.get("isolated_path") else None,
                subject_mask_path=REPO_ROOT / subject["mask_path"],
                diagram_root=out_dir() / setup["name"],
                diagram_generator=rectifier,
                outline_validator=validator,
                opencv_settings=models["opencv_prescreen"],
                initial_params={},
                max_attempts=int(comparison()["max_attempts"]),
            )
            row.update({
                "accepted": record["diagram_validation_status"] == "accepted",
                "attempts": record["diagram_attempt"],
                "combined_score": record["diagram_validation_score"],
                "opencv_flags": record["opencv_flags"],
                "validator_reason": str(record["outline_validator_reason"])[:300],
                "output_path": rel(record["diagram_path"]),
            })
        except Exception as exc:  # one bad subject must not sink the run
            row["error"] = f"{type(exc).__name__}: {exc}"[:500]
        row["subject_latency_s"] = round(time.perf_counter() - started, 3)
        if torch.cuda.is_available():
            row["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        row["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
        append_ndjson(out_dir() / "pipeline.ndjson", row)
        print(json.dumps({k: row.get(k) for k in ("sample_id", "category", "accepted", "attempts", "combined_score",
                                                   "subject_latency_s", "peak_vram_gb", "error")}), flush=True)
    rectifier.unload()


def _ollama_host(validator_cfg: dict[str, Any]) -> str:
    return str(validator_cfg.get("host") or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")


def _ollama_models(host: str) -> list[str] | None:
    try:
        with request.urlopen(f"{host}/api/tags", timeout=5) as response:
            return [m["name"] for m in json.loads(response.read().decode()).get("models", [])]
    except (error.URLError, OSError):
        return None


def ensure_ollama(validator_cfg: dict[str, Any]) -> None:
    """Start `ollama serve` if it's down; refuse to run if the model isn't pulled."""
    host = _ollama_host(validator_cfg)
    if _ollama_models(host) is None:
        log_path = out_dir() / "ollama_serve.log"
        subprocess.Popen(["ollama", "serve"], stdout=log_path.open("a"), stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(30):
            time.sleep(1)
            if _ollama_models(host) is not None:
                break
    available = _ollama_models(host)
    if available is None:
        raise SystemExit(f"Ollama not reachable at {host}; see {rel(out_dir() / 'ollama_serve.log')}")
    model = validator_cfg["ollama_model_name"]
    if model not in available and f"{model}:latest" not in available:
        raise SystemExit(f"Ollama model {model!r} is not pulled (have {available}). Pull it first: ollama pull {model}")


def cmd_pipeline(args: argparse.Namespace) -> None:
    setup = setup_by_name(args.setup)
    _, validator_cfg = setup_settings(setup, load_runtime_config(REPO_ROOT)["models"])
    if validator_cfg["backend"] == "ollama_semantic_validator":
        ensure_ollama(validator_cfg)
    command = [sys.executable, __file__, "_pipeline-one", "--setup", setup["name"]]
    if args.limit:
        command += ["--limit", str(args.limit)]
    peaks = {"sys_ram_gb": 0.0, "swap_gb": 0.0, "gpu_gb": 0.0}
    baseline_ram, baseline_swap = _system_memory_gb()
    done = threading.Event()

    def sample() -> None:
        while not done.wait(1.0):
            ram, swap = _system_memory_gb()
            peaks["sys_ram_gb"] = max(peaks["sys_ram_gb"], ram)
            peaks["swap_gb"] = max(peaks["swap_gb"], swap)
            peaks["gpu_gb"] = max(peaks["gpu_gb"], _gpu_used_gb() or 0.0)

    watcher = threading.Thread(target=sample, daemon=True)
    watcher.start()
    started = time.perf_counter()
    code = subprocess.run(command, cwd=REPO_ROOT).returncode
    oom_kills = 0
    # An OOM kill is a result, not the end of the run: count it and carry on
    # with the subjects that haven't finished (the one in flight is redone).
    while code == -9 and oom_kills < 3:
        oom_kills += 1
        print(f"!! OOM-killed ({oom_kills}); resuming", flush=True)
        code = subprocess.run(command + ["--resume"], cwd=REPO_ROOT).returncode
    done.set()
    watcher.join()
    summary = {
        "setup": setup["name"], "limit": args.limit, "exit_code": code, "oom_kills": oom_kills,
        "oom_killed": code == -9 or oom_kills > 0,
        "wall_s": round(time.perf_counter() - started, 1),
        "baseline_sys_ram_gb": round(baseline_ram, 2), "baseline_swap_gb": round(baseline_swap, 2),
        **{k: round(v, 2) for k, v in peaks.items()},
        "sys_ram_total_gb": round(float(Path("/proc/meminfo").read_text().split()[1]) / 1e6, 2),
    }
    append_ndjson(out_dir() / "pipeline_runs.ndjson", {**summary, **provenance()})
    print(json.dumps(summary, indent=2))


# ── evaluate ───────────────────────────────────────────────────────────────

def cmd_evaluate(args: argparse.Namespace) -> None:
    runtime = load_runtime_config(REPO_ROOT)
    subjects = subjects_by_id()
    categories = [c["name"] for c in runtime["datasets"]["categories"]]
    draw_model = em.calibrate_draw_time(REPO_ROOT / "isaac-sim")
    scorer = em.BioClipScorer(runtime["models"]["enrichment"], categories)
    rows = [r for r in read_ndjson(out_dir() / "attempts.ndjson") if r["kind"] == "generate" and r.get("output_path")]
    results_path = out_dir() / "metrics.ndjson"
    results_path.unlink(missing_ok=True)
    for row in rows:
        subject = subjects[row["sample_id"]]
        drawing = Image.open(REPO_ROOT / row["output_path"]).convert("L")
        ink = em.ink_mask(drawing)
        fill = em.filled_silhouette(ink)
        mask_image = _cropped_mask_image(Image.open(REPO_ROOT / subject["mask_path"]), subject.get("crop_bbox"), drawing.size)
        source_mask = (np.asarray(mask_image) > 0).astype(np.uint8)
        opencv = validate_with_opencv(REPO_ROOT / row["output_path"], runtime["models"]["opencv_prescreen"])
        crop = Image.open(REPO_ROOT / subject["crop_path"]).convert("RGB")
        photo = Image.composite(crop, Image.new("RGB", crop.size, "white"),
                                _cropped_mask_image(Image.open(REPO_ROOT / subject["mask_path"]), subject.get("crop_bbox"), crop.size))
        out = {
            "setup": row["setup"], "sample_id": row["sample_id"], "category": row["category"],
            "attempt": row["attempt"], "output_path": row["output_path"],
            "opencv_passed": opencv.passed, "opencv_score": round(opencv.score, 4),
            **{k: round(v, 4) for k, v in compute_opencv_metrics(drawing).items()},
            **em.structure_metrics(ink),
            **em.fidelity_metrics(fill, source_mask),
            **em.pose_metrics(fill, subject, runtime["viewpoints"]),
            **em.trajectory_metrics(drawing, ink, runtime["export"], draw_model),
            **scorer.score(drawing, photo, subject["category"]),
        }
        append_ndjson(results_path, out)
        print(json.dumps({k: out.get(k) for k in ("setup", "sample_id", "attempt", "silhouette_iou", "fill_ratio",
                                                   "angle_bucket_match", "clip_top1", "coverage", "draw_time_s")}), flush=True)
    if draw_model:
        (out_dir() / "draw_time_model.json").write_text(json.dumps(dataclasses.asdict(draw_model), indent=2))


# ── referee ────────────────────────────────────────────────────────────────

def final_rows() -> list[dict[str, Any]]:
    """Latest full-run final row per (setup, subject)."""
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_ndjson(out_dir() / "pipeline.ndjson"):
        latest[(row["setup"], row["sample_id"])] = row
    return list(latest.values())


def cmd_referee(args: argparse.Namespace) -> None:
    runtime = load_runtime_config(REPO_ROOT)
    models = runtime["models"]
    subjects = subjects_by_id()
    finals = [r for r in final_rows() if r.get("output_path")]
    names = args.referees or [r["name"] for r in comparison()["referees"]]
    results_path = out_dir() / "referee.ndjson"
    for name in names:
        entry = entry_by_name(comparison()["referees"], name)
        settings = {**models["outline_validator"], "force_semantic": True,
                    **{k: v for k, v in entry.items() if k not in ("name", "strong")}}
        if settings["backend"] == "ollama_semantic_validator":
            ensure_ollama(settings)
        judge = build_outline_validator(settings)
        for row in finals:
            path = REPO_ROOT / row["output_path"]
            opencv = validate_with_opencv(path, models["opencv_prescreen"])
            out = {"referee": name, "strong": bool(entry.get("strong")), "setup": row["setup"],
                   "sample_id": row["sample_id"], "category": row["category"], "output_path": row["output_path"]}
            try:
                result = judge.validate(subjects[row["sample_id"]], path, opencv)
                out.update({"passed": result.passed, "score": result.score, "reason": result.reason[:300],
                            "latency_s": result.metadata.get("latency_s"), "cost_usd": result.metadata.get("cost_usd"),
                            "parse_failed": bool(result.metadata.get("parse_failed"))})
            except Exception as exc:
                out["error"] = f"{type(exc).__name__}: {exc}"[:500]
            append_ndjson(results_path, out)
            print(json.dumps({k: out.get(k) for k in ("referee", "setup", "sample_id", "passed", "score", "error")}), flush=True)


# ── report ─────────────────────────────────────────────────────────────────

def _rate(values: list[Any]) -> float | None:
    values = [bool(v) for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def _summarize(values: list[Any], direction: str) -> dict[str, Any]:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"value": None}
    if all(isinstance(v, bool) for v in clean):
        data = [float(v) for v in clean]
        return {"value": _rate(clean), "ci": em.bootstrap_ci(data), "n": len(data)}
    data = [float(v) for v in clean]
    return {"value": round(sum(data) / len(data), 4), "ci": em.bootstrap_ci(data), "n": len(data)}


def build_summary() -> dict[str, Any]:
    setups = [s["name"] for s in comparison()["setups"]]
    finals = {(r["setup"], r["sample_id"]): r for r in final_rows()}
    metrics = read_ndjson(out_dir() / "metrics.ndjson")
    by_path = {m["output_path"]: m for m in metrics}
    attempts = read_ndjson(out_dir() / "attempts.ndjson")
    referee = [r for r in read_ndjson(out_dir() / "referee.ndjson") if "error" not in r]
    runs = {r["setup"]: r for r in read_ndjson(out_dir() / "pipeline_runs.ndjson") if not r.get("limit")}
    attempt_keys = [m.key for m in em.METRICS if m.level == "attempt" and m.key != "gen_latency_s"]
    referees = [r["name"] for r in comparison()["referees"]]
    strong = next((r["name"] for r in comparison()["referees"] if r.get("strong")), None)

    summary: dict[str, Any] = {"run_id": comparison()["run_id"], **provenance(), "setups": {}, "per_category": {},
                               "paired": {}, "referee_agreement": {}}
    for setup in setups:
        rows = [r for (s, _), r in finals.items() if s == setup]
        final_metrics = [by_path[r["output_path"]] for r in rows if r.get("output_path") in by_path]
        gens = [a for a in attempts if a["setup"] == setup and a["kind"] == "generate" and "error" not in a]
        judges = [a for a in attempts if a["setup"] == setup and a["kind"] == "judge" and not a.get("skipped")]
        run = runs.get(setup, {})
        out: dict[str, Any] = {
            "n_subjects": len(rows),
            "accepted": _summarize([r.get("accepted") for r in rows], "higher"),
            "first_attempt_pass": _summarize([r.get("accepted") and r.get("attempts") == 1 for r in rows if "error" not in r], "higher"),
            "attempts": _summarize([r.get("attempts") for r in rows], "lower"),
            "combined_score": _summarize([r.get("combined_score") for r in rows], "higher"),
            "error": _summarize(["error" in r for r in rows], "lower"),
            "subject_latency_s": _summarize([r.get("subject_latency_s") for r in rows], "lower"),
            "gen_latency_s": {"p50": _pct([a["latency_s"] for a in gens if not a.get("includes_model_load")], 0.5),
                              "p95": _pct([a["latency_s"] for a in gens if not a.get("includes_model_load")], 0.95),
                              "first_call_s": next((a["latency_s"] for a in gens if a.get("includes_model_load")), None),
                              "n": len(gens)},
            "own_judge_pass": _summarize([j.get("passed") for j in judges], "higher"),
            "parse_failed": _summarize([j.get("parse_failed") for j in judges], "lower"),
            "judge_latency_s": {"p50": _pct([j["latency_s"] for j in judges if j.get("latency_s") is not None], 0.5),
                                "p95": _pct([j["latency_s"] for j in judges if j.get("latency_s") is not None], 0.95),
                                "n": len(judges)},
            "cost_usd": round(sum(j.get("cost_usd") or 0.0 for j in judges), 5),
            "peak_vram_gb": max((r.get("peak_vram_gb") or 0 for r in rows), default=None),
            "peak_rss_gb": max((r.get("peak_rss_gb") or 0 for r in rows), default=None),
            "sys_ram_gb": run.get("sys_ram_gb"), "swap_gb": run.get("swap_gb"), "gpu_gb": run.get("gpu_gb"),
            "oom_killed": run.get("oom_killed"), "oom_kills": run.get("oom_kills"), "wall_s": run.get("wall_s"),
            "final": {k: _summarize([m.get(k) for m in final_metrics], "") for k in attempt_keys},
            "all_attempts": {k: _summarize([m.get(k) for m in metrics if m["setup"] == setup], "")
                             for k in ("silhouette_iou", "fill_ratio", "clip_top1", "opencv_passed")},
            "referee": {name: _summarize([r["passed"] for r in referee if r["setup"] == setup and r["referee"] == name], "higher")
                        for name in referees},
            "referee_score": {name: _summarize([r["score"] for r in referee if r["setup"] == setup and r["referee"] == name], "higher")
                              for name in referees},
        }
        summary["setups"][setup] = out
        per_cat = defaultdict(list)
        for r in rows:
            per_cat[r["category"]].append(r)
        summary["per_category"][setup] = {
            category: {
                "n": len(cat_rows),
                "accepted": _rate([r.get("accepted") for r in cat_rows]),
                "silhouette_iou": _mean([by_path[r["output_path"]]["silhouette_iou"] for r in cat_rows if r.get("output_path") in by_path]),
                "clip_top1": _rate([by_path[r["output_path"]]["clip_top1"] for r in cat_rows if r.get("output_path") in by_path]),
                "strong_referee_pass": _rate([x["passed"] for x in referee if x["setup"] == setup and x["referee"] == strong
                                              and x["sample_id"] in {r["sample_id"] for r in cat_rows}]),
            }
            for category, cat_rows in sorted(per_cat.items())
        }

    # Paired per-subject comparison of the first two setups.
    if len(setups) >= 2:
        a, b = setups[0], setups[1]
        shared = sorted({sid for (s, sid) in finals if s == a} & {sid for (s, sid) in finals if s == b})
        strong_scores = {(r["setup"], r["sample_id"]): r["score"] for r in referee if r["referee"] == strong}

        def value(setup: str, sid: str, key: str) -> float | None:
            if key == "strong_referee_score":
                return strong_scores.get((setup, sid))
            m = by_path.get(finals[(setup, sid)].get("output_path"))
            return None if m is None or m.get(key) is None else float(m[key])

        for key, higher in (("strong_referee_score", True), ("silhouette_iou", True), ("coverage", True),
                            ("clip_p_true", True), ("fill_ratio", False), ("chamfer_pct", False)):
            wins = losses = ties = 0
            for sid in shared:
                va, vb = value(a, sid, key), value(b, sid, key)
                if va is None or vb is None or abs(va - vb) < 1e-9:
                    ties += 1
                elif (va > vb) == higher:
                    wins += 1
                else:
                    losses += 1
            summary["paired"][key] = {"a": a, "b": b, "a_wins": wins, "b_wins": losses, "ties": ties,
                                      "sign_test_p": em.sign_test_p(wins, losses)}

    # Judge agreement over the same drawings.
    verdicts: dict[str, dict[tuple[str, str], bool]] = defaultdict(dict)
    for r in referee:
        verdicts[r["referee"]][(r["setup"], r["sample_id"])] = bool(r["passed"])
    for i, x in enumerate(referees):
        for y in referees[i + 1:]:
            keys = sorted(set(verdicts[x]) & set(verdicts[y]))
            summary["referee_agreement"][f"{x} vs {y}"] = {
                "kappa": em.cohen_kappa([verdicts[x][k] for k in keys], [verdicts[y][k] for k in keys]),
                "agreement": _rate([verdicts[x][k] == verdicts[y][k] for k in keys]), "n": len(keys)}
    model_path = out_dir() / "draw_time_model.json"
    summary["draw_time_model"] = json.loads(model_path.read_text()) if model_path.exists() else None
    summary["excluded_categories"] = json.loads(evalset_path().read_text()).get("excluded_categories", [])
    return summary


def _fmt(cell: Any) -> str:
    if isinstance(cell, dict):
        value = cell.get("value")
        ci = cell.get("ci")
        if value is None:
            return ""
        return f"{value} [{ci[0]}, {ci[1]}]" if ci and ci[0] is not None else str(value)
    return "" if cell is None else str(cell)


def contact_sheet(summary: dict[str, Any], destination: Path, cell: int = 180) -> None:
    setups = list(summary["setups"])
    finals = {(r["setup"], r["sample_id"]): r for r in final_rows()}
    metrics = {m["output_path"]: m for m in read_ndjson(out_dir() / "metrics.ndjson")}
    strong = next((r["name"] for r in comparison()["referees"] if r.get("strong")), None)
    ref = {(r["setup"], r["sample_id"]): r for r in read_ndjson(out_dir() / "referee.ndjson") if r.get("referee") == strong}
    subjects = list(subjects_by_id().values())
    header = 24
    sheet = Image.new("RGB", (cell * (1 + len(setups)), header + cell * len(subjects)), "white")
    draw = ImageDraw.Draw(sheet)
    for x, title in enumerate(["source"] + setups):
        draw.text((x * cell + 4, 6), title, fill="black")
    for y, subject in enumerate(subjects):
        top = header + y * cell
        photo = Image.open(REPO_ROOT / subject["crop_path"]).convert("RGB")
        photo.thumbnail((cell - 8, cell - 22))
        sheet.paste(photo, (4, top + 18))
        draw.text((4, top + 2), f"{subject['category']} {subject['angle_bucket']}", fill="black")
        for x, setup in enumerate(setups, start=1):
            row = finals.get((setup, subject["sample_id"]))
            if not row or not row.get("output_path"):
                continue
            thumb = Image.open(REPO_ROOT / row["output_path"]).convert("RGB")
            thumb.thumbnail((cell - 8, cell - 22))
            sheet.paste(thumb, (x * cell + 4, top + 18))
            m = metrics.get(row["output_path"], {})
            r = ref.get((setup, subject["sample_id"]), {})
            label = f"{'ok' if row.get('accepted') else 'rej'} a{row.get('attempts')} IoU {m.get('silhouette_iou')} ref {r.get('score')}"
            draw.text((x * cell + 4, top + 2), label, fill="black")
    sheet.save(destination)


# ── clean comparison tables (report + issue) ───────────────────────────────

RATE_KEYS = {"accepted", "first_attempt_pass", "error", "own_judge_pass", "parse_failed", "opencv_passed", "blank",
             "angle_bucket_match", "direction_match", "clip_top1", "trajectory_ok", "within_budget"}
SHORT = {"omnigen2-openrouter": "OmniGen2 + OpenRouter judge", "flux-ollama": "FLUX + local judge"}


def _num(value: float, key: str) -> str:
    if key in RATE_KEYS:
        return f"{100 * value:.0f}%"
    if key == "cost_usd":
        return f"${value:.4f}" if value else "$0"
    if key == "oom_kills":
        return str(int(value))
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _cell(stat: Any, key: str) -> tuple[str, float | None, tuple[float, float] | None]:
    """Display text, point value and CI for a summary cell."""
    if stat is None:
        return "–", None, None
    if not isinstance(stat, dict):
        return _num(float(stat), key) if isinstance(stat, (int, float)) and not isinstance(stat, bool) else str(stat), \
            (float(stat) if isinstance(stat, (int, float)) else None), None
    value, ci = stat.get("value"), stat.get("ci")
    if value is None:
        return "–", None, None
    text = _num(value, key)
    if ci and ci[0] is not None:
        text += f" ({_num(ci[0], key)}–{_num(ci[1], key)})"
        return text, float(value), (float(ci[0]), float(ci[1]))
    return text, float(value), None


def _winner(direction: str, key: str, cells: list[tuple[str, float | None, tuple[float, float] | None]], names: list[str]) -> str:
    values = [c[1] for c in cells]
    if direction not in ("higher", "lower", "target") or any(v is None for v in values) or len(values) != 2:
        return ""
    if direction == "target":
        target = 0.16 if key == "foreground_ratio" else None
        if target is None:
            return ""
        scores = [-abs(v - target) for v in values]
    else:
        scores = values if direction == "higher" else [-v for v in values]
    if abs(scores[0] - scores[1]) < 1e-9:
        return "tie"
    best = 0 if scores[0] > scores[1] else 1
    cis = [c[2] for c in cells]
    if all(cis) and cis[0][0] <= cis[1][1] and cis[1][0] <= cis[0][1]:
        return f"{names[best]} (CIs overlap)"
    return f"**{names[best]}**"


def clean_tables(summary: dict[str, Any]) -> str:
    setups = list(summary["setups"])
    names = [SHORT.get(s, s).split(" + ")[0] for s in setups]
    registry = {m.key: m for m in em.METRICS}
    arrows = {"higher": "↑", "lower": "↓", "target": "◎", "info": ""}
    targets = {"foreground_ratio": "◎ 0.16", "stroke_count": "◎ 1–6"}
    referees = comparison()["referees"]
    strong = next((r["name"] for r in referees if r.get("strong")), None)
    rows: list[list[str]] = []

    def add(label: str, direction: str, key: str, stats: list[Any]) -> None:
        cells = [_cell(stat, key) for stat in stats]
        rows.append([label, targets.get(key, arrows.get(direction, direction)), *[c[0] for c in cells],
                     _winner(direction, key, cells, names)])

    def group(title: str) -> None:
        rows.append([f"**{title}**", "", *[""] * len(setups), ""])

    data = [summary["setups"][s] for s in setups]
    group("A. Pipeline outcome")
    for key in ("accepted", "first_attempt_pass", "attempts", "combined_score", "error"):
        add(registry[key].name, registry[key].direction, key, [d[key] for d in data])
    group("F. Judges")
    judge_of = {s["name"]: s["validator"].get("openrouter_model_name") or s["validator"].get("ollama_model_name")
                for s in comparison()["setups"]}
    rows.append(["Own judge model", "", *[f"`{judge_of[s]}`" for s in setups], ""])
    for key in ("own_judge_pass", "parse_failed"):
        add(registry[key].name, registry[key].direction, key, [d[key] for d in data])
    add("Own-judge latency p50 (s)", "lower", "judge_p50", [d["judge_latency_s"]["p50"] for d in data])
    for referee in referees:
        label = f"Referee pass: `{referee['name']}`" + (" (strong)" if referee.get("strong") else "")
        add(label, "higher", "accepted", [d["referee"][referee["name"]] for d in data])
    agreement = summary["referee_agreement"]
    own_kappa = []
    ref_names = {r.get("openrouter_model_name") or r.get("ollama_model_name"): r["name"] for r in referees}
    for setup in setups:
        own = ref_names.get(judge_of[setup])
        pair = next((v for k, v in agreement.items() if own and own in k and strong in k), None)
        own_kappa.append(pair["kappa"] if pair else None)
    add("Own judge vs strong referee (Cohen's κ)", "higher", "kappa", own_kappa)
    for letter in ("B", "C", "D", "E", "G"):
        group(f"{letter}. {em.GROUPS[letter]}")
        for key, metric in registry.items():
            if metric.group == letter and key in data[0]["final"]:
                add(metric.name, metric.direction, key, [d["final"][key] for d in data])
    group("H. Cost and resources")
    add("Time per subject (s)", "lower", "subject_latency_s", [d["subject_latency_s"] for d in data])
    gen = []
    for setup in setups:
        calls = [a["latency_s"] for a in read_ndjson(out_dir() / "attempts.ndjson")
                 if a["setup"] == setup and a["kind"] == "generate" and "error" not in a]
        warm = summary["setups"][setup]["gen_latency_s"]["p50"]
        gen.append(f"{warm:.1f}" if warm is not None else f"{_pct(calls, 0.5):.1f} (incl. reload)")
    rows.append(["Generation p50 per attempt (s)", "↓", *gen, ""])
    add("Wall time, 20 subjects (min)", "lower", "wall", [round(d["wall_s"] / 60, 1) if d.get("wall_s") else None for d in data])
    for key, label in (("peak_vram_gb", "Peak VRAM, generation process (GB)"), ("sys_ram_gb", "Peak system RAM (GB)"),
                       ("swap_gb", "Peak swap (GB, of ~6.3)"), ("oom_kills", "OOM kills"), ("cost_usd", "API cost (USD)")):
        add(label, "lower", key, [d.get(key) for d in data])

    header = ["Metric", "Better", *[SHORT.get(s, s) for s in setups], "Better setup"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    out = ["Final drawing per subject, n = 20 per setup. Mean or rate, with bootstrap 95% CI in brackets. "
           "**Bold** = better and CIs don't overlap; \"(CIs overlap)\" = direction only.", "", "\n".join(lines)]

    out += ["", "**Per category** (A = " + names[0] + ", B = " + names[1] + ")", ""]
    cat_header = ["Category", "n", "Accepted A / B", f"Strong-referee pass A / B", "BioCLIP top-1 A / B", "Silhouette IoU A / B"]
    cat_lines = ["| " + " | ".join(cat_header) + " |", "|" + "---|" * len(cat_header)]
    per = summary["per_category"]
    for category in per[setups[0]]:
        a, b = per[setups[0]][category], per[setups[1]][category]

        def pair(key: str, rate: bool = True) -> str:
            fmt = (lambda v: "–" if v is None else f"{100 * v:.0f}%") if rate else (lambda v: "–" if v is None else f"{v:.2f}")
            return f"{fmt(a.get(key))} / {fmt(b.get(key))}"

        cat_lines.append(f"| {category} | {a['n']} | {pair('accepted')} | {pair('strong_referee_pass')} | "
                         f"{pair('clip_top1')} | {pair('silhouette_iou', rate=False)} |")
    out.append("\n".join(cat_lines))

    out += ["", "**Head-to-head per subject** (which setup's final drawing was better on each subject)", ""]
    labels = {"strong_referee_score": "Strong-referee score", "silhouette_iou": "Silhouette IoU",
              "coverage": "Trajectory ink coverage", "clip_p_true": "BioCLIP p(true category)",
              "fill_ratio": "Solid-fill ratio (lower wins)", "chamfer_pct": "Chamfer distance (lower wins)"}
    h_lines = ["| Metric | " + names[0] + " wins | " + names[1] + " wins | Ties | Sign-test p |", "|---|---|---|---|---|"]
    for key, v in summary["paired"].items():
        h_lines.append(f"| {labels.get(key, key)} | {v['a_wins']} | {v['b_wins']} | {v['ties']} | {v['sign_test_p']} |")
    out.append("\n".join(h_lines))
    return "\n".join(out)


def cmd_report(args: argparse.Namespace) -> None:
    summary = build_summary()
    out = out_dir()
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    contact_sheet(summary, out / "contact_sheet.png")
    setups = list(summary["setups"])
    registry = {m.key: m for m in em.METRICS}

    def table(keys: list[str], source: str) -> str:
        rows = []
        for key in keys:
            row = {"metric": registry[key].name if key in registry else key,
                   "better": {"higher": "↑", "lower": "↓", "target": "◎", "info": ""}.get(registry[key].direction, "") if key in registry else ""}
            for setup in setups:
                data = summary["setups"][setup]
                row[setup] = _fmt(data[source][key] if source else data.get(key))
            rows.append(row)
        return md_table(rows, ["metric", "better", *setups])

    final_keys = list(summary["setups"][setups[0]]["final"]) if setups else []
    outcome = ["accepted", "first_attempt_pass", "attempts", "combined_score", "error", "subject_latency_s",
               "own_judge_pass", "parse_failed", "cost_usd", "peak_vram_gb", "sys_ram_gb"]
    md = [
        f"# Setup comparison `{summary['run_id']}`",
        f"git `{summary['git_sha']}`, config hash `{summary['config_hash']}`. Metric definitions: "
        "`docs/benchmarks/eval_suite_r2.docx`. Values are means or rates over final drawings, with bootstrap 95% CI.",
        f"Excluded categories (no source masks): {', '.join(summary['excluded_categories']) or 'none'}.",
    ]
    findings = REPO_ROOT / "docs" / "benchmarks" / f"{summary['run_id']}_findings.md"
    if findings.exists():
        md += ["\n## Key findings", findings.read_text().strip()]
    md += ["\n## Results", clean_tables(summary)]
    md.append(f"\nContact sheet: `outputs/benchmarks/{summary['run_id']}/contact_sheet.png`.")
    text = "\n".join(md) + "\n"
    (out / "summary.md").write_text(text)
    (REPO_ROOT / "docs" / "benchmarks" / f"{summary['run_id']}.md").write_text(text)
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build-evalset")
    for name in ("pipeline", "_pipeline-one"):
        p = sub.add_parser(name)
        p.add_argument("--setup", required=True)
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--resume", action="store_true", help="Skip subjects this setup already finished.")
    sub.add_parser("evaluate")
    referee = sub.add_parser("referee")
    referee.add_argument("--referees", nargs="*")
    sub.add_parser("report")
    args = parser.parse_args()
    {
        "build-evalset": cmd_build_evalset, "pipeline": cmd_pipeline, "_pipeline-one": cmd_pipeline_one,
        "evaluate": cmd_evaluate, "referee": cmd_referee, "report": cmd_report,
    }[args.command](args)


if __name__ == "__main__":
    main()
