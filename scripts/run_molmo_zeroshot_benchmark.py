#!/usr/bin/env python3
"""Evaluate zero-shot Molmo stroke plans on Open Images subject crops.

This deliberately does not invoke Flux, SVG extraction, or Isaac Sim.  It
measures whether Molmo can produce a safe canonical trajectory at all before
we make claims about drawing quality or spend time on robot execution.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

from PIL import Image

from generate_molmo_stroke_plan import MolmoStrokePlanner
from wildtrace.bronze_stage import curate_bronze
from wildtrace.config import load_runtime_config
from wildtrace.io_utils import read_ndjson, resolve_repo_path
from wildtrace.pipeline import fetch_openimages
from wildtrace.silver_stage import enrich_and_crop_subjects, normalize_to_silver
from wildtrace.stage_io import silver_subjects_manifest_path
from wildtrace.trajectory_candidates import StrokePlanError, compile_stroke_plan, extract_json_plan


def _runtime(repo_root: Path, categories: list[str], source_limit: int) -> dict:
    runtime = deepcopy(load_runtime_config(repo_root))
    runtime["datasets"]["categories"] = [{"name": name, "limit": source_limit} for name in categories]
    runtime["datasets"]["fetch"]["official"]["splits"] = ["train", "validation"]
    # Cropping does not need BioCLIP; it would compete for GPU memory with Molmo.
    runtime["models"]["enrichment"]["backend"] = "heuristic_enrichment"
    root = "outputs/molmo_zeroshot"
    storage = runtime["storage"]
    storage["raw_root"] = root
    storage["processed_root"] = root
    storage["artifacts_root"] = root
    storage["bronze"] = {
        "images_dir": f"{root}/bronze/images",
        "masks_dir": f"{root}/bronze/masks",
        "metadata_dir": f"{root}/bronze/metadata",
        "logs_dir": f"{root}/bronze/logs",
        "manifests_dir": f"{root}/bronze/manifests",
    }
    storage["silver"] = {
        "images_dir": f"{root}/silver/images",
        "masks_dir": f"{root}/silver/masks",
        "isolated_dir": f"{root}/silver/isolated_subjects",
        "crops_dir": f"{root}/silver/crops",
        "diagrams_dir": f"{root}/silver/unused_diagrams",
        "checkpoints_dir": f"{root}/silver/checkpoints",
    }
    storage["gold"] = {
        "diagrams_dir": f"{root}/gold/unused_diagrams",
        "outlines_dir": f"{root}/gold/unused_outlines",
        "svg_dir": f"{root}/gold/unused_svg",
        "trajectories_dir": f"{root}/trajectories",
        "records_dir": f"{root}/gold/records",
    }
    storage["reports_dir"] = f"{root}/reports"
    runtime["datasets"]["fetch"]["manifest_path"] = f"{root}/bronze/manifests/openimages_fetch.ndjson"
    runtime["datasets"]["fetch"]["latest_view_path"] = f"{root}/bronze/manifests/openimages_fetch_latest.ndjson"
    return runtime


def _subjects(repo_root: Path, runtime: dict, categories: list[str], target: int) -> list[dict]:
    rows = [
        row
        for row in read_ndjson(silver_subjects_manifest_path(repo_root, runtime))
        if row.get("category") in categories and row.get("segmentation_status") == "accepted"
    ]
    grouped: dict[str, list[dict]] = {category: [] for category in categories}
    for row in rows:
        grouped[row["category"]].append(row)
    selected: list[dict] = []
    for category in categories:
        selected.extend(sorted(grouped[category], key=lambda row: row["sample_id"])[:target])
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", action="append", default=None, help="Repeat or comma-separate; default is all configured categories.")
    parser.add_argument("--images-per-category", type=int, default=5)
    parser.add_argument("--source-limit", type=int, default=20)
    parser.add_argument("--model", default="allenai/MolmoE-1B-0924")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.images_per_category < 1 or args.source_limit < args.images_per_category:
        parser.error("source-limit and images-per-category must be positive, with source-limit >= images-per-category")

    repo_root = Path(__file__).resolve().parents[1]
    configured = [entry["name"] for entry in load_runtime_config(repo_root)["datasets"]["categories"]]
    requested = args.category or configured
    categories = list(dict.fromkeys(name.strip() for value in requested for name in value.split(",") if name.strip()))
    unknown = sorted(set(categories) - set(configured))
    if unknown:
        parser.error(f"Unknown configured categories: {', '.join(unknown)}")
    runtime = _runtime(repo_root, categories, args.source_limit)

    for name, stage in (("fetch", fetch_openimages), ("curate", curate_bronze), ("normalize", normalize_to_silver), ("crop", enrich_and_crop_subjects)):
        rows = stage(repo_root, runtime)
        print(f"[MOLMO ZERO-SHOT] {name}: {len(rows)} rows", flush=True)
    subjects = _subjects(repo_root, runtime, categories, args.images_per_category)
    subject_counts = Counter(row["category"] for row in subjects)
    if any(subject_counts[category] < args.images_per_category for category in categories):
        print(f"[MOLMO ZERO-SHOT] insufficient accepted crops: {dict(subject_counts)}", flush=True)

    output_root = repo_root / "outputs" / "molmo_zeroshot"
    raw_root, trajectory_root = output_root / "raw", output_root / "trajectories"
    planner = MolmoStrokePlanner(args.model, load_in_4bit=True, local_files_only=args.local_files_only)
    records: list[dict] = []
    for index, subject in enumerate(subjects, start=1):
        category, sample_id = subject["category"], subject["sample_id"]
        raw_path = raw_root / category / f"{sample_id}.txt"
        trajectory_path = trajectory_root / category / f"{sample_id}.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        record = {"category": category, "sample_id": sample_id, "crop_path": subject["crop_path"], "raw_output_path": str(raw_path.relative_to(repo_root))}
        try:
            image = Image.open(resolve_repo_path(repo_root, subject["crop_path"])).convert("RGB")
            raw = planner.generate_text(image, args.max_new_tokens)
            raw_path.write_text(raw, encoding="utf-8")
            plan = extract_json_plan(raw)
            trajectory = compile_stroke_plan(plan, drawing_id=sample_id, max_control_points_per_stroke=24)
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(json.dumps(trajectory, indent=2) + "\n", encoding="utf-8")
            record.update({"status": "valid", "trajectory_path": str(trajectory_path.relative_to(repo_root)), "stroke_count": len(trajectory["strokes"]), "point_count": sum(len(stroke["points"]) for stroke in trajectory["strokes"])})
        except (StrokePlanError, ValueError, OSError, RuntimeError) as exc:
            record.update({"status": "invalid", "error": f"{type(exc).__name__}: {exc}"})
        record["latency_s"] = round(time.perf_counter() - started, 3)
        records.append(record)
        print(f"[MOLMO ZERO-SHOT] {index}/{len(subjects)} {category}/{sample_id}: {record['status']} ({record['latency_s']} s)", flush=True)

    valid_counts = Counter(row["category"] for row in records if row["status"] == "valid")
    report = {
        "model": args.model,
        "zero_shot": True,
        "uses_flux": False,
        "images_per_category_requested": args.images_per_category,
        "source_limit_per_category": args.source_limit,
        "categories": categories,
        "input_counts": {category: subject_counts[category] for category in categories},
        "valid_trajectory_counts": {category: valid_counts[category] for category in categories},
        "valid_rate": (sum(row["status"] == "valid" for row in records) / len(records)) if records else 0.0,
        "records": records,
    }
    report_path = output_root / "reports" / "zeroshot_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[MOLMO ZERO-SHOT] report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
