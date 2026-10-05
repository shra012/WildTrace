#!/usr/bin/env python3
"""Build an apples-to-apples VLM trajectory benchmark table.

The script consumes one or more model reports with a ``records`` list.  Every
record must at minimum contain a ``status`` and may include geometry, semantic,
GPU-memory, and Isaac metrics after a valid trajectory is generated.  Missing
measurements stay ``not measured``; they are never converted to zero.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterable


NOT_MEASURED = "not measured"
NOT_APPLICABLE = "not applicable"


def _percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percent / 100.0
    lower, upper = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def _value_or_na(value: float | int | str | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _numbers(records: list[dict[str, Any]], *keys: str) -> list[float]:
    values: list[float] = []
    for record in records:
        current: Any = record
        for key in keys:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        if isinstance(current, (int, float)) and not isinstance(current, bool):
            values.append(float(current))
    return values


def summarize(label: str, report: dict[str, Any]) -> dict[str, Any]:
    records = list(report.get("records", []))
    valid = [record for record in records if record.get("status") == "valid"]
    requested = int(report.get("images_per_category_requested", 0)) * len(report.get("categories", []))
    attempted = len(records)
    geometry_distance = _numbers(valid, "geometry", "symmetric_mean_distance_px")
    geometry_iou = _numbers(valid, "geometry", "pixel_iou")
    judge_scores = _numbers(valid, "judge", "score")
    isaac_rmse_mm = _numbers(valid, "isaac", "desired_to_executed_rmse_mm")
    gpu_memory_mib = _numbers(records, "gpu_peak_memory_mib")
    return {
        "label": label,
        "model": report.get("model", label),
        "requested_subjects": requested or None,
        "attempted_subjects": attempted,
        "valid_trajectories": len(valid),
        "invalid_outputs": attempted - len(valid),
        "subject_coverage_pct": (100.0 * attempted / requested) if requested else None,
        "valid_rate_pct": (100.0 * len(valid) / attempted) if attempted else None,
        "mean_latency_s": _mean(_numbers(records, "latency_s")),
        "p50_latency_s": _percentile(_numbers(records, "latency_s"), 50),
        "p95_latency_s": _percentile(_numbers(records, "latency_s"), 95),
        "trajectory_distance_px": _mean(geometry_distance),
        "trajectory_pixel_iou": _mean(geometry_iou),
        "judge_score": _mean(judge_scores),
        "isaac_tracking_rmse_mm": _mean(isaac_rmse_mm),
        "gpu_peak_memory_mib": max(gpu_memory_mib) if gpu_memory_mib else None,
        "geometry_status": "measured" if geometry_distance else (NOT_APPLICABLE if not valid else NOT_MEASURED),
        "isaac_status": "measured" if isaac_rmse_mm else (NOT_APPLICABLE if not valid else NOT_MEASURED),
        "report_path": report.get("_path"),
    }


def render_markdown(rows: list[dict[str, Any]], shared_subject_set: str | None) -> str:
    lines = [
        "# VLM trajectory benchmark",
        "",
        "A candidate receives source-geometry and Isaac tracking scores only after it emits a valid canonical trajectory. "
        "This prevents invalid-output failures from being misreported as zero geometry error.",
        "",
    ]
    if shared_subject_set:
        lines += [f"Shared-subject manifest: `{shared_subject_set}`.", ""]
    lines += [
        "| Model | Subjects attempted / requested | Coverage | Valid / invalid outputs | Valid rate | Mean / p50 / p95 latency (s) | Source distance (px) ↓ | Pixel IoU ↑ | Judge score ↑ | Isaac RMSE (mm) ↓ | GPU peak (MiB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        attempted = f"{row['attempted_subjects']}/{_value_or_na(row['requested_subjects'], 0)}"
        latency = " / ".join(_value_or_na(row[key]) for key in ("mean_latency_s", "p50_latency_s", "p95_latency_s"))
        lines.append(
            f"| {row['label']} | {attempted} | {_value_or_na(row['subject_coverage_pct'])}% | "
            f"{row['valid_trajectories']} / {row['invalid_outputs']} | "
            f"{_value_or_na(row['valid_rate_pct'])}% | {latency} | "
            f"{_value_or_na(row['trajectory_distance_px'])} | {_value_or_na(row['trajectory_pixel_iou'])} | "
            f"{_value_or_na(row['judge_score'])} | {_value_or_na(row['isaac_tracking_rmse_mm'])} | "
            f"{_value_or_na(row['gpu_peak_memory_mib'], 0)} |"
        )
    lines += [
        "",
        "## Measurement status",
        "",
    ]
    for row in rows:
        lines.append(
            f"- **{row['label']}** — source geometry: {row['geometry_status']}; Isaac tracking: {row['isaac_status']}."
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Repeat for each model, e.g. Molmo=outputs/molmo_zeroshot/reports/zeroshot_report.json",
    )
    parser.add_argument("--shared-subject-manifest", default=None)
    parser.add_argument(
        "--output-subject-manifest",
        default=None,
        help="Write the first report's attempted subject IDs for reuse by Flux and SANA.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    first_report: dict[str, Any] | None = None
    for value in args.report:
        if "=" not in value:
            parser.error(f"--report must be LABEL=PATH, got {value!r}")
        label, path_text = value.split("=", 1)
        path = Path(path_text)
        report = json.loads(path.read_text(encoding="utf-8"))
        report["_path"] = str(path)
        if first_report is None:
            first_report = report
        rows.append(summarize(label, report))
    shared_subject_manifest = args.shared_subject_manifest
    if args.output_subject_manifest:
        if first_report is None:
            parser.error("No report available to create a subject manifest")
        subject_path = Path(args.output_subject_manifest)
        subjects = [
            {key: record.get(key) for key in ("category", "sample_id", "crop_path")}
            for record in first_report.get("records", [])
        ]
        subject_path.parent.mkdir(parents=True, exist_ok=True)
        subject_path.write_text(json.dumps({"subjects": subjects}, indent=2) + "\n", encoding="utf-8")
        shared_subject_manifest = str(subject_path)
    payload = {"rows": rows, "shared_subject_manifest": shared_subject_manifest}
    output_json, output_markdown = Path(args.output_json), Path(args.output_markdown)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    output_markdown.write_text(render_markdown(rows, shared_subject_manifest), encoding="utf-8")
    print(f"[OK] {output_json}")
    print(f"[OK] {output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
