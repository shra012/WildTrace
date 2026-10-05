#!/usr/bin/env python3
"""Compare two trajectory generators using the same Isaac Sim run metrics.

The physical tracking metrics are each model's desired path versus the pen path
executed by the same controller.  The geometry metrics separately compare the
candidate's source trajectory against the reference source trajectory.  Keeping
those two questions separate avoids calling a well-tracked wrong drawing good.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "isaac-sim" / "src"))

from trajectory_loader import load_trajectory  # noqa: E402
from wildtrace.trajectory_candidates import symmetric_raster_distance  # noqa: E402


def _read_json(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in {path}")
    return payload


def _error_mm(metrics: Mapping[str, Any], name: str) -> dict[str, float | None]:
    values = metrics.get(name, {})
    if not isinstance(values, Mapping):
        return {"rmse_mm": None, "max_mm": None, "p95_mm": None}
    return {
        "rmse_mm": _metres_to_mm(values.get("rmse_m")),
        "max_mm": _metres_to_mm(values.get("max_error_m")),
        "p95_mm": _metres_to_mm(values.get("p95_error_m")),
    }


def _metres_to_mm(value: Any) -> float | None:
    return round(float(value) * 1000.0, 3) if value is not None else None


def _run_summary(label: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "label": label,
        "status": metrics.get("status", "unknown"),
        "drawing_id": metrics.get("drawing_id"),
        "stroke_count": metrics.get("stroke_count"),
        "pen_down_points": metrics.get("desired_pen_down_points"),
        "draw_time_s": metrics.get("simulation_duration_s"),
        "config_hash": metrics.get("config_hash"),
        "git_sha": metrics.get("git_sha"),
        "desired_to_executed": _error_mm(metrics, "desired_to_executed_nearest_path_error"),
        "executed_to_desired": _error_mm(metrics, "executed_to_desired_nearest_path_error"),
        "failure_reason": metrics.get("failure_reason"),
    }


def _format(value: Any, precision: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def _markdown(report: Mapping[str, Any]) -> str:
    reference, candidate = report["runs"]
    lines = [
        "# Trajectory-generator comparison",
        "",
        "Physical tracking uses the same Isaac controller's `desired_to_executed_nearest_path_error`. "
        "Source-geometry similarity is separate: it compares the candidate trajectory with the reference trajectory before simulation.",
        "",
        "| Generator | Status | Strokes | Pen-down pts | Tracking RMSE (mm) | Tracking max (mm) | Draw time (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in (reference, candidate):
        tracking = run["desired_to_executed"]
        lines.append(
            f"| {run['label']} | {run['status']} | {_format(run['stroke_count'])} | "
            f"{_format(run['pen_down_points'])} | {_format(tracking['rmse_mm'])} | "
            f"{_format(tracking['max_mm'])} | {_format(run['draw_time_s'], 1)} |"
        )
    geometry = report["source_geometry"]
    lines += [
        "",
        "## Candidate source drawing vs. reference source drawing",
        "",
        f"- Symmetric raster distance: {_format(geometry['symmetric_distance_px'])} px (lower is better)",
        f"- Pixel IoU: {_format(geometry['pixel_iou'])} (higher is better)",
        f"- Candidate → reference distance: {_format(geometry['candidate_to_reference_px'])} px",
        f"- Reference → candidate distance: {_format(geometry['reference_to_candidate_px'])} px",
        "",
        f"Comparable controller configuration: **{report['comparable_controller_config']}**.",
    ]
    if not report["both_finished"]:
        lines += ["", "**Do not report an accuracy improvement:** at least one Isaac run did not finish."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-label", default="Flux baseline")
    parser.add_argument("--candidate-label", default="Molmo candidate")
    parser.add_argument("--reference-trajectory", required=True)
    parser.add_argument("--candidate-trajectory", required=True)
    parser.add_argument("--reference-run", required=True, help="Controller *_run_metrics.json for reference")
    parser.add_argument("--candidate-run", required=True, help="Controller *_run_metrics.json for candidate")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--raster-size", type=int, default=512)
    args = parser.parse_args()

    reference_trajectory = load_trajectory(args.reference_trajectory)
    candidate_trajectory = load_trajectory(args.candidate_trajectory)
    reference_metrics = _read_json(args.reference_run)
    candidate_metrics = _read_json(args.candidate_run)
    source_geometry = symmetric_raster_distance(reference_trajectory, candidate_trajectory, size=args.raster_size)
    reference = _run_summary(args.reference_label, reference_metrics)
    candidate = _run_summary(args.candidate_label, candidate_metrics)
    hashes = {reference["config_hash"], candidate["config_hash"]}
    report = {
        "reference_trajectory": str(Path(args.reference_trajectory)),
        "candidate_trajectory": str(Path(args.candidate_trajectory)),
        "reference_run": str(Path(args.reference_run)),
        "candidate_run": str(Path(args.candidate_run)),
        "runs": [reference, candidate],
        "source_geometry": source_geometry,
        "comparable_controller_config": len(hashes) == 1 and None not in hashes,
        "both_finished": reference["status"] == "finished" and candidate["status"] == "finished",
    }
    for destination, content in (
        (Path(args.output_json), json.dumps(report, indent=2) + "\n"),
        (Path(args.output_markdown), _markdown(report)),
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    print(f"[OK] comparison JSON: {args.output_json}")
    print(f"[OK] comparison table: {args.output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
