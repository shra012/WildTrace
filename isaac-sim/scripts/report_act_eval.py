"""Summarize closed-loop evaluation runs written by isaac_batch.py eval into a Markdown report.

    python3 isaac-sim/scripts/report_act_eval.py --baseline ik --tags ik act_kin act_kin_phys

Per tag: finish rate, desired-to-executed nearest-path RMSE (the baseline-v1
headline metric), p95 and max error, heading RMSE, slew-clamped command
fraction and simulated draw time. A paired table compares each tag with the
baseline on drawings both finished.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(REPO_ROOT / "outputs" / "act_eval"))
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--baseline", default="ik")
    parser.add_argument("--out", default=str(REPO_ROOT / "outputs" / "act_eval" / "act_v1_results.md"))
    return parser.parse_args()


def load_runs(root: Path, tag: str) -> dict[str, dict]:
    runs = {}
    for path in sorted((root / tag).glob("*/*/run_metrics.json")):
        metrics = json.loads(path.read_text())
        path_error = metrics.get("desired_to_executed_nearest_path_error", {})
        runs[path.parent.name] = {
            "category": path.parent.parent.name,
            "finished": metrics.get("status") == "finished",
            "rmse_mm": _mm(path_error.get("rmse_m")),
            "p95_mm": _mm(path_error.get("p95_error_m")),
            "max_mm": _mm(path_error.get("max_error_m")),
            "heading_rmse_deg": metrics.get("heading_error", {}).get("rmse_deg"),
            "clamped_fraction": metrics.get("clamped_command_fraction"),
            "corner_mm": _mm(metrics.get("corner_tracking_error", {}).get("rmse_m")),
            "jerk_rms": metrics.get("pen_down_joint_jerk_rad_s3", {}).get("rms"),
            "sim_s": metrics.get("simulation_duration_s"),
            "failure": metrics.get("failure_reason", ""),
        }
    return runs


def _mm(value):
    return None if value is None else float(value) * 1e3


def _mean(values) -> str:
    values = [v for v in values if v is not None]
    return f"{np.mean(values):.3f}" if values else "-"


def summary_row(tag: str, runs: dict[str, dict]) -> list[str]:
    finished = [r for r in runs.values() if r["finished"]]
    return [
        tag,
        f"{len(finished)}/{len(runs)}",
        _mean(r["rmse_mm"] for r in finished),
        _mean(r["p95_mm"] for r in finished),
        _mean(r["max_mm"] for r in finished),
        _mean(r["heading_rmse_deg"] for r in finished),
        _mean(r["corner_mm"] for r in finished),
        _mean(r["jerk_rms"] for r in finished),
        _mean(r["sim_s"] for r in finished),
    ]


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def build_report(root: Path, tags: list[str], baseline: str) -> str:
    runs = {tag: load_runs(root, tag) for tag in tags}
    sections = ["# ACT closed-loop evaluation (gold test set, 1 g)", ""]
    sections.append(_table(
        ["Controller", "Finished", "RMSE mm", "p95 mm", "max mm", "Heading RMSE deg", "Corner RMSE mm",
         "Jerk RMS rad/s3", "Draw time s"],
        [summary_row(tag, runs[tag]) for tag in tags],
    ))
    if baseline in runs:
        paired = []
        for tag in tags:
            if tag == baseline:
                continue
            common = [k for k in runs[tag] if k in runs[baseline] and runs[tag][k]["finished"] and runs[baseline][k]["finished"]]
            deltas = [runs[tag][k]["rmse_mm"] - runs[baseline][k]["rmse_mm"] for k in common
                      if runs[tag][k]["rmse_mm"] is not None and runs[baseline][k]["rmse_mm"] is not None]
            wins = sum(d < 0 for d in deltas)
            paired.append([tag, str(len(deltas)), _mean(deltas), f"{wins}/{len(deltas)}"])
        sections += ["", f"## Paired against `{baseline}` (drawings both finished)", ""]
        sections.append(_table(["Controller", "Pairs", "Mean RMSE delta mm", "Lower RMSE"], paired))
    sections += ["", "## Per category (finished / RMSE mm)", ""]
    categories = sorted({r["category"] for tag_runs in runs.values() for r in tag_runs.values()})
    rows = []
    for category in categories:
        row = [category]
        for tag in tags:
            items = [r for r in runs[tag].values() if r["category"] == category]
            done = [r for r in items if r["finished"]]
            row.append(f"{len(done)}/{len(items)} · {_mean(r['rmse_mm'] for r in done)}")
        rows.append(row)
    sections.append(_table(["Category", *tags], rows))
    failures = defaultdict(list)
    for tag in tags:
        for name, run in runs[tag].items():
            if not run["finished"]:
                failures[tag].append(f"{name}: {run['failure'] or 'no metrics'}")
    if failures:
        sections += ["", "## Failures", ""]
        for tag, items in failures.items():
            sections.append(f"**{tag}**")
            sections += [f"- {item}" for item in items]
    return "\n".join(sections) + "\n"


def main() -> int:
    args = _parse_args()
    report = build_report(Path(args.root), args.tags, args.baseline)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
