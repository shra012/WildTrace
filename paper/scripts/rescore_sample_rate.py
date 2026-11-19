"""Re-score twin runs as if the pen tip were logged at a lower rate (the real arm logs at about 10 Hz).

The paper's path RMSE is the nearest-sample error from each planned pen-down point to the executed samples, so a
sparser log raises it even when the motion is unchanged. This script keeps the first executed pen-down sample in
each 1/rate window and recomputes the RMSE, reporting the ratio to the full-rate value.

Usage (from the repo root):
    uv run python paper/scripts/rescore_sample_rate.py outputs/paper_eval/paper_act outputs/paper_eval/paper_ik
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def rescore(run_dir: Path, rate_hz: float) -> tuple[float, float, float] | None:
    metrics = json.loads((run_dir / "run_metrics.json").read_text())
    if metrics["status"] != "finished":
        return None
    with open(run_dir / "desired_path.csv") as fh:
        desired = np.array(
            [[float(r["x_m"]), float(r["y_m"]), float(r["z_m"])] for r in csv.DictReader(fh) if r["pen_down"] == "1"]
        )
    with open(run_dir / "executed_path.csv") as fh:
        rows = [r for r in csv.DictReader(fh) if r["pen_down"] == "1"]
    t = np.array([float(r["simulation_time_s"]) for r in rows])
    tip = np.array([[float(r["tip_x_m"]), float(r["tip_y_m"]), float(r["tip_z_m"])] for r in rows])

    def rmse_mm(samples: np.ndarray) -> float:
        return float(np.sqrt(np.mean(cKDTree(samples).query(desired)[0] ** 2)) * 1e3)

    keep = np.unique(np.floor(t * rate_hz), return_index=True)[1]
    sub = tip[keep]
    spacing_mm = float(np.median(np.linalg.norm(np.diff(sub, axis=0), axis=1)) * 1e3)
    return rmse_mm(tip), rmse_mm(sub), spacing_mm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("batch_dirs", nargs="+", type=Path, help="batch output dirs, e.g. outputs/paper_eval/paper_act")
    parser.add_argument("--rate-hz", type=float, default=10.0)
    args = parser.parse_args()
    for batch in args.batch_dirs:
        results = [r for d in sorted(batch.glob("*/*/")) if (r := rescore(d, args.rate_hz))]
        full, sub, spacing = (np.array(x) for x in zip(*results))
        print(
            f"{batch.name}: {len(results)} runs, full-rate RMSE {full.mean():.3f} mm, "
            f"{args.rate_hz:g} Hz RMSE {sub.mean():.3f} mm, ratio {sub.mean() / full.mean():.2f}, "
            f"median sample spacing {np.median(spacing):.2f} mm"
        )


if __name__ == "__main__":
    main()
