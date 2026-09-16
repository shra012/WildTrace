"""Local (non-Isaac) helper: plot desired-vs-executed pen path for a finished
MCP drawing run, for a distortion-free top-down comparison against the
source diagram (the in-sim RTX capture is an oblique 3D view and not
suitable for judging silhouette fidelity)."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from coordinate_mapper import map_trajectory_to_plane
from trajectory_loader import load_trajectory

SURFACE_CENTER_XY_M = [0.45, 0.0]
SURFACE_SIZE_XY_M = [0.16, 0.12]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--executed-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    trajectory = load_trajectory(args.trajectory)
    mapped = map_trajectory_to_plane(
        trajectory,
        center_xy=SURFACE_CENTER_XY_M,
        size_xy=SURFACE_SIZE_XY_M,
        flip_image_y=True,
        max_step=0.005,
        smoothing_strength=0.08,
        corner_angle_degrees=35.0,
    )

    executed_by_stroke: dict[int, list[list[float]]] = {}
    with open(args.executed_csv, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["pen_down"] == "1":
                executed_by_stroke.setdefault(int(row["stroke_id"]), []).append(
                    [float(row["tip_x_m"]), float(row["tip_y_m"])]
                )

    figure, axis = plt.subplots(figsize=(6, 6))
    for stroke in mapped["strokes"]:
        points = np.asarray(stroke["points"])
        axis.plot(points[:, 0], points[:, 1], "-", color="#2ca02c", linewidth=1.2, alpha=0.8)
    for points in executed_by_stroke.values():
        arr = np.asarray(points)
        axis.plot(arr[:, 0], arr[:, 1], "-", color="#111111", linewidth=1.8)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("world X (m)")
    axis.set_ylabel("world Y (m)")
    axis.set_title(args.title or mapped["drawing_id"])
    axis.grid(True, alpha=0.2)
    from matplotlib.lines import Line2D

    axis.legend(
        handles=[
            Line2D([0], [0], color="#2ca02c", label="desired"),
            Line2D([0], [0], color="#111111", label="executed"),
        ]
    )
    figure.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    plt.close(figure)
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
