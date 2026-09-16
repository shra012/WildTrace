"""Validate and summarize the configured trajectory."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from coordinate_mapper import map_trajectory_to_plane
from drawing_state_machine import build_motion_sequence, flatten_desired_targets
from project_config import load_config
from trajectory_loader import load_trajectory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/xarm7_drawing.yaml")
    parser.add_argument("--trajectory", default=None, help="Override project.trajectory_path")
    parser.add_argument("--first-n-strokes", type=int, default=None, help="Only validate the first N strokes")
    args = parser.parse_args()
    config = load_config(args.config, PROJECT_ROOT)
    drawing = config["drawing"]
    trajectory_path = args.trajectory or config["project"]["trajectory_path"]
    trajectory = load_trajectory(trajectory_path, first_n_strokes=args.first_n_strokes)
    mapped = map_trajectory_to_plane(
        trajectory,
        center_xy=drawing["surface_center_xy_m"],
        size_xy=drawing["surface_size_xy_m"],
        flip_image_y=bool(drawing["flip_image_y"]),
        max_step=float(drawing["max_cartesian_step_m"]),
        smoothing_strength=float(drawing["smoothing_strength"]),
        corner_angle_degrees=float(drawing["corner_angle_degrees"]),
    )
    phases = build_motion_sequence(
        mapped["strokes"],
        pen_down_z=float(drawing["pen_down_z_m"]),
        pen_up_z=float(drawing["pen_up_z_m"]),
        approach_height=float(drawing["approach_height_m"]),
        max_cartesian_step=float(drawing["max_cartesian_step_m"]),
    )
    targets = flatten_desired_targets(phases)
    report = {
        "drawing_id": mapped["drawing_id"],
        "stroke_ids": [s["stroke_id"] for s in mapped["strokes"]],
        "points_per_stroke": [len(s["points"]) for s in mapped["strokes"]],
        "motion_target_count": len(targets),
        "mapping": mapped["mapping"],
        "state_target_counts": {phase.state: len(phase.targets) for phase in phases},
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

