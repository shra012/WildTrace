"""Compressed, self-describing deterministic demonstration recording."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


class DemonstrationRecorder:
    def __init__(self, drawing_id: str, lookahead_points: int, max_action_delta_m: float):
        self.drawing_id = str(drawing_id)
        self.lookahead_points = int(lookahead_points)
        self.max_action_delta_m = float(max_action_delta_m)
        self.rows: List[Dict[str, Any]] = []

    def append(
        self,
        *,
        stroke_id: int,
        waypoint_index: int,
        simulation_time_s: float,
        joint_positions_rad: Sequence[float],
        joint_velocities_rad_s: Sequence[float],
        pen_tip_position_m: Sequence[float],
        pen_tip_rotation_matrix: Sequence[Sequence[float]],
        current_target_m: Sequence[float],
        upcoming_targets_m: Sequence[Sequence[float]],
        pen_down: bool,
        state: str,
    ) -> None:
        tip = np.asarray(pen_tip_position_m, dtype=np.float64)
        target = np.asarray(current_target_m, dtype=np.float64)
        error = target - tip
        norm = float(np.linalg.norm(error))
        action = error if norm <= self.max_action_delta_m else error * (self.max_action_delta_m / norm)
        upcoming = np.asarray(upcoming_targets_m, dtype=np.float64)
        if upcoming.shape != (self.lookahead_points, 3):
            raise ValueError(f"Expected lookahead shape {(self.lookahead_points, 3)}, got {upcoming.shape}")
        self.rows.append(
            {
                "stroke_id": int(stroke_id),
                "waypoint_index": int(waypoint_index),
                "simulation_time_s": float(simulation_time_s),
                "joint_positions_rad": np.asarray(joint_positions_rad, dtype=np.float64),
                "joint_velocities_rad_s": np.asarray(joint_velocities_rad_s, dtype=np.float64),
                "pen_tip_position_m": tip,
                "pen_tip_rotation_matrix": np.asarray(pen_tip_rotation_matrix, dtype=np.float64),
                "current_target_m": target,
                "lookahead_target_m": upcoming,
                "pen_down": bool(pen_down),
                "cartesian_action_delta_m": action,
                "tracking_error_m": norm,
                "state": str(state),
            }
        )

    def save(self, path: str | Path) -> Path:
        if not self.rows:
            raise ValueError("No demonstration samples to save")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays: Dict[str, Any] = {
            "drawing_id": np.asarray([self.drawing_id] * len(self.rows)),
            "stroke_id": np.asarray([r["stroke_id"] for r in self.rows], dtype=np.int64),
            "waypoint_index": np.asarray([r["waypoint_index"] for r in self.rows], dtype=np.int64),
            "simulation_time_s": np.asarray([r["simulation_time_s"] for r in self.rows], dtype=np.float64),
            "joint_positions_rad": np.stack([r["joint_positions_rad"] for r in self.rows]),
            "joint_velocities_rad_s": np.stack([r["joint_velocities_rad_s"] for r in self.rows]),
            "pen_tip_position_m": np.stack([r["pen_tip_position_m"] for r in self.rows]),
            "pen_tip_rotation_matrix": np.stack([r["pen_tip_rotation_matrix"] for r in self.rows]),
            "current_target_m": np.stack([r["current_target_m"] for r in self.rows]),
            "lookahead_target_m": np.stack([r["lookahead_target_m"] for r in self.rows]),
            "pen_down": np.asarray([r["pen_down"] for r in self.rows], dtype=np.float32)[:, None],
            "cartesian_action_delta_m": np.stack([r["cartesian_action_delta_m"] for r in self.rows]),
            "tracking_error_m": np.asarray([r["tracking_error_m"] for r in self.rows], dtype=np.float64),
            "state": np.asarray([r["state"] for r in self.rows]),
        }
        features = make_policy_features(arrays)
        metadata = {
            "format_version": 1,
            "feature_order": [
                "joint_positions_rad[7]",
                "joint_velocities_rad_s[7]",
                "pen_tip_position_m[3]",
                "target_error_m[3]",
                f"lookahead_target_relative_m[{self.lookahead_points}x3]",
                "pen_down[1]",
            ],
            "action": "bounded Cartesian delta [dx,dy,dz] in metres",
            "max_action_delta_m": self.max_action_delta_m,
            "normalization": {
                "feature_mean": features.mean(axis=0).tolist(),
                "feature_std": np.maximum(features.std(axis=0), 1e-8).tolist(),
            },
        }
        arrays["feature_names_json"] = np.asarray(json.dumps(metadata["feature_order"]))
        arrays["units_json"] = np.asarray(
            json.dumps({key: _unit_for(key) for key in arrays if not key.endswith("_json")}, sort_keys=True)
        )
        arrays["normalization_metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
        np.savez_compressed(destination, **arrays)
        return destination


def _unit_for(name: str) -> str:
    if name.endswith("_rad"):
        return "rad"
    if name.endswith("_rad_s"):
        return "rad/s"
    if name.endswith("_m") or "target" in name or "action_delta" in name:
        return "m"
    if name == "simulation_time_s":
        return "s"
    return "dimensionless"


def make_policy_features(data: Dict[str, np.ndarray]) -> np.ndarray:
    tip = np.asarray(data["pen_tip_position_m"], dtype=np.float32)
    target = np.asarray(data["current_target_m"], dtype=np.float32)
    lookahead = np.asarray(data["lookahead_target_m"], dtype=np.float32) - tip[:, None, :]
    return np.concatenate(
        [
            np.asarray(data["joint_positions_rad"], dtype=np.float32),
            np.asarray(data["joint_velocities_rad_s"], dtype=np.float32),
            tip,
            target - tip,
            lookahead.reshape(len(tip), -1),
            np.asarray(data["pen_down"], dtype=np.float32).reshape(len(tip), 1),
        ],
        axis=1,
    )

