"""Observation features shared by demonstration recording, dataset building and ACT rollout.

One function builds the path window and one assembles the policy vectors, so
recorder, kinematic labeller, LeRobot converter and closed-loop runner cannot
drift apart. Pure numpy; safe to import inside Isaac Sim's python.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

PATH_WINDOW_POINTS = 10
PATH_WINDOW_SPACING_M = 0.002
# Enough raw targets to cover the window even after corner densification.
_MAX_TARGETS_SCANNED = 256

STATE_NAMES = (
    [f"q{i}" for i in range(1, 8)]
    + [f"qd{i}" for i in range(1, 8)]
    + [f"q{i}_prev_cmd" for i in range(1, 8)]
)
# Actions are steps from the previous joint command, not absolute targets.
# Absolute targets normalized over the workspace left ~0.002 rad (mm-level)
# error even when overfitting one drawing; per-tick steps are ~100x smaller.
ACTION_NAMES = [f"dq{i}_cmd" for i in range(1, 8)]


def environment_state_names(points: int = PATH_WINDOW_POINTS) -> list[str]:
    names = ["tip_x", "tip_y", "tip_z", "err_x", "err_y", "err_z"]
    for index in range(points):
        names += [f"w{index}_x", f"w{index}_y", f"w{index}_z"]
    return names + ["pen_down"]


def path_window(
    targets: np.ndarray,
    start_index: int,
    *,
    points: int = PATH_WINDOW_POINTS,
    spacing_m: float = PATH_WINDOW_SPACING_M,
    start_position: Sequence[float] | None = None,
) -> np.ndarray:
    """Upcoming path resampled at equal arc length, beginning at the current target.

    `targets` is the flattened (M, 3) sequence of every remaining waypoint, so
    the window runs across pen lifts and travel legs. `start_position`
    overrides the first point (the kinematic labeller chases an interpolated
    point between waypoints). Past the end of the path the last point repeats.
    """
    remaining = np.asarray(targets[start_index : start_index + _MAX_TARGETS_SCANNED], dtype=np.float64)
    if start_position is not None:
        remaining = np.vstack([np.asarray(start_position, dtype=np.float64)[None, :], remaining[1:]])
    if len(remaining) == 0:
        raise ValueError("path_window needs at least one target")
    if len(remaining) == 1:
        return np.repeat(remaining, points, axis=0)
    segment = np.linalg.norm(np.diff(remaining, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(segment)]
    query = np.minimum(np.arange(points, dtype=np.float64) * float(spacing_m), arc[-1])
    return np.stack([np.interp(query, arc, remaining[:, axis]) for axis in range(3)], axis=1)


def observation_state(
    joint_positions_rad: np.ndarray,
    joint_velocities_rad_s: np.ndarray,
    previous_command_rad: np.ndarray,
) -> np.ndarray:
    q = np.asarray(joint_positions_rad, dtype=np.float32).reshape(-1, 7)
    qd = np.asarray(joint_velocities_rad_s, dtype=np.float32).reshape(-1, 7)
    previous = np.asarray(previous_command_rad, dtype=np.float32).reshape(-1, 7)
    return np.concatenate([q, qd, previous], axis=1)


def previous_commands(commanded_rad: np.ndarray, first_measured_rad: np.ndarray) -> np.ndarray:
    """Command in force before each frame; the first frame starts from the measured pose."""
    commanded = np.asarray(commanded_rad, dtype=np.float64).reshape(-1, 7)
    return np.vstack([np.asarray(first_measured_rad, dtype=np.float64).reshape(1, 7), commanded[:-1]])


def environment_state(
    pen_tip_position_m: np.ndarray,
    current_target_m: np.ndarray,
    path_window_m: np.ndarray,
    pen_down: np.ndarray,
) -> np.ndarray:
    """[tip(3), target - tip(3), window - tip (N*3), pen_down(1)] per frame."""
    tip = np.asarray(pen_tip_position_m, dtype=np.float32).reshape(-1, 3)
    target = np.asarray(current_target_m, dtype=np.float32).reshape(-1, 3)
    window = np.asarray(path_window_m, dtype=np.float32).reshape(len(tip), -1, 3)
    relative = (window - tip[:, None, :]).reshape(len(tip), -1)
    pen = np.asarray(pen_down, dtype=np.float32).reshape(len(tip), 1)
    return np.concatenate([tip, target - tip, relative, pen], axis=1)


def act_frames(data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Policy arrays from a v2 demonstration (NPZ dict or recorder arrays).

    Call on frames already subsampled to the policy rate: the action is the
    step between consecutive kept commands, i.e. one control tick.
    """
    if "commanded_joint_positions_rad" not in data or "path_window_m" not in data:
        raise ValueError("ACT frames need a format_version 2 demonstration (commanded joints + path window)")
    commanded = np.asarray(data["commanded_joint_positions_rad"], dtype=np.float64).reshape(-1, 7)
    if "previous_command_rad" in data:
        # Noise-injected labels: the step is taken from the executed (perturbed)
        # command, not from the previous label. Recorded at the policy rate.
        previous = np.asarray(data["previous_command_rad"], dtype=np.float64).reshape(-1, 7)
    else:
        previous = previous_commands(commanded, np.asarray(data["joint_positions_rad"])[0])
    return {
        "observation.state": observation_state(data["joint_positions_rad"], data["joint_velocities_rad_s"], previous),
        "observation.environment_state": environment_state(
            data["pen_tip_position_m"], data["current_target_m"], data["path_window_m"], data["pen_down"]
        ),
        "action": (commanded - previous).astype(np.float32),
    }


def sample_augmentation(
    rng: np.random.Generator,
    center_xy: Sequence[float],
    size_xy: Sequence[float],
    stroke_count: int,
    *,
    center_jitter_m: float = 0.02,
    scale_range: tuple[float, float] = (0.8, 1.1),
    reverse_probability: float = 0.5,
) -> dict:
    """Random paper placement, drawing scale and per-stroke direction for one episode."""
    center = np.asarray(center_xy, dtype=np.float64) + rng.uniform(-center_jitter_m, center_jitter_m, size=2)
    scale = float(rng.uniform(*scale_range))
    reversed_strokes = [int(i) for i in range(stroke_count) if rng.random() < reverse_probability]
    return {
        "surface_center_xy_m": center.tolist(),
        "surface_size_xy_m": (np.asarray(size_xy, dtype=np.float64) * scale).tolist(),
        "scale": scale,
        "reversed_strokes": reversed_strokes,
    }


def reverse_strokes(trajectory: dict, stroke_indices: Sequence[int]) -> dict:
    """Copy of a loaded trajectory with the listed strokes (by position) drawn backwards."""
    flip = set(int(i) for i in stroke_indices)
    strokes = [
        {**stroke, "points": list(stroke["points"])[::-1]} if index in flip else stroke
        for index, stroke in enumerate(trajectory["strokes"])
    ]
    return {**trajectory, "strokes": strokes}


def path_arc_length(targets: np.ndarray) -> np.ndarray:
    positions = np.asarray(targets, dtype=np.float64)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]


def project_on_path(
    targets: np.ndarray, arc: np.ndarray, index: int, tip: Sequence[float], search: int = 40
) -> tuple[float, int]:
    """Arc position of the tip's projection near `index`, and the first waypoint ahead of it.

    Only segments from index-1 to index+search are considered, so a closed
    outline cannot match its far side. The returned waypoint never moves
    backwards past `index`.
    """
    positions = np.asarray(targets, dtype=np.float64)
    point = np.asarray(tip, dtype=np.float64)
    start = max(int(index) - 1, 0)
    stop = min(int(index) + search, len(positions) - 1)
    if stop <= start:
        return float(arc[min(int(index), len(arc) - 1)]), int(index)
    a = positions[start:stop]
    segment = positions[start + 1 : stop + 1] - a
    length_sq = np.maximum(np.einsum("ij,ij->i", segment, segment), 1e-18)
    t = np.clip(np.einsum("ij,ij->i", point - a, segment) / length_sq, 0.0, 1.0)
    distance = np.linalg.norm(a + t[:, None] * segment - point, axis=1)
    best = int(np.argmin(distance))
    s = float(arc[start + best] + t[best] * np.sqrt(length_sq[best]))
    ahead = int(np.searchsorted(arc, s, side="right"))
    return s, int(min(max(ahead, int(index)), len(positions) - 1))


def chased_index(targets: np.ndarray, arc: np.ndarray, index: int, tip: Sequence[float], search: int = 40) -> int:
    """First waypoint strictly ahead of the pen, never moving backwards.

    This is the same "chased waypoint" the kinematic labeller records, so a
    policy trained on those labels sees the same target semantics in closed
    loop, without the IK runner's stop-and-settle reach gate.
    """
    return project_on_path(targets, arc, index, tip, search)[1]
