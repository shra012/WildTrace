"""Observation features shared by demonstration recording, dataset building and ACT rollout.

One function builds the path window and one assembles the policy vectors, so
recorder, kinematic labeller, LeRobot converter and closed-loop runner cannot
drift apart. Pure numpy; safe to import inside Isaac Sim's python.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

PATH_WINDOW_POINTS = 10
# How far along the path (from the current target) the tip may be projected.
# A waypoint-count window (40 targets, 20+ mm where corners are densified) let
# the projection hop across narrow hairpins and cut e.g. a 162 deg toe by
# 8.5 mm. A 5 mm window still let it hop from one leg of a needle-sharp spike
# to the other once the pen was within ~2.5 mm of the tip, rounding tips by
# 2.0-2.6 mm. 1.5 mm caps that at ~0.75 mm and is still three times the pen's
# travel per 30 fps tick.
PROJECTION_AHEAD_M = 0.0015
PATH_WINDOW_SPACING_M = 0.002
# The pen inks while pen_down and the tip is within this of the target height. The
# twin's pen-down height is the calibrated contact height; the labeller's joint
# noise moves the tip about +-1 mm in z (10th-90th percentile), which a sprung pen
# rides out, so a 0.5 mm band marked 19% of pen-down frames as not inked (1.5 mm: 6%).
INK_CONTACT_M = 0.0015
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
    height_jitter_m: float = 0.0,
) -> dict:
    """Random paper placement, drawing scale and per-stroke direction for one episode.

    `height_jitter_m` also shifts the paper (and every paper-relative height)
    up or down, so one policy tolerates the sheet being re-placed between lab
    sessions. It is drawn last, so episodes sampled without it are unchanged.
    """
    center = np.asarray(center_xy, dtype=np.float64) + rng.uniform(-center_jitter_m, center_jitter_m, size=2)
    scale = float(rng.uniform(*scale_range))
    reversed_strokes = [int(i) for i in range(stroke_count) if rng.random() < reverse_probability]
    return {
        "surface_center_xy_m": center.tolist(),
        "surface_size_xy_m": (np.asarray(size_xy, dtype=np.float64) * scale).tolist(),
        "scale": scale,
        "reversed_strokes": reversed_strokes,
        "height_offset_m": float(rng.uniform(-height_jitter_m, height_jitter_m)) if height_jitter_m > 0 else 0.0,
    }


def shift_paper_heights(drawing: dict, offset_m: float) -> dict:
    """Copy of a drawing config with the paper and every pen height moved by `offset_m`."""
    keys = ("paper_top_z_m", "pen_down_z_m", "pen_up_z_m", "approach_height_m")
    return {**drawing, **{key: float(drawing[key]) + float(offset_m) for key in keys}}


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
    targets: np.ndarray,
    arc: np.ndarray,
    index: int,
    tip: Sequence[float],
    max_ahead_m: float = PROJECTION_AHEAD_M,
) -> tuple[float, int]:
    """Arc position of the tip's projection near `index`, and the first waypoint ahead of it.

    Only segments from index-1 up to `max_ahead_m` of path beyond waypoint
    `index` are considered, so neither a closed outline's far side nor a
    hairpin's return leg can be matched. The returned waypoint never moves
    backwards past `index`.
    """
    positions = np.asarray(targets, dtype=np.float64)
    point = np.asarray(tip, dtype=np.float64)
    start = max(int(index) - 1, 0)
    limit = float(arc[min(int(index), len(arc) - 1)]) + float(max_ahead_m)
    # At least one segment beyond the current target, even if it is long.
    stop = min(max(int(np.searchsorted(arc, limit, side="right")), int(index) + 1), len(positions) - 1)
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


def chased_index(
    targets: np.ndarray, arc: np.ndarray, index: int, tip: Sequence[float], max_ahead_m: float = PROJECTION_AHEAD_M
) -> int:
    """First waypoint strictly ahead of the pen, never moving backwards.

    This is the same "chased waypoint" the kinematic labeller records, so a
    policy trained on those labels sees the same target semantics in closed
    loop, without the IK runner's stop-and-settle reach gate.
    """
    return project_on_path(targets, arc, index, tip, max_ahead_m)[1]


class PathProgress:
    """Where a learned controller is along the path, with a stall release.

    Each tick the target is the first waypoint ahead of the pen's projection on
    the path (chased_index). A policy trained on perfect-tracking labels can end
    up hovering beside the path: close to it, but no longer moving along it, so
    the projection, and with it the target, never advances; on the real arm the
    pen then needed a manual nudge. If the projection has not moved forward by
    `min_progress_m` for `stall_s` while the pen is within `band_m` of the path,
    the target is stepped one waypoint ahead, which turns the policy's correction
    forward again. In normal running the projection keeps moving and the release
    never fires. stall_s <= 0 disables it (plain chased_index).
    """

    def __init__(self, targets: np.ndarray, arc: np.ndarray, stall_s: float = 0.5, band_m: float = 0.0015,
                 min_progress_m: float = 1e-4, max_ahead_m: float = PROJECTION_AHEAD_M):
        self.targets = np.asarray(targets, dtype=np.float64)
        self.arc = np.asarray(arc, dtype=np.float64)
        self.stall_s, self.band_m = float(stall_s), float(band_m)
        self.min_progress_m, self.max_ahead_m = float(min_progress_m), float(max_ahead_m)
        self.releases = 0
        self._best_s: float | None = None
        self._since = 0.0

    def update(self, index: int, tip: Sequence[float], now_s: float) -> int:
        s, ahead = project_on_path(self.targets, self.arc, index, tip, self.max_ahead_m)
        if self._best_s is None or s > self._best_s + self.min_progress_m:
            self._best_s, self._since = s, float(now_s)
            return ahead
        if self.stall_s <= 0 or now_s - self._since < self.stall_s or ahead >= len(self.targets) - 1:
            return ahead
        on_path = np.array([np.interp(s, self.arc, self.targets[:, axis]) for axis in range(3)])
        if np.linalg.norm(np.asarray(tip, dtype=np.float64) - on_path) > self.band_m:
            return ahead  # far off the path: a real tracking problem, left to the safety checks
        self.releases += 1
        self._since = float(now_s)
        return ahead + 1


def distance_to_path_behind(
    targets: np.ndarray, arc: np.ndarray, index: int, tip: Sequence[float], behind_m: float
) -> float:
    """Distance from the tip to the path between `behind_m` of arc before waypoint `index` and that waypoint.

    The policy safety check measures against this instead of the current target alone, so a
    pen that goes back along the line to redraw a gap is still known to be on the drawing.
    """
    positions = np.asarray(targets, dtype=np.float64)
    point = np.asarray(tip, dtype=np.float64)
    end = min(int(index), len(positions) - 1)
    start = max(int(np.searchsorted(arc, float(arc[end]) - float(behind_m), side="left")) - 1, 0)
    if end <= start:
        return float(np.linalg.norm(positions[end] - point))
    a = positions[start:end]
    segment = positions[start + 1 : end + 1] - a
    length_sq = np.maximum(np.einsum("ij,ij->i", segment, segment), 1e-18)
    t = np.clip(np.einsum("ij,ij->i", point - a, segment) / length_sq, 0.0, 1.0)
    return float(np.min(np.linalg.norm(a + t[:, None] * segment - point, axis=1)))


class SkipSchedule:
    """Where the pen fails to ink (#46): stretches of inked travel distance.

    Skips are placed along the distance the pen has travelled while in contact,
    `spacing_m` apart on average (exponential), each `length_m` long (uniform). The
    distance keeps growing on a redraw pass, so going back over a skip inks it, like
    a pen that bounced or ran dry for a moment. The kinematic labeller and the policy
    server's skip test use this same schedule.
    """

    def __init__(self, rng: np.random.Generator, spacing_m: float, length_m: tuple[float, float]):
        self.rng = rng
        self.spacing_m = float(spacing_m)
        self.length_m = (float(length_m[0]), float(length_m[1]))
        self.travel_m = 0.0
        self._last_xy = None
        self._skip: tuple[float, float] | None = None
        self.skips = 0
        if self.spacing_m > 0:
            self._schedule(0.0)

    def _schedule(self, after_m: float) -> None:
        start = after_m + float(self.rng.exponential(self.spacing_m))
        self._skip = (start, start + float(self.rng.uniform(*self.length_m)))

    def inks(self, contact: bool, tip_xy: Sequence[float]) -> bool:
        """Advance by one tick; True when this tick leaves ink."""
        if not contact:
            self._last_xy = None
            return False
        xy = np.asarray(tip_xy, dtype=np.float64)[:2]
        if self._last_xy is not None:
            self.travel_m += float(np.linalg.norm(xy - self._last_xy))
        self._last_xy = xy
        if self._skip is None:
            return True
        start, end = self._skip
        if self.travel_m >= end:
            self.skips += 1
            self._schedule(self.travel_m)
            start, end = self._skip
        return not (start <= self.travel_m < end)


# A pen behind its target on the path (going back to redraw a gap, #46) is shown the waypoint just
# ahead of where it is instead of the monotonic target further on. Otherwise the target-tip error of a
# redraw grows to several mm, which normal drawing never shows, and a policy learns "large error means
# go back" from the numbers instead of reading the gap off the page; in closed loop a 2 mm tracking lag
# then turned into a reverse and a stall.
RETURN_TRIGGER_M = 0.001
RETURN_BAND_M = 0.002
RETURN_BEHIND_M = 0.025
RETURN_MIN_AHEAD_M = 0.0005  # never a target at the pen itself, which a go-to-target policy stalls on


def observed_index(
    targets: np.ndarray,
    arc: np.ndarray,
    index: int,
    tip: Sequence[float],
    trigger_m: float = RETURN_TRIGGER_M,
    band_m: float = RETURN_BAND_M,
    behind_m: float = RETURN_BEHIND_M,
) -> int:
    """The waypoint a policy observation should use as its target: `index` itself, unless the tip is more
    than `trigger_m` from it and lies within `band_m` of the path up to `behind_m` of arc behind it, at
    least `trigger_m` back; then the first waypoint at least RETURN_MIN_AHEAD_M ahead of the tip's
    position on that stretch. Only
    path running the same way as the current segment counts, so at a needle spike the other leg (which
    runs the opposite way, a fraction of a millimetre beside the pen) is never taken for "behind"."""
    positions = np.asarray(targets, dtype=np.float64)
    point = np.asarray(tip, dtype=np.float64)
    index = min(int(index), len(positions) - 1)
    if np.linalg.norm(positions[index] - point) < trigger_m or index == 0:
        return index
    start = max(int(np.searchsorted(arc, float(arc[index]) - float(behind_m), side="left")) - 1, 0)
    a = positions[start:index]
    segment = positions[start + 1 : index + 1] - a
    length_sq = np.maximum(np.einsum("ij,ij->i", segment, segment), 1e-18)
    t = np.clip(np.einsum("ij,ij->i", point - a, segment) / length_sq, 0.0, 1.0)
    distance = np.linalg.norm(a + t[:, None] * segment - point, axis=1)
    heading = positions[index] - positions[index - 1]
    distance[np.einsum("ij,j->i", segment, heading) <= 0.0] = np.inf
    best = int(np.argmin(distance))
    s = float(arc[start + best] + t[best] * np.sqrt(length_sq[best]))
    if not np.isfinite(distance[best]) or distance[best] > band_m or s > float(arc[index]) - trigger_m:
        return index
    return int(min(np.searchsorted(arc, s + RETURN_MIN_AHEAD_M, side="left"), index))
