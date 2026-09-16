"""Pure-Python N-stroke drawing state machine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from coordinate_mapper import interpolate_segment, resample_polyline

HOME = "HOME"
FINISHED = "FINISHED"
FAILED = "FAILED"


@dataclass(frozen=True)
class MotionTarget:
    state: str
    position: np.ndarray
    stroke_id: int
    waypoint_index: int
    pen_down: bool


@dataclass(frozen=True)
class Phase:
    state: str
    targets: List[MotionTarget]


def _xyz(points_xy: Sequence[Sequence[float]], z: float) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    return np.c_[points, np.full(len(points), float(z))]


def build_motion_sequence(
    strokes: Sequence[dict],
    *,
    pen_down_z: float,
    pen_up_z: float,
    approach_height: float,
    max_cartesian_step: float,
) -> List[Phase]:
    """Build approach/lower/draw/lift phases for one or more strokes in order.

    A safe travel leg (lift height, then horizontal, then lower) is inserted
    between consecutive strokes; the first stroke is approached directly.
    """
    if len(strokes) < 1:
        raise ValueError("At least one stroke is required")
    if not pen_down_z < pen_up_z <= approach_height:
        raise ValueError("Require pen_down_z < pen_up_z <= approach_height")

    def targets(state: str, points: np.ndarray, stroke_id: int, pen_down: bool) -> List[MotionTarget]:
        return [
            MotionTarget(state, point.copy(), stroke_id, index, pen_down)
            for index, point in enumerate(np.asarray(points, dtype=np.float64))
        ]

    phases: List[Phase] = []
    previous_lift_xyz: Optional[np.ndarray] = None
    for stroke in strokes:
        stroke_id = int(stroke["stroke_id"])
        xy = np.asarray(stroke["points"], dtype=np.float64)
        if len(xy) < 2:
            raise ValueError(f"Stroke {stroke_id} needs at least two points")

        approach_point = np.r_[xy[0], approach_height]
        down_point = np.r_[xy[0], pen_down_z]

        if previous_lift_xyz is None:
            state = f"APPROACH_STROKE_{stroke_id}"
            phases.append(Phase(state, targets(state, np.asarray([approach_point]), stroke_id, False)))
        else:
            travel_start = previous_lift_xyz.copy()
            travel_start[2] = approach_height
            vertical = interpolate_segment(previous_lift_xyz, travel_start, max_cartesian_step)[1:]
            horizontal = interpolate_segment(travel_start, approach_point, max_cartesian_step)[1:]
            travel = np.vstack([part for part in (vertical, horizontal) if len(part)])
            state = f"MOVE_TO_STROKE_{stroke_id}"
            phases.append(Phase(state, targets(state, travel, stroke_id, False)))

        lower_state = f"LOWER_PEN_STROKE_{stroke_id}"
        phases.append(
            Phase(
                lower_state,
                targets(lower_state, interpolate_segment(approach_point, down_point, max_cartesian_step)[1:], stroke_id, False),
            )
        )
        draw_state = f"DRAW_STROKE_{stroke_id}"
        phases.append(
            Phase(draw_state, targets(draw_state, resample_polyline(_xyz(xy, pen_down_z), max_cartesian_step), stroke_id, True))
        )
        up_point = np.r_[xy[-1], pen_up_z]
        lift_state = f"LIFT_PEN_STROKE_{stroke_id}"
        phases.append(
            Phase(
                lift_state,
                targets(
                    lift_state,
                    interpolate_segment(np.r_[xy[-1], pen_down_z], up_point, max_cartesian_step)[1:],
                    stroke_id,
                    False,
                ),
            )
        )
        previous_lift_xyz = up_point

    for phase in phases:
        if not phase.targets:
            raise ValueError(f"State {phase.state} has no targets")
    return phases


class MultiStrokeStateMachine:
    """Advances HOME -> per-stroke approach/lower/draw/lift phases -> FINISHED.

    A waypoint only advances once the caller reports it reached (measured
    pen-tip error within tolerance); a stalled waypoint times out to FAILED.
    """

    def __init__(self, phases: Sequence[Phase], waypoint_timeout_s: float):
        if not phases:
            raise ValueError("At least one phase is required")
        if waypoint_timeout_s <= 0:
            raise ValueError("waypoint_timeout_s must be positive")
        self.phases = list(phases)
        self.waypoint_timeout_s = float(waypoint_timeout_s)
        self.phase_index = 0
        self.target_index = 0
        self.target_started_s = 0.0
        self.failure_reason = ""
        self.state = HOME
        self.visited_states = [HOME]

    @property
    def complete(self) -> bool:
        return self.state == FINISHED

    @property
    def failed(self) -> bool:
        return self.state == FAILED

    @property
    def current_target(self) -> Optional[MotionTarget]:
        if self.state in (HOME, FINISHED, FAILED):
            return None
        return self.phases[self.phase_index].targets[self.target_index]

    def mark_home_reached(self, simulation_time_s: float) -> None:
        if self.state != HOME:
            raise RuntimeError("Home can only be completed from HOME")
        self.phase_index = 0
        self.target_index = 0
        self._enter(self.phases[0].state, simulation_time_s)

    def update(self, simulation_time_s: float, reached: bool) -> None:
        if self.state in (HOME, FINISHED, FAILED):
            return
        if simulation_time_s - self.target_started_s > self.waypoint_timeout_s:
            target = self.current_target
            self.fail(f"Waypoint timeout in {self.state} at index {target.waypoint_index}")
            return
        if not reached:
            return
        current_phase = self.phases[self.phase_index]
        if self.target_index + 1 < len(current_phase.targets):
            self.target_index += 1
            self.target_started_s = float(simulation_time_s)
            return
        if self.phase_index + 1 < len(self.phases):
            self.phase_index += 1
            self.target_index = 0
            self._enter(self.phases[self.phase_index].state, simulation_time_s)
            return
        self._enter(FINISHED, simulation_time_s)

    def fail(self, reason: str) -> None:
        self.failure_reason = str(reason)
        self.state = FAILED
        self.visited_states.append(FAILED)

    def _enter(self, state: str, simulation_time_s: float) -> None:
        self.state = state
        self.target_started_s = float(simulation_time_s)
        self.visited_states.append(state)


def flatten_desired_targets(phases: Sequence[Phase]) -> List[MotionTarget]:
    return [target for phase in phases for target in phase.targets]
