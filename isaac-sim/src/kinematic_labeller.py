"""Physics-free IK demonstrations for ACT pretraining.

The same motion phases the Isaac runner executes are walked at a constant
tip speed per phase type. Each frame solves IK for where the pen should be
one frame later, slew-limits it from the previous command exactly like
SafeLulaIKController.apply, and treats the previous command as the measured
pose (one-frame lag, perfect servo). The IK solver and forward kinematics
are injected, so this module has no Isaac dependency and is unit-testable.

With `noise`, the executed command is perturbed by correlated joint noise
(DART-style). Progress along the path comes from projecting the arm's actual
tip onto it (act_features.project_on_path, as the closed-loop runner does),
and the label is "IK to the path point just ahead of that projection,
stepping from where the arm actually is". So every action carries a
correction, and the target is observable from the tip and path window; an
open-loop progress clock is not.

`noise` may also carry a servo offset: the measured joints sit a slowly
varying amount away from the executed command, the way gravity sag and
paper contact hold them off it in Isaac. The label then commands the IK
solution minus the observed offset (measured minus previous command), so the
policy learns to compensate instead of only ever seeing q == previous command. Perfect-tracking labels alone never show an off-path state, and a
policy trained only on them drifts in closed loop without recovering.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from act_features import path_window, project_on_path
from coordinate_mapper import interpolate_segment
from drawing_state_machine import MotionTarget, Phase, flatten_desired_targets
from path_geometry import limit_joint_delta

DEFAULT_SPEEDS_M_S = {
    "APPROACH": 0.030,
    "MOVE_TO": 0.020,
    "LOWER_PEN": 0.015,
    "LIFT_PEN": 0.015,
    "DRAW": 0.014,
}

SolveIK = Callable[[np.ndarray, np.ndarray], Optional[np.ndarray]]
ForwardKinematics = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]


class LabelError(RuntimeError):
    """The episode cannot be labelled safely and must be dropped."""


def phase_speed(state: str, speeds: Dict[str, float]) -> float:
    for prefix, speed in speeds.items():
        if state.startswith(prefix):
            return float(speed)
    raise KeyError(f"No speed configured for state {state}")


def turn_slowdown_speeds(
    targets: Sequence[MotionTarget],
    arc: np.ndarray,
    speeds: np.ndarray,
    *,
    angle_deg: float = 120.0,
    radius_m: float = 0.004,
    min_factor: float = 0.25,
    chord_m: float = 0.002,
) -> np.ndarray:
    """Slow pen-down labels into sharp turns so the policy learns to brake.

    At a constant 14 mm/s the policy rounded needle-sharp spike tips (175-180
    deg reversals) by up to 2.6 mm, where IK reaches them by stopping at every
    waypoint. The turn angle at each pen-down waypoint is measured over a
    `chord_m` chord on each side (robust to contour noise); within `radius_m`
    of arc length of any turn sharper than `angle_deg`, speed ramps linearly
    down to `min_factor` of its value at the apex.
    """
    positions = np.asarray([t.position for t in targets], dtype=np.float64)
    pen_down = np.asarray([t.pen_down for t in targets], dtype=bool)
    phase_key = [(t.state, t.stroke_id) for t in targets]
    result = np.asarray(speeds, dtype=np.float64).copy()
    apexes = []
    for i in np.flatnonzero(pen_down):
        same = [j for j in (i - 1, i + 1) if 0 <= j < len(targets) and phase_key[j] == phase_key[i]]
        if len(same) < 2:
            continue
        lo = i
        while lo > 0 and phase_key[lo - 1] == phase_key[i] and arc[i] - arc[lo] < chord_m:
            lo -= 1
        hi = i
        while hi < len(targets) - 1 and phase_key[hi + 1] == phase_key[i] and arc[hi] - arc[i] < chord_m:
            hi += 1
        back, ahead = positions[i] - positions[lo], positions[hi] - positions[i]
        norm = np.linalg.norm(back) * np.linalg.norm(ahead)
        if norm < 1e-12:
            continue
        turn = np.degrees(np.arccos(np.clip(float(back @ ahead) / norm, -1.0, 1.0)))
        if turn > angle_deg:
            apexes.append(float(arc[i]))
    for apex in apexes:
        near = pen_down & (np.abs(arc - apex) < radius_m)
        factor = min_factor + (1.0 - min_factor) * np.abs(arc[near] - apex) / radius_m
        result[near] = np.minimum(result[near], speeds[near] * factor)
    return result


def with_approach_from(phases: List[Phase], home_tip: np.ndarray, max_step: float) -> List[Phase]:
    """Replace the first phase with a straight leg from the home tip, as the runner does."""
    approach = phases[0]
    final = approach.targets[-1]
    points = interpolate_segment(home_tip, final.position, max_step)[1:]
    first = Phase(
        approach.state,
        [MotionTarget(approach.state, point, final.stroke_id, index, False) for index, point in enumerate(points)],
    )
    return [first, *phases[1:]]


def label_phases(
    phases: Sequence[Phase],
    *,
    home_q: np.ndarray,
    solve_ik: SolveIK,
    forward: ForwardKinematics,
    recorder,
    dt_s: float,
    max_joint_delta_rad: float,
    speeds_m_s: Dict[str, float] | None = None,
    max_ik_failures: int = 3,
    max_clamped_frames: int = 15,
    max_tip_error_m: float = 0.001,
    final_hold_s: float = 0.5,
    noise: Dict | None = None,
    turn_slowdown: Dict | None = None,
    reached_tolerance_m: float = 0.0015,
    max_duration_factor: float = 3.0,
) -> Dict[str, float]:
    """Append one episode of frames to `recorder` and return labelling statistics.

    Raises LabelError on repeated IK failure, a sustained slew-limited joint
    jump (an IK branch switch), pen-down tip error above `max_tip_error_m`
    (measured on the labelled command, so injected noise does not count), or
    an episode running past `max_duration_factor` times its nominal duration.

    Progress is the larger of the tip's projection and the previous aim point
    once the tip is within `reached_tolerance_m` of it. Projection alone can
    stall forever on a hairpin, where the point just ahead projects back onto
    the earlier side of the turn.

    `turn_slowdown` (keyword arguments for turn_slowdown_speeds, {} for the
    defaults) lowers the label speed into sharp pen-down turns.

    `noise` = {"rng": np.random.Generator, "sigma_rad": float, "correlation_s": float,
    optional "offset_sigma_rad": float, "offset_correlation_s": float} adds
    Ornstein-Uhlenbeck perturbations to every executed command and, if given,
    to the measured-versus-executed servo offset.
    """
    speeds = dict(DEFAULT_SPEEDS_M_S if speeds_m_s is None else speeds_m_s)
    targets = flatten_desired_targets(phases)
    positions = np.asarray([t.position for t in targets], dtype=np.float64)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
    target_speed = np.asarray([phase_speed(t.state, speeds) for t in targets], dtype=np.float64)
    if turn_slowdown is not None:
        target_speed = turn_slowdown_speeds(targets, arc, target_speed, **turn_slowdown)
    lookahead = int(recorder.lookahead_points)

    def point_at(s: float) -> np.ndarray:
        return np.asarray([np.interp(s, arc, positions[:, axis]) for axis in range(3)])


    q_meas = np.asarray(home_q, dtype=np.float64).copy()
    q_prev_meas = q_meas.copy()
    q_cmd = q_meas.copy()
    # Command the arm actually executed last tick (label plus injected noise).
    executed = q_meas.copy()
    perturbation = np.zeros_like(q_meas)
    offset = np.zeros_like(q_meas)
    offset_decay = offset_kick = 0.0
    if noise:
        decay = float(np.exp(-dt_s / float(noise["correlation_s"])))
        kick = float(noise["sigma_rad"]) * np.sqrt(1.0 - decay**2)
        if noise.get("offset_sigma_rad"):
            offset_decay = float(np.exp(-dt_s / float(noise["offset_correlation_s"])))
            offset_kick = float(noise["offset_sigma_rad"]) * np.sqrt(1.0 - offset_decay**2)
    s = 0.0
    time_s = 0.0
    hold_frames = int(round(final_hold_s / dt_s))
    ik_failures = clamped_run = clamped_frames = frames = 0
    pen_down_errors: List[float] = []
    index = 0
    previous_aim = None
    nominal_s = float(np.sum(np.diff(arc) / np.maximum(target_speed[1:], 1e-9))) + final_hold_s
    while True:
        if time_s > max_duration_factor * nominal_s:
            raise LabelError(f"Progress stalled at {targets[index].state} waypoint {targets[index].waypoint_index}")
        tip, rotation = forward(q_meas)
        projected, index = project_on_path(positions, arc, index, tip)
        # Progress never runs backwards, matching the runner's monotonic target.
        s = max(s, projected)
        if previous_aim is not None and np.linalg.norm(np.asarray(tip) - previous_aim[1]) <= reached_tolerance_m:
            s = max(s, previous_aim[0])
            index = max(index, int(min(np.searchsorted(arc, s, side="right"), len(targets) - 1)))
        target = targets[index]
        speed = float(target_speed[index])
        s_next = min(s + speed * dt_s, float(arc[-1]))
        desired_next = point_at(s_next)
        previous_aim = (s_next, desired_next)

        # What the servo currently holds the arm away from its command.
        observed_offset = q_meas - executed
        proposed = solve_ik(desired_next, q_meas)
        if proposed is None:
            ik_failures += 1
            if ik_failures >= max_ik_failures:
                raise LabelError(f"IK failed {ik_failures}x at {target.state} waypoint {target.waypoint_index}")
            proposed = q_meas
        else:
            ik_failures = 0
        proposed = np.asarray(proposed, dtype=np.float64) - observed_offset
        clamped = bool(np.any(np.abs(proposed - executed) > max_joint_delta_rad + 1e-12))
        clamped_run = clamped_run + 1 if clamped else 0
        clamped_frames += int(clamped)
        if clamped_run > max_clamped_frames:
            raise LabelError(f"Sustained joint jump (IK branch switch) at {target.state} waypoint {target.waypoint_index}")
        q_cmd = limit_joint_delta(executed, proposed, max_joint_delta_rad)

        if target.pen_down:
            labelled_tip, _ = forward(q_cmd + observed_offset)
            pen_down_errors.append(float(np.linalg.norm(np.asarray(labelled_tip) - desired_next)))
        upcoming = [t.position for t in targets[index + 1 : index + 1 + lookahead]]
        while len(upcoming) < lookahead:
            upcoming.append(upcoming[-1] if upcoming else target.position)
        recorder.append(
            stroke_id=target.stroke_id,
            waypoint_index=target.waypoint_index,
            simulation_time_s=time_s,
            joint_positions_rad=q_meas,
            joint_velocities_rad_s=(q_meas - q_prev_meas) / dt_s,
            pen_tip_position_m=tip,
            pen_tip_rotation_matrix=rotation,
            current_target_m=target.position,
            upcoming_targets_m=np.asarray(upcoming),
            pen_down=target.pen_down,
            state=target.state,
            commanded_joint_positions_rad=q_cmd.copy(),
            path_window_m=path_window(positions, index),
            previous_command_rad=executed.copy(),
        )
        frames += 1
        if noise:
            perturbation = decay * perturbation + kick * noise["rng"].standard_normal(perturbation.shape)
        executed = q_cmd + perturbation
        if offset_kick:
            offset = offset_decay * offset + offset_kick * noise["rng"].standard_normal(offset.shape)
        q_prev_meas, q_meas = q_meas, executed + offset
        time_s += dt_s
        if s_next >= arc[-1]:
            hold_frames -= 1
            if hold_frames < 0:
                break

    errors = np.asarray(pen_down_errors) if pen_down_errors else np.zeros(1)
    stats = {
        "frames": float(frames),
        "duration_s": float(time_s),
        "clamped_fraction": float(clamped_frames / max(frames, 1)),
        "pen_down_tip_error_rmse_m": float(np.sqrt(np.mean(errors**2))),
        "pen_down_tip_error_max_m": float(errors.max()),
    }
    if stats["pen_down_tip_error_max_m"] > max_tip_error_m:
        raise LabelError(f"Pen-down tip error {stats['pen_down_tip_error_max_m'] * 1e3:.3f} mm exceeds limit")
    return stats
