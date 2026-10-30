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

With `gaps` (#46), the labeller also tracks which pen-down waypoints have ink
near them and goes back to redraw a gap: a stretch of path behind the pen with
no ink within `cover_tol_m`, at least `min_gap_m` long. Ink fails where the
pen lifts off (INK_CONTACT_M) or where an injected skip (act_features.SkipSchedule)
says so. The redraw runs back along the line past the gap, then forward again
to where the pen left off. The teacher's progress stays frozen meanwhile; the
recorded observation shows the waypoint just ahead of the pen
(act_features.observed_index, as the closed-loop runner does), so the numbers
look like normal drawing and only a camera view of the page tells a policy
that a gap needs redrawing.

With `policy` (DAgger), a learned policy drives the arm instead of the labels:
each tick it is shown the same observation the recording stores and its joint
step is executed, while the label is still the IK teacher's command from that
state. The recording then covers the states the policy itself drifts into.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from act_features import (
    INK_CONTACT_M,
    SkipSchedule,
    environment_state,
    observation_state,
    observed_index,
    path_window,
    project_on_path,
)
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


# Teacher modes recorded per frame: drawing normally, going back to a gap, coming forward again.
MODE_NORMAL, MODE_BACK, MODE_FORWARD = 0, 1, 2
DEFAULT_GAPS = {
    "skip_spacing_m": 0.06,
    "skip_length_m": (0.003, 0.008),
    "cover_tol_m": 0.001,
    "min_gap_m": 0.0015,
    "detect_after_m": 0.001,
    "lookback_m": 0.015,
    "margin_m": 0.0005,
}


class _Coverage:
    """Which parts of the pen-down DRAW path have ink within `tol` of them, and the gaps behind the pen.

    Tracked on the path resampled every `step_m` (waypoints are up to 1.5 mm apart, as
    coarse as the gaps that matter)."""

    def __init__(self, targets, positions: np.ndarray, arc: np.ndarray, tol_m: float, step_m: float = 0.00025):
        self.tol = float(tol_m)
        self.arc = np.arange(0.0, float(arc[-1]) + step_m, step_m)
        self.points = np.stack([np.interp(self.arc, arc, positions[:, axis]) for axis in range(2)], axis=1)
        # A sample belongs to the segment ending at the first waypoint at or after it.
        ahead = np.minimum(np.searchsorted(arc, self.arc, side="left"), len(targets) - 1)
        before = np.maximum(ahead - 1, 0)
        draw = np.asarray([t.pen_down and t.state.startswith("DRAW") for t in targets])
        stroke = np.asarray([t.stroke_id for t in targets])
        self.drawable = draw[ahead] & draw[before] & (stroke[ahead] == stroke[before])
        self.stroke = stroke[ahead]
        self.covered = ~self.drawable
        self.attempted: set[int] = set()

    def ink(self, a: np.ndarray, b: np.ndarray, s: float) -> None:
        """Mark samples within tol of the inked segment a-b, searching around progress s."""
        lo = int(np.searchsorted(self.arc, s - 0.05))
        hi = int(np.searchsorted(self.arc, s + 0.01))
        p = self.points[lo:hi]
        d = b[:2] - a[:2]
        t = np.clip(((p - a[:2]) @ d) / max(float(d @ d), 1e-18), 0.0, 1.0)
        self.covered[lo:hi] |= np.linalg.norm(a[:2] + t[:, None] * d - p, axis=1) <= self.tol

    def gap_behind(self, s: float, stroke: int, min_gap_m: float, detect_after_m: float, lookback_m: float):
        """(start, end) arc of the earliest unattempted gap in this stroke behind progress s, or None."""
        lo = int(np.searchsorted(self.arc, s - lookback_m, side="left"))
        hi = int(np.searchsorted(self.arc, s - detect_after_m, side="right"))
        if hi <= lo:
            return None
        open_ = self.drawable[lo:hi] & (self.stroke[lo:hi] == stroke) & ~self.covered[lo:hi]
        edges = np.flatnonzero(np.diff(np.r_[0, open_.astype(np.int8), 0]))
        for first, last in zip(edges[::2] + lo, edges[1::2] + lo - 1):
            if self.arc[last] - self.arc[first] >= min_gap_m and int(first) not in self.attempted:
                self.attempted.add(int(first))
                return float(self.arc[first]), float(self.arc[last])
        return None

    def stroke_start_arc(self, stroke: int) -> float:
        same = np.flatnonzero(self.drawable & (self.stroke == stroke))
        return float(self.arc[same[0]]) if len(same) else 0.0

    def uncovered_fraction(self) -> float:
        return float(1.0 - self.covered[self.drawable].mean()) if self.drawable.any() else 0.0


def phase_speed(state: str, speeds: Dict[str, float]) -> float:
    for prefix, speed in speeds.items():
        if state.startswith(prefix):
            return float(speed)
    raise KeyError(f"No speed configured for state {state}")


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
    reached_tolerance_m: float = 0.0015,
    max_duration_factor: float = 3.0,
    gaps: Dict | None = None,
    policy: Dict | None = None,
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

    `noise` = {"rng": np.random.Generator, "sigma_rad": float, "correlation_s": float,
    optional "offset_sigma_rad": float, "offset_correlation_s": float} adds
    Ornstein-Uhlenbeck perturbations to every executed command and, if given,
    to the measured-versus-executed servo offset.

    `gaps` = {"rng": np.random.Generator, optional DEFAULT_GAPS keys} turns on ink
    tracking, injected skips and gap redraws; each frame then also records its
    ink_skip flag and teacher mode.

    `policy` = {"act": f(state, env, ink_skip) -> joint step, "share": float,
    "rng": np.random.Generator} executes the policy's step instead of the label on
    a `share` of ticks (DAgger); the label stays the teacher's.
    """
    speeds = dict(DEFAULT_SPEEDS_M_S if speeds_m_s is None else speeds_m_s)
    targets = flatten_desired_targets(phases)
    positions = np.asarray([t.position for t in targets], dtype=np.float64)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
    target_speed = np.asarray([phase_speed(t.state, speeds) for t in targets], dtype=np.float64)
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
    coverage = skips = redo = last_ink_tip = None
    mode = MODE_NORMAL
    redos = policy_steps = 0
    if gaps is not None:
        gap_config = {**DEFAULT_GAPS, **gaps}
        coverage = _Coverage(targets, positions, arc, gap_config["cover_tol_m"])
        skips = SkipSchedule(gap_config["rng"], gap_config["skip_spacing_m"], gap_config["skip_length_m"])
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
        # What a policy is shown: the teacher's target, or the waypoint just ahead of a pen that is back
        # on the path behind it (a redraw).
        seen_index = observed_index(positions, arc, index, tip)
        seen = targets[seen_index]

        skipped = False
        if coverage is not None:
            # Ink as the history renderer sees it: pen down, in contact, and not a skip.
            contact = bool(seen.pen_down) and float(tip[2] - seen.position[2]) < INK_CONTACT_M
            inked = skips.inks(contact, tip)
            skipped = contact and not inked
            if inked:
                coverage.ink(np.asarray(tip) if last_ink_tip is None else last_ink_tip, np.asarray(tip), s)
            last_ink_tip = np.asarray(tip, dtype=np.float64) if inked else None
            if mode == MODE_NORMAL and target.pen_down and target.state.startswith("DRAW"):
                gap = coverage.gap_behind(s, target.stroke_id, gap_config["min_gap_m"],
                                          gap_config["detect_after_m"], gap_config["lookback_m"])
                if gap is not None:
                    back = max(gap[0] - gap_config["margin_m"], coverage.stroke_start_arc(target.stroke_id))
                    redo = {"back": back, "resume": s, "aim": s,
                            "deadline": time_s + 2.0 + 3.0 * 2.0 * (s - back) / max(speed, 1e-9)}
                    mode, redos = MODE_BACK, redos + 1
            elif mode != MODE_NORMAL and time_s > redo["deadline"]:
                mode, previous_aim = MODE_NORMAL, None  # give the gap up rather than stall
            if mode == MODE_BACK and redo["aim"] <= redo["back"] and \
                    np.linalg.norm(np.asarray(tip) - point_at(redo["back"])) <= reached_tolerance_m:
                mode = MODE_FORWARD
            if mode == MODE_FORWARD and redo["aim"] >= redo["resume"] and \
                    np.linalg.norm(np.asarray(tip) - point_at(redo["resume"])) <= reached_tolerance_m:
                mode, previous_aim = MODE_NORMAL, None

        if mode == MODE_NORMAL:
            s_next = min(s + speed * dt_s, float(arc[-1]))
            desired_next = point_at(s_next)
            previous_aim = (s_next, desired_next)
        else:
            # Redraw: back along the line past the gap, then forward to where the pen left off.
            step = speed * dt_s if mode == MODE_FORWARD else -speed * dt_s
            redo["aim"] = float(np.clip(redo["aim"] + step, redo["back"], redo["resume"]))
            s_next = s
            desired_next = point_at(redo["aim"])

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
        window = path_window(positions, seen_index)
        upcoming = [t.position for t in targets[seen_index + 1 : seen_index + 1 + lookahead]]
        while len(upcoming) < lookahead:
            upcoming.append(upcoming[-1] if upcoming else seen.position)
        recorder.append(
            stroke_id=target.stroke_id,
            waypoint_index=target.waypoint_index,
            simulation_time_s=time_s,
            joint_positions_rad=q_meas,
            joint_velocities_rad_s=(q_meas - q_prev_meas) / dt_s,
            pen_tip_position_m=tip,
            pen_tip_rotation_matrix=rotation,
            current_target_m=seen.position,
            upcoming_targets_m=np.asarray(upcoming),
            pen_down=seen.pen_down,
            state=target.state,
            commanded_joint_positions_rad=q_cmd.copy(),
            path_window_m=window,
            previous_command_rad=executed.copy(),
            **({"ink_skip": skipped, "teacher_mode": mode} if coverage is not None else {}),
        )
        frames += 1
        if noise:
            perturbation = decay * perturbation + kick * noise["rng"].standard_normal(perturbation.shape)
        if policy is not None and policy["rng"].random() < policy["share"]:
            observed = observation_state(q_meas, (q_meas - q_prev_meas) / dt_s, executed)[0]
            env = environment_state(tip, seen.position, window[None], [[float(seen.pen_down)]])[0]
            step = np.asarray(policy["act"](observed, env, skipped), dtype=np.float64)
            policy_steps += 1
            executed = limit_joint_delta(executed, executed + step, max_joint_delta_rad)
        else:
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
    if policy is not None:
        stats["policy_share"] = float(policy_steps / max(frames, 1))
    if coverage is not None:
        stats.update(redos=float(redos), skips=float(skips.skips),
                     uncovered_fraction=coverage.uncovered_fraction())
    if stats["pen_down_tip_error_max_m"] > max_tip_error_m:
        raise LabelError(f"Pen-down tip error {stats['pen_down_tip_error_max_m'] * 1e3:.3f} mm exceeds limit")
    return stats
