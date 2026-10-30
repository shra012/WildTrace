"""Draw one trajectory on the physical xArm 7 with a trained ACT policy.

The policy is the one the Isaac runner serves (scripts/act_policy_server.py): joint
state plus the next 2 cm of pen path in, one joint step out, queried at 60 Hz at half
scale. Features come from src/act_features.py, so the model sees what it was trained on.
The path is built like the twin's (config/xarm7_drawing_real.yaml) but placed at the
paper centre measured on the real arm (config/xarm7_hardware.yaml).

Run it with the ACT environment (lerobot, torch, xarm-python-sdk):

    isaac-sim/.venv-act/Scripts/python.exe isaac-sim/scripts/run_xarm7_act_hardware.py --offline
    isaac-sim/.venv-act/Scripts/python.exe isaac-sim/scripts/run_xarm7_act_hardware.py --execute --dry-lift-mm 15
    isaac-sim/.venv-act/Scripts/python.exe isaac-sim/scripts/run_xarm7_act_hardware.py --execute

--offline never moves the arm. It rolls the policy out against the controller's own
forward kinematics with ideal joint tracking and reports path error, pen height and how
close each joint gets to its limit. Do that first. --dry-lift-mm raises the whole path so
the pen never touches the paper. A real run needs the arm at the policy's start pose
(act_start_joints_deg); the script joint-moves there first after a tip-height check.

--policy smolvla --checkpoint isaac-sim/models/smolvla_camera_v3 (or any train_smolvla
pretrained_model) runs a SmolVLA checkpoint instead. It cannot sample a chunk inside one
60 Hz tick, so it samples the next chunk in the background while the current one plays,
two ticks per action; a tick with no action ready holds still, and the run log counts
those holds.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from act_features import PathProgress, environment_state, observation_state, path_arc_length, path_window  # noqa: E402
from coordinate_mapper import interpolate_segment, map_trajectory_to_plane  # noqa: E402
from drawing_state_machine import MotionTarget, Phase, build_motion_sequence, flatten_desired_targets  # noqa: E402
from run_xarm7_hardware import RunLog, SafeStop, _jsonable  # noqa: E402
from trajectory_loader import load_trajectory  # noqa: E402

TWIN_CONFIG = PROJECT / "config" / "xarm7_drawing_real.yaml"
# xArm 7 joint limits in degrees (UFACTORY spec).
LIMITS_DEG = np.array([[-360, 360], [-118, 120], [-360, 360], [-11, 225], [-360, 360], [-97, 180], [-360, 360]], dtype=np.float64)
LIMIT_MARGIN_DEG = 3.0
CONTROL_HZ = 60.0
ACTION_SCALE = 0.5  # a 30 fps policy queried at 60 Hz, as in the Isaac runner
MAX_STEP_RAD = 0.006  # per 60 Hz tick, about 0.36 rad/s
MAX_TICK_S = 0.10
REAL_CHECK_TICKS = 6  # query the real pose at 10 Hz
MAX_LAG_M = 0.005  # real tip further than this from the commanded tip stops the run
# Values the SDK's report thread keeps up to date (3-11 Hz); reading them costs no request to the arm.
REPORT_FIELDS = ("angles", "position", "realtime_joint_speeds", "joints_torque", "currents", "temperatures",
                 "voltages", "error_code", "warn_code", "state", "mode")
ROBOT_CONFIG_FIELDS = ("version", "tcp_offset", "tcp_load", "collision_sensitivity", "teach_sensitivity",
                       "gravity_direction", "tcp_speed_limit", "joint_speed_limit")


def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


class UrdfTipFK:
    """Pen-tip forward kinematics from the calibrated real-arm URDF, in numpy.

    The SDK's state reports refresh at 3-11 Hz and a controller FK query takes ~10 ms, so neither
    fits a 60 Hz loop. Checked against the controller's FK at start-up (run_xarm7_act_hardware).
    """

    def __init__(self, urdf: Path, tip_link: str = "link_tcp") -> None:
        import xml.etree.ElementTree as ET

        joints = {j.find("child").get("link"): j for j in ET.parse(urdf).getroot().findall("joint")}
        chain, link = [], tip_link
        while link in joints:
            joint = joints[link]
            origin = joint.find("origin")
            xyz = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
            rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
            transform = np.eye(4)
            transform[:3, :3], transform[:3, 3] = _rpy(*rpy), xyz
            chain.append((transform, joint.get("type") == "revolute"))
            link = joint.find("parent").get("link")
        self.chain = chain[::-1]

    def tip(self, q_rad: np.ndarray) -> np.ndarray:
        pose, k = np.eye(4), 0
        for transform, revolute in self.chain:
            pose = pose @ transform
            if revolute:
                c, s = np.cos(q_rad[k]), np.sin(q_rad[k])
                pose = pose @ np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                k += 1
        return pose[:3, 3]


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_targets(
    hardware: dict, twin: dict, home_tip: np.ndarray, lift_m: float, skip: int, limit: int | None, floor_below_m: float = 0.0002
):
    drawing = twin["drawing"]
    trajectory_path = Path(hardware["trajectory_path"])
    trajectory = load_trajectory(trajectory_path if trajectory_path.is_absolute() else PROJECT / trajectory_path)
    paper = np.asarray(hardware["paper_center_mm"], dtype=np.float64) / 1000.0
    # The lift moves the draw plane and every height built from it, before the approach from the
    # pen's real position is added, so the path stays continuous from where the pen is.
    pen_down = float(paper[2]) + lift_m
    pen_up = pen_down + float(drawing["pen_up_z_m"]) - float(drawing["pen_down_z_m"])
    approach = pen_down + float(drawing["approach_height_m"]) - float(drawing["pen_down_z_m"])
    mapped = map_trajectory_to_plane(
        trajectory,
        center_xy=paper[:2].tolist(),
        size_xy=drawing["surface_size_xy_m"],
        flip_image_y=bool(drawing["flip_image_y"]),
        max_step=float(drawing["max_cartesian_step_m"]),
        smoothing_strength=float(drawing["smoothing_strength"]),
        corner_angle_degrees=float(drawing["corner_angle_degrees"]),
    )
    strokes = mapped["strokes"][skip:]
    if limit is not None:
        strokes = strokes[:limit]
    if not strokes:
        raise SafeStop("no strokes left after --skip-strokes / --max-strokes")
    phases = build_motion_sequence(
        strokes,
        pen_down_z=pen_down,
        pen_up_z=pen_up,
        approach_height=approach,
        max_cartesian_step=float(drawing["max_cartesian_step_m"]),
        corner_angle_degrees=float(drawing["corner_angle_degrees"]),
        corner_densify_window=int(drawing["corner_window_points"]),
        corner_densify_factor=int(drawing["corner_densify_factor"]),
    )
    first = phases[0]
    final = first.targets[-1]
    points = interpolate_segment(home_tip, final.position, float(drawing["max_cartesian_step_m"]))[1:]
    phases[0] = Phase(first.state, [MotionTarget(first.state, p, final.stroke_id, i, False) for i, p in enumerate(points)])
    targets = flatten_desired_targets(phases)
    positions = np.asarray([t.position for t in targets], dtype=np.float64)
    return {
        "positions": positions,
        "pen_down": np.asarray([bool(t.pen_down) for t in targets]),
        "states": [t.state for t in targets],
        "strokes": len(strokes),
        "paper_xy": paper[:2],
        "pen_down_z": pen_down,
        # The recorded touch height is a hard floor whatever the offset: below it the pen is pushed into the paper.
        "floor_z": float(paper[2]) - floor_below_m,
        "approach_z": approach,
        "size_xy": np.asarray(drawing["surface_size_xy_m"], dtype=np.float64),
    }


class OfflineArm:
    """Ideal joint tracking; the controller's own kinematics give the pen tip. Never moves the arm."""

    def __init__(self, api: Any, q0_rad: np.ndarray) -> None:
        self.api, self.q, self.q_prev = api, q0_rad.copy(), q0_rad.copy()

    def tip_m(self, q_rad: np.ndarray) -> np.ndarray:
        code, pose = self.api.get_forward_kinematics(np.degrees(q_rad).tolist(), input_is_radian=False, return_is_radian=False)
        if code != 0:
            raise SafeStop(f"forward kinematics failed with code {code}")
        return np.asarray(pose[:3], dtype=np.float64) / 1000.0

    def read(self, dt: float):
        qd = (self.q - self.q_prev) / dt
        return self.q.copy(), qd, self.tip_m(self.q)

    def send(self, q_rad: np.ndarray) -> None:
        self.q_prev, self.q = self.q, q_rad.copy()


def move_to_start(api: Any, log: RunLog, start_deg: np.ndarray, floor_z_m: float, speed_deg_s: float) -> None:
    """Position-mode joint move to the policy's start pose after a tip-height check along the path."""
    api.motion_enable(True)
    api.set_mode(0)
    api.set_state(0)
    time.sleep(0.5)
    current = np.asarray(api.get_servo_angle()[1][:7], dtype=np.float64)
    for fraction in np.linspace(0.0, 1.0, 41):
        angles = current + (start_deg - current) * fraction
        code, pose = api.get_forward_kinematics(angles.tolist(), input_is_radian=False, return_is_radian=False)
        if code != 0 or pose[2] / 1000.0 < floor_z_m:
            raise SafeStop(f"joint path to the start pose dips to z={pose[2] if code == 0 else 'n/a'} mm at t={fraction:.2f}")
    log.event("start_move", from_deg=current, to_deg=start_deg, speed_deg_s=speed_deg_s)
    code = api.set_servo_angle(angle=start_deg.tolist(), speed=speed_deg_s, mvacc=50, wait=True, timeout=180)
    if code != 0:
        raise SafeStop(f"start pose move failed with code {code}")
    gap = float(np.max(np.abs(np.asarray(api.get_servo_angle()[1][:7]) - start_deg)))
    log.event("start_reached", max_joint_gap_deg=round(gap, 3))
    if gap > 1.0:
        raise SafeStop(f"arm is {gap:.1f} deg from the start pose")


def lift_and_hold(api: Any, log: RunLog, lift_mm: float = 60.0) -> None:
    """Leave servo mode, then raise the pen straight up. Never raises."""
    try:
        api.clean_warn()
        api.clean_error()
        api.motion_enable(True)
        api.set_mode(0)
        api.set_state(0)
        time.sleep(0.4)
        x, y, z, roll, pitch, yaw = api.get_position()[1]
        code = api.set_position(x=x, y=y, z=z + lift_mm, roll=roll, pitch=pitch, yaw=yaw, speed=10, mvacc=50, wait=True, timeout=30)
        log.event("lift_and_hold", code=code, from_z_mm=round(z, 2), to_z_mm=round(z + lift_mm, 2))
    except Exception as exc:  # noqa: BLE001
        log.event("lift_failed", error=str(exc))


class HardwareArm:
    """Servo-mode streaming with ideal-tracking observations.

    The SDK's state reports refresh at 3-11 Hz, too slow and bursty for a 60 Hz policy (stale
    readings made it over-correct), so the policy sees the joints it last commanded and the tip
    from local FK of them, exactly as in the offline rollout. real_tip_m() queries the arm for the
    safety checks every REAL_CHECK_TICKS ticks.
    """

    def __init__(self, api: Any, fk: UrdfTipFK, q0_rad: np.ndarray) -> None:
        self.api, self.fk = api, fk
        self.q, self.q_prev = q0_rad.copy(), q0_rad.copy()

    def read(self, dt: float):
        return self.q.copy(), (self.q - self.q_prev) / dt, self.fk.tip(self.q)

    def send(self, q_rad: np.ndarray) -> None:
        code = self.api.set_servo_angle_j(np.degrees(q_rad).tolist(), is_radian=False)
        if code != 0:
            raise SafeStop(f"set_servo_angle_j returned {code}")
        self.q_prev, self.q = self.q, q_rad.copy()

    def real_tip_m(self) -> np.ndarray:
        code, pose = self.api.get_position()
        if code != 0:
            raise SafeStop(f"get_position returned {code}")
        return np.asarray(pose[:3], dtype=np.float64) / 1000.0

    def report(self) -> dict[str, Any]:
        """The arm's last reported joints, pose, torques, currents, temperatures and codes."""
        return {name: getattr(self.api, name, None) for name in REPORT_FIELDS}


def _fence_check(plan: dict, tip: np.ndarray, lift_m: float) -> str | None:
    cx, cy = plan["paper_xy"]
    hx, hy = plan["size_xy"] / 2.0 + 0.04
    floor = max(plan["pen_down_z"] - 0.001, plan["floor_z"])
    if not (cx - hx <= tip[0] <= cx + hx and cy - hy <= tip[1] <= cy + hy):
        return f"tip xy {tip[:2].round(4).tolist()} left the paper fence"
    if tip[2] < floor:
        return f"tip z {tip[2] * 1000:.2f} mm is below the floor {floor * 1000:.2f} mm"
    if tip[2] > plan["approach_z"] + 0.05:
        return f"tip z {tip[2] * 1000:.1f} mm is above the approach height"
    return None


def rollout(arm, runner, plan: dict, offset: np.ndarray, log: RunLog, csv_path: Path, args, lift_m: float) -> dict:
    positions, flags = plan["positions"], plan["pen_down"]
    arc = path_arc_length(positions)
    progress_tracker = PathProgress(positions, arc, stall_s=args.stall_release_s)
    dt = 1.0 / CONTROL_HZ
    q, qd, tip = arm.read(dt)
    prev_cmd = q.copy()
    index = 0
    last_tick = time.perf_counter()
    started = last_tick
    done_since = None
    errors, tick_ms, steps, progress, lags, real_errors, reports = [], [], [], [], [], [], []
    lo = np.radians(LIMITS_DEG[:, 0] + LIMIT_MARGIN_DEG)
    hi = np.radians(LIMITS_DEG[:, 1] - LIMIT_MARGIN_DEG)
    handle = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(handle)
    writer.writerow(["t_s", "index", "pen_down", "tip_x", "tip_y", "tip_z", "tgt_x", "tgt_y", "tgt_z", "err_mm", "max_step_rad", "real_lag_mm"] + [f"q{i}_deg" for i in range(1, 8)])
    outcome = "unfinished"
    try:
        for tick in range(int(args.max_seconds * CONTROL_HZ)):
            now = time.perf_counter()
            if args.execute and tick and now - last_tick > MAX_TICK_S:
                raise SafeStop(f"control tick took {(now - last_tick) * 1e3:.0f} ms")
            real_dt = dt if (not tick or not args.execute) else max(now - last_tick, 1e-3)
            last_tick = now
            q, qd, tip = arm.read(real_dt)
            problem = _fence_check(plan, tip, lift_m)
            if problem:
                raise SafeStop(problem)
            lag_mm = ""
            if args.execute and tick % REAL_CHECK_TICKS == 0:
                real = arm.real_tip_m()
                problem = _fence_check(plan, real, lift_m)
                if problem:
                    raise SafeStop(f"real arm: {problem}")
                lag = float(np.linalg.norm(real - tip))
                lags.append(lag)
                # Measured, not commanded: the real tip against the path target the policy was steering to.
                real_error = float(np.linalg.norm(real - positions[index]))
                if flags[index]:
                    real_errors.append(real_error)
                reports.append(dict(t_s=round(now - started, 4), tick=tick, index=index, pen_down=bool(flags[index]),
                                    real_tip_mm=(real * 1000).round(3), real_err_mm=round(real_error * 1e3, 3),
                                    lag_mm=round(lag * 1e3, 3), **arm.report()))
                lag_mm = f"{lag * 1e3:.3f}"
                if lag > MAX_LAG_M:
                    raise SafeStop(f"real tip is {lag * 1e3:.1f} mm from the commanded tip")
            if np.any(q < lo) or np.any(q > hi):
                joint = int(np.argmax((q < lo) | (q > hi))) + 1
                raise SafeStop(f"joint {joint} at {np.degrees(q[joint - 1]):.1f} deg is inside the {LIMIT_MARGIN_DEG} deg limit margin")
            index = progress_tracker.update(index, tip, now - started)
            target = positions[index]
            error = float(np.linalg.norm(tip - target))
            if flags[index] and error > args.max_error_mm / 1000.0:
                raise SafeStop(f"pen-down tip error {error * 1e3:.2f} mm exceeds {args.max_error_mm} mm")
            window = path_window(positions, index)
            t0 = time.perf_counter()
            state = observation_state(q - offset, qd, prev_cmd - offset)[0]
            env = environment_state(tip, target, window[None], [[float(flags[index])]])[0]
            predicted = np.asarray(runner.act(state, env), dtype=np.float64)
            tick_ms.append((time.perf_counter() - t0) * 1e3)
            if not np.isfinite(predicted).all():
                raise SafeStop("ACT produced a non-finite joint step")
            step = ACTION_SCALE * predicted
            if np.max(np.abs(step)) > MAX_STEP_RAD:
                log.event("step_guard", tick=tick, step_rad=step.round(5), qd_rad_s=qd.round(4), err_mm=round(error * 1e3, 3), index=index)
                raise SafeStop(f"ACT joint step {np.max(np.abs(step)):.4f} rad exceeds {MAX_STEP_RAD} rad per tick")
            cmd = prev_cmd + step
            arm.send(cmd)
            prev_cmd = cmd
            steps.append(float(np.max(np.abs(step))))
            if flags[index]:
                errors.append(error)
            writer.writerow([f"{now - started:.4f}", index, int(flags[index]), *tip.round(5), *target.round(5), f"{error * 1e3:.3f}", f"{steps[-1]:.5f}", lag_mm, *np.degrees(q).round(3)])
            if index >= len(positions) - 1:
                done_since = done_since or now
                if np.linalg.norm(tip - positions[-1]) < 0.002 or now - done_since > 3.0:
                    outcome = "finished"
                    break
            if tick % 300 == 0:
                # No disk I/O in the 60 Hz loop: an fsync here once stalled a tick for 140 ms.
                progress.append(dict(tick=tick, index=index, of=len(positions), tip_mm=(tip * 1000).round(2), err_mm=round(error * 1e3, 3), q_deg=np.degrees(q).round(1)))
            if args.execute:
                # Sleep, don't spin: a busy-wait holds the GIL and starves the SDK thread that reads the
                # arm's state, which then refreshed only every 200 ms. Spin only the last 0.5 ms.
                wake = last_tick + dt
                while True:
                    left = wake - time.perf_counter()
                    if left <= 0.0:
                        break
                    if left > 0.001:
                        time.sleep(left - 0.0005)
        else:
            outcome = "timeout"
    finally:
        handle.close()
        for item in progress:
            log.event("progress", **item)
        if reports:
            with (csv_path.parent / "robot_report.jsonl").open("w", encoding="utf-8") as stream:
                for item in reports:
                    stream.write(json.dumps(item, default=_jsonable) + "\n")
    log.event("stall_releases", count=progress_tracker.releases)
    if hasattr(runner, "holds"):
        log.event("policy_chunks", underruns=runner.underruns, holds=runner.holds, ticks=runner.tick)
    return {"outcome": outcome, "ticks": len(steps), "errors": np.asarray(errors), "tick_ms": np.asarray(tick_ms), "steps": np.asarray(steps), "lags": np.asarray(lags),
            "real_errors": np.asarray(real_errors), "last_q": q}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(PROJECT / "config" / "xarm7_hardware.yaml"))
    parser.add_argument("--checkpoint", default=str(PROJECT / "models" / "act_real_kinematic_v2"))
    parser.add_argument("--trajectory", default=None)
    parser.add_argument("--robot-ip", default=None)
    parser.add_argument("--offline", action="store_true", help="Roll out against the controller's FK. Never moves the arm.")
    parser.add_argument("--execute", action="store_true", help="Move the real arm.")
    parser.add_argument(
        "--z-offset-mm",
        "--dry-lift-mm",
        dest="z_offset_mm",
        type=float,
        default=0.0,
        help="Shift the whole path up. 15 or more is a dry run above the paper; 0.3 to 0.5 draws with a lighter pen.",
    )
    parser.add_argument("--start-speed-deg-s", type=float, default=8.0)
    parser.add_argument(
        "--floor-below-touch-mm",
        type=float,
        default=0.2,
        help="Stop if the commanded or real pen tip goes this far below the recorded touch height (paper_center_mm z)",
    )
    parser.add_argument("--max-strokes", type=int, default=None)
    parser.add_argument("--skip-strokes", type=int, default=0)
    parser.add_argument("--max-error-mm", type=float, default=3.0, help="Pen-down tip error that stops the run")
    parser.add_argument("--stall-release-s", type=float, default=0.0,
                        help="Step the target one waypoint ahead when the pen hovers within 1.5 mm of the path without "
                             "progressing for this long (act_features.PathProgress); 0 = off, the previous behaviour")
    parser.add_argument("--max-seconds", type=float, default=600.0)
    parser.add_argument("--policy", choices=["act", "smolvla"], default="act")
    parser.add_argument("--flow-steps", type=int, default=3, help="smolvla: flow-matching steps per chunk")
    parser.add_argument("--start-joints-deg", type=float, nargs=7, default=None, help="Real-arm joints (deg) for the policy's start pose")
    args = parser.parse_args()
    if args.offline == args.execute:
        parser.error("choose exactly one of --offline or --execute")
    if args.z_offset_mm < -1 or args.z_offset_mm > 40:
        parser.error("--z-offset-mm must be between -1 and 40")

    log = RunLog(PROJECT / "outputs" / "hardware_runs" / (datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.policy}"))
    outcome, api, motion_started = "started", None, False
    try:
        hardware = _load_yaml(Path(args.config))
        if args.trajectory:
            hardware["trajectory_path"] = args.trajectory
        twin = _load_yaml(TWIN_CONFIG)
        ip = args.robot_ip or str(hardware["robot_ip"])
        home_model = np.asarray(twin["robot"]["home_joint_positions_rad"], dtype=np.float64)
        start_deg = np.asarray(args.start_joints_deg if args.start_joints_deg else hardware["act_start_joints_deg"], dtype=np.float64)
        log.event("run_start", argv=sys.argv, ip=ip, checkpoint=args.checkpoint, start_joints_deg=start_deg, execute=args.execute)

        from xarm.wrapper import XArmAPI

        api = XArmAPI(ip, is_radian=False)
        if not api.connected:
            raise SafeStop(f"could not connect to {ip}")
        if api.get_err_warn_code()[1][0] not in (0, None):
            raise SafeStop(f"controller error {api.get_err_warn_code()[1]}")
        log.event("robot_config", **{name: getattr(api, name, None) for name in ROBOT_CONFIG_FIELDS})
        # The policy lives in the twin's joint convention; the real arm may sit 360 deg away on a joint.
        offset = np.radians(start_deg) - home_model
        offset = 2.0 * np.pi * np.round(offset / (2.0 * np.pi))
        log.event("joint_offset_deg", offset_deg=np.degrees(offset))
        if not np.allclose(np.radians(start_deg) - offset, home_model, atol=1e-3):
            raise SafeStop("act_start_joints_deg is not the twin home pose modulo 360 degrees")

        code, pose = api.get_forward_kinematics(start_deg.tolist(), input_is_radian=False, return_is_radian=False)
        home_tip = np.asarray(pose[:3]) / 1000.0
        fk = UrdfTipFK(PROJECT / "assets" / "xarm7" / "xarm7_real.urdf")
        fk_gap = float(np.linalg.norm(fk.tip(np.radians(start_deg)) - home_tip))
        log.event("urdf_fk_check", gap_mm=round(fk_gap * 1e3, 3))
        if fk_gap > 0.0005:
            raise SafeStop(f"URDF FK differs from the controller's by {fk_gap * 1e3:.2f} mm at the start pose")
        plan = build_targets(
            hardware, twin, home_tip, args.z_offset_mm / 1000.0, args.skip_strokes, args.max_strokes, args.floor_below_touch_mm / 1000.0
        )
        log.event("plan_ready", targets=len(plan["positions"]), strokes=plan["strokes"], home_tip_mm=(home_tip * 1000).round(2),
                  pen_down_z_mm=round(plan["pen_down_z"] * 1000, 3), lift_mm=args.z_offset_mm)

        from act_policy_server import ActRunner, SmolVlaRunner

        if args.policy == "smolvla":
            runner = SmolVlaRunner(Path(args.checkpoint).resolve(), "cuda", 1, "zero", 1, args.flow_steps,
                                   # a 30 fps action covers two 60 Hz ticks at ACTION_SCALE each
                                   async_chunks=True, ticks_per_action=round(CONTROL_HZ / 30.0), hold_on_underrun=True)
            warm_up = lambda: runner.chunk(np.zeros(21), np.zeros(37))  # noqa: E731
        else:
            runner = ActRunner(Path(args.checkpoint).resolve(), "cuda", "first", 0.01)
            warm_up = lambda: runner.act(np.zeros(21), np.zeros(37))  # noqa: E731
        # The first CUDA calls take over a second; do them before the 60 Hz loop starts.
        for _ in range(8):
            warm_up()
        runner.reset()
        csv_path = log.directory / "act_ticks.csv"

        if args.offline:
            arm = OfflineArm(api, np.radians(start_deg))
            result = rollout(arm, runner, plan, offset, log, csv_path, args, args.z_offset_mm / 1000.0)
        else:
            if hardware.get("paper_center_mm") is None:
                raise SafeStop("paper_center_mm is null; refusing to move")
            move_to_start(api, log, start_deg, plan["pen_down_z"] + 0.04, args.start_speed_deg_s)
            for remaining in range(5, 0, -1):
                log.event("countdown", seconds_left=remaining)
                time.sleep(1.0)
            api.set_mode(1)
            api.set_state(0)
            time.sleep(0.3)
            motion_started = True
            arm = HardwareArm(api, fk, np.radians(np.asarray(api.get_servo_angle()[1][:7], dtype=np.float64)))
            result = rollout(arm, runner, plan, offset, log, csv_path, args, args.z_offset_mm / 1000.0)
            lift_and_hold(api, log)

        errors = result["errors"] * 1000.0
        ticks = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        joints = np.array([[float(r[f"q{i}_deg"]) for i in range(1, 8)] for r in ticks])
        tip_z = np.array([float(r["tip_z"]) for r in ticks]) * 1000.0
        summary = {
            "outcome": result["outcome"],
            "ticks": result["ticks"],
            "seconds": round(result["ticks"] / CONTROL_HZ, 1),
            "pen_down_error_mm": {"rmse": round(float(np.sqrt(np.mean(errors**2))), 3), "max": round(float(errors.max()), 3)} if len(errors) else None,
            "tip_z_min_mm": round(float(tip_z.min()), 3),
            "pen_down_plane_mm": round(plan["pen_down_z"] * 1000, 3),
            "joint_min_deg": joints.min(axis=0).round(1).tolist(),
            "joint_max_deg": joints.max(axis=0).round(1).tolist(),
            "joint_limit_margin_deg": np.minimum(joints.min(axis=0) - LIMITS_DEG[:, 0], LIMITS_DEG[:, 1] - joints.max(axis=0)).round(1).tolist(),
            "max_joint_step_rad_per_tick": round(float(result["steps"].max()), 5),
            "policy_ms_mean": round(float(result["tick_ms"].mean()), 2),
            "real_tip_lag_mm": {"mean": round(float(result["lags"].mean() * 1e3), 3), "max": round(float(result["lags"].max() * 1e3), 3)}
            if len(result["lags"]) else None,
            # pen_down_error_mm is from the commanded joints; this one is the arm's measured tip, sampled at 10 Hz.
            "real_pen_down_error_mm": {"rmse": round(float(np.sqrt(np.mean(result["real_errors"] ** 2)) * 1e3), 3),
                                       "p95": round(float(np.percentile(result["real_errors"], 95) * 1e3), 3),
                                       "max": round(float(result["real_errors"].max() * 1e3), 3),
                                       "samples": int(len(result["real_errors"]))}
            if len(result["real_errors"]) else None,
        }
        log.event("summary", **summary)
        outcome = result["outcome"]
        (log.directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return 0 if outcome == "finished" else 2
    except SafeStop as exc:
        outcome = "stopped"
        log.event("safe_stop", reason=str(exc))
        if motion_started and api is not None:
            lift_and_hold(api, log)
        return 2
    except KeyboardInterrupt:
        outcome = "interrupted"
        log.event("safe_stop", reason="keyboard interrupt")
        if motion_started and api is not None:
            lift_and_hold(api, log)
        return 130
    except Exception as exc:  # noqa: BLE001
        outcome = "exception"
        log.event("exception", error=str(exc), traceback=traceback.format_exc())
        if motion_started and api is not None:
            lift_and_hold(api, log)
        return 1
    finally:
        if api is not None:
            try:
                api.disconnect()
            except Exception:  # noqa: BLE001
                pass
        (log.directory / "result.json").write_text(json.dumps({"outcome": outcome, "log_dir": str(log.directory)}, indent=2), encoding="utf-8")
        log.event("run_end", outcome=outcome)
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
