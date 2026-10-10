"""Replay one simulated drawing on a single physical xArm 7.

The pen path is built with the same loader, plane mapping, and stroke phases
as the Isaac run. Motion is sent only when --execute is present and
paper_center_mm has been measured in the robot base frame.
Before each session, touch the pen to the paper center and run
--set-paper-from-tcp. That stores the arm pose and a RealSense photo.
The arm must be at initial_tcp_mm_deg before --execute. The first target is
derived from that default: its height and wrist, over the first stroke.

Every step is appended to a local folder. The folder survives a disconnect
from the internet. Servos are left holding after a stop so the arm does not
go limp onto the table. Pass --release-servos only when the arm is supported.

    python isaac-sim/scripts/run_xarm7_hardware.py --plan-only
    python isaac-sim/scripts/run_xarm7_hardware.py
    python isaac-sim/scripts/run_xarm7_hardware.py --execute --max-strokes 1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from coordinate_mapper import map_trajectory_to_plane
from drawing_state_machine import build_motion_sequence
from trajectory_loader import load_trajectory


class RunLog:
    """Line-buffered text log plus one JSON record per step."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.text = (directory / "run.log").open("a", encoding="utf-8", buffering=1)
        self.events = (directory / "events.jsonl").open("a", encoding="utf-8", buffering=1)
        self.step = 0

    def event(self, name: str, **fields: Any) -> dict[str, Any]:
        self.step += 1
        record = {
            "step": self.step,
            "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": name,
            **fields,
        }
        line = json.dumps(record, default=_jsonable, ensure_ascii=True)
        self.events.write(line + "\n")
        self.events.flush()
        os.fsync(self.events.fileno())
        self.text.write(f"{record['step']:04d} {record['time']} {name} {_brief(fields)}\n")
        self.text.flush()
        os.fsync(self.text.fileno())
        print(f"{record['step']:04d} {name} {_brief(fields)}", flush=True)
        return record

    def close(self) -> None:
        self.text.close()
        self.events.close()


class SafeStop(Exception):
    """Motion must stop. The message is the failover reason."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


def _brief(fields: dict[str, Any]) -> str:
    parts = []
    for key, value in fields.items():
        text = json.dumps(value, default=_jsonable, ensure_ascii=True)
        if len(text) > 240:
            text = text[:237] + "..."
        parts.append(f"{key}={text}")
    return " ".join(parts)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration is not a mapping: {path}")
    return payload


def _optional_vector(value: Any, size: int, label: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be {size} finite numbers or null")
    return array


def _host_network_text() -> str:
    try:
        completed = subprocess.run(
            ["ipconfig"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"ipconfig failed: {exc}"
    return (completed.stdout or "") + (completed.stderr or "")


def _ping(ip: str) -> tuple[bool, str]:
    completed = subprocess.run(
        ["ping", "-n", "1", "-w", "1000", ip],
        capture_output=True,
        text=True,
        timeout=8,
        check=False,
    )
    output = ((completed.stdout or "") + (completed.stderr or "")).strip()
    # Windows ping exits 0 for "Destination net unreachable" from a gateway.
    # Only a reply from the robot itself counts.
    replied = f"Reply from {ip}:" in output and "unreachable" not in output.lower()
    return completed.returncode == 0 and replied, output


def _code(result: Any) -> int | None:
    if isinstance(result, int):
        return result
    if isinstance(result, tuple) and result:
        try:
            return int(result[0])
        except (TypeError, ValueError):
            return None
    return None


def _payload(result: Any) -> Any:
    if isinstance(result, tuple) and len(result) >= 2:
        return result[1]
    return result


class RobotSession:
    def __init__(self, log: RunLog, ip: str, connect_timeout_s: float) -> None:
        self.log = log
        self.ip = ip
        self.connect_timeout_s = connect_timeout_s
        self.arm: Any = None

    def connect_once(self) -> bool:
        from xarm.wrapper import XArmAPI

        holder: dict[str, Any] = {}

        def _open() -> None:
            try:
                arm = XArmAPI(self.ip, is_radian=False, do_not_open=True)
                arm.connect(port=self.ip)
                holder["arm"] = arm
            except Exception as exc:  # noqa: BLE001 - logged and retried by the caller
                holder["error"] = exc

        worker = threading.Thread(target=_open, name="xarm-connect", daemon=True)
        worker.start()
        worker.join(self.connect_timeout_s)
        if worker.is_alive():
            self.log.event("connect_timeout", ip=self.ip, timeout_s=self.connect_timeout_s)
            return False
        if "error" in holder:
            self.log.event("connect_exception", ip=self.ip, error=str(holder["error"]))
            return False
        arm = holder.get("arm")
        connected = bool(arm is not None and arm.connected)
        self.log.event(
            "connect_result",
            ip=self.ip,
            connected=connected,
            error_code=getattr(arm, "error_code", None),
            warn_code=getattr(arm, "warn_code", None),
        )
        if not connected:
            return False
        self.arm = arm
        return True

    def call(self, label: str, method: str, *args: Any, **kwargs: Any) -> Any:
        if self.arm is None:
            raise SafeStop(f"{label} called before connect")
        self.log.event("sdk_call", label=label, method=method)
        started = time.perf_counter()
        try:
            result = getattr(self.arm, method)(*args, **kwargs)
        except Exception as exc:
            self.log.event("sdk_exception", label=label, method=method, error=str(exc))
            raise SafeStop(f"{label} raised {exc}") from exc
        code = _code(result)
        self.log.event(
            "sdk_return",
            label=label,
            method=method,
            code=code,
            value=_payload(result),
            elapsed_s=round(time.perf_counter() - started, 3),
        )
        return result

    def read_snapshot(self, label: str) -> dict[str, Any]:
        state = self.call(f"{label}_state", "get_state")
        errors = self.call(f"{label}_err_warn", "get_err_warn_code")
        joints = self.call(f"{label}_joints", "get_servo_angle")
        tcp = self.call(f"{label}_tcp", "get_position")
        snapshot = {
            "label": label,
            "state_code": _code(state),
            "state": _payload(state),
            "err_code": _code(errors),
            "err_warn": _payload(errors),
            "joint_code": _code(joints),
            "joints_deg": _payload(joints),
            "tcp_code": _code(tcp),
            "tcp_mm_deg": _payload(tcp),
        }
        self.log.event("snapshot", **snapshot)
        return snapshot

    def stop_holding(self) -> None:
        if self.arm is None:
            self.log.event("stop_skipped", reason="no arm object")
            return
        self.call("stop_state", "set_state", 4)

    def disconnect(self) -> None:
        if self.arm is None:
            return
        try:
            self.arm.disconnect()
            self.log.event("disconnected", ip=self.ip)
        except Exception as exc:  # noqa: BLE001
            self.log.event("disconnect_exception", error=str(exc))
        self.arm = None


def build_plan(config: dict[str, Any], sim_drawing: dict[str, Any], max_strokes: int | None) -> list[dict[str, Any]]:
    trajectory_path = Path(config["trajectory_path"])
    if not trajectory_path.is_absolute():
        trajectory_path = PROJECT / trajectory_path
    trajectory = load_trajectory(trajectory_path)
    mapped = map_trajectory_to_plane(
        trajectory,
        center_xy=sim_drawing["surface_center_xy_m"],
        size_xy=sim_drawing["surface_size_xy_m"],
        flip_image_y=bool(sim_drawing["flip_image_y"]),
        max_step=float(config["hardware_step_m"]),
        smoothing_strength=float(sim_drawing["smoothing_strength"]),
        corner_angle_degrees=float(sim_drawing["corner_angle_degrees"]),
    )
    strokes = mapped["strokes"]
    if max_strokes is not None:
        if max_strokes < 1 or max_strokes > len(strokes):
            raise ValueError(f"--max-strokes {max_strokes} but {trajectory['drawing_id']} has {len(strokes)} strokes")
        strokes = strokes[:max_strokes]
    phases = build_motion_sequence(
        strokes,
        pen_down_z=float(sim_drawing["pen_down_z_m"]),
        pen_up_z=float(sim_drawing["pen_up_z_m"]),
        approach_height=float(sim_drawing["approach_height_m"]),
        max_cartesian_step=float(config["hardware_step_m"]),
        corner_angle_degrees=float(sim_drawing["corner_angle_degrees"]),
        corner_densify_window=int(sim_drawing["corner_window_points"]),
        corner_densify_factor=int(sim_drawing["corner_densify_factor"]),
    )
    paper = _optional_vector(config.get("paper_center_mm"), 3, "paper_center_mm")
    calibrated = paper is not None
    if paper is None:
        paper = np.zeros(3, dtype=np.float64)
    center_xy = np.asarray(sim_drawing["surface_center_xy_m"], dtype=np.float64)
    pen_down_z = float(sim_drawing["pen_down_z_m"])
    draw_speed = float(config["draw_speed_mm_s"])
    travel_speed = float(config["travel_speed_mm_s"])
    rows: list[dict[str, Any]] = []
    index = 0
    for phase in phases:
        for target in phase.targets:
            sim = np.asarray(target.position, dtype=np.float64)
            robot = np.array(
                [
                    paper[0] + (sim[0] - center_xy[0]) * 1000.0,
                    paper[1] + (sim[1] - center_xy[1]) * 1000.0,
                    paper[2] + (sim[2] - pen_down_z) * 1000.0,
                ],
                dtype=np.float64,
            )
            rows.append(
                {
                    "index": index,
                    "state": target.state,
                    "stroke_id": target.stroke_id,
                    "waypoint_index": target.waypoint_index,
                    "pen_down": int(target.pen_down),
                    "speed_mm_s": draw_speed if target.pen_down else travel_speed,
                    "sim_x_m": float(sim[0]),
                    "sim_y_m": float(sim[1]),
                    "sim_z_m": float(sim[2]),
                    "robot_x_mm": float(robot[0]),
                    "robot_y_mm": float(robot[1]),
                    "robot_z_mm": float(robot[2]),
                    "frame": "robot_base_mm" if calibrated else "uncalibrated_relative_mm",
                }
            )
            index += 1
    return rows


def _fence(config: dict[str, Any], sim_drawing: dict[str, Any], paper: np.ndarray) -> dict[str, float]:
    size = np.asarray(sim_drawing["surface_size_xy_m"], dtype=np.float64) * 1000.0
    margin = float(config["xy_margin_mm"])
    pen_down = float(sim_drawing["pen_down_z_m"])
    approach = float(sim_drawing["approach_height_m"])
    return {
        "x_min": float(paper[0] - size[0] / 2.0 - margin),
        "x_max": float(paper[0] + size[0] / 2.0 + margin),
        "y_min": float(paper[1] - size[1] / 2.0 - margin),
        "y_max": float(paper[1] + size[1] / 2.0 + margin),
        "z_min": float(paper[2] - float(config["z_below_paper_mm"])),
        "z_max": float(paper[2] + (approach - pen_down) * 1000.0 + float(config["z_above_approach_mm"])),
    }


def _inside(fence: dict[str, float], x: float, y: float, z: float) -> bool:
    return (
        fence["x_min"] <= x <= fence["x_max"]
        and fence["y_min"] <= y <= fence["y_max"]
        and fence["z_min"] <= z <= fence["z_max"]
    )


def _write_plan(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _with_derived_start(config: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Put the default height and wrist over the first stroke, then keep the pen aim."""
    initial = _optional_vector(config.get("initial_tcp_mm_deg"), 6, "initial_tcp_mm_deg")
    pen_rpy = _optional_vector(config.get("tcp_rpy_deg"), 3, "tcp_rpy_deg")
    stamped: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if pen_rpy is None:
            item["roll_deg"] = ""
            item["pitch_deg"] = ""
            item["yaw_deg"] = ""
        else:
            item["roll_deg"] = float(pen_rpy[0])
            item["pitch_deg"] = float(pen_rpy[1])
            item["yaw_deg"] = float(pen_rpy[2])
        stamped.append(item)
    if initial is None:
        return stamped
    first = stamped[0]
    start = {
        "index": 0,
        "state": "START",
        "stroke_id": -1,
        "waypoint_index": -1,
        "pen_down": 0,
        "speed_mm_s": float(config["travel_speed_mm_s"]),
        "sim_x_m": first["sim_x_m"],
        "sim_y_m": first["sim_y_m"],
        "sim_z_m": "",
        "robot_x_mm": float(first["robot_x_mm"]),
        "robot_y_mm": float(first["robot_y_mm"]),
        "robot_z_mm": float(initial[2]),
        "roll_deg": float(initial[3]),
        "pitch_deg": float(initial[4]),
        "yaw_deg": float(initial[5]),
        "frame": first["frame"],
    }
    derived = [start, *stamped]
    for index, row in enumerate(derived):
        row["index"] = index
    return derived


def _tcp_xyz(snapshot: dict[str, Any]) -> np.ndarray | None:
    tcp = snapshot.get("tcp_mm_deg")
    if not isinstance(tcp, (list, tuple)) or len(tcp) < 3:
        return None
    try:
        point = np.asarray(tcp[:3], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(point).all():
        return None
    return point


def _joints_deg(snapshot: dict[str, Any]) -> np.ndarray | None:
    joints = snapshot.get("joints_deg")
    if not isinstance(joints, (list, tuple)) or len(joints) < 7:
        return None
    try:
        angles = np.asarray(joints[:7], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(angles).all():
        return None
    return angles


def _row_rpy(row: dict[str, Any], fallback: np.ndarray) -> np.ndarray:
    roll = row.get("roll_deg")
    if roll == "" or roll is None:
        return fallback
    return np.asarray([row["roll_deg"], row["pitch_deg"], row["yaw_deg"]], dtype=np.float64)


def _error_warn(snapshot: dict[str, Any]) -> tuple[int | None, int | None]:
    pair = snapshot.get("err_warn")
    if not isinstance(pair, (list, tuple)) or len(pair) < 2:
        return None, None
    try:
        return int(pair[0]), int(pair[1])
    except (TypeError, ValueError):
        return None, None


def _clear_faults_once(session: RobotSession) -> dict[str, Any]:
    session.call("clear_warn", "clean_warn")
    session.call("clear_error", "clean_error")
    session.call("motion_enable_after_clear", "motion_enable", enable=True)
    session.call("mode_after_clear", "set_mode", 0)
    session.call("state_after_clear", "set_state", 0)
    time.sleep(0.5)
    return session.read_snapshot("after_clear")


def _prepare_motion(session: RobotSession) -> None:
    session.call("motion_enable", "motion_enable", enable=True)
    session.call("set_mode_position", "set_mode", 0)
    session.call("set_state_ready", "set_state", 0)
    time.sleep(0.5)
    snapshot = session.read_snapshot("ready")
    error_code, _warn = _error_warn(snapshot)
    state = snapshot.get("state")
    if error_code not in (0, None) or state not in (1, 2):
        session.log.event("ready_failover", error_code=error_code, state=state)
        snapshot = _clear_faults_once(session)
        error_code, _warn = _error_warn(snapshot)
        state = snapshot.get("state")
        if error_code not in (0, None) or state not in (1, 2):
            raise SafeStop(f"arm not ready after one fault clear: state={state} error={error_code}")


def _move_to(
    session: RobotSession,
    *,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    speed: float,
    acceleration: float,
    timeout_s: float,
) -> tuple[int | None, np.ndarray | None]:
    result = session.call(
        "set_position",
        "set_position",
        x=x,
        y=y,
        z=z,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        speed=speed,
        mvacc=acceleration,
        radius=-1,
        wait=True,
        timeout=timeout_s,
    )
    code = _code(result)
    snapshot = session.read_snapshot("after_move")
    return code, _tcp_xyz(snapshot)


def _lift_then_hold(session: RobotSession, z_mm: float, speed: float, acceleration: float, rpy: np.ndarray) -> None:
    snapshot = session.read_snapshot("before_failover_lift")
    current = _tcp_xyz(snapshot)
    if current is None:
        session.stop_holding()
        return
    session.log.event("failover_lift", from_mm=current.tolist(), to_z_mm=z_mm)
    try:
        _move_to(
            session,
            x=float(current[0]),
            y=float(current[1]),
            z=z_mm,
            roll=float(rpy[0]),
            pitch=float(rpy[1]),
            yaw=float(rpy[2]),
            speed=speed,
            acceleration=acceleration,
            timeout_s=20,
        )
    except SafeStop as exc:
        session.log.event("failover_lift_failed", error=str(exc))
    session.stop_holding()


def execute(session: RobotSession, rows: list[dict[str, Any]], config: dict[str, Any], fence: dict[str, float], rpy: np.ndarray) -> None:
    acceleration = float(config["move_acceleration_mm_s2"])
    tolerance = float(config["position_tolerance_mm"])
    attempts = max(1, int(config["waypoint_attempts"]))
    for row in rows:
        point = (float(row["robot_x_mm"]), float(row["robot_y_mm"]), float(row["robot_z_mm"]))
        if row["state"] != "START" and not _inside(fence, *point):
            raise SafeStop(f"waypoint {row['index']} is outside the paper fence")
        aim = _row_rpy(row, rpy)
        session.log.event(
            "waypoint_begin",
            **{key: row[key] for key in ("index", "state", "stroke_id", "pen_down", "speed_mm_s")},
            roll_deg=float(aim[0]),
            pitch_deg=float(aim[1]),
            yaw_deg=float(aim[2]),
        )
        speed = float(row["speed_mm_s"])
        reached = False
        for attempt in range(1, attempts + 1):
            code, measured = _move_to(
                session,
                x=point[0],
                y=point[1],
                z=point[2],
                roll=float(aim[0]),
                pitch=float(aim[1]),
                yaw=float(aim[2]),
                speed=speed,
                acceleration=acceleration,
                timeout_s=30,
            )
            if measured is None:
                error_mm = None
            else:
                error_mm = float(np.linalg.norm(measured - np.asarray(point)))
            session.log.event(
                "waypoint_check",
                index=row["index"],
                attempt=attempt,
                code=code,
                error_mm=None if error_mm is None else round(error_mm, 3),
                measured_mm=None if measured is None else [round(float(v), 3) for v in measured],
            )
            if code == 0 and error_mm is not None and error_mm <= tolerance:
                reached = True
                break
            snapshot = session.read_snapshot(f"retry_{row['index']}_{attempt}")
            error_code, _warn = _error_warn(snapshot)
            if error_code not in (0, None):
                cleared = _clear_faults_once(session)
                if _error_warn(cleared)[0] not in (0, None):
                    raise SafeStop(f"controller error {error_code} at waypoint {row['index']}")
            speed = max(5.0, speed / 2.0)
        if not reached:
            raise SafeStop(f"waypoint {row['index']} missed tolerance after {attempts} attempts")
    session.log.event("path_complete", waypoints=len(rows))
    session.stop_holding()


def _capture_camera(index: int, destination: Path) -> Path:
    import cv2

    capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        raise SafeStop(f"camera {index} did not open")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    frame = None
    for _ in range(8):
        ok, frame = capture.read()
        if ok:
            break
    capture.release()
    if frame is None:
        raise SafeStop(f"camera {index} returned no frame")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), frame):
        raise SafeStop(f"could not write {destination}")
    return destination


def _write_paper_pose(config_path: Path, xyz: Any, rpy: Any) -> None:
    text = config_path.read_text(encoding="utf-8")
    center = "[" + ", ".join(f"{float(v):.3f}" for v in xyz) + "]"
    angles = "[" + ", ".join(f"{float(v):.3f}" for v in rpy) + "]"
    replacements = {"paper_center_mm": center, "tcp_rpy_deg": angles}
    for key, literal in replacements.items():
        updated, count = re.subn(
            rf"(?m)^{key}:.*$",
            f"{key}: {literal}",
            text,
            count=1,
        )
        if count != 1:
            raise SafeStop(f"{config_path} has no {key} line to update")
        text = updated
    config_path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay one Isaac drawing on one physical xArm 7.")
    parser.add_argument("--config", default=str(PROJECT / "config" / "xarm7_hardware.yaml"))
    parser.add_argument("--robot-ip", default=None)
    parser.add_argument("--trajectory", default=None)
    parser.add_argument("--max-strokes", type=int, default=None)
    parser.add_argument("--plan-only", action="store_true", help="Write the plan and log, then exit before ping.")
    parser.add_argument("--execute", action="store_true", help="Enable motion. Requires a measured paper center.")
    parser.add_argument(
        "--set-paper-from-tcp",
        action="store_true",
        help="With the pen tip touching the paper center, save that TCP pose into the config and keep a RealSense photo. Does not move.",
    )
    parser.add_argument("--camera-index", type=int, default=2, help="OpenCV index of the RealSense color camera. 2 saw the table.")
    parser.add_argument("--release-servos", action="store_true", help="Turn servos off after the run. Default is to hold.")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = RunLog(PROJECT / "outputs" / "hardware_runs" / stamp)
    outcome = "started"
    session: RobotSession | None = None
    rpy_for_lift: np.ndarray | None = None
    lift_z: float | None = None
    motion_started = False
    config: dict[str, Any] = {}
    try:
        log.event(
            "run_start",
            argv=sys.argv,
            python=sys.version,
            hostname=socket.gethostname(),
            cwd=str(Path.cwd()),
            log_dir=str(log.directory),
        )
        network_text = _host_network_text()
        (log.directory / "ipconfig.txt").write_text(network_text, encoding="utf-8")
        log.event("host_network_saved", path=str(log.directory / "ipconfig.txt"), chars=len(network_text))

        config = _load_yaml(Path(args.config))
        if args.robot_ip:
            config["robot_ip"] = args.robot_ip
        if args.trajectory:
            config["trajectory_path"] = args.trajectory
        sim = _load_yaml(PROJECT / "config" / "xarm7_drawing.yaml")["drawing"]
        log.event("config_loaded", robot_ip=config["robot_ip"], trajectory=config["trajectory_path"], execute=args.execute)

        rows = _with_derived_start(config, build_plan(config, sim, args.max_strokes))
        if not rows:
            raise SafeStop("plan is empty")
        plan_path = log.directory / "plan.csv"
        _write_plan(plan_path, rows)
        robot_points = np.asarray([[row["robot_x_mm"], row["robot_y_mm"], row["robot_z_mm"]] for row in rows])
        log.event(
            "plan_ready",
            waypoints=len(rows),
            strokes=len({row["stroke_id"] for row in rows if int(row["stroke_id"]) >= 0}),
            frame=rows[0]["frame"],
            min_mm=robot_points.min(axis=0).round(2).tolist(),
            max_mm=robot_points.max(axis=0).round(2).tolist(),
            plan=str(plan_path),
        )
        if args.plan_only:
            outcome = "plan_only"
            return 0

        ip = str(config["robot_ip"])
        ping_ok = False
        for attempt in range(1, 4):
            log.event("ping_attempt", attempt=attempt, ip=ip)
            try:
                ping_ok, output = _ping(ip)
            except (OSError, subprocess.TimeoutExpired) as exc:
                ping_ok, output = False, str(exc)
            log.event("ping_result", attempt=attempt, ok=ping_ok, output=output[-500:])
            if ping_ok:
                break
            time.sleep(1.0)
        if not ping_ok:
            outcome = "ping_failed"
            raise SafeStop(f"no ping reply from {ip}")

        session = RobotSession(log, ip, float(config["connect_timeout_s"]))
        connected = False
        for attempt in range(1, int(config["connect_attempts"]) + 1):
            log.event("connect_attempt", attempt=attempt, ip=ip)
            connected = session.connect_once()
            if connected:
                break
            time.sleep(2.0)
        if not connected or session.arm is None:
            outcome = "connect_failed"
            raise SafeStop(f"could not connect to {ip}")

        snapshot = session.read_snapshot("connected")
        error_code, warn_code = _error_warn(snapshot)
        if warn_code not in (0, None):
            session.call("clear_warn_initial", "clean_warn")
            snapshot = session.read_snapshot("after_initial_warn_clear")
            error_code, warn_code = _error_warn(snapshot)
        if error_code not in (0, None):
            log.event("initial_fault_failover")
            snapshot = _clear_faults_once(session)
            error_code, _warn = _error_warn(snapshot)
            if error_code not in (0, None):
                outcome = "faulted"
                raise SafeStop(f"controller error {error_code} remains after one clear")
        if args.set_paper_from_tcp:
            tcp = snapshot.get("tcp_mm_deg")
            if not isinstance(tcp, (list, tuple)) or len(tcp) < 6:
                outcome = "tcp_unreadable"
                raise SafeStop("could not read the pen pose to store as the paper center")
            photo = _capture_camera(int(args.camera_index), log.directory / "paper_pose.png")
            _write_paper_pose(Path(args.config), tcp[:3], tcp[3:6])
            log.event(
                "paper_pose_saved",
                paper_center_mm=[round(float(v), 3) for v in tcp[:3]],
                tcp_rpy_deg=[round(float(v), 3) for v in tcp[3:6]],
                photo=str(photo),
                config=str(args.config),
            )
            outcome = "paper_pose_saved"
            return 0
        if not args.execute:
            outcome = "connected_read_only"
            log.event("motion_not_requested", hint="re-run with --execute after paper_center_mm is measured")
            return 0

        paper = _optional_vector(config.get("paper_center_mm"), 3, "paper_center_mm")
        if paper is None or rows[0]["frame"] != "robot_base_mm":
            outcome = "uncalibrated"
            raise SafeStop("paper_center_mm is null; refusing to move")
        fence = _fence(config, sim, paper)
        log.event("fence", **fence)
        outside = [
            row["index"]
            for row in rows
            if row["state"] != "START" and not _inside(fence, row["robot_x_mm"], row["robot_y_mm"], row["robot_z_mm"])
        ]
        if outside:
            outcome = "plan_outside_fence"
            raise SafeStop(f"{len(outside)} waypoints are outside the paper fence, first={outside[0]}")

        current = _tcp_xyz(snapshot)
        if current is None:
            snapshot = session.read_snapshot("tcp_reread")
            current = _tcp_xyz(snapshot)
        if current is None:
            outcome = "tcp_unreadable"
            raise SafeStop("could not read the current TCP position")
        initial = _optional_vector(config.get("initial_tcp_mm_deg"), 6, "initial_tcp_mm_deg")
        if initial is not None:
            home_gap = float(np.linalg.norm(current - initial[:3]))
            match_mm = float(config.get("initial_match_mm", 20))
            log.event("initial_pose_gap_mm", distance_mm=round(home_gap, 2), limit_mm=match_mm)
            if home_gap > match_mm:
                outcome = "not_at_initial_pose"
                raise SafeStop(
                    f"pen is {home_gap:.1f} mm from the recorded default; move the arm to its initial position first"
                )
            configured_joints = _optional_vector(config.get("initial_joints_deg"), 7, "initial_joints_deg")
            live_joints = _joints_deg(snapshot)
            if configured_joints is not None and live_joints is not None:
                joint_gap = float(np.max(np.abs(live_joints - configured_joints)))
                joint_limit = float(config.get("initial_joint_match_deg", 5))
                log.event("initial_joint_gap_deg", max_deg=round(joint_gap, 3), limit_deg=joint_limit)
                if joint_gap > joint_limit:
                    outcome = "not_at_initial_pose"
                    raise SafeStop(
                        f"a joint is {joint_gap:.1f} deg from the recorded default; move the arm to its initial position first"
                    )
            start = np.asarray([rows[0]["robot_x_mm"], rows[0]["robot_y_mm"], rows[0]["robot_z_mm"]], dtype=np.float64)
            log.event(
                "derived_start_mm",
                xyz_mm=[round(float(v), 3) for v in start],
                rpy_deg=[round(float(v), 3) for v in _row_rpy(rows[0], initial[3:6])],
                travel_mm=round(float(np.linalg.norm(start - initial[:3])), 2),
            )
        else:
            first = np.asarray([rows[0]["robot_x_mm"], rows[0]["robot_y_mm"], rows[0]["robot_z_mm"]], dtype=np.float64)
            jump = float(np.linalg.norm(current - first))
            log.event("first_jump_mm", distance_mm=round(jump, 2), limit_mm=float(config["max_first_jump_mm"]))
            if jump > float(config["max_first_jump_mm"]):
                outcome = "first_jump_too_far"
                raise SafeStop(
                    f"pen is {jump:.1f} mm from the first approach point; jog it closer than {config['max_first_jump_mm']} mm"
                )

        configured_rpy = _optional_vector(config.get("tcp_rpy_deg"), 3, "tcp_rpy_deg")
        tcp = snapshot.get("tcp_mm_deg")
        if configured_rpy is not None:
            rpy = configured_rpy
        elif isinstance(tcp, (list, tuple)) and len(tcp) >= 6:
            rpy = np.asarray(tcp[3:6], dtype=np.float64)
        else:
            outcome = "orientation_unreadable"
            raise SafeStop("could not read roll/pitch/yaw to hold")
        rpy_for_lift = rpy
        lift_z = float(fence["z_max"] - float(config["z_above_approach_mm"]))
        log.event("orientation_held_deg", roll=float(rpy[0]), pitch=float(rpy[1]), yaw=float(rpy[2]))

        for remaining in range(int(config["countdown_s"]), 0, -1):
            log.event("countdown", seconds_left=remaining)
            time.sleep(1.0)
        motion_started = True
        _prepare_motion(session)
        execute(session, rows, config, fence, rpy)
        outcome = "finished"
        return 0
    except SafeStop as exc:
        if outcome == "started":
            outcome = "stopped"
        log.event("safe_stop", outcome=outcome, reason=str(exc))
        if motion_started and session is not None and session.arm is not None and rpy_for_lift is not None and lift_z is not None:
            _lift_then_hold(
                session,
                lift_z,
                float(config.get("travel_speed_mm_s", 30.0)),
                float(config.get("move_acceleration_mm_s2", 100.0)),
                rpy_for_lift,
            )
        elif motion_started and session is not None:
            session.stop_holding()
        return 2
    except KeyboardInterrupt:
        outcome = "interrupted"
        log.event("safe_stop", outcome=outcome, reason="keyboard interrupt")
        if motion_started and session is not None and rpy_for_lift is not None and lift_z is not None:
            _lift_then_hold(
                session,
                lift_z,
                float(config.get("travel_speed_mm_s", 30.0)),
                float(config.get("move_acceleration_mm_s2", 100.0)),
                rpy_for_lift,
            )
        elif motion_started and session is not None:
            session.stop_holding()
        return 130
    except Exception as exc:  # noqa: BLE001
        outcome = "exception"
        log.event("exception", error=str(exc), traceback=traceback.format_exc())
        if session is not None:
            session.stop_holding()
        return 1
    finally:
        if session is not None and session.arm is not None and args.release_servos:
            try:
                session.call("release_servos", "motion_enable", enable=False)
            except SafeStop as exc:
                log.event("release_servos_failed", error=str(exc))
        if session is not None:
            session.disconnect()
        summary = {"outcome": outcome, "log_dir": str(log.directory)}
        (log.directory / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log.event("run_end", **summary)
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
