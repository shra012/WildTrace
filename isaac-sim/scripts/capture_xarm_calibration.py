"""Capture named xArm poses with both RealSense cameras, then derive a matching sim config.

Read-only: the script never enables motion, changes mode or sends a target.
Put the arm in each pose yourself (xArm Studio or manual mode), then run:

    python isaac-sim/scripts/capture_xarm_calibration.py --pose initial
    python isaac-sim/scripts/capture_xarm_calibration.py --pose paper_center      # pen tip touching the paper center
    python isaac-sim/scripts/capture_xarm_calibration.py --pose pen_down_start    # optional: pen touching where drawing starts
    python isaac-sim/scripts/capture_xarm_calibration.py --pose ready             # pen straight down, ~70 mm above the center
    python isaac-sim/scripts/capture_xarm_calibration.py --apply

Without a captured ready pose, solve one with Lula (Isaac's python), then --apply again:

    C:\\Isaac-Sim\\python.bat isaac-sim\\scripts\\capture_xarm_calibration.py --ready-from-ik

Runs start from 'ready' when there is one, else from 'initial' (taken from
xarm7_hardware.yaml if not captured).

Each capture stores joints, TCP pose, TCP offset and a color image plus aligned
depth from every connected RealSense under outputs/calibration/<session>/.
--apply writes config/xarm7_drawing_real.yaml (the Isaac config at the measured
geometry: robot base = sim world origin) and updates xarm7_hardware.yaml.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
CALIBRATION_ROOT = PROJECT / "outputs" / "calibration"
SIM_CONFIG = PROJECT / "config" / "xarm7_drawing.yaml"
REAL_SIM_CONFIG = PROJECT / "config" / "xarm7_drawing_real.yaml"
HARDWARE_CONFIG = PROJECT / "config" / "xarm7_hardware.yaml"
# xarm7_real.urdf puts pen_tip at the xArm Gripper TCP: 172 mm along the flange +Z.
REAL_TCP_OFFSET = [0.0, 0.0, 172.0, 0.0, 0.0, 0.0]
CAMERA_NAMES = {"109622070191": "wrist", "134322070512": "overhead"}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--pose", help="Name for this pose, e.g. initial, paper_center, pen_down_start")
    action.add_argument("--apply", action="store_true", help="Write the sim and hardware configs from the captures")
    action.add_argument(
        "--ready-from-ik",
        action="store_true",
        help="Store a 'ready' pose (pen straight down at approach height over the paper center) solved "
        "with Lula from the paper_center joints. Needs Isaac's python and a prior --apply.",
    )
    parser.add_argument("--session", default=datetime.now().strftime("%Y%m%d"), help="Capture folder name")
    parser.add_argument("--robot-ip", default=None, help="Defaults to robot_ip in xarm7_hardware.yaml")
    parser.add_argument("--no-cameras", action="store_true")
    return parser.parse_args()


def _wrap_deg(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def _read_arm(ip: str) -> dict[str, Any]:
    from xarm.wrapper import XArmAPI

    arm = XArmAPI(ip, is_radian=False)
    time.sleep(0.5)
    try:
        if not arm.connected:
            raise RuntimeError(f"could not connect to {ip}")
        code, joints = arm.get_servo_angle()
        code_tcp, tcp = arm.get_position()
        if code != 0 or code_tcp != 0:
            raise RuntimeError(f"read failed: joints code {code}, tcp code {code_tcp}")
        return {
            "robot_ip": ip,
            "version": arm.version,
            "state": arm.get_state()[1],
            "err_warn": arm.get_err_warn_code()[1],
            "joints_deg": [float(v) for v in joints[:7]],
            "joints_rad_wrapped": [math.radians(_wrap_deg(v)) for v in joints[:7]],
            "tcp_mm_deg": [float(v) for v in tcp[:6]],
            "tcp_offset": [float(v) for v in arm.tcp_offset],
        }
    finally:
        arm.disconnect()


def _grab(rs, serial: str):
    pipeline, config = rs.pipeline(), rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    profile = pipeline.start(config)
    try:
        align = rs.align(rs.stream.color)
        for _ in range(30):  # let auto exposure settle
            frames = pipeline.wait_for_frames(10000)
        frames = align.process(frames)
        color = np.asanyarray(frames.get_color_frame().get_data()).copy()
        scale = profile.get_device().first_depth_sensor().get_depth_scale()
        depth_m = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32) * scale
        intrinsics = frames.get_color_frame().profile.as_video_stream_profile().intrinsics
    finally:
        pipeline.stop()
    return color, depth_m, intrinsics


def _capture_cameras(directory: Path, pose: str) -> dict[str, Any]:
    import cv2
    import pyrealsense2 as rs

    shots: dict[str, Any] = {}
    for device in rs.context().query_devices():
        serial = device.get_info(rs.camera_info.serial_number)
        name = CAMERA_NAMES.get(serial, serial)
        # A camera started right after another sometimes misses its first frames.
        for attempt in range(1, 4):
            try:
                color, depth_m, intrinsics = _grab(rs, serial)
                break
            except RuntimeError as exc:
                # A stalled D435i stays stalled until its USB is reset.
                print(f"[WARN] {name} camera attempt {attempt}: {exc}; resetting it")
                device.hardware_reset()
                time.sleep(8.0)
        else:
            shots[name] = {"serial": serial, "error": "no frames after 3 attempts"}
            continue
        color_path = directory / f"{pose}_{name}.png"
        depth_path = directory / f"{pose}_{name}_depth.npy"
        cv2.imwrite(str(color_path), color)
        np.save(depth_path, depth_m)
        shots[name] = {
            "serial": serial,
            "color": color_path.name,
            "depth_m": depth_path.name,
            "intrinsics": {
                "width": intrinsics.width,
                "height": intrinsics.height,
                "fx": intrinsics.fx,
                "fy": intrinsics.fy,
                "ppx": intrinsics.ppx,
                "ppy": intrinsics.ppy,
                "coeffs": list(intrinsics.coeffs),
            },
        }
    return shots


def _load_captures(directory: Path) -> dict[str, Any]:
    path = directory / "calibration.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"poses": {}}


def capture(args, hardware: dict[str, Any]) -> int:
    directory = CALIBRATION_ROOT / args.session
    directory.mkdir(parents=True, exist_ok=True)
    record = {"captured_at": datetime.now().isoformat(timespec="seconds"), **_read_arm(args.robot_ip or hardware["robot_ip"])}
    if record["err_warn"] != [0, 0]:
        print(f"[WARN] Controller reports err/warn {record['err_warn']}")
    if not args.no_cameras:
        record["cameras"] = _capture_cameras(directory, args.pose)
    captures = _load_captures(directory)
    captures["poses"][args.pose] = record
    (directory / "calibration.json").write_text(json.dumps(captures, indent=2), encoding="utf-8")
    print(f"[OK] {args.pose}: joints {np.round(record['joints_deg'], 2).tolist()} deg")
    print(f"[OK] {args.pose}: tcp {np.round(record['tcp_mm_deg'], 2).tolist()} (mm, deg), tcp offset {record['tcp_offset']}")
    print(f"[OK] Cameras: {sorted(record.get('cameras', {}))} -> {directory}")
    return 0


def _set_line(text: str, key: str, literal: str) -> str:
    updated, count = re.subn(rf"(?m)^(\s*){key}:.*$", rf"\g<1>{key}: {literal}", text, count=1)
    if count != 1:
        raise SystemExit(f"[FAIL] no {key} line to update")
    return updated


def _vector(values, digits: int) -> str:
    return "[" + ", ".join(f"{float(v):.{digits}f}" for v in values) + "]"


def apply(args) -> int:
    directory = CALIBRATION_ROOT / args.session
    poses = _load_captures(directory)["poses"]
    if "paper_center" not in poses:
        raise SystemExit(f"[FAIL] capture paper_center first (session {args.session})")
    center = poses["paper_center"]
    initial_captured = "initial" in poses
    # Runs start from 'ready' (pen down, above the paper) when there is one;
    # 'initial' stays the controller rest pose recorded for the hardware script.
    ready = poses.get("ready")
    if initial_captured:
        initial = poses["initial"]
    else:
        # No capture: the controller default pose recorded in xarm7_hardware.yaml.
        hardware = yaml.safe_load(HARDWARE_CONFIG.read_text(encoding="utf-8"))
        joints = [float(v) for v in hardware["initial_joints_deg"]]
        initial = {
            "joints_deg": joints,
            "joints_rad_wrapped": [math.radians(_wrap_deg(v)) for v in joints],
            "tcp_mm_deg": [float(v) for v in hardware["initial_tcp_mm_deg"]],
        }
        print(f"[OK] Initial pose from {HARDWARE_CONFIG.name}: joints {joints} deg")
    contact = poses.get("pen_down_start", center)

    if any(abs(a - b) > 1e-3 for a, b in zip(center["tcp_offset"], REAL_TCP_OFFSET)):
        print(
            f"[WARN] Real TCP offset {center['tcp_offset']} differs from the {REAL_TCP_OFFSET} "
            "that xarm7_real.urdf puts the pen tip at; regenerate the URDF to match."
        )

    sim = yaml.safe_load(SIM_CONFIG.read_text(encoding="utf-8"))["drawing"]
    pen_down_z = contact["tcp_mm_deg"][2] / 1000.0
    paper_top_z = pen_down_z - (float(sim["pen_down_z_m"]) - float(sim["paper_top_z_m"]))
    shift = pen_down_z - float(sim["pen_down_z_m"])
    center_xy = [v / 1000.0 for v in center["tcp_mm_deg"][:2]]

    text = SIM_CONFIG.read_text(encoding="utf-8")
    # The lab arm itself: calibrated kinematics, gripper and pen at the 172 mm TCP
    # (scripts/generate_xarm7_urdf.py --real; check_environment.py imports the USD).
    text = _set_line(text, "urdf_path", "assets/xarm7/xarm7_real.urdf")
    text = _set_line(text, "usd_path", "assets/xarm7/xarm7_real.usd")
    home = ready or initial
    text = _set_line(text, "home_joint_positions_rad", _vector(home["joints_rad_wrapped"], 4))
    text = _set_line(text, "surface_center_xy_m", _vector(center_xy, 4))
    text = _set_line(text, "paper_top_z_m", f"{paper_top_z:.4f}")
    # Pen heights keep the sim's offsets above the paper.
    for key in ("pen_down_z_m", "pen_up_z_m", "approach_height_m"):
        text = _set_line(text, key, f"{float(sim[key]) + shift:.4f}")
    safety = yaml.safe_load(SIM_CONFIG.read_text(encoding="utf-8"))["safety"]
    low = [float(v) for v in safety["workspace_min_m"]]
    high = [float(v) for v in safety["workspace_max_m"]]
    low[2] = min(low[2] + shift, paper_top_z - 0.02)
    # The run starts from the sim tip at the home pose. The sim pen is shorter
    # than the real TCP offset, so pad the real home TCP generously.
    home_tip = np.asarray(initial["tcp_mm_deg"][:3], dtype=np.float64) / 1000.0
    low = np.minimum(low, home_tip - 0.08).tolist()
    high = np.maximum(high, home_tip + 0.08).tolist()
    text = _set_line(text, "workspace_min_m", _vector(low, 3))
    text = _set_line(text, "workspace_max_m", _vector(high, 3))
    header = (
        f"# Generated by scripts/capture_xarm_calibration.py --apply from outputs/calibration/{args.session}.\n"
        "# Same as xarm7_drawing.yaml except home pose, paper placement and pen heights,\n"
        "# which come from the real arm (robot base frame = sim world frame).\n"
    )
    REAL_SIM_CONFIG.write_text(header + text, encoding="utf-8")

    hardware_text = HARDWARE_CONFIG.read_text(encoding="utf-8")
    # Paper center in xy; height from the pen-down contact, which may be a separate capture.
    paper_mm = center["tcp_mm_deg"][:2] + [contact["tcp_mm_deg"][2]]
    hardware_text = _set_line(hardware_text, "paper_center_mm", _vector(paper_mm, 3))
    hardware_text = _set_line(hardware_text, "tcp_rpy_deg", _vector(contact["tcp_mm_deg"][3:6], 3))
    if initial_captured:
        hardware_text = _set_line(hardware_text, "initial_tcp_mm_deg", _vector(initial["tcp_mm_deg"], 3))
        hardware_text = _set_line(hardware_text, "initial_joints_deg", _vector(initial["joints_deg"], 3))
    HARDWARE_CONFIG.write_text(hardware_text, encoding="utf-8")

    print(f"[OK] Sim config: {REAL_SIM_CONFIG}")
    print(f"     home ({'ready' if ready else 'initial'}) rad {np.round(home['joints_rad_wrapped'], 3).tolist()}")
    print(f"     paper center xy {np.round(center_xy, 4).tolist()} m, paper top z {paper_top_z:.4f} m (shift {shift * 1000:+.1f} mm)")
    print(f"[OK] Hardware config updated: {HARDWARE_CONFIG}")
    return 0


def ready_from_ik(args) -> int:
    import lula

    directory = CALIBRATION_ROOT / args.session
    captures = _load_captures(directory)
    if "paper_center" not in captures["poses"]:
        raise SystemExit(f"[FAIL] capture paper_center first (session {args.session})")
    config = yaml.safe_load(REAL_SIM_CONFIG.read_text(encoding="utf-8"))
    robot, drawing = config["robot"], config["drawing"]
    kinematics = lula.load_robot(
        str(PROJECT / robot["robot_description_path"]), str(PROJECT / robot["urdf_path"])
    ).kinematics()
    seed = np.radians([_wrap_deg(v) for v in captures["poses"]["paper_center"]["joints_deg"]])
    target = np.array([*drawing["surface_center_xy_m"], drawing["approach_height_m"]], dtype=np.float64)
    w, x, y, z = (float(v) for v in drawing["orientation_wxyz"])
    ik = lula.CyclicCoordDescentIkConfig()
    ik.cspace_seeds = [seed]
    ik.position_tolerance = 1e-5
    ik.orientation_tolerance = 1e-3
    result = lula.compute_ik_ccd(kinematics, lula.Pose3(lula.Rotation3(w, x, y, z), target), robot["end_effector_frame"], ik)
    if not result.success:
        raise SystemExit("[FAIL] no IK solution for the ready pose")
    q = np.asarray(result.cspace_position, dtype=np.float64)
    tip = np.asarray(kinematics.pose(q, robot["end_effector_frame"]).translation) * 1000.0
    captures["poses"]["ready"] = {
        "source": "lula_ik",
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "joints_deg": np.degrees(q).tolist(),
        "joints_rad_wrapped": q.tolist(),
        "tcp_mm_deg": [*tip.tolist(), 180.0, 0.0, 0.0],
        "seed": "paper_center",
    }
    (directory / "calibration.json").write_text(json.dumps(captures, indent=2), encoding="utf-8")
    print(f"[OK] ready: joints {np.round(np.degrees(q), 2).tolist()} deg, pen tip {np.round(tip, 2).tolist()} mm")
    print(f"     max joint change from paper_center {np.degrees(np.max(np.abs(q - seed))):.1f} deg; run --apply next")
    return 0


def main() -> int:
    args = _parse_args()
    if args.ready_from_ik:
        return ready_from_ik(args)
    hardware = yaml.safe_load(HARDWARE_CONFIG.read_text(encoding="utf-8"))
    return apply(args) if args.apply else capture(args, hardware)


if __name__ == "__main__":
    raise SystemExit(main())
