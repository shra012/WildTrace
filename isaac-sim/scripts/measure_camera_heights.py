"""Measure how high each RealSense D435i sits above the paper, from its own depth.

For each camera: average depth frames aligned to colour, find the sheet by colour (low
saturation), deproject those pixels with the colour intrinsics and fit a plane with RANSAC.
The height is the perpendicular distance from the camera's optical centre to that plane, which
stays right when the camera is tilted. Also reported: the tilt between the optical axis and the
paper normal, the distance along the optical axis, and the plane fit residual. Never moves the arm;
the arm pose at capture time is recorded so the wrist-camera result can be tied to the TCP.

    isaac-sim/.venv-act/Scripts/python.exe isaac-sim/scripts/measure_camera_heights.py --label park
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
CAMERA_NAMES = {"109622070191": "wrist", "134322070512": "overhead"}


def grab(rs, serial: str, frames: int, width: int, height: int):
    pipeline, config = rs.pipeline(), rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    profile = pipeline.start(config)
    try:
        align = rs.align(rs.stream.color)
        scale = profile.get_device().first_depth_sensor().get_depth_scale()
        for _ in range(30):  # auto exposure settles
            pipeline.wait_for_frames(10000)
        stack, color, intrinsics = [], None, None
        for _ in range(frames):
            aligned = align.process(pipeline.wait_for_frames(10000))
            depth = np.asanyarray(aligned.get_depth_frame().get_data()).astype(np.float32) * scale
            stack.append(np.where(depth > 0, depth, np.nan))
            color = np.asanyarray(aligned.get_color_frame().get_data()).copy()
            intrinsics = aligned.get_color_frame().profile.as_video_stream_profile().intrinsics
    finally:
        pipeline.stop()
    with np.errstate(all="ignore"):
        depth_m = np.nanmedian(np.stack(stack), axis=0)
    return color, depth_m, intrinsics


def paper_mask(color: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 1] < 60) & (hsv[..., 2] > 90)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        return np.zeros_like(mask, dtype=bool)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    sheet = (labels == biggest).astype(np.uint8)
    return cv2.erode(sheet, np.ones((15, 15), np.uint8)) > 0  # keep away from the sheet's edges


def fit_plane(points: np.ndarray, threshold: float = 0.003, iterations: int = 500, seed: int = 0):
    """RANSAC, then least squares on the inliers. Plane n.p + d = 0 with |n| = 1."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        inliers = np.abs(points @ normal - normal @ a) < threshold
        if best is None or inliers.sum() > best.sum():
            best = inliers
    inl = points[best]
    centroid = inl.mean(axis=0)
    normal = np.linalg.svd(inl - centroid, full_matrices=False)[2][-1]  # full U would be N x N
    if normal[2] > 0:  # point the normal from the paper towards the camera (camera looks along +z)
        normal = -normal
    d = -normal @ centroid
    residual = inl @ normal + d
    return normal, d, best, float(np.sqrt(np.mean(residual**2)))


def measure(rs, serial: str, name: str, out: Path, label: str, frames: int, width: int, height: int, threshold: float = 0.003) -> dict:
    color, depth_m, k = grab(rs, serial, frames, width, height)
    sheet = paper_mask(color)
    valid = sheet & np.isfinite(depth_m) & (depth_m > 0.05) & (depth_m < 3.0)
    v, u = np.nonzero(valid)
    if len(u) < 500:
        raise RuntimeError(f"{name}: only {len(u)} paper pixels with depth")
    z = depth_m[v, u]
    points = np.c_[(u - k.ppx) / k.fx * z, (v - k.ppy) / k.fy * z, z]
    if len(points) > 60000:
        points = points[np.random.default_rng(1).choice(len(points), 60000, replace=False)]
    normal, d, inliers, rms = fit_plane(points, threshold=threshold, iterations=800)
    # Cross-check: the sheet lies on the table, so the dominant plane of the whole view (the tabletop)
    # must agree with it. White arm parts can leak into the sheet mask but are not on that plane.
    allv, allu = np.nonzero(np.isfinite(depth_m) & (depth_m > 0.05) & (depth_m < 3.0))
    allz = depth_m[allv, allu]
    scene = np.c_[(allu - k.ppx) / k.fx * allz, (allv - k.ppy) / k.fy * allz, allz]
    scene = scene[np.random.default_rng(2).choice(len(scene), min(60000, len(scene)), replace=False)]
    s_normal, s_d, s_inliers, s_rms = fit_plane(scene, threshold=max(threshold, 0.005), iterations=800)
    agree_deg = float(np.degrees(np.arccos(min(1.0, abs(float(normal @ s_normal))))))
    agree_mm = abs(abs(d) - abs(s_d)) * 1000
    height_m = abs(d)  # camera origin to plane
    tilt = float(np.degrees(np.arccos(min(1.0, abs(normal[2])))))
    axis_m = abs(d / normal[2])  # along the optical axis to the plane
    centre = depth_m[int(k.ppy) - 3 : int(k.ppy) + 4, int(k.ppx) - 3 : int(k.ppx) + 4]
    overlay = color.copy()
    residual = np.abs(np.c_[(u - k.ppx) / k.fx * z, (v - k.ppy) / k.fy * z, z] @ normal + d)
    on_plane = residual < threshold
    overlay[v[on_plane], u[on_plane]] = (0.5 * overlay[v[on_plane], u[on_plane]] + 0.5 * np.array([0, 200, 0])).astype(np.uint8)
    overlay[v[~on_plane], u[~on_plane]] = (0.5 * overlay[v[~on_plane], u[~on_plane]] + 0.5 * np.array([0, 0, 230])).astype(np.uint8)
    cv2.circle(overlay, (int(k.ppx), int(k.ppy)), 6, (0, 0, 255), 2)
    cv2.putText(overlay, f"{name}: {height_m * 1000:.1f} mm above paper, tilt {tilt:.1f} deg", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(str(out / f"{label}_{name}.png"), color)
    cv2.imwrite(str(out / f"{label}_{name}_paper.png"), overlay)
    np.save(out / f"{label}_{name}_depth.npy", depth_m)
    return {
        "camera": name,
        "serial": serial,
        "height_above_paper_mm": round(height_m * 1000, 1),
        "distance_along_optical_axis_mm": round(axis_m * 1000, 1),
        "tilt_from_vertical_deg": round(tilt, 2),
        "paper_normal_in_camera_frame": [round(float(x), 4) for x in normal],
        "plane_fit_rms_mm": round(rms * 1000, 2),
        "paper_points": int(len(points)),
        "inlier_fraction": round(float(inliers.mean()), 3),
        "table_plane": {"height_mm": round(abs(s_d) * 1000, 1), "tilt_deg": round(float(np.degrees(np.arccos(min(1.0, abs(s_normal[2]))))), 2),
                        "fit_rms_mm": round(s_rms * 1000, 2), "inlier_fraction": round(float(s_inliers.mean()), 3)},
        "paper_vs_table": {"angle_deg": round(agree_deg, 2), "height_difference_mm": round(agree_mm, 1)},
        "depth_at_image_centre_mm": round(float(np.nanmedian(centre)) * 1000, 1) if np.isfinite(centre).any() else None,
        "paper_depth_median_mm": round(float(np.median(z)) * 1000, 1),
        "intrinsics": {"width": k.width, "height": k.height, "fx": k.fx, "fy": k.fy, "ppx": k.ppx, "ppy": k.ppy, "coeffs": list(k.coeffs)},
    }


def arm_pose(ip: str) -> dict:
    from xarm.wrapper import XArmAPI

    arm = XArmAPI(ip, is_radian=False)
    time.sleep(0.3)
    try:
        return {"tcp_mm_deg": [round(float(v), 3) for v in arm.get_position()[1][:6]],
                "joints_deg": [round(float(v), 3) for v in arm.get_servo_angle()[1][:7]],
                "tcp_offset": [float(v) for v in arm.tcp_offset], "errors": arm.get_err_warn_code()[1]}
    finally:
        arm.disconnect()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="pose", help="Name for this capture, e.g. park")
    parser.add_argument("--frames", type=int, default=30, help="Depth frames to median per camera")
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera", choices=["wrist", "overhead", "both"], default="both")
    parser.add_argument("--threshold-mm", type=float, default=3.0, help="Plane inlier band; D435 noise grows with distance squared")
    parser.add_argument("--out", default=str(PROJECT / "outputs" / "camera_heights" / datetime.now().strftime("%Y%m%d")))
    args = parser.parse_args()

    import pyrealsense2 as rs

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hardware = yaml.safe_load((PROJECT / "config" / "xarm7_hardware.yaml").read_text(encoding="utf-8"))
    record = {"label": args.label, "time": datetime.now().isoformat(timespec="seconds"), "arm": arm_pose(str(hardware["robot_ip"])),
              "paper_touch_z_mm": hardware["paper_center_mm"][2], "cameras": {}}
    for device in rs.context().query_devices():
        serial = device.get_info(rs.camera_info.serial_number)
        name = CAMERA_NAMES.get(serial, serial)
        if args.camera != "both" and name != args.camera:
            continue
        for attempt in range(1, 4):
            try:
                record["cameras"][name] = measure(rs, serial, name, out, args.label, args.frames, args.width, args.height,
                                                  args.threshold_mm / 1000.0)
                break
            except RuntimeError as exc:
                print(f"[WARN] {name} attempt {attempt}: {exc}")
                if "paper pixels" in str(exc):
                    record["cameras"][name] = {"error": str(exc)}
                    break
                device.hardware_reset()
                time.sleep(8.0)
    (out / f"{args.label}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps({"arm_tcp": record["arm"]["tcp_mm_deg"], **{n: {k: c.get(k) for k in ("height_above_paper_mm", "distance_along_optical_axis_mm",
          "tilt_from_vertical_deg", "plane_fit_rms_mm", "inlier_fraction", "table_plane", "paper_vs_table", "error")} for n, c in record["cameras"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
