"""Solve each RealSense camera's pose relative to an A4 sheet from the sheet's four corners (PnP).

The sheet is found by colour (low saturation), its four edges are fitted as straight lines and
intersected, so clips or shadows at the corners do not pull them in. With the A4 size (297 x 210 mm)
and the colour intrinsics, cv2.solvePnP (IPPE, planar) gives the camera's position and orientation in
the sheet frame: origin at the sheet centre, X along the long edge, Y along the short edge, Z up
towards the camera. This is pure geometry, so it also checks the depth-based heights.
A rectangle is symmetric under a 180 deg turn, so X/Y signs are fixed by the image (X points to the
right-hand end of the long edge as seen by the camera); height and tilt do not depend on that choice.
Never moves the arm; the arm pose at capture is recorded.

    isaac-sim/.venv-act/Scripts/python.exe isaac-sim/scripts/camera_pose_from_sheet.py --camera wrist --label park_high
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
SHEETS_MM = {"a4": (297.0, 210.0), "letter": (279.4, 215.9)}
SHEET_MM = SHEETS_MM["a4"]  # set from --sheet


def grab_color(rs, serial: str, width: int, height: int):
    pipeline, config = rs.pipeline(), rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    try:
        for _ in range(40):
            frames = pipeline.wait_for_frames(10000)
        stack = []
        for _ in range(10):  # average frames to cut sensor noise at the edges
            stack.append(np.asanyarray(pipeline.wait_for_frames(10000).get_color_frame().get_data()).astype(np.float32))
        color = np.mean(stack, axis=0).astype(np.uint8)
        k = profile.get_stream(rs.stream.color).as_video_stream_profile().intrinsics
    finally:
        pipeline.stop()
    return color, k


def sheet_corners(color: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Four refined corners (pixels, clockwise from the top-left one) and the sheet mask."""
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    base = (hsv[..., 1] < 60) & (hsv[..., 2] > 90)
    # Glare can wash wood out to low saturation, but its hue stays warm. Keep the dominant hue of the
    # white pixels (the paper's tint under this camera's white balance) +-10.
    hue_mode = int(np.argmax(np.bincount(hsv[..., 0][base].ravel(), minlength=180)))
    mask = (base & (np.abs(hsv[..., 0].astype(int) - hue_mode) <= 10)).astype(np.uint8)
    # A strong opening cuts thin bridges to nearby white things (a notebook, the arm); a small closing
    # only fills pixel noise. A large closing merged the sheet with a notebook touching its corner.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    h, w = mask.shape
    target = SHEET_MM[0] / SHEET_MM[1]
    candidates = []
    for contour in contours:
        hull = cv2.convexHull(contour)
        if cv2.contourArea(hull) < 0.02 * mask.size:
            continue
        perimeter = cv2.arcLength(hull, True)
        for eps in np.linspace(0.01, 0.08, 15):
            poly = cv2.approxPolyDP(hull, eps * perimeter, True)
            if len(poly) == 4:
                quad = poly.reshape(4, 2).astype(np.float64)
                # A sheet cut off by the image border has a false edge along the border: skip it.
                if np.any((quad[:, 0] < 5) | (quad[:, 0] > w - 6) | (quad[:, 1] < 5) | (quad[:, 1] > h - 6)):
                    break
                edges = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
                pair_a, pair_b = (edges[0] + edges[2]) / 2, (edges[1] + edges[3]) / 2
                ratio = max(pair_a, pair_b) / min(pair_a, pair_b)
                # Other sheets and notebooks in view: keep the quad closest to the A4 aspect ratio.
                candidates.append((abs(ratio - target), -cv2.contourArea(poly), quad, contour.reshape(-1, 2)))
                break
    if not candidates:
        raise RuntimeError("no four-cornered sheet fully inside the image")
    _, _, rough, contour = min(candidates, key=lambda c: (c[0], c[1]))
    # Order clockwise from the corner nearest the image's top-left.
    centre = rough.mean(axis=0)
    angles = np.arctan2(rough[:, 1] - centre[1], rough[:, 0] - centre[0])
    rough = rough[np.argsort(angles)]
    start = int(np.argmin(rough.sum(axis=1)))
    rough = np.roll(rough, -start, axis=0)
    # Fit each edge to contour points away from the corners, then intersect neighbouring lines.
    lines = []
    for i in range(4):
        a, b = rough[i], rough[(i + 1) % 4]
        direction = (b - a) / np.linalg.norm(b - a)
        normal = np.array([-direction[1], direction[0]])
        rel = contour - a
        along = rel @ direction
        across = np.abs(rel @ normal)
        length = np.linalg.norm(b - a)
        near = (along > 0.15 * length) & (along < 0.85 * length) & (across < 6)
        pts = contour[near].astype(np.float32)
        if len(pts) < 20:
            raise RuntimeError(f"edge {i} has too few points ({len(pts)}); is the sheet fully in view?")
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
    corners = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        t = np.linalg.solve(np.c_[d1, -d2], p2 - p1)
        corners.append(p1 + t[0] * d1)
    return np.asarray(corners), mask


def solve(corners: np.ndarray, k) -> dict:
    K = np.array([[k.fx, 0, k.ppx], [0, k.fy, k.ppy], [0, 0, 1]], dtype=np.float64)
    dist = np.asarray(k.coeffs, dtype=np.float64)
    e01, e12 = np.linalg.norm(corners[1] - corners[0]), np.linalg.norm(corners[2] - corners[1])
    long_, short = SHEET_MM
    # The longer image edge between corners 0-1 or 1-2 is the sheet's long edge.
    if e01 >= e12:
        obj = np.array([[-long_ / 2, short / 2, 0], [long_ / 2, short / 2, 0], [long_ / 2, -short / 2, 0], [-long_ / 2, -short / 2, 0]])
    else:
        obj = np.array([[-short / 2, long_ / 2, 0], [short / 2, long_ / 2, 0], [short / 2, -long_ / 2, 0], [-short / 2, -long_ / 2, 0]])[:, [1, 0, 2]] * [1, -1, 1]
    best = None
    for flag in (cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_ITERATIVE):
        ok, rvec, tvec = cv2.solvePnP(obj, corners.astype(np.float64), K, dist, flags=flag)
        if not ok:
            continue
        rvec, tvec = cv2.solvePnPRefineLM(obj, corners.astype(np.float64), K, dist, rvec, tvec)
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        err = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - corners) ** 2, axis=1))))
        R, _ = cv2.Rodrigues(rvec)
        camera_in_sheet = (-R.T @ tvec).ravel()
        if camera_in_sheet[2] < 0:  # put the camera on the +Z side of the sheet
            continue
        if best is None or err < best[0]:
            best = (err, R, tvec.ravel(), camera_in_sheet)
    if best is None:
        raise RuntimeError("PnP found no pose with the camera above the sheet")
    err, R, tvec, camera_in_sheet = best
    optical_axis_in_sheet = R.T @ np.array([0.0, 0.0, 1.0])  # camera +Z (viewing direction) in the sheet frame
    tilt = float(np.degrees(np.arccos(np.clip(-optical_axis_in_sheet[2], -1, 1))))
    heading = float(np.degrees(np.arctan2(optical_axis_in_sheet[1], optical_axis_in_sheet[0])))
    # Where the optical axis meets the sheet, in sheet coordinates.
    s = -camera_in_sheet[2] / optical_axis_in_sheet[2]
    hit = camera_in_sheet + s * optical_axis_in_sheet
    image_x_in_sheet = R.T @ np.array([1.0, 0.0, 0.0])
    return {
        "camera_position_in_sheet_mm": [round(float(v), 1) for v in camera_in_sheet],
        "height_above_sheet_mm": round(float(camera_in_sheet[2]), 1),
        "distance_along_optical_axis_mm": round(float(s), 1),
        "tilt_from_vertical_deg": round(tilt, 2),
        "optical_axis_heading_in_sheet_deg": round(heading, 1),
        "optical_axis_hits_sheet_at_mm": [round(float(hit[0]), 1), round(float(hit[1]), 1)],
        "image_x_axis_heading_in_sheet_deg": round(float(np.degrees(np.arctan2(image_x_in_sheet[1], image_x_in_sheet[0]))), 1),
        "reprojection_rms_px": round(err, 3),
        "rotation_sheet_to_camera": np.round(R, 6).tolist(),
        "translation_sheet_to_camera_mm": np.round(tvec, 2).tolist(),
        "corners_px": np.round(corners, 2).tolist(),
        "edge_lengths_px": [round(float(np.linalg.norm(corners[(i + 1) % 4] - corners[i])), 1) for i in range(4)],
    }


def arm_pose(ip: str) -> dict:
    from xarm.wrapper import XArmAPI

    arm = XArmAPI(ip, is_radian=False)
    time.sleep(0.3)
    try:
        return {"tcp_mm_deg": [round(float(v), 3) for v in arm.get_position()[1][:6]],
                "joints_deg": [round(float(v), 3) for v in arm.get_servo_angle()[1][:7]], "errors": arm.get_err_warn_code()[1]}
    finally:
        arm.disconnect()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", choices=["wrist", "overhead"], required=True)
    parser.add_argument("--label", default="pose")
    parser.add_argument("--sheet", choices=sorted(SHEETS_MM), default="a4", help="Paper size; a wrong size shows as a large reprojection error")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--out", default=str(PROJECT / "outputs" / "camera_heights" / datetime.now().strftime("%Y%m%d")))
    args = parser.parse_args()

    import pyrealsense2 as rs

    global SHEET_MM
    SHEET_MM = SHEETS_MM[args.sheet]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hardware = yaml.safe_load((PROJECT / "config" / "xarm7_hardware.yaml").read_text(encoding="utf-8"))
    serial = next(s for s, n in CAMERA_NAMES.items() if n == args.camera)
    color, k = grab_color(rs, serial, args.width, args.height)
    corners, mask = sheet_corners(color)
    result = solve(corners, k)
    record = {"camera": args.camera, "serial": serial, "label": args.label, "sheet": args.sheet, "time": datetime.now().isoformat(timespec="seconds"),
              "sheet_mm": SHEET_MM, "arm": arm_pose(str(hardware["robot_ip"])), **result,
              "intrinsics": {"width": k.width, "height": k.height, "fx": k.fx, "fy": k.fy, "ppx": k.ppx, "ppy": k.ppy, "coeffs": list(k.coeffs)}}
    overlay = color.copy()
    for i, c in enumerate(corners):
        cv2.circle(overlay, tuple(int(v) for v in c), 8, (0, 0, 255), 2)
        cv2.line(overlay, tuple(int(v) for v in c), tuple(int(v) for v in corners[(i + 1) % 4]), (0, 255, 0), 2)
        cv2.putText(overlay, str(i), tuple(int(v) + 10 for v in c), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(overlay, f"{args.camera}: {result['height_above_sheet_mm']} mm above sheet, tilt {result['tilt_from_vertical_deg']} deg, "
                f"reproj {result['reprojection_rms_px']} px", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    stem = f"pnp_{args.label}_{args.camera}_{args.sheet}"
    cv2.imwrite(str(out / f"{stem}.png"), color)
    cv2.imwrite(str(out / f"{stem}_corners.png"), overlay)
    (out / f"{stem}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps({k2: record[k2] for k2 in ("camera", "label", "height_above_sheet_mm", "distance_along_optical_axis_mm", "tilt_from_vertical_deg",
                                                 "camera_position_in_sheet_mm", "optical_axis_hits_sheet_at_mm", "reprojection_rms_px", "edge_lengths_px")}
                     | {"arm_tcp": record["arm"]["tcp_mm_deg"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
