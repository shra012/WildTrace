"""Score a drawn sheet against its planned trajectory from a scan or photo.

The sheet is found as the largest bright quadrilateral and rectified to millimetres (US Letter by
default), the ink is segmented and thinned to a centre line, and the planned pen-down path is fitted
to it with a rigid transform (rotation + translation, no scale: the sheet gives the scale). Reported:

- plan_to_ink: distance from each planned sample to the nearest ink centre-line pixel (missed lines)
- ink_to_plan: distance from each ink pixel to the plan (stray or offset ink)
- coverage_at_mm: share of the plan with ink within that distance; precision_at_mm: the reverse
- chamfer_mm: mean of the two mean distances

Use a flatbed scan (300-600 dpi) for sub-millimetre numbers; an overhead photo works but its pixel
size bounds the resolution. Example:

    python3 isaac-sim/scripts/ink_fidelity.py --image scan.png --trajectory face.json \
        --out-dir outputs/ink_fidelity/run07
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from coordinate_mapper import map_trajectory_to_plane  # noqa: E402
from trajectory_loader import load_trajectory  # noqa: E402


def planned_path_mm(trajectory: Path, config: Path, step_mm: float) -> np.ndarray:
    drawing = yaml.safe_load(config.read_text())["drawing"]
    mapped = map_trajectory_to_plane(
        load_trajectory(trajectory),
        center_xy=[0.0, 0.0],
        size_xy=drawing["surface_size_xy_m"],
        flip_image_y=bool(drawing["flip_image_y"]),
        max_step=step_mm / 1000.0,
        smoothing_strength=float(drawing["smoothing_strength"]),
        corner_angle_degrees=float(drawing["corner_angle_degrees"]),
    )
    points = [np.asarray(p, dtype=np.float64)[:2] for s in mapped["strokes"] for p in s["points"]]
    return np.asarray(points) * 1000.0


def rectify_sheet(image: np.ndarray, sheet_mm: tuple[float, float], px_per_mm: float) -> np.ndarray:
    # Paper is bright and unsaturated; a wooden or coloured table is not. Ink inside the sheet is closed over.
    hsv = cv2.cvtColor(cv2.GaussianBlur(image, (5, 5), 0), cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 1] < 45) & (hsv[..., 2] > 110)).astype(np.uint8) * 255
    k = max(3, int(0.01 * max(image.shape[:2])) | 1)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    quad = None
    for eps in (0.01, 0.02, 0.03, 0.05):
        approx = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True).reshape(-1, 2)
        if len(approx) == 4:
            quad = approx.astype(np.float32)
            break
    if quad is None:  # fall back to the tightest rectangle around the sheet
        quad = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
    # Order corners: top-left, top-right, bottom-right, bottom-left.
    s, d = quad.sum(1), np.diff(quad, axis=1).ravel()
    quad = np.array([quad[np.argmin(s)], quad[np.argmin(d)], quad[np.argmax(s)], quad[np.argmax(d)]])
    w_px, h_px = np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[3] - quad[0])
    long_mm, short_mm = max(sheet_mm), min(sheet_mm)
    w_mm, h_mm = (long_mm, short_mm) if w_px >= h_px else (short_mm, long_mm)
    size = (int(round(w_mm * px_per_mm)), int(round(h_mm * px_per_mm)))
    dst = np.array([[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]], dtype=np.float32)
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(quad, dst), size)


def ink_centreline_mm(sheet: np.ndarray, px_per_mm: float, margin_mm: float, min_area_px: int) -> np.ndarray:
    gray = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY)
    m = int(margin_mm * px_per_mm)
    paper = int(np.median(gray))  # fill the border with paper tone: a white frame would itself read as an edge
    gray[:m, :], gray[-m:, :], gray[:, :m], gray[:, -m:] = paper, paper, paper, paper
    block = int(15 * px_per_mm) | 1
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 15)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = np.isin(labels, [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_area_px])
    rows, cols = np.nonzero(skeletonize(keep))
    return np.column_stack([cols, rows]).astype(np.float64) / px_per_mm


def rigid_icp(source: np.ndarray, target: np.ndarray, iterations: int = 60, keep: float = 0.8):
    """Best trimmed rigid fit of source onto target over 12 starting rotations, both handednesses.

    Fit the ink onto the plan, not the reverse: every ink pixel should lie on the plan, but parts of the plan
    may have no ink (a dry pen, a faint stroke), and those must not drag the fit. Each iteration keeps the
    closest `keep` share of pairs.
    """
    tree = cKDTree(target)
    best = None
    for start in np.arange(0.0, 360.0, 30.0):
        for mirror in (False, True):
            a = np.deg2rad(start)
            rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            if mirror:  # an image y axis pointing down mirrors the drawing frame
                rot = rot @ np.diag([1.0, -1.0])
            trans = target.mean(0) - source.mean(0) @ rot.T
            for _ in range(iterations):
                moved = source @ rot.T + trans
                dist, idx = tree.query(moved)
                inliers = dist <= np.quantile(dist, keep)
                src, matched = source[inliers], target[idx[inliers]]
                mu_s, mu_t = src.mean(0), matched.mean(0)
                u, _, vt = np.linalg.svd((src - mu_s).T @ (matched - mu_t))
                r = (u @ vt).T
                if np.linalg.det(r) * np.linalg.det(rot) < 0:  # keep the chosen handedness
                    r = (u @ np.diag([1.0, -1.0]) @ vt).T
                rot, trans = r, mu_t - mu_s @ r.T
            dist = tree.query(source @ rot.T + trans)[0]
            err = float(np.mean(np.sort(dist)[: int(keep * len(dist))]))
            if best is None or err < best[2]:
                best = (rot, trans, err)
    return best


def score(plan: np.ndarray, ink: np.ndarray, thresholds=(0.5, 1.0, 2.0)) -> dict:
    p2i = cKDTree(ink).query(plan)[0]
    i2p = cKDTree(plan).query(ink)[0]
    out = {
        "plan_to_ink_mm": {"mean": float(p2i.mean()), "rmse": float(np.sqrt(np.mean(p2i**2))),
                           "p95": float(np.percentile(p2i, 95)), "max": float(p2i.max())},
        "ink_to_plan_mm": {"mean": float(i2p.mean()), "rmse": float(np.sqrt(np.mean(i2p**2))),
                           "p95": float(np.percentile(i2p, 95))},
        "chamfer_mm": float((p2i.mean() + i2p.mean()) / 2),
        "plan_samples": int(len(plan)),
        "ink_pixels": int(len(ink)),
    }
    for t in thresholds:
        out[f"coverage_at_{t}mm"] = float(np.mean(p2i <= t))
        out[f"precision_at_{t}mm"] = float(np.mean(i2p <= t))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--trajectory", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=PROJECT / "config" / "xarm7_drawing_real.yaml",
                    help="Scene config whose drawing section sized the plan (default: lab-arm twin)")
    ap.add_argument("--sheet-mm", type=float, nargs=2, default=(279.4, 215.9))
    ap.add_argument("--px-per-mm", type=float, default=8.0, help="Rectified resolution")
    ap.add_argument("--margin-mm", type=float, default=6.0, help="Ignore this border (sheet edge, shadows)")
    ap.add_argument("--min-ink-area-mm2", type=float, default=0.5, help="Drop specks smaller than this")
    ap.add_argument("--already-rectified", action="store_true", help="The image is the sheet, edge to edge")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"cannot read {args.image}")
    if args.already_rectified:
        sheet = cv2.resize(image, (int(max(args.sheet_mm) * args.px_per_mm), int(min(args.sheet_mm) * args.px_per_mm)))
    else:
        sheet = rectify_sheet(image, tuple(args.sheet_mm), args.px_per_mm)
    ink = ink_centreline_mm(sheet, args.px_per_mm, args.margin_mm, int(args.min_ink_area_mm2 * args.px_per_mm**2))
    if len(ink) < 50:
        raise SystemExit("almost no ink found; check lighting or --margin-mm")
    plan = planned_path_mm(args.trajectory, args.config, step_mm=0.25)
    rot, trans, _ = rigid_icp(ink, plan)  # ink -> plan frame
    result = score(plan, ink @ rot.T + trans)
    aligned = (plan - trans) @ rot  # plan in sheet millimetres, for the overlay
    result["alignment"] = {"rotation_deg": float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))),
                           "mirrored": bool(np.linalg.det(rot) < 0), "translation_mm": trans.tolist()}
    result["image"], result["trajectory"], result["px_per_mm"] = str(args.image), str(args.trajectory), args.px_per_mm

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "ink_fidelity.json").write_text(json.dumps(result, indent=2))
    overlay = sheet.copy()
    for x, y in (aligned * args.px_per_mm).astype(int):
        cv2.circle(overlay, (int(x), int(y)), 1, (214, 120, 42), -1)
    cv2.imwrite(str(args.out_dir / "ink_fidelity_overlay.png"), overlay)
    print(json.dumps({k: v for k, v in result.items() if k not in ("image", "trajectory")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
