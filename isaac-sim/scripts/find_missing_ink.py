"""Find the parts of a drawing that got no ink, from a top-down photo of the sheet.

The planned pen-down path (robot frame, built exactly as run_xarm7_act_hardware.py builds it) is
registered onto the photo by chamfer matching against the ink that is there: a similarity
transform (scale, rotation, mirror, shift) found by grid search and a simplex refinement. Path
points with no dark pixel nearby are "missing". Missing runs are merged, padded with a little
overlap and written as mapped strokes (metres) for run_xarm7_act_hardware.py --mapped-strokes.

    python isaac-sim/scripts/find_missing_ink.py --photo sheet.png \
        --trajectory inputs/trajectories/Fish/ba3e1657db195fe4.json --out missing.json

Use a photo taken looking straight down (the wrist camera with the arm hovering over the paper).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from coordinate_mapper import map_trajectory_to_plane  # noqa: E402
from trajectory_loader import load_trajectory  # noqa: E402

CAP_PX = 10.0  # distances beyond this count the same, so missing ink does not drag the fit


def mapped_strokes(trajectory_path: Path, hardware: dict, twin: dict) -> list[dict]:
    drawing = twin["drawing"]
    paper = np.asarray(hardware["paper_center_mm"], dtype=np.float64) / 1000.0
    mapped = map_trajectory_to_plane(
        load_trajectory(trajectory_path),
        center_xy=paper[:2].tolist(),
        size_xy=drawing["surface_size_xy_m"],
        flip_image_y=bool(drawing["flip_image_y"]),
        max_step=float(drawing["max_cartesian_step_m"]),
        smoothing_strength=float(drawing["smoothing_strength"]),
        corner_angle_degrees=float(drawing["corner_angle_degrees"]),
    )
    return mapped["strokes"]


def ink_distance(
    image: np.ndarray, threshold: int, ignore: list[list[int]], before: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Distance (px) to the nearest ink pixel, and the ink mask. Faint show-through stays above threshold.
    With a `before` photo from the same camera pose, only ink that is new since then counts."""
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    # Auto-exposure makes paper and wood equally bright (~140); the wood is saturated, the paper is not.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    paper = ((hsv[..., 1] < 60) & (hsv[..., 2] > 90)).astype(np.uint8)
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(paper)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    hull = cv2.convexHull(np.argwhere(labels == biggest)[:, ::-1].astype(np.int32))
    sheet = np.zeros_like(paper)
    cv2.fillConvexPoly(sheet, hull, 1)
    ink = ((gray < threshold) & (sheet > 0)).astype(np.uint8)
    if before is not None:
        old = (cv2.GaussianBlur(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), (3, 3), 0) < threshold + 15).astype(np.uint8)
        ink[cv2.dilate(old, np.ones((5, 5), np.uint8)) > 0] = 0
    for x0, y0, x1, y1 in ignore:
        ink[y0:y1, x0:x1] = 0
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    distance = cv2.distanceTransform((1 - ink).astype(np.uint8), cv2.DIST_L2, 5)
    return distance, ink


def project(points: np.ndarray, center: np.ndarray, params: np.ndarray, mirror: float) -> np.ndarray:
    scale, angle, tx, ty = params
    local = (points - center) * np.array([1.0, mirror])
    c, s = np.cos(angle), np.sin(angle)
    return np.c_[c * local[:, 0] - s * local[:, 1], s * local[:, 0] + c * local[:, 1]] * scale + np.array([tx, ty])


def chamfer(distance: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    h, w = distance.shape
    x, y = np.round(pixels[:, 0]).astype(int), np.round(pixels[:, 1]).astype(int)
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    values = np.full(len(pixels), CAP_PX)
    values[inside] = np.minimum(distance[y[inside], x[inside]], CAP_PX)
    return values


def simplex(f, x0: np.ndarray, steps: np.ndarray, iterations: int = 400) -> np.ndarray:
    """Plain Nelder-Mead (no scipy in the project environments)."""
    points = [x0] + [x0 + np.eye(len(x0))[i] * steps[i] for i in range(len(x0))]
    values = [f(p) for p in points]
    for _ in range(iterations):
        order = np.argsort(values)
        points, values = [points[i] for i in order], [values[i] for i in order]
        centroid = np.mean(points[:-1], axis=0)
        reflected = centroid + (centroid - points[-1])
        fr = f(reflected)
        if fr < values[0]:
            expanded = centroid + 2.0 * (centroid - points[-1])
            fe = f(expanded)
            points[-1], values[-1] = (expanded, fe) if fe < fr else (reflected, fr)
        elif fr < values[-2]:
            points[-1], values[-1] = reflected, fr
        else:
            contracted = centroid + 0.5 * (points[-1] - centroid)
            fc = f(contracted)
            if fc < values[-1]:
                points[-1], values[-1] = contracted, fc
            else:
                points = [points[0] + 0.5 * (p - points[0]) for p in points]
                values = [f(p) for p in points]
    return points[int(np.argmin(values))]


def ink_to_path(ink_xy: np.ndarray, strokes_px: list[np.ndarray], shape: tuple[int, int]) -> float:
    """Mean capped distance from ink pixels to the projected path. Everything inked is part of the
    drawing, so this must be small even when half the drawing has no ink."""
    canvas = np.ones(shape, np.uint8)
    for pixels in strokes_px:
        cv2.polylines(canvas, [np.round(pixels).astype(np.int32)], False, 0, 1)
    distance = cv2.distanceTransform(canvas, cv2.DIST_L2, 5)
    return float(np.minimum(distance[ink_xy[:, 1], ink_xy[:, 0]], 3 * CAP_PX).mean())


def register(
    distance: np.ndarray, ink: np.ndarray, strokes: list[np.ndarray], center: np.ndarray, init: dict | None = None
) -> tuple[np.ndarray, float, float]:
    """Similarity transform robot XY -> image px. Scored mainly by ink-to-path distance: every inked
    pixel belongs to the drawing, while up to half the path may have no ink."""
    points = np.vstack(strokes)
    ys, xs = np.nonzero(ink)
    ink_xy = np.c_[xs, ys][:: max(1, len(xs) // 1500)]
    ink_sample = ink_xy[:: max(1, len(ink_xy) // 250)].astype(np.float64)
    ink_center = ink_xy.mean(axis=0)
    h, w = distance.shape
    # Path distance field in the drawing's own frame (relative to center), 0.25 mm per cell.
    cell, margin = 0.00025, 0.03
    lo = points.min(axis=0) - center - margin
    size = np.ceil((points.max(axis=0) - points.min(axis=0) + 2 * margin) / cell).astype(int) + 1
    canvas = np.ones((size[1], size[0]), np.uint8)
    for stroke in strokes:
        cv2.polylines(canvas, [np.round((stroke - center - lo) / cell).astype(np.int32)], False, 0, 1)
    path_field = cv2.distanceTransform(canvas, cv2.DIST_L2, 5) * cell  # metres

    def combined(params: np.ndarray, mirror: float) -> float:
        strokes_px = [project(s, center, params, mirror) for s in strokes]
        forward = chamfer(distance, np.vstack(strokes_px)).mean()
        return ink_to_path(ink_xy, strokes_px, (h, w)) + 0.3 * forward

    if init is not None:
        # Same camera pose as an earlier registration: only refine it (seconds instead of minutes).
        mirror = float(init["mirror"])
        start = np.array([init["scale_px_per_m"], init["angle_rad"], *init["shift_px"]], dtype=np.float64)
        start[2:] += project(center[None], np.asarray(init["center_m"]), np.r_[start[:2], 0.0, 0.0], mirror)[0]
        refined = simplex(lambda p: combined(p, mirror), start, np.array([30.0, np.radians(1.0), 3.0, 3.0]), iterations=200)
        return refined, mirror, combined(refined, mirror)

    shifts = np.stack(np.meshgrid(np.arange(-180, 181, 10), np.arange(-180, 181, 10)), -1).reshape(-1, 2).astype(np.float64)
    candidates = []
    for mirror in (1.0, -1.0):
        for angle in np.radians(np.arange(0, 360, 3)):
            c, s = np.cos(angle), np.sin(angle)
            for scale in np.arange(1500, 4001, 100):  # px per metre
                # Inverse map: image px -> drawing-frame metres, for every shift at once.
                rel = (ink_sample[None, :, :] - ink_center - shifts[:, None, :]) / scale
                local = np.stack([c * rel[..., 0] + s * rel[..., 1], -s * rel[..., 0] + c * rel[..., 1]], -1)
                local[..., 1] *= mirror
                cells = np.round((local - lo) / cell).astype(int)
                gx, gy = cells[..., 0], cells[..., 1]
                inside = (gx >= 0) & (gx < size[0]) & (gy >= 0) & (gy < size[1])
                metres = np.where(inside, path_field[np.clip(gy, 0, size[1] - 1), np.clip(gx, 0, size[0] - 1)], margin)
                cost = np.minimum(metres * scale, 3 * CAP_PX).mean(axis=1)  # back to px
                k = int(np.argmin(cost))
                candidates.append((float(cost[k]), np.array([scale, angle, *(ink_center + shifts[k])]), mirror))
    candidates.sort(key=lambda c: c[0])

    scored = sorted(((combined(p, m), p, m) for _, p, m in candidates[:150]), key=lambda c: c[0])
    best = (np.inf, None, None)
    for _, params, mirror in scored[:5]:
        refined = simplex(lambda p: combined(p, mirror), params, np.array([60.0, np.radians(2.0), 6.0, 6.0]), iterations=250)
        cost = combined(refined, mirror)
        if cost < best[0]:
            best = (cost, refined, mirror)
    cost, params, mirror = best
    return params, mirror, cost


def missing_runs(points: np.ndarray, missing: np.ndarray, bridge_m: float, overlap_m: float, min_m: float) -> list[np.ndarray]:
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    runs, start = [], None
    for i, flag in enumerate(np.r_[missing, False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append([start, i - 1])
            start = None
    merged: list[list[int]] = []
    for run in runs:
        if merged and arc[run[0]] - arc[merged[-1][1]] < bridge_m:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    out = []
    for a, b in merged:
        lo = int(np.searchsorted(arc, arc[a] - overlap_m))
        hi = int(np.searchsorted(arc, arc[b] + overlap_m, side="right")) - 1
        lo, hi = max(lo, 0), min(hi, len(points) - 1)
        if arc[hi] - arc[lo] >= min_m and hi > lo:
            out.append(points[lo : hi + 1])
    return out


def analyse(photo: str | Path, strokes: list[dict], *, ink_threshold: int = 100, miss_px: float = 4.0,
            ignore: list[list[int]] = (), bridge_mm: float = 4.0, overlap_mm: float = 3.0, min_mm: float = 3.0,
            init: dict | None = None, overlay: str | Path | None = None, before: str | Path | None = None) -> dict:
    """Register the path on the photo and measure its ink. Returns a summary, the missing strokes and a
    registration that can seed the next call (same camera pose)."""
    image = cv2.imread(str(photo))
    distance, ink = ink_distance(image, ink_threshold, list(ignore), None if before is None else cv2.imread(str(before)))
    arrays = [np.asarray(s["points"]) for s in strokes]
    center = np.vstack(arrays).mean(axis=0)
    if ink.sum() < 30 and init is not None:
        # No visible new ink (e.g. a sheet already covered in lines): keep the known camera pose.
        mirror = float(init["mirror"])
        params = np.array([init["scale_px_per_m"], init["angle_rad"], *init["shift_px"]], dtype=np.float64)
        params[2:] += project(center[None], np.asarray(init["center_m"]), np.r_[params[:2], 0.0, 0.0], mirror)[0]
        fit = float("nan")
    else:
        params, mirror, fit = register(distance, ink, arrays, center, init)
    result, canvas = [], image.copy()
    total = missing_total = 0.0
    for stroke, points in zip(strokes, arrays):
        pixels = project(points, center, params, mirror)
        missing = chamfer(distance, pixels) > miss_px
        seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
        total += seg.sum()
        missing_total += seg[missing[1:]].sum()
        for a, b, flag in zip(pixels[:-1], pixels[1:], missing[1:]):
            cv2.line(canvas, tuple(int(v) for v in a), tuple(int(v) for v in b), (0, 0, 255) if flag else (0, 200, 0), 2)
        for run in missing_runs(points, missing, bridge_mm / 1e3, overlap_mm / 1e3, min_mm / 1e3):
            result.append({"stroke_id": len(result), "source_stroke_id": int(stroke["stroke_id"]), "points": run.tolist()})
    if overlay:
        cv2.imwrite(str(overlay), canvas)
    return {
        "registration": {"scale_px_per_m": float(params[0]), "angle_rad": float(params[1]), "mirror": mirror,
                         "shift_px": [float(params[2]), float(params[3])], "center_m": center.tolist(), "fit_cost_px": round(fit, 2)},
        "path_mm": round(total * 1e3, 1),
        "missing_mm": round(missing_total * 1e3, 1),
        "ink_coverage_pct": round(100.0 * (1.0 - missing_total / total), 1) if total else None,
        "strokes": result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--photo", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--config", default=str(PROJECT / "config" / "xarm7_hardware.yaml"))
    parser.add_argument("--out", required=True, help="Mapped strokes JSON for run_xarm7_act_hardware.py --mapped-strokes")
    parser.add_argument("--overlay", default=None, help="Image with the registered path: green inked, red missing")
    parser.add_argument("--ink-threshold", type=int, default=100,
                        help="Gray level below which a pixel is ink (paper and show-through ~137-145, ink 15-90)")
    parser.add_argument("--miss-px", type=float, default=4.0, help="A path point is missing if no ink is this close")
    parser.add_argument("--ignore", type=int, nargs=4, action="append", default=[], metavar=("X0", "Y0", "X1", "Y1"),
                        help="Image box to ignore, e.g. the gripper fingers")
    parser.add_argument("--bridge-mm", type=float, default=4.0, help="Join missing runs separated by less inked path")
    parser.add_argument("--overlap-mm", type=float, default=3.0, help="Extend each run into the ink on both ends")
    parser.add_argument("--min-mm", type=float, default=3.0, help="Drop runs shorter than this")
    args = parser.parse_args()

    hardware = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    twin = yaml.safe_load((PROJECT / "config" / "xarm7_drawing_real.yaml").read_text(encoding="utf-8"))
    trajectory = Path(args.trajectory)
    strokes = mapped_strokes(trajectory if trajectory.is_absolute() else PROJECT / trajectory, hardware, twin)
    image = cv2.imread(args.photo)
    distance, ink = ink_distance(image, args.ink_threshold, args.ignore)
    all_points = np.vstack([np.asarray(s["points"]) for s in strokes])
    center = all_points.mean(axis=0)
    params, mirror, fit = register(distance, ink, [np.asarray(s["points"]) for s in strokes], center)

    result, overlay = [], image.copy()
    total = missing_total = 0.0
    for stroke in strokes:
        points = np.asarray(stroke["points"])
        pixels = project(points, center, params, mirror)
        missing = chamfer(distance, pixels) > args.miss_px
        seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
        total += seg.sum()
        missing_total += seg[missing[1:]].sum()
        for a, b, flag in zip(pixels[:-1], pixels[1:], missing[1:]):
            cv2.line(overlay, tuple(int(v) for v in a), tuple(int(v) for v in b), (0, 0, 255) if flag else (0, 200, 0), 2)
        for run in missing_runs(points, missing, args.bridge_mm / 1e3, args.overlap_mm / 1e3, args.min_mm / 1e3):
            result.append({"stroke_id": len(result), "source_stroke_id": int(stroke["stroke_id"]), "points": run.tolist()})
    if args.overlay:
        cv2.imwrite(args.overlay, overlay)
    summary = {
        "trajectory": str(args.trajectory),
        "photo": str(args.photo),
        "registration": {"px_per_mm": round(params[0] / 1000, 3), "angle_deg": round(float(np.degrees(params[1])) % 360, 2),
                         "mirror": mirror, "fit_cost_px": round(fit, 2)},
        "path_mm": round(total * 1e3, 1),
        "missing_mm": round(missing_total * 1e3, 1),
        "strokes_to_draw": len(result),
        "stroke_lengths_mm": [round(float(np.sum(np.linalg.norm(np.diff(np.asarray(r["points"]), axis=0), axis=1))) * 1e3, 1) for r in result],
    }
    Path(args.out).write_text(json.dumps({**summary, "strokes": result}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
