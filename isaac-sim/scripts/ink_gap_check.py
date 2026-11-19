"""Find missed ink on a drawn sheet and write a trajectory that redraws only the gaps.

Camera gap check for the physical arm (see the ACT section of isaac-sim/README.md). It covers failures the
hybrid supervisor's ink bookkeeping cannot see: a dry pen, a sheet lower than the surface model, a shifted sheet.
The camera stays outside the policy:

1. Photo of the drawn sheet (the runners already save the wrist-camera `_photo.jpg` from the park pose; a closer
   photo or a scan is better).
2. The sheet is rectified, the ink thinned and fitted to the plan exactly as in `ink_fidelity.py`. Every stretch
   of a planned stroke with no ink within `--gap-mm` and at least `--min-gap-mm` long is a gap; each gap is padded
   by `--pad-mm` on both sides so the redraw overlaps the ink it joins.
3. The output trajectory lists the gap segments first and then the original strokes unchanged. The runners map
   a trajectory to the sheet from the bounding box of all its strokes, so keeping the original strokes keeps the
   placement identical; draw only the gaps with `--max-strokes <gap count>`, as for paired run 09e.

This script only reads an image and writes files; it never moves the arm. Example:

    python3 isaac-sim/scripts/ink_gap_check.py --image run_photo.jpg \
        --trajectory isaac-sim/data/act_test/Cat/70bc30f8a5f918eb.json --out-dir outputs/ink_gaps/run07
    # then, at the arm, if the gaps look real on the overlay:
    python3 isaac-sim/scripts/run_xarm7_act_hardware.py --execute \
        --trajectory outputs/ink_gaps/run07/gap_trajectory.json --max-strokes <gap count> ...

Caution: the wrist photos (about 0.6 mm per pixel) miss faint lines from a fresh marker tip, so a gap list from
them can ask to redraw ink that is there. Check the overlay before redrawing, and calibrate `--gap-mm` against a
600 dpi scan of the same sheet.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trajectory_loader import load_trajectory  # noqa: E402


def plane_frame(strokes: list[np.ndarray], size_xy_m, flip_image_y: bool):
    """Centre and scale (normalised units to mm) of `map_trajectory_to_plane` with centre (0, 0)."""
    pts = np.vstack(strokes)
    lo, hi = pts.min(0), pts.max(0)
    span = hi - lo
    active = span > 1e-12
    scale_mm = float((np.asarray(size_xy_m, dtype=np.float64)[active] / span[active]).min()) * 1000.0
    return 0.5 * (lo + hi), scale_mm, flip_image_y


def densify(points: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Resample a polyline to at most `step` between samples; returns the samples and their arc length."""
    out, arc = [points[0]], [0.0]
    for a, b in zip(points[:-1], points[1:]):
        seg = float(np.linalg.norm(b - a))
        n = max(1, int(np.ceil(seg / step)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
            arc.append(arc[-1] + seg / n)
    return np.asarray(out), np.asarray(arc)


def find_gaps(arc_mm: np.ndarray, dist_mm: np.ndarray, own_dist_mm: np.ndarray, gap_mm: float, min_gap_mm: float,
              pad_mm: float, merge_mm: float, offset_mm: float) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Split a stroke's uncovered stretches (no ink within `gap_mm`, at least `min_gap_mm` long) into gaps to
    redraw and offsets to leave alone.

    `dist_mm` is each sample's distance to any ink, `own_dist_mm` to this stroke's own ink (pixels whose nearest
    planned point is on this stroke). A stretch whose median distance to its own ink is below `offset_mm` has ink
    running alongside it: the line was drawn, just displaced (or the photo is), and a redraw would add a second,
    parallel line. Gaps are padded by `pad_mm`, clipped to the stroke, and joined when closer than `merge_mm`.
    Returns (gaps, offsets)."""
    missing = dist_mm > gap_mm
    runs, offsets, start = [], [], None
    for i, miss in enumerate(missing):
        if miss and start is None:
            start = i
        if start is not None and (not miss or i == len(missing) - 1):
            end = i if miss else i - 1
            if arc_mm[end] - arc_mm[start] >= min_gap_mm:
                if float(np.median(own_dist_mm[start:end + 1])) < offset_mm:
                    offsets.append((float(arc_mm[start]), float(arc_mm[end])))
                else:
                    runs.append((max(0.0, arc_mm[start] - pad_mm), min(float(arc_mm[-1]), arc_mm[end] + pad_mm)))
            start = None
    merged: list[tuple[float, float]] = []
    for lo, hi in runs:
        if merged and lo - merged[-1][1] <= merge_mm:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged, offsets


def cut(points: np.ndarray, arc: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """The part of a densified polyline between arc lengths lo and hi, with interpolated ends."""
    inside = points[(arc > lo) & (arc < hi)]
    ends = [np.array([np.interp(v, arc, points[:, 0]), np.interp(v, arc, points[:, 1])]) for v in (lo, hi)]
    return np.vstack([ends[0], inside, ends[1]]) if len(inside) else np.vstack(ends)


def build_gap_trajectory(payload: dict, original: list[np.ndarray], segments: list[np.ndarray], drawing_id: str) -> dict:
    """Gap segments first (stroke ids 0..k-1), then every original stroke (ids k..), in normalised canvas units."""
    def as_points(arr):
        return [{"x": float(x), "y": float(y)} for x, y in arr]
    strokes = [{"stroke_id": i, "pen_state": "down", "points": as_points(s)} for i, s in enumerate(segments)]
    strokes += [{"stroke_id": len(segments) + i, "pen_state": "down", "points": as_points(s)}
                for i, s in enumerate(original)]
    out = {k: v for k, v in payload.items() if k not in ("strokes", "stroke_count", "point_count")}
    out.update({
        "drawing_id": drawing_id,
        "strokes": strokes,
        "stroke_count": len(strokes),
        "point_count": int(sum(len(s["points"]) for s in strokes)),
        "gap_strokes": len(segments),
        "note": f"Draw only the first {len(segments)} strokes (--max-strokes {len(segments)}); the rest keep the "
                "sheet placement of the original drawing.",
    })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--trajectory", required=True, type=Path, help="The trajectory that was drawn on the sheet")
    ap.add_argument("--config", type=Path, default=PROJECT / "config" / "xarm7_drawing_real.yaml")
    ap.add_argument("--sheet-mm", type=float, nargs=2, default=(279.4, 215.9))
    ap.add_argument("--px-per-mm", type=float, default=8.0)
    ap.add_argument("--margin-mm", type=float, default=6.0)
    ap.add_argument("--min-ink-area-mm2", type=float, default=0.5)
    ap.add_argument("--already-rectified", action="store_true")
    ap.add_argument("--gap-mm", type=float, default=1.0, help="Plan with no ink within this distance is missing")
    ap.add_argument("--min-gap-mm", type=float, default=1.5, help="Shorter misses are ignored (supervisor rule)")
    ap.add_argument("--pad-mm", type=float, default=2.0, help="Extend each redraw into the ink on both sides")
    ap.add_argument("--merge-mm", type=float, default=3.0, help="Join redraws closer than this")
    ap.add_argument("--offset-mm", type=float, default=4.0,
                    help="Uncovered stretches with ink running alongside closer than this are offsets, not gaps")
    ap.add_argument("--max-missing-share", type=float, default=0.5,
                    help="Refuse when more than this share of the plan reads as missing (likely a bad photo)")
    ap.add_argument("--force", action="store_true", help="Write the gap trajectory even above --max-missing-share")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    # Image-side imports here, so the gap logic above can be imported and tested without scipy/scikit-image.
    import cv2
    from scipy.spatial import cKDTree
    from ink_fidelity import ink_centreline_mm, ink_pixels_mm, planned_path_mm, rectify_sheet, rigid_icp

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"cannot read {args.image}")
    if args.already_rectified:
        sheet = cv2.resize(image, (int(max(args.sheet_mm) * args.px_per_mm), int(min(args.sheet_mm) * args.px_per_mm)))
    else:
        sheet = rectify_sheet(image, tuple(args.sheet_mm), args.px_per_mm)
    min_area_px = int(args.min_ink_area_mm2 * args.px_per_mm**2)
    ink = ink_centreline_mm(sheet, args.px_per_mm, args.margin_mm, min_area_px)
    if len(ink) < 50:
        raise SystemExit("almost no ink found; check lighting or --margin-mm")
    plan = planned_path_mm(args.trajectory, args.config, step_mm=0.25)
    rot, trans, fit_err = rigid_icp(ink, plan)  # fit on the centre line, as ink_fidelity.py does
    # Gaps are judged against every ink pixel: a wide or pooled line covers the plan across its whole width.
    ink_px = ink_pixels_mm(sheet, args.px_per_mm, args.margin_mm, min_area_px) @ rot.T + trans
    ink_tree = cKDTree(ink_px)

    drawing = yaml.safe_load(args.config.read_text())["drawing"]
    traj = load_trajectory(args.trajectory)
    original = [np.asarray(s["points"], dtype=np.float64) for s in traj["strokes"]]
    centre, scale_mm, flip = plane_frame(original, drawing["surface_size_xy_m"], bool(drawing["flip_image_y"]))

    dense_strokes, mm_strokes = [], []
    for pts in original:
        dense, _ = densify(pts, 0.25 / scale_mm)
        mm = (dense - centre) * scale_mm
        if flip:
            mm[:, 1] *= -1.0
        dense_strokes.append(dense)
        mm_strokes.append(mm)
    # Each ink pixel belongs to the stroke with the nearest planned point. Only a stroke's own ink can make an
    # uncovered stretch an offset; ink from a nearby eye or ear must not hide a missing outline.
    labels = np.concatenate([np.full(len(m), i) for i, m in enumerate(mm_strokes)])
    owner = labels[cKDTree(np.vstack(mm_strokes)).query(ink_px)[1]]

    segments, gaps, offsets, offset_segments, total_mm, missing_mm = [], [], [], [], 0.0, 0.0
    for stroke_index, (dense, mm) in enumerate(zip(dense_strokes, mm_strokes)):
        arc_mm = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(mm, axis=0), axis=1))])
        dist = ink_tree.query(mm)[0]
        own = ink_px[owner == stroke_index]
        own_dist = cKDTree(own).query(mm)[0] if len(own) else np.full(len(mm), np.inf)
        missing = dist > args.gap_mm
        total_mm += float(arc_mm[-1])
        missing_mm += float(np.sum(np.diff(arc_mm)[missing[1:]]))
        stroke_gaps, stroke_offsets = find_gaps(arc_mm, dist, own_dist, args.gap_mm, args.min_gap_mm,
                                                args.pad_mm, args.merge_mm, args.offset_mm)
        for lo, hi in stroke_gaps:
            segments.append(cut(dense, arc_mm, lo, hi))
            gaps.append({"stroke_index": stroke_index, "from_mm": round(lo, 2), "to_mm": round(hi, 2),
                         "length_mm": round(hi - lo, 2)})
        offset_segments += [cut(dense, arc_mm, lo, hi) for lo, hi in stroke_offsets]
        offsets += [{"stroke_index": stroke_index, "from_mm": round(lo, 2), "to_mm": round(hi, 2)}
                    for lo, hi in stroke_offsets]

    share = missing_mm / total_mm if total_mm else 0.0
    report = {
        "image": str(args.image), "trajectory": str(args.trajectory),
        "gap_mm": args.gap_mm, "min_gap_mm": args.min_gap_mm, "pad_mm": args.pad_mm,
        "plan_length_mm": round(total_mm, 1), "missing_share": round(share, 4),
        "fit_error_mm": round(float(fit_err), 3), "gap_count": len(gaps), "gaps": gaps,
        "offset_count": len(offsets), "offsets": offsets,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Overlay: plan in blue, gaps to redraw in red, offsets left alone in yellow, on the rectified sheet.
    overlay = sheet.copy()
    def to_px(norm_pts):
        mm = (norm_pts - centre) * scale_mm
        if flip:
            mm[:, 1] *= -1.0
        return (((mm - trans) @ rot) * args.px_per_mm).astype(int)
    for pts in original:
        for x, y in to_px(densify(pts, 0.25 / scale_mm)[0]):
            cv2.circle(overlay, (int(x), int(y)), 1, (214, 120, 42), -1)
    for seg in offset_segments:  # ink alongside: left alone (yellow)
        cv2.polylines(overlay, [to_px(densify(seg, 0.25 / scale_mm)[0]).reshape(-1, 1, 2)], False, (0, 200, 230), 3)
    for seg in segments:
        cv2.polylines(overlay, [to_px(densify(seg, 0.25 / scale_mm)[0]).reshape(-1, 1, 2)], False, (40, 40, 220), 3)
    cv2.imwrite(str(args.out_dir / "ink_gaps_overlay.png"), overlay)

    refused = share > args.max_missing_share and not args.force
    report["refused"] = refused
    if segments and not refused:
        payload = json.loads(args.trajectory.read_text())
        gap_traj = build_gap_trajectory(payload, original, segments, f"{traj['drawing_id']}_gaps")
        (args.out_dir / "gap_trajectory.json").write_text(json.dumps(gap_traj, indent=1))
        report["gap_trajectory"] = str(args.out_dir / "gap_trajectory.json")
        report["redraw_hint"] = f"--trajectory {report['gap_trajectory']} --max-strokes {len(segments)}"
    (args.out_dir / "ink_gaps.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("gaps", "offsets")}, indent=1))
    if refused:
        print(f"refused: {share:.0%} of the plan reads as missing, more than --max-missing-share "
              f"{args.max_missing_share:.0%}; check the overlay (bad photo or faint ink?) or pass --force",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
