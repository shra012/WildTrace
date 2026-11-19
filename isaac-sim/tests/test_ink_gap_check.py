"""Camera gap check: gap finding, offset handling and the redraw trajectory (no camera, no arm)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from coordinate_mapper import map_trajectory_to_plane  # noqa: E402
from ink_gap_check import build_gap_trajectory, cut, densify, find_gaps  # noqa: E402
from trajectory_loader import standardize_trajectory  # noqa: E402


def _line(n=401, step=0.25):
    arc = np.arange(n) * step  # 100 mm stroke
    return arc


def test_missing_stretch_becomes_one_padded_gap():
    arc = _line()
    dist = np.zeros_like(arc)
    dist[(arc > 40) & (arc < 50)] = 10.0  # 10 mm with no ink anywhere near
    gaps, offsets = find_gaps(arc, dist, dist, gap_mm=1.0, min_gap_mm=1.5, pad_mm=2.0, merge_mm=3.0, offset_mm=4.0)
    assert offsets == []
    assert len(gaps) == 1
    lo, hi = gaps[0]
    assert abs(lo - 38.25) < 0.3 and abs(hi - 51.75) < 0.3


def test_short_misses_are_ignored_and_close_gaps_merge():
    arc = _line()
    dist = np.zeros_like(arc)
    dist[(arc > 10) & (arc < 11)] = 10.0  # 1 mm: below min_gap_mm
    dist[(arc > 30) & (arc < 35)] = 10.0
    dist[(arc > 37) & (arc < 42)] = 10.0  # 2 mm apart: padded, they overlap and merge
    gaps, _ = find_gaps(arc, dist, dist, gap_mm=1.0, min_gap_mm=1.5, pad_mm=2.0, merge_mm=3.0, offset_mm=4.0)
    assert len(gaps) == 1 and gaps[0][0] < 30 and gaps[0][1] > 42


def test_ink_alongside_is_an_offset_not_a_gap():
    # The line was drawn 2 mm beside the plan: redrawing it would add a second, parallel line.
    arc = _line()
    dist = np.zeros_like(arc)
    dist[(arc > 20) & (arc < 35)] = 2.0
    gaps, offsets = find_gaps(arc, dist, dist, gap_mm=1.0, min_gap_mm=1.5, pad_mm=2.0, merge_mm=3.0, offset_mm=4.0)
    assert gaps == [] and len(offsets) == 1


def test_other_strokes_ink_does_not_hide_a_gap():
    # An eye 2 mm from a missing outline: close to some ink, but none of it belongs to this stroke.
    arc = _line()
    dist = np.zeros_like(arc)
    dist[(arc > 20) & (arc < 35)] = 2.0
    own = dist.copy()
    own[(arc > 20) & (arc < 35)] = np.inf
    gaps, offsets = find_gaps(arc, dist, own, gap_mm=1.0, min_gap_mm=1.5, pad_mm=2.0, merge_mm=3.0, offset_mm=4.0)
    assert offsets == [] and len(gaps) == 1


def test_cut_returns_the_interval_with_interpolated_ends():
    pts, arc = densify(np.array([[0.0, 0.0], [10.0, 0.0]]), 1.0)
    seg = cut(pts, arc, 2.5, 6.5)
    assert np.allclose(seg[0], [2.5, 0.0]) and np.allclose(seg[-1], [6.5, 0.0])
    assert np.all(np.diff(seg[:, 0]) > 0)


def test_gap_trajectory_keeps_the_sheet_placement():
    payload = {
        "drawing_id": "cat_test",
        "canvas_width": 580, "canvas_height": 447,
        "strokes": [
            {"stroke_id": 0, "pen_state": "down", "points": [{"x": 0.1, "y": 0.2}, {"x": 0.5, "y": 0.9}]},
            {"stroke_id": 1, "pen_state": "down", "points": [{"x": 0.6, "y": 0.1}, {"x": 0.9, "y": 0.4}]},
        ],
    }
    original = [np.array([[0.1, 0.2], [0.5, 0.9]]), np.array([[0.6, 0.1], [0.9, 0.4]])]
    pts, arc = densify(original[0], 0.01)
    gap = cut(pts, arc, 0.2, 0.4)
    out = build_gap_trajectory(payload, original, [gap], "cat_test_gaps")
    assert out["gap_strokes"] == 1 and out["stroke_count"] == 3 and out["canvas_width"] == 580

    def mapped(traj):
        return map_trajectory_to_plane(standardize_trajectory(traj), center_xy=[0.4, 0.0], size_xy=[0.16, 0.12],
                                       max_step=1.0)

    before = mapped(payload)["strokes"][0]["points"]
    after = mapped(out)
    # The gap stroke lies on the original stroke, in the same place on the sheet.
    a, b = np.asarray(before[0])[:2], np.asarray(before[-1])[:2]
    direction = (b - a) / np.linalg.norm(b - a)
    for p in np.asarray(after["strokes"][0]["points"])[:, :2]:
        cross = (p - a)[0] * direction[1] - (p - a)[1] * direction[0]
        assert abs(cross) < 1e-9
    # The original strokes follow unchanged, so the bounding box (and the mapping) is the original one.
    assert np.allclose(after["strokes"][1]["points"], before)
    assert after["mapping"]["uniform_scale"] == mapped(payload)["mapping"]["uniform_scale"]
