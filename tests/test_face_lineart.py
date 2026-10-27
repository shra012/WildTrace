from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wildtrace.face_lineart import (
    DEFAULT_SETTINGS,
    FaceLineArt,
    _chain,
    _clip_to_box,
    estimate_draw_seconds,
    generate_face_trajectory,
    order_strokes,
    to_trajectory,
)

MODEL_DIR = Path(DEFAULT_SETTINGS["model_dir"]).expanduser()
SAMPLE = MODEL_DIR / "portrait.jpg"


def test_chain_turns_edges_into_open_and_closed_polylines():
    open_chain = [(1, 2), (2, 3), (3, 4)]
    loop = [(10, 11), (11, 12), (12, 10)]
    paths = _chain(open_chain + loop)
    assert [1, 2, 3, 4] in paths or [4, 3, 2, 1] in paths
    closed = next(p for p in paths if 10 in p)
    assert closed[0] == closed[-1] and len(closed) == 4


def test_order_strokes_starts_with_face_outline_and_reverses_to_shorten_travel():
    strokes = [np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[5.0, 0.0], [1.1, 0.0]]), np.array([[9.0, 9.0], [9.0, 8.0]])]
    ordered, labels = order_strokes(strokes, ["nose", "eye", "face_outline"])
    assert labels[0] == "face_outline"
    # The eye stroke is reversed so it starts next to where the nose stroke ends.
    nose_end = ordered[labels.index("nose")][-1]
    eye_start = ordered[labels.index("eye")][0]
    assert np.linalg.norm(eye_start - nose_end) < 4.0 or labels.index("eye") < labels.index("nose")


def test_clip_to_box_splits_a_stroke_leaving_the_box():
    stroke = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0], [1.5, 0.0], [1.0, 0.0]])
    runs = _clip_to_box(stroke, (0.0, -1.0, 2.0, 1.0))
    assert len(runs) == 2 and all(len(r) == 2 for r in runs)


def test_trajectory_is_normalized_in_the_gold_schema():
    art = FaceLineArt(
        strokes=[np.array([[100.0, 100.0], [300.0, 100.0], [300.0, 500.0]]), np.array([[150.0, 200.0], [200.0, 220.0]])],
        labels=["face_outline", "nose"],
        size=(640, 640),
        backend="mediapipe (CPU)",
    )
    trajectory = to_trajectory(art, margin=0.05)
    points = np.array([[p["x"], p["y"]] for s in trajectory["strokes"] for p in s["points"]])
    assert points.min() >= 0.0 and points.max() <= 1.0
    assert trajectory["stroke_count"] == 2 and trajectory["point_count"] == 5
    assert trajectory["coordinate_frame"] == "normalized_canvas"
    assert all(s["pen_state"] == "down" for s in trajectory["strokes"])
    assert estimate_draw_seconds(trajectory) > 12.0  # at least the two pen lifts


@pytest.mark.skipif(
    not (SAMPLE.exists() and (MODEL_DIR / "face_landmarker.task").exists()),
    reason="MediaPipe models / sample portrait not cached (no downloads in tests)",
)
def test_sample_portrait_end_to_end(tmp_path):
    events = []
    result = generate_face_trajectory(image_path=SAMPLE, out_dir=tmp_path, emit=events.append)
    trajectory = json.loads(Path(result["trajectory_path"]).read_text())
    labels = set(trajectory["stroke_labels"])
    assert {"face_outline", "left_eye", "right_eye", "lips", "nose"} <= labels
    assert 8 <= trajectory["stroke_count"] <= 40
    assert [(e.name, e.status) for e in events] == [
        ("face_landmarks", "running"), ("face_landmarks", "done"), ("extract", "running"), ("extract", "done"),
    ]
    assert Path(result["svg_path"]).exists() and Path(result["lineart_path"]).exists()


def test_distance_to_open_polylines_ignores_the_closing_edge():
    from wildtrace.face_lineart import _distance_to_polylines

    arc = np.array([[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0]])  # open at the top
    points = np.array([[5.0, 0.0], [5.0, 9.0]])
    distances = _distance_to_polylines(points, [arc])
    assert distances[0] == pytest.approx(5.0)  # a closed polygon would say 0 (on the closing edge)
    assert distances[1] == pytest.approx(1.0)


def test_trim_forehead_keeps_the_outline_below_the_brows():
    from wildtrace.face_lineart import trim_forehead

    angles = np.linspace(0, 2 * np.pi, 37)
    oval = np.column_stack([100 + 40 * np.cos(angles), 100 + 60 * np.sin(angles)])  # y down, top at y=40
    runs = trim_forehead(oval, brow_y=80.0)
    assert len(runs) == 1 and len(runs[0]) > 20
    assert runs[0][:, 1].min() >= 80.0 and runs[0][:, 1].max() == pytest.approx(160.0, abs=1.0)


def test_hair_over_forehead_needs_hair_above_the_face():
    from wildtrace.face_lineart import HAIR, hair_over_forehead

    outline = np.array([[40.0, 50.0], [60.0, 50.0], [60.0, 90.0], [40.0, 90.0]])
    category = np.zeros((120, 100), dtype=np.uint8)
    assert not hair_over_forehead(category, outline)  # bald: keep the whole face outline
    category[30:55, 30:70] = HAIR
    assert hair_over_forehead(category, outline)


def test_chaikin_keeps_end_points_and_adds_points():
    from wildtrace.face_lineart import _chaikin

    line = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    smooth = _chaikin(line, 2)
    assert np.allclose(smooth[0], line[0]) and np.allclose(smooth[-1], line[-1])
    assert len(smooth) > len(line)
