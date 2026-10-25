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
