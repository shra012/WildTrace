from __future__ import annotations

import pytest

from wildtrace.trajectory_candidates import (
    StrokePlanError,
    compile_stroke_plan,
    extract_json_plan,
    symmetric_raster_distance,
)


def test_compile_stroke_plan_resamples_normalized_points() -> None:
    trajectory = compile_stroke_plan(
        {"drawing_id": "line", "strokes": [{"points": [[0, 0], [1, 0]]}]},
        max_segment_length=0.25,
    )
    assert trajectory["drawing_id"] == "line"
    assert len(trajectory["strokes"][0]["points"]) == 5
    assert trajectory["strokes"][0]["points"][-1] == [1.0, 0.0]


def test_compile_stroke_plan_rejects_out_of_canvas_point() -> None:
    with pytest.raises(StrokePlanError, match="outside normalized canvas"):
        compile_stroke_plan({"drawing_id": "bad", "strokes": [{"points": [[0, 0], [1.1, 0]]}]})


def test_extract_json_plan_accepts_a_fenced_model_response() -> None:
    plan = extract_json_plan('Here is the plan:\n```json\n{"strokes": []}\n```')
    assert plan == {"strokes": []}


def test_symmetric_raster_distance_identical_paths_is_zero() -> None:
    trajectory = compile_stroke_plan(
        {"drawing_id": "line", "strokes": [{"points": [[0.1, 0.1], [0.9, 0.9]]}]}
    )
    result = symmetric_raster_distance(trajectory, trajectory, size=64)
    assert result["symmetric_mean_distance_px"] == 0.0
    assert result["pixel_iou"] == 1.0
