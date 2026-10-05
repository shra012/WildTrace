"""Compile and score learned stroke plans against WildTrace trajectories.

A vision-language model should emit a small ordered stroke plan, not thousands
of raw waypoints.  This module validates that plan and deterministically
resamples it into WildTrace's existing normalized trajectory representation.
"""
from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


class StrokePlanError(ValueError):
    """Raised when a model response cannot be used safely as a drawing plan."""


def extract_json_plan(model_text: str) -> Mapping[str, Any]:
    """Extract one JSON object from a VLM answer without accepting prose as data."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(model_text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(model_text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return value
    raise StrokePlanError("Model response did not contain a JSON object")


def _point(value: Any) -> tuple[float, float]:
    if isinstance(value, Mapping):
        value = (value.get("x"), value.get("y"))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        raise StrokePlanError(f"Point must be [x, y], got {value!r}")
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise StrokePlanError(f"Point is not numeric: {value!r}") from exc
    if not (math.isfinite(x) and math.isfinite(y)):
        raise StrokePlanError(f"Point is not finite: {value!r}")
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise StrokePlanError(f"Point is outside normalized canvas [0, 1]: {value!r}")
    return x, y


def _resample_polyline(points: list[tuple[float, float]], max_segment_length: float) -> list[list[float]]:
    if max_segment_length <= 0:
        raise ValueError("max_segment_length must be positive")
    dense: list[list[float]] = [[points[0][0], points[0][1]]]
    for start, end in zip(points, points[1:]):
        distance = math.dist(start, end)
        count = max(1, math.ceil(distance / max_segment_length))
        for index in range(1, count + 1):
            alpha = index / count
            dense.append([
                start[0] + alpha * (end[0] - start[0]),
                start[1] + alpha * (end[1] - start[1]),
            ])
    return dense


def compile_stroke_plan(
    plan: Mapping[str, Any],
    *,
    drawing_id: str | None = None,
    max_segment_length: float = 0.01,
    max_control_points_per_stroke: int = 64,
) -> dict[str, Any]:
    """Convert a sparse model plan to the canonical WildTrace trajectory JSON.

    Expected plan schema::

        {"drawing_id": "optional", "strokes": [{"points": [[x, y], ...]}, ...]}

    Coordinates must be normalized to [0, 1], with y increasing downward, the
    same convention used by the existing SVG/trajectory exporter. Pen lifts are
    represented by separate stroke objects; the compiler never invents a pen-
    down transfer between strokes.
    """
    raw_strokes = plan.get("strokes")
    if not isinstance(raw_strokes, Sequence) or isinstance(raw_strokes, (str, bytes)):
        raise StrokePlanError("Plan must contain a strokes list")
    resolved_id = drawing_id or plan.get("drawing_id") or plan.get("id")
    if not isinstance(resolved_id, str) or not resolved_id.strip():
        raise StrokePlanError("A non-empty drawing_id is required")

    strokes: list[dict[str, Any]] = []
    for ordinal, raw_stroke in enumerate(raw_strokes):
        if not isinstance(raw_stroke, Mapping):
            raise StrokePlanError(f"Stroke {ordinal} is not an object")
        raw_points = raw_stroke.get("points")
        if not isinstance(raw_points, Sequence) or isinstance(raw_points, (str, bytes)):
            raise StrokePlanError(f"Stroke {ordinal} has no points list")
        if len(raw_points) > max_control_points_per_stroke:
            raise StrokePlanError(
                f"Stroke {ordinal} has {len(raw_points)} control points; limit is {max_control_points_per_stroke}"
            )
        points: list[tuple[float, float]] = []
        for raw_point in raw_points:
            point = _point(raw_point)
            if not points or point != points[-1]:
                points.append(point)
        if len(points) < 2:
            raise StrokePlanError(f"Stroke {ordinal} needs at least two distinct points")
        strokes.append({"stroke_id": ordinal, "points": _resample_polyline(points, max_segment_length)})
    if not strokes:
        raise StrokePlanError("Plan must contain at least one stroke")
    return {"drawing_id": resolved_id.strip(), "strokes": strokes}


def rasterize_trajectory(trajectory: Mapping[str, Any], size: int = 512) -> np.ndarray:
    """Render canonical normalized strokes into a binary image for comparison."""
    if size < 8:
        raise ValueError("size must be at least 8")
    image = np.zeros((size, size), dtype=np.uint8)
    # A dependency-free sampled rasterizer. It deliberately matches stroke
    # geometry, not line width, so candidate and baseline are compared fairly.
    for stroke in trajectory["strokes"]:
        points = [_point(point) for point in stroke["points"]]
        for start, end in zip(points, points[1:]):
            samples = max(2, int(math.ceil(math.dist(start, end) * size * 2)))
            for alpha in np.linspace(0.0, 1.0, samples):
                x = round((start[0] + alpha * (end[0] - start[0])) * (size - 1))
                y = round((start[1] + alpha * (end[1] - start[1])) * (size - 1))
                image[int(y), int(x)] = 1
    return image


def symmetric_raster_distance(reference: Mapping[str, Any], candidate: Mapping[str, Any], size: int = 512) -> dict[str, float]:
    """Return symmetric pixel distance and overlap for a candidate trajectory.

    The score is an offline geometry comparison only. Robot validity must still
    be measured by the existing Isaac Sim reach/contact metrics.
    """
    reference_image = rasterize_trajectory(reference, size)
    candidate_image = rasterize_trajectory(candidate, size)
    if not reference_image.any() or not candidate_image.any():
        raise StrokePlanError("Both trajectories must render at least one pixel")
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - OpenCV is a project dependency.
        raise RuntimeError("Trajectory comparison requires opencv-python-headless") from exc

    to_reference = cv2.distanceTransform((1 - reference_image).astype(np.uint8), cv2.DIST_L2, 3)
    to_candidate = cv2.distanceTransform((1 - candidate_image).astype(np.uint8), cv2.DIST_L2, 3)
    candidate_to_reference = float(to_reference[candidate_image.astype(bool)].mean())
    reference_to_candidate = float(to_candidate[reference_image.astype(bool)].mean())
    intersection = int(np.logical_and(reference_image, candidate_image).sum())
    union = int(np.logical_or(reference_image, candidate_image).sum())
    return {
        "raster_size_px": float(size),
        "candidate_to_reference_px": candidate_to_reference,
        "reference_to_candidate_px": reference_to_candidate,
        "symmetric_mean_distance_px": (candidate_to_reference + reference_to_candidate) / 2.0,
        "pixel_iou": intersection / union if union else 0.0,
    }
