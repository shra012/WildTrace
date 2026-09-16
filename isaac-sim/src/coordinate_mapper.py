"""Corner-preserving resampling and SI-unit drawing-plane mapping."""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


def remove_consecutive_duplicates(points: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return points.copy()
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > epsilon]
    return points[keep]


def resample_polyline(points: Sequence[Sequence[float]], max_step: float) -> np.ndarray:
    """Subdivide every segment, retaining every original vertex/corner."""
    if max_step <= 0:
        raise ValueError("max_step must be positive")
    source = remove_consecutive_duplicates(np.asarray(points, dtype=np.float64))
    if len(source) < 2:
        raise ValueError("Polyline needs two distinct points")
    result = [source[0]]
    for start, end in zip(source[:-1], source[1:]):
        distance = float(np.linalg.norm(end - start))
        subdivisions = max(1, int(math.ceil(distance / max_step)))
        for index in range(1, subdivisions + 1):
            result.append(start + (end - start) * (index / subdivisions))
    return np.asarray(result, dtype=np.float64)


def corner_preserving_smooth(
    points: Sequence[Sequence[float]], strength: float = 0.0, corner_angle_degrees: float = 35.0
) -> np.ndarray:
    """One conservative Laplacian pass; endpoints and sharp corners stay exact."""
    result = np.asarray(points, dtype=np.float64).copy()
    if not 0.0 <= strength <= 0.5:
        raise ValueError("smoothing strength must be in [0, 0.5]")
    if strength == 0.0 or len(result) < 3:
        return result
    original = result.copy()
    threshold = math.radians(corner_angle_degrees)
    for index in range(1, len(original) - 1):
        incoming = original[index] - original[index - 1]
        outgoing = original[index + 1] - original[index]
        ni, no = np.linalg.norm(incoming), np.linalg.norm(outgoing)
        if ni < 1e-12 or no < 1e-12:
            continue
        turn = math.acos(float(np.clip(np.dot(incoming, outgoing) / (ni * no), -1.0, 1.0)))
        if turn < threshold:
            neighbor_mean = 0.5 * (original[index - 1] + original[index + 1])
            result[index] = (1.0 - strength) * original[index] + strength * neighbor_mean
    return result


def map_trajectory_to_plane(
    trajectory: Dict[str, Any],
    *,
    center_xy: Sequence[float],
    size_xy: Sequence[float],
    flip_image_y: bool = True,
    max_step: float = 0.005,
    smoothing_strength: float = 0.0,
    corner_angle_degrees: float = 35.0,
) -> Dict[str, Any]:
    """Map all strokes into a centred XY rectangle in metres.

    A single global scale is used, so aspect ratio and relative stroke placement
    are preserved. Image +Y is flipped by default to become drawing-plane +Y.
    """
    center = np.asarray(center_xy, dtype=np.float64)
    size = np.asarray(size_xy, dtype=np.float64)
    if center.shape != (2,) or size.shape != (2,) or np.any(size <= 0):
        raise ValueError("center_xy and positive size_xy must each have two values")
    all_points = np.vstack([np.asarray(s["points"], dtype=np.float64) for s in trajectory["strokes"]])
    if not np.isfinite(all_points).all():
        raise ValueError("Non-finite trajectory coordinate")
    minimum, maximum = all_points.min(axis=0), all_points.max(axis=0)
    span = maximum - minimum
    active = span > 1e-12
    if not active.any():
        raise ValueError("Trajectory has zero extent")
    scales = size[active] / span[active]
    scale = float(scales.min())
    source_center = 0.5 * (minimum + maximum)

    mapped_strokes: List[Dict[str, Any]] = []
    for stroke in trajectory["strokes"]:
        points = np.asarray(stroke["points"], dtype=np.float64)
        mapped = (points - source_center) * scale
        if flip_image_y:
            mapped[:, 1] *= -1.0
        mapped += center
        mapped = corner_preserving_smooth(mapped, smoothing_strength, corner_angle_degrees)
        mapped = resample_polyline(mapped, max_step)
        mapped_strokes.append({"stroke_id": int(stroke["stroke_id"]), "points": mapped.tolist()})
    return {
        "drawing_id": str(trajectory["drawing_id"]),
        "strokes": mapped_strokes,
        "mapping": {
            "units": "metres",
            "center_xy": center.tolist(),
            "size_xy": size.tolist(),
            "source_bbox": [minimum.tolist(), maximum.tolist()],
            "uniform_scale": scale,
            "image_y_flipped": bool(flip_image_y),
        },
    }


def interpolate_segment(start: Iterable[float], end: Iterable[float], max_step: float) -> np.ndarray:
    return resample_polyline([list(start), list(end)], max_step)

