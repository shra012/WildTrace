"""Path tangent, heading-error, corner and joint-slew helpers.

Pure NumPy so the unit tests can exercise these without an Isaac Sim runtime.
Headings are drawing-plane XY tangents of the desired pen path; they are not a
mobile-base yaw. The pen orientation itself stays fixed by configuration.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def wrap_angle(angle):
    """Wrap to [-pi, pi] as atan2(sin, cos) so no branch cut is introduced."""
    values = np.asarray(angle, dtype=np.float64)
    wrapped = np.arctan2(np.sin(values), np.cos(values))
    return float(wrapped) if values.ndim == 0 else wrapped


def heading_error(desired_rad, actual_rad):
    """Wrapped desired-minus-actual heading difference."""
    return wrap_angle(np.asarray(desired_rad, dtype=np.float64) - np.asarray(actual_rad, dtype=np.float64))


def _xy(points: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(f"Expected an Nx2 or Nx3 path, got shape {array.shape}")
    return array[:, :2]


def polyline_headings(
    points: Sequence[Sequence[float]], min_segment_m: float = 0.0, trailing: bool = False
) -> np.ndarray:
    """Tangent heading per vertex, measured over a chord of at least min_segment_m.

    With the default of zero this is a plain forward difference. A positive
    min_segment_m instead takes the chord to the first vertex that far away,
    which matters when the polyline is densified well below the scale of the
    source data: adjacent 1 mm segments of a contour-extracted outline can
    reverse direction entirely, so point-to-point tangents report extraction
    noise rather than the path direction.

    ``trailing`` measures the chord arriving at each vertex rather than the
    one leaving it, which is what heading_from_history necessarily does since
    it cannot see the future. Comparing a leading desired tangent against a
    trailing measured heading offsets the two by the path curvature over
    twice the chord, so they must agree on the direction.

    Vertices with no qualifying chord inherit the nearest one that has it.
    """
    xy = _xy(points)
    count = len(xy)
    if count < 2:
        return np.zeros(count, dtype=np.float64)
    threshold = max(float(min_segment_m), 1e-12)
    headings = np.full(count, np.nan, dtype=np.float64)
    for index in range(count):
        others = range(index - 1, -1, -1) if trailing else range(index + 1, count)
        for other in others:
            delta = xy[index] - xy[other] if trailing else xy[other] - xy[index]
            if float(np.linalg.norm(delta)) >= threshold:
                headings[index] = float(np.arctan2(delta[1], delta[0]))
                break
    known = ~np.isnan(headings)
    if not known.any():
        return np.zeros(count, dtype=np.float64)
    # Forward-fill covers a leading chord running out at the end of the path;
    # the backfill covers a trailing chord having no history at the start.
    last = None
    for index in range(count):
        if known[index]:
            last = headings[index]
        elif last is not None:
            headings[index] = last
    return np.where(np.isnan(headings), headings[known][0], headings)


def heading_from_history(history_xy: Sequence[Sequence[float]], min_displacement_m: float):
    """Most recent travel direction that clears the measurement noise floor.

    Walking backwards returns the closest-in-time sample that is far enough
    away, which stays responsive through a corner while ignoring the
    sub-millimetre jitter of a waypoint settling in place. Consecutive-step
    differencing cannot do this: while the tip settles, its per-step motion
    points in essentially random directions.

    Returns None when the tip has not travelled far enough to have a
    meaningful direction, so the caller can carry the previous heading.
    """
    points = np.asarray(history_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2 or len(points) < 2:
        return None
    threshold = float(min_displacement_m)
    current = points[-1, :2]
    for index in range(len(points) - 2, -1, -1):
        displacement = current - points[index, :2]
        if float(np.linalg.norm(displacement)) >= threshold:
            return float(np.arctan2(displacement[1], displacement[0]))
    return None


def turn_angles(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Absolute direction change at each vertex; endpoints are zero."""
    xy = _xy(points)
    turns = np.zeros(len(xy), dtype=np.float64)
    if len(xy) < 3:
        return turns
    incoming = xy[1:-1] - xy[:-2]
    outgoing = xy[2:] - xy[1:-1]
    incoming_length = np.linalg.norm(incoming, axis=1)
    outgoing_length = np.linalg.norm(outgoing, axis=1)
    usable = (incoming_length > 1e-12) & (outgoing_length > 1e-12)
    cosine = np.ones(len(incoming), dtype=np.float64)
    cosine[usable] = np.sum(incoming[usable] * outgoing[usable], axis=1) / (
        incoming_length[usable] * outgoing_length[usable]
    )
    turns[1:-1] = np.arccos(np.clip(cosine, -1.0, 1.0))
    return turns


def corner_flags(
    points: Sequence[Sequence[float]], corner_angle_degrees: float, window: int = 0
) -> np.ndarray:
    """Mark vertices whose direction change reaches the threshold.

    ``window`` also marks neighbouring vertices, because overshoot appears on
    the waypoints just after a vertex rather than exactly on it.
    """
    if corner_angle_degrees <= 0:
        raise ValueError("corner_angle_degrees must be positive")
    if window < 0:
        raise ValueError("window must be non-negative")
    flags = turn_angles(points) >= math.radians(float(corner_angle_degrees))
    if not window or not flags.any():
        return flags
    spread = flags.copy()
    for offset in range(1, int(window) + 1):
        spread[offset:] |= flags[:-offset]
        spread[:-offset] |= flags[offset:]
    return spread


def densify_near_corners(
    points: Sequence[Sequence[float]],
    corner_angle_degrees: float,
    window: int,
    factor: int,
) -> np.ndarray:
    """Subdivide only the segments touching a corner, keeping every vertex.

    Straight sections keep their original spacing, so this adds gated stops
    through a turn without slowing the whole stroke.
    """
    source = np.asarray(points, dtype=np.float64)
    if int(factor) <= 1 or len(source) < 3:
        return source.copy()
    flags = corner_flags(source, corner_angle_degrees, window)
    result = [source[0]]
    for index in range(len(source) - 1):
        start, end = source[index], source[index + 1]
        subdivisions = int(factor) if (flags[index] or flags[index + 1]) else 1
        for step in range(1, subdivisions + 1):
            result.append(start + (end - start) * (step / subdivisions))
    return np.asarray(result, dtype=np.float64)


def limit_joint_delta(current_q, proposed_q, max_delta_rad) -> np.ndarray:
    """Clamp a commanded joint target to a per-step change around the measurement."""
    current = np.asarray(current_q, dtype=np.float64)
    proposed = np.asarray(proposed_q, dtype=np.float64)
    if current.shape != proposed.shape:
        raise ValueError(f"Shape mismatch: {current.shape} and {proposed.shape}")
    if max_delta_rad is None or float(max_delta_rad) <= 0.0:
        return proposed.copy()
    limit = float(max_delta_rad)
    return current + np.clip(proposed - current, -limit, limit)
