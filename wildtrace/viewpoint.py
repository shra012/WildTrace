from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from wildtrace.images import connected_components


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _binary_mask(mask: Image.Image) -> np.ndarray:
    arr = np.asarray(mask.convert("L"), dtype=np.uint8)
    return np.where(arr > 0, 1, 0).astype(np.uint8)


def _bbox(binary: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def extract_view_features(mask: Image.Image) -> dict[str, float | int]:
    binary = _binary_mask(mask)
    height, width = binary.shape
    total_pixels = float(height * width or 1)
    foreground = float(np.count_nonzero(binary))
    bbox = _bbox(binary)
    if bbox is None or foreground == 0:
        return {
            "component_count": 0,
            "largest_component_ratio": 0.0,
            "mask_coverage_ratio": 0.0,
            "bbox_area_ratio": 0.0,
            "bbox_fill_ratio": 0.0,
            "border_touch_ratio": 0.0,
            "centroid_x_offset": 0.0,
            "centroid_y_offset": 0.0,
            "symmetry_score": 0.0,
            "dominant_axis_angle_deg": 0.0,
            "direction_score": 0.0,
            "left_mass_ratio": 0.0,
            "right_mass_ratio": 0.0,
            "top_mass_ratio": 0.0,
            "bottom_mass_ratio": 0.0,
        }

    x_min, y_min, x_max, y_max = bbox
    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1
    bbox_area = float(bbox_width * bbox_height)

    components = connected_components(binary)
    component_sizes = sorted((len(component) for component in components), reverse=True)
    largest_component_ratio = float(component_sizes[0]) / foreground if component_sizes else 0.0

    touched_edges = 0
    touched_edges += int(binary[:, 0].any())
    touched_edges += int(binary[:, -1].any())
    touched_edges += int(binary[0, :].any())
    touched_edges += int(binary[-1, :].any())

    ys, xs = np.nonzero(binary)
    centroid_x = float(xs.mean())
    centroid_y = float(ys.mean())
    centroid_x_offset = (centroid_x - ((width - 1) / 2.0)) / max(width / 2.0, 1.0)
    centroid_y_offset = (centroid_y - ((height - 1) / 2.0)) / max(height / 2.0, 1.0)

    flipped = np.fliplr(binary)
    union = np.logical_or(binary, flipped).sum()
    intersection = np.logical_and(binary, flipped).sum()
    symmetry_score = float(intersection) / float(union or 1)

    points = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    centered_points = points - points.mean(axis=0, keepdims=True)
    covariance = np.cov(centered_points.T) if len(points) > 1 else np.eye(2, dtype=np.float32)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal = eigenvectors[:, int(np.argmax(eigenvalues))]
    dominant_axis_angle_deg = float(np.degrees(np.arctan2(principal[1], principal[0])))

    left_mass_ratio = float(binary[:, : width // 2].sum()) / foreground
    right_mass_ratio = float(binary[:, width // 2 :].sum()) / foreground
    top_mass_ratio = float(binary[: height // 2, :].sum()) / foreground
    bottom_mass_ratio = float(binary[height // 2 :, :].sum()) / foreground

    bbox_binary = binary[y_min : y_max + 1, x_min : x_max + 1]
    column_sums = bbox_binary.sum(axis=0).astype(np.float32)
    band_width = max(1, bbox_width // 5)
    left_tip_mass = float(column_sums[:band_width].sum())
    right_tip_mass = float(column_sums[-band_width:].sum())
    tip_total = left_tip_mass + right_tip_mass
    tip_bias = 0.0 if tip_total == 0 else (right_tip_mass - left_tip_mass) / tip_total
    bbox_centroid_x_offset = (centroid_x - ((x_min + x_max) / 2.0)) / max(bbox_width / 2.0, 1.0)
    direction_score = float(
        np.clip(
            -2.5 * ((0.7 * (left_mass_ratio - right_mass_ratio)) + (0.2 * -tip_bias) + (0.1 * -bbox_centroid_x_offset)),
            -1.0,
            1.0,
        )
    )

    return {
        "component_count": int(len(component_sizes)),
        "largest_component_ratio": largest_component_ratio,
        "mask_coverage_ratio": foreground / total_pixels,
        "bbox_area_ratio": bbox_area / total_pixels,
        "bbox_fill_ratio": foreground / max(bbox_area, 1.0),
        "border_touch_ratio": touched_edges / 4.0,
        "centroid_x_offset": float(centroid_x_offset),
        "centroid_y_offset": float(centroid_y_offset),
        "symmetry_score": symmetry_score,
        "dominant_axis_angle_deg": dominant_axis_angle_deg,
        "direction_score": direction_score,
        "left_mass_ratio": left_mass_ratio,
        "right_mass_ratio": right_mass_ratio,
        "top_mass_ratio": top_mass_ratio,
        "bottom_mass_ratio": bottom_mass_ratio,
    }


@dataclass(slots=True)
class ViewpointResult:
    bucket: str
    confidence: float
    margin: float
    scores: dict[str, float]
    status: str
    flags: list[str]
    allowed_for_outline: bool


class LocalScoreViewpointBackend:
    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = settings

    def classify(self, sample: dict[str, Any], config: dict[str, Any]) -> ViewpointResult:
        features = sample.get("view_features") or {}
        flags: list[str] = []
        prefilter = config["prefilter"]
        if prefilter.get("require_mask", True) and not sample.get("mask_available"):
            return ViewpointResult(
                bucket="unknown",
                confidence=0.0,
                margin=0.0,
                scores={},
                status="rejected",
                flags=["missing_mask"],
                allowed_for_outline=False,
            )
        if not features:
            return ViewpointResult(
                bucket="unknown",
                confidence=0.0,
                margin=0.0,
                scores={},
                status="rejected",
                flags=["missing_view_features"],
                allowed_for_outline=False,
            )

        if float(features["mask_coverage_ratio"]) < float(prefilter["min_mask_coverage_ratio"]):
            flags.append("low_mask_coverage")
        if float(features["border_touch_ratio"]) > float(prefilter["max_border_touch_ratio"]):
            flags.append("high_border_touch_ratio")
        if int(features["component_count"]) > int(prefilter["max_component_count"]):
            flags.append("too_many_components")
        if float(features["largest_component_ratio"]) < float(prefilter["min_largest_component_ratio"]):
            flags.append("low_largest_component_ratio")
        if float(features["bbox_fill_ratio"]) < float(prefilter["min_bbox_fill_ratio"]):
            flags.append("low_bbox_fill_ratio")

        if flags:
            return ViewpointResult(
                bucket="unknown",
                confidence=0.0,
                margin=0.0,
                scores={},
                status="rejected",
                flags=flags,
                allowed_for_outline=False,
            )

        scores = self._score_buckets(features, config["classifier"])
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_bucket, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        score_sum = sum(scores.values())
        confidence = float(top_score / max(score_sum, 1e-6))
        margin = float(top_score - second_score)

        status = "accepted"
        if confidence < float(config["min_confidence"]):
            status = "unknown"
            flags.append("low_confidence")
        if margin < float(config["min_margin"]):
            status = "unknown"
            flags.append("low_margin")
        if top_bucket in config.get("rejected_buckets", []):
            status = "rejected"
            flags.append("rejected_bucket")
        if top_bucket not in config.get("allowed_buckets", []):
            status = "unknown"
            flags.append("disallowed_bucket")

        return ViewpointResult(
            bucket=top_bucket if status == "accepted" else "unknown",
            confidence=confidence,
            margin=margin,
            scores=scores,
            status=status,
            flags=flags,
            allowed_for_outline=(status == "accepted"),
        )

    def _score_buckets(self, features: dict[str, Any], settings: dict[str, Any]) -> dict[str, float]:
        symmetry = _clamp01(features["symmetry_score"])
        centered = _clamp01(1.0 - abs(float(features["centroid_x_offset"])))
        direction = float(features["direction_score"])
        direction_abs = _clamp01(abs(direction))
        dominant_axis_angle = float(features["dominant_axis_angle_deg"])
        abs_angle = abs(dominant_axis_angle) % 180.0
        horizontal_distance = min(abs_angle, 180.0 - abs_angle)
        horizontalness = _clamp01(1.0 - (horizontal_distance / 90.0))
        verticalness = _clamp01(1.0 - (abs(abs_angle - 90.0) / 90.0))
        direction_deadzone = float(settings.get("direction_deadzone", 0.05))

        direction_strength = 0.0 if direction_abs <= direction_deadzone else _clamp01((direction_abs - direction_deadzone) / 0.10)
        left_signal = direction_strength if direction < 0 else 0.0
        right_signal = direction_strength if direction > 0 else 0.0

        front_bias = float(settings.get("front_symmetry_bias", 0.72))
        front_score = _clamp01(
            (front_bias * symmetry) + (0.10 * centered) + (0.10 * verticalness) - (0.35 * direction_abs) - (0.25 * horizontalness)
        )

        threeq_target = float(settings.get("three_quarter_symmetry_target", 0.58))
        threeq_tolerance = float(settings.get("three_quarter_symmetry_tolerance", 0.25))
        threeq_symmetry = _clamp01(1.0 - (abs(symmetry - threeq_target) / max(threeq_tolerance, 1e-6)))
        threeq_shape = _clamp01((0.6 * threeq_symmetry) + (0.4 * horizontalness))
        profile_shape = _clamp01((0.55 * horizontalness) + (0.45 * (1.0 - symmetry)))

        return {
            "left_profile": _clamp01(left_signal * profile_shape),
            "right_profile": _clamp01(right_signal * profile_shape),
            "front_left_3q": _clamp01(left_signal * threeq_shape),
            "front_right_3q": _clamp01(right_signal * threeq_shape),
            "front": front_score,
        }


def build_viewpoint_backend(config: dict[str, Any]) -> LocalScoreViewpointBackend:
    backend_name = config["classifier"]["backend"]
    if backend_name != "local_score_v1":
        raise ValueError(f"Unknown viewpoint backend: {backend_name}")
    return LocalScoreViewpointBackend(config["classifier"])
