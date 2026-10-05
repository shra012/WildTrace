from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from wildtrace import eval_metrics as em
from wildtrace.diagram import OpenCVValidationResult, OutlineRectifier, OutlineRectifierResult, SemanticValidationResult, SemanticValidator

ROOT = Path(__file__).resolve().parents[1]


def _disk(size: int = 200, center: tuple[int, int] = (100, 100), radius: int = 60) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, center, radius, 1, thickness=-1)
    return mask


def _ring(size: int = 200, center: tuple[int, int] = (100, 100), radius: int = 60) -> np.ndarray:
    ink = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(ink, center, radius, 1, thickness=3)
    return ink


def test_identical_silhouette_is_perfect_fidelity() -> None:
    fill = em.filled_silhouette(_ring())
    result = em.fidelity_metrics(fill, _disk(radius=61))
    assert result["silhouette_iou"] > 0.94  # ring's filled outer edge sits ~1px outside the disk
    assert result["chamfer_pct"] < 0.5
    assert result["boundary_f"] > 0.99
    assert result["centroid_offset_pct"] < 0.5


def test_shifted_shape_lowers_iou_and_raises_chamfer() -> None:
    mask = _disk()
    shifted = _disk(center=(130, 100))
    result = em.fidelity_metrics(shifted, mask)
    assert result["silhouette_iou"] < em.fidelity_metrics(mask, mask)["silhouette_iou"]
    assert result["chamfer_pct"] > 5.0
    assert result["centroid_offset_pct"] == pytest.approx(100 * 30 / np.hypot(200, 200), abs=0.2)


def test_fill_ratio_separates_solid_silhouette_from_outline() -> None:
    assert em.structure_metrics(_disk())["fill_ratio"] > 0.9
    assert em.structure_metrics(_ring())["fill_ratio"] < 0.05
    ring = _ring()
    true_width = ring.sum() / (2 * np.pi * 60)  # cv2 thickness=3 draws ~5px
    assert em.structure_metrics(ring)["stroke_width_px"] == pytest.approx(true_width, abs=0.75)
    assert em.structure_metrics(np.zeros((50, 50), dtype=np.uint8))["blank"] is True


def test_path_length_matches_simulator_paper_fit() -> None:
    # Unit square outline, normalized: uniform fit to 160x120 mm -> 120 mm side.
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
    pen_down, pen_up = em.path_lengths_mm([square])
    assert pen_down == pytest.approx(480.0)
    assert pen_up == 0.0
    two = [[(0.0, 0.0), (0.5, 0.0)], [(0.5, 1.0), (1.0, 1.0)]]
    assert em.path_lengths_mm(two)[1] == pytest.approx(120.0)


def test_coverage_is_full_when_ink_is_the_trajectory() -> None:
    ink = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(ink, (10, 10), (89, 89), 1, thickness=1)
    stroke = [(10 / 99, 10 / 99), (89 / 99, 10 / 99), (89 / 99, 89 / 99), (10 / 99, 89 / 99)]
    assert em.trajectory_coverage(ink, [stroke]) == 1.0
    cv2.circle(ink, (50, 50), 5, 1, thickness=-1)  # interior detail the robot won't draw
    assert em.trajectory_coverage(ink, [stroke]) < 1.0


def test_draw_time_fit_recovers_linear_model() -> None:
    samples = [(p, s, 0.35 * p + 26.0 * s + 5.0) for p, s in ((300, 2), (500, 4), (800, 5), (650, 3), (1000, 6))]
    model = em.fit_draw_time(samples)
    assert model.a_s_per_mm == pytest.approx(0.35)
    assert model.b_s_per_stroke == pytest.approx(26.0)
    assert model.r2 == pytest.approx(1.0)


def test_pose_metrics_detect_flipped_facing() -> None:
    body = np.zeros((120, 200), dtype=np.uint8)
    cv2.ellipse(body, (90, 70), (70, 30), 0, 0, 360, 1, -1)
    cv2.circle(body, (170, 45), 22, 1, -1)  # head on the right
    from wildtrace.viewpoint import extract_view_features

    source = {"view_features": extract_view_features(Image.fromarray(body * 255)), "angle_bucket": "right_profile"}
    from wildtrace.config import load_yaml

    config = load_yaml(ROOT / "configs" / "viewpoints.yaml")
    same = em.pose_metrics(body, source, config)
    flipped = em.pose_metrics(body[:, ::-1].copy(), source, config)
    assert same["direction_match"] is True
    assert same["direction_delta"] < 0.05
    assert flipped["direction_match"] is False


def test_statistics_helpers() -> None:
    assert em.sign_test_p(10, 0) == pytest.approx(2 / 1024, abs=1e-4)
    assert em.sign_test_p(3, 3) == 1.0
    assert em.cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert em.cohen_kappa([True, True, False, False], [True, False, True, False]) == 0.0
    low, high = em.bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert low < 3.0 < high


def test_metric_registry_is_complete() -> None:
    keys = [m.key for m in em.METRICS]
    assert len(keys) == len(set(keys))
    assert {m.group for m in em.METRICS} == set(em.GROUPS)
    assert all(m.direction in {"higher", "lower", "target", "info"} for m in em.METRICS)


def _load_setups_script():
    spec = importlib.util.spec_from_file_location("benchmark_setups", ROOT / "scripts" / "benchmark_setups.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "scripts"))
    spec.loader.exec_module(module)
    return module


def test_recording_proxies_log_every_call(tmp_path: Path, monkeypatch) -> None:
    bs = _load_setups_script()
    monkeypatch.setattr(bs, "REPO_ROOT", tmp_path)
    logged: list[dict] = []

    class StubRectifier(OutlineRectifier):
        def run(self, sample, subject_image, subject_mask, generated_path, destination, params):
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (8, 8), 255).save(destination)
            return OutlineRectifierResult(destination, {"backend": "stub", "prompt": "p"})

    class StubValidator(SemanticValidator):
        def validate(self, sample, diagram_path, opencv_result):
            return SemanticValidationResult(True, 0.9, "ok", {"latency_s": 0.1, "cost_usd": 0.0001})

        def suggest_params(self, *args):
            return {"strength": 0.8}

    rectifier = bs.RecordingRectifier(StubRectifier({}), logged.append)
    validator = bs.RecordingValidator(StubValidator({}), logged.append)
    sample = {"sample_id": "s1", "category": "Cat"}
    destination = tmp_path / "Cat" / "s1_attempt02.png"
    rectifier.run(sample, Image.new("RGB", (8, 8)), None, None, destination, {"strength": 0.9, "other": 1})
    validator.validate(sample, destination, OpenCVValidationResult(True, 0.8, [], {}))
    validator.suggest_params(sample, destination, destination, OpenCVValidationResult(True, 0.8, [], {}), {})

    assert [row["kind"] for row in logged] == ["generate", "judge", "feedback"]
    assert logged[0]["attempt"] == 2
    assert logged[0]["params"] == {"strength": 0.9}
    assert logged[0]["prompt_used"] == "p"
    assert logged[0]["output_path"] == "Cat/s1_attempt02.png"
    assert logged[1]["passed"] is True and logged[1]["cost_usd"] == 0.0001
    assert logged[2]["suggestion"] == {"strength": 0.8}
