from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw

from wildtrace import openrouter_image, studio_pipeline as sp
from wildtrace.diagram import (
    OpenCVValidationResult,
    OutlineRectifier,
    OutlineRectifierResult,
    SemanticValidationResult,
    SemanticValidator,
    build_line_drawing_prompt,
)
from wildtrace.openrouter_image import OpenRouterImageRectifier, extract_openrouter_image

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


# --------------------------------------------------------------------------- fixtures / fakes


def _animal_photo(size: tuple[int, int] = (320, 240)) -> tuple[Image.Image, Image.Image]:
    """Brown blob 'animal' on a green background, plus its mask."""
    image = Image.new("RGB", size, (60, 140, 60))
    mask = Image.new("L", size, 0)
    for canvas, fill in ((ImageDraw.Draw(image), (120, 80, 40)), (ImageDraw.Draw(mask), 255)):
        canvas.ellipse((70, 60, 250, 170), fill=fill)
        canvas.ellipse((200, 30, 270, 100), fill=fill)
        for x in (95, 125, 190, 220):
            canvas.rectangle((x, 150, x + 14, 215), fill=fill)
    return image, mask


def _line_art(size: tuple[int, int]) -> Image.Image:
    """Black outline drawing that passes the OpenCV prescreen."""
    width, height = size
    art = Image.new("L", size, 255)
    draw = ImageDraw.Draw(art)
    draw.ellipse((width * 0.1, height * 0.2, width * 0.8, height * 0.7), outline=0, width=4)
    draw.ellipse((width * 0.65, height * 0.08, width * 0.92, height * 0.38), outline=0, width=4)
    for fraction in (0.25, 0.6):
        x = width * fraction
        draw.line((x, height * 0.68, x, height * 0.97), fill=0, width=4)
    return art


class FakeRectifier(OutlineRectifier):
    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        super().__init__(settings or {"min_component_area": 80})
        self.calls: list[dict[str, Any]] = []

    def run(self, sample, subject_image, subject_mask, generated_path, destination, params) -> OutlineRectifierResult:
        self.calls.append({"sample_id": sample["sample_id"], "params": dict(params), "mask": subject_mask is not None})
        destination.parent.mkdir(parents=True, exist_ok=True)
        _line_art(subject_image.size).save(destination)
        return OutlineRectifierResult(diagram_path=destination, metadata={"backend": "fake", "prompt": build_line_drawing_prompt(sample)})


class ScriptedValidator(SemanticValidator):
    """Passes or fails each call from a script; repeats the last verdict."""

    def __init__(self, verdicts: list[bool]) -> None:
        super().__init__({"backend": "scripted", "model_name": "scripted-judge"})
        self.verdicts = list(verdicts)
        self.calls = 0

    def validate(self, sample, diagram_path, opencv_result) -> SemanticValidationResult:
        verdict = self.verdicts[min(self.calls, len(self.verdicts) - 1)]
        self.calls += 1
        return SemanticValidationResult(
            passed=verdict, score=0.9 if verdict else 0.2, reason="ok" if verdict else "not an animal", metadata={"model_name": "scripted-judge"}
        )


def _write_ndjson(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _silver_row(repo: Path, sample_id: str, category: str, angle: str, view_features: dict[str, float] | None = None) -> dict[str, Any]:
    image, mask = _animal_photo()
    image_rel = f"outputs/silver/images/{category}/{sample_id}_rgb.png"
    mask_rel = f"outputs/silver/masks/{category}/{sample_id}_mask.png"
    for rel, picture in ((image_rel, image), (mask_rel, mask)):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        picture.save(repo / rel)
    return {
        "sample_id": sample_id,
        "category": category,
        "subcategory": category.lower(),
        "angle_bucket": angle,
        "image_path": image_rel,
        "mask_path": mask_rel,
        "segmentation_status": "accepted",
        "view_features": view_features or {"symmetry_score": 0.1, "bottom_mass_ratio": 0.5, "direction_score": 0.8},
    }


@pytest.fixture()
def fake_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    rows = [
        _silver_row(repo, "used_cat", "Cat", "left_profile"),
        _silver_row(repo, "held_out_cat", "Cat", "left_profile"),
        _silver_row(repo, "free_left_cat", "Cat", "left_profile"),
        _silver_row(repo, "free_front_cat", "Cat", "front", {"symmetry_score": 0.6, "bottom_mass_ratio": 0.7}),
        _silver_row(repo, "free_dog", "Dog", "right_profile"),
    ]
    rejected = _silver_row(repo, "rejected_cat", "Cat", "front")
    rejected["segmentation_status"] = "rejected"
    missing = _silver_row(repo, "missing_mask_cat", "Cat", "front")
    (repo / missing["mask_path"]).unlink()
    _write_ndjson(repo / "outputs/silver/checkpoints/silver_subjects.ndjson", rows + [rejected, missing])
    bronze_only = _silver_row(repo, "bronze_only_cat", "Cat", "right_profile")
    bronze_only["curation_status"] = "accepted"
    _write_ndjson(repo / "outputs/bronze/manifests/bronze_accepted.ndjson", [bronze_only, {**rows[2], "curation_status": "accepted"}])
    for relative, sample_id in (("outputs/trajectories/Cat", "used_cat"), ("isaac-sim/data/act_test/Cat", "held_out_cat")):
        (repo / relative).mkdir(parents=True, exist_ok=True)
        (repo / relative / f"{sample_id}.json").write_text("{}", encoding="utf-8")
    return repo


# --------------------------------------------------------------------------- GPU + backend selection


def test_gpu_status_free_busy_and_reserved(monkeypatch, tmp_path: Path) -> None:
    status_file = tmp_path / "queue.status"
    monkeypatch.setattr(sp, "XRAYGPT_QUEUE_STATUS", status_file)
    monkeypatch.setattr(sp, "_query_nvidia_smi", lambda: [(3.0, 500.0, 16384.0)])
    free = sp.gpu_status()
    assert free["busy"] is False and free["utilization"] == 3.0 and "free" in free["note"]

    monkeypatch.setattr(sp, "_query_nvidia_smi", lambda: [(85.0, 500.0, 16384.0)])
    assert sp.gpu_status()["busy"] is True
    monkeypatch.setattr(sp, "_query_nvidia_smi", lambda: [(0.0, 9000.0, 16384.0)])
    assert "memory" in sp.gpu_status()["note"]

    monkeypatch.setattr(sp, "_query_nvidia_smi", lambda: [(0.0, 100.0, 16384.0)])
    status_file.write_text("START job_7\n", encoding="utf-8")
    reserved = sp.gpu_status()
    assert reserved["busy"] is True and "XRayGPT" in reserved["note"]
    status_file.write_text("START job_7\nQUEUE_DONE\n", encoding="utf-8")
    assert sp.gpu_status()["busy"] is False

    monkeypatch.setattr(sp, "_query_nvidia_smi", lambda: None)
    assert sp.gpu_status()["busy"] is True


def test_select_image_backend(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv(sp.IMAGE_BACKEND_ENV, raising=False)
    monkeypatch.setattr(sp, "_flux_dependencies_present", lambda: True)
    monkeypatch.setattr(sp, "gpu_status", lambda: {"busy": False, "utilization": 0.0, "note": ""})
    assert sp.select_image_backend(tmp_path, CONFIG_DIR) == "flux"
    monkeypatch.setattr(sp, "gpu_status", lambda: {"busy": True, "utilization": 90.0, "note": ""})
    assert sp.select_image_backend(tmp_path, CONFIG_DIR) == "openrouter"
    monkeypatch.setattr(sp, "gpu_status", lambda: {"busy": False, "utilization": 0.0, "note": ""})
    monkeypatch.setattr(sp, "_flux_dependencies_present", lambda: False)
    assert sp.select_image_backend(tmp_path, CONFIG_DIR) == "openrouter"
    monkeypatch.setenv(sp.IMAGE_BACKEND_ENV, "openrouter")
    monkeypatch.setattr(sp, "_flux_dependencies_present", lambda: True)
    assert sp.select_image_backend(tmp_path, CONFIG_DIR) == "openrouter"


def test_studio_validator_never_uses_ollama_without_gpu(monkeypatch) -> None:
    from wildtrace.config import load_runtime_config

    runtime = load_runtime_config(REPO_ROOT, CONFIG_DIR)
    runtime["models"]["outline_validator"]["backend"] = "ollama_semantic_validator"
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    validator = sp.build_studio_validator(runtime, "openrouter")
    assert type(validator).__name__ == "OpenRouterSemanticValidator"


# --------------------------------------------------------------------------- photo picking


def test_used_sample_ids(fake_repo: Path) -> None:
    assert sp.used_sample_ids(fake_repo) == {"used_cat", "held_out_cat"}


def test_pick_unused_photo_filters_used_rejected_and_missing(fake_repo: Path) -> None:
    import random

    picked = {sp.pick_unused_photo("cat", fake_repo, config_dir=CONFIG_DIR, rng=random.Random(seed))["sample_id"] for seed in range(20)}
    # Silver rows win over bronze-only ones; used, held-out, rejected and maskless rows never appear.
    assert picked == {"free_left_cat", "free_front_cat"}
    record = sp.pick_unused_photo("Dog", fake_repo, config_dir=CONFIG_DIR)
    assert record["sample_id"] == "free_dog"
    assert Path(record["resolved_image_path"]).is_file() and Path(record["resolved_mask_path"]).is_file()
    assert record["image_path"].startswith("outputs/silver/") and record["source_manifest"] == "silver_subjects"


def test_pick_unused_photo_hint_and_fallbacks(fake_repo: Path) -> None:
    assert sp.pick_unused_photo("Cat", fake_repo, hint="left profile", config_dir=CONFIG_DIR)["sample_id"] == "free_left_cat"
    assert sp.pick_unused_photo("Cat", fake_repo, hint="a sitting cat, front", config_dir=CONFIG_DIR)["sample_id"] == "free_front_cat"
    excluded = sp.pick_unused_photo("Cat", fake_repo, config_dir=CONFIG_DIR, exclude={"free_left_cat", "free_front_cat"})
    assert excluded["sample_id"] == "bronze_only_cat" and excluded["source_manifest"] == "bronze_accepted"
    with pytest.raises(LookupError):
        sp.pick_unused_photo("Frog", fake_repo, config_dir=CONFIG_DIR)


# --------------------------------------------------------------------------- segmentation


def test_consensus_mask_prefers_agreeing_plausible_mask() -> None:
    subject = np.zeros((100, 100), bool)
    subject[30:70, 20:80] = True
    shifted = np.zeros_like(subject)
    shifted[32:72, 22:82] = True
    background = np.ones_like(subject)  # whole frame: implausible
    empty = np.zeros_like(subject)
    stray = np.zeros_like(subject)
    stray[0:10, 0:10] = True
    assert sp.consensus_mask([background, empty, stray, subject, shifted, subject]) is subject
    assert sp.consensus_mask([background, empty]) is None


def test_segment_subject_falls_back_to_grabcut(monkeypatch) -> None:
    image, truth = _animal_photo()

    def broken(_rgb):
        raise RuntimeError("no mediapipe here")

    monkeypatch.setattr(sp, "_magic_touch_masks", broken)
    mask, label = sp.segment_subject(image)
    assert "grabcut" in label
    predicted = np.asarray(mask) > 0
    expected = np.asarray(truth) > 0
    assert (predicted & expected).sum() / (predicted | expected).sum() > 0.6


# --------------------------------------------------------------------------- stats


def test_estimated_draw_time_uses_sheet_scale_speed_and_pen_lifts() -> None:
    from wildtrace.face_lineart import estimate_draw_seconds

    trajectory = {"strokes": [{"points": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}]}, {"points": [{"x": 0.0, "y": 0.5}, {"x": 0.5, "y": 0.5}]}]}
    # 1.5 normalized units * 120 mm / 14 mm/s + 2 strokes * 6 s.
    assert estimate_draw_seconds(trajectory) == pytest.approx(1.5 * 120.0 / 14.0 + 12.0)


# --------------------------------------------------------------------------- full job


def _run_job(tmp_path: Path, fake_repo: Path, **kwargs: Any) -> tuple[dict[str, Any], list[sp.StageEvent]]:
    events: list[sp.StageEvent] = []
    result = sp.generate_animal_trajectory(
        out_dir=tmp_path / "job",
        repo_root=fake_repo,
        config_dir=CONFIG_DIR,
        category=kwargs.pop("category", "Cat"),
        emit=events.append,
        backend="openrouter",
        **kwargs,
    )
    return result, events


def _stage_trace(events: list[sp.StageEvent]) -> list[tuple[str, str]]:
    return [(event.name, event.status) for event in events]


def test_generate_from_picked_sample_writes_artifacts(tmp_path: Path, fake_repo: Path) -> None:
    sample = sp.pick_unused_photo("Cat", fake_repo, hint="left profile", config_dir=CONFIG_DIR)
    rectifier = FakeRectifier()
    validator = ScriptedValidator([True])
    result, events = _run_job(tmp_path, fake_repo, sample=sample, rectifier=rectifier, validator=validator)

    for key in ("source_path", "mask_path", "conditioning_path", "lineart_path", "svg_path", "trajectory_path"):
        assert Path(result[key]).is_file(), key
    assert rectifier.calls == [{"sample_id": "free_left_cat", "params": {}, "mask": True}]
    # Picked samples keep the Open Images mask rather than re-segmenting.
    assert np.array_equal(np.asarray(Image.open(result["mask_path"])) > 0, np.asarray(Image.open(sample["resolved_mask_path"])) > 0)
    conditioning = np.asarray(Image.open(result["conditioning_path"]))
    assert conditioning.shape == (512, 512) and (conditioning == 0).any()

    trajectory = json.loads(Path(result["trajectory_path"]).read_text())
    assert set(trajectory) == {
        "bounds", "canvas_height", "canvas_width", "coordinate_frame", "normalization_method",
        "point_count", "simplification_method", "stroke_count", "strokes",
    }
    assert trajectory["coordinate_frame"] == "normalized_canvas"
    assert set(trajectory["strokes"][0]) == {"stroke_id", "pen_state", "points"}
    assert 1 <= trajectory["stroke_count"] <= 6
    stats = result["stats"]
    assert stats["stroke_count"] == trajectory["stroke_count"] and stats["point_count"] == trajectory["point_count"]
    assert stats["est_draw_seconds"] > 6 * stats["stroke_count"]
    assert "<polyline" in Path(result["svg_path"]).read_text()
    assert result["validation"]["accepted"] is True and result["validation"]["attempts"] == 1

    # The caller reported pick_photo; every stage here is bracketed running -> done.
    assert _stage_trace(events) == [
        ("segment", "running"), ("segment", "done"),
        ("outline", "running"), ("outline", "done"),
        ("validate", "running"), ("validate", "done"),
        ("extract", "running"), ("extract", "done"),
    ]
    assert all(event.label for event in events)
    assert events[-1].artifact_path == Path(result["lineart_path"])
    assert json.loads((tmp_path / "job" / "result.json").read_text())["stats"] == stats


def test_text_job_picks_photo_itself(tmp_path: Path, fake_repo: Path) -> None:
    result, events = _run_job(tmp_path, fake_repo, category="Dog", rectifier=FakeRectifier(), validator=ScriptedValidator([True]))
    assert _stage_trace(events)[:2] == [("pick_photo", "running"), ("pick_photo", "done")]
    assert result["sample"]["sample_id"] == "free_dog"


def test_upload_is_segmented_on_cpu(tmp_path: Path, fake_repo: Path, monkeypatch) -> None:
    image, truth = _animal_photo()
    upload = tmp_path / "my cat.jpg"
    image.save(upload)
    monkeypatch.setattr(sp, "_magic_touch_masks", lambda rgb: [np.asarray(truth.resize((rgb.shape[1], rgb.shape[0]))) > 0])
    result, events = _run_job(tmp_path, fake_repo, image_path=upload, rectifier=FakeRectifier(), validator=ScriptedValidator([True]))
    segment_done = [event for event in events if event.name == "segment" and event.status == "done"][0]
    assert segment_done.backend == "mediapipe magic-touch (CPU)"
    assert result["sample"]["sample_id"] == "my_cat" and result["sample"]["angle_bucket"]
    assert (np.asarray(Image.open(result["mask_path"])) > 0).mean() > 0.2
    assert Path(result["trajectory_path"]).is_file()


def test_retry_loop_runs_until_validator_passes(tmp_path: Path, fake_repo: Path) -> None:
    sample = sp.pick_unused_photo("Dog", fake_repo, config_dir=CONFIG_DIR)
    rectifier = FakeRectifier()
    result, events = _run_job(tmp_path, fake_repo, sample=sample, rectifier=rectifier, validator=ScriptedValidator([False, True]), max_attempts=3)
    assert result["validation"]["accepted"] is True and result["validation"]["attempts"] == 2
    assert len(rectifier.calls) == 2
    # Retry params come from next_generator_params like the pipeline.
    assert rectifier.calls[1]["params"]["strength"] == pytest.approx(0.87)
    assert result["validation"]["diagram_path"].endswith("_attempt02.png")
    assert any(event.name == "validate" and event.detail and event.detail.startswith("retry 1") for event in events)
    assert ("validate", "done") in _stage_trace(events)


def test_rejected_attempts_still_extract_unless_required(tmp_path: Path, fake_repo: Path) -> None:
    sample = sp.pick_unused_photo("Dog", fake_repo, config_dir=CONFIG_DIR)
    result, events = _run_job(tmp_path, fake_repo, sample=sample, rectifier=FakeRectifier(), validator=ScriptedValidator([False]), max_attempts=2)
    assert result["validation"]["accepted"] is False and result["validation"]["attempts"] == 2
    assert ("validate", "failed") in _stage_trace(events) and ("extract", "done") in _stage_trace(events)

    with pytest.raises(RuntimeError):
        _run_job(tmp_path, fake_repo, sample=sample, rectifier=FakeRectifier(), validator=ScriptedValidator([False]), max_attempts=1, require_valid=True)


def test_failed_stage_is_reported(tmp_path: Path, fake_repo: Path) -> None:
    class Broken(FakeRectifier):
        def run(self, *args, **kwargs):
            raise RuntimeError("model unavailable")

    sample = sp.pick_unused_photo("Dog", fake_repo, config_dir=CONFIG_DIR)
    events: list[sp.StageEvent] = []
    with pytest.raises(RuntimeError):
        sp.generate_animal_trajectory(
            out_dir=tmp_path / "job", repo_root=fake_repo, config_dir=CONFIG_DIR, category="Dog",
            sample=sample, emit=events.append, backend="openrouter", rectifier=Broken(), validator=ScriptedValidator([True]),
        )
    assert (events[-1].name, events[-1].status) == ("outline", "failed")
    assert "model unavailable" in events[-1].detail


# --------------------------------------------------------------------------- OpenRouter rectifier


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def test_openrouter_rectifier_request_and_binarized_output(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("OPENROUTER_IMAGE_MODEL", raising=False)
    generated = _line_art((1024, 1024)).convert("RGB")
    captured: dict[str, Any] = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["auth"] = req.get_header("Authorization")
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse({"choices": [{"message": {"content": "", "images": [{"type": "image_url", "image_url": {"url": _data_url(generated)}}]}}], "usage": {"cost": 0.04}})

    monkeypatch.setattr(openrouter_image.request, "urlopen", fake_urlopen)
    image, mask = _animal_photo()
    sample = {"sample_id": "s1", "category": "Cat", "subcategory": "cat", "angle_bucket": "left_profile", "view_features": {}}
    rectifier = OpenRouterImageRectifier({"min_component_area": 80})
    destination = tmp_path / "out" / "s1_attempt01.png"
    result = rectifier.run(sample, image, mask, None, destination, {"strength": 0.85})

    body = captured["body"]
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["auth"] == "Bearer test-key"
    assert body["model"] == openrouter_image.DEFAULT_OPENROUTER_IMAGE_MODEL
    assert body["modalities"] == ["image", "text"]
    text_part, image_part = body["messages"][0]["content"]
    assert text_part["text"].endswith(build_line_drawing_prompt(sample))
    sent = Image.open(io.BytesIO(base64.b64decode(image_part["image_url"]["url"].split(",", 1)[1])))
    assert sent.size == (512, 512)  # the shared silhouette contour conditioning

    output = Image.open(result.diagram_path)
    assert output.size == image.size and set(np.unique(np.asarray(output))) <= {0, 255}
    assert result.metadata["model_name"] == openrouter_image.DEFAULT_OPENROUTER_IMAGE_MODEL
    assert result.metadata["cost_usd"] == 0.04
    assert "test-key" not in json.dumps(result.metadata)
    assert destination.with_name("s1_attempt01_raw.png").is_file()


def test_openrouter_rectifier_honours_model_override_and_prompt(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_IMAGE_MODEL", "google/gemini-2.5-flash-image")
    rectifier = OpenRouterImageRectifier({})
    assert rectifier.backend_label() == "openrouter: google/gemini-2.5-flash-image"
    assert rectifier.build_instruction({"category": "Cat"}, {"prompt": "custom"}) == "custom"
    assert OpenRouterImageRectifier({"openrouter_image_model": "x/y"})._model_name() == "x/y"


def test_openrouter_rectifier_requires_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OpenRouterImageRectifier({}).validate_ready()


def test_extract_openrouter_image_variants() -> None:
    art = _data_url(Image.new("RGB", (8, 8), "white"))
    assert extract_openrouter_image({"choices": [{"message": {"content": [{"type": "image_url", "image_url": {"url": art}}]}}]}).size == (8, 8)
    with pytest.raises(RuntimeError, match="no image"):
        extract_openrouter_image({"choices": [{"message": {"content": "I cannot draw that"}}]})
    with pytest.raises(RuntimeError):
        extract_openrouter_image({"error": {"message": "rate limited"}})


def test_downscale_line_art_keeps_thin_lines_connected() -> None:
    from wildtrace.diagram import _binarize_to_size, compute_opencv_metrics
    from wildtrace.openrouter_image import downscale_line_art

    big = Image.new("L", (1024, 1024), 255)
    ImageDraw.Draw(big).ellipse((100, 150, 900, 900), outline=0, width=2)
    nearest = compute_opencv_metrics(_binarize_to_size(big, (200, 300)))
    averaged = compute_opencv_metrics(downscale_line_art(big, (200, 300)))
    assert averaged["component_count"] == 1
    assert nearest["component_count"] > averaged["component_count"]


def test_judge_retries_a_rate_limited_call(monkeypatch):
    monkeypatch.setattr(sp, "RATE_LIMIT_BACKOFF_S", (0.0, 0.0))
    calls = []

    class FlakyJudge:
        settings: dict = {}

        def validate(self, sample, diagram_path, opencv_result):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError('OpenRouter semantic validation failed: {"error":{"code":429}}')
            return SimpleNamespace(score=0.9, passed=True, reason="ok")

    judge = sp._TrackingValidator(FlakyJudge())
    result = judge.validate({}, Path("x.png"), SimpleNamespace(score=1.0, passed=True, flags=[]))
    assert result.passed and len(calls) == 2
    assert judge.best["accepted"] is True


def test_judge_does_not_retry_real_errors(monkeypatch):
    monkeypatch.setattr(sp, "RATE_LIMIT_BACKOFF_S", (0.0, 0.0))

    class BrokenJudge:
        settings: dict = {}

        def validate(self, sample, diagram_path, opencv_result):
            raise RuntimeError("OpenRouter semantic validation failed: invalid api key")

    with pytest.raises(RuntimeError, match="invalid api key"):
        sp._TrackingValidator(BrokenJudge()).validate({}, Path("x.png"), SimpleNamespace(score=1.0, passed=True, flags=[]))
