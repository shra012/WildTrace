"""SDXL ControlNet integration checks. Inference is mocked."""

from __future__ import annotations

import sys
import types
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from wildtrace.diagram import (  # noqa: E402
    SDXLControlNetRectifier,
    _silhouette_contour_image,
    build_outline_rectifier,
)


def _blob_mask(size: tuple[int, int] = (80, 60), fill: int = 1) -> tuple[Image.Image, Image.Image]:
    subject = Image.new("RGB", size, "white")
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).ellipse((8, 6, size[0] - 8, size[1] - 6), fill=fill)
    return subject, mask


def _rectifier(**overrides: object) -> SDXLControlNetRectifier:
    settings: dict[str, object] = {
        "sdxl_model": SDXLControlNetRectifier.DEFAULT_MODEL,
        "sdxl_controlnet": SDXLControlNetRectifier.DEFAULT_CONTROLNET,
        "sdxl_vae": SDXLControlNetRectifier.DEFAULT_VAE,
        "sdxl_size": 64,
        "sdxl_contour_size": 64,
        "sdxl_num_inference_steps": 30,
        "sdxl_guidance_scale": 7.0,
        "sdxl_controlnet_conditioning_scale": 1.0,
        "sdxl_seed": 14,
        "min_component_area": 1,
    }
    settings.update(overrides)
    return SDXLControlNetRectifier(settings)


class _FakePipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        out = Image.new("RGB", (int(kwargs["width"]), int(kwargs["height"])), "white")
        ImageDraw.Draw(out).ellipse((8, 8, int(kwargs["width"]) - 8, int(kwargs["height"]) - 8), outline=(40, 40, 40), width=2)
        return type("FakeResult", (), {"images": [out]})()


def test_shared_contour_keeps_zero_one_masks() -> None:
    subject, mask = _blob_mask(fill=1)
    contour = _silhouette_contour_image(subject, mask, {}, {"min_component_area": 1}, size=128)
    arr = np.asarray(contour)
    assert arr.min() == 0
    assert arr.max() == 255

    _, mask_255 = _blob_mask(fill=255)
    contour_255 = _silhouette_contour_image(subject, mask_255, {}, {"min_component_area": 1}, size=128)
    assert np.asarray(contour_255).min() == 0


def test_sdxl_conditions_on_white_strokes_and_writes_binary_crop(monkeypatch, tmp_path: Path) -> None:
    subject, mask = _blob_mask()
    destination = tmp_path / "dog_attempt01.png"
    rectifier = _rectifier()
    pipe = _FakePipeline()
    monkeypatch.setattr(rectifier, "_load_pipeline", lambda: pipe)

    result = rectifier.run(
        {"sample_id": "dog-1", "category": "Dog", "subcategory": "dog", "angle_bucket": "side", "view_features": {}},
        subject,
        mask,
        None,
        destination,
        {},
    )

    control = pipe.calls[0]["image"]
    assert isinstance(control, Image.Image)
    assert control.size == (64, 64)
    control_arr = np.asarray(control)
    assert control_arr[0, 0].tolist() == [0, 0, 0]
    assert bool((control_arr == 255).all(axis=2).any())
    assert pipe.calls[0]["height"] == pipe.calls[0]["width"] == 64
    assert pipe.calls[0]["num_inference_steps"] == 30
    assert pipe.calls[0]["guidance_scale"] == 7.0
    assert pipe.calls[0]["controlnet_conditioning_scale"] == 1.0
    arr = np.asarray(Image.open(result.diagram_path).convert("L"))
    assert arr.shape == (60, 80)
    assert set(np.unique(arr)) == {0, 255}
    assert result.metadata["model"] == SDXLControlNetRectifier.DEFAULT_MODEL
    assert result.metadata["controlnet"] == SDXLControlNetRectifier.DEFAULT_CONTROLNET
    assert result.metadata["vae"] == SDXLControlNetRectifier.DEFAULT_VAE
    assert result.metadata["backend"] == "sdxl_controlnet_rectifier"


def test_sdxl_maps_retry_strength_onto_controlnet_scale(monkeypatch, tmp_path: Path) -> None:
    subject, mask = _blob_mask()
    rectifier = _rectifier()
    pipe = _FakePipeline()
    monkeypatch.setattr(rectifier, "_load_pipeline", lambda: pipe)

    rectifier.run({"sample_id": "dog-1", "category": "Dog"}, subject, mask, None, tmp_path / "a.png", {"strength": 0.90})
    rectifier.run({"sample_id": "dog-1", "category": "Dog"}, subject, mask, None, tmp_path / "b.png", {"strength": 0.80})

    assert pipe.calls[0]["controlnet_conditioning_scale"] == 0.8
    assert pipe.calls[1]["controlnet_conditioning_scale"] == 1.2


def test_sdxl_seeds_depend_on_sample_and_attempt_filename(monkeypatch, tmp_path: Path) -> None:
    subject, mask = _blob_mask()
    rectifier = _rectifier(sdxl_seed=14)
    pipe = _FakePipeline()
    monkeypatch.setattr(rectifier, "_load_pipeline", lambda: pipe)
    sample = {"sample_id": "dog-1", "category": "Dog"}
    first = tmp_path / "dog-1_attempt01.png"
    second = tmp_path / "dog-1_attempt02.png"

    again = rectifier.run(sample, subject, mask, None, first, {})
    repeat = rectifier.run(sample, subject, mask, None, first, {})
    retry = rectifier.run(sample, subject, mask, None, second, {})

    expected = 14 + zlib.crc32(f"dog-1:{first.name}".encode())
    assert again.metadata["seed"] == repeat.metadata["seed"] == expected
    assert retry.metadata["seed"] != again.metadata["seed"]
    assert pipe.calls[0]["generator"] is not None


def test_sdxl_checkpoint_selection_and_device_placement(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeControlNet:
        @staticmethod
        def from_pretrained(repo: str, **kwargs: object) -> str:
            captured["controlnet"] = (repo, kwargs.get("torch_dtype"))
            return "controlnet"

    class FakeVAE:
        @staticmethod
        def from_pretrained(repo: str, **kwargs: object) -> str:
            captured["vae"] = repo
            return "vae"

    class FakePipe:
        @staticmethod
        def from_pretrained(repo: str, **kwargs: object) -> "FakePipe":
            captured["model"] = repo
            captured["pipe_kwargs"] = kwargs
            return FakePipe()

        def enable_model_cpu_offload(self) -> None:
            captured["placed"] = "cuda-offload"

        def to(self, device: str) -> "FakePipe":
            captured["placed"] = device
            return self

    def install(cuda: bool, mps: bool) -> None:
        diffusers = types.ModuleType("diffusers")
        diffusers.ControlNetModel = FakeControlNet
        diffusers.AutoencoderKL = FakeVAE
        diffusers.StableDiffusionXLControlNetPipeline = FakePipe
        torch = types.ModuleType("torch")
        torch.float16 = "fp16"
        torch.float32 = "fp32"
        torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)
        torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
        monkeypatch.setitem(sys.modules, "diffusers", diffusers)
        monkeypatch.setitem(sys.modules, "torch", torch)

    install(cuda=True, mps=False)
    _rectifier()._load_pipeline()
    assert captured["model"] == SDXLControlNetRectifier.DEFAULT_MODEL
    assert captured["controlnet"] == (SDXLControlNetRectifier.DEFAULT_CONTROLNET, "fp16")
    assert captured["vae"] == SDXLControlNetRectifier.DEFAULT_VAE
    assert captured["pipe_kwargs"]["controlnet"] == "controlnet"
    assert captured["pipe_kwargs"]["vae"] == "vae"
    assert captured["placed"] == "cuda-offload"

    captured.clear()
    install(cuda=False, mps=True)
    SDXLControlNetRectifier({})._load_pipeline()
    assert captured["placed"] == "mps"
    assert captured["controlnet"][1] == "fp16"

    captured.clear()
    install(cuda=False, mps=False)
    SDXLControlNetRectifier({})._load_pipeline()
    assert captured["placed"] == "cpu"
    assert captured["controlnet"][1] == "fp32"


def test_build_outline_rectifier_registers_sdxl(monkeypatch) -> None:
    monkeypatch.setattr(SDXLControlNetRectifier, "validate_ready", lambda self: None)
    rectifier = build_outline_rectifier({"backend": "sdxl_controlnet_rectifier"})
    assert isinstance(rectifier, SDXLControlNetRectifier)


def test_benchmark_config_selection_changes_provenance(monkeypatch) -> None:
    import benchmark_models

    monkeypatch.delenv("WILDTRACE_BENCHMARK_CONFIG", raising=False)
    default_hash = benchmark_models.provenance()["config_hash"]
    default_bench = benchmark_models.load_bench()
    assert default_bench["comparison"]["run_id"] == "r2"

    monkeypatch.setenv("WILDTRACE_BENCHMARK_CONFIG", "configs/benchmark_sdxl.yaml")
    selected = benchmark_models.load_bench()
    comparison = selected["comparison"]
    assert comparison["run_id"] == "sdxl-r1"
    assert comparison["max_attempts"] == 3
    assert comparison["evalset"]["path"] == "benchmarks/evalset_v2.json"
    judges = {setup["validator"]["openrouter_model_name"] for setup in comparison["setups"]}
    assert judges == {"qwen/qwen3-vl-8b-instruct"}
    assert {setup["rectifier"] for setup in comparison["setups"]} == {"flux-schnell", "omnigen2", "sdxl-controlnet"}
    strong = [row for row in comparison["referees"] if row.get("strong")]
    assert len(strong) == 1
    assert "235b" in strong[0]["openrouter_model_name"]
    assert benchmark_models.provenance()["config_hash"] != default_hash

    from wildtrace.config import load_runtime_config

    sdxl_settings = benchmark_models.rectifier_settings(
        {"name": "sdxl-controlnet", "backend": "sdxl_controlnet_rectifier"},
        load_runtime_config(ROOT)["models"],
    )
    assert sdxl_settings["sdxl_model"] == SDXLControlNetRectifier.DEFAULT_MODEL
    assert sdxl_settings["backend"] == "sdxl_controlnet_rectifier"
    assert "gguf_repo" not in sdxl_settings


def test_pipeline_worker_forwards_resume_flag() -> None:
    import benchmark_setups

    resumed = benchmark_setups.pipeline_worker_command("sdxl-qwen3", limit=2, resume=True)
    assert resumed[:3] == [sys.executable, benchmark_setups.__file__, "_pipeline-one"]
    assert resumed[resumed.index("--setup") + 1] == "sdxl-qwen3"
    assert resumed[resumed.index("--limit") + 1] == "2"
    assert resumed[-1] == "--resume"

    fresh = benchmark_setups.pipeline_worker_command("flux-qwen3", limit=None, resume=False)
    assert "--resume" not in fresh
    assert "--limit" not in fresh


def test_sdxl_validate_ready_reports_missing_diffusers(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def blocked(name: str, *args: object, **kwargs: object) -> object:
        if name == "diffusers" or name.startswith("diffusers."):
            raise ImportError("diffusers missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RuntimeError, match="torch` and `diffusers"):
        SDXLControlNetRectifier({}).validate_ready()
