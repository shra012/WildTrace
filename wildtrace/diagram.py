from __future__ import annotations

import base64
from dataclasses import dataclass
import inspect
import json
import os
from pathlib import Path
from typing import Any, TypedDict
from urllib import error, request

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from huggingface_hub import hf_hub_download

from wildtrace.images import connected_components

try:
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover - exercised only when dependency is unavailable.
    END = "__end__"
    StateGraph = None

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only when dependency is unavailable.
    cv2 = None

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - exercised only when dependency is unavailable.
    ort = None


@dataclass(slots=True)
class DiagramBackendResult:
    diagram_path: Path
    metadata: dict[str, Any]


@dataclass(slots=True)
class OutlineRectifierResult:
    diagram_path: Path
    metadata: dict[str, Any]


@dataclass(slots=True)
class OutlineAttemptResult:
    diagram_path: Path
    generator_metadata: dict[str, Any]
    rectifier_metadata: dict[str, Any]


@dataclass(slots=True)
class OpenCVValidationResult:
    passed: bool
    score: float
    flags: list[str]
    metrics: dict[str, float]


@dataclass(slots=True)
class SemanticValidationResult:
    passed: bool
    score: float
    reason: str
    metadata: dict[str, Any]


class DiagramBackend:
    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = settings

    def run(self, image: Image.Image, destination: Path, params: dict[str, Any]) -> DiagramBackendResult:
        raise NotImplementedError


class OutlineRectifier:
    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = settings

    def validate_ready(self) -> None:
        return None

    def run(
        self,
        sample: dict[str, Any],
        subject_image: Image.Image,
        subject_mask: Image.Image | None,
        generated_path: Path,
        destination: Path,
        params: dict[str, Any],
    ) -> OutlineRectifierResult:
        raise NotImplementedError


def _filter_small_components(binary: np.ndarray, min_component_area: int) -> np.ndarray:
    if min_component_area <= 1:
        return binary
    if cv2 is not None:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        cleaned = np.zeros_like(binary)
        for index in range(1, count):
            if int(stats[index, cv2.CC_STAT_AREA]) >= min_component_area:
                cleaned[labels == index] = 1
        return cleaned
    cleaned = np.zeros_like(binary)
    for component in connected_components(binary.astype(np.uint8)):
        if len(component) >= min_component_area:
            for x, y in component:
                cleaned[y, x] = 1
    return cleaned


def _extract_simple_outline(
    binary: np.ndarray,
    outline_close_kernel: int,
    min_outline_area: float,
    max_outlines: int,
    simplify_ratio: float,
    stroke_width: int,
    smooth_sigma: float = 0.0,
) -> np.ndarray:
    if cv2 is None:
        return binary
    kernel_size = max(1, int(outline_close_kernel))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    if smooth_sigma > 0:
        blurred = cv2.GaussianBlur((closed * 255).astype(np.uint8), (0, 0), smooth_sigma)
        closed = (blurred > 127).astype(np.uint8)
    contours, _ = cv2.findContours((closed * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return binary

    selected = [contour for contour in sorted(contours, key=cv2.contourArea, reverse=True) if cv2.contourArea(contour) >= min_outline_area]
    if not selected:
        selected = [max(contours, key=cv2.contourArea)]

    outline = np.zeros_like(binary)
    for contour in selected[: max(1, int(max_outlines))]:
        perimeter = max(cv2.arcLength(contour, True), 1.0)
        simplified = cv2.approxPolyDP(contour, simplify_ratio * perimeter, True)
        cv2.drawContours(outline, [simplified], -1, 1, max(1, int(stroke_width)))
    return outline


def _subject_mask_from_image(
    image: Image.Image,
    subject_threshold: int,
    silhouette_close_kernel: int,
    min_component_area: int,
) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    binary = np.any(arr < int(subject_threshold), axis=2).astype(np.uint8)
    if cv2 is None:
        return _filter_small_components(binary, min_component_area)
    kernel_size = max(1, int(silhouette_close_kernel))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    return _filter_small_components(opened, min_component_area)


def _binary_mask_from_rendered_image(
    image: Image.Image,
    threshold: int,
    min_component_area: int,
    close_kernel: int,
) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    binary = np.any(arr < int(threshold), axis=2).astype(np.uint8)
    if cv2 is not None:
        kernel_size = max(1, int(close_kernel))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return _filter_small_components(binary, min_component_area)


def _mask_array_from_image(mask_image: Image.Image, size: tuple[int, int], min_component_area: int) -> np.ndarray:
    grayscale = mask_image.convert("L")
    if grayscale.size != size:
        grayscale = grayscale.resize(size, Image.Resampling.NEAREST)
    binary = (np.asarray(grayscale, dtype=np.uint8) > 0).astype(np.uint8)
    return _filter_small_components(binary, min_component_area)


def _cropped_mask_image(mask_image: Image.Image, crop_bbox: dict[str, Any] | None, size: tuple[int, int]) -> Image.Image:
    grayscale = mask_image.convert("L")
    if crop_bbox:
        grayscale = grayscale.crop(
            (
                int(crop_bbox["left"]),
                int(crop_bbox["top"]),
                int(crop_bbox["right"]),
                int(crop_bbox["bottom"]),
            )
        )
    if grayscale.size != size:
        grayscale = grayscale.resize(size, Image.Resampling.NEAREST)
    return grayscale


def _dilate_binary_mask(binary: np.ndarray, kernel_size: int) -> np.ndarray:
    if cv2 is None:
        return binary
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.dilate(binary.astype(np.uint8), kernel, iterations=1)


def _touches_all_borders(binary: np.ndarray) -> bool:
    if binary.size == 0:
        return False
    return bool(binary[0, :].any() and binary[-1, :].any() and binary[:, 0].any() and binary[:, -1].any())


class InformativeDrawingsBackend(DiagramBackend):
    def __init__(self, settings: dict[str, Any]) -> None:
        super().__init__(settings)
        self._session: Any | None = None

    def _model_backend(self) -> str:
        return str(self.settings.get("model_backend", "onnxruntime"))

    def _onnx_path(self) -> str:
        repo_id = self.settings.get("repo_id", "rocca/informative-drawings-line-art-onnx")
        filename = self.settings.get("filename", "model.onnx")
        cache_dir = os.path.expanduser(str(self.settings.get("cache_dir", "~/.cache/huggingface")))
        return hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)

    def _providers(self) -> list[str]:
        if ort is None:
            raise RuntimeError("Informative Drawings ONNX backend requires `onnxruntime`.")
        available = set(ort.get_available_providers())
        preferred = [
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]
        providers = [provider for provider in preferred if provider in available]
        if not providers:
            raise RuntimeError("No compatible ONNX Runtime execution provider is available.")
        return providers

    def _get_session(self) -> Any:
        if self._session is None:
            self._session = ort.InferenceSession(self._onnx_path(), providers=self._providers())
        return self._session

    def _prepare_input(self, image: Image.Image) -> tuple[np.ndarray, tuple[int, int]]:
        input_size = int(self.settings.get("input_size", 512))
        resized = image.convert("RGB").resize((input_size, input_size), Image.Resampling.LANCZOS)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))[None, ...]
        return arr, image.size

    def _subject_mask_from_image(self, image: Image.Image, params: dict[str, Any]) -> np.ndarray:
        return _subject_mask_from_image(
            image,
            int(params.get("subject_threshold", self.settings.get("subject_threshold", 248))),
            int(params.get("silhouette_close_kernel", self.settings.get("silhouette_close_kernel", 9))),
            int(self.settings.get("min_component_area", 80)),
        )

    def _render_outline_from_subject(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        mask = self._subject_mask_from_image(image, params)
        outline = _extract_simple_outline(
            mask,
            int(params.get("outline_close_kernel", self.settings.get("outline_close_kernel", 5))),
            float(params.get("min_outline_area", self.settings.get("min_outline_area", 300.0))),
            int(params.get("max_outlines", self.settings.get("max_outlines", 2))),
            float(params.get("outline_simplify_ratio", self.settings.get("outline_simplify_ratio", 0.003))),
            int(params.get("outline_stroke_width", self.settings.get("outline_stroke_width", 4))),
            float(params.get("outline_smooth_sigma", self.settings.get("outline_smooth_sigma", 0.0))),
        )
        return Image.fromarray(np.where(outline > 0, 0, 255).astype(np.uint8), mode="L")

    def _postprocess(self, output: np.ndarray, original_size: tuple[int, int], params: dict[str, Any]) -> Image.Image:
        line_strength = 1.0 - np.clip(output, 0.0, 1.0)
        grayscale = Image.fromarray(np.clip(line_strength * 255.0, 0, 255).astype(np.uint8), mode="L")
        blur_radius = float(params.get("blur_radius", self.settings.get("postprocess_blur_radius", 2.0)))
        if blur_radius > 0:
            grayscale = grayscale.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        threshold = int(params.get("threshold", self.settings.get("line_threshold", 26)))
        binary = (np.asarray(grayscale, dtype=np.uint8) >= threshold).astype(np.uint8)
        cleaned = _filter_small_components(binary, int(self.settings.get("min_component_area", 80)))
        outline = _extract_simple_outline(
            cleaned,
            int(params.get("outline_close_kernel", self.settings.get("outline_close_kernel", 5))),
            float(params.get("min_outline_area", self.settings.get("min_outline_area", 300.0))),
            int(params.get("max_outlines", self.settings.get("max_outlines", 2))),
            float(params.get("outline_simplify_ratio", self.settings.get("outline_simplify_ratio", 0.003))),
            int(params.get("outline_stroke_width", self.settings.get("outline_stroke_width", 4))),
            float(params.get("outline_smooth_sigma", self.settings.get("outline_smooth_sigma", 0.0))),
        )
        image = Image.fromarray(np.where(outline > 0, 0, 255).astype(np.uint8), mode="L")
        resized = image.resize(original_size, Image.Resampling.NEAREST)
        return resized.point(lambda value: 0 if value < 255 else 255)

    def _run_mock(self, image: Image.Image, destination: Path, params: dict[str, Any]) -> DiagramBackendResult:
        gray = ImageOps.grayscale(image)
        blur_radius = float(params.get("blur_radius", self.settings.get("blur_radius", 0.8)))
        threshold = int(params.get("threshold", self.settings.get("threshold", 150)))
        outline = gray.filter(ImageFilter.GaussianBlur(radius=blur_radius)).filter(ImageFilter.FIND_EDGES)
        arr = np.asarray(outline, dtype=np.uint8)
        binary = np.where(arr >= threshold, 0, 255).astype(np.uint8)
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(binary, mode="L").save(destination)
        return DiagramBackendResult(
            diagram_path=destination,
            metadata={
                "model_backend": "mock",
                "model_name": self.settings.get("model_name", "informative_drawings"),
                "model_version_or_checkpoint": self.settings.get("model_version_or_checkpoint", "mock-v1"),
                "execution_mode": "mock",
                "inference_params": {"blur_radius": blur_radius, "threshold": threshold},
            },
        )

    def run(self, image: Image.Image, destination: Path, params: dict[str, Any]) -> DiagramBackendResult:
        if self._model_backend() == "mock":
            return self._run_mock(image, destination, params)
        render_mode = str(params.get("render_mode", self.settings.get("render_mode", "silhouette_outline")))
        original_size = image.size
        if render_mode == "silhouette_outline":
            diagram = self._render_outline_from_subject(image, params)
        else:
            session = self._get_session()
            model_input, original_size = self._prepare_input(image)
            output_name = session.get_outputs()[0].name
            input_name = session.get_inputs()[0].name
            output = session.run([output_name], {input_name: model_input})[0][0, 0]
            diagram = self._postprocess(output, original_size, params)
        destination.parent.mkdir(parents=True, exist_ok=True)
        diagram.resize(original_size, Image.Resampling.NEAREST).point(lambda value: 0 if value > 0 else 255).save(destination)
        return DiagramBackendResult(
            diagram_path=destination,
            metadata={
                "model_backend": self._model_backend(),
                "model_name": self.settings.get("model_name", "informative_drawings"),
                "model_version_or_checkpoint": self.settings.get(
                    "model_version_or_checkpoint",
                    self.settings.get("repo_id", "rocca/informative-drawings-line-art-onnx"),
                ),
                "execution_mode": "onnxruntime",
                "inference_params": {
                    "render_mode": render_mode,
                    "input_size": int(self.settings.get("input_size", 512)),
                    "line_threshold": int(params.get("threshold", self.settings.get("line_threshold", 26))),
                    "postprocess_blur_radius": float(params.get("blur_radius", self.settings.get("postprocess_blur_radius", 2.0))),
                    "min_component_area": int(self.settings.get("min_component_area", 80)),
                    "subject_threshold": int(params.get("subject_threshold", self.settings.get("subject_threshold", 248))),
                    "silhouette_close_kernel": int(params.get("silhouette_close_kernel", self.settings.get("silhouette_close_kernel", 9))),
                    "outline_simplify_ratio": float(params.get("outline_simplify_ratio", self.settings.get("outline_simplify_ratio", 0.012))),
                    "outline_stroke_width": int(params.get("outline_stroke_width", self.settings.get("outline_stroke_width", 3))),
                    "max_outlines": int(params.get("max_outlines", self.settings.get("max_outlines", 2))),
                },
            },
        )


class SilhouetteOutlineRectifier(OutlineRectifier):
    def run(
        self,
        sample: dict[str, Any],
        subject_image: Image.Image,
        subject_mask: Image.Image | None,
        generated_path: Path,
        destination: Path,
        params: dict[str, Any],
    ) -> OutlineRectifierResult:
        min_component_area = int(self.settings.get("min_component_area", 80))
        subject_mask_array = (
            _mask_array_from_image(subject_mask, subject_image.size, min_component_area)
            if subject_mask is not None
            else _subject_mask_from_image(
                subject_image,
                int(params.get("subject_threshold", self.settings.get("subject_threshold", 248))),
                int(params.get("silhouette_close_kernel", self.settings.get("silhouette_close_kernel", 9))),
                min_component_area,
            )
        )
        outline = _extract_simple_outline(
            subject_mask_array,
            int(params.get("outline_close_kernel", self.settings.get("outline_close_kernel", 5))),
            float(params.get("min_outline_area", self.settings.get("min_outline_area", 300.0))),
            int(params.get("max_outlines", self.settings.get("max_outlines", 2))),
            float(params.get("outline_simplify_ratio", self.settings.get("outline_simplify_ratio", 0.003))),
            int(params.get("outline_stroke_width", self.settings.get("outline_stroke_width", 4))),
            float(params.get("outline_smooth_sigma", self.settings.get("outline_smooth_sigma", 0.0))),
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.where(outline > 0, 0, 255).astype(np.uint8), mode="L").save(destination)
        return OutlineRectifierResult(
            diagram_path=destination,
            metadata={
                "model_backend": self.settings.get("model_backend", "local"),
                "model_name": self.settings.get("model_name", "silhouette_outline_rectifier"),
                "model_version_or_checkpoint": self.settings.get("model_version_or_checkpoint", "local-v1"),
                "execution_mode": "subject_silhouette_rectification",
                "used_generator_input": generated_path.name,
                "inference_params": {
                    "mask_source": "silver_mask" if subject_mask is not None else "rgb_threshold",
                    "subject_threshold": int(params.get("subject_threshold", self.settings.get("subject_threshold", 248))),
                    "silhouette_close_kernel": int(params.get("silhouette_close_kernel", self.settings.get("silhouette_close_kernel", 9))),
                    "outline_close_kernel": int(params.get("outline_close_kernel", self.settings.get("outline_close_kernel", 5))),
                    "outline_simplify_ratio": float(params.get("outline_simplify_ratio", self.settings.get("outline_simplify_ratio", 0.012))),
                    "outline_stroke_width": int(params.get("outline_stroke_width", self.settings.get("outline_stroke_width", 3))),
                    "max_outlines": int(params.get("max_outlines", self.settings.get("max_outlines", 2))),
                },
            },
        )


class ControlNetLineArtRectifier(OutlineRectifier):
    def __init__(self, settings: dict[str, Any]) -> None:
        super().__init__(settings)
        self._pipeline: Any | None = None

    def _import_torch(self) -> Any:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("ControlNet lineart rectifier requires `torch` to be installed.") from exc
        return torch

    def _import_diffusers(self) -> tuple[type, type]:
        try:
            from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "ControlNet lineart rectifier requires `diffusers`, `transformers`, and `accelerate`. "
                "Install the project dependencies with `uv sync`."
            ) from exc
        return ControlNetModel, StableDiffusionControlNetPipeline

    def _device(self) -> str:
        return str(self.settings.get("device", "cuda"))

    def _torch_dtype_name(self) -> str:
        return str(self.settings.get("torch_dtype", "float16"))

    def _controlnet_repo_id(self) -> str:
        return str(self.settings.get("controlnet_model_name", "lllyasviel/control_v11p_sd15_lineart"))

    def _base_model_repo_id(self) -> str:
        return str(self.settings.get("base_model_name", "runwayml/stable-diffusion-v1-5"))

    def _cache_dir(self) -> str:
        return os.path.expanduser(str(self.settings.get("cache_dir", "~/.cache/huggingface")))

    def _hf_token(self) -> str | None:
        token = os.getenv("HF_TOKEN")
        return token.strip() if token else None

    def _torch_dtype(self, torch: Any) -> Any:
        dtype_name = self._torch_dtype_name()
        if not hasattr(torch, dtype_name):
            raise RuntimeError(f"Unsupported torch dtype for ControlNet lineart rectifier: {dtype_name}")
        return getattr(torch, dtype_name)

    def _ensure_cuda(self) -> None:
        torch = self._import_torch()
        if self._device().startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "ControlNet lineart rectifier requires a CUDA GPU, but CUDA is not available. "
                "Use a CUDA-capable machine or switch the rectifier backend."
            )

    def validate_ready(self) -> None:
        self._ensure_cuda()
        self._import_diffusers()

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        self.validate_ready()
        torch = self._import_torch()
        ControlNetModel, StableDiffusionControlNetPipeline = self._import_diffusers()
        load_kwargs: dict[str, Any] = {
            "torch_dtype": self._torch_dtype(torch),
            "cache_dir": self._cache_dir(),
        }
        token = self._hf_token()
        if token:
            load_kwargs["token"] = token
        try:
            controlnet = ControlNetModel.from_pretrained(self._controlnet_repo_id(), **load_kwargs)
            pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                self._base_model_repo_id(),
                controlnet=controlnet,
                **load_kwargs,
            )
        except Exception as exc:  # pragma: no cover - depends on environment
            message = str(exc)
            if "gated" in message.lower() or "401" in message or "403" in message or "access" in message.lower():
                raise RuntimeError(
                    "ControlNet lineart rectifier could not access one of the Hugging Face model repos. "
                    "If your chosen base model is gated, accept the license and set `HF_TOKEN`, or switch to a public model."
                ) from exc
            raise RuntimeError(f"Failed to load ControlNet lineart rectifier pipeline: {message}") from exc
        pipeline.to(self._device())
        if hasattr(pipeline, "set_progress_bar_config"):
            pipeline.set_progress_bar_config(disable=True)
        if hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing()
        if hasattr(pipeline, "safety_checker"):
            pipeline.safety_checker = None
        self._pipeline = pipeline
        return self._pipeline

    def _conditioning_source(self) -> str:
        return str(self.settings.get("conditioning_source", "generator_output"))

    def _subject_to_canny_conditioning(self, subject_image: Image.Image, params: dict[str, Any]) -> Image.Image:
        """Extract Canny edges from the real crop as ControlNet conditioning.

        This gives a structurally accurate input (faithful to the actual object shape)
        so ControlNet can simplify it via the text prompt rather than trying to fix
        a bad InformativeDrawings output.
        """
        if cv2 is None:
            return subject_image.convert("RGB")
        arr = np.asarray(subject_image.convert("L"), dtype=np.uint8)
        lo = int(params.get("canny_low_threshold", self.settings.get("canny_low_threshold", 100)))
        hi = int(params.get("canny_high_threshold", self.settings.get("canny_high_threshold", 200)))
        edges = cv2.Canny(arr, lo, hi)
        # ControlNet lineart convention: white background, black lines
        return Image.fromarray(255 - edges).convert("RGB")

    def _prompt_context(self, sample: dict[str, Any]) -> str:
        tags = [
            tag
            for tag in sample.get("tags", [])
            if tag
            and tag
            not in {"bioclip_enrichment", "drawable_subject", "photo_source", "segmented", "train"}
        ]
        prompt_parts = [sample.get("category", ""), sample.get("subcategory", ""), sample.get("angle_bucket", ""), ", ".join(tags[:4])]
        return ", ".join(part for part in prompt_parts if part)

    def _build_inference_kwargs(self, sample: dict[str, Any], conditioning_image: Image.Image, params: dict[str, Any]) -> dict[str, Any]:
        prompt_template = str(
            self.settings.get(
                "prompt",
                "a clean human-drawn style outline of {context}, full subject visible, minimal internal detail, white background, black ink lines",
            )
        )
        prompt = str(params.get("rectifier_prompt") or prompt_template.format(context=self._prompt_context(sample) or "the subject"))
        negative_prompt = str(
            params.get(
                "rectifier_negative_prompt",
                self.settings.get(
                    "negative_prompt",
                    "detailed texture, shading, hatching, feathers, fur, scales, interior detail, clutter, background objects, color fills",
                ),
            )
        )
        kwargs: dict[str, Any] = {
            "image": conditioning_image.convert("RGB"),
            "prompt": prompt,
            "guidance_scale": float(params.get("guidance_scale", self.settings.get("guidance_scale", 7.5))),
            "num_inference_steps": int(params.get("num_inference_steps", self.settings.get("num_inference_steps", 30))),
        }
        optional = {
            "negative_prompt": negative_prompt,
            "controlnet_conditioning_scale": float(
                params.get("controlnet_conditioning_scale", self.settings.get("controlnet_conditioning_scale", 1.0))
            ),
        }
        signature = inspect.signature(self._load_pipeline().__call__)
        for key, value in optional.items():
            if key in signature.parameters:
                kwargs[key] = value
        return kwargs

    def _postprocess_generated_output(
        self,
        edited_image: Image.Image,
        subject_image: Image.Image,
        subject_mask_image: Image.Image | None,
        params: dict[str, Any],
    ) -> Image.Image:
        if edited_image.size != subject_image.size:
            edited_image = edited_image.resize(subject_image.size, Image.Resampling.LANCZOS)
        min_component_area = int(self.settings.get("min_component_area", 80))
        subject_mask = (
            _mask_array_from_image(subject_mask_image, subject_image.size, min_component_area)
            if subject_mask_image is not None
            else _subject_mask_from_image(
                subject_image,
                int(params.get("subject_threshold", self.settings.get("subject_threshold", 248))),
                int(params.get("silhouette_close_kernel", self.settings.get("silhouette_close_kernel", 9))),
                min_component_area,
            )
        )
        edited_mask = _binary_mask_from_rendered_image(
            edited_image,
            int(params.get("rectifier_threshold", self.settings.get("rectifier_threshold", 235))),
            min_component_area,
            int(params.get("rectifier_close_kernel", self.settings.get("rectifier_close_kernel", 5))),
        )
        subject_ratio = float(subject_mask.mean())
        edited_ratio = float(edited_mask.mean())
        constrained_mask = np.logical_and(
            edited_mask > 0,
            _dilate_binary_mask(
                subject_mask,
                int(params.get("silhouette_close_kernel", self.settings.get("silhouette_close_kernel", 9))),
            )
            > 0,
        ).astype(np.uint8)
        constrained_ratio = float(constrained_mask.mean())

        # ControlNet sometimes returns an almost full-canvas non-white image, which thresholding turns
        # into a rectangular border. Treat the generated mask as a proposal and fall back to the
        # subject silhouette when it touches every border or diverges too far from the subject extent.
        if (
            edited_ratio > 0.85
            or _touches_all_borders(edited_mask)
            or constrained_ratio < max(0.01, subject_ratio * 0.15)
            or constrained_ratio > min(0.95, subject_ratio * 1.75)
        ):
            combined_mask = subject_mask
        else:
            combined_mask = np.maximum(subject_mask, constrained_mask)
        outline = _extract_simple_outline(
            combined_mask,
            int(params.get("outline_close_kernel", self.settings.get("outline_close_kernel", 5))),
            float(params.get("min_outline_area", self.settings.get("min_outline_area", 300.0))),
            int(params.get("max_outlines", self.settings.get("max_outlines", 2))),
            float(params.get("outline_simplify_ratio", self.settings.get("outline_simplify_ratio", 0.003))),
            int(params.get("outline_stroke_width", self.settings.get("outline_stroke_width", 4))),
            float(params.get("outline_smooth_sigma", self.settings.get("outline_smooth_sigma", 0.0))),
        )
        return Image.fromarray(np.where(outline > 0, 0, 255).astype(np.uint8), mode="L")

    def run(
        self,
        sample: dict[str, Any],
        subject_image: Image.Image,
        subject_mask: Image.Image | None,
        generated_path: Path,
        destination: Path,
        params: dict[str, Any],
    ) -> OutlineRectifierResult:
        pipeline = self._load_pipeline()
        if self._conditioning_source() == "subject_canny":
            conditioning_image = self._subject_to_canny_conditioning(subject_image, params)
        else:
            conditioning_image = Image.open(generated_path).convert("RGB")
        inference_kwargs = self._build_inference_kwargs(sample, conditioning_image, params)
        result = pipeline(**inference_kwargs)
        edited_image = result.images[0]
        outline = self._postprocess_generated_output(edited_image, subject_image, subject_mask, params)
        destination.parent.mkdir(parents=True, exist_ok=True)
        outline.save(destination)
        return OutlineRectifierResult(
            diagram_path=destination,
            metadata={
                "model_backend": self.settings.get("model_backend", "diffusers"),
                "model_name": self._base_model_repo_id(),
                "controlnet_model_name": self._controlnet_repo_id(),
                "model_version_or_checkpoint": self.settings.get("model_version_or_checkpoint", self._base_model_repo_id()),
                "execution_mode": "controlnet_lineart_diffusers",
                "conditioning_source": self._conditioning_source(),
                "used_generator_input": generated_path.name,
                "prompt": inference_kwargs.get("prompt"),
                "negative_prompt": inference_kwargs.get("negative_prompt"),
                "inference_params": {
                    "mask_source": "silver_mask" if subject_mask is not None else "rgb_threshold",
                    "device": self._device(),
                    "torch_dtype": self._torch_dtype_name(),
                    "guidance_scale": float(params.get("guidance_scale", self.settings.get("guidance_scale", 7.5))),
                    "num_inference_steps": int(params.get("num_inference_steps", self.settings.get("num_inference_steps", 30))),
                    "controlnet_conditioning_scale": float(
                        params.get("controlnet_conditioning_scale", self.settings.get("controlnet_conditioning_scale", 1.0))
                    ),
                    "rectifier_threshold": int(params.get("rectifier_threshold", self.settings.get("rectifier_threshold", 235))),
                    "rectifier_close_kernel": int(params.get("rectifier_close_kernel", self.settings.get("rectifier_close_kernel", 5))),
                    "subject_threshold": int(params.get("subject_threshold", self.settings.get("subject_threshold", 248))),
                    "silhouette_close_kernel": int(params.get("silhouette_close_kernel", self.settings.get("silhouette_close_kernel", 9))),
                    "outline_close_kernel": int(params.get("outline_close_kernel", self.settings.get("outline_close_kernel", 5))),
                    "outline_simplify_ratio": float(params.get("outline_simplify_ratio", self.settings.get("outline_simplify_ratio", 0.012))),
                    "outline_stroke_width": int(params.get("outline_stroke_width", self.settings.get("outline_stroke_width", 3))),
                    "max_outlines": int(params.get("max_outlines", self.settings.get("max_outlines", 2))),
                },
            },
        )


class SemanticValidator:
    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = settings

    def validate_ready(self) -> None:
        return None

    def validate(
        self,
        sample: dict[str, Any],
        diagram_path: Path,
        opencv_result: OpenCVValidationResult,
    ) -> SemanticValidationResult:
        raise NotImplementedError

    def suggest_params(
        self,
        sample: dict[str, Any],
        subject_path: Path,
        diagram_path: Path,
        opencv_result: OpenCVValidationResult,
        current_params: dict[str, Any],
    ) -> dict[str, Any]:
        return {}


class MockSemanticValidator(SemanticValidator):
    def validate(
        self,
        sample: dict[str, Any],
        diagram_path: Path,
        opencv_result: OpenCVValidationResult,
    ) -> SemanticValidationResult:
        passed = opencv_result.passed and opencv_result.score >= float(self.settings.get("min_score", 0.55))
        return SemanticValidationResult(
            passed=passed,
            score=opencv_result.score,
            reason="mock semantic validation",
            metadata={"model_backend": "mock", "model_name": self.settings.get("model_name", "mock_semantic_validator")},
        )


class OllamaSemanticValidator(SemanticValidator):
    def _model_name(self) -> str:
        return self.settings.get("ollama_model_name", self.settings.get("model_name", "qwen2.5vl:7b"))

    def _model_id(self) -> str:
        return self.settings.get("ollama_model_id", self.settings.get("model_id", self._model_name()))

    def _host(self) -> str:
        return os.getenv("OLLAMA_HOST", self.settings.get("host", "http://127.0.0.1:11434"))

    def _metadata(self, **extra: Any) -> dict[str, Any]:
        metadata = {
            "model_backend": "ollama",
            "model_name": self._model_name(),
            "model_id": self._model_id(),
            "ollama_host": self._host(),
        }
        metadata.update(extra)
        return metadata

    def _build_prompt(self, sample: dict[str, Any], opencv_result: OpenCVValidationResult) -> str:
        return (
            "You are validating simple outline art for a robot drawing dataset. "
            "Judge whether this image is a clean, simple, well-defined outer outline that is easy to draw. "
            "Prefer one or two clear contours with minimal internal detail. "
            "Do not judge whether it matches a specific animal, category, or subcategory. "
            "Return JSON only with keys: passed (boolean), score (0 to 1), reason (string). "
            f"OpenCV score: {opencv_result.score:.4f}. "
            f"OpenCV flags: {', '.join(opencv_result.flags) if opencv_result.flags else 'none'}."
        )

    def _request(self, diagram_path: Path, prompt: str) -> dict[str, Any]:
        return self._request_with_images([diagram_path], prompt)

    def validate_ready(self) -> None:
        req = request.Request(f"{self._host().rstrip('/')}/api/tags", method="GET")
        try:
            with request.urlopen(req, timeout=float(self.settings.get("timeout_seconds", 10))) as response:
                json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:  # pragma: no cover - network path
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Ollama validator is not ready: {detail or exc.reason}") from exc
        except error.URLError as exc:  # pragma: no cover - network path
            raise RuntimeError(
                f"Ollama validator is not ready at {self._host()}. Start `ollama serve` and verify the model is available."
            ) from exc

    def _request_with_images(self, image_paths: list[Path], prompt: str) -> dict[str, Any]:
        images = [base64.b64encode(path.read_bytes()).decode("ascii") for path in image_paths]
        body = {
            "model": self._model_name(),
            "prompt": prompt,
            "images": images,
            "stream": False,
            "options": {
                "temperature": float(self.settings.get("temperature", 0.0)),
                "num_predict": int(self.settings.get("max_tokens", 200)),
            },
        }
        req = request.Request(
            f"{self._host().rstrip('/')}/api/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=float(self.settings.get("timeout_seconds", 60))) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:  # pragma: no cover - network path
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Ollama semantic validation failed: {detail or exc.reason}") from exc
        except error.URLError as exc:  # pragma: no cover - network path
            raise RuntimeError(f"Ollama semantic validation failed: {exc.reason}") from exc

    def validate(
        self,
        sample: dict[str, Any],
        diagram_path: Path,
        opencv_result: OpenCVValidationResult,
    ) -> SemanticValidationResult:
        if not opencv_result.passed:
            return SemanticValidationResult(
                passed=False,
                score=0.0,
                reason="skipped semantic validation because OpenCV pre-screen failed",
                metadata=self._metadata(skipped=True),
            )
        payload = self._request(diagram_path, self._build_prompt(sample, opencv_result))
        result = _extract_json_object(str(payload.get("response", "")))
        score = float(result.get("score", 0.0))
        passed = bool(result.get("passed", False)) and score >= float(self.settings.get("min_score", 0.55))
        reason = str(result.get("reason", "ollama validation completed"))
        return SemanticValidationResult(
            passed=passed,
            score=score,
            reason=reason,
            metadata=self._metadata(),
        )

    def suggest_params(
        self,
        sample: dict[str, Any],
        subject_path: Path,
        diagram_path: Path,
        opencv_result: OpenCVValidationResult,
        current_params: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = (
            "Image 1 is the subject crop. Image 2 is the current outline diagram. "
            "Suggest generator parameters to make the result a complete full-subject outer outline only. "
            "Remove inner structures like fur, feathers, scales, and texture lines. "
            "Prefer a complete simple silhouette-style outline that is easy to draw. "
            "Return JSON only. Include any useful keys from this set: "
            "render_mode, subject_threshold, silhouette_close_kernel, outline_simplify_ratio, "
            "outline_stroke_width, max_outlines, threshold, blur_radius, reason. "
            f"OpenCV score: {opencv_result.score:.4f}. "
            f"OpenCV flags: {', '.join(opencv_result.flags) if opencv_result.flags else 'none'}. "
            f"Current params: {json.dumps(current_params, sort_keys=True)}."
        )
        payload = self._request_with_images([subject_path, diagram_path], prompt)
        result = _extract_json_object(str(payload.get("response", "")))
        allowed_keys = {
            "render_mode",
            "subject_threshold",
            "silhouette_close_kernel",
            "outline_simplify_ratio",
            "outline_stroke_width",
            "max_outlines",
            "threshold",
            "blur_radius",
            "reason",
        }
        return {key: value for key, value in result.items() if key in allowed_keys}


def _extract_json_object(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("{") and content.endswith("}"):
        return json.loads(content)
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Anthropic validator did not return a JSON object.")
    return json.loads(content[start : end + 1])


class AnthropicSemanticValidator(SemanticValidator):
    def validate_ready(self) -> None:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("Anthropic semantic validation requires `ANTHROPIC_API_KEY`.")

    def validate(
        self,
        sample: dict[str, Any],
        diagram_path: Path,
        opencv_result: OpenCVValidationResult,
    ) -> SemanticValidationResult:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Anthropic semantic validation requires `ANTHROPIC_API_KEY`.")

        model_name = self.settings.get("anthropic_model_name", self.settings.get("model_name", "claude-haiku-4-5"))
        api_base = self.settings.get("anthropic_api_base", "https://api.anthropic.com")
        image_bytes = diagram_path.read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        prompt = (
            "You are validating simple outline art for a robot drawing dataset. "
            "Judge whether this image is a clean, simple, well-defined outer outline that is easy to draw. "
            "Prefer one or two clear contours with minimal internal detail. "
            "Do not judge whether it matches a specific animal, category, or subcategory. "
            "Return JSON only with keys: passed (boolean), score (0 to 1), reason (string). "
            f"OpenCV score: {opencv_result.score:.4f}. "
            f"OpenCV flags: {', '.join(opencv_result.flags) if opencv_result.flags else 'none'}."
        )
        body = {
            "model": model_name,
            "max_tokens": int(self.settings.get("anthropic_max_tokens", 200)),
            "temperature": float(self.settings.get("anthropic_temperature", 0.0)),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64,
                            },
                        },
                    ],
                }
            ],
        }
        req = request.Request(
            f"{api_base.rstrip('/')}/v1/messages",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=float(self.settings.get("anthropic_timeout_seconds", 30))) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:  # pragma: no cover - network path
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Anthropic semantic validation failed: {detail or exc.reason}") from exc
        except error.URLError as exc:  # pragma: no cover - network path
            raise RuntimeError(f"Anthropic semantic validation failed: {exc.reason}") from exc

        content_blocks = payload.get("content", [])
        text = "\n".join(block.get("text", "") for block in content_blocks if block.get("type") == "text")
        result = _extract_json_object(text)
        score = float(result.get("score", 0.0))
        passed = bool(result.get("passed", False)) and score >= float(self.settings.get("min_score", 0.55))
        reason = str(result.get("reason", "anthropic validation completed"))
        return SemanticValidationResult(
            passed=passed,
            score=score,
            reason=reason,
            metadata={
                "model_backend": "anthropic",
                "model_name": model_name,
                "api_base": api_base,
            },
        )


def build_outline_generator(config: dict[str, Any]) -> DiagramBackend:
    backend_name = config.get("backend", "informative_drawings")
    if backend_name == "informative_drawings":
        return InformativeDrawingsBackend(config)
    raise ValueError(f"Unknown outline generator backend: {backend_name}")


def build_outline_rectifier(config: dict[str, Any]) -> OutlineRectifier:
    backend_name = config.get("backend", "silhouette_outline_rectifier")
    if backend_name == "silhouette_outline_rectifier":
        rectifier = SilhouetteOutlineRectifier(config)
        rectifier.validate_ready()
        return rectifier
    if backend_name == "controlnet_lineart_rectifier":
        rectifier = ControlNetLineArtRectifier(config)
        rectifier.validate_ready()
        return rectifier
    raise ValueError(f"Unknown outline rectifier backend: {backend_name}")


def build_outline_validator(config: dict[str, Any]) -> SemanticValidator:
    backend_name = config.get("backend", "auto")
    if backend_name == "auto":
        backend_name = "anthropic_semantic_validator" if os.getenv("ANTHROPIC_API_KEY") else "ollama_semantic_validator"
    if backend_name == "mock_semantic_validator":
        validator = MockSemanticValidator(config)
        validator.validate_ready()
        return validator
    if backend_name == "ollama_semantic_validator":
        validator = OllamaSemanticValidator(config)
        validator.validate_ready()
        return validator
    if backend_name == "anthropic_semantic_validator":
        validator = AnthropicSemanticValidator(config)
        validator.validate_ready()
        return validator
    raise ValueError(f"Unknown outline validator backend: {backend_name}")


def build_diagram_backend(config: dict[str, Any]) -> DiagramBackend:
    return build_outline_generator(config)


def build_semantic_validator(config: dict[str, Any]) -> SemanticValidator:
    return build_outline_validator(config)


def run_outline_pass(
    sample: dict[str, Any],
    subject_image: Image.Image,
    subject_mask: Image.Image | None,
    destination: Path,
    params: dict[str, Any],
    generator: DiagramBackend,
    rectifier: OutlineRectifier,
) -> OutlineAttemptResult:
    raw_destination = destination.with_name(f"{destination.stem}_generated{destination.suffix}")
    generated = generator.run(subject_image, raw_destination, params)
    try:
        rectified = rectifier.run(sample, subject_image, subject_mask, generated.diagram_path, destination, params)
    finally:
        if raw_destination.exists():
            raw_destination.unlink()
    return OutlineAttemptResult(
        diagram_path=rectified.diagram_path,
        generator_metadata=generated.metadata,
        rectifier_metadata=rectified.metadata,
    )


def compute_opencv_metrics(diagram: Image.Image) -> dict[str, float]:
    arr = np.asarray(diagram.convert("L"), dtype=np.uint8)
    # Diagrams are black lines on white background; treat dark pixels as foreground (lines).
    binary = np.where(arr < 128, 255, 0).astype(np.uint8)
    foreground_ratio = float(np.count_nonzero(binary)) / float(binary.size or 1)
    if cv2 is not None:
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour_count = float(len(contours))
        component_count = float(cv2.connectedComponents((binary > 0).astype(np.uint8))[0] - 1)
        small_contours = float(sum(1 for contour in contours if cv2.contourArea(contour) < 12.0))
    else:
        components = connected_components(np.where(binary > 0, 1, 0).astype(np.uint8))
        contour_count = float(len(components))
        component_count = float(len(components))
        small_contours = float(sum(1 for component in components if len(component) < 12))
    return {
        "foreground_ratio": foreground_ratio,
        "contour_count": contour_count,
        "component_count": component_count,
        "small_contour_ratio": small_contours / max(contour_count, 1.0),
    }


def validate_with_opencv(diagram_path: Path, settings: dict[str, Any]) -> OpenCVValidationResult:
    image = Image.open(diagram_path).convert("L")
    metrics = compute_opencv_metrics(image)
    flags: list[str] = []
    if metrics["foreground_ratio"] < float(settings.get("min_foreground_ratio", 0.03)):
        flags.append("low_foreground_ratio")
    if metrics["foreground_ratio"] > float(settings.get("max_foreground_ratio", 0.32)):
        flags.append("high_foreground_ratio")
    if metrics["component_count"] > float(settings.get("max_component_count", 18)):
        flags.append("too_many_components")
    if metrics["small_contour_ratio"] > float(settings.get("max_small_contour_ratio", 0.7)):
        flags.append("too_many_small_contours")
    score = max(
        0.0,
        1.0
        - (0.8 * abs(metrics["foreground_ratio"] - float(settings.get("target_foreground_ratio", 0.16))))
        - (0.03 * max(metrics["component_count"] - 3.0, 0.0))
        - (0.6 * metrics["small_contour_ratio"]),
    )
    return OpenCVValidationResult(passed=not flags, score=score, flags=flags, metrics=metrics)


def next_generator_params(current: dict[str, Any], opencv_flags: list[str], semantic_passed: bool) -> dict[str, Any]:
    updated = dict(current)
    threshold = int(updated.get("threshold", 150))
    blur_radius = float(updated.get("blur_radius", 0.8))
    if "high_foreground_ratio" in opencv_flags:
        threshold = min(threshold + 18, 235)
    elif "low_foreground_ratio" in opencv_flags:
        threshold = max(threshold - 18, 40)
    elif not semantic_passed:
        blur_radius = min(blur_radius + 0.35, 3.0)
        threshold = min(threshold + 8, 235)
    updated["threshold"] = threshold
    updated["blur_radius"] = blur_radius
    return updated


class ValidationGraphState(TypedDict, total=False):
    sample: dict[str, Any]
    subject_path: str
    subject_mask_path: str | None
    diagram_root: str
    max_attempts: int
    current_params: dict[str, Any]
    attempt_count: int
    attempt_offset: int
    latest_diagram_path: str
    latest_generator_metadata: dict[str, Any]
    latest_rectifier_metadata: dict[str, Any]
    opencv_result: dict[str, Any]
    semantic_result: dict[str, Any]
    improver_feedback: dict[str, Any]
    accepted: bool
    final_record: dict[str, Any]
    outline_generator: DiagramBackend
    outline_rectifier: OutlineRectifier
    outline_validator: SemanticValidator
    opencv_settings: dict[str, Any]


def _generate_attempt(state: ValidationGraphState) -> ValidationGraphState:
    subject_image = Image.open(state["subject_path"]).convert("RGB")
    subject_mask = None
    if state.get("subject_mask_path"):
        subject_mask = _cropped_mask_image(
            Image.open(state["subject_mask_path"]),
            state["sample"].get("crop_bbox"),
            subject_image.size,
        )
    attempt_count = int(state.get("attempt_count", 0)) + 1
    display_attempt = attempt_count + int(state.get("attempt_offset", 0))
    diagram_root = Path(state["diagram_root"])
    sample = state["sample"]
    destination = diagram_root / sample["category"] / f"{sample['sample_id']}_attempt{display_attempt:02d}.png"
    result = run_outline_pass(
        sample,
        subject_image,
        subject_mask,
        destination,
        state["current_params"],
        state["outline_generator"],
        state["outline_rectifier"],
    )
    return {
        "attempt_count": attempt_count,
        "latest_diagram_path": str(result.diagram_path),
        "latest_generator_metadata": result.generator_metadata,
        "latest_rectifier_metadata": result.rectifier_metadata,
    }


def _opencv_validate(state: ValidationGraphState) -> ValidationGraphState:
    result = validate_with_opencv(Path(state["latest_diagram_path"]), state["opencv_settings"])
    return {
        "opencv_result": {
        "passed": result.passed,
        "score": result.score,
        "flags": result.flags,
        "metrics": result.metrics,
        }
    }


def _semantic_validate(state: ValidationGraphState) -> ValidationGraphState:
    opencv_result = OpenCVValidationResult(**state["opencv_result"])
    result = state["outline_validator"].validate(state["sample"], Path(state["latest_diagram_path"]), opencv_result)
    semantic_result = {
        "passed": result.passed,
        "score": result.score,
        "reason": result.reason,
        "metadata": result.metadata,
    }
    accepted = opencv_result.passed and result.passed
    final_record = {
        "sample_id": state["sample"]["sample_id"],
        "category": state["sample"]["category"],
        "subcategory": state["sample"]["subcategory"],
        "angle_bucket": state["sample"]["angle_bucket"],
        "diagram_path": state["latest_diagram_path"],
        "diagram_attempt": int(state["attempt_count"]) + int(state.get("attempt_offset", 0)),
        "outline_generator_backend": state["latest_generator_metadata"],
        "outline_rectifier_backend": state["latest_rectifier_metadata"],
        "diagram_validation_status": "accepted" if accepted else "rejected",
        "diagram_validation_score": round((opencv_result.score * 0.5) + (result.score * 0.5), 4),
        "opencv_flags": opencv_result.flags,
        "outline_validator_model": result.metadata.get("model_name", "qwen2.5vl:7b"),
        "outline_validator_reason": result.reason,
        "selected_for_gold": False,
        "selection_rank": None,
        "lineage": {
            "subject_sample_id": state["sample"]["sample_id"],
            "diagram_attempt_count": state["attempt_count"],
        },
    }
    return {
        "semantic_result": semantic_result,
        "accepted": accepted,
        "final_record": final_record,
    }


def _route_after_semantic(state: ValidationGraphState) -> str:
    if state.get("accepted"):
        return "finish"
    if int(state["attempt_count"]) >= int(state["max_attempts"]):
        return "finish"
    return "feedback"


def _vlm_feedback(state: ValidationGraphState) -> ValidationGraphState:
    suggest = getattr(state["outline_validator"], "suggest_params", None)
    if suggest is None:
        return {"improver_feedback": {}}
    feedback = suggest(
        state["sample"],
        Path(state["subject_path"]),
        Path(state["latest_diagram_path"]),
        OpenCVValidationResult(**state["opencv_result"]),
        state["current_params"],
    )
    return {"improver_feedback": feedback}


def _prepare_retry(state: ValidationGraphState) -> ValidationGraphState:
    updated = next_generator_params(
            state["current_params"],
            list(state["opencv_result"].get("flags", [])),
            bool(state["semantic_result"].get("passed", False)),
        )
    feedback = dict(state.get("improver_feedback") or {})
    feedback.pop("reason", None)
    updated.update(feedback)
    return {"current_params": updated}


def run_langgraph_validation_loop(
    sample: dict[str, Any],
    subject_path: Path,
    subject_mask_path: Path | None,
    diagram_root: Path,
    outline_generator: DiagramBackend,
    outline_rectifier: OutlineRectifier,
    outline_validator: SemanticValidator,
    opencv_settings: dict[str, Any],
    initial_params: dict[str, Any],
    max_attempts: int,
    attempt_offset: int = 0,
) -> dict[str, Any]:
    state: ValidationGraphState = {
        "sample": sample,
        "subject_path": str(subject_path),
        "subject_mask_path": str(subject_mask_path) if subject_mask_path else None,
        "diagram_root": str(diagram_root),
        "outline_generator": outline_generator,
        "outline_rectifier": outline_rectifier,
        "outline_validator": outline_validator,
        "opencv_settings": opencv_settings,
        "current_params": dict(initial_params),
        "attempt_count": 0,
        "attempt_offset": attempt_offset,
        "max_attempts": max_attempts,
    }
    if StateGraph is None:
        while True:  # pragma: no cover - exercised only without langgraph installed.
            state.update(_generate_attempt(state))
            state.update(_opencv_validate(state))
            state.update(_semantic_validate(state))
            route = _route_after_semantic(state)
            if route == "finish":
                return state["final_record"]
            state.update(_vlm_feedback(state))
            state.update(_prepare_retry(state))
        return state["final_record"]

    graph = StateGraph(ValidationGraphState)
    graph.add_node("generate", _generate_attempt)
    graph.add_node("opencv", _opencv_validate)
    graph.add_node("semantic", _semantic_validate)
    graph.add_node("feedback", _vlm_feedback)
    graph.add_node("prepare_retry", _prepare_retry)
    graph.set_entry_point("generate")
    graph.add_edge("generate", "opencv")
    graph.add_edge("opencv", "semantic")
    graph.add_conditional_edges(
        "semantic",
        _route_after_semantic,
        {
            "feedback": "feedback",
            "finish": END,
        },
    )
    graph.add_edge("feedback", "prepare_retry")
    graph.add_edge("prepare_retry", "generate")
    app = graph.compile()
    result = app.invoke(state)
    return dict(result["final_record"])
