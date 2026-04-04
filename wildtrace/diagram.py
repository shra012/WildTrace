from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, TypedDict
from urllib import error, request

import numpy as np
from PIL import Image, ImageOps
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


@dataclass(slots=True)
class DiagramBackendResult:
    diagram_path: Path
    metadata: dict[str, Any]


@dataclass(slots=True)
class OutlineRectifierResult:
    diagram_path: Path
    metadata: dict[str, Any]


@dataclass(slots=True)
class GeneratedDiagramResult:
    diagram_path: Path
    metadata: dict[str, Any]


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
        generated_path: Path | None,
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


class SilhouetteOutlineRectifier(OutlineRectifier):
    def run(
        self,
        sample: dict[str, Any],
        subject_image: Image.Image,
        subject_mask: Image.Image | None,
        generated_path: Path | None,
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
        return str(self.settings.get("host") or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))

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
            "You are validating a simple cartoon doodle drawing for an animal drawing dataset. "
            "Judge whether this image shows a recognizable animal with black ink strokes on a mostly white background. "
            "A good drawing has a clear full-body outline, simple readable body parts, and minimal interior clutter. "
            "Do not judge whether it matches a specific species or subcategory. "
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
        try:
            result = _extract_json_object(str(payload.get("response", "")))
        except (ValueError, json.JSONDecodeError):
            return SemanticValidationResult(
                passed=False, score=0.0,
                reason="ollama validator returned unparseable response",
                metadata=self._metadata(),
            )
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
            "Image 1 is the subject animal photo. Image 2 is the current cartoon doodle drawing. "
            "Suggest FLUX img2img generator parameters to improve the drawing so it becomes a simple cartoon doodle "
            "with a complete body outline, simple readable body parts, black ink strokes, and minimal interior clutter. "
            "Return JSON only. Include any useful keys from this set: "
            "strength, prompt, reason. "
            f"OpenCV score: {opencv_result.score:.4f}. "
            f"OpenCV flags: {', '.join(opencv_result.flags) if opencv_result.flags else 'none'}. "
            f"Current params: {json.dumps(current_params, sort_keys=True)}."
        )
        payload = self._request_with_images([subject_path, diagram_path], prompt)
        try:
            result = _extract_json_object(str(payload.get("response", "")))
        except (ValueError, json.JSONDecodeError):
            return {}
        allowed_keys = {"strength", "prompt", "reason"}
        return {key: value for key, value in result.items() if key in allowed_keys}


def _extract_json_object(content: str) -> dict[str, Any]:
    content = content.strip()
    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    if content.startswith("```"):
        first_newline = content.find("\n")
        content = content[first_newline + 1 :] if first_newline != -1 else content[3:]
        if content.endswith("```"):
            content = content[: content.rfind("```")]
        content = content.strip()
    if content.startswith("{") and content.endswith("}"):
        return json.loads(content)
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Validator did not return a JSON object. Got: {content[:120]!r}")
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
            "You are validating a simple cartoon doodle drawing for an animal drawing dataset. "
            "Judge whether this image shows a recognizable animal with black ink strokes on a mostly white background. "
            "A good drawing has a clear full-body outline, simple readable body parts, and minimal interior clutter. "
            "Do not judge whether it matches a specific species or subcategory. "
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
        try:
            result = _extract_json_object(text)
        except (ValueError, json.JSONDecodeError):
            return SemanticValidationResult(
                passed=False, score=0.0,
                reason="anthropic validator returned unparseable response",
                metadata={"model_backend": "anthropic", "model_name": model_name, "api_base": api_base},
            )
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



def build_outline_rectifier(config: dict[str, Any]) -> OutlineRectifier:
    backend_name = config.get("backend", "silhouette_outline_rectifier")
    if backend_name == "silhouette_outline_rectifier":
        rectifier = SilhouetteOutlineRectifier(config)
        rectifier.validate_ready()
        return rectifier
    if backend_name == "flux_silhouette_rectifier":
        rectifier = FluxSilhouetteRectifier(config)
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



def build_semantic_validator(config: dict[str, Any]) -> SemanticValidator:
    return build_outline_validator(config)


def run_drawn_diagram_pass(
    sample: dict[str, Any],
    subject_image: Image.Image,
    subject_mask: Image.Image | None,
    destination: Path,
    params: dict[str, Any],
    generator: OutlineRectifier,
) -> GeneratedDiagramResult:
    generated = generator.run(sample, subject_image, subject_mask, None, destination, params)
    return GeneratedDiagramResult(
        diagram_path=generated.diagram_path,
        metadata=generated.metadata,
    )


def compute_opencv_metrics(diagram: Image.Image) -> dict[str, float]:
    arr = np.asarray(diagram.convert("L"), dtype=np.uint8)
    # Diagrams are black lines on white background; treat dark pixels as foreground (lines).
    binary = np.where(arr < 128, 255, 0).astype(np.uint8)
    foreground_ratio = float(np.count_nonzero(binary)) / float(binary.size or 1)
    h, w = binary.shape

    # Bounding-box bottom reach: how far down (0–1) the drawn content extends.
    # A value well below 1.0 indicates the body is truncated / cropped at the bottom.
    row_has_content = (binary > 0).any(axis=1)
    if row_has_content.any():
        body_bottom_reach = float(np.where(row_has_content)[0].max() + 1) / float(h)
    else:
        body_bottom_reach = 0.0

    # Lower-body fragment count: connected components in the bottom 35% of the image.
    # Many small disconnected blobs there indicate broken/stub legs.
    lower_start = int(h * 0.65)
    lower_binary = binary[lower_start:, :]
    if cv2 is not None:
        lower_component_count = float(cv2.connectedComponents((lower_binary > 0).astype(np.uint8))[0] - 1)
    else:
        lower_comps = connected_components(np.where(lower_binary > 0, 1, 0).astype(np.uint8))
        lower_component_count = float(len(lower_comps))

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
        "body_bottom_reach": body_bottom_reach,
        "lower_body_fragment_count": lower_component_count,
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
    # Body completeness checks
    if metrics["body_bottom_reach"] < float(settings.get("min_body_bottom_reach", 0.60)):
        flags.append("body_truncated")
    if metrics["lower_body_fragment_count"] > float(settings.get("max_lower_body_fragments", 5)):
        flags.append("too_many_lower_fragments")
    score = max(
        0.0,
        1.0
        - (0.8 * abs(metrics["foreground_ratio"] - float(settings.get("target_foreground_ratio", 0.16))))
        - (0.03 * max(metrics["component_count"] - 3.0, 0.0))
        - (0.6 * metrics["small_contour_ratio"])
        - (0.5 * max(0.0, float(settings.get("min_body_bottom_reach", 0.70)) - metrics["body_bottom_reach"]))
        - (0.04 * max(metrics["lower_body_fragment_count"] - 3.0, 0.0)),
    )
    return OpenCVValidationResult(passed=not flags, score=score, flags=flags, metrics=metrics)


def next_generator_params(current: dict[str, Any], opencv_flags: list[str], semantic_passed: bool) -> dict[str, Any]:
    """Adjust FLUX img2img retry params based on OpenCV flags.

    The primary knob for FLUX is `strength` (0–1).  Higher strength = more creative
    freedom (good for too-empty diagrams); lower strength = closer to the silhouette
    contour input (good for overfilled / noisy diagrams).
    """
    updated = dict(current)
    strength = float(updated.get("strength", 0.90))
    if "high_foreground_ratio" in opencv_flags or "too_many_components" in opencv_flags:
        strength = max(round(strength - 0.05, 2), 0.60)
    elif "low_foreground_ratio" in opencv_flags or "body_truncated" in opencv_flags:
        strength = min(round(strength + 0.05, 2), 0.98)
    elif "too_many_small_contours" in opencv_flags or "too_many_lower_fragments" in opencv_flags or not semantic_passed:
        strength = max(round(strength - 0.03, 2), 0.60)
    updated["strength"] = strength
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
    latest_diagram_metadata: dict[str, Any]
    opencv_result: dict[str, Any]
    semantic_result: dict[str, Any]
    improver_feedback: dict[str, Any]
    accepted: bool
    final_record: dict[str, Any]
    diagram_generator: OutlineRectifier
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
    result = run_drawn_diagram_pass(
        sample,
        subject_image,
        subject_mask,
        destination,
        state["current_params"],
        state["diagram_generator"],
    )
    return {
        "attempt_count": attempt_count,
        "latest_diagram_path": str(result.diagram_path),
        "latest_diagram_metadata": result.metadata,
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
        "diagram_generator_backend": state["latest_diagram_metadata"],
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
    diagram_generator: OutlineRectifier,
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
        "diagram_generator": diagram_generator,
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
class FluxSilhouetteRectifier(OutlineRectifier):
    def __init__(self, settings: dict[str, Any]) -> None:
        super().__init__(settings)
        self._pipeline: Any | None = None

    def _import_torch(self) -> Any:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("FluxSilhouetteRectifier requires `torch`.") from exc
        return torch

    def _import_diffusers(self) -> tuple[Any, Any, Any]:
        try:
            from diffusers import FluxImg2ImgPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
        except ImportError as exc:
            raise RuntimeError(
                "FluxSilhouetteRectifier requires `diffusers` (>=0.31), `transformers`, `accelerate`, and `gguf`. "
                "Install with `uv add diffusers transformers accelerate gguf`."
            ) from exc
        return FluxImg2ImgPipeline, FluxTransformer2DModel, GGUFQuantizationConfig

    def validate_ready(self) -> None:
        self._import_torch()
        self._import_diffusers()

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        
        torch = self._import_torch()
        FluxImg2ImgPipeline, FluxTransformer2DModel, GGUFQuantizationConfig = self._import_diffusers()

        gguf_repo = str(self.settings.get("gguf_repo", "city96/FLUX.1-schnell-gguf"))
        gguf_file = str(self.settings.get("gguf_file", "flux1-schnell-Q4_K_S.gguf"))
        base_repo = str(self.settings.get("base_model", "black-forest-labs/FLUX.1-schnell"))

        try:
            gguf_path = hf_hub_download(gguf_repo, gguf_file)
            transformer = FluxTransformer2DModel.from_single_file(
                gguf_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                torch_dtype=torch.bfloat16,
            )
            pipe = FluxImg2ImgPipeline.from_pretrained(
                base_repo,
                transformer=transformer,
                torch_dtype=torch.bfloat16,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load FLUX GGUF pipeline: {exc}") from exc

        pipe.enable_model_cpu_offload()
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None
        
        self._pipeline = pipe
        return self._pipeline

    def run(
        self,
        sample: dict[str, Any],
        subject_image: Image.Image,
        subject_mask: Image.Image | None,
        generated_path: Path | None,
        destination: Path,
        params: dict[str, Any],
    ) -> OutlineRectifierResult:
        import cv2

        # --- Stage 1: Silhouette contour extraction ---
        target_w, target_h = 512, 512
        min_component_area = int(self.settings.get("min_component_area", 80))
        
        if subject_mask is not None:
            mask_arr = _mask_array_from_image(subject_mask, subject_image.size, min_component_area)
        else:
            mask_arr = _subject_mask_from_image(
                subject_image,
                int(params.get("subject_threshold", self.settings.get("subject_threshold", 248))),
                int(params.get("silhouette_close_kernel", 9)),
                min_component_area,
            )
        
        mask_resized = np.asarray(
            Image.fromarray(mask_arr, mode="L").resize((target_w, target_h), Image.NEAREST),
            dtype=np.uint8,
        )
        mask_bin = (mask_resized > 128).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_closed = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel, iterations=3)

        contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = np.ones((target_h, target_w), dtype=np.uint8) * 255

        for cnt in contours:
            if cv2.contourArea(cnt) > 200:
                epsilon = 0.003 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                cv2.drawContours(contour_img, [approx], -1, 0, thickness=2)

        silhouette = Image.fromarray(contour_img, mode="L")

        # --- Stage 2: FLUX img2img Refinement ---
        pipe = self._load_pipeline()
        
        input_img = silhouette.convert("RGB")
        
        species = str(sample.get("category", "animal")).lower()
        prompt_template = str(self.settings.get(
            "prompt", 
            "simple clean line drawing of a {species}, coloring book page for children, "
            "cute {species} with visible features, black outline on white background, "
            "minimalist simple doodle, no shading, no fill, no color"
        ))
        prompt = prompt_template.format(species=species)
        
        strength = float(params.get("strength", self.settings.get("strength", 0.90)))
        
        result = pipe(
            prompt=prompt,
            image=input_img,
            strength=strength,
            num_inference_steps=4,
            guidance_scale=0.0,
        ).images[0]

        # Binarize output back to black lines on white
        r_gray = ImageOps.grayscale(result)
        r_arr = np.asarray(r_gray, dtype=np.uint8)
        r_bin = np.where(r_arr < 128, 0, 255).astype(np.uint8)
        
        # Resize to match original subject shape
        final = Image.fromarray(r_bin, mode="L").resize(subject_image.size, Image.NEAREST)
        
        destination.parent.mkdir(parents=True, exist_ok=True)
        final.save(destination)

        return OutlineRectifierResult(
            diagram_path=destination,
            metadata={
                "backend": "flux_silhouette_rectifier",
                "strength": strength,
                "prompt": prompt,
            },
        )
