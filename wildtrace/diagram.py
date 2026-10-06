from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import os
import time
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

    def unload(self) -> None:
        """Free any loaded model; the next run() reloads it."""
        return None

    def prepare_conditioning_image(
        self,
        crop_image: Image.Image,
        isolated_image: Image.Image | None = None,
    ) -> Image.Image:
        return crop_image

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
            # Default keep_alive leaves the ~6GB vision model resident in RAM
            # between calls. That's fine on its own, but concurrently with the
            # Flux pipeline's ~20GB host-RAM working set it pushes this box
            # over its 23GB and gets OOM-killed. Unload immediately after each
            # response so Ollama's footprint only spikes during the call.
            # Benchmarks override this to measure warm latency with nothing
            # else resident.
            "keep_alive": self.settings.get("keep_alive", "0s"),
        }
        # Thinking-capable models (qwen3.x) otherwise spend the num_predict
        # budget reasoning and return no JSON.
        if self.settings.get("think") is not None:
            body["think"] = bool(self.settings["think"])
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
        if not opencv_result.passed and not self.settings.get("force_semantic", False):
            return SemanticValidationResult(
                passed=False,
                score=0.0,
                reason="skipped semantic validation because OpenCV pre-screen failed",
                metadata=self._metadata(skipped=True),
            )
        started = time.perf_counter()
        payload = self._request(diagram_path, self._build_prompt(sample, opencv_result))
        call_metadata = self._metadata(
            latency_s=round(time.perf_counter() - started, 3),
            prompt_tokens=payload.get("prompt_eval_count"),
            output_tokens=payload.get("eval_count"),
            cost_usd=payload.get("cost_usd"),
        )
        try:
            result = _extract_json_object(str(payload.get("response", "")))
        except (ValueError, json.JSONDecodeError):
            return SemanticValidationResult(
                passed=False, score=0.0,
                reason="ollama validator returned unparseable response",
                metadata={**call_metadata, "parse_failed": True},
            )
        score = float(result.get("score", 0.0))
        verdict = bool(result.get("passed", False))
        passed = verdict and score >= float(self.settings.get("min_score", 0.55))
        reason = str(result.get("reason", "ollama validation completed"))
        return SemanticValidationResult(
            passed=passed,
            score=score,
            reason=reason,
            metadata={**call_metadata, "raw_verdict": verdict},
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


class OpenRouterSemanticValidator(OllamaSemanticValidator):
    """Same prompt, parsing and retry feedback as the Ollama validator, served
    by OpenRouter's OpenAI-compatible API instead of a local model."""

    def _model_name(self) -> str:
        return self.settings.get("openrouter_model_name", "qwen/qwen3-vl-8b-instruct")

    def _model_id(self) -> str:
        return self._model_name()

    def _api_base(self) -> str:
        return str(self.settings.get("openrouter_api_base", "https://openrouter.ai/api/v1"))

    def _metadata(self, **extra: Any) -> dict[str, Any]:
        return {"model_backend": "openrouter", "model_name": self._model_name(), "api_base": self._api_base(), **extra}

    def validate_ready(self) -> None:
        if not os.getenv("OPENROUTER_API_KEY"):
            raise RuntimeError("OpenRouter semantic validation requires `OPENROUTER_API_KEY`.")

    def _request_with_images(self, image_paths: list[Path], prompt: str) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in image_paths:
            image_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
            media_type = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
            content.append({"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_b64}"}})
        body = {
            "model": self._model_name(),
            "messages": [{"role": "user", "content": content}],
            "temperature": float(self.settings.get("temperature", 0.0)),
            "max_tokens": int(self.settings.get("max_tokens", 200)),
            "usage": {"include": True},
        }
        req = request.Request(
            f"{self._api_base().rstrip('/')}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY', '')}",
                "x-title": "WildTrace",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=float(self.settings.get("openrouter_timeout_seconds", 60))) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:  # pragma: no cover - network path
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenRouter semantic validation failed: {detail or exc.reason}") from exc
        except error.URLError as exc:  # pragma: no cover - network path
            raise RuntimeError(f"OpenRouter semantic validation failed: {exc.reason}") from exc
        if "error" in payload:
            raise RuntimeError(f"OpenRouter semantic validation failed: {payload['error']}")
        usage = payload.get("usage") or {}
        # Reshape to the Ollama /api/generate payload the parent class reads.
        return {
            "response": payload["choices"][0]["message"].get("content") or "",
            "prompt_eval_count": usage.get("prompt_tokens"),
            "eval_count": usage.get("completion_tokens"),
            "cost_usd": usage.get("cost"),
        }


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
    backends: dict[str, type[OutlineRectifier]] = {
        "silhouette_outline_rectifier": SilhouetteOutlineRectifier,
        "flux_silhouette_rectifier": FluxSilhouetteRectifier,
        "flux_kontext_rectifier": FluxKontextRectifier,
        "sd_controlnet_lineart_rectifier": SDControlNetLineartRectifier,
        "sdxl_controlnet_rectifier": SDXLControlNetRectifier,
        "informative_drawings_rectifier": InformativeDrawingsRectifier,
        "omnigen2_rectifier": OmniGen2Rectifier,
    }
    if backend_name not in backends:
        raise ValueError(f"Unknown outline rectifier backend: {backend_name}")
    rectifier = backends[backend_name](config)
    rectifier.validate_ready()
    return rectifier


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
    if backend_name == "openrouter_semantic_validator":
        validator = OpenRouterSemanticValidator(config)
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
    conditioning_path: str | None
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
    crop_image = Image.open(state["subject_path"]).convert("RGB")
    isolated_image = None
    if state.get("conditioning_path"):
        isolated_image = Image.open(state["conditioning_path"]).convert("RGB")
    subject_image = state["diagram_generator"].prepare_conditioning_image(crop_image, isolated_image)
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
    conditioning_path: Path | None,
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
        "conditioning_path": str(conditioning_path) if conditioning_path else None,
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

    app = build_validation_langgraph()
    result = app.invoke(state)
    return dict(result["final_record"])


def build_validation_langgraph():
    if StateGraph is None:
        raise RuntimeError("langgraph is not installed.")

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
    return graph.compile()


def _silhouette_contour_image(
    subject_image: Image.Image,
    subject_mask: Image.Image | None,
    params: dict[str, Any],
    settings: dict[str, Any],
    size: int = 512,
) -> Image.Image:
    """Black 2px outer contour of the subject mask on white, size x size.
    Shared conditioning input so every generative rectifier starts from the
    same shape."""
    import cv2

    min_component_area = int(settings.get("min_component_area", 80))
    if subject_mask is not None:
        mask_arr = _mask_array_from_image(subject_mask, subject_image.size, min_component_area)
    else:
        mask_arr = _subject_mask_from_image(
            subject_image,
            int(params.get("subject_threshold", settings.get("subject_threshold", 248))),
            int(params.get("silhouette_close_kernel", 9)),
            min_component_area,
        )

    mask_resized = np.asarray(
        Image.fromarray(mask_arr, mode="L").resize((size, size), Image.NEAREST),
        dtype=np.uint8,
    )
    # Masks that reach this helper are 0/1 after component filtering. A
    # threshold of 128 treated every pixel as background and erased the contour
    # for FLUX, SD1.5, and SDXL alike.
    mask_bin = (mask_resized > 0).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_closed = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = np.ones((size, size), dtype=np.uint8) * 255
    for cnt in contours:
        if cv2.contourArea(cnt) > 200:
            epsilon = 0.003 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(contour_img, [approx], -1, 0, thickness=2)
    return Image.fromarray(contour_img, mode="L")


def _binarize_to_size(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Black lines on white at `size`, the format every rectifier must emit."""
    arr = np.asarray(ImageOps.grayscale(image), dtype=np.uint8)
    binary = np.where(arr < 128, 0, 255).astype(np.uint8)
    return Image.fromarray(binary, mode="L").resize(size, Image.NEAREST)


def _prepared(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def build_line_drawing_prompt(sample: dict[str, Any]) -> str:
    """Shared rectifier prompt: species/breed, pose from view_features, angle."""
    species = str(sample.get("category", "animal")).lower()
    subcategory = str(sample.get("subcategory", "")).lower()
    angle = str(sample.get("angle_bucket", "")).replace("_", " ")
    view_features = sample.get("view_features", {})

    # Breed/subspecies term — use subcategory only when it adds info beyond the species.
    # Strip common redundant prefixes like "domestic_cat" when species is already "cat".
    clean_sub = subcategory.replace("domestic_", "").replace("_", " ")
    breed = clean_sub if clean_sub and clean_sub != species else ""

    # Pose from view_features
    symmetry = float(view_features.get("symmetry_score", 0.0))
    bottom_mass = float(view_features.get("bottom_mass_ratio", 0.5))
    direction = float(view_features.get("direction_score", 0.0))

    # Sitting: high symmetry + mass concentrated in lower half (not applicable to fish/bird)
    no_legs = species in ("fish", "snake", "worm")
    if no_legs:
        pose = "swimming" if species == "fish" else "moving"
    elif symmetry > 0.35 and bottom_mass > 0.6:
        pose = "sitting"
    elif symmetry > 0.35:
        pose = "standing, front view"
    elif direction < -0.3:
        pose = "walking, facing left"
    elif direction > 0.3:
        pose = "walking, facing right"
    else:
        pose = "standing"

    # Viewing angle descriptor
    angle_desc = angle if angle else "side view"

    subject = f"{breed} {species}".strip() if breed else species

    parts = [
        f"simple clean line drawing of a {subject}, {pose}, {angle_desc}",
        "coloring book page for children",
        f"full body visible, {subject} with distinct features matching the pose",
        "black outline on white background",
        "minimalist doodle, no shading, no fill, no color",
    ]
    return ", ".join(parts)


class FluxSilhouetteRectifier(OutlineRectifier):
    def __init__(self, settings: dict[str, Any]) -> None:
        super().__init__(settings)
        self._pipeline: Any | None = None

    def prepare_conditioning_image(
        self,
        crop_image: Image.Image,
        isolated_image: Image.Image | None = None,
    ) -> Image.Image:
        mode = str(self.settings.get("conditioning_mode", "isolated_only")).strip().lower()
        if mode == "crop_only":
            return crop_image.copy()
        if isolated_image is None:
            raise ValueError("Flux conditioning requires an isolated subject image.")

        crop = crop_image.convert("RGB")
        isolated = isolated_image.convert("RGB")
        if isolated.size != crop.size:
            isolated = isolated.resize(crop.size, Image.Resampling.LANCZOS)
        if mode == "isolated_only":
            return isolated
        if mode == "blended":
            return Image.blend(crop, isolated, alpha=0.5)
        raise ValueError(
            f"Unknown Flux conditioning mode: {mode}. Expected `crop_only`, `isolated_only`, or `blended`."
        )

    def _build_flux_prompt(self, sample: dict[str, Any]) -> str:
        return build_line_drawing_prompt(sample)

    def _import_torch(self) -> Any:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(f"{type(self).__name__} requires `torch`.") from exc
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

        # enable_sequential_cpu_offload() would give a much smaller steady
        # -state footprint, but accelerate's per-tensor device hook chokes on
        # this pipeline's GGUF-quantized transformer (KeyError on the quant
        # type when moving a tensor to the "meta" device), so it's not usable
        # here. Stuck with model_cpu_offload's ~20GB host-RAM footprint --
        # see the swap-headroom note in generate_line_diagrams/validate_and_
        # retry_diagrams docs for how the concurrent-Ollama OOM is handled.
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
        # --- Stage 1: Silhouette contour extraction ---
        silhouette = _silhouette_contour_image(subject_image, subject_mask, params, self.settings)

        # --- Stage 2: FLUX img2img Refinement ---
        pipe = self._load_pipeline()

        input_img = silhouette.convert("RGB")
        prompt = self._build_flux_prompt(sample)
        strength = float(params.get("strength", self.settings.get("strength", 0.85)))

        # Sequential calls on a cached pipeline otherwise accumulate host RAM
        # (accelerate's cpu-offload hooks + autograd bookkeeping never release
        # between calls) until the process is OOM-killed partway through a
        # multi-sample batch. inference_mode stops graph retention; the
        # explicit cache/gc pass after each call is the standard diffusers
        # workaround for the offload-hook growth.
        torch = self._import_torch()
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                image=input_img,
                strength=strength,
                num_inference_steps=4,
                guidance_scale=0.0,
            ).images[0]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc

        gc.collect()

        _binarize_to_size(result, subject_image.size).save(_prepared(destination))

        return OutlineRectifierResult(
            diagram_path=destination,
            metadata={
                "backend": "flux_silhouette_rectifier",
                "strength": strength,
                "prompt": prompt,
            },
        )

    def unload(self) -> None:
        _release_pipeline(self)


def _release_pipeline(rectifier: Any) -> None:
    """Drop a cached pipeline and return its GPU/host memory, so the next
    model in a benchmark sweep starts from a clean device."""
    rectifier._pipeline = None
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover - torch is a core dependency
        pass


class FluxKontextRectifier(FluxSilhouetteRectifier):
    """FLUX.1-Kontext-dev (GGUF) instruction edit of the silhouette contour.

    Text encoders and VAE are FLUX-family components identical to the
    FLUX.1-schnell ones already cached for the baseline rectifier, so only
    the quantized transformer and Kontext's small config files are fetched
    (the 24GB bf16 transformer never is)."""

    def _import_diffusers(self) -> tuple[Any, Any, Any]:
        try:
            from diffusers import FluxKontextPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
        except ImportError as exc:
            raise RuntimeError("FluxKontextRectifier requires diffusers>=0.35 with FluxKontextPipeline.") from exc
        return FluxKontextPipeline, FluxTransformer2DModel, GGUFQuantizationConfig

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        torch = self._import_torch()
        kontext_pipeline_cls, transformer_cls, gguf_config_cls = self._import_diffusers()
        from transformers import T5EncoderModel

        kontext_repo = str(self.settings.get("kontext_repo", "black-forest-labs/FLUX.1-Kontext-dev"))
        encoder_repo = str(self.settings.get("text_encoder_repo", "black-forest-labs/FLUX.1-schnell"))
        try:
            gguf_path = hf_hub_download(
                str(self.settings.get("gguf_repo", "QuantStack/FLUX.1-Kontext-dev-GGUF")),
                str(self.settings.get("gguf_file", "flux1-kontext-dev-Q4_K_S.gguf")),
            )
            transformer = transformer_cls.from_single_file(
                gguf_path,
                quantization_config=gguf_config_cls(compute_dtype=torch.bfloat16),
                torch_dtype=torch.bfloat16,
                config=kontext_repo,
                subfolder="transformer",
            )
            text_encoder_2 = T5EncoderModel.from_pretrained(
                encoder_repo, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
            )
            pipe = kontext_pipeline_cls.from_pretrained(
                kontext_repo,
                transformer=transformer,
                text_encoder_2=text_encoder_2,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load FLUX Kontext GGUF pipeline: {exc}") from exc
        pipe.enable_model_cpu_offload()
        self._pipeline = pipe
        return self._pipeline

    def _build_instruction(self, sample: dict[str, Any]) -> str:
        return (
            "Turn this silhouette outline into a finished drawing, keeping exactly the same pose and outer shape: "
            + build_line_drawing_prompt(sample)
        )

    def run(
        self,
        sample: dict[str, Any],
        subject_image: Image.Image,
        subject_mask: Image.Image | None,
        generated_path: Path | None,
        destination: Path,
        params: dict[str, Any],
    ) -> OutlineRectifierResult:
        size = int(self.settings.get("size", 512))
        silhouette = _silhouette_contour_image(subject_image, subject_mask, params, self.settings, size=size)
        pipe = self._load_pipeline()
        instruction = str(params.get("prompt") or self._build_instruction(sample))
        steps = int(params.get("num_inference_steps", self.settings.get("num_inference_steps", 20)))
        guidance = float(params.get("guidance_scale", self.settings.get("guidance_scale", 2.5)))
        torch = self._import_torch()
        with torch.inference_mode():
            result = pipe(
                image=silhouette.convert("RGB"),
                prompt=instruction,
                guidance_scale=guidance,
                num_inference_steps=steps,
                height=size,
                width=size,
                _auto_resize=False,
            ).images[0]
        _release_cuda_cache(torch)
        _binarize_to_size(result, subject_image.size).save(_prepared(destination))
        return OutlineRectifierResult(
            diagram_path=destination,
            metadata={
                "backend": "flux_kontext_rectifier",
                "prompt": instruction,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
            },
        )


class SDControlNetLineartRectifier(OutlineRectifier):
    """SD1.5 coloring-book checkpoint + ControlNet lineart, conditioned on the
    silhouette contour. Small enough to sit fully on the GPU."""

    def __init__(self, settings: dict[str, Any]) -> None:
        super().__init__(settings)
        self._pipeline: Any | None = None

    def validate_ready(self) -> None:
        try:
            import torch  # noqa: F401
            from diffusers import ControlNetModel, StableDiffusionControlNetPipeline  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("SDControlNetLineartRectifier requires `torch` and `diffusers`.") from exc

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        import torch
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

        try:
            controlnet = ControlNetModel.from_pretrained(
                str(self.settings.get("controlnet_repo", "lllyasviel/control_v11p_sd15_lineart")),
                variant="fp16",
                torch_dtype=torch.float16,
            )
            checkpoint = hf_hub_download(
                str(self.settings.get("checkpoint_repo", "artificialguybr/ColoringBookSD")),
                str(self.settings.get("checkpoint_file", "VARPJ1.safetensors")),
            )
            pipe = StableDiffusionControlNetPipeline.from_single_file(
                checkpoint,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                config=str(self.settings.get("base_config", "stable-diffusion-v1-5/stable-diffusion-v1-5")),
                safety_checker=None,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load SD ControlNet lineart pipeline: {exc}") from exc
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        self._pipeline = pipe
        return self._pipeline

    def unload(self) -> None:
        _release_pipeline(self)

    def run(
        self,
        sample: dict[str, Any],
        subject_image: Image.Image,
        subject_mask: Image.Image | None,
        generated_path: Path | None,
        destination: Path,
        params: dict[str, Any],
    ) -> OutlineRectifierResult:
        import torch

        size = int(self.settings.get("size", 512))
        silhouette = _silhouette_contour_image(subject_image, subject_mask, params, self.settings, size=size)
        # ControlNet lineart is trained on white lines over black.
        control = ImageOps.invert(silhouette).convert("RGB")
        pipe = self._load_pipeline()
        prompt = str(params.get("prompt") or build_line_drawing_prompt(sample))
        steps = int(params.get("num_inference_steps", self.settings.get("num_inference_steps", 20)))
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=str(self.settings.get("negative_prompt", "color, shading, gray, texture, background, text")),
                image=control,
                num_inference_steps=steps,
                guidance_scale=float(self.settings.get("guidance_scale", 7.0)),
                controlnet_conditioning_scale=float(self.settings.get("controlnet_conditioning_scale", 1.0)),
                height=size,
                width=size,
            ).images[0]
        _release_cuda_cache(torch)
        _binarize_to_size(result, subject_image.size).save(_prepared(destination))
        return OutlineRectifierResult(
            diagram_path=destination,
            metadata={"backend": "sd_controlnet_lineart_rectifier", "prompt": prompt, "num_inference_steps": steps},
        )


class SDXLControlNetRectifier(OutlineRectifier):
    """SDXL base plus scribble ControlNet, conditioned on the shared silhouette.

    Checkpoints are read from ``sdxl_`` settings so FLUX GGUF keys in the same
    config block are not reused. The production default stays FLUX.
    """

    DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
    DEFAULT_CONTROLNET = "xinsir/controlnet-scribble-sdxl-1.0"
    DEFAULT_VAE = "madebyollin/sdxl-vae-fp16-fix"

    def __init__(self, settings: dict[str, Any]) -> None:
        super().__init__(settings)
        self._pipeline: Any | None = None

    def prepare_conditioning_image(
        self,
        crop_image: Image.Image,
        isolated_image: Image.Image | None = None,
    ) -> Image.Image:
        mode = str(self.settings.get("conditioning_mode", "isolated_only")).strip().lower()
        if mode == "crop_only":
            return crop_image.copy()
        if isolated_image is None:
            raise ValueError("SDXL conditioning requires an isolated subject image.")
        crop = crop_image.convert("RGB")
        isolated = isolated_image.convert("RGB")
        if isolated.size != crop.size:
            isolated = isolated.resize(crop.size, Image.Resampling.LANCZOS)
        if mode == "isolated_only":
            return isolated
        if mode == "blended":
            return Image.blend(crop, isolated, alpha=0.5)
        raise ValueError(
            f"Unknown SDXL conditioning mode: {mode}. Expected `crop_only`, `isolated_only`, or `blended`."
        )

    def validate_ready(self) -> None:
        try:
            import torch  # noqa: F401
            from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetPipeline  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("SDXLControlNetRectifier requires `torch` and `diffusers`.") from exc

    def _setting(self, name: str, default: Any) -> Any:
        prefixed = f"sdxl_{name}"
        if prefixed in self.settings:
            return self.settings[prefixed]
        return default

    def _seed_base(self) -> int:
        if "sdxl_seed" in self.settings:
            return int(self.settings["sdxl_seed"])
        return int(self.settings.get("seed", 0))

    def _conditioning_scale(self, params: dict[str, Any]) -> float:
        """Translate the shared retry ``strength`` knob into ControlNet scale.

        Higher strength means freer of the contour. ControlNet scale is the
        inverse, so nominal strength 0.85 keeps the configured scale and each
        retry step of 0.05 moves the scale by 0.20.
        """
        base = float(
            params.get(
                "controlnet_conditioning_scale",
                self._setting("controlnet_conditioning_scale", 1.0),
            )
        )
        if "strength" not in params:
            return base
        adjusted = base - 4.0 * (float(params["strength"]) - 0.85)
        return round(min(max(adjusted, 0.35), 1.5), 2)

    def _execution_device(self, torch: Any) -> str:
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        import torch
        from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetPipeline

        device = self._execution_device(torch)
        dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
        model_id = str(self._setting("model", self.DEFAULT_MODEL))
        controlnet_id = str(self._setting("controlnet", self.DEFAULT_CONTROLNET))
        vae_id = str(self._setting("vae", self.DEFAULT_VAE))
        try:
            controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype)
            vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=dtype)
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                model_id,
                controlnet=controlnet,
                vae=vae,
                torch_dtype=dtype,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load SDXL ControlNet pipeline: {exc}") from exc
        if device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        self._pipeline = pipe
        return self._pipeline

    def unload(self) -> None:
        _release_pipeline(self)

    def run(
        self,
        sample: dict[str, Any],
        subject_image: Image.Image,
        subject_mask: Image.Image | None,
        generated_path: Path | None,
        destination: Path,
        params: dict[str, Any],
    ) -> OutlineRectifierResult:
        import torch
        import zlib

        contour_size = int(self._setting("contour_size", 512))
        size = int(params.get("size", self._setting("size", 1024)))
        silhouette = _silhouette_contour_image(
            subject_image, subject_mask, params, self.settings, size=contour_size
        )
        if silhouette.size != (size, size):
            silhouette = silhouette.resize((size, size), Image.Resampling.NEAREST)
        # Scribble ControlNet is trained on white strokes over black.
        control = ImageOps.invert(silhouette).convert("RGB")
        pipe = self._load_pipeline()
        prompt = str(params.get("prompt") or build_line_drawing_prompt(sample))
        negative_prompt = str(
            params.get("negative_prompt", self._setting("negative_prompt", "color, shading, gray, texture, background, text"))
        )
        steps = int(params.get("num_inference_steps", self._setting("num_inference_steps", 30)))
        guidance = float(params.get("guidance_scale", self._setting("guidance_scale", 7.0)))
        conditioning_scale = self._conditioning_scale(params)
        seed = self._seed_base() + zlib.crc32(f"{sample.get('sample_id')}:{destination.name}".encode())
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=control,
                num_inference_steps=steps,
                guidance_scale=guidance,
                controlnet_conditioning_scale=conditioning_scale,
                height=size,
                width=size,
                generator=generator,
            ).images[0]
        _release_cuda_cache(torch)
        _binarize_to_size(result, subject_image.size).save(_prepared(destination))
        return OutlineRectifierResult(
            diagram_path=destination,
            metadata={
                "backend": "sdxl_controlnet_rectifier",
                "model": str(self._setting("model", self.DEFAULT_MODEL)),
                "controlnet": str(self._setting("controlnet", self.DEFAULT_CONTROLNET)),
                "vae": str(self._setting("vae", self.DEFAULT_VAE)),
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": seed,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "controlnet_conditioning_scale": conditioning_scale,
                "size": size,
                "contour_size": contour_size,
                "strength": params.get("strength"),
            },
        )


class InformativeDrawingsRectifier(OutlineRectifier):
    """Photo -> line-art GAN (Chan et al., "Learning to generate line
    drawings that convey geometry and semantics"), ONNX. Unlike the other
    rectifiers it reads the subject crop, not the silhouette contour: it is a
    photo-to-sketch network with no text or shape conditioning."""

    def __init__(self, settings: dict[str, Any]) -> None:
        super().__init__(settings)
        self._pipeline: Any | None = None

    def validate_ready(self) -> None:
        try:
            import onnxruntime  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("InformativeDrawingsRectifier requires `onnxruntime`.") from exc

    def prepare_conditioning_image(
        self,
        crop_image: Image.Image,
        isolated_image: Image.Image | None = None,
    ) -> Image.Image:
        # The crop, not the isolated image: isolated_path is the full frame
        # while the mask is cropped to crop_bbox and sized to what this
        # returns, so only the crop lines up with it. run() whites out the
        # background with the mask, which gives the isolated crop.
        return crop_image.convert("RGB")

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        import onnxruntime

        model_path = hf_hub_download(
            str(self.settings.get("onnx_repo", "rocca/informative-drawings-line-art-onnx")),
            str(self.settings.get("onnx_file", "model.onnx")),
        )
        self._pipeline = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        return self._pipeline

    def unload(self) -> None:
        _release_pipeline(self)

    def run(
        self,
        sample: dict[str, Any],
        subject_image: Image.Image,
        subject_mask: Image.Image | None,
        generated_path: Path | None,
        destination: Path,
        params: dict[str, Any],
    ) -> OutlineRectifierResult:
        size = int(self.settings.get("size", 512))
        source = subject_image.convert("RGB")
        if subject_mask is not None:
            # White out everything outside the subject so only its lines remain.
            mask = subject_mask.convert("L").resize(source.size, Image.NEAREST)
            source = Image.composite(source, Image.new("RGB", source.size, (255, 255, 255)), mask)
        batch = np.asarray(source.resize((size, size), Image.BICUBIC), dtype=np.float32) / 255.0
        batch = batch.transpose(2, 0, 1)[None, ...]
        session = self._load_pipeline()
        output = session.run(None, {session.get_inputs()[0].name: batch})[0][0, 0]
        line_art = Image.fromarray((np.clip(output, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
        _binarize_to_size(line_art, subject_image.size).save(_prepared(destination))
        return OutlineRectifierResult(
            diagram_path=destination,
            metadata={"backend": "informative_drawings_rectifier", "input": "masked_subject_crop"},
        )


class OmniGen2Rectifier(OutlineRectifier):
    """OmniGen2 (Qwen2.5-VL-3B front-end + 4B diffusion decoder) instruction
    edit: reads the masked subject photo, or the silhouette contour with
    `conditioning_input: contour`, and redraws it as a line drawing.

    The pipeline code is not on PyPI or in diffusers, and its pinned
    requirements (torch 2.6, transformers 4.51) clash with ours, so it is
    imported from a git checkout (scripts/install_omnigen2.sh) put on sys.path
    rather than installed."""

    DEFAULT_NEGATIVE_PROMPT = (
        "color, shading, gray fill, texture, background, scenery, text, watermark, blurry, "
        "deformed, bad anatomy, extra limbs, messy drawing"
    )

    def __init__(self, settings: dict[str, Any]) -> None:
        super().__init__(settings)
        self._pipeline: Any | None = None

    def _code_dir(self) -> Path:
        return Path(str(self.settings.get("omnigen2_code_dir", "~/.cache/wildtrace/OmniGen2"))).expanduser()

    def _import_omnigen2(self) -> tuple[Any, Any, Any]:
        import sys

        code_dir = self._code_dir()
        if not (code_dir / "omnigen2").is_dir():
            raise RuntimeError(f"OmniGen2 code not found at {code_dir}. Run scripts/install_omnigen2.sh.")
        if str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))
        try:
            from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
            from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
            from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
        except ImportError as exc:
            raise RuntimeError(f"Failed to import OmniGen2 from {code_dir}: {exc}") from exc
        return OmniGen2Pipeline, OmniGen2Transformer2DModel, DPMSolverMultistepScheduler

    def validate_ready(self) -> None:
        self._import_omnigen2()

    def prepare_conditioning_image(
        self,
        crop_image: Image.Image,
        isolated_image: Image.Image | None = None,
    ) -> Image.Image:
        # See InformativeDrawingsRectifier: only the crop aligns with the mask.
        return crop_image.convert("RGB")

    def _bf16_dir(self) -> Path:
        return Path(str(self.settings.get("omnigen2_bf16_dir", "~/.cache/wildtrace/OmniGen2-bf16"))).expanduser()

    def _load_component(self, name: str) -> Any:
        """Load one pipeline component in bf16, from the local bf16 export when
        present (scripts/install_omnigen2.sh), else from the fp32 hub weights."""
        import torch
        from diffusers import AutoencoderKL
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

        _, transformer_cls, _ = self._import_omnigen2()
        local = self._bf16_dir()
        if (local / name).is_dir():
            source, extra = str(local), {}
        else:
            source = str(self.settings.get("omnigen2_repo", "OmniGen2/OmniGen2"))
            extra = {"revision": self.settings.get("omnigen2_revision")}
        if name == "transformer":
            return transformer_cls.from_pretrained(source, subfolder=name, torch_dtype=torch.bfloat16, **extra)
        if name == "vae":
            return AutoencoderKL.from_pretrained(source, subfolder=name, torch_dtype=torch.bfloat16, **extra)
        if name == "mllm":
            return Qwen2_5_VLForConditionalGeneration.from_pretrained(source, subfolder=name, dtype=torch.bfloat16, **extra)
        if name == "processor":
            return Qwen2_5_VLProcessor.from_pretrained(source, subfolder=name, **extra)
        raise ValueError(f"Unknown OmniGen2 component: {name}")

    def export_bf16(self) -> Path:
        """One-time: the hub weights are fp32 (31GB) and casting them at every
        load peaks host RAM past this box's 23GB. Save a bf16 copy (~16GB),
        one component at a time so only one fp32 copy is ever in memory."""
        import gc

        out_dir = self._bf16_dir()
        for name in ("processor", "vae", "mllm", "transformer"):
            if (out_dir / name).is_dir():
                continue
            component = self._load_component(name)
            component.save_pretrained(out_dir / f"{name}.partial")
            (out_dir / f"{name}.partial").rename(out_dir / name)
            del component
            gc.collect()
        return out_dir

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        import torch

        pipeline_cls, _, dpm_scheduler_cls = self._import_omnigen2()
        try:
            # Built from components rather than pipeline_cls.from_pretrained,
            # which imports the repo's model_index modules as remote code.
            pipe = pipeline_cls(
                transformer=self._load_component("transformer"),
                vae=self._load_component("vae"),
                scheduler=dpm_scheduler_cls(
                    algorithm_type="dpmsolver++",
                    solver_type="midpoint",
                    solver_order=2,
                    prediction_type="flow_prediction",
                ),
                mllm=self._load_component("mllm"),
                processor=self._load_component("processor"),
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load OmniGen2 pipeline: {exc}") from exc
        offload = str(self.settings.get("offload", "model"))
        if offload == "model":
            pipe.enable_model_cpu_offload()
        elif offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        elif offload == "encoder_cpu":
            # The Qwen2.5-VL-3B mllm only encodes the ~100-token prompt, so it
            # stays on CPU (see _encode_on_cpu) and the transformer + VAE stay
            # resident on GPU: host RAM holds 7.5GB of weights instead of all 16.
            pipe.transformer.to("cuda")
            pipe.vae.to("cuda")
        else:
            pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        self._pipeline = pipe
        return self._pipeline

    def unload(self) -> None:
        _release_pipeline(self)

    def _build_instruction(self, sample: dict[str, Any], style: str | None = None) -> str:
        # An edit instruction ("Convert this photo into ..."): phrased as a
        # description ("Redraw this animal as ...") OmniGen2 returned the
        # input photo unchanged. A judge's retry prompt is a text-to-image
        # style description, so it is appended rather than replacing the edit.
        species = str(sample.get("category", "animal")).lower()
        instruction = (
            f"Convert this photo into a black and white line drawing of the {species}: clean black outlines only, "
            "no shading, no fill, no color. Keep the same pose, size and position. Plain white background."
        )
        return f"{instruction} Style: {style}" if style else instruction

    @staticmethod
    def _encode_on_cpu(pipe: Any, prompt: str, negative_prompt: str, guided: bool) -> dict[str, Any]:
        prompt_embeds, prompt_mask, negative_embeds, negative_mask = pipe.encode_prompt(
            prompt, guided, negative_prompt=negative_prompt, device="cpu"
        )
        device = pipe.transformer.device
        return {
            "prompt_embeds": prompt_embeds.to(device),
            "prompt_attention_mask": prompt_mask.to(device),
            "negative_prompt_embeds": negative_embeds.to(device) if negative_embeds is not None else None,
            "negative_prompt_attention_mask": negative_mask.to(device) if negative_mask is not None else None,
        }

    def _image_guidance(self, params: dict[str, Any]) -> float:
        # The retry loop steers `strength` (higher = freer from the input,
        # nominal 0.85); the matching OmniGen2 knob is image guidance,
        # inverted and nudged around the configured base.
        base = float(params.get("image_guidance_scale", self.settings.get("image_guidance_scale", 1.6)))
        if "strength" in params:
            return round(min(max(base - 4.0 * (float(params["strength"]) - 0.85), 1.1), 3.0), 2)
        return base

    def run(
        self,
        sample: dict[str, Any],
        subject_image: Image.Image,
        subject_mask: Image.Image | None,
        generated_path: Path | None,
        destination: Path,
        params: dict[str, Any],
    ) -> OutlineRectifierResult:
        import torch

        size = int(self.settings.get("size", 768))
        conditioning_input = str(self.settings.get("conditioning_input", "subject"))
        if conditioning_input == "contour":
            source = _silhouette_contour_image(subject_image, subject_mask, params, self.settings, size=size).convert("RGB")
        else:
            source = subject_image.convert("RGB")
            if subject_mask is not None:
                mask = subject_mask.convert("L").resize(source.size, Image.NEAREST)
                source = Image.composite(source, Image.new("RGB", source.size, (255, 255, 255)), mask)
        # Letterbox onto a white size x size square and crop the drawing back
        # out afterwards: on elongated crops (e.g. 768x432) OmniGen2 returned
        # a blank white page for most subjects, while square inputs of the
        # same subjects came back as faithful line drawings.
        scale = size / max(source.size)
        inner = (max(1, round(source.width * scale)), max(1, round(source.height * scale)))
        offset = ((size - inner[0]) // 2, (size - inner[1]) // 2)
        square = Image.new("RGB", (size, size), (255, 255, 255))
        square.paste(source.resize(inner, Image.BICUBIC), offset)
        source, width, height = square, size, size

        pipe = self._load_pipeline()
        instruction = self._build_instruction(sample, params.get("prompt"))
        steps = int(params.get("num_inference_steps", self.settings.get("num_inference_steps", 30)))
        text_guidance = float(params.get("text_guidance_scale", self.settings.get("text_guidance_scale", 5.0)))
        image_guidance = self._image_guidance(params)
        negative_prompt = str(self.settings.get("negative_prompt", self.DEFAULT_NEGATIVE_PROMPT))
        # Guidance only over this fraction of the schedule; later steps run
        # a single unguided pass, which roughly halves the cost of those steps.
        cfg_range = tuple(float(v) for v in self.settings.get("cfg_range", (0.0, 1.0)))
        # A fresh, reproducible seed per subject and attempt: OmniGen2 lands in
        # a blank-page mode for some seeds, so a fixed seed made every retry of
        # such a subject blank too (FLUX draws fresh noise on every call).
        import zlib

        seed = int(self.settings.get("seed", 0)) + zlib.crc32(f"{sample.get('sample_id')}:{destination.name}".encode())
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.inference_mode():
            embeds: dict[str, Any] = {}
            if str(self.settings.get("offload", "model")) == "encoder_cpu":
                embeds = self._encode_on_cpu(pipe, instruction, negative_prompt, text_guidance > 1.0)
            result = pipe(
                prompt=instruction,
                negative_prompt=negative_prompt,
                **embeds,
                input_images=[source],
                width=width,
                height=height,
                max_pixels=size * size,
                max_input_image_side_length=size,
                num_inference_steps=steps,
                text_guidance_scale=text_guidance,
                image_guidance_scale=image_guidance,
                cfg_range=cfg_range,
                generator=generator,
                output_type="pil",
            ).images[0]
        _release_cuda_cache(torch)
        result = result.resize((size, size), Image.BICUBIC).crop(
            (offset[0], offset[1], offset[0] + inner[0], offset[1] + inner[1])
        )
        _binarize_to_size(result, subject_image.size).save(_prepared(destination))
        return OutlineRectifierResult(
            diagram_path=destination,
            metadata={
                "backend": "omnigen2_rectifier",
                "input": conditioning_input,
                "seed": seed,
                "prompt": instruction,
                "num_inference_steps": steps,
                "text_guidance_scale": text_guidance,
                "image_guidance_scale": image_guidance,
                "cfg_range": list(cfg_range),
            },
        )


def _release_cuda_cache(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc

    gc.collect()
