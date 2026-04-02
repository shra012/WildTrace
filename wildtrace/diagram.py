from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, TypedDict
from urllib import error, request

import numpy as np
from PIL import Image, ImageFilter, ImageOps

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


class InformativeDrawingsBackend(DiagramBackend):
    def run(self, image: Image.Image, destination: Path, params: dict[str, Any]) -> DiagramBackendResult:
        gray = ImageOps.grayscale(image)
        blur_radius = float(params.get("blur_radius", self.settings.get("blur_radius", 0.8)))
        threshold = int(params.get("threshold", self.settings.get("threshold", 150)))
        outline = gray.filter(ImageFilter.GaussianBlur(radius=blur_radius)).filter(ImageFilter.FIND_EDGES)
        arr = np.asarray(outline, dtype=np.uint8)
        binary = np.where(arr >= threshold, 255, 0).astype(np.uint8)
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(binary, mode="L").save(destination)
        return DiagramBackendResult(
            diagram_path=destination,
            metadata={
                "model_backend": "local",
                "model_name": self.settings.get("model_name", "informative_drawings"),
                "model_version_or_checkpoint": self.settings.get("model_version_or_checkpoint", "heuristic-emulation-v1"),
                "execution_mode": "heuristic_emulation",
                "inference_params": {"blur_radius": blur_radius, "threshold": threshold},
            },
        )


class SemanticValidator:
    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = settings

    def validate(
        self,
        sample: dict[str, Any],
        diagram_path: Path,
        opencv_result: OpenCVValidationResult,
    ) -> SemanticValidationResult:
        raise NotImplementedError


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
            "You are validating line art for a robot drawing dataset. "
            "Judge whether this diagram is simple, elegant, recognizable, clean, well-defined, and easy to draw. "
            "Prefer sparse line art over noisy or cluttered detail. "
            "Return JSON only with keys: passed (boolean), score (0 to 1), reason (string). "
            f"Category: {sample['category']}. "
            f"Subcategory: {sample['subcategory']}. "
            f"Angle bucket: {sample['angle_bucket']}. "
            f"OpenCV score: {opencv_result.score:.4f}. "
            f"OpenCV flags: {', '.join(opencv_result.flags) if opencv_result.flags else 'none'}."
        )

    def _request(self, diagram_path: Path, prompt: str) -> dict[str, Any]:
        image_b64 = base64.b64encode(diagram_path.read_bytes()).decode("ascii")
        body = {
            "model": self._model_name(),
            "prompt": prompt,
            "images": [image_b64],
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
            "You are validating line art for a robot drawing dataset. "
            "Judge whether this diagram is simple, recognizable, clean, well-defined, and easy to draw. "
            "Prefer sparse elegant line art over noisy detail. "
            "Return JSON only with keys: passed (boolean), score (0 to 1), reason (string). "
            f"Category: {sample['category']}. "
            f"Subcategory: {sample['subcategory']}. "
            f"Angle bucket: {sample['angle_bucket']}. "
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


def build_diagram_backend(config: dict[str, Any]) -> DiagramBackend:
    backend_name = config.get("backend", "informative_drawings")
    if backend_name == "informative_drawings":
        return InformativeDrawingsBackend(config)
    raise ValueError(f"Unknown diagram backend: {backend_name}")


def build_semantic_validator(config: dict[str, Any]) -> SemanticValidator:
    backend_name = config.get("backend", "auto")
    if backend_name == "auto":
        backend_name = "anthropic_semantic_validator" if os.getenv("ANTHROPIC_API_KEY") else "ollama_semantic_validator"
    if backend_name == "ollama_semantic_validator":
        return OllamaSemanticValidator(config)
    if backend_name == "anthropic_semantic_validator":
        return AnthropicSemanticValidator(config)
    raise ValueError(f"Unknown semantic validator backend: {backend_name}")


def compute_opencv_metrics(diagram: Image.Image) -> dict[str, float]:
    arr = np.asarray(diagram.convert("L"), dtype=np.uint8)
    binary = np.where(arr > 0, 255, 0).astype(np.uint8)
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
    diagram_root: str
    max_attempts: int
    current_params: dict[str, Any]
    attempt_count: int
    attempt_offset: int
    latest_diagram_path: str
    latest_metadata: dict[str, Any]
    opencv_result: dict[str, Any]
    semantic_result: dict[str, Any]
    accepted: bool
    final_record: dict[str, Any]
    generator: DiagramBackend
    validator: SemanticValidator
    opencv_settings: dict[str, Any]


def _generate_attempt(state: ValidationGraphState) -> ValidationGraphState:
    subject_image = Image.open(state["subject_path"]).convert("RGB")
    attempt_count = int(state.get("attempt_count", 0)) + 1
    state["attempt_count"] = attempt_count
    display_attempt = attempt_count + int(state.get("attempt_offset", 0))
    diagram_root = Path(state["diagram_root"])
    sample = state["sample"]
    destination = diagram_root / sample["category"] / f"{sample['sample_id']}_attempt{display_attempt:02d}.png"
    result = state["generator"].run(subject_image, destination, state["current_params"])
    state["latest_diagram_path"] = str(result.diagram_path)
    state["latest_metadata"] = result.metadata
    return state


def _opencv_validate(state: ValidationGraphState) -> ValidationGraphState:
    result = validate_with_opencv(Path(state["latest_diagram_path"]), state["opencv_settings"])
    state["opencv_result"] = {
        "passed": result.passed,
        "score": result.score,
        "flags": result.flags,
        "metrics": result.metrics,
    }
    return state


def _semantic_validate(state: ValidationGraphState) -> ValidationGraphState:
    opencv_result = OpenCVValidationResult(**state["opencv_result"])
    result = state["validator"].validate(state["sample"], Path(state["latest_diagram_path"]), opencv_result)
    state["semantic_result"] = {
        "passed": result.passed,
        "score": result.score,
        "reason": result.reason,
        "metadata": result.metadata,
    }
    accepted = opencv_result.passed and result.passed
    state["accepted"] = accepted
    state["final_record"] = {
        "sample_id": state["sample"]["sample_id"],
        "category": state["sample"]["category"],
        "subcategory": state["sample"]["subcategory"],
        "angle_bucket": state["sample"]["angle_bucket"],
        "diagram_path": state["latest_diagram_path"],
        "diagram_attempt": int(state["attempt_count"]) + int(state.get("attempt_offset", 0)),
        "diagram_generator_backend": state["latest_metadata"],
        "diagram_validation_status": "accepted" if accepted else "rejected",
        "diagram_validation_score": round((opencv_result.score * 0.5) + (result.score * 0.5), 4),
        "opencv_flags": opencv_result.flags,
        "semantic_validator_model": result.metadata.get("model_name", "qwen2.5vl:7b"),
        "semantic_validator_reason": result.reason,
        "selected_for_gold": False,
        "selection_rank": None,
        "lineage": {
            "subject_sample_id": state["sample"]["sample_id"],
            "diagram_attempt_count": state["attempt_count"],
        },
    }
    return state


def _route_after_semantic(state: ValidationGraphState) -> str:
    if state.get("accepted"):
        return "finish"
    if int(state["attempt_count"]) >= int(state["max_attempts"]):
        return "finish"
    state["current_params"] = next_generator_params(
        state["current_params"],
        list(state["opencv_result"].get("flags", [])),
        bool(state["semantic_result"].get("passed", False)),
    )
    return "retry"


def run_langgraph_validation_loop(
    sample: dict[str, Any],
    subject_path: Path,
    diagram_root: Path,
    generator: DiagramBackend,
    validator: SemanticValidator,
    opencv_settings: dict[str, Any],
    initial_params: dict[str, Any],
    max_attempts: int,
    attempt_offset: int = 0,
) -> dict[str, Any]:
    state: ValidationGraphState = {
        "sample": sample,
        "subject_path": str(subject_path),
        "diagram_root": str(diagram_root),
        "generator": generator,
        "validator": validator,
        "opencv_settings": opencv_settings,
        "current_params": dict(initial_params),
        "attempt_count": 0,
        "attempt_offset": attempt_offset,
        "max_attempts": max_attempts,
    }
    if StateGraph is None:
        while True:  # pragma: no cover - exercised only without langgraph installed.
            state = _generate_attempt(state)
            state = _opencv_validate(state)
            state = _semantic_validate(state)
            route = _route_after_semantic(state)
            if route == "finish":
                return state["final_record"]
        return state["final_record"]

    graph = StateGraph(dict)
    graph.add_node("generate", _generate_attempt)
    graph.add_node("opencv", _opencv_validate)
    graph.add_node("semantic", _semantic_validate)
    graph.set_entry_point("generate")
    graph.add_edge("generate", "opencv")
    graph.add_edge("opencv", "semantic")
    graph.add_conditional_edges(
        "semantic",
        _route_after_semantic,
        {
            "retry": "generate",
            "finish": END,
        },
    )
    app = graph.compile()
    result = app.invoke(state)
    return dict(result["final_record"])
