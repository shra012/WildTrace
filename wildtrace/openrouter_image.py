"""OpenRouter-hosted image model as an outline rectifier.

Same contract as the local rectifiers in `wildtrace.diagram`: it takes the
shared silhouette contour conditioning image and the shared line-drawing
prompt, and writes black-on-white binarized line art at the subject size. The
model runs remotely, so this backend is the CPU-only fallback when the GPU is
busy (no FLUX, torch or diffusers load).
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib import error, request

from PIL import Image, ImageFilter

from wildtrace.diagram import (
    OutlineRectifier,
    OutlineRectifierResult,
    _binarize_to_size,
    _prepared,
    _silhouette_contour_image,
    build_line_drawing_prompt,
)

# Gemini 3.1 Flash Lite Image: the cheapest current Gemini image model on
# OpenRouter that accepts an input image (edit mode). ~$0.04 per output image.
DEFAULT_OPENROUTER_IMAGE_MODEL = "google/gemini-3.1-flash-lite-image"
DEFAULT_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

# Edit models need to be told what to do with the conditioning image; this is
# the same wrapper FluxKontextRectifier puts around build_line_drawing_prompt.
_EDIT_INSTRUCTION = "Turn this silhouette outline into a finished drawing, keeping exactly the same pose and outer shape: "


def _image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _decode_data_url(url: str) -> Image.Image:
    if not url.startswith("data:"):
        raise RuntimeError("OpenRouter image generation returned a non-inline image URL.")
    _, _, encoded = url.partition(",")
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")


def extract_openrouter_image(payload: dict[str, Any]) -> Image.Image:
    """First generated image of a chat-completions response.

    OpenRouter returns generated images in `message.images`; some providers
    put them in `message.content` as `image_url` parts instead."""
    if "error" in payload:
        raise RuntimeError(f"OpenRouter image generation failed: {payload['error']}")
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("OpenRouter image generation returned no choices.")
    message = choices[0].get("message") or {}
    parts: list[Any] = list(message.get("images") or [])
    if isinstance(message.get("content"), list):
        parts.extend(message["content"])
    for part in parts:
        if not isinstance(part, dict):
            continue
        image_url = part.get("image_url")
        url = image_url.get("url") if isinstance(image_url, dict) else image_url
        if isinstance(url, str) and url:
            return _decode_data_url(url)
    text = message.get("content") if isinstance(message.get("content"), str) else ""
    raise RuntimeError(f"OpenRouter image generation returned no image. Text: {str(text)[:160]!r}")


def downscale_line_art(image: Image.Image, size: tuple[int, int], line_threshold: int = 200) -> Image.Image:
    """Area-average the ~1024px model output down to the subject size, keep
    every pixel a line touched, then binarize with the shared helper.

    Hosted models draw 2-3px lines at 1024px; the nearest-neighbour resize in
    _binarize_to_size alone breaks those into dozens of fragments once the
    subject crop is a few hundred pixels, which the OpenCV prescreen rejects
    as too_many_components. FLUX emits 512px with heavier lines, so it does
    not need this."""
    gray = image.convert("L")
    # Widen lines by about half the shrink factor first so a thin diagonal
    # still covers most of each destination pixel it crosses.
    shrink = max(gray.width / max(size[0], 1), gray.height / max(size[1], 1))
    if shrink > 2.0:
        gray = gray.filter(ImageFilter.MinFilter(2 * int(round(shrink / 2.0)) + 1))
    gray = gray.resize(size, Image.Resampling.BOX)
    lines = gray.point(lambda value: 0 if value < int(line_threshold) else 255)
    return _binarize_to_size(lines, size)


class OpenRouterImageRectifier(OutlineRectifier):
    """Silhouette contour + shared prompt -> OpenRouter image model -> line art."""

    def _model_name(self) -> str:
        return str(
            self.settings.get("openrouter_image_model")
            or os.getenv("OPENROUTER_IMAGE_MODEL")
            or DEFAULT_OPENROUTER_IMAGE_MODEL
        )

    def _api_base(self) -> str:
        return str(
            self.settings.get("openrouter_api_base") or os.getenv("OPENROUTER_API_BASE") or DEFAULT_OPENROUTER_API_BASE
        )

    def backend_label(self) -> str:
        return f"openrouter: {self._model_name()}"

    def validate_ready(self) -> None:
        if not os.getenv("OPENROUTER_API_KEY"):
            raise RuntimeError("OpenRouterImageRectifier requires `OPENROUTER_API_KEY` (environment or .env).")

    def build_instruction(self, sample: dict[str, Any], params: dict[str, Any] | None = None) -> str:
        # A VLM-suggested prompt from the retry loop wins, as in FluxKontextRectifier.
        override = (params or {}).get("prompt")
        return str(override) if override else _EDIT_INSTRUCTION + build_line_drawing_prompt(sample)

    def _request_body(self, instruction: str, conditioning: Image.Image) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._model_name(),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": _image_to_data_url(conditioning)}},
                    ],
                }
            ],
            # Image-only models (FLUX, Seedream) reject "text"; Gemini needs both.
            "modalities": list(self.settings.get("openrouter_image_modalities", ["image", "text"])),
            "usage": {"include": True},
        }
        aspect_ratio = self.settings.get("openrouter_image_aspect_ratio", "1:1")
        if aspect_ratio:
            body["image_config"] = {"aspect_ratio": str(aspect_ratio)}
        return body

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        # The key is only ever placed in the request header, never logged or
        # copied into metadata or error messages.
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
            with request.urlopen(req, timeout=float(self.settings.get("openrouter_image_timeout_seconds", 120))) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:  # pragma: no cover - network path
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenRouter image generation failed ({exc.code}): {detail[:300] or exc.reason}") from exc
        except error.URLError as exc:  # pragma: no cover - network path
            raise RuntimeError(f"OpenRouter image generation failed: {exc.reason}") from exc

    def run(
        self,
        sample: dict[str, Any],
        subject_image: Image.Image,
        subject_mask: Image.Image | None,
        generated_path: Path | None,
        destination: Path,
        params: dict[str, Any],
    ) -> OutlineRectifierResult:
        self.validate_ready()
        size = int(self.settings.get("size", 512))
        silhouette = _silhouette_contour_image(subject_image, subject_mask, params, self.settings, size=size)
        instruction = self.build_instruction(sample, params)
        started = time.perf_counter()
        payload = self._post(self._request_body(instruction, silhouette))
        generated = extract_openrouter_image(payload)
        latency = round(time.perf_counter() - started, 3)
        # Keep the raw model output next to the binarized line art for debugging.
        raw_path = destination.with_name(destination.stem + "_raw.png")
        generated.save(_prepared(raw_path))
        line_threshold = int(self.settings.get("openrouter_line_threshold", 200))
        downscale_line_art(generated, subject_image.size, line_threshold).save(_prepared(destination))
        usage = payload.get("usage") or {}
        return OutlineRectifierResult(
            diagram_path=destination,
            metadata={
                "backend": "openrouter_image_rectifier",
                "model_backend": "openrouter",
                "model_name": self._model_name(),
                "prompt": instruction,
                # Strength has no remote equivalent; recorded so retry params stay traceable.
                "strength": float(params.get("strength", self.settings.get("strength", 0.85))),
                "raw_output_path": str(raw_path),
                "latency_s": latency,
                "cost_usd": usage.get("cost"),
            },
        )
