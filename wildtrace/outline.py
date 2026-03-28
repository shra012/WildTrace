from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from wildtrace.images import local_edge_outline


@dataclass(slots=True)
class OutlineBackendResult:
    outline_path: Path
    metadata: dict[str, Any]


class OutlineBackend:
    def __init__(self, name: str, settings: dict[str, Any]) -> None:
        self.name = name
        self.settings = settings

    def run(self, image: Image.Image, destination: Path) -> OutlineBackendResult:
        raise NotImplementedError


class LocalEdgesBackend(OutlineBackend):
    def run(self, image: Image.Image, destination: Path) -> OutlineBackendResult:
        threshold_percentile = int(self.settings.get("threshold_percentile", 75))
        outline = local_edge_outline(image, threshold_percentile=threshold_percentile)
        destination.parent.mkdir(parents=True, exist_ok=True)
        outline.save(destination)
        nonzero_ratio = sum(1 for value in outline.getdata() if value > 0) / float(outline.width * outline.height or 1)
        return OutlineBackendResult(
            outline_path=destination,
            metadata={
                "model_backend": "local",
                "model_name": self.name,
                "model_version_or_checkpoint": self.settings.get("model_version_or_checkpoint", "baseline-v1"),
                "inference_params": {"threshold_percentile": threshold_percentile},
                "inference_timestamp": datetime.now(UTC).isoformat(),
                "foreground_ratio": nonzero_ratio,
            },
        )


class ApiStubBackend(OutlineBackend):
    def run(self, image: Image.Image, destination: Path) -> OutlineBackendResult:
        raise RuntimeError("The api_stub backend is defined for schema compatibility only and is not implemented.")


def build_backend(name: str, settings: dict[str, Any]) -> OutlineBackend:
    if name == "local_edges":
        return LocalEdgesBackend(name, settings)
    if name == "api_stub":
        return ApiStubBackend(name, settings)
    raise ValueError(f"Unknown outline backend: {name}")
