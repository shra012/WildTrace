from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(slots=True)
class EnrichmentResult:
    subcategory: str
    tags: list[str]
    confidence: float
    metadata: dict[str, Any]


class EnrichmentBackend:
    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = settings

    def enrich(self, sample: dict[str, Any], image: Image.Image) -> EnrichmentResult:
        raise NotImplementedError


_DEFAULT_CANDIDATE_LABELS = {
    "bird": ["flamingo", "eagle", "owl", "parrot", "penguin", "bird"],
    "cat": ["domestic cat", "cat"],
    "dog": ["domestic dog", "dog"],
    "horse": ["horse"],
    "butterfly": ["butterfly"],
    "fish": ["fish"],
}

_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any, Any, Any]] = {}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _metadata_text(sample: dict[str, Any]) -> str:
    return " ".join(
        str(sample.get(key, "") or "")
        for key in ("title", "category", "label_name", "source_image_id", "original_url", "original_landing_url")
    ).lower()


def _heuristic_subcategory(category: str, sample: dict[str, Any]) -> str:
    text = _metadata_text(sample)
    if category.lower() == "bird":
        if "flamingo" in text:
            return "flamingo"
        if "eagle" in text:
            return "eagle"
        if "owl" in text:
            return "owl"
        return "bird"
    if category.lower() == "cat":
        return "domestic_cat"
    if category.lower() == "dog":
        return "domestic_dog"
    if category.lower() == "horse":
        return "horse"
    if category.lower() == "butterfly":
        return "butterfly"
    if category.lower() == "fish":
        return "fish"
    return _slug(category)


def _candidate_labels(sample: dict[str, Any], settings: dict[str, Any]) -> list[str]:
    category_key = str(sample["category"]).lower()
    configured = settings.get("candidate_labels", {}).get(category_key, [])
    labels = list(configured or _DEFAULT_CANDIDATE_LABELS.get(category_key, [sample["category"]]))
    text = _metadata_text(sample)
    for label in list(labels):
        if label.lower() in text:
            labels.insert(0, labels.pop(labels.index(label)))
            break
    return list(dict.fromkeys(labels))


def _base_tags(category: str, sample: dict[str, Any], image: Image.Image) -> set[str]:
    tags = {
        _slug(category),
        "drawable_subject",
        "segmented",
        "photo_source",
        "wide" if image.width >= image.height else "tall",
    }
    split = sample.get("split")
    if split:
        tags.add(str(split))
    angle_bucket = sample.get("angle_bucket")
    if angle_bucket:
        tags.add(str(angle_bucket))
    return tags


def _build_tags(
    category: str,
    sample: dict[str, Any],
    image: Image.Image,
    ranked_labels: list[str],
    source_tag: str,
) -> list[str]:
    tags = _base_tags(category, sample, image)
    tags.add(source_tag)
    tags.update(_slug(label) for label in ranked_labels if label)
    return sorted(tags)


class HeuristicEnrichmentBackend(EnrichmentBackend):
    def enrich(self, sample: dict[str, Any], image: Image.Image) -> EnrichmentResult:
        category = str(sample["category"])
        subcategory = _heuristic_subcategory(category, sample)
        tags = _build_tags(category, sample, image, [subcategory], "heuristic_enrichment")
        metadata = {
            "model_backend": "local",
            "model_name": self.settings.get("model_name", "heuristic_enrichment"),
            "model_id": self.settings.get("model_id", "heuristic_enrichment"),
        }
        return EnrichmentResult(
            subcategory=subcategory,
            tags=tags,
            confidence=float(self.settings.get("default_confidence", 0.72)),
            metadata=metadata,
        )


class BioCLIPHuggingFaceBackend(EnrichmentBackend):
    def _load_runtime(self) -> tuple[Any, Any, Any, Any]:
        repo_id = str(self.settings.get("model_id", "imageomics/bioclip"))
        device = str(self.settings.get("device", "cuda"))
        cache_key = (repo_id, device)
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        try:
            import open_clip
            import torch
        except ImportError as exc:  # pragma: no cover - depends on optional runtime deps
            raise RuntimeError(
                "BioCLIP enrichment requires `open-clip-torch` and `torch`. Run `uv sync` to install them."
            ) from exc

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "BioCLIP enrichment is configured for CUDA, but no GPU is available. "
                "Set `BIOCLIP_DEVICE` only to a valid CUDA device such as `cuda` or `cuda:0` on a GPU-enabled machine."
            )

        model, _, preprocess = open_clip.create_model_and_transforms(f"hf-hub:{repo_id}")
        tokenizer = open_clip.get_tokenizer(f"hf-hub:{repo_id}")
        model = model.to(device)
        model.eval()
        runtime = (model, preprocess, tokenizer, torch)
        _MODEL_CACHE[cache_key] = runtime
        return runtime

    def _predict_labels(self, sample: dict[str, Any], image: Image.Image) -> tuple[list[str], float]:
        model, preprocess, tokenizer, torch = self._load_runtime()
        candidate_labels = _candidate_labels(sample, self.settings)
        prompts = [str(self.settings.get("prompt_template", "a photo of a {}")).format(label) for label in candidate_labels]
        device = str(self.settings.get("device", "cuda"))

        with torch.no_grad():
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            text_tensor = tokenizer(prompts).to(device)
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]

        top_k = min(int(self.settings.get("top_k_tags", 3)), len(candidate_labels))
        scores, indices = similarities.topk(top_k)
        ranked_labels = [candidate_labels[int(index)] for index in indices.tolist()]
        confidence = float(scores[0].item()) if len(scores) else 0.0
        return ranked_labels, confidence

    def enrich(self, sample: dict[str, Any], image: Image.Image) -> EnrichmentResult:
        category = str(sample["category"])
        ranked_labels, confidence = self._predict_labels(sample, image)
        subcategory = _slug(ranked_labels[0]) if ranked_labels else _heuristic_subcategory(category, sample)
        tags = _build_tags(category, sample, image, ranked_labels, "bioclip_enrichment")
        metadata = {
            "model_backend": "huggingface",
            "model_name": self.settings.get("model_name", "imageomics/bioclip"),
            "model_id": self.settings.get("model_id", "imageomics/bioclip"),
            "cache_dir": str(Path(str(self.settings.get("cache_dir", "~/.cache/huggingface"))).expanduser()),
            "device": self.settings.get("device", "cuda"),
            "candidate_labels": _candidate_labels(sample, self.settings),
        }
        return EnrichmentResult(
            subcategory=subcategory,
            tags=tags,
            confidence=confidence,
            metadata=metadata,
        )


def build_enrichment_backend(config: dict[str, Any]) -> EnrichmentBackend:
    backend_name = config.get("backend", "bioclip_huggingface")
    if backend_name == "bioclip_huggingface":
        return BioCLIPHuggingFaceBackend(config)
    if backend_name == "heuristic_enrichment":
        return HeuristicEnrichmentBackend(config)
    raise ValueError(f"Unknown enrichment backend: {backend_name}")
