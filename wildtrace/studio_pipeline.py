"""WildTrace Studio backend for animal jobs.

Runs one photo through the same silver/gold stages the data pipeline uses
(silhouette conditioning -> outline rectifier -> OpenCV prescreen + semantic
validation with the LangGraph retry loop -> stroke extraction), writing every
artifact into a per-job directory instead of the medallion manifests. Nothing
here changes the pipeline; it calls the stage helpers as they are.

The image backend follows the pipeline default (FLUX-schnell on the GPU) when
the GPU is free, and falls back to an OpenRouter image model otherwise, so a
job never competes with another GPU workload.
"""

from __future__ import annotations

import importlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from importlib import util as importlib_util
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from wildtrace.bronze_stage import _classify_angle
from wildtrace.config import load_runtime_config
from wildtrace.diagram import (
    FluxSilhouetteRectifier,
    OutlineRectifier,
    OutlineRectifierResult,
    SemanticValidator,
    _silhouette_contour_image,
    build_outline_validator,
    next_generator_params,
    run_drawn_diagram_pass,
    run_langgraph_validation_loop,
    validate_with_opencv,
)
from wildtrace.face_lineart import estimate_draw_seconds
from wildtrace.gold_stage import _build_trajectory_payload
from wildtrace.images import (
    crop_to_mask_bbox,
    isolate_subject,
    normalize_mask,
    open_image,
    resize_longest_side,
    save_image,
    write_svg,
)
from wildtrace.io_utils import read_ndjson, resolve_repo_path
from wildtrace.openrouter_image import OpenRouterImageRectifier
from wildtrace.stage_io import bronze_accepted_manifest_path, silver_subjects_manifest_path
from wildtrace.viewpoint import build_viewpoint_backend, extract_view_features

try:
    import cv2
except ImportError:  # pragma: no cover - opencv is a core dependency.
    cv2 = None


# Another long job owns the GPU until its queue status file reports QUEUE_DONE.
XRAYGPT_QUEUE_STATUS = Path.home() / "Workspace/XRayGpt/experiments/experiment2/logs/phase7_test_queue.status"
GPU_BUSY_UTILIZATION_PCT = 20.0
GPU_BUSY_MEMORY_FRACTION = 0.25
# Optional manual override of select_image_backend: "flux" or "openrouter".
IMAGE_BACKEND_ENV = "WILDTRACE_STUDIO_IMAGE_BACKEND"

# Trajectories already drawn or held out; new jobs should not reuse their photos.
USED_TRAJECTORY_DIRS = ("outputs/trajectories", "isaac-sim/data/act_test")

MAGIC_TOUCH_URL = (
    "https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/latest/magic_touch.tflite"
)
MEDIAPIPE_CACHE_DIR = Path(os.getenv("WILDTRACE_MEDIAPIPE_DIR", str(Path.home() / ".cache/wildtrace/mediapipe")))


@dataclass
class StageEvent:
    name: str  # "segment", "outline", "validate", "extract" (and "pick_photo" for text jobs)
    label: str  # human readable
    status: str  # "running" | "done" | "failed" | "skipped"
    backend: str | None = None  # e.g. "flux-schnell (GPU)", "openrouter: <model>", "opencv", "<judge model>"
    detail: str | None = None
    artifact_path: Path | None = None  # image produced by the stage, if any


STAGE_LABELS = {
    "pick_photo": "Pick an unused photo",
    "segment": "Segment the animal",
    "outline": "Generate line art",
    "validate": "Validate the drawing",
    "extract": "Extract strokes",
}


# --------------------------------------------------------------------------- GPU / backend


def _query_nvidia_smi() -> list[tuple[float, float, float]] | None:
    """(utilization %, memory used MiB, memory total MiB) per GPU, or None
    when nvidia-smi is missing or fails. Read-only, no CUDA context."""
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    gpus: list[tuple[float, float, float]] = []
    for line in completed.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        try:
            gpus.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except (IndexError, ValueError):
            continue
    return gpus or None


def _gpu_reservation_active(status_path: Path | None = None) -> bool:
    status_path = status_path or XRAYGPT_QUEUE_STATUS
    if not status_path.exists():
        return False
    try:
        return "QUEUE_DONE" not in status_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return True


def gpu_status() -> dict:
    """{"busy", "utilization", "note"}; utilization is the busiest GPU's
    percent. Busy means: do not start GPU work (high load, high memory, no
    GPU at all, or the XRayGPT reservation is still running)."""
    gpus = _query_nvidia_smi()
    reserved = _gpu_reservation_active()
    if not gpus:
        note = "no NVIDIA GPU visible"
        return {"busy": True, "utilization": 0.0, "note": note + ("; XRayGPT reservation active" if reserved else "")}
    utilization, used, total = max(gpus, key=lambda gpu: (gpu[0], gpu[1]))
    memory_fraction = used / total if total else 1.0
    reasons: list[str] = []
    if reserved:
        reasons.append("reserved by XRayGPT evaluation")
    if utilization >= GPU_BUSY_UTILIZATION_PCT:
        reasons.append(f"utilization {utilization:.0f}%")
    if memory_fraction >= GPU_BUSY_MEMORY_FRACTION:
        reasons.append(f"memory {used / 1024:.1f}/{total / 1024:.1f} GiB in use")
    if reasons:
        note = "GPU busy: " + ", ".join(reasons)
    else:
        note = f"GPU free: utilization {utilization:.0f}%, memory {used / 1024:.1f}/{total / 1024:.1f} GiB"
    return {"busy": bool(reasons), "utilization": float(utilization), "note": note}


def _flux_dependencies_present() -> bool:
    # find_spec only locates the packages; nothing is imported or loaded.
    return all(importlib_util.find_spec(name) is not None for name in ("torch", "diffusers", "gguf"))


def select_image_backend(repo_root: Path, config_dir: Path) -> str:
    """"flux" (the normal pipeline rectifier) when the GPU is free, else "openrouter"."""
    # Loads .env so the override and OPENROUTER_API_KEY are visible to the job.
    load_runtime_config(Path(repo_root), Path(config_dir) if config_dir else None)
    override = os.getenv(IMAGE_BACKEND_ENV, "").strip().lower()
    if override in ("flux", "openrouter"):
        return override
    if gpu_status()["busy"] or not _flux_dependencies_present():
        return "openrouter"
    return "flux"


# --------------------------------------------------------------------------- photo picking


def used_sample_ids(repo_root: Path, dirs: tuple[str, ...] = USED_TRAJECTORY_DIRS) -> set[str]:
    """sample_ids that already have a trajectory (file stem of <dir>/<Category>/<id>.json)."""
    used: set[str] = set()
    for relative in dirs:
        root = Path(repo_root) / relative
        if root.is_dir():
            used.update(path.stem for path in root.glob("*/*.json"))
    return used


_ANGLE_HINTS = {
    "left_profile": ("left profile", "left side", "facing left", "left"),
    "right_profile": ("right profile", "right side", "facing right", "right"),
    "front_left_3q": ("front left", "three quarter left", "3/4 left", "3q left"),
    "front_right_3q": ("front right", "three quarter right", "3/4 right", "3q right"),
    "front": ("front", "facing me", "facing the camera", "head on", "head-on"),
}
_POSE_HINTS = ("sitting", "standing", "walking", "lying", "swimming", "flying")


def _hint_pose(sample: dict[str, Any]) -> str:
    """Pose word build_line_drawing_prompt would derive from the view features."""
    features = sample.get("view_features") or {}
    symmetry = float(features.get("symmetry_score", 0.0))
    bottom_mass = float(features.get("bottom_mass_ratio", 0.5))
    direction = abs(float(features.get("direction_score", 0.0)))
    if symmetry > 0.35 and bottom_mass > 0.6:
        return "sitting"
    if symmetry <= 0.35 and direction > 0.3:
        return "walking"
    return "standing"


def _hint_angle(text: str) -> str | None:
    # Longest phrases first so "front left" is not read as plain "front".
    phrases = sorted(((phrase, bucket) for bucket, items in _ANGLE_HINTS.items() for phrase in items), key=lambda item: -len(item[0]))
    for phrase, bucket in phrases:
        if f" {phrase} " in text:
            return bucket
    return None


def _hint_score(sample: dict[str, Any], hint: str | None) -> int:
    """2 for a matching angle bucket, 1 for a matching pose; 0 without a hint."""
    if not hint:
        return 0
    text = " " + re.sub(r"[^a-z0-9/]+", " ", hint.lower()) + " "
    score = 0
    angle = _hint_angle(text)
    if angle and sample.get("angle_bucket") == angle:
        score += 2
    pose = _hint_pose(sample)
    if any(word in text for word in _POSE_HINTS):
        wanted = {word for word in _POSE_HINTS if word in text}
        # Walking and standing look alike in mask features; treat them as one.
        if pose in wanted or (pose in ("walking", "standing") and wanted & {"walking", "standing"}):
            score += 1
    return score


def pick_unused_photo(
    category: str,
    repo_root: Path,
    hint: str | None = None,
    *,
    config_dir: Path | None = None,
    exclude: set[str] | None = None,
    rng: random.Random | None = None,
) -> dict:
    """An OpenImages subject of `category` (with its mask) that no trajectory uses yet.

    Accepted silver subjects come first (normalized image, mask, crop and
    view features ready); accepted bronze rows are the fallback. `hint`
    (e.g. "sitting", "left profile") steers by angle bucket and pose; ties are
    broken at random. Returns the manifest record plus `resolved_image_path`,
    `resolved_mask_path` and `source_manifest`. Raises LookupError when none
    is left."""
    repo_root = Path(repo_root)
    runtime = load_runtime_config(repo_root, Path(config_dir) if config_dir else None)
    used = used_sample_ids(repo_root) | set(exclude or ())
    wanted = category.strip().lower()
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    sources = (
        ("silver_subjects", silver_subjects_manifest_path(repo_root, runtime)),
        ("bronze_accepted", bronze_accepted_manifest_path(repo_root, runtime)),
    )
    for source_name, manifest in sources:
        for row in read_ndjson(manifest):
            sample_id = row.get("sample_id")
            if not sample_id or sample_id in used or sample_id in seen:
                continue
            if str(row.get("category", "")).lower() != wanted or not row.get("mask_path"):
                continue
            if source_name == "silver_subjects" and row.get("segmentation_status") != "accepted":
                continue
            if source_name == "bronze_accepted" and row.get("curation_status", "accepted") != "accepted":
                continue
            image_path = resolve_repo_path(repo_root, row["image_path"])
            mask_path = resolve_repo_path(repo_root, row["mask_path"])
            if not image_path.exists() or not mask_path.exists():
                continue
            seen.add(sample_id)
            candidates.append(
                {
                    **row,
                    "source_manifest": source_name,
                    "resolved_image_path": str(image_path),
                    "resolved_mask_path": str(mask_path),
                }
            )
    if not candidates:
        raise LookupError(f"No unused {category} photo with a mask is left in the bronze/silver manifests.")
    scored = [(_hint_score(row, hint), row["source_manifest"] == "silver_subjects", row) for row in candidates]
    best = max((score, silver) for score, silver, _ in scored)
    pool = [row for score, silver, row in scored if (score, silver) == best]
    return (rng or random.Random()).choice(pool)


# --------------------------------------------------------------------------- segmentation


def _magic_touch_model() -> Path:
    path = MEDIAPIPE_CACHE_DIR / "magic_touch.tflite"
    if not path.exists():
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(MAGIC_TOUCH_URL, partial)
        partial.replace(path)
    return path


def _largest_component(binary: np.ndarray) -> np.ndarray:
    if cv2 is None or not binary.any():
        return binary
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    if count <= 2:
        return binary.astype(bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def _grabcut_mask(rgb: np.ndarray, seed: np.ndarray | None = None, work_size: int = 384) -> np.ndarray:
    """OpenCV GrabCut on a downscaled copy (CPU, a second or two). Seeded from
    `seed` when given, else from a rectangle just inside the frame."""
    height, width = rgb.shape[:2]
    scale = min(1.0, work_size / max(height, width))
    small = cv2.resize(rgb, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    mask = np.zeros(small.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    if seed is not None and seed.any():
        seed_small = cv2.resize(seed.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask[:] = cv2.GC_PR_BGD
        mask[seed_small > 0] = cv2.GC_PR_FGD
        mask[cv2.erode(seed_small, np.ones((7, 7), np.uint8)) > 0] = cv2.GC_FGD
        cv2.grabCut(bgr, mask, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    else:
        margin = max(1, int(0.04 * min(small.shape[:2])))
        rect = (margin, margin, small.shape[1] - 2 * margin, small.shape[0] - 2 * margin)
        cv2.grabCut(bgr, mask, rect, bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
    foreground = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    return cv2.resize(foreground, (width, height), interpolation=cv2.INTER_NEAREST) > 0


# Prompt points for MagicTouch: the centre plus a ring around it, since an
# off-centre animal or a background patch under the centre is common.
_PROMPT_POINTS = ((0.5, 0.5), (0.4, 0.5), (0.6, 0.5), (0.5, 0.4), (0.5, 0.6), (0.35, 0.4), (0.65, 0.4), (0.35, 0.6), (0.65, 0.6))


def _magic_touch_masks(rgb: np.ndarray, points: tuple[tuple[float, float], ...] = _PROMPT_POINTS) -> list[np.ndarray]:
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.components.containers.keypoint import NormalizedKeypoint
    from mediapipe.tasks.python.core.base_options import BaseOptions

    options = vision.InteractiveSegmenterLegacyOptions(
        base_options=BaseOptions(model_asset_path=str(_magic_touch_model())),
        output_confidence_masks=True,
        output_category_mask=False,
    )
    roi_cls = vision.InteractiveSegmenterLegacyRegionOfInterest
    segmenter = vision.InteractiveSegmenterLegacy.create_from_options(options)
    masks: list[np.ndarray] = []
    try:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        for x, y in points:
            result = segmenter.segment(image, roi_cls(format=roi_cls.Format.KEYPOINT, keypoint=NormalizedKeypoint(x, y)))
            masks.append(np.squeeze(result.confidence_masks[0].numpy_view()) > 0.5)
    finally:
        segmenter.close()
    return masks


def _plausible_subject(mask: np.ndarray) -> bool:
    """Not empty, not the whole frame, and not mostly hugging the border."""
    coverage = float(mask.mean())
    if coverage < 0.01 or coverage > 0.85:
        return False
    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    return float(border.mean()) < 0.5


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum()) / float(max((a | b).sum(), 1))


def consensus_mask(masks: list[np.ndarray]) -> np.ndarray | None:
    """The plausible mask that agrees most (mean IoU) with the other plausible ones."""
    plausible = [mask for mask in masks if _plausible_subject(mask)]
    if not plausible:
        return None
    return max(plausible, key=lambda mask: np.mean([_iou(mask, other) for other in plausible]))


def segment_subject(image: Image.Image, method: str = "auto") -> tuple[Image.Image, str]:
    """CPU subject mask for an uploaded animal photo -> (L mask 0/255, backend label).

    MediaPipe's MagicTouch interactive segmenter (6 MB tflite, CPU, ~0.15 s
    per prompt) is prompted at nine points around the centre and the mask the
    prompts agree on is kept. On 36 Open Images silver photos this scored a
    mean IoU of 0.53 against the annotated masks, versus 0.18 for a single
    centre prompt and 0.38 for GrabCut. GrabCut is the fallback when MediaPipe
    is missing or finds nothing plausible (`method="grabcut"` forces it)."""
    if cv2 is None:  # pragma: no cover - opencv is a core dependency.
        raise RuntimeError("Subject segmentation requires opencv.")
    rgb = np.asarray(image.convert("RGB"))
    chosen: np.ndarray | None = None
    label = "opencv grabcut (CPU)"
    if method in ("auto", "mediapipe"):
        try:
            chosen = consensus_mask(_magic_touch_masks(rgb))
            if chosen is not None:
                label = "mediapipe magic-touch (CPU)"
        except Exception:
            if method == "mediapipe":
                raise
    if chosen is None:
        chosen = _grabcut_mask(rgb)
    chosen = _largest_component(chosen)
    mask = Image.fromarray(np.where(chosen, 255, 0).astype(np.uint8), mode="L")
    return mask, label


# --------------------------------------------------------------------------- job helpers


class _EventingGenerator:
    """Pass-through rectifier wrapper that reports each retry attempt."""

    def __init__(self, inner: OutlineRectifier, on_attempt: Callable[[int], None]) -> None:
        self.inner = inner
        self.settings = inner.settings
        self.attempts = 0
        self._on_attempt = on_attempt

    def prepare_conditioning_image(self, crop_image: Image.Image, isolated_image: Image.Image | None = None) -> Image.Image:
        return self.inner.prepare_conditioning_image(crop_image, isolated_image)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        self.attempts += 1
        self._on_attempt(self.attempts)
        return self.inner.run(*args, **kwargs)


class _ProcessPerAttemptRectifier:
    """Runs each outline attempt of a rectifier in a fresh Python process.

    A cached FLUX pipeline with model CPU offload keeps growing in host RAM from
    one call to the next (see FluxSilhouetteRectifier.run), so the third
    attempt of a validation retry loop gets OOM-killed on a 24 GB machine. One
    process per attempt costs a model load (~15 s) but every attempt starts
    from the footprint of the first. The attempt itself is the inner
    rectifier's own run(), unchanged.
    """

    def __init__(self, inner: OutlineRectifier) -> None:
        self.inner = inner
        self.settings = inner.settings

    def validate_ready(self) -> None:
        self.inner.validate_ready()

    def prepare_conditioning_image(self, crop_image: Image.Image, isolated_image: Image.Image | None = None) -> Image.Image:
        return self.inner.prepare_conditioning_image(crop_image, isolated_image)

    def run(self, sample: dict[str, Any], subject_image: Image.Image, subject_mask: Image.Image | None,
            generated_path: Path | None, destination: Path, params: dict[str, Any]) -> OutlineRectifierResult:
        work = Path(destination).parent / f".{Path(destination).stem}_attempt"
        work.mkdir(parents=True, exist_ok=True)
        subject_image.save(work / "subject.png")
        if subject_mask is not None:
            subject_mask.save(work / "mask.png")
        request = {
            "rectifier": f"{type(self.inner).__module__}:{type(self.inner).__qualname__}",
            "settings": self.settings,
            "sample": sample,
            "generated_path": str(generated_path) if generated_path else None,
            "destination": str(destination),
            "params": params,
            "work": str(work),
        }
        (work / "request.json").write_text(json.dumps(request, default=str), encoding="utf-8")
        completed = subprocess.run([sys.executable, "-m", "wildtrace.studio_pipeline", "--outline-attempt", str(work / "request.json")],
                                   cwd=Path(__file__).resolve().parents[1], stdout=subprocess.DEVNULL)
        result_path = work / "result.json"
        if completed.returncode != 0 or not result_path.exists():
            if completed.returncode in (-9, 137):
                raise RuntimeError("The outline attempt was stopped by the system for running out of memory "
                                   "(FLUX-schnell needs about 20 GB of free RAM).")
            raise RuntimeError(f"The outline attempt failed (exit code {completed.returncode}).")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        shutil.rmtree(work, ignore_errors=True)
        return OutlineRectifierResult(diagram_path=Path(result["diagram_path"]), metadata=result["metadata"])


def _outline_attempt_main(request_path: Path) -> int:
    """Child side of _ProcessPerAttemptRectifier: one rectifier.run() call."""
    request = json.loads(request_path.read_text(encoding="utf-8"))
    module_name, class_name = request["rectifier"].split(":")
    rectifier = getattr(importlib.import_module(module_name), class_name)(request["settings"])
    work = Path(request["work"])
    mask_path = work / "mask.png"
    result = rectifier.run(
        request["sample"],
        Image.open(work / "subject.png"),
        Image.open(mask_path) if mask_path.exists() else None,
        Path(request["generated_path"]) if request["generated_path"] else None,
        Path(request["destination"]),
        request["params"],
    )
    (work / "result.json").write_text(json.dumps({"diagram_path": str(result.diagram_path), "metadata": result.metadata},
                                                 default=str), encoding="utf-8")
    return 0


RATE_LIMIT_BACKOFF_S = (3.0, 8.0, 20.0)


def _transient(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in ("429", "rate-limit", "rate limit", "temporarily", " 502", " 503", "timed out"))


class _TrackingValidator:
    """Pass-through semantic validator that remembers the best-scoring attempt,
    scored like the pipeline (0.5 * OpenCV + 0.5 * semantic)."""

    def __init__(self, inner: SemanticValidator) -> None:
        self.inner = inner
        self.settings = inner.settings
        self.best: dict[str, Any] | None = None

    def validate_ready(self) -> None:
        self.inner.validate_ready()

    def validate(self, sample: dict[str, Any], diagram_path: Path, opencv_result: Any) -> Any:
        result = self._validate_with_retries(sample, diagram_path, opencv_result)
        score = round(opencv_result.score * 0.5 + result.score * 0.5, 4)
        accepted = bool(opencv_result.passed and result.passed)
        if self.best is None or (accepted, score) > (self.best["accepted"], self.best["score"]):
            self.best = {
                "diagram_path": str(diagram_path),
                "score": score,
                "accepted": accepted,
                "opencv_flags": list(opencv_result.flags),
                "semantic_reason": result.reason,
            }
        return result

    def suggest_params(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.inner.suggest_params(*args, **kwargs)

    def _validate_with_retries(self, sample: dict[str, Any], diagram_path: Path, opencv_result: Any) -> Any:
        # Shared OpenRouter judge models get rate limited upstream (429) for a few seconds at a
        # time; one such blip should not throw away a generated drawing.
        for delay_s in RATE_LIMIT_BACKOFF_S:
            try:
                return self.inner.validate(sample, diagram_path, opencv_result)
            except RuntimeError as exc:
                if not _transient(exc):
                    raise
                time.sleep(delay_s)
        return self.inner.validate(sample, diagram_path, opencv_result)


def _rectifier_settings(runtime: dict[str, Any], backend: str) -> dict[str, Any]:
    settings = dict(runtime["models"]["outline_rectifier"])
    settings["backend"] = "flux_silhouette_rectifier" if backend == "flux" else "openrouter_image_rectifier"
    if backend == "flux":
        # The studio always uses the production FLUX path, whatever .env points the batch backend at.
        settings["conditioning_mode"] = "isolated_only"
    return settings


def build_studio_rectifier(runtime: dict[str, Any], backend: str) -> OutlineRectifier:
    settings = _rectifier_settings(runtime, backend)
    rectifier: OutlineRectifier = FluxSilhouetteRectifier(settings) if backend == "flux" else OpenRouterImageRectifier(settings)
    rectifier.validate_ready()
    # FLUX attempts run one per process; the OpenRouter stand-in holds no model, so it stays in-process.
    return _ProcessPerAttemptRectifier(rectifier) if backend == "flux" else rectifier


def build_studio_validator(runtime: dict[str, Any], backend: str) -> SemanticValidator:
    """The pipeline's configured semantic validator, except that a GPU-free job
    never falls through to the local Ollama model."""
    settings = dict(runtime["models"]["outline_validator"])
    name = settings.get("backend", "auto")
    if name == "auto":
        name = "anthropic_semantic_validator" if os.getenv("ANTHROPIC_API_KEY") else "ollama_semantic_validator"
    if name == "ollama_semantic_validator" and backend != "flux":
        name = "openrouter_semantic_validator" if os.getenv("OPENROUTER_API_KEY") else "anthropic_semantic_validator"
    settings["backend"] = name
    return build_outline_validator(settings)


def _rectifier_label(rectifier: Any, backend: str) -> str:
    if backend == "flux":
        return "flux-schnell (GPU)"
    label = getattr(rectifier, "backend_label", None)
    return label() if callable(label) else f"openrouter: {type(rectifier).__name__}"


def _validator_label(validator: Any) -> str:
    settings = getattr(validator, "settings", {}) or {}
    model = getattr(validator, "_model_name", None)
    name = model() if callable(model) else settings.get("model_name") or type(validator).__name__
    backend = settings.get("backend", "")
    prefix = "openrouter: " if "openrouter" in str(backend) else ""
    return f"opencv + {prefix}{name}"


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "upload"


def _prepare_upload_sample(
    *,
    out_dir: Path,
    runtime: dict[str, Any],
    category: str,
    image_path: Path,
    mask_path: Path | None,
    sample: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Normalize + isolate + crop like silver normalize/enrich, into out_dir.
    Returns the silver-shaped sample and the written file paths."""
    image = resize_longest_side(open_image(image_path), int(runtime["quality"]["max_resize"]))
    mask = normalize_mask(Image.open(mask_path), image.size)
    isolated = isolate_subject(image, mask)
    crop, bbox = crop_to_mask_bbox(isolated, mask)
    paths = {
        "source": out_dir / "source.png",
        "mask": out_dir / "mask.png",
        "isolated": out_dir / "isolated.png",
        "crop": out_dir / "crop.png",
    }
    save_image(paths["source"], image)
    save_image(paths["mask"], mask)
    save_image(paths["isolated"], isolated)
    save_image(paths["crop"], crop)

    base = dict(sample or {})
    view_features = extract_view_features(mask)
    if base.get("angle_bucket") and base.get("view_features"):
        angle_bucket = base["angle_bucket"]
        view_features = base["view_features"]
    else:
        angle_bucket, _, _, _ = _classify_angle(
            image, paths["mask"], runtime["viewpoints"], build_viewpoint_backend(runtime["viewpoints"])
        )
    sample_id = str(base.get("sample_id") or _safe_id(Path(image_path).stem))
    prepared = {
        **base,
        "sample_id": sample_id,
        "category": base.get("category") or category,
        "subcategory": base.get("subcategory") or category.lower(),
        "angle_bucket": angle_bucket,
        "view_features": view_features,
        "crop_bbox": {"left": bbox[0], "top": bbox[1], "right": bbox[2], "bottom": bbox[3]},
    }
    return prepared, paths


def _emit_stage(
    emit: Callable[[StageEvent], None],
    name: str,
    status: str,
    backend: str | None = None,
    detail: str | None = None,
    artifact_path: Path | None = None,
) -> None:
    emit(StageEvent(name=name, label=STAGE_LABELS[name], status=status, backend=backend, detail=detail, artifact_path=artifact_path))


def _sample_file(repo_root: Path, sample: dict[str, Any], key: str) -> Path:
    resolved = sample.get(f"resolved_{key}")
    return Path(resolved).resolve() if resolved else resolve_repo_path(repo_root, sample[key]).resolve()


# --------------------------------------------------------------------------- main entry


def generate_animal_trajectory(
    *,
    out_dir: Path,
    repo_root: Path,
    config_dir: Path,
    category: str,
    image_path: Path | None = None,
    mask_path: Path | None = None,
    sample: dict | None = None,
    emit: Callable[[StageEvent], None] = lambda e: None,
    backend: str | None = None,
    hint: str | None = None,
    max_attempts: int | None = None,
    require_valid: bool = False,
    rectifier: OutlineRectifier | None = None,
    validator: SemanticValidator | None = None,
    segment_method: str = "auto",
) -> dict:
    """Photo -> validated line art -> robot trajectory, all artifacts in out_dir.

    Inputs, in order of precedence: an uploaded `image_path` (mask built on
    CPU unless `mask_path` is given), a picked manifest `sample` (from
    pick_unused_photo), or neither, in which case an unused photo of
    `category` is picked here (steered by `hint`).

    `max_attempts` counts every generation including the first (pipeline
    default: outline_rectifier.max_attempts or 5). When no attempt passes
    validation the best-scoring one is still extracted and the validate stage
    is reported as failed, unless `require_valid` is set, which raises
    instead. `rectifier` / `validator` override the backends (tests)."""
    out_dir = Path(out_dir)
    repo_root = Path(repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime = load_runtime_config(repo_root, Path(config_dir) if config_dir else None)
    backend = backend or select_image_backend(repo_root, config_dir)
    if backend not in ("flux", "openrouter"):
        raise ValueError(f"Unknown studio image backend: {backend}")

    current = "pick_photo"
    try:
        # --- pick_photo (text jobs) ----------------------------------------
        if image_path is None and sample is None:
            _emit_stage(emit, "pick_photo", "running", "bronze/silver manifests", f"hint: {hint}" if hint else None)
            sample = pick_unused_photo(category, repo_root, hint, config_dir=config_dir)
            _emit_stage(
                emit, "pick_photo", "done", "bronze/silver manifests",
                f"{sample['sample_id']} ({sample.get('angle_bucket', 'unknown')}, {sample['source_manifest']})",
                Path(sample["resolved_image_path"]),
            )
        # A sample passed in was picked (and its pick_photo stage reported) by the caller.

        # --- segment ---------------------------------------------------------
        current = "segment"
        sample_image = _sample_file(repo_root, sample, "image_path") if sample else None
        if sample is not None and mask_path is None and (image_path is None or Path(image_path).resolve() == sample_image):
            # A picked manifest sample keeps its Open Images mask.
            image_path = None
        if image_path is not None:
            source = Path(image_path)
            if mask_path is not None:
                seg_backend, seg_detail = "provided mask", "using the mask supplied with the upload"
            else:
                seg_backend, seg_detail = "cpu segmentation", None
        else:
            source = sample_image
            mask_path = _sample_file(repo_root, sample, "mask_path")
            seg_backend, seg_detail = "OpenImages mask", "segmentation mask from the Open Images annotations"
        _emit_stage(emit, "segment", "running", seg_backend, seg_detail)
        if mask_path is None:
            mask_image, seg_backend = segment_subject(open_image(source), method=segment_method)
            mask_path = out_dir / "mask_raw.png"
            save_image(mask_path, mask_image)
        prepared, paths = _prepare_upload_sample(
            out_dir=out_dir, runtime=runtime, category=category, image_path=source, mask_path=mask_path, sample=sample
        )
        coverage = float((np.asarray(Image.open(paths["mask"])) > 0).mean())
        if coverage < 0.005:
            raise RuntimeError("No subject found in the photo (empty mask).")
        _emit_stage(
            emit, "segment", "done", seg_backend,
            f"mask covers {coverage:.0%} of the photo; view {prepared['angle_bucket'].replace('_', ' ')}",
            paths["isolated"],
        )

        # --- outline ---------------------------------------------------------
        current = "outline"
        diagram_generator = rectifier or build_studio_rectifier(runtime, backend)
        outline_backend = _rectifier_label(diagram_generator, backend)
        _emit_stage(emit, "outline", "running", outline_backend, "silhouette contour + line-drawing prompt")
        crop_image = open_image(paths["crop"])
        isolated_image = open_image(paths["isolated"])
        subject_image = diagram_generator.prepare_conditioning_image(crop_image, isolated_image)
        bbox = prepared["crop_bbox"]
        subject_mask = Image.open(paths["mask"]).crop(
            (int(bbox["left"]), int(bbox["top"]), int(bbox["right"]), int(bbox["bottom"]))
        ).resize(subject_image.size, Image.Resampling.NEAREST)
        params = dict(runtime["models"]["outline_rectifier"].get("default_params", {}))
        conditioning_path = out_dir / "conditioning.png"
        # Same helper and arguments every rectifier uses for its input.
        _silhouette_contour_image(
            subject_image, subject_mask, params, diagram_generator.settings, size=int(diagram_generator.settings.get("size", 512))
        ).save(conditioning_path)
        diagram_root = out_dir / "diagrams"
        first_destination = diagram_root / prepared["category"] / f"{prepared['sample_id']}_attempt01.png"
        initial = run_drawn_diagram_pass(prepared, subject_image, subject_mask, first_destination, params, diagram_generator)
        _emit_stage(emit, "outline", "done", outline_backend, "attempt 1", initial.diagram_path)

        # --- validate --------------------------------------------------------
        current = "validate"
        judge = _TrackingValidator(validator or build_studio_validator(runtime, backend))
        validate_backend = _validator_label(judge.inner)
        _emit_stage(emit, "validate", "running", validate_backend, "OpenCV prescreen + semantic check")
        opencv_settings = runtime["models"]["opencv_prescreen"]
        opencv_result = validate_with_opencv(initial.diagram_path, opencv_settings)
        semantic_result = judge.validate(prepared, initial.diagram_path, opencv_result)
        accepted = opencv_result.passed and semantic_result.passed
        total_attempts = int(max_attempts or runtime["models"]["outline_rectifier"].get("max_attempts", 5))
        attempts_used = 1
        if not accepted and total_attempts > 1:
            retry_generator = _EventingGenerator(
                diagram_generator,
                lambda n: _emit_stage(emit, "validate", "running", validate_backend,
                                      f"retry {n} of {total_attempts - 1}: regenerating with {outline_backend}"),
            )
            final_record = run_langgraph_validation_loop(
                sample=prepared,
                subject_path=paths["crop"],
                conditioning_path=paths["isolated"],
                subject_mask_path=paths["mask"],
                diagram_root=diagram_root,
                diagram_generator=retry_generator,
                outline_validator=judge,
                opencv_settings=opencv_settings,
                initial_params=next_generator_params(params, opencv_result.flags, semantic_result.passed),
                max_attempts=total_attempts - 1,
                attempt_offset=1,
            )
            attempts_used = 1 + retry_generator.attempts
            accepted = final_record["diagram_validation_status"] == "accepted"
        best = judge.best or {}
        if accepted and best.get("accepted"):
            final_diagram = Path(best["diagram_path"])
        else:
            final_diagram = Path(best.get("diagram_path") or initial.diagram_path)
        validation = {
            "accepted": bool(accepted),
            "score": best.get("score"),
            "opencv_flags": best.get("opencv_flags", list(opencv_result.flags)),
            "reason": best.get("semantic_reason", semantic_result.reason),
            "attempts": attempts_used,
            "diagram_path": str(final_diagram),
        }
        detail = f"{'passed' if accepted else 'no attempt passed'} after {attempts_used} attempt(s); score {validation['score']}"
        if validation["opencv_flags"]:
            detail += "; flags: " + ", ".join(validation["opencv_flags"])
        if not accepted:
            _emit_stage(emit, "validate", "failed", validate_backend, detail + f"; {validation['reason']}", final_diagram)
            if require_valid:
                raise RuntimeError("No generated line art passed validation.")
        else:
            _emit_stage(emit, "validate", "done", validate_backend, detail, final_diagram)

        # --- extract ---------------------------------------------------------
        current = "extract"
        _emit_stage(emit, "extract", "running", "opencv contours", "gold-stage stroke sampling")
        export = runtime["export"]
        lineart_path = out_dir / "lineart.png"
        shutil.copy2(final_diagram, lineart_path)
        diagram = Image.open(lineart_path).convert("L")
        trajectory_info = _build_trajectory_payload(diagram, export)
        payload = trajectory_info["payload"]
        if not payload["strokes"]:
            raise RuntimeError("The line art produced no drawable strokes.")
        svg_path = out_dir / "drawing.svg"
        trajectory_path = out_dir / "trajectory.json"
        write_svg(
            svg_path,
            trajectory_info["strokes"],
            width=diagram.width,
            height=diagram.height,
            stroke_width=int(export["svg_stroke_width"]),
            stroke_color=str(export["svg_stroke_color"]),
        )
        trajectory_path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        stats = {
            "stroke_count": int(payload["stroke_count"]),
            "point_count": int(payload["point_count"]),
            "est_draw_seconds": round(float(estimate_draw_seconds(payload)), 1),
        }
        _emit_stage(
            emit, "extract", "done", "opencv contours",
            f"{stats['stroke_count']} strokes, {stats['point_count']} points, ~{stats['est_draw_seconds']:.0f} s to draw",
            lineart_path,
        )
    except Exception as exc:
        _emit_stage(emit, current, "failed", None, str(exc)[:300])
        raise

    result = {
        "source_path": str(paths["source"]),
        "mask_path": str(paths["mask"]),
        "conditioning_path": str(conditioning_path),
        "lineart_path": str(lineart_path),
        "svg_path": str(svg_path),
        "trajectory_path": str(trajectory_path),
        "stats": stats,
        "backend": backend,
        "sample": {
            key: value
            for key, value in prepared.items()
            if key in ("sample_id", "category", "subcategory", "angle_bucket", "crop_bbox", "source_manifest")
        },
        "validation": validation,
        "outline_metadata": {key: value for key, value in initial.metadata.items() if key != "raw_output_path"},
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--outline-attempt":
        raise SystemExit(_outline_attempt_main(Path(sys.argv[2])))
    raise SystemExit("usage: python -m wildtrace.studio_pipeline --outline-attempt <request.json>")
