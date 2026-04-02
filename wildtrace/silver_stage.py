from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from wildtrace.diagram import (
    build_outline_generator,
    build_outline_rectifier,
    build_outline_validator,
    next_generator_params,
    run_langgraph_validation_loop,
    run_outline_pass,
    validate_with_opencv,
)
from wildtrace.images import (
    blur_score,
    crop_to_mask_bbox,
    detect_grayscale,
    grayscale,
    isolate_subject,
    mask_coverage,
    normalize_mask,
    open_image,
    resize_longest_side,
    save_image,
)
from wildtrace.io_utils import read_ndjson, resolve_repo_path, write_ndjson
from wildtrace.pipeline import config_hash, utc_now
from wildtrace.stage_io import (
    bronze_accepted_manifest_path,
    line_diagram_attempts_manifest_path,
    selected_diagrams_manifest_path,
    silver_samples_manifest_path,
    silver_subjects_manifest_path,
    validated_diagrams_manifest_path,
)
from wildtrace.viewpoint import extract_view_features


def _silver_category_paths(repo_root: Path, runtime: dict[str, Any], category: str) -> dict[str, Path]:
    silver_storage = runtime["storage"]["silver"]
    return {
        "images": resolve_repo_path(repo_root, silver_storage["images_dir"]) / category,
        "masks": resolve_repo_path(repo_root, silver_storage["masks_dir"]) / category,
        "isolated": resolve_repo_path(repo_root, silver_storage["isolated_dir"]) / category,
        "crops": resolve_repo_path(repo_root, silver_storage["crops_dir"]) / category,
        "diagrams": resolve_repo_path(repo_root, silver_storage["diagrams_dir"]) / category,
    }


def _normalize_mask_assets(
    bronze: dict[str, Any],
    resized: Image.Image,
    repo_root: Path,
    category_paths: dict[str, Path],
) -> tuple[str | None, str | None, float | None, dict[str, Any]]:
    if not bronze.get("mask_path"):
        return None, None, None, {}
    mask = Image.open(resolve_repo_path(repo_root, bronze["mask_path"]))
    normalized_mask = normalize_mask(mask, resized.size)
    sample_id = bronze["sample_id"]

    mask_path = category_paths["masks"] / f"{sample_id}_mask.png"
    save_image(mask_path, normalized_mask)

    isolated = isolate_subject(resized, normalized_mask)
    isolated_path = category_paths["isolated"] / f"{sample_id}_isolated.png"
    save_image(isolated_path, isolated)

    return (
        str(mask_path.relative_to(repo_root)),
        str(isolated_path.relative_to(repo_root)),
        mask_coverage(normalized_mask),
        extract_view_features(normalized_mask),
    )


def _base_sample_fields(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "category": sample["category"],
        "subcategory": sample["subcategory"],
        "angle_bucket": sample["angle_bucket"],
    }


def _accepted_subject_samples(repo_root: Path, runtime: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        sample
        for sample in read_ndjson(silver_subjects_manifest_path(repo_root, runtime))
        if sample["segmentation_status"] == "accepted"
    ]


def _diagram_validation_score(opencv_score: float, semantic_score: float) -> float:
    return round((opencv_score * 0.5) + (semantic_score * 0.5), 4)


def _validator_model_name(runtime: dict[str, Any]) -> str:
    validator_cfg = runtime["models"]["outline_validator"]
    return str(
        validator_cfg.get("ollama_model_name")
        or validator_cfg.get("anthropic_model_name")
        or validator_cfg.get("model_name")
        or "outline_validator"
    )


def _build_normalized_sample(
    bronze: dict[str, Any],
    repo_root: Path,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    quality = runtime["quality"]
    category_paths = _silver_category_paths(repo_root, runtime, bronze["category"])
    image = open_image(resolve_repo_path(repo_root, bronze["image_path"]))
    resized = resize_longest_side(image, int(quality["max_resize"]))
    gray = grayscale(resized)
    mask_relative, isolated_relative, mask_ratio, view_features = _normalize_mask_assets(
        bronze,
        resized,
        repo_root,
        category_paths,
    )

    sample_id = bronze["sample_id"]
    rgb_path = category_paths["images"] / f"{sample_id}_rgb.png"
    gray_path = category_paths["images"] / f"{sample_id}_gray.png"
    save_image(rgb_path, resized)
    save_image(gray_path, gray)
    return {
        "sample_id": bronze["sample_id"],
        "category": bronze["category"],
        "subcategory": bronze["subcategory"],
        "tags": bronze["tags"],
        "label_source": bronze["label_source"],
        "label_confidence": bronze["label_confidence"],
        "angle_bucket": bronze["angle_bucket"],
        "angle_feasible": bronze["angle_feasible"],
        "view_confidence": bronze["view_confidence"],
        "image_path": str(rgb_path.relative_to(repo_root)),
        "grayscale_path": str(gray_path.relative_to(repo_root)),
        "mask_path": mask_relative,
        "isolated_path": isolated_relative,
        "mask_coverage_ratio": mask_ratio,
        "blur_score": blur_score(resized),
        "grayscale_detected": detect_grayscale(resized),
        "view_features": view_features,
        "quality_status": "accepted",
        "quality_flags": [],
        "lineage": {"bronze_sample_id": bronze["sample_id"], "silver_config_hash": config_hash({"quality": quality})},
    }


def normalize_to_silver(repo_root: Path, runtime: dict[str, Any]) -> list[dict[str, Any]]:
    accepted_records = read_ndjson(bronze_accepted_manifest_path(repo_root, runtime))
    rows = [_build_normalized_sample(bronze, repo_root, runtime) for bronze in accepted_records]
    write_ndjson(silver_samples_manifest_path(repo_root, runtime), rows)
    return rows


def _build_subject_record(sample: dict[str, Any], repo_root: Path, runtime: dict[str, Any]) -> dict[str, Any]:
    category_paths = _silver_category_paths(repo_root, runtime, sample["category"])
    isolated_path = resolve_repo_path(repo_root, sample["isolated_path"] or sample["image_path"])
    image = open_image(isolated_path)
    bbox = (0, 0, image.width, image.height)
    segmentation_score = 0.65
    segmentation_status = "accepted"
    if sample.get("mask_path"):
        mask = Image.open(resolve_repo_path(repo_root, sample["mask_path"]))
        crop, bbox = crop_to_mask_bbox(image, mask)
        bbox_fill = float(sample["view_features"].get("bbox_fill_ratio", 0.0))
        segmentation_score = min(1.0, 0.45 + bbox_fill)
        if segmentation_score < float(runtime["quality"].get("minimum_segmentation_score", 0.45)):
            segmentation_status = "rejected"
    else:
        crop = image
        segmentation_status = "rejected"

    crop_path = category_paths["crops"] / f"{sample['sample_id']}_crop.png"
    save_image(crop_path, crop)
    return {
        **sample,
        "crop_path": str(crop_path.relative_to(repo_root)),
        "crop_bbox": {"left": bbox[0], "top": bbox[1], "right": bbox[2], "bottom": bbox[3]},
        "segmentation_status": segmentation_status,
        "segmentation_score": segmentation_score,
        "lineage": {**sample["lineage"], "subject_stage": "enrich_and_crop_subjects"},
    }


def enrich_and_crop_subjects(repo_root: Path, runtime: dict[str, Any]) -> list[dict[str, Any]]:
    silver_records = read_ndjson(silver_samples_manifest_path(repo_root, runtime))
    rows = [_build_subject_record(sample, repo_root, runtime) for sample in silver_records]
    write_ndjson(silver_subjects_manifest_path(repo_root, runtime), rows)
    return rows


def _build_initial_diagram_attempt(
    sample: dict[str, Any],
    repo_root: Path,
    runtime: dict[str, Any],
    outline_generator: Any,
    outline_rectifier: Any,
) -> dict[str, Any]:
    generator_cfg = runtime["models"]["outline_generator"]
    category_paths = _silver_category_paths(repo_root, runtime, sample["category"])
    subject_path = resolve_repo_path(repo_root, sample["crop_path"])
    image = open_image(subject_path)
    params = dict(generator_cfg.get("default_params", {}))
    destination = category_paths["diagrams"] / f"{sample['sample_id']}_attempt01.png"
    result = run_outline_pass(image, destination, params, outline_generator, outline_rectifier)
    return {
        **_base_sample_fields(sample),
        "diagram_path": str(result.diagram_path.relative_to(repo_root)),
        "diagram_attempt": 1,
        "outline_params": params,
        "outline_generator_backend": result.generator_metadata,
        "outline_rectifier_backend": result.rectifier_metadata,
        "lineage": {"subject_sample_id": sample["sample_id"], "generated_at": utc_now()},
    }


def generate_line_diagrams(repo_root: Path, runtime: dict[str, Any]) -> list[dict[str, Any]]:
    outline_generator = build_outline_generator(runtime["models"]["outline_generator"])
    outline_rectifier = build_outline_rectifier(runtime["models"]["outline_rectifier"])
    samples = _accepted_subject_samples(repo_root, runtime)
    rows = [
        _build_initial_diagram_attempt(sample, repo_root, runtime, outline_generator, outline_rectifier)
        for sample in samples
    ]
    write_ndjson(line_diagram_attempts_manifest_path(repo_root, runtime), rows)
    return rows


def _accepted_diagram_record(
    sample: dict[str, Any],
    runtime: dict[str, Any],
    initial: dict[str, Any],
    opencv_result: Any,
    semantic_result: Any,
) -> dict[str, Any]:
    return {
        **_base_sample_fields(sample),
        "diagram_path": initial["diagram_path"],
        "diagram_attempt": 1,
        "outline_generator_backend": initial["outline_generator_backend"],
        "outline_rectifier_backend": initial["outline_rectifier_backend"],
        "diagram_validation_status": "accepted",
        "diagram_validation_score": _diagram_validation_score(opencv_result.score, semantic_result.score),
        "opencv_flags": opencv_result.flags,
        "outline_validator_model": semantic_result.metadata.get(
            "model_name",
            _validator_model_name(runtime),
        ),
        "outline_validator_reason": semantic_result.reason,
        "selected_for_gold": False,
        "selection_rank": None,
        "lineage": {"subject_sample_id": sample["sample_id"], "diagram_attempt_count": 1},
    }


def _validate_single_diagram(
    sample: dict[str, Any],
    initial: dict[str, Any],
    repo_root: Path,
    runtime: dict[str, Any],
    outline_generator: Any,
    outline_rectifier: Any,
    outline_validator: Any,
) -> dict[str, Any]:
    model_cfg = runtime["models"]
    initial_path = resolve_repo_path(repo_root, initial["diagram_path"])
    opencv_result = validate_with_opencv(initial_path, model_cfg["opencv_prescreen"])
    semantic_result = outline_validator.validate(sample, initial_path, opencv_result)
    if opencv_result.passed and semantic_result.passed:
        return _accepted_diagram_record(sample, runtime, initial, opencv_result, semantic_result)

    retry_params = next_generator_params(initial["outline_params"], opencv_result.flags, semantic_result.passed)
    retried = run_langgraph_validation_loop(
        sample=sample,
        subject_path=resolve_repo_path(repo_root, sample["crop_path"]),
        diagram_root=_silver_category_paths(repo_root, runtime, sample["category"])["diagrams"].parent,
        outline_generator=outline_generator,
        outline_rectifier=outline_rectifier,
        outline_validator=outline_validator,
        opencv_settings=model_cfg["opencv_prescreen"],
        initial_params=retry_params,
        max_attempts=max(int(model_cfg["outline_generator"].get("max_attempts", 5)) - 1, 1),
        attempt_offset=1,
    )
    retried["diagram_path"] = str(Path(retried["diagram_path"]).relative_to(repo_root))
    return retried


def validate_and_retry_diagrams(repo_root: Path, runtime: dict[str, Any]) -> list[dict[str, Any]]:
    model_cfg = runtime["models"]
    outline_validator = build_outline_validator(model_cfg["outline_validator"])
    outline_generator = build_outline_generator(model_cfg["outline_generator"])
    outline_rectifier = build_outline_rectifier(model_cfg["outline_rectifier"])
    initial_attempts = {row["sample_id"]: row for row in read_ndjson(line_diagram_attempts_manifest_path(repo_root, runtime))}
    samples = _accepted_subject_samples(repo_root, runtime)
    rows = [
        _validate_single_diagram(
            sample,
            initial_attempts[sample["sample_id"]],
            repo_root,
            runtime,
            outline_generator,
            outline_rectifier,
            outline_validator,
        )
        for sample in samples
    ]
    write_ndjson(validated_diagrams_manifest_path(repo_root, runtime), rows)
    return rows


def _apply_selection_ranks(rows: list[dict[str, Any]]) -> None:
    rows.sort(key=lambda row: row["diagram_validation_score"], reverse=True)
    for index, row in enumerate(rows, start=1):
        row["selected_for_gold"] = index == 1
        row["selection_rank"] = index


def select_final_by_angle(repo_root: Path, runtime: dict[str, Any]) -> list[dict[str, Any]]:
    validated = [row for row in read_ndjson(validated_diagrams_manifest_path(repo_root, runtime)) if row["diagram_validation_status"] == "accepted"]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in validated:
        grouped[(row["category"], row["subcategory"], row["angle_bucket"])].append(row)
    for rows in grouped.values():
        _apply_selection_ranks(rows)
    all_rows = sorted(validated, key=lambda row: (row["category"], row["subcategory"], row["angle_bucket"], -row["diagram_validation_score"]))
    write_ndjson(selected_diagrams_manifest_path(repo_root, runtime), all_rows)
    return all_rows
