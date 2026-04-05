from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from wildtrace.diagram import (
    build_outline_rectifier,
    build_outline_validator,
    next_generator_params,
    run_langgraph_validation_loop,
    run_drawn_diagram_pass,
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


def _row_input_hash(row: dict[str, Any]) -> str:
    return config_hash(row)


def _paths_exist(repo_root: Path, *values: str | None) -> bool:
    for value in values:
        if value and not resolve_repo_path(repo_root, value).exists():
            return False
    return True


def _normalize_config_hash(runtime: dict[str, Any]) -> str:
    return config_hash({"quality": runtime["quality"]})


def _subject_config_hash(runtime: dict[str, Any]) -> str:
    return config_hash({"quality": runtime["quality"]})


def _outline_generation_config_hash(runtime: dict[str, Any]) -> str:
    return config_hash(
        {
            "diagram_generator": runtime["models"]["outline_rectifier"],
        }
    )


def _outline_validation_config_hash(runtime: dict[str, Any]) -> str:
    return config_hash(
        {
            "diagram_generator": runtime["models"]["outline_rectifier"],
            "outline_validator": runtime["models"]["outline_validator"],
            "opencv_prescreen": runtime["models"]["opencv_prescreen"],
        }
    )


def _sample_row_reusable(row: dict[str, Any], bronze: dict[str, Any], repo_root: Path, stage_hash: str) -> bool:
    lineage = row.get("lineage", {})
    if lineage.get("silver_config_hash") != stage_hash:
        return False
    if lineage.get("bronze_input_hash") != _row_input_hash(bronze):
        return False
    return _paths_exist(repo_root, row.get("image_path"), row.get("grayscale_path"), row.get("mask_path"), row.get("isolated_path"))


def _subject_row_reusable(row: dict[str, Any], sample: dict[str, Any], repo_root: Path, stage_hash: str) -> bool:
    lineage = row.get("lineage", {})
    if lineage.get("subject_config_hash") != stage_hash:
        return False
    if lineage.get("silver_input_hash") != _row_input_hash(sample):
        return False
    return _paths_exist(repo_root, row.get("crop_path"))


def _diagram_attempt_reusable(row: dict[str, Any], sample: dict[str, Any], repo_root: Path, stage_hash: str) -> bool:
    lineage = row.get("lineage", {})
    if lineage.get("outline_generation_config_hash") != stage_hash:
        return False
    if lineage.get("subject_input_hash") != _row_input_hash(sample):
        return False
    return _paths_exist(repo_root, row.get("diagram_path"))


def _validated_row_reusable(row: dict[str, Any], initial: dict[str, Any], repo_root: Path, stage_hash: str) -> bool:
    lineage = row.get("lineage", {})
    if lineage.get("validation_config_hash") != stage_hash:
        return False
    if lineage.get("initial_attempt_hash") != _row_input_hash(initial):
        return False
    return _paths_exist(repo_root, row.get("diagram_path"))


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
        "lineage": {
            "bronze_sample_id": bronze["sample_id"],
            "bronze_input_hash": _row_input_hash(bronze),
            "silver_config_hash": _normalize_config_hash(runtime),
        },
    }


def normalize_to_silver(repo_root: Path, runtime: dict[str, Any]) -> list[dict[str, Any]]:
    accepted_records = read_ndjson(bronze_accepted_manifest_path(repo_root, runtime))
    existing_rows = {row["sample_id"]: row for row in read_ndjson(silver_samples_manifest_path(repo_root, runtime))}
    stage_hash = _normalize_config_hash(runtime)
    expected_ids = {row["sample_id"] for row in accepted_records}
    changed = set(existing_rows) != expected_ids
    rows: list[dict[str, Any]] = []
    for bronze in accepted_records:
        existing = existing_rows.get(bronze["sample_id"])
        if existing and _sample_row_reusable(existing, bronze, repo_root, stage_hash):
            rows.append(existing)
            continue
        changed = True
        rows.append(_build_normalized_sample(bronze, repo_root, runtime))
    if changed or not silver_samples_manifest_path(repo_root, runtime).exists():
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
        "lineage": {
            **sample["lineage"],
            "silver_input_hash": _row_input_hash(sample),
            "subject_config_hash": _subject_config_hash(runtime),
            "subject_stage": "enrich_and_crop_subjects",
        },
    }


def enrich_and_crop_subjects(repo_root: Path, runtime: dict[str, Any]) -> list[dict[str, Any]]:
    silver_records = read_ndjson(silver_samples_manifest_path(repo_root, runtime))
    existing_rows = {row["sample_id"]: row for row in read_ndjson(silver_subjects_manifest_path(repo_root, runtime))}
    stage_hash = _subject_config_hash(runtime)
    expected_ids = {row["sample_id"] for row in silver_records}
    changed = set(existing_rows) != expected_ids
    rows: list[dict[str, Any]] = []
    for sample in silver_records:
        existing = existing_rows.get(sample["sample_id"])
        if existing and _subject_row_reusable(existing, sample, repo_root, stage_hash):
            rows.append(existing)
            continue
        changed = True
        rows.append(_build_subject_record(sample, repo_root, runtime))
    if changed or not silver_subjects_manifest_path(repo_root, runtime).exists():
        write_ndjson(silver_subjects_manifest_path(repo_root, runtime), rows)
    return rows


def _build_initial_diagram_attempt(
    sample: dict[str, Any],
    repo_root: Path,
    runtime: dict[str, Any],
    diagram_generator: Any,
) -> dict[str, Any]:
    generator_cfg = runtime["models"]["outline_rectifier"]
    category_paths = _silver_category_paths(repo_root, runtime, sample["category"])
    subject_path = resolve_repo_path(repo_root, sample["crop_path"])
    crop_image = open_image(subject_path)
    isolated_image = None
    if sample.get("isolated_path"):
        isolated_image = open_image(resolve_repo_path(repo_root, sample["isolated_path"]))
    image = diagram_generator.prepare_conditioning_image(crop_image, isolated_image)
    subject_mask = None
    if sample.get("mask_path"):
        subject_mask = Image.open(resolve_repo_path(repo_root, sample["mask_path"])).crop(
            (
                int(sample["crop_bbox"]["left"]),
                int(sample["crop_bbox"]["top"]),
                int(sample["crop_bbox"]["right"]),
                int(sample["crop_bbox"]["bottom"]),
            )
        ).resize(image.size, Image.Resampling.NEAREST)
    params = dict(generator_cfg.get("default_params", {}))
    destination = category_paths["diagrams"] / f"{sample['sample_id']}_attempt01.png"
    result = run_drawn_diagram_pass(sample, image, subject_mask, destination, params, diagram_generator)
    return {
        **_base_sample_fields(sample),
        "diagram_path": str(result.diagram_path.relative_to(repo_root)),
        "diagram_attempt": 1,
        "outline_params": params,
        "diagram_generator_backend": result.metadata,
        "lineage": {
            "subject_sample_id": sample["sample_id"],
            "subject_input_hash": _row_input_hash(sample),
            "outline_generation_config_hash": _outline_generation_config_hash(runtime),
            "generated_at": utc_now(),
        },
    }


def generate_line_diagrams(repo_root: Path, runtime: dict[str, Any]) -> list[dict[str, Any]]:
    samples = _accepted_subject_samples(repo_root, runtime)
    existing_rows = {row["sample_id"]: row for row in read_ndjson(line_diagram_attempts_manifest_path(repo_root, runtime))}
    stage_hash = _outline_generation_config_hash(runtime)
    expected_ids = {row["sample_id"] for row in samples}
    changed = set(existing_rows) != expected_ids
    rows: list[dict[str, Any]] = []
    missing_samples: list[dict[str, Any]] = []
    for sample in samples:
        existing = existing_rows.get(sample["sample_id"])
        if existing and _diagram_attempt_reusable(existing, sample, repo_root, stage_hash):
            rows.append(existing)
            continue
        changed = True
        missing_samples.append(sample)
    if missing_samples:
        diagram_generator = build_outline_rectifier(runtime["models"]["outline_rectifier"])
        rows.extend(
            _build_initial_diagram_attempt(sample, repo_root, runtime, diagram_generator)
            for sample in missing_samples
        )
        rows.sort(key=lambda row: row["sample_id"])
    if changed or not line_diagram_attempts_manifest_path(repo_root, runtime).exists():
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
        "diagram_generator_backend": initial["diagram_generator_backend"],
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
        "lineage": {
            "subject_sample_id": sample["sample_id"],
            "initial_attempt_hash": _row_input_hash(initial),
            "validation_config_hash": _outline_validation_config_hash(runtime),
            "diagram_attempt_count": 1,
        },
    }


def _validate_single_diagram(
    sample: dict[str, Any],
    initial: dict[str, Any],
    repo_root: Path,
    runtime: dict[str, Any],
    diagram_generator: Any,
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
        conditioning_path=resolve_repo_path(repo_root, sample["isolated_path"]) if sample.get("isolated_path") else None,
        subject_mask_path=resolve_repo_path(repo_root, sample["mask_path"]) if sample.get("mask_path") else None,
        diagram_root=_silver_category_paths(repo_root, runtime, sample["category"])["diagrams"].parent,
        diagram_generator=diagram_generator,
        outline_validator=outline_validator,
        opencv_settings=model_cfg["opencv_prescreen"],
        initial_params=retry_params,
        max_attempts=max(int(model_cfg["outline_rectifier"].get("max_attempts", 5)) - 1, 1),
        attempt_offset=1,
    )
    retried["diagram_path"] = str(Path(retried["diagram_path"]).relative_to(repo_root))
    return retried


def validate_and_retry_diagrams(repo_root: Path, runtime: dict[str, Any]) -> list[dict[str, Any]]:
    model_cfg = runtime["models"]
    initial_attempts = {row["sample_id"]: row for row in read_ndjson(line_diagram_attempts_manifest_path(repo_root, runtime))}
    samples = _accepted_subject_samples(repo_root, runtime)
    existing_rows = {row["sample_id"]: row for row in read_ndjson(validated_diagrams_manifest_path(repo_root, runtime))}
    stage_hash = _outline_validation_config_hash(runtime)
    expected_ids = {row["sample_id"] for row in samples}
    changed = set(existing_rows) != expected_ids
    rows: list[dict[str, Any]] = []
    missing_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for sample in samples:
        initial = initial_attempts[sample["sample_id"]]
        existing = existing_rows.get(sample["sample_id"])
        if existing and _validated_row_reusable(existing, initial, repo_root, stage_hash):
            rows.append(existing)
            continue
        changed = True
        missing_pairs.append((sample, initial))
    if missing_pairs:
        outline_validator = build_outline_validator(model_cfg["outline_validator"])
        diagram_generator = build_outline_rectifier(model_cfg["outline_rectifier"])
        rows.extend(
            _validate_single_diagram(
                sample,
                initial,
                repo_root,
                runtime,
                diagram_generator,
                outline_validator,
            )
            for sample, initial in missing_pairs
        )
        rows.sort(key=lambda row: row["sample_id"])
    if changed or not validated_diagrams_manifest_path(repo_root, runtime).exists():
        write_ndjson(validated_diagrams_manifest_path(repo_root, runtime), rows)
    return rows


def _apply_selection_ranks(rows: list[dict[str, Any]]) -> None:
    rows.sort(key=lambda row: row["diagram_validation_score"], reverse=True)
    for index, row in enumerate(rows, start=1):
        row["selected_for_gold"] = index == 1
        row["selection_rank"] = index


def select_final_by_angle(repo_root: Path, runtime: dict[str, Any]) -> list[dict[str, Any]]:
    validated = [row for row in read_ndjson(validated_diagrams_manifest_path(repo_root, runtime)) if row["diagram_validation_status"] == "accepted"]
    existing_rows = read_ndjson(selected_diagrams_manifest_path(repo_root, runtime))
    selection_source_hash = config_hash(
        [
            {
                "sample_id": row["sample_id"],
                "category": row["category"],
                "subcategory": row["subcategory"],
                "angle_bucket": row["angle_bucket"],
                "diagram_validation_score": row["diagram_validation_score"],
            }
            for row in sorted(validated, key=lambda item: item["sample_id"])
        ]
    )
    if existing_rows:
        reusable = len(existing_rows) == len(validated) and all(
            row.get("lineage", {}).get("selection_source_hash") == selection_source_hash
            for row in existing_rows
        )
        if reusable:
            return existing_rows
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in validated:
        grouped[(row["category"], row["subcategory"], row["angle_bucket"])].append(row)
    for rows in grouped.values():
        _apply_selection_ranks(rows)
    all_rows = sorted(validated, key=lambda row: (row["category"], row["subcategory"], row["angle_bucket"], -row["diagram_validation_score"]))
    for row in all_rows:
        row["lineage"] = {**row.get("lineage", {}), "selection_source_hash": selection_source_hash}
    write_ndjson(selected_diagrams_manifest_path(repo_root, runtime), all_rows)
    return all_rows
