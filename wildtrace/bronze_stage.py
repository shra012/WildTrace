from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from wildtrace.enrichment import build_enrichment_backend
from wildtrace.images import blur_score, normalize_mask, open_image
from wildtrace.io_utils import read_ndjson, resolve_repo_path, sha256_file, write_ndjson
from wildtrace.pipeline import config_hash, fetch_latest_view_path, relative_local_uri, utc_now
from wildtrace.stage_io import bronze_accepted_manifest_path, bronze_curated_manifest_path
from wildtrace.viewpoint import build_viewpoint_backend, extract_view_features


def _bronze_config_hash(runtime: dict[str, Any]) -> str:
    return config_hash({"quality": runtime["quality"], "viewpoints": runtime["viewpoints"], "models": runtime["models"]})


def _bronze_row_reusable(
    row: dict[str, Any],
    record: dict[str, Any],
    repo_root: Path,
    bronze_hash: str,
) -> bool:
    lineage = row.get("lineage", {})
    if lineage.get("fetch_record_id") != record["record_id"]:
        return False
    if lineage.get("bronze_config_hash") != bronze_hash:
        return False
    if row.get("curation_status") != "accepted":
        return True
    image_path = resolve_repo_path(repo_root, row["image_path"])
    if not image_path.exists():
        return False
    if row.get("mask_path") and not resolve_repo_path(repo_root, row["mask_path"]).exists():
        return False
    if row.get("metadata_path") and not resolve_repo_path(repo_root, row["metadata_path"]).exists():
        return False
    return True


def _base_curation_state(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "curation_status": "accepted",
        "curation_flags": [],
        "deleted_paths": [],
        "checksum": record.get("image_checksum"),
        "angle_bucket": "unknown",
        "angle_feasible": False,
        "view_confidence": 0.0,
        "subcategory": record["category"].lower(),
        "tags": [],
        "label_confidence": 0.0,
        "enrichment_metadata": {},
        "image": None,
    }


def _reject(state: dict[str, Any], flag: str) -> None:
    state["curation_status"] = "rejected"
    state["curation_flags"].append(flag)


def _delete_if_present(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    path.unlink()
    return str(path)


def _bronze_delete(record: dict[str, Any], repo_root: Path) -> list[str]:
    deleted: list[str] = []
    for key in ("image_path", "mask_path", "metadata_path"):
        value = record.get(key)
        if not value:
            continue
        deleted_path = _delete_if_present(resolve_repo_path(repo_root, value))
        if deleted_path:
            deleted.append(deleted_path)
    return deleted


def _check_image_asset(
    record: dict[str, Any],
    image_path: Path,
    seen_checksums: dict[str, dict[str, Any]],
    state: dict[str, Any],
) -> None:
    if not image_path.exists():
        _reject(state, "missing_image")
        return
    checksum = state["checksum"] or sha256_file(image_path)
    state["checksum"] = checksum
    if checksum in seen_checksums:
        _reject(state, "duplicate_image_checksum")
        return
    seen_checksums[checksum] = record


def _load_and_score_image(image_path: Path, quality: dict[str, Any], state: dict[str, Any]) -> None:
    if state["curation_status"] != "accepted":
        return
    try:
        image = open_image(image_path)
    except Exception:  # noqa: BLE001
        _reject(state, "unreadable_image")
        return

    state["image"] = image
    if image.width < int(quality["minimum_width"]) or image.height < int(quality["minimum_height"]):
        _reject(state, "below_minimum_resolution")
    if blur_score(image) < float(quality["blur_variance_threshold"]):
        _reject(state, "low_blur_variance")


def _classify_angle(
    image: Image.Image,
    mask_path: Path | None,
    viewpoints: dict[str, Any],
    backend: Any,
) -> tuple[str, bool, float, list[str]]:
    if mask_path is None or not mask_path.exists():
        return "unknown", False, 0.0, ["missing_mask"]
    normalized_mask = normalize_mask(Image.open(mask_path), image.size)
    view_features = extract_view_features(normalized_mask)
    result = backend.classify({"mask_available": True, "view_features": view_features}, viewpoints)
    if result.bucket == "unknown" and result.scores:
        angle_bucket = max(result.scores.items(), key=lambda item: item[1])[0]
    else:
        angle_bucket = result.bucket
    angle_feasible = angle_bucket in set(viewpoints.get("allowed_buckets", []))
    flags = list(result.flags or [])
    if result.status == "rejected" or angle_bucket in set(viewpoints.get("rejected_buckets", [])):
        return angle_bucket, False, result.confidence, flags or ["angle_not_feasible"]
    if result.status != "accepted":
        return angle_bucket, angle_feasible, result.confidence, flags or ["low_view_confidence"]
    return angle_bucket, angle_feasible, result.confidence, flags


def _apply_angle_classification(
    mask_path: Path | None,
    viewpoints: dict[str, Any],
    viewpoint_backend: Any,
    state: dict[str, Any],
) -> None:
    if state["curation_status"] != "accepted" or state["image"] is None:
        return
    angle_bucket, angle_feasible, view_confidence, angle_flags = _classify_angle(
        state["image"],
        mask_path,
        viewpoints,
        viewpoint_backend,
    )
    state["angle_bucket"] = angle_bucket
    state["angle_feasible"] = angle_feasible
    state["view_confidence"] = view_confidence
    state["curation_flags"].extend(angle_flags)
    if not angle_feasible:
        state["curation_status"] = "rejected"


def _apply_enrichment(record: dict[str, Any], enrichment_backend: Any, state: dict[str, Any]) -> None:
    if state["curation_status"] != "accepted" or state["image"] is None:
        return
    enriched_sample = dict(record)
    enriched_sample["angle_bucket"] = state["angle_bucket"]
    enriched = enrichment_backend.enrich(enriched_sample, state["image"])
    state["subcategory"] = enriched.subcategory
    state["tags"] = enriched.tags
    state["label_confidence"] = enriched.confidence
    state["enrichment_metadata"] = enriched.metadata


def _finalize_curation_record(
    record: dict[str, Any],
    repo_root: Path,
    runtime: dict[str, Any],
    image_path: Path,
    mask_path: Path | None,
    state: dict[str, Any],
) -> dict[str, Any]:
    quality = runtime["quality"]
    viewpoints = runtime["viewpoints"]
    model_cfg = runtime["models"]["enrichment"]
    if state["curation_status"] != "accepted":
        state["deleted_paths"] = _bronze_delete(record, repo_root)

    return {
        "sample_id": record["sample_id"],
        "category": record["category"],
        "subcategory": state["subcategory"],
        "tags": state["tags"],
        "label_source": model_cfg.get("model_name", "bioclip"),
        "label_confidence": state["label_confidence"],
        "angle_bucket": state["angle_bucket"],
        "angle_feasible": state["angle_feasible"],
        "view_confidence": state["view_confidence"],
        "curation_status": state["curation_status"],
        "curation_flags": sorted(set(state["curation_flags"])),
        "image_path": record["image_path"],
        "mask_path": record.get("mask_path"),
        "metadata_path": record.get("metadata_path"),
        "image_uri": record.get("image_uri") or relative_local_uri(repo_root, image_path),
        "mask_uri": record.get("mask_uri") or relative_local_uri(repo_root, mask_path),
        "source_dataset": record["source_dataset"],
        "source_image_id": record["source_image_id"],
        "split": record.get("split"),
        "image_checksum": state["checksum"],
        "deleted_paths": state["deleted_paths"],
        "lineage": {
            "fetch_record_id": record["record_id"],
            "bronze_config_hash": _bronze_config_hash(runtime),
            "enrichment_model": state["enrichment_metadata"],
            "curated_at": utc_now(),
        },
    }


def _curate_single_record(
    record: dict[str, Any],
    repo_root: Path,
    runtime: dict[str, Any],
    viewpoint_backend: Any,
    enrichment_backend: Any,
    seen_checksums: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    quality = runtime["quality"]
    viewpoints = runtime["viewpoints"]
    image_path = resolve_repo_path(repo_root, record["image_path"])
    mask_path = resolve_repo_path(repo_root, record["mask_path"]) if record.get("mask_path") else None
    state = _base_curation_state(record)
    _check_image_asset(record, image_path, seen_checksums, state)
    _load_and_score_image(image_path, quality, state)
    _apply_angle_classification(mask_path, viewpoints, viewpoint_backend, state)
    _apply_enrichment(record, enrichment_backend, state)
    return _finalize_curation_record(record, repo_root, runtime, image_path, mask_path, state)


def curate_bronze(repo_root: Path, runtime: dict[str, Any]) -> list[dict[str, Any]]:
    fetch_records = read_ndjson(fetch_latest_view_path(repo_root, runtime))
    viewpoint_backend = build_viewpoint_backend(runtime["viewpoints"])
    enrichment_backend = build_enrichment_backend(runtime["models"]["enrichment"])
    bronze_hash = _bronze_config_hash(runtime)
    existing_rows = {row["sample_id"]: row for row in read_ndjson(bronze_curated_manifest_path(repo_root, runtime))}
    seen_checksums: dict[str, dict[str, Any]] = {}
    expected_ids = {record["sample_id"] for record in fetch_records}
    changed = set(existing_rows) != expected_ids
    curated: list[dict[str, Any]] = []
    for record in sorted(fetch_records, key=lambda row: row["sample_id"]):
        existing = existing_rows.get(record["sample_id"])
        if existing and _bronze_row_reusable(existing, record, repo_root, bronze_hash):
            curated.append(existing)
            if existing.get("curation_status") == "accepted" and existing.get("image_checksum"):
                seen_checksums[existing["image_checksum"]] = record
            continue
        changed = True
        curated.append(_curate_single_record(record, repo_root, runtime, viewpoint_backend, enrichment_backend, seen_checksums))
    accepted = [row for row in curated if row["curation_status"] == "accepted"]
    if changed or not bronze_accepted_manifest_path(repo_root, runtime).exists():
        write_ndjson(bronze_curated_manifest_path(repo_root, runtime), curated)
        write_ndjson(bronze_accepted_manifest_path(repo_root, runtime), accepted)
    return curated
