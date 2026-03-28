from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from PIL import Image

from wildtrace.config import load_runtime_config
from wildtrace.images import (
    blur_score,
    detect_grayscale,
    grayscale,
    isolate_subject,
    mask_coverage,
    normalize_mask,
    open_image,
    resize_longest_side,
    sample_outline_strokes,
    save_image,
    write_svg,
)
from wildtrace.io_utils import (
    append_ndjson,
    ensure_parent,
    fetch_to_path,
    load_records,
    read_ndjson,
    resolve_repo_path,
    sha256_file,
    stable_sample_id,
    write_ndjson,
)
from wildtrace.outline import build_backend
from wildtrace.viewpoint import build_viewpoint_backend, extract_view_features


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def config_hash(runtime: dict[str, Any]) -> str:
    payload = json.dumps(runtime, sort_keys=True)
    return stable_sample_id(payload)


def parse_common_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--repo-root", default=".", help="Repo root for relative paths.")
    parser.add_argument("--config-dir", default=None, help="Config directory. Defaults to <repo-root>/configs.")
    return parser.parse_args()


def load_runtime(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    repo_root = Path(args.repo_root).resolve()
    config_dir = Path(args.config_dir).resolve() if args.config_dir else None
    return repo_root, load_runtime_config(repo_root, config_dir)


def category_limits(runtime: dict[str, Any]) -> dict[str, int]:
    return {item["name"]: int(item.get("limit", 0) or 0) for item in runtime["datasets"].get("categories", [])}


def bronze_images_root(repo_root: Path, runtime: dict[str, Any]) -> Path:
    return resolve_repo_path(repo_root, runtime["storage"]["bronze"]["images_dir"])


def bronze_masks_root(repo_root: Path, runtime: dict[str, Any]) -> Path:
    return resolve_repo_path(repo_root, runtime["storage"]["bronze"]["masks_dir"])


def bronze_metadata_root(repo_root: Path, runtime: dict[str, Any]) -> Path:
    return resolve_repo_path(repo_root, runtime["storage"]["bronze"]["metadata_dir"])


def fetch_manifest_path(repo_root: Path, runtime: dict[str, Any]) -> Path:
    return resolve_repo_path(repo_root, runtime["datasets"]["fetch"]["manifest_path"])


def fetch_latest_view_path(repo_root: Path, runtime: dict[str, Any]) -> Path:
    return resolve_repo_path(repo_root, runtime["datasets"]["fetch"]["latest_view_path"])


def ingest_manifest_path(repo_root: Path, runtime: dict[str, Any]) -> Path:
    return resolve_repo_path(repo_root, runtime["datasets"]["ingest"]["manifest_path"])


def ingest_latest_view_path(repo_root: Path, runtime: dict[str, Any]) -> Path:
    return resolve_repo_path(repo_root, runtime["datasets"]["ingest"]["latest_view_path"])


def silver_samples_manifest_path(repo_root: Path, runtime: dict[str, Any]) -> Path:
    return resolve_repo_path(repo_root, runtime["storage"]["silver"]["qa_dir"]) / "silver_samples.ndjson"


def silver_viewpoints_manifest_path(repo_root: Path, runtime: dict[str, Any]) -> Path:
    return resolve_repo_path(repo_root, runtime["storage"]["silver"]["qa_dir"]) / "silver_viewpoints.ndjson"


def latest_successful_records(records: list[dict[str, Any]], status_field: str) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get(status_field) != "success":
            continue
        sample_id = record["sample_id"]
        current_version = int(record.get("asset_version", 0) or 0)
        previous = latest.get(sample_id)
        if previous is None or current_version >= int(previous.get("asset_version", 0) or 0):
            latest[sample_id] = record
    return latest


def write_latest_success_view(ledger_path: Path, latest_view_path: Path, status_field: str) -> list[dict[str, Any]]:
    latest = latest_successful_records(read_ndjson(ledger_path), status_field)
    rows = sorted(latest.values(), key=lambda row: (row.get("category", ""), row["sample_id"]))
    write_ndjson(latest_view_path, rows)
    return rows


def relative_local_uri(repo_root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    return f"local://{path.relative_to(repo_root).as_posix()}"


def infer_suffix(value: str | None, default: str) -> str:
    if not value:
        return default
    suffix = Path(urlparse(value).path).suffix
    return suffix or default


def sample_identity(record: dict[str, Any], source_dataset: str) -> str:
    logical_key = str(
        record.get("instance_key")
        or record.get("mask_path_in_archive")
        or record.get("mask_path")
        or record.get("source_image_id")
    )
    return stable_sample_id(source_dataset, str(record.get("split", "unknown")), record["category"], logical_key)


def selected_record_catalog(runtime: dict[str, Any], repo_root: Path) -> list[dict[str, Any]]:
    fetch_cfg = runtime["datasets"]["fetch"]
    records_path = fetch_cfg.get("records_path")
    records = list(fetch_cfg.get("records", []))
    if records_path:
        records.extend(load_records(resolve_repo_path(repo_root, records_path)))
    limits = category_limits(runtime)
    counts: Counter[str] = Counter()
    chosen: list[dict[str, Any]] = []
    for record in records:
        category = record.get("category")
        if category not in limits:
            continue
        if limits[category] and counts[category] >= limits[category]:
            continue
        normalized = dict(record)
        normalized["source_url"] = record.get("source_url") or record.get("image_url")
        normalized["mask_source_url"] = record.get("mask_source_url") or record.get("mask_url")
        normalized["sample_id"] = sample_identity(normalized, runtime["datasets"]["source_dataset"])
        counts[category] += 1
        chosen.append(normalized)
    return chosen


def cache_remote_file(url: str, destination: Path, timeout_seconds: int, user_agent: str) -> Path:
    if destination.exists():
        return destination
    fetch_to_path(url, destination, timeout_seconds=timeout_seconds, user_agent=user_agent)
    return destination


def discover_official_openimages(runtime: dict[str, Any], repo_root: Path) -> list[dict[str, Any]]:
    storage = runtime["storage"]
    datasets = runtime["datasets"]
    fetch_cfg = datasets["fetch"]
    official = fetch_cfg["official"]
    metadata_root = bronze_metadata_root(repo_root, runtime) / "catalog"
    metadata_root.mkdir(parents=True, exist_ok=True)
    timeout = int(fetch_cfg["timeout_seconds"])
    user_agent = str(fetch_cfg["user_agent"])

    class_cache = metadata_root / Path(urlparse(official["class_descriptions_url"]).path).name
    cache_remote_file(official["class_descriptions_url"], class_cache, timeout, user_agent)
    display_to_label: dict[str, str] = {}
    with class_cache.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2:
                display_to_label[row[1]] = row[0]

    limits = category_limits(runtime)
    target_labels = {display_to_label[name]: name for name in limits if name in display_to_label}
    discovered_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    selected_ids_by_split: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()

    for split in official["splits"]:
        ann_url = official["segmentation_annotation_urls"][split]
        ann_cache = metadata_root / Path(urlparse(ann_url).path).name
        cache_remote_file(ann_url, ann_cache, timeout, user_agent)
        with ann_cache.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                label_name = row.get("LabelName")
                category = target_labels.get(label_name)
                if not category:
                    continue
                if limits[category] and counts[category] >= limits[category]:
                    continue
                record = {
                    "source_image_id": row["ImageID"],
                    "category": category,
                    "split": split,
                    "label_name": label_name,
                    "box_id": row.get("BoxID"),
                    "mask_path_in_archive": row.get("MaskPath"),
                    "instance_key": row.get("MaskPath") or row["ImageID"],
                    "source_url": official["image_base_url_template"].format(split=split, image_id=row["ImageID"]),
                    "mask_source_url": None,
                    "mask_archive_url": official["mask_zip_url_templates"][split].format(prefix=row["ImageID"][0].lower()),
                }
                record["sample_id"] = sample_identity(record, datasets["source_dataset"])
                discovered_by_split[split].append(record)
                selected_ids_by_split[split].add(row["ImageID"])
                counts[category] += 1
                if all(limit and counts[name] >= limit for name, limit in limits.items()):
                    break

    enriched: list[dict[str, Any]] = []
    for split, records in discovered_by_split.items():
        if not records:
            continue
        info_url = official["image_info_urls"][split]
        info_cache = metadata_root / Path(urlparse(info_url).path).name
        cache_remote_file(info_url, info_cache, timeout, user_agent)
        selected_ids = selected_ids_by_split[split]
        info_map: dict[str, dict[str, str]] = {}
        with info_cache.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                image_id = row.get("ImageID")
                if image_id in selected_ids:
                    info_map[image_id] = row
        for record in records:
            info = info_map.get(record["source_image_id"], {})
            record.update(
                {
                    "license": info.get("License", "unknown"),
                    "original_url": info.get("OriginalURL"),
                    "original_landing_url": info.get("OriginalLandingURL"),
                    "author": info.get("Author"),
                    "title": info.get("Title"),
                }
            )
            enriched.append(record)
    return enriched


def discovered_fetch_records(runtime: dict[str, Any], repo_root: Path) -> list[dict[str, Any]]:
    fetch_cfg = runtime["datasets"]["fetch"]
    if fetch_cfg.get("source_mode") == "records" or fetch_cfg.get("records_path") or fetch_cfg.get("records"):
        return selected_record_catalog(runtime, repo_root)
    return discover_official_openimages(runtime, repo_root)


def current_file_checksum(source_url: str | None) -> str | None:
    if not source_url:
        return None
    parsed = urlparse(source_url)
    if parsed.scheme == "file":
        path = Path(parsed.path)
    elif parsed.scheme == "":
        path = Path(source_url)
    else:
        return None
    if not path.exists():
        return None
    return sha256_file(path)


def fetch_record_is_current(repo_root: Path, record: dict[str, Any]) -> bool:
    image_path = resolve_repo_path(repo_root, record["image_path"])
    if not image_path.exists() or sha256_file(image_path) != record.get("image_checksum"):
        return False
    mask_path_value = record.get("mask_path")
    mask_checksum = record.get("mask_checksum")
    if mask_path_value:
        mask_path = resolve_repo_path(repo_root, mask_path_value)
        if not mask_path.exists() or sha256_file(mask_path) != mask_checksum:
            return False
    return True


def next_asset_version(record: dict[str, Any] | None) -> int:
    return int(record.get("asset_version", 0) or 0) + 1 if record else 1


def versioned_asset_paths(repo_root: Path, runtime: dict[str, Any], record: dict[str, Any], version: int) -> tuple[Path, Path | None, Path]:
    image_suffix = infer_suffix(record.get("source_url"), ".jpg")
    mask_suffix = infer_suffix(record.get("mask_source_url") or record.get("mask_path_in_archive"), ".png")
    image_path = bronze_images_root(repo_root, runtime) / record["category"] / record["sample_id"] / f"v{version:04d}{image_suffix}"
    mask_path = None
    if record.get("mask_source_url") or record.get("mask_path_in_archive"):
        mask_path = bronze_masks_root(repo_root, runtime) / record["category"] / record["sample_id"] / f"v{version:04d}{mask_suffix}"
    metadata_path = bronze_metadata_root(repo_root, runtime) / record["category"] / record["sample_id"] / f"v{version:04d}.json"
    return image_path, mask_path, metadata_path


def fetch_mask_from_archive(mask_archive_url: str, member_name: str, archive_cache_path: Path, destination: Path, timeout_seconds: int, user_agent: str) -> None:
    cache_remote_file(mask_archive_url, archive_cache_path, timeout_seconds, user_agent)
    ensure_parent(destination)
    with zipfile.ZipFile(archive_cache_path, "r") as archive, destination.open("wb") as handle:
        handle.write(archive.read(member_name))


def fetch_openimages(repo_root: Path, runtime: dict[str, Any]) -> list[dict[str, Any]]:
    datasets = runtime["datasets"]
    fetch_cfg = datasets["fetch"]
    timeout = int(fetch_cfg["timeout_seconds"])
    user_agent = str(fetch_cfg["user_agent"])
    ledger_path = fetch_manifest_path(repo_root, runtime)
    latest_view = fetch_latest_view_path(repo_root, runtime)
    metadata_root = bronze_metadata_root(repo_root, runtime)
    archive_root = metadata_root / "mask_archives"
    existing = read_ndjson(ledger_path)
    latest_success = latest_successful_records(existing, "fetch_status")
    fetched_records: list[dict[str, Any]] = []

    for record in discovered_fetch_records(runtime, repo_root):
        latest = latest_success.get(record["sample_id"])
        current_source_checksum = current_file_checksum(record.get("source_url"))
        current_mask_source_checksum = current_file_checksum(record.get("mask_source_url"))
        if latest and fetch_record_is_current(repo_root, latest):
            image_changed = current_source_checksum is not None and current_source_checksum != latest.get("image_checksum")
            mask_changed = current_mask_source_checksum is not None and current_mask_source_checksum != latest.get("mask_checksum")
            if not image_changed and not mask_changed:
                continue
        version = next_asset_version(latest)
        image_path, mask_path, metadata_path = versioned_asset_paths(repo_root, runtime, record, version)
        failure_reason = None
        try:
            fetch_to_path(record["source_url"], image_path, timeout_seconds=timeout, user_agent=user_agent)
            if mask_path and record.get("mask_source_url"):
                fetch_to_path(record["mask_source_url"], mask_path, timeout_seconds=timeout, user_agent=user_agent)
            elif mask_path and record.get("mask_archive_url") and record.get("mask_path_in_archive"):
                archive_path = archive_root / record["split"] / f"{record['source_image_id'][0].lower()}.zip"
                fetch_mask_from_archive(
                    record["mask_archive_url"],
                    record["mask_path_in_archive"],
                    archive_path,
                    mask_path,
                    timeout_seconds=timeout,
                    user_agent=user_agent,
                )
            image_checksum = sha256_file(image_path)
            mask_checksum = sha256_file(mask_path) if mask_path and mask_path.exists() else None
            metadata_payload = dict(record)
            metadata_payload["asset_version"] = version
            metadata_payload["image_path"] = str(image_path.relative_to(repo_root))
            metadata_payload["mask_path"] = str(mask_path.relative_to(repo_root)) if mask_path else None
            metadata_payload["image_checksum"] = image_checksum
            metadata_payload["mask_checksum"] = mask_checksum
            metadata_payload["storage_backend"] = "local"
            metadata_payload["fetched_at"] = utc_now()
            ensure_parent(metadata_path)
            metadata_path.write_text(json.dumps(metadata_payload, sort_keys=True, indent=2), encoding="utf-8")
            fetch_record = {
                "record_id": stable_sample_id(record["sample_id"], str(version), image_checksum),
                "sample_id": record["sample_id"],
                "task_type": datasets["task_type"],
                "source_dataset": datasets["source_dataset"],
                "source_image_id": record["source_image_id"],
                "category": record["category"],
                "split": record.get("split"),
                "asset_version": version,
                "fetch_status": "success",
                "image_checksum": image_checksum,
                "mask_checksum": mask_checksum,
                "image_path": str(image_path.relative_to(repo_root)),
                "mask_path": str(mask_path.relative_to(repo_root)) if mask_path else None,
                "metadata_path": str(metadata_path.relative_to(repo_root)),
                "image_uri": relative_local_uri(repo_root, image_path),
                "mask_uri": relative_local_uri(repo_root, mask_path),
                "storage_backend": "local",
                "source_url": record.get("source_url"),
                "mask_source_url": record.get("mask_source_url"),
                "mask_archive_url": record.get("mask_archive_url"),
                "mask_path_in_archive": record.get("mask_path_in_archive"),
                "license": record.get("license", "unknown"),
                "failure_reason": None,
                "fetched_at": utc_now(),
            }
        except Exception as exc:  # noqa: BLE001
            failure_reason = str(exc)
            fetch_record = {
                "record_id": stable_sample_id(record["sample_id"], str(version), failure_reason),
                "sample_id": record["sample_id"],
                "task_type": datasets["task_type"],
                "source_dataset": datasets["source_dataset"],
                "source_image_id": record["source_image_id"],
                "category": record["category"],
                "split": record.get("split"),
                "asset_version": version,
                "fetch_status": "failed",
                "image_checksum": None,
                "mask_checksum": None,
                "image_path": str(image_path.relative_to(repo_root)),
                "mask_path": str(mask_path.relative_to(repo_root)) if mask_path else None,
                "metadata_path": str(metadata_path.relative_to(repo_root)),
                "image_uri": relative_local_uri(repo_root, image_path),
                "mask_uri": relative_local_uri(repo_root, mask_path),
                "storage_backend": "local",
                "source_url": record.get("source_url"),
                "mask_source_url": record.get("mask_source_url"),
                "mask_archive_url": record.get("mask_archive_url"),
                "mask_path_in_archive": record.get("mask_path_in_archive"),
                "license": record.get("license", "unknown"),
                "failure_reason": failure_reason,
                "fetched_at": utc_now(),
            }
        append_ndjson(ledger_path, fetch_record)
        fetched_records.append(fetch_record)
    write_latest_success_view(ledger_path, latest_view, "fetch_status")
    return fetched_records


def ingest_openimages(repo_root: Path, runtime: dict[str, Any]) -> list[dict[str, Any]]:
    storage = runtime["storage"]
    datasets = runtime["datasets"]
    fetch_latest_path = fetch_latest_view_path(repo_root, runtime)
    ingest_ledger_path = ingest_manifest_path(repo_root, runtime)
    ingest_latest_path = ingest_latest_view_path(repo_root, runtime)
    ingest_hash = config_hash({"datasets": datasets, "storage": storage})
    fetched_latest = read_ndjson(fetch_latest_path)
    existing_ingest = read_ndjson(ingest_ledger_path)
    latest_ingest = latest_successful_records(existing_ingest, "ingest_status")
    appended: list[dict[str, Any]] = []

    for fetch_record in fetched_latest:
        sample_id = fetch_record["sample_id"]
        image_path = resolve_repo_path(repo_root, fetch_record["image_path"])
        current_checksum = sha256_file(image_path)
        latest = latest_ingest.get(sample_id)
        if latest and latest.get("checksum") == current_checksum:
            continue
        image = open_image(image_path)
        ingest_record = {
            "record_id": stable_sample_id(sample_id, str(fetch_record["asset_version"]), current_checksum),
            "sample_id": sample_id,
            "task_type": datasets["task_type"],
            "source_dataset": datasets["source_dataset"],
            "source_image_id": fetch_record["source_image_id"],
            "category": fetch_record["category"],
            "split": fetch_record.get("split"),
            "asset_version": int(fetch_record["asset_version"]),
            "image_path": fetch_record["image_path"],
            "mask_path": fetch_record.get("mask_path"),
            "metadata_path": fetch_record.get("metadata_path"),
            "image_uri": fetch_record.get("image_uri"),
            "mask_uri": fetch_record.get("mask_uri"),
            "storage_backend": fetch_record.get("storage_backend", "local"),
            "width": image.width,
            "height": image.height,
            "license": fetch_record.get("license", "unknown"),
            "checksum": current_checksum,
            "ingest_status": "success",
            "failure_reason": None,
            "ingest_timestamp": utc_now(),
            "lineage": {
                "fetch_record_id": fetch_record["record_id"],
                "ingest_config_hash": ingest_hash,
            },
        }
        append_ndjson(ingest_ledger_path, ingest_record)
        appended.append(ingest_record)
    write_latest_success_view(ingest_ledger_path, ingest_latest_path, "ingest_status")
    return appended


def validate_bronze(repo_root: Path, runtime: dict[str, Any]) -> list[dict[str, Any]]:
    storage = runtime["storage"]
    ingest_manifest = ingest_latest_view_path(repo_root, runtime)
    output_path = resolve_repo_path(repo_root, storage["bronze"]["manifests_dir"]) / "bronze_validation.ndjson"
    records = read_ndjson(ingest_manifest)
    results: list[dict[str, Any]] = []
    for record in records:
        image_path = resolve_repo_path(repo_root, record["image_path"])
        result = {
            "sample_id": record["sample_id"],
            "category": record["category"],
            "status": "accepted",
            "issues": [],
            "image_exists": image_path.exists(),
            "mask_exists": bool(record.get("mask_path") and resolve_repo_path(repo_root, record["mask_path"]).exists()),
            "validated_timestamp": utc_now(),
        }
        if record["ingest_status"] != "success":
            result["status"] = "rejected"
            result["issues"].append("ingest_failed")
        if not image_path.exists():
            result["status"] = "rejected"
            result["issues"].append("missing_image_path")
        else:
            try:
                image = open_image(image_path)
                if image.width != record.get("width") or image.height != record.get("height"):
                    result["issues"].append("dimension_mismatch")
                    result["status"] = "flagged"
                if not record.get("checksum"):
                    result["issues"].append("missing_checksum")
                    result["status"] = "flagged"
            except Exception:  # noqa: BLE001
                result["status"] = "rejected"
                result["issues"].append("unreadable_image")
        results.append(result)
    write_ndjson(output_path, results)
    summary = {
        "generated_at": utc_now(),
        "accepted": sum(1 for row in results if row["status"] == "accepted"),
        "flagged": sum(1 for row in results if row["status"] == "flagged"),
        "rejected": sum(1 for row in results if row["status"] == "rejected"),
    }
    report_path = resolve_repo_path(repo_root, storage["reports_dir"]) / "bronze_validation_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# Bronze Validation Report",
                f"- Generated: {summary['generated_at']}",
                f"- Accepted: {summary['accepted']}",
                f"- Flagged: {summary['flagged']}",
                f"- Rejected: {summary['rejected']}",
            ]
        ),
        encoding="utf-8",
    )
    return results


def normalize_to_silver(repo_root: Path, runtime: dict[str, Any]) -> list[dict[str, Any]]:
    storage = runtime["storage"]
    quality = runtime["quality"]
    ingest_records = {row["sample_id"]: row for row in read_ndjson(ingest_latest_view_path(repo_root, runtime))}
    validations = read_ndjson(resolve_repo_path(repo_root, storage["bronze"]["manifests_dir"]) / "bronze_validation.ndjson")
    output_path = silver_samples_manifest_path(repo_root, runtime)
    silver_hash = config_hash({"quality": quality, "storage": storage})
    results: list[dict[str, Any]] = []
    for validation in validations:
        if validation["status"] == "rejected":
            continue
        bronze = ingest_records[validation["sample_id"]]
        image = open_image(resolve_repo_path(repo_root, bronze["image_path"]))
        resized = resize_longest_side(image, int(quality["max_resize"]))
        gray = grayscale(resized)
        mask_available = bool(bronze.get("mask_path"))
        mask_ratio = None
        mask_relative = None
        isolated_relative = None
        if mask_available:
            mask = Image.open(resolve_repo_path(repo_root, bronze["mask_path"]))
            normalized = normalize_mask(mask, resized.size)
            mask_ratio = mask_coverage(normalized)
            view_features = extract_view_features(normalized)
            mask_path = resolve_repo_path(repo_root, storage["silver"]["masks_dir"]) / bronze["category"] / f"{bronze['sample_id']}_mask.png"
            save_image(mask_path, normalized)
            mask_relative = str(mask_path.relative_to(repo_root))
            isolated = isolate_subject(resized, normalized)
            isolated_path = resolve_repo_path(repo_root, storage["silver"]["isolated_dir"]) / bronze["category"] / f"{bronze['sample_id']}_isolated.png"
            save_image(isolated_path, isolated)
            isolated_relative = str(isolated_path.relative_to(repo_root))
        else:
            view_features = {}
        rgb_path = resolve_repo_path(repo_root, storage["silver"]["images_dir"]) / bronze["category"] / f"{bronze['sample_id']}_rgb.png"
        gray_path = resolve_repo_path(repo_root, storage["silver"]["images_dir"]) / bronze["category"] / f"{bronze['sample_id']}_gray.png"
        save_image(rgb_path, resized)
        save_image(gray_path, gray)
        grayscale_detected = detect_grayscale(resized)
        blur_value = blur_score(resized)
        aspect_ratio = resized.width / max(resized.height, 1)
        issues: list[str] = []
        status = "accepted"
        if resized.width < int(quality["minimum_width"]) or resized.height < int(quality["minimum_height"]):
            status = "rejected"
            issues.append("below_minimum_resolution")
        if mask_ratio is not None and mask_ratio < float(quality["low_mask_coverage_threshold"]):
            status = "flagged" if status != "rejected" else status
            issues.append("low_mask_coverage")
        if max(aspect_ratio, 1 / max(aspect_ratio, 1e-6)) > float(quality["extreme_aspect_ratio_threshold"]):
            status = "flagged" if status != "rejected" else status
            issues.append("extreme_aspect_ratio")
        if blur_value < float(quality["blur_variance_threshold"]):
            status = "flagged" if status != "rejected" else status
            issues.append("low_blur_variance")
        results.append(
            {
                "sample_id": bronze["sample_id"],
                "category": bronze["category"],
                "image_path": str(rgb_path.relative_to(repo_root)),
                "grayscale_path": str(gray_path.relative_to(repo_root)),
                "mask_path": mask_relative,
                "isolated_path": isolated_relative,
                "width": resized.width,
                "height": resized.height,
                "aspect_ratio": aspect_ratio,
                "mask_available": mask_available,
                "mask_coverage_ratio": mask_ratio,
                "grayscale_detected": grayscale_detected,
                "blur_score": blur_value,
                "duplicate_score": None,
                "view_features": view_features,
                "quality_status": status,
                "quality_flags": issues,
                "lineage": {"silver_config_hash": silver_hash, "bronze_sample_id": bronze["sample_id"]},
            }
        )
    write_ndjson(output_path, results)
    return results


def filter_viewpoints(repo_root: Path, runtime: dict[str, Any]) -> list[dict[str, Any]]:
    storage = runtime["storage"]
    viewpoints = runtime["viewpoints"]
    silver_records = read_ndjson(silver_samples_manifest_path(repo_root, runtime))
    output_path = silver_viewpoints_manifest_path(repo_root, runtime)
    backend = build_viewpoint_backend(viewpoints)
    viewpoint_hash = config_hash({"viewpoints": viewpoints})
    preliminary: list[dict[str, Any]] = []

    for silver in silver_records:
        if silver["quality_status"] == "rejected":
            preliminary.append(
                {
                    "sample_id": silver["sample_id"],
                    "category": silver["category"],
                    "image_path": silver["image_path"],
                    "isolated_path": silver["isolated_path"],
                    "view_bucket": "unknown",
                    "view_confidence": 0.0,
                    "view_margin": 0.0,
                    "view_scores": {},
                    "view_status": "rejected",
                    "view_flags": ["quality_rejected"],
                    "allowed_for_outline": False,
                    "quality_status": silver["quality_status"],
                    "view_features": silver.get("view_features", {}),
                    "lineage": {"silver_sample_id": silver["sample_id"], "viewpoint_config_hash": viewpoint_hash},
                }
            )
            continue

        result = backend.classify(silver, viewpoints)
        preliminary.append(
            {
                "sample_id": silver["sample_id"],
                "category": silver["category"],
                "image_path": silver["image_path"],
                "isolated_path": silver["isolated_path"],
                "view_bucket": result.bucket,
                "view_confidence": result.confidence,
                "view_margin": result.margin,
                "view_scores": result.scores,
                "view_status": result.status,
                "view_flags": result.flags,
                "allowed_for_outline": result.allowed_for_outline,
                "quality_status": silver["quality_status"],
                "view_features": silver.get("view_features", {}),
                "lineage": {"silver_sample_id": silver["sample_id"], "viewpoint_config_hash": viewpoint_hash},
            }
        )

    quotas = viewpoints.get("target_per_bucket", {})
    default_quota = int(quotas.get("default", 0) or 0)
    for category in {row["category"] for row in preliminary}:
        for bucket in viewpoints.get("allowed_buckets", []):
            target = int(quotas.get(category, {}).get(bucket, default_quota) if isinstance(quotas.get(category), dict) else default_quota)
            if target <= 0:
                continue
            accepted = [
                row
                for row in preliminary
                if row["category"] == category and row["view_bucket"] == bucket and row["view_status"] == "accepted"
            ]
            accepted.sort(key=lambda row: row["view_confidence"], reverse=True)
            for overflow in accepted[target:]:
                overflow["view_status"] = "blocked_quota"
                overflow["allowed_for_outline"] = False
                overflow["view_flags"] = list(overflow["view_flags"]) + ["quota_exceeded"]

    write_ndjson(output_path, preliminary)
    return preliminary


def run_outline_inference(repo_root: Path, runtime: dict[str, Any]) -> list[dict[str, Any]]:
    storage = runtime["storage"]
    models = runtime["models"]
    output_path = resolve_repo_path(repo_root, storage["gold"]["outlines_dir"]) / "outline_inference.ndjson"
    silver_records = read_ndjson(silver_viewpoints_manifest_path(repo_root, runtime))
    backend_name = models["active_backend"]
    backend = build_backend(backend_name, models["backends"][backend_name])
    results: list[dict[str, Any]] = []
    for silver in silver_records:
        if not silver.get("allowed_for_outline", False):
            continue
        image_path = silver["isolated_path"] or silver["image_path"]
        image = open_image(resolve_repo_path(repo_root, image_path))
        outline_path = resolve_repo_path(repo_root, storage["gold"]["outlines_dir"]) / silver["category"] / f"{silver['sample_id']}_outline.png"
        try:
            backend_result = backend.run(image, outline_path)
            status = "accepted"
            failure_reason = None
        except Exception as exc:  # noqa: BLE001
            backend_result = None
            status = "rejected"
            failure_reason = str(exc)
        results.append(
            {
                "sample_id": silver["sample_id"],
                "category": silver["category"],
                "status": status,
                "outline_path": str(outline_path.relative_to(repo_root)) if backend_result else None,
                "failure_reason": failure_reason,
                "view_bucket": silver["view_bucket"],
                "view_confidence": silver["view_confidence"],
                "view_status": silver["view_status"],
                "outline_model": backend_result.metadata if backend_result else {"model_name": backend_name},
                "lineage": {"silver_sample_id": silver["sample_id"], "viewpoint_sample_id": silver["sample_id"]},
            }
        )
    write_ndjson(output_path, results)
    return results


def refine_outlines(repo_root: Path, runtime: dict[str, Any]) -> list[dict[str, Any]]:
    storage = runtime["storage"]
    export = runtime["export"]
    output_path = resolve_repo_path(repo_root, storage["gold"]["trajectories_dir"]) / "refined_outlines.ndjson"
    outline_records = read_ndjson(resolve_repo_path(repo_root, storage["gold"]["outlines_dir"]) / "outline_inference.ndjson")
    results: list[dict[str, Any]] = []
    for outline_record in outline_records:
        if outline_record["status"] != "accepted":
            continue
        outline_path = resolve_repo_path(repo_root, outline_record["outline_path"])
        outline = Image.open(outline_path).convert("L")
        strokes = sample_outline_strokes(
            outline,
            max_strokes=int(export["max_strokes"]),
            max_points_per_stroke=int(export["max_points_per_stroke"]),
            min_points_per_stroke=int(export["min_points_per_stroke"]),
        )
        svg_path = resolve_repo_path(repo_root, storage["gold"]["svg_dir"]) / outline_record["category"] / f"{outline_record['sample_id']}.svg"
        trajectory_path = resolve_repo_path(repo_root, storage["gold"]["trajectories_dir"]) / outline_record["category"] / f"{outline_record['sample_id']}.json"
        write_svg(
            svg_path,
            strokes,
            width=outline.width,
            height=outline.height,
            stroke_width=int(export["svg_stroke_width"]),
            stroke_color=str(export["svg_stroke_color"]),
        )
        trajectory = {
            "coordinate_frame": export["coordinate_frame"],
            "canvas_width": outline.width,
            "canvas_height": outline.height,
            "stroke_count": len(strokes),
            "point_count": sum(len(stroke) for stroke in strokes),
            "bounds": {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0},
            "normalization_method": "width_height_normalized",
            "simplification_method": export["simplification_method"],
            "strokes": [
                {
                    "stroke_id": index,
                    "pen_state": "down",
                    "points": [{"x": x, "y": y} for x, y in stroke],
                }
                for index, stroke in enumerate(strokes)
            ],
        }
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory_path.write_text(json.dumps(trajectory, sort_keys=True, indent=2), encoding="utf-8")
        results.append(
            {
                "sample_id": outline_record["sample_id"],
                "category": outline_record["category"],
                "status": "accepted" if strokes else "flagged",
                "outline_path": outline_record["outline_path"],
                "svg_path": str(svg_path.relative_to(repo_root)),
                "trajectory_path": str(trajectory_path.relative_to(repo_root)),
                "stroke_count": trajectory["stroke_count"],
                "point_count": trajectory["point_count"],
                "lineage": {"outline_sample_id": outline_record["sample_id"]},
            }
        )
    write_ndjson(output_path, results)
    return results


def export_gold_ndjson(repo_root: Path, runtime: dict[str, Any]) -> list[dict[str, Any]]:
    storage = runtime["storage"]
    export = runtime["export"]
    ingest_records = {row["sample_id"]: row for row in read_ndjson(ingest_latest_view_path(repo_root, runtime))}
    validation_records = {row["sample_id"]: row for row in read_ndjson(resolve_repo_path(repo_root, storage["bronze"]["manifests_dir"]) / "bronze_validation.ndjson")}
    silver_records = {row["sample_id"]: row for row in read_ndjson(resolve_repo_path(repo_root, storage["silver"]["qa_dir"]) / "silver_samples.ndjson")}
    viewpoint_records = {row["sample_id"]: row for row in read_ndjson(silver_viewpoints_manifest_path(repo_root, runtime))}
    inference_records = {row["sample_id"]: row for row in read_ndjson(resolve_repo_path(repo_root, storage["gold"]["outlines_dir"]) / "outline_inference.ndjson")}
    refined_records = {row["sample_id"]: row for row in read_ndjson(resolve_repo_path(repo_root, storage["gold"]["trajectories_dir"]) / "refined_outlines.ndjson")}
    output_path = resolve_repo_path(repo_root, storage["gold"]["ndjson_dir"]) / "gold_samples.ndjson"
    results: list[dict[str, Any]] = []
    for sample_id, ingest in ingest_records.items():
        silver = silver_records.get(sample_id)
        viewpoint = viewpoint_records.get(sample_id)
        refined = refined_records.get(sample_id)
        inference = inference_records.get(sample_id)
        if not silver or not viewpoint or not refined or not inference:
            continue
        trajectory = json.loads(resolve_repo_path(repo_root, refined["trajectory_path"]).read_text(encoding="utf-8"))
        results.append(
            {
                "sample_id": sample_id,
                "task_type": ingest["task_type"],
                "category": ingest["category"],
                "source_dataset": ingest["source_dataset"],
                "source_refs": {
                    "source_image_id": ingest["source_image_id"],
                    "source_url": ingest.get("image_uri"),
                    "license": ingest["license"],
                },
                "bronze_refs": {
                    "image_path": ingest["image_path"],
                    "mask_path": ingest["mask_path"],
                    "validation_status": validation_records.get(sample_id, {}).get("status"),
                },
                "silver_refs": {
                    "image_path": silver["image_path"],
                    "grayscale_path": silver["grayscale_path"],
                    "mask_path": silver["mask_path"],
                    "isolated_path": silver["isolated_path"],
                },
                "gold_refs": {
                    "outline_path": inference["outline_path"],
                    "svg_path": refined["svg_path"],
                    "trajectory_path": refined["trajectory_path"],
                },
                "outline_model": inference["outline_model"],
                "qa_scores": {
                    "mask_coverage_ratio": silver["mask_coverage_ratio"],
                    "blur_score": silver["blur_score"],
                    "view_confidence": viewpoint["view_confidence"],
                    "stroke_count": refined["stroke_count"],
                    "point_count": refined["point_count"],
                },
                "status": refined["status"],
                "view_bucket": viewpoint["view_bucket"],
                "view_confidence": viewpoint["view_confidence"],
                "view_status": viewpoint["view_status"],
                "split": ingest["split"],
                "lineage": {
                    "ndjson_schema_version": export["ndjson_schema_version"],
                    "bronze": ingest["lineage"],
                    "silver": silver["lineage"],
                    "viewpoint": viewpoint["lineage"],
                    "outline": inference["lineage"],
                    "refine": refined["lineage"],
                },
                "trajectory": trajectory,
            }
        )
    write_ndjson(output_path, results)
    return results


def generate_dataset_report(repo_root: Path, runtime: dict[str, Any]) -> dict[str, Any]:
    storage = runtime["storage"]
    viewpoints = runtime["viewpoints"]
    records = read_ndjson(resolve_repo_path(repo_root, storage["gold"]["ndjson_dir"]) / "gold_samples.ndjson")
    viewpoint_records = read_ndjson(silver_viewpoints_manifest_path(repo_root, runtime))
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    viewpoint_status_by_category: dict[str, Counter[str]] = defaultdict(Counter)
    viewpoint_bucket_by_category: dict[str, Counter[str]] = defaultdict(Counter)
    blocked_reasons: Counter[str] = Counter()
    model_counter: Counter[str] = Counter()
    stroke_counts: list[int] = []
    point_counts: list[int] = []
    for record in records:
        by_category[record["category"]][record["status"]] += 1
        model_counter[record["outline_model"].get("model_name", "unknown")] += 1
        stroke_counts.append(int(record["trajectory"]["stroke_count"]))
        point_counts.append(int(record["trajectory"]["point_count"]))
    for record in viewpoint_records:
        viewpoint_status_by_category[record["category"]][record["view_status"]] += 1
        viewpoint_bucket_by_category[record["category"]][record["view_bucket"]] += 1
        for flag in record.get("view_flags", []):
            blocked_reasons[flag] += 1

    quotas = viewpoints.get("target_per_bucket", {})
    default_quota = int(quotas.get("default", 0) or 0)
    viewpoint_shortfalls: dict[str, dict[str, int]] = {}
    for category in viewpoint_bucket_by_category:
        shortfalls: dict[str, int] = {}
        category_quota = quotas.get(category, {})
        for bucket in viewpoints.get("allowed_buckets", []):
            target = int(category_quota.get(bucket, default_quota) if isinstance(category_quota, dict) else default_quota)
            if target <= 0:
                continue
            accepted = sum(
                1
                for row in viewpoint_records
                if row["category"] == category and row["view_bucket"] == bucket and row["view_status"] == "accepted"
            )
            shortfall = max(target - accepted, 0)
            if shortfall:
                shortfalls[bucket] = shortfall
        if shortfalls:
            viewpoint_shortfalls[category] = shortfalls

    summary = {
        "generated_at": utc_now(),
        "total_records": len(records),
        "categories": {category: dict(counter) for category, counter in by_category.items()},
        "viewpoint_status_by_category": {category: dict(counter) for category, counter in viewpoint_status_by_category.items()},
        "viewpoint_bucket_by_category": {category: dict(counter) for category, counter in viewpoint_bucket_by_category.items()},
        "blocked_view_flags": dict(blocked_reasons),
        "viewpoint_shortfalls": viewpoint_shortfalls,
        "model_backend_usage": dict(model_counter),
        "average_stroke_count": (sum(stroke_counts) / len(stroke_counts)) if stroke_counts else 0.0,
        "average_point_count": (sum(point_counts) / len(point_counts)) if point_counts else 0.0,
        "unresolved_flagged_samples": sum(1 for record in records if record["status"] == "flagged"),
    }
    report_root = resolve_repo_path(repo_root, storage["reports_dir"])
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "dataset_report.json").write_text(json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8")
    lines = ["# Dataset Report", f"- Generated: {summary['generated_at']}", f"- Total records: {summary['total_records']}"]
    for category, counts in summary["categories"].items():
        lines.append(f"- {category}: {counts}")
    for category, counts in summary["viewpoint_status_by_category"].items():
        lines.append(f"- {category} viewpoint status: {counts}")
    for category, counts in summary["viewpoint_bucket_by_category"].items():
        lines.append(f"- {category} viewpoint buckets: {counts}")
    lines.append(f"- Blocked view flags: {summary['blocked_view_flags']}")
    lines.append(f"- Viewpoint shortfalls: {summary['viewpoint_shortfalls']}")
    lines.append(f"- Model backend usage: {summary['model_backend_usage']}")
    lines.append(f"- Average stroke count: {summary['average_stroke_count']:.2f}")
    lines.append(f"- Average point count: {summary['average_point_count']:.2f}")
    lines.append(f"- Unresolved flagged samples: {summary['unresolved_flagged_samples']}")
    (report_root / "dataset_report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def fetch_main() -> int:
    repo_root, runtime = load_runtime(parse_common_args("Fetch Open Images assets into bronze storage."))
    fetch_openimages(repo_root, runtime)
    return 0


def ingest_main() -> int:
    repo_root, runtime = load_runtime(parse_common_args("Ingest fetched bronze Open Images assets into the canonical manifest."))
    ingest_openimages(repo_root, runtime)
    return 0


def validate_main() -> int:
    repo_root, runtime = load_runtime(parse_common_args("Validate bronze ingest records."))
    validate_bronze(repo_root, runtime)
    return 0


def normalize_main() -> int:
    repo_root, runtime = load_runtime(parse_common_args("Normalize bronze records into silver assets."))
    normalize_to_silver(repo_root, runtime)
    return 0


def outline_main() -> int:
    repo_root, runtime = load_runtime(parse_common_args("Generate outline proposals from silver assets."))
    run_outline_inference(repo_root, runtime)
    return 0


def filter_viewpoints_main() -> int:
    repo_root, runtime = load_runtime(parse_common_args("Filter silver samples into drawing-friendly viewpoint buckets."))
    filter_viewpoints(repo_root, runtime)
    return 0


def refine_main() -> int:
    repo_root, runtime = load_runtime(parse_common_args("Refine outline proposals into SVG and trajectories."))
    refine_outlines(repo_root, runtime)
    return 0


def export_main() -> int:
    repo_root, runtime = load_runtime(parse_common_args("Export canonical gold NDJSON records."))
    export_gold_ndjson(repo_root, runtime)
    return 0


def report_main() -> int:
    repo_root, runtime = load_runtime(parse_common_args("Generate dataset summary reports."))
    generate_dataset_report(repo_root, runtime)
    return 0
