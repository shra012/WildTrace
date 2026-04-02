from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from wildtrace.config import load_runtime_config
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
        mask_path = resolve_repo_path(repo_root, record["mask_path"])
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
            fetch_record = {
                "record_id": stable_sample_id(record["sample_id"], str(version), str(exc)),
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
                "failure_reason": str(exc),
                "fetched_at": utc_now(),
            }
        append_ndjson(ledger_path, fetch_record)
        fetched_records.append(fetch_record)
    write_latest_success_view(ledger_path, latest_view, "fetch_status")
    return fetched_records


def fetch_main() -> int:
    repo_root, runtime = load_runtime(parse_common_args("Fetch Open Images assets into bronze storage."))
    fetch_openimages(repo_root, runtime)
    return 0
