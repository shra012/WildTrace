from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from wildtrace.io_utils import resolve_repo_path
from wildtrace.pipeline import load_runtime, parse_common_args


def bronze_manifests_dir(repo_root: Path, runtime: dict[str, Any]) -> Path:
    return resolve_repo_path(repo_root, runtime["storage"]["bronze"]["manifests_dir"])


def _storage_path(section: dict[str, Any], preferred_key: str, legacy_key: str | None = None) -> str:
    if preferred_key in section:
        return section[preferred_key]
    if legacy_key and legacy_key in section:
        return section[legacy_key]
    raise KeyError(f"Missing storage path config: {preferred_key}")


def silver_checkpoints_dir(repo_root: Path, runtime: dict[str, Any]) -> Path:
    silver = runtime["storage"]["silver"]
    return resolve_repo_path(repo_root, _storage_path(silver, "checkpoints_dir", "qa_dir"))


def gold_records_dir(repo_root: Path, runtime: dict[str, Any]) -> Path:
    gold = runtime["storage"]["gold"]
    return resolve_repo_path(repo_root, _storage_path(gold, "records_dir", "ndjson_dir"))


def bronze_curated_manifest_path(repo_root: Path, runtime: dict[str, Any]) -> Path:
    return bronze_manifests_dir(repo_root, runtime) / "bronze_curated.ndjson"


def bronze_accepted_manifest_path(repo_root: Path, runtime: dict[str, Any]) -> Path:
    return bronze_manifests_dir(repo_root, runtime) / "bronze_accepted.ndjson"


def silver_samples_manifest_path(repo_root: Path, runtime: dict[str, Any]) -> Path:
    return silver_checkpoints_dir(repo_root, runtime) / "silver_samples.ndjson"


def silver_subjects_manifest_path(repo_root: Path, runtime: dict[str, Any]) -> Path:
    return silver_checkpoints_dir(repo_root, runtime) / "silver_subjects.ndjson"


def line_diagram_attempts_manifest_path(repo_root: Path, runtime: dict[str, Any]) -> Path:
    return silver_checkpoints_dir(repo_root, runtime) / "line_diagram_attempts.ndjson"


def validated_diagrams_manifest_path(repo_root: Path, runtime: dict[str, Any]) -> Path:
    return silver_checkpoints_dir(repo_root, runtime) / "validated_diagrams.ndjson"


def selected_diagrams_manifest_path(repo_root: Path, runtime: dict[str, Any]) -> Path:
    return silver_checkpoints_dir(repo_root, runtime) / "selected_diagrams.ndjson"


def diagram_trajectories_manifest_path(repo_root: Path, runtime: dict[str, Any]) -> Path:
    return resolve_repo_path(repo_root, runtime["storage"]["gold"]["trajectories_dir"]) / "diagram_trajectories.ndjson"


def run_stage_main(description: str, stage_fn: Callable[[Path, dict[str, Any]], object]) -> int:
    repo_root, runtime = load_runtime(parse_common_args(description))
    stage_fn(repo_root, runtime)
    return 0
