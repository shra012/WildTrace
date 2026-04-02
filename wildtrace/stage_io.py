from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from wildtrace.io_utils import resolve_repo_path
from wildtrace.pipeline import load_runtime, parse_common_args


def bronze_manifests_dir(repo_root: Path, runtime: dict[str, Any]) -> Path:
    return resolve_repo_path(repo_root, runtime["storage"]["bronze"]["manifests_dir"])


def silver_qa_dir(repo_root: Path, runtime: dict[str, Any]) -> Path:
    return resolve_repo_path(repo_root, runtime["storage"]["silver"]["qa_dir"])


def bronze_curated_manifest_path(repo_root: Path, runtime: dict[str, Any]) -> Path:
    return bronze_manifests_dir(repo_root, runtime) / "bronze_curated.ndjson"


def bronze_accepted_manifest_path(repo_root: Path, runtime: dict[str, Any]) -> Path:
    return bronze_manifests_dir(repo_root, runtime) / "bronze_accepted.ndjson"


def silver_samples_manifest_path(repo_root: Path, runtime: dict[str, Any]) -> Path:
    return silver_qa_dir(repo_root, runtime) / "silver_samples.ndjson"


def silver_subjects_manifest_path(repo_root: Path, runtime: dict[str, Any]) -> Path:
    return silver_qa_dir(repo_root, runtime) / "silver_subjects.ndjson"


def line_diagram_attempts_manifest_path(repo_root: Path, runtime: dict[str, Any]) -> Path:
    return silver_qa_dir(repo_root, runtime) / "line_diagram_attempts.ndjson"


def validated_diagrams_manifest_path(repo_root: Path, runtime: dict[str, Any]) -> Path:
    return silver_qa_dir(repo_root, runtime) / "validated_diagrams.ndjson"


def selected_diagrams_manifest_path(repo_root: Path, runtime: dict[str, Any]) -> Path:
    return silver_qa_dir(repo_root, runtime) / "selected_diagrams.ndjson"


def diagram_trajectories_manifest_path(repo_root: Path, runtime: dict[str, Any]) -> Path:
    return resolve_repo_path(repo_root, runtime["storage"]["gold"]["trajectories_dir"]) / "diagram_trajectories.ndjson"


def run_stage_main(description: str, stage_fn: Callable[[Path, dict[str, Any]], object]) -> int:
    repo_root, runtime = load_runtime(parse_common_args(description))
    stage_fn(repo_root, runtime)
    return 0
