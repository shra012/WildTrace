from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_FILES = (
    "datasets.yaml",
    "storage.yaml",
    "quality.yaml",
    "models.yaml",
    "export.yaml",
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping.")
    return data


def load_runtime_config(repo_root: Path, config_dir: Path | None = None) -> dict[str, Any]:
    config_root = config_dir or repo_root / "configs"
    runtime: dict[str, Any] = {"repo_root": str(repo_root)}
    for filename in DEFAULT_CONFIG_FILES:
        runtime[filename.removesuffix(".yaml")] = load_yaml(config_root / filename)
    return runtime
