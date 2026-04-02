from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import yaml


DEFAULT_CONFIG_FILES = (
    "datasets.yaml",
    "storage.yaml",
    "quality.yaml",
    "models.yaml",
    "export.yaml",
    "viewpoints.yaml",
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping.")
    return data


def _resolve_env_placeholders(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_env_placeholders(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env_placeholders(item) for item in value]
    if not isinstance(value, str):
        return value
    if not value.startswith("${") or not value.endswith("}"):
        return value
    inner = value[2:-1]
    if ":-" in inner:
        env_name, default = inner.split(":-", 1)
        return os.getenv(env_name, default)
    return os.environ[inner]


def load_runtime_config(repo_root: Path, config_dir: Path | None = None) -> dict[str, Any]:
    config_root = config_dir or repo_root / "configs"
    load_dotenv(repo_root / ".env")
    runtime: dict[str, Any] = {"repo_root": str(repo_root)}
    for filename in DEFAULT_CONFIG_FILES:
        runtime[filename.removesuffix(".yaml")] = _resolve_env_placeholders(load_yaml(config_root / filename))
    return runtime
