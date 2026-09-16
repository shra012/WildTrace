"""Configuration loading with project-relative path resolution."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path, project_root: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    root = Path(project_root).resolve()
    if not config_path.is_absolute():
        config_path = root / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Configuration is not a mapping: {config_path}")
    config["_config_path"] = str(config_path.resolve())
    config["_project_root"] = str(root)
    for section, key in (
        ("project", "trajectory_path"),
        ("project", "output_dir"),
        ("robot", "urdf_path"),
        ("robot", "usd_path"),
        ("robot", "robot_description_path"),
        ("recording", "demo_path"),
        ("training", "checkpoint_path"),
        ("training", "tensorboard_dir"),
    ):
        value = Path(config[section][key])
        config[section][key] = str(value if value.is_absolute() else (root / value).resolve())
    return config

