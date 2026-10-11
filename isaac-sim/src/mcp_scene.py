"""Which config the MCP-driven Isaac scene uses.

The wildtrace-sim MCP server writes outputs/mcp_sessions/scene.json before it
builds the scene; scripts/setup_scene.py and scripts/mcp_drawing_controller.py
both read it at import, so the scene, the controller and the run report always
agree. Without the file the original sim config is used, so the existing
"draw a <category>" flow is unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_CONFIG = "config/xarm7_drawing.yaml"


def scene_file(project_root: Path) -> Path:
    return Path(project_root) / "outputs" / "mcp_sessions" / "scene.json"


def scene_config_path(project_root: Path) -> Path:
    """Absolute path of the config the current MCP scene was built from."""
    root = Path(project_root)
    try:
        relative = json.loads(scene_file(root).read_text(encoding="utf-8")).get("config", DEFAULT_CONFIG)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        relative = DEFAULT_CONFIG
    path = Path(relative)
    return path if path.is_absolute() else root / path
