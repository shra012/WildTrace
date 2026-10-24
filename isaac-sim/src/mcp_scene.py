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


# How far the pen tip sticks out past the calibrated TCP. The calibration
# measured the TCP (172 mm, fingertip-pad centre) at 49.61 mm with the pen on
# the paper, so the tool frame and every drawing height stay as measured; the
# protrusion only places the visible pen tip, the sheet and the table d lower.
# Unmeasured on the lab arm so far; 20-60 mm is the plausible range.
DEFAULT_PEN_PROTRUSION_M = 0.035
PEN_PROTRUSION_RANGE_M = (0.020, 0.060)


def pen_protrusion_m(project_root: Path) -> float:
    try:
        value = json.loads(scene_file(Path(project_root)).read_text(encoding="utf-8")).get("pen_protrusion_m")
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        value = None
    # Absent means the scene's IK frame already is the pen tip (the original
    # sim's stylus); only the lab-arm twin, where it is the gripper TCP, sets it.
    return 0.0 if value is None else float(value)


def paper_surface_z(drawing: dict, protrusion_m: float) -> float:
    """Where the sheet really is: the calibrated pen-down TCP height minus the protrusion."""
    return float(drawing["pen_down_z_m"]) - float(protrusion_m)
