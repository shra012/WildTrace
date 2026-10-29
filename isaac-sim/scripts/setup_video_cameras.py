"""Run inside Isaac Sim (reload_script): add the video cameras for the active MCP scene.

/World/VideoCams/WideFront looks at the paper from the front and above with
the whole arm in frame; /World/VideoCams/PaperTop is a close-up of the sheet
from low and to the side (straight down, the wrist hides it). Both are placed from the scene config in outputs/mcp_sessions/scene.json
(see src/mcp_scene.py), so they frame the lab-arm twin and the default sim alike.
"""
from __future__ import annotations

import sys
from pathlib import Path

def _wt_project_root():
    """isaac-sim/ as Isaac (Windows) sees it, over \\\\wsl.localhost. Isaac runs this file as a script,
    so __file__ can be missing: WILDTRACE_ISAAC_SIM_ROOT overrides, then the script's own location,
    then the WildTrace checkout in the WSL user's home (user name from wsl.exe)."""
    import os
    import subprocess
    from pathlib import Path

    candidates = [os.environ.get("WILDTRACE_ISAAC_SIM_ROOT")]
    try:
        candidates.append(str(Path(__file__).resolve().parents[1]))
    except (NameError, IndexError):
        pass
    for candidate in candidates:
        if candidate and (Path(candidate) / "src" / "project_config.py").exists():
            return Path(candidate)
    user = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04", "-e", "whoami"], capture_output=True, text=True,
                          timeout=30).stdout.strip()
    return Path(rf"\\wsl.localhost\Ubuntu-24.04\home\{user}\Workspace\WildTrace\isaac-sim")


PROJECT_ROOT = _wt_project_root()
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import importlib  # noqa: E402

import mcp_scene  # noqa: E402

# Kit keeps modules imported across reload_script/graph rebuilds; reload so
# edits to src/mcp_scene.py take effect without restarting Isaac Sim.
importlib.reload(mcp_scene)
from mcp_scene import paper_surface_z, pen_protrusion_m, scene_config_path  # noqa: E402
from project_config import load_config  # noqa: E402

CONFIG = load_config(scene_config_path(PROJECT_ROOT), PROJECT_ROOT)
DRAWING = CONFIG["drawing"]
HORIZONTAL_APERTURE_MM = 20.955  # USD camera default


def _camera(stage, path, eye, target, focal_length_mm, up=(0.0, 0.0, 1.0)):
    from pxr import Gf, UsdGeom

    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr().Set(float(focal_length_mm))
    camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
    # A USD camera looks down its -Z axis with +Y up; the inverse of a
    # look-at view matrix is exactly that pose in the world.
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up))
    xform = UsdGeom.Xformable(camera)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(view.GetInverse())


def main():
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    cx, cy = (float(v) for v in DRAWING["surface_center_xy_m"])
    protrusion = pen_protrusion_m(PROJECT_ROOT)
    paper_z = paper_surface_z(DRAWING, protrusion) if protrusion > 0 else float(DRAWING["paper_top_z_m"])
    _camera(stage, "/World/VideoCams/WideFront", (cx + 0.75, cy - 0.55, paper_z + 0.55), (cx - 0.12, cy, paper_z + 0.12), 18.0)
    # Low and to the side: straight above, the wrist hides the sheet.
    _camera(stage, "/World/VideoCams/PaperTop", (cx + 0.30, cy - 0.24, paper_z + 0.20), (cx - 0.01, cy + 0.01, paper_z), 24.0)
    vla_camera = CONFIG.get("vla_camera")
    if vla_camera:
        # Straight down, image up = world +y, as src/vla_features.py renders it. The 16:9 frame fits
        # the horizontal aperture, so the vertical one (and the centre square) spans view_size_m.
        vx, vy = (float(v) for v in vla_camera["center_xy_m"])
        height = float(vla_camera["height_above_paper_m"])
        focal = HORIZONTAL_APERTURE_MM * 9.0 / 16.0 * height / float(vla_camera["view_size_m"])
        _camera(stage, "/World/VideoCams/PaperOverhead", (vx, vy, paper_z + height), (vx, vy, paper_z), focal,
                up=(0.0, 1.0, 0.0))
    print(f"[VideoCams] cameras placed around paper centre ({cx:.3f}, {cy:.3f}, {paper_z:.3f})")


main()
