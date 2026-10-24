"""Run inside Isaac Sim (reload_script): add the video cameras for the active MCP scene.

/World/VideoCams/WideFront looks at the paper from the front and above with
the whole arm in frame; /World/VideoCams/PaperTop is a close-up of the sheet
from low and to the side (straight down, the wrist hides it). Both are placed from the scene config in outputs/mcp_sessions/scene.json
(see src/mcp_scene.py), so they frame the lab-arm twin and the default sim alike.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(r"\\wsl.localhost\Ubuntu-24.04\home\shravan\Workspace\WildTrace\isaac-sim")
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

DRAWING = load_config(scene_config_path(PROJECT_ROOT), PROJECT_ROOT)["drawing"]


def _camera(stage, path, eye, target, focal_length_mm):
    from pxr import Gf, UsdGeom

    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr().Set(float(focal_length_mm))
    camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
    # A USD camera looks down its -Z axis with +Y up; the inverse of a
    # look-at view matrix is exactly that pose in the world.
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0.0, 0.0, 1.0))
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
    print(f"[VideoCams] cameras placed around paper centre ({cx:.3f}, {cy:.3f}, {paper_z:.3f})")


main()
