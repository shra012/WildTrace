"""One-shot scene builder, driven by config/xarm7_drawing.yaml -- the same
config file mcp_drawing_controller.py's robot/drawing constants mirror.
Run via reload_script (executes once, top to bottom; no setup()/compute(),
so it is NOT a ScriptNode -- reload_script's "standalone controller" mode).

Builds the config-driven physical parts: the project's own xArm7-with-pen
USD at the configured prim path, and a paper/table/stand stage sized from
drawing.surface_*/paper_top_z_m so they can never drift out of sync with
the values the controller solves IK against. Table color/lighting are this
script's own reasonable defaults -- the config has no opinion on stage
aesthetics, only on the physically load-bearing drawing geometry.

Physics scene, camera, and the Action Graph are deliberately NOT built
here -- those go through the isaac-sim MCP server's own tools
(create_physics_scene, create_camera adopting an existing viewport camera,
create_action_graph), which already handle Kit-specific render-pipeline and
graph-wiring behavior this script has no business reimplementing in raw
pxr/omni.graph calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(r"\\wsl.localhost\Ubuntu-24.04\home\shravan\Workspace\WildTrace\isaac-sim")
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from project_config import load_config

CONFIG = load_config(PROJECT_ROOT / "config" / "xarm7_drawing.yaml", PROJECT_ROOT)
ROBOT = CONFIG["robot"]
DRAWING = CONFIG["drawing"]


def _load_robot():
    from isaacsim.core.utils.stage import add_reference_to_stage

    usd_path = str((PROJECT_ROOT / "assets" / "xarm7" / "xarm7.usd").resolve())
    add_reference_to_stage(usd_path, ROBOT["prim_path"])


def _create_stage_dressing():
    from pxr import Gf, UsdGeom
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    center_x, center_y = DRAWING["surface_center_xy_m"]
    size_x, size_y = DRAWING["surface_size_xy_m"]
    paper_top_z = float(DRAWING["paper_top_z_m"])
    paper_thickness = 0.01
    margin = 0.02  # matches run_two_stroke_drawing.py's FixedCuboid margin (+0.04 total)

    def _cube(path, position, scale, color):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr().Set(2.0)
        xform = UsdGeom.Xformable(cube)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        xform.AddScaleOp().Set(Gf.Vec3f(*scale))
        cube.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])

    # Drawing surface (paper): top face at paper_top_z_m, matching every
    # pen_down_z_m/pen_up_z_m/approach_height_m the controller solves IK against.
    _cube(
        "/World/DrawingSurface",
        (center_x, center_y, paper_top_z - paper_thickness / 2),
        ((size_x + margin) / 2, (size_y + margin) / 2, paper_thickness / 2),
        (0.92, 0.92, 0.88),
    )
    # Low tabletop the robot base and paper stand sit on, top flush at z=0
    # (the robot's own base mounting plane).
    _cube("/World/TableTop", (0.2, 0.0, -0.015), (0.55, 0.45, 0.015), (0.45, 0.3, 0.18))
    # Paper stand: rises from the tabletop (z=0) to the paper's underside.
    _cube(
        "/World/PaperStand",
        (center_x, center_y, (paper_top_z - paper_thickness) / 2),
        (0.03, 0.025, (paper_top_z - paper_thickness) / 2),
        (0.25, 0.25, 0.27),
    )


def _create_lights():
    from pxr import Gf, UsdGeom, UsdLux
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    center_x, center_y = DRAWING["surface_center_xy_m"]

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr().Set(1000.0)

    distant = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    distant.CreateIntensityAttr().Set(3000.0)
    distant_xform = UsdGeom.Xformable(distant)
    distant_xform.ClearXformOpOrder()
    distant_xform.AddTranslateOp().Set(Gf.Vec3d(center_x, center_y, 2.0))

    rect = UsdLux.RectLight.Define(stage, "/World/DrawingAreaLight")
    rect.CreateIntensityAttr().Set(5000.0)
    rect.CreateColorAttr().Set(Gf.Vec3f(1.0, 0.98, 0.95))
    rect_xform = UsdGeom.Xformable(rect)
    rect_xform.ClearXformOpOrder()
    rect_xform.AddTranslateOp().Set(Gf.Vec3d(center_x, center_y, 0.7))
    rect_xform.AddRotateXYZOp().Set(Gf.Vec3f(180.0, 0.0, 0.0))


def main():
    _load_robot()
    _create_stage_dressing()
    _create_lights()
    print(f"[SceneSetup] Built from {CONFIG['_config_path']}")
    print(f"[SceneSetup] Robot prim: {ROBOT['prim_path']}, paper top z: {DRAWING['paper_top_z_m']}")
    print("[SceneSetup] Physics scene, camera, and Action Graph still need the MCP tools "
          "(create_physics_scene, create_camera on an existing viewport camera, create_action_graph).")


main()
