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

import importlib  # noqa: E402

import mcp_scene  # noqa: E402

# Kit keeps modules imported across reload_script/graph rebuilds; reload so
# edits to src/mcp_scene.py take effect without restarting Isaac Sim.
importlib.reload(mcp_scene)
from mcp_scene import paper_surface_z, pen_protrusion_m, scene_config_path

# The MCP server picks the config (default sim or the lab-arm twin) via
# outputs/mcp_sessions/scene.json; see src/mcp_scene.py.
CONFIG = load_config(scene_config_path(PROJECT_ROOT), PROJECT_ROOT)
ROBOT = CONFIG["robot"]
DRAWING = CONFIG["drawing"]
# 0 for the original sim (its IK frame is the stylus tip); the lab-arm twin's
# IK frame is the calibrated gripper TCP and the pen sticks out past it.
PEN_PROTRUSION_M = pen_protrusion_m(PROJECT_ROOT)


def _load_robot():
    from isaacsim.core.utils.stage import add_reference_to_stage

    add_reference_to_stage(ROBOT["usd_path"], ROBOT["prim_path"])


def _create_stage_dressing():
    """Paper, table, floor and robot pedestal, all visual only.

    With a pen protrusion d, the sheet sits at the calibrated pen-down TCP
    height minus d and lies directly on the tabletop. The floor drops below
    the table and a pedestal carries the robot base, which stays at z = 0
    (the calibration frame), so the sheet may end up below the base plane.
    """
    from pxr import Gf, UsdGeom
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    center_x, center_y = DRAWING["surface_center_xy_m"]
    size_x, size_y = DRAWING["surface_size_xy_m"]
    margin = 0.02  # matches run_two_stroke_drawing.py's FixedCuboid margin (+0.04 total)

    def _cube(path, center, half_extents, color):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr().Set(2.0)
        xform = UsdGeom.Xformable(cube)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*center))
        xform.AddScaleOp().Set(Gf.Vec3f(*half_extents))
        cube.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])

    if PEN_PROTRUSION_M > 0:
        paper_top = paper_surface_z(DRAWING, PEN_PROTRUSION_M)
        paper_thickness = 0.001
    else:
        paper_top = float(DRAWING["paper_top_z_m"])
        paper_thickness = 0.01
    table_top = paper_top - paper_thickness
    table_thickness = 0.03
    floor_z = min(0.0, table_top) - 0.70

    # Sheet (top face at the drawing surface).
    _cube(
        "/World/DrawingSurface",
        (center_x, center_y, paper_top - paper_thickness / 2),
        ((size_x + margin) / 2, (size_y + margin) / 2, paper_thickness / 2),
        (0.97, 0.97, 0.95),
    )
    # Tabletop the sheet lies on; it stops short of the robot base.
    table_x0, table_x1, table_y0, table_y1 = 0.20, 0.85, -0.40, 0.50
    _cube(
        "/World/TableTop",
        ((table_x0 + table_x1) / 2, (table_y0 + table_y1) / 2, table_top - table_thickness / 2),
        ((table_x1 - table_x0) / 2, (table_y1 - table_y0) / 2, table_thickness / 2),
        (0.45, 0.3, 0.18),
    )
    leg_height = table_top - table_thickness - floor_z
    for index, (lx, ly) in enumerate(((table_x0 + 0.04, table_y0 + 0.04), (table_x1 - 0.04, table_y0 + 0.04),
                                      (table_x0 + 0.04, table_y1 - 0.04), (table_x1 - 0.04, table_y1 - 0.04))):
        _cube(f"/World/TableLeg{index}", (lx, ly, floor_z + leg_height / 2), (0.02, 0.02, leg_height / 2), (0.35, 0.23, 0.14))
    # Pedestal from the floor up to the robot base mounting plane (z = 0).
    _cube("/World/RobotPedestal", (0.0, 0.0, floor_z / 2), (0.09, 0.09, -floor_z / 2), (0.3, 0.3, 0.32))
    # Floor: move the physics scene's ground plane under the table legs.
    ground = stage.GetPrimAtPath("/World/groundPlane")
    if ground and ground.IsValid():
        ground_xform = UsdGeom.Xformable(ground)
        ground_xform.ClearXformOpOrder()
        ground_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, floor_z))
    # The old paper stand is replaced by the table.
    if stage.GetPrimAtPath("/World/PaperStand"):
        stage.RemovePrim("/World/PaperStand")


def _add_pen_protrusion():
    """Visible pen tip sticking out d past the gripper TCP, rigidly on the pen link."""
    if PEN_PROTRUSION_M <= 0:
        return
    from pxr import Gf, UsdGeom
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    pen_body = next(
        (prim for prim in stage.Traverse()
         if str(prim.GetPath()).startswith(ROBOT["prim_path"]) and prim.GetName() == "pen_body"),
        None,
    )
    if pen_body is None:
        print("[SceneSetup] No pen_body prim; pen protrusion not drawn")
        return
    tip = UsdGeom.Cylinder.Define(stage, pen_body.GetPath().AppendChild("PenProtrusion"))
    tip.CreateAxisAttr().Set("Z")
    tip.CreateHeightAttr().Set(float(PEN_PROTRUSION_M))
    tip.CreateRadiusAttr().Set(0.005)
    tip.CreateDisplayColorAttr().Set([Gf.Vec3f(0.1, 0.2, 0.9)])
    xform = UsdGeom.Xformable(tip)
    xform.ClearXformOpOrder()
    # The TCP frame's +Z points out of the gripper toward the paper.
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, PEN_PROTRUSION_M / 2))


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


def _prepare_drives():
    """Author the same PhysX drive type, gains and solver iterations the
    headless runner authors, so an MCP run matches the evaluated sim."""
    import omni.usd

    from articulation_control import prepare_articulation_solver
    from xarm7_loader import find_articulation_root

    stage = omni.usd.get_context().get_stage()
    prepare_articulation_solver(stage, find_articulation_root(stage, ROBOT["prim_path"]), ROBOT)


def main():
    _load_robot()
    _prepare_drives()
    _add_pen_protrusion()
    _create_stage_dressing()
    _create_lights()
    print(f"[SceneSetup] Built from {CONFIG['_config_path']}")
    print(f"[SceneSetup] Robot prim: {ROBOT['prim_path']}, paper top z: {DRAWING['paper_top_z_m']}, "
          f"pen protrusion: {PEN_PROTRUSION_M * 1e3:.0f} mm")
    print("[SceneSetup] Physics scene, camera, and Action Graph still need the MCP tools "
          "(create_physics_scene, create_camera on an existing viewport camera, create_action_graph).")


main()
