"""Launch Isaac Sim, verify required APIs, import xArm 7, and inspect articulation."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))
PORTABLE_ROOT = PROJECT_ROOT / "outputs" / "isaac_portable"
PORTABLE_ROOT.mkdir(parents=True, exist_ok=True)
if "--portable-root" not in sys.argv:
    sys.argv.extend(["--portable-root", str(PORTABLE_ROOT)])


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/xarm7_drawing.yaml")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--portable-root", default=str(PORTABLE_ROOT), help=argparse.SUPPRESS)
    parser.add_argument("--skip-import", action="store_true", help="Inspect an existing USD instead of regenerating it")
    return parser.parse_args()


ARGS = _parse_args()

# Required ordering: no Omniverse/Isaac runtime imports before this instance.
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": ARGS.headless, "width": 1280, "height": 720})


def _installed_version() -> tuple[str, str]:
    candidates = []
    isaac_path = os.environ.get("ISAAC_PATH")
    if isaac_path:
        candidates.append(Path(isaac_path) / "VERSION")
    executable = Path(sys.executable).resolve()
    candidates.extend([executable.parent / "VERSION", executable.parent.parent / "VERSION"])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8").strip(), str(candidate)
    return "unknown", "not found"


def main() -> int:
    import numpy as np
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.extensions import enable_extension
    from isaacsim.core.utils.stage import create_new_stage
    from pxr import Usd, UsdGeom

    from project_config import load_config
    from xacro_builder import generate_xarm7_urdf
    from xarm7_loader import add_robot_reference, find_articulation_root, import_urdf_to_usd, inspect_urdf

    config = load_config(ARGS.config, PROJECT_ROOT)
    version, version_file = _installed_version()
    print(f"[INFO] Isaac version: {version} ({version_file})")
    if not version.startswith("6.0.1"):
        raise RuntimeError(f"Isaac Sim 6.0.1 is required; this python.bat reports {version}")

    enable_extension("isaacsim.asset.importer.urdf")
    enable_extension("isaacsim.robot_motion.motion_generation")
    for _ in range(10):
        simulation_app.update()
    from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver

    print(f"[OK] Motion APIs: {LulaKinematicsSolver.__name__}, {ArticulationKinematicsSolver.__name__}")
    urdf_path = generate_xarm7_urdf(PROJECT_ROOT)
    description = inspect_urdf(urdf_path)
    usd_path = Path(config["robot"]["usd_path"])
    if not ARGS.skip_import or not usd_path.is_file():
        import_urdf_to_usd(
            urdf_path,
            usd_path,
            drive_type=config["robot"]["drive_type"],
            drive_stiffness=float(config["robot"]["drive_stiffness"]),
            drive_damping=float(config["robot"]["drive_damping"]),
        )
        print(f"[OK] Imported reusable USD: {usd_path}")
    else:
        print(f"[OK] Existing reusable USD: {usd_path}")

    asset_stage = Usd.Stage.Open(str(usd_path))
    if not asset_stage:
        raise RuntimeError(f"Could not open generated USD: {usd_path}")
    units = float(UsdGeom.GetStageMetersPerUnit(asset_stage))
    up_axis = str(UsdGeom.GetStageUpAxis(asset_stage))
    if abs(units - 1.0) > 1e-12 or up_axis.upper() != "Z":
        raise RuntimeError(f"Expected metre/Z-up USD, got metresPerUnit={units}, upAxis={up_axis}")

    create_new_stage()
    world = World(stage_units_in_meters=1.0, physics_dt=float(config["safety"]["physics_dt_s"]))
    world.scene.add_default_ground_plane()
    reference_root = add_robot_reference(usd_path, config["robot"]["prim_path"])
    for _ in range(10):
        simulation_app.update()
    articulation_root = find_articulation_root(world.stage, reference_root)
    robot = world.scene.add(SingleArticulation(prim_path=articulation_root, name="xarm7_environment_check"))
    world.reset()
    actual_names = list(robot.dof_names)
    expected = list(config["robot"]["expected_joint_names"])
    if actual_names != expected:
        raise RuntimeError(f"Expected seven arm DOFs {expected}, imported {actual_names}")
    home = np.asarray(config["robot"]["home_joint_positions_rad"], dtype=np.float64)
    robot.set_joint_positions(home)
    for _ in range(5):
        world.step(render=not ARGS.headless)

    report = {
        "isaac_version": version,
        "usd_path": str(usd_path),
        "articulation_root": articulation_root,
        "root_link": description.root_link,
        "flange_link": description.flange_link,
        "pen_tip_link": description.pen_tip_link,
        "arm_joint_names": actual_names,
        "gripper_joint_names": description.gripper_joint_names,
        "meters_per_unit": units,
        "up_axis": up_axis,
        "home_joint_positions_rad": home.tolist(),
        "drive_type": config["robot"]["drive_type"],
        "drive_stiffness": float(config["robot"]["drive_stiffness"]),
        "drive_damping": float(config["robot"]["drive_damping"]),
    }
    output = Path(config["project"]["output_dir"]) / "environment_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] Articulation root: {articulation_root}")
    print(f"[OK] Arm joints ({len(actual_names)}): {actual_names}")
    print(f"[OK] Flange: {description.flange_link}; IK frame: {description.pen_tip_link}")
    print(f"[OK] Gripper joints: {description.gripper_joint_names or 'none'}")
    print(f"[OK] Units/up-axis: metres ({units} m/unit), {up_axis}-up")
    print(f"[OK] Report: {output}")
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close(exit_code=exit_code)
    raise SystemExit(exit_code)
