"""Inspect the generated URDF and, when available, its imported USD articulation."""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
PORTABLE_ROOT = PROJECT_ROOT / "outputs" / "isaac_portable"
PORTABLE_ROOT.mkdir(parents=True, exist_ok=True)
if "--portable-root" not in sys.argv:
    sys.argv.extend(["--portable-root", str(PORTABLE_ROOT)])

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config/xarm7_drawing.yaml")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--portable-root", default=str(PORTABLE_ROOT), help=argparse.SUPPRESS)
args = parser.parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})


def main() -> int:
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import create_new_stage

    from project_config import load_config
    from xarm7_loader import add_robot_reference, find_articulation_root, inspect_urdf

    config = load_config(args.config, PROJECT_ROOT)
    description = inspect_urdf(config["robot"]["urdf_path"])
    print(f"robot={description.robot_name} root_link={description.root_link}")
    print(f"flange={description.flange_link} pen_tip={description.pen_tip_link}")
    for name in description.arm_joint_names:
        limit = description.joint_limits[name]
        print(
            f"{name}: lower={limit.lower_rad:.8f} rad upper={limit.upper_rad:.8f} rad "
            f"velocity={limit.velocity_rad_s:.5f} rad/s effort={limit.effort:g}"
        )
    print(f"gripper_joints={description.gripper_joint_names or 'none'}")
    usd_path = Path(config["robot"]["usd_path"])
    if not usd_path.is_file():
        raise FileNotFoundError(f"Run check_environment.py to import the USD first: {usd_path}")
    create_new_stage()
    world = World(stage_units_in_meters=1.0)
    reference = add_robot_reference(usd_path, config["robot"]["prim_path"])
    for _ in range(10):
        simulation_app.update()
    articulation_root = find_articulation_root(world.stage, reference)
    robot = world.scene.add(SingleArticulation(prim_path=articulation_root, name="xarm7_inspection"))
    world.reset()
    print(f"articulation_root={articulation_root}")
    print(f"articulation_dofs={list(robot.dof_names)}")
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close(exit_code=code)
    raise SystemExit(code)
