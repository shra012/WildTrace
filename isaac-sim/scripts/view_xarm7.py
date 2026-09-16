"""Open the xArm 7 in the Isaac Sim GUI and idle so the arm can be inspected."""
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
parser.add_argument("--usd", default=None, help="Defaults to robot.usd_path from the config")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--portable-root", default=str(PORTABLE_ROOT), help=argparse.SUPPRESS)
args = parser.parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})


def main() -> int:
    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import create_new_stage

    from project_config import load_config
    from xarm7_loader import add_robot_reference, find_articulation_root

    config = load_config(args.config, PROJECT_ROOT)
    robot_config, safety = config["robot"], config["safety"]
    usd_path = Path(args.usd) if args.usd else Path(robot_config["usd_path"])
    if not usd_path.is_file():
        raise FileNotFoundError(f"No such USD: {usd_path}")

    create_new_stage()
    world = World(stage_units_in_meters=1.0, physics_dt=float(safety["physics_dt_s"]))
    world.get_physics_context().set_gravity(float(safety["gravity_m_s2"]))
    world.scene.add_default_ground_plane()

    reference_root = add_robot_reference(usd_path, robot_config["prim_path"])
    for _ in range(10):
        simulation_app.update()
    articulation_root = find_articulation_root(world.stage, reference_root)
    robot = world.scene.add(SingleArticulation(prim_path=articulation_root, name="xarm7_viewer"))
    world.reset()
    robot.set_joint_positions(np.asarray(robot_config["home_joint_positions_rad"], dtype=np.float64))

    print(f"[OK] Loaded {usd_path}")
    print(f"[OK] Articulation root: {articulation_root}")
    print(f"[OK] Joints: {list(robot.dof_names)}")
    print("[INFO] Close the Isaac Sim window to exit.")
    while simulation_app.is_running():
        world.step(render=not args.headless)
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
