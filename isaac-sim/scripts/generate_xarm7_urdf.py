"""Generate a simulation-only URDF from official UFACTORY xacro sources."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xacro_builder import generate_xarm7_urdf
from xarm7_loader import inspect_urdf

GRIPPER_JOINTS = (
    "drive_joint",
    "left_finger_joint",
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "right_finger_joint",
    "right_inner_knuckle_joint",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--real",
        action="store_true",
        help="Build xarm7_real.urdf: calibrated kinematics of the lab arm, xArm Gripper, pen at the gripper TCP",
    )
    args = parser.parse_args()
    if args.real:
        path = generate_xarm7_urdf(
            PROJECT_ROOT,
            force=args.force,
            entry_name="xarm7_real.urdf.xacro",
            output_name="xarm7_real.urdf",
            freeze_joints=GRIPPER_JOINTS,
        )
    else:
        path = generate_xarm7_urdf(PROJECT_ROOT, force=args.force)
    description = inspect_urdf(path)
    print(f"[OK] Generated: {path}")
    print(f"[OK] Root link: {description.root_link}")
    print(f"[OK] Flange: {description.flange_link}; pen tip: {description.pen_tip_link}")
    print(f"[OK] Arm joints ({len(description.arm_joint_names)}): {description.arm_joint_names}")
    print(f"[OK] Gripper joints: {description.gripper_joint_names or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

