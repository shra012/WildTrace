"""Generate a simulation-only URDF from official UFACTORY xacro sources."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xacro_builder import generate_xarm7_urdf
from xarm7_loader import inspect_urdf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
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

