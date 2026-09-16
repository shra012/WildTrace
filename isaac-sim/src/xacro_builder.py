"""Offline xacro conversion for the vendored official UFACTORY description."""
from __future__ import annotations

import sys
from pathlib import Path


def generate_xarm7_urdf(project_root: str | Path, force: bool = False) -> Path:
    root = Path(project_root).resolve()
    asset_dir = root / "assets" / "xarm7"
    source_dir = asset_dir / "xarm_ros2_source" / "xarm_description"
    xacro_vendor = asset_dir / "xacro_vendor"
    entry = asset_dir / "xarm7_isaac.urdf.xacro"
    output = asset_dir / "xarm7_with_pen.urdf"
    if output.is_file() and not force:
        return output
    for required in (source_dir, xacro_vendor, entry):
        if not required.exists():
            raise FileNotFoundError(f"Required xArm generation input is missing: {required}")

    sys.path.insert(0, str(xacro_vendor))
    try:
        import xacro
        import xacro.substitution_args as substitution_args

        def project_find(package: str) -> str:
            if package != "xarm_description":
                raise RuntimeError(f"Hardware-free xacro wrapper only permits xarm_description, got {package}")
            return str(source_dir)

        substitution_args._eval_find = project_find
        substitution_args._eval_dict["find"] = project_find
        document = xacro.process_file(str(entry), mappings={})
        text = document.toprettyxml(indent="  ")
    finally:
        try:
            sys.path.remove(str(xacro_vendor))
        except ValueError:
            pass

    forbidden = ("UFRobotSystemHardware", "robot_ip", "<ros2_control", "<gazebo")
    present = [token for token in forbidden if token.lower() in text.lower()]
    if present:
        raise RuntimeError(f"Refusing to write URDF containing hardware/control tokens: {present}")
    output.write_text(text, encoding="utf-8")
    return output
