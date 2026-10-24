"""Offline xacro conversion for the vendored official UFACTORY description."""
from __future__ import annotations

import math
import sys
from pathlib import Path


def _rotation_rpy(roll: float, pitch: float, yaw: float):
    """URDF fixed-axis roll/pitch/yaw as a 3x3 matrix (R = Rz Ry Rx)."""
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _rotation_axis(axis, angle: float):
    x, y, z = axis
    norm = math.sqrt(x * x + y * y + z * z)
    x, y, z = x / norm, y / norm, z / norm
    c, s, t = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]


def _matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _rpy_from_rotation(r):
    pitch = math.asin(max(-1.0, min(1.0, -r[2][0])))
    if abs(math.cos(pitch)) > 1e-9:
        return math.atan2(r[2][1], r[2][2]), pitch, math.atan2(r[1][0], r[0][0])
    return math.atan2(-r[1][2], r[1][1]), pitch, 0.0


def freeze_at(joint, angle: float) -> None:
    """Bake a revolute joint's rotation at `angle` into its origin (before it becomes fixed)."""
    axes = joint.getElementsByTagName("axis")
    if not axes or angle == 0.0:
        return
    axis = [float(v) for v in axes[0].getAttribute("xyz").split()]
    origins = [child for child in joint.childNodes if getattr(child, "tagName", None) == "origin"]
    origin = origins[0] if origins else joint.ownerDocument.createElement("origin")
    if not origins:
        origin.setAttribute("xyz", "0 0 0")
        joint.appendChild(origin)
    rpy = [float(v) for v in (origin.getAttribute("rpy") or "0 0 0").split()]
    rotation = _matmul(_rotation_rpy(*rpy), _rotation_axis(axis, angle))
    origin.setAttribute("rpy", " ".join(f"{v:.9g}" for v in _rpy_from_rotation(rotation)))


def generate_xarm7_urdf(
    project_root: str | Path,
    force: bool = False,
    entry_name: str = "xarm7_isaac.urdf.xacro",
    output_name: str = "xarm7_with_pen.urdf",
    freeze_joints: tuple[str, ...] = (),
    freeze_drive_position: float = 0.0,
) -> Path:
    """Expand the xacro entry into a URDF.

    freeze_joints become fixed joints. They are frozen at the pose where the
    joint they mimic (or the joint itself, for the drive) is at
    `freeze_drive_position`, e.g. the xArm Gripper closed on the pen; 0.0
    freezes them at their zero pose (gripper fully open).
    """
    root = Path(project_root).resolve()
    asset_dir = root / "assets" / "xarm7"
    source_dir = asset_dir / "xarm_ros2_source" / "xarm_description"
    xacro_vendor = asset_dir / "xacro_vendor"
    entry = asset_dir / entry_name
    output = asset_dir / output_name
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
        document = xacro.process_file(str(entry), mappings={"asset_dir": str(asset_dir)})
        # Frozen joints (a gripper that only holds the pen) become fixed at
        # their zero pose so the articulation keeps exactly the 7 arm joints.
        for joint in document.getElementsByTagName("joint"):
            if joint.getAttribute("name") in freeze_joints:
                mimics = joint.getElementsByTagName("mimic")
                angle = float(freeze_drive_position)
                if mimics:
                    angle = float(mimics[0].getAttribute("multiplier") or 1.0) * angle + float(
                        mimics[0].getAttribute("offset") or 0.0
                    )
                freeze_at(joint, angle)
                joint.setAttribute("type", "fixed")
                for tag in ("mimic", "limit", "axis", "dynamics"):
                    for child in list(joint.getElementsByTagName(tag)):
                        joint.removeChild(child)
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
