"""xArm 7 description inspection and Isaac URDF/USD loading helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class JointLimit:
    lower_rad: float
    upper_rad: float
    velocity_rad_s: float
    effort: float


@dataclass(frozen=True)
class RobotDescription:
    robot_name: str
    root_link: str
    flange_link: str
    pen_tip_link: str
    arm_joint_names: List[str]
    joint_limits: Dict[str, JointLimit]
    gripper_joint_names: List[str]


EXPECTED_ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]


def inspect_urdf(urdf_path: str | Path) -> RobotDescription:
    root = ET.parse(Path(urdf_path)).getroot()
    links = {link.attrib["name"] for link in root.findall("link")}
    child_links = {child.attrib["link"] for joint in root.findall("joint") for child in joint.findall("child")}
    roots = sorted(links - child_links)
    if len(roots) != 1:
        raise ValueError(f"Expected one URDF root link, found {roots}")
    limits: Dict[str, JointLimit] = {}
    movable_names: List[str] = []
    for joint in root.findall("joint"):
        if joint.attrib.get("type") in {"revolute", "continuous", "prismatic"}:
            name = joint.attrib["name"]
            movable_names.append(name)
            limit = joint.find("limit")
            if limit is None:
                raise ValueError(f"Movable joint {name} lacks a limit")
            limits[name] = JointLimit(
                float(limit.attrib.get("lower", "-inf")),
                float(limit.attrib.get("upper", "inf")),
                float(limit.attrib.get("velocity", "inf")),
                float(limit.attrib.get("effort", "inf")),
            )
    missing = [name for name in EXPECTED_ARM_JOINTS if name not in movable_names]
    if missing:
        raise ValueError(f"Missing expected xArm 7 joints: {missing}")
    if "link_eef" not in links or "pen_tip" not in links:
        raise ValueError("Generated drawing URDF must contain link_eef and pen_tip")
    gripper = [name for name in movable_names if name not in EXPECTED_ARM_JOINTS]
    return RobotDescription(
        robot_name=root.attrib.get("name", ""),
        root_link=roots[0],
        flange_link="link_eef",
        pen_tip_link="pen_tip",
        arm_joint_names=EXPECTED_ARM_JOINTS.copy(),
        joint_limits={name: limits[name] for name in EXPECTED_ARM_JOINTS},
        gripper_joint_names=gripper,
    )


def import_urdf_to_usd(
    urdf_path: str | Path,
    usd_path: str | Path,
    *,
    drive_type: str = "acceleration",
    drive_stiffness: float = 400.0,
    drive_damping: float = 40.0,
) -> Tuple[bool, str]:
    """Import with the Isaac 6 URDF extension. Call only after SimulationApp."""
    import shutil
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

    source = str(Path(urdf_path).resolve())
    destination = Path(usd_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    import_config = URDFImporterConfig(
        urdf_path=source,
        usd_path=str(destination.parent),
        merge_fixed_joints=False,
        merge_mesh=False,
        collision_from_visuals=False,
        allow_self_collision=False,
        fix_base=None,
        joint_drive_type=drive_type,
        joint_target_type="position",
        override_joint_stiffness=float(drive_stiffness),
        override_joint_damping=float(drive_damping),
        run_asset_transformer=False,
        run_multi_physics_conversion=True,
    )
    generated_path = Path(URDFImporter(import_config).import_urdf()).resolve()
    if not generated_path.is_file():
        raise RuntimeError(f"URDF import failed for {source}")
    if generated_path != destination:
        shutil.copy2(generated_path, destination)
    from pxr import Usd

    imported_stage = Usd.Stage.Open(str(destination))
    if not imported_stage:
        raise RuntimeError(f"Isaac wrote an unreadable USD: {destination}")
    if not imported_stage.GetDefaultPrim().IsValid():
        roots = list(imported_stage.GetPseudoRoot().GetChildren())
        if len(roots) != 1:
            raise RuntimeError(f"Imported USD has no default prim and {len(roots)} root prims")
        imported_stage.SetDefaultPrim(roots[0])
        imported_stage.GetRootLayer().Save()
    return True, f"/{Path(source).stem}"


def add_robot_reference(usd_path: str | Path, prim_path: str = "/World/xarm7") -> str:
    from isaacsim.core.utils.stage import add_reference_to_stage

    add_reference_to_stage(str(Path(usd_path).resolve()), prim_path)
    return prim_path


def find_articulation_root(stage, below_path: str) -> str:
    from pxr import UsdPhysics

    candidates = []
    prefix = below_path.rstrip("/") + "/"
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if (path == below_path or path.startswith(prefix)) and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one articulation root below {below_path}, found {candidates}")
    return candidates[0]
