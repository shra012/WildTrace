"""Shared Isaac Sim articulation-drive configuration for the xArm drawing tasks."""
from __future__ import annotations

from typing import Any

import numpy as np


def _as_numpy(values: Any) -> np.ndarray:
    """Convert NumPy, Torch, or Isaac backend arrays to a flat NumPy array."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    elif hasattr(values, "numpy"):
        values = values.numpy()
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _joint_vector(value: Any, joint_count: int, name: str) -> np.ndarray:
    """Resolve a scalar or per-joint setting and validate it."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(joint_count, float(array), dtype=np.float64)
    else:
        array = array.reshape(-1)
    if array.shape != (joint_count,):
        raise ValueError(f"robot.{name} must be a scalar or contain {joint_count} values")
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError(f"robot.{name} must contain finite, non-negative values")
    return array


def _solver_iterations(robot_config: dict) -> tuple[int, int]:
    position_iterations = int(robot_config.get("solver_position_iterations", 16))
    velocity_iterations = int(robot_config.get("solver_velocity_iterations", 4))
    if position_iterations <= 0 or velocity_iterations <= 0:
        raise ValueError("robot solver iteration counts must be positive")
    return position_iterations, velocity_iterations


def prepare_articulation_solver(stage, articulation_root: str, robot_config: dict) -> None:
    """Author PhysX articulation/drive settings before World.reset initializes it."""
    from pxr import PhysxSchema, UsdPhysics

    prim = stage.GetPrimAtPath(articulation_root)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid articulation root: {articulation_root}")
    if not prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
        PhysxSchema.PhysxArticulationAPI.Apply(prim)
    physx_articulation = PhysxSchema.PhysxArticulationAPI(prim)
    position_iterations, velocity_iterations = _solver_iterations(robot_config)
    physx_articulation.CreateSolverPositionIterationCountAttr(position_iterations)
    physx_articulation.CreateSolverVelocityIterationCountAttr(velocity_iterations)

    joint_names = list(robot_config["expected_joint_names"])
    joint_count = len(joint_names)
    max_forces = _joint_vector(
        robot_config["sim_position_drive_max_force"],
        joint_count,
        "sim_position_drive_max_force",
    )
    kps = _joint_vector(robot_config["drive_stiffness"], joint_count, "drive_stiffness")
    kds = _joint_vector(robot_config["drive_damping"], joint_count, "drive_damping")
    drive_type = str(robot_config.get("drive_type", "acceleration")).strip().lower()
    if drive_type not in {"acceleration", "force"}:
        raise ValueError("robot.drive_type must be 'acceleration' or 'force'")

    joint_prims: dict[str, list] = {name: [] for name in joint_names}
    for candidate in stage.Traverse():
        name = candidate.GetName()
        if name in joint_prims and candidate.IsA(UsdPhysics.RevoluteJoint):
            joint_prims[name].append(candidate)
    invalid = {name: len(prims) for name, prims in joint_prims.items() if len(prims) != 1}
    if invalid:
        raise RuntimeError(f"Expected exactly one revolute joint prim per arm joint; found {invalid}")

    # Rotational drive gains stored in USD use degree units. Isaac's runtime
    # controller exposes radian-unit gains, hence the pi/180 conversion.
    radians_per_degree = np.pi / 180.0
    for index, name in enumerate(joint_names):
        joint_prim = joint_prims[name][0]
        drive = UsdPhysics.DriveAPI(joint_prim, "angular")
        if not drive.GetMaxForceAttr().IsValid():
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
        drive.CreateTypeAttr(drive_type)
        drive.CreateMaxForceAttr(float(max_forces[index]))
        drive.CreateStiffnessAttr(float(kps[index] * radians_per_degree))
        drive.CreateDampingAttr(float(kds[index] * radians_per_degree))


def configure_position_drive(robot, robot_config: dict, *, prefix: str = "[CONTROL]"):
    """Configure the simulated low-level position servo used beneath IK/RL.

    The real xArm's internal joint servos support commanded positions against
    gravity. This function models that layer with a configured PhysX position
    drive; the finite simulation force ceilings prevent the imported URDF
    limits from prematurely saturating the simulated servo.
    """
    joint_count = len(robot.dof_names)
    controller = robot.get_articulation_controller()
    drive_type = str(robot_config.get("drive_type", "acceleration")).strip().lower()
    if drive_type not in {"acceleration", "force"}:
        raise ValueError("robot.drive_type must be 'acceleration' or 'force'")

    controller.switch_control_mode("position")
    kps = _joint_vector(robot_config["drive_stiffness"], joint_count, "drive_stiffness")
    kds = _joint_vector(robot_config["drive_damping"], joint_count, "drive_damping")
    controller.set_gains(kps=kps, kds=kds)

    configured_max_force = robot_config.get("sim_position_drive_max_force")
    if configured_max_force is not None:
        max_forces = _joint_vector(
            configured_max_force, joint_count, "sim_position_drive_max_force"
        )
        if np.any(max_forces <= 0.0):
            raise ValueError("robot.sim_position_drive_max_force values must be greater than zero")
        controller.set_max_efforts(max_forces)

    position_iterations, velocity_iterations = _solver_iterations(robot_config)

    actual_kps, actual_kds = controller.get_gains()
    actual_max_forces = controller.get_max_efforts()
    actual_modes = controller.get_effort_modes()
    if any(str(mode) != drive_type for mode in actual_modes):
        raise RuntimeError(
            f"Expected {drive_type} effort mode for every joint, got {actual_modes}"
        )
    print(
        f"{prefix} position servo mode={drive_type}, "
        f"kp={np.round(_as_numpy(actual_kps), 3).tolist()}, "
        f"kd={np.round(_as_numpy(actual_kds), 3).tolist()}, "
        f"max_force={np.round(_as_numpy(actual_max_forces), 3).tolist()}, "
        f"solver_iterations={position_iterations}/{velocity_iterations}"
    )
    return controller


def generalized_gravity_forces(robot) -> np.ndarray:
    """Read the current generalized gravity load for controller diagnostics."""
    source = robot
    if not hasattr(source, "get_generalized_gravity_forces"):
        source = getattr(robot, "_articulation_view", None)
    if source is None or not hasattr(source, "get_generalized_gravity_forces"):
        raise RuntimeError("The Isaac articulation does not expose generalized gravity forces")
    return _as_numpy(source.get_generalized_gravity_forces())[: len(robot.dof_names)]
