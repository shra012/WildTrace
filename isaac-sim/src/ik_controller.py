"""Lula IK adapter with explicit safety checks."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

from path_geometry import limit_joint_delta


class SafeLulaIKController:
    """Position-only articulation commands; never emits torque or hardware I/O."""

    def __init__(
        self,
        articulation,
        robot_description_path: str | Path,
        urdf_path: str | Path,
        end_effector_frame: str,
        joint_limits: Dict[str, Tuple[float, float]],
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
        max_joint_delta_rad: float | None = None,
    ):
        from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver

        self.articulation = articulation
        self.lula = LulaKinematicsSolver(str(robot_description_path), str(urdf_path))
        base_position, base_orientation = articulation.get_world_pose()
        self.lula.set_robot_base_pose(np.asarray(base_position), np.asarray(base_orientation))
        self.solver = ArticulationKinematicsSolver(articulation, self.lula, end_effector_frame)
        self.articulation_controller = articulation.get_articulation_controller()
        self.joint_names = list(self.lula.get_joint_names())
        self.lower = np.asarray([joint_limits[name][0] for name in self.joint_names], dtype=np.float64)
        self.upper = np.asarray([joint_limits[name][1] for name in self.joint_names], dtype=np.float64)
        self.position_tolerance_m = float(position_tolerance_m)
        self.orientation_tolerance_rad = float(orientation_tolerance_rad)
        self.max_joint_delta_rad = None if max_joint_delta_rad is None else float(max_joint_delta_rad)
        self.last_commanded_delta_rad = np.zeros(len(self.joint_names), dtype=np.float64)
        self.last_command_was_clamped = False

    def end_effector_pose(self):
        return self.solver.compute_end_effector_pose(position_only=False)

    def solve(self, target_position: Sequence[float], target_orientation_wxyz: Sequence[float]):
        action, success = self.solver.compute_inverse_kinematics(
            np.asarray(target_position, dtype=np.float64),
            np.asarray(target_orientation_wxyz, dtype=np.float64),
            position_tolerance=self.position_tolerance_m,
            orientation_tolerance=self.orientation_tolerance_rad,
        )
        if not success or action.joint_positions is None:
            return action, False
        proposed = np.asarray(action.joint_positions, dtype=np.float64)
        if not np.isfinite(proposed).all() or np.any(proposed < self.lower) or np.any(proposed > self.upper):
            return action, False
        return action, True

    def apply(self, action) -> None:
        """Apply the solution, slew-limited around the measured joint state.

        Clamping toward a limit-checked solution from the measured position
        keeps the command inside the joint limits that solve() already
        verified, so no second limit check is required.
        """
        self.last_command_was_clamped = False
        if self.max_joint_delta_rad and action.joint_positions is not None:
            measured = np.asarray(self.articulation.get_joint_positions(), dtype=np.float64)
            indices = getattr(action, "joint_indices", None)
            if indices is not None:
                measured = measured[np.asarray(indices, dtype=int)]
            proposed = np.asarray(action.joint_positions, dtype=np.float64)
            limited = limit_joint_delta(measured, proposed, self.max_joint_delta_rad)
            self.last_commanded_delta_rad = limited - measured
            self.last_command_was_clamped = bool(
                np.any(np.abs(proposed - measured) > self.max_joint_delta_rad + 1e-12)
            )
            action.joint_positions = limited
        self.articulation_controller.apply_action(action)
