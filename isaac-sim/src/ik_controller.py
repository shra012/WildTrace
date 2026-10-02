"""Lula IK adapter with explicit safety checks."""
from __future__ import annotations

from copy import copy
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
        self._last_commanded_positions = None
        self.reset_command_state()

    def reset_command_state(self, positions=None) -> None:
        """Synchronize the setpoint-rate limiter after a teleport/reset."""
        if positions is None:
            positions = self.articulation.get_joint_positions()
        values = np.asarray(positions, dtype=np.float64).reshape(-1)
        self._last_commanded_positions = values.copy()
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
        """Apply the solution, slew-limited from the previous commanded target.

        Clamping toward a limit-checked solution from the previous setpoint
        preserves restoring authority under gravity while keeping the command
        inside the limits already verified by solve().
        """
        self.last_command_was_clamped = False
        # Never mutate the caller's IK/home action. Reusing a mutated action
        # under gravity makes the target follow the drifting measured pose on
        # every frame, so the position drive can never recover the true goal.
        command_action = action
        if action.joint_positions is not None:
            measured = np.asarray(self.articulation.get_joint_positions(), dtype=np.float64).reshape(-1)
            indices = getattr(action, "joint_indices", None)
            command_indices = (
                np.arange(measured.size, dtype=int)
                if indices is None
                else np.asarray(indices, dtype=int).reshape(-1)
            )
            proposed = np.asarray(action.joint_positions, dtype=np.float64).reshape(-1)
            if command_indices.size != proposed.size:
                raise RuntimeError(
                    f"Position action has {proposed.size} values for {command_indices.size} joint indices"
                )
            if self._last_commanded_positions is None or self._last_commanded_positions.shape != measured.shape:
                self._last_commanded_positions = measured.copy()
            previous_command = self._last_commanded_positions[command_indices]
            limited = proposed
            if self.max_joint_delta_rad:
                limited = limit_joint_delta(previous_command, proposed, self.max_joint_delta_rad)
                self.last_command_was_clamped = bool(
                    np.any(np.abs(proposed - previous_command) > self.max_joint_delta_rad + 1e-12)
                )
            self.last_commanded_delta_rad = np.zeros_like(self._last_commanded_positions)
            self.last_commanded_delta_rad[command_indices] = limited - previous_command
            self._last_commanded_positions[command_indices] = limited
            if self.max_joint_delta_rad:
                command_action = copy(action)
                command_action.joint_positions = limited
        self.articulation_controller.apply_action(command_action)
