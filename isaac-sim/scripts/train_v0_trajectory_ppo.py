"""Train and evaluate a trajectory-conditioned PPO v0 drawing baseline.

The policy receives the current reference waypoint and lookahead waypoints. It
outputs bounded residual joint velocities around the repository's IK and
gravity-capable position-drive controller. Curriculum training, episode
metrics, deterministic evaluation, and resumable checkpoints make this a
measurable baseline before moving to image-only control.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaacsim import SimulationApp


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/xarm7_drawing.yaml")
    parser.add_argument("--trajectory", default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--updates", type=int, default=None, help="Override v0_ppo.ppo_updates")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint to evaluate, resume from, or overwrite. Defaults to v0_ppo.checkpoint_path.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Restore policy, optimizer, curriculum stage, and update counter from --checkpoint.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Render one deterministic full-path rollout; no training or checkpoint writes occur.",
    )
    parser.add_argument(
        "--scene-check",
        action="store_true",
        help="Open the initialized robot, paper, lights, and camera without training or evaluation.",
    )
    parser.add_argument(
        "--compare-zero-action",
        action="store_true",
        help="In evaluation, also measure the IK-only baseline with zero RL residual.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=15.0,
        help="How long to leave the completed GUI evaluation visible (ignored when headless).",
    )
    return parser.parse_args()


ARGS = _parse_args()
simulation_app = SimulationApp({"headless": ARGS.headless, "width": 960, "height": 720})


class TrajectoryPPOEnv:
    """One xArm reference-tracking environment using stable position drives."""

    def __init__(self, config, trajectory_path: str | Path):
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import FixedCuboid
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.extensions import enable_extension
        from isaacsim.core.utils.stage import create_new_stage

        from coordinate_mapper import map_trajectory_to_plane
        from articulation_control import configure_position_drive, prepare_articulation_solver
        from drawing_state_machine import build_motion_sequence, flatten_desired_targets
        from ik_controller import SafeLulaIKController
        from trajectory_loader import load_trajectory
        from xarm7_loader import add_robot_reference, find_articulation_root, inspect_urdf

        self.config = config
        self.robot_config = config["robot"]
        self.drawing = config["drawing"]
        self.safety = config["safety"]
        self.v0 = config["v0_ppo"]
        trajectory = load_trajectory(trajectory_path, first_n_strokes=int(self.v0["first_n_strokes"]))
        self.mapped = map_trajectory_to_plane(
            trajectory,
            center_xy=self.drawing["surface_center_xy_m"],
            size_xy=self.drawing["surface_size_xy_m"],
            flip_image_y=bool(self.drawing["flip_image_y"]),
            max_step=float(self.drawing["max_cartesian_step_m"]),
            smoothing_strength=float(self.drawing["smoothing_strength"]),
            corner_angle_degrees=float(self.drawing["corner_angle_degrees"]),
        )
        phases = build_motion_sequence(
            self.mapped["strokes"],
            pen_down_z=float(self.drawing["pen_down_z_m"]),
            pen_up_z=float(self.drawing["pen_up_z_m"]),
            approach_height=float(self.drawing["approach_height_m"]),
            max_cartesian_step=float(self.drawing["max_cartesian_step_m"]),
            corner_angle_degrees=float(self.drawing["corner_angle_degrees"]),
            corner_densify_window=int(self.drawing["corner_window_points"]),
            corner_densify_factor=int(self.drawing["corner_densify_factor"]),
        )
        self.full_targets = flatten_desired_targets(phases)
        if not self.full_targets:
            raise RuntimeError("Mapped trajectory produced no reference targets")
        self.curriculum_counts, self.curriculum_tolerances = self._resolve_curriculum()

        enable_extension("isaacsim.asset.importer.urdf")
        enable_extension("isaacsim.robot_motion.motion_generation")
        enable_extension("isaacsim.util.debug_draw")
        create_new_stage()
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=float(self.safety["physics_dt_s"]),
            rendering_dt=float(self.safety["physics_dt_s"]),
        )
        self.world.get_physics_context().set_gravity(float(self.safety["gravity_m_s2"]))
        # Avoid add_default_ground_plane(): Isaac resolves that visual asset
        # through the online asset root. A local fixed cuboid is sufficient for
        # this scene and keeps training available offline.
        self.world.scene.add(FixedCuboid(
            prim_path="/World/Ground",
            name="ground",
            position=np.asarray([0.0, 0.0, -0.01]),
            scale=np.asarray([8.0, 8.0, 0.02]),
            color=np.asarray([0.08, 0.16, 0.22]),
        ))
        paper_size = np.asarray(self.drawing["surface_size_xy_m"], dtype=np.float64) + 0.04
        paper_thickness = 0.01
        self.world.scene.add(FixedCuboid(
            prim_path="/World/DrawingSurface",
            name="drawing_surface",
            position=np.asarray([
                *self.drawing["surface_center_xy_m"],
                float(self.drawing["paper_top_z_m"]) - paper_thickness / 2,
            ]),
            scale=np.asarray([paper_size[0], paper_size[1], paper_thickness]),
            color=np.asarray([0.92, 0.92, 0.88]),
        ))
        urdf_path = Path(self.robot_config["urdf_path"])
        description = inspect_urdf(urdf_path)
        root = add_robot_reference(self.robot_config["usd_path"], self.robot_config["prim_path"])
        for _ in range(10):
            simulation_app.update()
        articulation_root = find_articulation_root(self.world.stage, root)
        prepare_articulation_solver(self.world.stage, articulation_root, self.robot_config)
        self.robot = self.world.scene.add(
            SingleArticulation(prim_path=articulation_root, name="xarm7_v0_ppo")
        )
        self.world.reset()
        self.controller = configure_position_drive(
            self.robot, self.robot_config, prefix="[V0 CONTROL]"
        )
        joint_limits = {
            name: (description.joint_limits[name].lower_rad, description.joint_limits[name].upper_rad)
            for name in description.arm_joint_names
        }
        self.joint_lower = np.asarray([joint_limits[name][0] for name in description.arm_joint_names])
        self.joint_upper = np.asarray([joint_limits[name][1] for name in description.arm_joint_names])
        self.ik = SafeLulaIKController(
            self.robot,
            self.robot_config["robot_description_path"],
            urdf_path,
            self.robot_config["end_effector_frame"],
            joint_limits,
            float(self.drawing["ik_position_tolerance_m"]),
            float(self.drawing["orientation_tolerance_rad"]),
            max_joint_delta_rad=float(self.robot_config["max_joint_delta_rad"]),
        )
        self.orientation = np.asarray(self.drawing["orientation_wxyz"], dtype=np.float64)
        self.home = np.asarray(self.robot_config["home_joint_positions_rad"], dtype=np.float64)
        self.dt = float(self.safety["physics_dt_s"])
        self.control_substeps = int(self.v0["control_substeps"])
        self.lookahead = int(self.v0["lookahead_points"])
        self.max_steps = int(self.v0["max_episode_steps"])
        self.residual_velocity = float(self.v0["residual_joint_velocity_rad_s"])
        self.target_index = 0
        self.step_count = 0
        self.observation = None
        self.previous_distance = 0.0
        self.previous_action = np.zeros(7, dtype=np.float32)
        self.executed_points = []
        self.completed_episode_metrics = []
        self._reset_episode_accumulators()
        self.curriculum_stage = 0
        self.targets = []
        self.target_tolerance = float(self.v0["target_tolerance_m"])
        self.set_curriculum_stage(0, announce=False)
        try:
            from isaacsim.util.debug_draw import _debug_draw

            self.debug_draw = _debug_draw.acquire_debug_draw_interface()
        except Exception as exc:
            print(f"[V0] debug draw unavailable: {exc}")
            self.debug_draw = None
        # q[7], qd[7], current error[3], relative lookahead[5*3], pen state[1]
        self.observation_dim = 7 + 7 + 3 + 3 * self.lookahead + 1
        self._configure_gui_scene()
        print(
            f"[V0] full_targets={len(self.full_targets)}, observation_dim={self.observation_dim}, "
            f"control=IK + residual joint velocity, gravity={self.safety['gravity_m_s2']}"
        )
        self._announce_stage()

    def _configure_gui_scene(self):
        """Add local lighting and frame the robot/canvas in the interactive viewport."""
        if ARGS.headless:
            return

        from isaacsim.core.rendering_manager import ViewportManager
        from pxr import Gf, UsdGeom, UsdLux

        stage = self.world.stage
        center_x, center_y = self.drawing["surface_center_xy_m"]

        # Debug-draw lines remain visible without lights, but USD geometry does
        # not. Explicit local lights keep the robot and paper visible even when
        # the default Isaac stage/viewport lighting mode changes.
        dome = UsdLux.DomeLight.Define(stage, "/World/V0DomeLight")
        dome.CreateIntensityAttr().Set(1000.0)

        key = UsdLux.RectLight.Define(stage, "/World/V0DrawingAreaLight")
        key.CreateIntensityAttr().Set(5000.0)
        key.CreateColorAttr().Set(Gf.Vec3f(1.0, 0.98, 0.95))
        key.CreateWidthAttr().Set(2.0)
        key.CreateHeightAttr().Set(2.0)
        key_xform = UsdGeom.Xformable(key)
        key_xform.ClearXformOpOrder()
        key_xform.AddTranslateOp().Set(Gf.Vec3d(center_x, center_y, 1.5))
        key_xform.AddRotateXYZOp().Set(Gf.Vec3f(180.0, 0.0, 0.0))

        eye = list(self.v0.get("gui_camera_eye_m", [1.35, 1.20, 1.15]))
        target = list(self.v0.get("gui_camera_target_m", [0.28, 0.0, 0.42]))
        ViewportManager.set_camera_view(
            "/OmniverseKit_Persp",
            eye=eye,
            target=target,
        )
        for _ in range(8):
            simulation_app.update()
        print(f"[V0 VIEW] camera_eye={eye}, camera_target={target}, local_lights=on")

    def _resolve_curriculum(self):
        raw_counts = list(self.v0.get("curriculum_target_counts", [-1]))
        raw_tolerances = list(self.v0.get("curriculum_tolerances_m", []))
        fallback = float(self.v0["target_tolerance_m"])
        if not raw_tolerances:
            raw_tolerances = [fallback] * len(raw_counts)
        if len(raw_counts) != len(raw_tolerances):
            raise ValueError("v0_ppo curriculum target and tolerance lists must have equal lengths")
        counts, tolerances = [], []
        full_count = len(self.full_targets)
        for raw_count, raw_tolerance in zip(raw_counts, raw_tolerances):
            count = full_count if int(raw_count) <= 0 else min(int(raw_count), full_count)
            if count <= 0:
                continue
            if counts and count <= counts[-1]:
                if count == counts[-1]:
                    tolerances[-1] = float(raw_tolerance)
                continue
            counts.append(count)
            tolerances.append(float(raw_tolerance))
        if not counts or counts[-1] < full_count:
            counts.append(full_count)
            tolerances.append(fallback)
        return counts, tolerances

    def _announce_stage(self):
        print(
            f"[CURRICULUM] stage={self.curriculum_stage + 1}/{len(self.curriculum_counts)} "
            f"targets={len(self.targets)}/{len(self.full_targets)} "
            f"tolerance_m={self.target_tolerance:.4f} max_steps={self.max_steps}"
        )

    def set_curriculum_stage(self, stage_index: int, announce: bool = True):
        self.curriculum_stage = int(np.clip(stage_index, 0, len(self.curriculum_counts) - 1))
        count = self.curriculum_counts[self.curriculum_stage]
        self.targets = self.full_targets[:count]
        self.target_tolerance = self.curriculum_tolerances[self.curriculum_stage]
        self.target_index = 0
        self.step_count = 0
        self.observation = None
        self.executed_points = []
        self._reset_episode_accumulators()
        if announce:
            self._announce_stage()

    @property
    def current_target(self):
        return self.targets[min(self.target_index, len(self.targets) - 1)]

    def _lookahead_positions(self):
        values = []
        for offset in range(1, self.lookahead + 1):
            values.append(self.targets[min(self.target_index + offset, len(self.targets) - 1)].position)
        return np.asarray(values, dtype=np.float64)

    def _update_visualization(self):
        if self.debug_draw is None:
            return
        self.debug_draw.clear_lines()
        z = float(self.drawing["pen_down_z_m"]) + 0.001
        for stroke in self.mapped["strokes"]:
            points = [(float(p[0]), float(p[1]), z) for p in stroke["points"]]
            if len(points) >= 2:
                self.debug_draw.draw_lines_spline(points, (0.1, 0.9, 0.2, 1.0), 2.0, False)
        if len(self.executed_points) >= 2:
            self.debug_draw.draw_lines_spline(
                [(float(p[0]), float(p[1]), float(p[2])) for p in self.executed_points],
                (0.95, 0.15, 0.75, 1.0),
                3.0,
                False,
            )

    def _observation(self):
        tip, _ = self.ik.end_effector_pose()
        tip = np.asarray(tip, dtype=np.float64)
        q = np.asarray(self.robot.get_joint_positions(), dtype=np.float64)
        qd = np.asarray(self.robot.get_joint_velocities(), dtype=np.float64)
        target = self.current_target
        scale = 0.25
        return np.concatenate([
            q / np.pi,
            qd / 3.14,
            (target.position - tip) / scale,
            (self._lookahead_positions() - tip[None, :]).reshape(-1) / scale,
            [float(target.pen_down)],
        ]).astype(np.float32)

    def _reset_episode_accumulators(self):
        self.episode_reward = 0.0
        self.episode_distances = []
        self.episode_pen_down_steps = 0
        self.episode_contact_steps = 0
        self.episode_action_sq_sum = 0.0
        self.episode_action_delta_sq_sum = 0.0

    def _finish_episode(self, termination: str):
        distances = np.asarray(self.episode_distances, dtype=np.float64)
        mean_error = float(distances.mean()) if distances.size else float("nan")
        p95_error = float(np.percentile(distances, 95)) if distances.size else float("nan")
        contact_rate = (
            self.episode_contact_steps / self.episode_pen_down_steps
            if self.episode_pen_down_steps
            else float("nan")
        )
        denominator = max(1, self.step_count * 7)
        self.completed_episode_metrics.append({
            "return": float(self.episode_reward),
            "length": int(self.step_count),
            "targets_reached": int(self.target_index),
            "target_count": int(len(self.targets)),
            "target_fraction": float(self.target_index / max(1, len(self.targets))),
            "complete": bool(self.target_index >= len(self.targets)),
            "mean_tracking_error_m": mean_error,
            "p95_tracking_error_m": p95_error,
            "contact_rate": float(contact_rate),
            "action_rms": float(np.sqrt(self.episode_action_sq_sum / denominator)),
            "action_delta_rms": float(np.sqrt(self.episode_action_delta_sq_sum / denominator)),
            "termination": termination,
            "curriculum_stage": int(self.curriculum_stage),
            "target_tolerance_m": float(self.target_tolerance),
        })

    def drain_episode_metrics(self):
        metrics = self.completed_episode_metrics
        self.completed_episode_metrics = []
        return metrics

    def reset(self):
        from isaacsim.core.utils.types import ArticulationAction

        self.robot.set_joint_positions(self.home)
        self.robot.set_joint_velocities(np.zeros(7))
        self.ik.reset_command_state(self.home)
        home_action = ArticulationAction(joint_positions=self.home)
        settle_steps = max(1, int(self.robot_config.get("gravity_home_settle_steps", 20)))
        for _ in range(settle_steps):
            self.ik.apply(home_action)
            self.world.step(render=not ARGS.headless)
        self.target_index = 0
        self.step_count = 0
        self.previous_action.fill(0.0)
        self.executed_points = []
        self._reset_episode_accumulators()
        tip, _ = self.ik.end_effector_pose()
        self.previous_distance = float(np.linalg.norm(np.asarray(tip) - self.current_target.position))
        self._update_visualization()
        self.observation = self._observation()
        return self.observation.copy()

    def step(self, action):
        from isaacsim.core.utils.types import ArticulationAction

        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        target = self.current_target
        ik_action, success = self.ik.solve(target.position, self.orientation)
        if not success or ik_action.joint_positions is None:
            reward = -2.0
            self.step_count += 1
            self.episode_reward += reward
            self.episode_distances.append(float(self.previous_distance))
            self._finish_episode("ik_failure")
            self.observation = self._observation()
            return self.observation.copy(), reward, True, {
                "failure": "ik",
                "distance_m": float(self.previous_distance),
                "target_index": self.target_index,
                "target_count": len(self.targets),
                "complete": False,
                "contact": False,
            }
        q = np.asarray(self.robot.get_joint_positions(), dtype=np.float64)
        nominal = np.asarray(ik_action.joint_positions, dtype=np.float64)
        residual = action * self.residual_velocity * self.dt * self.control_substeps
        requested = np.clip(nominal + residual, self.joint_lower, self.joint_upper)
        drive_action = ArticulationAction(joint_positions=requested)
        for _ in range(max(1, self.control_substeps)):
            self.ik.apply(drive_action)
            self.world.step(render=not ARGS.headless)
        tip, _ = self.ik.end_effector_pose()
        tip = np.asarray(tip, dtype=np.float64)
        distance = float(np.linalg.norm(tip - target.position))
        contact_error = abs(float(tip[2]) - float(self.drawing["pen_down_z_m"]))
        contact = (not target.pen_down) or contact_error <= self.target_tolerance
        reached = distance <= self.target_tolerance and contact
        progress = self.previous_distance - distance
        reward = float(self.v0["tracking_progress_weight"]) * progress
        reward -= float(self.v0["tracking_distance_weight"]) * distance
        reward -= float(self.v0["action_l2_weight"]) * float(np.dot(action, action))
        action_delta = action - self.previous_action
        reward -= float(self.v0["action_smoothness_weight"]) * float(np.dot(action_delta, action_delta))
        if target.pen_down:
            reward -= float(self.v0["contact_penalty_weight"]) * min(1.0, contact_error / 0.005)
            self.episode_pen_down_steps += 1
            if contact:
                self.episode_contact_steps += 1
                self.executed_points.append(tip.copy())
        if reached:
            reward += float(self.v0["reached_bonus"])
            self.target_index += 1
        self.episode_reward += reward
        self.episode_distances.append(distance)
        self.episode_action_sq_sum += float(np.dot(action, action))
        self.episode_action_delta_sq_sum += float(np.dot(action_delta, action_delta))
        self.previous_action = action.astype(np.float32)
        self.step_count += 1
        complete = self.target_index >= len(self.targets)
        self.previous_distance = 0.0 if complete else float(np.linalg.norm(tip - self.current_target.position))
        done = complete or self.step_count >= self.max_steps
        if complete:
            completion_bonus = float(self.v0.get("completion_bonus", 5.0))
            reward += completion_bonus
            self.episode_reward += completion_bonus
        if self.step_count % 5 == 0:
            self._update_visualization()
        if done:
            self._finish_episode("complete" if complete else "time_limit")
        self.observation = self._observation()
        return self.observation.copy(), float(reward), bool(done), {
            "distance_m": distance,
            "target_index": self.target_index,
            "target_count": len(self.targets),
            "complete": complete,
            "contact": contact,
        }


def _finite_mean(values, default=float("nan")):
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float(default)


def summarize_episodes(episodes):
    if not episodes:
        return None
    return {
        "episodes": len(episodes),
        "success_rate": float(np.mean([episode["complete"] for episode in episodes])),
        "return_mean": float(np.mean([episode["return"] for episode in episodes])),
        "targets_mean": float(np.mean([episode["targets_reached"] for episode in episodes])),
        "target_count": int(episodes[-1]["target_count"]),
        "target_fraction": float(np.mean([episode["target_fraction"] for episode in episodes])),
        "tracking_error_m": _finite_mean([episode["mean_tracking_error_m"] for episode in episodes]),
        "p95_error_m": _finite_mean([episode["p95_tracking_error_m"] for episode in episodes]),
        "contact_rate": _finite_mean([episode["contact_rate"] for episode in episodes]),
        "action_rms": float(np.mean([episode["action_rms"] for episode in episodes])),
        "action_delta_rms": float(np.mean([episode["action_delta_rms"] for episode in episodes])),
    }


def print_episode_summary(prefix, summary):
    if summary is None:
        print(f"{prefix} episodes=0 (the current episode crosses this logging interval)")
        return
    contact = "n/a" if not np.isfinite(summary["contact_rate"]) else f"{summary['contact_rate']:.1%}"
    print(
        f"{prefix} episodes={summary['episodes']} success={summary['success_rate']:.1%} "
        f"targets_mean={summary['targets_mean']:.1f}/{summary['target_count']} "
        f"return_mean={summary['return_mean']:.3f} "
        f"mean_error_mm={summary['tracking_error_m'] * 1000:.2f} "
        f"p95_error_mm={summary['p95_error_m'] * 1000:.2f} contact={contact} "
        f"action_rms={summary['action_rms']:.3f} smooth_rms={summary['action_delta_rms']:.3f}"
    )


def run_deterministic_episode(env, policy, device, zero_action=False):
    import torch

    env.drain_episode_metrics()
    observation = env.reset()
    was_training = policy.training
    policy.eval()
    for _ in range(env.max_steps):
        if zero_action:
            action = np.zeros(7, dtype=np.float32)
        else:
            tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mean, _ = policy(tensor)
            action = torch.tanh(mean).squeeze(0).cpu().numpy()
        observation, _, done, _ = env.step(action)
        if done:
            break
    if was_training:
        policy.train()
    episodes = env.drain_episode_metrics()
    env.observation = None
    if not episodes:
        raise RuntimeError("Deterministic evaluation ended without episode metrics")
    return episodes[-1]


def validate_policy_dimensions(policy, observation_dim):
    if policy.body[0].in_features != observation_dim or policy.actor_mean.out_features != 7:
        raise RuntimeError("Checkpoint dimensions do not match the current V0 environment")


def hold_gui(prefix, seconds):
    if ARGS.headless or seconds <= 0:
        return

    import time

    print(f"{prefix} keeping the GUI visible for {seconds:g} seconds")
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        simulation_app.update()
        time.sleep(0.01)


def main():
    import torch

    from project_config import load_config
    from rl_policy import (
        DrawingActorCritic,
        collect_rollout,
        compute_gae,
        load_policy_checkpoint,
        ppo_update,
        save_policy,
    )

    np.random.seed(ARGS.seed)
    torch.manual_seed(ARGS.seed)
    config = load_config(ARGS.config, PROJECT_ROOT)
    trajectory = ARGS.trajectory or config["project"]["trajectory_path"]
    env = TrajectoryPPOEnv(config, trajectory)
    v0 = config["v0_ppo"]
    cuda_requested = ARGS.device == "cuda"
    device = torch.device("cuda" if cuda_requested and torch.cuda.is_available() else "cpu")
    if cuda_requested and device.type != "cuda":
        print("[V0] CUDA was requested but is unavailable; using CPU")
    checkpoint_path = Path(ARGS.checkpoint or v0["checkpoint_path"])
    best_checkpoint_path = Path(v0.get("best_checkpoint_path", "outputs/v0_trajectory_ppo_best.pt"))

    def validate_checkpoint_gravity(payload, path):
        metadata = payload.get("metadata", {})
        saved_gravity = metadata.get("gravity_m_s2")
        configured_gravity = float(config["safety"]["gravity_m_s2"])
        if saved_gravity is None:
            print(
                f"[V0] warning: {path} has no gravity metadata; compatibility cannot be verified"
            )
            return
        if not np.isclose(float(saved_gravity), configured_gravity, atol=1e-6, rtol=0.0):
            raise RuntimeError(
                f"Checkpoint {path} was trained at gravity={saved_gravity} m/s^2, but the "
                f"current task uses {configured_gravity} m/s^2. Start a fresh 1-g checkpoint "
                "instead of resuming across different dynamics."
            )

    if ARGS.scene_check:
        if ARGS.headless:
            raise ValueError("--scene-check requires the GUI; remove --headless")
        env.reset()
        env._update_visualization()
        hold_gui("[V0 SCENE CHECK]", ARGS.hold_seconds)
        return

    if ARGS.evaluate:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"V0 checkpoint not found: {checkpoint_path}")
        policy, payload = load_policy_checkpoint(checkpoint_path, device)
        validate_checkpoint_gravity(payload, checkpoint_path)
        validate_policy_dimensions(policy, env.observation_dim)
        env.set_curriculum_stage(len(env.curriculum_counts) - 1)
        print(f"[V0 EVAL] checkpoint={checkpoint_path}, metadata={payload.get('metadata', {})}")
        if ARGS.compare_zero_action:
            baseline = run_deterministic_episode(env, policy, device, zero_action=True)
            print_episode_summary("[V0 EVAL IK-ONLY]", summarize_episodes([baseline]))
        learned = run_deterministic_episode(env, policy, device, zero_action=False)
        print_episode_summary("[V0 EVAL POLICY]", summarize_episodes([learned]))
        env._update_visualization()
        hold_gui("[V0 EVAL]", ARGS.hold_seconds)
        return

    policy = DrawingActorCritic(env.observation_dim, action_dim=7).to(device)
    resume_payload = {}
    global_update = 0
    stage_index = 0
    best_eval_score = float("-inf")
    if ARGS.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Cannot resume: V0 checkpoint not found: {checkpoint_path}")
        policy, resume_payload = load_policy_checkpoint(checkpoint_path, device)
        validate_checkpoint_gravity(resume_payload, checkpoint_path)
        validate_policy_dimensions(policy, env.observation_dim)
        training_state = resume_payload.get("training_state", {})
        global_update = int(training_state.get("global_update", 0))
        stage_index = int(training_state.get("curriculum_stage", 0))
        best_eval_score = float(training_state.get("best_eval_score", float("-inf")))
        print(f"[V0] resuming policy from {checkpoint_path} at global_update={global_update}")
    env.set_curriculum_stage(stage_index)
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(v0["ppo_learning_rate"]))
    optimizer_state = resume_payload.get("optimizer_state_dict")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        print("[V0] restored Adam optimizer state")
    elif ARGS.resume:
        print(
            "[V0] legacy checkpoint: optimizer/update/curriculum history was not stored; "
            "the policy weights are preserved and new training state starts now"
        )

    update_count = int(v0["ppo_updates"] if ARGS.updates is None else ARGS.updates)
    if update_count <= 0:
        raise ValueError("--updates must be positive")
    log_interval = int(v0.get("log_interval_updates", 10))
    eval_interval = int(v0.get("curriculum_eval_interval_updates", 10))
    eval_episodes = int(v0.get("curriculum_eval_episodes", 1))
    success_threshold = float(v0.get("curriculum_success_rate", 1.0))
    recent_episodes = []

    def checkpoint_metadata():
        return {
            "baseline": "trajectory_conditioned_residual_joint_velocity_ppo",
            "gravity_m_s2": float(config["safety"]["gravity_m_s2"]),
            "trajectory_conditioned": True,
            "curriculum_stage": int(env.curriculum_stage),
            "active_targets": int(len(env.targets)),
            "full_targets": int(len(env.full_targets)),
            "target_tolerance_m": float(env.target_tolerance),
        }

    def training_state():
        return {
            "global_update": int(global_update),
            "curriculum_stage": int(env.curriculum_stage),
            "best_eval_score": float(best_eval_score),
        }

    def save_training_checkpoint(destination):
        save_policy(
            destination,
            policy,
            env.observation_dim,
            checkpoint_metadata(),
            optimizer=optimizer,
            training_state=training_state(),
        )

    for _ in range(update_count):
        rollout = collect_rollout(env, policy, int(v0["ppo_steps_per_update"]), device)
        recent_episodes.extend(env.drain_episode_metrics())
        advantages, returns = compute_gae(
            rollout, float(v0["ppo_gamma"]), float(v0["ppo_gae_lambda"])
        )
        stats = ppo_update(
            policy,
            optimizer,
            rollout,
            advantages,
            returns,
            float(v0["ppo_clip_ratio"]),
            int(v0["ppo_epochs"]),
            int(v0["ppo_minibatch_size"]),
            float(v0["ppo_entropy_weight"]),
            device,
        )
        global_update += 1
        log_due = global_update % log_interval == 0
        eval_due = global_update % eval_interval == 0

        if log_due:
            print(
                f"[V0 TRAIN] global_update={global_update} stage={env.curriculum_stage + 1}/"
                f"{len(env.curriculum_counts)} step_reward={rollout.rewards.mean():.5f} "
                f"policy_loss={stats['policy_loss']:.5f} value_loss={stats['value_loss']:.5f} "
                f"kl={stats['approx_kl']:.5f} clip_fraction={stats['clip_fraction']:.3f}"
            )
            print_episode_summary("[V0 TRAIN EPISODES]", summarize_episodes(recent_episodes))
            recent_episodes = []

        if eval_due:
            evaluation = [
                run_deterministic_episode(env, policy, device, zero_action=False)
                for _ in range(eval_episodes)
            ]
            evaluation_summary = summarize_episodes(evaluation)
            print_episode_summary("[V0 CURRICULUM EVAL]", evaluation_summary)
            score = (
                env.curriculum_stage * 10000.0
                + evaluation_summary["success_rate"] * 1000.0
                + evaluation_summary["target_fraction"] * 100.0
                - evaluation_summary["tracking_error_m"] * 1000.0
            )
            if score > best_eval_score:
                best_eval_score = score
                save_training_checkpoint(best_checkpoint_path)
                print(f"[V0] new best checkpoint: {best_checkpoint_path} score={score:.3f}")
            if (
                evaluation_summary["success_rate"] >= success_threshold
                and env.curriculum_stage + 1 < len(env.curriculum_counts)
            ):
                env.set_curriculum_stage(env.curriculum_stage + 1)
                recent_episodes = []
                print("[V0] curriculum advanced after deterministic success")

        if log_due or eval_due:
            save_training_checkpoint(checkpoint_path)

    save_training_checkpoint(checkpoint_path)
    print(
        f"[OK] V0 latest checkpoint: {checkpoint_path}; best checkpoint: {best_checkpoint_path}; "
        f"global_update={global_update}"
    )


if __name__ == "__main__":
    exit_code = 1
    try:
        main()
        exit_code = 0
    except Exception as exc:
        import traceback

        print(f"[V0 ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
    finally:
        simulation_app.close(exit_code=exit_code)
    raise SystemExit(exit_code)
