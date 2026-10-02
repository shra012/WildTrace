"""Train a goal-conditioned Cartesian drawing policy in Isaac Sim.

This is intentionally a small first RL experiment, not a production trainer.
The policy sees a target raster and a raster of what it has drawn so far. It
does not receive trajectory points or waypoint indices. It outputs a bounded
Cartesian delta and a pen command; SafeLulaIKController remains underneath.

Run with the Isaac Sim Python, after generating/validating the xArm USD:
    python.bat scripts\\train_rl_drawing.py --config config\\xarm7_drawing.yaml --headless
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Isaac Sim provides a matched PyTorch build through its bundled extensions.
# Do not prepend the project's .vendor Torch here: mixing its Python modules
# with Isaac's compiled torch._C causes torch.optim/torch._dynamo import errors.
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaacsim import SimulationApp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/xarm7_drawing.yaml")
    parser.add_argument("--trajectory", default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--reach-test", action="store_true", help="Move the pen to the paper without running RL")
    return parser.parse_args()


ARGS = parse_args()
simulation_app = SimulationApp({"headless": ARGS.headless, "width": 640, "height": 480})


class IsaacDrawingEnv:
    """Single-environment goal-conditioned drawing task."""

    def __init__(self, config, trajectory_path: str | Path):
        import torch
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import FixedCuboid
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.extensions import enable_extension
        from isaacsim.core.utils.stage import create_new_stage

        from coordinate_mapper import map_trajectory_to_plane
        from articulation_control import configure_position_drive, prepare_articulation_solver
        from ik_controller import SafeLulaIKController
        from project_config import load_config
        from trajectory_loader import load_trajectory
        from xarm7_loader import add_robot_reference, find_articulation_root, inspect_urdf

        self.config = config
        self.robot_config = config["robot"]
        self.drawing = config["drawing"]
        self.safety = config["safety"]
        self.rl = config["rl"]
        self.width = int(self.rl["target_width_px"])
        self.height = int(self.rl["target_height_px"])
        self.max_steps = int(self.rl["max_episode_steps"])
        self.action_limit = float(self.rl["action_delta_m"])
        self.pen_down_threshold = float(self.rl["pen_down_threshold"])
        self.control_substeps = int(self.rl.get("control_substeps", 1))
        self.contact_z_tolerance = float(self.rl.get("contact_z_tolerance_m", 0.002))
        self.step_count = 0
        self.observation = None
        self.previous_tip = None
        self.previous_action = np.zeros(4, dtype=np.float32)
        self.canvas = np.zeros((self.height, self.width), dtype=np.float32)
        self.executed_points = []

        trajectory = load_trajectory(trajectory_path, first_n_strokes=2)
        self.mapped = map_trajectory_to_plane(
            trajectory,
            center_xy=self.drawing["surface_center_xy_m"],
            size_xy=self.drawing["surface_size_xy_m"],
            flip_image_y=bool(self.drawing["flip_image_y"]),
            max_step=float(self.drawing["max_cartesian_step_m"]),
            smoothing_strength=float(self.drawing["smoothing_strength"]),
            corner_angle_degrees=float(self.drawing["corner_angle_degrees"]),
        )
        self.target_mask = self._rasterize_strokes(self.mapped["strokes"])
        self.target_pixels = np.argwhere(self.target_mask > 0.5).astype(np.float32)

        enable_extension("isaacsim.asset.importer.urdf")
        enable_extension("isaacsim.robot_motion.motion_generation")
        create_new_stage()
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=float(self.safety["physics_dt_s"]),
            rendering_dt=float(self.safety["physics_dt_s"]),
        )
        self.world.get_physics_context().set_gravity(float(self.safety["gravity_m_s2"]))
        # A local ground avoids an unnecessary online Isaac asset-root lookup.
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
            prim_path="/World/DrawingSurface", name="drawing_surface",
            position=np.asarray([*self.drawing["surface_center_xy_m"], float(self.drawing["paper_top_z_m"]) - paper_thickness / 2]),
            scale=np.asarray([paper_size[0], paper_size[1], paper_thickness]),
            color=np.asarray([0.92, 0.92, 0.88]),
        ))
        urdf_path = Path(self.robot_config["urdf_path"])
        description = inspect_urdf(urdf_path)
        reference_root = add_robot_reference(self.robot_config["usd_path"], self.robot_config["prim_path"])
        for _ in range(10):
            simulation_app.update()
        articulation_root = find_articulation_root(self.world.stage, reference_root)
        prepare_articulation_solver(self.world.stage, articulation_root, self.robot_config)
        self.robot = self.world.scene.add(SingleArticulation(prim_path=articulation_root, name="xarm7_rl"))
        self.world.reset()
        try:
            from isaacsim.util.debug_draw import _debug_draw

            self.debug_draw = _debug_draw.acquire_debug_draw_interface()
        except Exception as exc:
            print(f"[RL] visual debug draw unavailable: {exc}")
            self.debug_draw = None
        self.controller = configure_position_drive(
            self.robot, self.robot_config, prefix="[RL CONTROL]"
        )
        limits = {name: (description.joint_limits[name].lower_rad, description.joint_limits[name].upper_rad)
                  for name in description.arm_joint_names}
        self.ik = SafeLulaIKController(
            self.robot, self.robot_config["robot_description_path"], urdf_path,
            self.robot_config["end_effector_frame"], limits,
            float(self.drawing["ik_position_tolerance_m"]),
            float(self.drawing["orientation_tolerance_rad"]),
            max_joint_delta_rad=float(self.robot_config["max_joint_delta_rad"]),
        )
        self.orientation = np.asarray(self.drawing["orientation_wxyz"], dtype=np.float64)
        self.home = np.asarray(self.robot_config["home_joint_positions_rad"], dtype=np.float64)
        self.last_score = 0.0
        self.last_proximity = 0.0
        self.observation_dim = self.width * self.height * 2 + 7
        print(f"[RL] observation_dim={self.observation_dim}, gravity={self.safety['gravity_m_s2']} m/s^2")

    def _apply_joint_target(self, ik_action):
        """Apply an IK target through the validated gravity-capable position servo."""
        self.ik.apply(ik_action)

    def _raster_xy(self, xy):
        center = np.asarray(self.drawing["surface_center_xy_m"], dtype=np.float64)
        size = np.asarray(self.drawing["surface_size_xy_m"], dtype=np.float64)
        normalized = (np.asarray(xy, dtype=np.float64) - center + size / 2) / size
        return np.asarray([normalized[0] * (self.width - 1), (1.0 - normalized[1]) * (self.height - 1)])

    def _rasterize_strokes(self, strokes):
        mask = np.zeros((self.height, self.width), dtype=np.float32)
        for stroke in strokes:
            points = np.asarray(stroke["points"], dtype=np.float64)
            for start, end in zip(points[:-1], points[1:]):
                distance = np.linalg.norm(end - start)
                count = max(2, int(np.ceil(distance / 0.001)))
                for point in np.linspace(start, end, count):
                    px, py = np.rint(self._raster_xy(point)).astype(int)
                    if 0 <= px < self.width and 0 <= py < self.height:
                        mask[py, px] = 1.0
        return mask

    def _draw_segment(self, start, end):
        a, b = self._raster_xy(start[:2]), self._raster_xy(end[:2])
        count = max(2, int(np.ceil(np.linalg.norm(b - a))))
        for point in np.linspace(a, b, count):
            px, py = np.rint(point).astype(int)
            if 0 <= px < self.width and 0 <= py < self.height:
                self.canvas[py, px] = 1.0

    def _score(self):
        target = self.target_mask > 0.5
        drawn = self.canvas > 0.5
        covered = float(np.logical_and(target, drawn).sum()) / max(1, int(target.sum()))
        outside = float(np.logical_and(~target, drawn).sum()) / max(1, int(drawn.sum()))
        return covered - float(self.rl["reward_outside_weight"]) * outside

    def _tip_proximity(self, tip):
        """Dense reward signal: closeness of the tip to any target pixel."""
        if len(self.target_pixels) == 0:
            return 0.0
        px = self._raster_xy(np.asarray(tip, dtype=np.float64)[:2]).astype(np.float32)
        distance = float(np.min(np.linalg.norm(self.target_pixels - px[None, :], axis=1)))
        return float(np.exp(-distance / 4.0))

    def _update_visualization(self):
        if self.debug_draw is None:
            return
        self.debug_draw.clear_lines()
        target_color = (0.1, 0.9, 0.2, 1.0)
        executed_color = (0.95, 0.15, 0.75, 1.0)
        z_target = float(self.drawing["pen_down_z_m"]) + 0.001
        for stroke in self.mapped["strokes"]:
            points = [(float(p[0]), float(p[1]), z_target) for p in stroke["points"]]
            if len(points) >= 2:
                self.debug_draw.draw_lines_spline(points, target_color, 2.0, False)
        if len(self.executed_points) >= 2:
            self.debug_draw.draw_lines_spline(
                [(float(p[0]), float(p[1]), float(p[2])) for p in self.executed_points],
                executed_color, 3.0, False,
            )

    def _get_observation(self):
        tip, _ = self.ik.end_effector_pose()
        tip = np.asarray(tip, dtype=np.float32)
        velocity = np.zeros(3, dtype=np.float32) if self.previous_tip is None else (tip - self.previous_tip) / float(self.safety["physics_dt_s"])
        self.previous_tip = tip.copy()
        center = np.asarray([*self.drawing["surface_center_xy_m"], self.drawing["pen_down_z_m"]], dtype=np.float32)
        tip_relative = tip - center
        pen_down = float(tip[2] < (float(self.drawing["pen_up_z_m"]) + float(self.drawing["pen_down_z_m"])) / 2)
        return np.concatenate([self.target_mask.ravel(), self.canvas.ravel(), tip_relative, velocity, [pen_down]]).astype(np.float32)

    def reset(self):
        from isaacsim.core.utils.types import ArticulationAction

        self.robot.set_joint_positions(self.home)
        self.robot.set_joint_velocities(np.zeros(7))
        self.ik.reset_command_state(self.home)
        self.canvas.fill(0.0)
        self.executed_points = []
        self.step_count = 0
        self.previous_tip = None
        self.previous_action.fill(0.0)
        home_action = ArticulationAction(joint_positions=self.home)
        settle_steps = int(self.robot_config.get("gravity_home_settle_steps", 20))
        for _ in range(max(20, settle_steps)):
            self._apply_joint_target(home_action)
            self.world.step(render=not ARGS.headless)
        self.last_score = self._score()
        tip, _ = self.ik.end_effector_pose()
        self.last_proximity = self._tip_proximity(tip)
        self._update_visualization()
        self.observation = self._get_observation()
        return self.observation.copy()

    def _hold_cartesian_target(self, target, max_steps=240):
        """Reach one fixed Cartesian target and report measured tip error."""
        target = np.asarray(target, dtype=np.float64)
        last_error = float("inf")
        for step in range(int(max_steps)):
            ik_action, success = self.ik.solve(target, self.orientation)
            if not success:
                print(f"[REACH TEST] IK failed at step={step}, target={target.tolist()}")
                return False
            if step == 0:
                print(
                    "[REACH TEST] first IK command="
                    f"{np.round(np.asarray(ik_action.joint_positions), 4).tolist()} "
                    f"measured_q={np.round(np.asarray(self.robot.get_joint_positions()), 4).tolist()}"
                )
            for _ in range(max(1, self.control_substeps)):
                self._apply_joint_target(ik_action)
                self.world.step(render=not ARGS.headless)
            tip, _ = self.ik.end_effector_pose()
            tip = np.asarray(tip, dtype=np.float64)
            last_error = float(np.linalg.norm(tip - target))
            if step % 20 == 0:
                print(f"[REACH TEST] step={step} tip={tip.tolist()} error_m={last_error:.6f}")
            if last_error <= max(self.contact_z_tolerance, 0.002):
                print(f"[REACH TEST] reached target with error_m={last_error:.6f}")
                return True
        print(f"[REACH TEST] FAILED; final error_m={last_error:.6f}, tip={tip.tolist()}")
        return False

    def run_reach_test(self):
        """Check home -> above paper -> paper contact under gravity."""
        from articulation_control import generalized_gravity_forces

        self.reset()
        gravity_load = generalized_gravity_forces(self.robot)
        available = np.asarray(self.controller.get_max_efforts(), dtype=np.float64).reshape(-1)[:7]
        home_q = np.asarray(self.robot.get_joint_positions(), dtype=np.float64)
        home_qd = np.asarray(self.robot.get_joint_velocities(), dtype=np.float64)
        home_tip, _ = self.ik.end_effector_pose()
        actual_kps, actual_kds = self.controller.get_gains()
        applied = self.controller.get_applied_action()
        print(
            "[REACH TEST] home gravity_load="
            f"{np.round(gravity_load, 3).tolist()} max_force={np.round(available, 3).tolist()} "
            f"q={np.round(home_q, 4).tolist()} q_error="
            f"{np.round(home_q - self.home, 4).tolist()} "
            f"qd={np.round(home_qd, 4).tolist()} tip={np.round(home_tip, 4).tolist()}"
        )
        print(
            "[REACH TEST] post-settle drive "
            f"kp={np.round(np.asarray(actual_kps), 3).tolist()} "
            f"kd={np.round(np.asarray(actual_kds), 3).tolist()} "
            f"position_target={np.round(np.asarray(applied.joint_positions), 4).tolist()}"
        )
        center_x, center_y = self.drawing["surface_center_xy_m"]
        approach = np.asarray([center_x, center_y, self.drawing["approach_height_m"]], dtype=np.float64)
        down = np.asarray([center_x, center_y, self.drawing["pen_down_z_m"]], dtype=np.float64)
        print(f"[REACH TEST] approach target={approach.tolist()}")
        if not self._hold_cartesian_target(approach):
            return False
        print(f"[REACH TEST] pen-down target={down.tolist()}")
        return self._hold_cartesian_target(down)

    def step(self, action):
        from isaacsim.core.utils.types import ArticulationAction

        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        tip, _ = self.ik.end_effector_pose()
        tip = np.asarray(tip, dtype=np.float64)
        desired = tip + action[:3] * self.action_limit
        pen_down = bool((action[3] + 1.0) / 2.0 > self.pen_down_threshold)
        desired[2] = float(self.drawing["pen_down_z_m"] if pen_down else self.drawing["pen_up_z_m"])
        desired = np.clip(desired, np.asarray(self.safety["workspace_min_m"]), np.asarray(self.safety["workspace_max_m"]))
        # Hold the same Cartesian target for several physics frames. A single
        # physics frame is not enough for the gravity-loaded position drives to
        # settle, so advancing the policy every frame makes the pen lag and
        # visibly draw above the paper.
        ik_action, success = self.ik.solve(desired, self.orientation)
        for _ in range(max(1, self.control_substeps)):
            if success:
                self._apply_joint_target(ik_action)
            self.world.step(render=not ARGS.headless)
        new_tip, _ = self.ik.end_effector_pose()
        new_tip = np.asarray(new_tip, dtype=np.float64)
        contact_error = abs(float(new_tip[2]) - float(self.drawing["pen_down_z_m"]))
        in_contact = contact_error <= self.contact_z_tolerance
        if pen_down and success and in_contact:
            self._draw_segment(tip, new_tip)
            self.executed_points.append(new_tip.copy())
        score = self._score()
        progress = score - self.last_score
        proximity = self._tip_proximity(new_tip)
        proximity_progress = proximity - self.last_proximity
        smoothness = float(np.linalg.norm(action - self.previous_action))
        reward = progress - float(self.rl["reward_action_weight"]) * float(np.linalg.norm(action[:3]))
        reward -= float(self.rl["reward_smoothness_weight"]) * smoothness
        reward += float(self.rl["dense_distance_weight"]) * 10.0 * proximity_progress
        if pen_down:
            reward -= float(self.rl.get("contact_penalty_weight", 0.25)) * min(
                1.0, contact_error / max(self.contact_z_tolerance, 1e-6)
            )
        if not success:
            reward -= 1.0
        self.last_score = score
        self.last_proximity = proximity
        self.previous_action = action.astype(np.float32)
        self.step_count += 1
        if self.step_count % max(1, int(self.rl["visualization_stride"])) == 0:
            self._update_visualization()
        done = self.step_count >= self.max_steps or score >= 0.98
        self.observation = self._get_observation()
        return self.observation.copy(), float(reward), bool(done), {
            "score": score,
            "ik_success": success,
            "contact": in_contact,
            "contact_error_m": contact_error,
            "tip_z_m": float(new_tip[2]),
        }


def main():
    import torch
    from project_config import load_config
    from rl_policy import DrawingActorCritic, collect_rollout, compute_gae, ppo_update, save_policy

    np.random.seed(ARGS.seed)
    torch.manual_seed(ARGS.seed)
    config = load_config(ARGS.config, PROJECT_ROOT)
    trajectory = ARGS.trajectory or config["project"]["trajectory_path"]
    env = IsaacDrawingEnv(config, trajectory)
    if ARGS.reach_test:
        if not env.run_reach_test():
            raise RuntimeError("The arm could not reach the paper under gravity; do not train RL yet")
        return
    device = torch.device(ARGS.device if ARGS.device == "cuda" and torch.cuda.is_available() else "cpu")
    policy = DrawingActorCritic(env.observation_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(config["rl"]["ppo_learning_rate"]))
    rl = config["rl"]
    for update in range(int(rl["ppo_updates"])):
        rollout = collect_rollout(env, policy, int(rl["ppo_steps_per_update"]), device)
        advantages, returns = compute_gae(rollout, float(rl["ppo_gamma"]), float(rl["ppo_gae_lambda"]))
        stats = ppo_update(policy, optimizer, rollout, advantages, returns,
                           float(rl["ppo_clip_ratio"]), int(rl["ppo_epochs"]),
                           int(rl["ppo_minibatch_size"]), float(rl["ppo_entropy_weight"]), device)
        if update % 10 == 0:
            print(f"[RL] update={update} mean_reward={rollout.rewards.mean():.5f} "
                  f"score={env.last_score:.3f} policy_loss={stats['policy_loss']:.5f}")
            save_policy(config["rl"]["checkpoint_path"], policy, env.observation_dim,
                        {"gravity_m_s2": config["safety"]["gravity_m_s2"], "trajectory_used_only_to_make_target": True})
    save_policy(config["rl"]["checkpoint_path"], policy, env.observation_dim,
                {"gravity_m_s2": config["safety"]["gravity_m_s2"], "trajectory_used_only_to_make_target": True})
    print(f"[OK] RL checkpoint: {config['rl']['checkpoint_path']}")


if __name__ == "__main__":
    code = 1
    try:
        main()
        code = 0
    except Exception as exc:
        import traceback

        print(f"[RL ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
    finally:
        simulation_app.close(exit_code=code)
    raise SystemExit(code)
