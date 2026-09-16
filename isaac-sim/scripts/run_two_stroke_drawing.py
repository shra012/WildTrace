"""Standalone deterministic two-stroke xArm 7 drawing baseline for Isaac Sim 6.0.1."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))
PORTABLE_ROOT = PROJECT_ROOT / "outputs" / "isaac_portable"
PORTABLE_ROOT.mkdir(parents=True, exist_ok=True)
if "--portable-root" not in sys.argv:
    sys.argv.extend(["--portable-root", str(PORTABLE_ROOT)])


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/xarm7_drawing.yaml")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--record-demonstration", action="store_true")
    parser.add_argument("--portable-root", default=str(PORTABLE_ROOT), help=argparse.SUPPRESS)
    parser.add_argument("--max-steps", type=int, default=0, help="Diagnostic hard stop; 0 disables it")
    parser.add_argument("--trajectory", default=None, help="Override project.trajectory_path")
    parser.add_argument("--first-n-strokes", type=int, default=None, help="Only draw the first N strokes")
    return parser.parse_args()


ARGS = _parse_args()

# Required ordering: launch before any Omniverse/Isaac runtime module import.
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": ARGS.headless, "width": 1280, "height": 720})


def _create_curve(stage, path: str, strokes, z: float, color, width: float) -> None:
    from pxr import Gf, UsdGeom

    points = []
    counts = []
    for stroke in strokes:
        stroke_points = [[float(p[0]), float(p[1]), float(z)] for p in stroke["points"]]
        points.extend(stroke_points)
        counts.append(len(stroke_points))
    curve = UsdGeom.BasisCurves.Define(stage, path)
    curve.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
    curve.CreateCurveVertexCountsAttr().Set(counts)
    curve.CreatePointsAttr().Set([Gf.Vec3f(*point) for point in points])
    curve.CreateWidthsAttr().Set([float(width)])
    curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    curve.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])


def _workspace_check(targets, minimum, maximum) -> None:
    import numpy as np

    low = np.asarray(minimum, dtype=np.float64)
    high = np.asarray(maximum, dtype=np.float64)
    for index, target in enumerate(targets):
        if not np.isfinite(target.position).all() or np.any(target.position < low) or np.any(target.position > high):
            raise RuntimeError(f"Target {index} in {target.state.value} is outside workspace: {target.position}")


def _lookahead(machine, count: int):
    import numpy as np

    target = machine.current_target
    values = machine.phases[machine.state]
    selected = [value.position for value in values[machine.target_index + 1 : machine.target_index + 1 + count]]
    pad = target.position if not selected else selected[-1]
    while len(selected) < count:
        selected.append(np.asarray(pad, dtype=np.float64))
    return np.asarray(selected, dtype=np.float64)


def _draw_debug_trace(debug_draw, desired_strokes, actual_by_stroke) -> None:
    if debug_draw is None:
        return
    debug_draw.clear_lines()
    for stroke in desired_strokes:
        points = [tuple(map(float, point)) for point in stroke]
        if len(points) >= 2:
            debug_draw.draw_lines_spline(points, (0.1, 0.9, 0.2, 1.0), 2.0, False)
    colors = [(0.95, 0.15, 0.75, 1.0), (0.15, 0.55, 1.0, 1.0)]
    for ordinal, (_stroke_id, samples) in enumerate(sorted(actual_by_stroke.items())):
        if len(samples) >= 2:
            debug_draw.draw_lines_spline(
                [tuple(map(float, point)) for point in samples], colors[ordinal % len(colors)], 3.0, False
            )


def _write_comparison(path: Path, desired_strokes, executed_by_stroke) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axis = plt.subplots(figsize=(8, 6))
    for ordinal, (stroke_id, points) in enumerate(desired_strokes.items()):
        array = np.asarray(points)
        axis.plot(array[:, 0], array[:, 1], "--", linewidth=2, label=f"desired stroke {stroke_id}")
        executed = np.asarray(executed_by_stroke.get(stroke_id, []))
        if len(executed):
            axis.plot(executed[:, 0], executed[:, 1], linewidth=1.2, label=f"executed stroke {stroke_id}")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("world X (m)")
    axis.set_ylabel("world Y (m)")
    axis.set_title("xArm 7 deterministic two-stroke drawing")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> int:
    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCuboid, VisualSphere
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.extensions import enable_extension
    from isaacsim.core.utils.stage import create_new_stage
    from isaacsim.core.utils.types import ArticulationAction

    from coordinate_mapper import interpolate_segment, map_trajectory_to_plane
    from demonstration import DemonstrationRecorder
    from drawing_state_machine import (
        HOME,
        MotionTarget,
        MultiStrokeStateMachine,
        Phase,
        build_motion_sequence,
        flatten_desired_targets,
    )
    from ik_controller import SafeLulaIKController
    from metrics import nearest_path_errors, summarize_errors, write_csv, write_metrics
    from project_config import load_config
    from trajectory_loader import load_trajectory
    from xarm7_loader import add_robot_reference, find_articulation_root, inspect_urdf

    config = load_config(ARGS.config, PROJECT_ROOT)
    robot_config, drawing, safety = config["robot"], config["drawing"], config["safety"]
    output_dir = Path(config["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    urdf_path, usd_path = Path(robot_config["urdf_path"]), Path(robot_config["usd_path"])
    if not urdf_path.is_file():
        raise FileNotFoundError(f"Generate the URDF first with scripts\\generate_xarm7_urdf.py: {urdf_path}")
    if not usd_path.is_file():
        raise FileNotFoundError(f"Run scripts\\check_environment.py to import the reusable USD first: {usd_path}")
    description = inspect_urdf(urdf_path)
    if robot_config["end_effector_frame"] != description.pen_tip_link:
        raise RuntimeError("Configured IK frame does not match the generated fixed pen-tip frame")

    trajectory_path = ARGS.trajectory or config["project"]["trajectory_path"]
    trajectory = load_trajectory(trajectory_path, first_n_strokes=ARGS.first_n_strokes)
    mapped = map_trajectory_to_plane(
        trajectory,
        center_xy=drawing["surface_center_xy_m"],
        size_xy=drawing["surface_size_xy_m"],
        flip_image_y=bool(drawing["flip_image_y"]),
        max_step=float(drawing["max_cartesian_step_m"]),
        smoothing_strength=float(drawing["smoothing_strength"]),
        corner_angle_degrees=float(drawing["corner_angle_degrees"]),
    )

    enable_extension("isaacsim.asset.importer.urdf")
    enable_extension("isaacsim.robot_motion.motion_generation")
    enable_extension("isaacsim.util.debug_draw")
    for _ in range(10):
        simulation_app.update()

    create_new_stage()
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=float(safety["physics_dt_s"]),
        rendering_dt=float(safety["physics_dt_s"]),
    )
    world.get_physics_context().set_gravity(float(safety["gravity_m_s2"]))
    world.scene.add_default_ground_plane()
    paper_size = np.asarray(drawing["surface_size_xy_m"], dtype=np.float64) + 0.04
    paper_thickness = 0.01
    world.scene.add(
        FixedCuboid(
            prim_path="/World/DrawingSurface",
            name="drawing_surface",
            position=np.asarray(
                [*drawing["surface_center_xy_m"], float(drawing["paper_top_z_m"]) - paper_thickness / 2]
            ),
            scale=np.asarray([paper_size[0], paper_size[1], paper_thickness]),
            color=np.asarray([0.92, 0.92, 0.88]),
        )
    )
    reference_root = add_robot_reference(usd_path, robot_config["prim_path"])
    for _ in range(10):
        simulation_app.update()
    articulation_root = find_articulation_root(world.stage, reference_root)
    robot = world.scene.add(SingleArticulation(prim_path=articulation_root, name="xarm7"))

    _create_curve(
        world.stage,
        "/World/DesiredDrawing",
        mapped["strokes"],
        float(drawing["pen_down_z_m"]),
        (0.1, 0.85, 0.2),
        0.002,
    )
    first_point = np.r_[mapped["strokes"][0]["points"][0], drawing["approach_height_m"]]
    target_marker = world.scene.add(
        VisualSphere(
            prim_path="/World/CurrentTarget",
            name="current_target",
            position=first_point,
            radius=0.007,
            color=np.asarray([1.0, 0.2, 0.05]),
        )
    )
    world.reset()
    actual_names = list(robot.dof_names)
    expected_names = list(robot_config["expected_joint_names"])
    if actual_names != expected_names:
        raise RuntimeError(f"Expected arm joints {expected_names}, imported {actual_names}")
    home = np.asarray(robot_config["home_joint_positions_rad"], dtype=np.float64)
    articulation_controller = robot.get_articulation_controller()
    articulation_controller.switch_control_mode("position")
    articulation_controller.set_gains(
        kps=np.full(len(expected_names), float(robot_config["drive_stiffness"]), dtype=np.float64),
        kds=np.full(len(expected_names), float(robot_config["drive_damping"]), dtype=np.float64),
    )
    actual_kps, actual_kds = articulation_controller.get_gains()
    print(f"[OK] Position-drive gains kp={actual_kps.tolist()}, kd={actual_kds.tolist()}")
    robot.set_joint_positions(home)
    home_action = ArticulationAction(joint_positions=home)
    for _ in range(10):
        articulation_controller.apply_action(home_action)
        world.step(render=not ARGS.headless)

    joint_limits = {
        name: (description.joint_limits[name].lower_rad, description.joint_limits[name].upper_rad)
        for name in description.arm_joint_names
    }
    ik = SafeLulaIKController(
        robot,
        robot_config["robot_description_path"],
        urdf_path,
        robot_config["end_effector_frame"],
        joint_limits,
        float(drawing["target_tolerance_m"]),
        float(drawing["orientation_tolerance_rad"]),
    )
    phases = build_motion_sequence(
        mapped["strokes"],
        pen_down_z=float(drawing["pen_down_z_m"]),
        pen_up_z=float(drawing["pen_up_z_m"]),
        approach_height=float(drawing["approach_height_m"]),
        max_cartesian_step=float(drawing["max_cartesian_step_m"]),
    )
    # Enforce max step from the actual simulated home pen pose to first approach.
    home_tip, _ = ik.end_effector_pose()
    approach_phase = phases[0]
    final_approach = approach_phase.targets[-1]
    approach_points = interpolate_segment(home_tip, final_approach.position, float(drawing["max_cartesian_step_m"]))[1:]
    phases[0] = Phase(
        approach_phase.state,
        [
            MotionTarget(approach_phase.state, point, final_approach.stroke_id, index, False)
            for index, point in enumerate(approach_points)
        ],
    )
    machine = MultiStrokeStateMachine(phases, float(safety["waypoint_timeout_s"]))
    desired_targets = flatten_desired_targets(phases)
    _workspace_check(desired_targets, safety["workspace_min_m"], safety["workspace_max_m"])

    desired_rows = [
        {
            "sequence_index": index,
            "state": target.state.value,
            "stroke_id": target.stroke_id,
            "waypoint_index": target.waypoint_index,
            "pen_down": int(target.pen_down),
            "x_m": float(target.position[0]),
            "y_m": float(target.position[1]),
            "z_m": float(target.position[2]),
        }
        for index, target in enumerate(desired_targets)
    ]
    write_csv(
        output_dir / "desired_path.csv",
        desired_rows,
        ["sequence_index", "state", "stroke_id", "waypoint_index", "pen_down", "x_m", "y_m", "z_m"],
    )

    try:
        from isaacsim.util.debug_draw import _debug_draw

        debug_draw = _debug_draw.acquire_debug_draw_interface()
    except Exception as exc:
        print(f"[WARN] Viewport debug trace unavailable: {exc}")
        debug_draw = None

    desired_down_by_stroke = {
        int(stroke["stroke_id"]): [[*point, float(drawing["pen_down_z_m"])] for point in stroke["points"]]
        for stroke in mapped["strokes"]
    }
    actual_by_stroke = {int(stroke["stroke_id"]): [] for stroke in mapped["strokes"]}
    executed_rows = []
    reached_target_errors = []
    step_count = 0
    start_time = float(world.current_time)
    home_start = start_time
    orientation = np.asarray(drawing["orientation_wxyz"], dtype=np.float64)
    recorder = None
    last_home_error = None
    if ARGS.record_demonstration:
        recorder = DemonstrationRecorder(
            mapped["drawing_id"],
            int(config["recording"]["lookahead_points"]),
            float(config["training"]["max_action_delta_m"]),
        )

    print(f"[OK] Drawing {mapped['drawing_id']} with exactly two strokes")
    print(f"[OK] Articulation root: {articulation_root}; joints: {actual_names}")
    print(f"[OK] Flange: {description.flange_link}; controlled fixed frame: {description.pen_tip_link}")
    while simulation_app.is_running() and not machine.complete and not machine.failed:
        sim_time = float(world.current_time)
        if sim_time - start_time > float(safety["run_timeout_s"]):
            machine.fail("Overall run timeout")
            break
        if ARGS.max_steps and step_count >= ARGS.max_steps:
            machine.fail(f"Diagnostic max-steps limit ({ARGS.max_steps})")
            break

        if machine.state == HOME:
            articulation_controller.apply_action(home_action)
            q = np.asarray(robot.get_joint_positions(), dtype=np.float64)
            last_home_error = float(np.max(np.abs(q - home)))
            if last_home_error <= float(robot_config["home_joint_tolerance_rad"]):
                machine.mark_home_reached(sim_time)
                print("[STATE] HOME -> APPROACH_STROKE_1")
            elif sim_time - home_start > float(safety["home_timeout_s"]):
                machine.fail(f"Home joint-position timeout (max error {last_home_error:.6f} rad, q={q.tolist()})")
        else:
            target = machine.current_target
            target_marker.set_world_pose(position=target.position)
            tip_position, tip_rotation = ik.end_effector_pose()
            error = float(np.linalg.norm(np.asarray(tip_position) - target.position))
            action, success = ik.solve(target.position, orientation)
            if not success:
                machine.fail(
                    f"IK failed in {machine.state.value}, stroke={target.stroke_id}, waypoint={target.waypoint_index}"
                )
                print(f"[SAFE STOP] {machine.failure_reason}")
                break
            ik.apply(action)
            q = np.asarray(robot.get_joint_positions(), dtype=np.float64)
            qd = np.asarray(robot.get_joint_velocities(), dtype=np.float64)
            executed_rows.append(
                {
                    "simulation_time_s": sim_time,
                    "state": machine.state.value,
                    "stroke_id": target.stroke_id,
                    "waypoint_index": target.waypoint_index,
                    "pen_down": int(target.pen_down),
                    "target_x_m": float(target.position[0]),
                    "target_y_m": float(target.position[1]),
                    "target_z_m": float(target.position[2]),
                    "tip_x_m": float(tip_position[0]),
                    "tip_y_m": float(tip_position[1]),
                    "tip_z_m": float(tip_position[2]),
                    "tracking_error_m": error,
                }
            )
            if target.pen_down:
                actual_by_stroke[target.stroke_id].append(np.asarray(tip_position).tolist())
            if recorder is not None:
                recorder.append(
                    stroke_id=target.stroke_id,
                    waypoint_index=target.waypoint_index,
                    simulation_time_s=sim_time,
                    joint_positions_rad=q,
                    joint_velocities_rad_s=qd,
                    pen_tip_position_m=tip_position,
                    pen_tip_rotation_matrix=tip_rotation,
                    current_target_m=target.position,
                    upcoming_targets_m=_lookahead(machine, recorder.lookahead_points),
                    pen_down=target.pen_down,
                    state=machine.state.value,
                )
            previous_state = machine.state
            reached = error <= float(drawing["target_tolerance_m"])
            if reached:
                reached_target_errors.append(error)
            machine.update(sim_time, reached)
            if machine.state != previous_state:
                print(f"[STATE] {previous_state.value} -> {machine.state.value}")

        world.step(render=not ARGS.headless)
        step_count += 1
        if step_count % 5 == 0:
            _draw_debug_trace(debug_draw, desired_down_by_stroke.values(), actual_by_stroke)

    fields = [
        "simulation_time_s",
        "state",
        "stroke_id",
        "waypoint_index",
        "pen_down",
        "target_x_m",
        "target_y_m",
        "target_z_m",
        "tip_x_m",
        "tip_y_m",
        "tip_z_m",
        "tracking_error_m",
    ]
    write_csv(output_dir / "executed_path.csv", executed_rows, fields)
    _write_comparison(
        output_dir / "two_stroke_comparison.png",
        {key: np.asarray(value)[:, :2] for key, value in desired_down_by_stroke.items()},
        {key: np.asarray(value)[:, :2] if len(value) else [] for key, value in actual_by_stroke.items()},
    )
    executed_down = [point for points in actual_by_stroke.values() for point in points]
    desired_down = [point for points in desired_down_by_stroke.values() for point in points]
    metrics = {
        "status": "finished" if machine.complete else "failed",
        "failure_reason": machine.failure_reason,
        "drawing_id": mapped["drawing_id"],
        "requested_stroke_count": 2,
        "visited_states": [state.value for state in machine.visited_states],
        "simulation_steps": step_count,
        "simulation_duration_s": float(world.current_time) - start_time,
        "desired_pen_down_points": len(desired_down),
        "executed_pen_down_samples": len(executed_down),
    }
    if reached_target_errors:
        metrics["waypoint_reach_error"] = summarize_errors(reached_target_errors)
    if executed_down:
        metrics["executed_to_desired_nearest_path_error"] = summarize_errors(
            nearest_path_errors(desired_down, executed_down)
        )
        metrics["desired_to_executed_nearest_path_error"] = summarize_errors(
            nearest_path_errors(executed_down, desired_down)
        )
    write_metrics(output_dir / "run_metrics.json", metrics)
    # A failed or partial run is never a behavioral-cloning demonstration.
    if recorder is not None and recorder.rows and machine.complete:
        saved_demo = recorder.save(config["recording"]["demo_path"])
        print(f"[OK] Demonstration: {saved_demo}")
    elif recorder is not None and recorder.rows:
        print("[WARN] Partial run was not saved as a demonstration")
    print(json.dumps(metrics, indent=2))
    print(f"[OK] Outputs: {output_dir}")
    return 0 if machine.complete else 2


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close(exit_code=code)
    raise SystemExit(code)
