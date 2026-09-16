"""ScriptNode controller: drive the xArm7 pen through any project trajectory.

Runs inside the Isaac Sim Kit process via an Action Graph
(OnPlaybackTick -> ScriptNode), driven live through the isaacsim-mcp-server
MCP connection. Only the IK solve and articulation I/O touch Isaac Sim APIs;
trajectory mapping and sequencing reuse this project's pure-Python
src/ modules (coordinate_mapper, drawing_state_machine, trajectory_loader)
so the same logic that is unit-tested offline drives the real robot.

Which trajectory to draw is read from JOB_FILE at the start of each run (INIT
state), not hardcoded — write a new {"trajectory_path": ..., "prefix": ...}
there and Stop+Play to run a different animal without editing/reloading this
script. On completion, both the desired (mapped) and executed pen-down paths
are authored as permanent BasisCurves on the stage under /World/DrawnResult
so a capture_image afterward actually shows the drawing, not just a bare pen
over blank paper.

setup() runs once; compute() runs every tick. State resets on timeline STOP
so the next Play re-inits and re-homes.
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(r"\\wsl.localhost\Ubuntu-24.04\home\shravan\Workspace\WildTrace\isaac-sim")
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import carb
import numpy as np
import omni.timeline

from coordinate_mapper import interpolate_segment, map_trajectory_to_plane
from drawing_state_machine import MotionTarget, MultiStrokeStateMachine, Phase, build_motion_sequence
from metrics import nearest_path_errors, summarize_errors, write_csv, write_metrics
from trajectory_loader import load_trajectory

# ── Config (mirrors config/xarm7_drawing.yaml) ───────────────
ROBOT_PRIM_PATH = "/World/xarm7"
END_EFFECTOR_FRAME = "pen_tip"
ROBOT_DESCRIPTION_PATH = str(PROJECT_ROOT / "assets" / "xarm7" / "robot_description.yaml")
URDF_PATH = str(PROJECT_ROOT / "assets" / "xarm7" / "xarm7_with_pen.urdf")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mcp_sessions"
JOB_FILE = OUTPUT_DIR / "job.json"
DEFAULT_TRAJECTORY_PATH = str(PROJECT_ROOT / "inputs" / "trajectories" / "Cat" / "70bc30f8a5f918eb.json")
DEFAULT_PREFIX = "cat"
DRAWN_RESULT_PRIM = "/World/DrawnResult"

SURFACE_CENTER_XY_M = [0.45, 0.0]
SURFACE_SIZE_XY_M = [0.16, 0.12]
FLIP_IMAGE_Y = True
# Raised from 0.08 and step halved from 0.005: contour-extraction noise plus
# large per-step direction changes at corners caused overshoot loops ("knots")
# in a tolerance-gated per-waypoint IK controller. See config/xarm7_drawing.yaml
# for the matching rationale.
SMOOTHING_STRENGTH = 0.25
CORNER_ANGLE_DEGREES = 35.0
PEN_DOWN_Z_M = 0.201
PEN_UP_Z_M = 0.235
APPROACH_HEIGHT_M = 0.27
MAX_CARTESIAN_STEP_M = 0.002
TARGET_TOLERANCE_M = 0.003
ORIENTATION_TOLERANCE_RAD = 0.04
ORIENTATION_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])

HOME_JOINT_POSITIONS = np.array([0.0, -0.35, 0.0, 1.20, 0.0, 1.55, 0.0])
HOME_JOINT_TOLERANCE_RAD = 0.02
# The USD's URDF-imported drive gains are far weaker than these; without
# reasserting them at runtime the arm creeps toward a target asymptotically
# and misses the waypoint timeout well before closing the last few mm.
DRIVE_STIFFNESS = 400.0
DRIVE_DAMPING = 40.0
JOINT_LOWER = np.array(
    [-6.283185307179586, -2.058999960451334, -6.283185307179586, -0.1919799925803282,
     -6.283185307179586, -1.6929700402247723, -6.283185307179586]
)
JOINT_UPPER = np.array(
    [6.283185307179586, 2.094400029241212, 6.283185307179586, 3.9269998717349477,
     6.283185307179586, 3.141592653589793, 6.283185307179586]
)

WAYPOINT_TIMEOUT_S = 4.0
HOME_TIMEOUT_S = 12.0
RUN_TIMEOUT_S = 300.0
PHYSICS_DT_S = 1.0 / 60.0
WARMUP_FRAMES = 30

WARMUP, INIT, HOME, DRAWING, FINISHED, FAILED = "WARMUP", "INIT", "HOME", "DRAWING", "FINISHED", "FAILED"

# ── Persistent state (module globals; ScriptNode "proper mode") ─
_state = WARMUP
_frame = 0
_sim_time = 0.0
_world = None
_robot = None
_solver = None
_machine = None
_mapped = None
_prefix = DEFAULT_PREFIX
_home_action = None
_tl_sub = None
_executed_rows = []
_reached_errors = []
_report_written = False


def _log(msg):
    carb.log_warn(f"[Drawing] {msg}")


def _go(new_state):
    global _state
    if new_state != _state:
        _log(f"{_state} -> {new_state}")
        _state = new_state


def _reset(_event=None):
    global _state, _frame, _sim_time, _world, _robot, _solver, _machine, _mapped, _prefix
    global _home_action, _executed_rows, _reached_errors, _report_written
    _state = WARMUP
    _frame = 0
    _sim_time = 0.0
    _robot = None
    _solver = None
    _machine = None
    _mapped = None
    _prefix = DEFAULT_PREFIX
    _home_action = None
    _executed_rows = []
    _reached_errors = []
    _report_written = False
    if _world is not None:
        _world.clear_instance()
        _world = None
    _log("Reset")


def _on_timeline_event(event):
    if event.type == int(omni.timeline.TimelineEventType.STOP):
        _reset()


def setup(db=None):
    global _tl_sub
    if _tl_sub is None:
        tl = omni.timeline.get_timeline_interface()
        _tl_sub = tl.get_timeline_event_stream().create_subscription_to_pop(_on_timeline_event)
        _log("setup()")


def _load_job():
    try:
        job = json.loads(JOB_FILE.read_text(encoding="utf-8"))
        return str(job.get("trajectory_path", DEFAULT_TRAJECTORY_PATH)), str(job.get("prefix", DEFAULT_PREFIX))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return DEFAULT_TRAJECTORY_PATH, DEFAULT_PREFIX


def _build_machine(trajectory_path):
    trajectory = load_trajectory(trajectory_path)
    mapped = map_trajectory_to_plane(
        trajectory,
        center_xy=SURFACE_CENTER_XY_M,
        size_xy=SURFACE_SIZE_XY_M,
        flip_image_y=FLIP_IMAGE_Y,
        max_step=MAX_CARTESIAN_STEP_M,
        smoothing_strength=SMOOTHING_STRENGTH,
        corner_angle_degrees=CORNER_ANGLE_DEGREES,
    )
    phases = build_motion_sequence(
        mapped["strokes"],
        pen_down_z=PEN_DOWN_Z_M,
        pen_up_z=PEN_UP_Z_M,
        approach_height=APPROACH_HEIGHT_M,
        max_cartesian_step=MAX_CARTESIAN_STEP_M,
    )
    return mapped, MultiStrokeStateMachine(phases, WAYPOINT_TIMEOUT_S)


def _clear_drawn_result():
    """Wipe any previous run's ink off the paper. Called at the start of
    INIT (not just after a run finishes) so a new Play doesn't draw the new
    animal on top of the old one still sitting on the stage."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    old_prim = stage.GetPrimAtPath(DRAWN_RESULT_PRIM)
    if old_prim and old_prim.IsValid():
        stage.RemovePrim(DRAWN_RESULT_PRIM)


def _author_drawn_curves():
    """Author the desired (green) and executed (black) pen-down paths as
    permanent BasisCurves on the stage so a capture_image shows the actual
    drawing instead of a bare pen over blank paper."""
    from pxr import Gf, UsdGeom
    import omni.usd

    _clear_drawn_result()
    stage = omni.usd.get_context().get_stage()

    def _curve(name, strokes_points, z, color, width):
        points, counts = [], []
        for stroke_points in strokes_points:
            if len(stroke_points) < 2:
                continue
            points.extend([Gf.Vec3f(float(p[0]), float(p[1]), float(z)) for p in stroke_points])
            counts.append(len(stroke_points))
        if not counts:
            return
        curve = UsdGeom.BasisCurves.Define(stage, f"{DRAWN_RESULT_PRIM}/{name}")
        curve.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
        curve.CreateCurveVertexCountsAttr().Set(counts)
        curve.CreatePointsAttr().Set(points)
        curve.CreateWidthsAttr().Set([float(width)])
        curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        curve.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])

    desired_strokes = [stroke["points"] for stroke in _mapped["strokes"]] if _mapped else []
    executed_by_stroke = {}
    for row in _executed_rows:
        if row["pen_down"]:
            executed_by_stroke.setdefault(row["stroke_id"], []).append(
                [row["tip_x_m"], row["tip_y_m"]]
            )
    _curve("Desired", desired_strokes, PEN_DOWN_Z_M + 0.0005, (0.15, 0.7, 0.15), 0.0015)
    _curve("Executed", list(executed_by_stroke.values()), PEN_DOWN_Z_M + 0.001, (0.05, 0.05, 0.05), 0.0018)


def _write_report():
    global _report_written
    if _report_written:
        return
    _report_written = True
    fields = [
        "simulation_time_s", "state", "stroke_id", "waypoint_index", "pen_down",
        "target_x_m", "target_y_m", "target_z_m", "tip_x_m", "tip_y_m", "tip_z_m", "tracking_error_m",
    ]
    write_csv(OUTPUT_DIR / f"{_prefix}_executed_path.csv", _executed_rows, fields)
    try:
        _author_drawn_curves()
    except Exception as exc:
        import traceback

        traceback.print_exc()
        _log(f"Curve authoring error: {exc}")
    desired_down = [
        [*point, PEN_DOWN_Z_M] for stroke in _mapped["strokes"] for point in stroke["points"]
    ] if _mapped else []
    executed_down = [
        [row["tip_x_m"], row["tip_y_m"], row["tip_z_m"]] for row in _executed_rows if row["pen_down"]
    ]
    metrics = {
        "status": "finished" if _machine.complete else "failed",
        "failure_reason": _machine.failure_reason,
        "drawing_id": _mapped["drawing_id"] if _mapped else None,
        "stroke_count": len(_mapped["strokes"]) if _mapped else 0,
        "visited_states": _machine.visited_states,
        "simulation_duration_s": _sim_time,
        "executed_sample_count": len(_executed_rows),
        "desired_pen_down_points": len(desired_down),
        "executed_pen_down_samples": len(executed_down),
    }
    if _reached_errors:
        metrics["waypoint_reach_error"] = summarize_errors(_reached_errors)
    if executed_down and desired_down:
        metrics["executed_to_desired_nearest_path_error"] = summarize_errors(
            nearest_path_errors(desired_down, executed_down)
        )
        metrics["desired_to_executed_nearest_path_error"] = summarize_errors(
            nearest_path_errors(executed_down, desired_down)
        )
    write_metrics(OUTPUT_DIR / f"{_prefix}_run_metrics.json", metrics)
    _log(f"Report written: {metrics['status']}, {len(_executed_rows)} samples")


def compute(db=None):
    global _state, _frame, _sim_time, _world, _robot, _solver, _machine, _mapped, _prefix, _home_action

    if _state == WARMUP:
        _frame += 1
        if _frame >= WARMUP_FRAMES:
            from isaacsim.core.api import World

            _world = World.instance()
            if _world is None:
                _world = World(physics_dt=PHYSICS_DT_S, stage_units_in_meters=1.0)
            _world.initialize_physics()
            _log("World + physics initialized")
            _go(INIT)
        return True

    if _state == INIT:
        try:
            from isaacsim.core.prims import SingleArticulation
            from isaacsim.core.utils.types import ArticulationAction
            from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver

            _robot = SingleArticulation(prim_path=ROBOT_PRIM_PATH, name="xarm7_mcp_cat")
            _robot.initialize()

            articulation_controller = _robot.get_articulation_controller()
            articulation_controller.switch_control_mode("position")
            num_dof = len(_robot.dof_names)
            articulation_controller.set_gains(
                kps=np.full(num_dof, DRIVE_STIFFNESS, dtype=np.float64),
                kds=np.full(num_dof, DRIVE_DAMPING, dtype=np.float64),
            )
            _readback_kps, _readback_kds = articulation_controller.get_gains()
            _log(f"Gains set, readback kps={_readback_kps.tolist()} kds={_readback_kds.tolist()}")

            lula = LulaKinematicsSolver(ROBOT_DESCRIPTION_PATH, URDF_PATH)
            base_position, base_orientation = _robot.get_world_pose()
            lula.set_robot_base_pose(np.asarray(base_position), np.asarray(base_orientation))
            _solver = ArticulationKinematicsSolver(_robot, lula, END_EFFECTOR_FRAME)

            trajectory_path, _prefix = _load_job()
            _mapped, _machine = _build_machine(trajectory_path)
            try:
                _clear_drawn_result()
            except Exception as exc:
                _log(f"Clear old drawing error: {exc}")
            _home_action = ArticulationAction(joint_positions=HOME_JOINT_POSITIONS)
            _sim_time = 0.0
            _log(f"[{_prefix}] Loaded '{_mapped['drawing_id']}' ({len(_mapped['strokes'])} strokes, {len(_machine.phases)} phases) from {trajectory_path}")
            _go(HOME)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            _log(f"Init error: {exc}")
            _go(FAILED)
        return True

    if _state == HOME:
        _robot.get_articulation_controller().apply_action(_home_action)
        q = np.asarray(_robot.get_joint_positions(), dtype=np.float64)
        _sim_time += PHYSICS_DT_S
        err = float(np.max(np.abs(q - HOME_JOINT_POSITIONS)))
        if err <= HOME_JOINT_TOLERANCE_RAD:
            # Subdivide the home -> first-approach leg from the actual settled
            # pen-tip pose; build_motion_sequence only knows the single
            # end-of-leg point, and a one-shot jump there exceeds the
            # effort-capped drives' reach within a waypoint timeout.
            home_tip, _ = _solver.compute_end_effector_pose(position_only=False)
            approach_phase = _machine.phases[0]
            final_approach = approach_phase.targets[-1]
            approach_points = interpolate_segment(home_tip, final_approach.position, MAX_CARTESIAN_STEP_M)[1:]
            _machine.phases[0] = Phase(
                approach_phase.state,
                [
                    MotionTarget(approach_phase.state, point, final_approach.stroke_id, index, False)
                    for index, point in enumerate(approach_points)
                ],
            )
            _machine.mark_home_reached(_sim_time)
            _log(f"HOME reached (err={err:.4f}) -> {_machine.state} ({len(approach_points)} approach waypoints)")
            _go(DRAWING)
        elif _sim_time > HOME_TIMEOUT_S:
            _log(f"Home timeout (err={err:.4f})")
            _go(FAILED)
        return True

    if _state == DRAWING:
        _sim_time += PHYSICS_DT_S
        if _sim_time > RUN_TIMEOUT_S:
            _machine.fail("Overall run timeout")
        else:
            target = _machine.current_target
            if target is not None:
                action, success = _solver.compute_inverse_kinematics(
                    target.position,
                    ORIENTATION_WXYZ,
                    position_tolerance=TARGET_TOLERANCE_M,
                    orientation_tolerance=ORIENTATION_TOLERANCE_RAD,
                )
                if not success or action.joint_positions is None:
                    _machine.fail(f"IK failed in {_machine.state}, stroke={target.stroke_id}, waypoint={target.waypoint_index}")
                else:
                    proposed = np.asarray(action.joint_positions, dtype=np.float64)
                    if not np.isfinite(proposed).all() or np.any(proposed < JOINT_LOWER) or np.any(proposed > JOINT_UPPER):
                        _machine.fail(f"IK out-of-limits in {_machine.state}")
                    else:
                        _robot.get_articulation_controller().apply_action(action)
                        tip_position, _tip_rot = _solver.compute_end_effector_pose(position_only=False)
                        error = float(np.linalg.norm(np.asarray(tip_position) - target.position))
                        _executed_rows.append(
                            {
                                "simulation_time_s": _sim_time,
                                "state": _machine.state,
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
                        reached = error <= TARGET_TOLERANCE_M
                        if reached:
                            _reached_errors.append(error)
                        previous_phase = _machine.state
                        _machine.update(_sim_time, reached)
                        if _machine.state != previous_phase:
                            _log(f"{previous_phase} -> {_machine.state}")
        if _machine.complete:
            _write_report()
            _go(FINISHED)
        elif _machine.failed:
            _log(f"FAILED: {_machine.failure_reason}")
            _write_report()
            _go(FAILED)
        return True

    if _state in (FINISHED, FAILED):
        return True

    return True
