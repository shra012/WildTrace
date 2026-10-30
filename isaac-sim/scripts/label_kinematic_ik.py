"""Label every generated trajectory with physics-free Lula IK demonstrations for ACT pretraining.

Runs in Isaac Sim's python (Lula ships there) but needs no SimulationApp:

    call C:\\isaac-sim-6.0.1\\python.bat \\\\wsl.localhost\\Ubuntu-24.04\\...\\isaac-sim\\scripts\\label_kinematic_ik.py ^
        --augmentations 3 --workers 16

Writes one format-version-2 NPZ per (trajectory, augmentation) to
<out>/<Category>/<sample_id>_a<k>.npz; a=0 is the canonical placement and
a>=1 use act_features.sample_augmentation. Existing files are skipped, so a
run can be resumed. Gold trajectories are excluded so they stay a clean test set.

--gaps (#46) tracks ink, injects skips and labels gap redraws. With
--policy-ports (DAgger), a policy served by scripts/act_policy_server.py on
each listed port (one per worker) drives the arm and the IK teacher labels the
states it reaches; --first-augmentation picks fresh placements for those.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from act_features import reverse_strokes, sample_augmentation, shift_paper_heights  # noqa: E402
from coordinate_mapper import map_trajectory_to_plane  # noqa: E402
from demonstration import DemonstrationRecorder  # noqa: E402
from drawing_state_machine import build_motion_sequence, flatten_desired_targets  # noqa: E402
from kinematic_labeller import LabelError, label_phases, with_approach_from  # noqa: E402
from project_config import load_config  # noqa: E402
from trajectory_loader import load_trajectory  # noqa: E402
from xarm7_loader import inspect_urdf  # noqa: E402

_WORKER: dict = {}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/xarm7_drawing.yaml")
    parser.add_argument("--trajectories", default=str(REPO_ROOT / "outputs" / "trajectories"))
    parser.add_argument("--exclude-dir", default=str(PROJECT_ROOT / "data" / "act_test"))
    parser.add_argument("--out", default=str(REPO_ROOT / "outputs" / "act_demos" / "kinematic_v2"))
    parser.add_argument("--augmentations", type=int, default=3, help="Episodes per trajectory (a0 = canonical)")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="Only the first N trajectories (0 = all)")
    parser.add_argument(
        "--noise-sigma-rad",
        type=float,
        default=0.002,
        help="DART joint noise on augmented episodes (a>=1); ~1.7 mm median, ~3.4 mm p95 at the tip. 0 disables",
    )
    parser.add_argument("--noise-correlation-s", type=float, default=0.5)
    parser.add_argument(
        "--offset-sigma-rad",
        type=float,
        default=0.0015,
        help="Measured-vs-commanded servo offset on augmented episodes (Isaac 1 g sag is up to ~0.0014 rad)",
    )
    parser.add_argument("--offset-correlation-s", type=float, default=2.0)
    parser.add_argument(
        "--noise-sigma-per-augmentation",
        default="",
        help="Comma list overriding --noise-sigma-rad per augmentation index (a1, a2, ...), e.g. 0.001,0.0005",
    )
    parser.add_argument(
        "--gaps",
        action="store_true",
        help="Track ink and redraw gaps (#46); records ink_skip and teacher_mode per frame",
    )
    parser.add_argument(
        "--skip-spacing-per-augmentation",
        default="",
        help="With --gaps: mean inked distance between injected skips per augmentation (a1, a2, ...), 0 = none",
    )
    parser.add_argument("--policy-ports", default="",
                        help="DAgger: comma list of policy server ports, one per worker (sets --workers)")
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-share", type=float, default=1.0, help="DAgger: share of ticks the policy executes")
    parser.add_argument("--first-augmentation", type=int, default=0,
                        help="label augmentations first..first+augmentations-1 (fresh placements for DAgger)")
    parser.add_argument("--gap-cover-tol-m", type=float, default=0.001,
                        help="With --gaps: a path point counts as drawn with ink this close to it")
    parser.add_argument(
        "--paper-height-jitter-m",
        type=float,
        default=0.015,
        help="Random paper height shift on augmented episodes (a>=1), so a policy survives the sheet being re-placed",
    )
    return parser.parse_args()


def _seed(sample_id: str, augmentation: int) -> int:
    return int(hashlib.sha256(f"{sample_id}:{augmentation}".encode()).hexdigest()[:8], 16)


def _init_worker(
    config_path: str,
    fps: float,
    noise_sigma: float,
    noise_correlation: float,
    offset_sigma: float,
    offset_correlation: float,
    paper_height_jitter: float,
    noise_per_augmentation: tuple = (),
    gaps: dict | None = None,
    policy: dict | None = None,
) -> None:
    import multiprocessing

    import lula

    config = load_config(config_path, PROJECT_ROOT)
    robot_config, drawing = config["robot"], config["drawing"]
    robot = lula.load_robot(robot_config["robot_description_path"], robot_config["urdf_path"])
    kinematics = robot.kinematics()
    frame = robot_config["end_effector_frame"]
    description = inspect_urdf(robot_config["urdf_path"])
    names = [robot.c_space_coord_name(i) for i in range(robot.num_c_space_coords())]
    lower = np.asarray([description.joint_limits[n].lower_rad for n in names])
    upper = np.asarray([description.joint_limits[n].upper_rad for n in names])
    w, x, y, z = (float(v) for v in drawing["orientation_wxyz"])
    rotation = lula.Rotation3(w, x, y, z)
    ik_config = lula.CyclicCoordDescentIkConfig()
    ik_config.position_tolerance = float(drawing["ik_position_tolerance_m"])
    ik_config.orientation_tolerance = float(drawing["orientation_tolerance_rad"])

    def solve(target, seed):
        ik_config.cspace_seeds = [np.asarray(seed, dtype=np.float64)]
        result = lula.compute_ik_ccd(kinematics, lula.Pose3(rotation, np.asarray(target)), frame, ik_config)
        if not result.success:
            return None
        q = np.asarray(result.cspace_position, dtype=np.float64)
        if not np.isfinite(q).all() or np.any(q < lower) or np.any(q > upper):
            return None
        return q

    def forward(q):
        pose = kinematics.pose(np.asarray(q, dtype=np.float64), frame)
        return np.asarray(pose.translation), np.asarray(pose.rotation.matrix())

    physics_dt = float(config["safety"]["physics_dt_s"])
    _WORKER.update(
        config=config,
        solve=solve,
        forward=forward,
        dt=1.0 / fps,
        # The runner's slew cap is per physics step; scale it to the label rate.
        max_joint_delta=float(robot_config["max_joint_delta_rad"]) * (1.0 / fps) / physics_dt,
        noise_sigma=float(noise_sigma),
        noise_correlation=float(noise_correlation),
        offset_sigma=float(offset_sigma),
        offset_correlation=float(offset_correlation),
        paper_height_jitter=float(paper_height_jitter),
        noise_per_augmentation=tuple(noise_per_augmentation),
        gaps=gaps,
        policy=None,
    )
    if policy:
        from act_client import ActPolicyClient

        # One server per worker: each serves one connection at a time.
        worker = (multiprocessing.current_process()._identity or (1,))[0] - 1
        port = policy["ports"][worker % len(policy["ports"])]
        _WORKER["policy"] = {"client": ActPolicyClient(f"{policy['host']}:{port}", timeout_s=120.0),
                             "share": policy["share"]}


def _per_augmentation(values: tuple, augmentation: int, default: float) -> float:
    """values[k - 1] for augmentation k >= 1, else the default."""
    return float(values[augmentation - 1]) if 1 <= augmentation <= len(values) else float(default)


def _label_one(job):
    path, category, augmentation, destination = job
    config = _WORKER["config"]
    robot_config, drawing, safety = config["robot"], config["drawing"], config["safety"]
    started = time.time()
    record = {"path": path, "category": category, "augmentation": augmentation, "output": destination}
    noise_sigma = _per_augmentation(_WORKER["noise_per_augmentation"], augmentation, _WORKER["noise_sigma"])
    gap_config = recorder = None
    if _WORKER["gaps"] is not None:
        spacing = _per_augmentation(_WORKER["gaps"]["skip_spacing_per_augmentation"], augmentation, 0.0)
        gap_config = {"cover_tol_m": _WORKER["gaps"]["cover_tol_m"],
                      "skip_spacing_m": spacing if augmentation > 0 else 0.0}
    try:
        trajectory = load_trajectory(path)
        center, size = drawing["surface_center_xy_m"], drawing["surface_size_xy_m"]
        placement = None
        if augmentation > 0:
            placement = sample_augmentation(
                np.random.default_rng(_seed(trajectory["drawing_id"], augmentation)),
                center,
                size,
                len(trajectory["strokes"]),
                height_jitter_m=_WORKER["paper_height_jitter"],
            )
            center, size = placement["surface_center_xy_m"], placement["surface_size_xy_m"]
            drawing = shift_paper_heights(drawing, placement["height_offset_m"])
            trajectory = reverse_strokes(trajectory, placement["reversed_strokes"])
        mapped = map_trajectory_to_plane(
            trajectory,
            center_xy=center,
            size_xy=size,
            flip_image_y=bool(drawing["flip_image_y"]),
            max_step=float(drawing["max_cartesian_step_m"]),
            smoothing_strength=float(drawing["smoothing_strength"]),
            corner_angle_degrees=float(drawing["corner_angle_degrees"]),
        )
        phases = build_motion_sequence(
            mapped["strokes"],
            pen_down_z=float(drawing["pen_down_z_m"]),
            pen_up_z=float(drawing["pen_up_z_m"]),
            approach_height=float(drawing["approach_height_m"]),
            max_cartesian_step=float(drawing["max_cartesian_step_m"]),
            corner_angle_degrees=float(drawing["corner_angle_degrees"]),
            corner_densify_window=int(drawing["corner_window_points"]),
            corner_densify_factor=int(drawing["corner_densify_factor"]),
            short_hop_m=float(drawing.get("short_hop_m", 0.0)),
        )
        home = np.asarray(robot_config["home_joint_positions_rad"], dtype=np.float64)
        home_tip, _ = _WORKER["forward"](home)
        phases = with_approach_from(phases, home_tip, float(drawing["max_cartesian_step_m"]))
        low, high = np.asarray(safety["workspace_min_m"]), np.asarray(safety["workspace_max_m"])
        for phase in phases:
            for target in phase.targets:
                if np.any(target.position < low) or np.any(target.position > high):
                    raise LabelError(f"Target outside workspace: {target.position}")
        recorder = DemonstrationRecorder(
            mapped["drawing_id"],
            int(config["recording"]["lookahead_points"]),
            float(config["training"]["max_action_delta_m"]),
            metadata={
                "source": "kinematic_lula",
                "category": category,
                "sample_id": mapped["drawing_id"],
                "trajectory_path": str(path),
                "augmentation_index": augmentation,
                "augmentation": placement,
                "surface_center_xy_m": list(map(float, center)),
                "surface_size_xy_m": list(map(float, size)),
                "label_dt_s": _WORKER["dt"],
                "noise_sigma_rad": noise_sigma if augmentation > 0 else 0.0,
                "gaps": gap_config,
                "noise_correlation_s": _WORKER["noise_correlation"],
                "offset_sigma_rad": _WORKER["offset_sigma"] if augmentation > 0 else 0.0,
            },
        )
        noise = None
        if augmentation > 0 and noise_sigma > 0:
            noise = {
                "rng": np.random.default_rng(_seed(mapped["drawing_id"], 1000 + augmentation)),
                "sigma_rad": noise_sigma,
                "correlation_s": _WORKER["noise_correlation"],
                "offset_sigma_rad": _WORKER["offset_sigma"],
                "offset_correlation_s": _WORKER["offset_correlation"],
            }
        policy = None
        if _WORKER["policy"]:
            client = _WORKER["policy"]["client"]
            flat = flatten_desired_targets(phases)
            client.reset(np.asarray([[t.position[0], t.position[1], float(t.pen_down)] for t in flat]))
            policy = {"act": client.act, "share": _WORKER["policy"]["share"],
                      "rng": np.random.default_rng(_seed(mapped["drawing_id"], 3000 + augmentation))}
        stats = label_phases(
            phases,
            home_q=home,
            solve_ik=_WORKER["solve"],
            forward=_WORKER["forward"],
            recorder=recorder,
            dt_s=_WORKER["dt"],
            max_joint_delta_rad=_WORKER["max_joint_delta"],
            noise=noise,
            gaps=None if gap_config is None else {
                **gap_config, "rng": np.random.default_rng(_seed(mapped["drawing_id"], 2000 + augmentation))},
            policy=policy,
        )
        recorder.save(destination)
        record.update(status="ok", sample_id=mapped["drawing_id"], **stats)
    except LabelError as exc:
        record.update(status="dropped", reason=str(exc))
        if _WORKER["policy"] and recorder is not None and len(recorder.rows) > 300:
            # A policy-driven episode that went wrong still shows the teacher's corrections up to there.
            recorder.save(destination)
            record.update(status="partial", frames=len(recorder.rows))
    except Exception as exc:  # noqa: BLE001 - a bad file must not stop the batch
        record.update(status="error", reason=repr(exc), traceback=traceback.format_exc(limit=3))
    record["seconds"] = round(time.time() - started, 3)
    return record


def _floats(text: str) -> tuple:
    return tuple(float(v) for v in text.split(",") if v.strip())


def main() -> int:
    args = _parse_args()
    excluded = {p.stem for p in Path(args.exclude_dir).rglob("*.json")} if args.exclude_dir else set()
    sources = sorted(p for p in Path(args.trajectories).rglob("*.json") if p.stem not in excluded)
    if args.limit:
        sources = sources[: args.limit]
    out = Path(args.out)
    jobs = []
    for path in sources:
        for augmentation in range(args.first_augmentation, args.first_augmentation + args.augmentations):
            destination = out / path.parent.name / f"{path.stem}_a{augmentation}.npz"
            if not destination.exists():
                jobs.append((str(path), path.parent.name, augmentation, str(destination)))
    print(f"[OK] {len(sources)} trajectories ({len(excluded)} gold excluded); {len(jobs)} episodes to label")
    out.mkdir(parents=True, exist_ok=True)
    policy = None
    if args.policy_ports:
        ports = [int(v) for v in args.policy_ports.split(",") if v.strip()]
        policy = {"ports": ports, "host": args.policy_host, "share": args.policy_share}
        args.workers = len(ports)
    manifest = out / "labels.ndjson"
    counts = {"ok": 0, "dropped": 0, "partial": 0, "error": 0}
    started = time.time()
    with Pool(args.workers, initializer=_init_worker, initargs=(
            args.config,
            args.fps,
            args.noise_sigma_rad,
            args.noise_correlation_s,
            args.offset_sigma_rad,
            args.offset_correlation_s,
            args.paper_height_jitter_m,
            _floats(args.noise_sigma_per_augmentation),
            {"cover_tol_m": args.gap_cover_tol_m,
             "skip_spacing_per_augmentation": _floats(args.skip_spacing_per_augmentation)} if args.gaps else None,
            policy,
        )) as pool, manifest.open(
        "a", encoding="utf-8"
    ) as log:
        for done, record in enumerate(pool.imap_unordered(_label_one, jobs, chunksize=4), start=1):
            counts[record["status"]] += 1
            log.write(json.dumps(record) + "\n")
            if done % 100 == 0 or done == len(jobs):
                rate = done / max(time.time() - started, 1e-9)
                print(f"[{done}/{len(jobs)}] {counts} {rate:.1f} episodes/s", flush=True)
    print(f"[OK] Done: {counts}; manifest {manifest}")
    return 0 if counts["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
