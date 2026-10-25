"""Replay / preview a drawing run without Isaac Sim.

When Isaac Sim cannot run (no GPU to spare), WildTrace Studio still needs a
run to watch. ReplayRun animates the planned pen path of a trajectory and
writes the same live status and trace files as the Action Graph controller
(src/live_telemetry.py), so the UI cannot tell the difference except for
"source": "replay" in the status.

The path is the controller's own: the trajectory is mapped onto the drawing
plane (coordinate_mapper.map_trajectory_to_plane) and sequenced into
approach/lower/draw/lift phases (drawing_state_machine.build_motion_sequence)
with the scene config's settings, then timed at a constant tip speed per
phase type (kinematic_labeller.DEFAULT_SPEEDS_M_S). Sim time runs at
`speedup` times wall time; the robot's homing leg is skipped.

With the frames directory of a recorded run (C:\\wildtrace\\videos\\<prefix>,
one sub-directory of NNNNN.jpg per camera, or a single camera directory) the
status also names the frame that matches the current progress. With that
run's <prefix>_executed_path.csv the match uses the recorded progress over
sim time, and error_mm is the recorded tracking error at that progress;
without it frames are spread evenly over the planned duration and error_mm
is 0.

Usage (manual test; writes outputs/mcp_sessions/live/<prefix>.json):
    python src/run_replay.py data/act_test/Cat/1163bceaf6134cf5.json --speedup 8 \\
      --frames-dir /mnt/c/wildtrace/videos/cat_1163bcea \\
      --executed-csv outputs/mcp_sessions/cat_1163bcea_executed_path.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

SRC = Path(__file__).resolve().parent
PROJECT_ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_telemetry  # noqa: E402
from coordinate_mapper import map_trajectory_to_plane  # noqa: E402
from drawing_state_machine import build_motion_sequence  # noqa: E402
from kinematic_labeller import DEFAULT_SPEEDS_M_S, phase_speed  # noqa: E402
from mcp_scene import scene_config_path  # noqa: E402
from project_config import load_config  # noqa: E402
from trajectory_loader import load_trajectory  # noqa: E402

DEFAULT_LIVE_DIR = live_telemetry.live_dir(PROJECT_ROOT / "outputs" / "mcp_sessions")
# Capture-node frame timing (scripts/video_capture_node.py and the
# controller): one frame every `every` ticks from Play, while the
# controller's sim clock only starts after WARMUP (30 ticks) and INIT (1).
DEFAULT_FRAME_EVERY = 4
CONTROLLER_START_TICKS = 31


def plan_phases(trajectory: Mapping[str, Any], drawing: Mapping[str, Any]):
    """The mapped strokes and motion phases, built exactly as mcp_drawing_controller._build_machine does."""
    max_step = float(drawing["max_cartesian_step_m"])
    corner_angle = float(drawing["corner_angle_degrees"])
    mapped = map_trajectory_to_plane(
        trajectory,
        center_xy=list(drawing["surface_center_xy_m"]),
        size_xy=list(drawing["surface_size_xy_m"]),
        flip_image_y=bool(drawing["flip_image_y"]),
        max_step=max_step,
        smoothing_strength=float(drawing["smoothing_strength"]),
        corner_angle_degrees=corner_angle,
    )
    phases = build_motion_sequence(
        mapped["strokes"],
        pen_down_z=float(drawing["pen_down_z_m"]),
        pen_up_z=float(drawing["pen_up_z_m"]),
        approach_height=float(drawing["approach_height_m"]),
        max_cartesian_step=max_step,
        corner_angle_degrees=corner_angle,
        corner_densify_window=int(drawing["corner_window_points"]),
        corner_densify_factor=int(drawing["corner_densify_factor"]),
    )
    return mapped, phases


class PlannedPath:
    """Every motion target with the planned time the tip reaches it."""

    def __init__(self, phases, speeds_m_s: Optional[Mapping[str, float]] = None):
        speeds = dict(DEFAULT_SPEEDS_M_S if speeds_m_s is None else speeds_m_s)
        progress = live_telemetry.PenDownProgress(phases)
        self.targets = []
        fractions = []
        for phase_index, phase in enumerate(phases):
            for target_index, target in enumerate(phase.targets):
                self.targets.append(target)
                fractions.append(progress.at(phase_index, target_index))
        self.positions = np.asarray([t.position for t in self.targets], dtype=np.float64)
        lengths = np.linalg.norm(np.diff(self.positions, axis=0), axis=1)
        speed = np.asarray([phase_speed(t.state, speeds) for t in self.targets[1:]], dtype=np.float64)
        self.times = np.r_[0.0, np.cumsum(lengths / np.maximum(speed, 1e-9))]
        self.progress = np.asarray(fractions, dtype=np.float64)
        self.duration_s = float(self.times[-1])

    def index_at(self, sim_time: float) -> int:
        """Index of the last target reached at sim_time."""
        return int(np.clip(np.searchsorted(self.times, sim_time, side="right") - 1, 0, len(self.targets) - 1))

    def position_at(self, sim_time: float) -> np.ndarray:
        return np.asarray([np.interp(sim_time, self.times, self.positions[:, axis]) for axis in range(3)])

    def progress_at(self, sim_time: float) -> float:
        return float(np.interp(sim_time, self.times, self.progress))


class RecordedRun:
    """Pen-down progress over sim time from a past run's executed_path.csv."""

    def __init__(self, csv_path: str | Path):
        times, errors, drawn = [], [], []
        total = 0.0
        previous = None
        with open(csv_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                xy = (float(row["tip_x_m"]), float(row["tip_y_m"]))
                key = (row.get("stroke_id"), row.get("state"))
                if int(float(row["pen_down"])):
                    if previous is not None and previous[0] == key:
                        total += float(np.hypot(xy[0] - previous[1][0], xy[1] - previous[1][1]))
                    previous = (key, xy)
                else:
                    previous = None
                times.append(float(row["simulation_time_s"]))
                errors.append(float(row["tracking_error_m"]))
                drawn.append(total)
        if not times:
            raise ValueError(f"No rows in {csv_path}")
        self.times = np.asarray(times)
        self.errors = np.asarray(errors)
        self.progress = np.asarray(drawn) / total if total > 0 else np.zeros(len(drawn))
        self.duration_s = float(self.times[-1])

    def row_at_time(self, sim_time: float) -> int:
        return int(min(np.searchsorted(self.times, sim_time, side="left"), len(self.times) - 1))


def time_warp(path: PlannedPath, recorded: RecordedRun, levels: int = 201):
    """(replay times, recorded times) anchors that match equal pen-down
    progress, so np.interp maps a replay instant to the recorded one, travel
    and pen-up legs included."""
    grid = np.linspace(0.0, 1.0, levels)
    replay = path.times[np.minimum(np.searchsorted(path.progress, grid, side="left"), len(path.times) - 1)]
    actual = recorded.times[np.minimum(np.searchsorted(recorded.progress, grid, side="left"), len(recorded.times) - 1)]
    xp = np.maximum.accumulate(np.r_[0.0, replay, path.duration_s])
    fp = np.maximum.accumulate(np.r_[0.0, actual, recorded.duration_s])
    return xp, fp


def load_frames(frames_dir: str | Path) -> Dict[str, List[Path]]:
    """{camera: sorted frame paths} for a run's frames directory or one camera's directory."""
    root = Path(frames_dir)
    own = sorted(root.glob("*.jpg"))
    if own:
        return {root.name: own}
    return {
        sub.name: sorted(sub.glob("*.jpg"))
        for sub in sorted(p for p in root.iterdir() if p.is_dir())
        if any(sub.glob("*.jpg"))
    }


class ReplayRun:
    """Animate one trajectory into live telemetry files from a background thread.

    run = ReplayRun("data/act_test/Cat/1163bceaf6134cf5.json", speedup=8).start()
    ... poll live_telemetry.read_status(run.status_file) or run.status() ...
    run.stop()  # early stop marks the run "failed" ("replay stopped")

    step(sim_time) advances the replay deterministically (the thread calls it
    with the wall clock times speedup), so tests need no sleeping."""

    def __init__(
        self,
        trajectory_path: str | Path,
        *,
        prefix: Optional[str] = None,
        live_dir: str | Path = DEFAULT_LIVE_DIR,
        config: Optional[str | Path] = None,
        speedup: float = 1.0,
        frames_dir: Optional[str | Path] = None,
        executed_csv: Optional[str | Path] = None,
        tick_s: float = 0.1,
        speeds_m_s: Optional[Mapping[str, float]] = None,
        frame_every: int = DEFAULT_FRAME_EVERY,
        controller: str = "replay",
        clock=time.monotonic,
    ):
        if speedup <= 0:
            raise ValueError("speedup must be positive")
        self.prefix = prefix or f"replay_{uuid.uuid4().hex[:8]}"
        self.speedup = float(speedup)
        self.tick_s = float(tick_s)
        self.clock = clock
        config_path = Path(config) if config else scene_config_path(PROJECT_ROOT)
        cfg = load_config(config_path, PROJECT_ROOT)
        self.physics_dt_s = float(cfg["safety"]["physics_dt_s"])
        self.frame_every = int(frame_every)
        trajectory = load_trajectory(trajectory_path)
        self.mapped, phases = plan_phases(trajectory, cfg["drawing"])
        self.path = PlannedPath(phases, speeds_m_s)
        self.recorded = RecordedRun(executed_csv) if executed_csv else None
        self._warp = time_warp(self.path, self.recorded) if self.recorded else None
        self.frames = load_frames(frames_dir) if frames_dir else {}
        self.telemetry = live_telemetry.LiveTelemetry(
            live_dir,
            self.prefix,
            self.mapped["mapping"],
            phase=self.path.targets[0].state,
            period_s=None,
            extra={
                "source": "replay",
                "mode": "recorded" if self.recorded else "planned",
                "controller": controller,
                "speedup": self.speedup,
                "duration_s": round(self.path.duration_s, 3),
                "drawing_id": self.mapped["drawing_id"],
                "config_path": str(config_path),
            },
        )
        self.status_file = self.telemetry.status_file
        self.trace_file = self.telemetry.trace_file
        self._index = -1
        self._sim_time = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def finished(self) -> bool:
        return self.telemetry.finished

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> Optional[Dict[str, Any]]:
        return live_telemetry.read_status(self.status_file)

    def recorded_time(self, sim_time: float) -> Optional[float]:
        """The recorded run's sim time at the same progress as replay time sim_time."""
        return float(np.interp(sim_time, *self._warp)) if self._warp is not None else None

    def current_frames(self, sim_time: Optional[float] = None) -> Dict[str, Optional[str]]:
        """{camera: frame path} matching replay time sim_time (default: now)."""
        if not self.frames:
            return {}
        sim_time = self._sim_time if sim_time is None else float(sim_time)
        recorded_time = self.recorded_time(sim_time)
        frames = {}
        for camera, paths in self.frames.items():
            if recorded_time is not None:
                index = int(round((CONTROLLER_START_TICKS + recorded_time / self.physics_dt_s) / self.frame_every))
            else:
                fraction = sim_time / self.path.duration_s if self.path.duration_s > 0 else 1.0
                index = int(round(fraction * (len(paths) - 1)))
            frames[camera] = str(paths[min(max(index, 0), len(paths) - 1)]) if paths else None
        return frames

    def step(self, sim_time: float) -> Dict[str, Any]:
        """Advance to sim_time: trace every target passed since the last step,
        write the status once, and finish at the end of the path."""
        if self.finished:
            return self.telemetry.status
        path = self.path
        sim_time = min(max(float(sim_time), self._sim_time), path.duration_s)
        reached = path.index_at(sim_time)
        for index in range(self._index + 1, reached + 1):
            self._sample(float(path.times[index]), index, path.positions[index], float(path.progress[index]))
        self._index = max(self._index, reached)
        self._sim_time = sim_time
        heading = min(reached + 1, len(path.targets) - 1)
        self._sample(sim_time, heading, path.position_at(sim_time), path.progress_at(sim_time))
        if self.frames:
            self.telemetry.update(frame=self.current_frames(sim_time))
        if sim_time >= path.duration_s:
            self.telemetry.finish(sim_time, live_telemetry.STATUS_FINISHED)
        else:
            self.telemetry.flush()
        return self.telemetry.status

    def _sample(self, sim_time: float, index: int, position, progress: float) -> None:
        target = self.path.targets[index]
        error_m = 0.0
        if self.recorded is not None:
            error_m = float(self.recorded.errors[self.recorded.row_at_time(self.recorded_time(sim_time))])
        self.telemetry.sample(
            sim_time, target.state, position, error_m=error_m, progress=progress,
            pen_down=bool(target.pen_down), stroke_id=int(target.stroke_id),
        )

    def run(self) -> None:
        """Blocking replay at `speedup` x real time until finished or stopped."""
        started = self.clock()
        while not self._stop.is_set() and not self.finished:
            self.step((self.clock() - started) * self.speedup)
            self._stop.wait(self.tick_s)
        if not self.finished:
            self.telemetry.finish(self._sim_time, live_telemetry.STATUS_FAILED, "replay stopped")

    def start(self) -> "ReplayRun":
        if self._thread is None:
            self.telemetry.flush()  # the status exists as soon as start() returns
            self._thread = threading.Thread(target=self.run, name=f"replay-{self.prefix}", daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout: Optional[float] = 5.0) -> None:
        self._stop.set()
        self.join(timeout)

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trajectory", help="Trajectory JSON (pipeline schema).")
    parser.add_argument("--prefix", default=None, help="Run prefix (default replay_<8 hex>).")
    parser.add_argument("--speedup", type=float, default=4.0, help="Sim seconds per wall second.")
    parser.add_argument("--frames-dir", default=None, help="Recorded frames directory of a past run.")
    parser.add_argument("--executed-csv", default=None, help="That run's <prefix>_executed_path.csv.")
    parser.add_argument("--config", default=None, help="Scene config (default: the current MCP scene's).")
    parser.add_argument("--live-dir", default=str(DEFAULT_LIVE_DIR), help="Where to write the live files.")
    parser.add_argument("--tick-s", type=float, default=0.1, help="Wall seconds between status writes.")
    args = parser.parse_args()

    run = ReplayRun(
        args.trajectory, prefix=args.prefix, live_dir=args.live_dir, config=args.config, speedup=args.speedup,
        frames_dir=args.frames_dir, executed_csv=args.executed_csv, tick_s=args.tick_s,
    )
    print(f"replay {run.prefix}: {run.path.duration_s:.1f} sim s at {run.speedup:g}x -> {run.status_file}")
    run.start()
    try:
        while run.running:
            status = run.status() or {}
            print(json.dumps({key: status.get(key) for key in ("phase", "progress", "tip_canvas", "sim_time", "status", "frame")}))
            time.sleep(1.0)
    except KeyboardInterrupt:
        run.stop()
    final = run.status() or {}
    print(json.dumps({key: final.get(key) for key in ("status", "progress", "sim_time", "trace_samples")}))
    return 0 if final.get("status") == live_telemetry.STATUS_FINISHED else 1


if __name__ == "__main__":
    sys.exit(main())
