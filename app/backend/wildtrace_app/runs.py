"""Drawing runs: live in Isaac Sim through the wildtrace-sim MCP server, or a replay.

Both write the same live status and trace files (isaac-sim/src/live_telemetry.py);
a monitor thread per run turns them into WebSocket telemetry and Run updates.

live   - wildtrace_sim_server.draw_trajectory(wait_s=0), then run_status(prefix).
replay - isaac-sim/src/run_replay.ReplayRun over the job's planned pen path. For a
         gold drawing with a recorded lab-arm twin run of the same controller, the
         replay follows that run's progress, tracking error, metrics and camera frames.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import health, settings
from .store import bus, file_path, store

if str(settings.ISAAC_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(settings.ISAAC_ROOT / "src"))

import live_telemetry  # noqa: E402  (stdlib only)

SESSIONS_DIR = settings.ISAAC_ROOT / "outputs" / "mcp_sessions"
LIVE_DIR = live_telemetry.live_dir(SESSIONS_DIR)
VIDEO_ROOT = Path(os.environ.get("WILDTRACE_VIDEO_ROOT", "/mnt/c/wildtrace/videos"))
# Replays are compressed to about this many wall seconds (never slower than real time).
REPLAY_SECONDS = float(os.environ.get("WILDTRACE_REPLAY_SECONDS", "45"))
POLL_S = 0.1

_active: dict[str, dict[str, Any]] = {}   # run_id -> {"prefix", "replay"?, "trace_file", "status_file"}
_lock = threading.Lock()


# Recorded runs -------------------------------------------------------------------------------

def _summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    error = metrics.get("desired_to_executed_nearest_path_error") or {}
    jerk = metrics.get("pen_down_joint_jerk_rad_s3") or {}
    out = {
        "rmse_mm": round(error["rmse_m"] * 1000, 3) if "rmse_m" in error else None,
        "max_error_mm": round(error["max_error_m"] * 1000, 3) if "max_error_m" in error else None,
        "sim_seconds": round(metrics["simulation_duration_s"], 1) if metrics.get("simulation_duration_s") else None,
        "jerk_rms": round(jerk["rms"], 1) if "rms" in jerk else None,
    }
    return {k: v for k, v in out.items() if v is not None}


@lru_cache(maxsize=1)
def _recordings() -> dict[tuple[str, str], dict[str, Any]]:
    """(drawing_id, controller) -> a finished past run in the studio's scene config, best first:
    MCP runs with camera frames, then MCP runs, then the headless twin evaluation runs."""
    config_name = Path(settings.DRAW_CONFIG).name
    found: dict[tuple[str, str], dict[str, Any]] = {}

    def offer(metrics_path: Path, csv_path: Path, frames_dir: Path | None, rank: int) -> None:
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        config = str(metrics.get("config_path", "")).replace("\\", "/").rsplit("/", 1)[-1]
        if metrics.get("status") != "finished" or not csv_path.exists() or (config and config != config_name):
            return
        key = (str(metrics.get("drawing_id")), str(metrics.get("controller")))
        if key not in found or rank < found[key]["rank"]:
            found[key] = {"metrics": _summarize_metrics(metrics), "csv": csv_path, "frames_dir": frames_dir, "rank": rank}

    for metrics_path in SESSIONS_DIR.glob("*_run_metrics.json"):
        prefix = metrics_path.name.removesuffix("_run_metrics.json")
        frames = VIDEO_ROOT / prefix
        has_frames = frames.is_dir() and any(frames.iterdir())
        offer(metrics_path, SESSIONS_DIR / f"{prefix}_executed_path.csv", frames if has_frames else None, 0 if has_frames else 1)
    from .catalog import RECORDINGS
    for name in RECORDINGS.values():
        for metrics_path in (settings.REPO_ROOT / "outputs" / "act_eval" / name).glob("*/*/run_metrics.json"):
            offer(metrics_path, metrics_path.parent / "executed_path.csv", None, 2)
    return found


def recording_for(job: dict[str, Any], controller: str) -> dict[str, Any] | None:
    sample_id = job["request"].get("sample_id")
    return _recordings().get((sample_id, controller)) if sample_id else None


# Starting runs -------------------------------------------------------------------------------

def _trajectory_path(job: dict[str, Any]) -> Path:
    return file_path(job["artifacts"]["trajectory_url"])


def start_run(job: dict[str, Any], controller: str, mode: str | None) -> dict[str, Any]:
    if mode is None:
        mode = "live" if health.isaac().get("running") else "replay"
    run = store.create_run(job["id"], controller, mode)
    try:
        if mode == "live":
            _start_live(run, job, controller)
        else:
            _start_replay(run, job, controller)
    except Exception as exc:  # reported on the run; nothing else to clean up yet
        bus.log(f"Could not start the {mode} run: {exc}", "error")
        return store.update_run(run["id"], status="failed", error=str(exc))
    threading.Thread(target=_monitor, args=(run["id"],), name=f"monitor-{run['id']}", daemon=True).start()
    return store.get_run(run["id"])


def _start_live(run: dict[str, Any], job: dict[str, Any], controller: str) -> None:
    server = health.sim_server()
    if server is None:
        raise RuntimeError("the wildtrace-sim MCP server cannot be imported (install the app extra)")
    reply = json.loads(server.draw_trajectory(
        trajectory_path=str(_trajectory_path(job)), controller=controller, config=settings.DRAW_CONFIG,
        record_video=True, wait_s=0, title=job["title"],
    ))
    if reply.get("status") != "started":
        raise RuntimeError(reply.get("reason") or reply.get("status"))
    with _lock:
        _active[run["id"]] = {"prefix": reply["prefix"], "status_file": Path(reply["telemetry_file"]),
                              "trace_file": Path(reply["trace_file"]), "server": server}
    store.update_run(run["id"], status="starting", phase="LAUNCH", cameras=reply.get("cameras", []), prefix=reply["prefix"])
    bus.log(f"Isaac Sim is drawing {job['title'].lower()} ({controller.upper()}), prefix {reply['prefix']}")


def _start_replay(run: dict[str, Any], job: dict[str, Any], controller: str) -> None:
    from run_replay import ReplayRun

    recorded = recording_for(job, controller)
    replay = ReplayRun(
        _trajectory_path(job),
        prefix=f"studio_replay_{run['id'].removeprefix('run_')}",
        live_dir=LIVE_DIR,
        config=settings.ISAAC_ROOT / settings.DRAW_CONFIG,
        frames_dir=recorded["frames_dir"] if recorded else None,
        executed_csv=recorded["csv"] if recorded else None,
        controller=controller,
    )
    replay.speedup = max(1.0, replay.path.duration_s / REPLAY_SECONDS)
    with _lock:
        _active[run["id"]] = {"prefix": replay.prefix, "status_file": replay.status_file,
                              "trace_file": replay.trace_file, "replay": replay, "recorded": recorded}
    store.update_run(run["id"], status="running", phase="REPLAY", cameras=sorted(replay.frames), prefix=replay.prefix,
                     source="recorded" if recorded else "planned")
    replay.start()
    what = ("a recorded twin run" if recorded else "the planned pen path")
    bus.log(f"Isaac Sim is not running: replaying {what} at {replay.speedup:.1f}x")


# Monitoring ----------------------------------------------------------------------------------

def _monitor(run_id: str) -> None:
    with _lock:
        active = _active.get(run_id)
    if active is None:
        return
    last_publish, last_phase, last_launch_poll = 0.0, None, 0.0
    while True:
        status = live_telemetry.read_status(active["status_file"])
        now = time.monotonic()
        server = active.get("server")
        if status is None and server is not None and now - last_launch_poll > 2:
            # Still launching Isaac / building the scene / starting the policy server.
            last_launch_poll = now
            reply = json.loads(server.run_status(active["prefix"]))
            if reply.get("status") == "failed":
                store.update_run(run_id, status="failed", error=reply.get("reason") or "launch failed")
                break
            stage = reply.get("launch_stage") or "launch"
            if stage != last_phase:
                last_phase = stage
                store.update_run(run_id, phase=f"LAUNCH: {stage}")
        if status is not None:
            tip = status.get("tip_canvas")
            bus.publish({"type": "telemetry", "run_id": run_id, "t": status.get("sim_time", 0.0),
                         "phase": status.get("phase"), "progress": status.get("progress", 0.0),
                         "tip": tip, "error_mm": status.get("error_mm"), "frame": bool(status.get("frame"))})
            state = status.get("status")
            if state in (live_telemetry.STATUS_FINISHED, live_telemetry.STATUS_FAILED):
                _finish(run_id, active, status)
                break
            if status.get("phase") != last_phase or now - last_publish > 1.0:
                last_phase, last_publish = status.get("phase"), now
                store.update_run(run_id, status="running", phase=status.get("phase"),
                                 progress=round(float(status.get("progress") or 0.0), 4))
        time.sleep(POLL_S)


def _finish(run_id: str, active: dict[str, Any], status: dict[str, Any]) -> None:
    metrics = None
    if active.get("server") is not None:
        # The report lands a moment after the controller's final status.
        for _ in range(30):
            reply = json.loads(active["server"].run_status(active["prefix"]))
            if reply.get("metrics"):
                metrics = {"rmse_mm": reply["metrics"].get("rmse_mm"), "max_error_mm": reply["metrics"].get("max_error_mm"),
                           "sim_seconds": reply["metrics"].get("simulation_duration_s")}
                break
            time.sleep(1)
    elif active.get("recorded"):
        metrics = active["recorded"]["metrics"]
    failed = status.get("status") == live_telemetry.STATUS_FAILED
    store.update_run(run_id, status="failed" if failed else "finished", phase=status.get("phase"),
                     progress=1.0 if not failed else float(status.get("progress") or 0.0),
                     metrics=metrics, error=status.get("failure_reason") if failed else None)
    job = store.get_job(store.get_run(run_id)["job_id"])
    if failed:
        bus.log(f"Drawing {job['title'].lower()} failed: {status.get('failure_reason')}", "error")
    else:
        rmse = f", RMSE {metrics['rmse_mm']:.3f} mm" if metrics and metrics.get("rmse_mm") is not None else ""
        bus.log(f"Finished drawing {job['title'].lower()}{rmse}")


def stop_all() -> None:
    with _lock:
        replays = [a["replay"] for a in _active.values() if a.get("replay") is not None]
    for replay in replays:
        replay.stop(timeout=1.0)


# Trace and frames ----------------------------------------------------------------------------

def trace(run: dict[str, Any]) -> dict[str, Any]:
    job = store.get_job(run["job_id"])
    data = json.loads(_trajectory_path(job).read_text(encoding="utf-8"))
    desired = [[[p["x"], p["y"]] for p in stroke["points"]] for stroke in data["strokes"]]
    prefix = run.get("prefix")
    executed: list[list[list[float]]] = []
    if prefix:
        samples, _ = live_telemetry.read_trace(live_telemetry.trace_path(LIVE_DIR, prefix))
        executed = live_telemetry.trace_polylines(samples)
    width, height = float(data.get("canvas_width") or 1), float(data.get("canvas_height") or 1)
    return {"desired": desired, "executed": executed, "aspect": round(width / height, 4)}


def latest_frame(run_id: str, camera: str) -> tuple[bytes | None, bool]:
    """(newest complete JPEG for the camera, whether the run is over)."""
    run = store.get_run(run_id)
    done = run is None or run["status"] in ("finished", "failed")
    with _lock:
        active = _active.get(run_id)
    path = None
    if active and active.get("server") is not None:
        cameras = active["server"]._camera_frames(active["prefix"])
        path = (cameras.get(camera) or {}).get("latest_frame")
    elif active and active.get("replay") is not None:
        status = live_telemetry.read_status(active["status_file"]) or {}
        path = (status.get("frame") or {}).get(camera)
    if not path:
        return None, done
    try:
        return Path(path).read_bytes(), done
    except OSError:
        return None, done
