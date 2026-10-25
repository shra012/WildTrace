# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp>=1.2,<2"]
# ///
"""wildtrace-sim: one MCP server that takes "draw a <category>" from nothing
running to a finished, measured drawing -- launch Isaac Sim, build the xArm7
scene, queue a gold trajectory, run it, wait, plot, report.

Runs inside WSL next to the rest of WildTrace. Two boundary facts drive the
design:

- WSL2 here is in NAT networking mode, so WSL cannot reach Windows'
  localhost:8766 where the isaac.sim.mcp_extension socket listens. Every
  extension command is therefore relayed through a one-shot Windows Python
  process (WIN_PYTHON) that speaks the extension's JSON-over-TCP protocol
  ({"type", "params"} in, {"status", "result"|"message"} out). The extension
  serves each client on its own thread, so this coexists with the regular
  isaac-sim MCP server.
- Isaac Sim itself is a Windows GUI app, launched detached via PowerShell
  Start-Process using the isaacsim-mcp-server checkout's run_isaac_sim.ps1,
  pinned to ISAAC_SIM_ROOT (there are two installs on this machine and the
  launcher's auto-discovery picks the wrong one).

The draw flow is the same one .claude/skills/draw-animal/SKILL.md documents,
just automated end to end.

Register with Claude Code:
    claude mcp add wildtrace-sim -s user \\
      -e WILDTRACE_PYTHON=/home/shravan/anaconda3/bin/python3 \\
      -- uv run --script /home/shravan/Workspace/WildTrace/isaac-sim/mcp_server/wildtrace_sim_server.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from mcp.server.fastmcp import FastMCP

ISAAC_SIM_PROJECT = Path(__file__).resolve().parents[1]
WILDTRACE_ROOT = ISAAC_SIM_PROJECT.parent
sys.path.insert(0, str(ISAAC_SIM_PROJECT / "scripts"))
from provision_and_draw import GOLD_TRAJECTORIES_DIR, MCP_SESSIONS_DIR, _to_windows_path, provision  # noqa: E402
sys.path.insert(0, str(ISAAC_SIM_PROJECT / "src"))
import live_telemetry  # noqa: E402  (stdlib only, like this script's environment)

WIN_MCP_REPO = os.environ.get("ISAAC_MCP_REPO_WIN", r"C:\wildtrace\isaacsim-mcp-server")
ISAAC_SIM_ROOT = os.environ.get("ISAAC_SIM_ROOT_WIN", r"C:\isaac-sim-6.0.1")
WIN_PYTHON = os.environ.get("WIN_PYTHON", "/mnt/c/wildtrace/isaacsim-mcp-server/.venv/Scripts/python.exe")
EXT_PORT = int(os.environ.get("ISAAC_MCP_PORT", "8766"))
LAUNCH_LOG = r"C:\wildtrace\isaac_sim_launch.log"
LAUNCH_ERR_LOG = r"C:\wildtrace\isaac_sim_launch_err.log"
PLOT_PYTHON = os.environ.get("WILDTRACE_PYTHON", "python3")

ROBOT_PRIM = "/World/xarm7"
GRAPH_PATH = "/World/DrawingGraph"
CAPTURE_GRAPH_PATH = "/World/CaptureGraph"
CAMERA_PRIM = "/OmniverseKit_Persp"
VIDEO_CAMERAS = ["/World/VideoCams/WideFront", "/World/VideoCams/PaperTop"]
DEFAULT_CONFIG = "config/xarm7_drawing.yaml"
TWIN_CONFIG = "config/xarm7_drawing_real.yaml"
SCENE_FILE = MCP_SESSIONS_DIR / "scene.json"
CAPTURE_JOB_FILE = MCP_SESSIONS_DIR / "capture_job.json"
# ACT checkpoints per scene config (scripts/act_policy_server.py serves them).
ACT_CHECKPOINTS = {
    DEFAULT_CONFIG: ISAAC_SIM_PROJECT / "models" / "act_kinematic_v4",
    TWIN_CONFIG: ISAAC_SIM_PROJECT / "models" / "act_real_kinematic_v2",
}
ACT_PYTHON = ISAAC_SIM_PROJECT / ".venv-act" / "bin" / "python"
ACT_PORT = int(os.environ.get("WILDTRACE_ACT_PORT", "8790"))
VIDEO_ROOT_WIN = r"C:\wildtrace\videos"
VIDEO_ROOT_WSL = Path("/mnt/c/wildtrace/videos")
VIDEO_OUT = ISAAC_SIM_PROJECT / "outputs" / "videos"
POLL_S = 5  # seconds between checks for a run's report

# Runs under Windows Python: read one command from stdin, send it to the
# extension, print its JSON reply. Read-until-it-parses mirrors
# isaac_mcp.connection.receive_full_response.
_RELAY = r"""
import json, socket, sys
port = int(sys.argv[1])
payload = sys.stdin.read().encode("utf-8")
try:
    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
except OSError as exc:
    print(json.dumps({"status": "unreachable", "message": str(exc)}))
    sys.exit(0)
sock.settimeout(300)
sock.sendall(payload)
buf = b""
while True:
    chunk = sock.recv(65536)
    if not chunk:
        break
    buf += chunk
    try:
        json.loads(buf.decode("utf-8"))
        break
    except ValueError:
        continue
sock.close()
print(buf.decode("utf-8") or json.dumps({"status": "error", "message": "empty reply"}))
"""

mcp = FastMCP("wildtrace-sim")


def _isaac(command_type: str, params: dict | None = None, timeout_s: float = 330) -> dict:
    """Send one command to the Isaac Sim extension. Returns the reply dict:
    status "success"/"error" from the extension, or "unreachable" when
    nothing is listening (Isaac Sim not running or still booting)."""
    try:
        proc = subprocess.run(
            [WIN_PYTHON, "-c", _RELAY, str(EXT_PORT)],
            input=json.dumps({"type": command_type, "params": params or {}}),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": f"{command_type} got no reply within {timeout_s}s"}
    out = proc.stdout.strip()
    if not out:
        return {"status": "error", "message": f"relay produced no output: {proc.stderr.strip()[-500:]}"}
    return json.loads(out.splitlines()[-1])


def _ok(reply: dict) -> bool:
    return reply.get("status") == "success"


def _cmd(command: str, timeout_s: float = 60) -> str:
    """Run a Windows command through cmd.exe. PowerShell startup hangs on this
    machine when called from WSL (even `1+1`, stdin closed), so nothing here
    depends on it."""
    proc = subprocess.run(
        ["cmd.exe", "/c", command], capture_output=True, text=True, errors="replace",
        timeout=timeout_s, stdin=subprocess.DEVNULL, cwd="/mnt/c",
    )
    return proc.stdout.strip()


def _kit_running() -> bool:
    # tasklist.exe directly with an argv list: through cmd.exe the quoted
    # filter reaches tasklist split apart ("Invalid argument/option - 'eq'").
    proc = subprocess.run(
        ["tasklist.exe", "/FI", "IMAGENAME eq kit.exe", "/FO", "CSV", "/NH"],
        capture_output=True, text=True, errors="replace", timeout=60, stdin=subprocess.DEVNULL, cwd="/mnt/c",
    )
    return "kit.exe" in proc.stdout.lower()


def _scene_ready() -> bool:
    return _ok(_isaac("robots.get_info", {"prim_path": ROBOT_PRIM}))


def _timeline() -> str | None:
    reply = _isaac("simulation.get_state", timeout_s=30)
    return reply.get("result", {}).get("timeline_state") if _ok(reply) else None


def _set_timeline(command_type: str, wanted: str, attempts: int = 2) -> bool:
    """Send stop/play and confirm the timeline reached `wanted`, retrying once."""
    for _ in range(attempts):
        _isaac(command_type, timeout_s=60)
        for _ in range(10):
            if _timeline() == wanted:
                return True
            time.sleep(1)
    return False


def _unc(path: Path) -> str:
    return _to_windows_path(path)


@mcp.tool()
def sim_status() -> str:
    """Report whether Isaac Sim is running, whether its MCP extension socket
    answers, whether the xArm7 drawing scene is built, and the timeline state."""
    reply = _isaac("simulation.get_state", timeout_s=30)
    status = {
        "kit_process_running": _kit_running(),
        "extension_reachable": reply.get("status") != "unreachable",
        "timeline": reply.get("result") if _ok(reply) else reply.get("message"),
    }
    status["scene_ready"] = status["extension_reachable"] and _scene_ready()
    return json.dumps(status, indent=2)


@mcp.tool()
def sim_start(wait_s: int = 300) -> str:
    """Launch Isaac Sim on Windows with the MCP extension enabled (no-op if the
    extension already answers), then wait up to wait_s seconds for it to boot.
    Boot takes roughly 1-3 minutes. Launch output goes to
    C:\\wildtrace\\isaac_sim_launch.log."""
    if _ok(_isaac("simulation.get_state", timeout_s=30)):
        return json.dumps({"status": "already_running"})

    launched = False
    if not _kit_running():
        # A detached child with no inherited pipes, in its own session, the way
        # isaac_batch.py starts headless runs. Same arguments as the MCP repo's
        # scripts/run_isaac_sim.ps1, which needs PowerShell (it hangs here).
        launch = (
            f"{ISAAC_SIM_ROOT}\\isaac-sim.bat --ext-folder {WIN_MCP_REPO} --enable isaac.sim.mcp_extension "
            f"--/exts/isaac.sim.mcp/server.port={EXT_PORT} --/exts/isaac.sim.mcp/server.host=localhost"
        )
        subprocess.Popen(
            ["cmd.exe", "/c", f"{launch} > {LAUNCH_LOG} 2> {LAUNCH_ERR_LOG}"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            cwd="/mnt/c", start_new_session=True,
        )
        launched = True

    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        time.sleep(5)
        # The socket opens before the stage exists ("still starting up -- no
        # stage yet"); scene commands only work once get_state succeeds.
        if _ok(_isaac("simulation.get_state", timeout_s=30)):
            return json.dumps({"status": "started", "launched_new_process": launched}, indent=2)
    return json.dumps(
        {
            "status": "timeout",
            "launched_new_process": launched,
            "kit_process_running": _kit_running(),
            "hint": f"extension not answering after {wait_s}s; check {LAUNCH_LOG} and {LAUNCH_ERR_LOG}, "
            "or call sim_start again to keep waiting",
        },
        indent=2,
    )


@mcp.tool()
def sim_shutdown() -> str:
    """Close Isaac Sim (terminates the Windows kit.exe process). Any unsaved
    stage state is lost; the scene is rebuilt automatically on the next draw."""
    _cmd("taskkill /IM kit.exe /F")
    time.sleep(3)
    return json.dumps({"kit_process_running": _kit_running()})


def _scene_config() -> str:
    try:
        return json.loads(SCENE_FILE.read_text(encoding="utf-8")).get("config", DEFAULT_CONFIG)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return DEFAULT_CONFIG


def _scene_protrusion() -> float:
    sys.path.insert(0, str(ISAAC_SIM_PROJECT / "src"))
    from mcp_scene import pen_protrusion_m

    return pen_protrusion_m(ISAAC_SIM_PROJECT)


def _paper_surface(config: str) -> float:
    """Height of the sheet in the scene: calibrated pen-down TCP height minus the pen protrusion."""
    sys.path.insert(0, str(ISAAC_SIM_PROJECT / "src"))
    from mcp_scene import paper_surface_z

    drawing = _load_scene_config(config)["drawing"]
    protrusion = _scene_protrusion()
    return paper_surface_z(drawing, protrusion) if protrusion > 0 else float(drawing["paper_top_z_m"])


def _load_scene_config(config: str) -> dict:
    sys.path.insert(0, str(ISAAC_SIM_PROJECT / "src"))
    from project_config import load_config

    return load_config(config, ISAAC_SIM_PROJECT)


@mcp.tool()
def setup_scene(force: bool = False, config: str = "", gravity: float = -1.0, pen_protrusion_mm: float = -1.0) -> str:
    """Build the xArm7 drawing scene from a config: physics scene, the
    config-driven robot/paper/table from scripts/setup_scene.py, the two video
    cameras, the viewport camera, and the Action Graph running
    scripts/mcp_drawing_controller.py.

    config: "config/xarm7_drawing.yaml" (default sim) or
    "config/xarm7_drawing_real.yaml" (the lab-arm twin: calibrated URDF with
    gripper, measured paper placement). Empty keeps the current scene's config.
    gravity: m/s^2; negative means "from the config" (the default sim keeps
    its legacy zero-gravity MCP scene, the twin uses Earth gravity).
    pen_protrusion_mm: twin only, 20-60 mm: how far the pen tip sticks out past
    the calibrated gripper TCP. The calibration (TCP at the measured contact
    height) and all control stay as measured; this only draws the pen tip and
    puts the sheet and table that much lower. Negative keeps the current value
    (35 mm default until the lab arm is measured).
    Skipped when the scene is already built with this config, unless force=True.
    Scene state does not survive an Isaac Sim restart."""
    config = config or _scene_config()
    if config != _scene_config():
        force = True
    protrusion = 0.0
    if config == TWIN_CONFIG:
        current = _scene_protrusion() if config == _scene_config() else 0.0
        protrusion = pen_protrusion_mm / 1000.0 if pen_protrusion_mm >= 0 else (current or 0.035)
        if not 0.020 <= protrusion <= 0.060:
            return json.dumps({"status": "failed", "reason": f"pen_protrusion_mm must be 20-60, got {protrusion * 1e3:.1f}"})
        if abs(protrusion - current) > 1e-9:
            force = True
    if not force and _scene_ready():
        return json.dumps({"status": "already_built", "config": config})
    if gravity < 0:
        gravity = 0.0 if config == DEFAULT_CONFIG else float(_load_scene_config(config)["safety"]["gravity_m_s2"])

    MCP_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    scene = {"config": config, "gravity_m_s2": gravity}
    if protrusion > 0:
        scene["pen_protrusion_m"] = protrusion
    SCENE_FILE.write_text(json.dumps(scene, indent=2), encoding="utf-8")
    if _scene_ready():
        cleared = _isaac("scene.clear")
        if not _ok(cleared):
            return json.dumps({"status": "failed", "step": "scene.clear", "reply": cleared}, indent=2)

    scripts_unc = _unc(ISAAC_SIM_PROJECT / "scripts")
    steps = [
        ("scene.create_physics", {"scene_name": "PhysicsScene", "gravity": [0, 0, -gravity]}),
        ("simulation.reload_script", {"file_path": scripts_unc + "\\setup_scene.py"}),
        ("simulation.reload_script", {"file_path": scripts_unc + "\\setup_video_cameras.py"}),
        ("sensors.create_camera", {"prim_path": CAMERA_PRIM, "resolution": [1024, 1024]}),
        (
            "graphs.create_action_graph",
            {"graph_path": GRAPH_PATH, "evaluator": "execution", "script_file": scripts_unc + "\\mcp_drawing_controller.py"},
        ),
    ]
    for command_type, params in steps:
        reply = _isaac(command_type, params)
        if not _ok(reply):
            return json.dumps({"status": "failed", "step": command_type, "reply": reply}, indent=2)

    if not _scene_ready():
        return json.dumps({"status": "failed", "step": "verify", "reply": _isaac("robots.get_info", {"prim_path": ROBOT_PRIM})})
    _aim_viewport(config)
    return json.dumps({"status": "built", "config": config, "gravity_m_s2": gravity, "pen_protrusion_mm": protrusion * 1e3})


def _aim_viewport(config: str) -> None:
    """Point the GUI viewport at the arm and paper; recreating the viewport
    camera leaves it at a default pose looking into the robot/table."""
    drawing = _load_scene_config(config)["drawing"]
    cx, cy = (float(v) for v in drawing["surface_center_xy_m"])
    z = _paper_surface(config)
    _isaac("simulation.execute_script", {"code": (
        "from isaacsim.core.utils.viewports import set_camera_view\n"
        f"set_camera_view(eye=[{cx + 0.85}, {cy - 0.65}, {z + 0.6}], target=[{cx - 0.12}, {cy}, {z + 0.12}], "
        f"camera_prim_path='{CAMERA_PRIM}')\n"
    )}, timeout_s=60)


def _port_open(port: int) -> bool:
    import socket

    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def _ensure_policy_server(checkpoint: Path) -> dict:
    """Start scripts/act_policy_server.py for `checkpoint` unless it is already serving it."""
    marker = MCP_SESSIONS_DIR / "act_server.json"
    try:
        running = json.loads(marker.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        running = {}
    if _port_open(ACT_PORT) and running.get("checkpoint") == str(checkpoint):
        return {"status": "already_running", "checkpoint": str(checkpoint)}
    if running.get("pid"):
        try:
            os.kill(int(running["pid"]), 15)
            time.sleep(2)
        except OSError:
            pass
    log = open(MCP_SESSIONS_DIR / "act_server.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [str(ACT_PYTHON), "-u", str(ISAAC_SIM_PROJECT / "scripts" / "act_policy_server.py"),
         "--checkpoint", str(checkpoint), "--execution", "first", "--port", str(ACT_PORT)],
        stdout=log, stderr=subprocess.STDOUT, cwd=WILDTRACE_ROOT, start_new_session=True,
    )
    marker.write_text(json.dumps({"pid": proc.pid, "checkpoint": str(checkpoint)}), encoding="utf-8")
    for _ in range(120):
        if _port_open(ACT_PORT):
            return {"status": "started", "checkpoint": str(checkpoint), "pid": proc.pid}
        if proc.poll() is not None:
            break
        time.sleep(1)
    return {"status": "failed", "log": (MCP_SESSIONS_DIR / "act_server.log").read_text(encoding="utf-8")[-1500:]}


def _attach_capture(prefix: str, config: str, every: int) -> dict:
    """Write the capture job for this run and (re)build the capture Action Graph."""
    drawing = _load_scene_config(config)["drawing"]
    cx, cy = (float(v) for v in drawing["surface_center_xy_m"])
    sx, sy = (float(v) for v in drawing["surface_size_xy_m"])
    paper_z, pen_z = float(drawing["paper_top_z_m"]), float(drawing["pen_down_z_m"])
    out_dir_win = VIDEO_ROOT_WIN + "\\" + prefix
    shutil.rmtree(VIDEO_ROOT_WSL / prefix, ignore_errors=True)
    CAPTURE_JOB_FILE.write_text(json.dumps({
        "out_dir": out_dir_win,
        "every": every,
        "cameras": VIDEO_CAMERAS,
        "resolution": [1280, 720],
        "robot_prim": ROBOT_PRIM,
        "sheet_xy": [cx - sx / 2 - 0.005, cx + sx / 2 + 0.005, cy - sy / 2 - 0.005, cy + sy / 2 + 0.005],
        "pen_down_z": [paper_z - 0.005, pen_z + 0.0035],
        "ink_z": _paper_surface(config) + 0.0005,
    }, indent=2), encoding="utf-8")
    _detach_capture()
    return _isaac("graphs.create_action_graph", {
        "graph_path": CAPTURE_GRAPH_PATH, "evaluator": "execution",
        "script_file": _unc(ISAAC_SIM_PROJECT / "scripts") + "\\video_capture_node.py",
    })


def _detach_capture() -> None:
    _isaac("simulation.execute_script", {"code": (
        "import omni.usd\n"
        f"stage = omni.usd.get_context().get_stage()\n"
        f"if stage.GetPrimAtPath('{CAPTURE_GRAPH_PATH}'):\n"
        f"    stage.RemovePrim('{CAPTURE_GRAPH_PATH}')\n"
    )}, timeout_s=60)


def _assemble_videos(prefix: str, label: str, sim_seconds: float, every: int) -> list[str]:
    out = VIDEO_OUT / prefix
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["uv", "run", "--no-project", "--with", "imageio-ffmpeg", "--with", "pillow", "python",
         str(ISAAC_SIM_PROJECT / "scripts" / "make_drawing_videos.py"),
         "--frames", str(VIDEO_ROOT_WSL / prefix), "--label", label,
         "--sim-seconds", f"{sim_seconds:.1f}", "--every", str(every), "--out", str(out), "--target-seconds", "45"],
        capture_output=True, text=True, cwd=WILDTRACE_ROOT, timeout=900,
    )
    videos = []
    for video in sorted(out.glob("*.mp4")):
        # make_drawing_videos.py names files after the caption label; keep a
        # plain <prefix>_<camera>.mp4 instead.
        camera = video.stem.rsplit("_", 1)[-1]
        target = out / f"{prefix}_{camera}.mp4"
        if video != target:
            video.replace(target)
        videos.append(str(target))
    return videos or [f"video assembly failed: {(proc.stdout + proc.stderr)[-500:]}"]


def _metrics_summary(prefix: str, identity: dict | None = None) -> dict:
    """The headline fields of <prefix>_run_metrics.json (no plot)."""
    metrics_path = MCP_SESSIONS_DIR / f"{prefix}_run_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    err = metrics.get("desired_to_executed_nearest_path_error", {})
    return {
        "status": metrics.get("status"),
        "failure_reason": metrics.get("failure_reason") or None,
        **(identity or {}),
        "prefix": prefix,
        "stroke_count": metrics.get("stroke_count", metrics.get("requested_stroke_count")),
        "pen_down_points": metrics.get("desired_pen_down_points"),
        "rmse_mm": round(err.get("rmse_m", float("nan")) * 1000, 3),
        "mean_error_mm": round(err.get("mean_error_m", float("nan")) * 1000, 3),
        "max_error_mm": round(err.get("max_error_m", float("nan")) * 1000, 3),
        "simulation_duration_s": metrics.get("simulation_duration_s"),
        "git_sha": metrics.get("git_sha"),
        "config_hash": metrics.get("config_hash"),
        "metrics_file": str(metrics_path),
    }


def _plot(prefix: str, trajectory: Path, title: str) -> str:
    plot_path = MCP_SESSIONS_DIR / f"{prefix}_comparison.png"
    plot = subprocess.run(
        [
            PLOT_PYTHON,
            str(ISAAC_SIM_PROJECT / "scripts" / "plot_drawing_comparison.py"),
            "--trajectory", str(trajectory),
            "--executed-csv", str(MCP_SESSIONS_DIR / f"{prefix}_executed_path.csv"),
            "--output", str(plot_path),
            "--title", title,
        ],
        capture_output=True,
        text=True,
        cwd=ISAAC_SIM_PROJECT,
    )
    return str(plot_path) if plot.returncode == 0 else f"plot failed: {plot.stderr.strip()[-300:]}"


def _result(category: str, sample_id: str, prefix: str) -> dict:
    result = _metrics_summary(prefix, {"category": category, "sample_id": sample_id})
    result["comparison_plot"] = _plot(prefix, GOLD_TRAJECTORIES_DIR / category / f"{sample_id}.json", f"{category} ({sample_id})")
    return result


def _launch(job_data: dict, controller: str, config: str, record_video: bool, video_every: int, on_stage=None) -> dict:
    """Shared by draw and draw_trajectory: start Isaac Sim and build the scene
    if needed, start the ACT policy server, write job.json, attach or detach
    the capture graph and restart the timeline. Returns {"status": "started",
    "scene_config", "run_started"} or a {"status": "failed", "step", ...} dict."""
    stage = on_stage or (lambda name: None)
    stage("sim_start")
    started = json.loads(sim_start())
    if started["status"] == "timeout":
        return {"status": "failed", "step": "sim_start", "detail": started}
    stage("setup_scene")
    built = json.loads(setup_scene(config=config))
    if built["status"] == "failed":
        return {"status": "failed", "step": "setup_scene", "detail": built}
    scene_config = _scene_config()

    job_data["controller"] = controller
    if controller == "act":
        checkpoint = ACT_CHECKPOINTS.get(scene_config)
        if checkpoint is None:
            return {"status": "failed", "reason": f"no ACT checkpoint for config {scene_config}"}
        stage("act_policy_server")
        server = _ensure_policy_server(checkpoint)
        if server["status"] == "failed":
            return {"status": "failed", "step": "act_policy_server", "detail": server}
        job_data["policy_address"] = f"127.0.0.1:{ACT_PORT}"
    (MCP_SESSIONS_DIR / "job.json").write_text(json.dumps(job_data, indent=2), encoding="utf-8")
    if record_video:
        stage("attach_capture")
        attached = _attach_capture(job_data["prefix"], scene_config, video_every)
        if not _ok(attached):
            return {"status": "failed", "step": "attach_capture", "reply": attached}
    else:
        _detach_capture()

    # With allow_repeat an earlier run's report may already sit at this path;
    # only a file written after this run starts counts as its result.
    run_started = time.time()

    # The controller re-reads job.json and clears the previous drawing at INIT,
    # so a stop/play cycle is all it takes to start the queued job. Confirm each
    # transition from the timeline itself: a play once hung without effect
    # right after the Action Graph was first created.
    stage("timeline")
    if not _set_timeline("simulation.stop", "stopped"):
        return {"status": "failed", "step": "simulation.stop", "timeline": _timeline()}
    if not _set_timeline("simulation.play", "playing"):
        return {"status": "failed", "step": "simulation.play", "timeline": _timeline()}
    return {"status": "started", "scene_config": scene_config, "run_started": run_started}


def _await_run(
    prefix: str, run_started: float, wait_s: float, make_result, *,
    record_video: bool, video_every: int, video_label: str, stop_on_live_failure: bool = False,
) -> dict | None:
    """Wait up to wait_s for this run's metrics, then build its result (and
    videos). None when the wait ran out. With stop_on_live_failure, a failed
    live status that no report follows (INIT/HOME failures, a Stop) ends the
    wait early with that failure."""
    metrics_path = MCP_SESSIONS_DIR / f"{prefix}_run_metrics.json"
    live_file = live_telemetry.status_path(_live_dir(), prefix)
    failed_seen = None
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if metrics_path.exists() and metrics_path.stat().st_mtime >= run_started:
            time.sleep(1)  # let the writer finish
            result = make_result()
            if record_video:
                time.sleep(3)  # let the last frames flush
                _set_timeline("simulation.pause", "paused")
                _detach_capture()
                result["videos"] = _assemble_videos(
                    prefix, video_label, float(result.get("simulation_duration_s") or 0.0), video_every,
                )
            return result
        if stop_on_live_failure:
            live = live_telemetry.read_status(live_file)
            if live and live.get("status") == "failed" and float(live.get("updated_at") or 0) >= run_started:
                if failed_seen is not None and time.monotonic() - failed_seen >= 5:
                    if record_video:
                        _set_timeline("simulation.pause", "paused")
                        _detach_capture()
                    return {"status": "failed", "prefix": prefix, "step": live.get("phase"),
                            "failure_reason": live.get("failure_reason")}
                failed_seen = failed_seen or time.monotonic()
        time.sleep(POLL_S)
    return None


@mcp.tool()
def draw(
    category: str,
    allow_repeat: bool = False,
    wait_s: int = 900,
    controller: str = "ik",
    config: str = "",
    record_video: bool = False,
    video_every: int = 4,
) -> str:
    """Draw a gold WildTrace trajectory for `category` (e.g. "Frog") with the
    xArm7 in Isaac Sim, end to end: start Isaac Sim if needed, build the scene
    if needed, queue a not-yet-drawn gold sample, restart the timeline, wait up
    to wait_s seconds for the run to finish, then plot and report tracking
    error. Categories without gold trajectories are reported, never generated.
    If the wait runs out, returns status "running" -- call draw_result with the
    returned category/sample_id/prefix later.

    controller: "ik" (Lula IK) or "act" (the trained ACT policy for the scene
    config, served by scripts/act_policy_server.py, started automatically).
    config: scene config, e.g. "config/xarm7_drawing_real.yaml" for the lab-arm
    twin; empty keeps the current scene's config.
    record_video: record the run from two cameras and assemble MP4s (one per
    camera) under isaac-sim/outputs/videos/<prefix>/; video_every is the
    number of physics ticks between saved frames."""
    controller = controller.lower()
    if controller not in ("ik", "act"):
        return json.dumps({"status": "failed", "reason": f"unknown controller {controller!r}"})
    if _RUN_LOCK.locked():
        return json.dumps({"status": "busy", "reason": "a draw_trajectory run is in progress", "prefix": _active_prefix()})
    job = provision(category, allow_repeat=allow_repeat)
    if job["status"] != "ok":
        return json.dumps(job, indent=2)

    job_data = json.loads((MCP_SESSIONS_DIR / "job.json").read_text(encoding="utf-8"))
    launched = _launch(job_data, controller, config, record_video, video_every)
    if launched["status"] != "started":
        return json.dumps(launched, indent=2)
    scene_config = launched["scene_config"]

    def make_result():
        result = _result(job["category"], job["sample_id"], job["prefix"])
        result.update({"controller": controller, "config": scene_config})
        return result

    result = _await_run(
        job["prefix"], launched["run_started"], wait_s, make_result, record_video=record_video,
        video_every=video_every, video_label=f"{job['category']} ({controller.upper()})",
    )
    if result is not None:
        return json.dumps(result, indent=2)
    return json.dumps(
        {"status": "running", "category": job["category"], "sample_id": job["sample_id"], "prefix": job["prefix"]},
        indent=2,
    )


@mcp.tool()
def draw_result(category: str, sample_id: str, prefix: str) -> str:
    """Fetch metrics and generate the comparison plot for a draw that was still
    running when `draw` returned. Returns status "running" if not finished yet."""
    if not (MCP_SESSIONS_DIR / f"{prefix}_run_metrics.json").exists():
        state = _isaac("simulation.get_state", timeout_s=30)
        return json.dumps({"status": "running", "timeline": state.get("result", state.get("message"))}, indent=2)
    return json.dumps(_result(category, sample_id, prefix), indent=2)


# ── WildTrace Studio: draw any trajectory file, poll live status. ───────────

# How long a background (wait_s=0) studio run waits for its report before
# giving up on assembling videos; the drawing itself keeps going in Isaac.
STUDIO_WAIT_S = int(os.environ.get("WILDTRACE_STUDIO_WAIT_S", "3600"))
STUDIO_VIDEO_EVERY = 4
_PREFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
# One drawing at a time: Isaac has one robot and one timeline.
_RUN_LOCK = threading.Lock()
_LAUNCHES: dict[str, dict] = {}
_LAUNCH_THREADS: dict[str, threading.Thread] = {}


def _live_dir() -> Path:
    return live_telemetry.live_dir(MCP_SESSIONS_DIR)


def _active_prefix() -> str | None:
    for prefix, launch in _LAUNCHES.items():
        if launch.get("stage") not in ("done", "failed", "timed_out"):
            return prefix
    return None


def _resolve_trajectory(trajectory_path: str) -> Path | None:
    """A WSL path (absolute, or relative to the repo or isaac-sim/) or a
    \\\\wsl.localhost\\<distro>\\... / \\\\wsl$\\<distro>\\... UNC path, as a WSL Path."""
    raw = trajectory_path.strip()
    lowered = raw.lower().replace("/", "\\")
    for unc in ("\\\\wsl.localhost\\", "\\\\wsl$\\"):
        if lowered.startswith(unc):
            parts = raw.replace("/", "\\")[len(unc):].split("\\")[1:]  # drop the distro name
            raw = "/" + "/".join(part for part in parts if part)
            break
    path = Path(raw).expanduser()
    candidates = [path] if path.is_absolute() else [WILDTRACE_ROOT / path, ISAAC_SIM_PROJECT / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _studio_file(prefix: str, kind: str) -> Path:
    return MCP_SESSIONS_DIR / f"{prefix}_studio_{kind}.json"


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _camera_frames(prefix: str) -> dict:
    """Per camera: the WSL and Windows frames directories, the frame count and
    the newest frame known to be completely written (the capture node saves
    JPEGs on a thread pool, so the very newest file may still be growing)."""
    cameras = {}
    for camera in VIDEO_CAMERAS:
        name = camera.rsplit("/", 1)[-1]
        frames_dir = VIDEO_ROOT_WSL / prefix / name
        try:
            frames = sorted(entry for entry in os.listdir(frames_dir) if entry.endswith(".jpg"))
        except OSError:
            frames = []
        latest = frames[-2] if len(frames) >= 2 else None
        cameras[name] = {
            "frames_dir": str(frames_dir),
            "frames_dir_win": f"{VIDEO_ROOT_WIN}\\{prefix}\\{name}",
            "frame_count": len(frames),
            "latest_frame": str(frames_dir / latest) if latest else None,
        }
    return cameras


def _fail_launch(prefix: str, launched: dict) -> None:
    """A run that never reached the controller: record it where both
    run_status and a UI that only reads the live status file will see it."""
    _LAUNCHES[prefix].update(stage="failed", result=launched)
    _studio_file(prefix, "result").write_text(json.dumps(launched, indent=2), encoding="utf-8")
    detail = next((launched[key] for key in ("reason", "detail", "reply", "timeline") if launched.get(key)), "")
    live_telemetry.write_status(live_telemetry.status_path(_live_dir(), prefix), {
        "prefix": prefix, "phase": "LAUNCH", "progress": 0.0, "tip_world": None, "tip_canvas": None,
        "error_mm": None, "sim_time": 0.0, "status": live_telemetry.STATUS_FAILED, "source": "isaac",
        "failure_reason": f"{launched.get('step') or 'launch'}: {detail}", "updated_at": round(time.time(), 3),
    })


def _studio_run(prefix: str, job_data: dict, trajectory: Path, controller: str, config: str,
                record_video: bool, title: str, wait_s: float) -> dict:
    """Launch and wait for one studio run. The caller holds _RUN_LOCK; this releases it."""
    try:
        launch = _LAUNCHES[prefix]
        launched = _launch(job_data, controller, config, record_video, STUDIO_VIDEO_EVERY,
                           on_stage=lambda name: launch.update(stage=name))
        if launched["status"] != "started":
            launched["prefix"] = prefix
            _fail_launch(prefix, launched)
            return launched
        scene_config = launched["scene_config"]
        launch.update(stage="running", run_started=launched["run_started"], config=scene_config)

        def make_result():
            result = _metrics_summary(prefix, {"title": title, "trajectory_path": str(trajectory)})
            result["comparison_plot"] = _plot(prefix, trajectory, title)
            result.update({"controller": controller, "config": scene_config})
            return result

        result = _await_run(
            prefix, launched["run_started"], wait_s, make_result, record_video=record_video,
            video_every=STUDIO_VIDEO_EVERY, video_label=f"{title} ({controller.upper()})", stop_on_live_failure=True,
        )
        if result is None:
            launch.update(stage="timed_out")
            return {"status": "running", "prefix": prefix, "title": title, "trajectory_path": str(trajectory)}
        launch.update(stage="done", result=result)
        _studio_file(prefix, "result").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    except Exception as exc:  # a background thread has nobody else to report to
        failed = {"status": "failed", "prefix": prefix, "step": _LAUNCHES.get(prefix, {}).get("stage"), "reason": repr(exc)}
        _fail_launch(prefix, failed)
        return failed
    finally:
        _RUN_LOCK.release()


@mcp.tool()
def draw_trajectory(
    trajectory_path: str,
    controller: str = "act",
    config: str = "",
    record_video: bool = True,
    wait_s: int = 0,
    title: str = "",
) -> str:
    """Draw any trajectory JSON (the pipeline schema: strokes of {x, y} points
    in normalized canvas units) with the xArm7 in Isaac Sim, the same way
    `draw` does, under a new prefix "studio_<8 hex>".

    trajectory_path: WSL path (absolute, or relative to the repo or isaac-sim/)
    or a \\\\wsl.localhost UNC path.
    controller: "act" (default) or "ik". config: scene config; empty keeps the
    current scene's. record_video: capture both cameras (live JPEG frames under
    C:\\wildtrace\\videos\\<prefix>\\<Camera>\\) and assemble MP4s at the end.
    wait_s: 0 returns at once with {"status": "started", "prefix", ...} while
    the launch and run continue in the background (poll run_status(prefix) or
    the live status file); > 0 waits like `draw` and returns its result fields,
    or {"status": "running"} if the wait runs out. Only one run at a time:
    a second call while one is active returns {"status": "busy"}."""
    controller = controller.lower()
    if controller not in ("ik", "act"):
        return json.dumps({"status": "failed", "reason": f"unknown controller {controller!r}"})
    trajectory = _resolve_trajectory(trajectory_path)
    if trajectory is None:
        return json.dumps({"status": "failed", "reason": f"trajectory file not found: {trajectory_path}"})
    sys.path.insert(0, str(ISAAC_SIM_PROJECT / "src"))
    from trajectory_loader import load_trajectory

    try:
        loaded = load_trajectory(trajectory)
    except Exception as exc:
        return json.dumps({"status": "failed", "reason": f"invalid trajectory: {exc}"})
    if not _RUN_LOCK.acquire(blocking=False):
        return json.dumps({"status": "busy", "reason": "another run is in progress", "prefix": _active_prefix()})

    prefix = f"studio_{uuid.uuid4().hex[:8]}"
    title = title.strip() or trajectory.stem
    job_data = {"trajectory_path": _unc(trajectory), "prefix": prefix, "title": title, "source": "studio"}
    info = {
        "prefix": prefix,
        "title": title,
        "trajectory_path": str(trajectory),
        "trajectory_path_win": job_data["trajectory_path"],
        "controller": controller,
        "config": config or _scene_config(),
        "record_video": record_video,
        "stroke_count": len(loaded["strokes"]),
        "point_count": sum(len(stroke["points"]) for stroke in loaded["strokes"]),
        "telemetry_file": str(live_telemetry.status_path(_live_dir(), prefix)),
        "trace_file": str(live_telemetry.trace_path(_live_dir(), prefix)),
        "frames_dir": str(VIDEO_ROOT_WSL / prefix) if record_video else None,
        "cameras": [camera.rsplit("/", 1)[-1] for camera in VIDEO_CAMERAS] if record_video else [],
        "created_at": round(time.time(), 3),
    }
    try:
        MCP_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        _studio_file(prefix, "job").write_text(json.dumps(info, indent=2), encoding="utf-8")
    except OSError as exc:
        _RUN_LOCK.release()
        return json.dumps({"status": "failed", "reason": f"cannot write studio job: {exc}"})
    _LAUNCHES[prefix] = {**info, "stage": "queued"}
    args = (prefix, job_data, trajectory, controller, config, record_video, title)
    if wait_s <= 0:
        thread = threading.Thread(target=_studio_run, args=(*args, STUDIO_WAIT_S), name=f"studio-{prefix}", daemon=True)
        _LAUNCH_THREADS[prefix] = thread
        thread.start()
        return json.dumps({"status": "started", **info}, indent=2)
    return json.dumps({**info, **_studio_run(*args, wait_s)}, indent=2)


@mcp.tool()
def run_status(prefix: str) -> str:
    """Live state of a run by prefix (from draw_trajectory or draw): the latest
    live telemetry, the launch stage while Isaac/scene/policy are still being
    prepared, run metrics once the report exists, the result (plot, videos)
    once finalized, and each camera's frames directory with its newest frame.
    "status" is "launching", "drawing", "finished", "failed" or "unknown"."""
    if not _PREFIX_RE.match(prefix or ""):
        return json.dumps({"status": "failed", "reason": f"invalid prefix {prefix!r}"})
    live = live_telemetry.read_status(live_telemetry.status_path(_live_dir(), prefix))
    launch = {key: value for key, value in _LAUNCHES.get(prefix, {}).items() if key != "result"}
    job = launch or _read_json(_studio_file(prefix, "job")) or {}
    result = _LAUNCHES.get(prefix, {}).get("result") or _read_json(_studio_file(prefix, "result"))
    metrics_path = MCP_SESSIONS_DIR / f"{prefix}_run_metrics.json"
    metrics = None
    run_started = launch.get("run_started")
    # 1 s of slack: file mtimes are coarser than time.time().
    if metrics_path.exists() and (run_started is None or metrics_path.stat().st_mtime >= run_started - 1):
        try:
            metrics = _metrics_summary(prefix)
        except (OSError, ValueError):
            metrics = None  # mid-write; the next poll gets it

    stage = launch.get("stage")
    if stage == "failed":
        status = "failed"
    elif live is not None and (run_started is None or float(live.get("updated_at") or 0) >= run_started):
        status = live.get("status", "unknown")
    elif stage is not None and stage not in ("done", "timed_out"):
        status = "launching"
    elif metrics is not None:
        status = metrics["status"]
    else:
        status = "unknown"
    videos = sorted(str(path) for path in (VIDEO_OUT / prefix).glob("*.mp4")) if (VIDEO_OUT / prefix).is_dir() else []
    return json.dumps({
        "status": status,
        "prefix": prefix,
        "launch_stage": stage,
        # Results (plot, videos) are still being produced after the drawing ends.
        "finalizing": status in ("finished", "failed") and stage == "running",
        "telemetry": live,
        "metrics": metrics,
        "result": result,
        "job": job or None,
        "cameras": _camera_frames(prefix),
        "videos": videos,
        "trace_file": str(live_telemetry.trace_path(_live_dir(), prefix)),
    }, indent=2)


if __name__ == "__main__":
    mcp.run()
