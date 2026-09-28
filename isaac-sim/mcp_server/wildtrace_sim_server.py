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
import subprocess
import sys
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

ISAAC_SIM_PROJECT = Path(__file__).resolve().parents[1]
WILDTRACE_ROOT = ISAAC_SIM_PROJECT.parent
sys.path.insert(0, str(ISAAC_SIM_PROJECT / "scripts"))
from provision_and_draw import GOLD_TRAJECTORIES_DIR, MCP_SESSIONS_DIR, _to_windows_path, provision  # noqa: E402

WIN_MCP_REPO = os.environ.get("ISAAC_MCP_REPO_WIN", r"C:\wildtrace\isaacsim-mcp-server")
ISAAC_SIM_ROOT = os.environ.get("ISAAC_SIM_ROOT_WIN", r"C:\isaac-sim-6.0.1")
WIN_PYTHON = os.environ.get("WIN_PYTHON", "/mnt/c/wildtrace/isaacsim-mcp-server/.venv/Scripts/python.exe")
EXT_PORT = int(os.environ.get("ISAAC_MCP_PORT", "8766"))
LAUNCH_LOG = r"C:\wildtrace\isaac_sim_launch.log"
LAUNCH_ERR_LOG = r"C:\wildtrace\isaac_sim_launch_err.log"
PLOT_PYTHON = os.environ.get("WILDTRACE_PYTHON", "python3")

ROBOT_PRIM = "/World/xarm7"
GRAPH_PATH = "/World/DrawingGraph"
CAMERA_PRIM = "/OmniverseKit_Persp"

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


def _powershell(command: str, timeout_s: float = 60) -> str:
    proc = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return proc.stdout.strip()


def _kit_running() -> bool:
    return bool(_powershell("Get-Process -Name kit -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id"))


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
        # Win32_Process.Create, not Start-Process: a Start-Process child still
        # shares WSL interop's console and got a Ctrl+C ~25s into boot when an
        # unrelated WSL shell exited. A WMI-created process has no such tie.
        launcher = WIN_MCP_REPO + r"\scripts\run_isaac_sim.ps1"
        command_line = (
            f"cmd.exe /c powershell.exe -NoProfile -ExecutionPolicy Bypass -File {launcher} "
            f"-IsaacSimRoot {ISAAC_SIM_ROOT} -Port {EXT_PORT} > {LAUNCH_LOG} 2> {LAUNCH_ERR_LOG}"
        )
        _powershell(
            "Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
            f"-Arguments @{{CommandLine='{command_line}'; CurrentDirectory='C:\\wildtrace'}} | Out-Null"
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
    _powershell("Stop-Process -Name kit -ErrorAction SilentlyContinue")
    time.sleep(3)
    return json.dumps({"kit_process_running": _kit_running()})


@mcp.tool()
def setup_scene(force: bool = False) -> str:
    """Build the xArm7 drawing scene: zero-gravity physics scene, the
    config-driven robot/paper/table from scripts/setup_scene.py, viewport
    camera, and the Action Graph running scripts/mcp_drawing_controller.py.
    Skipped when the robot prim already exists, unless force=True. Scene state
    does not survive an Isaac Sim restart."""
    if not force and _scene_ready():
        return json.dumps({"status": "already_built"})

    scripts_unc = _unc(ISAAC_SIM_PROJECT / "scripts")
    steps = [
        ("scene.create_physics", {"scene_name": "PhysicsScene", "gravity": [0, 0, 0]}),
        ("simulation.reload_script", {"file_path": scripts_unc + "\\setup_scene.py"}),
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
    return json.dumps({"status": "built"})


def _result(category: str, sample_id: str, prefix: str) -> dict:
    metrics_path = MCP_SESSIONS_DIR / f"{prefix}_run_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    plot_path = MCP_SESSIONS_DIR / f"{prefix}_comparison.png"
    plot = subprocess.run(
        [
            PLOT_PYTHON,
            str(ISAAC_SIM_PROJECT / "scripts" / "plot_drawing_comparison.py"),
            "--trajectory", str(GOLD_TRAJECTORIES_DIR / category / f"{sample_id}.json"),
            "--executed-csv", str(MCP_SESSIONS_DIR / f"{prefix}_executed_path.csv"),
            "--output", str(plot_path),
            "--title", f"{category} ({sample_id})",
        ],
        capture_output=True,
        text=True,
        cwd=ISAAC_SIM_PROJECT,
    )
    err = metrics.get("desired_to_executed_nearest_path_error", {})
    return {
        "status": metrics.get("status"),
        "failure_reason": metrics.get("failure_reason") or None,
        "category": category,
        "sample_id": sample_id,
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
        "comparison_plot": str(plot_path) if plot.returncode == 0 else f"plot failed: {plot.stderr.strip()[-300:]}",
    }


@mcp.tool()
def draw(category: str, allow_repeat: bool = False, wait_s: int = 900) -> str:
    """Draw a gold WildTrace trajectory for `category` (e.g. "Frog") with the
    xArm7 in Isaac Sim, end to end: start Isaac Sim if needed, build the scene
    if needed, queue a not-yet-drawn gold sample, restart the timeline, wait up
    to wait_s seconds for the run to finish, then plot and report tracking
    error. Categories without gold trajectories are reported, never generated.
    If the wait runs out, returns status "running" -- call draw_result with the
    returned category/sample_id/prefix later."""
    job = provision(category, allow_repeat=allow_repeat)
    if job["status"] != "ok":
        return json.dumps(job, indent=2)

    started = json.loads(sim_start())
    if started["status"] == "timeout":
        return json.dumps({"status": "failed", "step": "sim_start", "detail": started}, indent=2)
    built = json.loads(setup_scene())
    if built["status"] == "failed":
        return json.dumps({"status": "failed", "step": "setup_scene", "detail": built}, indent=2)

    # With allow_repeat an earlier run's report may already sit at this path;
    # only a file written after this run starts counts as its result.
    metrics_path = MCP_SESSIONS_DIR / f"{job['prefix']}_run_metrics.json"
    run_started = time.time()

    # The controller re-reads job.json and clears the previous drawing at INIT,
    # so a stop/play cycle is all it takes to start the queued job. Confirm each
    # transition from the timeline itself: a play once hung without effect
    # right after the Action Graph was first created.
    if not _set_timeline("simulation.stop", "stopped"):
        return json.dumps({"status": "failed", "step": "simulation.stop", "timeline": _timeline()}, indent=2)
    if not _set_timeline("simulation.play", "playing"):
        return json.dumps({"status": "failed", "step": "simulation.play", "timeline": _timeline()}, indent=2)

    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if metrics_path.exists() and metrics_path.stat().st_mtime >= run_started:
            time.sleep(1)  # let the writer finish
            return json.dumps(_result(job["category"], job["sample_id"], job["prefix"]), indent=2)
        time.sleep(5)
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


if __name__ == "__main__":
    mcp.run()
