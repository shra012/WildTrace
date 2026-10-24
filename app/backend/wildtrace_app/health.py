"""GPU / Isaac / backend status for /api/health and the navbar pills (cached, cheap to call)."""
from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from . import chat, settings

_cache: dict[str, tuple[float, Any]] = {}
_lock = threading.Lock()


def _cached(key: str, ttl_s: float, compute):
    with _lock:
        hit = _cache.get(key)
        if hit and time.monotonic() - hit[0] < ttl_s:
            return hit[1]
    value = compute()
    with _lock:
        _cache[key] = (time.monotonic(), value)
    return value


def gpu() -> dict[str, Any]:
    def compute() -> dict[str, Any]:
        try:
            from wildtrace.studio_pipeline import gpu_status
            return gpu_status()
        except ImportError:
            pass
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout.strip()
            utilization = float(out.splitlines()[0])
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return {"busy": True, "utilization": 0.0, "note": "nvidia-smi unavailable"}
        return {"busy": utilization > 20, "utilization": utilization, "note": f"{utilization:.0f}% used"}
    return _cached("gpu", 10, compute)


def sim_server():
    """The wildtrace-sim MCP server module, imported for its helpers (None if `mcp` is missing)."""
    import sys
    scripts = str(settings.ISAAC_ROOT / "mcp_server")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    try:
        import wildtrace_sim_server
    except ImportError:
        return None
    return wildtrace_sim_server


def isaac() -> dict[str, Any]:
    def compute() -> dict[str, Any]:
        server = sim_server()
        if server is None:
            return {"running": False, "scene_ready": False, "note": "mcp package not installed"}
        try:
            running = server._kit_running()
            scene_ready = running and server._scene_ready()
            config = server._scene_config() if scene_ready else None
        except Exception as exc:  # the relay to Windows can fail in many ways; report, do not raise
            return {"running": False, "scene_ready": False, "note": str(exc)[:120]}
        return {"running": running, "scene_ready": scene_ready, "config": config}
    return _cached("isaac", 15, compute)


def image_backend() -> str:
    def compute() -> str:
        try:
            from wildtrace.studio_pipeline import select_image_backend
            choice = select_image_backend(settings.REPO_ROOT, settings.CONFIG_DIR)
        except ImportError:
            choice = "openrouter"
        if choice == "flux":
            return "flux-schnell (GPU)"
        from wildtrace.openrouter_image import DEFAULT_OPENROUTER_IMAGE_MODEL
        return f"openrouter: {os.environ.get('OPENROUTER_IMAGE_MODEL') or DEFAULT_OPENROUTER_IMAGE_MODEL}"
    return _cached("image", 10, compute)


def policy() -> dict[str, Any]:
    checkpoint = "act_real_kinematic_v2" if "real" in Path(settings.DRAW_CONFIG).name else "act_kinematic_v4"
    return {"checkpoint": checkpoint, "device": "cpu" if gpu().get("busy") else "cuda"}


def snapshot() -> dict[str, Any]:
    from wildtrace import face_lineart
    face_device = face_lineart.DEFAULT_SETTINGS["device"]
    return {
        "gpu": gpu(),
        "isaac": isaac(),
        "backends": {"image": image_backend(), "chat": chat.backend_name(), "face": f"mediapipe ({face_device.upper()})"},
        "policy": policy(),
    }
