"""Paths and switches for the studio backend, overridable through environment variables."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("WILDTRACE_REPO_ROOT", Path(__file__).resolve().parents[3]))
CONFIG_DIR = REPO_ROOT / "configs"
ISAAC_ROOT = REPO_ROOT / "isaac-sim"
STUDIO_DIR = Path(os.environ.get("WILDTRACE_STUDIO_DIR", REPO_ROOT / "outputs" / "studio"))
JOBS_DIR = STUDIO_DIR / "jobs"
RUNS_DIR = STUDIO_DIR / "runs"

# Robot side defaults: the lab-arm twin and the ACT policy trained for it.
DRAW_CONFIG = os.environ.get("WILDTRACE_DRAW_CONFIG", "config/xarm7_drawing_real.yaml")
DEFAULT_CONTROLLER = os.environ.get("WILDTRACE_CONTROLLER", "act")

# OpenRouter model used to understand chat messages (any chat model works).
CHAT_MODEL = os.environ.get("WILDTRACE_CHAT_MODEL", "openai/gpt-4o-mini")
OPENROUTER_BASE = os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")

CORS_ORIGINS = os.environ.get("WILDTRACE_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")

for directory in (JOBS_DIR, RUNS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def load_dotenv() -> None:
    """Read REPO_ROOT/.env into os.environ without overriding values already set."""
    path = REPO_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()
