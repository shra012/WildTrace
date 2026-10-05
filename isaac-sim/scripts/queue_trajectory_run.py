#!/usr/bin/env python3
"""Queue one canonical trajectory for the existing Isaac ScriptNode controller.

This does not run Isaac Sim itself.  It validates a trajectory and writes the
job file read by ``mcp_drawing_controller.py`` at the beginning of a run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ISAAC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ISAAC_ROOT / "src"))

from trajectory_loader import load_trajectory  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, help="Canonical JSON trajectory to execute")
    parser.add_argument("--prefix", required=True, help="Unique output prefix, e.g. molmo_cat_v0")
    parser.add_argument(
        "--job-file",
        default=str(ISAAC_ROOT / "outputs" / "mcp_sessions" / "job.json"),
        help="Controller job file; normally leave as the default.",
    )
    args = parser.parse_args()

    trajectory_path = Path(args.trajectory).resolve()
    trajectory = load_trajectory(trajectory_path)
    job_path = Path(args.job_file)
    job_path.parent.mkdir(parents=True, exist_ok=True)
    job_path.write_text(
        json.dumps({"trajectory_path": str(trajectory_path), "prefix": args.prefix}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[OK] queued {trajectory['drawing_id']} "
        f"({len(trajectory['strokes'])} strokes) as '{args.prefix}': {job_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
