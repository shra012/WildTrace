"""v1 bridge: "I want a <category>" -> check WildTrace's curated gold
trajectories -> queue one for Isaac Sim, or say plainly that it isn't
available.

Entirely read-only against WildTrace: never invokes a pipeline stage script,
never touches the validator/GPU pipeline, never writes into WildTrace's own
manifests. It only reads outputs/gold/trajectories/<Category>/*.json (the
same files a human has been hand-picking all session) and writes this
project's own outputs/mcp_sessions/job.json, which
scripts/mcp_cat_drawing_controller.py already knows how to read.

On-demand generation of genuinely new trajectories (bypassing WildTrace's
per-angle-bucket gold selection, or running the fetch/curate/diagram/validate
pipeline live) is a deliberately separate, deferred enhancement -- see the
project plan. This script only ever serves what's already in gold.

Usage:
    python provision_and_draw.py --category Dog
    python provision_and_draw.py --category Dog --allow-repeat
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ISAAC_SIM_ROOT = Path(__file__).resolve().parents[1]
WILDTRACE_ROOT = ISAAC_SIM_ROOT.parent
GOLD_TRAJECTORIES_DIR = WILDTRACE_ROOT / "outputs" / "gold" / "trajectories"


def _to_windows_path(path: Path) -> str:
    """job.json's trajectory_path is read by Windows-native Isaac Sim, not by
    this script itself -- when this script runs inside WSL (its natural
    home, next to the rest of WildTrace), plain WSL paths like
    /home/shravan/... resolve to nothing on the Windows side. Convert to the
    \\\\wsl.localhost\\<distro>\\... form Windows can actually open. A no-op
    when this script is ever run from Windows Python instead (no
    WSL_DISTRO_NAME set), where the path is already Windows-native."""
    distro = os.environ.get("WSL_DISTRO_NAME")
    if not distro:
        return str(path)
    return "\\\\wsl.localhost\\" + distro + str(path).replace("/", "\\")
MCP_SESSIONS_DIR = ISAAC_SIM_ROOT / "outputs" / "mcp_sessions"
JOB_FILE = MCP_SESSIONS_DIR / "job.json"


def _resolve_category(requested: str) -> str | None:
    """Case-insensitive match against the gold trajectory category folders."""
    if not GOLD_TRAJECTORIES_DIR.is_dir():
        return None
    wanted = requested.strip().lower()
    for entry in GOLD_TRAJECTORIES_DIR.iterdir():
        if entry.is_dir() and entry.name.lower() == wanted:
            return entry.name
    return None


def _already_drawn(sample_id: str) -> bool:
    """A sample counts as drawn if any prior run_metrics.json mentions it in
    its prefix -- the same "<category>_<sample_id-prefix>" convention every
    manual run this session already used. 6 chars matches the shortest
    prefix used by hand-picked runs earlier this session (e.g.
    "dog_96ab6d_run_metrics.json"); this script's own prefixes use 8 for a
    little more uniqueness going forward, but the check stays lenient so it
    still recognizes those older runs."""
    if not MCP_SESSIONS_DIR.is_dir():
        return False
    short_id = sample_id[:6]
    return any(short_id in path.name for path in MCP_SESSIONS_DIR.glob("*_run_metrics.json"))


def provision(category: str, allow_repeat: bool = False) -> dict:
    matched_category = _resolve_category(category)
    if matched_category is None:
        return {
            "status": "not_available",
            "requested_category": category,
            "reason": "no gold trajectories for this category",
        }

    category_dir = GOLD_TRAJECTORIES_DIR / matched_category
    candidates = sorted(p for p in category_dir.glob("*.json"))
    if not candidates:
        return {
            "status": "not_available",
            "requested_category": category,
            "reason": "category folder exists but has no trajectory files",
        }

    chosen = None
    if not allow_repeat:
        for candidate in candidates:
            if not _already_drawn(candidate.stem):
                chosen = candidate
                break
    if chosen is None:
        chosen = candidates[0]

    trajectory = json.loads(chosen.read_text(encoding="utf-8"))
    sample_id = chosen.stem
    prefix = f"{matched_category.lower()}_{sample_id[:8]}"
    was_drawn_before = _already_drawn(sample_id)
    windows_trajectory_path = _to_windows_path(chosen)

    MCP_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    JOB_FILE.write_text(
        json.dumps({"trajectory_path": windows_trajectory_path, "prefix": prefix}, indent=2),
        encoding="utf-8",
    )

    return {
        "status": "ok",
        "category": matched_category,
        "sample_id": sample_id,
        "trajectory_path": windows_trajectory_path,
        "prefix": prefix,
        "stroke_count": trajectory.get("stroke_count"),
        "point_count": trajectory.get("point_count"),
        "already_drawn_before": was_drawn_before,
        "job_file": str(JOB_FILE),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True, help='Category to draw, e.g. "Dog".')
    parser.add_argument(
        "--allow-repeat",
        action="store_true",
        help="Skip the not-yet-drawn preference and just take the first available sample.",
    )
    args = parser.parse_args()

    result = provision(args.category, allow_repeat=args.allow_repeat)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
