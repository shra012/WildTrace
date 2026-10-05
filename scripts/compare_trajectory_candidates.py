#!/usr/bin/env python3
"""Compare a learned trajectory candidate with an existing trajectory."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ISAAC_SRC = ROOT / "isaac-sim" / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ISAAC_SRC))

from trajectory_loader import load_trajectory  # noqa: E402
from wildtrace.trajectory_candidates import symmetric_raster_distance  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="Current Flux-derived trajectory JSON")
    parser.add_argument("--candidate", required=True, help="Molmo candidate trajectory JSON")
    parser.add_argument("--output", required=True, help="Metrics JSON")
    parser.add_argument("--raster-size", type=int, default=512)
    args = parser.parse_args()

    reference = load_trajectory(args.reference)
    candidate = load_trajectory(args.candidate)
    metrics = symmetric_raster_distance(reference, candidate, size=args.raster_size)
    metrics.update({
        "reference": str(Path(args.reference)),
        "candidate": str(Path(args.candidate)),
        "reference_strokes": len(reference["strokes"]),
        "candidate_strokes": len(candidate["strokes"]),
    })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
