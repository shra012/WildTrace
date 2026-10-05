#!/usr/bin/env python3
"""Compile a sparse VLM stroke plan into an Isaac-compatible trajectory JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wildtrace.trajectory_candidates import StrokePlanError, compile_stroke_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, help="Sparse model plan JSON")
    parser.add_argument("--output", required=True, help="Canonical trajectory JSON")
    parser.add_argument("--drawing-id", default=None)
    parser.add_argument("--max-segment-length", type=float, default=0.01)
    args = parser.parse_args()

    try:
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        trajectory = compile_stroke_plan(
            plan,
            drawing_id=args.drawing_id,
            max_segment_length=args.max_segment_length,
        )
    except (OSError, json.JSONDecodeError, StrokePlanError) as exc:
        parser.error(str(exc))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(trajectory, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] wrote {output}: {len(trajectory['strokes'])} strokes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
