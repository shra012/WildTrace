"""Run the deterministic baseline with compressed NPZ demonstration recording."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

script = Path(__file__).with_name("run_two_stroke_drawing.py")
if "--record-demonstration" not in sys.argv:
    sys.argv.append("--record-demonstration")
runpy.run_path(str(script), run_name="__main__")

