#!/usr/bin/env python3
"""CPU-only OpenCV robustness check on committed line drawings.

Blurs, resizes, and noising each drawing, then rescores it with the production
OpenCV prescreen. No VLM and no diffusion model is loaded. Results are written
to docs/presentation/validator_robustness.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from wildtrace.config import load_runtime_config  # noqa: E402
from wildtrace.diagram import validate_with_opencv  # noqa: E402
DRAWING_ROOT = REPO_ROOT / "isaac-sim" / "inputs" / "diagrams"
OUTPUT_PATH = REPO_ROOT / "docs" / "presentation" / "validator_robustness.json"
RNG = np.random.default_rng(14)


def _score(image: Image.Image, settings: dict[str, object], scratch: Path) -> dict[str, object]:
    scratch.parent.mkdir(parents=True, exist_ok=True)
    image.save(scratch)
    result = validate_with_opencv(scratch, settings)
    return {"passed": result.passed, "score": round(result.score, 4), "flags": result.flags}


def _variants(source: Image.Image) -> dict[str, Image.Image]:
    gray = source.convert("L")
    width, height = gray.size
    small = gray.resize((max(1, width // 2), max(1, height // 2)), Image.Resampling.BILINEAR)
    noisy = np.asarray(gray, dtype=np.uint8).copy()
    flip = RNG.random(noisy.shape) < 0.02
    salt = RNG.random(noisy.shape) < 0.5
    noisy[flip & salt] = 255
    noisy[flip & ~salt] = 0
    return {
        "original": gray,
        "blur_radius_2": gray.filter(ImageFilter.GaussianBlur(radius=2)),
        "resize_half_and_back": small.resize((width, height), Image.Resampling.BILINEAR),
        "salt_pepper_2pct": Image.fromarray(noisy, mode="L"),
    }


def main() -> None:
    settings = load_runtime_config(REPO_ROOT)["models"]["opencv_prescreen"]
    drawings = sorted(DRAWING_ROOT.glob("*/*.png"))
    rows = []
    scratch = REPO_ROOT / "outputs" / "presentation" / "_robustness_scratch.png"
    for path in drawings:
        source = Image.open(path)
        record: dict[str, object] = {
            "drawing": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "category": path.parent.name,
            "variants": {},
        }
        for name, image in _variants(source).items():
            record["variants"][name] = _score(image, settings, scratch)
        rows.append(record)
    if scratch.exists():
        scratch.unlink()

    summary: dict[str, dict[str, object]] = {}
    for name in ("original", "blur_radius_2", "resize_half_and_back", "salt_pepper_2pct"):
        scored = [row["variants"][name] for row in rows]
        summary[name] = {
            "n": len(scored),
            "pass_rate": (sum(1 for item in scored if item["passed"]) / len(scored)) if scored else None,
            "mean_score": (sum(item["score"] for item in scored) / len(scored)) if scored else None,
        }
    report = {
        "experiment": "opencv_prescreen_robustness",
        "device": "cpu",
        "drawings": str(DRAWING_ROOT.relative_to(REPO_ROOT)).replace("\\", "/"),
        "perturbations": ["original", "blur_radius_2", "resize_half_and_back", "salt_pepper_2pct"],
        "summary": summary,
        "rows": rows,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUTPUT_PATH.relative_to(REPO_ROOT)} ({len(rows)} drawings)")


if __name__ == "__main__":
    main()
