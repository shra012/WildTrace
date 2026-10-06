#!/usr/bin/env python3
"""Check whether the SDXL comparison can run, without inventing results.

Writes docs/benchmarks/sdxl_preflight.json. Quality and performance metrics
stay null until scripts/benchmark_setups.py actually finishes sdxl-r1.
"""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "benchmark_sdxl.yaml"
OUTPUT_PATH = REPO_ROOT / "docs" / "benchmarks" / "sdxl_preflight.json"
UNMEASURED = (
    "acceptance_rate",
    "first_attempt_pass_rate",
    "attempts",
    "combined_validation_score",
    "strong_referee_pass_rate",
    "bioclip_recognizability",
    "silhouette_iou",
    "drawing_structure",
    "blank_output_rate",
    "exportable_trajectory_rate",
    "ink_coverage",
    "estimated_drawing_time_s",
    "generation_latency_s",
    "peak_memory",
    "robot_tracking_rmse",
)


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _cuda_available() -> bool:
    if not _installed("torch"):
        return False
    import torch

    return bool(torch.cuda.is_available())


def _count_existing(subjects: list[dict[str, object]], key: str) -> int:
    return sum(1 for subject in subjects if (REPO_ROOT / str(subject[key])).is_file())


def build_report() -> dict[str, object]:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    comparison = config["comparison"]
    evalset = json.loads((REPO_ROOT / comparison["evalset"]["path"]).read_text())
    subjects = evalset["subjects"]
    expected = len(subjects)
    packages = {
        "diffusers": _installed("diffusers"),
        "accelerate": _installed("accelerate"),
        "gguf": _installed("gguf"),
        "open_clip": _installed("open_clip"),
    }
    requirements = {
        "cuda_gpu": _cuda_available(),
        "benchmark_crops": {"available": _count_existing(subjects, "crop_path"), "expected": expected},
        "benchmark_masks": {"available": _count_existing(subjects, "mask_path"), "expected": expected},
        "isolated_subject_images": {"available": _count_existing(subjects, "isolated_path"), "expected": expected},
        "packages": packages,
        "openrouter_judge_key": bool(os.environ.get("OPENROUTER_API_KEY", "").strip()),
    }
    ready = (
        requirements["cuda_gpu"]
        and requirements["benchmark_crops"]["available"] == expected
        and requirements["benchmark_masks"]["available"] == expected
        and requirements["isolated_subject_images"]["available"] == expected
        and all(packages.values())
        and requirements["openrouter_judge_key"]
    )
    return {
        "experiment": comparison["run_id"],
        "config": "configs/benchmark_sdxl.yaml",
        "executed": False,
        "ready_to_run": ready,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "requirements": requirements,
        "metrics": {name: None for name in UNMEASURED},
        "note": (
            "SDXL quality and performance are unmeasured. Historical r2 figures describe "
            "the original experiment and are not measurements of the corrected contour helper."
        ),
    }


def main() -> None:
    report = build_report()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    requirements = report["requirements"]
    print(f"experiment: {report['experiment']} executed={report['executed']} ready={report['ready_to_run']}")
    print(f"cuda_gpu: {requirements['cuda_gpu']}")
    for key in ("benchmark_crops", "benchmark_masks", "isolated_subject_images"):
        row = requirements[key]
        print(f"{key}: {row['available']} of {row['expected']}")
    print("packages: " + ", ".join(f"{name}={'yes' if ok else 'no'}" for name, ok in requirements["packages"].items()))
    print(f"openrouter_judge_key: {requirements['openrouter_judge_key']}")
    print(f"wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
