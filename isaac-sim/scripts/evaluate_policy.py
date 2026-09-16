"""Offline BC checkpoint evaluation on held demonstration NPZ files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR = PROJECT_ROOT / ".vendor"
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch

from behavior_cloning import load_checkpoint, load_demonstrations
from project_config import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/xarm7_drawing.yaml")
    parser.add_argument("--demo-glob", action="append", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    config = load_config(args.config, PROJECT_ROOT)
    features, actions, drawing_ids = load_demonstrations(args.demo_glob)
    model, metadata = load_checkpoint(config["training"]["checkpoint_path"], args.device)
    with torch.no_grad():
        predictions = model(torch.from_numpy(features).to(args.device)).cpu().numpy()
    errors = predictions - actions
    report = {
        "sample_count": len(features),
        "drawing_ids": sorted(set(drawing_ids.tolist())),
        "mse_m2": float(np.mean(errors**2)),
        "rmse_m": float(np.sqrt(np.mean(errors**2))),
        "maximum_action_norm_m": float(np.max(np.linalg.norm(predictions, axis=1))),
        "checkpoint_metadata": metadata,
        "scope": "offline imitation evaluation; not an Isaac closed-loop success claim",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
