"""Train the bounded Cartesian-delta policy from recorded demonstrations."""
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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from behavior_cloning import (
    CartesianDeltaPolicy,
    load_demonstrations,
    save_checkpoint,
    set_reproducible_seed,
    split_by_drawing_id,
)
from project_config import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/xarm7_drawing.yaml")
    parser.add_argument("--demo-glob", action="append", default=[])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    config = load_config(args.config, PROJECT_ROOT)
    training = config["training"]
    patterns = args.demo_glob or [str(Path(config["project"]["output_dir"]) / "demonstration_*.npz")]
    features, actions, drawing_ids = load_demonstrations(patterns)
    if len(set(drawing_ids.tolist())) < 2:
        raise RuntimeError(
            "Behavioral cloning is not yet trainable: record at least two complete drawing IDs "
            "so validation is split by drawing rather than by correlated frames."
        )
    seed = int(training["seed"])
    set_reproducible_seed(seed)
    train_indices, validation_indices = split_by_drawing_id(
        drawing_ids, float(training["validation_fraction"]), seed
    )
    mean = features[train_indices].mean(axis=0)
    std = np.maximum(features[train_indices].std(axis=0), 1e-8)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(device_name)
    model = CartesianDeltaPolicy(
        features.shape[1],
        training["hidden_layers"],
        float(training["max_action_delta_m"]),
        mean,
        std,
    ).to(device)
    train_dataset = TensorDataset(torch.from_numpy(features[train_indices]), torch.from_numpy(actions[train_indices]))
    validation_dataset = TensorDataset(
        torch.from_numpy(features[validation_indices]), torch.from_numpy(actions[validation_indices])
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=int(training["batch_size"]), shuffle=True, generator=generator
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=int(training["batch_size"]), shuffle=False
    )
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError("TensorBoard is required; install it into a project-local target, not globally") from exc
    writer = SummaryWriter(log_dir=training["tensorboard_dir"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    loss_function = nn.MSELoss()
    best_loss, stale_epochs = float("inf"), 0
    checkpoint = Path(training["checkpoint_path"])
    try:
        for epoch in range(int(training["max_epochs"])):
            model.train()
            train_sum = 0.0
            for batch_features, batch_actions in train_loader:
                batch_features, batch_actions = batch_features.to(device), batch_actions.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(model(batch_features), batch_actions)
                loss.backward()
                optimizer.step()
                train_sum += float(loss.item()) * len(batch_features)
            train_loss = train_sum / len(train_dataset)
            model.eval()
            validation_sum = 0.0
            with torch.no_grad():
                for batch_features, batch_actions in validation_loader:
                    batch_features, batch_actions = batch_features.to(device), batch_actions.to(device)
                    validation_sum += float(loss_function(model(batch_features), batch_actions).item()) * len(batch_features)
            validation_loss = validation_sum / len(validation_dataset)
            writer.add_scalars("mse", {"train": train_loss, "validation": validation_loss}, epoch)
            print(f"epoch={epoch:03d} train_mse={train_loss:.8g} validation_mse={validation_loss:.8g}")
            if validation_loss < best_loss - 1e-10:
                best_loss, stale_epochs = validation_loss, 0
                save_checkpoint(
                    checkpoint,
                    model,
                    best_validation_mse=best_loss,
                    epoch=epoch,
                    seed=seed,
                    train_drawing_ids=sorted(set(drawing_ids[train_indices].tolist())),
                    validation_drawing_ids=sorted(set(drawing_ids[validation_indices].tolist())),
                )
            else:
                stale_epochs += 1
                if stale_epochs >= int(training["patience"]):
                    print(f"[EARLY STOP] no validation improvement for {stale_epochs} epochs")
                    break
    finally:
        writer.close()
    print(json.dumps({"checkpoint": str(checkpoint), "best_validation_mse": best_loss, "device": str(device)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
