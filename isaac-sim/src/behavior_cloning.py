"""Behavioral-cloning model and dataset helpers (Cartesian deltas only)."""
from __future__ import annotations

import glob
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from demonstration import make_policy_features


class CartesianDeltaPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: Sequence[int] = (256, 256, 128),
        action_limit_m: float = 0.005,
        input_mean: Sequence[float] | None = None,
        input_std: Sequence[float] | None = None,
    ):
        super().__init__()
        if list(hidden_layers) != [256, 256, 128]:
            raise ValueError("This stage requires MLP hidden layers [256, 256, 128]")
        mean = torch.zeros(input_dim) if input_mean is None else torch.as_tensor(input_mean, dtype=torch.float32)
        std = torch.ones(input_dim) if input_std is None else torch.as_tensor(input_std, dtype=torch.float32)
        if mean.shape != (input_dim,) or std.shape != (input_dim,):
            raise ValueError("Normalization vectors must match input_dim")
        self.register_buffer("input_mean", mean)
        self.register_buffer("input_std", torch.clamp(std, min=1e-8))
        self.action_limit_m = float(action_limit_m)
        widths = [input_dim, *hidden_layers, 3]
        modules: List[nn.Module] = []
        for index, (in_features, out_features) in enumerate(zip(widths[:-1], widths[1:])):
            modules.append(nn.Linear(in_features, out_features))
            if index < len(widths) - 2:
                modules.append(nn.ReLU())
        self.network = nn.Sequential(*modules)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.input_mean) / self.input_std
        return torch.tanh(self.network(normalized)) * self.action_limit_m


def load_demonstrations(patterns: Iterable[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths: List[Path] = []
    for pattern in patterns:
        matches = [Path(value) for value in glob.glob(pattern)]
        paths.extend(matches if matches else [Path(pattern)])
    paths = sorted(set(path.resolve() for path in paths if path.is_file()))
    if not paths:
        raise FileNotFoundError("No NPZ demonstrations matched")
    feature_chunks, action_chunks, id_chunks = [], [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "drawing_id",
                "joint_positions_rad",
                "joint_velocities_rad_s",
                "pen_tip_position_m",
                "current_target_m",
                "lookahead_target_m",
                "pen_down",
                "cartesian_action_delta_m",
            }
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"{path} lacks fields {sorted(missing)}")
            mapping = {name: data[name] for name in required}
            feature_chunks.append(make_policy_features(mapping))
            action_chunks.append(np.asarray(data["cartesian_action_delta_m"], dtype=np.float32))
            id_chunks.append(np.asarray(data["drawing_id"]).astype(str))
    return np.concatenate(feature_chunks), np.concatenate(action_chunks), np.concatenate(id_chunks)


def split_by_drawing_id(
    drawing_ids: Sequence[str], validation_fraction: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(drawing_ids).astype(str)
    unique = sorted(set(ids.tolist()))
    if len(unique) < 2:
        raise ValueError("Training/validation split requires demonstrations from at least two complete drawing IDs")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0,1)")
    rng = random.Random(seed)
    rng.shuffle(unique)
    validation_count = min(len(unique) - 1, max(1, round(len(unique) * validation_fraction)))
    validation_ids = set(unique[:validation_count])
    validation_mask = np.asarray([value in validation_ids for value in ids])
    return np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask)


def save_checkpoint(path: str | Path, model: CartesianDeltaPolicy, **metadata) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "input_dim": int(model.input_mean.numel()),
        "hidden_layers": [256, 256, 128],
        "action_limit_m": model.action_limit_m,
        "metadata": metadata,
    }
    torch.save(payload, destination)
    return destination


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    try:
        payload = torch.load(Path(path), map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(Path(path), map_location=device)
    model = CartesianDeltaPolicy(
        payload["input_dim"], payload["hidden_layers"], payload["action_limit_m"]
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload.get("metadata", {})


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

