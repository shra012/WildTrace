"""Path tracking metrics and artifact writers."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


def pointwise_errors(desired: Sequence[Sequence[float]], executed: Sequence[Sequence[float]]) -> np.ndarray:
    desired_array = np.asarray(desired, dtype=np.float64)
    executed_array = np.asarray(executed, dtype=np.float64)
    if desired_array.shape != executed_array.shape or desired_array.ndim != 2:
        raise ValueError(f"Expected matching NxD paths, got {desired_array.shape} and {executed_array.shape}")
    return np.linalg.norm(desired_array - executed_array, axis=1)


def nearest_path_errors(reference: Sequence[Sequence[float]], samples: Sequence[Sequence[float]]) -> np.ndarray:
    reference_array = np.asarray(reference, dtype=np.float64)
    samples_array = np.asarray(samples, dtype=np.float64)
    if reference_array.ndim != 2 or samples_array.ndim != 2 or reference_array.shape[1] != samples_array.shape[1]:
        raise ValueError("Paths must be NxD arrays with the same D")
    if len(reference_array) == 0 or len(samples_array) == 0:
        raise ValueError("Paths must be non-empty")
    return np.min(np.linalg.norm(samples_array[:, None, :] - reference_array[None, :, :], axis=2), axis=1)


def summarize_errors(errors: Sequence[float]) -> Dict[str, float]:
    values = np.asarray(errors, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Errors must be finite and non-empty")
    return {
        "mean_error_m": float(values.mean()),
        "rmse_m": float(np.sqrt(np.mean(values**2))),
        "max_error_m": float(values.max()),
        "p95_error_m": float(np.percentile(values, 95)),
        "sample_count": int(values.size),
    }


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metrics(path: str | Path, values: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dict(values), indent=2, sort_keys=True), encoding="utf-8")

