"""Trajectory loading and validation without Isaac Sim dependencies."""
from __future__ import annotations

import csv
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


Trajectory = Dict[str, Any]


class TrajectoryError(ValueError):
    """Raised when trajectory data cannot be standardized safely."""


def _point_xy(value: Any) -> List[float]:
    if isinstance(value, Mapping):
        if "x" not in value or "y" not in value:
            raise TrajectoryError(f"Point mapping lacks x/y: {value!r}")
        pair = (value["x"], value["y"])
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        pair = (value[0], value[1])
    else:
        raise TrajectoryError(f"Unsupported point representation: {value!r}")
    try:
        x, y = float(pair[0]), float(pair[1])
    except (TypeError, ValueError) as exc:
        raise TrajectoryError(f"Non-numeric point: {value!r}") from exc
    if not (math.isfinite(x) and math.isfinite(y)):
        raise TrajectoryError(f"NaN or infinite point: {value!r}")
    return [x, y]


def standardize_trajectory(payload: Mapping[str, Any], *, drop_consecutive_duplicates: bool = True) -> Trajectory:
    """Convert a Gold/WildTrace mapping to the project canonical representation.

    Point order and stroke boundaries are retained. Exact consecutive duplicate
    points are removed by default; non-consecutive repeats (including deliberate
    closed-loop endpoints) are preserved.
    """
    if "trajectory" in payload and isinstance(payload["trajectory"], Mapping):
        payload = payload["trajectory"]
    drawing_id = payload.get("drawing_id", payload.get("sample_id", payload.get("id")))
    if drawing_id is None or str(drawing_id).strip() == "":
        raise TrajectoryError("Missing drawing_id/sample_id")
    raw_strokes = payload.get("strokes")
    if not isinstance(raw_strokes, Sequence) or isinstance(raw_strokes, (str, bytes)):
        raise TrajectoryError("Trajectory must contain a strokes list")

    strokes: List[Dict[str, Any]] = []
    seen_ids = set()
    for ordinal, raw in enumerate(raw_strokes):
        if not isinstance(raw, Mapping):
            raise TrajectoryError(f"Stroke {ordinal} is not a mapping")
        stroke_id = int(raw.get("stroke_id", ordinal))
        if stroke_id in seen_ids:
            raise TrajectoryError(f"Duplicate stroke_id {stroke_id}")
        seen_ids.add(stroke_id)
        raw_points = raw.get("points", raw.get("trajectory"))
        if not isinstance(raw_points, Sequence) or isinstance(raw_points, (str, bytes)):
            raise TrajectoryError(f"Stroke {stroke_id} has no points list")
        points: List[List[float]] = []
        for raw_point in raw_points:
            point = _point_xy(raw_point)
            if drop_consecutive_duplicates and points and point == points[-1]:
                continue
            points.append(point)
        if not points:
            raise TrajectoryError(f"Stroke {stroke_id} is empty")
        if len(points) < 2:
            raise TrajectoryError(f"Stroke {stroke_id} needs at least two distinct points")
        strokes.append({"stroke_id": stroke_id, "points": points})
    if not strokes:
        raise TrajectoryError("Trajectory has no strokes")
    return {"drawing_id": str(drawing_id), "strokes": strokes}


def _from_rows(rows: Iterable[Mapping[str, Any]], fallback_id: str) -> Trajectory:
    grouped: Dict[int, List[List[float]]] = {}
    drawing_id = fallback_id
    implicit_stroke = 0
    previous_pen_down = False
    for row_number, row in enumerate(rows, 2):
        if row.get("drawing_id") not in (None, ""):
            drawing_id = str(row["drawing_id"])
        pen_value = str(row.get("pen_state", row.get("pen_down", "down"))).strip().lower()
        pen_down = pen_value not in {"0", "false", "up", "pen_up"}
        if not pen_down:
            previous_pen_down = False
            continue
        if row.get("stroke_id") not in (None, ""):
            stroke_id = int(row["stroke_id"])
        else:
            if not previous_pen_down and grouped:
                implicit_stroke += 1
            stroke_id = implicit_stroke
        try:
            point = _point_xy(row)
        except TrajectoryError as exc:
            raise TrajectoryError(f"CSV row {row_number}: {exc}") from exc
        grouped.setdefault(stroke_id, []).append(point)
        previous_pen_down = True
    return standardize_trajectory(
        {"drawing_id": drawing_id, "strokes": [{"stroke_id": k, "points": v} for k, v in grouped.items()]}
    )


def _numpy_payload(path: Path) -> Any:
    import numpy as np

    value = np.load(path, allow_pickle=True)
    if isinstance(value, np.lib.npyio.NpzFile):
        try:
            if "trajectory_json" in value.files:
                return json.loads(str(value["trajectory_json"].item()))
            if "trajectory" in value.files:
                obj = value["trajectory"]
                return obj.item() if obj.shape == () else obj.tolist()
            if {"x", "y"}.issubset(value.files):
                ids = value["stroke_id"] if "stroke_id" in value.files else np.zeros_like(value["x"], dtype=int)
                rows = [
                    {"drawing_id": path.stem, "stroke_id": int(s), "x": float(x), "y": float(y)}
                    for x, y, s in zip(value["x"], value["y"], ids)
                ]
                return _from_rows(rows, path.stem)
            raise TrajectoryError(f"Unsupported NPZ fields: {value.files}")
        finally:
            value.close()
    if getattr(value, "shape", None) == ():
        return value.item()
    array = np.asarray(value)
    if array.ndim == 2 and array.shape[1] >= 2:
        return {"drawing_id": path.stem, "strokes": [{"stroke_id": 0, "points": array[:, :2].tolist()}]}
    return value.tolist()


def load_trajectory(path: str | Path, *, first_n_strokes: int | None = None) -> Trajectory:
    """Load JSON/CSV/NPY/NPZ/pickle/Parquet trajectory data.

    Pickle should only be used for trusted local data because unpickling executes
    Python object constructors.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    suffix = source.suffix.lower()
    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping) and not any(
            payload.get(key) not in (None, "") for key in ("drawing_id", "sample_id", "id")
        ):
            payload = {**payload, "drawing_id": source.stem}
    elif suffix == ".csv":
        with source.open("r", newline="", encoding="utf-8-sig") as stream:
            payload = _from_rows(csv.DictReader(stream), source.stem)
    elif suffix in {".npy", ".npz"}:
        payload = _numpy_payload(source)
    elif suffix in {".pkl", ".pickle"}:
        with source.open("rb") as stream:
            payload = pickle.load(stream)  # nosec: documented trusted-local-data support
    elif suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise TrajectoryError("Parquet support requires project-local pandas/pyarrow") from exc
        payload = _from_rows(pd.read_parquet(source).to_dict("records"), source.stem)
    else:
        raise TrajectoryError(f"Unsupported trajectory extension: {suffix}")

    trajectory = payload if _is_canonical(payload) else standardize_trajectory(payload)
    if first_n_strokes is not None:
        if first_n_strokes < 1 or len(trajectory["strokes"]) < first_n_strokes:
            raise TrajectoryError(
                f"Requested {first_n_strokes} strokes but {trajectory['drawing_id']} has {len(trajectory['strokes'])}"
            )
        trajectory = {"drawing_id": trajectory["drawing_id"], "strokes": trajectory["strokes"][:first_n_strokes]}
    return trajectory


def _is_canonical(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"drawing_id", "strokes"}:
        return False
    # Revalidate even canonical-looking external data.
    return False

