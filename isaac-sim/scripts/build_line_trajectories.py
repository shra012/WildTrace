"""Turn each stored line diagram into a pen trajectory the arm can finish.

Both edges of the ink are kept, curves stay smooth, and hairpin corners become
separate strokes so the pen can lift. This matches the finished ea59 dog run.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from coordinate_mapper import map_trajectory_to_plane
from drawing_state_machine import build_motion_sequence
from trajectory_loader import load_trajectory

DIAGRAMS = PROJECT / "inputs" / "diagrams"
OUT = PROJECT / "data" / "line_runs"


def _spaced(points: np.ndarray, step: float = 2.2) -> np.ndarray:
    kept = [points[0]]
    for point in points[1:]:
        if np.linalg.norm(point - kept[-1]) >= step:
            kept.append(point)
    if np.linalg.norm(points[-1] - kept[-1]) > 0.4:
        kept.append(points[-1])
    if len(kept) >= 2 and np.linalg.norm(kept[0] - kept[-1]) < step:
        kept = kept[:-1]
    return np.asarray(kept, dtype=np.float64)


def _split_hairpins(points: np.ndarray, limit: float = 95.0) -> list[np.ndarray]:
    if len(points) < 4:
        return [points]
    cuts = []
    for index in range(1, len(points) - 1):
        incoming = points[index] - points[index - 1]
        outgoing = points[index + 1] - points[index]
        incoming_norm = np.linalg.norm(incoming)
        outgoing_norm = np.linalg.norm(outgoing)
        if incoming_norm < 1e-9 or outgoing_norm < 1e-9:
            continue
        turn = math.degrees(
            math.acos(float(np.clip(np.dot(incoming, outgoing) / (incoming_norm * outgoing_norm), -1, 1)))
        )
        if turn >= limit:
            cuts.append(index)
    pieces = []
    start = 0
    for cut in cuts:
        pieces.append(points[start : cut + 1])
        start = cut + 1
    pieces.append(points[start:])
    return pieces


def trajectory_from_diagram(path: Path) -> dict:
    image = Image.open(path).convert("L")
    array = np.asarray(image)
    height, width = array.shape
    binary = np.where(array < 180, 255, 0).astype(np.uint8)
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    strokes = []
    for contour in contours:
        if len(contour) < 8:
            continue
        curve = _spaced(contour.reshape(-1, 2).astype(np.float64))
        if len(curve) < 2:
            continue
        for piece in _split_hairpins(curve):
            normalized = []
            for x_coord, y_coord in piece:
                point = [
                    round(float(x_coord) / max(width - 1, 1), 6),
                    round(float(y_coord) / max(height - 1, 1), 6),
                ]
                if not normalized or abs(point[0] - normalized[-1][0]) > 1e-5 or abs(point[1] - normalized[-1][1]) > 1e-5:
                    normalized.append(point)
            if len(normalized) >= 2:
                strokes.append(normalized)
    return {
        "drawing_id": f"{path.parent.name}_{path.stem[:8]}",
        "sample_id": path.stem,
        "category": path.parent.name,
        "coordinate_frame": "normalized_canvas",
        "canvas_width": int(width),
        "canvas_height": int(height),
        "source_diagram": str(path.relative_to(PROJECT)).replace("\\", "/"),
        "stroke_count": len(strokes),
        "point_count": sum(len(stroke) for stroke in strokes),
        "strokes": [
            {"stroke_id": index, "pen_state": "down", "points": stroke}
            for index, stroke in enumerate(strokes)
        ],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for diagram in sorted(DIAGRAMS.glob("*/*.png")):
        payload = trajectory_from_diagram(diagram)
        destination = OUT / f"{diagram.parent.name}_{diagram.stem}.json"
        destination.write_text(json.dumps(payload), encoding="utf-8")
        loaded = load_trajectory(destination)
        mapped = map_trajectory_to_plane(
            loaded,
            center_xy=[0.45, 0.0],
            size_xy=[0.16, 0.12],
            flip_image_y=True,
            max_step=0.0015,
            smoothing_strength=0.40,
            corner_angle_degrees=28.0,
        )
        build_motion_sequence(
            mapped["strokes"],
            pen_down_z=0.201,
            pen_up_z=0.235,
            approach_height=0.27,
            max_cartesian_step=0.0015,
            corner_angle_degrees=28.0,
            corner_densify_window=5,
            corner_densify_factor=3,
        )
        print(f"{payload['drawing_id']} strokes={payload['stroke_count']} points={payload['point_count']}")


if __name__ == "__main__":
    main()
