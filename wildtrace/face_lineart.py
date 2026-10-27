"""Face photo -> robot-drawable line art and stroke trajectory.

The face counterpart of the animal outline stages: MediaPipe face landmarks
give clean, ordered feature lines (face outline, eyes, irises, brows, nose,
lips) and a multiclass selfie segmenter gives the hair (and optionally the
shoulders). Lines are chained into pen strokes, simplified, ordered to keep
pen-up travel short, and written in the same trajectory JSON schema as the
gold stage (normalized canvas, one entry per pen-down stroke), so the robot
side draws a face exactly like an animal.

MediaPipe's Python package runs on CPU on Linux; `device: auto` uses the GPU
delegate only where MediaPipe supports it and falls back to CPU otherwise.
"""
from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageOps

from wildtrace.images import write_svg

MODEL_URLS = {
    "face_landmarker": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "selfie_multiclass": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
}
# Categories of the selfie multiclass segmenter.
HAIR, BODY_SKIN, FACE_SKIN, CLOTHES = 1, 2, 3, 4

DEFAULT_SETTINGS: dict[str, Any] = {
    "device": "cpu",                     # cpu (MediaPipe's normal Python mode) | gpu | auto (try GPU, fall back)
    "model_dir": "~/.cache/wildtrace/mediapipe",
    "max_side_px": 1024,                 # photos are resized so the long side is at most this
    "include_irises": True,
    # sketch: single-line brows, a one-stroke nose (bridge side + nostril base), hairline instead of
    # the forehead edge, and neck/collar lines. mesh: MediaPipe's landmark outlines as they are.
    "feature_style": "sketch",
    "include_hair": True,
    "include_shoulders": True,
    "simplify_epsilon_px": 1.2,          # Douglas-Peucker tolerance on the resized photo
    "min_stroke_length_px": 18.0,
    "hair_face_gap_px": 6.0,             # hair outline points this close to the face outline are dropped
    "max_points_per_stroke": 128,        # export.yaml max_points_per_stroke
    "margin": 0.04,                      # normalized-canvas margin around the drawing
    # Group photos: faces are searched in the whole photo and in overlapping tiles (fractions of
    # the photo size, each resized to detect_tile_px) so small faces are found too.
    "detect_tile_fractions": (1.0, 0.5, 0.35),
    "detect_tile_px": 640,
    "max_faces": 8,
    "crop_min_side_px": 768,             # small face crops are upscaled to at least this long side
    # Head-and-shoulders crop around the face, in face widths/heights; strokes
    # leaving it are clipped so the face, not the torso, fills the drawing.
    "portrait_half_width_faces": 1.6,
    "portrait_below_chin_faces": 0.9,
    "portrait_above_brow_faces": 0.8,
}


@dataclass
class FaceStageEvent:
    name: str
    label: str
    status: str
    backend: str | None = None
    detail: str | None = None
    artifact_path: Path | None = None


@dataclass
class FaceLineArt:
    """Strokes in pixel coordinates of the resized photo, plus what produced them."""

    strokes: list[np.ndarray]
    labels: list[str]
    size: tuple[int, int]
    backend: str
    notes: list[str] = field(default_factory=list)


class NoFaceFound(RuntimeError):
    """The photo has no detectable face."""


def _model_path(name: str, settings: dict[str, Any]) -> Path:
    directory = Path(str(settings["model_dir"])).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / Path(MODEL_URLS[name]).name
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(MODEL_URLS[name], temporary)
        temporary.replace(path)
    return path


def _base_options(model: Path, device: str):
    from mediapipe.tasks.python import BaseOptions

    delegate = BaseOptions.Delegate.GPU if device == "gpu" else BaseOptions.Delegate.CPU
    return BaseOptions(model_asset_path=str(model), delegate=delegate)


def _create(factory: Callable[[str], Any], device: str) -> tuple[Any, str]:
    """Build a MediaPipe task on the requested device; `auto` tries GPU, then CPU."""
    if device in ("gpu", "auto"):
        try:
            return factory("gpu"), "mediapipe (GPU)"
        except Exception:
            if device == "gpu":
                raise
    return factory("cpu"), "mediapipe (CPU)"


def load_photo(path: Path, max_side_px: int) -> np.ndarray:
    """RGB uint8 array, EXIF-rotated, long side capped."""
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    scale = min(1.0, float(max_side_px) / max(image.size))
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    return np.asarray(image)


def _chain(connections) -> list[list[int]]:
    """Turn an undirected landmark edge set into ordered polylines (closed loops repeat their start)."""
    adjacency: dict[int, set[int]] = {}
    for connection in connections:
        a, b = (connection.start, connection.end) if hasattr(connection, "start") else connection
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    unused = {tuple(sorted(edge)) for a in adjacency for edge in ((a, b) for b in adjacency[a])}
    paths: list[list[int]] = []

    def walk(start: int) -> list[int]:
        path = [start]
        current = start
        while True:
            options = [n for n in sorted(adjacency[current]) if tuple(sorted((current, n))) in unused]
            if not options:
                return path
            following = options[0]
            unused.discard(tuple(sorted((current, following))))
            path.append(following)
            current = following

    # Open chains start at an end point; whatever remains is a closed loop.
    for node in sorted(n for n, ns in adjacency.items() if len(ns) == 1):
        if any(tuple(sorted((node, n))) in unused for n in adjacency[node]):
            paths.append(walk(node))
    while unused:
        paths.append(walk(min(unused)[0]))
    return [p for p in paths if len(p) >= 2]


# Face mesh landmark indices for the sketch style (MediaPipe's "left" is the subject's left).
BROW_ROWS = {  # upper and lower edge of each brow, outer end first
    "left_brow": ([300, 293, 334, 296, 336], [276, 283, 282, 295, 285]),
    "right_brow": ([70, 63, 105, 66, 107], [46, 53, 52, 65, 55]),
}
# One stroke the way people draw a nose: down one side of the bridge, a hook into the nostril,
# then along the base of the nose to the other nostril.
NOSE_SKETCH = [193, 122, 188, 114, 217, 209, 49, 48, 64, 98, 97, 2, 326, 327, 294, 278, 279]


def _chaikin(points: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Corner-cutting smoothing for short landmark polylines (keeps the end points)."""
    for _ in range(iterations):
        if len(points) < 3:
            return points
        q = 0.75 * points[:-1] + 0.25 * points[1:]
        r = 0.25 * points[:-1] + 0.75 * points[1:]
        points = np.vstack([points[:1], np.column_stack([q, r]).reshape(-1, 2), points[-1:]])
    return points


def _feature_strokes(landmarks, size: tuple[int, int], include_irises: bool,
                     style: str = "sketch") -> tuple[list[np.ndarray], list[str]]:
    from mediapipe.tasks.python.vision import FaceLandmarksConnections as C

    width, height = size
    points = np.asarray([[lm.x * width, lm.y * height] for lm in landmarks], dtype=np.float64)
    groups = [
        ("face_outline", C.FACE_LANDMARKS_FACE_OVAL),
        ("left_eye", C.FACE_LANDMARKS_LEFT_EYE),
        ("right_eye", C.FACE_LANDMARKS_RIGHT_EYE),
        ("lips", C.FACE_LANDMARKS_LIPS),
    ]
    if style == "mesh":
        groups += [("left_brow", C.FACE_LANDMARKS_LEFT_EYEBROW), ("right_brow", C.FACE_LANDMARKS_RIGHT_EYEBROW),
                   ("nose", C.FACE_LANDMARKS_NOSE)]
    if include_irises and len(points) > 473:
        groups += [("left_iris", C.FACE_LANDMARKS_LEFT_IRIS), ("right_iris", C.FACE_LANDMARKS_RIGHT_IRIS)]
    strokes, labels = [], []
    for label, connections in groups:
        for path in _chain(connections):
            stroke = points[path]
            if style != "mesh" and label != "face_outline":
                # Eyes, irises and lips are a handful of landmarks; corner cutting turns the
                # diamonds and polygons into the curves a person would draw.
                stroke = _chaikin(stroke, 3)
            strokes.append(stroke)
            labels.append(label)
    if style != "mesh":
        for label, (upper, lower) in BROW_ROWS.items():
            strokes.append(_chaikin((points[upper] + points[lower]) / 2))
            labels.append(label)
        strokes.append(_chaikin(points[NOSE_SKETCH]))
        labels.append("nose")
    return strokes, labels


def _mask_contours(mask: np.ndarray, min_area: float) -> list[np.ndarray]:
    mask = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [c.reshape(-1, 2).astype(np.float64) for c in contours if cv2.contourArea(c) >= min_area]


def _runs(points: np.ndarray, keep: np.ndarray, closed: bool = True, max_gap_px: float = 4.0) -> list[np.ndarray]:
    """Split a polyline into the runs of kept points, also breaking where consecutive points jump."""
    if closed and keep.all():
        return [np.vstack([points, points[:1]])]
    if closed:
        # Rotate so the polyline starts in a dropped section; then no run wraps around.
        start = int(np.argmin(keep))
        points, keep = np.roll(points, -start, axis=0), np.roll(keep, -start)
    pieces, current = [], []
    for point, kept in zip(points, keep):
        if kept and current and np.hypot(*(point - current[-1])) > max_gap_px:
            pieces.append(current)
            current = []
        if kept:
            current.append(point)
        elif current:
            pieces.append(current)
            current = []
    if current:
        pieces.append(current)
    return [np.asarray(piece) for piece in pieces if len(piece) >= 2]


def _distance_to(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    if len(polyline) == 0:
        return np.full(len(points), np.inf)
    poly = polyline.astype(np.float32).reshape(-1, 1, 2)
    return np.asarray([abs(cv2.pointPolygonTest(poly, (float(x), float(y)), True)) for x, y in points])


def _distance_to_polylines(points: np.ndarray, polylines: list[np.ndarray]) -> np.ndarray:
    """Distance from each point to the nearest segment of any open polyline."""
    best = np.full(len(points), np.inf)
    for line in polylines:
        if len(line) < 2:
            continue
        a, b = line[:-1], line[1:]
        ab = b - a
        length2 = np.maximum((ab ** 2).sum(axis=1), 1e-9)
        for start in range(0, len(points), 2048):
            chunk = points[start:start + 2048, None, :]
            t = np.clip(((chunk - a) * ab).sum(axis=2) / length2, 0.0, 1.0)
            nearest = a + t[..., None] * ab
            best[start:start + 2048] = np.minimum(best[start:start + 2048], np.sqrt(((chunk - nearest) ** 2).sum(axis=2)).min(axis=1))
    return best


def trim_forehead(face_outline: np.ndarray, brow_y: float) -> list[np.ndarray]:
    """The face outline from temple to temple round the chin: the part above the brows is where
    the hairline goes in a sketch, so it is left out."""
    # Landmarks are ~20 px apart, so no splitting on jumps (that is for dense mask contours).
    return _runs(face_outline, face_outline[:, 1] >= brow_y, closed=True, max_gap_px=np.inf)


def hair_over_forehead(category: np.ndarray, face_outline: np.ndarray) -> bool:
    """Whether hair covers the top of the head (bald heads keep the full face outline)."""
    low, high = face_outline.min(axis=0), face_outline.max(axis=0)
    width, height = high - low
    x0, x1 = int(low[0] + 0.25 * width), int(high[0] - 0.25 * width)
    y0, y1 = int(max(0, low[1] - 0.35 * height)), int(low[1] + 0.1 * height)
    band = category[y0:y1, max(0, x0):max(0, x1)]
    return band.size > 0 and float((band == HAIR).mean()) > 0.3


def person_columns(face: tuple[float, float, float, float], others, width: int) -> tuple[float, float]:
    """x range of the photo that belongs to `face` when other people stand beside it: the
    boundary to each neighbour divides the gap between face centres in proportion to face width."""
    center, face_w = (face[0] + face[2]) / 2, face[2] - face[0]
    left, right = 0.0, float(width)
    for other in others:
        other_center, other_w = (other[0] + other[2]) / 2, other[2] - other[0]
        boundary = center + (other_center - center) * face_w / (face_w + other_w)
        if other_center < center:
            left = max(left, boundary)
        else:
            right = min(right, boundary)
    return left, right


def _smooth(points: np.ndarray, window: int = 5) -> np.ndarray:
    if len(points) <= window:
        return points
    kernel = np.ones(window) / window
    padded = np.pad(points, ((window // 2, window // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(2)], axis=1)


def _segmentation_strokes(category: np.ndarray, face_outline: np.ndarray, settings: dict[str, Any],
                          columns: tuple[float, float] | None = None,
                          drawn_outline: list[np.ndarray] | None = None) -> tuple[list[np.ndarray], list[str]]:
    """Hair, neck and shoulder contours. Parts that retrace `drawn_outline` (the face outline as
    it will be drawn; default the whole of face_outline) are left out. With a forehead-trimmed
    outline the hair contour's lower edge stays in, and that edge is the hairline."""
    height, width = category.shape
    left, right = columns or (0.0, float(width))
    category = category.copy()
    category[:, : max(0, int(np.floor(left)))] = 0  # other people in a group photo
    category[:, min(width, int(np.ceil(right))):] = 0
    gap = float(settings["hair_face_gap_px"])
    area = category.size
    outline = drawn_outline if drawn_outline is not None else [face_outline]
    sketch = settings.get("feature_style", "sketch") != "mesh"
    drawn_contours: list[np.ndarray] = []

    def keep(contour: np.ndarray) -> np.ndarray:
        # Drop edges that only exist because the photo, the crop or a neighbour cut the mask,
        # and the parts that would retrace the face outline or a contour already drawn.
        x, y = contour[:, 0], contour[:, 1]
        kept = (x > max(2, left + 2)) & (x < min(width - 3, right - 3)) & (y > 2) & (y < height - 3)
        return kept & (_distance_to_polylines(contour, outline + drawn_contours) > gap)

    strokes, labels = [], []
    masks = []
    if settings["include_hair"]:
        masks.append(("hair", category == HAIR, 0.004, False))
    if settings["include_shoulders"]:
        if sketch:
            # Neck sides and the collar: the outline of the visible neck and chest skin.
            masks.append(("neck", category == BODY_SKIN, 0.002, True))
        masks.append(("shoulders", (category == BODY_SKIN) | (category == CLOTHES), 0.01, True))
    for label, mask, min_fraction, below in masks:
        contours = [c for c in _mask_contours(mask, min_area=area * min_fraction) if _near_face(c, face_outline, below=below)]
        for contour in contours:
            for piece in _runs(contour, keep(contour)):
                strokes.append(_smooth(piece))
                labels.append(label)
        if label == "neck":
            drawn_contours += [np.vstack([c, c[:1]]) for c in contours]
    return strokes, labels


def _clip_to_box(stroke: np.ndarray, box: tuple[float, float, float, float]) -> list[np.ndarray]:
    """Split a stroke into the runs that lie inside box = (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box
    inside = (stroke[:, 0] >= x0) & (stroke[:, 0] <= x1) & (stroke[:, 1] >= y0) & (stroke[:, 1] <= y1)
    runs, current = [], []
    for point, kept in zip(stroke, inside):
        if kept:
            current.append(point)
        elif current:
            runs.append(np.asarray(current))
            current = []
    if current:
        runs.append(np.asarray(current))
    return [run for run in runs if len(run) >= 2]


def portrait_box(face_outline: np.ndarray, settings: dict[str, Any]) -> tuple[float, float, float, float]:
    low, high = face_outline.min(axis=0), face_outline.max(axis=0)
    face_w, face_h = high - low
    center_x = (low[0] + high[0]) / 2
    half = float(settings["portrait_half_width_faces"]) * face_w
    return (
        center_x - half,
        low[1] - float(settings["portrait_above_brow_faces"]) * face_h,
        center_x + half,
        high[1] + float(settings["portrait_below_chin_faces"]) * face_h,
    )


def _simplify(stroke: np.ndarray, epsilon: float, max_points: int) -> np.ndarray:
    closed = len(stroke) > 2 and np.linalg.norm(stroke[0] - stroke[-1]) < 1e-6
    simplified = cv2.approxPolyDP(stroke.astype(np.float32).reshape(-1, 1, 2), epsilon, closed).reshape(-1, 2)
    if closed and len(simplified) > 1:
        simplified = np.vstack([simplified, simplified[:1]])
    if len(simplified) > max_points:
        simplified = simplified[np.linspace(0, len(simplified) - 1, max_points).round().astype(int)]
    return simplified.astype(np.float64)


def _length(stroke: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(stroke, axis=0), axis=1).sum()) if len(stroke) > 1 else 0.0


def order_strokes(strokes: list[np.ndarray], labels: list[str]) -> tuple[list[np.ndarray], list[str]]:
    """Greedy nearest-neighbour order (each stroke may be reversed) to keep pen-up travel short.

    Starts with the face outline so the drawing reads as a face from the first stroke.
    """
    remaining = list(range(len(strokes)))
    first = labels.index("face_outline") if "face_outline" in labels else 0
    order, ordered = [first], [strokes[first]]
    remaining.remove(first)
    position = strokes[first][-1]
    while remaining:
        best, best_distance, best_reversed = None, math.inf, False
        for index in remaining:
            for reverse in (False, True):
                start = strokes[index][-1] if reverse else strokes[index][0]
                distance = float(np.linalg.norm(start - position))
                if distance < best_distance:
                    best, best_distance, best_reversed = index, distance, reverse
        stroke = strokes[best][::-1] if best_reversed else strokes[best]
        order.append(best)
        ordered.append(stroke)
        position = stroke[-1]
        remaining.remove(best)
    return ordered, [labels[i] for i in order]


def _landmarker(settings: dict[str, Any], num_faces: int):
    from mediapipe.tasks.python import vision

    model = _model_path("face_landmarker", settings)
    return _create(
        lambda d: vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(base_options=_base_options(model, d), num_faces=num_faces,
                                         min_face_detection_confidence=0.4, min_face_presence_confidence=0.4)
        ),
        str(settings["device"]).lower(),
    )


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _tiles(width: int, height: int, fractions) -> list[tuple[int, int, int, int]]:
    """(x0, y0, w, h) windows: the whole photo, then grids of smaller windows overlapping by half."""
    tiles = []
    for fraction in fractions:
        tile_w, tile_h = max(1, int(width * fraction)), max(1, int(height * fraction))
        steps = 1 if fraction >= 1.0 else int(np.ceil((1.0 - fraction) / (fraction / 2))) + 1
        for iy in range(steps):
            for ix in range(steps):
                x0 = 0 if steps == 1 else round(ix * (width - tile_w) / (steps - 1))
                y0 = 0 if steps == 1 else round(iy * (height - tile_h) / (steps - 1))
                tiles.append((x0, y0, tile_w, tile_h))
    return tiles


def detect_faces(photo: np.ndarray, settings: dict[str, Any] | None = None) -> tuple[list[tuple[float, float, float, float]], str]:
    """Face boxes (x0, y0, x1, y1) in photo pixels, left to right, and the backend that found them."""
    import mediapipe as mp

    settings = {**DEFAULT_SETTINGS, **(settings or {})}
    height, width = photo.shape[:2]
    tile_px = float(settings["detect_tile_px"])
    found = []
    landmarker, backend = _landmarker(settings, int(settings["max_faces"]))
    with landmarker:
        for x0, y0, tile_w, tile_h in _tiles(width, height, settings["detect_tile_fractions"]):
            tile = photo[y0:y0 + tile_h, x0:x0 + tile_w]
            scale = tile_px / max(tile_w, tile_h)
            if abs(scale - 1.0) > 0.05:
                tile = np.asarray(Image.fromarray(tile).resize((round(tile_w * scale), round(tile_h * scale)), Image.LANCZOS))
            result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(tile)))
            for landmarks in result.face_landmarks:
                xs = np.asarray([p.x for p in landmarks]) * tile_w + x0
                ys = np.asarray([p.y for p in landmarks]) * tile_h + y0
                found.append((float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())))
    # The same face seen in several windows: keep the largest box of each overlapping group.
    kept: list[tuple[float, float, float, float]] = []
    for box in sorted(found, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        if all(_iou(box, other) < 0.3 for other in kept):
            kept.append(box)
    return sorted(kept[: int(settings["max_faces"])], key=lambda b: (b[0] + b[2]) / 2), backend


def crop_portrait(photo: np.ndarray, face_box: tuple[float, float, float, float],
                  settings: dict[str, Any] | None = None) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Head-and-shoulders crop around one face (with room to spare), upscaled when small.

    Returns the crop and its (x0, y0, x1, y1) box in the photo."""
    settings = {**DEFAULT_SETTINGS, **(settings or {})}
    height, width = photo.shape[:2]
    x0, y0, x1, y1 = face_box
    face_w, face_h = x1 - x0, y1 - y0
    pad_x = (float(settings["portrait_half_width_faces"]) + 0.3) * face_w
    box = (
        int(max(0, (x0 + x1) / 2 - pad_x)),
        int(max(0, y0 - (float(settings["portrait_above_brow_faces"]) + 0.3) * face_h)),
        int(min(width, (x0 + x1) / 2 + pad_x)),
        int(min(height, y1 + (float(settings["portrait_below_chin_faces"]) + 0.3) * face_h)),
    )
    crop = photo[box[1]:box[3], box[0]:box[2]]
    scale = float(settings["crop_min_side_px"]) / max(crop.shape[:2])
    if scale > 1.0:
        crop = np.asarray(Image.fromarray(crop).resize((round(crop.shape[1] * scale), round(crop.shape[0] * scale)), Image.LANCZOS))
    return crop, box


def _central_face(faces, size: tuple[int, int]):
    """In a portrait crop the subject is the big face near the middle; neighbours are cut off at the sides."""
    width, height = size

    def score(landmarks) -> float:
        xs, ys = np.asarray([p.x for p in landmarks]), np.asarray([p.y for p in landmarks])
        area = (xs.max() - xs.min()) * (ys.max() - ys.min())
        offset = abs((xs.min() + xs.max()) / 2 - 0.5)
        return float(area * (1.0 - offset))

    return max(faces, key=score)


def _near_face(contour: np.ndarray, face_outline: np.ndarray, below: bool) -> bool:
    """Whether a segmentation contour belongs to this face's person: it overlaps the face box grown
    sideways (hair), or for the body, the band under the face reaching down to the photo border."""
    if len(face_outline) == 0:
        return True
    low, high = face_outline.min(axis=0), face_outline.max(axis=0)
    face_w, face_h = high - low
    if below:
        region = (low[0] - 0.6 * face_w, high[1] - 0.2 * face_h, high[0] + 0.6 * face_w, np.inf)
    else:
        region = (low[0] - 0.35 * face_w, low[1] - 0.6 * face_h, high[0] + 0.35 * face_w, high[1] + 0.6 * face_h)
    c_low, c_high = contour.min(axis=0), contour.max(axis=0)
    return c_low[0] < region[2] and c_high[0] > region[0] and c_low[1] < region[3] and c_high[1] > region[1]


def _box(landmarks, size: tuple[int, int]) -> tuple[float, float, float, float]:
    xs = np.asarray([p.x for p in landmarks]) * size[0]
    ys = np.asarray([p.y for p in landmarks]) * size[1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def extract_face_lineart(photo: np.ndarray, settings: dict[str, Any] | None = None,
                         other_faces=(), target_face: tuple[float, float, float, float] | None = None) -> FaceLineArt:
    """Detect the face and build feature, hair and shoulder strokes in photo pixel coordinates.

    With several faces in the photo the subject is the big central one; `other_faces` adds
    known neighbour face boxes (photo pixels) so their hair and shoulders are left out, and
    `target_face` (photo pixels) picks the subject when it is known from find_faces."""
    import mediapipe as mp
    from mediapipe.tasks.python import vision

    settings = {**DEFAULT_SETTINGS, **(settings or {})}
    height, width = photo.shape[:2]
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(photo))
    device = str(settings["device"]).lower()
    landmarker, backend = _landmarker(settings, 4)
    with landmarker:
        result = landmarker.detect(image)
    if not result.face_landmarks:
        raise NoFaceFound("No face found in the photo")
    if target_face is not None:
        landmarks = max(result.face_landmarks, key=lambda face: _iou(_box(face, (width, height)), target_face))
    else:
        landmarks = _central_face(result.face_landmarks, (width, height))
    others = [_box(face, (width, height)) for face in result.face_landmarks if face is not landmarks] + list(other_faces)
    strokes, labels = _feature_strokes(landmarks, (width, height), bool(settings["include_irises"]),
                                       str(settings.get("feature_style", "sketch")))
    face_outline = next((s for s, l in zip(strokes, labels) if l == "face_outline"), np.zeros((0, 2)))

    notes = []
    if settings["include_hair"] or settings["include_shoulders"]:
        segmenter_model = _model_path("selfie_multiclass", settings)
        segmenter, _ = _create(
            lambda d: vision.ImageSegmenter.create_from_options(
                vision.ImageSegmenterOptions(base_options=_base_options(segmenter_model, d), output_category_mask=True)
            ),
            device,
        )
        with segmenter:
            category = segmenter.segment(image).category_mask.numpy_view().copy()
        if category.shape != (height, width):
            category = cv2.resize(category, (width, height), interpolation=cv2.INTER_NEAREST)
        columns = person_columns(_box(landmarks, (width, height)), others, width)
        drawn_outline = None
        if str(settings.get("feature_style", "sketch")) != "mesh" and settings["include_hair"] and len(face_outline) \
                and hair_over_forehead(category, face_outline):
            brow_y = float(np.mean([s[:, 1].mean() for s, l in zip(strokes, labels) if l.endswith("_brow")]))
            drawn_outline = trim_forehead(face_outline, brow_y)
            index = labels.index("face_outline")
            strokes[index:index + 1] = drawn_outline
            labels[index:index + 1] = ["face_outline"] * len(drawn_outline)
        extra, extra_labels = _segmentation_strokes(category, face_outline, settings, columns, drawn_outline)
        strokes += extra
        labels += extra_labels
        notes.append(f"hair strokes: {extra_labels.count('hair')}, neck strokes: {extra_labels.count('neck')}, "
                     f"shoulder strokes: {extra_labels.count('shoulders')}")

    if len(face_outline):
        box = portrait_box(face_outline, settings)
        clipped = [(run, label) for stroke, label in zip(strokes, labels) for run in _clip_to_box(stroke, box)]
        strokes, labels = [s for s, _ in clipped], [l for _, l in clipped]

    epsilon, max_points = float(settings["simplify_epsilon_px"]), int(settings["max_points_per_stroke"])
    kept = [
        (_simplify(s, epsilon, max_points), l)
        for s, l in zip(strokes, labels)
        if _length(s) >= float(settings["min_stroke_length_px"])
    ]
    strokes, labels = order_strokes([s for s, _ in kept], [l for _, l in kept])
    return FaceLineArt(strokes=strokes, labels=labels, size=(width, height), backend=backend, notes=notes)


def to_trajectory(art: FaceLineArt, margin: float) -> dict[str, Any]:
    """Normalize to a square-ish canvas around the drawing in the gold-stage trajectory schema."""
    stacked = np.vstack(art.strokes)
    low, high = stacked.min(axis=0), stacked.max(axis=0)
    span = float(max(high - low))
    center = (low + high) / 2
    scale = (1.0 - 2 * margin) / span
    strokes = []
    for stroke in art.strokes:
        normalized = (stroke - center) * scale + 0.5
        strokes.append({"pen_state": "down", "points": [{"x": float(x), "y": float(y)} for x, y in normalized]})
    point_count = sum(len(s["points"]) for s in strokes)
    canvas = int(round(span / (1.0 - 2 * margin)))
    return {
        "bounds": {"x_max": 1.0, "x_min": 0.0, "y_max": 1.0, "y_min": 0.0},
        "canvas_height": canvas,
        "canvas_width": canvas,
        "coordinate_frame": "normalized_canvas",
        "normalization_method": "width_height_normalized",
        "point_count": point_count,
        "simplification_method": "douglas_peucker",
        "source": "face_landmarks",
        "stroke_count": len(strokes),
        "stroke_labels": list(art.labels),
        "strokes": strokes,
    }


def estimate_draw_seconds(trajectory: dict[str, Any], sheet_mm: tuple[float, float] = (160.0, 120.0),
                          draw_mm_s: float = 14.0, per_stroke_s: float = 6.0) -> float:
    """Pen-down length on the sheet at the drawing speed, plus a pen lift and travel per stroke."""
    scale = min(sheet_mm)  # the mapper fits the drawing into the sheet with one uniform scale
    length = 0.0
    for stroke in trajectory["strokes"]:
        points = np.asarray([[p["x"], p["y"]] for p in stroke["points"]])
        length += _length(points) * scale
    return length / draw_mm_s + per_stroke_s * len(trajectory["strokes"])


def render_lineart(trajectory: dict[str, Any], size: int = 1024, thickness: int = 3) -> Image.Image:
    canvas = np.full((size, size), 255, dtype=np.uint8)
    for stroke in trajectory["strokes"]:
        points = np.asarray([[p["x"] * (size - 1), p["y"] * (size - 1)] for p in stroke["points"]], dtype=np.int32)
        cv2.polylines(canvas, [points.reshape(-1, 1, 2)], False, 0, thickness, lineType=cv2.LINE_AA)
    return Image.fromarray(canvas)


def find_faces(
    *,
    image_path: Path,
    out_dir: Path,
    settings: dict[str, Any] | None = None,
    emit: Callable[[FaceStageEvent], None] = lambda event: None,
) -> list[dict[str, Any]]:
    """Find every face in a photo and save a head-and-shoulders crop of each (face_01.png, ...).

    Returns [{"index", "crop_path", "box"}] left to right, box = (x0, y0, x1, y1) of the face
    normalized to the photo. Raises NoFaceFound when there is none."""
    settings = {**DEFAULT_SETTINGS, **(settings or {})}
    out_dir.mkdir(parents=True, exist_ok=True)
    emit(FaceStageEvent("detect_faces", "Find faces", "running"))
    # Detection runs on a larger copy than the line art so small faces in group photos survive.
    photo = load_photo(Path(image_path), max(2048, int(settings["max_side_px"])))
    try:
        boxes, backend = detect_faces(photo, settings)
    except Exception as exc:
        emit(FaceStageEvent("detect_faces", "Find faces", "failed", detail=str(exc)))
        raise
    if not boxes:
        emit(FaceStageEvent("detect_faces", "Find faces", "failed", detail="no face found"))
        raise NoFaceFound("No face found in the photo")
    height, width = photo.shape[:2]
    faces = []
    for index, box in enumerate(boxes, start=1):
        crop, (cx0, cy0, cx1, cy1) = crop_portrait(photo, box, settings)
        crop_path = out_dir / f"face_{index:02d}.png"
        Image.fromarray(crop).save(crop_path)
        # Neighbours inside this crop, normalized to it, so their hair and shoulders are left out.
        others = [
            [round((o[0] - cx0) / (cx1 - cx0), 4), round((o[1] - cy0) / (cy1 - cy0), 4),
             round((o[2] - cx0) / (cx1 - cx0), 4), round((o[3] - cy0) / (cy1 - cy0), 4)]
            for o in boxes if o is not box and o[2] > cx0 and o[0] < cx1 and o[3] > cy0 and o[1] < cy1
        ]
        faces.append({
            "index": index,
            "crop_path": crop_path,
            "box": [round(box[0] / width, 4), round(box[1] / height, 4), round(box[2] / width, 4), round(box[3] / height, 4)],
            "box_in_crop": [round((box[0] - cx0) / (cx1 - cx0), 4), round((box[1] - cy0) / (cy1 - cy0), 4),
                            round((box[2] - cx0) / (cx1 - cx0), 4), round((box[3] - cy0) / (cy1 - cy0), 4)],
            "others_in_crop": others,
        })
    emit(FaceStageEvent(
        "detect_faces", "Find faces", "done", backend=backend,
        detail=f"{len(faces)} face{'s' if len(faces) != 1 else ''} found",
        artifact_path=faces[0]["crop_path"] if len(faces) == 1 else None,
    ))
    return faces


def generate_face_trajectory(
    *,
    image_path: Path,
    out_dir: Path,
    settings: dict[str, Any] | None = None,
    emit: Callable[[FaceStageEvent], None] = lambda event: None,
    other_faces: list[list[float]] = (),
    target_face: list[float] | None = None,
) -> dict[str, Any]:
    """Photo -> line art PNG, strokes SVG and trajectory JSON in out_dir.

    other_faces / target_face: neighbour and subject face boxes normalized to the photo
    (find_faces "others_in_crop" / "box_in_crop").

    Emits stage events ("face_landmarks", "extract") like the animal pipeline,
    and returns artifact paths plus stroke statistics.
    """
    settings = {**DEFAULT_SETTINGS, **(settings or {})}
    out_dir.mkdir(parents=True, exist_ok=True)
    photo = load_photo(Path(image_path), int(settings["max_side_px"]))
    source_path = out_dir / "source.png"
    Image.fromarray(photo).save(source_path)

    emit(FaceStageEvent("face_landmarks", "Find face and hair", "running"))
    try:
        height, width = photo.shape[:2]
        others = [(o[0] * width, o[1] * height, o[2] * width, o[3] * height) for o in other_faces]
        target = (target_face[0] * width, target_face[1] * height, target_face[2] * width, target_face[3] * height) if target_face else None
        art = extract_face_lineart(photo, settings, others, target)
    except Exception as exc:
        emit(FaceStageEvent("face_landmarks", "Find face and hair", "failed", detail=str(exc)))
        raise
    emit(FaceStageEvent(
        "face_landmarks", "Find face and hair", "done", backend=art.backend,
        detail=f"{len(art.strokes)} strokes ({', '.join(sorted(set(art.labels)))})",
    ))

    emit(FaceStageEvent("extract", "Build pen strokes", "running"))
    trajectory = to_trajectory(art, float(settings["margin"]))
    trajectory_path = out_dir / "trajectory.json"
    trajectory_path.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
    svg_path = out_dir / "strokes.svg"
    write_svg(
        svg_path,
        [[(p["x"], p["y"]) for p in stroke["points"]] for stroke in trajectory["strokes"]],
        1024, 1024, stroke_width=3, stroke_color="#000000",
    )
    lineart_path = out_dir / "lineart.png"
    render_lineart(trajectory).save(lineart_path)
    stats = {
        "stroke_count": trajectory["stroke_count"],
        "point_count": trajectory["point_count"],
        "est_draw_seconds": round(estimate_draw_seconds(trajectory), 1),
    }
    emit(FaceStageEvent(
        "extract", "Build pen strokes", "done", backend="opencv",
        detail=f"{stats['stroke_count']} strokes, {stats['point_count']} points, ~{stats['est_draw_seconds']:.0f} s to draw",
        artifact_path=lineart_path,
    ))
    return {
        "source_path": source_path,
        "lineart_path": lineart_path,
        "svg_path": svg_path,
        "trajectory_path": trajectory_path,
        "stats": stats,
        "backend": art.backend,
    }
