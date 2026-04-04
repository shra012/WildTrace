from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageFilter, ImageOps


def open_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_longest_side(image: Image.Image, max_side: int) -> Image.Image:
    output = image.copy()
    output.thumbnail((max_side, max_side))
    return output


def save_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def grayscale(image: Image.Image) -> Image.Image:
    return ImageOps.grayscale(image)


def normalize_mask(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    mask = ImageOps.grayscale(mask)
    resized = mask.resize(size, Image.Resampling.NEAREST)
    return resized.point(lambda value: 255 if value > 0 else 0)


def isolate_subject(image: Image.Image, mask: Image.Image) -> Image.Image:
    white = Image.new("RGB", image.size, color="white")
    return Image.composite(image, white, mask)


def crop_to_mask_bbox(image: Image.Image, mask: Image.Image, padding_ratio: float = 0.08) -> tuple[Image.Image, tuple[int, int, int, int]]:
    arr = np.asarray(mask.convert("L"), dtype=np.uint8)
    ys, xs = np.nonzero(arr)
    if len(xs) == 0 or len(ys) == 0:
        return image.copy(), (0, 0, image.width, image.height)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    pad_x = max(1, int((x_max - x_min + 1) * padding_ratio))
    pad_y = max(1, int((y_max - y_min + 1) * padding_ratio))
    left = max(0, x_min - pad_x)
    top = max(0, y_min - pad_y)
    right = min(image.width, x_max + pad_x + 1)
    bottom = min(image.height, y_max + pad_y + 1)
    return image.crop((left, top, right, bottom)), (left, top, right, bottom)


def mask_coverage(mask: Image.Image) -> float:
    arr = np.asarray(mask, dtype=np.uint8)
    return float(np.count_nonzero(arr)) / float(arr.size or 1)


def detect_grayscale(image: Image.Image) -> bool:
    arr = np.asarray(image, dtype=np.int16)
    return bool(np.all(np.abs(arr[:, :, 0] - arr[:, :, 1]) < 2) and np.all(np.abs(arr[:, :, 1] - arr[:, :, 2]) < 2))


def blur_score(image: Image.Image) -> float:
    gray = np.asarray(grayscale(image), dtype=np.float32) / 255.0
    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    return float(np.var(dx) + np.var(dy))


def local_edge_outline(image: Image.Image, threshold_percentile: int) -> Image.Image:
    edge = grayscale(image).filter(ImageFilter.FIND_EDGES)
    arr = np.asarray(edge, dtype=np.uint8)
    threshold = np.percentile(arr, threshold_percentile)
    binary = np.where(arr >= threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary, mode="L")


def connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for y in range(height):
        for x in range(width):
            if mask[y, x] == 0 or visited[y, x]:
                continue
            queue = deque([(x, y)])
            visited[y, x] = True
            points: list[tuple[int, int]] = []
            while queue:
                cx, cy = queue.popleft()
                points.append((cx, cy))
                for dx, dy in offsets:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] != 0 and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((nx, ny))
            components.append(points)
    return components


def border_points(component: Iterable[tuple[int, int]], mask: np.ndarray) -> list[tuple[int, int]]:
    points = set(component)
    height, width = mask.shape
    borders: list[tuple[int, int]] = []
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for x, y in points:
        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= width or ny >= height or mask[ny, nx] == 0:
                borders.append((x, y))
                break
    return borders


def sample_outline_strokes(
    outline: Image.Image,
    max_strokes: int,
    max_points_per_stroke: int,
    min_points_per_stroke: int,
) -> list[list[tuple[float, float]]]:
    arr = np.asarray(outline, dtype=np.uint8)
    binary = np.where(arr < 180, 1, 0).astype(np.uint8)
    components = connected_components(binary)
    components.sort(key=len, reverse=True)
    strokes: list[list[tuple[float, float]]] = []
    height, width = binary.shape
    for component in components[:max_strokes]:
        borders = border_points(component, binary)
        if len(borders) < min_points_per_stroke:
            continue
        pts = np.asarray(borders, dtype=np.float32)
        centroid = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        ordered = pts[np.argsort(angles)]
        if len(ordered) > max_points_per_stroke:
            indices = np.linspace(0, len(ordered) - 1, max_points_per_stroke, dtype=int)
            ordered = ordered[indices]
        stroke = [(float(x) / max(width - 1, 1), float(y) / max(height - 1, 1)) for x, y in ordered]
        strokes.append(stroke)
    return strokes


def write_svg(path: Path, strokes: list[list[tuple[float, float]]], width: int, height: int, stroke_width: int, stroke_color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" fill="none">',
    ]
    for stroke in strokes:
        points = " ".join(f"{x * (width - 1):.2f},{y * (height - 1):.2f}" for x, y in stroke)
        lines.append(
            f'<polyline points="{points}" stroke="{stroke_color}" stroke-width="{stroke_width}" '
            'stroke-linecap="round" stroke-linejoin="round" />'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")
