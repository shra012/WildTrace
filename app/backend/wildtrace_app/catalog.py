"""Index of drawable trajectories for the gallery and for text "draw a <animal>" requests.

Sources, best first: the 48-drawing gold set (isaac-sim/data/act_test), the
full generated set (outputs/trajectories, with breed / view angle metadata
from diagram_trajectories.ndjson), and trajectories made in the studio (faces
and new animals).
"""
from __future__ import annotations

import json
import random
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import settings
from .store import file_path, store

CATEGORIES = ["Bird", "Cat", "Dog", "Fish", "Frog", "Horse", "Face"]
_lock = threading.Lock()


def _stroke_count(path: Path) -> int:
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("stroke_count", 0))
    except (OSError, json.JSONDecodeError):
        return 0


@lru_cache(maxsize=1)
def _base_index() -> list[dict[str, Any]]:
    """Gold and generated trajectories (static for the life of the server)."""
    metadata: dict[str, dict[str, Any]] = {}
    for manifest in (settings.REPO_ROOT / "outputs" / "trajectories" / "diagram_trajectories.ndjson",
                     settings.ISAAC_ROOT / "data" / "act_test" / "diagram_trajectories.ndjson"):
        if not manifest.exists():
            continue
        for line in manifest.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            metadata[row.get("sample_id", "")] = row
    items: dict[str, dict[str, Any]] = {}
    sources = [
        ("gold", settings.ISAAC_ROOT / "data" / "act_test"),
        ("generated", settings.REPO_ROOT / "outputs" / "trajectories"),
    ]
    for source, root in sources:
        for path in sorted(root.glob("*/*.json")):
            sample_id = path.stem
            if sample_id in items:
                continue
            meta = metadata.get(sample_id, {})
            items[sample_id] = {
                "sample_id": sample_id,
                "category": path.parent.name,
                "subcategory": meta.get("subcategory"),
                "angle": meta.get("angle_bucket"),
                "score": meta.get("diagram_validation_score"),
                "source": source,
                "path": str(path),
            }
    return list(items.values())


def _studio_items() -> list[dict[str, Any]]:
    items = []
    for job in store.list_jobs():
        trajectory_url = job["artifacts"].get("trajectory_url")
        if job["status"] != "ready" or not trajectory_url or job["kind"] == "pick":
            continue
        path = file_path(trajectory_url)
        items.append({
            "sample_id": job["id"],
            "category": "Face" if job["kind"] == "face" else (job["request"].get("category") or "Animal"),
            "subcategory": job["title"],
            "angle": None,
            "score": None,
            "source": "studio",
            "path": str(path),
        })
    return items


def all_items() -> list[dict[str, Any]]:
    return _studio_items() + _base_index()


def find(sample_id: str) -> dict[str, Any] | None:
    return next((i for i in all_items() if i["sample_id"] == sample_id), None)


def as_api(item: dict[str, Any]) -> dict[str, Any]:
    path = Path(item["path"])
    with _lock:
        strokes = _stroke_count(path)
    return {
        "sample_id": item["sample_id"],
        "category": item["category"],
        "subcategory": item.get("subcategory"),
        "angle": item.get("angle"),
        "stroke_count": strokes,
        "thumbnail_url": f"/api/catalog/{item['sample_id']}/svg",
        "trajectory_url": f"/api/catalog/{item['sample_id']}/trajectory",
        "source": item["source"],
    }


def listing(category: str | None, limit: int) -> list[dict[str, Any]]:
    items = all_items()
    if category:
        items = [i for i in items if i["category"].lower() == category.lower()]
    # Studio drawings first, then the validated gold set, then the rest.
    order = {"studio": 0, "gold": 1, "generated": 2}
    items.sort(key=lambda i: (order[i["source"]], i["category"], i["sample_id"]))
    return [as_api(i) for i in items[:limit]]


_HINT_WORDS = {
    "left": "left", "right": "right", "front": "front", "profile": "profile", "side": "profile",
    "three quarter": "3q", "3/4": "3q",
}


def pick(category: str, hint: str | None = None, avoid: set[str] | None = None) -> dict[str, Any] | None:
    """A trajectory of `category`, steered by breed / view words in `hint`, preferring the gold set."""
    candidates = [i for i in _base_index() if i["category"].lower() == category.lower()]
    if not candidates:
        return None
    avoid = avoid or set()
    fresh = [i for i in candidates if i["sample_id"] not in avoid] or candidates
    if hint:
        words = re.findall(r"[a-z0-9/]+", hint.lower())
        tokens = {_HINT_WORDS.get(w, w) for w in words}
        def score(item: dict[str, Any]) -> int:
            text = f"{item.get('subcategory') or ''} {item.get('angle') or ''}".lower()
            return sum(1 for t in tokens if t and t in text)
        best = max(score(i) for i in fresh)
        if best > 0:
            fresh = [i for i in fresh if score(i) == best]
    gold = [i for i in fresh if i["source"] == "gold"]
    return random.choice(gold or fresh)


# Recorded lab-arm twin runs of the gold set, used to replay real results when Isaac is not running.
RECORDINGS = {"act": "act_real_v2_stiff", "ik": "ik_real_stiff"}


def source_image(item: dict[str, Any]) -> Path | None:
    """The cropped Open Images photo a generated trajectory came from."""
    path = settings.REPO_ROOT / "outputs" / "silver" / "crops" / item["category"] / f"{item['sample_id']}_crop.png"
    return path if path.exists() else None


def recording(sample_id: str, category: str, controller: str) -> Path | None:
    """Folder with desired_path.csv / executed_path.csv / run_metrics.json from a past twin run."""
    path = settings.REPO_ROOT / "outputs" / "act_eval" / RECORDINGS.get(controller, "") / category / sample_id
    return path if (path / "executed_path.csv").exists() else None


def canvas_size(trajectory: dict[str, Any], size: int) -> tuple[int, int]:
    """Pixel size with the trajectory's aspect ratio (normalized x and y are scaled by width and height)."""
    width = float(trajectory.get("canvas_width") or 1)
    height = float(trajectory.get("canvas_height") or 1)
    scale = size / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def svg_for(path: Path, size: int = 256, stroke: str = "#111827") -> str:
    """Inline SVG preview of a trajectory (normalized canvas, y down)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    width, height = canvas_size(data, size)
    polylines = []
    for item in data["strokes"]:
        points = " ".join(f"{p['x'] * width:.1f},{p['y'] * height:.1f}" for p in item["points"])
        polylines.append(f'<polyline points="{points}" fill="none" stroke="{stroke}" stroke-width="2" '
                         'stroke-linecap="round" stroke-linejoin="round"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="-4 -4 {width + 8} {height + 8}" width="{width}" '
            f'height="{height}"><rect x="-4" y="-4" width="{width + 8}" height="{height + 8}" fill="#ffffff"/>'
            f'{"".join(polylines)}</svg>')
