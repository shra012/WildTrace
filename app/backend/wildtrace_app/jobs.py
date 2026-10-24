"""Job runners: text or photo in, robot-drawable trajectory out, using the pipeline stages.

pick   - an existing trajectory from the catalog (no model work).
face   - wildtrace.face_lineart (MediaPipe landmarks + hair outline).
animal - wildtrace.studio_pipeline (silhouette -> outline -> validate -> extract), the
         same stages as the data pipeline with FLUX-schnell or the OpenRouter stand-in.
Jobs run in a small thread pool and report every stage through the store.
"""
from __future__ import annotations

import json
import re
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from . import catalog, settings
from .store import bus, file_path, file_url, store

_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="job")

STAGES = {
    "pick": [("understand", "Understand request"), ("pick", "Pick a drawing")],
    "face": [("detect_faces", "Find faces"), ("choose_face", "Choose a face"),
             ("face_landmarks", "Find face and hair"), ("extract", "Build pen strokes")],
    "animal": [("segment", "Isolate the subject"), ("outline", "Generate line art"),
               ("validate", "Validate the drawing"), ("extract", "Build pen strokes")],
}


def _stages(kind: str, extra: list[tuple[str, str]] = ()) -> list[dict[str, Any]]:
    return [{"name": n, "label": l, "status": "pending"} for n, l in [*extra, *STAGES[kind]]]


def _on_event(job_id: str):
    """Adapter from the pipelines' StageEvent / FaceStageEvent to store stages."""
    def emit(event) -> None:
        store.update_stage(job_id, event.name, label=event.label, status=event.status, backend=event.backend,
                           detail=event.detail, artifact_url=file_url(event.artifact_path))
        if event.status == "failed":
            bus.log(f"{event.label} failed: {event.detail}", "warn")
    return emit


def _finish(job_id: str, result: dict[str, Any]) -> None:
    store.update_job(
        job_id,
        status="ready",
        artifacts={
            "source_image_url": file_url(result.get("source_path")),
            "lineart_url": file_url(result.get("lineart_path")),
            "svg_url": file_url(result.get("svg_path")),
            "trajectory_url": file_url(result.get("trajectory_path")),
        },
        trajectory=result["stats"],
    )
    job = store.get_job(job_id)
    bus.log(f"{job['title']} is ready: {result['stats']['stroke_count']} strokes")


def _fail(job_id: str, exc: BaseException) -> None:
    job = store.get_job(job_id) or {"stages": []}
    for stage in job["stages"]:
        if stage["status"] == "running":
            store.update_stage(job_id, stage["name"], status="failed", detail=str(exc)[:200])
    store.update_job(job_id, status="failed", error=str(exc) or type(exc).__name__)
    bus.log(f"Job failed: {exc}", "error")


def _submit(job_id: str, work) -> None:
    def run() -> None:
        store.update_job(job_id, status="running")
        try:
            result = work()
            if result is not None:  # None: the job paused for the user (awaiting_input)
                _finish(job_id, result)
        except Exception as exc:  # reported to the UI; the traceback goes to the server log
            traceback.print_exc()
            _fail(job_id, exc)
    _pool.submit(run)


# Pick ---------------------------------------------------------------------------------------

def render_lineart(trajectory: dict[str, Any], size: int = 1024, thickness: int = 3):
    """Black strokes on white, keeping the trajectory's aspect ratio."""
    import cv2
    import numpy as np
    from PIL import Image

    width, height = catalog.canvas_size(trajectory, size - 32)
    canvas = np.full((height + 32, width + 32), 255, dtype=np.uint8)
    for stroke in trajectory["strokes"]:
        points = np.asarray([[16 + p["x"] * (width - 1), 16 + p["y"] * (height - 1)] for p in stroke["points"]], dtype=np.int32)
        cv2.polylines(canvas, [points.reshape(-1, 1, 2)], False, 0, thickness, lineType=cv2.LINE_AA)
    return Image.fromarray(canvas)


def _materialize_pick(job_id: str, item: dict[str, Any]) -> dict[str, Any]:
    from wildtrace.face_lineart import estimate_draw_seconds
    from wildtrace.images import write_svg

    out = store.job_dir(job_id)
    trajectory = json.loads(Path(item["path"]).read_text(encoding="utf-8"))
    trajectory_path = out / "trajectory.json"
    shutil.copyfile(item["path"], trajectory_path)
    svg_path = out / "strokes.svg"
    width, height = catalog.canvas_size(trajectory, 1024)
    write_svg(svg_path, [[(p["x"], p["y"]) for p in s["points"]] for s in trajectory["strokes"]],
              width, height, stroke_width=3, stroke_color="#000000")
    lineart_path = out / "lineart.png"
    render_lineart(trajectory).save(lineart_path)
    return {
        "source_path": catalog.source_image(item),
        "lineart_path": lineart_path,
        "svg_path": svg_path,
        "trajectory_path": trajectory_path,
        "stats": {
            "stroke_count": int(trajectory.get("stroke_count", len(trajectory["strokes"]))),
            "point_count": int(trajectory.get("point_count", sum(len(s["points"]) for s in trajectory["strokes"]))),
            "est_draw_seconds": round(estimate_draw_seconds(trajectory), 1),
        },
    }


def pick_job(category: str, hint: str | None, text: str, sample_id: str | None = None,
             understood_by: str = "rules") -> dict[str, Any]:
    item = catalog.find(sample_id) if sample_id else None
    category = item["category"] if item else category
    title = f"{(hint + ' ') if hint else ''}{category.lower()}".strip().capitalize()
    job = store.create_job("pick", title, {"text": text, "category": category}, _stages("pick"))

    def work() -> dict[str, Any]:
        store.update_stage(job["id"], "understand", status="done", backend=understood_by,
                           detail=f"{category}{' - ' + hint if hint else ''}")
        store.update_stage(job["id"], "pick", status="running")
        used = {j["request"].get("sample_id") for j in store.list_jobs() if j["kind"] == "pick"}
        chosen = item or catalog.pick(category, hint, avoid=used)
        if chosen is None:
            raise RuntimeError(f"No {category} drawings in the catalog")
        store.update_job(job["id"], request={**job["request"], "sample_id": chosen["sample_id"]})
        result = _materialize_pick(job["id"], chosen)
        detail = ", ".join(x for x in (chosen.get("subcategory"), chosen.get("angle"), chosen["source"]) if x)
        store.update_stage(job["id"], "pick", status="done", backend="catalog", detail=f"{chosen['sample_id']} ({detail})",
                           artifact_url=file_url(result["lineart_path"]))
        return result

    _submit(job["id"], work)
    return job


# Face ---------------------------------------------------------------------------------------

def _face_choices(faces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"index": f["index"], "image_url": file_url(f["crop_path"]), "box": f["box"],
             "box_in_crop": f["box_in_crop"], "others_in_crop": f["others_in_crop"]} for f in faces]


def _detect_work(job_id: str, image_path: Path):
    """Find the faces; with one, draw it straight away, with several, wait for the user's pick."""
    def work() -> dict[str, Any] | None:
        from wildtrace.face_lineart import find_faces

        faces = find_faces(image_path=image_path, out_dir=store.job_dir(job_id) / "faces", emit=_on_event(job_id))
        choices = _face_choices(faces)
        if len(choices) == 1:
            store.update_job(job_id, choices=choices, title="Your portrait")
            store.update_stage(job_id, "choose_face", status="skipped", detail="one face")
            return _draw_face(job_id, choices[0])
        store.update_stage(job_id, "choose_face", status="running", detail=f"{len(choices)} faces found - pick one")
        store.update_job(job_id, status="awaiting_input", choices=choices, title=f"Group photo ({len(choices)} faces)")
        bus.log(f"Found {len(choices)} faces. Pick the one to draw.")
        return None
    return work


def _draw_face(job_id: str, choice: dict[str, Any]) -> dict[str, Any]:
    from wildtrace.face_lineart import generate_face_trajectory

    return generate_face_trajectory(
        image_path=file_path(choice["image_url"]), out_dir=store.job_dir(job_id), emit=_on_event(job_id),
        other_faces=choice["others_in_crop"], target_face=choice["box_in_crop"],
    )


def face_job(image_path: Path, title: str = "Your portrait") -> dict[str, Any]:
    job = store.create_job("face", title, {"image_url": file_url(image_path)}, _stages("face"))
    _submit(job["id"], _detect_work(job["id"], image_path))
    return job


def choose_face(job_id: str, index: int) -> dict[str, Any]:
    """Continue a face job with the face the user picked (1-based, left to right)."""
    job = store.get_job(job_id)
    if job is None or job["status"] != "awaiting_input":
        raise ValueError("This job is not waiting for a face choice")
    choice = next((c for c in job.get("choices") or [] if c["index"] == index), None)
    if choice is None:
        raise ValueError(f"No face {index}; pick 1 to {len(job.get('choices') or [])}")
    total = len(job["choices"])
    store.update_stage(job_id, "choose_face", status="done", backend="you", detail=f"face {index} of {total}",
                       artifact_url=choice["image_url"])
    job = store.update_job(job_id, status="queued", title=f"Portrait {index} of {total}",
                           request={**job["request"], "face_index": index})
    _submit(job_id, lambda: _draw_face(job_id, choice))
    return job


# Animal -------------------------------------------------------------------------------------

def _animal_work(job_id: str, category: str, image_path: Path | None, hint: str | None):
    def work() -> dict[str, Any]:
        from wildtrace import studio_pipeline as sp

        emit = _on_event(job_id)
        sample = None
        if image_path is None:
            emit(sp.StageEvent("pick_photo", "Pick an unused photo", "running"))
            sample = sp.pick_unused_photo(category, settings.REPO_ROOT, hint=hint)
            emit(sp.StageEvent("pick_photo", "Pick an unused photo", "done", backend="open images",
                               detail=str(sample.get("sample_id", "")), artifact_path=sample.get("resolved_image_path")))
        backend = sp.select_image_backend(settings.REPO_ROOT, settings.CONFIG_DIR)
        return sp.generate_animal_trajectory(
            out_dir=store.job_dir(job_id), repo_root=settings.REPO_ROOT, config_dir=settings.CONFIG_DIR,
            category=category, image_path=image_path if sample is None else None,
            mask_path=None, sample=sample, emit=emit, backend=backend,
        )
    return work


def animal_job(category: str, hint: str | None = None, image_path: Path | None = None, text: str | None = None) -> dict[str, Any]:
    extra = [] if image_path else [("pick_photo", "Pick an unused photo")]
    title = f"New {(hint + ' ') if hint else ''}{category.lower()}"
    request = {"category": category, "text": text, "image_url": file_url(image_path) if image_path else None}
    job = store.create_job("animal", title, {k: v for k, v in request.items() if v}, _stages("animal", extra))
    _submit(job["id"], _animal_work(job["id"], category, image_path, hint))
    return job


def auto_image_job(image_path: Path, category: str | None) -> dict[str, Any]:
    """A face if MediaPipe finds one, otherwise an animal of `category`."""
    if category:
        return animal_job(category, image_path=image_path)
    job = store.create_job("face", "Your photo", {"image_url": file_url(image_path)}, _stages("face"))
    detect = _detect_work(job["id"], image_path)

    def work() -> dict[str, Any] | None:
        from wildtrace.face_lineart import NoFaceFound
        try:
            return detect()
        except NoFaceFound:
            raise RuntimeError("No face found in this photo. For an animal, pick Animal and say which one "
                               "(Bird, Cat, Dog, Fish, Frog or Horse).") from None

    _submit(job["id"], work)
    return job


# Text ---------------------------------------------------------------------------------------

_SAMPLE_ID = re.compile(r"\b([A-Za-z0-9_-]{12,})\b")


def text_job(prompt: str) -> dict[str, Any] | None:
    """A job for a text request, or None when the text is not a drawing request."""
    from . import chat

    # Gallery clicks send "draw <category> <sample_id>".
    for token in _SAMPLE_ID.findall(prompt):
        item = catalog.find(token)
        if item:
            return pick_job(item["category"], None, prompt, sample_id=token, understood_by="gallery")
    intent = chat.parse(prompt, [], store.latest_ready_job() is not None)
    return job_for_intent(intent, prompt)


def job_for_intent(intent, text: str) -> dict[str, Any] | None:
    if intent.action == "pick" and intent.category:
        return pick_job(intent.category, intent.hint, text, understood_by=intent.backend)
    if intent.action == "generate" and intent.category:
        return animal_job(intent.category, hint=intent.hint, text=text)
    return None


def save_upload(data: bytes, filename: str) -> Path:
    suffix = Path(filename or "upload.jpg").suffix.lower()
    suffix = suffix if suffix in (".jpg", ".jpeg", ".png", ".webp", ".bmp") else ".jpg"
    directory = settings.STUDIO_DIR / "uploads"
    directory.mkdir(parents=True, exist_ok=True)
    from .store import new_id
    path = directory / f"{new_id('img')}{suffix}"
    path.write_bytes(data)
    return path
