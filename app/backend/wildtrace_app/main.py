"""FastAPI app for WildTrace Studio (contract: app/API.md).

Run: uv run --extra app uvicorn wildtrace_app.main:app --app-dir app/backend --port 8000
"""
from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from . import settings

if str(settings.REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(settings.REPO_ROOT))

from . import catalog, chat, health, jobs, runs  # noqa: E402
from .store import bus, store  # noqa: E402

_sessions: dict[str, list[dict[str, str]]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    bus.bind(asyncio.get_running_loop())
    ticker = asyncio.create_task(_health_ticker())
    yield
    ticker.cancel()
    runs.stop_all()


async def _health_ticker() -> None:
    while True:
        await asyncio.sleep(10)
        bus.publish({"type": "health", "health": await asyncio.to_thread(health.snapshot)})


app = FastAPI(title="WildTrace Studio", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])


def _job_or_404(job_id: str) -> dict[str, Any]:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"No job {job_id}")
    return job


def _run_or_404(run_id: str) -> dict[str, Any]:
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(404, f"No run {run_id}")
    return run


@app.get("/api/health")
def get_health() -> dict[str, Any]:
    return health.snapshot()


# Chat ---------------------------------------------------------------------------------------

class ChatIn(BaseModel):
    message: str
    session_id: str | None = None


@app.post("/api/chat")
def post_chat(body: ChatIn) -> dict[str, Any]:
    history = _sessions.setdefault(body.session_id or "default", [])
    waiting = next((j for j in store.list_jobs() if j["status"] == "awaiting_input"), None)
    if waiting is not None:
        index = chat.parse_face_choice(body.message, len(waiting.get("choices") or []))
        if index is not None:
            job = jobs.choose_face(waiting["id"], index)
            reply = f"Drawing face {index} of {len(waiting['choices'])}. The line art is on its way."
            history += [{"role": "user", "content": body.message}, {"role": "assistant", "content": reply}]
            return {"reply": reply, "job": job, "suggestions": ["Draw it", "Draw it with IK"]}
    ready = store.latest_ready_job()
    intent = chat.parse(body.message, history, ready is not None)
    out: dict[str, Any] = {"reply": intent.reply or ""}
    if intent.action == "draw":
        if ready is None:
            out.update(reply="Nothing is ready to draw yet. Ask for an animal or attach a photo first.",
                       suggestions=chat.SUGGESTIONS)
        else:
            run = runs.start_run(ready, intent.controller or settings.DEFAULT_CONTROLLER, None)
            out["run"] = run
            out["reply"] = intent.reply or (f"Drawing {ready['title'].lower()} with {run['controller'].upper()} "
                                            f"({'live in Isaac Sim' if run['mode'] == 'live' else 'replay'}).")
    elif intent.action in ("pick", "generate"):
        job = jobs.job_for_intent(intent, body.message)
        out["job"] = job
        if not intent.reply:
            out["reply"] = (f"Picking a {intent.category.lower()} drawing." if intent.action == "pick" else
                            f"Making a new {(intent.hint + ' ') if intent.hint else ''}{intent.category.lower()} "
                            "from a fresh photo. Follow the stages on the right.")
        out["suggestions"] = ["Draw it", "Draw it with IK", "Pick another"]
    else:
        out["reply"] = intent.reply or ("I can draw a Bird, Cat, Dog, Fish, Frog or Horse, make a new one from a fresh "
                                        "photo, or turn your face photo into line art.")
        out["suggestions"] = chat.SUGGESTIONS
    history += [{"role": "user", "content": body.message}, {"role": "assistant", "content": out["reply"]}]
    del history[:-20]
    return out


# Jobs ---------------------------------------------------------------------------------------

class TextJobIn(BaseModel):
    prompt: str


@app.post("/api/jobs/text")
def post_text_job(body: TextJobIn) -> dict[str, Any]:
    job = jobs.text_job(body.prompt)
    if job is None:
        raise HTTPException(422, "Not a drawing request. Try 'draw a cat' or 'make a new sitting dog'.")
    return job


@app.post("/api/jobs/image")
async def post_image_job(file: UploadFile = File(...), kind: str = Form("auto"), category: str | None = Form(None)) -> dict[str, Any]:
    data = await file.read()
    if not data:
        raise HTTPException(422, "Empty upload")
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(413, "Photo is larger than 25 MB")
    path = jobs.save_upload(data, file.filename or "upload.jpg")
    category = chat.ANIMALS.get((category or "").strip().lower()) if category else None
    if kind == "face":
        return jobs.face_job(path)
    if kind == "animal":
        if not category:
            raise HTTPException(422, "Say which animal it is (Bird, Cat, Dog, Fish, Frog or Horse)")
        return jobs.animal_job(category, image_path=path)
    return jobs.auto_image_job(path, category)


@app.get("/api/jobs")
def get_jobs() -> list[dict[str, Any]]:
    return store.list_jobs()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return _job_or_404(job_id)


class DrawIn(BaseModel):
    controller: str | None = None
    mode: str | None = None


@app.post("/api/jobs/{job_id}/draw")
def post_draw(job_id: str, body: DrawIn | None = None) -> dict[str, Any]:
    job = _job_or_404(job_id)
    if job["status"] != "ready":
        raise HTTPException(409, f"Job is {job['status']}, not ready to draw")
    body = body or DrawIn()
    controller = body.controller if body.controller in ("act", "ik") else settings.DEFAULT_CONTROLLER
    mode = body.mode if body.mode in ("live", "replay") else None
    return runs.start_run(job, controller, mode)


class ChooseIn(BaseModel):
    index: int


@app.post("/api/jobs/{job_id}/choose")
def post_choose(job_id: str, body: ChooseIn) -> dict[str, Any]:
    _job_or_404(job_id)
    try:
        return jobs.choose_face(job_id, body.index)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None


# Runs ---------------------------------------------------------------------------------------

@app.get("/api/runs")
def get_runs(active: bool = False, limit: int = 20) -> list[dict[str, Any]]:
    runs_ = [r for r in store.list_runs() if not active or r["status"] in ("starting", "running")]
    return runs_[: max(1, min(limit, 200))]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    return _run_or_404(run_id)


@app.get("/api/runs/{run_id}/trace")
def get_trace(run_id: str) -> dict[str, Any]:
    return runs.trace(_run_or_404(run_id))


@app.get("/api/runs/{run_id}/stream/{camera}")
async def get_stream(run_id: str, camera: str) -> StreamingResponse:
    run = _run_or_404(run_id)
    if camera not in run.get("cameras", []):
        raise HTTPException(404, f"Run has no camera {camera}")

    async def frames():
        last = None
        while True:
            frame, done = await asyncio.to_thread(runs.latest_frame, run_id, camera)
            if frame is not None and frame != last:
                last = frame
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode()
                       + b"\r\n\r\n" + frame + b"\r\n")
            if done:
                return
            await asyncio.sleep(0.1)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})


# Catalog and files --------------------------------------------------------------------------

@app.get("/api/catalog")
def get_catalog(category: str | None = None, limit: int = 48) -> list[dict[str, Any]]:
    return catalog.listing(category, max(1, min(limit, 500)))


@app.get("/api/catalog/{sample_id}/svg")
def get_catalog_svg(sample_id: str) -> Response:
    item = catalog.find(sample_id)
    if item is None:
        raise HTTPException(404, f"No catalog item {sample_id}")
    return Response(catalog.svg_for(Path(item["path"])), media_type="image/svg+xml",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/catalog/{sample_id}/trajectory")
def get_catalog_trajectory(sample_id: str) -> FileResponse:
    item = catalog.find(sample_id)
    if item is None:
        raise HTTPException(404, f"No catalog item {sample_id}")
    return FileResponse(item["path"], media_type="application/json")


_SERVED_ROOTS = ("outputs/", "isaac-sim/data/", "isaac-sim/inputs/")


@app.get("/files/{path:path}")
def get_file(path: str) -> FileResponse:
    if path.startswith("studio/"):
        root = settings.STUDIO_DIR.resolve()
        target = (root / path.removeprefix("studio/")).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise HTTPException(404)
        return FileResponse(target)
    target = (settings.REPO_ROOT / path).resolve()
    try:
        relative = target.relative_to(settings.REPO_ROOT).as_posix()
    except ValueError:
        raise HTTPException(404) from None
    if not relative.startswith(_SERVED_ROOTS) or not target.is_file():
        raise HTTPException(404)
    return FileResponse(target)


# WebSocket ----------------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket(ws: WebSocket) -> None:
    await ws.accept()
    queue = bus.subscribe()
    try:
        await ws.send_json({"type": "health", "health": await asyncio.to_thread(health.snapshot)})
        while True:
            await ws.send_json(await queue.get())
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        bus.unsubscribe(queue)
