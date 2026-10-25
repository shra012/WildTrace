"""Jobs and runs (app/API.md types), persisted as JSON, with an event bus for the WebSocket.

Worker threads update jobs and runs; every change is published to all
connected WebSocket clients through asyncio queues on the server loop.
"""
from __future__ import annotations

import asyncio
import json
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import settings


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def file_url(path: Path | str | None) -> str | None:
    """URL under /files for an artifact in the studio dir or the repo (None for anything else)."""
    if not path:
        return None
    resolved = Path(path).resolve()
    for root, prefix in ((settings.STUDIO_DIR.resolve(), "studio/"), (settings.REPO_ROOT, "")):
        try:
            return f"/files/{prefix}{resolved.relative_to(root).as_posix()}"
        except ValueError:
            continue
    return None


def file_path(url: str | None) -> Path | None:
    """Inverse of file_url."""
    if not url or not url.startswith("/files/"):
        return None
    relative = url.removeprefix("/files/")
    if relative.startswith("studio/"):
        return settings.STUDIO_DIR / relative.removeprefix("studio/")
    return settings.REPO_ROOT / relative


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    def publish(self, event: dict[str, Any]) -> None:
        """Thread-safe: callable from worker threads."""
        if self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._fan_out, event)
        except RuntimeError:
            pass  # the server is shutting down

    def _fan_out(self, event: dict[str, Any]) -> None:
        for queue in list(self._subscribers):
            if queue.full():
                try:
                    queue.get_nowait()  # drop the oldest rather than block a slow client
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(event)

    def log(self, message: str, level: str = "info") -> None:
        self.publish({"type": "log", "level": level, "message": message, "at": now()})


bus = EventBus()


class Store:
    """In-memory jobs and runs, written through to outputs/studio/{jobs,runs}/<id>/<kind>.json."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.runs: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        for kind, directory, table in (("job", settings.JOBS_DIR, self.jobs), ("run", settings.RUNS_DIR, self.runs)):
            for path in directory.glob(f"*/{kind}.json"):
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                # Anything still in flight when the server stopped did not finish.
                if record.get("status") in ("queued", "running", "starting"):
                    record["status"] = "failed"
                    record["error"] = record.get("error") or "Interrupted by a server restart"
                table[record["id"]] = record

    def _save(self, kind: str, record: dict[str, Any]) -> None:
        directory = (settings.JOBS_DIR if kind == "job" else settings.RUNS_DIR) / record["id"]
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / f"{kind}.json.tmp"
        temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
        temporary.replace(directory / f"{kind}.json")

    # Jobs ---------------------------------------------------------------------------------
    def create_job(self, kind: str, title: str, request: dict[str, Any], stages: list[dict[str, Any]]) -> dict[str, Any]:
        job = {
            "id": new_id("job"),
            "kind": kind,
            "status": "queued",
            "title": title,
            "request": request,
            "stages": stages,
            "artifacts": {},
            "trajectory": None,
            "error": None,
            "created_at": now(),
            "updated_at": now(),
        }
        with self._lock:
            self.jobs[job["id"]] = job
            self._save("job", job)
        bus.publish({"type": "job", "job": deepcopy(job)})
        return deepcopy(job)

    def job_dir(self, job_id: str) -> Path:
        directory = settings.JOBS_DIR / job_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def update_job(self, job_id: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            job = self.jobs[job_id]
            job.update(fields)
            job["updated_at"] = now()
            self._save("job", job)
            snapshot = deepcopy(job)
        bus.publish({"type": "job", "job": snapshot})
        return snapshot

    def update_stage(self, job_id: str, name: str, **fields: Any) -> None:
        with self._lock:
            job = self.jobs[job_id]
            stage = next((s for s in job["stages"] if s["name"] == name), None)
            if stage is None:
                stage = {"name": name, "label": fields.get("label", name), "status": "pending"}
                job["stages"].append(stage)
            if fields.get("status") == "running" and not stage.get("started_at"):
                stage["started_at"] = now()
            if fields.get("status") in ("done", "failed", "skipped"):
                stage["finished_at"] = now()
            stage.update({k: v for k, v in fields.items() if v is not None})
        self.update_job(job_id)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self.jobs.get(job_id)
            return deepcopy(job) if job else None

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return sorted((deepcopy(j) for j in self.jobs.values()), key=lambda j: j["created_at"], reverse=True)

    def latest_ready_job(self) -> dict[str, Any] | None:
        return next((j for j in self.list_jobs() if j["status"] == "ready"), None)

    # Runs ---------------------------------------------------------------------------------
    def create_run(self, job_id: str, controller: str, mode: str) -> dict[str, Any]:
        run = {
            "id": new_id("run"),
            "job_id": job_id,
            "controller": controller,
            "mode": mode,
            "status": "starting",
            "phase": None,
            "progress": 0.0,
            "metrics": None,
            "cameras": [],
            "error": None,
            "started_at": now(),
            "finished_at": None,
        }
        with self._lock:
            self.runs[run["id"]] = run
            self._save("run", run)
        bus.publish({"type": "run", "run": deepcopy(run)})
        return deepcopy(run)

    def run_dir(self, run_id: str) -> Path:
        directory = settings.RUNS_DIR / run_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def update_run(self, run_id: str, publish: bool = True, **fields: Any) -> dict[str, Any]:
        with self._lock:
            run = self.runs[run_id]
            run.update(fields)
            if fields.get("status") in ("finished", "failed") and not run.get("finished_at"):
                run["finished_at"] = now()
            self._save("run", run)
            snapshot = deepcopy(run)
        if publish:
            bus.publish({"type": "run", "run": snapshot})
        return snapshot

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            return sorted((deepcopy(r) for r in self.runs.values()), key=lambda r: r["started_at"], reverse=True)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            run = self.runs.get(run_id)
            return deepcopy(run) if run else None


store = Store()
