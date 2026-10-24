"""WildTrace Studio backend (app/backend): chat routing, pick jobs, replay runs, files. No network, no GPU."""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
REPO_ROOT = Path(__file__).resolve().parents[1]
GOLD_CAT = "1163bceaf6134cf5"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("WILDTRACE_STUDIO_DIR", str(tmp_path / "studio"))
    monkeypatch.setenv("WILDTRACE_REPLAY_SECONDS", "2")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")  # rule-based chat only
    sys.path.insert(0, str(REPO_ROOT / "app" / "backend"))
    for name in [m for m in sys.modules if m.startswith("wildtrace_app")]:
        del sys.modules[name]
    main = importlib.import_module("wildtrace_app.main")
    health = importlib.import_module("wildtrace_app.health")
    # Keep the tests away from nvidia-smi and the Windows relay.
    monkeypatch.setattr(health, "gpu", lambda: {"busy": True, "utilization": 0.0, "note": "test"})
    monkeypatch.setattr(health, "isaac", lambda: {"running": False, "scene_ready": False})
    monkeypatch.setattr(health, "image_backend", lambda: "openrouter")
    from fastapi.testclient import TestClient
    with TestClient(main.app) as test_client:
        yield test_client


def _wait(client, url, done=("ready", "failed", "finished"), timeout_s=60.0):
    deadline = time.monotonic() + timeout_s
    while True:
        record = client.get(url).json()
        if record["status"] in done or time.monotonic() > deadline:
            return record
        time.sleep(0.1)


def test_rule_based_chat_intents():
    sys.path.insert(0, str(REPO_ROOT / "app" / "backend"))
    from wildtrace_app import chat

    assert chat.parse_rules("draw a cat", False).action == "pick"
    generate = chat.parse_rules("make a new sitting dog", False)
    assert (generate.action, generate.category, generate.hint) == ("generate", "Dog", "sitting")
    start = chat.parse_rules("draw it with ik", True)
    assert (start.action, start.controller) == ("draw", "ik")
    assert chat.parse_rules("what can you do?", False).action == "chat"


def test_health_reports_backends(client):
    health = client.get("/api/health").json()
    assert health["backends"]["chat"] == "rules"
    assert health["backends"]["face"].startswith("mediapipe")
    assert health["isaac"]["running"] is False


@pytest.mark.skipif(not (REPO_ROOT / "isaac-sim" / "data" / "act_test" / "Cat" / f"{GOLD_CAT}.json").exists(),
                    reason="gold test set not present")
def test_pick_job_then_replay_run(client):
    job = client.post("/api/jobs/text", json={"prompt": f"draw Cat {GOLD_CAT}"}).json()
    job = _wait(client, f"/api/jobs/{job['id']}")
    assert job["status"] == "ready", job["error"]
    assert job["request"]["sample_id"] == GOLD_CAT
    assert [s["status"] for s in job["stages"]] == ["done", "done"]
    for url in job["artifacts"].values():
        if url:
            assert client.get(url).status_code == 200

    run = client.post(f"/api/jobs/{job['id']}/draw", json={"mode": "replay", "controller": "act"}).json()
    assert run["mode"] == "replay"
    run = _wait(client, f"/api/runs/{run['id']}", done=("finished", "failed"))
    assert run["status"] == "finished", run["error"]
    assert run["progress"] == 1.0
    trace = client.get(f"/api/runs/{run['id']}/trace").json()
    assert len(trace["desired"]) == job["trajectory"]["stroke_count"]
    assert trace["executed"] and all(0.0 <= u <= 1.0 and 0.0 <= v <= 1.0 for line in trace["executed"] for u, v in line)


def test_chat_draw_without_a_job_explains(client):
    reply = client.post("/api/chat", json={"message": "draw it"}).json()
    assert "run" not in reply and "Nothing is ready" in reply["reply"]


def test_files_route_stays_inside_served_roots(client):
    assert client.get("/files/pyproject.toml").status_code == 404
    assert client.get("/files/studio/../../etc/passwd").status_code == 404


def test_catalog_lists_gold_first(client):
    items = client.get("/api/catalog?category=Cat&limit=5").json()
    assert items and items[0]["source"] == "gold"
    svg = client.get(items[0]["thumbnail_url"])
    assert svg.status_code == 200 and svg.text.startswith("<svg")


def test_face_choice_parsing():
    sys.path.insert(0, str(REPO_ROOT / "app" / "backend"))
    from wildtrace_app import chat

    assert chat.parse_face_choice("2", 4) == 2
    assert chat.parse_face_choice("the third one please", 4) == 3
    assert chat.parse_face_choice("left", 3) == 1
    assert chat.parse_face_choice("rightmost", 3) == 3
    assert chat.parse_face_choice("middle", 3) == 2
    assert chat.parse_face_choice("7", 3) is None
    assert chat.parse_face_choice("draw a cat", 3) is None


GROUP_PHOTO = REPO_ROOT / "app" / "data" / "faces" / "images.jpg"
FACE_MODELS = Path("~/.cache/wildtrace/mediapipe").expanduser()


@pytest.mark.skipif(not (GROUP_PHOTO.exists() and (FACE_MODELS / "face_landmarker.task").exists()
                         and (FACE_MODELS / "selfie_multiclass_256x256.tflite").exists()),
                    reason="group photo or MediaPipe models not present (no downloads in tests)")
def test_group_photo_waits_for_a_face_choice(client):
    with GROUP_PHOTO.open("rb") as handle:
        job = client.post("/api/jobs/image", files={"file": ("group.jpg", handle, "image/jpeg")}, data={"kind": "auto"}).json()
    job = _wait(client, f"/api/jobs/{job['id']}", done=("awaiting_input", "ready", "failed"))
    assert job["status"] == "awaiting_input", job["error"]
    assert len(job["choices"]) == 4
    assert [c["index"] for c in job["choices"]] == [1, 2, 3, 4]
    assert all(client.get(c["image_url"]).status_code == 200 for c in job["choices"])
    assert client.post(f"/api/jobs/{job['id']}/draw", json={}).status_code == 409

    reply = client.post("/api/chat", json={"message": "the second one"}).json()
    assert reply["job"]["request"]["face_index"] == 2
    job = _wait(client, f"/api/jobs/{job['id']}")
    assert job["status"] == "ready", job["error"]
    stages = {s["name"]: s["status"] for s in job["stages"]}
    assert stages == {"detect_faces": "done", "choose_face": "done", "face_landmarks": "done", "extract": "done"}
    assert client.post(f"/api/jobs/{job['id']}/choose", json={"index": 1}).status_code == 409
