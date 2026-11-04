"""Opt-in judge-driven glasses redraw in generate_face_trajectory (judge and landmarks mocked, no network)."""
import json

import numpy as np
import pytest
from PIL import Image

from wildtrace import face_fidelity, face_lineart as module


def _art(glasses: bool) -> module.FaceLineArt:
    strokes = [np.array([[10.0, 10.0], [90.0, 12.0], [50.0, 80.0]])]
    labels = ["face_outline"]
    if glasses:
        strokes.append(np.array([[20.0, 30.0], [40.0, 31.0], [60.0, 30.0]]))
        labels.append("glasses")
    return module.FaceLineArt(strokes=strokes, labels=labels, size=(100, 100), backend="fixture")


def _verdict(photo: str, drawing: str = "absent", resemblance: int = 4) -> dict:
    features = {n: {"photo": "absent", "drawing": "absent"} for n in face_fidelity.FEATURES}
    features["glasses"] = {"photo": photo, "drawing": drawing}
    return {"features": features, "resemblance": resemblance, "missing": ["glasses"] if photo == "present" else [],
            "extra": [], "provenance": {"model": "mock"}}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    photo = tmp_path / "p.png"
    Image.fromarray(np.full((100, 100, 3), 200, dtype=np.uint8)).save(photo)
    calls = {"extract": []}

    def extract(img, settings, others=(), target=None):
        calls["extract"].append(bool(settings.get("include_glasses")))
        return _art(bool(settings.get("include_glasses")))

    monkeypatch.setattr(module, "extract_face_lineart", extract)
    return photo, tmp_path / "out", calls, monkeypatch


def _run(setup, settings, judge):
    photo, out, calls, mp = setup
    mp.setattr(face_fidelity, "judge_face_fidelity", judge)
    result = module.generate_face_trajectory(image_path=photo, out_dir=out, settings=settings)
    return result, calls, json.loads((out / "trajectory.json").read_text())


def test_default_never_calls_the_judge(setup):
    def boom(*a, **k):
        raise AssertionError("judge must not run")
    result, calls, traj = _run(setup, None, boom)
    assert result["fidelity"] is None and "fidelity" not in traj and calls["extract"] == [False]


def test_auto_adds_glasses_when_judge_sees_them_missing(setup):
    result, calls, traj = _run(setup, {"fidelity_check": True}, lambda *a, **k: _verdict("present"))
    assert calls["extract"] == [False, True]
    assert "glasses" in traj["stroke_labels"]
    fid = traj["fidelity"]
    assert fid["status"] == "ok" and fid["glasses_added"] and fid["resemblance"] == 4 and fid["missing"] == ["glasses"]
    assert result["fidelity"]["glasses_added"]


@pytest.mark.parametrize("photo", ["absent", "uncertain"])
def test_auto_leaves_drawing_alone_without_glasses(setup, photo):
    result, calls, traj = _run(setup, {"fidelity_check": True}, lambda *a, **k: _verdict(photo))
    assert calls["extract"] == [False] and "glasses" not in traj["stroke_labels"]
    assert traj["fidelity"]["status"] == "ok" and not traj["fidelity"]["glasses_added"]


def test_auto_redraws_when_judge_misreads_the_drawing(setup):
    # The judge claimed glasses in a drawing that has none (seen on a real face): the stroke labels,
    # not the judge, decide what the drawing contains, so glasses are still added.
    _, calls, traj = _run(setup, {"fidelity_check": True}, lambda *a, **k: _verdict("present", "present"))
    assert calls["extract"] == [False, True] and traj["fidelity"]["glasses_added"]
    assert traj["fidelity"]["glasses_drawing_judge"] == "present" and not traj["fidelity"]["glasses_drawing"]


def test_judge_failure_keeps_original_and_records_skip(setup):
    def down(*a, **k):
        raise ConnectionError("refused")
    result, calls, traj = _run(setup, {"fidelity_check": True}, down)
    assert calls["extract"] == [False]
    assert traj["fidelity"]["status"] == "skipped" and "refused" in traj["fidelity"]["reason"]
    assert result["stats"]["stroke_count"] == 1


def test_judge_is_always_local_ollama(setup):
    seen = {}
    def judge(*a, **k):
        seen.update(k)
        return _verdict("absent")
    _run(setup, {"fidelity_check": True}, judge)
    assert seen["backend"] == "ollama" and "127.0.0.1" in seen["ollama_host"]


def test_on_forces_glasses_and_off_never_adds(setup):
    _, calls, traj = _run(setup, {"include_glasses": True}, lambda *a, **k: 1 / 0)
    assert calls["extract"] == [True] and "glasses" in traj["stroke_labels"]
    photo, out, calls2, mp = setup
    calls2["extract"].clear()
    mp.setattr(face_fidelity, "judge_face_fidelity", lambda *a, **k: 1 / 0)
    module.generate_face_trajectory(image_path=photo, out_dir=out, settings={"include_glasses": False})
    assert calls2["extract"] == [False]
