from __future__ import annotations

import copy
import json
from unittest.mock import patch

import pytest
from PIL import Image

from wildtrace.face_fidelity import FEATURES, LOCAL_MODEL, _checklist, judge_face_fidelity


@pytest.fixture
def checklist():
    return {"features": {name: {"photo": "absent", "drawing": "absent"} for name in FEATURES},
            "hair_length": {"photo": "short", "drawing": "short"},
            "expression": {"photo": "smiling", "drawing": "neutral"},
            "resemblance": 3, "missing": ["smile"], "extra": []}


@pytest.mark.parametrize("presence,expected", [("present", True), ("absent", False), ("uncertain", None)])
def test_glasses_setting_uses_only_source_verdict(checklist, presence, expected):
    checklist["features"]["glasses"] = {"photo": presence, "drawing": "present"}
    assert _checklist(checklist)["include_glasses"] is expected


@pytest.mark.parametrize("score", [True, 0, 6, 3.5, "4"])
def test_invalid_resemblance_fails_closed(checklist, score):
    checklist["resemblance"] = score
    with pytest.raises(ValueError, match="Resemblance"):
        _checklist(checklist)


def test_malformed_presence_and_feature_lists_fail_closed(checklist):
    bad = copy.deepcopy(checklist)
    bad["features"]["glasses"]["photo"] = {"present": True}
    with pytest.raises(ValueError, match="glasses"):
        _checklist(bad)
    checklist["missing"] = "not a list"
    with pytest.raises(ValueError, match="missing"):
        _checklist(checklist)


@pytest.mark.parametrize("photo,drawing,field", [("present", "absent", "missing"), ("absent", "present", "extra")])
def test_feature_lists_include_explicit_presence_disagreements(checklist, photo, drawing, field):
    checklist["missing"] = []
    checklist["features"]["glasses"] = {"photo": photo, "drawing": drawing}
    result = _checklist(checklist)
    assert "glasses" in result[field]
    assert result["model_" + field] == []


def test_hosted_private_source_is_rejected_before_network():
    with patch("wildtrace.face_fidelity.OpenRouterSemanticValidator") as remote:
        with pytest.raises(ValueError, match="Open Images"):
            judge_face_fidelity("private-source.png", "drawing.png", backend="openrouter")
        remote.assert_not_called()


def test_remote_ollama_host_is_rejected_before_reading_photos():
    with pytest.raises(ValueError, match="this machine"):
        judge_face_fidelity("private.png", "art.png", ollama_host="http://192.0.2.1:11434")


def test_default_is_local_and_does_not_follow_remote_ollama_environment(tmp_path, checklist, monkeypatch):
    photo, art = tmp_path / "photo.png", tmp_path / "art.png"
    for path in (photo, art):
        Image.new("RGB", (8, 8), "white").save(path)
    monkeypatch.setenv("OLLAMA_HOST", "https://example.com")
    with patch("wildtrace.face_fidelity.OllamaSemanticValidator") as local, \
            patch("wildtrace.face_fidelity.OpenRouterSemanticValidator") as remote:
        local.return_value._request_with_images.return_value = {"response": json.dumps(checklist)}
        result = judge_face_fidelity(photo, art)
    settings = local.call_args.args[0]
    assert settings["ollama_model_name"] == LOCAL_MODEL
    assert settings["host"] == "http://127.0.0.1:11434"
    assert settings["think"] is False
    assert settings["format"]["required"] == ["features", "hair_length", "expression", "resemblance", "missing", "extra"]
    assert result["provenance"]["backend"] == "ollama"
    assert len(result["provenance"]["source_sha256"]) == 64
    remote.assert_not_called()
