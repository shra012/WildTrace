from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from wildtrace.diagram import OllamaSemanticValidator, OpenRouterSemanticValidator


def reply(payload):
    return io.BytesIO(json.dumps(payload).encode())


def test_local_request_passes_explicit_json_schema():
    schema = {"type": "object", "properties": {"resemblance": {"type": "integer"}}}
    judge = OllamaSemanticValidator({"format": schema, "think": False})
    with patch("wildtrace.diagram.request.urlopen", return_value=reply({"response": "{}"})) as call:
        judge._request_with_images([], "test")
    body = json.loads(call.call_args.args[0].data)
    assert body["format"] == schema
    assert body["think"] is False


def test_rate_limit_retries_same_model_and_honours_retry_after():
    judge = OpenRouterSemanticValidator({"openrouter_model_name": "qwen/qwen3-vl-8b-instruct", "openrouter_max_retries": 2})
    limited = HTTPError("https://test", 429, "limited", {"Retry-After": "7"}, io.BytesIO(b"limited"))
    success = {"choices": [{"message": {"content": '{"passed":true}'}}]}
    with patch("wildtrace.diagram.request.urlopen", side_effect=[limited, reply(success)]) as call, \
            patch("wildtrace.diagram.time.sleep") as sleep:
        assert judge._request_with_images([], "test")["response"] == '{"passed":true}'
    sleep.assert_called_once_with(7)
    assert call.call_count == 2
    bodies = [json.loads(c.args[0].data) for c in call.call_args_list]
    assert bodies[0] == bodies[1]
    assert bodies[0]["model"] == "qwen/qwen3-vl-8b-instruct"


def test_authentication_error_is_not_retried():
    judge = OpenRouterSemanticValidator({"openrouter_max_retries": 3})
    denied = HTTPError("https://test", 401, "denied", {}, io.BytesIO(b"denied"))
    with patch("wildtrace.diagram.request.urlopen", side_effect=denied) as call, \
            patch("wildtrace.diagram.time.sleep") as sleep, pytest.raises(RuntimeError, match="denied"):
        judge._request_with_images([], "test")
    assert call.call_count == 1
    sleep.assert_not_called()


def test_embedded_transient_error_retries_and_stops_at_limit():
    judge = OpenRouterSemanticValidator({"openrouter_max_retries": 1})
    with patch("wildtrace.diagram.request.urlopen", side_effect=lambda *a, **kw: reply({"error": {"code": 503}})) as call, \
            patch("wildtrace.diagram.time.sleep"), pytest.raises(RuntimeError, match="503"):
        judge._request_with_images([], "test")
    assert call.call_count == 2


def test_supervisor_retries_latest_errors_and_missing_artifacts(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/run_paper_generator_track.py"
    spec = importlib.util.spec_from_file_location("generator_track", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "good.png").touch()
    rows = [
        {"setup": "s", "sample_id": "good", "output_path": "good.png"},
        {"setup": "s", "sample_id": "failed", "output_path": "good.png"},
        {"setup": "s", "sample_id": "failed", "error": "429"},
        {"setup": "s", "sample_id": "missing", "output_path": "missing.png"},
    ]
    (tmp_path / "pipeline.ndjson").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with patch.object(module, "REPO", tmp_path), patch.object(module, "RESULTS", tmp_path):
        assert module.completed_subjects("s") == {"good"}


def test_supervisor_probes_before_waiting_for_api_recovery(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/run_paper_generator_track.py"
    spec = importlib.util.spec_from_file_location("generator_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVAL = tmp_path
    from types import SimpleNamespace
    with patch.object(module.subprocess, "run", side_effect=[SimpleNamespace(returncode=1), SimpleNamespace(returncode=0)]), \
            patch.object(module.time, "sleep") as sleep, patch.object(module, "log"):
        module.wait_for_judge()
    sleep.assert_called_once_with(300)


def test_paper_table_rejects_partial_generator_and_judge_denominators():
    path = Path(__file__).resolve().parents[1] / "paper/scripts/make_generator_table.py"
    spec = importlib.util.spec_from_file_location("generator_table", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    setup = {"final": {key: {"value": 0.5, "n": 20} for key in ("clip_top1", "blank", "trajectory_ok")},
             "referee": {module.STRONG: {"value": 0.5, "n": 20}}}
    import copy
    summary = {"setups": {key: copy.deepcopy(setup) for key in module.GENERATORS},
               "referee_agreement": {f"{key} vs {module.STRONG}": {"n": 80, "kappa": 0.5, "agreement": 0.9}
                                     for key in module.JUDGES}}
    assert "Interim" not in module.table(summary)
    summary["setups"]["flux-qwen3"]["final"]["clip_top1"]["n"] = 3
    with pytest.raises(ValueError, match="incomplete"):
        module.table(summary)
    assert "Interim" in module.table(summary, allow_incomplete=True)
    summary["setups"]["flux-qwen3"]["final"]["clip_top1"]["n"] = 20
    summary["referee_agreement"][f"qwen3-vl-8b vs {module.STRONG}"]["n"] = 79
    with pytest.raises(ValueError, match="incomplete"):
        module.table(summary)


def test_latest_referee_result_matches_current_image(tmp_path, monkeypatch):
    import sys
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    import benchmark_setups as module
    rows = [
        {"referee": "judge", "setup": "s", "sample_id": "a", "output_path": "current.png", "passed": True},
        {"referee": "judge", "setup": "s", "sample_id": "a", "output_path": "current.png", "error": "429"},
        {"referee": "judge", "setup": "s", "sample_id": "a", "output_path": "old.png", "passed": True},
    ]
    (tmp_path / "referee.ndjson").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with patch.object(module, "out_dir", return_value=tmp_path), \
            patch.object(module, "final_rows", return_value=[{"setup": "s", "sample_id": "a", "output_path": "current.png"}]):
        assert module.referee_rows() == [rows[1]]
