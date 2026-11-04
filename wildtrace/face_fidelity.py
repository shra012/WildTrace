"""Compare visible portrait features locally; hosted judging requires a public-source opt-in."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from wildtrace.diagram import OllamaSemanticValidator, OpenRouterSemanticValidator, _extract_json_object

FEATURES = ("ears", "glasses", "beard", "hat", "earrings", "hair_strands_braids")
PRESENCE = {"present", "absent", "uncertain"}
# The unqualified 8b tag is a thinking model; its budget can expire before any JSON.
LOCAL_MODEL = "qwen3-vl:8b-instruct"
HOSTED_MODEL = "qwen/qwen3-vl-8b-instruct"
PAIR_SCHEMA = {"type": "object", "properties": {k: {"type": "string"} for k in ("photo", "drawing")},
               "required": ["photo", "drawing"], "additionalProperties": False}
FEATURE_SCHEMA = {"type": "object", "properties": {k: {"type": "string", "enum": sorted(PRESENCE)}
                                                       for k in ("photo", "drawing")},
                  "required": ["photo", "drawing"], "additionalProperties": False}
FACE_SCHEMA = {
    "type": "object", "properties": {
        "features": {"type": "object", "properties": {name: FEATURE_SCHEMA for name in FEATURES},
                     "required": list(FEATURES), "additionalProperties": False},
        "hair_length": PAIR_SCHEMA, "expression": PAIR_SCHEMA,
        "resemblance": {"type": "integer", "minimum": 1, "maximum": 5},
        "missing": {"type": "array", "items": {"type": "string"}},
        "extra": {"type": "array", "items": {"type": "string"}},
    }, "required": ["features", "hair_length", "expression", "resemblance", "missing", "extra"],
    "additionalProperties": False,
}
SCHEMA_HASH = hashlib.sha256(json.dumps(FACE_SCHEMA, sort_keys=True).encode()).hexdigest()
PROMPT = """Image 1 is a source photograph; image 2 is a pen-stroke portrait of it.
Compare visible features, allowing for simple line-art abstraction. Do not identify the person.
If a feature is occluded or ambiguous, say uncertain; do not invent it.
Return JSON only, in this structure:
{"features":{"ears":{"photo":"present|absent|uncertain","drawing":"present|absent|uncertain"},
"glasses":{"photo":"present|absent|uncertain","drawing":"present|absent|uncertain"},
"beard":{"photo":"present|absent|uncertain","drawing":"present|absent|uncertain"},
"hat":{"photo":"present|absent|uncertain","drawing":"present|absent|uncertain"},
"earrings":{"photo":"present|absent|uncertain","drawing":"present|absent|uncertain"},
"hair_strands_braids":{"photo":"present|absent|uncertain","drawing":"present|absent|uncertain"}},
"hair_length":{"photo":"short description","drawing":"short description"},
"expression":{"photo":"short description","drawing":"short description"},
"resemblance":1,"missing":["visible missing features"],"extra":["features added by the drawing"]}
For hair_strands_braids judge distinctive locks or braids, not every individual hair.
Resemblance is an integer: 1 poor, 2 weak, 3 moderate, 4 good, 5 very close.
Glasses must describe actual visible frames, not eyes or eyebrows. Empty missing/extra lists are valid.
"""


def _checklist(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or not isinstance(raw.get("features"), dict):
        raise ValueError("Face fidelity judge did not return a feature checklist")
    features = {}
    for name in FEATURES:
        feature = raw["features"].get(name)
        if not isinstance(feature, dict) or any(not isinstance(feature.get(k), str) or feature[k] not in PRESENCE
                                                for k in ("photo", "drawing")):
            raise ValueError(f"Invalid presence verdict for {name}")
        features[name] = {k: feature[k] for k in ("photo", "drawing")}
    score = raw.get("resemblance")
    if type(score) is not int or not 1 <= score <= 5:
        raise ValueError("Resemblance must be an integer from 1 to 5")
    result: dict[str, Any] = {"features": features, "resemblance": score}
    for name in ("hair_length", "expression"):
        value = raw.get(name)
        if not isinstance(value, dict) or any(not isinstance(value.get(k), str) for k in ("photo", "drawing")):
            raise ValueError(f"Invalid {name} comparison")
        result[name] = {k: value[k][:300] for k in ("photo", "drawing")}
    for name in ("missing", "extra"):
        value = raw.get("model_" + name, raw.get(name))
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"Invalid {name} feature list")
        result["model_" + name] = [item[:300] for item in value[:30]]
        result[name] = list(result["model_" + name])
    # Reconcile the free-text lists with the explicit presence checklist.
    for name, feature in features.items():
        target = "missing" if feature == {"photo": "present", "drawing": "absent"} else \
                 "extra" if feature == {"photo": "absent", "drawing": "present"} else None
        if target and name not in result[target]:
            result[target].append(name)
    glasses = features["glasses"]["photo"]
    result["include_glasses"] = {"present": True, "absent": False, "uncertain": None}[glasses]
    return result


def judge_face_fidelity(
    source_png: Path | str,
    lineart_png: Path | str,
    *,
    backend: str = "ollama",
    model: str | None = None,
    source_is_open_images: bool = False,
    ollama_host: str = "http://127.0.0.1:11434",
) -> dict[str, Any]:
    """Return a validated JSON checklist, never silently switching to a hosted model.

    Hosted calls are permitted only when the caller confirms an Open Images source.
    An uncertain glasses verdict returns include_glasses=None; callers keep their default.
    """
    if backend not in {"ollama", "openrouter"}:
        raise ValueError("Face fidelity backend must be ollama or openrouter")
    if backend == "openrouter" and not source_is_open_images:
        raise ValueError("Hosted face fidelity judging is restricted to Open Images test faces")
    host = urlsplit(ollama_host)
    if backend == "ollama" and (host.hostname not in {"localhost", "127.0.0.1", "::1"}
                                or host.scheme != "http" or host.username or host.password
                                or host.path not in {"", "/"} or host.query or host.fragment):
        raise ValueError("Local face fidelity requires an Ollama URL on this machine")
    paths = [Path(source_png), Path(lineart_png)]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    settings = {"temperature": 0.0, "max_tokens": 1200, "timeout_seconds": 180}
    if backend == "ollama":
        model = model or LOCAL_MODEL
        judge = OllamaSemanticValidator({**settings, "ollama_model_name": model,
                                        "host": ollama_host, "keep_alive": "0s", "think": False, "format": FACE_SCHEMA})
    else:
        model = model or HOSTED_MODEL
        judge = OpenRouterSemanticValidator({**settings, "openrouter_model_name": model,
                                            "openrouter_timeout_seconds": 180, "openrouter_max_retries": 4})
    judge.validate_ready()
    response = judge._request_with_images(paths, PROMPT)
    result = _checklist(_extract_json_object(str(response.get("response", ""))))
    result["provenance"] = {
        "schema_version": 2, "backend": backend, "model": model,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "response_schema_sha256": SCHEMA_HASH,
        "source_sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
        "lineart_sha256": hashlib.sha256(paths[1].read_bytes()).hexdigest(),
    }
    if backend == "ollama":
        result["provenance"]["ollama_host"] = ollama_host
    return result
