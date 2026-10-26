"""Run one animal job (wildtrace.studio_pipeline) in its own process.

FLUX-schnell with model CPU offload holds about 20 GB of host RAM. Inside the
server process that both risks the OOM killer taking the whole backend down and
keeps the memory after the job. A child process per job fails alone and gives
everything back when it exits.

Protocol: argv[1] is a JSON request file; stdout carries one JSON object per
line, {"event": {...StageEvent}} while running, then {"result": {...}} or
{"error": "..."}. Library logs and progress bars go to stderr.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path


def _send(message: dict) -> None:
    sys.stdout.write(json.dumps(message, default=str) + "\n")
    sys.stdout.flush()


def main() -> int:
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    # Keep the protocol channel clean: anything else printing to stdout goes to stderr.
    protocol, sys.stdout = sys.stdout, sys.stderr
    try:
        from wildtrace import studio_pipeline as sp

        def emit(event) -> None:
            sys.stdout, saved = protocol, sys.stdout
            try:
                _send({"event": asdict(event)})
            finally:
                sys.stdout = saved

        result = sp.generate_animal_trajectory(
            out_dir=Path(request["out_dir"]),
            repo_root=Path(request["repo_root"]),
            config_dir=Path(request["config_dir"]),
            category=request["category"],
            image_path=Path(request["image_path"]) if request.get("image_path") else None,
            mask_path=None,
            sample=request.get("sample"),
            emit=emit,
            backend=request.get("backend"),
        )
        sys.stdout = protocol
        _send({"result": result})
        return 0
    except Exception as exc:  # reported to the parent, which fails the job
        sys.stdout = protocol
        _send({"error": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
