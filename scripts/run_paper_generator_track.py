#!/usr/bin/env python3
"""Resume the paper generator queue, preserving completed setups and logging RAM waits.

    nohup python3 scripts/run_paper_generator_track.py > outputs/paper_eval/logs/gen_supervisor.log 2>&1 &

Keep the existing 18 GiB available-RAM gate: FLUX CPU offload is memory hungry.
The robot queue is independent and retains its current concurrency setting.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "outputs" / "paper_eval"
RESULTS = REPO / "outputs" / "benchmarks" / "paper-r3"
SETUPS = ("informative-qwen3", "sd15cn-qwen3", "flux-qwen3", "omnigen2-qwen3")
REFEREES = ("qwen2.5-vl-7b", "qwen3-vl-8b", "qwen3-vl-30b-a3b", "gemma-3-12b", "qwen3-vl-235b")
MIN_AVAILABLE_GIB = 18


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def log(message):
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}", flush=True)


def check_handoff():
    if (EVAL / "CLAUDE_TAKEOVER.json").exists():
        log("PAUSED for Claude takeover; see outputs/paper_eval/HANDOFF.md")
        raise SystemExit(0)


def completed_subjects(setup):
    latest = {r["sample_id"]: r for r in rows(RESULTS / "pipeline.ndjson") if r["setup"] == setup}
    return {sid for sid, row in latest.items() if "error" not in row and row.get("output_path")
            and (REPO / row["output_path"]).is_file()}


def available_gib():
    fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(fields["MemAvailable"].split()[0]) / 1024**2


def wait_ready():
    last_log = 0
    while True:
        check_handoff()
        training = subprocess.run(
            ["pgrep", "-f", "[t]rain_act.py|[l]erobot-train"], capture_output=True
        ).returncode == 0
        available = available_gib()
        if not training and available >= MIN_AVAILABLE_GIB:
            log(f"READY available_gib={available:.2f}")
            return
        if time.monotonic() - last_log >= 300:
            log(f"WAIT training={training} available_gib={available:.2f} required_gib={MIN_AVAILABLE_GIB}")
            last_log = time.monotonic()
        time.sleep(30)


def stage(*args):
    check_handoff()
    log("START " + " ".join(args))
    result = subprocess.run(["uv", "run", "python", "scripts/benchmark_setups.py", *args], cwd=REPO)
    if result.returncode:
        raise RuntimeError(f"{' '.join(args)} exited {result.returncode}")
    log("END " + " ".join(args))


def wait_for_judge():
    """Keep API outages from repeatedly loading models and regenerating drawings."""
    while True:
        check_handoff()
        result = subprocess.run(["uv", "run", "python", "scripts/benchmark_setups.py", "probe"], cwd=REPO)
        if result.returncode == 0:
            return
        log("WAIT hosted judge unavailable; retry cached-image probe in 300 s")
        time.sleep(300)


def main():
    os.chdir(REPO)
    os.environ["WILDTRACE_BENCHMARK_CONFIG"] = "configs/benchmark_paper.yaml"
    EVAL.mkdir(parents=True, exist_ok=True)
    check_handoff()
    with (EVAL / "generator_track.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Generator supervisor already running")
        lock.write(str(os.getpid()))
        lock.flush()
        subjects = {s["sample_id"] for s in json.loads((REPO / "benchmarks/evalset_v2.json").read_text())["subjects"]}
        for setup in SETUPS:
            done = completed_subjects(setup)
            if done == subjects:
                log(f"SKIP {setup} complete={len(done)}/{len(subjects)}")
                continue
            log(f"PENDING {setup} complete={len(done)}/{len(subjects)}")
            if any(r.get("output_path") and "error" not in r for r in rows(RESULTS / "pipeline.ndjson")):
                wait_for_judge()
            wait_ready()
            stage("pipeline", "--setup", setup, "--resume")
            done = completed_subjects(setup)
            runs = [r for r in rows(RESULTS / "pipeline_runs.ndjson") if r["setup"] == setup]
            if done != subjects or not runs or runs[-1]["exit_code"] != 0:
                raise RuntimeError(f"{setup} incomplete or worker failed; inspect pipeline_runs.ndjson")
        wait_ready()
        stage("evaluate")
        # All generators have exited before the local Ollama judge loads.
        if subprocess.run(["ollama", "list"], capture_output=True).returncode:
            with (EVAL / "logs/ollama_serve.log").open("a") as output:
                server = subprocess.Popen(["ollama", "serve"], stdout=output, stderr=subprocess.STDOUT,
                                          start_new_session=True)
            for _ in range(30):
                if subprocess.run(["ollama", "list"], capture_output=True).returncode == 0:
                    break
                if server.poll() is not None:
                    raise RuntimeError("Ollama server exited; inspect logs/ollama_serve.log")
                time.sleep(1)
            else:
                raise RuntimeError("Ollama server did not become ready")
        subprocess.run(["ollama", "pull", "qwen2.5vl:7b"], check=True)
        wait_for_judge()
        stage("referee", "--resume")
        stage("report")
        finals = {(r["setup"], r["sample_id"]): r for r in rows(RESULTS / "pipeline.ndjson")}
        expected = {(judge, r["setup"], r["sample_id"], r["output_path"])
                    for judge in REFEREES for r in finals.values() if r.get("output_path")}
        latest = {(r["referee"], r["setup"], r["sample_id"], r["output_path"]): r
                  for r in rows(RESULTS / "referee.ndjson")}
        verdicts = {key for key, row in latest.items() if "error" not in row and not row.get("parse_failed")}
        if expected - verdicts:
            raise RuntimeError(f"Report saved, but {len(expected - verdicts)} referee verdicts need retry; rerun this supervisor")
        subprocess.run(["python3", "paper/scripts/make_generator_table.py"], cwd=REPO, check=True)
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                       cwd=REPO / "paper", check=True)
        match = re.search(r"Output written on main.pdf \((\d+) pages?", (REPO / "paper/main.log").read_text())
        pages = int(match.group(1)) if match else None
        if pages != 8:
            raise RuntimeError(f"Benchmark complete and paper rebuilt, but layout needs review: {pages} pages (target 8)")
        (EVAL / "GENERATOR_COMPLETE.json").write_text(json.dumps({
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "final_images": len(finals), "referee_verdicts": len(expected), "paper_pages": pages,
        }, indent=2) + "\n")
        log("CHAIN DONE")


if __name__ == "__main__":
    main()
