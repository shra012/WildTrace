"""Run the headless Isaac drawing runner over many trajectories, one process per drawing.

Runs in WSL and launches Windows Isaac Sim through cmd.exe. One process per
drawing costs ~20 s of startup against minutes of drawing and keeps every
run isolated. Finished outputs are skipped, so a batch can be resumed.

Record 1 g IK demonstrations for ACT fine-tuning (a0 canonical, a1.. augmented):

    python3 isaac-sim/scripts/isaac_batch.py record --sample 300 --augmentations 1

Closed-loop evaluation on the gold test set:

    python3 isaac-sim/scripts/isaac_batch.py eval --controller ik --tag ik
    python3 isaac-sim/scripts/isaac_batch.py eval --controller act --tag act_kin   # server must be running
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_ISAAC = r"C:\isaac-sim-6.0.1\python.bat"


def _parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--isaac-python", default=os.environ.get("ISAAC_PYTHON_BAT", DEFAULT_ISAAC))
    common.add_argument("--distro", default=os.environ.get("WSL_DISTRO_NAME", "Ubuntu-24.04"))
    common.add_argument("--limit", type=int, default=0)
    common.add_argument("--timeout-s", type=int, default=3600, help="Wall-clock limit per drawing")
    common.add_argument("--dry-run", action="store_true")
    common.add_argument("--runner-arg", action="append", default=[], help="Extra runner flag, e.g. --runner-arg=--gravity=0")

    record = sub.add_parser("record", parents=[common])
    record.add_argument("--trajectories", default=str(REPO_ROOT / "outputs" / "trajectories"),
                        help="Directory of trajectory JSONs, or a single file")
    record.add_argument("--exclude-dir", default=str(PROJECT_ROOT / "data" / "act_test"))
    record.add_argument("--sample", type=int, default=300, help="Drawings to record, stratified by category")
    record.add_argument("--augmentations", type=int, default=1)
    record.add_argument("--seed", type=int, default=17)
    record.add_argument("--out", default=str(REPO_ROOT / "outputs" / "act_demos" / "physics"))

    evaluate = sub.add_parser("eval", parents=[common])
    evaluate.add_argument("--trajectories", default=str(PROJECT_ROOT / "data" / "act_test"),
                          help="Directory of trajectory JSONs, or a single file")
    evaluate.add_argument("--controller", choices=["ik", "act"], required=True)
    evaluate.add_argument("--tag", required=True, help="Result folder name, e.g. ik, act_kin, act_kin_phys")
    evaluate.add_argument("--policy-address", default="127.0.0.1:8790")
    evaluate.add_argument("--out", default=str(REPO_ROOT / "outputs" / "act_eval"))
    return parser.parse_args()


def windows_path(path: Path, distro: str) -> str:
    resolved = str(Path(path).resolve())
    if resolved.startswith("/mnt/") and len(resolved) > 6 and resolved[6] == "/":
        return f"{resolved[5].upper()}:\\" + resolved[7:].replace("/", "\\")
    return f"\\\\wsl.localhost\\{distro}" + resolved.replace("/", "\\")


def stratified_sample(paths: list[Path], count: int, seed: int) -> list[Path]:
    """Round-robin across categories so small categories (Fish) are represented."""
    by_category: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        by_category[path.parent.name].append(path)
    rng = random.Random(seed)
    for items in by_category.values():
        rng.shuffle(items)
    chosen: list[Path] = []
    while len(chosen) < count and any(by_category.values()):
        for category in sorted(by_category):
            if by_category[category] and len(chosen) < count:
                chosen.append(by_category[category].pop())
    return chosen


def _jobs(args) -> list[dict]:
    jobs = []
    if args.mode == "record":
        source = Path(args.trajectories)
        if source.is_file():
            chosen = [source]
        else:
            excluded = {p.stem for p in Path(args.exclude_dir).rglob("*.json")}
            sources = sorted(p for p in source.glob("*/*.json") if p.stem not in excluded)
            chosen = stratified_sample(sources, args.sample, args.seed)
        for path in chosen:
            for augmentation in range(args.augmentations):
                name = f"{path.stem}_p{augmentation}"
                demo = Path(args.out) / path.parent.name / f"{name}.npz"
                extra = ["--record-demonstration", "--demo-path", demo]
                if augmentation > 0:
                    extra += ["--augment-seed", str(args.seed * 1000 + augmentation)]
                jobs.append({"trajectory": path, "name": name, "done": demo,
                             "run_dir": Path(args.out) / "runs" / name, "extra": extra})
    else:
        source = Path(args.trajectories)
        for path in [source] if source.is_file() else sorted(source.glob("*/*.json")):
            run_dir = Path(args.out) / args.tag / path.parent.name / path.stem
            extra = ["--controller", args.controller]
            if args.controller == "act":
                extra += ["--policy-address", args.policy_address]
            jobs.append({"trajectory": path, "name": path.stem, "done": run_dir / "run_metrics.json",
                         "run_dir": run_dir, "extra": extra + list(args.runner_arg)})
    return jobs[: args.limit] if args.limit else jobs


def _already_failed(job) -> bool:
    """A finished-but-failed IK run is deterministic, so retrying it only repeats the failure.

    Crashed runs (no metrics) are retried. Delete the run folder to force a retry.
    """
    metrics = Path(job["run_dir"]) / "run_metrics.json"
    return metrics.exists() and json.loads(metrics.read_text()).get("status") == "failed"


def _command(args, job) -> str:
    runner = PROJECT_ROOT / "scripts" / "run_two_stroke_drawing.py"
    parts = [args.isaac_python, windows_path(runner, args.distro), "--headless",
             "--trajectory", windows_path(job["trajectory"], args.distro),
             "--output-dir", windows_path(job["run_dir"], args.distro)]
    for value in job["extra"]:
        parts.append(windows_path(value, args.distro) if isinstance(value, Path) else str(value))
    return " ".join(f'"{part}"' if " " in part else part for part in parts)


def main() -> int:
    args = _parse_args()
    jobs = _jobs(args)
    pending = [job for job in jobs if not Path(job["done"]).exists() and not _already_failed(job)]
    out = Path(args.out) / (args.tag if args.mode == "eval" else "")
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "batch_log.ndjson"
    print(f"[OK] {len(jobs)} jobs, {len(pending)} pending; log {log_path}", flush=True)
    counts = defaultdict(int)
    for number, job in enumerate(pending, start=1):
        Path(job["run_dir"]).mkdir(parents=True, exist_ok=True)
        command = _command(args, job)
        if args.dry_run:
            print(command)
            continue
        started = time.time()
        try:
            # Kit blocks at startup waiting on stdin (EULA prompt) unless stdin is
            # closed and the EULA is pre-accepted; a hung run used no CPU for 35 min.
            result = subprocess.run(["cmd.exe", "/c", f"set OMNI_KIT_ACCEPT_EULA=YES&& {command}"], cwd="/mnt/c",
                                    stdin=subprocess.DEVNULL, capture_output=True, text=True, errors="replace",
                                    timeout=args.timeout_s)
            code = result.returncode
            tail = (result.stdout + result.stderr)[-2000:]
        except subprocess.TimeoutExpired:
            code, tail = -1, "wall-clock timeout"
        (Path(job["run_dir"]) / "isaac_stdout_tail.txt").write_text(tail, encoding="utf-8")
        metrics_path = Path(job["run_dir"]) / "run_metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        status = metrics.get("status", "crashed")
        counts[status] += 1
        record = {"name": job["name"], "category": job["trajectory"].parent.name, "exit_code": code,
                  "status": status, "wall_s": round(time.time() - started, 1),
                  "sim_s": metrics.get("simulation_duration_s"),
                  "rmse_m": metrics.get("desired_to_executed_nearest_path_error", {}).get("rmse_m")}
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(record) + "\n")
        rtf = (record["sim_s"] or 0) / max(record["wall_s"], 1e-9)
        print(f"[{number}/{len(pending)}] {job['name']} {status} wall {record['wall_s']}s "
              f"sim {record['sim_s']} (x{rtf:.2f}) rmse {record['rmse_m']} | {dict(counts)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
