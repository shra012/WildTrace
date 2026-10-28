"""Draw N drawings per class on the physical xArm 7 with ACT and record metrics for every one.

1. Pre-check: each candidate runs offline first (run_xarm7_act_hardware.py --offline). Only drawings
   that finish with the joints well clear of their limits are drawn. Candidates come from the
   held-out test set (data/act_test) first, then from the training pool (flagged in the results).
2. Draw: between drawings the arm parks high over the paper and the wrist camera waits for a blank
   sheet that has stayed still for a few seconds. Then the drawing runs, the arm returns to the park
   pose, and the photo is compared with the planned path (find_missing_ink.analyse) for ink coverage.

Results go to <batch>/results.csv and results.json; every run keeps its own log, per-tick CSV and
the per-tick NPZ in the policy's training format. Create <batch>/STOP to stop after the current
drawing. Any safety stop ends the batch.

    .venv/Scripts/python.exe isaac-sim/scripts/run_act_eval_batch.py --per-class 5
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from find_missing_ink import analyse, ink_distance, mapped_strokes  # noqa: E402

ACT_PYTHON = PROJECT / ".venv-act" / "Scripts" / "python.exe"
RUNNER = PROJECT / "scripts" / "run_xarm7_act_hardware.py"
PARK = (476.9, 88.0, 260.0, -180.0, 0.0, 0.0)  # wrist camera sees the whole sheet from here
WRIST_CAMERA = 0
IGNORE = [[740, 640, 1000, 720], [1030, 0, 1280, 440], [440, 140, 480, 190]]  # gripper fingers, notebook, clip
CLASSES = ["Bird", "Cat", "Dog", "Fish", "Frog", "Horse"]


def say(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def candidates(per_class_pool: int, seed: int) -> dict[str, list[tuple[Path, str]]]:
    rng = random.Random(seed)
    out = {}
    for name in CLASSES:
        test = sorted((PROJECT / "data" / "act_test" / name).glob("*.json"))
        held = {p.stem for p in test}
        pool = [p for p in sorted((REPO / "outputs" / "trajectories" / name).glob("*.json")) if p.stem not in held]
        rng.shuffle(pool)
        out[name] = [(p, "test") for p in test] + [(p, "train_pool") for p in pool[:per_class_pool]]
    return out


def runner(args: list[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as handle:
        return subprocess.call([str(ACT_PYTHON), str(RUNNER), *args], stdout=handle, stderr=subprocess.STDOUT, cwd=str(PROJECT))


def precheck_one(path: Path, run_dir: Path, draw_args: list[str]) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    runner(["--offline", "--trajectory", str(path), "--run-dir", str(run_dir), "--max-seconds", "300", *draw_args], run_dir / "stdout.txt")
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        reason = json.loads((run_dir / "result.json").read_text())["outcome"] if (run_dir / "result.json").exists() else "no result"
        return {"ok": False, "reason": reason}
    summary = json.loads(summary_path.read_text())
    margin = min(summary["joint_limit_margin_deg"])
    ok = summary["outcome"] == "finished" and margin >= 5.0
    return {"ok": ok, "reason": "ok" if ok else f"{summary['outcome']}, joint margin {margin}", **summary}


def precheck(batch: Path, per_class: int, pool: dict, draw_args: list[str], workers: int) -> list[dict]:
    chosen, rows = [], []
    for name in CLASSES:
        accepted, queue = [], list(pool[name])
        with ThreadPoolExecutor(max_workers=workers) as executor:
            while len(accepted) < per_class and queue:
                batch_items = [queue.pop(0) for _ in range(min(workers, per_class - len(accepted), len(queue)))]
                futures = [(p, src, executor.submit(precheck_one, p, batch / "precheck" / f"{name}_{p.stem}", draw_args)) for p, src in batch_items]
                for p, src, future in futures:
                    result = future.result()
                    rows.append({"class": name, "drawing_id": p.stem, "source": src, **{k: result.get(k) for k in ("ok", "reason", "seconds", "pen_down_error_mm", "joint_limit_margin_deg", "stall_nudges")}})
                    say(f"precheck {name} {p.stem} ({src}): {result['reason']}, nudges {result.get('stall_nudges')}")
                    if result["ok"] and len(accepted) < per_class:
                        accepted.append({"class": name, "drawing_id": p.stem, "source": src, "path": str(p), "offline": result})
        if len(accepted) < per_class:
            say(f"only {len(accepted)} {name} drawings passed the pre-check")
        chosen += accepted
    (batch / "precheck.json").write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
    return chosen


class Arm:
    def __init__(self, ip: str, paper_z: float) -> None:
        from xarm.wrapper import XArmAPI

        self.api = XArmAPI(ip, is_radian=False)
        self.paper_z = paper_z

    def ready(self) -> None:
        errors = self.api.get_err_warn_code()[1]
        if errors[0] != 0:
            raise RuntimeError(f"controller error {errors}; release the e-stop / clear it first")
        self.api.motion_enable(True)
        self.api.set_mode(0)
        self.api.set_state(0)
        time.sleep(0.4)

    def park(self) -> None:
        """Straight up to the park height, then over the sheet centre with the pen vertical."""
        self.ready()
        x, y, z, roll, pitch, yaw = self.api.get_position()[1]
        if z < PARK[2]:
            self._check(self.api.set_position(x=x, y=y, z=PARK[2], roll=roll, pitch=pitch, yaw=yaw, speed=20, mvacc=100, wait=True, timeout=60), "lift to park height")
        x0, y0, z0, r0, p0, w0 = PARK
        self._check(self.api.set_position(x=x0, y=y0, z=z0, roll=r0, pitch=p0, yaw=w0, speed=20, mvacc=100, wait=True, timeout=60), "move to park")

    @staticmethod
    def _check(code: int, what: str) -> None:
        if code != 0:
            raise RuntimeError(f"{what} failed with code {code}")

    def close(self) -> None:
        self.api.disconnect()


def photo(path: Path, camera: int = WRIST_CAMERA) -> np.ndarray:
    capture = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    frame = None
    for _ in range(20):
        ok, image = capture.read()
        if ok:
            frame = image
    capture.release()
    if frame is None:
        raise RuntimeError(f"camera {camera} returned no frame")
    cv2.imwrite(str(path), frame)
    return frame


def sheet_state(frame: np.ndarray) -> tuple[int, np.ndarray]:
    """Ink pixel count on the sheet and the sheet centroid (px)."""
    _, ink = ink_distance(frame, 100, IGNORE)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    paper = ((hsv[..., 1] < 60) & (hsv[..., 2] > 90)).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(paper)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return int(ink.sum()), centroids[biggest]


def wait_for_blank_sheet(stop_file: Path, reference: np.ndarray | None, still_s: float = 5.0, blank_px: int = 150) -> np.ndarray:
    say("waiting for a blank sheet: replace the drawn sheet (do not stack sheets), then keep hands clear")
    capture = cv2.VideoCapture(WRIST_CAMERA, cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    since, last_center, last_report = None, None, 0.0
    try:
        while True:
            if stop_file.exists():
                raise KeyboardInterrupt("STOP file")
            ok, frame = capture.read()
            if not ok:
                continue
            ink, center = sheet_state(frame)
            placed = reference is None or np.linalg.norm(center - reference) < 25
            steady = last_center is not None and np.linalg.norm(center - last_center) < 3
            last_center = center
            if ink < blank_px and placed and steady:
                since = since or time.time()
                if time.time() - since >= still_s:
                    return center
            else:
                since = None
            if time.time() - last_report > 15:
                say(f"  sheet check: ink px {ink}, sheet offset {0 if reference is None else round(float(np.linalg.norm(center - reference)), 1)} px")
                last_report = time.time()
            time.sleep(0.2)
    finally:
        capture.release()


def on_ac_power() -> bool:
    out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Battery).BatteryStatus"], capture_output=True, text=True)
    status = out.stdout.strip()
    return status != "1"  # 1 = discharging; no battery or anything else is fine


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=5)
    parser.add_argument("--pool", type=int, default=6, help="Extra training-pool candidates per class if test drawings fail")
    parser.add_argument("--z-offset-mm", type=float, default=0.0)
    parser.add_argument("--floor-below-touch-mm", type=float, default=0.6)
    parser.add_argument("--workers", type=int, default=3, help="Parallel offline pre-checks")
    parser.add_argument("--batch-dir", default=None)
    parser.add_argument("--precheck-only", action="store_true")
    parser.add_argument("--same-sheet", action="store_true", help="Draw everything on one sheet; ink coverage counts only new ink")
    parser.add_argument("--resume", action="store_true", help="Skip drawings that already have a finished run in --batch-dir")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    batch = Path(args.batch_dir).resolve() if args.batch_dir else PROJECT / "outputs" / "act_eval_batches" / datetime.now().strftime("%Y%m%d_%H%M%S")
    batch.mkdir(parents=True, exist_ok=True)
    stop_file = batch / "STOP"
    draw_args = ["--z-offset-mm", str(args.z_offset_mm), "--floor-below-touch-mm", str(args.floor_below_touch_mm)]
    hardware = yaml.safe_load((PROJECT / "config" / "xarm7_hardware.yaml").read_text(encoding="utf-8"))
    twin = yaml.safe_load((PROJECT / "config" / "xarm7_drawing_real.yaml").read_text(encoding="utf-8"))
    (batch / "settings.json").write_text(json.dumps({**vars(args), "paper_center_mm": hardware["paper_center_mm"]}, indent=1), encoding="utf-8")

    chosen_path = batch / "chosen.json"
    if args.resume and chosen_path.exists():
        chosen = json.loads(chosen_path.read_text())
    else:
        say(f"pre-checking candidates offline into {batch}")
        chosen = precheck(batch, args.per_class, candidates(args.pool, args.seed), draw_args, args.workers)
        chosen_path.write_text(json.dumps(chosen, indent=1, default=str), encoding="utf-8")
    say(f"{len(chosen)} drawings to draw: " + ", ".join(f"{c['class']}/{c['drawing_id']}" for c in chosen))
    if args.precheck_only:
        return 0

    results_path = batch / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    done = {r["drawing_id"] for r in results if r.get("outcome") == "finished"}
    arm = Arm(str(hardware["robot_ip"]), float(hardware["paper_center_mm"][2]))
    registration_path = batch / "registration.json"
    registration = json.loads(registration_path.read_text()) if registration_path.exists() else None
    reference = None
    try:
        arm.park()
        for number, item in enumerate(chosen, 1):
            if item["drawing_id"] in done:
                continue
            if stop_file.exists():
                say("STOP file found; ending the batch")
                break
            if args.same_sheet:
                time.sleep(3.0)  # drawings go on top of each other; ink coverage counts only new ink
            else:
                reference = wait_for_blank_sheet(stop_file, reference)
            while not on_ac_power():
                say("laptop is on battery: plug in the charger to continue")
                time.sleep(10)
            name = f"{number:02d}_{item['class']}_{item['drawing_id']}"
            run_dir = batch / "runs" / name
            run_dir.mkdir(parents=True, exist_ok=True)
            photo(run_dir / "before_wrist.png")
            say(f"drawing {number}/{len(chosen)}: {item['class']} {item['drawing_id']} ({item['source']})")
            code = runner(["--execute", "--trajectory", item["path"], "--run-dir", str(run_dir), "--max-seconds", "300", *draw_args], run_dir / "stdout.txt")
            arm.park()
            after = run_dir / "after_wrist.png"
            photo(after)
            summary = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
            outcome = summary.get("outcome") or json.loads((run_dir / "result.json").read_text())["outcome"]
            strokes = mapped_strokes(Path(item["path"]), hardware, twin)
            ink = analyse(after, strokes, ignore=IGNORE, init=registration, overlay=run_dir / "ink_overlay.png",
                          before=run_dir / "before_wrist.png" if args.same_sheet else None)
            if ink["registration"]["fit_cost_px"] < 6.0:
                registration = ink["registration"]
                registration_path.write_text(json.dumps(registration, indent=1), encoding="utf-8")
            (run_dir / "missing_strokes.json").write_text(json.dumps(ink, indent=1), encoding="utf-8")
            row = {
                "order": number, "class": item["class"], "drawing_id": item["drawing_id"], "source": item["source"],
                "outcome": outcome, "exit_code": code, "seconds": summary.get("seconds"),
                "rmse_mm": (summary.get("pen_down_error_mm") or {}).get("rmse"), "max_err_mm": (summary.get("pen_down_error_mm") or {}).get("max"),
                "tip_z_min_mm": summary.get("tip_z_min_mm"), "pen_plane_mm": summary.get("pen_down_plane_mm"),
                "min_joint_margin_deg": min(summary["joint_limit_margin_deg"]) if summary.get("joint_limit_margin_deg") else None,
                "real_lag_mean_mm": (summary.get("real_tip_lag_mm") or {}).get("mean"), "real_lag_max_mm": (summary.get("real_tip_lag_mm") or {}).get("max"),
                "policy_ms": summary.get("policy_ms_mean"), "longest_stall_s": summary.get("longest_stall_s"), "stall_nudges": summary.get("stall_nudges"),
                "late_ticks": summary.get("late_ticks_over_50ms"), "worst_tick_ms": summary.get("worst_tick_ms"), "ink_coverage_pct": ink["ink_coverage_pct"], "missing_mm": ink["missing_mm"],
                "path_mm": ink["path_mm"], "ink_fit_px": ink["registration"]["fit_cost_px"],
                "offline_rmse_mm": (item["offline"].get("pen_down_error_mm") or {}).get("rmse"), "run_dir": str(run_dir),
            }
            results.append(row)
            results_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
            with (batch / "results.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerows(results)
            say(f"  {outcome}: rmse {row['rmse_mm']} mm, ink coverage {row['ink_coverage_pct']}%, {row['seconds']} s")
            if outcome != "finished":
                say("safety stop during the drawing; ending the batch so it can be checked")
                break
    except KeyboardInterrupt as exc:
        say(f"stopped: {exc}")
    finally:
        try:
            arm.park()
        except Exception as exc:  # noqa: BLE001
            say(f"could not park: {exc}")
        arm.close()
    say(f"results: {batch / 'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
