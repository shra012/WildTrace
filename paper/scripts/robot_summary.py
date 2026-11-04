"""Summarize the paper's twin gold-set runs (outputs/paper_eval/<tag>) into robot_summary.json.

    ~/anaconda3/bin/python3 paper/scripts/robot_summary.py paper_ik paper_ikstream paper_act ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "outputs" / "paper_eval"
EXPECTED_IDS = {p.stem for p in (REPO / "isaac-sim" / "data" / "act_test").glob("*/*.json")}
RNG = np.random.default_rng(0)


def boot(x, n=10000):
    x = np.asarray(x, float)
    return [float(v) for v in np.percentile(RNG.choice(x, (n, len(x))).mean(1), [2.5, 97.5])]


def runs(tag):
    out = {}
    for p in sorted((EVAL / tag).glob("*/*/run_metrics.json")):
        out[p.parent.name] = (p.parent.parent.name, json.loads(p.read_text()))
    return out


def tag_stats(tag):
    r = runs(tag)
    fin = {d: (c, m) for d, (c, m) in r.items() if m.get("status") == "finished"}
    g = lambda m, k, s: m[k][s] * 1e3
    rm = [g(m, "desired_to_executed_nearest_path_error", "rmse_m") for _, m in fin.values()]
    st = {
        "finished": len(fin), "total": len(r),
        "expected": len(EXPECTED_IDS), "complete": bool(EXPECTED_IDS) and set(r) == EXPECTED_IDS,
        "pending": sorted(EXPECTED_IDS - set(r)),
        "rmse_mm": float(np.mean(rm)) if rm else None, "rmse_ci": boot(rm) if rm else None,
        "p95_mm": float(np.mean([g(m, "desired_to_executed_nearest_path_error", "p95_error_m") for _, m in fin.values()])) if fin else None,
        "max_mm": float(np.mean([g(m, "desired_to_executed_nearest_path_error", "max_error_m") for _, m in fin.values()])) if fin else None,
        "reverse_rmse_mm": float(np.mean([g(m, "executed_to_desired_nearest_path_error", "rmse_m") for _, m in fin.values()])) if fin else None,
        "corner_rmse_mm": float(np.mean([g(m, "corner_tracking_error", "rmse_m") for _, m in fin.values() if m.get("corner_tracking_error")])) if fin else None,
        "jerk_rms": float(np.mean([m["pen_down_joint_jerk_rad_s3"]["rms"] for _, m in fin.values()])) if fin else None,
        "settled_fraction": float(np.mean([m.get("velocity_settled_fraction", np.nan) for _, m in fin.values()])) if fin else None,
        "median_tip_speed_mm_s": float(np.mean([m["tip_speed_mm_s_stats"]["median"] for _, m in fin.values()])) if fin else None,
        "sim_time_s": float(np.mean([m["simulation_duration_s"] for _, m in fin.values()])) if fin else None,
        "failures": {d: m.get("failure_reason", "") for d, (c, m) in r.items() if m.get("status") != "finished"},
        "per_class_rmse_mm": {},
    }
    for cls in sorted({c for c, _ in r.values()}):
        v = [g(m, "desired_to_executed_nearest_path_error", "rmse_m") for c, m in fin.values() if c == cls]
        n = sum(1 for c, _ in r.values() if c == cls)
        st["per_class_rmse_mm"][cls] = {"finished": len(v), "total": n, "rmse": float(np.mean(v)) if v else None}
    return st, fin


def paired(base_fin, fin):
    keys = sorted(set(base_fin) & set(fin))
    if not keys:
        return None
    rm = lambda m: m["desired_to_executed_nearest_path_error"]["rmse_m"] * 1e3
    d = np.array([rm(fin[k][1]) - rm(base_fin[k][1]) for k in keys])
    return {"pairs": len(keys), "mean_delta_mm": float(d.mean()), "delta_ci": boot(d),
            "lower_than_base": int((d < 0).sum()), "wilcoxon_p": float(wilcoxon(d).pvalue) if np.any(d) else 1.0}


def main():
    tags = sys.argv[1:] or ["paper_ik", "paper_ikstream", "paper_ikstream_slow", "paper_act", "paper_smolvla", "paper_act_seed1", "paper_act_seed2"]
    summary, fins = {}, {}
    for t in tags:
        if (EVAL / t).exists():
            summary[t], fins[t] = tag_stats(t)
    if "paper_ik" in fins:
        for t in fins:
            if t != "paper_ik":
                summary[t]["paired_vs_ik"] = paired(fins["paper_ik"], fins[t])
    seed_tags = ("paper_act", "paper_act_seed1", "paper_act_seed2")
    completed = [t for t in seed_tags if t in summary and summary[t]["complete"]]
    shared = sorted(set.intersection(*(set(fins[t]) for t in completed))) if completed else []
    # Use completed batches and the same successful drawings for every seed.
    # Otherwise an in-flight one-drawing seed is compared with a full test set.
    if shared:
        seeds = [float(np.mean([fins[t][k][1]["desired_to_executed_nearest_path_error"]["rmse_m"] * 1e3
                                for k in shared])) for t in completed]
        summary["act_seed_spread"] = {
            "n": len(seeds), "complete": len(completed) == len(seed_tags), "tags": completed,
            "pending_tags": [t for t in seed_tags if t not in completed], "paired_drawings": len(shared),
            "mean": float(np.mean(seeds)), "std": float(np.std(seeds, ddof=1)) if len(seeds) > 1 else None,
            "values": seeds,
        }
    (EVAL / "robot_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
