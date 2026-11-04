"""Build the paper's figures and print the numbers its tables use.

Reads committed results under isaac-sim/results/ and, for the per-sample error
distributions and path overlays, the local executed paths under outputs/act_eval/.

    ~/anaconda3/bin/python3 paper/scripts/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde, wilcoxon

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "isaac-sim" / "results"
OUT = REPO / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Categorical slots 1-3 of the reference palette (validated all-pairs, light surface).
ACT, IK, VLA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
COL_W, FULL_W = 3.5, 7.16  # IEEE two-column widths, inches
ZOOM_MM = 14.0

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.4,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})


def load_runs(tag: str, root: Path | None = None) -> pd.DataFrame:
    rows = []
    for path in sorted(((root or RESULTS / "act_eval") / tag).glob("*/*/run_metrics.json")):
        m = json.loads(path.read_text())
        err = m.get("desired_to_executed_nearest_path_error") or {}
        rows.append({
            "category": path.parent.parent.name,
            "drawing": path.parent.name,
            "finished": m.get("status") == "finished",
            "failure": m.get("failure_reason", ""),
            "rmse_mm": err.get("rmse_m", np.nan) * 1e3,
            "p95_mm": err.get("p95_error_m", np.nan) * 1e3,
            "max_mm": err.get("max_error_m", np.nan) * 1e3,
            "corner_mm": (m.get("corner_tracking_error") or {}).get("rmse_m", np.nan) * 1e3,
            "jerk": (m.get("pen_down_joint_jerk_rad_s3") or {}).get("rms", np.nan),
            "heading_deg": (m.get("heading_error") or {}).get("rmse_deg", np.nan),
            "heading_med_deg": (m.get("heading_error") or {}).get("median_abs_error_deg", np.nan),
            "sim_s": m.get("simulation_duration_s", np.nan),
        })
    return pd.DataFrame(rows)


def per_sample_errors(tag: str, drawings: set[str], root: Path | None = None) -> np.ndarray:
    """Nearest-path error (mm) of every pen-down desired sample to the executed pen-down tip path."""
    out = []
    for d in sorted(((root or REPO / "outputs" / "act_eval") / tag).glob("*/*")):
        if d.name not in drawings or not (d / "executed_path.csv").exists():
            continue
        des = pd.read_csv(d / "desired_path.csv")
        ex = pd.read_csv(d / "executed_path.csv")
        des = des[des.state.str.startswith("DRAW")]
        ex = ex[ex.state.str.startswith("DRAW")]
        e, _ = cKDTree(ex[["tip_x_m", "tip_y_m", "tip_z_m"]].values).query(des[["x_m", "y_m", "z_m"]].values)
        out.append(e * 1e3)
    return np.concatenate(out)


def kde_panel(ax, series, xlabel, xmax, letter, log=False):
    grid = np.geomspace(xmax / 400, xmax, 600) if log else np.linspace(0, xmax, 600)
    peak = 0.0
    series = [tuple(x) + ("-",) * (4 - len(x)) for x in series]
    for label, values, color, style in series:
        values = np.asarray(values)
        values = values[np.isfinite(values)]
        fit = np.log10(values) if log else values
        kde = gaussian_kde(fit, bw_method="scott")
        dens = kde(np.log10(grid)) if log else kde(grid)
        peak = max(peak, dens.max())
        ax.fill_between(grid, dens, color=color, alpha=0.10 if style != "-" else 0.16, linewidth=0)
        ax.plot(grid, dens, color=color, linestyle=style, label=label)
        med = np.median(values)
        ax.axvline(med, color=color, linestyle=(0, (1, 1.5)), linewidth=0.9)
    for label, values, color, _ in series:
        if len(values) < 200:  # rug for per-drawing panels
            ax.plot(values, np.full(len(values), -0.04 * peak), "|", color=color, markersize=5, mew=0.8)
    ax.set_ylim(-0.08 * peak, peak * 1.12)
    if log:
        ax.set_xscale("log")
    ax.set_xlim(grid[0], grid[-1])
    ax.set_xlabel(xlabel)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    n = [int(np.isfinite(np.asarray(v, float)).sum()) for _, v, _, _ in series]
    short = (lambda k: f"{k / 1000:.0f}k") if min(n) >= 10000 else str
    ax.set_title(letter, loc="left", color=INK)  # sample counts go in the caption
    return [np.median(np.asarray(v)[np.isfinite(v)]) for _, v, _, _ in series]


EVAL = REPO / "outputs" / "paper_eval"
CONTROLLERS = [("IK", "paper_ik", IK, "-"), ("Streaming IK", "paper_ikstream", IK, (0, (4, 1.5))),
               ("ACT", "paper_act", ACT, "-"), ("SmolVLA", "paper_smolvla", VLA, "-")]


def figure_kde() -> dict:
    """KDEs over the gold set for the four controllers, all from the same-code runs in outputs/paper_eval."""
    runs = {lab: load_runs(tag, EVAL) for lab, tag, _, _ in CONTROLLERS}
    fin = {lab: df[df.finished] for lab, df in runs.items()}
    samples = {lab: per_sample_errors(tag, set(fin[lab].drawing), EVAL) for lab, tag, _, _ in CONTROLLERS}
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.3), gridspec_kw={"wspace": 0.14})
    spec = [(lab, c, st) for lab, _, c, st in CONTROLLERS]
    kde_panel(axes[0], [(l, fin[l].rmse_mm, c, st) for l, c, st in spec], "Per-drawing path RMSE (mm)", 0.42,
              "(a) Path RMSE per drawing")
    meds = kde_panel(axes[1], [(l, samples[l], c, st) for l, c, st in spec], "Per-sample nearest-path error (mm, log)",
                     2.0, "(b) Pen-down samples", log=True)
    kde_panel(axes[2], [(l, fin[l].jerk, c, st) for l, c, st in spec], r"Pen-down joint jerk RMS (rad s$^{-3}$)", 1300,
              "(c) Joint jerk per drawing")
    axes[0].plot([], [], color=MUTED, linestyle=(0, (1, 1.5)), linewidth=0.9, label="median")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.08), handlelength=2.0)
    fig.savefig(OUT / "kde_gold.pdf")
    plt.close(fig)
    out = {}
    for (lab, _, _, _), med in zip(CONTROLLERS, meds):
        out[lab] = {"n_samples": int(len(samples[lab])), "sample_median": float(med),
                    "sample_p95": float(np.percentile(samples[lab], 95))}
    return out


def figure_paths(drawing: str, category: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(FULL_W, 2.2), gridspec_kw={"width_ratios": [1, 1, 0.62, 0.62], "wspace": 0.08})
    zoom = None
    for i, (tag, label, color) in enumerate([("paper_act", "ACT", ACT), ("paper_ik", "IK", IK)]):
        d = EVAL / tag / category / drawing
        des = pd.read_csv(d / "desired_path.csv")
        ex = pd.read_csv(d / "executed_path.csv")
        m = json.loads((d / "run_metrics.json").read_text())["desired_to_executed_nearest_path_error"]
        des = des[des.state.str.startswith("DRAW")]
        ex = ex[ex.state.str.startswith("DRAW")]
        cx, cy = des.x_m.mean(), des.y_m.mean()
        for ax in (axes[i], axes[2 + i]):
            for _, s in des.groupby("stroke_id"):
                ax.plot((s.x_m - cx) * 1e3, (s.y_m - cy) * 1e3, color="#9a9994", linewidth=2.2, solid_capstyle="round")
            for _, s in ex.groupby("stroke_id"):
                ax.plot((s.tip_x_m - cx) * 1e3, (s.tip_y_m - cy) * 1e3, color=color, linewidth=0.7)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        if zoom is None:
            # Zoom on the head (eye, nose and ears: the tightest curves), a fixed feature rather than a worst case.
            x, y = (des.x_m - cx) * 1e3, (des.y_m - cy) * 1e3
            head = x < x.min() + 2 * ZOOM_MM
            zx, zy = x.min() + ZOOM_MM / 2, float(np.median(y[head]))
            zoom = (zx - ZOOM_MM / 2, zx + ZOOM_MM / 2, zy - ZOOM_MM / 2, zy + ZOOM_MM / 2)
        axes[i].add_patch(Rectangle((zoom[0], zoom[2]), ZOOM_MM, ZOOM_MM, fill=False, edgecolor=INK, linewidth=0.6))
        axes[2 + i].set_xlim(zoom[0], zoom[1])
        axes[2 + i].set_ylim(zoom[2], zoom[3])
        for sp in axes[2 + i].spines.values():
            sp.set_visible(True)
            sp.set_color(MUTED)
            sp.set_linewidth(0.6)
        axes[i].set_title(f"({'ab'[i]}) {label}: RMSE {m['rmse_m'] * 1e3:.3f} mm", loc="left")
        axes[2 + i].set_title(f"({'cd'[i]}) {label}, head, {ZOOM_MM:.0f} mm", loc="left")
    axes[0].plot([], [], color="#9a9994", linewidth=2.2, label="planned")
    axes[0].plot([], [], color=ACT, linewidth=0.9, label="ACT pen tip")
    axes[0].plot([], [], color=IK, linewidth=0.9, label="IK pen tip")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="lower center", ncol=3, fontsize=8,
               handlelength=1.6, bbox_to_anchor=(0.5, 0.02))
    fig.savefig(OUT / "paths_cat.pdf")
    plt.close(fig)


def figure_twin_real() -> None:
    classes = ["Bird", "Cat", "Dog", "Fish", "Frog", "Horse"]
    twin = [0.192, 0.197, 0.194, 0.198, 0.187, 0.203]  # act_real_v2_stiff_results.md, per category
    real = [0.579, 0.488, 0.536, 0.549, 0.480, 0.559]  # same file, physical arm batch 2026-10-27
    lo = [0.549, 0.456, 0.484, 0.507, 0.421, 0.442]
    hi = [0.594, 0.539, 0.587, 0.595, 0.569, 0.622]
    fig, ax = plt.subplots(figsize=(COL_W, 2.15))
    y = np.arange(len(classes))[::-1]
    for yi, t, r, a, b in zip(y, twin, real, lo, hi):
        ax.plot([t, r], [yi, yi], color=GRID, linewidth=1.0, zorder=1)
        ax.plot([a, b], [yi, yi], color=ACT, linewidth=2.6, alpha=0.35, solid_capstyle="round", zorder=2)
    ax.scatter(twin, y, s=26, color=ACT, zorder=3, label="Twin (48 gold drawings)")
    ax.scatter(real, y, s=26, facecolor="white", edgecolor=ACT, linewidth=1.2, marker="s", zorder=3,
               label="Physical arm (5 per class), bar = range")
    for yi, t, r in zip(y, twin, real):
        ax.text(0.665, yi, f"{r / t:.1f}x", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(classes)
    ax.set_xlim(0.1, 0.72)
    ax.text(0.665, y[0] + 0.75, "real/twin", fontsize=8, color=MUTED)
    ax.set_xlabel("ACT path RMSE (mm)")
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=2, fontsize=8, handletextpad=0.3,
              columnspacing=1.0)
    fig.savefig(OUT / "twin_vs_real.pdf")
    plt.close(fig)


def figure_photos() -> None:
    isaac = Image.open(REPO / "docs/readme/isaac.jpg").convert("RGB")
    isaac.crop((24, 60, 664, 420)).save(OUT / "twin_scene.jpg", quality=92)
    isaac.crop((720, 60, 1360, 420)).save(OUT / "twin_closeup.jpg", quality=92)
    real = Image.open(REPO / "paper/figures/src/real_bird_frame.jpg").convert("RGB")
    real.crop((0, 120, 464, 740)).save(OUT / "real_arm.jpg", quality=92)
    face = Image.open(RESULTS / "portrait_hardware_eval/20261102/07_face_e67870b5_act_markerB_topdown.png").convert("RGB")
    face.rotate(-90, expand=True).crop((140, 130, 350, 390)).save(OUT / "real_face.jpg", quality=92)


def robustness_table(ik: pd.DataFrame, act: pd.DataFrame) -> None:
    print("\nRobustness (16 drawings): placement, controller, finished, mean RMSE, mean p95")
    names = None
    for place, suffix in [("nominal", None), ("12 mm lower", "zlow"), ("12 mm higher", "zhigh"), ("(+15,-15) mm", "xy")]:
        for ctrl, base in (("IK", ik), ("ACT", act)):
            df = load_runs(f"robust_{ctrl.lower()}_{suffix}") if suffix else None
            if df is None:
                names = names or set(load_runs("robust_act_xy").drawing)
                df = base[base.drawing.isin(names)]
            f = df[df.finished]
            print(f"  {place:14s} {ctrl:4s} {len(f)}/{len(df)}  {f.rmse_mm.mean():.3f}  {f.p95_mm.mean():.3f}  "
                  f"jerk {f.jerk.mean():.0f}")


def gold_set_stats() -> None:
    strokes, points, lengths = [], [], []
    for p in (REPO / "isaac-sim/data/act_test").glob("*/*.json"):
        t = json.loads(p.read_text())
        st = t["strokes"]
        strokes.append(len(st))
        pts = [np.asarray([[q["x"], q["y"]] for q in s["points"]], float) for s in st]
        points.append(sum(len(x) for x in pts))
        lengths.append(sum(np.linalg.norm(np.diff(x, axis=0), axis=1).sum() for x in pts if len(x) > 1))
    print(f"\nGold set: {len(strokes)} drawings, strokes {np.mean(strokes):.1f} (range {min(strokes)}-{max(strokes)}), "
          f"points {np.mean(points):.0f}, normalized length {np.mean(lengths):.2f}")


def main() -> None:
    ik, act = load_runs("ik_real_stiff"), load_runs("act_real_v2_stiff")
    for name, df in (("IK", ik), ("ACT", act)):
        f = df[df.finished]
        print(f"{name}: {len(f)}/{len(df)} rmse {f.rmse_mm.mean():.3f} p95 {f.p95_mm.mean():.3f} max {f.max_mm.mean():.3f} "
              f"corner {f.corner_mm.mean():.3f} heading med {f.heading_med_deg.mean():.2f} jerk {f.jerk.mean():.0f} "
              f"sim {f.sim_s.mean():.1f}")
    print(json.dumps(figure_kde(), indent=1))
    figure_paths("1163bceaf6134cf5", "Cat")
    figure_twin_real()
    figure_photos()
    robustness_table(ik, act)
    gold_set_stats()


if __name__ == "__main__":
    main()


def figure_speed() -> None:
    """Speed sweep, results/speed/README.md (cat 323bd30d, MCP runner, lab-arm twin)."""
    act = [(0.5, 48.7, 0.206), (0.75, 37.1, 0.219), (1.0, 30.3, 0.228), (1.5, 22.0, 0.318)]
    vla = [(0.5, 49.8, 0.252), (1.0, 29.8, 0.300), (1.5, 21.3, 0.514)]
    fig, ax = plt.subplots(figsize=(COL_W, 2.25))
    for pts, color, label, offs in ((act, ACT, "ACT", (2, -11)), (vla, VLA, "SmolVLA (CUDA graph)", (3, 5))):
        t, e = [p[1] for p in pts], [p[2] for p in pts]
        ax.plot(t, e, color=color, marker="o", markersize=4.5, label=label)
        for s, x, y in pts:
            ax.annotate(f"{s:g}x", (x, y), textcoords="offset points", xytext=offs, fontsize=8, color=MUTED)
    ax.annotate("ACT at 2.0x: overshoot,\nsafety stop after 5.4 s", xy=(33, 0.45), fontsize=8, color=MUTED, ha="left")
    ax.set_xlabel("Simulated drawing time (s)")
    ax.set_ylabel("Path RMSE (mm)")
    ax.set_xlim(17, 54)
    ax.set_ylim(0.15, 0.6)
    ax.grid(color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(OUT / "speed.pdf")
    plt.close(fig)


def motion_stats() -> None:
    for tag in ("ik_real_stiff", "act_real_v2_stiff"):
        ms = [json.loads(p.read_text()) for p in (RESULTS / "act_eval" / tag).glob("*/*/run_metrics.json")]
        ms = [m for m in ms if m.get("status") == "finished"]
        print(f"{tag}: settled {np.mean([m['velocity_settled_fraction'] for m in ms]):.3f}, "
              f"median tip speed {np.mean([m['tip_speed_mm_s_stats']['median'] for m in ms]):.1f} mm/s, "
              f"exec->plan rmse {np.mean([m['executed_to_desired_nearest_path_error']['rmse_m'] for m in ms]) * 1e3:.3f}")


if __name__ == "__main__":
    figure_speed()
    motion_stats()


PAPER_FACES = ["58265341041a89bb", "fb1b8bcaf2415905", "cdb9aedb5778896b", "f4d9516ce50ff090", "e00984bb56fa1ba6"]
REAL_FACES_DIR = REPO / "isaac-sim" / "results" / "portrait_hardware_eval" / "paper_faces"


def figure_faces() -> None:
    """Source photo, generated line art and (when present) the sheet drawn by the physical arm, per face.

    Real drawings are read from REAL_FACES_DIR/<sample_id>.(jpg|png); a face without one keeps an empty slot.
    """
    work = REPO / "outputs" / "faces_source" / "work"
    real = {f: next((REAL_FACES_DIR / f"{f}{ext}" for ext in (".jpg", ".png") if (REAL_FACES_DIR / f"{f}{ext}").exists()), None)
            for f in PAPER_FACES}
    rows = [("Photo", "source.png"), ("Line art", "lineart.png")] + ([("Physical arm", "real")] if any(real.values()) else [])
    fig, axes = plt.subplots(len(rows), len(PAPER_FACES), figsize=(FULL_W, 1.45 * len(rows)),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.06})
    for c, face in enumerate(PAPER_FACES):
        for r, (label, name) in enumerate(rows):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            path = real[face] if name == "real" else work / face / name
            if path is not None and Path(path).exists():
                im = Image.open(path).convert("RGB")
                im.thumbnail((700, 700))
                ax.imshow(np.asarray(im))
            ax.set_aspect("equal")
            if c == 0:
                ax.set_ylabel(label, color=INK)
    fig.savefig(OUT / "faces.pdf", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    figure_faces()
