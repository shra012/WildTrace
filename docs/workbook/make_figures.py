#!/usr/bin/env python3
"""Figures for Workbook 1 (docs/workbook/). Run with any Python that has matplotlib:

    /home/shravan/anaconda3/bin/python3 docs/workbook/make_figures.py

Numbers are copied from the manifests and benchmark summaries cited in the
workbook text; keep the two in sync when either changes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# Reference palette (dataviz skill), validated on white: slot 1 blue, slot 2 orange.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 220,
})


def _bar_axes(ax, xmax, xlabel):
    ax.set_xlim(0, xmax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlabel, color=INK2)


def funnel() -> None:
    stages = [
        ("Fetched (latest view)", 540),
        ("Curated subjects (bronze accepted)", 186),
        ("Validated line diagrams", 132),
        ("Gold trajectories", 39),
        ("Drawn in Isaac Sim (distinct samples)", 9),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    labels = [s for s, _ in stages][::-1]
    values = [v for _, v in stages][::-1]
    bars = ax.barh(labels, values, color=BLUE, height=0.62, edgecolor="white", linewidth=2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + 8, bar.get_y() + bar.get_height() / 2, f"{value}",
                va="center", color=INK, fontsize=10)
    _bar_axes(ax, 600, "Samples")
    fig.tight_layout()
    fig.savefig(OUT / "fig_funnel.png")
    plt.close(fig)


def controller_before_after() -> None:
    names = ["bird_c3a528aa", "cat_70bc30f8", "cat_ab7beb03", "dog_96ab6db6", "fish_ba3e1657", "horse_7c54b4c7"]
    before = [1.554, 1.163, 1.495, 1.638, 1.604, 1.244]
    after = [0.315, 0.312, 0.314, 0.366, 0.394, 0.316]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    y = range(len(names))[::-1]
    h = 0.36
    b1 = ax.barh([i + h / 2 for i in y], before, height=h, color=ORANGE, edgecolor="white", linewidth=2,
                 label="Pre-baseline controller (c3408fc)")
    b2 = ax.barh([i - h / 2 for i in y], after, height=h, color=BLUE, edgecolor="white", linewidth=2,
                 label="baseline-v1 (74e32f9)")
    for bars, vals in ((b1, before), (b2, after)):
        for bar, value in zip(bars, vals):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2, f"{value:.2f}",
                    va="center", color=INK2, fontsize=8.5)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names)
    _bar_axes(ax, 1.9, "Tracking RMSE, desired → executed (mm)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig_controller_before_after.png")
    plt.close(fig)


def r2_generators() -> None:
    metrics = ["Accepted by pipeline", "First-attempt pass", "Strong-referee pass (235B)",
               "BioCLIP top-1 correct", "Exportable trajectory", "Draw time within 600 s"]
    omni = [35, 5, 30, 30, 60, 55]
    flux = [65, 45, 100, 75, 100, 90]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = range(len(metrics))[::-1]
    h = 0.36
    b1 = ax.barh([i + h / 2 for i in y], flux, height=h, color=BLUE, edgecolor="white", linewidth=2,
                 label="FLUX.1-schnell + qwen2.5vl:7b (local)")
    b2 = ax.barh([i - h / 2 for i in y], omni, height=h, color=ORANGE, edgecolor="white", linewidth=2,
                 label="OmniGen2 + qwen3-vl-8b (OpenRouter)")
    for bars, vals in ((b1, flux), (b2, omni)):
        for bar, value in zip(bars, vals):
            ax.text(bar.get_width() + 1.2, bar.get_y() + bar.get_height() / 2, f"{value}%",
                    va="center", color=INK2, fontsize=8.5)
    ax.set_yticks(list(y))
    ax.set_yticklabels(metrics)
    _bar_axes(ax, 112, "% of 20 evaluation subjects (final drawing per subject)")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig_r2_generators.png")
    plt.close(fig)


def architecture() -> None:
    """Snake layout: each lane hands off to the next with one vertical arrow."""
    fig, ax = plt.subplots(figsize=(10, 7.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 80)
    ax.axis("off")

    lanes = [
        # Title sits on the side away from the lane's incoming arrow.
        (68, 11, "SOURCES", "#f3f6fb", "left"),
        (52, 14, "BRONZE  ·  outputs/bronze", "#f6f5f1", "right"),
        (33, 17, "SILVER  ·  outputs/silver  ·  agentic LangGraph loop", "#f3f6fb", "left"),
        (17, 14, "GOLD  ·  outputs/gold", "#f6f5f1", "right"),
        (1, 14, "EXECUTION  ·  Windows · Isaac Sim 6.0.1 · xArm7  ·  driven over MCP", "#f3f6fb", "left"),
    ]
    for y0, hgt, title, fill, side in lanes:
        ax.add_patch(FancyBboxPatch((0.5, y0), 99, hgt, boxstyle="round,pad=0,rounding_size=1.2",
                                    facecolor=fill, edgecolor=AXIS, linewidth=0.8))
        ax.text(1.8 if side == "left" else 98.2, y0 + hgt - 1.2, title, fontsize=8.5, color=INK2,
                fontweight="bold", va="top", ha=side)

    def box(x, y, w, h, text, model=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                    facecolor="white", edgecolor=BLUE, linewidth=1.3))
        ax.text(x + w / 2, y + h / 2 + (1.0 if model else 0), text, ha="center", va="center", fontsize=8.2, color=INK)
        if model:
            ax.text(x + w / 2, y + h / 2 - 1.6, model, ha="center", va="center", fontsize=7, color=ORANGE, style="italic")
        return (x, y, w, h)

    def pt(bx, side):
        x, y, w, h = bx
        return {"r": (x + w, y + h / 2), "l": (x, y + h / 2), "b": (x + w / 2, y), "t": (x + w / 2, y + h)}[side]

    def drop(a, b):
        """Vertical hand-off between lanes, at the middle of the two boxes' overlap."""
        x = (max(a[0], b[0]) + min(a[0] + a[2], b[0] + b[2])) / 2
        ax.add_patch(FancyArrowPatch((x, a[1]), (x, b[1] + b[3]), arrowstyle="-|>", mutation_scale=10,
                                     color=INK2, linewidth=1.0, shrinkA=1, shrinkB=1))

    def arrow(a, b, side_a, side_b, color=INK2):
        ax.add_patch(FancyArrowPatch(pt(a, side_a), pt(b, side_b), arrowstyle="-|>", mutation_scale=10,
                                     color=color, linewidth=1.0, shrinkA=1, shrinkB=1))

    src1 = box(4, 69.2, 27, 6, "Open Images v7", "images + instance masks (CC BY 2.0)")
    box(36.5, 69.2, 27, 6, "configs/*.yaml + .env", "categories, thresholds, model choice")
    box(69, 69.2, 27, 6, "Model hubs", "Hugging Face · Ollama · OpenRouter")

    # Bronze: left -> right
    b1 = box(4, 53.5, 22, 7, "fetch_openimages", "versioned ledger, 540 latest")
    b2 = box(33, 53.5, 28, 7, "curate_bronze", "BioCLIP · viewpoint scorer")
    b3 = box(69, 53.5, 27, 7, "curated subjects", "186 accepted")
    # Silver: right -> left
    s1 = box(77, 34.5, 19, 7, "normalize + crop")
    s2 = box(52, 34.5, 21, 7, "generate diagram", "FLUX.1-schnell img2img")
    s3 = box(26, 34.5, 22, 7, "validate", "OpenCV prescreen + VLM judge")
    s4 = box(4, 34.5, 18, 7, "select by angle", "132 → 39 picks")
    # Gold: left -> right
    g1 = box(4, 18.5, 27, 7, "extract_trajectories", "≤ 6 strokes, ≤ 128 pts each")
    g2 = box(36.5, 18.5, 27, 7, "export_gold_ndjson", "39 gold records")
    g3 = box(69, 18.5, 27, 7, "gold trajectories", "JSON · SVG · PNG")
    # Execution: right -> left
    e1 = box(74, 2.5, 22, 8, "provision_and_draw", "category → job.json")
    e2 = box(49, 2.5, 21, 8, "ScriptNode controller", "Lula IK · state machine")
    e3 = box(26.5, 2.5, 18.5, 8, "run_metrics.json", "RMSE · time · provenance")
    e4 = box(4, 2.5, 18.5, 8, "plots + capture", "desired vs executed")

    drop(src1, b1)
    arrow(b1, b2, "r", "l"); arrow(b2, b3, "r", "l")
    drop(b3, s1)
    arrow(s1, s2, "l", "r"); arrow(s2, s3, "l", "r"); arrow(s3, s4, "l", "r")
    ax.annotate("", xy=(62.5, 41.5), xytext=(37, 41.5),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, linewidth=1.0, connectionstyle="arc3,rad=-0.4"))
    ax.text(49.8, 43.4, "retry, ≤ 5 attempts", ha="center", fontsize=7, color=ORANGE)
    drop(s4, g1)
    arrow(g1, g2, "r", "l"); arrow(g2, g3, "r", "l")
    drop(g3, e1)
    arrow(e1, e2, "l", "r"); arrow(e2, e3, "l", "r"); arrow(e3, e4, "l", "r")

    fig.savefig(OUT / "fig_architecture.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    funnel()
    controller_before_after()
    r2_generators()
    architecture()
    print("\n".join(sorted(p.name for p in OUT.glob("*.png"))))
