"""Plot only the pen-down tip path from an Isaac Sim executed_path.csv."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    strokes: dict[int, list[tuple[float, float]]] = defaultdict(list)
    order: list[int] = []
    with Path(args.csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["pen_down"] != "1":
                continue
            stroke_id = int(row["stroke_id"])
            if stroke_id not in strokes:
                order.append(stroke_id)
            strokes[stroke_id].append((float(row["tip_x_m"]), float(row["tip_y_m"])))
    figure, axis = plt.subplots(figsize=(10, 6.2), facecolor="white")
    for stroke_id in order:
        points = np.asarray(strokes[stroke_id])
        if len(points) < 2:
            continue
        axis.plot(points[:, 0], points[:, 1], color="black", linewidth=1.6, solid_capstyle="round", solid_joinstyle="round")
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")
    figure.tight_layout(pad=0.2)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
