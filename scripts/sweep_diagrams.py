#!/usr/bin/env python3
"""Sweep the diagram pipeline across all silver subjects and build a contact sheet.

Reads samples from outputs/silver/checkpoints/silver_subjects.ndjson — the same manifest
that the pipeline stages use. Uses the configured outline_rectifier from configs/models.yaml.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from wildtrace.diagram import build_outline_rectifier


def load_subjects(repo_root: Path) -> list[dict]:
    """Load accepted subjects from the silver checkpoint manifest."""
    manifest = repo_root / "outputs" / "silver" / "checkpoints" / "silver_subjects.ndjson"
    if not manifest.exists():
        raise FileNotFoundError(
            f"Silver subjects manifest not found: {manifest}\n"
            "Run enrich_and_crop_subjects.py first."
        )
    return [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep diagram pipeline across silver subjects.")
    parser.add_argument("--limit", type=int, default=None, help="Max number of subjects to process.")
    args = parser.parse_args()

    OUT = REPO_ROOT / "outputs" / "flux_sweep"
    OUT.mkdir(parents=True, exist_ok=True)

    with open(REPO_ROOT / "configs" / "models.yaml") as f:
        config = yaml.safe_load(f)

    print("Initializing OutlineRectifier pipeline...")
    rectifier = build_outline_rectifier(config["outline_rectifier"])

    subjects = load_subjects(REPO_ROOT)
    if args.limit:
        subjects = subjects[: args.limit]
    print(f"Processing {len(subjects)} subjects.")

    results = []

    for subj in subjects:
        sid = subj["sample_id"]
        species = subj["category"]
        crop_path = REPO_ROOT / subj["crop_path"]
        mask_path = REPO_ROOT / subj["mask_path"]

        if not crop_path.exists() or not mask_path.exists():
            print(f"SKIP {species}/{sid} — missing files")
            continue

        print(f"Processing {species}/{sid}...")

        crop = Image.open(crop_path)
        mask = Image.open(mask_path)
        destination = OUT / f"{species}_{sid}_final.png"

        result = rectifier.run(
            sample=subj,
            subject_image=crop,
            subject_mask=mask,
            generated_path=None,
            destination=destination,
            params={},
        )

        final = Image.open(result.diagram_path)
        results.append((species, sid, crop, final))
        print(f"  ✓ Saved {destination.name}")

    if not results:
        print("No results to show.")
        return

    # --- Contact sheet ---
    print("\nBuilding contact sheet...")
    N = len(results)
    CELL = 300
    PADDING = 10
    HEADER = 40
    cols = N
    sheet_w = cols * (CELL + PADDING) + PADDING
    sheet_h = 2 * (CELL + PADDING) + PADDING + HEADER  # row 0: crop, row 1: final
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)

    for i, (species, sid, crop, final) in enumerate(results):
        x = PADDING + i * (CELL + PADDING)
        draw.text((x + 5, 5), species, fill="black")

        for row_idx, img in enumerate([crop, final]):
            y = HEADER + PADDING + row_idx * (CELL + PADDING)
            thumb = img.convert("RGB")
            thumb.thumbnail((CELL, CELL), Image.LANCZOS)
            ox = x + (CELL - thumb.width) // 2
            oy = y + (CELL - thumb.height) // 2
            sheet.paste(thumb, (ox, oy))

    sheet_dest = OUT / "contact_sheet.png"
    sheet.save(sheet_dest)
    print(f"Done! Contact sheet: {sheet_dest}")


if __name__ == "__main__":
    main()
