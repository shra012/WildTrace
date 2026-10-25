"""Split trajectory JSONs into train and test sets, stratified by category, with a fixed seed.

    python isaac-sim/scripts/split_trajectories.py --source outputs/trajectories --out outputs/act_split --test-fraction 0.1

Copies <source>/<Category>/<id>.json to <out>/{train,test}/<Category>/<id>.json and
writes <out>/split.json. Label with --trajectories <out>/train --exclude-dir <out>/test
and evaluate on <out>/test, so no test drawing is ever trained on.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Folder of <Category>/<id>.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    source, out = Path(args.source), Path(args.out)
    if out.exists():
        raise SystemExit(f"[FAIL] {out} exists; remove it to re-split")
    rng = random.Random(args.seed)
    split = {"source": str(source.resolve()), "seed": args.seed, "test_fraction": args.test_fraction, "categories": {}}
    for category in sorted(p for p in source.iterdir() if p.is_dir()):
        files = sorted(category.glob("*.json"))
        rng.shuffle(files)
        test_count = max(1, round(len(files) * args.test_fraction))
        for name, chosen in (("test", files[:test_count]), ("train", files[test_count:])):
            destination = out / name / category.name
            destination.mkdir(parents=True, exist_ok=True)
            for path in chosen:
                shutil.copy2(path, destination / path.name)
        split["categories"][category.name] = {
            "train": sorted(p.stem for p in files[test_count:]),
            "test": sorted(p.stem for p in files[:test_count]),
        }
        print(f"[OK] {category.name}: {len(files) - test_count} train, {test_count} test")
    (out / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    totals = {name: sum(len(c[name]) for c in split["categories"].values()) for name in ("train", "test")}
    print(f"[OK] {totals['train']} train, {totals['test']} test -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
