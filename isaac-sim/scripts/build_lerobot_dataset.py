"""Convert format-version-2 demonstration NPZs into LeRobot datasets for ACT.

Runs in the LeRobot environment (isaac-sim/.venv-act):

    isaac-sim/.venv-act/bin/python isaac-sim/scripts/build_lerobot_dataset.py \
        --inputs "outputs/act_demos/kinematic/*/*.npz" --name wildtrace_kin_v1

Writes <root>/<name>_train and <root>/<name>_val. Every augmentation of a
drawing lands in the same split (grouped by sample_id, stratified by
category), so validation measures generalization to unseen drawings.
Frames are subsampled to --fps from the recording rate (60 Hz physics,
30 Hz kinematic labels). Features come from act_features.act_frames, the
same code the closed-loop runner uses.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from act_features import ACTION_NAMES, STATE_NAMES, act_frames, environment_state_names  # noqa: E402

TASK = "trace the upcoming pen path"


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="NPZ globs")
    parser.add_argument("--name", required=True)
    parser.add_argument("--root", default=str(REPO_ROOT / "outputs" / "lerobot"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--no-split", action="store_true", help="Write a single <name> dataset (test sets)")
    parser.add_argument("--exclude-dir", default=str(PROJECT_ROOT / "data" / "act_test"))
    parser.add_argument("--limit", type=int, default=0, help="Only the first N episodes (smoke tests)")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _features(env_dim: int) -> dict:
    return {
        "observation.state": {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES},
        "observation.environment_state": {
            "dtype": "float32",
            "shape": (env_dim,),
            "names": environment_state_names((env_dim - 7) // 3),
        },
        "action": {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES},
    }


def _episode_info(path: Path) -> dict:
    with np.load(path) as data:
        metadata = json.loads(str(data["recording_metadata_json"])) if "recording_metadata_json" in data.files else {}
        drawing_id = str(data["drawing_id"][0])
    return {
        "path": str(path),
        "sample_id": metadata.get("sample_id", drawing_id),
        "category": metadata.get("category", path.parent.name),
        "source": metadata.get("source", "unknown"),
    }


def _is_val(sample_id: str, fraction: float) -> bool:
    bucket = int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < fraction


def _split(episodes: list[dict], fraction: float) -> dict[str, list[dict]]:
    """Hash split per category so each category keeps roughly the same val share."""
    by_category: dict[str, list[dict]] = defaultdict(list)
    for episode in episodes:
        by_category[episode["category"]].append(episode)
    splits: dict[str, list[dict]] = {"train": [], "val": []}
    for category, items in sorted(by_category.items()):
        samples = sorted({e["sample_id"] for e in items})
        val_samples = {s for s in samples if _is_val(s, fraction)}
        if not val_samples and len(samples) > 1:
            val_samples = {samples[0]}
        for episode in items:
            splits["val" if episode["sample_id"] in val_samples else "train"].append(episode)
    return splits


def _subsample(data: dict, fps: int) -> dict:
    times = np.asarray(data["simulation_time_s"], dtype=np.float64)
    step = float(np.median(np.diff(times))) if len(times) > 1 else 1.0 / fps
    stride = max(1, int(round((1.0 / fps) / step)))
    return {key: value[::stride] if getattr(value, "ndim", 0) >= 1 and len(value) == len(times) else value
            for key, value in data.items()}


def _write(dataset_root: Path, repo_id: str, episodes: list[dict], fps: int) -> dict:
    from lerobot.datasets import dataset_writer
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # embed_images walks every row through datasets.map (~650 rows/s) even when
    # there are no image features, which made conversion take hours. These
    # datasets are state-only by construction, so it is a no-op here.
    dataset_writer.embed_images = lambda dataset: dataset

    first = act_frames(_subsample(dict(np.load(episodes[0]["path"])), fps))
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=_features(first["observation.environment_state"].shape[1]),
        root=dataset_root,
        robot_type="xarm7_pen",
        use_videos=False,
    )
    frames_total = 0
    started = time.time()
    for number, episode in enumerate(episodes, start=1):
        frames = act_frames(_subsample(dict(np.load(episode["path"])), fps))
        size = len(frames["action"])
        dataset.save_episode(
            episode_data={
                "size": size,
                "task": [TASK] * size,
                "episode_index": dataset.meta.total_episodes,
                "frame_index": np.arange(size, dtype=np.int64),
                "timestamp": (np.arange(size, dtype=np.float64) / fps).astype(np.float32),
                "index": [],
                "task_index": [],
                **frames,
            }
        )
        frames_total += size
        if number % 200 == 0 or number == len(episodes):
            print(f"  {repo_id}: {number}/{len(episodes)} episodes, {frames_total} frames, "
                  f"{number / max(time.time() - started, 1e-9):.1f} ep/s", flush=True)
    dataset.finalize()
    return {"episodes": len(episodes), "frames": frames_total, "root": str(dataset_root)}


def main() -> int:
    args = _parse_args()
    paths = sorted({Path(p) for pattern in args.inputs for p in glob.glob(pattern)})
    excluded = {p.stem for p in Path(args.exclude_dir).rglob("*.json")} if args.exclude_dir and not args.no_split else set()
    episodes = [info for info in map(_episode_info, paths) if info["sample_id"] not in excluded]
    if args.limit:
        episodes = episodes[: args.limit]
    if not episodes:
        raise SystemExit("No episodes matched --inputs")
    print(f"[OK] {len(episodes)} episodes from {len({e['sample_id'] for e in episodes})} drawings "
          f"({len(paths) - len(episodes)} skipped as gold/limit)")
    splits = {args.name: episodes} if args.no_split else {
        f"{args.name}_{split}": items for split, items in _split(episodes, args.val_fraction).items()
    }
    root = Path(args.root)
    summary = {}
    for name, items in splits.items():
        destination = root / name
        if destination.exists():
            if not args.overwrite:
                raise SystemExit(f"{destination} exists; pass --overwrite to rebuild it")
            shutil.rmtree(destination)
        summary[name] = _write(destination, f"wildtrace/{name}", items, args.fps)
        summary[name]["sample_ids"] = sorted({e["sample_id"] for e in items})
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{args.name}_splits.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "sample_ids"} for k, v in summary.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
