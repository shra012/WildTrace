"""Fine-tune SmolVLA (lerobot/smolvla_base) on the ACT drawing datasets.

    isaac-sim/.venv-act/bin/python isaac-sim/scripts/train_smolvla.py \
        --dataset wildtrace_real_kin_v2_train --val-dataset wildtrace_real_kin_v2_val \
        --output outputs/smolvla/real_kinematic_v2 --steps 20000

The datasets are the ones ACT trains on (scripts/build_lerobot_dataset.py).
Each frame's SmolVLA observation is built from them on the fly by
src/vla_features.py: a 28-value state and a rendered top-down view of the
upcoming path, plus the dataset's task string. Same frames, same actions
(7 joint steps from the previous command), so ACT and SmolVLA compare like
for like.

LeRobot's own trainer needs the image in the dataset; rendering here instead
avoids a second 20M-frame dataset with video. Checkpoints are written like
lerobot-train's (<output>/checkpoints/<step>/pretrained_model and last ->),
so scripts/act_policy_server.py --policy smolvla serves them.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import smolvla_direct_state  # noqa: E402
import vla_features  # noqa: E402

DATASETS_ROOT = PROJECT_ROOT.parent / "outputs" / "lerobot"
ACT_STATE, ACT_ENV, ACTION = "observation.state", "observation.environment_state", "action"


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="wildtrace_real_kin_v2_train")
    parser.add_argument("--val-dataset", default="wildtrace_real_kin_v2_val",
                        help='"train" validates on training frames (overfit tests with --episodes)')
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "smolvla" / "real_kinematic_v2"))
    parser.add_argument("--init", default="lerobot/smolvla_base", help="pretrained SmolVLA to start from")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=10, help="same 1/3 s chunk as ACT")
    parser.add_argument("--episodes", type=int, default=0, help="train on the first N episodes only (0 = all)")
    parser.add_argument("--episode-stride", type=int, default=1,
                        help="use one episode in N; 3 keeps one recording of each drawing (the datasets hold "
                             "3 recordings per drawing, stored next to each other: a0 clean, a1-a2 with DART noise, "
                             "servo offset and paper shift), rotating through them so all three kinds stay in")
    parser.add_argument("--image-size", type=int, default=0,
                        help="resize the path image to this square size inside the model (0 = the checkpoint's, 512); "
                             "the vision encoder dominates step time")
    parser.add_argument("--vlm-layers", type=int, default=0,
                        help="keep only the first N language-model layers (0 = the checkpoint's); fewer is faster")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--val-freq", type=int, default=2000)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resume", action="store_true", help="continue from <output>/checkpoints/last")
    parser.add_argument("--train-vlm", action="store_true",
                        help="also train the language model layers (the vision encoder stays frozen); the state "
                             "token passes through them on its way to the action expert")
    parser.add_argument("--direct-state", action="store_true",
                        help="also feed the state straight to the action expert (src/smolvla_direct_state.py)")
    parser.add_argument("--state", choices=["compact", "full"], default="compact",
                        help="full adds the 30-value path window to the state (58 values, max_state_dim 64)")
    return parser.parse_args()


def episode_subset(total: int, stride: int) -> list[int]:
    """One episode in each group of `stride` neighbours, at offset group % stride. A plain
    range(0, total, 3) would keep only each drawing's clean a0 recording and never show a recovery."""
    stride = max(1, stride)
    return [i for i in range(total) if i % stride == (i // stride) % stride]


class VlaFrames(torch.utils.data.Dataset):
    """An ACT LeRobotDataset seen as SmolVLA frames."""

    def __init__(self, name: str, chunk_size: int, episodes: int = 0, full_state: bool = False, stride: int = 1):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.full_state = full_state
        root = DATASETS_ROOT / name
        info = json.loads((root / "meta" / "info.json").read_text())
        fps = info["fps"]
        selected = episode_subset(episodes or info["total_episodes"], stride)
        self.frames = LeRobotDataset(
            f"wildtrace/{name}", root=root,
            episodes=selected if (episodes or stride > 1) else None,
            delta_timestamps={ACTION: [i / fps for i in range(chunk_size)]},
        )

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> dict:
        frame = self.frames[index]
        env = frame[ACT_ENV].numpy()
        return {
            "observation.state": torch.from_numpy(
                vla_features.vla_state(frame[ACT_STATE].numpy(), env, self.full_state)[0]),
            vla_features.IMAGE_KEY: torch.from_numpy(vla_features.image_tensor_layout(vla_features.path_view(env))),
            ACTION: frame[ACTION],
            "action_is_pad": frame["action_is_pad"],
            "task": frame.get("task", vla_features.TASK),
        }


def vla_stats(act_stats: dict, full_state: bool = False) -> dict:
    """SmolVLA normalization statistics from the ACT dataset's (state = ACT state + 7 env values,
    then the path window with `full_state`), in vla_features.vla_state's order."""
    pick = list(range(6)) + [-1] + (list(range(6, 36)) if full_state else [])
    stats = {}
    for key in ("mean", "std", "min", "max"):
        state = np.asarray(act_stats[ACT_STATE][key], dtype=np.float32).reshape(-1)
        env = np.asarray(act_stats[ACT_ENV][key], dtype=np.float32).reshape(-1)[pick]
        stats.setdefault("observation.state", {})[key] = torch.from_numpy(np.concatenate([state, env]))
        stats.setdefault(ACTION, {})[key] = torch.as_tensor(np.asarray(act_stats[ACTION][key], dtype=np.float32))
    return stats


FULL_MAX_STATE_DIM = 64


def widen_state_input(policy, max_state_dim: int, keep: int = vla_features.STATE_DIM) -> None:
    """Grow SmolVLA's state projection to max_state_dim inputs. The first `keep` columns (the
    compact state the checkpoint was trained on) keep their weights; the rest, which only ever
    saw zero padding, start at zero. The policy starts out exactly as before on the compact
    values and learns to use the added ones."""
    old = policy.model.state_proj
    if old.in_features >= max_state_dim:
        return
    new = torch.nn.Linear(max_state_dim, old.out_features, device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :keep] = old.weight[:, :keep]
        new.bias.copy_(old.bias)
    new.weight.requires_grad = new.bias.requires_grad = old.weight.requires_grad
    policy.model.state_proj = new
    policy.config.max_state_dim = max_state_dim
    policy.model.config.max_state_dim = max_state_dim


def make_policy(init: str, chunk_size: int, device: str, full_state: bool = False, train_vlm: bool = False,
                vlm_layers: int = 0, image_size: int = 0):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config = PreTrainedConfig.from_pretrained(init)
    config.input_features = {
        "observation.state": PolicyFeature(
            type=FeatureType.STATE, shape=(vla_features.FULL_STATE_DIM if full_state else vla_features.STATE_DIM,)),
        vla_features.IMAGE_KEY: PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, vla_features.IMAGE_SIZE_PX, vla_features.IMAGE_SIZE_PX)
        ),
    }
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    config.chunk_size = chunk_size
    config.n_action_steps = chunk_size
    config.device = device
    config.push_to_hub = False
    if vlm_layers > 0:
        config.num_vlm_layers = vlm_layers
    if image_size > 0:
        config.resize_imgs_with_padding = (image_size, image_size)
    if train_vlm:
        config.train_expert_only = False
        config.freeze_vision_encoder = True
    policy = SmolVLAPolicy.from_pretrained(init, config=config).to(device)
    if full_state:
        widen_state_input(policy, FULL_MAX_STATE_DIM)
    return policy, policy.config


def _batches(loader):
    while True:
        yield from loader


@torch.no_grad()
def first_action_error(policy, preprocessor, batch: dict) -> float:
    """Mean |predicted - true| first action over mean |true| (normalized), as for the ACT overfit gate."""
    batch = preprocessor(batch)
    policy.reset()
    # Zero starting noise, so checkpoints are compared without sampling luck.
    noise = torch.zeros(len(batch[ACTION]), policy.config.chunk_size, policy.config.max_action_dim,
                        device=batch[ACTION].device)
    predicted = policy.predict_action_chunk(batch, noise=noise)[:, 0]
    true = batch[ACTION][:, 0]
    policy.train()
    return float((predicted - true).abs().mean() / true.abs().mean().clamp_min(1e-6))


def save(policy, preprocessor, postprocessor, output: Path, step: int) -> None:
    target = output / "checkpoints" / f"{step:06d}" / "pretrained_model"
    target.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(target)
    if hasattr(policy.model, "state_to_expert"):
        smolvla_direct_state.mark_checkpoint(target)
    preprocessor.save_pretrained(target)
    postprocessor.save_pretrained(target)
    (target.parent / "training_state.json").write_text(json.dumps({"step": step}))
    last = output / "checkpoints" / "last"
    if last.is_symlink() or last.exists():
        last.unlink()
    last.symlink_to(target.parent.name)


def main() -> int:
    args = _parse_args()
    from lerobot.policies.factory import make_pre_post_processors

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    full_state = args.state == "full"
    train = VlaFrames(args.dataset, args.chunk_size, args.episodes, full_state, args.episode_stride)
    stats = vla_stats(train.frames.meta.stats, full_state)
    start_step = 0
    if args.resume:
        last = output / "checkpoints" / "last"
        init = str(last / "pretrained_model")
        start_step = json.loads((last / "training_state.json").read_text())["step"]
    else:
        init = args.init
    policy, config = make_policy(init, args.chunk_size, device, full_state, args.train_vlm, args.vlm_layers,
                                 args.image_size)
    if not smolvla_direct_state.load_direct_state(policy, Path(init)) and args.direct_state:
        smolvla_direct_state.add_direct_state(policy)
    if args.resume:
        preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=init)
    else:
        preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)

    trainable = [p for p in policy.parameters() if p.requires_grad]
    print(f"[OK] {len(train)} frames; {sum(p.numel() for p in trainable) / 1e6:.1f}M of "
          f"{sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M parameters train", flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=config.optimizer_betas, eps=config.optimizer_eps,
                                  weight_decay=config.optimizer_weight_decay)

    def lr_factor(step: int) -> float:  # linear warm-up, cosine decay to 2.5 % (SmolVLA's preset shape)
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        progress = min(1.0, (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps))
        return 0.025 + 0.975 * 0.5 * (1 + math.cos(math.pi * progress))

    for group in optimizer.param_groups:  # LambdaLR needs this to start part-way through (--resume)
        group.setdefault("initial_lr", args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor, last_epoch=start_step - 1 if start_step else -1)
    loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                         pin_memory=device == "cuda", drop_last=True, persistent_workers=True)
    val_batch = None
    if args.val_dataset:
        val = train if args.val_dataset == "train" else VlaFrames(args.val_dataset, args.chunk_size,
                                                                     full_state=full_state)
        generator = torch.Generator().manual_seed(args.seed)
        val_batch = next(iter(torch.utils.data.DataLoader(val, batch_size=64, shuffle=True, generator=generator)))

    log = (output / "train_log.ndjson").open("a")
    policy.train()
    batches = _batches(loader)
    started, load_time = time.perf_counter(), 0.0
    for step in range(start_step, args.steps):
        load_started = time.perf_counter()
        batch = preprocessor(next(batches))
        load_time += time.perf_counter() - load_started
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            loss, _ = policy.forward(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, config.optimizer_grad_clip_norm)
        optimizer.step()
        scheduler.step()
        done = step + 1
        record = None
        if done % args.log_freq == 0:
            elapsed = time.perf_counter() - started
            record = {"step": done, "loss": float(loss), "grad_norm": float(grad_norm),
                      "lr": scheduler.get_last_lr()[0], "samples_s": args.log_freq * args.batch_size / elapsed,
                      "load_share": load_time / elapsed}
            started, load_time = time.perf_counter(), 0.0
        if val_batch is not None and (done % args.val_freq == 0 or done == args.steps):
            record = record or {"step": done}
            record["val_first_action_rel_error"] = first_action_error(policy, preprocessor, dict(val_batch))
        if record:
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + "\n")
            log.flush()
        if done % args.save_freq == 0 or done == args.steps:
            save(policy, preprocessor, postprocessor, output, done)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
