"""Run one ACT training stage from config/act_drawing.yaml through LeRobot's trainer.

    isaac-sim/.venv-act/bin/python isaac-sim/scripts/train_act.py --stage kinematic
    isaac-sim/.venv-act/bin/python isaac-sim/scripts/train_act.py --stage physics_finetune
    isaac-sim/.venv-act/bin/python isaac-sim/scripts/train_act.py --stage overfit_one --dry-run

Checkpoints land in <output_root>/<stage>/checkpoints/. A stage with
`init_from` starts from that stage's last pretrained_model, keeping its
normalization statistics.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "act_drawing.yaml"))
    parser.add_argument("--stage", required=True)
    parser.add_argument("--steps", type=int, default=None, help="Override the stage's step count")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Continue the stage's existing run")
    parser.add_argument("--policy", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a policy setting, e.g. --policy use_vae=false --policy chunk_size=10")
    parser.add_argument("--output-name", default=None, help="Run folder name (default: the stage name)")
    parser.add_argument("--dry-run", action="store_true", help="Print the lerobot-train command only")
    return parser.parse_args()


def _flag(name: str, value) -> str:
    if isinstance(value, bool):
        value = str(value).lower()
    elif isinstance(value, dict):
        value = json.dumps(value)
    return f"--{name}={value}"


def build_command(
    config: dict, stage_name: str, *, steps=None, batch_size=None, resume=False, overrides=(), output_name=None
) -> list[str]:
    stages = config["stages"]
    if stage_name not in stages:
        raise SystemExit(f"Unknown stage {stage_name}; choose from {sorted(stages)}")
    stage = stages[stage_name]
    datasets_root = (PROJECT_ROOT / config["datasets_root"]).resolve()
    output_dir = (PROJECT_ROOT / config["output_root"]).resolve() / (output_name or stage_name)
    train = dict(config["train"])
    if batch_size:
        train["batch_size"] = batch_size
    if sys.platform == "win32":
        # Same trainer, with checkpoints/last as a junction instead of a symlink.
        launcher = [sys.executable, str(Path(__file__).with_name("lerobot_train_windows.py"))]
    else:
        launcher = [str(Path(sys.executable).with_name("lerobot-train"))]
    command = [
        *launcher,
        _flag("dataset.repo_id", f"wildtrace/{stage['dataset']}"),
        _flag("dataset.root", datasets_root / stage["dataset"]),
        _flag("output_dir", output_dir),
        _flag("job_name", f"act_{stage_name}"),
        _flag("steps", steps or stage["steps"]),
        _flag("wandb.enable", False),
        *(_flag(key, value) for key, value in train.items()),
    ]
    policy = dict(config["policy"])
    policy.update(stage.get("policy", {}))
    for item in overrides:
        key, _, value = item.partition("=")
        policy[key] = yaml.safe_load(value)
    if "optimizer_lr" in stage:
        policy["optimizer_lr"] = stage["optimizer_lr"]
    if stage.get("init_from"):
        source = (PROJECT_ROOT / config["output_root"]).resolve() / stage["init_from"]
        pretrained = source / "checkpoints" / "last" / "pretrained_model"
        if not pretrained.is_dir():
            raise SystemExit(f"Stage {stage['init_from']} has no checkpoint yet: {pretrained}")
        # The architecture comes from the checkpoint; only runtime knobs are overridden.
        command.append(_flag("policy.path", pretrained))
        for key in ("device", "push_to_hub", "optimizer_lr", "use_amp"):
            command.append(_flag(f"policy.{key}", policy[key]))
    else:
        command += [_flag(f"policy.{key}", value) for key, value in policy.items()]
    if resume:
        command += [_flag("resume", True), _flag("config_path", output_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json")]
    return command


def main() -> int:
    args = _parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    command = build_command(
        config, args.stage, steps=args.steps, batch_size=args.batch_size, resume=args.resume,
        overrides=args.policy, output_name=args.output_name,
    )
    print(" ".join(command), flush=True)
    if args.dry_run:
        return 0
    output_dir = Path(next(c.split("=", 1)[1] for c in command if c.startswith("--output_dir=")))
    if output_dir.exists() and not args.resume:
        # lerobot-train refuses an existing output_dir without --resume.
        backup = output_dir.with_name(output_dir.name + ".previous")
        if backup.exists():
            shutil.rmtree(backup)
        output_dir.rename(backup)
        print(f"[OK] Previous run moved to {backup}")
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
