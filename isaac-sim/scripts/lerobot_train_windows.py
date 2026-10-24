"""Run lerobot-train on Windows, where creating checkpoints/last as a symlink needs admin or Developer Mode.

A directory junction needs neither and resolves the same way for resume and
for the pretrained_model paths the rest of this project reads. train_act.py
launches this instead of lerobot-train.exe on Windows; arguments pass through.
"""
from __future__ import annotations

import _winapi
import os
from pathlib import Path

import lerobot.scripts.lerobot_train as lerobot_train
from lerobot.common.train_utils import LAST_CHECKPOINT_LINK


def update_last_checkpoint(checkpoint_dir: Path) -> Path:
    last = Path(checkpoint_dir).parent / LAST_CHECKPOINT_LINK
    if last.is_junction() or last.is_symlink():
        os.rmdir(last) if last.is_junction() else last.unlink()
    _winapi.CreateJunction(str(Path(checkpoint_dir).resolve()), str(last))
    return last


lerobot_train.update_last_checkpoint = update_last_checkpoint

if __name__ == "__main__":
    lerobot_train.main()
