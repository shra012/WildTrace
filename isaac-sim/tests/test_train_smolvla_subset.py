"""train_smolvla's episode subset keeps every kind of recording, not only the clean one."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_smolvla import episode_subset  # noqa: E402


def test_stride_one_keeps_everything():
    assert episode_subset(7, 1) == list(range(7))


def test_one_recording_per_drawing_rotating_through_augmentations():
    # Episodes 3k, 3k+1, 3k+2 are drawing k's a0 (clean), a1 and a2 (noisy) recordings.
    selected = episode_subset(300, 3)
    assert len(selected) == 100
    assert [i // 3 for i in selected] == list(range(100))  # one per drawing
    kinds = [i % 3 for i in selected]
    assert all(kinds.count(kind) in (33, 34) for kind in range(3))
