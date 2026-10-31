"""DAgger round 2 (#46): picking drawings for the labeller and mixing and weighting its recordings in training."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from label_kinematic_ik import select_sources  # noqa: E402


def _paths(*stems: str) -> list[Path]:
    return [Path("trajectories") / "Cat" / f"{stem}.json" for stem in stems]


def test_only_keeps_listed_stems_in_order(tmp_path):
    only = tmp_path / "only.txt"
    only.write_text("c\na\n\nzz\n")
    assert select_sources(_paths("a", "b", "c", "d"), str(only)) == _paths("a", "c")


def test_only_then_every_then_limit(tmp_path):
    only = tmp_path / "only.txt"
    only.write_text("a b c d e")
    sources = _paths("a", "b", "c", "d", "e", "f")
    assert select_sources(sources, str(only), every=2) == _paths("a", "c", "e")
    assert select_sources(sources, str(only), every=2, limit=2) == _paths("a", "c")
    assert select_sources(sources) == sources


def _train_smolvla():
    pytest.importorskip("torch")
    import train_smolvla

    return train_smolvla


def test_sample_shares_split_extra_by_frame_count():
    shares = _train_smolvla().sample_shares(0.3, [1000, 300, 100])
    assert shares == pytest.approx([0.7, 0.225, 0.075])
    assert _train_smolvla().sample_shares(0.3, [1000]) == [1.0]


def _history(tip: np.ndarray, target: np.ndarray, starts: np.ndarray) -> SimpleNamespace:
    # History rows as vla_features.history_rows: tip xy, target xy, ink, target on paper.
    rows = np.zeros((len(tip), 6), np.float32)
    rows[:, 0:2], rows[:, 2:4], rows[:, 4], rows[:, 5] = tip, target, 1.0, 1.0
    return SimpleNamespace(history=rows, starts=starts)


def test_retrace_flags_a_path_running_back_over_fresh_ink():
    retrace_frames = _train_smolvla().retrace_frames
    # Out 20 mm along x and back over the same line, 0.5 mm per frame, target 1 mm ahead of the tip.
    out = np.stack([np.arange(40) * 0.0005, np.zeros(40)], 1)
    tip = np.concatenate([out, out[::-1]])
    heading = np.r_[np.ones(40), -np.ones(40)]
    target = tip + np.stack([0.001 * heading, np.zeros(80)], 1)
    flags = retrace_frames(_history(tip, target, np.zeros(80, np.int64)), lag=10)
    assert not flags[:40].any()  # on the way out the target is past the ink
    assert flags[50:78].all()  # on the way back the target sits on ink laid more than 10 frames ago


def test_retrace_ignores_a_straight_line_and_skipped_episodes():
    retrace_frames = _train_smolvla().retrace_frames
    tip = np.stack([np.arange(60) * 0.0005, np.zeros(60)], 1)
    target = tip + [0.001, 0.0]
    starts = np.r_[np.zeros(30, np.int64), np.full(30, 30, np.int64)]
    assert not retrace_frames(_history(tip, target, starts), lag=5).any()
    # A hairpin in the second episode only counts when that episode is searched.
    tip[30:] = np.concatenate([tip[0:15], tip[0:15][::-1]])
    target[30:] = tip[30:] + np.stack([0.001 * np.r_[np.ones(15), -np.ones(15)], np.zeros(30)], 1)
    dataset = _history(tip, target, starts)
    assert retrace_frames(dataset, lag=5)[50:58].all()
    assert not retrace_frames(dataset, lag=5, episodes=np.array([0])).any()
