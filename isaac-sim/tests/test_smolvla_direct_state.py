"""SmolVLA's direct state path (src/smolvla_direct_state.py) on a stand-in model, no LeRobot needed."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import smolvla_direct_state  # noqa: E402

WIDTH, STATE_DIM, CHUNK = 8, 6, 3


class FakeFlowModel(torch.nn.Module):
    """The parts of VLAFlowMatching the patch touches."""

    def __init__(self):
        super().__init__()
        self.vlm_with_expert = types.SimpleNamespace(expert_hidden_size=WIDTH)
        self.action_in_proj = torch.nn.Linear(4, WIDTH)
        self.seen = []

    def embed_suffix(self, noisy_actions, timestep):
        embs = self.action_in_proj(noisy_actions)
        return embs, torch.ones(embs.shape[:2], dtype=torch.bool), torch.ones(embs.shape[:2])

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):
        embs, _, _ = self.embed_suffix(actions, time)
        self.seen.append(embs)
        return embs

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None):
        embs, _, _ = self.embed_suffix(noise, None)
        return embs


def _policy():
    return types.SimpleNamespace(model=FakeFlowModel(), config=types.SimpleNamespace(max_state_dim=STATE_DIM))


def test_patched_model_starts_identical_and_then_uses_the_state():
    policy = _policy()
    actions, state = torch.randn(2, CHUNK, 4), torch.randn(2, STATE_DIM)
    before = policy.model.forward(None, None, None, None, state, actions)
    smolvla_direct_state.add_direct_state(policy)
    after = policy.model.forward(None, None, None, None, state, actions)
    assert torch.allclose(before, after)  # zero-initialized output layer
    assert policy.model._direct_state is None  # cleared after the call

    torch.nn.init.normal_(policy.model.state_to_expert[2].weight)
    moved = policy.model.sample_actions(None, None, None, None, state, noise=actions)
    other = policy.model.sample_actions(None, None, None, None, torch.zeros_like(state), noise=actions)
    assert not torch.allclose(moved, before) and not torch.allclose(moved, other)
    # The same state shift is added to every action token of a sample.
    shift = moved - before
    assert torch.allclose(shift[:, 0], shift[:, -1], atol=1e-6)


def test_patch_is_idempotent_and_marker_round_trips(tmp_path):
    policy = _policy()
    smolvla_direct_state.add_direct_state(policy)
    mlp = policy.model.state_to_expert
    smolvla_direct_state.add_direct_state(policy)
    assert policy.model.state_to_expert is mlp
    assert not smolvla_direct_state.load_direct_state(_policy(), tmp_path)  # no marker, no patch

    from safetensors.torch import save_file

    torch.nn.init.normal_(mlp[2].weight)
    save_file({f"model.state_to_expert.{k}": v.contiguous() for k, v in mlp.state_dict().items()},
              str(tmp_path / "model.safetensors"))
    smolvla_direct_state.mark_checkpoint(tmp_path)
    restored = _policy()
    assert smolvla_direct_state.load_direct_state(restored, tmp_path)
    assert torch.allclose(restored.model.state_to_expert[2].weight, mlp[2].weight)
