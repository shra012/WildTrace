"""Give SmolVLA's action expert the robot state directly.

Stock SmolVLA turns the state into one token of the VLM prefix. With the VLM
frozen (train_expert_only), that token passes through 16 frozen layers among
~100 image and language tokens before the action expert can attend to it, and
on a single drawing the policy could not fit the label closer than ~0.3
relative first-action error, whatever the state held.

add_direct_state() adds a small MLP from the (padded) state to the expert's
width and adds its output to every action token in embed_suffix, so the
expert sees the exact numbers at every denoising step. Its last layer starts
at zero: the patched policy starts out identical to the checkpoint it was
loaded from.

The MLP's weights are saved in model.safetensors with the rest of the policy;
a marker file next to it tells loaders to patch before restoring them
(load_direct_state). Needs torch only on top of LeRobot.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

MARKER_FILE = "wildtrace_direct_state.json"
_PREFIX = "model.state_to_expert."


def add_direct_state(policy) -> None:
    """Patch a SmolVLAPolicy in place (no-op if already patched)."""
    model = policy.model
    if hasattr(model, "state_to_expert"):
        return
    width = model.vlm_with_expert.expert_hidden_size
    reference = model.action_in_proj.weight
    mlp = nn.Sequential(nn.Linear(policy.config.max_state_dim, width), nn.SiLU(), nn.Linear(width, width))
    nn.init.zeros_(mlp[2].weight)
    nn.init.zeros_(mlp[2].bias)
    model.state_to_expert = mlp.to(device=reference.device, dtype=reference.dtype)
    model._direct_state = None

    cls = type(model)

    def embed_suffix(noisy_actions, timestep):
        embs, pad_masks, att_masks = cls.embed_suffix(model, noisy_actions, timestep)
        if model._direct_state is not None:
            state = model._direct_state.to(device=embs.device, dtype=model.state_to_expert[0].weight.dtype)
            embs = embs + model.state_to_expert(state)[:, None, :].to(embs.dtype)
        return embs, pad_masks, att_masks

    def with_state(method):
        def call(images, img_masks, lang_tokens, lang_masks, state, *args, **kwargs):
            model._direct_state = state
            try:
                return method(model, images, img_masks, lang_tokens, lang_masks, state, *args, **kwargs)
            finally:
                model._direct_state = None
        return call

    model.embed_suffix = embed_suffix
    model.forward = with_state(cls.forward)
    model.sample_actions = with_state(cls.sample_actions)


def mark_checkpoint(pretrained_dir: Path) -> None:
    (Path(pretrained_dir) / MARKER_FILE).write_text(json.dumps({"direct_state": True}), encoding="utf-8")


def load_direct_state(policy, pretrained_dir: Path) -> bool:
    """Patch and restore the MLP if `pretrained_dir` was saved with it. Returns whether it was."""
    pretrained_dir = Path(pretrained_dir)
    if not (pretrained_dir / MARKER_FILE).exists():
        return False
    from safetensors.torch import load_file

    add_direct_state(policy)
    weights = {key[len(_PREFIX):]: value for key, value in load_file(str(pretrained_dir / "model.safetensors")).items()
               if key.startswith(_PREFIX)}
    policy.model.state_to_expert.load_state_dict(weights)
    return True
