"""Run SmolVLA's sampling as one CUDA graph.

A SmolVLA call launches about 2,600 small kernels (vision encoder, VLM prefix,
then one action-expert pass per flow step). The GPU work is about 11 ms, but
under WSL each launch costs ~30 us of CPU, so a 3-flow-step chunk took ~225 ms.
Capturing sample_actions once and replaying it cuts that to ~30 ms, the same
actions up to bf16 rounding (first-action error unchanged on the val frames).

use_cuda_graph() replaces policy.model.sample_actions with a wrapper that copies
the inputs into static tensors and replays the graph, capturing it on the first
call (and again if the input shapes change, e.g. a different sample count).
Capturing needs three changes, all value-preserving:
- the flow-matching loop builds its timestep tensors on the GPU up front
  (lerobot's euler_integrate makes one from a Python float every step);
- constants made with torch.tensor(..., device=cuda) and the vision encoder's
  torch.ones mask are made without a CPU-to-GPU copy during capture;
- the vision embeddings' boolean-mask assignment (a CPU sync) becomes torch.where.
Needs torch only on top of LeRobot; works with smolvla_direct_state.
"""
from __future__ import annotations

import torch


def _capturable_vision_embeddings(module) -> None:
    """SmolVLMVisionEmbeddings.forward with torch.where in place of the boolean-mask assignment."""

    def forward(pixel_values, patch_attention_mask):
        batch_size = pixel_values.shape[0]
        embeddings = module.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        n = module.num_patches_per_side
        boundaries = torch.arange(1 / n, 1.0, 1 / n, device=pixel_values.device)
        step_h = 1.0 / patch_attention_mask[:, :, 0].sum(dim=1)
        step_w = 1.0 / patch_attention_mask[:, 0, :].sum(dim=1)
        h = torch.arange(patch_attention_mask.size(1), device=pixel_values.device, dtype=torch.float32)
        w = torch.arange(patch_attention_mask.size(2), device=pixel_values.device, dtype=torch.float32)
        coords_h = torch.clamp(h[None, :] * step_h[:, None], max=1.0 - 1e-6).to(pixel_values.dtype)
        coords_w = torch.clamp(w[None, :] * step_w[:, None], max=1.0 - 1e-6).to(pixel_values.dtype)
        bucket_h = torch.bucketize(coords_h, boundaries, right=True)
        bucket_w = torch.bucketize(coords_w, boundaries, right=True)
        pos_ids = (bucket_h[:, :, None] * n + bucket_w[:, None, :]).reshape(batch_size, -1)
        position_ids = torch.where(patch_attention_mask.view(batch_size, -1), pos_ids, 0)
        return embeddings + module.position_embedding(position_ids)

    module.forward = forward


def use_cuda_graph(policy) -> None:
    """Patch a SmolVLAPolicy on a CUDA device in place to sample through a CUDA graph."""
    model = policy.model
    for module in model.modules():
        if type(module).__name__ == "SmolVLMVisionEmbeddings":
            _capturable_vision_embeddings(module)
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

    static: dict = {}

    def sample(images, img_masks, lang_tokens, lang_masks, state, noise):
        """SmolVLAFlowMatching.sample_actions on the static tensors."""
        if hasattr(model, "state_to_expert"):
            model._direct_state = state
        embs, pad, att = model.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
        _, past_key_values = model.vlm_with_expert.forward(
            attention_mask=make_att_2d_masks(pad, att), position_ids=torch.cumsum(pad, dim=1) - 1,
            past_key_values=None, inputs_embeds=[embs, None], use_cache=model.config.use_cache)
        x = noise
        for timestep in static["timesteps"]:  # forward Euler from t=1 (noise) to t=0, as euler_integrate
            x = x + (-1.0 / len(static["timesteps"])) * model.denoise_step(
                x_t=x, prefix_pad_masks=pad, past_key_values=past_key_values, timestep=timestep)
        return x

    def capture(inputs: list[torch.Tensor]) -> None:
        static["inputs"] = [tensor.clone() for tensor in inputs]
        n, batch, device = model.config.num_steps, inputs[-1].shape[0], inputs[-1].device
        static["timesteps"] = [torch.full((batch,), 1.0 - step / n, dtype=torch.float32, device=device)
                               for step in range(n)]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                static["unpack"](static["inputs"])
        torch.cuda.current_stream().wait_stream(side)
        static["graph"] = torch.cuda.CUDAGraph()
        plain_tensor, plain_ones, static["constants"] = torch.tensor, torch.ones, []

        def tensor(data, *args, device=None, **kwargs):
            if device is None or torch.device(device).type != "cuda" or torch.is_tensor(data):
                return plain_tensor(data, *args, device=device, **kwargs)
            host = plain_tensor(data, *args, **kwargs).pin_memory()
            static["constants"].append(host)  # the graph copies from it on every replay
            return host.to(device, non_blocking=True)

        def ones(*size, **kwargs):  # only the vision encoder's patch mask, which it moves to the GPU
            return plain_ones(*size, **kwargs) if kwargs.get("device") is not None \
                else plain_ones(*size, device=device, **kwargs)

        torch.tensor, torch.ones = tensor, ones
        try:
            with torch.cuda.graph(static["graph"]):
                static["output"] = static["unpack"](static["inputs"])
        finally:
            torch.tensor, torch.ones = plain_tensor, plain_ones
            if hasattr(model, "state_to_expert"):
                model._direct_state = None
        static["shapes"] = [tuple(tensor.shape) for tensor in inputs]

    @torch.no_grad()
    def sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
        if noise is None:
            noise = model.sample_noise((state.shape[0], model.config.chunk_size, model.config.max_action_dim),
                                       state.device)
        count = len(images)
        inputs = [*images, *img_masks, lang_tokens, lang_masks, state, noise]
        static["unpack"] = lambda t: sample(t[:count], t[count:2 * count], *t[2 * count:])
        if static.get("shapes") != [tuple(tensor.shape) for tensor in inputs]:
            capture(inputs)
        for target, source in zip(static["inputs"], inputs):
            target.copy_(source)
        static["graph"].replay()
        return static["output"].clone()

    model.sample_actions = sample_actions
