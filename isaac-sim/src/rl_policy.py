"""Small PPO implementation for the goal-conditioned drawing prototype.

The policy outputs Cartesian deltas, not joint torques. Isaac Sim and the
existing safe IK controller remain responsible for robot-level control.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


class DrawingActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int = 4, hidden: int = 256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(observation_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))

    def forward(self, observation: torch.Tensor):
        features = self.body(observation)
        return self.actor_mean(features), self.critic(features).squeeze(-1)

    def distribution(self, observation: torch.Tensor):
        mean, value = self(observation)
        return Normal(mean, self.log_std.exp()), value


@dataclass
class Rollout:
    observations: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    last_observation: np.ndarray
    last_value: float


def collect_rollout(env, policy: DrawingActorCritic, steps: int, device: torch.device) -> Rollout:
    observations, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
    observation = env.reset() if env.observation is None else env.observation
    for _ in range(int(steps)):
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            distribution, value = policy.distribution(tensor)
            action = distribution.sample()
            log_prob = distribution.log_prob(action).sum(dim=-1)
        action_np = torch.tanh(action).squeeze(0).cpu().numpy()
        next_observation, reward, done, _ = env.step(action_np)
        observations.append(observation.copy())
        actions.append(action.squeeze(0).cpu().numpy())
        log_probs.append(float(log_prob.item()))
        rewards.append(float(reward))
        dones.append(float(done))
        values.append(float(value.item()))
        observation = env.reset() if done else next_observation
    with torch.no_grad():
        last_value = float(policy(torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0))[1].item())
    env.observation = observation
    return Rollout(
        np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32),
        np.asarray(log_probs, dtype=np.float32), np.asarray(rewards, dtype=np.float32),
        np.asarray(dones, dtype=np.float32), np.asarray(values, dtype=np.float32),
        observation.copy(), last_value,
    )


def compute_gae(rollout: Rollout, gamma: float, gae_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rollout.rewards)
    running = 0.0
    next_value = rollout.last_value
    for index in range(len(rollout.rewards) - 1, -1, -1):
        nonterminal = 1.0 - rollout.dones[index]
        delta = rollout.rewards[index] + gamma * next_value * nonterminal - rollout.values[index]
        running = delta + gamma * gae_lambda * nonterminal * running
        advantages[index] = running
        next_value = rollout.values[index]
    return advantages, advantages + rollout.values


def ppo_update(policy, optimizer, rollout, advantages, returns, clip_ratio, epochs, minibatch_size, entropy_weight, device):
    observations = torch.as_tensor(rollout.observations, dtype=torch.float32, device=device)
    actions = torch.as_tensor(rollout.actions, dtype=torch.float32, device=device)
    old_log_probs = torch.as_tensor(rollout.log_probs, dtype=torch.float32, device=device)
    advantages = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    returns = torch.as_tensor(returns, dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    count = len(observations)
    stats = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "batches": 0,
    }
    for _ in range(int(epochs)):
        order = torch.randperm(count, device=device)
        for start in range(0, count, int(minibatch_size)):
            index = order[start:start + int(minibatch_size)]
            distribution, value = policy.distribution(observations[index])
            log_prob = distribution.log_prob(actions[index]).sum(dim=-1)
            ratio = (log_prob - old_log_probs[index]).exp()
            clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
            policy_loss = -torch.minimum(ratio * advantages[index], clipped * advantages[index]).mean()
            value_loss = 0.5 * (returns[index] - value).pow(2).mean()
            entropy = distribution.entropy().sum(dim=-1).mean()
            with torch.no_grad():
                log_ratio = log_prob - old_log_probs[index]
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > clip_ratio).float().mean()
            loss = policy_loss + value_loss - entropy_weight * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            stats["policy_loss"] += float(policy_loss.item())
            stats["value_loss"] += float(value_loss.item())
            stats["entropy"] += float(entropy.item())
            stats["approx_kl"] += float(approx_kl.item())
            stats["clip_fraction"] += float(clip_fraction.item())
            stats["batches"] += 1
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"):
        stats[key] /= max(1, stats["batches"])
    return stats


def save_policy(
    path: str | Path,
    policy: DrawingActorCritic,
    observation_dim: int,
    metadata: Dict,
    optimizer=None,
    training_state: Dict | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": policy.state_dict(),
        "observation_dim": int(observation_dim),
        "action_dim": int(policy.log_std.numel()),
        "metadata": metadata,
        "training_state": dict(training_state or {}),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, destination)


def load_policy_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    """Load a policy and the complete checkpoint payload.

    Older checkpoints without optimizer/training state remain supported.
    """
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    policy = DrawingActorCritic(
        int(payload["observation_dim"]), action_dim=int(payload.get("action_dim", 4))
    ).to(device)
    policy.load_state_dict(payload["state_dict"])
    policy.eval()
    return policy, payload


def load_policy(path: str | Path, device: str | torch.device = "cpu"):
    policy, payload = load_policy_checkpoint(path, device)
    return policy, payload.get("metadata", {})
