"""Serve a trained ACT (or SmolVLA) checkpoint to the Isaac Sim runner over TCP.

Runs in WSL inside isaac-sim/.venv-act; Isaac (Windows) reaches it through
WSL's localhost forwarding:

    isaac-sim/.venv-act/bin/python isaac-sim/scripts/act_policy_server.py \
        --checkpoint outputs/act/kinematic/checkpoints/last/pretrained_model

Protocol: one JSON object per line, one client at a time.
    {"reset": true}                               -> {"ok": true}
    {"state": [21 floats], "env": [37 floats]}    -> {"action": [7 floats], "ms": float}
Every call predicts a fresh chunk. --execution ensemble blends it with the
earlier overlapping chunks (ACT temporal ensembling); --execution first
returns only its first action. Either way each reply is the action for the
current control tick.

--policy smolvla serves a scripts/train_smolvla.py checkpoint behind the same
protocol: the request's ACT state and env are turned into SmolVLA's state and
path image (src/vla_features.py) here, so Isaac's side does not change.
SmolVLA samples its chunk with 10 flow-matching steps, so --replan-every can
execute several actions of a chunk before sampling the next one. Each chunk
starts from Gaussian noise; with fresh noise every chunk, consecutive chunks
disagree slightly and the pen wobbles. --noise zero starts every chunk from the
same (zero) noise, and --samples N averages N chunks sampled together.

--async-chunks is for the real arm, which cannot wait for a chunk: every reply
comes straight from the current chunk, while a background thread keeps sampling
the next one from the latest observation. A finished chunk replaces the queue,
minus the actions whose ticks passed while it was being sampled; the count of
ticks that still had to wait (underruns) is printed when the client leaves.
--ticks-per-action 2 plays each chunk action for two ticks, for a 30 fps policy
driven by a 60 Hz loop at half scale, so a chunk lasts as long as it was trained
to. --hold-on-underrun never waits: a tick with no action ready gets a zero
joint step (the arm holds still) and is counted as a hold, which keeps every
tick inside a real-time loop's budget.
"""
from __future__ import annotations

import argparse
import json
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="pretrained_model directory")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--execution",
        choices=["ensemble", "first"],
        default="first",
        help="first: re-plan every tick and execute only the first action (tracks far better on the "
        "overfit gate); ensemble: ACT temporal ensembling",
    )
    parser.add_argument("--temporal-ensemble-coeff", type=float, default=0.01, help="ACT paper default")
    parser.add_argument("--policy", choices=["act", "smolvla"], default="act")
    parser.add_argument("--replan-every", type=int, default=1, help="smolvla: actions executed per sampled chunk")
    parser.add_argument("--noise", choices=["random", "zero"], default="random", help="smolvla: starting noise of each chunk")
    parser.add_argument("--samples", type=int, default=1, help="smolvla: chunks sampled and averaged per query")
    parser.add_argument("--async-chunks", action="store_true",
                        help="smolvla: sample the next chunk in the background while the current one runs")
    parser.add_argument("--flow-steps", type=int, default=0,
                        help="smolvla: flow-matching steps per chunk (0 = the checkpoint's, 10); fewer is faster")
    parser.add_argument("--ticks-per-action", type=int, default=1,
                        help="smolvla --async-chunks: control ticks each chunk action is played for")
    parser.add_argument("--hold-on-underrun", action="store_true",
                        help="smolvla --async-chunks: return a zero step instead of waiting for a chunk")
    return parser.parse_args()


class ActRunner:
    def __init__(self, checkpoint: Path, device: str, execution: str, ensemble_coeff: float):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        config = PreTrainedConfig.from_pretrained(checkpoint)
        config.device = device
        # Both modes query a fresh chunk every tick; "first" keeps only its first
        # action, so corrections are not averaged with predictions made up to a
        # chunk ago.
        config.n_action_steps = 1
        config.temporal_ensemble_coeff = ensemble_coeff if execution == "ensemble" else None
        self.policy = ACTPolicy.from_pretrained(checkpoint, config=config)
        self.policy.to(device).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(config, pretrained_path=str(checkpoint))
        self.device = device

    def reset(self) -> None:
        self.policy.reset()

    @torch.no_grad()
    def act(self, state, env) -> list[float]:
        observation = {
            "observation.state": torch.as_tensor(state, dtype=torch.float32),
            "observation.environment_state": torch.as_tensor(env, dtype=torch.float32),
        }
        batch = self.preprocessor(observation)
        action = self.postprocessor(self.policy.select_action(batch))
        return action.reshape(-1).float().cpu().tolist()


class SmolVlaRunner:
    def __init__(self, checkpoint: Path, device: str, replan_every: int, noise: str = "random", samples: int = 1,
                 flow_steps: int = 0, async_chunks: bool = False, ticks_per_action: int = 1,
                 hold_on_underrun: bool = False):
        import sys

        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        import vla_features

        self.features = vla_features
        config = PreTrainedConfig.from_pretrained(checkpoint)
        config.device = device
        if flow_steps > 0:
            config.num_steps = flow_steps
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config)
        import smolvla_direct_state

        smolvla_direct_state.load_direct_state(self.policy, checkpoint)
        self.policy.to(device).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(config, pretrained_path=str(checkpoint))
        self.replan_every = max(1, min(replan_every, config.chunk_size))
        self.noise, self.samples, self.device = noise, max(1, samples), device
        self.noise_shape = (self.samples, config.chunk_size, config.max_action_dim)
        # Checkpoints trained with --state full take the path window as numbers too.
        self.full_state = config.input_features["observation.state"].shape[0] == vla_features.FULL_STATE_DIM
        self.queue: list[list[float]] = []
        self.async_chunks = async_chunks
        self.ticks_per_action = max(1, ticks_per_action)
        self.hold_on_underrun = hold_on_underrun
        self._executor = ThreadPoolExecutor(max_workers=1) if async_chunks else None
        self._pending = None  # (future, tick whose observation it was sampled from)
        self.tick = 0
        self.underruns = 0
        self.holds = 0

    def reset(self) -> None:
        if self._pending is not None:
            self._pending[0].result()  # let an in-flight chunk finish before clearing state
        self._pending = None
        self.policy.reset()
        self.queue = []
        self.tick = 0
        self.underruns = 0
        self.holds = 0

    @torch.no_grad()
    def chunk(self, state, env) -> torch.Tensor:
        """The next (chunk_size, 7) actions, unnormalized, averaged over `samples` sampled chunks."""
        image = self.features.path_view(env)
        observation = {
            "observation.state": torch.from_numpy(self.features.vla_state(state, env, self.full_state)[0]),
            self.features.IMAGE_KEY: torch.from_numpy(self.features.image_tensor_layout(image)),
            "task": self.features.TASK,
        }
        batch = self.preprocessor(observation)
        if self.samples > 1:
            batch = {key: value.repeat(self.samples, *[1] * (value.dim() - 1))
                     if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == 1 else value
                     for key, value in batch.items()}
        noise = torch.zeros(self.noise_shape, device=self.device) if self.noise == "zero" else None
        self.policy.reset()
        actions = self.policy.predict_action_chunk(batch, noise=noise).mean(dim=0, keepdim=True)
        return self.postprocessor(actions)[0].float().cpu()

    def act(self, state, env) -> list[float]:
        if not self.async_chunks:
            if not self.queue:
                self.queue = self.chunk(state, env)[: self.replan_every].tolist()
            return self.queue.pop(0)
        wait = not self.hold_on_underrun
        waited = False
        if self._pending is not None and (self._pending[0].done() or (wait and not self.queue)):
            future, launched = self._pending
            self._pending = None
            waited = not future.done()
            # chunk[0] was meant for the tick it was sampled at; skip the ticks that have passed since.
            # A chunk that is entirely in the past is dropped.
            self.queue = self._ticks(future.result())[self.tick - launched:]
        if not self.queue and wait:
            waited = waited or self.tick > 0
            self.queue = self._ticks(self.chunk(state, env))
        self.underruns += int(waited)
        if self.queue:
            action = self.queue.pop(0)
        else:
            self.holds += 1
            action = [0.0] * 7
        self._pending = self._pending or (self._executor.submit(self.chunk, list(state), list(env)), self.tick)
        self.tick += 1
        return action

    def _ticks(self, chunk: torch.Tensor) -> list[list[float]]:
        """A chunk's actions, one per control tick."""
        return [action for action in chunk.tolist() for _ in range(self.ticks_per_action)]


def _serve(runner, host: str, port: int) -> None:
    with socket.create_server((host, port), reuse_port=False) as server:
        print(f"[OK] {type(runner).__name__} policy server on {host}:{port}", flush=True)
        while True:
            connection, address = server.accept()
            print(f"[OK] Client {address}", flush=True)
            runner.reset()
            with connection, connection.makefile("rwb") as stream:
                for line in stream:
                    request = json.loads(line)
                    if request.get("reset"):
                        runner.reset()
                        reply = {"ok": True}
                    else:
                        started = time.perf_counter()
                        reply = {
                            "action": runner.act(request["state"], request["env"]),
                            "ms": (time.perf_counter() - started) * 1e3,
                        }
                    stream.write((json.dumps(reply) + "\n").encode())
                    stream.flush()
            print(f"[OK] Client {address} disconnected"
                  + (f" ({runner.underruns} underruns, {runner.holds} holds in {runner.tick} ticks)"
                     if getattr(runner, "async_chunks", False)
                     else ""), flush=True)


def main() -> int:
    args = _parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    if args.policy == "smolvla":
        runner = SmolVlaRunner(checkpoint, args.device, args.replan_every, args.noise, args.samples, args.flow_steps,
                               args.async_chunks, args.ticks_per_action, args.hold_on_underrun)
        print(f"[OK] SmolVLA, executing {runner.replan_every} action(s) per chunk, noise {args.noise}, "
              f"{runner.samples} sample(s) per chunk, {runner.policy.config.num_steps} flow steps", flush=True)
    else:
        runner = ActRunner(checkpoint, args.device, args.execution, args.temporal_ensemble_coeff)
        print(f"[OK] Execution mode: {args.execution}", flush=True)
    _serve(runner, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
