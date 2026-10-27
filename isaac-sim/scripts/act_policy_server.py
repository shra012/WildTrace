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
execute several actions of a chunk before sampling the next one.
"""
from __future__ import annotations

import argparse
import json
import socket
import time
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
    def __init__(self, checkpoint: Path, device: str, replan_every: int):
        import sys

        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        import vla_features

        self.features = vla_features
        config = PreTrainedConfig.from_pretrained(checkpoint)
        config.device = device
        config.n_action_steps = max(1, min(replan_every, config.chunk_size))
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config)
        self.policy.to(device).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(config, pretrained_path=str(checkpoint))

    def reset(self) -> None:
        self.policy.reset()

    @torch.no_grad()
    def act(self, state, env) -> list[float]:
        image = self.features.path_view(env)
        observation = {
            "observation.state": torch.from_numpy(self.features.vla_state(state, env)[0]),
            self.features.IMAGE_KEY: torch.from_numpy(self.features.image_tensor_layout(image)),
            "task": self.features.TASK,
        }
        batch = self.preprocessor(observation)
        action = self.postprocessor(self.policy.select_action(batch))
        return action.reshape(-1).float().cpu().tolist()


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
            print(f"[OK] Client {address} disconnected", flush=True)


def main() -> int:
    args = _parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    if args.policy == "smolvla":
        runner = SmolVlaRunner(checkpoint, args.device, args.replan_every)
        print(f"[OK] SmolVLA, executing {runner.policy.config.n_action_steps} action(s) per chunk", flush=True)
    else:
        runner = ActRunner(checkpoint, args.device, args.execution, args.temporal_ensemble_coeff)
        print(f"[OK] Execution mode: {args.execution}", flush=True)
    _serve(runner, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
