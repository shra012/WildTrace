"""Serve a trained ACT checkpoint to the Isaac Sim runner over TCP.

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
"""
from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

import torch


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


def _serve(runner: ActRunner, host: str, port: int) -> None:
    with socket.create_server((host, port), reuse_port=False) as server:
        print(f"[OK] ACT policy server on {host}:{port}", flush=True)
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
    runner = ActRunner(Path(args.checkpoint).resolve(), args.device, args.execution, args.temporal_ensemble_coeff)
    print(f"[OK] Execution mode: {args.execution}", flush=True)
    _serve(runner, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
