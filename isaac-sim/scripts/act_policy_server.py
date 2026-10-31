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

import numpy as np
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
    parser.add_argument("--cuda-graph", action="store_true",
                        help="smolvla: sample each chunk by replaying a CUDA graph (src/smolvla_cuda_graph.py), "
                             "about 7x faster per chunk under WSL")
    parser.add_argument("--ticks-per-action", type=int, default=1,
                        help="smolvla --async-chunks: control ticks each chunk action is played for")
    parser.add_argument("--hold-on-underrun", action="store_true",
                        help="smolvla --async-chunks: return a zero step instead of waiting for a chunk")
    parser.add_argument("--skip-test-spacing-m", type=float, default=0.0,
                        help="smolvla camera: inject ink skips into the rendered page every this many mm of inked "
                             "travel on average (act_features.SkipSchedule), to test gap redraws (#46); 0 = off")
    parser.add_argument("--skip-test-length-m", default="0.003,0.008", help="skip length range, min,max")
    parser.add_argument("--skip-test-seed", type=int, default=0)
    parser.add_argument("--history-log", default="",
                        help="smolvla camera: directory to save each drawing's page history (history_<n>.npz)")
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
    def act(self, state, env, ink_skip: bool = False, mode: int = 0) -> list[float]:
        """`ink_skip` and `mode` are for camera policies (#46); ACT has no page or mode state."""
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
                 hold_on_underrun: bool = False, skip_test: dict | None = None, history_log: str = "",
                 cuda_graph: bool = False):
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
        if cuda_graph:
            import smolvla_cuda_graph

            smolvla_cuda_graph.use_cuda_graph(self.policy)
        self.preprocessor, self.postprocessor = make_pre_post_processors(config, pretrained_path=str(checkpoint))
        self.replan_every = max(1, min(replan_every, config.chunk_size))
        self.noise, self.samples, self.device = noise, max(1, samples), device
        self.noise_shape = (self.samples, config.chunk_size, config.max_action_dim)
        # Checkpoints trained with --state full take the path window as numbers too, and with
        # --mode-state the redraw mode (sent by the drawing controller's supervisor) after it.
        state_dim = config.input_features["observation.state"].shape[0]
        self.full_state = state_dim >= vla_features.FULL_STATE_DIM
        self.mode_state = state_dim == vla_features.FULL_STATE_DIM + vla_features.MODE_STATE_DIM
        self.mode = 0
        # Checkpoints trained with --view camera see the paper; they render it from the ticks since reset.
        self.camera = vla_features.OVERHEAD_KEY in config.input_features
        # Tracing-mode checkpoints also show the whole drawing, sent with the reset.
        self.uses_template = bool(vla_features.read_views(checkpoint).get("template"))
        self.template = None
        self.history: list[np.ndarray] = []
        self.queue: list[list[float]] = []
        self.async_chunks = async_chunks
        self.ticks_per_action = max(1, ticks_per_action)
        self.hold_on_underrun = hold_on_underrun
        self._executor = ThreadPoolExecutor(max_workers=1) if async_chunks else None
        self._pending = None  # (future, tick whose observation it was sampled from)
        self.tick = 0
        self.underruns = 0
        self.holds = 0
        # Skip test (#46): the page leaves out ink where a SkipSchedule says the pen skipped.
        self.skip_test = skip_test
        self.skips = None
        self.history_log = Path(history_log) if history_log else None
        self.drawings = 0

    def _new_skips(self):
        from act_features import SkipSchedule

        if not self.skip_test:
            return None
        rng = np.random.default_rng(self.skip_test["seed"] + self.drawings)
        return SkipSchedule(rng, self.skip_test["spacing_m"], self.skip_test["length_m"])

    def save_history(self) -> None:
        if self.history_log is None or not self.history:
            return
        self.history_log.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.history_log / f"history_{self.drawings:03d}.npz", rows=np.concatenate(self.history),
                            skip_test=json.dumps(self.skip_test or {}))

    def reset(self, drawing=None) -> None:
        """`drawing`: (M, 3) x, y, pen_down of the whole path, for tracing-mode checkpoints."""
        if self._pending is not None:
            self._pending[0].result()  # let an in-flight chunk finish before clearing state
        self.template = None if drawing is None else np.asarray(drawing, dtype=np.float64).reshape(-1, 3)
        self._pending = None
        self.policy.reset()
        if self.history:
            self.save_history()
            self.drawings += 1
        self.skips = self._new_skips()
        self.history = []
        self.mode = 0
        self.queue = []
        self.tick = 0
        self.underruns = 0
        self.holds = 0

    @torch.no_grad()
    def chunk(self, state, env, history: np.ndarray | None = None) -> torch.Tensor:
        """The next (chunk_size, 7) actions, unnormalized, averaged over `samples` sampled chunks.
        Camera checkpoints render the paper from `history` (rows up to and including this tick;
        default: the ticks since reset)."""
        if self.camera:
            if history is None:
                history = np.concatenate(self.history) if self.history else self.features.history_rows(env)
            if self.uses_template and self.template is None:
                raise RuntimeError("this checkpoint was trained in tracing mode; send the drawing with the reset")
            images = self.features.camera_views(history, env, template=self.template if self.uses_template else None)
        else:
            images = {self.features.IMAGE_KEY: self.features.image_tensor_layout(self.features.path_view(env))}
        observation = {
            "observation.state": torch.from_numpy(
                self.features.vla_state(state, env, self.full_state, self.mode if self.mode_state else None)[0]),
            **{key: torch.from_numpy(image) for key, image in images.items()},
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

    def act(self, state, env, ink_skip: bool = False, mode: int = 0) -> list[float]:
        """`mode`: the redraw supervisor's teacher mode (0 normal, 1 back, 2 forward), for --mode-state checkpoints."""
        self.mode = int(mode)
        if self.camera:
            row = self.features.history_rows(env, np.asarray([float(ink_skip)]))
            if self.skips is not None:
                contact = bool(row[0, 4] > 0.5)
                if not self.skips.inks(contact, row[0, 0:2]) and contact:
                    row[0, 4] = 0.0
            self.history.append(row)
            if self.history_log is not None and len(self.history) % 300 == 0:
                self.save_history()
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
        self._pending = self._pending or (self._executor.submit(
            self.chunk, list(state), list(env), np.concatenate(self.history) if self.camera else None), self.tick)
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
                        if "drawing" in request and isinstance(runner, SmolVlaRunner):
                            runner.reset(request["drawing"])
                        else:
                            runner.reset()
                        reply = {"ok": True}
                    else:
                        started = time.perf_counter()
                        reply = {
                            "action": runner.act(request["state"], request["env"],
                                                 **{key: request[key] for key in ("ink_skip", "mode") if key in request}),
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
        lengths = tuple(float(v) for v in args.skip_test_length_m.split(","))
        skip_test = {"spacing_m": args.skip_test_spacing_m, "length_m": lengths, "seed": args.skip_test_seed} \
            if args.skip_test_spacing_m > 0 else None
        runner = SmolVlaRunner(checkpoint, args.device, args.replan_every, args.noise, args.samples, args.flow_steps,
                               args.async_chunks, args.ticks_per_action, args.hold_on_underrun, skip_test,
                               args.history_log, args.cuda_graph)
        print(f"[OK] SmolVLA, executing {runner.replan_every} action(s) per chunk, noise {args.noise}, "
              f"{runner.samples} sample(s) per chunk, {runner.policy.config.num_steps} flow steps"
              + (", CUDA graph" if args.cuda_graph else ""), flush=True)
    else:
        runner = ActRunner(checkpoint, args.device, args.execution, args.temporal_ensemble_coeff)
        print(f"[OK] Execution mode: {args.execution}", flush=True)
    _serve(runner, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
