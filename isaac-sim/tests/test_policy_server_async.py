"""SmolVLA server's background chunk sampling (--async-chunks) with a stand-in model, no GPU."""
from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from act_policy_server import SmolVlaRunner  # noqa: E402

CHUNK = 10


def _runner(latency_s: float, ticks_per_action: int = 1, hold: bool = False) -> SmolVlaRunner:
    """A runner whose chunk k (sampled at tick t) is [[t*100 + i] * 7 for i in range(CHUNK)]."""
    runner = SmolVlaRunner.__new__(SmolVlaRunner)
    runner.async_chunks = True
    runner._executor = ThreadPoolExecutor(max_workers=1)
    runner._pending = None
    runner.queue = []
    runner.tick = 0
    runner.underruns = 0
    runner.holds = 0
    runner.ticks_per_action = ticks_per_action
    runner.hold_on_underrun = hold
    runner.replan_every = 1
    runner.camera = False
    runner.history = []
    runner.skip_test = runner.skips = runner.history_log = None
    runner.drawings = 0
    runner.policy = type("P", (), {"reset": lambda self: None})()
    seen_ticks = []
    lock = threading.Lock()

    def chunk(state, env, history=None, modes=None):
        with lock:
            tick = int(state[0])
            seen_ticks.append(tick)
        time.sleep(latency_s)
        return torch.tensor([[tick * 100 + i] * 7 for i in range(CHUNK)], dtype=torch.float32)

    runner.chunk = chunk
    runner.seen_ticks = seen_ticks
    return runner


def _drive(runner: SmolVlaRunner, ticks: int, tick_s: float) -> list[float]:
    actions = []
    for tick in range(ticks):
        actions.append(runner.act([float(tick)] + [0.0] * 20, [0.0] * 37)[0])
        time.sleep(tick_s)
    return actions


def test_fast_model_never_waits_and_skips_the_ticks_that_passed():
    runner = _runner(latency_s=0.02)
    actions = _drive(runner, ticks=30, tick_s=0.01)
    assert runner.underruns == 0
    assert actions[0] == 0.0  # first chunk sampled synchronously at tick 0
    # Every later action comes from a chunk sampled at an earlier tick t, at index (now - t).
    for now, value in enumerate(actions):
        sampled_at, index = divmod(int(value), 100)
        assert sampled_at <= now and sampled_at + index == now
    assert len(set(int(v) // 100 for v in actions)) > 3  # chunks really were replaced in the background


def test_slow_model_counts_underruns_instead_of_returning_stale_moves():
    runner = _runner(latency_s=0.15)
    actions = _drive(runner, ticks=40, tick_s=0.0)
    assert runner.underruns > 0
    for now, value in enumerate(actions):
        sampled_at, index = divmod(int(value), 100)
        assert sampled_at + index == now  # still the move meant for this tick


def test_two_ticks_per_action_play_each_action_twice():
    runner = _runner(latency_s=0.02, ticks_per_action=2)
    actions = _drive(runner, ticks=40, tick_s=0.01)
    assert runner.underruns == 0
    for now, value in enumerate(actions):
        sampled_at, index = divmod(int(value), 100)
        assert index == (now - sampled_at) // 2  # chunk action i covers ticks 2i and 2i + 1 after sampling


def test_hold_mode_never_waits_and_returns_zero_steps_instead():
    runner = _runner(latency_s=0.15, hold=True)
    started = time.perf_counter()
    calls = []
    for tick in range(40):
        t0 = time.perf_counter()
        calls.append(runner.act([float(tick)] + [0.0] * 20, [0.0] * 37))
        assert time.perf_counter() - t0 < 0.05  # no tick waits for the model
        time.sleep(0.01)
    assert runner.underruns == 0 and runner.holds > 0
    assert calls[0] == [0.0] * 7  # nothing ready at tick 0: hold
    assert time.perf_counter() - started < 1.5
    for now, action in enumerate(calls):
        if action != [0.0] * 7:
            sampled_at, index = divmod(int(action[0]), 100)
            assert sampled_at + index == now


def test_reset_waits_for_the_chunk_in_flight():
    runner = _runner(latency_s=0.05)
    _drive(runner, ticks=3, tick_s=0.0)
    runner.reset()
    assert runner._pending is None and runner.queue == [] and runner.tick == 0


def test_redraw_route_checkpoints_see_the_route_back_while_going_back():
    """A --redraw-route camera checkpoint is shown the reversed traced stroke when the request's mode is 1."""
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import vla_features

    runner = SmolVlaRunner.__new__(SmolVlaRunner)
    runner.features = vla_features
    runner.camera = runner.redraw_route = runner.full_state = runner.mode_state = True
    runner.uses_template, runner.template = False, None
    runner.async_chunks, runner.replan_every, runner.samples = False, 1, 1
    runner.noise, runner.noise_shape, runner.device = "zero", (1, CHUNK, 32), "cpu"
    runner.skip_test = runner.skips = runner.history_log = None
    runner.history, runner.modes, runner.queue, runner.mode = [], [], [], 0
    seen = []
    runner.preprocessor = lambda observation: seen.append(observation["observation.state"].numpy()) or observation
    runner.postprocessor = lambda actions: actions
    runner.policy = type("P", (), {"reset": lambda self: None,
                                   "predict_action_chunk": lambda self, batch, noise=None: torch.zeros(1, CHUNK, 7)})()
    window = vla_features.FULL_STATE_DIM - vla_features.STATE_DIM
    for tick, mode in enumerate([0] * 12 + [1] * 3):
        x = 0.47 + 0.0005 * (tick if mode == 0 else 22 - tick)
        env = np.r_[[x, 0.08, 0.0496], [0.0005, 0.0, 0.0], np.tile([0.0005, 0.0, 0.0], window // 3), 1.0]
        runner.act([0.0] * 21, env.tolist(), False, mode)
    assert runner.modes == [0] * 12 + [1] * 3
    assert seen[11][24] > 0 and seen[-1][24] < 0  # target error x: ahead while drawing, behind going back
    assert seen[-1][-2:].tolist() == [1.0, 0.0]  # mode state: going back
