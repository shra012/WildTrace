"""Minimal client for scripts/act_policy_server.py (stdlib only, runs in Isaac's python)."""
from __future__ import annotations

import json
import socket
from typing import Sequence

import numpy as np


class ActPolicyClient:
    def __init__(self, address: str, timeout_s: float = 30.0):
        host, _, port = address.rpartition(":")
        self._socket = socket.create_connection((host or "127.0.0.1", int(port)), timeout=timeout_s)
        self._stream = self._socket.makefile("rwb")
        self.latencies_ms: list[float] = []

    def _call(self, payload: dict) -> dict:
        self._stream.write((json.dumps(payload) + "\n").encode())
        self._stream.flush()
        line = self._stream.readline()
        if not line:
            raise ConnectionError("ACT policy server closed the connection")
        return json.loads(line)

    def reset(self) -> None:
        self._call({"reset": True})

    def act(self, state: Sequence[float], env: Sequence[float]) -> np.ndarray:
        reply = self._call({"state": [float(v) for v in state], "env": [float(v) for v in env]})
        self.latencies_ms.append(float(reply.get("ms", 0.0)))
        return np.asarray(reply["action"], dtype=np.float64)

    def close(self) -> None:
        try:
            self._stream.close()
        finally:
            self._socket.close()
