"""Live run status for a drawing in progress, readable by a web UI.

The Action Graph controller (scripts/mcp_drawing_controller.py) and the
offline replay driver (src/run_replay.py) both write, per run prefix, under
outputs/mcp_sessions/live/:

- <prefix>.json: the latest status, replaced atomically (temp file + replace)
  about 10 times per simulated second:
      {"prefix", "phase", "progress", "tip_world": [x, y, z],
       "tip_canvas": [u, v], "error_mm", "sim_time", "status", "updated_at", ...}
  "status" is "drawing", "finished" or "failed"; "progress" is the fraction
  (0..1) of the pen-down path drawn; "tip_canvas" is in the trajectory's own
  canvas units (normalized 0..1, y down, for pipeline trajectories).
- <prefix>_trace.ndjson: append-only, one JSON line per pen-down tip sample
  thinned to ~1 mm spacing: {"stroke_id", "segment", "u", "v", "x", "y", "sim_time"}.

Standard library only, so it imports unchanged in Isaac Sim's Python, the
wildtrace-sim MCP server's uv script environment and the web backend. Writes
never raise: a full disk or a flaky \\\\wsl.localhost share must not stop a
drawing run.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

LIVE_SUBDIR = "live"
STATUS_DRAWING, STATUS_FINISHED, STATUS_FAILED = "drawing", "finished", "failed"
DEFAULT_PERIOD_S = 0.1
DEFAULT_TRACE_SPACING_M = 0.001


def live_dir(sessions_dir: str | Path) -> Path:
    """outputs/mcp_sessions/live for a given outputs/mcp_sessions directory."""
    return Path(sessions_dir) / LIVE_SUBDIR


def status_path(directory: str | Path, prefix: str) -> Path:
    return Path(directory) / f"{prefix}.json"


def trace_path(directory: str | Path, prefix: str) -> Path:
    return Path(directory) / f"{prefix}_trace.ndjson"


# ── Plane mapping inverse. map_trajectory_to_plane applies one affine map to
# every stroke before smoothing/resampling: world = (canvas - source_center) *
# uniform_scale, with y negated when image_y_flipped, plus center_xy.


def _mapping_terms(mapping: Mapping[str, Any]) -> Tuple[float, float, float, float, float, bool]:
    (min_u, min_v), (max_u, max_v) = mapping["source_bbox"]
    cx, cy = (float(v) for v in mapping["center_xy"])
    scale = float(mapping["uniform_scale"])
    return cx, cy, 0.5 * (float(min_u) + float(max_u)), 0.5 * (float(min_v) + float(max_v)), scale, bool(
        mapping.get("image_y_flipped", True)
    )


def world_to_canvas(mapping: Mapping[str, Any], xy: Sequence[float]) -> List[float]:
    """Drawing-plane metres -> trajectory canvas coordinates (the inverse of the plane mapping)."""
    cx, cy, su, sv, scale, flipped = _mapping_terms(mapping)
    dy = float(xy[1]) - cy
    if flipped:
        dy = -dy
    return [(float(xy[0]) - cx) / scale + su, dy / scale + sv]


def canvas_to_world(mapping: Mapping[str, Any], uv: Sequence[float]) -> List[float]:
    """Trajectory canvas coordinates -> drawing-plane metres, as map_trajectory_to_plane does."""
    cx, cy, su, sv, scale, flipped = _mapping_terms(mapping)
    dy = (float(uv[1]) - sv) * scale
    if flipped:
        dy = -dy
    return [(float(uv[0]) - su) * scale + cx, dy + cy]


# ── Files.


def write_status(path: str | Path, status: Mapping[str, Any]) -> bool:
    """Replace `path` with `status` atomically. Returns False instead of raising."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    text = json.dumps(status)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
        return True
    except OSError:
        pass
    try:
        tmp.unlink()
    except OSError:
        pass
    # Some shares refuse a rename over an existing file; a plain rewrite is
    # still better than a status that stops updating (read_status returns
    # None for the rare torn read, and the next poll gets the full file).
    try:
        path.write_text(text, encoding="utf-8")
        return True
    except OSError:
        return False


def read_status(path: str | Path) -> Optional[Dict[str, Any]]:
    """The latest status dict, or None when there is none (yet) or it cannot be read."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def read_trace(path: str | Path, offset: int = 0) -> Tuple[List[Dict[str, Any]], int]:
    """Trace samples from byte `offset` on, and the offset to resume from.

    Only complete lines are consumed, so a poll that races the writer simply
    picks the rest up next time. A missing file reads as empty."""
    try:
        with open(path, "rb") as handle:
            handle.seek(offset)
            data = handle.read()
    except OSError:
        return [], offset
    end = data.rfind(b"\n") + 1
    samples = []
    for line in data[:end].splitlines():
        try:
            samples.append(json.loads(line))
        except ValueError:
            continue
    return samples, offset + end


def trace_polylines(samples: Iterable[Mapping[str, Any]]) -> List[List[List[float]]]:
    """Group trace samples into one [[u, v], ...] polyline per pen-down segment."""
    lines: Dict[Tuple[Any, Any], List[List[float]]] = {}
    for sample in samples:
        if sample.get("u") is None:
            continue
        lines.setdefault((sample.get("segment"), sample.get("stroke_id")), []).append([sample["u"], sample["v"]])
    return list(lines.values())


# ── Progress along the pen-down path.


class PenDownProgress:
    """Fraction of pen-down path length drawn, by (phase index, target index).

    `phases` are drawing_state_machine.Phase objects (duck typed: .targets with
    .position and .pen_down). Inside a pen-down phase the fraction grows with
    arc length to the current target; any other phase reports what the earlier
    pen-down phases add up to, so replacing a travel phase's targets (as the
    controller does for the home -> approach leg) changes nothing."""

    def __init__(self, phases: Sequence[Any]):
        self._base: List[float] = []
        self._arcs: List[Optional[List[float]]] = []
        total = 0.0
        for phase in phases:
            self._base.append(total)
            targets = list(phase.targets)
            if targets and all(bool(t.pen_down) for t in targets):
                arc = [0.0]
                for previous, current in zip(targets[:-1], targets[1:]):
                    arc.append(arc[-1] + math.dist(list(previous.position)[:2], list(current.position)[:2]))
                self._arcs.append(arc)
                total += arc[-1]
            else:
                self._arcs.append(None)
        self.total_m = total

    def at(self, phase_index: int, target_index: int) -> float:
        if self.total_m <= 0.0 or not self._base:
            return 0.0
        phase_index = min(max(int(phase_index), 0), len(self._base) - 1)
        drawn = self._base[phase_index]
        arc = self._arcs[phase_index]
        if arc is not None:
            drawn += arc[min(max(int(target_index), 0), len(arc) - 1)]
        return min(max(drawn / self.total_m, 0.0), 1.0)


# ── Writer.


class LiveTelemetry:
    """Writes one run's status file and executed trace.

    Call sample() for every controller tick (or replay step); the status file
    is rewritten whenever `period_s` of sim time has passed since the last
    write (None: only on flush()/finish()). Pen-down samples go to the trace
    once they are `trace_spacing_m` from the last kept one, and the end of each
    pen-down segment is always kept. Trace lines are buffered and appended
    together with each status write, so a run costs two small file operations
    per period. Construction clears any earlier trace for the prefix."""

    def __init__(
        self,
        directory: str | Path,
        prefix: str,
        mapping: Optional[Mapping[str, Any]] = None,
        *,
        phase: str = "INIT",
        period_s: Optional[float] = DEFAULT_PERIOD_S,
        trace_spacing_m: float = DEFAULT_TRACE_SPACING_M,
        extra: Optional[Mapping[str, Any]] = None,
        clock=time.time,
    ):
        self.prefix = str(prefix)
        self.mapping = dict(mapping) if mapping else None
        self.status_file = status_path(directory, self.prefix)
        self.trace_file = trace_path(directory, self.prefix)
        self.period_s = period_s
        self.trace_spacing_m = float(trace_spacing_m)
        self.extra = dict(extra or {})
        self.clock = clock
        self.finished = False
        self.trace_count = 0
        self._pending: List[str] = []
        self._last_kept: Optional[Tuple[float, float]] = None
        self._held: Optional[Dict[str, Any]] = None
        self._pen_down = False
        self._segment = -1
        self._last_write_sim: Optional[float] = None
        self._status: Dict[str, Any] = {
            "prefix": self.prefix,
            "phase": str(phase),
            "progress": 0.0,
            "tip_world": None,
            "tip_canvas": None,
            "error_mm": None,
            "sim_time": 0.0,
            "status": STATUS_DRAWING,
        }
        try:
            self.trace_file.parent.mkdir(parents=True, exist_ok=True)
            self.trace_file.write_text("", encoding="utf-8")
        except OSError:
            pass

    @property
    def status(self) -> Dict[str, Any]:
        return dict(self._status)

    def canvas(self, xy: Sequence[float]) -> Optional[List[float]]:
        return world_to_canvas(self.mapping, xy) if self.mapping else None

    def sample(
        self,
        sim_time: float,
        phase: str,
        tip_world: Sequence[float],
        *,
        error_m: Optional[float] = None,
        progress: Optional[float] = None,
        pen_down: bool = False,
        stroke_id: Optional[int] = None,
        **fields: Any,
    ) -> bool:
        """Record one tip sample. Returns True when the status file was written."""
        if self.finished:
            return False
        tip = [float(v) for v in tip_world]
        uv = self.canvas(tip[:2])
        if pen_down:
            self._trace(sim_time, tip, uv, stroke_id)
        elif self._pen_down:
            self._end_segment()
        self._pen_down = bool(pen_down)
        status = self._status
        status.update(fields)
        status.update(
            {
                "phase": str(phase),
                "tip_world": [round(v, 6) for v in tip],
                "tip_canvas": [round(v, 6) for v in uv] if uv else None,
                "error_mm": None if error_m is None else round(float(error_m) * 1000.0, 4),
                "sim_time": round(float(sim_time), 4),
                "pen_down": bool(pen_down),
                "stroke_id": stroke_id,
            }
        )
        if progress is not None:
            # Never runs backwards, even if the controller's index does.
            status["progress"] = round(max(float(status["progress"]), min(max(float(progress), 0.0), 1.0)), 6)
        due = self.period_s is not None and (
            self._last_write_sim is None or float(sim_time) - self._last_write_sim >= self.period_s
        )
        if due:
            self.flush()
        return due

    def update(self, **fields: Any) -> None:
        """Set extra status fields for the next write (e.g. the replay's matching frame)."""
        self._status.update(fields)

    def flush(self) -> bool:
        """Append buffered trace lines and rewrite the status file now."""
        if self._pending:
            try:
                with open(self.trace_file, "a", encoding="utf-8") as handle:
                    handle.write("".join(self._pending))
                self._pending = []
            except OSError:
                pass
        self._last_write_sim = float(self._status["sim_time"])
        return write_status(self.status_file, self._payload())

    def finish(self, sim_time: float, status: str, failure_reason: Optional[str] = None, **fields: Any) -> bool:
        """Final status: "finished" (progress 1) or "failed". Later calls are ignored."""
        if self.finished:
            return False
        self._end_segment()
        self._status.update(fields)
        self._status["sim_time"] = round(float(sim_time), 4)
        self._status["status"] = str(status)
        self._status["phase"] = fields.get("phase", "FINISHED" if status == STATUS_FINISHED else "FAILED")
        if status == STATUS_FINISHED:
            self._status["progress"] = 1.0
        if failure_reason:
            self._status["failure_reason"] = str(failure_reason)
        written = self.flush()
        self.finished = True
        return written

    def _payload(self) -> Dict[str, Any]:
        payload = dict(self.extra)
        payload.update(self._status)
        payload["trace_samples"] = self.trace_count
        payload["updated_at"] = round(float(self.clock()), 3)
        return payload

    def _trace(self, sim_time: float, tip: List[float], uv: Optional[List[float]], stroke_id: Optional[int]) -> None:
        if not self._pen_down:
            self._segment += 1
            self._last_kept = None
        point = {
            "stroke_id": stroke_id,
            "segment": self._segment,
            "u": round(uv[0], 6) if uv else None,
            "v": round(uv[1], 6) if uv else None,
            "x": round(tip[0], 6),
            "y": round(tip[1], 6),
            "sim_time": round(float(sim_time), 4),
        }
        if self._last_kept is None or math.dist(self._last_kept, tip[:2]) >= self.trace_spacing_m:
            self._keep(point)
            self._held = None
        else:
            self._held = point

    def _end_segment(self) -> None:
        if self._held is not None:
            self._keep(self._held)
            self._held = None

    def _keep(self, point: Dict[str, Any]) -> None:
        self._last_kept = (point["x"], point["y"])
        self._pending.append(json.dumps(point) + "\n")
        self.trace_count += 1
