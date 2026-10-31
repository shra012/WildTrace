"""Hybrid policy control (#46): a learned policy draws, IK steps in where it stalls or leaves a gap.

Both hand-overs come from the run's own bookkeeping, not from the policy:
- stall: the target has not advanced for `stall_s`. Camera policies do this at ink they already drew
  ahead of the pen (the cat 323bd30d cheek notch on the twin, camera_v3 to camera_v5): IK then drives
  the next `assist_m` of path and hands back to the policy.
- gap: a stretch of path behind the pen with no ink, by the labeller's own rule
  (kinematic_labeller._Coverage). IK goes back along the line past the gap and forward again to where
  the pen left off, as the labeller teaches, while the run's progress stays put.
Ink is tracked the way the camera page renders it (pen down, tip within INK_CONTACT_M of the target
height), with optional injected skips (act_features.SkipSchedule) for the skip test. Pure numpy; safe
to import inside Isaac Sim's python.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from act_features import INK_CONTACT_M, SkipSchedule
from kinematic_labeller import DEFAULT_GAPS, MODE_BACK, MODE_FORWARD, MODE_NORMAL, _Coverage

POLICY, IK, REDRAW = "policy", "ik", "redraw"


class PolicyAssist:
    """Per tick: record ink with `ink`, then ask `mode` who drives and, for a redraw, where to."""

    def __init__(self, targets, positions: np.ndarray, arc: np.ndarray, *, stall_s: float = 1.5,
                 assist_m: float = 0.005, gaps: dict | None = None, skips: SkipSchedule | None = None,
                 redraw_step_m: float = 0.0005, reach_m: float = 0.0006, redraw_speed_m_s: float = 0.0025):
        self.targets = list(targets)
        self.positions = np.asarray(positions, dtype=np.float64)
        self.arc = np.asarray(arc, dtype=np.float64)
        self.stall_s, self.assist_m = float(stall_s), float(assist_m)
        self.gap_config = {**DEFAULT_GAPS, **(gaps or {})}
        self.coverage = _Coverage(self.targets, self.positions, self.arc, self.gap_config["cover_tol_m"])
        self.skips = skips
        self.redraw_step_m, self.reach_m, self.redraw_speed_m_s = float(redraw_step_m), float(reach_m), float(redraw_speed_m_s)
        self.assist_until: int | None = None
        self.redraw: np.ndarray | None = None
        self.redraw_at = self.redraw_turn = 0
        self.redraw_deadline = 0.0
        self.best_index, self.since = -1, 0.0
        self.last_ink: np.ndarray | None = None
        self.stall_assists = self.redraws = self.redraws_given_up = 0

    def point_at(self, s: float) -> np.ndarray:
        return np.array([np.interp(s, self.arc, self.positions[:, axis]) for axis in range(3)])

    def ink(self, index: int, tip: Sequence[float]) -> bool:
        """Record this tick's ink at waypoint `index`; True when the pen touched the paper but left no ink."""
        target = self.targets[index]
        tip = np.asarray(tip, dtype=np.float64)
        contact = bool(target.pen_down) and float(tip[2] - target.position[2]) < INK_CONTACT_M
        inked = self.skips.inks(contact, tip[:2]) if self.skips is not None else contact
        if inked:
            self.coverage.ink(tip if self.last_ink is None else self.last_ink, tip, float(self.arc[index]))
        self.last_ink = tip if inked else None
        return contact and not inked

    def mode(self, now_s: float, index: int, tip: Sequence[float]) -> tuple[str, np.ndarray | None]:
        """(POLICY | IK | REDRAW, the redraw point to drive to or None) for this tick at waypoint `index`."""
        tip = np.asarray(tip, dtype=np.float64)
        if self.redraw is not None:
            if now_s > self.redraw_deadline:
                self.redraws_given_up += 1
            else:
                while self.redraw_at < len(self.redraw) and np.linalg.norm(tip - self.redraw[self.redraw_at]) <= self.reach_m:
                    self.redraw_at += 1
                if self.redraw_at < len(self.redraw):
                    return REDRAW, self.redraw[self.redraw_at]
            self.redraw = None
            self.best_index, self.since = index, float(now_s)  # the redraw's time is not a stall
        if index > self.best_index:
            self.best_index, self.since = index, float(now_s)
        target = self.targets[index]
        if target.pen_down and target.state.startswith("DRAW"):
            s = float(self.arc[index])
            gap = self.coverage.gap_behind(s, target.stroke_id, self.gap_config["min_gap_m"],
                                           self.gap_config["detect_after_m"], self.gap_config["lookback_m"])
            if gap is not None:
                back = max(gap[0] - self.gap_config["margin_m"], self.coverage.stroke_start_arc(target.stroke_id))
                count = max(int(np.ceil((s - back) / self.redraw_step_m)), 1)
                arcs = np.r_[np.linspace(s, back, count + 1), np.linspace(back, s, count + 1)[1:]]
                self.redraw = np.stack([self.point_at(a) for a in arcs])
                self.redraw_at, self.redraw_turn = 0, count
                self.redraw_deadline = float(now_s) + 2.0 + 2.0 * (s - back) / self.redraw_speed_m_s
                self.assist_until = None
                self.redraws += 1
                return REDRAW, self.redraw[0]
        if self.assist_until is not None:
            if index < self.assist_until:
                return IK, None
            self.assist_until = None
            self.since = float(now_s)
        elif now_s - self.since > self.stall_s and index < len(self.targets) - 1:
            limit = float(self.arc[index]) + self.assist_m
            self.assist_until = int(min(max(np.searchsorted(self.arc, limit, side="left"), index + 1),
                                        len(self.targets) - 1))
            self.stall_assists += 1
            return IK, None
        return POLICY, None

    @property
    def teacher_mode(self) -> int:
        """The labeller's mode for this tick (kinematic_labeller.MODE_*): 1 going back to the gap, 2 coming
        forward again, 0 otherwise; what a --mode-state policy is shown when it does the redraw itself."""
        if self.redraw is None:
            return MODE_NORMAL
        return MODE_BACK if self.redraw_at <= self.redraw_turn else MODE_FORWARD

    def stats(self) -> dict:
        return {"stall_assists": self.stall_assists, "redraws": self.redraws, "redraws_given_up": self.redraws_given_up,
                "skips": self.skips.skips if self.skips is not None else 0,
                "uncovered_fraction": self.coverage.uncovered_fraction()}
