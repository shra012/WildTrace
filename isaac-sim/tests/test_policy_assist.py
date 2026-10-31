from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from act_features import path_arc_length  # noqa: E402
from drawing_state_machine import build_motion_sequence, flatten_desired_targets  # noqa: E402
from policy_assist import IK, POLICY, REDRAW, PolicyAssist  # noqa: E402

PEN_DOWN_Z = 0.201


def _assist(**kwargs):
    strokes = [{"stroke_id": 0, "points": [[0.40, 0.00], [0.44, 0.00]]}]
    phases = build_motion_sequence(strokes, pen_down_z=PEN_DOWN_Z, pen_up_z=0.235, approach_height=0.27,
                                   max_cartesian_step=0.0005)
    targets = flatten_desired_targets(phases)
    positions = np.asarray([t.position for t in targets])
    return PolicyAssist(targets, positions, path_arc_length(positions), **kwargs), targets


def _draw_indices(targets):
    return [i for i, t in enumerate(targets) if t.pen_down and t.state.startswith("DRAW")]


class PolicyAssistTests(unittest.TestCase):
    def test_policy_drives_while_the_pen_inks_and_progresses(self):
        assist, targets = _assist()
        for tick, index in enumerate(_draw_indices(targets)):
            tip = targets[index].position
            self.assertFalse(assist.ink(index, tip))
            self.assertEqual(assist.mode(tick * 0.02, index, tip), (POLICY, None))
        self.assertEqual(assist.stats()["redraws"], 0)
        self.assertEqual(assist.stats()["stall_assists"], 0)

    def test_stall_hands_the_next_stretch_to_ik_then_back(self):
        assist, targets = _assist(stall_s=1.0, assist_m=0.005)
        draw = _draw_indices(targets)
        for earlier in draw[:21]:
            assist.ink(earlier, targets[earlier].position)
        index = draw[20]
        tip = targets[index].position
        self.assertEqual(assist.mode(0.0, index, tip)[0], POLICY)
        self.assertEqual(assist.mode(0.9, index, tip)[0], POLICY)
        self.assertEqual(assist.mode(1.1, index, tip)[0], IK)
        end = assist.assist_until
        self.assertAlmostEqual(assist.arc[end] - assist.arc[index], 0.005, delta=0.0006)
        for step, ahead in enumerate(range(index + 1, end + 1)):
            assist.ink(ahead, targets[ahead].position)
            self.assertEqual(assist.mode(1.2 + 0.02 * step, ahead, targets[ahead].position)[0],
                             POLICY if ahead == end else IK)
        self.assertEqual(assist.stats()["stall_assists"], 1)

    def test_gap_behind_the_pen_is_redrawn_back_and_forth(self):
        assist, targets = _assist()
        draw = _draw_indices(targets)
        lifted = set(draw[10:20])  # ~5 mm with the pen off the paper
        mode = POLICY
        for tick, index in enumerate(draw[:40]):
            tip = targets[index].position + (np.array([0.0, 0.0, 0.003]) if index in lifted else 0.0)
            assist.ink(index, tip)
            mode, point = assist.mode(tick * 0.02, index, tip)
            if mode == REDRAW:
                break
        self.assertEqual(mode, REDRAW)
        self.assertEqual(assist.teacher_mode, 1)
        redraw = assist.redraw
        s = assist.arc[index]
        along = np.interp(redraw[:, 0], assist.positions[draw, 0], assist.arc[draw])
        turn = int(np.argmin(along))
        # Back to the start of the gap (ink reaches cover_tol_m into it), then forward again.
        self.assertLessEqual(along[turn], assist.arc[draw[10]] + 1e-6)
        self.assertTrue(np.all(np.diff(along[: turn + 1]) <= 1e-9))
        self.assertTrue(np.all(np.diff(along[turn:]) >= -1e-9))
        self.assertAlmostEqual(along[-1], s, delta=1e-6)
        np.testing.assert_allclose(redraw[:, 2], PEN_DOWN_Z)
        # Following the points to the end hands back to the policy and inks the gap.
        now, seen_modes = 1.0, [1]
        while True:
            assist.ink(index, point)
            mode, point = assist.mode(now, index, point if point is not None else redraw[-1])
            now += 0.02
            if mode != REDRAW:
                break
            if assist.teacher_mode != seen_modes[-1]:
                seen_modes.append(assist.teacher_mode)
        self.assertEqual(mode, POLICY)
        self.assertEqual(assist.teacher_mode, 0)
        self.assertEqual(seen_modes, [1, 2])
        self.assertEqual(assist.stats()["redraws"], 1)
        self.assertIsNone(assist.coverage.gap_behind(s, 0, 0.0015, 0.001, 0.015))

    def test_redraw_that_cannot_finish_is_given_up(self):
        assist, targets = _assist()
        draw = _draw_indices(targets)
        for tick, index in enumerate(draw[:40]):
            tip = targets[index].position + (np.array([0.0, 0.0, 0.003]) if 10 <= tick < 20 else 0.0)
            assist.ink(index, tip)
            if assist.mode(tick * 0.02, index, tip)[0] == REDRAW:
                break
        stuck = assist.redraw[0] + np.array([0.0, 0.01, 0.0])
        self.assertEqual(assist.mode(1.0, index, stuck)[0], REDRAW)
        self.assertEqual(assist.mode(60.0, index, stuck)[0], POLICY)
        self.assertEqual(assist.stats()["redraws_given_up"], 1)

    def test_skip_schedule_reports_skipped_contact_ticks(self):
        from act_features import SkipSchedule

        skips = SkipSchedule(np.random.default_rng(0), 0.004, (0.003, 0.004))
        assist, targets = _assist(skips=skips)
        flags = [assist.ink(index, targets[index].position) for index in _draw_indices(targets)]
        self.assertTrue(any(flags))
        self.assertFalse(all(flags))
        self.assertGreater(assist.stats()["skips"], 0)


if __name__ == "__main__":
    unittest.main()
