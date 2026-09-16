from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from coordinate_mapper import map_trajectory_to_plane, resample_polyline
from drawing_state_machine import FINISHED, HOME, MultiStrokeStateMachine, build_motion_sequence, flatten_desired_targets
from trajectory_loader import TrajectoryError, load_trajectory, standardize_trajectory


class TrajectoryTests(unittest.TestCase):
    def test_puppy_sample_has_exactly_two_strokes(self):
        trajectory = load_trajectory(PROJECT_ROOT / "data" / "puppy_two_stroke.json")
        self.assertEqual(trajectory["drawing_id"], "puppy_two_stroke")
        self.assertEqual([stroke["stroke_id"] for stroke in trajectory["strokes"]], [0, 1])
        self.assertGreaterEqual(len(trajectory["strokes"][0]["points"]), 40)
        self.assertGreaterEqual(len(trajectory["strokes"][1]["points"]), 40)

    def test_load_existing_gold_json(self):
        source = Path(r"C:\isaacsim\wildtrace_samples\cat_full.json")
        if not source.is_file():
            self.skipTest("Read-only prior sample is unavailable")
        trajectory = load_trajectory(source, first_n_strokes=2)
        self.assertEqual(trajectory["drawing_id"], "demo_cat_full")
        self.assertEqual([stroke["stroke_id"] for stroke in trajectory["strokes"]], [0, 1])
        self.assertEqual([len(stroke["points"]) for stroke in trajectory["strokes"]], [49, 7])

    def test_load_fallback_json(self):
        trajectory = load_trajectory(PROJECT_ROOT / "data" / "fallback_two_stroke.json")
        self.assertEqual(trajectory["drawing_id"], "synthetic_two_stroke_test")
        self.assertEqual(len(trajectory["strokes"]), 2)

    def test_selected_sample_has_exactly_two_strokes(self):
        trajectory = load_trajectory(PROJECT_ROOT / "data" / "sample_two_stroke.json")
        self.assertEqual(len(trajectory["strokes"]), 2)

    def test_non_finite_rejected_and_consecutive_duplicates_removed(self):
        with self.assertRaises(TrajectoryError):
            standardize_trajectory(
                {"drawing_id": "bad", "strokes": [{"stroke_id": 0, "points": [[0, 0], [float("nan"), 1]]}]}
            )
        cleaned = standardize_trajectory(
            {"drawing_id": "dup", "strokes": [{"stroke_id": 0, "points": [[0, 0], [0, 0], [1, 1]]}]}
        )
        self.assertEqual(cleaned["strokes"][0]["points"], [[0.0, 0.0], [1.0, 1.0]])

    def test_normalization_preserves_aspect_and_workspace(self):
        trajectory = load_trajectory(PROJECT_ROOT / "data" / "fallback_two_stroke.json")
        mapped = map_trajectory_to_plane(
            trajectory, center_xy=[0.45, 0.0], size_xy=[0.16, 0.12], max_step=0.01, flip_image_y=True
        )
        points = np.vstack([stroke["points"] for stroke in mapped["strokes"]])
        self.assertGreaterEqual(points[:, 0].min(), 0.45 - 0.08 - 1e-12)
        self.assertLessEqual(points[:, 0].max(), 0.45 + 0.08 + 1e-12)
        self.assertGreaterEqual(points[:, 1].min(), -0.06 - 1e-12)
        self.assertLessEqual(points[:, 1].max(), 0.06 + 1e-12)
        self.assertAlmostEqual(mapped["mapping"]["uniform_scale"], min(0.16 / 0.5, 0.12 / 0.35))
        self.assertGreater(mapped["strokes"][0]["points"][0][1], mapped["strokes"][0]["points"][1][1])

    def test_resampling_retains_corner_and_caps_step(self):
        points = resample_polyline([[0, 0], [1, 0], [1, 1]], 0.3)
        self.assertTrue(any(np.allclose(point, [1, 0]) for point in points))
        self.assertLessEqual(np.linalg.norm(np.diff(points, axis=0), axis=1).max(), 0.3 + 1e-12)


class StateMachineTests(unittest.TestCase):
    def setUp(self):
        self.strokes = [
            {"stroke_id": 7, "points": [[0.4, 0.0], [0.41, 0.0]]},
            {"stroke_id": 9, "points": [[0.43, 0.01], [0.44, 0.01]]},
        ]
        self.phases = build_motion_sequence(
            self.strokes, pen_down_z=0.2, pen_up_z=0.23, approach_height=0.26, max_cartesian_step=0.005
        )

    def test_pen_up_transition_inserted(self):
        order = flatten_desired_targets(self.phases)
        first_draw_last = max(i for i, target in enumerate(order) if target.state == "DRAW_STROKE_7")
        second_draw_first = min(i for i, target in enumerate(order) if target.state == "DRAW_STROKE_9")
        between = order[first_draw_last + 1 : second_draw_first]
        self.assertTrue(between)
        self.assertTrue(all(not target.pen_down for target in between))
        self.assertTrue(any(target.state == "LIFT_PEN_STROKE_7" for target in between))
        self.assertTrue(any(target.state == "MOVE_TO_STROKE_9" for target in between))

    def test_state_machine_transitions_only_when_reached(self):
        machine = MultiStrokeStateMachine(self.phases, waypoint_timeout_s=1.0)
        machine.mark_home_reached(0.0)
        machine.update(0.1, reached=False)
        self.assertEqual(machine.state, "APPROACH_STROKE_7")
        time_s = 0.1
        while not machine.complete:
            machine.update(time_s, reached=True)
            time_s += 0.01
        self.assertEqual(machine.visited_states, [HOME] + [phase.state for phase in self.phases] + [FINISHED])

    def test_timeout_fails(self):
        machine = MultiStrokeStateMachine(self.phases, waypoint_timeout_s=0.2)
        machine.mark_home_reached(0.0)
        machine.update(0.21, reached=False)
        self.assertTrue(machine.failed)

    def test_five_stroke_sequence_completes_in_order(self):
        strokes = [
            {"stroke_id": i, "points": [[0.3 + 0.01 * i, 0.0], [0.31 + 0.01 * i, 0.0], [0.32 + 0.01 * i, 0.01]]}
            for i in range(5)
        ]
        phases = build_motion_sequence(
            strokes, pen_down_z=0.2, pen_up_z=0.23, approach_height=0.26, max_cartesian_step=0.005
        )
        draw_states = [phase.state for phase in phases if phase.state.startswith("DRAW_STROKE_")]
        self.assertEqual(draw_states, [f"DRAW_STROKE_{i}" for i in range(5)])
        machine = MultiStrokeStateMachine(phases, waypoint_timeout_s=1.0)
        machine.mark_home_reached(0.0)
        time_s = 0.0
        while not machine.complete and not machine.failed:
            time_s += 0.01
            machine.update(time_s, reached=True)
        self.assertTrue(machine.complete)
        self.assertEqual(machine.visited_states[-1], FINISHED)


if __name__ == "__main__":
    unittest.main()
