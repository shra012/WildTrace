from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drawing_state_machine import build_motion_sequence
from path_geometry import (
    corner_flags,
    densify_near_corners,
    heading_error,
    heading_from_history,
    limit_joint_delta,
    polyline_headings,
    turn_angles,
    wrap_angle,
)


class WrapAngleTests(unittest.TestCase):
    def test_scalar_wraps_into_range(self):
        self.assertAlmostEqual(wrap_angle(0.4), 0.4)
        # The two half-turns are both valid representations of pi.
        self.assertAlmostEqual(abs(wrap_angle(3 * math.pi)), math.pi)
        self.assertAlmostEqual(abs(wrap_angle(-3 * math.pi)), math.pi)

    def test_array_wraps_elementwise(self):
        wrapped = wrap_angle([0.0, 2 * math.pi + 0.25, -2 * math.pi - 0.25])
        np.testing.assert_allclose(wrapped, [0.0, 0.25, -0.25], atol=1e-12)

    def test_heading_error_takes_short_way_round(self):
        # 170 deg desired against -170 deg actual is a 20 deg turn, not 340.
        error = heading_error(math.radians(170.0), math.radians(-170.0))
        self.assertAlmostEqual(math.degrees(error), -20.0, places=9)
        self.assertLessEqual(abs(error), math.pi)

    def test_heading_error_sign_follows_turn_direction(self):
        self.assertGreater(heading_error(math.radians(10.0), math.radians(-10.0)), 0.0)
        self.assertLess(heading_error(math.radians(-10.0), math.radians(10.0)), 0.0)


class HeadingTests(unittest.TestCase):
    def test_headings_follow_segments(self):
        headings = polyline_headings([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        np.testing.assert_allclose(headings, [0.0, math.pi / 2, math.pi / 2], atol=1e-12)

    def test_degenerate_segment_inherits_previous_heading(self):
        headings = polyline_headings([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        np.testing.assert_allclose(headings, np.zeros(4), atol=1e-12)

    def test_single_point_has_no_heading(self):
        np.testing.assert_allclose(polyline_headings([[0.5, 0.5]]), [0.0])

    def test_three_dimensional_input_uses_xy(self):
        headings = polyline_headings([[0.0, 0.0, 0.2], [0.0, 1.0, 0.2]])
        np.testing.assert_allclose(headings, [math.pi / 2, math.pi / 2], atol=1e-12)

    def test_chord_length_averages_out_zigzag_noise(self):
        # A straight +x path carrying lateral noise larger than the spacing,
        # so point-to-point tangents swing wildly but the path is straight.
        points = [[0.001 * i, 0.0006 * (i % 2)] for i in range(21)]
        point_to_point = np.degrees(np.abs(polyline_headings(points)))
        scale_matched = np.degrees(np.abs(polyline_headings(points, min_segment_m=0.005)))
        self.assertGreater(point_to_point[:-1].max(), 25.0)
        self.assertLess(scale_matched[:-1].max(), 10.0)

    def test_chord_tangent_matches_forward_difference_on_a_clean_path(self):
        points = [[0.0, 0.0], [0.01, 0.0], [0.02, 0.0], [0.03, 0.0]]
        np.testing.assert_allclose(
            polyline_headings(points, min_segment_m=0.005), np.zeros(4), atol=1e-12
        )

    def test_trailing_matches_leading_on_a_straight_path(self):
        points = [[0.001 * i, 0.0] for i in range(20)]
        np.testing.assert_allclose(
            polyline_headings(points, 0.004, trailing=True),
            polyline_headings(points, 0.004, trailing=False),
            atol=1e-12,
        )

    def test_trailing_reports_the_arriving_direction(self):
        # Ten mm east then ten mm north. At the turn the arriving direction is
        # still east, while the leaving direction is already north.
        points = [[0.001 * i, 0.0] for i in range(11)] + [[0.010, 0.001 * i] for i in range(1, 11)]
        corner = 10
        trailing = polyline_headings(points, 0.004, trailing=True)
        leading = polyline_headings(points, 0.004, trailing=False)
        self.assertAlmostEqual(math.degrees(trailing[corner]), 0.0, places=6)
        self.assertAlmostEqual(math.degrees(leading[corner]), 90.0, places=6)

    def test_trailing_start_backfills_instead_of_reporting_zero(self):
        points = [[0.0, 0.0], [0.0, 0.001], [0.0, 0.002], [0.0, 0.010]]
        trailing = polyline_headings(points, 0.004, trailing=True)
        np.testing.assert_allclose(np.degrees(trailing), np.full(4, 90.0), atol=1e-9)


class HeadingFromHistoryTests(unittest.TestCase):
    def test_none_until_threshold_is_cleared(self):
        history = [[0.0, 0.0], [0.0001, 0.0]]
        self.assertIsNone(heading_from_history(history, 0.0005))
        self.assertIsNone(heading_from_history([[0.0, 0.0]], 0.0005))

    def test_direction_measured_once_far_enough(self):
        history = [[0.0, 0.0], [0.0003, 0.0], [0.0006, 0.0]]
        self.assertAlmostEqual(heading_from_history(history, 0.0005), 0.0)

    def test_settling_jitter_is_ignored(self):
        # Real travel along +x, then sub-threshold jitter that includes a
        # backwards step. Consecutive differencing would report 180 degrees.
        history = [[0.0, 0.0], [0.002, 0.0], [0.002_05, 0.0], [0.002_02, 0.0]]
        heading = heading_from_history(history, 0.0005)
        self.assertAlmostEqual(math.degrees(heading), 0.0, places=6)

    def test_most_recent_qualifying_sample_wins(self):
        # Travelled +x then turned +y; the reported heading is the new leg.
        history = [[0.0, 0.0], [0.01, 0.0], [0.01, 0.001]]
        self.assertAlmostEqual(math.degrees(heading_from_history(history, 0.0005)), 90.0)

    def test_three_dimensional_history_uses_xy(self):
        history = [[0.0, 0.0, 0.2], [0.0, 0.001, 0.2]]
        self.assertAlmostEqual(math.degrees(heading_from_history(history, 0.0005)), 90.0)


class CornerTests(unittest.TestCase):
    def test_right_angle_is_a_corner_and_endpoints_are_not(self):
        points = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
        np.testing.assert_allclose(turn_angles(points), [0.0, math.pi / 2, 0.0], atol=1e-12)
        np.testing.assert_array_equal(corner_flags(points, 35.0), [False, True, False])

    def test_straight_line_has_no_corners(self):
        points = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
        self.assertFalse(corner_flags(points, 35.0).any())

    def test_window_marks_neighbours(self):
        points = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [2.0, 2.0]]
        np.testing.assert_array_equal(
            corner_flags(points, 35.0, window=1), [False, True, True, True, False]
        )

    def test_invalid_arguments_rejected(self):
        with self.assertRaises(ValueError):
            corner_flags([[0.0, 0.0], [1.0, 0.0]], 0.0)
        with self.assertRaises(ValueError):
            corner_flags([[0.0, 0.0], [1.0, 0.0]], 35.0, window=-1)


class DensifyTests(unittest.TestCase):
    def setUp(self):
        self.points = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [2.0, 2.0]]

    def test_factor_one_is_a_no_op(self):
        np.testing.assert_allclose(
            densify_near_corners(self.points, 35.0, 0, 1), np.asarray(self.points)
        )

    def test_corner_segments_subdivided_and_vertices_preserved(self):
        dense = densify_near_corners(self.points, 35.0, 0, 2)
        for vertex in self.points:
            self.assertTrue(any(np.allclose(vertex, point) for point in dense))
        self.assertGreater(len(dense), len(self.points))
        # Only the two segments touching the corner gain a midpoint.
        self.assertEqual(len(dense), len(self.points) + 2)

    def test_straight_path_is_untouched(self):
        straight = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
        np.testing.assert_allclose(
            densify_near_corners(straight, 35.0, 0, 3), np.asarray(straight)
        )

    def test_step_never_exceeds_original_spacing(self):
        dense = densify_near_corners(self.points, 35.0, 1, 2)
        steps = np.linalg.norm(np.diff(dense, axis=0), axis=1)
        self.assertLessEqual(steps.max(), 1.0 + 1e-12)


class JointSlewTests(unittest.TestCase):
    def test_large_step_is_clamped_per_joint(self):
        current = np.zeros(3)
        proposed = np.asarray([0.5, -0.5, 0.01])
        limited = limit_joint_delta(current, proposed, 0.1)
        np.testing.assert_allclose(limited, [0.1, -0.1, 0.01])

    def test_small_step_passes_through(self):
        current = np.asarray([1.0, 2.0])
        proposed = np.asarray([1.01, 1.99])
        np.testing.assert_allclose(limit_joint_delta(current, proposed, 0.1), proposed)

    def test_disabled_limit_returns_proposal(self):
        proposed = np.asarray([5.0, -5.0])
        np.testing.assert_allclose(limit_joint_delta(np.zeros(2), proposed, None), proposed)
        np.testing.assert_allclose(limit_joint_delta(np.zeros(2), proposed, 0.0), proposed)

    def test_clamped_command_stays_between_measurement_and_proposal(self):
        current = np.asarray([0.2, -1.0])
        proposed = np.asarray([0.9, -2.5])
        limited = limit_joint_delta(current, proposed, 0.1)
        self.assertTrue(np.all((limited - current) * (proposed - current) >= 0.0))
        self.assertTrue(np.all(np.abs(limited - current) <= np.abs(proposed - current)))

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            limit_joint_delta(np.zeros(3), np.zeros(2), 0.1)


class MotionSequenceDensificationTests(unittest.TestCase):
    def _draw_targets(self, **kwargs):
        strokes = [{"stroke_id": 0, "points": [[0.40, 0.00], [0.44, 0.00], [0.44, 0.04]]}]
        phases = build_motion_sequence(
            strokes,
            pen_down_z=0.2,
            pen_up_z=0.23,
            approach_height=0.26,
            max_cartesian_step=0.005,
            **kwargs,
        )
        return next(phase for phase in phases if phase.state == "DRAW_STROKE_0").targets

    def test_default_sequence_is_unchanged(self):
        self.assertEqual(len(self._draw_targets()), len(self._draw_targets(corner_densify_factor=1)))

    def test_densification_adds_corner_targets_only(self):
        baseline = self._draw_targets()
        dense = self._draw_targets(corner_densify_window=1, corner_densify_factor=2)
        self.assertGreater(len(dense), len(baseline))
        corner = np.asarray([0.44, 0.00, 0.2])
        self.assertTrue(any(np.allclose(target.position, corner) for target in dense))


if __name__ == "__main__":
    unittest.main()
