from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from act_features import (  # noqa: E402
    PATH_WINDOW_POINTS,
    act_frames,
    chased_index,
    path_arc_length,
    environment_state,
    environment_state_names,
    path_window,
    reverse_strokes,
    sample_augmentation,
    shift_paper_heights,
)
from demonstration import DemonstrationRecorder  # noqa: E402
from drawing_state_machine import build_motion_sequence  # noqa: E402
from kinematic_labeller import LabelError, label_phases, with_approach_from  # noqa: E402


def _row(recorder, **overrides):
    values = dict(
        stroke_id=0,
        waypoint_index=0,
        simulation_time_s=0.0,
        joint_positions_rad=np.zeros(7),
        joint_velocities_rad_s=np.zeros(7),
        pen_tip_position_m=np.zeros(3),
        pen_tip_rotation_matrix=np.eye(3),
        current_target_m=np.asarray([0.001, 0.0, 0.0]),
        upcoming_targets_m=np.zeros((recorder.lookahead_points, 3)),
        pen_down=True,
        state="DRAW_STROKE_0",
    )
    values.update(overrides)
    recorder.append(**values)


class PathWindowTests(unittest.TestCase):
    def test_equal_arc_spacing_along_straight_line(self):
        targets = np.c_[np.linspace(0.0, 0.1, 101), np.zeros(101), np.zeros(101)]
        window = path_window(targets, 10, points=5, spacing_m=0.002)
        np.testing.assert_allclose(window[:, 0], [0.010, 0.012, 0.014, 0.016, 0.018], atol=1e-12)

    def test_pads_with_last_point_past_path_end(self):
        targets = np.asarray([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]])
        window = path_window(targets, 0, points=4, spacing_m=0.002)
        np.testing.assert_allclose(window[1:], np.repeat([[0.001, 0.0, 0.0]], 3, axis=0))

    def test_single_remaining_target_repeats(self):
        window = path_window(np.ones((3, 3)), 2, points=PATH_WINDOW_POINTS)
        self.assertEqual(window.shape, (PATH_WINDOW_POINTS, 3))

    def test_environment_state_layout_matches_names(self):
        tip = np.asarray([[1.0, 2.0, 3.0]])
        window = np.ones((1, PATH_WINDOW_POINTS, 3)) * 2.0
        env = environment_state(tip, tip + 0.5, window, np.asarray([[1.0]]))
        self.assertEqual(env.shape[1], len(environment_state_names()))
        np.testing.assert_allclose(env[0, 3:6], [0.5, 0.5, 0.5])
        np.testing.assert_allclose(env[0, 6:9], [1.0, 0.0, -1.0])
        self.assertEqual(env[0, -1], 1.0)


class ChasedIndexTests(unittest.TestCase):
    def setUp(self):
        self.targets = np.c_[np.arange(0.0, 0.0101, 0.001), np.zeros(11), np.zeros(11)]
        self.arc = path_arc_length(self.targets)

    def test_matches_first_waypoint_ahead_of_projected_tip(self):
        self.assertEqual(chased_index(self.targets, self.arc, 4, [0.0045, 0.0003, 0.0]), 5)

    def test_never_moves_backwards(self):
        self.assertEqual(chased_index(self.targets, self.arc, 7, [0.0015, 0.0, 0.0]), 7)

    def test_clamps_to_last_waypoint(self):
        self.assertEqual(chased_index(self.targets, self.arc, 9, [0.02, 0.0, 0.0]), 10)

    def test_hairpin_does_not_jump_to_return_side(self):
        # Outward leg along x, return leg 0.6 mm away. A tip that drifted toward
        # the return side near the base is still on the outward leg: the return
        # side is ~17 mm further along the path, beyond the arc search window.
        hairpin = np.r_[self.targets, self.targets[::-1][1:] + [0.0, 0.0006, 0.0]]
        self.assertEqual(chased_index(hairpin, path_arc_length(hairpin), 2, [0.0025, 0.0004, 0.0]), 3)


class SpikeTipTests(unittest.TestCase):
    def test_spike_tip_is_not_cut_short(self):
        # Needle tip: out 10 mm along x, back 0.3 mm away. A pen 1 mm before the
        # apex, drifted toward the return leg, is still short of the tip; the
        # return leg is only ~2 mm further along the path from there.
        out = np.c_[np.arange(0.0, 0.0101, 0.0005), np.zeros(21), np.zeros(21)]
        back = out[::-1][1:] + [0.0, 0.0003, 0.0]
        spike = np.r_[out, back]
        arc = path_arc_length(spike)
        index = chased_index(spike, arc, 18, [0.009, 0.00025, 0.0])
        self.assertLessEqual(index, 20, "progress jumped past the apex onto the return leg")


class AugmentationTests(unittest.TestCase):
    def test_sampling_is_seeded_and_bounded(self):
        first = sample_augmentation(np.random.default_rng(3), [0.45, 0.0], [0.16, 0.12], 4)
        second = sample_augmentation(np.random.default_rng(3), [0.45, 0.0], [0.16, 0.12], 4)
        self.assertEqual(first, second)
        self.assertTrue(0.8 <= first["scale"] <= 1.1)
        self.assertLessEqual(abs(first["surface_center_xy_m"][0] - 0.45), 0.02)

    def test_height_jitter_is_opt_in_and_shifts_every_paper_height(self):
        plain = sample_augmentation(np.random.default_rng(3), [0.45, 0.0], [0.16, 0.12], 4)
        self.assertEqual(plain["height_offset_m"], 0.0)
        jittered = sample_augmentation(np.random.default_rng(3), [0.45, 0.0], [0.16, 0.12], 4, height_jitter_m=0.015)
        # Drawn last, so the rest of the episode is unchanged.
        self.assertEqual({k: v for k, v in jittered.items() if k != "height_offset_m"},
                         {k: v for k, v in plain.items() if k != "height_offset_m"})
        self.assertLessEqual(abs(jittered["height_offset_m"]), 0.015)
        drawing = {"paper_top_z_m": 0.2, "pen_down_z_m": 0.201, "pen_up_z_m": 0.235, "approach_height_m": 0.27, "x": 1}
        moved = shift_paper_heights(drawing, -0.01)
        self.assertAlmostEqual(moved["pen_down_z_m"] - moved["paper_top_z_m"], 0.001)
        self.assertAlmostEqual(moved["approach_height_m"], 0.26)
        self.assertEqual(moved["x"], 1)

    def test_reverse_only_listed_strokes(self):
        trajectory = {"drawing_id": "d", "strokes": [{"stroke_id": 0, "points": [[0, 0], [1, 0]]},
                                                     {"stroke_id": 1, "points": [[0, 1], [1, 1]]}]}
        flipped = reverse_strokes(trajectory, [1])
        self.assertEqual(flipped["strokes"][0]["points"], [[0, 0], [1, 0]])
        self.assertEqual(flipped["strokes"][1]["points"], [[1, 1], [0, 1]])
        self.assertEqual(trajectory["strokes"][1]["points"], [[0, 1], [1, 1]])


class RecorderV2Tests(unittest.TestCase):
    def test_v1_recording_still_saves_without_act_fields(self):
        recorder = DemonstrationRecorder("d", 5, 0.005)
        _row(recorder)
        with tempfile.TemporaryDirectory() as tmp:
            data = np.load(recorder.save(Path(tmp) / "demo.npz"))
            self.assertEqual(json.loads(str(data["normalization_metadata_json"]))["format_version"], 1)
            self.assertNotIn("commanded_joint_positions_rad", data.files)
            with self.assertRaises(ValueError):
                act_frames(dict(data))

    def test_v2_round_trip_to_act_frames(self):
        recorder = DemonstrationRecorder("d", 5, 0.005, metadata={"source": "test"})
        for index in range(3):
            _row(
                recorder,
                simulation_time_s=index / 30,
                commanded_joint_positions_rad=np.full(7, index, dtype=np.float64),
                path_window_m=np.zeros((PATH_WINDOW_POINTS, 3)),
            )
        with tempfile.TemporaryDirectory() as tmp:
            data = dict(np.load(recorder.save(Path(tmp) / "demo.npz")))
        self.assertEqual(json.loads(str(data["normalization_metadata_json"]))["format_version"], 2)
        self.assertEqual(json.loads(str(data["recording_metadata_json"])), {"source": "test"})
        frames = act_frames(data)
        self.assertEqual(frames["observation.state"].shape, (3, 21))
        self.assertEqual(frames["observation.environment_state"].shape, (3, 6 + 3 * PATH_WINDOW_POINTS + 1))
        # Commands 0, 1, 2 from a measured start of 0: steps of 0, 1, 1 and the
        # previous command appended to the state.
        np.testing.assert_allclose(frames["action"][:, 0], [0, 1, 1])
        np.testing.assert_allclose(frames["observation.state"][:, 14], [0, 0, 1])

    def test_mixing_formats_is_rejected(self):
        recorder = DemonstrationRecorder("d", 5, 0.005)
        _row(recorder)
        with self.assertRaises(ValueError):
            _row(recorder, commanded_joint_positions_rad=np.zeros(7), path_window_m=np.zeros((PATH_WINDOW_POINTS, 3)))
        with self.assertRaises(ValueError):
            _row(recorder, commanded_joint_positions_rad=np.zeros(7))


def _toy_kinematics():
    """Tip = first three joints; orientation fixed. Exact, branch-free IK."""

    def solve(target, seed):
        return np.r_[np.asarray(target, dtype=np.float64), np.asarray(seed, dtype=np.float64)[3:]]

    def forward(q):
        return np.asarray(q[:3], dtype=np.float64), np.eye(3)

    return solve, forward


def _phases():
    strokes = [
        {"stroke_id": 0, "points": [[0.40, 0.00], [0.42, 0.00], [0.42, 0.02]]},
        {"stroke_id": 1, "points": [[0.45, 0.00], [0.46, 0.01]]},
    ]
    return build_motion_sequence(
        strokes, pen_down_z=0.201, pen_up_z=0.235, approach_height=0.27, max_cartesian_step=0.0015
    )


class KinematicLabellerTests(unittest.TestCase):
    def test_labels_full_episode_with_monotonic_time_and_slew_limit(self):
        solve, forward = _toy_kinematics()
        home = np.asarray([0.38, 0.0, 0.30, 0.0, 0.0, 0.0, 0.0])
        phases = with_approach_from(_phases(), forward(home)[0], 0.0015)
        recorder = DemonstrationRecorder("toy", 5, 0.005)
        stats = label_phases(
            phases, home_q=home, solve_ik=solve, forward=forward, recorder=recorder,
            dt_s=1 / 30, max_joint_delta_rad=0.07,
        )
        times = np.asarray([row["simulation_time_s"] for row in recorder.rows])
        commands = np.stack([row["commanded_joint_positions_rad"] for row in recorder.rows])
        self.assertTrue(np.all(np.diff(times) > 0))
        self.assertLessEqual(np.abs(np.diff(commands, axis=0)).max(), 0.07 + 1e-12)
        self.assertNotIn("HOME", {row["state"] for row in recorder.rows})
        self.assertIn("DRAW_STROKE_1", {row["state"] for row in recorder.rows})
        self.assertLess(stats["pen_down_tip_error_max_m"], 1e-3)
        # The measured pose lags the command by exactly one frame.
        measured = np.stack([row["joint_positions_rad"] for row in recorder.rows])
        np.testing.assert_allclose(measured[1:], commands[:-1])

    def test_noise_injection_labels_correct_back_to_path(self):
        solve, forward = _toy_kinematics()
        home = np.asarray([0.38, 0.0, 0.30, 0.0, 0.0, 0.0, 0.0])
        phases = with_approach_from(_phases(), forward(home)[0], 0.0015)
        recorder = DemonstrationRecorder("toy", 5, 0.005)
        noise = {"rng": np.random.default_rng(0), "sigma_rad": 0.003, "correlation_s": 0.5}
        label_phases(
            phases, home_q=home, solve_ik=solve, forward=forward, recorder=recorder,
            dt_s=1 / 30, max_joint_delta_rad=0.07, noise=noise,
        )
        with tempfile.TemporaryDirectory() as tmp:
            data = dict(np.load(recorder.save(Path(tmp) / "demo.npz")))
        frames = act_frames(data)
        measured = data["joint_positions_rad"][:, :3]
        commanded = data["commanded_joint_positions_rad"][:, :3]
        # The arm wanders off the labelled path ...
        self.assertGreater(np.abs(measured[1:] - commanded[:-1]).max(), 1e-3)
        # ... and each action steps from the executed pose to the IK target.
        np.testing.assert_allclose(
            frames["action"], data["commanded_joint_positions_rad"] - data["previous_command_rad"], atol=1e-6
        )
        np.testing.assert_allclose(data["previous_command_rad"][1:], data["joint_positions_rad"][1:])

    def test_servo_offset_is_compensated_in_labels(self):
        solve, forward = _toy_kinematics()
        home = np.asarray([0.38, 0.0, 0.30, 0.0, 0.0, 0.0, 0.0])
        phases = with_approach_from(_phases(), forward(home)[0], 0.0015)
        recorder = DemonstrationRecorder("toy", 5, 0.005)
        noise = {"rng": np.random.default_rng(1), "sigma_rad": 0.0, "correlation_s": 0.5,
                 "offset_sigma_rad": 0.002, "offset_correlation_s": 2.0}
        stats = label_phases(
            phases, home_q=home, solve_ik=solve, forward=forward, recorder=recorder,
            dt_s=1 / 30, max_joint_delta_rad=0.07, noise=noise,
        )
        measured = np.stack([r["joint_positions_rad"] for r in recorder.rows])
        previous = np.stack([r["previous_command_rad"] for r in recorder.rows])
        # The arm is held away from its command, as gravity does in Isaac ...
        self.assertGreater(np.abs(measured - previous).max(), 5e-4)
        # ... and commanding IK minus that offset still lands on the path.
        self.assertLess(stats["pen_down_tip_error_max_m"], 1e-3)

    def test_hairpin_does_not_stall_progress(self):
        solve, forward = _toy_kinematics()
        hairpin = [{"stroke_id": 0, "points": [[0.40, 0.0], [0.41, 0.0], [0.4005, 0.0003]]}]
        phases = build_motion_sequence(
            hairpin, pen_down_z=0.201, pen_up_z=0.235, approach_height=0.27, max_cartesian_step=0.0015
        )
        home = np.asarray([0.38, 0.0, 0.30, 0.0, 0.0, 0.0, 0.0])
        phases = with_approach_from(phases, forward(home)[0], 0.0015)
        noise = {"rng": np.random.default_rng(2), "sigma_rad": 0.0005, "correlation_s": 0.5}
        recorder = DemonstrationRecorder("toy", 5, 0.005)
        label_phases(
            phases, home_q=home, solve_ik=solve, forward=forward, recorder=recorder,
            dt_s=1 / 30, max_joint_delta_rad=0.07, noise=noise,
        )
        self.assertIn("LIFT_PEN_STROKE_0", {row["state"] for row in recorder.rows})

    def test_frozen_arm_raises_instead_of_looping(self):
        _, forward = _toy_kinematics()
        home = np.asarray([0.38, 0.0, 0.30, 0.0, 0.0, 0.0, 0.0])
        phases = with_approach_from(_phases(), forward(home)[0], 0.0015)
        with self.assertRaises(LabelError):
            label_phases(
                phases, home_q=home, solve_ik=lambda target, seed: np.asarray(seed), forward=forward,
                recorder=DemonstrationRecorder("toy", 5, 0.005), dt_s=1 / 30, max_joint_delta_rad=0.07,
            )

    def test_repeated_ik_failure_drops_episode(self):
        _, forward = _toy_kinematics()
        home = np.asarray([0.40, 0.0, 0.27, 0.0, 0.0, 0.0, 0.0])
        with self.assertRaises(LabelError):
            label_phases(
                _phases(), home_q=home, solve_ik=lambda target, seed: None, forward=forward,
                recorder=DemonstrationRecorder("toy", 5, 0.005), dt_s=1 / 30, max_joint_delta_rad=0.07,
            )

    def test_sustained_joint_jump_drops_episode(self):
        solve, forward = _toy_kinematics()
        home = np.asarray([0.40, 0.0, 0.27, 0.0, 0.0, 0.0, 0.0])

        def flipping(target, seed):
            q = solve(target, seed)
            q[6] = 3.0  # a far IK branch the slew limit can only creep toward
            return q

        with self.assertRaises(LabelError):
            label_phases(
                _phases(), home_q=home, solve_ik=flipping, forward=forward,
                recorder=DemonstrationRecorder("toy", 5, 0.005), dt_s=1 / 30, max_joint_delta_rad=0.07,
            )


if __name__ == "__main__":
    unittest.main()
