from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ik_controller import SafeLulaIKController, gravity_offset_rad


class GravityOffsetTest(unittest.TestCase):
    def test_offset_is_minus_torque_over_stiffness(self):
        offset = gravity_offset_rad([10.0, -20.0, 0.0], 1000.0, max_offset_rad=1.0)
        np.testing.assert_allclose(offset, [-0.01, 0.02, 0.0])

    def test_per_joint_stiffness_and_bound(self):
        offset = gravity_offset_rad([10.0, 10.0], [1000.0, 100.0], max_offset_rad=0.05)
        np.testing.assert_allclose(offset, [-0.01, -0.05])

    def test_invalid_input_gives_zero_offset(self):
        np.testing.assert_array_equal(gravity_offset_rad([np.nan, 1.0], 100.0), [0.0, 0.0])
        np.testing.assert_array_equal(gravity_offset_rad([1.0, 1.0], 0.0), [0.0, 0.0])

    def test_apply_adds_offset_to_drive_but_not_to_state(self):
        sent = []
        ctrl = SafeLulaIKController.__new__(SafeLulaIKController)
        ctrl.articulation = SimpleNamespace(get_joint_positions=lambda: np.zeros(2))
        ctrl.articulation_controller = SimpleNamespace(apply_action=sent.append)
        ctrl.joint_names = ["a", "b"]
        ctrl.lower = np.full(2, -1.0)
        ctrl.upper = np.full(2, 1.0)
        ctrl.max_joint_delta_rad = None
        ctrl._last_commanded_positions = np.zeros(2)
        ctrl.last_commanded_delta_rad = np.zeros(2)
        ctrl.last_command_was_clamped = False
        ctrl.last_gravity_offset_rad = np.zeros(2)
        ctrl.gravity_offset_fn = lambda: np.array([0.01, -0.02])
        action = SimpleNamespace(joint_positions=np.array([0.5, 0.5]), joint_indices=None)
        ctrl.apply(action)
        np.testing.assert_allclose(sent[0].joint_positions, [0.51, 0.48])
        np.testing.assert_allclose(ctrl._last_commanded_positions, [0.5, 0.5])
        np.testing.assert_allclose(action.joint_positions, [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
