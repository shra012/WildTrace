from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from metrics import nearest_path_errors, pointwise_errors, summarize_errors
from xarm7_loader import EXPECTED_ARM_JOINTS, inspect_urdf


class MetricsAndRobotTests(unittest.TestCase):
    def test_path_error_metrics(self):
        desired = [[0.0, 0.0], [1.0, 0.0]]
        actual = [[0.0, 0.1], [1.0, 0.2]]
        errors = pointwise_errors(desired, actual)
        np.testing.assert_allclose(errors, [0.1, 0.2])
        summary = summarize_errors(errors)
        self.assertAlmostEqual(summary["mean_error_m"], 0.15)
        np.testing.assert_allclose(nearest_path_errors(desired, actual), [0.1, 0.2])

    def test_generated_urdf_has_verified_arm_and_pen_frames(self):
        description = inspect_urdf(PROJECT_ROOT / "assets" / "xarm7" / "xarm7_with_pen.urdf")
        self.assertEqual(description.arm_joint_names, EXPECTED_ARM_JOINTS)
        self.assertEqual(description.flange_link, "link_eef")
        self.assertEqual(description.pen_tip_link, "pen_tip")
        self.assertEqual(description.gripper_joint_names, [])
        self.assertAlmostEqual(description.joint_limits["joint4"].lower_rad, -0.19198)
        self.assertAlmostEqual(description.joint_limits["joint6"].upper_rad, np.pi)


if __name__ == "__main__":
    unittest.main()

