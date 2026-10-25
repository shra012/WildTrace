from __future__ import annotations

import math
import sys
import unittest
import xml.dom.minidom as md
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xacro_builder import _rotation_axis, _rotation_rpy, _rpy_from_rotation, freeze_at  # noqa: E402


def _close(a, b, tol=1e-9):
    return all(abs(x - y) < tol for row_a, row_b in zip(a, b) for x, y in zip(row_a, row_b))


class FreezeAtAngleTests(unittest.TestCase):
    def test_rpy_round_trip(self):
        for rpy in ((0.3, -0.2, 1.1), (0.0, 0.0, 0.0), (-1.2, 0.4, -2.5)):
            self.assertTrue(_close(_rotation_rpy(*_rpy_from_rotation(_rotation_rpy(*rpy))), _rotation_rpy(*rpy)))

    def test_freeze_bakes_axis_rotation_into_origin(self):
        joint = md.parseString(
            '<joint name="j" type="revolute"><origin rpy="0.1 0 0" xyz="0 0.035 0.06"/><axis xyz="-1 0 0"/></joint>'
        ).documentElement
        freeze_at(joint, 0.76)
        origin = joint.getElementsByTagName("origin")[0]
        rpy = [float(v) for v in origin.getAttribute("rpy").split()]
        self.assertAlmostEqual(rpy[0], 0.1 - 0.76, places=9)
        self.assertEqual(origin.getAttribute("xyz"), "0 0.035 0.06")

    def test_general_axis_matches_matrix_product(self):
        joint = md.parseString(
            '<joint name="j" type="revolute"><origin rpy="0.2 -0.3 0.5" xyz="0 0 0"/><axis xyz="0 1 1"/></joint>'
        ).documentElement
        freeze_at(joint, 0.4)
        rpy = [float(v) for v in joint.getElementsByTagName("origin")[0].getAttribute("rpy").split()]
        expected = [[sum(a * b for a, b in zip(row, col)) for col in zip(*_rotation_axis((0, 1, 1), 0.4))]
                    for row in _rotation_rpy(0.2, -0.3, 0.5)]
        self.assertTrue(_close(_rotation_rpy(*rpy), expected))

    def test_zero_angle_leaves_origin_alone(self):
        joint = md.parseString('<joint name="j" type="revolute"><origin rpy="0 0 0" xyz="0 0 0"/><axis xyz="1 0 0"/></joint>').documentElement
        freeze_at(joint, 0.0)
        self.assertEqual(joint.getElementsByTagName("origin")[0].getAttribute("rpy"), "0 0 0")


if __name__ == "__main__":
    unittest.main()
