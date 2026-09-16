from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR = PROJECT_ROOT / ".vendor"
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import torch

    from behavior_cloning import CartesianDeltaPolicy, load_checkpoint, save_checkpoint, split_by_drawing_id
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this Python environment")
class BehaviorCloningTests(unittest.TestCase):
    def test_tensor_dimensions_and_action_bound(self):
        # 7 q + 7 qd + 3 tip + 3 error + (5*3) lookahead + 1 pen state.
        model = CartesianDeltaPolicy(36, [256, 256, 128], 0.005)
        output = model(torch.randn(11, 36))
        self.assertEqual(tuple(output.shape), (11, 3))
        self.assertLessEqual(float(output.abs().max()), 0.005 + 1e-7)

    def test_checkpoint_save_load(self):
        model = CartesianDeltaPolicy(36, [256, 256, 128], 0.005)
        sample = torch.randn(3, 36)
        expected = model(sample).detach().numpy()
        with tempfile.TemporaryDirectory() as directory:
            path = save_checkpoint(Path(directory) / "policy.pt", model, epoch=4)
            loaded, metadata = load_checkpoint(path)
            actual = loaded(sample).detach().numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
        self.assertEqual(metadata["epoch"], 4)

    def test_split_is_by_complete_drawing_id(self):
        ids = np.asarray(["a", "a", "b", "b", "c", "c"])
        train, validation = split_by_drawing_id(ids, 0.34, 10)
        self.assertFalse(set(ids[train]).intersection(set(ids[validation])))
        self.assertEqual(set(train).union(set(validation)), set(range(len(ids))))


if __name__ == "__main__":
    unittest.main()
