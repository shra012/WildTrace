from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcp_scene import DEFAULT_CONFIG, scene_config_path, scene_file  # noqa: E402


class SceneConfigTests(unittest.TestCase):
    def test_defaults_to_the_original_sim_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(scene_config_path(Path(tmp)), Path(tmp) / DEFAULT_CONFIG)

    def test_reads_the_config_the_mcp_server_chose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_file(root).parent.mkdir(parents=True)
            scene_file(root).write_text(json.dumps({"config": "config/xarm7_drawing_real.yaml"}))
            self.assertEqual(scene_config_path(root), root / "config" / "xarm7_drawing_real.yaml")

    def test_unreadable_scene_file_falls_back_to_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_file(root).parent.mkdir(parents=True)
            scene_file(root).write_text("{not json")
            self.assertEqual(scene_config_path(root), root / DEFAULT_CONFIG)


if __name__ == "__main__":
    unittest.main()
