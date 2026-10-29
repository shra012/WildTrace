"""SmolVLA observations derived from the ACT ones (src/vla_features.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import vla_features  # noqa: E402
from act_features import PATH_WINDOW_POINTS  # noqa: E402


def _env(window_xyz: np.ndarray, error=(0.0, 0.0, 0.0), pen_down=1.0) -> np.ndarray:
    return np.r_[[0.4, 0.1, 0.05], error, np.asarray(window_xyz, dtype=float).reshape(-1), pen_down]


def test_state_keeps_joints_tip_error_and_pen():
    act_state = np.arange(21, dtype=np.float32)
    env = _env(np.zeros((PATH_WINDOW_POINTS, 3)), error=(0.001, -0.002, 0.0), pen_down=1.0)
    state = vla_features.vla_state(act_state, env)
    assert state.shape == (1, vla_features.STATE_DIM) == (1, len(vla_features.STATE_NAMES))
    assert np.allclose(state[0, :21], act_state)
    assert np.allclose(state[0, 21:27], env[:6]) and state[0, 27] == 1.0


def test_path_view_draws_the_path_ahead_of_the_tip():
    # Straight path along +x from the tip, level with it.
    window = np.c_[np.arange(PATH_WINDOW_POINTS) * 0.002, np.zeros(PATH_WINDOW_POINTS), np.zeros(PATH_WINDOW_POINTS)]
    image = vla_features.path_view(_env(window))
    size = vla_features.IMAGE_SIZE_PX
    assert image.shape == (size, size, 3) and image.dtype == np.uint8
    red = image[..., 0]
    center = size // 2
    assert red[center, center + 40] > 100  # on the path, right of the tip
    assert red[center, center - 40] == 0  # nothing behind the tip
    assert red[center - 40, center + 40] == 0  # nor off to the side
    # Level path: blue is mid-grey where the path is.
    assert 100 < image[center, center + 40, 2] < 160
    # The nearest dot is brighter than the farthest.
    scale = (size / 2) / vla_features.VIEW_HALF_EXTENT_M
    near = image[center, int(center + 0.002 * scale), 1]
    far = image[center, int(center + 0.018 * scale), 1]
    assert near > far > 0


def test_path_view_shows_a_lift_in_blue():
    window = np.c_[np.zeros(PATH_WINDOW_POINTS), np.arange(PATH_WINDOW_POINTS) * 0.002,
                   np.arange(PATH_WINDOW_POINTS) * 0.004]
    image = vla_features.path_view(_env(window))
    center = vla_features.IMAGE_SIZE_PX // 2
    column = image[:center, center, 2]  # +y is up in the image
    assert column.max() > 200


def test_image_layout_is_channels_first_unit_range():
    image = np.full((4, 5, 3), 255, dtype=np.uint8)
    tensor = vla_features.image_tensor_layout(image)
    assert tensor.shape == (3, 4, 5) and tensor.dtype == np.float32 and tensor.max() == 1.0


def test_full_state_appends_the_path_window_after_the_compact_state():
    act_state = np.arange(21, dtype=np.float32)
    window = np.arange(PATH_WINDOW_POINTS * 3, dtype=float).reshape(PATH_WINDOW_POINTS, 3) * 1e-3
    env = _env(window, error=(0.001, -0.002, 0.0))
    compact = vla_features.vla_state(act_state, env)
    full = vla_features.vla_state(act_state, env, full=True)
    assert full.shape == (1, vla_features.FULL_STATE_DIM) == (1, len(vla_features.FULL_STATE_NAMES))
    assert np.allclose(full[:, : vla_features.STATE_DIM], compact)
    assert np.allclose(full[0, vla_features.STATE_DIM :], window.reshape(-1))


def test_widening_the_state_input_keeps_the_old_outputs():
    import types

    import pytest

    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_smolvla import widen_state_input

    old = torch.nn.Linear(32, 8)
    config = types.SimpleNamespace(max_state_dim=32)
    policy = types.SimpleNamespace(config=config, model=types.SimpleNamespace(state_proj=old, config=config))
    compact = torch.cat([torch.randn(4, 28), torch.zeros(4, 4)], dim=1)  # 28 values, zero padding
    before = old(compact)
    widen_state_input(policy, 64, keep=28)
    wide = policy.model.state_proj
    assert wide.in_features == 64 and config.max_state_dim == 64
    full = torch.cat([compact[:, :28], torch.randn(4, 36)], dim=1)  # window values land on zero weights
    assert torch.allclose(wide(full), before, atol=1e-6)


def _stroke_env(tip_xy, target_xy, pen_down=1.0, tip_z=0.0496, target_z=0.0496) -> np.ndarray:
    error = (target_xy[0] - tip_xy[0], target_xy[1] - tip_xy[1], target_z - tip_z)
    env = np.r_[[tip_xy[0], tip_xy[1], tip_z], error, np.zeros(3 * PATH_WINDOW_POINTS), pen_down]
    env[6 : 6 + 3 * PATH_WINDOW_POINTS] = np.tile(error, PATH_WINDOW_POINTS)
    return env


def test_history_inks_only_with_the_pen_down_on_the_paper():
    rows = vla_features.history_rows(np.stack([
        _stroke_env((0.47, 0.08), (0.47, 0.08)),                               # drawing
        _stroke_env((0.47, 0.08), (0.47, 0.08), tip_z=0.0536),                  # pen down, 4 mm above
        _stroke_env((0.47, 0.08), (0.48, 0.08), pen_down=0.0, tip_z=0.0836),   # travel
    ]))
    assert rows.shape == (3, vla_features.HISTORY_COLUMNS)
    assert rows[:, 4].tolist() == [1.0, 0.0, 0.0]  # ink
    assert rows[:, 5].tolist() == [1.0, 1.0, 0.0]  # target is part of the drawing
    assert np.allclose(rows[2, 2:4], (0.48, 0.08))


def test_camera_views_show_ink_where_drawn_and_target_where_missed():
    cx, cy = vla_features.OVERHEAD_CENTER_XY_M
    xs = np.linspace(cx - 0.04, cx + 0.04, 200)
    envs = [_stroke_env((x, cy), (x, cy)) for x in xs]
    rows = vla_features.history_rows(np.stack(envs))
    rows[100:, 4] = 0.0  # second half of the line never inked: a gap
    views = vla_features.camera_views(rows, envs[-1])
    overhead = (views[vla_features.OVERHEAD_KEY].transpose(1, 2, 0) * 255).astype(int)
    size = vla_features.CAMERA_IMAGE_SIZE_PX
    scale = (size / 2) / vla_features.OVERHEAD_HALF_EXTENT_M
    row = size // 2
    inked = overhead[row, int(size / 2 - 0.02 * scale)]
    missed = overhead[row, int(size / 2 + 0.02 * scale)]
    blank = overhead[int(size / 2 - 0.03 * scale), size // 2]
    assert inked.max() < 80                                   # black ink
    assert missed[2] > 200 and missed[0] < 160                # blue target, no ink
    assert blank.min() == 255                                 # white paper elsewhere
    crop = views[vla_features.TIP_CROP_KEY]
    assert crop.shape == (3, size, size) and crop.dtype == np.float32


def test_synthetic_overhead_matches_the_twin_camera_config():
    import yaml

    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "xarm7_drawing_real.yaml").read_text())
    camera = config["vla_camera"]
    assert np.allclose(camera["center_xy_m"], vla_features.OVERHEAD_CENTER_XY_M)
    assert np.isclose(camera["view_size_m"], 2 * vla_features.OVERHEAD_HALF_EXTENT_M)
    assert camera["image_size_px"] == vla_features.CAMERA_IMAGE_SIZE_PX
