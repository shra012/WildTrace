"""SmolVLA observations built from the ACT ones (src/act_features.py).

SmolVLA reads images, a state vector of at most 32 values and a language
instruction; it has no slot for ACT's 37-value environment_state. The drawing
scene has no task camera either, so the upcoming path is shown to it as a
rendered top-down view centred on the pen tip:

- red:   the path from the current target through the 10 window points,
- green: the window points as dots, brighter the nearer they are (direction),
- blue:  the same path, its brightness the path height above the tip
         (128 = level), so lifts and lowers are visible.

The numbers that need sub-millimetre precision (tip, target - tip, pen_down)
stay in the state: [q(7), qd(7), previous command(7), tip(3), target-tip(3),
pen_down(1)] = 28 values. Everything here is a pure function of the ACT
observation, so the same LeRobot datasets train both policies and the policy
server takes the same request from Isaac for either. Pure numpy + OpenCV.
"""
from __future__ import annotations

import cv2
import numpy as np

from act_features import PATH_WINDOW_POINTS

IMAGE_KEY = "observation.images.path_view"
IMAGE_SIZE_PX = 224
# The window reaches (points - 1) * 2 mm = 18 mm ahead, plus the target error.
VIEW_HALF_EXTENT_M = 0.0224
HEIGHT_SCALE_PER_MM = 3.0          # blue channel counts per mm of height above the tip
SUBPIXEL_BITS = 4                  # cv2 fixed-point drawing, 1/16 px
STATE_DIM = 28
TASK = "trace the upcoming pen path"

STATE_NAMES = (
    [f"q{i}" for i in range(1, 8)]
    + [f"qd{i}" for i in range(1, 8)]
    + [f"q{i}_prev_cmd" for i in range(1, 8)]
    + ["tip_x", "tip_y", "tip_z", "err_x", "err_y", "err_z", "pen_down"]
)


def vla_state(act_state: np.ndarray, environment_state: np.ndarray) -> np.ndarray:
    """[q, qd, previous command] from the ACT state + [tip, target - tip, pen_down] from its env state."""
    state = np.asarray(act_state, dtype=np.float32).reshape(-1, 21)
    env = np.asarray(environment_state, dtype=np.float32).reshape(len(state), -1)
    return np.concatenate([state, env[:, :6], env[:, -1:]], axis=1)


def path_view(environment_state: np.ndarray, size: int = IMAGE_SIZE_PX,
              half_extent_m: float = VIEW_HALF_EXTENT_M) -> np.ndarray:
    """Render one frame's upcoming path as a (size, size, 3) uint8 image, tip at the centre, +y up."""
    env = np.asarray(environment_state, dtype=np.float64).reshape(-1)
    window = env[6 : 6 + 3 * PATH_WINDOW_POINTS].reshape(PATH_WINDOW_POINTS, 3)
    path = np.vstack([env[3:6][None, :], window])  # current target first, all relative to the tip
    scale = (size / 2.0) / half_extent_m
    pixels = np.empty((len(path), 2))
    pixels[:, 0] = size / 2.0 + path[:, 0] * scale
    pixels[:, 1] = size / 2.0 - path[:, 1] * scale
    fixed = np.round(pixels * (1 << SUBPIXEL_BITS)).astype(np.int32)
    red, green, blue = (np.zeros((size, size), dtype=np.uint8) for _ in range(3))
    # Each segment is drawn on its own so the blue channel can carry its height.
    for index in range(len(path) - 1):
        a, b = tuple(fixed[index]), tuple(fixed[index + 1])
        height_mm = 1e3 * (path[index, 2] + path[index + 1, 2]) / 2.0
        cv2.line(red, a, b, 255, 1, cv2.LINE_AA, SUBPIXEL_BITS)
        cv2.line(blue, a, b, int(np.clip(128 + HEIGHT_SCALE_PER_MM * height_mm, 1, 255)), 1, cv2.LINE_AA,
                 SUBPIXEL_BITS)
    for index, point in enumerate(fixed[1:]):
        brightness = int(255 - index * (200 / max(1, PATH_WINDOW_POINTS - 1)))
        cv2.circle(green, tuple(point), 2 << SUBPIXEL_BITS, brightness, -1, cv2.LINE_AA, SUBPIXEL_BITS)
    return np.stack([red, green, blue], axis=2)


def image_tensor_layout(image: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 -> (3, H, W) float32 in [0, 1], LeRobot's image layout."""
    return np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / 255.0
