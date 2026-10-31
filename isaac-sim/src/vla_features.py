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
pen_down(1)] = 28 values. The "full" state appends the 30-value path window
as numbers too (58 values, the whole ACT observation); it needs SmolVLA's
max_state_dim raised to 64. Its first 28 values are the compact state.
Everything here is a pure function of the ACT observation, so the same LeRobot datasets train both policies and the policy
server takes the same request from Isaac for either. Pure numpy + OpenCV.
"""
from __future__ import annotations

import cv2
import numpy as np

from act_features import INK_CONTACT_M, PATH_WINDOW_POINTS

IMAGE_KEY = "observation.images.path_view"
IMAGE_SIZE_PX = 224
# The window reaches (points - 1) * 2 mm = 18 mm ahead, plus the target error.
VIEW_HALF_EXTENT_M = 0.0224
HEIGHT_SCALE_PER_MM = 3.0          # blue channel counts per mm of height above the tip
SUBPIXEL_BITS = 4                  # cv2 fixed-point drawing, 1/16 px
STATE_DIM = 28
FULL_STATE_DIM = 58
TASK = "trace the upcoming pen path"

STATE_NAMES = (
    [f"q{i}" for i in range(1, 8)]
    + [f"qd{i}" for i in range(1, 8)]
    + [f"q{i}_prev_cmd" for i in range(1, 8)]
    + ["tip_x", "tip_y", "tip_z", "err_x", "err_y", "err_z", "pen_down"]
)
FULL_STATE_NAMES = STATE_NAMES + [f"w{i}_{axis}" for i in range(PATH_WINDOW_POINTS) for axis in "xyz"]


# With a mode state (#46), after the full state: going back to a gap, coming forward again (one-hot;
# both zero when drawing normally), so the policy is told a redraw is on instead of guessing it from ink.
MODE_STATE_DIM = 2


def mode_state(teacher_mode) -> np.ndarray:
    """(N,) teacher modes (kinematic_labeller.MODE_*) -> (N, 2) one-hot back / forward."""
    mode = np.asarray(teacher_mode).reshape(-1)
    return np.stack([mode == 1, mode == 2], axis=1).astype(np.float32)


def vla_state(act_state: np.ndarray, environment_state: np.ndarray, full: bool = False,
              mode=None) -> np.ndarray:
    """[q, qd, previous command] from the ACT state + [tip, target - tip, pen_down] from its env state,
    with `full` the path window (relative to the tip) after them, and with `mode` (teacher modes,
    full state only) the mode state last."""
    state = np.asarray(act_state, dtype=np.float32).reshape(-1, 21)
    env = np.asarray(environment_state, dtype=np.float32).reshape(len(state), -1)
    parts = [state, env[:, :6], env[:, -1:]]
    if full:
        parts.append(env[:, 6 : 6 + 3 * PATH_WINDOW_POINTS])
    if mode is not None:
        parts.append(mode_state(np.broadcast_to(np.asarray(mode).reshape(-1), (len(state),))))
    return np.concatenate(parts, axis=1)


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


# Camera view of the paper (#46): what a camera over the sheet would see, rendered from the
# pen tip history instead of the path window alone. Two images:
# - overhead: a fixed top-down camera over the drawing area (config `vla_camera`), the whole sheet;
# - tip crop: the same paper zoomed around the pen tip, with the upcoming path drawn over it.
# Both show the target traced so far (light blue) under the ink drawn so far (black), so a
# missed or offset segment is visible as blue without black. Built only from per-tick tip,
# target and pen-down values, so training (from the episode's earlier frames) and the policy
# server (from the ticks since reset) render the same pictures.
OVERHEAD_KEY = "observation.images.paper_overhead"
TIP_CROP_KEY = "observation.images.paper_tip"
CAMERA_IMAGE_SIZE_PX = 256
OVERHEAD_CENTER_XY_M = (0.4770, 0.0882)  # config/xarm7_drawing_real.yaml drawing.surface_center_xy_m
OVERHEAD_HALF_EXTENT_M = 0.088           # 176 mm square around the 160 x 120 mm drawing area
TIP_CROP_HALF_EXTENT_M = VIEW_HALF_EXTENT_M
INK_WIDTH_M = 0.0005
PAPER_RGB = (255, 255, 255)
TARGET_RGB = (90, 140, 255)
INK_RGB = (0, 0, 0)
UPCOMING_RGB = (255, 0, 0)
TIP_RGB = (255, 140, 0)
TEMPLATE_RGB = (205, 205, 205)  # the whole drawing, faint, as on a printed sheet (tracing mode)
VIEWS_FILE = "wildtrace_views.json"  # checkpoint marker: which camera extras the policy was trained with
HISTORY_COLUMNS = 6  # tip x, tip y, target x, target y, ink (0/1), on paper (0/1)


def history_rows(environment_state: np.ndarray, ink_skip: np.ndarray | None = None) -> np.ndarray:
    """(N, 37) ACT env states -> (N, 6) history rows: tip xy, target xy, ink, target on paper.
    A frame inks when the pen is down, the tip is within INK_CONTACT_M of the target height and
    the frame is not an ink skip (gap-labelled data, #46); its target counts as part of the
    drawing whenever the pen is down."""
    env = np.asarray(environment_state, dtype=np.float32).reshape(-1, 6 + 3 * PATH_WINDOW_POINTS + 1)
    target = env[:, 0:3] + env[:, 3:6]
    pen_down = env[:, -1] > 0.5
    ink = pen_down & (env[:, 2] - target[:, 2] < INK_CONTACT_M)
    if ink_skip is not None:
        ink &= np.asarray(ink_skip).reshape(-1) < 0.5
    return np.c_[env[:, 0:2], target[:, 0:2], ink, pen_down].astype(np.float32)


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index ranges where flags is true."""
    edges = np.flatnonzero(np.diff(np.r_[0, flags.astype(np.int8), 0]))
    return list(zip(edges[::2], edges[1::2]))


def paper_view(history: np.ndarray, center_xy, half_extent_m: float, size: int = CAMERA_IMAGE_SIZE_PX,
               environment_state: np.ndarray | None = None, ink_rgb=INK_RGB,
               ink_width_m: float = INK_WIDTH_M, template: np.ndarray | None = None) -> np.ndarray:
    """Top-down (size, size, 3) uint8 view of the paper centred on center_xy, +y up: target traced
    so far, ink over it, the pen tip (last history row) as a dot, and with `environment_state`
    the upcoming path from the current target through the window."""
    history = np.asarray(history, dtype=np.float64).reshape(-1, HISTORY_COLUMNS)
    scale = (size / 2.0) / half_extent_m
    cx, cy = float(center_xy[0]), float(center_xy[1])

    def pixels(xy: np.ndarray) -> np.ndarray:
        px = np.c_[size / 2.0 + (xy[:, 0] - cx) * scale, size / 2.0 - (xy[:, 1] - cy) * scale]
        return np.round(px * (1 << SUBPIXEL_BITS)).astype(np.int32)

    image = np.empty((size, size, 3), dtype=np.uint8)
    image[:] = PAPER_RGB
    if template is not None:
        # (M, 3) x, y, pen_down of the whole drawing: its pen-down runs, under everything else.
        template = np.asarray(template, dtype=np.float64).reshape(-1, 3)
        lines = [pixels(template[a:b, 0:2]) for a, b in _runs(template[:, 2] > 0.5) if b - a > 1]
        if lines:
            cv2.polylines(image, lines, False, TEMPLATE_RGB, max(1, int(round(INK_WIDTH_M * scale))),
                          cv2.LINE_AA, SUBPIXEL_BITS)
    for column, xy, color, width_m in ((5, history[:, 2:4], TARGET_RGB, INK_WIDTH_M),
                                       (4, history[:, 0:2], tuple(ink_rgb), ink_width_m)):
        lines = [pixels(xy[a:b]) for a, b in _runs(history[:, column] > 0.5) if b - a > 1]
        if lines:
            cv2.polylines(image, lines, False, color, max(1, int(round(width_m * scale))), cv2.LINE_AA,
                          SUBPIXEL_BITS)
    if environment_state is not None:
        env = np.asarray(environment_state, dtype=np.float64).reshape(-1)
        tip = env[0:3]
        window = env[6 : 6 + 3 * PATH_WINDOW_POINTS].reshape(PATH_WINDOW_POINTS, 3) + tip
        path = pixels(np.vstack([(tip + env[3:6])[None, :2], window[:, :2]]))
        cv2.polylines(image, [path], False, UPCOMING_RGB, 1, cv2.LINE_AA, SUBPIXEL_BITS)
        for point in path[1:]:
            cv2.circle(image, tuple(point), 2 << SUBPIXEL_BITS, UPCOMING_RGB, -1, cv2.LINE_AA, SUBPIXEL_BITS)
    if len(history):
        cv2.circle(image, tuple(pixels(history[-1:, 0:2])[0]), 3 << SUBPIXEL_BITS, TIP_RGB, -1, cv2.LINE_AA,
                   SUBPIXEL_BITS)
    return image


def camera_views(history: np.ndarray, environment_state: np.ndarray,
                 rng: np.random.Generator | None = None, template: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Both camera images for one frame, in LeRobot layout. `history` holds this frame's row last.
    With `rng` (training only) the ink style varies and each image is perturbed the way a real
    camera would be (augment_camera_image). With `template` ((M, 3) x, y, pen_down of the whole
    drawing, tracing mode) both views show the drawing faintly under the ink."""
    tip = np.asarray(environment_state, dtype=np.float64).reshape(-1)[0:2]
    style = {"template": template}
    if rng is not None:
        grey = int(rng.integers(0, 100))
        style.update(ink_rgb=(grey, grey, grey), ink_width_m=float(rng.uniform(0.0003, 0.0012)))
    overhead = paper_view(history, OVERHEAD_CENTER_XY_M, OVERHEAD_HALF_EXTENT_M, **style)
    crop = paper_view(history, tip, TIP_CROP_HALF_EXTENT_M, environment_state=environment_state, **style)
    if rng is not None:
        overhead = augment_camera_image(overhead, rng, occlude=True)
        crop = augment_camera_image(crop, rng, occlude=False)
    return {OVERHEAD_KEY: image_tensor_layout(overhead), TIP_CROP_KEY: image_tensor_layout(crop)}


def augment_camera_image(image: np.ndarray, rng: np.random.Generator, occlude: bool = False) -> np.ndarray:
    """Training-time camera realism (#46, step 5): a small misregistration (shift, rotation,
    scale, as from an imperfect camera calibration), lighting and white balance, blur, sensor
    noise and, for the overhead, sometimes a dark shape over one edge where the arm would be."""
    size = image.shape[0]
    angle = float(rng.uniform(-1.5, 1.5))
    scale = float(rng.uniform(0.98, 1.02))
    matrix = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), angle, scale)
    matrix[:, 2] += rng.uniform(-3.0, 3.0, size=2)
    out = cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if occlude and rng.random() < 0.3:
        # The arm: a grey band entering from one edge.
        edge = int(rng.integers(4))
        width = int(rng.integers(size // 10, size // 4))
        depth = int(rng.integers(size // 6, size // 2))
        offset = int(rng.integers(0, size - width))
        grey = int(rng.integers(60, 180))
        y0, y1, x0, x1 = [(0, depth, offset, offset + width), (size - depth, size, offset, offset + width),
                          (offset, offset + width, 0, depth), (offset, offset + width, size - depth, size)][edge]
        out[y0:y1, x0:x1] = grey
    out = out.astype(np.float32)
    out = out * rng.uniform(0.8, 1.15) * rng.uniform(0.93, 1.07, size=3) + rng.uniform(-20, 20)
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.3, 1.0)))
    out += rng.normal(0.0, float(rng.uniform(0.0, 6.0)), size=out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def load_history(dataset_root) -> tuple[np.ndarray, np.ndarray]:
    """History rows for every frame of a LeRobot dataset, indexed by its global `index`, and each
    frame's episode start index. (19.8M frames: about 0.6 GB.) Gap-labelled datasets' ink_skip
    column leaves skipped frames uninked."""
    from pathlib import Path

    import pyarrow.parquet as pq

    files = sorted(Path(dataset_root).glob("data/*/*.parquet"))
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    rows = np.zeros((total, HISTORY_COLUMNS), dtype=np.float32)
    starts = np.zeros(total, dtype=np.int64)
    for path in files:  # one file at a time: the full env column is about 3 GB
        names = pq.ParquetFile(path).schema_arrow.names
        columns = ["observation.environment_state", "frame_index", "index"]
        table = pq.read_table(path, columns=columns + (["ink_skip"] if "ink_skip" in names else []))
        index = table.column("index").to_numpy()
        env = table.column("observation.environment_state").combine_chunks().flatten().to_numpy()
        skip = None
        if "ink_skip" in names:
            column = table.column("ink_skip").combine_chunks()
            skip = (column.flatten() if hasattr(column, "flatten") else column).to_numpy()
        rows[index] = history_rows(env.reshape(len(index), -1), skip)
        starts[index] = index - table.column("frame_index").to_numpy()
    return rows, starts


def template_from_history(history: np.ndarray) -> np.ndarray:
    """(M, 3) template (target x, y, pen_down) from a whole episode's history rows; in training
    the drawing is the path the targets traced, at drawing time it is the planned path."""
    history = np.asarray(history).reshape(-1, HISTORY_COLUMNS)
    return np.c_[history[:, 2:4], history[:, 5]]


def read_views(pretrained_dir) -> dict:
    """The camera extras a checkpoint was trained with ({"template": bool}); {} for older ones."""
    import json
    from pathlib import Path

    path = Path(pretrained_dir) / VIEWS_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
