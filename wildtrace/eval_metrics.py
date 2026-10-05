"""Benchmark metrics for generated line diagrams (issue #14, run r2).

Pure functions over binary images plus a metric registry (`METRICS`) that the
benchmark report and the eval-suite document are both generated from, so the
written definitions cannot drift from the code. Spec:
docs/benchmarks/eval_suite_r2.docx.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np
from PIL import Image

from wildtrace.gold_stage import _build_trajectory_payload
from wildtrace.viewpoint import build_viewpoint_backend, extract_view_features

# Paper the xArm7 draws on (isaac-sim/config/xarm7_drawing.yaml surface_size_xy_m).
SURFACE_MM = (160.0, 120.0)
DRAW_BUDGET_S = 600.0
BLANK_INK_RATIO = 0.002


@dataclass(frozen=True, slots=True)
class Metric:
    key: str
    group: str
    name: str
    direction: str  # "higher", "lower", "target" or "info"
    formula: str
    meaning: str
    level: str = "attempt"  # "attempt", "subject" or "setup"


GROUPS = {
    "A": "Pipeline outcome",
    "B": "Drawing structure",
    "C": "Shape fidelity to the source",
    "D": "Pose / view consistency",
    "E": "Semantic recognizability",
    "F": "Judges",
    "G": "Robot drawability",
    "H": "Cost and resources",
}

METRICS: tuple[Metric, ...] = (
    # A. Pipeline outcome
    Metric("accepted", "A", "Acceptance rate", "higher",
           "share of subjects whose final attempt passed OpenCV prescreen AND the setup's own judge",
           "How often the production loop ends with a usable drawing within max_attempts.", "subject"),
    Metric("first_attempt_pass", "A", "First-attempt pass rate", "higher",
           "share of subjects accepted on attempt 1",
           "Quality without retries; retries cost time and hide a weak generator.", "subject"),
    Metric("attempts", "A", "Attempts used", "lower",
           "number of generate calls for the subject (1..max_attempts)",
           "Retry pressure. Equal to max_attempts for every rejected subject.", "subject"),
    Metric("combined_score", "A", "Combined score", "higher",
           "0.5 x OpenCV score + 0.5 x own-judge score (production diagram_validation_score)",
           "The number production ranks drawings by. Not comparable across judges on its own.", "subject"),
    Metric("error", "A", "Error rate", "lower",
           "share of subjects where generation or judging raised an exception",
           "Robustness: crashes, API failures, out-of-memory.", "subject"),
    # B. Drawing structure
    Metric("opencv_passed", "B", "OpenCV prescreen pass", "higher",
           "no flag from validate_with_opencv (foreground, components, small contours, truncation, lower fragments)",
           "Cheap structural sanity check that gates the judge in production."),
    Metric("opencv_score", "B", "OpenCV score", "higher",
           "1 - 0.8|fg - 0.16| - 0.03 max(components - 3, 0) - 0.6 small_ratio - 0.5 truncation - 0.04 excess fragments",
           "Continuous version of the prescreen; 1.0 is a clean, complete, uncluttered outline."),
    Metric("foreground_ratio", "B", "Ink ratio", "target",
           "dark pixels / all pixels (target 0.16, allowed 0.01-0.32)",
           "Too low = faint or empty drawing; too high = filled or cluttered."),
    Metric("component_count", "B", "Connected components", "lower",
           "8-connected ink blobs",
           "Fragmentation. A robot-friendly outline has few pieces."),
    Metric("small_contour_ratio", "B", "Small-contour ratio", "lower",
           "contours with area < 12 px / all contours",
           "Speckle and noise left by binarization."),
    Metric("body_bottom_reach", "B", "Bottom reach", "higher",
           "lowest ink row / image height",
           "Below ~0.6 means the body (legs, tail) is cut off."),
    Metric("fill_ratio", "B", "Solid-fill ratio", "lower",
           "ink surviving a 7x7 morphological opening / all ink",
           "Line drawings have ~0; solid silhouettes or black patches approach 1. The judge prompt forbids fill."),
    Metric("stroke_width_px", "B", "Mean stroke width (px)", "info",
           "2 x ink area / ink perimeter",
           "Line weight. Very thick strokes read as fill; very thin ones break up when binarized."),
    Metric("blank", "B", "Blank output", "lower",
           f"ink ratio < {BLANK_INK_RATIO}",
           "The generator returned (almost) nothing."),
    # C. Shape fidelity
    Metric("silhouette_iou", "C", "Silhouette IoU", "higher",
           "|filled drawing AND source mask| / |filled drawing OR source mask|",
           "Does the drawing occupy the same shape and place as the animal in the photo? 1 = identical silhouette."),
    Metric("chamfer_pct", "C", "Chamfer distance (% of diagonal)", "lower",
           "mean of the two directed mean nearest-boundary distances, / image diagonal x 100",
           "Average outline misplacement; robust to small gaps where IoU is not."),
    Metric("boundary_f", "C", "Boundary F-score @2%", "higher",
           "F1 of drawing-boundary vs mask-boundary pixels matched within 2% of the diagonal",
           "Precision = drawn outline lies on the real outline; recall = real outline was drawn."),
    Metric("aspect_error", "C", "Aspect-ratio error", "lower",
           "|w/h of drawing bbox - w/h of mask bbox| / (w/h of mask bbox)",
           "Stretched or squashed body proportions."),
    Metric("centroid_offset_pct", "C", "Centroid offset (% of diagonal)", "lower",
           "distance between drawing and mask centroids / diagonal x 100",
           "Whole drawing shifted relative to the subject."),
    # D. Pose / view
    Metric("angle_bucket_match", "D", "Angle-bucket match", "higher",
           "viewpoint classifier bucket on the drawing's filled silhouette == source angle_bucket",
           "Keeps the pose label the gold dataset ships with truthful (left profile stays left profile)."),
    Metric("direction_match", "D", "Facing-direction match", "higher",
           "sign(direction_score) equal for drawing and source (|score| < 0.05 counts as neutral)",
           "Did the animal keep facing the same way?"),
    Metric("direction_delta", "D", "|Δ direction score|", "lower",
           "|direction_score(drawing) - direction_score(source)|",
           "Continuous facing difference; less brittle than the bucket."),
    Metric("symmetry_delta", "D", "|Δ symmetry|", "lower",
           "|symmetry_score(drawing) - symmetry_score(source)|",
           "Front-on vs profile change."),
    # E. Semantics
    Metric("clip_top1", "E", "BioCLIP category top-1", "higher",
           "argmax over 7 categories of BioCLIP(\"a line drawing of a {c}\") == true category",
           "An independent model can tell which animal was drawn (Butterfly kept as a distractor)."),
    Metric("clip_p_true", "E", "BioCLIP p(true category)", "higher",
           "softmax(100 x cosine) probability of the true category",
           "Confidence version of top-1."),
    Metric("clip_image_cosine", "E", "Drawing-photo cosine", "higher",
           "cosine of BioCLIP image embeddings of drawing and isolated subject photo",
           "Overall resemblance to this particular animal, not just the category."),
    # F. Judges
    Metric("own_judge_pass", "F", "Own-judge pass rate", "higher",
           "share of judge calls with passed=true and score >= min_score",
           "What the setup's own validator thinks. Different setups use different judges.", "setup"),
    Metric("parse_failed", "F", "Judge parse-failure rate", "lower",
           "share of judge replies with no parseable JSON",
           "Judge reliability; a parse failure counts as a rejection.", "setup"),
    Metric("judge_latency_s", "F", "Judge latency p50 / p95 (s)", "lower",
           "wall time per validate() call", "Judge speed, including Ollama cold loads.", "setup"),
    Metric("referee_pass", "F", "Cross-referee pass rate", "higher",
           "share of final drawings passed by a fixed referee (each judge and a strong 235B referee)",
           "Apples-to-apples quality: every setup is scored by the same judges.", "setup"),
    Metric("referee_kappa", "F", "Judge agreement (Cohen's κ)", "higher",
           "Cohen's kappa between two judges' pass/fail on the same drawings",
           "Is the judge consistent with others? κ < 0.2 means the verdicts are close to chance.", "setup"),
    # G. Robot drawability
    Metric("stroke_count", "G", "Trajectory strokes", "target",
           "outer contours kept by sample_outline_strokes (max 6)",
           "Pen lifts. 1-6 is exportable; 0 means nothing to draw."),
    Metric("trajectory_ok", "G", "Trajectory exportable", "higher",
           "1 <= stroke_count <= export.max_strokes",
           "The drawing produces a valid gold trajectory."),
    Metric("coverage", "G", "Ink coverage by trajectory", "higher",
           "ink pixels within 3 px of the trajectory polylines / all ink pixels",
           "How much of the drawing the robot will actually draw (interior detail beyond 6 contours is dropped)."),
    Metric("path_mm", "G", "Pen-down path (mm)", "lower",
           "sum of segment lengths after the simulator's uniform fit to 160x120 mm paper",
           "Ink the robot must lay down; drives drawing time."),
    Metric("pen_up_mm", "G", "Pen-up travel (mm)", "lower",
           "sum of distances from each stroke end to the next stroke start, on paper",
           "Wasted travel between strokes."),
    Metric("draw_time_s", "G", "Estimated draw time (s)", "lower",
           "a x path_mm + b x strokes + c, least-squares fit on past Isaac Sim runs",
           "Predicted simulator time to draw; see calibration R²."),
    Metric("within_budget", "G", "Within 10-min budget", "higher",
           f"draw_time_s <= {int(DRAW_BUDGET_S)}",
           "The real-robot run budget the tracking gates were tuned for."),
    # H. Cost
    Metric("gen_latency_s", "H", "Generation time per attempt (s)", "lower",
           "wall time of rectifier.run(); first call includes model load",
           "Throughput of the drawing model.", "attempt"),
    Metric("subject_latency_s", "H", "Time per subject (s)", "lower",
           "wall time of the whole loop for one subject (all attempts, judges, feedback)",
           "End-to-end cost per dataset row.", "subject"),
    Metric("peak_vram_gb", "H", "Peak VRAM (GB)", "lower",
           "torch.cuda.max_memory_allocated in the generation process",
           "GPU headroom; Ollama VRAM is measured separately by the system watcher.", "setup"),
    Metric("sys_ram_gb", "H", "Peak system RAM (GB)", "lower",
           "max over 1 s samples of MemTotal - MemAvailable, all processes",
           "OOM risk on this 24.6 GB machine (plus 6 GB swap).", "setup"),
    Metric("cost_usd", "H", "API cost (USD)", "lower",
           "OpenRouter usage.cost summed over judge calls",
           "Money spent; local models are $0.", "setup"),
)


# ── image helpers ──────────────────────────────────────────────────────────

def ink_mask(image: Image.Image, threshold: int = 128) -> np.ndarray:
    """Dark pixels of a black-on-white drawing as a 0/1 uint8 array."""
    return (np.asarray(image.convert("L"), dtype=np.uint8) < threshold).astype(np.uint8)


def filled_silhouette(ink: np.ndarray, close_kernel: int = 5) -> np.ndarray:
    """Fill the outer contours of an outline drawing to get a solid mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    closed = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(closed)
    if contours:
        cv2.drawContours(filled, contours, -1, 1, thickness=-1)
    return filled


def _boundary(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    return (mask.astype(np.uint8) - cv2.erode(mask.astype(np.uint8), kernel)).astype(np.uint8)


def _distance_to(mask: np.ndarray) -> np.ndarray:
    """Per-pixel distance to the nearest nonzero pixel of `mask`."""
    if not mask.any():
        return np.full(mask.shape, float(np.hypot(*mask.shape)), dtype=np.float32)
    return cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


# ── B. structure ───────────────────────────────────────────────────────────

def structure_metrics(ink: np.ndarray) -> dict[str, Any]:
    area = int(ink.sum())
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    solid = int(cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel).sum())
    contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    return {
        "fill_ratio": round(solid / area, 4) if area else 0.0,
        "stroke_width_px": round(2.0 * area / perimeter, 2) if perimeter else 0.0,
        "blank": area / float(ink.size or 1) < BLANK_INK_RATIO,
    }


# ── C. fidelity ────────────────────────────────────────────────────────────

def fidelity_metrics(drawing_fill: np.ndarray, source_mask: np.ndarray, tolerance: float = 0.02) -> dict[str, Any]:
    if drawing_fill.shape != source_mask.shape:
        raise ValueError("drawing and mask must share a canvas")
    diag = float(np.hypot(*source_mask.shape))
    union = np.logical_or(drawing_fill, source_mask).sum()
    iou = float(np.logical_and(drawing_fill, source_mask).sum() / union) if union else 0.0

    b_draw, b_mask = _boundary(drawing_fill), _boundary(source_mask)
    if not b_draw.any() or not b_mask.any():
        return {"silhouette_iou": round(iou, 4), "chamfer_pct": 100.0, "boundary_f": 0.0,
                "aspect_error": None, "centroid_offset_pct": None}
    d_to_mask = _distance_to(b_mask)[b_draw > 0]
    d_to_draw = _distance_to(b_draw)[b_mask > 0]
    chamfer = 0.5 * (float(d_to_mask.mean()) + float(d_to_draw.mean()))
    tol = tolerance * diag
    precision = float((d_to_mask <= tol).mean())
    recall = float((d_to_draw <= tol).mean())
    f_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    (dx0, dy0, dx1, dy1), (mx0, my0, mx1, my1) = _bbox(drawing_fill), _bbox(source_mask)
    ar_draw = (dx1 - dx0 + 1) / (dy1 - dy0 + 1)
    ar_mask = (mx1 - mx0 + 1) / (my1 - my0 + 1)
    cd = np.argwhere(drawing_fill).mean(axis=0)
    cm = np.argwhere(source_mask).mean(axis=0)
    return {
        "silhouette_iou": round(iou, 4),
        "chamfer_pct": round(100.0 * chamfer / diag, 3),
        "boundary_f": round(f_score, 4),
        "aspect_error": round(abs(ar_draw - ar_mask) / ar_mask, 4),
        "centroid_offset_pct": round(100.0 * float(np.hypot(*(cd - cm))) / diag, 3),
    }


# ── D. pose ────────────────────────────────────────────────────────────────

def _view_features(mask: np.ndarray, max_side: int = 256) -> dict[str, Any]:
    # extract_view_features uses a pure-Python component search; all its
    # features are ratios, so a downscaled mask gives the same answer faster.
    image = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    image.thumbnail((max_side, max_side), Image.NEAREST)
    return extract_view_features(image)


def pose_metrics(drawing_fill: np.ndarray, source: dict[str, Any], viewpoint_config: dict[str, Any]) -> dict[str, Any]:
    features = _view_features(drawing_fill)
    result = build_viewpoint_backend(viewpoint_config).classify(
        {"mask_available": True, "view_features": features}, viewpoint_config
    )
    source_features = source.get("view_features") or {}
    src_dir = float(source_features.get("direction_score", 0.0))
    drw_dir = float(features.get("direction_score", 0.0))

    def side(value: float) -> int:
        return 0 if abs(value) < 0.05 else (1 if value > 0 else -1)

    return {
        "drawing_angle_bucket": result.bucket,
        "angle_bucket_match": result.bucket == source.get("angle_bucket"),
        "direction_match": side(drw_dir) == side(src_dir),
        "direction_delta": round(abs(drw_dir - src_dir), 4),
        "symmetry_delta": round(abs(float(features.get("symmetry_score", 0.0))
                                    - float(source_features.get("symmetry_score", 0.0))), 4),
    }


# ── G. robot drawability ───────────────────────────────────────────────────

def paper_strokes(strokes: Sequence[Sequence[tuple[float, float]]], surface_mm: tuple[float, float] = SURFACE_MM) -> list[np.ndarray]:
    """Normalized strokes mapped to paper mm exactly like the simulator's
    map_trajectory_to_plane: one uniform scale fitting the bbox to the paper."""
    arrays = [np.asarray(s, dtype=np.float64) for s in strokes if len(s) >= 2]
    if not arrays:
        return []
    points = np.vstack(arrays)
    span = points.max(axis=0) - points.min(axis=0)
    active = span > 1e-12
    if not active.any():
        return []
    scale = float((np.asarray(surface_mm)[active] / span[active]).min())
    return [a * scale for a in arrays]


def path_lengths_mm(strokes: Sequence[Sequence[tuple[float, float]]]) -> tuple[float, float]:
    mapped = paper_strokes(strokes)
    pen_down = sum(float(np.linalg.norm(np.diff(s, axis=0), axis=1).sum()) for s in mapped)
    pen_up = sum(float(np.linalg.norm(b[0] - a[-1])) for a, b in zip(mapped, mapped[1:]))
    return pen_down, pen_up


def trajectory_coverage(ink: np.ndarray, strokes: Sequence[Sequence[tuple[float, float]]], radius_px: int = 3) -> float:
    if not ink.any():
        return 0.0
    height, width = ink.shape
    drawn = np.zeros_like(ink)
    for stroke in strokes:
        pts = np.round(np.asarray(stroke) * [max(width - 1, 1), max(height - 1, 1)]).astype(np.int32)
        if len(pts) >= 2:
            cv2.polylines(drawn, [pts.reshape(-1, 1, 2)], True, 1, thickness=1)
    near = cv2.dilate(drawn, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1,) * 2))
    return round(float((near[ink > 0] > 0).mean()), 4)


@dataclass(frozen=True, slots=True)
class DrawTimeModel:
    a_s_per_mm: float
    b_s_per_stroke: float
    c_s: float
    r2: float
    n: int

    def predict(self, path_mm: float, strokes: int) -> float:
        return self.a_s_per_mm * path_mm + self.b_s_per_stroke * strokes + self.c_s


def fit_draw_time(samples: Sequence[tuple[float, int, float]]) -> DrawTimeModel:
    """Least squares duration ~ a*path_mm + b*strokes + c over (path_mm, strokes, duration_s)."""
    x = np.array([[p, s, 1.0] for p, s, _ in samples], dtype=np.float64)
    y = np.array([d for *_, d in samples], dtype=np.float64)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ coef
    total = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((residual ** 2).sum()) / total if total else 0.0
    return DrawTimeModel(float(coef[0]), float(coef[1]), float(coef[2]), round(r2, 3), len(samples))


def calibrate_draw_time(isaac_root: Path) -> DrawTimeModel | None:
    """Fit on simulator runs made with the current drawing config (those
    whose run_metrics record config_path), joined to their input trajectory."""
    import json

    samples = []
    for metrics_path in sorted((isaac_root / "outputs" / "mcp_sessions").glob("*run_metrics.json")):
        run = json.loads(metrics_path.read_text())
        if "config_path" not in run or run.get("status") != "finished":
            continue
        matches = list((isaac_root / "inputs" / "trajectories").glob(f"*/{run['drawing_id']}.json"))
        if not matches:
            continue
        trajectory = json.loads(matches[0].read_text())
        strokes = [[(p["x"], p["y"]) for p in s["points"]] for s in trajectory["strokes"]]
        samples.append((path_lengths_mm(strokes)[0], len(strokes), float(run["simulation_duration_s"])))
    return fit_draw_time(samples) if len(samples) >= 4 else None


def trajectory_metrics(image: Image.Image, ink: np.ndarray, export: dict[str, Any], model: DrawTimeModel | None) -> dict[str, Any]:
    built = _build_trajectory_payload(image, export)
    strokes = built["strokes"]
    pen_down, pen_up = path_lengths_mm(strokes)
    draw_time = round(model.predict(pen_down, len(strokes)), 1) if model and strokes else None
    return {
        "stroke_count": len(strokes),
        "point_count": built["payload"]["point_count"],
        "trajectory_ok": 1 <= len(strokes) <= int(export["max_strokes"]),
        "coverage": trajectory_coverage(ink, strokes),
        "path_mm": round(pen_down, 1),
        "pen_up_mm": round(pen_up, 1),
        "draw_time_s": draw_time,
        "within_budget": draw_time is not None and draw_time <= DRAW_BUDGET_S,
    }


# ── E. semantics ───────────────────────────────────────────────────────────

class BioClipScorer:
    """Zero-shot category probabilities and drawing-photo similarity, reusing
    the enrichment stage's cached BioCLIP runtime."""

    def __init__(self, enrichment_settings: dict[str, Any], categories: Sequence[str]) -> None:
        from wildtrace.enrichment import BioCLIPHuggingFaceBackend

        self.categories = list(categories)
        self._model, self._preprocess, tokenizer, self._torch = BioCLIPHuggingFaceBackend(enrichment_settings)._load_runtime()
        device = next(self._model.parameters()).device
        with self._torch.no_grad():
            tokens = tokenizer([f"a line drawing of a {c.lower()}" for c in self.categories]).to(device)
            text = self._model.encode_text(tokens)
        self._text = text / text.norm(dim=-1, keepdim=True)
        self._device = device

    def _embed(self, image: Image.Image) -> Any:
        batch = self._preprocess(image.convert("RGB")).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            features = self._model.encode_image(batch)
        return features / features.norm(dim=-1, keepdim=True)

    def score(self, drawing: Image.Image, photo: Image.Image | None, category: str) -> dict[str, Any]:
        embedding = self._embed(drawing)
        probs = (100.0 * embedding @ self._text.T).softmax(dim=-1)[0].tolist()
        top = self.categories[int(np.argmax(probs))]
        out = {
            "clip_top1": top == category,
            "clip_pred": top,
            "clip_p_true": round(probs[self.categories.index(category)], 4) if category in self.categories else None,
        }
        if photo is not None:
            out["clip_image_cosine"] = round(float((embedding @ self._embed(photo).T)[0, 0]), 4)
        return out


# ── statistics ─────────────────────────────────────────────────────────────

def bootstrap_ci(values: Sequence[float], stat: Callable[[np.ndarray], float] = np.mean,
                 n: int = 2000, alpha: float = 0.05, seed: int = 14) -> tuple[float | None, float | None]:
    data = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if len(data) < 2:
        return (None, None)
    rng = np.random.default_rng(seed)
    stats = [stat(data[rng.integers(0, len(data), len(data))]) for _ in range(n)]
    return (round(float(np.quantile(stats, alpha / 2)), 4), round(float(np.quantile(stats, 1 - alpha / 2)), 4))


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided sign test; ties are excluded by the caller."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return round(min(1.0, 2 * tail), 4)


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> float | None:
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return round((observed - expected) / (1 - expected), 4) if expected < 1 else 1.0
