"""Reproducible synthetic-data demonstration of edge/diffuse delamination detection.

This example generates an artificial TWLI-like image sequence with *known*
ground-truth geometry (edge fronts, diffuse blobs, matrix cracks, and one
disconnected "distractor" region), runs the normal public DelaDect workflow
(:meth:`~deladect.detection.delamination.core.DelaminationDetector.detect_both_delaminations`)
on it, and compares the detected masks against the exact ground truth with
IoU, Dice, and area-error metrics.

It is a documentation example, not a claim of full experimental validation:
synthetic verification demonstrates implementation correctness and controlled
robustness under known conditions. It does not prove performance on all
experimental TWLI data and does not replace validation using manually
annotated experimental images.

Generation (:func:`generate_synthetic_sequence`) is kept separate from
detection/plotting so it can be imported and exercised on its own, e.g. from
``tests/test_synthetic_delamination.py``, without running the detector or
matplotlib.

Regenerate everything with::

    python examples/synthetic_delamination_validation.py
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage.transform import resize

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results" / "synthetic_validation"
DOCS_STATIC_DIR = REPO_ROOT / "docs" / "source" / "_static" / "delamination"

# ---------------------------------------------------------------------------
# Fixed generation parameters (one seed, one parameter set for the whole run)
# ---------------------------------------------------------------------------

SEED = 20260825
N_FRAMES = 30
HEIGHT, WIDTH = 400, 1300
SHAPE = (HEIGHT, WIDTH)
SCALE_PX_MM = 41.03328366  # reused from example_images/sample-3 for realistic mm^2 scale

# Rendering formula: I_n = g_n * B * (1 - a_e*E_n - a_d*D_n - a_c*C_n - a_x*X_n) + noise
ALPHA_EDGE = 0.35
ALPHA_DIFFUSE = 0.35
ALPHA_CRACK = 0.55
ALPHA_DISTRACTOR = 0.40
DRIFT_AMPLITUDE = 0.02  # |g_n - 1| stays well under hard_floor's ~0.10 margin
NOISE_STD = 4.0
BLUR_SIGMA = 1.0

CRACK_APPEAR_FRAME = 2  # frames < this: no cracks at all (frame 0 is undamaged)

EDGE_START_FRAME = 4
EDGE_END_FRAME = N_FRAMES - 1
EDGE_MAX_H = 70.0
EDGE_SIN_AMPLITUDE = 25.0
EDGE_SIN_LAMBDA = 400.0
EDGE_SIN_PHASE = 80.0  # sin peaks near x=180, aligned with crack "E" below

DISTRACTOR_START_FRAME = 5
DISTRACTOR_Y = (90, 120)
DISTRACTOR_X = (1150, 1230)

# Detection parameters used for the *entire* sequence -- one fixed set, no
# per-frame tuning, and never adjusted based on the ground truth. Reuses the
# known-good primary-edge parameter shape from examples/02_multi_interface_edge_delamination.py,
# with window sizes rescaled down for this synthetic image's smaller features.
DETECTION_EDGE_PARAMS: Dict[str, Any] = {
    "window_edge": (1, 40),
    "gaussian_filters": (0.5, 15.0),
    "hard_floor": 0.90,
    "scale_min_percentile": 10,
    "scale_max_percentile": 95,
    "seed_ratio": 0.01,
    "post_threshold_closing_px": 10,
}
# window_diffuse must be smaller than the perpendicular width of the smallest
# detectable diffuse blob, or the max/min-filter closing erases it entirely
# (verified empirically: the real-data default (0, 60) wipes out blobs
# narrower than ~60 px in the direction perpendicular to the crack).
DETECTION_DIFFUSE_PARAMS: Dict[str, Any] = {
    "diffuse_dx": 70.0,
    "diffuse_dy": 70.0,
    "window_diffuse": (1, 20),
    "gaussian_filters": (0.5, 15.0),
    "hard_floor": 0.90,
    "scale_min_percentile": 10,
    "scale_max_percentile": 99,
    "post_threshold_closing_px": 4,
}
# The combined-precedence rule (D_final = D_raw \ E, M = E U D_final) is only
# exactly the documented rule when edge isn't pre-dilated -- detect_both_delaminations
# dilates the edge mask by edge_exclusion_px (default 5) *before* arbitration.
EDGE_EXCLUSION_PX = 0


@dataclass(frozen=True)
class CrackSpec:
    """One ground-truth crack + its associated diffuse-delamination growth."""

    name: str
    y: float
    x1: float
    x2: float
    kind: str  # "ellipse", "band", "two_lobed"
    start_frame: int
    max_ry: float = 0.0
    max_rx: float = 0.0
    max_half_width: float = 0.0
    wave_amp: float = 0.10
    wave_freq: float = 7.0
    lobe_offset: float = 0.0


CRACKS: Tuple[CrackSpec, ...] = (
    CrackSpec(name="E", y=55.0, x1=150.0, x2=210.0, kind="ellipse",
              start_frame=8, max_ry=50.0, max_rx=55.0, wave_amp=0.12, wave_freq=5.0),
    CrackSpec(name="A", y=170.0, x1=350.0, x2=430.0, kind="ellipse",
              start_frame=6, max_ry=40.0, max_rx=65.0, wave_amp=0.10, wave_freq=7.0),
    CrackSpec(name="B", y=190.0, x1=520.0, x2=660.0, kind="band",
              start_frame=5, max_half_width=35.0, wave_amp=0.08, wave_freq=10.0),
    CrackSpec(name="C", y=150.0, x1=780.0, x2=840.0, kind="two_lobed",
              start_frame=7, max_ry=28.0, max_rx=28.0, wave_amp=0.10, wave_freq=6.0, lobe_offset=15.0),
    CrackSpec(name="D1", y=230.0, x1=950.0, x2=1030.0, kind="ellipse",
              start_frame=9, max_ry=30.0, max_rx=50.0, wave_amp=0.09, wave_freq=8.0),
    CrackSpec(name="D2", y=235.0, x1=1000.0, x2=1080.0, kind="ellipse",
              start_frame=10, max_ry=30.0, max_rx=50.0, wave_amp=0.09, wave_freq=9.0),
)


# ---------------------------------------------------------------------------
# Growth helpers
# ---------------------------------------------------------------------------

def _smoothstep(t: float) -> float:
    """Smooth, monotonic 0->1 easing (3t^2 - 2t^3) for a monotonic input t."""
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _growth_fraction(n: int, start: int, end: int) -> float:
    """Return 0 at/before ``start``, 1 at/after ``end``, linear in between."""
    if end <= start:
        return 1.0 if n >= end else 0.0
    return float(np.clip((n - start) / (end - start), 0.0, 1.0))


def _crack_growth(spec: CrackSpec, n: int, n_frames: int) -> float:
    return _smoothstep(_growth_fraction(n, spec.start_frame, n_frames - 1))


def _edge_growth(n: int, n_frames: int) -> float:
    return _smoothstep(_growth_fraction(n, EDGE_START_FRAME, n_frames - 1))


# ---------------------------------------------------------------------------
# Mask geometry primitives
# ---------------------------------------------------------------------------

def _wavy_ellipse_mask(
    shape: Tuple[int, int],
    cy: float,
    cx: float,
    ry: float,
    rx: float,
    *,
    wave_amp: float,
    wave_freq: float,
    phase: float = 0.0,
) -> np.ndarray:
    """Ellipse mask with a small deterministic angular boundary perturbation."""
    if ry <= 0.0 or rx <= 0.0:
        return np.zeros(shape, dtype=bool)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    dy = (yy - cy) / ry
    dx = (xx - cx) / rx
    r = np.sqrt(dy * dy + dx * dx)
    theta = np.arctan2(dy, dx)
    perturb = 1.0 + wave_amp * np.sin(wave_freq * theta + phase)
    return r <= perturb


def _wavy_band_mask(
    shape: Tuple[int, int],
    cy: float,
    x1: float,
    x2: float,
    half_width: float,
    *,
    wave_amp: float,
    wave_freq: float,
) -> np.ndarray:
    """Stadium-shaped band along a horizontal crack, with wavy edges."""
    if half_width <= 0.0:
        return np.zeros(shape, dtype=bool)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    xc = np.clip(xx, x1, x2)
    dy = yy - cy
    dx = xx - xc
    dist = np.sqrt(dy.astype(np.float64) ** 2 + dx.astype(np.float64) ** 2)
    theta = np.arctan2(dy, xx - (x1 + x2) / 2.0)
    perturb = half_width * (1.0 + wave_amp * np.sin(wave_freq * theta))
    return dist <= perturb


def diffuse_blob_mask(shape: Tuple[int, int], spec: CrackSpec, n: int, n_frames: int = N_FRAMES) -> np.ndarray:
    """Return the raw diffuse-delamination mask grown from one crack at frame n."""
    g = _crack_growth(spec, n, n_frames)
    cx = (spec.x1 + spec.x2) / 2.0
    if spec.kind == "ellipse":
        return _wavy_ellipse_mask(
            shape, spec.y, cx, spec.max_ry * g, spec.max_rx * g,
            wave_amp=spec.wave_amp, wave_freq=spec.wave_freq,
        )
    if spec.kind == "band":
        return _wavy_band_mask(
            shape, spec.y, spec.x1, spec.x2, spec.max_half_width * g,
            wave_amp=spec.wave_amp, wave_freq=spec.wave_freq,
        )
    if spec.kind == "two_lobed":
        left = _wavy_ellipse_mask(
            shape, spec.y, cx - spec.lobe_offset, spec.max_ry * g, spec.max_rx * g,
            wave_amp=spec.wave_amp, wave_freq=spec.wave_freq, phase=0.0,
        )
        right = _wavy_ellipse_mask(
            shape, spec.y, cx + spec.lobe_offset, spec.max_ry * g, spec.max_rx * g,
            wave_amp=spec.wave_amp, wave_freq=spec.wave_freq, phase=np.pi,
        )
        return left | right
    raise ValueError(f"Unknown crack kind: {spec.kind!r}")


def edge_mask_for_frame(shape: Tuple[int, int], n: int, n_frames: int = N_FRAMES) -> np.ndarray:
    """Non-uniform, monotonically growing edge-delamination ground truth."""
    h, w = shape
    g = _edge_growth(n, n_frames)
    h0 = EDGE_MAX_H * g
    xx = np.arange(w, dtype=np.float64)
    front = h0 + g * EDGE_SIN_AMPLITUDE * np.sin(2.0 * np.pi * (xx - EDGE_SIN_PHASE) / EDGE_SIN_LAMBDA)
    front = np.clip(front, 0.0, None)
    front_px = np.round(front).astype(int)
    yy = np.arange(h)[:, None]
    top_mask = yy < front_px[None, :]
    bottom_mask = yy >= (h - front_px[None, :])
    return top_mask | bottom_mask


def distractor_mask_for_frame(shape: Tuple[int, int], n: int) -> np.ndarray:
    """Isolated dark region near, but disconnected from, the top edge."""
    mask = np.zeros(shape, dtype=bool)
    if n < DISTRACTOR_START_FRAME:
        return mask
    y0, y1 = DISTRACTOR_Y
    x0, x1 = DISTRACTOR_X
    mask[y0:y1, x0:x1] = True
    return mask


def crack_line_mask_for_frame(shape: Tuple[int, int], n: int) -> np.ndarray:
    """Thin rendered crack lines (visual only; not the diffuse ROI ground truth)."""
    mask = np.zeros(shape, dtype=bool)
    if n < CRACK_APPEAR_FRAME:
        return mask
    h, w = shape
    for spec in CRACKS:
        y = int(round(spec.y))
        x1 = max(0, int(round(spec.x1)))
        x2 = min(w, int(round(spec.x2)))
        y_lo, y_hi = max(0, y - 1), min(h, y + 2)
        mask[y_lo:y_hi, x1:x2] = True
    return mask


def cracks_for_frame(n: int) -> np.ndarray:
    """Ground-truth crack endpoints in the (y, x) convention DelaDect expects."""
    if n < CRACK_APPEAR_FRAME:
        return np.empty((0, 2, 2), dtype=np.float64)
    segments = np.array(
        [[[spec.y, spec.x1], [spec.y, spec.x2]] for spec in CRACKS],
        dtype=np.float64,
    )
    return segments


# ---------------------------------------------------------------------------
# Background + rendering
# ---------------------------------------------------------------------------

def generate_background(rng: np.random.Generator, shape: Tuple[int, int]) -> np.ndarray:
    """Undamaged, transmitted-light-like background B(x, y), in [0, 255]."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)

    gradient = 205.0 + 20.0 * (xx / w) - 10.0 * (yy / h)

    small = rng.normal(0.0, 1.0, size=(9, 13)).astype(np.float64)
    texture = resize(small, (h, w), order=3, mode="reflect", anti_aliasing=False)
    texture *= 7.0

    stitching = 4.0 * np.sin(2.0 * np.pi * xx / 47.0) * (0.5 + 0.5 * np.sin(2.0 * np.pi * yy / 181.0))

    background = gradient + texture + stitching
    return np.clip(background, 140.0, 245.0)


def render_frame(
    n: int,
    background: np.ndarray,
    edge_mask: np.ndarray,
    diffuse_raw_mask: np.ndarray,
    crack_mask: np.ndarray,
    distractor_mask: np.ndarray,
    noise_rng: np.random.Generator,
) -> np.ndarray:
    """Render one 8-bit frame from the additive/multiplicative damage model."""
    g_n = 1.0 + DRIFT_AMPLITUDE * np.sin(n / 6.0)
    attenuation = (
        1.0
        - ALPHA_EDGE * edge_mask
        - ALPHA_DIFFUSE * diffuse_raw_mask
        - ALPHA_CRACK * crack_mask
        - ALPHA_DISTRACTOR * distractor_mask
    )
    attenuation = np.clip(attenuation, 0.0, 1.0)
    frame = g_n * background * attenuation
    frame = ndi.gaussian_filter(frame, BLUR_SIGMA)
    frame = frame + noise_rng.normal(0.0, NOISE_STD, size=frame.shape)
    return np.clip(frame, 0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Sequence assembly
# ---------------------------------------------------------------------------

def generate_synthetic_sequence(
    *,
    seed: int = SEED,
    n_frames: int = N_FRAMES,
    shape: Tuple[int, int] = SHAPE,
) -> Dict[str, Any]:
    """Generate the full deterministic synthetic sequence and its ground truth.

    Returns a dict with (all keyed/indexed by frame number 0..n_frames-1):

    - ``frames``: list of ``uint8`` rendered images.
    - ``background``: the undamaged reference image (frame 0 == this, exactly).
    - ``cracks_by_frame``: list of ``(k, 2, 2)`` arrays of exact ``(y, x)``
      crack endpoints, passed directly to diffuse detection.
    - ``edge_gt``, ``diffuse_raw_gt``, ``diffuse_final_gt``, ``combined_gt``:
      per-frame boolean ground-truth masks (already cumulative -- see below).
    - ``edge_gt_cum`` etc: explicit cumulative-union versions of the above
      (``G_n^cum = union_{j<=n} G_j``); included even though the masks above
      are already monotonically growing by construction, so cumulative union
      is a no-op -- this makes the cumulative-comparison contract explicit.
    - ``distractor_gt``: the disconnected dark region, deliberately excluded
      from ``edge_gt``.
    - ``params``: every generation parameter used, plus the seed.
    """
    rng_bg = np.random.default_rng(seed)
    rng_noise = np.random.default_rng(seed + 1)

    background = generate_background(rng_bg, shape)

    frames: List[np.ndarray] = []
    cracks_by_frame: List[np.ndarray] = []
    edge_gt: List[np.ndarray] = []
    diffuse_raw_gt: List[np.ndarray] = []
    diffuse_final_gt: List[np.ndarray] = []
    combined_gt: List[np.ndarray] = []
    distractor_gt: List[np.ndarray] = []
    crack_render_gt: List[np.ndarray] = []

    for n in range(n_frames):
        edge_n = edge_mask_for_frame(shape, n, n_frames)
        diffuse_raw_n = np.zeros(shape, dtype=bool)
        for spec in CRACKS:
            diffuse_raw_n |= diffuse_blob_mask(shape, spec, n, n_frames)
        distractor_n = distractor_mask_for_frame(shape, n)
        crack_render_n = crack_line_mask_for_frame(shape, n)

        # Ground-truth precedence rule mirrors DelaminationDetector's
        # _apply_edge_precedence exactly (with edge_exclusion_px=0, the
        # detector's "edge_exclusion" mask equals the raw edge mask).
        diffuse_final_n = diffuse_raw_n & ~edge_n
        combined_n = edge_n | diffuse_final_n

        frame = render_frame(
            n, background, edge_n, diffuse_raw_n, crack_render_n, distractor_n, rng_noise,
        )

        frames.append(frame)
        cracks_by_frame.append(cracks_for_frame(n))
        edge_gt.append(edge_n)
        diffuse_raw_gt.append(diffuse_raw_n)
        diffuse_final_gt.append(diffuse_final_n)
        combined_gt.append(combined_n)
        distractor_gt.append(distractor_n)
        crack_render_gt.append(crack_render_n)

    def _cumulative(masks: List[np.ndarray]) -> List[np.ndarray]:
        cum: List[np.ndarray] = []
        running = np.zeros(shape, dtype=bool)
        for m in masks:
            running = running | m
            cum.append(running.copy())
        return cum

    params = {
        "seed": seed,
        "n_frames": n_frames,
        "shape": shape,
        "scale_px_mm": SCALE_PX_MM,
        "alpha_edge": ALPHA_EDGE,
        "alpha_diffuse": ALPHA_DIFFUSE,
        "alpha_crack": ALPHA_CRACK,
        "alpha_distractor": ALPHA_DISTRACTOR,
        "drift_amplitude": DRIFT_AMPLITUDE,
        "noise_std": NOISE_STD,
        "blur_sigma": BLUR_SIGMA,
        "edge_start_frame": EDGE_START_FRAME,
        "edge_max_h": EDGE_MAX_H,
        "edge_sin_amplitude": EDGE_SIN_AMPLITUDE,
        "edge_sin_lambda": EDGE_SIN_LAMBDA,
        "crack_appear_frame": CRACK_APPEAR_FRAME,
        "distractor_start_frame": DISTRACTOR_START_FRAME,
        "detection_edge_params": DETECTION_EDGE_PARAMS,
        "detection_diffuse_params": DETECTION_DIFFUSE_PARAMS,
        "edge_exclusion_px": EDGE_EXCLUSION_PX,
    }

    return {
        "frames": frames,
        "background": background,
        "cracks_by_frame": cracks_by_frame,
        "edge_gt": edge_gt,
        "diffuse_raw_gt": diffuse_raw_gt,
        "diffuse_final_gt": diffuse_final_gt,
        "combined_gt": combined_gt,
        "edge_gt_cum": _cumulative(edge_gt),
        "diffuse_raw_gt_cum": _cumulative(diffuse_raw_gt),
        "diffuse_final_gt_cum": _cumulative(diffuse_final_gt),
        "combined_gt_cum": _cumulative(combined_gt),
        "distractor_gt": distractor_gt,
        "crack_render_gt": crack_render_gt,
        "params": params,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection over union; 1.0 when both masks are empty."""
    inter = np.count_nonzero(pred & gt)
    union = np.count_nonzero(pred | gt)
    return 1.0 if union == 0 else inter / union


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice coefficient; 1.0 when both masks are empty."""
    inter = np.count_nonzero(pred & gt)
    denom = int(np.count_nonzero(pred)) + int(np.count_nonzero(gt))
    return 1.0 if denom == 0 else 2.0 * inter / denom


def boundary_iou(pred: np.ndarray, gt: np.ndarray, *, dilation_px: int = 5) -> float:
    """Boundary IoU (Cheng et al., 2021): IoU restricted to a boundary band.

    Standard mask IoU is dominated by interior area and can stay high even
    when a large mask's *boundary* is systematically off. Boundary IoU
    instead only scores the ``dilation_px``-wide band just inside each
    mask's border: ``pred_band = pred & ~erode(pred, dilation_px)`` (and
    likewise for ``gt``), then IoU is computed between the two bands. 1.0
    when both boundary bands are empty (both masks are empty or fill the
    whole frame).
    """
    from scipy import ndimage as ndi

    def _boundary_band(mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            return np.zeros_like(mask, dtype=bool)
        eroded = ndi.binary_erosion(mask, iterations=max(1, int(dilation_px)))
        return mask & ~eroded

    pred_band = _boundary_band(np.asarray(pred, dtype=bool))
    gt_band = _boundary_band(np.asarray(gt, dtype=bool))
    return iou_score(pred_band, gt_band)


def front_heights(mask: np.ndarray, *, side: str = "top") -> np.ndarray:
    """Per-column front position (in px from the border) of a border-connected mask.

    Assumes the mask grows as a contiguous run from the specified border
    inward in each column (true by construction for the edge ground truth,
    and for DelaDect's edge-connected reconstruction output). Column ``x``'s
    height is the number of leading ``True`` pixels from that border; a
    fully undamaged column returns 0, a fully damaged column returns the
    image height.
    """
    if side not in {"top", "bottom"}:
        raise ValueError("side must be 'top' or 'bottom'")
    m = mask if side == "top" else mask[::-1, :]
    h, _w = m.shape
    all_true = m.all(axis=0)
    all_false = ~m.any(axis=0)
    first_false = np.argmax(~m, axis=0)  # index of first False row per column
    heights = np.where(all_true, h, np.where(all_false, 0, first_false))
    return heights.astype(np.int64)


def edge_front_error(
    pred_edge: np.ndarray,
    gt_edge: np.ndarray,
    *,
    scale_px_mm: float,
) -> Dict[str, float]:
    """Column-wise front-position error between detected and ground-truth edge masks.

    Computes ``e_front = sqrt(mean_x[(y_det(x) - y_GT(x))^2])`` (and the
    corresponding mean absolute error) pooling columns from both the top and
    bottom fronts, in pixels and in mm (via ``scale_px_mm``). Unlike area
    IoU, this is sensitive to a front that is uniformly offset by a few
    pixels even when the total detected/GT areas are nearly equal.
    """
    diffs: List[np.ndarray] = []
    for side in ("top", "bottom"):
        det_h = front_heights(pred_edge, side=side)
        gt_h = front_heights(gt_edge, side=side)
        diffs.append((det_h - gt_h).astype(np.float64))
    all_diffs = np.concatenate(diffs)
    mae_px = float(np.mean(np.abs(all_diffs)))
    rmse_px = float(np.sqrt(np.mean(all_diffs ** 2)))
    return {
        "mae_px": mae_px,
        "rmse_px": rmse_px,
        "mae_mm": mae_px / scale_px_mm,
        "rmse_mm": rmse_px / scale_px_mm,
    }


def area_errors(pred_px: int, gt_px: int) -> Tuple[int, int, float]:
    """Return (signed_px, abs_px, relative) area error; relative is NaN if gt_px == 0."""
    signed = int(pred_px) - int(gt_px)
    absolute = abs(signed)
    relative = float(signed) / gt_px if gt_px > 0 else float("nan")
    return signed, absolute, relative


def px_to_mm2(pixels: int, scale_px_mm: float) -> float:
    """Convert a pixel count to mm^2 using A = N_pixels / s^2."""
    return float(pixels) / (scale_px_mm ** 2)


def compute_metrics(
    *,
    frame_indices: List[int],
    det_masks: Dict[str, Dict[str, np.ndarray]],
    gt_cum: Dict[str, List[np.ndarray]],
    scale_px_mm: float,
) -> pd.DataFrame:
    """Build the per-frame metrics table for edge / diffuse (final) / combined."""
    channels = {
        "edge": ("edge_exclusion", "edge_gt_cum"),
        "diffuse": ("diffuse", "diffuse_final_gt_cum"),
        "combined": ("combined", "combined_gt_cum"),
    }
    rows: List[Dict[str, Any]] = []
    for n in frame_indices:
        frame_key = f"frame_{n:04d}"
        row: Dict[str, Any] = {"frame": n}
        for label, (det_key, gt_key) in channels.items():
            pred = np.asarray(det_masks[det_key][frame_key], dtype=bool)
            gt = np.asarray(gt_cum[gt_key][n], dtype=bool)
            pred_px = int(np.count_nonzero(pred))
            gt_px = int(np.count_nonzero(gt))
            signed, absolute, relative = area_errors(pred_px, gt_px)
            row[f"{label}_gt_px"] = gt_px
            row[f"{label}_det_px"] = pred_px
            row[f"{label}_gt_mm2"] = px_to_mm2(gt_px, scale_px_mm)
            row[f"{label}_det_mm2"] = px_to_mm2(pred_px, scale_px_mm)
            row[f"{label}_iou"] = iou_score(pred, gt)
            row[f"{label}_dice"] = dice_score(pred, gt)
            row[f"{label}_boundary_iou"] = boundary_iou(pred, gt)
            row[f"{label}_signed_err_px"] = signed
            row[f"{label}_abs_err_px"] = absolute
            row[f"{label}_rel_err"] = relative
            if label == "edge":
                front_err = edge_front_error(pred, gt, scale_px_mm=scale_px_mm)
                row["edge_front_mae_px"] = front_err["mae_px"]
                row["edge_front_rmse_px"] = front_err["rmse_px"]
                row["edge_front_mae_mm"] = front_err["mae_mm"]
                row["edge_front_rmse_mm"] = front_err["rmse_mm"]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Detection run + figure + CLI entry point (imports matplotlib/deladect lazily
# so generation/metrics stay importable without the full detection stack)
# ---------------------------------------------------------------------------

def _write_frames_to_disk(frames: List[np.ndarray], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for n, frame in enumerate(frames):
        Image.fromarray(frame, mode="L").save(out_dir / f"{n:04d}_sc.png")


def make_standalone_owner(avg_crack_width_px: float = 8.0) -> Any:
    """Build a minimal "owner" so EdgeDetector/DiffuseDetector can run without a Specimen.

    Both detectors are written against an owning :class:`DelaminationDetector`
    (``self.owner``), but when frames are supplied directly via
    ``processed_stack=`` (skipping preprocessing/overlays/debug output), the
    only owner state actually touched is ``specimen.avg_crack_width_px``,
    ``_uses_stack_overrides()``/``_select_stacks()`` (both stubbed to the
    full-frame, non-region-override path), and the threshold helpers
    ``_images_threshold``/``_kmeans_threshold`` -- which don't reference
    ``self`` in :class:`DelaminationDetector` at all, so the real
    implementations are reused unbound.

    Used by the parameter-sweep studies in
    ``examples/synthetic_delamination_robustness.py``, where constructing a
    real :class:`~deladect.specimen.Specimen` (and writing PNGs to disk) for
    every one of hundreds of grid cells would dominate runtime.
    """
    from deladect.detection.delamination.core import DelaminationDetector

    class _StandaloneOwner:
        def __init__(self) -> None:
            self.specimen = type("_Specimen", (), {"image_stack_full": None, "avg_crack_width_px": avg_crack_width_px})()
            self.interface = type("_Interface", (), {"name": "standalone"})()
            self.save_preprocess_outputs = False
            self._notice_flags: Dict[str, bool] = {}

        def _uses_stack_overrides(self) -> bool:
            return False

        def _select_stacks(self) -> Dict[str, Any]:
            return {"upper": None, "lower": None, "middle": None, "full": None}

        _images_threshold = DelaminationDetector._images_threshold
        _kmeans_threshold = DelaminationDetector._kmeans_threshold

    return _StandaloneOwner()


def run_detection(sequence: Dict[str, Any], *, results_root: Path = RESULTS_ROOT) -> Dict[str, Any]:
    """Write frames to disk and run the normal public DelaDect workflow on them."""
    from deladect.detection.delamination.core import DelaminationDetector
    from deladect.specimen import Specimen

    frames_dir = results_root / "frames" / "full"
    _write_frames_to_disk(sequence["frames"], frames_dir)

    specimen = Specimen(
        name="synthetic-validation",
        scale_px_mm=sequence["params"]["scale_px_mm"],
        path_full=str(frames_dir),
        sorting_key="_sc",
        image_types=["png"],
        results_root=str(results_root),
        avg_crack_width_px=8.0,
    )
    specimen.add_ply(name="ply_0", orientation_deg=0.0, avg_crack_width_px=8.0, min_crack_length_px=20.0)
    specimen.add_ply(name="ply_1", orientation_deg=90.0, avg_crack_width_px=8.0, min_crack_length_px=20.0)
    specimen.add_interface(name="synthetic", upper_ply=0, lower_ply=1)

    detector = DelaminationDetector(specimen, specimen.interfaces[0])

    result = detector.detect_both_delaminations(
        cracks=sequence["cracks_by_frame"],
        edge_exclusion_px=EDGE_EXCLUSION_PX,
        track_cracks=False,
        return_masks=True,
        save_overlays=False,
        save_masks=True,
        save_metrics=False,
        debug=False,
        edge_params=DETECTION_EDGE_PARAMS,
        diffuse_params=DETECTION_DIFFUSE_PARAMS,
    )
    return {"specimen": specimen, "detector": detector, "result": result}


def _configure_style() -> None:
    """Self-contained matplotlib style (no external plot_style/LaTeX dependency).

    Deliberately does not use the machine-local ``my_plots/plot_style`` helper
    used by some other example scripts: that module lives outside this
    repository and its ``usetex=True`` mode requires a LaTeX install, neither
    of which is guaranteed on a clean checkout. This mimics the same general
    sizing/typography conventions (serif font, ~160 mm page width, small
    consistent font sizes) without either dependency.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "CMU Serif", "Times New Roman"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })


def _classified_rgba(edge_mask: np.ndarray, diffuse_mask: np.ndarray) -> np.ndarray:
    """RGBA overlay: red=edge, green=diffuse, alpha=0 elsewhere."""
    h, w = edge_mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[edge_mask, 0] = 0.89
    rgba[edge_mask, 3] = 0.55
    rgba[diffuse_mask, 1] = 0.65
    rgba[diffuse_mask, 3] = 0.55
    return rgba


TP_COLOR: Tuple[float, float, float, float] = (0.12, 0.47, 0.71, 0.85)  # tab:blue
FP_COLOR: Tuple[float, float, float, float] = (1.00, 0.50, 0.05, 0.85)  # tab:orange
FN_COLOR: Tuple[float, float, float, float] = (0.58, 0.40, 0.74, 0.85)  # tab:purple


def _error_map_rgba(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """RGBA pixelwise comparison: blue=TP, orange=FP, purple=FN.

    Deliberately disjoint from the red/green edge/diffuse classification
    palette used in panels (b)/(d) of the main figure, so the same colours
    never carry two different meanings across adjacent panels.
    """
    h, w = pred.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt
    rgba[tp] = TP_COLOR
    rgba[fp] = FP_COLOR
    rgba[fn] = FN_COLOR
    return rgba


def make_figure(
    sequence: Dict[str, Any],
    metrics: pd.DataFrame,
    det_masks: Dict[str, Dict[str, np.ndarray]],
    *,
    display_frame: int,
    output_dir: Path,
) -> Path:
    """Build the six-panel explanatory figure and save PNG + PDF."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    _configure_style()

    n = display_frame
    frame_key = f"frame_{n:04d}"
    background = sequence["background"]
    raw_frame = sequence["frames"][n]
    edge_gt = sequence["edge_gt_cum"][n]
    diffuse_gt = sequence["diffuse_final_gt_cum"][n]
    combined_gt = sequence["combined_gt_cum"][n]

    det_edge = np.asarray(det_masks["edge_exclusion"][frame_key], dtype=bool)
    det_diffuse = np.asarray(det_masks["diffuse"][frame_key], dtype=bool)
    det_combined = np.asarray(det_masks["combined"][frame_key], dtype=bool)

    # This figure is normally shown at full documentation-column width.  Keep
    # its text substantially larger than Matplotlib's compact plotting defaults
    # so panel labels and legends remain readable after the image is scaled by
    # the HTML/PDF documentation.  These are exactly 1.8x the previous sizes.
    title_fontsize = 27.0
    label_fontsize = 23.4
    tick_fontsize = 21.6
    legend_fontsize = 21.6

    fig, axes = plt.subplots(2, 3, figsize=(18.0, 11.0))
    fig.subplots_adjust(left=0.04, right=0.99, top=0.93, bottom=0.22, hspace=0.38, wspace=0.12)
    ax = axes.ravel()

    ax[0].imshow(background, cmap="gray", vmin=0, vmax=255)
    ax[0].set_title("(a) Undamaged background", fontsize=title_fontsize)

    ax[1].imshow(background, cmap="gray", vmin=0, vmax=255)
    ax[1].imshow(_classified_rgba(edge_gt, diffuse_gt))
    for spec in CRACKS:
        ax[1].plot([spec.x1, spec.x2], [spec.y, spec.y], color="black", linewidth=1.0)
    ax[1].set_title("(b) Ground truth", fontsize=title_fontsize)

    ax[2].imshow(raw_frame, cmap="gray", vmin=0, vmax=255)
    ax[2].set_title("(c) Rendered damaged frame", fontsize=title_fontsize)

    ax[3].imshow(raw_frame, cmap="gray", vmin=0, vmax=255)
    ax[3].imshow(_classified_rgba(det_edge, det_diffuse))
    ax[3].set_title("(d) DelaDect\nclassified output", fontsize=title_fontsize)

    ax[4].imshow(raw_frame, cmap="gray", vmin=0, vmax=255)
    ax[4].imshow(_error_map_rgba(det_combined, combined_gt))
    ax[4].set_title("(e) Pixelwise comparison", fontsize=title_fontsize)

    for a in ax[:5]:
        a.set_xticks([])
        a.set_yticks([])

    ax_area = ax[5]
    frames_axis = metrics["frame"].to_numpy()
    ax_area.plot(frames_axis, metrics["edge_gt_mm2"], color="#e41a1c", linestyle="-")
    ax_area.plot(frames_axis, metrics["edge_det_mm2"], color="#e41a1c", linestyle="--")
    ax_area.plot(frames_axis, metrics["diffuse_gt_mm2"], color="#4daf4a", linestyle="-")
    ax_area.plot(frames_axis, metrics["diffuse_det_mm2"], color="#4daf4a", linestyle="--")
    ax_area.set_xlabel("Frame", fontsize=label_fontsize)
    ax_area.set_ylabel(r"Area [mm$^2$]", fontsize=label_fontsize)
    ax_area.set_title("(f) Area vs. frame", fontsize=title_fontsize)
    ax_area.tick_params(axis="both", labelsize=tick_fontsize)
    type_handles = [
        Line2D([0], [0], color="black", linestyle="-", label="Ground truth"),
        Line2D([0], [0], color="black", linestyle="--", label="Detected"),
    ]
    ax_area.legend(
        handles=type_handles, loc="upper left", frameon=False,
        fontsize=legend_fontsize,
    )

    # Give the large y-axis label clear space between panels (e) and (f)
    # without changing the font size or the width of any image panel.
    area_position = ax_area.get_position()
    area_left_inset = 0.035
    ax_area.set_position([
        area_position.x0 + area_left_inset,
        area_position.y0,
        area_position.width - area_left_inset,
        area_position.height,
    ])

    classified_handles = [
        Patch(facecolor=(0.89, 0.10, 0.11, 0.55), label="Edge delamination"),
        Patch(facecolor=(0.0, 0.65, 0.0, 0.55), label="Diffuse delamination"),
        Line2D([0], [0], color="black", linewidth=1.0, label="Matrix crack"),
    ]
    error_handles = [
        Patch(facecolor=TP_COLOR, label="True positive"),
        Patch(facecolor=FP_COLOR, label="False positive"),
        Patch(facecolor=FN_COLOR, label="False negative"),
    ]
    fig.legend(
        handles=classified_handles, loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.50, 0.075), fontsize=legend_fontsize,
    )
    fig.legend(
        handles=error_handles, loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.50, 0.012), fontsize=legend_fontsize,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "synthetic_validation.png"
    pdf_path = output_dir / "synthetic_validation.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def main() -> None:
    print(f"Generating synthetic sequence (seed={SEED}, n_frames={N_FRAMES}, shape={SHAPE})...")
    sequence = generate_synthetic_sequence()

    print("Running DelaDect detection (detect_both_delaminations)...")
    run = run_detection(sequence)
    result = run["result"]
    det_masks = result["masks"]

    frame_indices = list(range(N_FRAMES))
    metrics = compute_metrics(
        frame_indices=frame_indices,
        det_masks=det_masks,
        gt_cum={
            "edge_gt_cum": sequence["edge_gt_cum"],
            "diffuse_final_gt_cum": sequence["diffuse_final_gt_cum"],
            "combined_gt_cum": sequence["combined_gt_cum"],
        },
        scale_px_mm=sequence["params"]["scale_px_mm"],
    )

    metrics_dir = RESULTS_ROOT / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "synthetic_validation_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"Metrics written to: {metrics_path}")

    DOCS_STATIC_DIR.mkdir(parents=True, exist_ok=True)
    docs_metrics_path = DOCS_STATIC_DIR / "synthetic_validation_metrics.csv"
    shutil.copy2(metrics_path, docs_metrics_path)

    display_frame = N_FRAMES - 1
    figure_dir = RESULTS_ROOT / "documentation"
    png_path = make_figure(
        sequence, metrics, det_masks, display_frame=display_frame, output_dir=figure_dir,
    )
    for suffix in (".png", ".pdf"):
        shutil.copy2(figure_dir / f"synthetic_validation{suffix}", DOCS_STATIC_DIR / f"synthetic_validation{suffix}")
    print(f"Figure written to: {png_path}")
    print(f"Documentation copies: {DOCS_STATIC_DIR / 'synthetic_validation.png'}, "
          f"{DOCS_STATIC_DIR / 'synthetic_validation.pdf'}")

    summary_cols = [
        "frame", "edge_iou", "diffuse_iou", "combined_iou",
        "edge_dice", "diffuse_dice", "combined_dice",
        "combined_rel_err",
    ]
    print("\nSummary (last frame):")
    print(metrics[summary_cols].tail(1).to_string(index=False))
    print("\nMean IoU / Dice over frames where ground truth is non-empty (combined):")
    nonzero = metrics[metrics["combined_gt_px"] > 0]
    print(f"  IoU:  {nonzero['combined_iou'].mean():.4f}")
    print(f"  Dice: {nonzero['combined_dice'].mean():.4f}")


if __name__ == "__main__":
    main()
