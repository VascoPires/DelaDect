"""Helpers shared by the delamination modules."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, unsharp_mask

from deladect.specimen import Specimen, sanitize_path_token

CrackAnalysisMapping = Mapping[str, Mapping[str, Any]]
CrackInput = Union[Sequence[np.ndarray], CrackAnalysisMapping]

PROGRESS_MILESTONES: Tuple[int, ...] = (25, 50, 75, 90)


def _result_key_token(value: Any) -> str:
    """Turn a name into a safe folder name."""
    return sanitize_path_token(value, fallback="interface")


class _Progress:
    """Print start, 25/50/75/90% and done lines for a frame loop, if enabled."""

    def __init__(self, stage: str, total_frames: int, enabled: bool) -> None:
        self.stage = stage
        self.total = max(0, int(total_frames))
        self.enabled = enabled
        self._pending = list(PROGRESS_MILESTONES)
        if enabled:
            print(f"[progress] {stage}: start ({self.total} frames)", flush=True)

    def update(self, completed_frames: int) -> None:
        if not self.enabled:
            return
        total = max(1, self.total)
        completed = max(0, int(completed_frames))
        percent = 100.0 * completed / total
        while self._pending and percent >= self._pending[0]:
            print(f"[progress] {self.stage}: {self._pending.pop(0)}% ({completed}/{total})", flush=True)

    def done(self) -> None:
        if self.enabled:
            print(f"[progress] {self.stage}: done ({self.total}/{self.total})", flush=True)


def _ensure_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert a frame to single-channel ``uint8``; images with values in ``[0, 1]`` are scaled to 255."""
    frame_float = frame.astype(np.float32)
    if frame_float.ndim == 3:
        frame_float = frame_float.mean(axis=2)
    if frame_float.max() <= 1.0:
        frame_float = frame_float * 255.0
    return np.clip(frame_float, 0, 255).astype(np.uint8)


def _frame_to_float(frame: np.ndarray) -> np.ndarray:
    """Convert a frame to single-channel ``float32`` in ``[0, 1]``."""
    frame_float = frame.astype(np.float32)
    if frame_float.ndim == 3:
        frame_float = frame_float.mean(axis=2)
    if frame_float.max() > 1.0:
        frame_float = frame_float / 255.0
    return frame_float


def _smooth_for_threshold(
    image_uint8: np.ndarray,
    window: Tuple[int, int],
    gaussian_sigmas: Tuple[float, float],
    avg_crack_width_px: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare an image for thresholding.

    A max filter followed by a min filter over ``window`` removes thin
    cracks, then an unsharp mask and a Gaussian blur smooth the result.
    Returns ``(filtered_max, filtered_min, sharpened, smoothed)``.
    """
    size = (max(1, int(window[0])), max(1, int(window[1])))
    filtered_max = ndi.maximum_filter(image_uint8, size=size, mode="reflect")
    filtered_min = ndi.minimum_filter(filtered_max, size=size, mode="reflect")
    sharpened = unsharp_mask(filtered_min, radius=float(avg_crack_width_px), amount=2.0, preserve_range=True)
    smoothed = ndi.gaussian_filter(sharpened, gaussian_sigmas)
    return filtered_max, filtered_min, sharpened, smoothed


def _percentile_range(image: np.ndarray, pct_min: Optional[float], pct_max: Optional[float]) -> Optional[Tuple[float, float]]:
    """The ``pct_min`` and ``pct_max`` percentiles of ``image``, or ``None`` if unset or equal."""
    if pct_min is None or pct_max is None:
        return None
    p_min = float(np.percentile(image, float(pct_min)))
    p_max = float(np.percentile(image, float(pct_max)))
    if np.isfinite(p_min) and np.isfinite(p_max) and p_max > p_min:
        return p_min, p_max
    return None


def _scale_to_unit(image: np.ndarray, scale_min: float, scale_max: float) -> np.ndarray:
    """Map ``[scale_min, scale_max]`` linearly to ``[0, 1]``, clipped, as float32."""
    if scale_max > scale_min:
        return np.clip((image.astype(np.float32) - scale_min) / (scale_max - scale_min), 0.0, 1.0)
    return np.zeros_like(image, dtype=np.float32)


def _hard_floor_mask(smoothed: np.ndarray, hard_floor: Optional[float]) -> Tuple[np.ndarray, Optional[float]]:
    """Pixels dark enough to be damage (``smoothed / 255 <= hard_floor``), or all pixels without a floor."""
    if hard_floor is None:
        return np.ones_like(smoothed, dtype=bool), None
    hard_floor = float(hard_floor)
    return smoothed.astype(np.float32) / 255.0 <= hard_floor, hard_floor


def _minmax_otsu_threshold(image: np.ndarray, window: Tuple[int, int]) -> float:
    """Otsu threshold of ``image`` after a max and a min filter over ``window``."""
    size = (max(1, int(window[0])), max(1, int(window[1])))
    filtered = ndi.minimum_filter(ndi.maximum_filter(image, size=size), size=size)
    return float(threshold_otsu(filtered))


def _kmeans_split(image: np.ndarray, *, max_iter: int = 20, tol: float = 1e-2) -> Optional[float]:
    """Midpoint between the two k-means centroids of the values, or ``None`` if they can't be split."""
    values = np.asarray(image, dtype=np.float32).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    v_min = float(values.min())
    v_max = float(values.max())
    if v_max - v_min < 1e-3:
        return None

    centroids = np.array([v_min, v_max], dtype=np.float32)
    for _ in range(max_iter):
        # Ties go to the dark cluster.
        is_bright = np.abs(values - centroids[1]) < np.abs(values - centroids[0])
        new_centroids = centroids.copy()
        updated = False
        for idx, members in enumerate((values[~is_bright], values[is_bright])):
            if members.size == 0:
                new_centroids[idx] = v_min if idx == 0 else v_max
                continue
            candidate = float(members.mean())
            if abs(candidate - centroids[idx]) > tol:
                updated = True
            new_centroids[idx] = candidate
        centroids = new_centroids
        if not updated:
            break

    dark, bright = sorted(float(value) for value in centroids)
    if abs(bright - dark) < 1e-3:
        return None
    return float(0.5 * (dark + bright))


def _resolve_hard_floor_ratio(value: Any) -> Optional[float]:
    """Return ``hard_floor`` as a ratio; old 8-bit values (> 1) are divided by 255 with a warning."""
    if value is None:
        return None
    hard_floor = float(value)
    if hard_floor > 1.0:
        warnings.warn(
            "hard_floor > 1.0 detected; interpreting as 8-bit intensity and "
            "converting to ratio (value / 255). Use ratio values such as 0.90.",
            DeprecationWarning,
            stacklevel=3,
        )
        hard_floor = hard_floor / 255.0
    return float(hard_floor)


def _resolve_pair(value: Any, *, name: str, caster: Callable[[Any], Any]) -> Tuple[Any, Any]:
    """Cast a two-value parameter such as a window size, raising if it doesn't have two values."""
    pair = tuple(value)
    if len(pair) != 2:
        raise ValueError(f"{name} must be a tuple/list with 2 values.")
    return (caster(pair[0]), caster(pair[1]))


def _resolve_optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def _resolve_pos_scale(value: Any) -> Optional[float]:
    return None if value is None else max(0.0, float(value))


def _fetch_region_override_stacks(
    owner: Any,
    *,
    domain: str,
    max_frames: Optional[int],
    extra_frame_counts: Optional[Dict[str, int]] = None,
) -> Tuple[Sequence[np.ndarray], Sequence[np.ndarray], Sequence[np.ndarray], Optional[Sequence[np.ndarray]], int]:
    """Return the upper, middle, lower and full stacks and the number of frames to process.

    All stacks (and ``extra_frame_counts``, e.g. the crack frames) must
    have the same number of frames; ``max_frames`` caps it.
    """
    stacks = owner._select_stacks()
    upper_stack = stacks.get("upper")
    middle_stack = stacks.get("middle")
    lower_stack = stacks.get("lower")
    raw_stack = getattr(owner.specimen, "image_stack_full", None)

    if upper_stack is None or middle_stack is None or lower_stack is None:
        raise ValueError("Region override mode requires upper/middle/lower stacks to be available.")

    counts = {"upper": len(upper_stack), "middle": len(middle_stack), "lower": len(lower_stack)}
    if extra_frame_counts:
        counts.update(extra_frame_counts)
    total_frames = _require_equal_frame_counts(counts)
    if max_frames is not None:
        total_frames = min(total_frames, max(0, int(max_frames)))
    if total_frames <= 0:
        raise ValueError(f"No frames available for region-overridden {domain} detection.")

    return upper_stack, middle_stack, lower_stack, raw_stack, total_frames


def _region_override_raw_frame(
    raw_stack: Optional[Sequence[np.ndarray]],
    idx: int,
    target_shape: Tuple[int, int],
    upper_frame: np.ndarray,
    middle_frame: np.ndarray,
    lower_frame: np.ndarray,
) -> np.ndarray:
    """Full frame ``idx`` for display, or the three region frames stacked if its shape doesn't match."""
    if raw_stack is not None and idx < len(raw_stack):
        raw_candidate = _ensure_uint8(raw_stack[idx])
        if raw_candidate.shape[:2] == target_shape:
            return raw_candidate
    return np.vstack([_ensure_uint8(upper_frame), _ensure_uint8(middle_frame), _ensure_uint8(lower_frame)])


def _auto_preprocess_cache_paths(
    owner: Any,
    *,
    save_overlays: bool,
    max_frames: Optional[int],
    progress: bool,
    key_prefix: str,
    reference_mode: str = "static",
) -> List[Path]:
    """Preprocess the full stack when the caller passed no processed frames.

    Preprocess previews are saved whenever overlays are requested.
    """
    stack = getattr(owner.specimen, "image_stack_full", None)
    if stack is None:
        raise ValueError("Specimen has no full image stack to preprocess.")
    restore_preprocess_outputs = None
    if save_overlays and not owner.save_preprocess_outputs:
        restore_preprocess_outputs = owner.save_preprocess_outputs
        owner.save_preprocess_outputs = True
    try:
        return owner.preprocess_stack_to_disk(
            stack,
            key=f"{key_prefix}_{_result_key_token(owner.interface.name)}",
            max_frames=max_frames,
            cache_dirname="Preprocessor_cache",
            reference_mode=reference_mode,
            progress=progress,
        )["cache_paths"]
    finally:
        if restore_preprocess_outputs is not None:
            owner.save_preprocess_outputs = restore_preprocess_outputs


def _crack_input_frame_count(cracks: Any) -> int:
    """Number of frames in the crack input (a list, an array or a :func:`crack_analysis` result)."""
    if isinstance(cracks, Mapping):
        if not cracks:
            raise ValueError("Crack analysis results must contain at least one orientation.")

        orientation_counts: Dict[str, int] = {}
        for orientation, payload in cracks.items():
            if not isinstance(payload, Mapping) or "cracks" not in payload:
                raise ValueError(
                    f"Crack analysis result '{orientation}' must be a mapping "
                    "containing a 'cracks' field."
                )
            orientation_counts[str(orientation)] = _crack_input_frame_count(payload["cracks"])

        unique_counts = set(orientation_counts.values())
        if len(unique_counts) != 1:
            details = ", ".join(f"{orientation}={count}" for orientation, count in orientation_counts.items())
            raise ValueError(
                "Crack analysis orientations must have equal frame counts; "
                f"received {details}."
            )

        frame_count = next(iter(unique_counts))
        if frame_count <= 0:
            raise ValueError("Crack analysis results must contain at least one crack frame.")
        return frame_count

    if isinstance(cracks, np.ndarray):
        if cracks.ndim == 4 and cracks.shape[-2:] == (2, 2):
            return int(cracks.shape[0])
        if cracks.ndim == 3 and cracks.shape[-2:] == (2, 2):
            return 1
        if cracks.ndim == 1 and cracks.dtype == object:
            return int(len(cracks))

    try:
        return int(len(cracks))
    except TypeError as exc:
        raise TypeError(
            "cracks must be a per-frame sequence, NumPy array, or "
            "orientation-keyed crack_analysis result."
        ) from exc


def _require_equal_frame_counts(counts: Dict[str, int]) -> int:
    """Return the frame count shared by all inputs, or raise naming the ones that differ."""
    unique_counts = set(counts.values())
    if len(unique_counts) > 1:
        details = ", ".join(f"{name}={count}" for name, count in counts.items())
        raise ValueError(
            f"Frame count mismatch between inputs: {details}. Refusing to silently "
            "truncate to the shortest input; verify that all regions/crack input "
            "were produced from the same set of frames."
        )
    return next(iter(unique_counts))


def _coerce_cracks_by_frame(cracks: Any, frame_count: int) -> List[Any]:
    """Return the crack input as a list with one entry per frame.

    Accepts a list per frame, a ragged object array, a
    ``(frames, cracks, 2, 2)`` array, a ``(cracks, 2, 2)`` array for a
    single frame, or a :func:`crack_analysis` result (all orientations
    are merged).
    """
    if isinstance(cracks, Mapping):
        analysis_frame_count = _crack_input_frame_count(cracks)
        orientation_frames = [
            _coerce_cracks_by_frame(payload["cracks"], analysis_frame_count)
            for payload in cracks.values()
        ]
        frame_cracks = Specimen.join_cracks(*orientation_frames)
    elif isinstance(cracks, np.ndarray):
        if cracks.ndim == 4 and cracks.shape[-2:] == (2, 2):
            frame_cracks = list(cracks)
        elif cracks.ndim == 3 and cracks.shape[-2:] == (2, 2):
            if frame_count == 1:
                frame_cracks = [cracks]
            elif cracks.shape[0] == frame_count:
                frame_cracks = [cracks[idx : idx + 1] for idx in range(frame_count)]
            else:
                raise ValueError(
                    "A (cracks, 2, 2) array is only unambiguous for one frame; "
                    "for multiple frames provide a per-frame sequence or an "
                    "array shaped (frames, cracks, 2, 2)."
                )
        elif cracks.ndim == 1 and cracks.dtype == object:
            frame_cracks = list(cracks)
        else:
            raise ValueError(
                "Unsupported cracks array shape. Expected an object array by frame, "
                "(frames, cracks, 2, 2), or (cracks, 2, 2) for one frame."
            )
    else:
        try:
            frame_cracks = list(cracks)
        except TypeError as exc:
            raise TypeError("cracks must be a per-frame sequence or NumPy array.") from exc

    if len(frame_cracks) != frame_count:
        raise ValueError(
            f"Crack input has {len(frame_cracks)} frame(s) but {frame_count} frame(s) "
            "were expected from the image stack being processed; refusing to silently "
            "truncate or pad with empty frames. Verify that crack detection and "
            "delamination detection were run on the same set of frames."
        )
    return frame_cracks
