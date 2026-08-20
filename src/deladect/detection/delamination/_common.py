"""Shared low-level primitives used across the delamination detection package.

No dependency on any other submodule in this package.
"""


from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from deladect.specimen import (
    Specimen,
    sanitize_path_token,
)

logger = logging.getLogger(__name__)

EDGE_OVERLAY_RGBA = (1.0, 0.0, 0.0, 0.35)
DIFFUSE_OVERLAY_RGBA = (0.0, 1.0, 0.0, 0.35)
MULTI_INTERFACE_DEFAULT_COLORS: Tuple[Tuple[float, float, float, float], ...] = (
    (0.89, 0.10, 0.11, 0.35),
    (0.12, 0.47, 0.71, 0.35),
    (0.20, 0.63, 0.17, 0.35),
    (1.00, 0.50, 0.05, 0.35),
    (0.58, 0.40, 0.74, 0.35),
    (0.55, 0.34, 0.29, 0.35),
    (0.89, 0.47, 0.76, 0.35),
    (0.50, 0.50, 0.50, 0.35),
)

PROGRESS_MILESTONES: Tuple[int, ...] = (25, 50, 75, 90)
PREPROCESS_MANIFEST_FILENAME = "preprocess_manifest.json"
DIFFUSE_CRACK_FRAME_POLICIES: Tuple[str, ...] = ("current", "reference_latest", "reference_midpoint")
CRACK_OVERLAY_RGBA: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.95)
CrackAnalysisMapping = Mapping[str, Mapping[str, Any]]
CrackInput = Union[Sequence[np.ndarray], CrackAnalysisMapping]


def _result_key_token(value: Any) -> str:
    """Convert a display label into one safe result-directory component."""
    return sanitize_path_token(value, fallback="interface")


def _progress_init(stage: str, total_frames: int, enabled: bool) -> Optional[Dict[int, bool]]:
    """Initialize milestone tracking and print start banner."""
    if not enabled:
        return None
    total = max(0, int(total_frames))
    print(f"[progress] {stage}: start ({total} frames)", flush=True)
    return {milestone: False for milestone in PROGRESS_MILESTONES}


def _progress_update(
    stage: str,
    completed_frames: int,
    total_frames: int,
    state: Optional[Dict[int, bool]],
) -> None:
    """Emit milestone progress updates at 25/50/75/90%."""
    if state is None:
        return
    total = max(1, int(total_frames))
    completed = max(0, int(completed_frames))
    percent = 100.0 * float(completed) / float(total)

    for milestone in PROGRESS_MILESTONES:
        already = bool(state.get(milestone, False))
        if not already and percent >= float(milestone):
            print(f"[progress] {stage}: {milestone}% ({completed}/{total})", flush=True)
            state[milestone] = True


def _progress_done(stage: str, total_frames: int, enabled: bool) -> None:
    """Emit completion banner for a progress-tracked stage."""
    if not enabled:
        return
    total = max(0, int(total_frames))
    print(f"[progress] {stage}: done ({total}/{total})", flush=True)


def _ensure_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert input frame to single-channel ``uint8``."""
    frame_float = frame.astype(np.float32)
    if frame_float.ndim == 3:
        frame_float = frame_float.mean(axis=2)
    if frame_float.max() <= 1.0:
        frame_float = frame_float * 255.0
    return np.clip(frame_float, 0, 255).astype(np.uint8)


def _frame_to_float(frame: np.ndarray) -> np.ndarray:
    """Convert input frame to single-channel ``float32`` in ``[0, 1]``."""
    frame_float = frame.astype(np.float32)
    if frame_float.ndim == 3:
        frame_float = frame_float.mean(axis=2)
    if frame_float.max() > 1.0:
        frame_float = frame_float / 255.0
    return frame_float


def _resolve_hard_floor_ratio(value: Any) -> Optional[float]:
    """Normalize ``hard_floor`` to ratio space.

    The preferred input is a normalized ratio (for example ``0.90``). Legacy
    8-bit values greater than ``1.0`` are interpreted as pixel intensities and
    converted via ``value / 255``.
    """
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
    """Validate and cast a 2-element tuple/list parameter (e.g. a window size).

    Shared by the edge/diffuse parameter resolvers for ``window_edge`` /
    ``window_diffuse`` / ``gaussian_filters``. Raises ``ValueError`` naming
    the parameter if it isn't exactly length 2.
    """
    pair = tuple(value)
    if len(pair) != 2:
        raise ValueError(f"{name} must be a tuple/list with 2 values.")
    return (caster(pair[0]), caster(pair[1]))


def _resolve_optional_float(value: Any) -> Optional[float]:
    """Return ``None`` unchanged, otherwise cast to ``float``."""
    return None if value is None else float(value)


def _resolve_pos_scale(value: Any) -> Optional[float]:
    """Return ``None`` unchanged, otherwise a non-negative ``float``."""
    return None if value is None else max(0.0, float(value))


def _fetch_region_override_stacks(
    owner: Any,
    *,
    domain: str,
    max_frames: Optional[int],
    extra_frame_counts: Optional[Dict[str, int]] = None,
) -> Tuple[Sequence[np.ndarray], Sequence[np.ndarray], Sequence[np.ndarray], Optional[Sequence[np.ndarray]], int]:
    """Fetch and validate upper/middle/lower stacks for a region-override detection path.

    Shared setup for ``EdgeDetector._detect_primary_region_overrides`` and
    ``DiffuseDetector._diffuse_delamination_region_overrides``: fetch the
    three region stacks plus the raw full-frame stack, require all three
    regions to be present, reconcile frame counts (optionally including
    caller-supplied extras such as a crack count), and clamp against
    ``max_frames``.
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
    """Return the raw display frame for a region-override overlay at ``idx``.

    Prefers a same-shaped slice of ``raw_stack`` (the true full-frame image);
    falls back to vertically stacking the three region frames when no raw
    stack is available or its shape doesn't match the mask being overlaid.
    """
    if raw_stack is not None and idx < len(raw_stack):
        raw_candidate = _ensure_uint8(raw_stack[idx])
        if raw_candidate.shape[:2] == target_shape:
            return raw_candidate
    return np.vstack(
        [
            _ensure_uint8(upper_frame),
            _ensure_uint8(middle_frame),
            _ensure_uint8(lower_frame),
        ]
    )


def _auto_preprocess_cache_paths(
    owner: Any,
    *,
    save_overlays: bool,
    max_frames: Optional[int],
    progress: bool,
    key_prefix: str,
    reference_mode: str = "static",
) -> List[Path]:
    """Auto-preprocess the specimen's full stack when no cache/stack was supplied.

    Shared by ``EdgeDetector.detect_primary`` and
    ``DiffuseDetector.diffuse_delamination``: resolves ``image_stack_full``,
    temporarily forces ``save_preprocess_outputs=True`` when overlays are
    requested and it wasn't already on, and restores it afterward regardless
    of outcome.
    """
    stacks = owner._select_stacks()
    stack = getattr(owner.specimen, "image_stack_full", None) or stacks.get("full")
    if stack is None:
        raise ValueError("Specimen has no full image stack to preprocess.")
    restore_preprocess_outputs = None
    if save_overlays and not owner.save_preprocess_outputs:
        restore_preprocess_outputs = owner.save_preprocess_outputs
        owner.save_preprocess_outputs = True
    try:
        auto_key = f"{key_prefix}_{_result_key_token(owner.interface.name)}"
        return owner.preprocess_stack_to_disk(
            stack,
            key=auto_key,
            max_frames=max_frames,
            cache_dirname="Preprocessor_cache",
            reference_mode=reference_mode,
            progress=progress,
        )["cache_paths"]
    finally:
        if restore_preprocess_outputs is not None:
            owner.save_preprocess_outputs = restore_preprocess_outputs


_PRIMARY_EDGE_DEBUG_KEYS: Tuple[str, ...] = (
    "smoothed",
    "constant_scaled",
    "closed",
    "threshold",
    "hard_floor_eff",
    "close_radius",
    "min_object_px",
    "binary",
    "binary_closed",
    "mask",
    "primary_edge_snapshot",
    "status",
)


def _build_primary_debug_payload(
    processed: np.ndarray,
    upper_result: Dict[str, Any],
    lower_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Build one frame's debug payload for primary edge detection.

    Shared by ``EdgeDetector.detect_primary`` and
    ``EdgeDetector._detect_primary_region_overrides``, which differ only in
    how ``processed`` itself is assembled (a direct frame copy vs. a stitched
    upper/middle/lower stack) -- both already pass in the finished array.
    """
    return {
        "processed": processed,
        "upper": {key: upper_result[key] for key in _PRIMARY_EDGE_DEBUG_KEYS},
        "lower": {key: lower_result[key] for key in _PRIMARY_EDGE_DEBUG_KEYS},
    }


def _normalize_rgba(color: Sequence[float], *, default_alpha: float = 0.35) -> Tuple[float, float, float, float]:
    """Normalize arbitrary color input to a clipped RGBA tuple."""
    values = [float(v) for v in color]
    if len(values) == 3:
        values.append(float(default_alpha))
    if len(values) != 4:
        return (1.0, 0.0, 0.0, float(default_alpha))
    return (
        float(np.clip(values[0], 0.0, 1.0)),
        float(np.clip(values[1], 0.0, 1.0)),
        float(np.clip(values[2], 0.0, 1.0)),
        float(np.clip(values[3], 0.0, 1.0)),
    )


def _rgba_close(
    left: Sequence[float],
    right: Sequence[float],
    *,
    tolerance: float = 1e-6,
) -> bool:
    """Return ``True`` when two RGBA colors are numerically close."""
    if len(left) != 4 or len(right) != 4:
        return False
    return all(abs(float(left[idx]) - float(right[idx])) <= tolerance for idx in range(4))


def _mask_px(mask: np.ndarray) -> int:
    """Count non-zero pixels in a boolean-like mask."""
    return int(np.count_nonzero(mask))


def _crack_input_frame_count(cracks: Any) -> int:
    """Return the source frame count for raw or orientation-keyed crack input."""
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
            details = ", ".join(
                f"{orientation}={count}"
                for orientation, count in orientation_counts.items()
            )
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
    """Return the shared frame count across named inputs, or raise naming the mismatch.

    Region-override detection reads several independently-loaded stacks (and
    optionally crack input); silently taking ``min(...)`` across them would
    mask a missing frame in one region as a shorter-but-successful run. This
    makes that mismatch a hard, named error instead.
    """
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
    """Return CrackDect-style crack output as a frame-indexed Python list.

    CrackDect outputs may be Python sequences, object arrays (for ragged frame
    results), or dense numeric arrays with shape ``(frames, cracks, 2, 2)``.
    A ``(cracks, 2, 2)`` array is also accepted for a single-frame stack.
    Orientation-keyed mappings returned by :func:`crack_analysis` are merged
    frame by frame across every orientation present in the mapping.
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
            frame_cracks = [cracks[idx] for idx in range(cracks.shape[0])]
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
