"""Reference/history normalization, preprocess caching, and preview plots.

Provides :class:`PreprocessingMixin`, mixed into
:class:`~deladect.detection.delamination.core.DelaminationDetector`.
Depends only on :mod:`._common`.
"""


from __future__ import annotations

from collections import deque
import json
import logging
from pathlib import Path
import warnings
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import numpy as np


logger = logging.getLogger(__name__)

from ._common import (
    PREPROCESS_MANIFEST_FILENAME,
    _ensure_uint8,
    _frame_to_float,
    _progress_done,
    _progress_init,
    _progress_update,
)


def _preprocess_manifest_path(cache_dir: Path) -> Path:
    """Return manifest path for one preprocess cache directory."""
    return cache_dir / PREPROCESS_MANIFEST_FILENAME


def _write_preprocess_manifest(cache_dir: Path, manifest: Dict[str, Any]) -> None:
    """Persist cache-level preprocessing metadata to JSON."""
    manifest_path = _preprocess_manifest_path(cache_dir)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _as_scalar(value: Any, default: Any = None) -> Any:
    """Convert numpy scalar/array values to Python scalars safely."""
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        if value.shape == ():
            return value.item()
        if value.size == 1:
            return value.reshape(()).item()
        return value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _reference_window_bounds(
    frame_idx: int,
    *,
    reference_mode: str,
    reference_window: int,
    reference_skip: int,
) -> Tuple[int, int]:
    """Return [start, end) indices used as normalization reference for one frame."""
    idx = max(0, int(frame_idx))
    mode = str(reference_mode)
    window = max(1, int(reference_window))
    skip = max(0, int(reference_skip))

    if mode == "rolling_median":
        end_idx = max(0, idx - skip)
        start_idx = max(0, end_idx - window)
        return int(start_idx), int(end_idx)

    if mode == "static":
        if idx < skip:
            return idx, idx + 1
        return skip, skip + 1

    return idx, idx + 1


def _reference_anchor_index(
    frame_idx: int,
    *,
    start_idx: int,
    end_idx: int,
    policy: str,
) -> int:
    """Select one crack anchor index from a [start, end) reference window."""
    idx = max(0, int(frame_idx))
    start = max(0, int(start_idx))
    end = max(start, int(end_idx))
    policy_name = str(policy)

    if policy_name == "current":
        return idx
    if end <= start:
        return idx
    if policy_name == "reference_latest":
        return end - 1
    if policy_name == "reference_midpoint":
        return start + (end - start - 1) // 2
    return idx


def _build_frame_reference_metadata(
    frame_idx: int,
    *,
    reference_mode: str,
    reference_window: int,
    reference_skip: int,
) -> Dict[str, Any]:
    """Build per-frame preprocessing metadata used for crack-frame alignment."""
    start_idx, end_idx = _reference_window_bounds(
        frame_idx,
        reference_mode=reference_mode,
        reference_window=reference_window,
        reference_skip=reference_skip,
    )
    anchor_idx = _reference_anchor_index(
        frame_idx,
        start_idx=start_idx,
        end_idx=end_idx,
        policy="reference_midpoint",
    )
    return {
        "ref_start_idx": int(start_idx),
        "ref_end_idx": int(end_idx),
        "ref_anchor_idx": int(anchor_idx),
        "reference_mode": str(reference_mode),
        "reference_window": int(reference_window),
        "reference_skip": int(reference_skip),
    }


def _extract_preprocess_frame_metadata(payload: Any, frame_idx: int) -> Dict[str, Any]:
    """Read per-frame preprocess metadata from cached payload with safe fallbacks."""
    default_meta = _build_frame_reference_metadata(
        frame_idx,
        reference_mode="static",
        reference_window=1,
        reference_skip=0,
    )

    try:
        reference_mode = str(_as_scalar(payload["reference_mode"], default_meta["reference_mode"]))
    except Exception:
        reference_mode = str(default_meta["reference_mode"])
    try:
        reference_window = int(_as_scalar(payload["reference_window"], default_meta["reference_window"]))
    except Exception:
        reference_window = int(default_meta["reference_window"])
    try:
        reference_skip = int(_as_scalar(payload["reference_skip"], default_meta["reference_skip"]))
    except Exception:
        reference_skip = int(default_meta["reference_skip"])

    base_meta = _build_frame_reference_metadata(
        frame_idx,
        reference_mode=reference_mode,
        reference_window=reference_window,
        reference_skip=reference_skip,
    )

    for key in ("ref_start_idx", "ref_end_idx", "ref_anchor_idx"):
        try:
            base_meta[key] = int(_as_scalar(payload[key], base_meta[key]))
        except Exception:
            pass
    return base_meta


def _reference_settings_from_cache_paths(
    processed_cache_paths: Optional[Sequence[Path]],
) -> Dict[str, Any]:
    """Best-effort extraction of reference settings from cached preprocess payloads."""
    defaults = {
        "reference_mode": "static",
        "reference_window": 10,
        "reference_skip": 0,
    }
    if not processed_cache_paths:
        return defaults

    first_path = Path(processed_cache_paths[0])
    if not first_path.exists():
        return defaults

    try:
        with np.load(first_path, allow_pickle=False) as payload:
            meta = _extract_preprocess_frame_metadata(payload, 0)
    except Exception:
        return defaults

    return {
        "reference_mode": str(meta.get("reference_mode", defaults["reference_mode"])),
        "reference_window": max(1, int(meta.get("reference_window", defaults["reference_window"]))),
        "reference_skip": max(0, int(meta.get("reference_skip", defaults["reference_skip"]))),
    }


def _prepare_preprocess_figure(image_shape: Optional[Tuple[int, int]], reference_mode: str = "rolling_median"):
    """Create reusable matplotlib artists for preprocess triplet previews."""
    import matplotlib.pyplot as plt

    _DPI = 100
    if image_shape is not None and len(image_shape) >= 2:
        height, width = image_shape[:2]
        placeholder_shape = (int(height), int(width))
        figsize = (3 * int(width) / _DPI, int(height) / _DPI)
    else:
        placeholder_shape = (2, 2)
        figsize = plt.rcParams["figure.figsize"]

    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=_DPI, constrained_layout=True)
    axes = list(axes)

    baseline_title = "Rolling median baseline" if reference_mode == "rolling_median" else "Static baseline"
    im_raw = axes[0].imshow(np.zeros(placeholder_shape), cmap="gray", vmin=0, vmax=255, aspect="equal")
    axes[0].set_title("Raw")
    im_base = axes[1].imshow(np.zeros(placeholder_shape), cmap="gray", vmin=0.0, vmax=1.0, aspect="equal")
    axes[1].set_title(baseline_title)
    im_proc = axes[2].imshow(np.zeros(placeholder_shape), cmap="gray", vmin=0, vmax=255, aspect="equal")
    axes[2].set_title("Processed")
    for ax in axes:
        ax.axis("off")

    return (fig, axes, {"raw": im_raw, "baseline": im_base, "processed": im_proc})


def _update_preprocess_figure(
    plot_state,
    raw: np.ndarray,
    baseline: np.ndarray,
    processed: np.ndarray,
    frame_idx: int,
    save_path,
) -> None:
    """Update and save one preprocess preview figure."""
    fig, axes, artists = plot_state
    artists["raw"].set_data(raw)
    artists["baseline"].set_data(baseline)
    artists["processed"].set_data(processed)

    for key, frame in ("raw", raw), ("baseline", baseline), ("processed", processed):
        height, width = frame.shape[:2]
        ax = artists[key].axes
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)

    axes[0].set_xlabel(f"idx={frame_idx}")
    axes[1].set_xlabel("baseline")
    axes[2].set_xlabel("processed")
    fig.suptitle(f"Preprocessing - frame {frame_idx}", fontsize=12)
    fig.savefig(save_path, dpi=fig.get_dpi())


def _close_preprocess_figure(plot_state) -> None:
    """Close preprocess preview figure and release matplotlib resources."""
    import matplotlib.pyplot as plt

    fig, _, _ = plot_state
    plt.close(fig)



class PreprocessingMixin:
    """Reference-normalization and on-disk preprocess caching for :class:`DelaminationDetector`.

    Mixed into ``DelaminationDetector`` so ``detector.preprocess_stack_to_disk(...)``
    and friends keep working exactly as before the package split.
    """

    def apply_minimum_history(
        self,
        stack: Optional[List[np.ndarray]],
        *,
        key: str,
        history_buffers: Dict[str, Any],
        mode: str = "running",
        window_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Apply minimum history clamp across an entire stack.

        Returns
        -------
        dict[str, Any]
            ``{"frames": list[np.ndarray]}``.
        """
        if stack is None:
            raise ValueError("A valid image stack is required for minimum history processing.")
        if mode not in {"running", "rolling"}:
            raise ValueError("mode must be 'running' or 'rolling'.")

        effective_window = 10 if window_size is None else max(1, int(window_size))
        if mode == "rolling" and window_size is None and not self._notice_flags.get(key):
            warnings.warn(
                f"Using rolling minimum history with default window size N={effective_window}.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._notice_flags[key] = True

        buffer = history_buffers.get(key)
        if mode == "rolling" and (buffer is None or not isinstance(buffer, deque)):
            buffer = deque(maxlen=effective_window)

        processed: List[np.ndarray] = []
        for frame in stack:
            if mode == "running":
                history = history_buffers.get(key)
                if history is None:
                    history = frame.copy()
                else:
                    history = np.minimum(history, frame)
                history_buffers[key] = history
                processed.append(np.minimum(frame, history))
            else:
                if buffer is None:
                    buffer = deque(maxlen=effective_window)
                buffer.append(frame)
                history_buffers[key] = buffer
                processed.append(np.minimum.reduce(list(buffer)))

        return {"frames": processed}

    def apply_reference_normalization(
        self,
        stack: Optional[List[np.ndarray]],
        *,
        reference_mode: str = "static",
        reference_window: int = 10,
        reference_skip: int = 0,
        output_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Apply reference normalisation and return processed stacks.

        Returns
        -------
        dict[str, Any]
            ``{"processed_frames": list[np.ndarray], "baseline_frames": list[np.ndarray | None]}``.
        """
        if stack is None:
            raise ValueError("A valid image stack is required for reference normalisation.")

        raw_stack = [_ensure_uint8(frame) for frame in stack]
        processed_frames: List[np.ndarray] = []
        baseline_frames: List[Optional[np.ndarray]] = []

        plot_state = None
        output_dir = self._resolve_preprocess_output_dir(output_key)
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            image_shape = raw_stack[0].shape if raw_stack else None
            plot_state = _prepare_preprocess_figure(image_shape, reference_mode=reference_mode)

        reference_window = max(1, int(reference_window))
        reference_skip = max(0, int(reference_skip))

        baseline_static_float: Optional[np.ndarray] = None

        for idx, frame in enumerate(raw_stack):
            frame_float = _frame_to_float(frame)
            baseline_float: Optional[np.ndarray] = None

            if reference_mode == "rolling_median":
                end_idx = max(0, idx - reference_skip)
                start_idx = max(0, end_idx - reference_window)
                window_frames: List[np.ndarray] = []
                for ref_idx in range(start_idx, end_idx):
                    if ref_idx < 0 or ref_idx >= len(raw_stack):
                        continue
                    window_frames.append(_frame_to_float(raw_stack[ref_idx]))
                baseline_float = np.median(np.stack(window_frames, axis=0), axis=0) if window_frames else frame_float
            elif reference_mode == "static":
                if baseline_static_float is None and idx >= reference_skip:
                    baseline_static_float = frame_float
                baseline_float = baseline_static_float if baseline_static_float is not None else frame_float

            processed, baseline_uint8 = self._normalize_reference_frame(
                frame_uint8=frame,
                frame_float=frame_float,
                baseline_float=baseline_float,
            )

            processed_frames.append(processed)
            baseline_frames.append(baseline_uint8)

            self._maybe_save_preprocess_plot(
                plot_state=plot_state,
                output_dir=output_dir,
                frame_idx=idx,
                raw_frame=frame,
                frame_float=frame_float,
                baseline_float=baseline_float,
                processed=processed,
            )

        if plot_state is not None:
            _close_preprocess_figure(plot_state)

        return {"processed_frames": processed_frames, "baseline_frames": baseline_frames}

    def normalize_delamination_stack(
        self,
        stack: Optional[List[np.ndarray]],
        *,
        key: str,
        history_buffers: Dict[str, Any],
        history_mode: str = "running",
        history_window_size: Optional[int] = None,
        reference_mode: str = "static",
        reference_window: int = 10,
        reference_skip: int = 0,
    ) -> Dict[str, Any]:
        """Apply minimum history then reference normalization, returning processed frames only.

        Returns
        -------
        dict[str, Any]
            ``{"frames": list[np.ndarray]}``.
        """
        if stack is None:
            raise ValueError("A valid image stack is required for delamination normalization.")

        history_stack = (
            self.apply_minimum_history(
                stack,
                key=key,
                history_buffers=history_buffers,
                mode=history_mode,
                window_size=history_window_size,
            )["frames"]
            if self.history_clamp
            else [_ensure_uint8(frame) for frame in stack]
        )

        normalization_result = self.apply_reference_normalization(
            history_stack,
            reference_mode=reference_mode,
            reference_window=reference_window,
            reference_skip=reference_skip,
            output_key=key,
        )
        return {"frames": normalization_result["processed_frames"]}

    def preprocess_stack_to_disk(
        self,
        stack: Optional[Iterable[np.ndarray]],
        *,
        key: str,
        max_frames: Optional[int] = None,
        history_mode: str = "running",
        history_window_size: Optional[int] = None,
        reference_mode: str = "static",
        reference_window: int = 10,
        reference_skip: int = 0,
        cache_dirname: str = "Preprocessor_cache",
        progress: bool = False,
    ) -> Dict[str, Any]:
        """Preprocess a stack and persist processed/baseline frames to ``.npz``.

        This method runs the same normalization stack used by detectors, but stores
        each frame payload on disk so downstream detection can be repeated without
        re-running preprocessing.

        Parameters
        ----------
        stack:
            Raw grayscale-compatible frames.
        key:
            Cache key used to build ``<results>/<cache_dirname>/<key>/``.
        max_frames:
            Optional cap on number of frames written.
        history_mode:
            ``"running"`` minimum history or ``"rolling"`` minimum history.
        history_window_size:
            Rolling history window size when ``history_mode="rolling"``.
        reference_mode:
            Reference normalization mode.  Two options are supported:

            ``"static"``
                Use a fixed early-frame baseline for the entire stack.  This is the
                standard mode for all general delamination detection
                (:meth:`detect_primary`, :meth:`detect_both_delaminations`,
                :meth:`~DelaminationDetector.detect_diffuse_delamination`).

            ``"rolling_median"``
                Use a rolling median of recent frames as the reference.  Reserved
                for multi-interface edge detection (:meth:`detect_edge_multi`),
                where the propagating damage front requires an adaptive baseline.
                Do not use this mode with single-interface detection methods.

        reference_window:
            Reference window size for rolling median.

            Tip: ``reference_mode="rolling_median"`` with ``reference_window=1``
            behaves like a single-frame rolling reference (legacy ``rolling``-style
            baseline): after warmup, frame ``n`` is normalized against roughly
            ``n - (reference_skip + 1)``.
        reference_skip:
            Number of newest prior frames skipped from rolling reference.

            With ``reference_window=1`` this becomes an explicit lag control:

            - ``reference_skip=0`` -> previous frame (``n-1``)
            - ``reference_skip=1`` -> two frames behind (``n-2``)
            - ``reference_skip=2`` -> three frames behind (``n-3``)

            Early frames with insufficient history fall back to the current frame.
        cache_dirname:
            Root cache folder under specimen results.

        Returns
        -------
        dict[str, Any]
            ``{"cache_paths": list[pathlib.Path]}`` — ordered list of written ``.npz`` paths.
        """
        if stack is None:
            raise ValueError("A valid image stack is required for preprocessing.")
        if history_mode not in {"running", "rolling"}:
            raise ValueError("history_mode must be 'running' or 'rolling'.")

        cache_dir = self._resolve_preprocess_cache_dir(cache_dirname, key)
        cache_dir.mkdir(parents=True, exist_ok=True)

        output_dir = self._resolve_preprocess_output_dir(key)
        plot_state = None

        stack_seq: Sequence[np.ndarray]
        if hasattr(stack, "__len__") and hasattr(stack, "__getitem__"):
            stack_seq = cast(Sequence[np.ndarray], stack)
        else:
            stack_seq = list(stack)
        total_frames = len(stack_seq)
        limit = total_frames if max_frames is None else min(max_frames, total_frames)
        progress_state = _progress_init("preprocess_stack", limit, progress)

        history: Optional[np.ndarray] = None
        history_buffer: Optional[Deque[np.ndarray]] = None

        reference_window = max(1, int(reference_window))
        reference_skip = max(0, int(reference_skip))
        ref_buffer: Deque[np.ndarray] = deque(maxlen=reference_window + reference_skip + 1)
        baseline_static_float: Optional[np.ndarray] = None

        cache_paths: List[Path] = []

        for idx in range(limit):
            raw = _ensure_uint8(stack_seq[idx])
            if self.history_clamp:
                if history_mode == "running":
                    history = raw if history is None else np.minimum(history, raw)
                    history_frame = np.minimum(raw, history)
                else:
                    if history_buffer is None:
                        history_buffer = deque(maxlen=history_window_size or 10)
                    history_buffer.append(raw)
                    history_frame = np.minimum.reduce(list(history_buffer))
            else:
                history_frame = raw

            frame_float = _frame_to_float(history_frame)
            baseline_float: Optional[np.ndarray] = None
            if reference_mode == "rolling_median":
                buffer_list = list(ref_buffer)
                end = max(0, len(buffer_list) - reference_skip)
                start = max(0, end - reference_window)
                window_frames = buffer_list[start:end]
                baseline_float = np.median(np.stack(window_frames, axis=0), axis=0) if window_frames else frame_float
            elif reference_mode == "static":
                if baseline_static_float is None and idx >= reference_skip:
                    baseline_static_float = frame_float
                baseline_float = baseline_static_float if baseline_static_float is not None else frame_float

            processed, baseline_uint8 = self._normalize_reference_frame(
                frame_uint8=history_frame,
                frame_float=frame_float,
                baseline_float=baseline_float,
            )

            frame_meta = _build_frame_reference_metadata(
                idx,
                reference_mode=reference_mode,
                reference_window=reference_window,
                reference_skip=reference_skip,
            )

            cache_path = cache_dir / f"preprocess_{idx:04d}.npz"
            np.savez_compressed(
                cache_path,
                processed=processed,
                baseline=baseline_uint8 if baseline_uint8 is not None else np.array([]),
                ref_start_idx=np.int32(frame_meta["ref_start_idx"]),
                ref_end_idx=np.int32(frame_meta["ref_end_idx"]),
                ref_anchor_idx=np.int32(frame_meta["ref_anchor_idx"]),
                reference_mode=np.array(frame_meta["reference_mode"]),
                reference_window=np.int32(frame_meta["reference_window"]),
                reference_skip=np.int32(frame_meta["reference_skip"]),
                history_mode=np.array(str(history_mode)),
                history_window_size=np.int32(-1 if history_window_size is None else int(history_window_size)),
            )
            cache_paths.append(cache_path)

            if output_dir is not None and plot_state is None:
                plot_state = _prepare_preprocess_figure(raw.shape, reference_mode=reference_mode)
            self._maybe_save_preprocess_plot(
                plot_state=plot_state,
                output_dir=output_dir,
                frame_idx=idx,
                raw_frame=raw,
                frame_float=frame_float,
                baseline_float=baseline_float,
                processed=processed,
            )

            if reference_mode == "rolling_median":
                ref_buffer.append(frame_float)

            _progress_update("preprocess_stack", idx + 1, limit, progress_state)

        if plot_state is not None:
            _close_preprocess_figure(plot_state)

        manifest = {
            "version": 1,
            "frame_count": int(limit),
            "history_mode": str(history_mode),
            "history_window_size": None if history_window_size is None else int(history_window_size),
            "reference_mode": str(reference_mode),
            "reference_window": int(reference_window),
            "reference_skip": int(reference_skip),
        }
        _write_preprocess_manifest(cache_dir, manifest)

        _progress_done("preprocess_stack", limit, progress)

        return {"cache_paths": cache_paths}

    def iter_preprocessed_cache(self, cache_paths: List[Path]):
        """Yield (index, processed_frame) from cached preprocess frames."""
        for idx, path in enumerate(cache_paths):
            with np.load(path, allow_pickle=False) as payload:
                processed = payload["processed"]
            yield idx, processed

    def iter_preprocessed_cache_with_metadata(self, cache_paths: List[Path]):
        """Yield (index, processed_frame, metadata) from cached preprocess frames."""
        for idx, path in enumerate(cache_paths):
            with np.load(path, allow_pickle=False) as payload:
                processed = payload["processed"]
                metadata = _extract_preprocess_frame_metadata(payload, idx)
            yield idx, processed, metadata

    def _resolve_preprocess_cache_dir(self, cache_dirname: str, key: str) -> Path:
        """Resolve cache output directory for a preprocessing run key."""
        return self.specimen.results_dir(cache_dirname, key)

    def _resolve_preprocess_output_dir(self, output_key: Optional[str]) -> Optional[Path]:
        """Return preprocess preview directory when preview export is enabled."""
        if not self.save_preprocess_outputs:
            return None
        parts = [self.preprocess_outputs_dirname]
        if output_key:
            parts.append(str(output_key))
        return self.specimen.results_dir(*parts)

    def _normalize_reference_frame(
        self,
        *,
        frame_uint8: np.ndarray,
        frame_float: np.ndarray,
        baseline_float: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply ratio-based normalization against a baseline reference frame."""
        if baseline_float is None:
            return frame_uint8, None
        denominator = np.maximum(baseline_float, 1e-3)
        ratio = np.clip(frame_float / denominator, 0.0, 1.0)
        processed = (ratio * 255.0).astype(np.uint8)
        baseline_uint8 = (baseline_float * 255.0).astype(np.uint8)
        return processed, baseline_uint8

    def _maybe_save_preprocess_plot(
        self,
        *,
        plot_state,
        output_dir: Optional[Path],
        frame_idx: int,
        raw_frame: np.ndarray,
        frame_float: np.ndarray,
        baseline_float: Optional[np.ndarray],
        processed: np.ndarray,
    ) -> None:
        """Save one preprocessing preview panel when plotting is enabled."""
        if plot_state is None or output_dir is None:
            return
        baseline_display = baseline_float if baseline_float is not None else frame_float
        save_path = output_dir / f"preprocess_{frame_idx:04d}.png"
        _update_preprocess_figure(
            plot_state,
            raw_frame,
            baseline_display,
            processed,
            frame_idx,
            save_path,
        )
