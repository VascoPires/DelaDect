"""Frame preprocessing: minimum-history clamp, normalization by a
reference frame, and the on-disk preprocess cache.

Provides :class:`PreprocessingMixin` for :class:`DelaminationDetector`.
"""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import warnings
from typing import Any, Deque, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np

from ._common import _Progress, _ensure_uint8, _frame_to_float

PREPROCESS_MANIFEST_FILENAME = "preprocess_manifest.json"


def _as_scalar(value: Any, default: Any = None) -> Any:
    """Turn numpy scalars and one-element arrays into Python scalars."""
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
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
    """Frame range ``[start, end)`` used as the reference of one frame."""
    idx = max(0, int(frame_idx))
    window = max(1, int(reference_window))
    skip = max(0, int(reference_skip))

    if reference_mode == "rolling_median":
        end_idx = max(0, idx - skip)
        return max(0, end_idx - window), end_idx
    if reference_mode == "static" and idx >= skip:
        return skip, skip + 1
    return idx, idx + 1


def _reference_anchor_index(frame_idx: int, *, start_idx: int, end_idx: int, policy: str) -> int:
    """Frame of a ``[start, end)`` reference window to take the cracks from, according to ``policy``."""
    idx = max(0, int(frame_idx))
    start = max(0, int(start_idx))
    end = max(start, int(end_idx))

    if policy == "current" or end <= start:
        return idx
    if policy == "reference_latest":
        return end - 1
    if policy == "reference_midpoint":
        return start + (end - start - 1) // 2
    return idx


def _build_frame_reference_metadata(
    frame_idx: int,
    *,
    reference_mode: str,
    reference_window: int,
    reference_skip: int,
) -> Dict[str, Any]:
    """Reference window metadata stored with each cached frame, used to align cracks with frames."""
    start_idx, end_idx = _reference_window_bounds(
        frame_idx,
        reference_mode=reference_mode,
        reference_window=reference_window,
        reference_skip=reference_skip,
    )
    return {
        "ref_start_idx": int(start_idx),
        "ref_end_idx": int(end_idx),
        "ref_anchor_idx": _reference_anchor_index(
            frame_idx, start_idx=start_idx, end_idx=end_idx, policy="reference_midpoint"
        ),
        "reference_mode": str(reference_mode),
        "reference_window": int(reference_window),
        "reference_skip": int(reference_skip),
    }


def _extract_preprocess_frame_metadata(payload: Any, frame_idx: int) -> Dict[str, Any]:
    """Read the reference metadata of a cached frame, with defaults for missing or bad values."""

    def read(key: str, cast_to: Any, default: Any) -> Any:
        try:
            return cast_to(_as_scalar(payload[key], default))
        except Exception:
            return default

    meta = _build_frame_reference_metadata(
        frame_idx,
        reference_mode=read("reference_mode", str, "static"),
        reference_window=read("reference_window", int, 1),
        reference_skip=read("reference_skip", int, 0),
    )
    for key in ("ref_start_idx", "ref_end_idx", "ref_anchor_idx"):
        meta[key] = read(key, int, meta[key])
    return meta


class _ReferenceBaseline:
    """Normalization baseline for each frame of a stack, in order.

    ``"static"`` uses frame ``reference_skip`` for every later frame.
    ``"rolling_median"`` uses the median of the ``reference_window``
    frames that end ``reference_skip`` frames before the current one.
    Frames without enough history are their own baseline; other modes
    give no baseline.
    """

    def __init__(self, reference_mode: str, reference_window: int, reference_skip: int) -> None:
        self.mode = reference_mode
        self.window = reference_window
        self.skip = reference_skip
        self._history: Deque[np.ndarray] = deque(maxlen=reference_window + reference_skip + 1)
        self._static: Optional[np.ndarray] = None

    def next(self, idx: int, frame_float: np.ndarray) -> Optional[np.ndarray]:
        baseline: Optional[np.ndarray] = None
        if self.mode == "rolling_median":
            history = list(self._history)
            end = max(0, len(history) - self.skip)
            window_frames = history[max(0, end - self.window):end]
            baseline = np.median(np.stack(window_frames, axis=0), axis=0) if window_frames else frame_float
            self._history.append(frame_float)
        elif self.mode == "static":
            if self._static is None and idx >= self.skip:
                self._static = frame_float
            baseline = self._static if self._static is not None else frame_float
        return baseline


def _normalize_reference_frame(
    frame_uint8: np.ndarray,
    frame_float: np.ndarray,
    baseline_float: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return ``frame / baseline`` (clipped to 1) and the baseline, both as ``uint8``."""
    if baseline_float is None:
        return frame_uint8, None
    ratio = np.clip(frame_float / np.maximum(baseline_float, 1e-3), 0.0, 1.0)
    return (ratio * 255.0).astype(np.uint8), (baseline_float * 255.0).astype(np.uint8)


class _PreviewWriter:
    """Save raw / baseline / processed previews side by side, reusing one figure."""

    _DPI = 100

    def __init__(self, output_dir: Optional[Path], reference_mode: str) -> None:
        self.output_dir = output_dir
        self.reference_mode = reference_mode
        self._figure = None

    def _create_figure(self, image_shape: Tuple[int, ...]) -> None:
        height, width = int(image_shape[0]), int(image_shape[1])
        fig, axes = plt.subplots(1, 3, figsize=(3 * width / self._DPI, height / self._DPI),
                                 dpi=self._DPI, constrained_layout=True)
        baseline_title = "Rolling median baseline" if self.reference_mode == "rolling_median" else "Static baseline"
        placeholder = np.zeros((height, width))
        artists = {
            "raw": axes[0].imshow(placeholder, cmap="gray", vmin=0, vmax=255, aspect="equal"),
            "baseline": axes[1].imshow(placeholder, cmap="gray", vmin=0.0, vmax=1.0, aspect="equal"),
            "processed": axes[2].imshow(placeholder, cmap="gray", vmin=0, vmax=255, aspect="equal"),
        }
        for ax, title in zip(axes, ("Raw", baseline_title, "Processed")):
            ax.set_title(title)
            ax.axis("off")
        self._figure = (fig, axes, artists)

    def save(self, frame_idx: int, raw: np.ndarray, baseline: np.ndarray, processed: np.ndarray) -> None:
        if self.output_dir is None:
            return
        if self._figure is None:
            self._create_figure(raw.shape)
        fig, axes, artists = self._figure
        for key, frame in (("raw", raw), ("baseline", baseline), ("processed", processed)):
            artists[key].set_data(frame)
            height, width = frame.shape[:2]
            artists[key].axes.set_xlim(-0.5, width - 0.5)
            artists[key].axes.set_ylim(height - 0.5, -0.5)

        axes[0].set_xlabel(f"idx={frame_idx}")
        axes[1].set_xlabel("baseline")
        axes[2].set_xlabel("processed")
        fig.suptitle(f"Preprocessing - frame {frame_idx}", fontsize=12)
        fig.savefig(self.output_dir / f"preprocess_{frame_idx:04d}.png", dpi=fig.get_dpi())

    def close(self) -> None:
        if self._figure is not None:
            plt.close(self._figure[0])


class PreprocessingMixin:
    """Preprocessing and preprocess-cache methods of :class:`DelaminationDetector`."""

    def apply_minimum_history(
        self,
        stack: Optional[List[np.ndarray]],
        *,
        key: str,
        history_buffers: Dict[str, Any],
        mode: str = "running",
        window_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Clamp each frame to the darkest value seen so far at each pixel.

        ``"running"`` uses all previous frames, ``"rolling"`` the last
        ``window_size`` (default 10). ``history_buffers[key]`` keeps the
        history between calls, so a stack can be processed in chunks.

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

        processed: List[np.ndarray] = []
        if mode == "running":
            for frame in stack:
                history = history_buffers.get(key)
                history = frame.copy() if history is None else np.minimum(history, frame)
                history_buffers[key] = history
                processed.append(np.minimum(frame, history))
        else:
            buffer = history_buffers.get(key)
            if not isinstance(buffer, deque):
                buffer = deque(maxlen=effective_window)
            for frame in stack:
                buffer.append(frame)
                history_buffers[key] = buffer
                processed.append(np.minimum.reduce(list(buffer)))

        return {"frames": processed}

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
        """Preprocess a stack and save each frame to ``.npz``.

        Frames are clamped to their minimum history and divided by a
        reference frame. The cache lets detection run again without
        repeating this step.

        Parameters
        ----------
        stack:
            Raw frames.
        key:
            Cache name; frames go to ``<results>/<cache_dirname>/<key>/``.
        max_frames:
            Process only the first ``max_frames`` frames (at least 1).
        history_mode:
            ``"running"`` (all previous frames) or ``"rolling"`` minimum history.
        history_window_size:
            Window of the ``"rolling"`` history (default 10).
        reference_mode:
            ``"static"`` divides every frame by one early frame. Use it for
            :meth:`~EdgeDetector.detect_primary`,
            :meth:`~DiffuseDetector.diffuse_delamination` and
            :meth:`detect_both_delaminations`.

            ``"rolling_median"`` divides by the median of recent frames, so
            only new changes stand out. It is meant for the deeper interfaces
            of :meth:`~EdgeDetector.detect_edge_multi`.
        reference_window:
            Number of frames in the rolling median. With ``1``, frame ``n``
            is divided by frame ``n - reference_skip - 1``.
        reference_skip:
            Number of most recent frames left out of the rolling median.
            Frames without enough history are divided by themselves.
        cache_dirname:
            Cache folder under the specimen results.
        progress:
            Print progress.

        Returns
        -------
        dict[str, Any]
            ``{"cache_paths": list[pathlib.Path]}``, one file per frame.
        """
        cache_paths = self._preprocess_to_disk(
            stack,
            key=key,
            max_frames=max_frames,
            history_mode=history_mode,
            history_window_size=history_window_size,
            reference_mode=reference_mode,
            reference_window=reference_window,
            reference_skip=reference_skip,
            cache_dirname=cache_dirname,
            progress=progress,
            save_previews=self.save_preprocess_outputs,
        )
        return {"cache_paths": cache_paths}

    def _preprocess_to_disk(
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
        save_previews: bool = False,
    ) -> List[Path]:
        """:meth:`preprocess_stack_to_disk` with previews switched by ``save_previews``; returns the cache paths."""
        from ._frames import check_max_frames

        if stack is None:
            raise ValueError("A valid image stack is required for preprocessing.")
        if history_mode not in {"running", "rolling"}:
            raise ValueError("history_mode must be 'running' or 'rolling'.")
        max_frames = check_max_frames(max_frames)

        cache_dir = self.specimen.results_dir(cache_dirname, key)
        previews = _PreviewWriter(self._preview_dir(key) if save_previews else None, reference_mode)

        if hasattr(stack, "__len__") and hasattr(stack, "__getitem__"):
            frames = cast(Sequence[np.ndarray], stack)
        else:
            frames = list(stack)
        limit = len(frames) if max_frames is None else min(max_frames, len(frames))
        progress_log = _Progress("preprocess_stack", limit, progress)

        reference_window = max(1, int(reference_window))
        reference_skip = max(0, int(reference_skip))
        baselines = _ReferenceBaseline(reference_mode, reference_window, reference_skip)
        history: Optional[np.ndarray] = None
        history_buffer: Deque[np.ndarray] = deque(maxlen=history_window_size or 10)

        cache_paths: List[Path] = []
        for idx in range(limit):
            raw = _ensure_uint8(frames[idx])
            if not self.history_clamp:
                history_frame = raw
            elif history_mode == "running":
                history = raw if history is None else np.minimum(history, raw)
                history_frame = np.minimum(raw, history)
            else:
                history_buffer.append(raw)
                history_frame = np.minimum.reduce(list(history_buffer))

            frame_float = _frame_to_float(history_frame)
            baseline_float = baselines.next(idx, frame_float)
            processed, baseline_uint8 = _normalize_reference_frame(history_frame, frame_float, baseline_float)
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

            previews.save(idx, raw, frame_float if baseline_float is None else baseline_float, processed)
            progress_log.update(idx + 1)

        previews.close()

        manifest = {
            "version": 1,
            "frame_count": int(limit),
            "history_mode": str(history_mode),
            "history_window_size": None if history_window_size is None else int(history_window_size),
            "reference_mode": str(reference_mode),
            "reference_window": int(reference_window),
            "reference_skip": int(reference_skip),
        }
        (cache_dir / PREPROCESS_MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        progress_log.done()
        return cache_paths

    def iter_preprocessed_cache(self, cache_paths: List[Path]) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield ``(index, processed_frame)`` for each cached frame."""
        from ._frames import PreprocessedFrames

        return iter(PreprocessedFrames.from_cache(cache_paths))

    def iter_preprocessed_cache_with_metadata(
        self, cache_paths: List[Path]
    ) -> Iterator[Tuple[int, np.ndarray, Dict[str, Any]]]:
        """Yield ``(index, processed_frame, reference_metadata)`` for each cached frame."""
        from ._frames import PreprocessedFrames

        return PreprocessedFrames.from_cache(cache_paths).with_metadata()

    def _preview_dir(self, key: str) -> Path:
        """Folder for the preprocess previews of cache ``key``."""
        return self.specimen.results_dir(self.preprocess_outputs_dirname, str(key))
