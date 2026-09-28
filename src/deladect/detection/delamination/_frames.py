"""Preprocessed frames, read from the preprocess cache or held in memory.

Every detection method gets its frames through :func:`resolve_frames`, so
the checks on ``processed_cache_paths`` / ``processed_stack``, automatic
preprocessing and ``max_frames`` are done in one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ._common import _result_key_token
from ._preprocess import _extract_preprocess_frame_metadata

if TYPE_CHECKING:
    from .core import DelaminationDetector

DEFAULT_REFERENCE = {"reference_mode": "static", "reference_window": 10, "reference_skip": 0}


def check_max_frames(max_frames: Optional[int]) -> Optional[int]:
    """Return ``max_frames`` as an int, or ``None`` for all frames. Raises if it is below 1."""
    if max_frames is None:
        return None
    value = int(max_frames)
    if value < 1:
        raise ValueError(f"max_frames must be at least 1 (or None for all frames); got {max_frames!r}.")
    return value


class PreprocessedFrames:
    """A sequence of preprocessed frames.

    Frames come either from cache files written by
    :meth:`~deladect.detection.DelaminationDetector.preprocess_stack_to_disk`
    or from arrays. Only cache frames carry reference metadata (the frames
    each one was normalized against); for arrays it is ``None``.
    """

    def __init__(
        self,
        *,
        cache_paths: Optional[Sequence[Path]] = None,
        arrays: Optional[Sequence[np.ndarray]] = None,
        count: Optional[int] = None,
        crop: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        if (cache_paths is None) == (arrays is None):
            raise ValueError("Give exactly one of cache_paths or arrays.")
        self.cache_paths = None if cache_paths is None else list(cache_paths)
        self._arrays = arrays
        available = len(self.cache_paths) if self.cache_paths is not None else len(arrays)
        self._count = available if count is None else min(available, count)
        self._crop = crop

    @classmethod
    def from_cache(cls, cache_paths: Sequence[Path]) -> "PreprocessedFrames":
        return cls(cache_paths=cache_paths)

    @classmethod
    def from_arrays(cls, arrays: Sequence[np.ndarray]) -> "PreprocessedFrames":
        return cls(arrays=arrays)

    def __len__(self) -> int:
        return self._count

    def _derive(self, *, count: Optional[int] = None, crop: Optional[Callable] = None) -> "PreprocessedFrames":
        return PreprocessedFrames(
            cache_paths=self.cache_paths,
            arrays=self._arrays,
            count=self._count if count is None else min(self._count, count),
            crop=crop if crop is not None else self._crop,
        )

    def limit(self, max_frames: Optional[int]) -> "PreprocessedFrames":
        """The first ``max_frames`` frames (all of them for ``None``)."""
        return self if max_frames is None else self._derive(count=max_frames)

    def rows(self, start: int, stop: int) -> "PreprocessedFrames":
        """The same frames, cut to rows ``start:stop``."""
        return self._derive(crop=lambda frame: frame[start:stop])

    def with_metadata(self) -> Iterator[Tuple[int, np.ndarray, Optional[Dict[str, Any]]]]:
        """Yield ``(index, frame, reference_metadata)``; the metadata is ``None`` for arrays."""
        for idx in range(self._count):
            if self.cache_paths is not None:
                with np.load(self.cache_paths[idx], allow_pickle=False) as payload:
                    frame = payload["processed"]
                    meta: Optional[Dict[str, Any]] = _extract_preprocess_frame_metadata(payload, idx)
            else:
                frame, meta = self._arrays[idx], None
            yield idx, (frame if self._crop is None else self._crop(frame)), meta

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield ``(index, frame)``."""
        for idx, frame, _ in self.with_metadata():
            yield idx, frame

    def to_list(self) -> List[np.ndarray]:
        return [frame for _, frame in self]

    def reference_settings(self) -> Dict[str, Any]:
        """Reference settings of the cache, read from its first frame (defaults for arrays)."""
        settings = dict(DEFAULT_REFERENCE)
        if not self.cache_paths or not Path(self.cache_paths[0]).exists():
            return settings
        try:
            with np.load(self.cache_paths[0], allow_pickle=False) as payload:
                meta = _extract_preprocess_frame_metadata(payload, 0)
        except Exception:
            return settings
        return {
            "reference_mode": str(meta["reference_mode"]),
            "reference_window": max(1, int(meta["reference_window"])),
            "reference_skip": max(0, int(meta["reference_skip"])),
        }


def resolve_frames(
    owner: "DelaminationDetector",
    *,
    processed_cache_paths: Optional[Sequence[Path]],
    processed_stack: Optional[Sequence[np.ndarray]],
    max_frames: Optional[int],
    auto_key: str,
    save_previews: bool = False,
    progress: bool = False,
    reference: Optional[Dict[str, Any]] = None,
) -> PreprocessedFrames:
    """Frames from the cache or arrays given, or from preprocessing the full stack.

    The full stack is preprocessed (with ``reference``, static by default)
    only when neither input is given, into the cache key
    ``<auto_key>_<interface>``. Preprocess previews are saved when
    ``save_previews`` or the detector's ``save_preprocess_outputs`` is set.
    The result is cut to ``max_frames``.
    """
    if processed_cache_paths and processed_stack:
        raise ValueError("Provide either processed_cache_paths or processed_stack, not both.")
    max_frames = check_max_frames(max_frames)

    if processed_stack is not None:
        frames = PreprocessedFrames.from_arrays(processed_stack)
    elif processed_cache_paths is not None:
        frames = PreprocessedFrames.from_cache(processed_cache_paths)
    else:
        stack = getattr(owner.specimen, "image_stack_full", None)
        if stack is None:
            raise ValueError("Specimen has no full image stack to preprocess.")
        reference = {**DEFAULT_REFERENCE, **(reference or {})}
        cache_paths = owner._preprocess_to_disk(
            stack,
            key=f"{auto_key}_{_result_key_token(owner.interface.name)}",
            max_frames=max_frames,
            reference_mode=str(reference["reference_mode"]),
            reference_window=int(reference["reference_window"]),
            reference_skip=int(reference["reference_skip"]),
            progress=progress,
            save_previews=save_previews or owner.save_preprocess_outputs,
        )
        frames = PreprocessedFrames.from_cache(cache_paths)
    return frames.limit(max_frames)


__all__ = ["PreprocessedFrames", "check_max_frames", "resolve_frames"]
