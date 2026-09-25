"""Helpers for crack segments, stored as ``[[row0, col0], [row1, col1]]``."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np


def crack_mid_point(crack: Sequence[Sequence[float]]) -> List[Optional[float]]:
    """Midpoint ``[row, col]`` of a segment, or ``[None, None]`` if it isn't a 2x2 array."""
    crack_array = np.asarray(crack, dtype=float)
    if crack_array.shape != (2, 2):
        return [None, None]
    return [float(crack_array[:, 0].mean()), float(crack_array[:, 1].mean())]


def crack_length(crack: Sequence[Sequence[float]]) -> float:
    """Length of a segment in pixels."""
    if crack is None or len(crack) != 2:
        return 0.0
    (row0, col0), (row1, col1) = crack
    return float(np.hypot(float(row1) - float(row0), float(col1) - float(col0)))


def draw_crack_segments(
    ax: Any,
    cracks: Optional[Sequence[np.ndarray]],
    *,
    color: Any,
    linewidth: float,
) -> None:
    """Draw each segment on ``ax`` as a line, skipping malformed ones."""
    if cracks is None:
        return
    for segment in cracks:
        try:
            arr = np.asarray(segment, dtype=float).reshape(-1, 2)
        except Exception:
            continue
        if arr.shape[0] < 2:
            continue
        (y0, x0), (y1, x1) = arr[:2]
        ax.plot((x0, x1), (y0, y1), color=color, linewidth=linewidth, linestyle="-")


__all__ = ["crack_length", "crack_mid_point", "draw_crack_segments"]
