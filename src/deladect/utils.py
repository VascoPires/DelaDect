"""General utilities shared across crack and specimen workflows."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np


def crack_mid_point(crack: Sequence[Sequence[float]]) -> List[Optional[float]]:
    """Return midpoint of a crack segment ``[[row0, col0], [row1, col1]]``."""
    crack_array = np.asarray(crack, dtype=float)
    if crack_array.shape != (2, 2):
        return [None, None]
    mid_row = float(crack_array[:, 0].mean())
    mid_col = float(crack_array[:, 1].mean())
    return [mid_row, mid_col]


def crack_length(crack: Sequence[Sequence[float]]) -> float:
    """Return the Euclidean length of a crack segment."""
    if crack is None or len(crack) != 2:
        return 0.0
    (row0, col0), (row1, col1) = crack
    return float(np.hypot(float(row1) - float(row0), float(col1) - float(col0)))


def crack_px_mm(crack_list: Sequence[Sequence[Sequence[float]]], scale: float) -> List[List[List[float]]]:
    """Scale crack coordinates from pixels to millimetres.

    Input and output coordinates use ``[row, col]`` ordering.
    """
    if not crack_list or scale == 0:
        return [[[float(row), float(col)] for row, col in segment] for segment in crack_list]
    scaled: List[List[List[float]]] = []
    for segment in crack_list:
        scaled.append([[row / scale, col / scale] for row, col in segment])
    return scaled


__all__ = [
    "crack_length",
    "crack_mid_point",
    "crack_px_mm",
]
