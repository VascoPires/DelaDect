"""Where edge and diffuse detection look in a frame.

Without region folders, edge detection splits each frame into its upper
and lower half and diffuse detection uses the whole frame. With region
folders (``path_upper_border``, ``path_middle``, ``path_lower_border``),
the regions are row ranges of the full frame: edge detection uses the
upper and lower rows and diffuse detection the middle rows. The region
images must be exact crops of the full frames; only their heights are
used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from deladect.specimen import Specimen


@dataclass(frozen=True)
class RegionLayout:
    """Row heights of the upper, middle and lower region, or all ``None`` for full-frame mode."""

    upper: Optional[int] = None
    middle: Optional[int] = None
    lower: Optional[int] = None
    width: Optional[int] = None

    @classmethod
    def from_specimen(cls, specimen: Specimen) -> "RegionLayout":
        stacks = [getattr(specimen, f"image_stack_{name}", None) for name in ("upper", "middle", "lower")]
        if not all(
            path is not None
            for path in (specimen.path_upper_border, specimen.path_middle, specimen.path_lower_border)
        ) or any(stack is None for stack in stacks):
            return cls()
        upper, middle, lower = (int(np.asarray(stack[0]).shape[0]) for stack in stacks)
        return cls(upper, middle, lower, int(np.asarray(stacks[1][0]).shape[1]))

    @property
    def region_mode(self) -> bool:
        return self.upper is not None

    def edge_halves(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """``(upper, lower, gap)``: the two edge parts of ``frame`` and the rows between them.

        ``lower`` is not flipped.
        """
        if not self.region_mode:
            split_row = frame.shape[0] // 2
            return frame[:split_row, :], frame[split_row:, :], 0
        lower_start = self.upper + self.middle
        return frame[: self.upper, :], frame[lower_start:, :], self.middle

    @staticmethod
    def join_edge_masks(upper: np.ndarray, lower_flipped: np.ndarray, gap: int = 0) -> np.ndarray:
        """Full-frame mask from the upper mask, ``gap`` empty rows and the flipped lower mask."""
        upper = np.asarray(upper, dtype=bool)
        lower = np.flipud(np.asarray(lower_flipped, dtype=bool))
        full = np.zeros((upper.shape[0] + gap + lower.shape[0], upper.shape[1]), dtype=bool)
        full[: upper.shape[0], :] = upper
        full[upper.shape[0] + gap :, :] = lower
        return full

    def diffuse_rows(self) -> Optional[Tuple[int, int]]:
        """Rows ``(start, stop)`` searched for diffuse damage, or ``None`` for the whole frame."""
        if not self.region_mode:
            return None
        return self.upper, self.upper + self.middle

    def diffuse_to_full(self, mask: np.ndarray) -> np.ndarray:
        """Place a mask of the diffuse rows into a full-frame mask."""
        if not self.region_mode:
            return mask
        full = np.zeros((self.upper + mask.shape[0] + self.lower, mask.shape[1]), dtype=bool)
        full[self.upper : self.upper + mask.shape[0], :] = mask
        return full

    def check_cracks_in_middle(self, cracks_by_frame: Sequence[Any]) -> None:
        """In region mode, raise if a crack of the last frame lies outside the middle region.

        Diffuse detection runs on the middle region, so the cracks must be in
        its coordinates (as :func:`~deladect.detection.crack_analysis` gives
        them in region mode). The last frame is checked because it has the
        most cracks.
        """
        if not self.region_mode or not cracks_by_frame:
            return
        last = cracks_by_frame[-1]
        if last is None or len(last) == 0:
            return
        points = np.asarray(last, dtype=float).reshape(-1, 2)
        rows, cols = points[:, 0], points[:, 1]
        if rows.min() < 0 or rows.max() > self.middle or cols.min() < 0 or cols.max() > self.width:
            raise ValueError(
                f"Cracks were found outside the middle region ({self.middle}x{self.width} px): rows "
                f"{rows.min():g} to {rows.max():g}, columns {cols.min():g} to {cols.max():g} in the last "
                "frame. In region mode the cracks must come from the middle region stack; check the "
                "cracks you passed in."
            )

    def clear_edge_rows(self, mask: np.ndarray) -> np.ndarray:
        """Copy of a full-frame mask with the upper and lower region rows cleared."""
        if not self.region_mode:
            return mask
        mask = mask.copy()
        mask[: self.upper, :] = False
        if self.lower > 0:
            mask[-self.lower :, :] = False
        return mask

    def cracks_for_display(
        self,
        cracks: Optional[Sequence[np.ndarray]],
        crack_coordinate_space: str,
    ) -> Optional[List[np.ndarray]]:
        """Crack segments in full-frame coordinates, for overlays.

        In region mode, cracks detected on the middle region
        (``crack_coordinate_space="middle"``) are moved down by the upper
        region height. Full-frame mode returns the cracks unchanged.
        """
        if cracks is None or not self.region_mode:
            return cracks
        prepared: List[np.ndarray] = []
        for segment in cracks:
            try:
                arr = np.asarray(segment, dtype=float).reshape(-1, 2)
            except Exception:
                continue
            if arr.shape[0] >= 2:
                prepared.append(arr)
        if crack_coordinate_space == "middle" and self.upper > 0:
            offset = np.array([float(self.upper), 0.0])
            return [arr + offset for arr in prepared]
        return prepared


__all__ = ["RegionLayout"]
