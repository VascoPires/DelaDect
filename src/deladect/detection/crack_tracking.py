"""Crack tracking primitives for the baseline-normalised diffuse workflow.

Public API
----------
CrackDetection
    Immutable descriptor for a single detected crack segment.
CrackTrack
    Mutable tracker object accumulating per-frame history.
normalize_detections(raw_cracks)
    Convert raw segment arrays to :class:`CrackDetection` lists.
match_tracks(tracks, detections, ...)
    Greedy one-to-one track–detection assignment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrackDetection:
    """Geometric descriptor for a single detected crack segment."""

    segment: np.ndarray
    center_yx: Tuple[float, float]
    length_px: float
    angle_deg: float
    bbox: Tuple[int, int, int, int]


@dataclass
class CrackTrack:
    """Mutable tracker accumulating detections across frames.

    The ``active`` flag is set to ``False`` when a track is terminated
    (no matching detection in a new frame). ``history`` is mutated in-place
    during the tracking loop; each entry is a plain dict with at least
    ``"frame_abs"`` and ``"status"`` keys.
    """

    track_id: int
    first_frame_abs: int
    baseline_frame_abs: int
    baseline_segment: np.ndarray
    baseline_length_px: float
    baseline_bbox: Tuple[int, int, int, int]
    last_frame_abs: int
    last_segment: np.ndarray
    last_length_px: float
    last_bbox: Tuple[int, int, int, int]
    active: bool = True
    history: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------


def _segment_features(seg: np.ndarray) -> CrackDetection:
    seg = np.asarray(seg, dtype=np.float32).reshape(2, 2)
    y0, x0 = seg[0]
    y1, x1 = seg[1]
    cy = float((y0 + y1) / 2.0)
    cx = float((x0 + x1) / 2.0)
    dy = float(y1 - y0)
    dx = float(x1 - x0)
    length = float(math.hypot(dy, dx))
    angle = float(math.degrees(math.atan2(dy, dx)))
    y_min = int(math.floor(min(y0, y1)))
    x_min = int(math.floor(min(x0, x1)))
    y_max = int(math.ceil(max(y0, y1))) + 1
    x_max = int(math.ceil(max(x0, x1))) + 1
    return CrackDetection(
        segment=seg,
        center_yx=(cy, cx),
        length_px=length,
        angle_deg=angle,
        bbox=(y_min, x_min, y_max, x_max),
    )


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    iy0 = max(ay0, by0)
    ix0 = max(ax0, bx0)
    iy1 = min(ay1, by1)
    ix1 = min(ax1, bx1)
    inter = float(max(0, iy1 - iy0)) * float(max(0, ix1 - ix0))
    if inter <= 0:
        return 0.0
    area_a = float(max(0, ay1 - ay0)) * float(max(0, ax1 - ax0))
    area_b = float(max(0, by1 - by0)) * float(max(0, bx1 - bx0))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def normalize_detections(raw_cracks: Sequence[np.ndarray]) -> List[CrackDetection]:
    """Convert a list of raw segment arrays to :class:`CrackDetection` objects.

    Parameters
    ----------
    raw_cracks:
        Each element is an ``(N, 2, 2)`` or ``(2, 2)`` float array of
        ``[[y0, x0], [y1, x1]]`` endpoint pairs.

    Returns
    -------
    list[CrackDetection]
        One entry per segment found across all arrays.
    """
    detections: List[CrackDetection] = []
    for crack in raw_cracks:
        segs = np.asarray(crack, dtype=np.float32).reshape(-1, 2, 2)
        for seg in segs:
            detections.append(_segment_features(seg))
    return detections


def match_tracks(
    tracks: List[CrackTrack],
    detections: List[CrackDetection],
    *,
    max_center_px: float,
    max_angle_deg: float,
    max_cost: float,
) -> Tuple[Dict[int, int], List[int], List[int]]:
    """Greedy one-to-one assignment of active tracks to new detections.

    Parameters
    ----------
    tracks:
        All known tracks (active and closed).
    detections:
        Detections for the current frame.
    max_center_px:
        Maximum centre-to-centre distance to consider a match.
    max_angle_deg:
        Maximum angle difference (in degrees, 0–90) to consider a match.
    max_cost:
        Assignments with cost above this value are rejected.

    Raises
    ------
    ValueError
        If a matching threshold is not finite, or if either distance/angle
        threshold is non-positive.

    Returns
    -------
    matched : dict[int, int]
        Map from track index to detection index for accepted matches.
    unmatched_tracks : list[int]
        Indices of active tracks with no match.
    unmatched_dets : list[int]
        Indices of detections not assigned to any track.
    """
    max_center_px = float(max_center_px)
    max_angle_deg = float(max_angle_deg)
    max_cost = float(max_cost)
    if not math.isfinite(max_center_px) or max_center_px <= 0:
        raise ValueError("max_center_px must be a finite value greater than zero.")
    if not math.isfinite(max_angle_deg) or max_angle_deg <= 0:
        raise ValueError("max_angle_deg must be a finite value greater than zero.")
    if not math.isfinite(max_cost) or max_cost < 0:
        raise ValueError("max_cost must be a finite value greater than or equal to zero.")

    candidates: List[Tuple[float, int, int]] = []
    eps = 1e-6
    for ti, track in enumerate(tracks):
        if not track.active:
            continue
        t_feat = _segment_features(track.last_segment)
        t_center = t_feat.center_yx
        t_len = max(track.last_length_px, eps)
        for di, det in enumerate(detections):
            cy, cx = det.center_yx
            ty, tx = t_center
            center_dist = float(math.hypot(cy - ty, cx - tx))
            angle_diff = abs(det.angle_deg - t_feat.angle_deg)
            angle_diff = min(angle_diff, 180.0 - angle_diff)
            if center_dist > max_center_px or angle_diff > max_angle_deg:
                continue
            iou = _bbox_iou(track.last_bbox, det.bbox)
            length_ratio = det.length_px / t_len
            if length_ratio <= 0:
                continue
            if iou <= 0.0 and center_dist > 0.5 * max_center_px:
                continue
            cost = (
                1.25 * (center_dist / max_center_px)
                + 0.75 * (angle_diff / max_angle_deg)
                + 0.75 * (1.0 - iou)
                + 0.25 * abs(math.log(length_ratio))
            )
            if cost <= max_cost:
                candidates.append((cost, ti, di))

    candidates.sort(key=lambda item: item[0])
    matched: Dict[int, int] = {}
    used_tracks: Set[int] = set()
    used_dets: Set[int] = set()
    for _, ti, di in candidates:
        if ti in used_tracks or di in used_dets:
            continue
        used_tracks.add(ti)
        used_dets.add(di)
        matched[ti] = di

    unmatched_tracks = [
        ti for ti in range(len(tracks))
        if ti not in used_tracks and tracks[ti].active
    ]
    unmatched_dets = [di for di in range(len(detections)) if di not in used_dets]
    return matched, unmatched_tracks, unmatched_dets
