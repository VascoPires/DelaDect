"""Diffuse delamination detection: :class:`DiffuseDetector`."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import closing, disk

from deladect.io import layout as layout_io

from ._common import (
    CrackInput,
    _Progress,
    _cracks_by_frame,
    _ensure_uint8,
    _frame_to_float,
    _hard_floor_mask,
    _kmeans_split,
    _percentile_range,
    _resolve_hard_floor_ratio,
    _resolve_optional_float,
    _resolve_pair,
    _resolve_pos_scale,
    _scale_to_unit,
    _smooth_for_threshold,
)
from ._overlays import _save_diffuse_overlay
from ._frames import PreprocessedFrames, resolve_frames
from ._preprocess import _reference_anchor_index, _reference_window_bounds

if TYPE_CHECKING:
    from .core import DelaminationDetector

logger = logging.getLogger(__name__)

DIFFUSE_CRACK_FRAME_POLICIES: Tuple[str, ...] = ("current", "reference_latest", "reference_midpoint")

Bounds = Tuple[int, int, int, int]


def _safe_otsu_threshold(values: np.ndarray) -> float:
    """Otsu threshold that also works for empty, constant or non-finite values."""
    values = np.asarray(values).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.5
    v_min = float(values.min())
    v_max = float(values.max())
    if v_max - v_min < 1e-3:
        return v_min
    try:
        return float(threshold_otsu(values))
    except ValueError:
        return float(np.median(values))


def _splat_mask(full_mask: np.ndarray, mask_bbox: np.ndarray, bounds: Bounds) -> None:
    """OR ``mask_bbox`` into ``full_mask`` at ``bounds``, clipped to the frame."""
    height, width = full_mask.shape[:2]
    y_lo, y_hi, x_lo, x_hi = bounds
    out_y0, out_y1 = max(0, y_lo), min(height, y_hi)
    out_x0, out_x1 = max(0, x_lo), min(width, x_hi)
    src_y0 = out_y0 - y_lo
    src_x0 = out_x0 - x_lo
    full_mask[out_y0:out_y1, out_x0:out_x1] |= mask_bbox[
        src_y0: src_y0 + (out_y1 - out_y0),
        src_x0: src_x0 + (out_x1 - out_x0),
    ]


class DiffuseDetector:
    """Diffuse delamination detection around cracks.

    :meth:`diffuse_delamination` looks at each frame's cracks on their
    own; :meth:`diffuse_crack_tracking` follows the cracks over time.
    """

    def __init__(self, owner: DelaminationDetector) -> None:
        """Create a diffuse detector for the parent :class:`DelaminationDetector`."""
        self.owner = owner

    def diffuse_delamination(
        self,
        *,
        cracks: Optional[CrackInput] = None,
        processed_cache_paths: Optional[List[Path]] = None,
        processed_stack: Optional[List[np.ndarray]] = None,
        save_overlays: bool = False,
        overlay_dirname: str = "delamination",
        max_frames: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        progress: bool = False,
        crack_coordinate_space: str = "middle",
    ) -> Dict[str, Any]:
        """Detect diffuse delamination in a region around each crack.

        The regions are filtered like in edge detection, one threshold is
        computed per frame from all regions together, and the masks are
        latched over time.

        Parameters
        ----------
        cracks:
            Cracks per frame, or the result of
            :func:`~deladect.detection.crack_analysis` (all orientations are
            used). There must be one entry per frame.
        processed_cache_paths, processed_stack:
            Preprocessed frames, as cache files or arrays. If neither is
            given, the full stack is preprocessed with a static reference.
            In region mode the middle rows are cut from these full frames.
        save_overlays:
            Save a diffuse overlay per frame.
        overlay_dirname:
            Output folder under the specimen results.
        max_frames:
            Process only the first ``max_frames`` frames (at least 1), of
            both the images and the cracks.
        params:
            Diffuse parameter overrides. ``crack_frame_policy`` picks which
            frame's cracks are used for each frame: ``"current"``,
            ``"reference_latest"`` or ``"reference_midpoint"`` of the
            reference window stored in the cache.
        debug:
            Return the threshold and regions of each frame.
        progress:
            Print progress.
        crack_coordinate_space:
            ``"middle"`` if the cracks were detected on the middle region
            stack, ``"full"`` if on the full frames. Only used to place the
            cracks on overlays in region mode. Detection in region mode needs
            cracks from the middle region stack; if the last frame has cracks
            outside the middle region, a ``ValueError`` is raised.

        Returns
        -------
        dict[str, Any]
            ``{"masks": {frame_key: mask}, "debug": {...} or None}``, with
            keys like ``"frame_0003"``.
        """
        if cracks is None:
            raise ValueError("Diffuse delamination requires `cracks` to be provided.")
        if crack_coordinate_space not in {"middle", "full"}:
            raise ValueError("crack_coordinate_space must be one of: 'middle', 'full'.")
        raw_stack = getattr(self.owner.specimen, "image_stack_full", None)
        if save_overlays and raw_stack is None:
            raise ValueError("Cannot save overlays without a full raw image stack.")

        frames = resolve_frames(
            self.owner,
            processed_cache_paths=processed_cache_paths,
            processed_stack=processed_stack,
            max_frames=max_frames,
            auto_key="diffuse_auto",
            save_previews=save_overlays,
            progress=progress,
        )
        cracks_by_frame = _cracks_by_frame(cracks, len(frames), max_frames)
        self.owner.layout.check_cracks_in_middle(cracks_by_frame)
        return self._detect(
            frames,
            cracks_by_frame,
            params=self._resolve_diffuse_params(params),
            save_overlays=save_overlays,
            overlay_dirname=overlay_dirname,
            debug=debug,
            progress=progress,
            crack_coordinate_space=crack_coordinate_space,
        )

    def _detect(
        self,
        frames: PreprocessedFrames,
        cracks_list: List[Any],
        *,
        params: Dict[str, Any],
        save_overlays: bool,
        overlay_dirname: str,
        debug: bool,
        progress: bool,
        crack_coordinate_space: str,
    ) -> Dict[str, Any]:
        """Detect diffuse damage in the diffuse rows of every frame and latch the masks."""
        layout = self.owner.layout
        rows = layout.diffuse_rows()
        row_offset = 0 if rows is None else rows[0]
        search_frames = frames if rows is None else frames.rows(*rows)
        raw_stack = getattr(self.owner.specimen, "image_stack_full", None)
        overlay_dir = layout_io.overlay_dir(self.owner.specimen, overlay_dirname, "diffuse") if save_overlays else None
        diffuse_masks: Dict[str, np.ndarray] = {}
        debug_payloads = self._new_debug_payload(params) if debug else None
        latched: Optional[np.ndarray] = None
        progress_log = _Progress("diffuse_delamination", len(frames), progress)

        for idx, processed, frame_meta in search_frames.with_metadata():
            crack_idx, ref_start, ref_end = self._resolve_diffuse_crack_index(
                frame_idx=idx,
                cracks_count=len(cracks_list),
                params=params,
                frame_meta=frame_meta,
            )
            frame_cracks = cracks_list[crack_idx] if 0 <= crack_idx < len(cracks_list) else []
            if frame_cracks is None:
                frame_cracks = []

            mask, bounds_list, hard_floors, threshold = self._frame_roi_mask(processed, frame_cracks, params)
            latched = mask if latched is None else mask | latched
            mask_full = layout.diffuse_to_full(latched)

            frame_key = f"frame_{idx:04d}"
            diffuse_masks[frame_key] = mask_full

            if overlay_dir is not None:
                _save_diffuse_overlay(
                    _ensure_uint8(raw_stack[idx]),
                    mask_full,
                    overlay_dir / layout_io.overlay_name("diffuse", idx),
                    cracks=layout.cracks_for_display(frame_cracks, crack_coordinate_space),
                )

            if debug_payloads is not None:
                hard_floor_values = [float(value) for value in hard_floors if value is not None]
                debug_payloads["frames"][frame_key] = {
                    "crack_count": len(frame_cracks),
                    "crack_idx_used": int(crack_idx),
                    "reference_window": [int(ref_start), int(ref_end)],
                    "roi_bounds": [
                        (y_lo + row_offset, y_hi + row_offset, x_lo, x_hi) for y_lo, y_hi, x_lo, x_hi in bounds_list
                    ],
                    "threshold": threshold,
                    "hard_floor_eff_min": float(np.min(hard_floor_values)) if hard_floor_values else None,
                    "hard_floor_eff_max": float(np.max(hard_floor_values)) if hard_floor_values else None,
                }

            progress_log.update(idx + 1)

        progress_log.done()
        return {"masks": diffuse_masks, "debug": debug_payloads}

    @staticmethod
    def _new_debug_payload(diffuse_params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "frames": {},
            "params": diffuse_params,
            "threshold_strategy": "kmeans",
            "threshold_mode": "per_frame_roi_union",
        }

    def _frame_roi_mask(
        self,
        processed: np.ndarray,
        frame_cracks: Sequence[np.ndarray],
        params: Dict[str, Any],
    ) -> Tuple[np.ndarray, List[Bounds], List[Optional[float]], float]:
        """Threshold the regions around every crack of one frame with one shared threshold.

        The threshold comes from the pooled (downsampled) values of all
        regions. Returns the frame mask, the bounds of each non-empty region,
        the hard floor used for each region and the threshold.
        """
        avg_crack_width_px = self.owner.specimen.avg_crack_width_px
        step = params["threshold_downsample"]
        image = np.asarray(processed, dtype=np.float32)

        rois: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        samples: List[np.ndarray] = []
        for crack in frame_cracks:
            geom = self._diffuse_roi_geometry(image, crack, dx=params["diffuse_dx"], dy=params["diffuse_dy"])
            if geom is None:
                continue
            pre = self._diffuse_prethreshold_image(geom["patch"], params=params, avg_crack_width_px=avg_crack_width_px)
            rois.append((geom, pre))
            closed_sample = pre["closed"][::step, ::step] if step > 1 else pre["closed"]
            samples.append(closed_sample.reshape(-1))

        values = np.concatenate(samples) if samples else np.array([], dtype=np.float32)
        threshold = self._compute_frame_diffuse_threshold(values, max_samples=params["threshold_max_samples"])

        mask = np.zeros_like(processed, dtype=bool)
        bounds_list: List[Bounds] = []
        for geom, pre in rois:
            roi_mask, bounds = self._diffuse_mask_from_preprocessed(
                geom=geom,
                closed=pre["closed"],
                floor_mask=pre["floor_mask"],
                threshold=threshold,
                params=params,
                avg_crack_width_px=avg_crack_width_px,
            )
            if roi_mask.size == 0:
                continue
            y_lo, y_hi, x_lo, x_hi = bounds
            mask[y_lo:y_hi, x_lo:x_hi] |= roi_mask
            bounds_list.append(bounds)
        return mask, bounds_list, [pre["hard_floor_eff"] for _, pre in rois], threshold

    def _resolve_diffuse_params(self, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Fill in the default diffuse parameters and check the given ones.

        ``window_edge`` is accepted in place of ``window_diffuse``.
        ``hard_floor`` is a fraction of full intensity: pixels brighter than
        it are never damage. The default 0.90 comes from tuning on our
        specimens; Glud/Bender crack detection is often run at about 0.96.
        """
        resolved = {
            "diffuse_dx": 20.0,
            "diffuse_dy": 20.0,
            "threshold_max_samples": 200000,
            "threshold_downsample": 2,
            "crack_frame_policy": "reference_midpoint",
            "window_diffuse": (0, 60),
            "gaussian_filters": (0.5, 15.0),
            "scale_min": 150.0,
            "scale_max": 255.0,
            "scale_min_percentile": 10.0,
            "scale_max_percentile": 99.0,
            "hard_floor": 0.90,
            "post_threshold_closing_px": 4,
            "post_threshold_closing_scale": None,
            "reference_mode": None,
            "reference_window": None,
            "reference_skip": None,
        }
        if params:
            if "window_edge" in params and "window_diffuse" not in params:
                params = {**params, "window_diffuse": params["window_edge"]}
            resolved.update(params)

        def optional(value: Any, cast_to: Any) -> Any:
            return None if value is None else cast_to(value)

        resolved.update(
            diffuse_dx=float(resolved["diffuse_dx"]),
            diffuse_dy=float(resolved["diffuse_dy"]),
            threshold_max_samples=max(1, int(resolved["threshold_max_samples"])),
            threshold_downsample=max(1, int(resolved["threshold_downsample"])),
            window_diffuse=_resolve_pair(resolved["window_diffuse"], name="window_diffuse", caster=int),
            gaussian_filters=_resolve_pair(resolved["gaussian_filters"], name="gaussian_filters", caster=float),
            scale_min=float(resolved["scale_min"]),
            scale_max=float(resolved["scale_max"]),
            scale_min_percentile=_resolve_optional_float(resolved["scale_min_percentile"]),
            scale_max_percentile=_resolve_optional_float(resolved["scale_max_percentile"]),
            hard_floor=_resolve_hard_floor_ratio(resolved["hard_floor"]),
            post_threshold_closing_px=max(0, int(resolved["post_threshold_closing_px"])),
            post_threshold_closing_scale=_resolve_pos_scale(resolved["post_threshold_closing_scale"]),
        )
        policy = str(resolved["crack_frame_policy"]).strip().lower()
        if policy not in DIFFUSE_CRACK_FRAME_POLICIES:
            allowed = ", ".join(DIFFUSE_CRACK_FRAME_POLICIES)
            raise ValueError(f"crack_frame_policy must be one of: {allowed}")
        resolved.update(
            crack_frame_policy=policy,
            reference_mode=optional(resolved["reference_mode"], str),
            reference_window=optional(resolved["reference_window"], lambda value: max(1, int(value))),
            reference_skip=optional(resolved["reference_skip"], lambda value: max(0, int(value))),
        )
        return resolved

    def _resolve_diffuse_crack_index(
        self,
        *,
        frame_idx: int,
        cracks_count: int,
        params: Dict[str, Any],
        frame_meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, int, int]:
        """Choose which frame's cracks to use for frame ``frame_idx``.

        Returns ``(crack frame, reference start, reference end)``. The
        reference window comes from the cache metadata when available.
        """
        policy = str(params.get("crack_frame_policy", "reference_midpoint")).strip().lower()
        reference_mode = ""
        if frame_meta is not None and frame_meta.get("reference_mode") is not None:
            reference_mode = str(frame_meta.get("reference_mode")).strip().lower()
        elif params.get("reference_mode") is not None:
            reference_mode = str(params.get("reference_mode")).strip().lower()

        if reference_mode == "static" and policy == "reference_midpoint":
            notice_key = "static_reference_midpoint_policy_override"
            if not self.owner._notice_flags.get(notice_key, False):
                logger.warning(
                    "crack_frame_policy='reference_midpoint' with reference_mode='static' anchors cracks "
                    "to the static baseline frame; overriding crack_frame_policy to 'current'."
                )
                self.owner._notice_flags[notice_key] = True
            policy = "current"

        if frame_meta is not None:
            start_idx = int(frame_meta.get("ref_start_idx", frame_idx))
            end_idx = int(frame_meta.get("ref_end_idx", frame_idx + 1))
        elif params.get("reference_mode") is None:
            start_idx, end_idx = int(frame_idx), int(frame_idx) + 1
        else:
            start_idx, end_idx = _reference_window_bounds(
                frame_idx,
                reference_mode=str(params["reference_mode"]),
                reference_window=int(params.get("reference_window") or 1),
                reference_skip=int(params.get("reference_skip") or 0),
            )

        if policy == "reference_midpoint" and frame_meta is not None and "ref_anchor_idx" in frame_meta:
            anchor_idx = int(frame_meta["ref_anchor_idx"])
        else:
            anchor_idx = _reference_anchor_index(frame_idx, start_idx=start_idx, end_idx=end_idx, policy=policy)

        if cracks_count <= 0:
            return int(frame_idx), int(start_idx), int(end_idx)

        anchor_idx = max(0, min(int(anchor_idx), int(cracks_count) - 1))
        return int(anchor_idx), int(start_idx), int(end_idx)

    def _compute_frame_diffuse_threshold(
        self,
        values: np.ndarray,
        *,
        max_samples: int,
    ) -> float:
        """K-means threshold of the pooled region values, or Otsu if k-means can't split them."""
        values = np.asarray(values).reshape(-1)
        if values.size == 0:
            return 0.5
        if values.size > max_samples:
            values = values[:: max(1, values.size // max_samples)]
        threshold = _kmeans_split(values)
        return _safe_otsu_threshold(values) if threshold is None else threshold

    def _diffuse_roi_geometry(
        self,
        image: np.ndarray,
        crack: np.ndarray,
        *,
        dx: float,
        dy: float,
    ) -> Optional[Dict[str, Any]]:
        """Cut out a region around ``crack``, rotated to follow the crack.

        The region extends ``dx`` pixels to each side of the crack and ``dy``
        past its ends. Returns the patch and the transform back to the frame,
        or ``None`` if the region falls outside the image.
        """
        h, w = image.shape[:2]
        crack_arr = np.asarray(crack, dtype=np.float64).reshape(-1, 2)
        if crack_arr.shape[0] < 2:
            return None

        (y1, x1), (y2, x2) = crack_arr[:2]
        vy = float(y2 - y1)
        vx = float(x2 - x1)
        seg_len = float(np.hypot(vy, vx))

        if not np.isfinite(seg_len) or seg_len < 1e-6:
            y_lo = int(max(0, min(y1, y2) - dy))
            y_hi = int(min(h, max(y1, y2) + dy))
            x_lo = int(max(0, min(x1, x2) - dx))
            x_hi = int(min(w, max(x1, x2) + dx))
            if x_hi <= x_lo or y_hi <= y_lo:
                return None
            patch = _ensure_uint8(image[y_lo:y_hi, x_lo:x_hi].copy())
            valid_mask = np.ones_like(patch, dtype=bool)
            return {
                "bounds": (y_lo, y_hi, x_lo, x_hi),
                "patch": patch,
                "matrix": np.eye(2, dtype=np.float64),
                "offset": np.array([float(y_lo), float(x_lo)], dtype=np.float64),
                "rotated": False,
                "valid_mask": valid_mask,
            }

        center = np.array([(y1 + y2) / 2.0, (x1 + x2) / 2.0], dtype=np.float64)
        u_parallel = np.array([vy, vx], dtype=np.float64) / seg_len
        u_perp = np.array([-u_parallel[1], u_parallel[0]], dtype=np.float64)

        half_len = max(seg_len / 2.0, 0.5) + float(dy)
        half_width = max(1.0, float(dx))

        roi_height = max(1, int(np.ceil(2.0 * half_len)) + 2)
        roi_width = max(1, int(np.ceil(2.0 * half_width)) + 2)

        matrix = np.array(
            [[u_parallel[0], u_perp[0]], [u_parallel[1], u_perp[1]]],
            dtype=np.float64,
        )
        half_len_pix = (roi_height - 1) / 2.0
        half_width_pix = (roi_width - 1) / 2.0
        offset = center - u_parallel * half_len_pix - u_perp * half_width_pix

        roi_patch = ndi.affine_transform(
            image.astype(np.float32, copy=False),
            matrix=matrix,
            offset=offset,
            output_shape=(roi_height, roi_width),
            order=1,
            mode="constant",
            cval=0.0,
        )
        coverage = ndi.affine_transform(
            np.ones_like(image, dtype=np.float32),
            matrix=matrix,
            offset=offset,
            output_shape=(roi_height, roi_width),
            order=1,
            mode="constant",
            cval=0.0,
        )
        valid_mask = coverage > 1e-6
        patch = np.clip(roi_patch, 0.0, 255.0).astype(np.uint8)
        patch = np.where(valid_mask, patch, 255)

        corners_local = np.array(
            [
                [0.0, 0.0],
                [roi_height - 1.0, 0.0],
                [roi_height - 1.0, roi_width - 1.0],
                [0.0, roi_width - 1.0],
            ],
            dtype=np.float64,
        )
        corners_global = (corners_local @ matrix.T) + offset

        y_min = float(np.min(corners_global[:, 0]))
        y_max = float(np.max(corners_global[:, 0]))
        x_min = float(np.min(corners_global[:, 1]))
        x_max = float(np.max(corners_global[:, 1]))

        y_lo = max(0, int(np.floor(y_min)))
        y_hi = min(h, int(np.ceil(y_max)) + 1)
        x_lo = max(0, int(np.floor(x_min)))
        x_hi = min(w, int(np.ceil(x_max)) + 1)

        if x_hi <= x_lo or y_hi <= y_lo:
            return None

        return {
            "bounds": (y_lo, y_hi, x_lo, x_hi),
            "patch": patch,
            "matrix": matrix,
            "offset": offset,
            "rotated": True,
            "valid_mask": valid_mask,
        }

    @staticmethod
    def _apply_roi_geometry(frame: np.ndarray, geom: Dict[str, Any]) -> np.ndarray:
        """Sample the region of ``geom`` from another frame, e.g. the current frame for a baseline region."""
        h_out, w_out = geom["patch"].shape[:2]
        arr = _frame_to_float(frame) * 255.0
        roi = ndi.affine_transform(
            arr.astype(np.float32, copy=False),
            matrix=geom["matrix"],
            offset=geom["offset"],
            output_shape=(h_out, w_out),
            order=1,
            mode="constant",
            cval=0.0,
        )
        return np.clip(roi, 0.0, 255.0).astype(np.uint8)

    def _diffuse_baseline_normalized_roi(
        self,
        baseline_frame: np.ndarray,
        current_frame: np.ndarray,
        crack_segment: np.ndarray,
        *,
        dx: float,
        dy: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]]:
        """Cut the same crack region out of the baseline and the current frame and divide them.

        The region is placed using ``crack_segment`` on ``baseline_frame``.
        Returns ``(ratio, baseline, current, geom)`` as ``uint8`` patches plus
        the region geometry, or ``None`` if the region can't be built. In the
        ratio, dark pixels are new damage and bright ones are unchanged.
        """
        geom = self._diffuse_roi_geometry(baseline_frame, crack_segment, dx=dx, dy=dy)
        if geom is None:
            return None

        roi_baseline_u8 = geom["patch"]
        roi_current_u8 = self._apply_roi_geometry(current_frame, geom)

        roi_baseline_f = roi_baseline_u8.astype(np.float32) / 255.0
        roi_current_f = roi_current_u8.astype(np.float32) / 255.0

        roi_ratio = np.clip(
            roi_current_f / np.maximum(roi_baseline_f, 1e-3), 0.0, 1.0
        )
        roi_ratio_u8 = (roi_ratio * 255.0).astype(np.uint8)
        return roi_ratio_u8, roi_baseline_u8, roi_current_u8, geom

    def _diffuse_prethreshold_image(
        self,
        image: np.ndarray,
        *,
        params: Dict[str, Any],
        avg_crack_width_px: float,
    ) -> Dict[str, Any]:
        """Filter and scale a region patch before thresholding.

        When the percentile settings are valid they override
        ``scale_min``/``scale_max``.
        """
        filtered_max, filtered_min, sharpened, smoothed = _smooth_for_threshold(
            _ensure_uint8(image), params["window_diffuse"], params["gaussian_filters"], avg_crack_width_px
        )
        scale_range = _percentile_range(
            smoothed, params.get("scale_min_percentile"), params.get("scale_max_percentile")
        ) or (float(params.get("scale_min", 0)), float(params.get("scale_max", 255)))
        constant_scaled = _scale_to_unit(smoothed, *scale_range)
        floor_mask, hard_floor_eff = _hard_floor_mask(smoothed, params.get("hard_floor"))

        return {
            "filtered_max": filtered_max,
            "filtered_min": filtered_min,
            "sharpened": sharpened,
            "smoothed": smoothed,
            "constant_scaled": constant_scaled,
            "closed": constant_scaled,
            "floor_mask": floor_mask,
            "hard_floor_eff": hard_floor_eff,
        }

    def _diffuse_mask_from_preprocessed(
        self,
        *,
        geom: Dict[str, Any],
        closed: np.ndarray,
        floor_mask: np.ndarray,
        threshold: float,
        params: Dict[str, Any],
        avg_crack_width_px: float,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Threshold a region, close small gaps, and map the mask back to frame coordinates."""
        bounds = geom["bounds"]
        valid_mask = geom.get("valid_mask")

        roi_mask_aligned = (closed < float(threshold)) & np.asarray(floor_mask, dtype=bool)
        close_px = params.get("post_threshold_closing_px")
        if close_px is not None:
            close_radius = max(0, int(close_px))
        else:
            close_scale = params.get("post_threshold_closing_scale")
            if close_scale is None:
                close_radius = 4
            else:
                close_radius = max(1, int(round(max(0.5, float(close_scale)) * avg_crack_width_px)))
        roi_mask_aligned = closing(roi_mask_aligned, disk(close_radius)).astype(bool)
        if valid_mask is not None:
            roi_mask_aligned = np.where(valid_mask, roi_mask_aligned, False)

        if not geom["rotated"]:
            return roi_mask_aligned.astype(bool), bounds

        matrix = geom["matrix"]
        offset = geom["offset"]
        y_lo, y_hi, x_lo, x_hi = bounds

        global_offset = np.array([y_lo, x_lo], dtype=np.float64)
        matrix_back = matrix.T
        offset_back = matrix_back @ (global_offset - offset)

        projected = ndi.affine_transform(
            roi_mask_aligned.astype(np.float32, copy=False),
            matrix=matrix_back,
            offset=offset_back,
            output_shape=(y_hi - y_lo, x_hi - x_lo),
            order=1,
            mode="constant",
            cval=0.0,
        )
        roi_mask_bbox = projected > 0.5
        if valid_mask is not None:
            valid_projected = ndi.affine_transform(
                valid_mask.astype(np.float32, copy=False),
                matrix=matrix_back,
                offset=offset_back,
                output_shape=(y_hi - y_lo, x_hi - x_lo),
                order=1,
                mode="constant",
                cval=0.0,
            )
            roi_mask_bbox = np.where(valid_projected > 0.5, roi_mask_bbox, False)
        return roi_mask_bbox.astype(bool), bounds

    def diffuse_crack_tracking(
        self,
        processed_frames: List[np.ndarray],
        crack_frames: List[List[Any]],
        selected_indices: List[int],
        *,
        avg_crack_width_px: float,
        diffuse_params: Dict[str, Any],
        max_center_px: Optional[float] = None,
        max_angle_deg: float = 15.0,
        max_cost: float = 1.8,
        return_intermediates: bool = False,
    ) -> Dict[str, Any]:
        """Follow cracks over time and detect diffuse damage around them.

        Each crack region is divided by the same region in the frame where
        the crack first appeared, so only changes since then show up. This
        is done for every matched crack and for cracks that vanish (they may
        have been hidden by delamination).

        Parameters
        ----------
        processed_frames:
            Preprocessed frames, in the order of ``selected_indices``.
        crack_frames:
            :class:`~deladect.detection.CrackDetection` lists per frame, from
            :func:`~deladect.detection.normalize_detections`.
        selected_indices:
            Frame numbers of ``processed_frames``.
        avg_crack_width_px:
            Crack width used when filtering the regions.
        diffuse_params:
            Resolved diffuse parameters (``diffuse_dx``, ``diffuse_dy``,
            ``window_diffuse``, ``gaussian_filters``, ...).
        max_center_px:
            Largest center distance for a crack match. Defaults to
            ``max(12, 2.5 * avg_crack_width_px)``.
        max_angle_deg:
            Largest angle difference for a crack match.
        max_cost:
            Largest match cost, see :func:`~deladect.detection.match_tracks`.
        return_intermediates:
            Include the region images in each stats entry (uses a lot of
            memory).

        Returns
        -------
        dict[str, Any]
            ``tracks`` (all :class:`~deladect.detection.CrackTrack` objects),
            ``events`` (new, matched and terminated tracks),
            ``vanishing_stats`` and ``diffuse_stats`` (one entry per region
            checked), ``frame_masks`` (``{frame: mask}``) and
            ``frame_detection_track_ids`` (track id of each detection).
        """
        from deladect.detection.crack_tracking import CrackTrack, match_tracks

        dx = float(diffuse_params["diffuse_dx"])
        dy = float(diffuse_params["diffuse_dy"])
        downsample = int(diffuse_params.get("threshold_downsample", 2))
        max_samples = int(diffuse_params.get("threshold_max_samples", 400_000))
        if max_center_px is None:
            max_center_px = max(12.0, 2.5 * avg_crack_width_px)
        frame_pos_by_abs = {int(frame_abs): pos for pos, frame_abs in enumerate(selected_indices)}

        tracks: List[CrackTrack] = []
        events: List[Dict[str, Any]] = []
        vanishing_stats: List[Dict[str, Any]] = []
        diffuse_stats: List[Dict[str, Any]] = []
        frame_masks: Dict[int, np.ndarray] = {}
        frame_detection_track_ids: Dict[int, List[Optional[int]]] = {}

        def segment_roi(track: CrackTrack, frame_abs: int, frame: np.ndarray, segment: np.ndarray) -> Optional[Dict[str, Any]]:
            """Detect damage around ``segment`` in ``frame``, relative to the track's first frame.

            The mask is added to ``frame_masks[frame_abs]``.
            """
            base_pos = frame_pos_by_abs.get(int(track.baseline_frame_abs))
            if base_pos is None:
                return None
            roi = self._diffuse_baseline_normalized_roi(processed_frames[base_pos], frame, segment, dx=dx, dy=dy)
            if roi is None:
                return None
            roi_ratio_u8, roi_baseline_u8, roi_current_u8, geom = roi
            pre = self._diffuse_prethreshold_image(
                roi_ratio_u8, params=diffuse_params, avg_crack_width_px=avg_crack_width_px
            )
            threshold = self._compute_frame_diffuse_threshold(
                pre["closed"][::downsample, ::downsample].reshape(-1), max_samples=max_samples
            )
            mask_bbox, bounds = self._diffuse_mask_from_preprocessed(
                geom=geom,
                closed=pre["closed"],
                floor_mask=pre["floor_mask"],
                threshold=threshold,
                params=diffuse_params,
                avg_crack_width_px=avg_crack_width_px,
            )
            if frame_abs not in frame_masks:
                frame_masks[frame_abs] = np.zeros(frame.shape[:2], dtype=bool)
            _splat_mask(frame_masks[frame_abs], mask_bbox, bounds)
            return {
                "threshold": float(threshold),
                "mask_frac": float(np.mean(mask_bbox)),
                "floor_mask_frac": float(np.mean(pre["floor_mask"])),
                "intermediates": {
                    "roi_ratio_u8": roi_ratio_u8,
                    "roi_baseline_u8": roi_baseline_u8,
                    "roi_current_u8": roi_current_u8,
                    "pre": pre,
                    "mask_bbox": mask_bbox,
                    "bounds": bounds,
                },
            }

        for frame_abs, detections, proc_frame in zip(selected_indices, crack_frames, processed_frames):
            frame_abs = int(frame_abs)
            matched, unmatched_tracks_idx, unmatched_det_idx = match_tracks(
                tracks,
                detections,
                max_center_px=float(max_center_px),
                max_angle_deg=max_angle_deg,
                max_cost=max_cost,
            )

            # A crack that vanishes may have been swallowed by diffuse damage:
            # check its last known position against its baseline frame.
            for ti in unmatched_tracks_idx:
                track = tracks[ti]
                track.active = False
                events.append({"frame_abs": frame_abs, "track_id": int(track.track_id), "status": "terminated"})

                was_matched = any(entry["status"] == "matched" for entry in track.history)
                if not was_matched or track.first_frame_abs == frame_abs:
                    continue
                roi = segment_roi(track, frame_abs, proc_frame, track.last_segment)
                if roi is None:
                    continue
                entry = {
                    "track_id": int(track.track_id),
                    "termination_frame_abs": frame_abs,
                    "baseline_frame_abs": int(track.baseline_frame_abs),
                    "threshold": roi["threshold"],
                    "mask_frac": roi["mask_frac"],
                    "floor_mask_frac": roi["floor_mask_frac"],
                }
                if return_intermediates:
                    entry.update({k: v for k, v in roi["intermediates"].items() if k != "bounds"})
                vanishing_stats.append(entry)

            det_to_track: Dict[int, int] = {di: ti for ti, di in matched.items()}

            for di in unmatched_det_idx:
                det = detections[di]
                track = CrackTrack(
                    track_id=len(tracks) + 1,
                    first_frame_abs=frame_abs,
                    baseline_frame_abs=frame_abs,
                    baseline_segment=det.segment.copy(),
                    baseline_length_px=float(det.length_px),
                    baseline_bbox=det.bbox,
                    last_frame_abs=frame_abs,
                    last_segment=det.segment.copy(),
                    last_length_px=float(det.length_px),
                    last_bbox=det.bbox,
                )
                track.history.append({"frame_abs": frame_abs, "status": "new"})
                tracks.append(track)
                det_to_track[di] = len(tracks) - 1

            for ti, di in matched.items():
                track = tracks[ti]
                det = detections[di]
                growth_ratio = (
                    0.0 if track.baseline_length_px <= 0
                    else float(det.length_px / track.baseline_length_px - 1.0)
                )
                events.append({
                    "frame_abs": frame_abs,
                    "track_id": int(track.track_id),
                    "status": "matched",
                    "growth_ratio": float(growth_ratio),
                    "baseline_frame_abs": int(track.baseline_frame_abs),
                    "baseline_segment_y0": float(track.baseline_segment[0, 0]),
                    "baseline_segment_x0": float(track.baseline_segment[0, 1]),
                    "baseline_segment_y1": float(track.baseline_segment[1, 0]),
                    "baseline_segment_x1": float(track.baseline_segment[1, 1]),
                    "current_segment_y0": float(det.segment[0, 0]),
                    "current_segment_x0": float(det.segment[0, 1]),
                    "current_segment_y1": float(det.segment[1, 0]),
                    "current_segment_x1": float(det.segment[1, 1]),
                })
                track.last_frame_abs = frame_abs
                track.last_segment = det.segment.copy()
                track.last_length_px = float(det.length_px)
                track.last_bbox = det.bbox
                track.history.append({"frame_abs": frame_abs, "status": "matched", "growth_ratio": float(growth_ratio)})

                roi = segment_roi(track, frame_abs, proc_frame, track.baseline_segment)
                if roi is None:
                    continue
                entry = {
                    "track_id": int(track.track_id),
                    "frame_abs": frame_abs,
                    "baseline_frame_abs": int(track.baseline_frame_abs),
                    "threshold": roi["threshold"],
                    "floor_mask_frac": roi["floor_mask_frac"],
                    "mask_frac": roi["mask_frac"],
                }
                if return_intermediates:
                    entry.update(roi["intermediates"])
                diffuse_stats.append(entry)

            frame_detection_track_ids[frame_abs] = [
                tracks[det_to_track[di]].track_id if di in det_to_track else None
                for di in range(len(detections))
            ]

        for track in tracks:
            track.active = False

        return {
            "tracks": tracks,
            "events": events,
            "vanishing_stats": vanishing_stats,
            "diffuse_stats": diffuse_stats,
            "frame_masks": frame_masks,
            "frame_detection_track_ids": frame_detection_track_ids,
        }
