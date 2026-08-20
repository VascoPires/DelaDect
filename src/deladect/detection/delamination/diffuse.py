"""Diffuse-focused delamination detection: :class:`DiffuseDetector`.

Depends on :mod:`._common`, :mod:`._overlays`, and :mod:`._preprocess`.
"""


from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, unsharp_mask
from skimage.morphology import closing, disk


logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING

from ._common import (
    DIFFUSE_CRACK_FRAME_POLICIES,
    CrackInput,
    _auto_preprocess_cache_paths,
    _coerce_cracks_by_frame,
    _crack_input_frame_count,
    _ensure_uint8,
    _fetch_region_override_stacks,
    _frame_to_float,
    _progress_done,
    _progress_init,
    _progress_update,
    _region_override_raw_frame,
    _resolve_hard_floor_ratio,
    _resolve_optional_float,
    _resolve_pair,
    _resolve_pos_scale,
    _result_key_token,
)
from ._overlays import _save_diffuse_overlay
from ._preprocess import (
    _reference_anchor_index,
    _reference_settings_from_cache_paths,
    _reference_window_bounds,
)

if TYPE_CHECKING:
    from .core import DelaminationDetector


def _safe_otsu_threshold(values: np.ndarray) -> float:
    """Compute Otsu threshold safely on degenerate or low-variance arrays."""
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



class DiffuseDetector:
    """Diffuse-focused delamination workflows.

    The class exposes:

    - :meth:`diffuse_delamination` for crack-guided diffuse detection.
    - :meth:`diffuse_crack_tracking` for optional track-based diffuse detection in
      :meth:`DelaminationDetector.detect_both_delaminations`.
    """

    def __init__(self, owner: DelaminationDetector) -> None:
        """Create a diffuse detector bound to its parent delamination detector."""
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
        """Detect diffuse delamination masks using crack-guided ROIs.

        This workflow builds one threshold per frame from the union of ROI values,
        then applies that threshold inside each crack-guided ROI after edge-style
        filtering. Frame masks are latched over time (logical OR).

        Parameters
        ----------
        cracks:
            Per-frame crack segments or the orientation-keyed result returned by
            :func:`deladect.detection.crack_analysis`. Every orientation present
            in a structured analysis result is merged into the local diffuse ROIs.
        processed_cache_paths, processed_stack:
            Exactly one preprocessed source. If omitted, preprocessing is executed
            automatically with static reference mode.
        save_overlays:
            If ``True``, save per-frame diffuse overlays.
        overlay_dirname:
            Root folder used under specimen results for diffuse outputs.
        max_frames:
            Optional cap on processed frames.
        params:
            Optional override dictionary for diffuse thresholds/filtering settings.
            ``crack_frame_policy`` controls which frame's cracks are used for each
            analyzed frame (``current``, ``reference_latest``, ``reference_midpoint``).
            When preprocessed cache metadata is available, reference-aligned crack
            indices are derived from that metadata.
        debug:
            If ``True``, return per-frame thresholds and ROI bounds.
        crack_coordinate_space:
            Coordinate space of ``cracks``, used for full-frame overlays in
            region-override mode: ``"middle"`` (default) if crack detection
            ran on the middle-region stack, or ``"full"`` if it was forced
            onto the full-frame stack. Never inferred from coordinate values.

        Returns
        -------
        dict[str, Any]
            ``{"masks": dict[str, np.ndarray], "debug": dict[str, Any] | None}``
            where masks are keyed as ``frame_XXXX``.
        """
        if cracks is None:
            raise ValueError("Diffuse delamination requires `cracks` to be provided.")
        if processed_cache_paths and processed_stack:
            raise ValueError("Provide either processed_cache_paths or processed_stack, not both.")
        if crack_coordinate_space not in {"middle", "full"}:
            raise ValueError("crack_coordinate_space must be one of: 'middle', 'full'.")

        if self.owner._uses_stack_overrides():
            return self._diffuse_delamination_region_overrides(
                cracks=cracks,
                processed_cache_paths=processed_cache_paths,
                save_overlays=save_overlays,
                overlay_dirname=overlay_dirname,
                max_frames=max_frames,
                params=params,
                debug=debug,
                progress=progress,
                crack_coordinate_space=crack_coordinate_space,
            )

        stacks = self.owner._select_stacks()
        raw_stack = getattr(self.owner.specimen, "image_stack_full", None) or stacks.get("full")
        if save_overlays and raw_stack is None:
            raise ValueError("Cannot save overlays without a full raw image stack.")

        crack_input_frames = _crack_input_frame_count(cracks)
        cracks_list = _coerce_cracks_by_frame(cracks, crack_input_frames)
        if processed_cache_paths is None and processed_stack is None:
            processed_cache_paths = _auto_preprocess_cache_paths(
                self.owner,
                save_overlays=save_overlays,
                max_frames=max_frames,
                progress=progress,
                key_prefix="diffuse_auto",
            )

        diffuse_masks: Dict[str, np.ndarray] = {}
        debug_payloads: Optional[Dict[str, Any]] = {"frames": {}} if debug else None

        diffuse_params = self._resolve_diffuse_params(params)
        if debug_payloads is not None:
            debug_payloads["params"] = diffuse_params
            debug_payloads["threshold_strategy"] = "kmeans"
            debug_payloads["threshold_mode"] = "per_frame_roi_union"

        total_frames = self._resolve_diffuse_frame_count(
            processed_stack=processed_stack,
            processed_cache_paths=processed_cache_paths,
            cracks=cracks_list,
            max_frames=max_frames,
        )
        progress_state = _progress_init("diffuse_delamination", total_frames, progress)
        if processed_stack is not None:
            processed_iter = enumerate(processed_stack)
            cache_metadata_iter = False
        else:
            if processed_cache_paths is None:
                raise ValueError("No processed frames available for diffuse detection.")
            processed_iter = self.owner.iter_preprocessed_cache_with_metadata(processed_cache_paths)
            cache_metadata_iter = True

        prev_latched: Optional[np.ndarray] = None

        for item in processed_iter:
            item_any = cast(Any, item)
            if cache_metadata_iter:
                idx = int(item_any[0])
                processed = item_any[1]
                frame_meta = item_any[2]
            else:
                idx = int(item_any[0])
                processed = item_any[1]
                frame_meta = None
            if idx >= total_frames:
                break

            crack_idx, ref_start, ref_end = self._resolve_diffuse_crack_index(
                frame_idx=idx,
                cracks_count=len(cracks_list),
                params=diffuse_params,
                frame_meta=frame_meta,
            )
            frame_cracks = cracks_list[crack_idx] if 0 <= crack_idx < len(cracks_list) else []
            if frame_cracks is None:
                frame_cracks = []

            mask_full = np.zeros_like(processed, dtype=bool)
            bounds_list: List[Tuple[int, int, int, int]] = []
            roi_entries: List[Dict[str, Any]] = []
            roi_values: List[np.ndarray] = []

            for crack in frame_cracks:
                geom = self._diffuse_roi_geometry(
                    processed,
                    crack,
                    dx=diffuse_params["diffuse_dx"],
                    dy=diffuse_params["diffuse_dy"],
                )
                if geom is None:
                    continue
                preprocessed = self._diffuse_prethreshold_image(
                    geom["patch"],
                    params=diffuse_params,
                    avg_crack_width_px=self.owner.specimen.avg_crack_width_px,
                )
                closed = preprocessed["closed"]
                roi_entries.append(
                    {
                        "geom": geom,
                        "closed": closed,
                        "floor_mask": preprocessed["floor_mask"],
                        "hard_floor_eff": preprocessed["hard_floor_eff"],
                    }
                )
                closed_sample = closed
                if diffuse_params["threshold_downsample"] > 1:
                    closed_sample = closed_sample[:: diffuse_params["threshold_downsample"], :: diffuse_params["threshold_downsample"]]
                roi_values.append(closed_sample.reshape(-1))

            if roi_values:
                values = np.concatenate(roi_values)
            else:
                values = np.array([], dtype=np.float32)
            frame_threshold = self._compute_frame_diffuse_threshold(
                values,
                max_samples=diffuse_params["threshold_max_samples"],
            )

            for entry in roi_entries:
                roi_mask, bounds = self._diffuse_mask_from_preprocessed(
                    geom=entry["geom"],
                    closed=entry["closed"],
                    floor_mask=entry["floor_mask"],
                    threshold=frame_threshold,
                    params=diffuse_params,
                    avg_crack_width_px=self.owner.specimen.avg_crack_width_px,
                )
                if roi_mask.size == 0:
                    continue
                y_lo, y_hi, x_lo, x_hi = bounds
                mask_full[y_lo:y_hi, x_lo:x_hi] |= roi_mask
                bounds_list.append(bounds)

            if prev_latched is not None:
                mask_full |= prev_latched
            prev_latched = mask_full.copy()

            frame_key = f"frame_{idx:04d}"
            diffuse_masks[frame_key] = mask_full

            if save_overlays and raw_stack is not None:
                raw_frame = _ensure_uint8(raw_stack[idx])
                overlay_dir = self.owner.specimen.results_dir(overlay_dirname, "diffuse", "overlays")
                overlay_path = overlay_dir / f"diffuse_overlay_{idx:04d}.png"
                _save_diffuse_overlay(raw_frame, mask_full, overlay_path, cracks=frame_cracks)

            if debug_payloads is not None:
                hard_floor_values = [
                    float(val)
                    for val in (entry.get("hard_floor_eff") for entry in roi_entries)
                    if val is not None
                ]
                debug_payloads["frames"][frame_key] = {
                    "crack_count": len(frame_cracks),
                    "crack_idx_used": int(crack_idx),
                    "reference_window": [int(ref_start), int(ref_end)],
                    "roi_bounds": bounds_list,
                    "threshold": frame_threshold,
                    "hard_floor_eff_min": (None if not hard_floor_values else float(np.min(hard_floor_values))),
                    "hard_floor_eff_max": (None if not hard_floor_values else float(np.max(hard_floor_values))),
                }

            _progress_update("diffuse_delamination", idx + 1, total_frames, progress_state)

        _progress_done("diffuse_delamination", total_frames, progress)

        return {"masks": diffuse_masks, "debug": debug_payloads}

    def _diffuse_delamination_region_overrides(
        self,
        *,
        cracks: CrackInput,
        processed_cache_paths: Optional[List[Path]] = None,
        save_overlays: bool = False,
        overlay_dirname: str = "delamination",
        max_frames: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        progress: bool = False,
        crack_coordinate_space: str = "middle",
    ) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, Any]]]:
        """Diffuse detection path that prioritizes explicit upper/middle/lower stacks."""
        crack_input_frames = _crack_input_frame_count(cracks)
        cracks_list = _coerce_cracks_by_frame(cracks, crack_input_frames)
        if not cracks_list:
            raise ValueError("Diffuse delamination requires at least one crack frame.")

        upper_stack, middle_stack, lower_stack, raw_stack, total_frames = _fetch_region_override_stacks(
            self.owner,
            domain="diffuse",
            max_frames=max_frames,
            extra_frame_counts={"cracks": len(cracks_list)},
        )

        diffuse_params = self._resolve_diffuse_params(params)
        if diffuse_params.get("reference_mode") is None:
            ref_from_cache = _reference_settings_from_cache_paths(processed_cache_paths)
            diffuse_params["reference_mode"] = ref_from_cache["reference_mode"]
            diffuse_params["reference_window"] = ref_from_cache["reference_window"]
            diffuse_params["reference_skip"] = ref_from_cache["reference_skip"]

        middle_key = f"diffuse_middle_auto_{_result_key_token(self.owner.interface.name)}"
        middle_cache_paths = self.owner.preprocess_stack_to_disk(
            middle_stack,
            key=middle_key,
            max_frames=total_frames,
            cache_dirname="Preprocessor_cache",
            history_mode="running",
            history_window_size=None,
            reference_mode=str(diffuse_params.get("reference_mode") or "static"),
            reference_window=int(diffuse_params.get("reference_window") or 1),
            reference_skip=int(diffuse_params.get("reference_skip") or 0),
            progress=progress,
        )["cache_paths"]

        diffuse_masks: Dict[str, np.ndarray] = {}
        debug_payloads: Optional[Dict[str, Any]] = {"frames": {}} if debug else None
        if debug_payloads is not None:
            debug_payloads["params"] = diffuse_params
            debug_payloads["threshold_strategy"] = "kmeans"
            debug_payloads["threshold_mode"] = "per_frame_roi_union"

        prev_latched_middle: Optional[np.ndarray] = None
        progress_state = _progress_init("diffuse_delamination", total_frames, progress)

        for idx, processed_middle, frame_meta in self.owner.iter_preprocessed_cache_with_metadata(middle_cache_paths):
            if idx >= total_frames:
                break

            upper_h = int(np.asarray(_ensure_uint8(upper_stack[idx])).shape[0])
            middle_h = int(np.asarray(processed_middle).shape[0])
            lower_h = int(np.asarray(_ensure_uint8(lower_stack[idx])).shape[0])
            width = int(np.asarray(processed_middle).shape[1])

            crack_idx, ref_start, ref_end = self._resolve_diffuse_crack_index(
                frame_idx=idx,
                cracks_count=len(cracks_list),
                params=diffuse_params,
                frame_meta=frame_meta,
            )
            frame_cracks = cracks_list[crack_idx] if 0 <= crack_idx < len(cracks_list) else []

            mask_middle = np.zeros_like(processed_middle, dtype=bool)
            bounds_list: List[Tuple[int, int, int, int]] = []
            roi_entries: List[Dict[str, Any]] = []
            roi_values: List[np.ndarray] = []

            for crack in frame_cracks:
                geom = self._diffuse_roi_geometry(
                    processed_middle,
                    crack,
                    dx=diffuse_params["diffuse_dx"],
                    dy=diffuse_params["diffuse_dy"],
                )
                if geom is None:
                    continue
                preprocessed = self._diffuse_prethreshold_image(
                    geom["patch"],
                    params=diffuse_params,
                    avg_crack_width_px=self.owner.specimen.avg_crack_width_px,
                )
                closed = preprocessed["closed"]
                roi_entries.append(
                    {
                        "geom": geom,
                        "closed": closed,
                        "floor_mask": preprocessed["floor_mask"],
                        "hard_floor_eff": preprocessed["hard_floor_eff"],
                    }
                )
                closed_sample = closed
                if diffuse_params["threshold_downsample"] > 1:
                    closed_sample = closed_sample[
                        :: diffuse_params["threshold_downsample"],
                        :: diffuse_params["threshold_downsample"],
                    ]
                roi_values.append(closed_sample.reshape(-1))

            if roi_values:
                values = np.concatenate(roi_values)
            else:
                values = np.array([], dtype=np.float32)
            threshold = self._compute_frame_diffuse_threshold(
                values,
                max_samples=diffuse_params["threshold_max_samples"],
            )

            for entry in roi_entries:
                roi_mask, bounds = self._diffuse_mask_from_preprocessed(
                    geom=entry["geom"],
                    closed=entry["closed"],
                    floor_mask=entry["floor_mask"],
                    threshold=threshold,
                    params=diffuse_params,
                    avg_crack_width_px=self.owner.specimen.avg_crack_width_px,
                )
                if roi_mask.size == 0:
                    continue
                y_lo, y_hi, x_lo, x_hi = bounds
                if y_hi > y_lo and x_hi > x_lo:
                    mask_middle[y_lo:y_hi, x_lo:x_hi] |= roi_mask
                bounds_list.append((y_lo + upper_h, y_hi + upper_h, x_lo, x_hi))

            if prev_latched_middle is None:
                prev_latched_middle = np.zeros_like(mask_middle, dtype=bool)
            prev_latched_middle = np.logical_or(prev_latched_middle, mask_middle)

            full_shape = (upper_h + middle_h + lower_h, width)
            mask_full = np.zeros(full_shape, dtype=bool)
            mask_full[upper_h:upper_h + middle_h, :] = prev_latched_middle

            frame_key = f"frame_{idx:04d}"
            diffuse_masks[frame_key] = mask_full

            if save_overlays:
                raw_frame = _region_override_raw_frame(
                    raw_stack, idx, mask_full.shape[:2], upper_stack[idx], middle_stack[idx], lower_stack[idx]
                )
                overlay_cracks = self.owner._cracks_for_full_overlay(
                    frame_cracks,
                    shift=(crack_coordinate_space == "middle"),
                    upper_height=upper_h,
                )
                overlay_dir = self.owner.specimen.results_dir(overlay_dirname, "diffuse", "overlays")
                overlay_path = overlay_dir / f"diffuse_overlay_{idx:04d}.png"
                _save_diffuse_overlay(raw_frame, mask_full, overlay_path, cracks=overlay_cracks)

            if debug_payloads is not None:
                debug_payloads["frames"][frame_key] = {
                    "threshold": threshold,
                    "bounds": bounds_list,
                    "roi_count": len(roi_entries),
                    "crack_frame_index": int(crack_idx),
                    "reference_window": [int(ref_start), int(ref_end)],
                    "frame_meta": frame_meta,
                }

            _progress_update("diffuse_delamination", idx + 1, total_frames, progress_state)

        _progress_done("diffuse_delamination", total_frames, progress)
        return {"masks": diffuse_masks, "debug": debug_payloads}

    def _resolve_diffuse_params(self, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge diffuse defaults with optional user-provided overrides.

        ``hard_floor`` is expressed as an intensity fraction in normalized space
        (``0.0``-``1.0``). The default ``0.90`` is chosen after internal
        delamination tuning; for reference, Glud/Bender-style crack workflows are
        often configured around ``0.96``.
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
                params = dict(params)
                params["window_diffuse"] = params["window_edge"]
            resolved.update(params)
        resolved["diffuse_dx"] = float(resolved["diffuse_dx"])
        resolved["diffuse_dy"] = float(resolved["diffuse_dy"])
        resolved["threshold_max_samples"] = max(1, int(resolved["threshold_max_samples"]))
        resolved["threshold_downsample"] = max(1, int(resolved["threshold_downsample"]))
        resolved["window_diffuse"] = _resolve_pair(resolved["window_diffuse"], name="window_diffuse", caster=int)
        resolved["gaussian_filters"] = _resolve_pair(resolved["gaussian_filters"], name="gaussian_filters", caster=float)
        resolved["scale_min"] = float(resolved["scale_min"])
        resolved["scale_max"] = float(resolved["scale_max"])
        resolved["scale_min_percentile"] = _resolve_optional_float(resolved["scale_min_percentile"])
        resolved["scale_max_percentile"] = _resolve_optional_float(resolved["scale_max_percentile"])
        hard_floor = resolved.get("hard_floor")
        resolved["hard_floor"] = _resolve_hard_floor_ratio(hard_floor)
        resolved["post_threshold_closing_px"] = max(0, int(resolved.get("post_threshold_closing_px", 4)))
        resolved["post_threshold_closing_scale"] = _resolve_pos_scale(resolved.get("post_threshold_closing_scale"))
        policy = str(resolved.get("crack_frame_policy", "reference_midpoint")).strip().lower()
        if policy not in DIFFUSE_CRACK_FRAME_POLICIES:
            allowed = ", ".join(DIFFUSE_CRACK_FRAME_POLICIES)
            raise ValueError(f"crack_frame_policy must be one of: {allowed}")
        resolved["crack_frame_policy"] = policy
        if resolved.get("reference_mode") is not None:
            resolved["reference_mode"] = str(resolved["reference_mode"])
        if resolved.get("reference_window") is not None:
            resolved["reference_window"] = max(1, int(resolved["reference_window"]))
        if resolved.get("reference_skip") is not None:
            resolved["reference_skip"] = max(0, int(resolved["reference_skip"]))
        return resolved

    def _resolve_diffuse_crack_index(
        self,
        *,
        frame_idx: int,
        cracks_count: int,
        params: Dict[str, Any],
        frame_meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, int, int]:
        """Resolve crack-frame index aligned to preprocessing reference metadata."""
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
            if policy == "reference_midpoint":
                anchor_idx = int(
                    frame_meta.get(
                        "ref_anchor_idx",
                        _reference_anchor_index(
                            frame_idx,
                            start_idx=start_idx,
                            end_idx=end_idx,
                            policy=policy,
                        ),
                    )
                )
            else:
                anchor_idx = _reference_anchor_index(
                    frame_idx,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    policy=policy,
                )
        else:
            reference_mode_raw = params.get("reference_mode")
            if reference_mode_raw is None:
                start_idx, end_idx = int(frame_idx), int(frame_idx) + 1
            else:
                reference_mode = str(reference_mode_raw)
                reference_window = int(params.get("reference_window") or 1)
                reference_skip = int(params.get("reference_skip") or 0)
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
                policy=policy,
            )

        if cracks_count <= 0:
            return int(frame_idx), int(start_idx), int(end_idx)

        anchor_idx = max(0, min(int(anchor_idx), int(cracks_count) - 1))
        return int(anchor_idx), int(start_idx), int(end_idx)

    def _resolve_diffuse_frame_count(
        self,
        *,
        processed_stack: Optional[Sequence[np.ndarray]],
        processed_cache_paths: Optional[Sequence[Path]],
        cracks: Sequence[np.ndarray],
        max_frames: Optional[int],
    ) -> int:
        """Determine the effective number of frames for diffuse processing."""
        if processed_stack is not None:
            total = len(processed_stack)
        elif processed_cache_paths is not None:
            total = len(processed_cache_paths)
        else:
            total = 0
        total = min(total, len(cracks)) if cracks else total
        if max_frames is not None:
            total = min(total, max_frames)
        return total

    def _compute_frame_diffuse_threshold(
        self,
        values: np.ndarray,
        *,
        max_samples: int,
    ) -> float:
        """Compute one threshold from the union of ROI values in a frame."""
        values = np.asarray(values).reshape(-1)
        if values.size == 0:
            return 0.5
        if values.size > max_samples:
            stride = max(1, values.size // max_samples)
            values = values[::stride]
        fallback = _safe_otsu_threshold(values)
        return self.owner._kmeans_threshold(values, fallback)

    def _diffuse_roi_geometry(
        self,
        image: np.ndarray,
        crack: np.ndarray,
        *,
        dx: float,
        dy: float,
    ) -> Optional[Dict[str, Any]]:
        """Build a local crack-aligned ROI and mapping metadata."""
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
        """Apply a pre-computed ROI affine geometry to a different frame.

        Re-uses the ``matrix`` and ``offset`` from :meth:`_diffuse_roi_geometry`
        to sample the same physical region from a frame other than the one used
        to build the geometry.  Useful for extracting the current-frame ROI with
        the same coordinate frame as the baseline ROI.
        """
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
        """Extract crack-aligned ROIs from baseline and current frames.

        Returns a 4-tuple ``(roi_ratio_u8, roi_baseline_u8, roi_current_u8, geom)``:

        - *roi_ratio_u8*: per-pixel ratio ``clip(current / baseline, 0, 1)`` as
          ``uint8``, ready for :meth:`_diffuse_prethreshold_image`.  Values near
          0 (dark) indicate damage; values near 255 (bright) indicate no change.
          This polarity is consistent with the dark-tail thresholding in
          :meth:`_diffuse_mask_from_preprocessed` (``closed < threshold``) and
          with the ``hard_floor`` semantics that suppress background pixels.
        - *roi_baseline_u8*: the baseline ROI patch (``uint8``).
        - *roi_current_u8*: the current-frame ROI sampled with the same affine
          geometry as the baseline (``uint8``).
        - *geom*: the affine geometry dict produced by :meth:`_diffuse_roi_geometry`.

        The geometry is derived from *crack_segment* on *baseline_frame* so both
        frames cover the same physical region.

        Returns ``None`` if the geometry cannot be built (degenerate segment or
        segment outside frame bounds).
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
        """Apply diffuse pre-threshold filtering and intensity scaling."""
        img_uint8 = _ensure_uint8(image)
        wy, wx = max(1, int(params["window_diffuse"][0])), max(1, int(params["window_diffuse"][1]))
        filtered_max = ndi.maximum_filter(img_uint8, size=(wy, wx), mode="reflect")
        filtered_min = ndi.minimum_filter(filtered_max, size=(wy, wx), mode="reflect")
        sharpened = unsharp_mask(
            filtered_min,
            radius=float(avg_crack_width_px),
            amount=2.0,
            preserve_range=True,
        )
        smoothed = ndi.gaussian_filter(sharpened, params["gaussian_filters"])

        scale_min = float(params.get("scale_min", 0))
        scale_max = float(params.get("scale_max", 255))
        pct_min = params.get("scale_min_percentile")
        pct_max = params.get("scale_max_percentile")
        if pct_min is not None and pct_max is not None:
            p_min = float(np.percentile(smoothed, float(pct_min)))
            p_max = float(np.percentile(smoothed, float(pct_max)))
            if np.isfinite(p_min) and np.isfinite(p_max) and p_max > p_min:
                scale_min, scale_max = p_min, p_max

        if scale_max > scale_min:
            constant_scaled = np.clip((smoothed.astype(np.float32) - scale_min) / (scale_max - scale_min), 0.0, 1.0)
        else:
            constant_scaled = np.zeros_like(smoothed, dtype=np.float32)

        hard_floor = params.get("hard_floor")
        hard_floor_eff: Optional[float]
        if hard_floor is None:
            hard_floor_eff = None
            floor_mask = np.ones_like(smoothed, dtype=bool)
        else:
            hard_floor_eff = float(hard_floor)
            smoothed_norm = smoothed.astype(np.float32) / 255.0
            floor_mask = smoothed_norm <= hard_floor_eff

        closed = constant_scaled

        return {
            "filtered_max": filtered_max,
            "filtered_min": filtered_min,
            "sharpened": sharpened,
            "smoothed": smoothed,
            "constant_scaled": constant_scaled,
            "closed": closed,
            "floor_mask": floor_mask,
            "hard_floor_eff": None if hard_floor_eff is None else float(hard_floor_eff),
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
        """Threshold and map one ROI mask back to full-frame coordinates."""
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
        """Track crack segments across frames and run baseline-normalised diffuse analysis.

        This method encapsulates the full crack-tracking + vanishing-crack diffuse
        + per-track diffuse history workflow.  It consumes preprocessed frames and
        normalised crack detections (see :func:`.crack_tracking.normalize_detections`),
        runs the tracking loop, computes baseline-normalised ROI ratios for each
        matched track position and for every terminated (vanishing) crack, and
        assembles per-frame binary damage masks.

        Parameters
        ----------
        processed_frames:
            Preprocessed grayscale frames in the same order as *selected_indices*.
        crack_frames:
            Per-frame lists of :class:`.crack_tracking.CrackDetection` objects,
            aligned with *selected_indices*.
        selected_indices:
            Absolute frame indices corresponding to entries in the two lists above.
        avg_crack_width_px:
            Average crack width in pixels; passed to the diffuse preprocessing step.
        diffuse_params:
            Parameter dict forwarded to :meth:`_diffuse_prethreshold_image` and
            :meth:`_diffuse_mask_from_preprocessed`.  Required keys: ``diffuse_dx``,
            ``diffuse_dy``, ``window_diffuse``, ``gaussian_filters``.  Optional keys:
            ``hard_floor``, ``scale_min_percentile``, ``scale_max_percentile``,
            ``post_threshold_closing_px``, ``threshold_downsample``,
            ``threshold_max_samples``.
        max_center_px:
            Maximum centre-to-centre distance for track–detection assignment.
            Defaults to ``max(12.0, 2.5 * avg_crack_width_px)``.
        max_angle_deg:
            Maximum angular difference (degrees) for a valid assignment.
        max_cost:
            Cost ceiling for :func:`.crack_tracking.match_tracks`.
        return_intermediates:
            If ``True``, include large intermediate arrays (``roi_ratio_u8``,
            ``roi_baseline_u8``, ``roi_current_u8``, ``pre``, ``mask_bbox``,
            ``bounds``) in each ``vanishing_stats`` / ``diffuse_stats`` entry.
            Defaults to ``False`` to keep memory usage low in batch runs.

        Returns
        -------
        dict with keys:

        ``"tracks"`` : list[CrackTrack]
            All tracks in creation order (active at the end → ``active=False`` before
            return).
        ``"events"`` : list[dict]
            One entry per frame-level event (new, matched, terminated).
        ``"vanishing_stats"`` : list[dict]
            Diffuse results for every terminated track.
        ``"diffuse_stats"`` : list[dict]
            Diffuse results for every matched-track position.
        ``"frame_masks"`` : dict[int, np.ndarray]
            ``{frame_abs: bool_mask}`` assembled from all per-ROI masks.
        """
        from deladect.detection.crack_tracking import CrackTrack, match_tracks

        dx = float(diffuse_params["diffuse_dx"])
        dy = float(diffuse_params["diffuse_dy"])
        ds = int(diffuse_params.get("threshold_downsample", 2))
        max_samples = int(diffuse_params.get("threshold_max_samples", 400_000))
        max_center_px_eff = float(
            max_center_px if max_center_px is not None else max(12.0, 2.5 * avg_crack_width_px)
        )

        frame_pos_by_abs = {int(a): i for i, a in enumerate(selected_indices)}

        tracks: List[CrackTrack] = []
        events: List[Dict[str, Any]] = []
        vanishing_stats: List[Dict[str, Any]] = []
        diffuse_stats: List[Dict[str, Any]] = []
        frame_masks: Dict[int, np.ndarray] = {}
        frame_detection_track_ids: Dict[int, List[Optional[int]]] = {}
        next_track_id = 1

        def _splat_mask(
            full_mask: np.ndarray,
            mask_bbox: np.ndarray,
            bounds: Tuple[int, int, int, int],
        ) -> None:
            H, W = full_mask.shape[:2]
            y_lo, y_hi, x_lo, x_hi = bounds
            out_y0, out_y1 = max(0, y_lo), min(H, y_hi)
            out_x0, out_x1 = max(0, x_lo), min(W, x_hi)
            src_y0 = out_y0 - y_lo
            src_x0 = out_x0 - x_lo
            full_mask[out_y0:out_y1, out_x0:out_x1] |= mask_bbox[
                src_y0: src_y0 + (out_y1 - out_y0),
                src_x0: src_x0 + (out_x1 - out_x0),
            ]

        for frame_abs, detections, proc_frame in zip(
            selected_indices, crack_frames, processed_frames
        ):
            frame_abs = int(frame_abs)

            matched, unmatched_tracks_idx, unmatched_det_idx = match_tracks(
                tracks,
                detections,
                max_center_px=max_center_px_eff,
                max_angle_deg=max_angle_deg,
                max_cost=max_cost,
            )

            # ------------------------------------------------------------------
            # Terminate unmatched tracks and run vanishing-crack diffuse check
            # ------------------------------------------------------------------
            for ti in unmatched_tracks_idx:
                track = tracks[ti]
                track.active = False
                events.append({
                    "frame_abs": frame_abs,
                    "track_id": int(track.track_id),
                    "status": "terminated",
                })

                has_matched = any(h["status"] == "matched" for h in track.history)
                if not has_matched or track.first_frame_abs == frame_abs:
                    continue
                base_pos = frame_pos_by_abs.get(int(track.baseline_frame_abs))
                if base_pos is None:
                    continue

                v_result = self._diffuse_baseline_normalized_roi(
                    processed_frames[base_pos],
                    proc_frame,
                    track.last_segment,
                    dx=dx,
                    dy=dy,
                )
                if v_result is None:
                    continue

                v_roi_ratio_u8, _v_baseline_u8, _v_current_u8, v_geom = v_result
                v_pre = self._diffuse_prethreshold_image(
                    v_roi_ratio_u8,
                    params=diffuse_params,
                    avg_crack_width_px=avg_crack_width_px,
                )
                v_threshold = self._compute_frame_diffuse_threshold(
                    v_pre["closed"][::ds, ::ds].reshape(-1),
                    max_samples=max_samples,
                )
                v_mask_bbox, v_bounds = self._diffuse_mask_from_preprocessed(
                    geom=v_geom,
                    closed=v_pre["closed"],
                    floor_mask=v_pre["floor_mask"],
                    threshold=v_threshold,
                    params=diffuse_params,
                    avg_crack_width_px=avg_crack_width_px,
                )
                H, W = proc_frame.shape[:2]
                if frame_abs not in frame_masks:
                    frame_masks[frame_abs] = np.zeros((H, W), dtype=bool)
                _splat_mask(frame_masks[frame_abs], v_mask_bbox, v_bounds)

                v_entry: Dict[str, Any] = {
                    "track_id": int(track.track_id),
                    "termination_frame_abs": frame_abs,
                    "baseline_frame_abs": int(track.baseline_frame_abs),
                    "threshold": float(v_threshold),
                    "mask_frac": float(np.mean(v_mask_bbox)),
                    "floor_mask_frac": float(np.mean(v_pre["floor_mask"])),
                }
                if return_intermediates:
                    v_entry.update({
                        "roi_ratio_u8": v_roi_ratio_u8,
                        "roi_baseline_u8": _v_baseline_u8,
                        "roi_current_u8": _v_current_u8,
                        "pre": v_pre,
                        "mask_bbox": v_mask_bbox,
                    })
                vanishing_stats.append(v_entry)

            det_to_track: Dict[int, int] = {di: ti for ti, di in matched.items()}

            # ------------------------------------------------------------------
            # Spawn new tracks for unmatched detections
            # ------------------------------------------------------------------
            for di in unmatched_det_idx:
                det = detections[di]
                track = CrackTrack(
                    track_id=next_track_id,
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
                next_track_id += 1

            # ------------------------------------------------------------------
            # Update matched tracks and run per-track diffuse
            # ------------------------------------------------------------------
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
                track.history.append({
                    "frame_abs": frame_abs,
                    "status": "matched",
                    "growth_ratio": float(growth_ratio),
                })

                # Per-track diffuse analysis
                base_pos = frame_pos_by_abs.get(int(track.baseline_frame_abs))
                if base_pos is None:
                    continue
                f_pos = frame_pos_by_abs.get(frame_abs)
                if f_pos is None:
                    continue

                base_seg = track.baseline_segment
                result = self._diffuse_baseline_normalized_roi(
                    processed_frames[base_pos],
                    proc_frame,
                    base_seg,
                    dx=dx,
                    dy=dy,
                )
                if result is None:
                    continue

                roi_ratio_u8, roi_baseline_u8, roi_current_u8, geom = result
                pre = self._diffuse_prethreshold_image(
                    roi_ratio_u8,
                    params=diffuse_params,
                    avg_crack_width_px=avg_crack_width_px,
                )
                threshold = self._compute_frame_diffuse_threshold(
                    pre["closed"][::ds, ::ds].reshape(-1),
                    max_samples=max_samples,
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
                    frame_masks[frame_abs] = np.zeros(proc_frame.shape[:2], dtype=bool)
                _splat_mask(frame_masks[frame_abs], mask_bbox, bounds)

                d_entry: Dict[str, Any] = {
                    "track_id": int(track.track_id),
                    "frame_abs": frame_abs,
                    "baseline_frame_abs": int(track.baseline_frame_abs),
                    "threshold": float(threshold),
                    "floor_mask_frac": float(np.mean(pre["floor_mask"])),
                    "mask_frac": float(np.mean(mask_bbox)),
                }
                if return_intermediates:
                    d_entry.update({
                        "roi_ratio_u8": roi_ratio_u8,
                        "roi_baseline_u8": roi_baseline_u8,
                        "roi_current_u8": roi_current_u8,
                        "pre": pre,
                        "mask_bbox": mask_bbox,
                        "bounds": bounds,
                    })
                diffuse_stats.append(d_entry)

            frame_detection_track_ids[frame_abs] = [
                tracks[det_to_track[di]].track_id if di in det_to_track else None
                for di in range(len(detections))
            ]

        # Close all remaining active tracks
        for track in tracks:
            if track.active:
                track.active = False

        return {
            "tracks": tracks,
            "events": events,
            "vanishing_stats": vanishing_stats,
            "diffuse_stats": diffuse_stats,
            "frame_masks": frame_masks,
            "frame_detection_track_ids": frame_detection_track_ids,
        }
