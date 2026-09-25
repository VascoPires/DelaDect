"""Edge delamination detection: :class:`EdgeDetector`."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import closing, disk

from deladect.io.delamination import save_mask_bundle, store_interface_masks
from deladect.specimen import Interface

from ._common import (
    _Progress,
    _auto_preprocess_cache_paths,
    _ensure_uint8,
    _fetch_region_override_stacks,
    _hard_floor_mask,
    _kmeans_split,
    _minmax_otsu_threshold,
    _percentile_range,
    _region_override_raw_frame,
    _resolve_hard_floor_ratio,
    _resolve_optional_float,
    _resolve_pair,
    _resolve_pos_scale,
    _result_key_token,
    _scale_to_unit,
    _smooth_for_threshold,
)
from ._overlays import (
    _interface_legend_label,
    _resolve_multi_interface_colors,
    _save_edge_debug_frame,
    _save_edge_multi_debug_panels,
    _save_edge_overlay,
    _save_multi_level_overlay,
)
from ._preprocess import _reference_settings_from_cache_paths

if TYPE_CHECKING:
    from .core import DelaminationDetector

_SIDES = ("upper", "lower")

_PRIMARY_DEBUG_KEYS: Tuple[str, ...] = (
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


def _rebuild_edge_connected_directional(
    mask: np.ndarray,
    *,
    seed_depth: int,
    lateral_drift_px: int,
) -> np.ndarray:
    """Keep the part of ``mask`` that grows down from the first ``seed_depth`` rows.

    A pixel is kept only if a kept pixel in the row above lies within
    ``lateral_drift_px`` columns, so growth can't jump over empty rows.
    """
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim != 2:
        raise ValueError("Directional edge reconstruction expects a 2D mask.")

    height = mask_bool.shape[0]
    if height == 0:
        return mask_bool

    seed_rows = min(max(1, int(seed_depth)), height)
    drift = max(0, int(lateral_drift_px))

    rebuilt = np.zeros_like(mask_bool, dtype=bool)
    rebuilt[:seed_rows, :] = mask_bool[:seed_rows, :]
    if drift == 0:
        for row in range(seed_rows, height):
            rebuilt[row, :] = mask_bool[row, :] & rebuilt[row - 1, :]
        return rebuilt

    support_structure = np.ones((2 * drift + 1,), dtype=bool)
    for row in range(seed_rows, height):
        support = ndi.binary_dilation(rebuilt[row - 1, :], structure=support_structure)
        rebuilt[row, :] = mask_bool[row, :] & support
    return rebuilt


def _rebuild_edge_connected_columnwise(mask: np.ndarray, *, seed_depth: int) -> np.ndarray:
    """Like the directional version with no drift: each column grows on its own."""
    return _rebuild_edge_connected_directional(mask, seed_depth=seed_depth, lateral_drift_px=0)


def _remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove connected regions smaller than ``min_size`` pixels."""
    cleaned = np.asarray(mask, dtype=bool)
    if max(0, int(min_size)) <= 1:
        return cleaned

    labels, count = ndi.label(cleaned)
    if count <= 0:
        return cleaned
    keep = np.bincount(labels.ravel()) >= int(min_size)
    keep[0] = False
    return keep[labels]


def _filter_specimen_edge_connected(mask: np.ndarray) -> np.ndarray:
    """Keep only the connected regions of ``mask`` that touch row 0, the specimen edge."""
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return mask
    labeled, _ = ndi.label(mask)
    edge_labels = set(np.unique(labeled[0, :])) - {0}
    if not edge_labels:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(labeled, list(edge_labels))


def _assemble_full_mask(upper: np.ndarray, lower_flipped: np.ndarray, middle_height: int = 0) -> np.ndarray:
    """Stack the upper half, ``middle_height`` empty rows and the lower half flipped back."""
    upper = np.asarray(upper, dtype=bool)
    lower = np.flipud(np.asarray(lower_flipped, dtype=bool))
    full = np.zeros((upper.shape[0] + middle_height + lower.shape[0], upper.shape[1]), dtype=bool)
    full[: upper.shape[0], :] = upper
    full[upper.shape[0] + middle_height :, :] = lower
    return full


def _primary_debug_payload(
    processed: np.ndarray,
    upper_result: Dict[str, Any],
    lower_result: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "processed": processed,
        "upper": {key: upper_result[key] for key in _PRIMARY_DEBUG_KEYS},
        "lower": {key: lower_result[key] for key in _PRIMARY_DEBUG_KEYS},
    }


class _EdgeFrame(NamedTuple):
    """The two edge halves of one frame, as passed to :meth:`EdgeDetector._process_edge_slice`."""

    idx: int
    upper: np.ndarray
    lower: np.ndarray  # not yet flipped
    middle_height: int  # rows between the halves (region-override mode only)
    raw_frame: Callable[[Tuple[int, int]], np.ndarray]  # display frame for a mask of the given shape


class EdgeDetector:
    """Edge delamination detection.

    :meth:`detect_primary` detects one interface; :meth:`detect_edge_multi`
    splits the damage between several interfaces.
    """

    def __init__(self, owner: DelaminationDetector) -> None:
        """Create an edge detector for the parent :class:`DelaminationDetector`."""
        self.owner = owner

    def detect_primary(
        self,
        *,
        processed_cache_paths: Optional[List[Path]] = None,
        processed_stack: Optional[List[np.ndarray]] = None,
        save_overlays: bool = False,
        overlay_dirname: str = "delamination",
        overlay_view: str = "mask",
        max_frames: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        progress: bool = False,
        save_debug_outputs: bool = False,
        debug_dirname: str = "edge_accumulation_debug",
    ) -> Dict[str, Any]:
        """Detect edge delamination in every frame.

        Each frame is split into an upper and a lower half, and the lower
        half is flipped so that both grow from row 0. Damage must connect to
        the specimen edge, and masks are latched over time.

        Parameters
        ----------
        processed_cache_paths, processed_stack:
            Preprocessed frames, as cache files or arrays. If neither is
            given, the full stack is preprocessed with
            ``reference_mode="static"``. Frames you pass should also use a
            static reference.
        save_overlays:
            Save an edge overlay per frame.
        overlay_dirname:
            Output folder under the specimen results.
        overlay_view:
            ``"mask"``, ``"line"`` or ``"both"``.
        max_frames:
            Process only the first ``max_frames`` frames.
        params:
            Edge parameter overrides.
        debug:
            Return intermediate images and thresholds per frame.
        progress:
            Print progress.
        save_debug_outputs:
            Save every intermediate image to disk.
        debug_dirname:
            Folder for those images.

        Returns
        -------
        dict[str, Any]
            ``{"masks": {frame_key: mask}, "debug": {...} or None}``, with
            keys like ``"frame_0003"``.
        """
        if processed_cache_paths and processed_stack:
            raise ValueError("Provide either processed_cache_paths or processed_stack, not both.")
        if overlay_view not in {"mask", "line", "both"}:
            raise ValueError("overlay_view must be one of: 'mask', 'line', 'both'.")

        if self.owner._uses_stack_overrides():
            edge_params = self._resolve_primary_params(params)
            frames, total_frames = self._region_override_frames(processed_cache_paths, max_frames, params, progress)
        else:
            raw_stack = getattr(self.owner.specimen, "image_stack_full", None)
            if save_overlays and raw_stack is None:
                raise ValueError("Cannot save overlays without a full raw image stack.")
            if processed_cache_paths is None and processed_stack is None:
                processed_cache_paths = _auto_preprocess_cache_paths(
                    self.owner,
                    save_overlays=save_overlays,
                    max_frames=max_frames,
                    progress=progress,
                    key_prefix="edge_primary_auto",
                )
            edge_params = self._resolve_primary_params(params)

            if processed_stack is not None:
                processed_iter: Iterable[Tuple[int, np.ndarray]] = enumerate(processed_stack)
                total_frames = len(processed_stack)
            else:
                processed_iter = self.owner.iter_preprocessed_cache(processed_cache_paths)
                total_frames = len(processed_cache_paths)
            if max_frames is not None:
                total_frames = min(total_frames, max_frames)
            frames = self._full_frame_halves(processed_iter, total_frames, raw_stack)

        return self._accumulate_primary(
            frames,
            total_frames,
            edge_params=edge_params,
            save_overlays=save_overlays,
            overlay_dirname=overlay_dirname,
            overlay_view=overlay_view,
            debug=debug,
            progress=progress,
            debug_root=self.owner.specimen.results_dir(debug_dirname) if save_debug_outputs else None,
        )

    @staticmethod
    def _full_frame_halves(
        processed_iter: Iterable[Tuple[int, np.ndarray]],
        total_frames: int,
        raw_stack: Optional[Sequence[np.ndarray]],
    ) -> Iterator[_EdgeFrame]:
        def raw_frame(idx: int, processed: np.ndarray, _shape: Tuple[int, int]) -> np.ndarray:
            return _ensure_uint8(raw_stack[idx]) if raw_stack is not None else processed

        for idx, processed in processed_iter:
            if idx >= total_frames:
                break
            split_row = processed.shape[0] // 2
            yield _EdgeFrame(
                idx,
                processed[:split_row, :],
                processed[split_row:, :],
                0,
                partial(raw_frame, idx, processed),
            )

    def _region_override_frames(
        self,
        processed_cache_paths: Optional[List[Path]],
        max_frames: Optional[int],
        params: Optional[Dict[str, Any]],
        progress: bool,
    ) -> Tuple[Iterator[_EdgeFrame], int]:
        """Preprocess the upper and lower region stacks; return their frames and the frame count."""
        upper_stack, middle_stack, lower_stack, raw_stack, total_frames = _fetch_region_override_stacks(
            self.owner, domain="edge", max_frames=max_frames
        )

        reference = _reference_settings_from_cache_paths(processed_cache_paths)
        reference.update({key: (params or {})[key] for key in reference if key in (params or {})})
        token = _result_key_token(self.owner.interface.name)
        upper_cache_paths, lower_cache_paths = (
            self.owner.preprocess_stack_to_disk(
                stack,
                key=f"edge_{side}_auto_{token}",
                max_frames=total_frames,
                cache_dirname="Preprocessor_cache",
                history_mode="running",
                history_window_size=None,
                reference_mode=str(reference["reference_mode"]),
                reference_window=int(reference["reference_window"]),
                reference_skip=int(reference["reference_skip"]),
                progress=progress,
            )["cache_paths"]
            for side, stack in (("upper", upper_stack), ("lower", lower_stack))
        )

        def raw_frame(idx: int, middle_raw: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
            return _region_override_raw_frame(raw_stack, idx, shape, upper_stack[idx], middle_raw, lower_stack[idx])

        def frames() -> Iterator[_EdgeFrame]:
            upper_iter = self.owner.iter_preprocessed_cache(upper_cache_paths)
            lower_iter = self.owner.iter_preprocessed_cache(lower_cache_paths)
            for (idx, upper_processed), (_, lower_processed) in zip(upper_iter, lower_iter):
                if idx >= total_frames:
                    break
                middle_raw = _ensure_uint8(middle_stack[idx])
                yield _EdgeFrame(
                    idx,
                    _ensure_uint8(upper_processed),
                    _ensure_uint8(lower_processed),
                    int(middle_raw.shape[0]),
                    partial(raw_frame, idx, middle_raw),
                )

        return frames(), total_frames

    def _accumulate_primary(
        self,
        frames: Iterable[_EdgeFrame],
        total_frames: int,
        *,
        edge_params: Dict[str, Any],
        save_overlays: bool,
        overlay_dirname: str,
        overlay_view: str,
        debug: bool,
        progress: bool,
        debug_root: Optional[Path],
    ) -> Dict[str, Any]:
        """Detect edge damage in both halves of every frame and latch the masks."""
        overlay_dir = self.owner.specimen.results_dir(overlay_dirname, "edge", "overlays") if save_overlays else None
        primary_masks: Dict[str, np.ndarray] = {}
        debug_payloads: Optional[Dict[str, Any]] = {} if debug else None
        upper_state: Optional[np.ndarray] = None
        lower_state: Optional[np.ndarray] = None
        progress_log = _Progress("edge_primary", total_frames, progress)

        for frame in frames:
            idx = frame.idx
            lower_flipped = np.flipud(frame.lower)
            upper_result, lower_result = self._process_halves(
                frame.upper, lower_flipped, upper_state, lower_state, edge_params
            )
            upper_state = upper_result["primary_latched"]
            lower_state = lower_result["primary_latched"]

            primary_full = _assemble_full_mask(upper_state, lower_state, frame.middle_height)
            frame_key = f"frame_{idx:04d}"
            primary_masks[frame_key] = primary_full

            if overlay_dir is None and debug_root is None and not debug:
                progress_log.update(idx + 1)
                continue

            processed_full = None
            if debug_root is not None or debug_payloads is not None:
                middle_gap = np.zeros((frame.middle_height, frame.upper.shape[1]), dtype=frame.upper.dtype)
                processed_full = np.vstack([frame.upper, middle_gap, frame.lower])

            if overlay_dir is not None:
                _save_edge_overlay(
                    frame.raw_frame(primary_full.shape[:2]),
                    primary_full,
                    overlay_dir / f"edge_overlay_{idx:04d}.png",
                    view=overlay_view,
                )

            if debug_root is not None:
                frame_dir = debug_root / f"frame_{idx:04d}"
                frame_dir.mkdir(parents=True, exist_ok=True)
                _save_edge_debug_frame(
                    frame_dir=frame_dir,
                    raw_frame=frame.raw_frame(primary_full.shape[:2]),
                    processed=processed_full,
                    upper_slice=frame.upper,
                    lower_slice=lower_flipped,
                    upper_result=upper_result,
                    lower_result=lower_result,
                    lower_latched_unflipped=np.flipud(lower_state),
                    full_latched=primary_full,
                )

            if debug_payloads is not None:
                debug_payloads[frame_key] = _primary_debug_payload(processed_full, upper_result, lower_result)

            progress_log.update(idx + 1)

        progress_log.done()
        return {"masks": primary_masks, "debug": debug_payloads}

    def _process_halves(
        self,
        upper: np.ndarray,
        lower_flipped: np.ndarray,
        upper_prev: Optional[np.ndarray],
        lower_prev: Optional[np.ndarray],
        params: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run :meth:`_process_edge_slice` on the upper half, then on the flipped lower half."""
        avg_crack_width_px = self.owner.specimen.avg_crack_width_px
        upper_result = self._process_edge_slice(
            upper, prev_latched=upper_prev, params=params, avg_crack_width_px=avg_crack_width_px
        )
        lower_result = self._process_edge_slice(
            lower_flipped, prev_latched=lower_prev, params=params, avg_crack_width_px=avg_crack_width_px
        )
        return upper_result, lower_result

    def detect_edge_multi(
        self,
        *,
        interfaces: Sequence[Interface],
        processed_cache_paths: Optional[List[Path]] = None,
        processed_stack: Optional[List[np.ndarray]] = None,
        secondary_cache_paths: Optional[List[Path]] = None,
        save_overlays: bool = False,
        overlay_dirname: str = "delamination",
        save_masks: bool = True,
        masks_dirname: str = "masks",
        max_frames: Optional[int] = None,
        primary_params: Optional[Dict[str, Any]] = None,
        secondary_edge_params: Optional[Dict[str, Any]] = None,
        secondary_params: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        return_masks: bool = True,
        debug: bool = False,
        debug_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Split edge delamination between several interfaces.

        The first interface is detected as in :meth:`detect_primary`. Each
        deeper interface gets the new damage that appears inside the mask of
        the interface above it, as that mask was ``reference_window`` frames
        earlier (default 7). Only damage connected to the specimen edge
        counts.

        Parameters
        ----------
        interfaces:
            Interfaces from the shallowest to the deepest.
        processed_cache_paths, processed_stack:
            Preprocessed frames for the first interface. If neither is given,
            the full stack is preprocessed with a rolling median reference.
            A static-reference cache gives a first-interface mask that
            matches :meth:`DelaminationDetector.detect_both_delaminations`.
        secondary_cache_paths:
            Optional rolling-median cache with the same frames. If given,
            damage for the deeper interfaces is detected on it; otherwise on
            the primary frames.
        save_overlays:
            Save one overlay per frame with a color per interface.
        overlay_dirname:
            Output folder under the specimen results.
        save_masks:
            Save the masks of each interface to ``.npz`` and record them on
            the interface.
        masks_dirname:
            Folder for the mask files.
        max_frames:
            Process only the first ``max_frames`` frames.
        primary_params:
            Edge parameters for the first interface (``window_edge``,
            ``hard_floor``, ``seed_ratio``, ...).
        secondary_edge_params:
            Edge parameters for the ``secondary_cache_paths`` frames.
        secondary_params:
            ``secondary_start_frame``: no damage is given to deeper interfaces
            before this frame. ``secondary_similarity_threshold`` and
            ``min_primary_frac_for_secondary`` are accepted but not used yet.
        params:
            Older single-dict form; used as the base of the other parameter
            dicts.
        return_masks:
            Include the masks in the result.
        debug:
            Include per-interface diagnostics in the result.
        debug_dir:
            Save a debug figure per frame and deeper interface here.

        Returns
        -------
        dict[str, Any]
            ``interfaces`` (key, name, label and color of each),
            ``frame_indices``, ``frame_level_maps`` (deepest interface per
            pixel), ``paths`` and ``params``, plus ``inclusive_masks``,
            ``exclusive_masks`` and ``debug`` when requested.
        """
        if processed_cache_paths and processed_stack:
            raise ValueError("Provide either processed_cache_paths or processed_stack, not both.")

        primary_params = {**(params or {}), **(primary_params or {})}
        if processed_cache_paths is None and processed_stack is None:
            auto_stack = getattr(self.owner.specimen, "image_stack_full", None)
            if auto_stack is None:
                raise ValueError(
                    "detect_edge_multi: no full image stack available for automatic "
                    "preprocessing. Provide processed_cache_paths or processed_stack."
                )
            interface_token = _result_key_token(interfaces[0].name if interfaces else "i0")
            processed_cache_paths = self.owner.preprocess_stack_to_disk(
                auto_stack,
                key=f"edge_multi_auto_{interface_token}",
                max_frames=max_frames,
                cache_dirname="Preprocessor_cache",
                reference_mode="rolling_median",
                reference_window=int(primary_params.get("reference_window", 10)),
                reference_skip=int(primary_params.get("reference_skip", 1)),
            )["cache_paths"]

        interface_list = list(interfaces)
        if not interface_list:
            raise ValueError("detect_edge_multi requires at least one interface.")

        raw_stack = getattr(self.owner.specimen, "image_stack_full", None)
        if save_overlays and raw_stack is None:
            raise ValueError("Cannot save overlays without a full raw image stack.")

        edge_params = self._resolve_primary_params(primary_params)
        multi_params = self._resolve_multi_params({**(params or {}), **(secondary_params or {})})
        # The secondary (rolling-median) cache, when given, supplies the
        # candidate masks for the deeper levels; otherwise the primary pass does.
        secondary_edge_params_resolved = (
            self._resolve_primary_params({**(params or {}), **(secondary_edge_params or {})})
            if secondary_cache_paths is not None
            else None
        )

        if processed_stack is not None:
            processed_iter: Iterable[Tuple[int, np.ndarray]] = enumerate(processed_stack)
        else:
            processed_iter = self.owner.iter_preprocessed_cache(processed_cache_paths)
        secondary_iter = (
            self.owner.iter_preprocessed_cache(secondary_cache_paths) if secondary_cache_paths is not None else None
        )
        keep_debug = debug_dir is not None

        frame_indices: List[int] = []
        split_rows: List[int] = []
        primary_state: Dict[str, Optional[np.ndarray]] = {side: None for side in _SIDES}
        secondary_state: Dict[str, Optional[np.ndarray]] = {side: None for side in _SIDES}
        primary_frames: Dict[str, List[np.ndarray]] = {side: [] for side in _SIDES}
        rolling_frames: Dict[str, List[np.ndarray]] = {side: [] for side in _SIDES}
        candidate_frames: Dict[str, List[np.ndarray]] = {side: [] for side in _SIDES}
        debug_processed: List[np.ndarray] = []
        debug_results: Dict[str, List[Dict[str, Any]]] = {side: [] for side in _SIDES}
        debug_sec_processed: List[np.ndarray] = []
        debug_sec_results: Dict[str, List[Dict[str, Any]]] = {side: [] for side in _SIDES}

        for idx, processed in processed_iter:
            if max_frames is not None and len(frame_indices) >= max(0, int(max_frames)):
                break

            processed_uint8 = _ensure_uint8(processed)
            split_row = processed_uint8.shape[0] // 2
            results = dict(zip(_SIDES, self._process_halves(
                processed_uint8[:split_row, :],
                np.flipud(processed_uint8[split_row:, :]),
                primary_state["upper"],
                primary_state["lower"],
                edge_params,
            )))
            for side in _SIDES:
                primary_state[side] = np.asarray(results[side]["primary_latched"], dtype=bool)
                primary_frames[side].append(primary_state[side].copy())

            frame_indices.append(int(idx))
            split_rows.append(split_row)
            if keep_debug:
                debug_processed.append(processed_uint8.copy())
                for side in _SIDES:
                    debug_results[side].append(results[side])

            if secondary_iter is not None:
                _, sec_processed = next(secondary_iter)
                sec_uint8 = _ensure_uint8(sec_processed)
                candidate_results = dict(zip(_SIDES, self._process_halves(
                    sec_uint8[:split_row, :],
                    np.flipud(sec_uint8[split_row:, :]),
                    secondary_state["upper"],
                    secondary_state["lower"],
                    secondary_edge_params_resolved,
                )))
                for side in _SIDES:
                    secondary_state[side] = np.asarray(candidate_results[side]["primary_latched"], dtype=bool)
                    rolling_frames[side].append(secondary_state[side].copy())
                if keep_debug:
                    debug_sec_processed.append(sec_uint8.copy())
                    for side in _SIDES:
                        debug_sec_results[side].append(candidate_results[side])
            else:
                candidate_results = results
                for side in _SIDES:
                    rolling_frames[side].append(primary_state[side].copy())
            for side in _SIDES:
                candidate_frames[side].append(np.asarray(candidate_results[side]["mask"], dtype=bool))

        if not frame_indices:
            raise ValueError("No processed frames available for multi-interface edge detection.")

        levels: Dict[str, List[List[np.ndarray]]] = {side: [primary_frames[side]] for side in _SIDES}
        debug_levels: Dict[str, Any] = {}
        parent_delay = int((secondary_edge_params_resolved or edge_params).get("reference_window", 7))
        start_frame = multi_params["secondary_start_frame"]

        for level_idx in range(1, len(interface_list)):
            latched: Dict[str, List[np.ndarray]] = {}
            diagnostics: Dict[str, List[Dict[str, Any]]] = {}
            for side in _SIDES:
                latched[side], diagnostics[side] = self._attribute_deeper_level(
                    candidate_frames[side],
                    parent=levels[side][level_idx - 1],
                    frame_indices=frame_indices,
                    parent_delay=parent_delay,
                    start_frame=start_frame,
                    keep_masks=keep_debug,
                )
                levels[side].append(latched[side])

            if debug:
                debug_levels[f"level_{level_idx + 1}"] = diagnostics

            if debug_dir is not None:
                _save_edge_multi_debug_panels(
                    debug_dir=debug_dir,
                    frame_indices=frame_indices,
                    processed_frames=debug_processed,
                    upper_results=debug_results["upper"],
                    lower_results=debug_results["lower"],
                    upper_latched=latched["upper"],
                    lower_latched=latched["lower"],
                    upper_diag=diagnostics["upper"],
                    lower_diag=diagnostics["lower"],
                    split_rows=split_rows,
                    level_idx=level_idx,
                    sec_processed_frames=debug_sec_processed,
                    sec_upper_results=debug_sec_results["upper"],
                    sec_lower_results=debug_sec_results["lower"],
                    upper_rolling_frames=rolling_frames["upper"],
                    lower_rolling_frames=rolling_frames["lower"],
                )

        result_keys = self._build_interface_result_keys(interface_list)
        display_colors = _resolve_multi_interface_colors(interface_list)
        inclusive_masks: Dict[str, Dict[str, np.ndarray]] = {key: {} for key in result_keys}
        exclusive_masks: Dict[str, Dict[str, np.ndarray]] = {key: {} for key in result_keys}
        frame_level_maps: Dict[str, np.ndarray] = {}

        # Deeper levels overwrite shallower ones, so each pixel is labelled with the deepest level reaching it.
        for frame_pos, frame_idx in enumerate(frame_indices):
            frame_key = f"frame_{frame_idx:04d}"
            frame_level: Optional[np.ndarray] = None
            for level, key in enumerate(result_keys, start=1):
                full_mask = _assemble_full_mask(
                    levels["upper"][level - 1][frame_pos], levels["lower"][level - 1][frame_pos]
                )
                inclusive_masks[key][frame_key] = full_mask
                if frame_level is None:
                    frame_level = np.zeros(full_mask.shape, dtype=np.uint8)
                frame_level[full_mask] = np.uint8(level)

            frame_level_maps[frame_key] = frame_level
            for level, key in enumerate(result_keys, start=1):
                exclusive_masks[key][frame_key] = frame_level == np.uint8(level)

        paths: Dict[str, Any] = {"inclusive_masks": {}, "exclusive_masks": {}, "overlays": None}

        if save_masks:
            masks_root = self.owner.specimen.results_dir(overlay_dirname, "edge_multi", masks_dirname)
            for key, interface in zip(result_keys, interface_list):
                inclusive_path = save_mask_bundle(inclusive_masks[key], masks_root / f"{key}_inclusive.npz")
                exclusive_path = save_mask_bundle(exclusive_masks[key], masks_root / f"{key}_exclusive.npz")
                paths["inclusive_masks"][key] = str(inclusive_path)
                paths["exclusive_masks"][key] = str(exclusive_path)
                store_interface_masks(interface, primary_path=inclusive_path, secondary_path=exclusive_path)

        labels = [_interface_legend_label(self.owner.specimen, interface) for interface in interface_list]
        if save_overlays:
            overlay_dir = self.owner.specimen.results_dir(overlay_dirname, "edge_multi", "overlays")
            for frame_idx in frame_indices:
                frame_key = f"frame_{frame_idx:04d}"
                _save_multi_level_overlay(
                    raw_frame=_ensure_uint8(raw_stack[frame_idx]),
                    level_masks=[exclusive_masks[key][frame_key] for key in result_keys],
                    labels=labels,
                    colors=display_colors,
                    save_path=overlay_dir / f"edge_multi_overlay_{frame_idx:04d}.png",
                )
            paths["overlays"] = str(overlay_dir)

        result: Dict[str, Any] = {
            "interfaces": [
                {"key": key, "name": interface.name, "label": label, "color_rgba": color}
                for key, interface, label, color in zip(result_keys, interface_list, labels, display_colors)
            ],
            "frame_indices": frame_indices,
            "frame_level_maps": frame_level_maps,
            "paths": paths,
            "params": {
                "secondary_similarity_threshold": multi_params["secondary_similarity_threshold"],
            },
        }
        if return_masks:
            result["inclusive_masks"] = inclusive_masks
            result["exclusive_masks"] = exclusive_masks
        if debug:
            result["debug"] = debug_levels
        return result

    @staticmethod
    def _attribute_deeper_level(
        candidates: List[np.ndarray],
        *,
        parent: List[np.ndarray],
        frame_indices: List[int],
        parent_delay: int,
        start_frame: Optional[int],
        keep_masks: bool,
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """Latch the candidate pixels that lie inside the parent interface's mask.

        The parent mask is taken ``parent_delay`` frames back, so its
        still-growing front is left out. Only regions touching the specimen
        edge are kept.
        """
        accumulated = np.zeros_like(candidates[0], dtype=bool)
        latched: List[np.ndarray] = []
        diagnostics: List[Dict[str, Any]] = []
        for frame_pos, candidate in enumerate(candidates):
            if start_frame is not None and frame_indices[frame_pos] < start_frame:
                latched.append(accumulated.copy())
                diagnostics.append({"_masks": {}, "connected_pixels": 0})
                continue

            settled_parent = np.asarray(parent[max(0, frame_pos - parent_delay)], dtype=bool)
            connected = _filter_specimen_edge_connected(candidate & settled_parent)
            accumulated = _filter_specimen_edge_connected(accumulated | connected)
            latched.append(accumulated.copy())
            diagnostics.append({
                "_masks": {"connected_mask": connected} if keep_masks else {},
                "connected_pixels": int(connected.sum()),
            })
        return latched, diagnostics

    def _resolve_multi_params(self, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Defaults and type checks for ``secondary_params``."""
        resolved: Dict[str, Any] = {
            "secondary_similarity_threshold": 0.6,
            "min_primary_frac_for_secondary": 0.0,
            "secondary_start_frame": None,
        }
        resolved.update({key: params[key] for key in resolved if params and key in params})
        resolved["secondary_similarity_threshold"] = float(resolved["secondary_similarity_threshold"])
        resolved["min_primary_frac_for_secondary"] = float(resolved["min_primary_frac_for_secondary"])
        if resolved["secondary_start_frame"] is not None:
            resolved["secondary_start_frame"] = int(resolved["secondary_start_frame"])
        return resolved

    @staticmethod
    def _build_interface_result_keys(interfaces: Sequence[Interface]) -> List[str]:
        """Unique, file-safe keys for the interfaces, based on their names."""
        seen: Dict[str, int] = {}
        keys: List[str] = []
        for idx, interface in enumerate(interfaces):
            raw_base = str(interface.name).strip() or f"interface_{idx + 1}"
            base = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in raw_base)
            base = base.strip("_") or f"interface_{idx + 1}"
            count = seen.get(base, 0)
            seen[base] = count + 1
            keys.append(base if count == 0 else f"{base}_{count + 1}")
        return keys

    def _resolve_primary_params(self, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Fill in the default edge parameters and check the given ones.

        ``hard_floor`` is a fraction of full intensity: pixels brighter than
        it are never damage. The default 0.90 comes from tuning on our
        specimens; Glud/Bender crack detection is often run at about 0.96.
        Unless ``scale_min`` or ``scale_max`` is set, the image is scaled
        between two percentiles.
        """
        resolved = {
            "window_edge": (0, 60),
            "threshold_strategy": "kmeans",
            "gaussian_filters": (0.5, 15.0),
            "scale_min": None,
            "scale_max": None,
            "scale_min_percentile": 10.0,
            "scale_max_percentile": 99.0,
            "seed_ratio": 0.01,
            "connectivity_mode": "directional",
            "directional_lateral_drift_px": None,
            "directional_lateral_drift_scale": 0.25,
            "hard_floor": 0.90,
            "post_threshold_closing_px": 4,
            "post_threshold_closing_scale": None,
            "post_threshold_closing_radius": None,
            "pre_threshold_closing_radius": None,
            "min_object_px": 0,
        }
        if params:
            resolved.update(params)

        seed_ratio = float(resolved["seed_ratio"])
        if seed_ratio <= 0:
            raise ValueError("seed_ratio must be > 0.")

        connectivity_mode = str(resolved["connectivity_mode"]).strip().lower()
        if connectivity_mode == "legacy_flood":
            raise ValueError(
                "connectivity_mode='legacy_flood' has been removed; "
                "use connectivity_mode='directional' or 'columnwise'."
            )
        if connectivity_mode not in {"directional", "columnwise"}:
            raise ValueError("connectivity_mode must be 'directional' or 'columnwise'.")

        def optional_int(value: Any) -> Optional[int]:
            return None if value is None else int(value)

        lateral_px = resolved["directional_lateral_drift_px"]
        resolved.update(
            window_edge=_resolve_pair(resolved["window_edge"], name="window_edge", caster=int),
            gaussian_filters=_resolve_pair(resolved["gaussian_filters"], name="gaussian_filters", caster=float),
            scale_min=_resolve_optional_float(resolved["scale_min"]),
            scale_max=_resolve_optional_float(resolved["scale_max"]),
            scale_min_percentile=_resolve_optional_float(resolved["scale_min_percentile"]),
            scale_max_percentile=_resolve_optional_float(resolved["scale_max_percentile"]),
            seed_ratio=seed_ratio,
            connectivity_mode=connectivity_mode,
            directional_lateral_drift_px=None if lateral_px is None else max(0, int(lateral_px)),
            directional_lateral_drift_scale=max(0.0, float(resolved["directional_lateral_drift_scale"])),
            hard_floor=_resolve_hard_floor_ratio(resolved["hard_floor"]),
            post_threshold_closing_px=max(0, int(resolved["post_threshold_closing_px"])),
            post_threshold_closing_scale=_resolve_pos_scale(resolved["post_threshold_closing_scale"]),
            post_threshold_closing_radius=optional_int(resolved["post_threshold_closing_radius"]),
            pre_threshold_closing_radius=optional_int(resolved["pre_threshold_closing_radius"]),
            min_object_px=max(0, int(resolved["min_object_px"])),
        )
        return resolved

    def _process_edge_slice(
        self,
        slice_img: np.ndarray,
        *,
        prev_latched: Optional[np.ndarray],
        params: Dict[str, Any],
        avg_crack_width_px: float,
    ) -> Dict[str, Any]:
        """Detect edge damage in one half-frame (specimen edge at row 0) and latch it onto ``prev_latched``.

        All intermediate images are returned too, for debugging and figures.
        """
        filtered_max, filtered_min, sharpened, smoothed = _smooth_for_threshold(
            _ensure_uint8(slice_img), params["window_edge"], params["gaussian_filters"], avg_crack_width_px
        )

        # Explicit scale bounds win; otherwise stretch between two percentiles of the slice.
        scale_min = params.get("scale_min")
        scale_max = params.get("scale_max")
        if scale_min is None and scale_max is None:
            percentiles = _percentile_range(
                smoothed, params.get("scale_min_percentile"), params.get("scale_max_percentile")
            )
            if percentiles is not None:
                scale_min, scale_max = percentiles
        constant_scaled = _scale_to_unit(
            smoothed,
            150.0 if scale_min is None else float(scale_min),
            255.0 if scale_max is None else float(scale_max),
        )
        closed = constant_scaled

        thresh = _kmeans_split(closed) if params["threshold_strategy"] == "kmeans" else None
        if thresh is None:
            thresh = _minmax_otsu_threshold(closed, params["window_edge"])

        floor_mask, hard_floor_eff = _hard_floor_mask(smoothed, params.get("hard_floor"))

        binary = (closed < thresh) & floor_mask

        close_radius = params.get("post_threshold_closing_radius")
        if close_radius is None:
            close_radius = params.get("post_threshold_closing_px")
        if close_radius is None:
            close_radius = params.get("pre_threshold_closing_radius")
        if close_radius is None:
            close_scale = params.get("post_threshold_closing_scale")
            close_radius = 4 if close_scale is None else round(float(close_scale) * avg_crack_width_px)
        close_radius = int(close_radius)

        if close_radius > 0:
            binary_closed = closing(binary, disk(close_radius)).astype(bool)
        else:
            binary_closed = np.asarray(binary, dtype=bool)

        min_object_px = int(params.get("min_object_px", 0))
        if min_object_px > 0:
            binary_closed = _remove_small_components(binary_closed, min_object_px)
        mask = np.asarray(binary_closed, dtype=bool)

        combined_upper = mask.copy()
        if prev_latched is not None:
            combined_upper = np.asarray(prev_latched, dtype=bool) | combined_upper

        seed_depth = max(1, int(round(float(params["seed_ratio"]) * combined_upper.shape[0])))
        connectivity_mode = str(params.get("connectivity_mode", "directional"))
        lateral_drift_px = params.get("directional_lateral_drift_px")
        if lateral_drift_px is None:
            drift_scale = float(params.get("directional_lateral_drift_scale", 0.25))
            lateral_drift_px = max(1, int(round(drift_scale * float(avg_crack_width_px))))
        else:
            lateral_drift_px = max(0, int(lateral_drift_px))

        primary_seed = np.zeros_like(combined_upper, dtype=np.uint8)
        primary_seed[:seed_depth, :] = combined_upper[:seed_depth, :].astype(np.uint8)
        if connectivity_mode == "columnwise":
            primary_edge_snapshot = _rebuild_edge_connected_columnwise(combined_upper, seed_depth=seed_depth)
        else:
            primary_edge_snapshot = _rebuild_edge_connected_directional(
                combined_upper, seed_depth=seed_depth, lateral_drift_px=lateral_drift_px
            )

        if prev_latched is None:
            primary_latched = primary_edge_snapshot.copy()
        else:
            primary_latched = np.asarray(prev_latched, dtype=bool) | primary_edge_snapshot

        return {
            "status": "ok",
            "filtered_max": filtered_max,
            "filtered_min": filtered_min,
            "sharpened": sharpened,
            "smoothed": smoothed,
            "constant_scaled": constant_scaled,
            "closed": closed,
            "threshold": float(thresh),
            "hard_floor_eff": hard_floor_eff,
            "binary": binary,
            "binary_closed": binary_closed,
            "close_radius": close_radius,
            "min_object_px": min_object_px,
            "mask": mask,
            "combined_upper": combined_upper,
            "primary_seed": primary_seed,
            "primary_edge_snapshot": primary_edge_snapshot,
            "primary_latched": primary_latched,
            "connectivity_mode": connectivity_mode,
            "directional_lateral_drift_px": int(lateral_drift_px),
        }
