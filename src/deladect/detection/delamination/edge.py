"""Edge-focused delamination detection: :class:`EdgeDetector`.

Depends on :mod:`._common`, :mod:`._overlays`, and :mod:`._preprocess`.
"""


from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import unsharp_mask
from skimage.morphology import closing, disk

from deladect.io.delamination import (
    save_mask_bundle,
    store_interface_masks,
)
from deladect.specimen import (
    Interface,
)

logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING

from ._common import (
    _auto_preprocess_cache_paths,
    _build_primary_debug_payload,
    _ensure_uint8,
    _fetch_region_override_stacks,
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


def _rebuild_edge_connected_directional(
    mask: np.ndarray,
    *,
    seed_depth: int,
    lateral_drift_px: int,
) -> np.ndarray:
    """Rebuild an edge-connected mask using vertical-biased propagation.

    Growth is causal from top to bottom: each row can activate only candidate
    pixels supported by the preceding accepted row within ``lateral_drift_px``
    columns. Empty rows cannot be jumped.
    """
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim != 2:
        raise ValueError("Directional edge reconstruction expects a 2D mask.")

    height, _ = mask_bool.shape
    if height == 0:
        return np.asarray(mask_bool, dtype=bool)

    seed_rows = min(max(1, int(seed_depth)), height)
    drift = max(0, int(lateral_drift_px))

    rebuilt = np.zeros_like(mask_bool, dtype=bool)
    rebuilt[:seed_rows, :] = mask_bool[:seed_rows, :]
    if seed_rows >= height:
        return rebuilt

    if drift <= 0:
        for row in range(seed_rows, height):
            rebuilt[row, :] = mask_bool[row, :] & rebuilt[row - 1, :]
        return rebuilt

    support_structure = np.ones((2 * drift + 1,), dtype=bool)
    for row in range(seed_rows, height):
        support = ndi.binary_dilation(rebuilt[row - 1, :], structure=support_structure)
        rebuilt[row, :] = mask_bool[row, :] & support
    return rebuilt


def _rebuild_edge_connected_columnwise(
    mask: np.ndarray,
    *,
    seed_depth: int,
) -> np.ndarray:
    """Rebuild an edge-connected mask without lateral or diagonal support.

    Seed-strip candidates are accepted directly. Each subsequent candidate is
    accepted only if the pixel immediately above it in the same column was
    accepted. A gap therefore terminates growth in that column.
    """
    return _rebuild_edge_connected_directional(
        mask,
        seed_depth=seed_depth,
        lateral_drift_px=0,
    )


def _remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove connected components smaller than ``min_size`` pixels."""
    min_px = max(0, int(min_size))
    cleaned = np.asarray(mask, dtype=bool)
    if min_px <= 1:
        return cleaned

    labels, count = ndi.label(cleaned)
    if count <= 0:
        return cleaned

    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[labels]


def _filter_specimen_edge_connected(mask: np.ndarray) -> np.ndarray:
    """Keep only mask pixels whose connected component touches row 0 (specimen free edge)."""
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return mask
    labeled, _ = ndi.label(mask)
    edge_labels = set(np.unique(labeled[0, :])) - {0}
    if not edge_labels:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(labeled, list(edge_labels)).astype(bool)



class EdgeDetector:
    """Edge-focused delamination workflows.

    The class exposes:

    - :meth:`detect_primary` for single-interface edge tracking.
    - :meth:`detect_edge_multi` for hierarchical multi-interface delamination.
    """

    def __init__(self, owner: DelaminationDetector) -> None:
        """Create an edge detector bound to its parent delamination detector."""
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
        """Detect primary edge delamination masks.

        The frame is split into upper/lower halves. The lower half is flipped so both
        sides are treated with the same edge-seeding convention. Masks are latched over
        time and then assembled back into full-frame outputs.

        Parameters
        ----------
        processed_cache_paths, processed_stack:
            Optional preprocessed source.  If omitted, preprocessing is performed
            automatically with ``reference_mode="static"``.  When providing
            pre-computed frames, they must have been produced with static reference;
            rolling-median preprocessed frames are only appropriate for
            :meth:`detect_edge_multi`.
        save_overlays:
            If ``True``, save per-frame edge overlays.
        overlay_dirname:
            Output folder root under specimen results.
        overlay_view:
            One of ``"mask"``, ``"line"``, or ``"both"``.
        max_frames:
            Optional cap on processed frames.
        params:
            Optional edge parameter overrides.
        debug:
            If ``True``, return per-frame intermediate arrays and thresholds.
        save_debug_outputs:
            If ``True``, persist intermediate edge arrays per frame to disk.
        debug_dirname:
            Output folder name for saved debug images.

        Returns
        -------
        dict[str, Any]
            ``{"masks": dict[str, np.ndarray], "debug": dict[str, Any] | None}``
            keyed by ``frame_XXXX``.
        """
        if processed_cache_paths and processed_stack:
            raise ValueError("Provide either processed_cache_paths or processed_stack, not both.")
        if overlay_view not in {"mask", "line", "both"}:
            raise ValueError("overlay_view must be one of: 'mask', 'line', 'both'.")

        if self.owner._uses_stack_overrides():
            return self._detect_primary_region_overrides(
                processed_cache_paths=processed_cache_paths,
                save_overlays=save_overlays,
                overlay_dirname=overlay_dirname,
                overlay_view=overlay_view,
                max_frames=max_frames,
                params=params,
                debug=debug,
                progress=progress,
                save_debug_outputs=save_debug_outputs,
                debug_dirname=debug_dirname,
            )

        stacks = self.owner._select_stacks()
        raw_stack = getattr(self.owner.specimen, "image_stack_full", None) or stacks.get("full")
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

        primary_masks: Dict[str, np.ndarray] = {}
        debug_payloads: Optional[Dict[str, Any]] = {} if debug else None

        upper_state: Optional[np.ndarray] = None
        lower_state: Optional[np.ndarray] = None

        edge_params = self._resolve_primary_params(params)

        debug_root: Optional[Path] = None
        if save_debug_outputs:
            debug_root = self.owner.specimen.results_dir(debug_dirname)
            debug_root.mkdir(parents=True, exist_ok=True)

        if processed_stack is not None:
            processed_iter = enumerate(processed_stack)
            total_frames = len(processed_stack)
        else:
            if processed_cache_paths is None:
                raise ValueError("No processed frames available for edge detection.")
            processed_iter = self.owner.iter_preprocessed_cache(processed_cache_paths)
            total_frames = len(processed_cache_paths)

        if max_frames is not None:
            total_frames = min(total_frames, max_frames)
        progress_state = _progress_init("edge_primary", total_frames, progress)

        for idx, processed in processed_iter:
            if idx >= total_frames:
                break
            split_row = processed.shape[0] // 2
            upper_slice = processed[:split_row, :]
            lower_slice = processed[split_row:, :]
            lower_prepared = np.flipud(lower_slice)

            upper_result = self._process_edge_slice(
                upper_slice,
                prev_latched=upper_state,
                params=edge_params,
                avg_crack_width_px=self.owner.specimen.avg_crack_width_px,
            )
            lower_result = self._process_edge_slice(
                lower_prepared,
                prev_latched=lower_state,
                params=edge_params,
                avg_crack_width_px=self.owner.specimen.avg_crack_width_px,
            )

            upper_state = upper_result["primary_latched"]
            lower_state = lower_result["primary_latched"]

            primary_full = np.zeros_like(processed, dtype=bool)
            primary_full[:split_row, :] = upper_state
            lower_unflipped = np.flipud(lower_state) if lower_state is not None else np.zeros_like(lower_slice)
            primary_full[split_row:, :] = lower_unflipped

            frame_key = f"frame_{idx:04d}"
            primary_masks[frame_key] = primary_full

            if save_overlays and raw_stack is not None:
                raw_frame = _ensure_uint8(raw_stack[idx])
                overlay_dir = self.owner.specimen.results_dir(overlay_dirname, "edge", "overlays")
                overlay_path = overlay_dir / f"edge_overlay_{idx:04d}.png"
                _save_edge_overlay(raw_frame, primary_full, overlay_path, view=overlay_view)

            if debug_root is not None:
                raw_frame = _ensure_uint8(raw_stack[idx]) if raw_stack is not None else processed
                frame_dir = debug_root / f"frame_{idx:04d}"
                frame_dir.mkdir(parents=True, exist_ok=True)
                _save_edge_debug_frame(
                    frame_dir=frame_dir,
                    raw_frame=raw_frame,
                    processed=processed,
                    upper_slice=upper_slice,
                    lower_slice=lower_prepared,
                    upper_result=upper_result,
                    lower_result=lower_result,
                    lower_latched_unflipped=np.flipud(lower_state) if lower_state is not None else None,
                    full_latched=primary_full,
                )

            if debug and debug_payloads is not None:
                debug_payloads[frame_key] = _build_primary_debug_payload(
                    processed.copy(), upper_result, lower_result
                )

            _progress_update("edge_primary", idx + 1, total_frames, progress_state)

        _progress_done("edge_primary", total_frames, progress)

        return {"masks": primary_masks, "debug": debug_payloads}

    def _detect_primary_region_overrides(
        self,
        *,
        processed_cache_paths: Optional[List[Path]] = None,
        save_overlays: bool = False,
        overlay_dirname: str = "delamination",
        overlay_view: str = "mask",
        max_frames: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        progress: bool = False,
        save_debug_outputs: bool = False,
        debug_dirname: str = "edge_accumulation_debug",
    ) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, Any]]]:
        """Edge detection path that prioritizes explicit upper/lower/middle stacks."""
        upper_stack, middle_stack, lower_stack, raw_stack, total_frames = _fetch_region_override_stacks(
            self.owner, domain="edge", max_frames=max_frames
        )

        edge_params = self._resolve_primary_params(params)
        reference_defaults = _reference_settings_from_cache_paths(processed_cache_paths)
        reference_mode = str((params or {}).get("reference_mode", reference_defaults["reference_mode"]))
        reference_window = int((params or {}).get("reference_window", reference_defaults["reference_window"]))
        reference_skip = int((params or {}).get("reference_skip", reference_defaults["reference_skip"]))

        upper_cache_paths = self.owner.preprocess_stack_to_disk(
            upper_stack,
            key=f"edge_upper_auto_{_result_key_token(self.owner.interface.name)}",
            max_frames=total_frames,
            cache_dirname="Preprocessor_cache",
            history_mode="running",
            history_window_size=None,
            reference_mode=reference_mode,
            reference_window=reference_window,
            reference_skip=reference_skip,
            progress=progress,
        )["cache_paths"]
        lower_cache_paths = self.owner.preprocess_stack_to_disk(
            lower_stack,
            key=f"edge_lower_auto_{_result_key_token(self.owner.interface.name)}",
            max_frames=total_frames,
            cache_dirname="Preprocessor_cache",
            history_mode="running",
            history_window_size=None,
            reference_mode=reference_mode,
            reference_window=reference_window,
            reference_skip=reference_skip,
            progress=progress,
        )["cache_paths"]

        primary_masks: Dict[str, np.ndarray] = {}
        debug_payloads: Optional[Dict[str, Any]] = {} if debug else None

        upper_state: Optional[np.ndarray] = None
        lower_state: Optional[np.ndarray] = None

        debug_root: Optional[Path] = None
        if save_debug_outputs:
            debug_root = self.owner.specimen.results_dir(debug_dirname)
            debug_root.mkdir(parents=True, exist_ok=True)

        progress_state = _progress_init("edge_primary", total_frames, progress)

        upper_iter = self.owner.iter_preprocessed_cache(upper_cache_paths)
        lower_iter = self.owner.iter_preprocessed_cache(lower_cache_paths)

        for (idx_u, upper_processed), (idx_l, lower_processed) in zip(upper_iter, lower_iter):
            idx = int(min(idx_u, idx_l))
            if idx >= total_frames:
                break

            upper_processed = _ensure_uint8(upper_processed)
            lower_processed = _ensure_uint8(lower_processed)
            lower_prepared = np.flipud(lower_processed)

            upper_result = self._process_edge_slice(
                upper_processed,
                prev_latched=upper_state,
                params=edge_params,
                avg_crack_width_px=self.owner.specimen.avg_crack_width_px,
            )
            lower_result = self._process_edge_slice(
                lower_prepared,
                prev_latched=lower_state,
                params=edge_params,
                avg_crack_width_px=self.owner.specimen.avg_crack_width_px,
            )

            upper_state = upper_result["primary_latched"]
            lower_state = lower_result["primary_latched"]

            middle_raw = _ensure_uint8(middle_stack[idx])
            upper_h, width = upper_processed.shape[:2]
            middle_h = int(middle_raw.shape[0])
            lower_h = int(lower_processed.shape[0])

            primary_full = np.zeros((upper_h + middle_h + lower_h, width), dtype=bool)
            primary_full[:upper_h, :] = np.asarray(upper_state, dtype=bool)
            lower_unflipped = np.flipud(np.asarray(lower_state, dtype=bool))
            primary_full[upper_h + middle_h :, :] = lower_unflipped

            frame_key = f"frame_{idx:04d}"
            primary_masks[frame_key] = primary_full

            if save_overlays:
                raw_frame = _region_override_raw_frame(
                    raw_stack, idx, primary_full.shape[:2], upper_stack[idx], middle_raw, lower_stack[idx]
                )
                overlay_dir = self.owner.specimen.results_dir(overlay_dirname, "edge", "overlays")
                overlay_path = overlay_dir / f"edge_overlay_{idx:04d}.png"
                _save_edge_overlay(raw_frame, primary_full, overlay_path, view=overlay_view)

            if debug_root is not None:
                raw_frame = _region_override_raw_frame(
                    raw_stack, idx, primary_full.shape[:2], upper_stack[idx], middle_raw, lower_stack[idx]
                )

                processed_full = np.vstack(
                    [
                        upper_processed,
                        np.zeros((middle_h, width), dtype=np.uint8),
                        lower_processed,
                    ]
                )

                frame_dir = debug_root / f"frame_{idx:04d}"
                frame_dir.mkdir(parents=True, exist_ok=True)
                _save_edge_debug_frame(
                    frame_dir=frame_dir,
                    raw_frame=raw_frame,
                    processed=processed_full,
                    upper_slice=upper_processed,
                    lower_slice=lower_prepared,
                    upper_result=upper_result,
                    lower_result=lower_result,
                    lower_latched_unflipped=lower_unflipped,
                    full_latched=primary_full,
                )

            if debug and debug_payloads is not None:
                debug_payloads[frame_key] = _build_primary_debug_payload(
                    np.vstack(
                        [
                            upper_processed,
                            np.zeros((middle_h, width), dtype=np.uint8),
                            lower_processed,
                        ]
                    ),
                    upper_result,
                    lower_result,
                )

            _progress_update("edge_primary", idx + 1, total_frames, progress_state)

        _progress_done("edge_primary", total_frames, progress)
        return {"masks": primary_masks, "debug": debug_payloads}

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
        """Detect hierarchical edge delamination across multiple interfaces.

        The first level uses primary edge detection. Each deeper level is attributed from
        its parent level using workbook-inspired candidate logic, similarity gating,
        persistence confirmation, and edge-connected reconstruction.

        Parameters
        ----------
        interfaces:
            Ordered interface list from shallow to deep.
        processed_cache_paths, processed_stack:
            Preprocessed source for the **primary** (level-0) edge detection.  If omitted,
            rolling-median preprocessing is executed automatically.  For best results pass
            a *static*-preprocessed cache so the accumulated primary mask matches
            ``detect_both_delaminations``.
        secondary_cache_paths:
            Optional separate cache for the **secondary** binary/mask step.  When provided,
            rolling-median-preprocessed frames from this cache are used to compute the
            binary and mask used for hierarchical attribution, while
            ``processed_cache_paths`` drives the primary latched accumulation only.
            Both caches must have the same frame count.
        save_overlays:
            If ``True``, save one classified overlay per frame with an external legend.
        overlay_dirname:
            Output folder root under specimen results.
        save_masks:
            If ``True``, save inclusive/exclusive masks for each interface as ``.npz``.
        masks_dirname:
            Mask bundle subfolder name.
        max_frames:
            Optional cap on processed frames.
        primary_params:
            Edge detection parameters for the primary (level-0) pass: window_edge,
            gaussian_filters, hard_floor, closing_px, seed_ratio, etc.
        secondary_edge_params:
            Edge detection parameters applied to ``secondary_cache_paths`` frames when
            computing binary/mask for the attribution logic.  Ignored if
            ``secondary_cache_paths`` is not set.
        secondary_params:
            Attribution parameters for deeper levels: secondary_similarity_threshold.
        params:
            Legacy single-dict interface (backward compat). Keys are merged into
            primary_params as a base; primary_params and secondary_params take precedence.
        return_masks:
            If ``True``, include mask dictionaries in the returned payload.
        debug:
            If ``True``, include per-level diagnostics.

        Returns
        -------
        dict[str, Any]
            Contains interface descriptors, frame-level maps, output paths,
            effective parameters, and optional masks/debug payloads.
        """
        if processed_cache_paths and processed_stack:
            raise ValueError("Provide either processed_cache_paths or processed_stack, not both.")

        if processed_cache_paths is None and processed_stack is None:
            stacks_for_preprocess = self.owner._select_stacks()
            auto_stack = (
                getattr(self.owner.specimen, "image_stack_full", None)
                or stacks_for_preprocess.get("full")
            )
            if auto_stack is None:
                raise ValueError(
                    "detect_edge_multi: no full image stack available for automatic "
                    "preprocessing. Provide processed_cache_paths or processed_stack."
                )
            _p = dict(params or {})
            _p.update(primary_params or {})
            _ref_window = int(_p.get("reference_window", 10))
            _ref_skip = int(_p.get("reference_skip", 1))
            interface_token = _result_key_token(interfaces[0].name if interfaces else "i0")
            auto_key = f"edge_multi_auto_{interface_token}"
            processed_cache_paths = self.owner.preprocess_stack_to_disk(
                auto_stack,
                key=auto_key,
                max_frames=max_frames,
                cache_dirname="Preprocessor_cache",
                reference_mode="rolling_median",
                reference_window=_ref_window,
                reference_skip=_ref_skip,
            )["cache_paths"]

        interface_list = list(interfaces)
        if not interface_list:
            raise ValueError("detect_edge_multi requires at least one interface.")

        # Load baselines from cache .npz files for debug panels.
        _debug_baselines: List[np.ndarray] = []
        if debug_dir is not None and processed_cache_paths:
            for p in processed_cache_paths:
                try:
                    with np.load(p, allow_pickle=False) as z:
                        _debug_baselines.append(z["baseline"])
                except Exception:
                    _debug_baselines.append(np.array([]))

        stacks = self.owner._select_stacks()
        raw_stack = getattr(self.owner.specimen, "image_stack_full", None) or stacks.get("full")
        if save_overlays and raw_stack is None:
            raise ValueError("Cannot save overlays without a full raw image stack.")

        _primary = dict(params or {})
        _primary.update(primary_params or {})
        _secondary = dict(params or {})
        _secondary.update(secondary_params or {})
        edge_params = self._resolve_primary_params(_primary)
        multi_params = self._resolve_multi_params(_secondary)

        # Resolve edge params for the secondary binary/mask step (rolling_median cache).
        _sec_edge_params = self._resolve_primary_params(
            {**(params or {}), **(secondary_edge_params or {})}
        ) if secondary_cache_paths is not None else None

        if processed_stack is not None:
            processed_iter = enumerate(processed_stack)
        else:
            processed_iter = self.owner.iter_preprocessed_cache(processed_cache_paths)

        _secondary_cache_iter = (
            self.owner.iter_preprocessed_cache(secondary_cache_paths)
            if secondary_cache_paths is not None
            else None
        )

        upper_state: Optional[np.ndarray] = None
        lower_state: Optional[np.ndarray] = None
        upper_rolling_state: Optional[np.ndarray] = None
        lower_rolling_state: Optional[np.ndarray] = None
        upper_rolling_frames: List[np.ndarray] = []
        lower_rolling_frames: List[np.ndarray] = []

        frame_indices: List[int] = []
        frame_shapes: List[Tuple[int, int]] = []
        split_rows: List[int] = []
        upper_primary_frames: List[np.ndarray] = []
        lower_primary_frames: List[np.ndarray] = []
        _debug_upper_results: List[Dict[str, Any]] = []
        _debug_lower_results: List[Dict[str, Any]] = []
        _debug_processed: List[np.ndarray] = []
        _debug_sec_upper_results: List[Dict[str, Any]] = []
        _debug_sec_lower_results: List[Dict[str, Any]] = []
        _debug_sec_processed: List[np.ndarray] = []
        upper_binary_frames: List[np.ndarray] = []
        lower_binary_frames: List[np.ndarray] = []
        upper_mask_frames: List[np.ndarray] = []
        lower_mask_frames: List[np.ndarray] = []

        for idx, processed in processed_iter:
            if max_frames is not None and len(frame_indices) >= max(0, int(max_frames)):
                break

            processed_uint8 = _ensure_uint8(processed)
            split_row = processed_uint8.shape[0] // 2
            upper_slice = processed_uint8[:split_row, :]
            lower_slice = processed_uint8[split_row:, :]
            lower_prepared = np.flipud(lower_slice)

            upper_result = self._process_edge_slice(
                upper_slice,
                prev_latched=upper_state,
                params=edge_params,
                avg_crack_width_px=self.owner.specimen.avg_crack_width_px,
            )
            lower_result = self._process_edge_slice(
                lower_prepared,
                prev_latched=lower_state,
                params=edge_params,
                avg_crack_width_px=self.owner.specimen.avg_crack_width_px,
            )

            upper_curr = np.asarray(upper_result["primary_latched"], dtype=bool)
            lower_curr = np.asarray(lower_result["primary_latched"], dtype=bool)
            upper_state = upper_curr
            lower_state = lower_curr

            if debug_dir is not None:
                _debug_upper_results.append(upper_result)
                _debug_lower_results.append(lower_result)
                _debug_processed.append(processed_uint8.copy())

            frame_indices.append(int(idx))
            frame_shapes.append((int(processed_uint8.shape[0]), int(processed_uint8.shape[1])))
            split_rows.append(split_row)
            upper_primary_frames.append(upper_curr.copy())
            lower_primary_frames.append(lower_curr.copy())

            if _secondary_cache_iter is not None:
                _, sec_processed = next(_secondary_cache_iter)
                sec_uint8 = _ensure_uint8(sec_processed)
                sec_upper = self._process_edge_slice(
                    sec_uint8[:split_row, :],
                    prev_latched=upper_rolling_state,
                    params=_sec_edge_params,
                    avg_crack_width_px=self.owner.specimen.avg_crack_width_px,
                )
                sec_lower = self._process_edge_slice(
                    np.flipud(sec_uint8[split_row:, :]),
                    prev_latched=lower_rolling_state,
                    params=_sec_edge_params,
                    avg_crack_width_px=self.owner.specimen.avg_crack_width_px,
                )
                upper_rolling_state = np.asarray(sec_upper["primary_latched"], dtype=bool)
                lower_rolling_state = np.asarray(sec_lower["primary_latched"], dtype=bool)
                upper_rolling_frames.append(upper_rolling_state.copy())
                lower_rolling_frames.append(lower_rolling_state.copy())
                if debug_dir is not None:
                    _debug_sec_upper_results.append(sec_upper)
                    _debug_sec_lower_results.append(sec_lower)
                    _debug_sec_processed.append(sec_uint8.copy())
                upper_binary_frames.append(np.asarray(sec_upper["binary"], dtype=bool))
                lower_binary_frames.append(np.asarray(sec_lower["binary"], dtype=bool))
                upper_mask_frames.append(np.asarray(sec_upper["mask"], dtype=bool))
                lower_mask_frames.append(np.asarray(sec_lower["mask"], dtype=bool))
            else:
                upper_rolling_frames.append(upper_curr.copy())
                lower_rolling_frames.append(lower_curr.copy())
                upper_binary_frames.append(np.asarray(upper_result["binary"], dtype=bool))
                lower_binary_frames.append(np.asarray(lower_result["binary"], dtype=bool))
                upper_mask_frames.append(np.asarray(upper_result["mask"], dtype=bool))
                lower_mask_frames.append(np.asarray(lower_result["mask"], dtype=bool))

        if not frame_indices:
            raise ValueError("No processed frames available for multi-interface edge detection.")

        upper_levels: List[List[np.ndarray]] = [upper_primary_frames]
        lower_levels: List[List[np.ndarray]] = [lower_primary_frames]
        upper_rolling_levels: List[List[np.ndarray]] = [upper_rolling_frames]
        lower_rolling_levels: List[List[np.ndarray]] = [lower_rolling_frames]
        debug_levels: Dict[str, Any] = {}

        # Per-frame gate: suppress secondary if static primary area is below threshold.
        _min_prim_frac = multi_params["min_primary_frac_for_secondary"]
        if _min_prim_frac > 0.0:
            _upper_active: Optional[List[bool]] = []
            _lower_active: Optional[List[bool]] = []
            for _fp in range(len(upper_primary_frames)):
                u = np.asarray(upper_primary_frames[_fp], dtype=bool)
                l = np.asarray(lower_primary_frames[_fp], dtype=bool)
                _upper_active.append(u.sum() / max(1, u.size) >= _min_prim_frac)
                _lower_active.append(l.sum() / max(1, l.size) >= _min_prim_frac)
        else:
            _upper_active = None
            _lower_active = None

        for level_idx in range(1, len(interface_list)):
            upper_latched: List[np.ndarray] = []
            lower_latched: List[np.ndarray] = []
            upper_diag: List[Dict[str, Any]] = []
            lower_diag: List[Dict[str, Any]] = []
            acc_u = np.zeros_like(upper_mask_frames[0], dtype=bool)
            acc_l = np.zeros_like(lower_mask_frames[0], dtype=bool)

            # Use the established mask of the level directly above (delayed by ~rolling-
            # median window) so the growing front is excluded: only ROLLING mask pixels
            # inside the settled parent-level area accumulate here, catching the interior
            # darkening event. For level 1 the parent is the primary (level 0); for level
            # 2+ the parent is the previous level's own accumulated attribution, so each
            # deeper level is recursively gated by the level immediately above it.
            primary_upper = upper_levels[level_idx - 1]
            primary_lower = lower_levels[level_idx - 1]
            secondary_reference_params = _sec_edge_params or edge_params
            sec_ref_window = int(secondary_reference_params.get("reference_window", 7))
            _sec_start = multi_params.get("secondary_start_frame")

            for frame_pos in range(len(upper_mask_frames)):
                # Gate: suppress secondary accumulation before the configured onset frame.
                if _sec_start is not None and frame_indices[frame_pos] < _sec_start:
                    upper_latched.append(acc_u.copy())
                    lower_latched.append(acc_l.copy())
                    upper_diag.append({"_masks": {}, "connected_pixels": 0})
                    lower_diag.append({"_masks": {}, "connected_pixels": 0})
                    continue

                mask_u = np.asarray(upper_mask_frames[frame_pos], dtype=bool)
                mask_l = np.asarray(lower_mask_frames[frame_pos], dtype=bool)

                delayed_pos = max(0, frame_pos - sec_ref_window)
                est_u = np.asarray(primary_upper[delayed_pos], dtype=bool)
                est_l = np.asarray(primary_lower[delayed_pos], dtype=bool)

                conn_u = _filter_specimen_edge_connected(mask_u & est_u)
                conn_l = _filter_specimen_edge_connected(mask_l & est_l)

                acc_u = _filter_specimen_edge_connected(acc_u | conn_u)
                acc_l = _filter_specimen_edge_connected(acc_l | conn_l)

                upper_latched.append(acc_u.copy())
                lower_latched.append(acc_l.copy())

                _mu = {"connected_mask": conn_u} if debug_dir is not None else {}
                upper_diag.append({"_masks": _mu, "connected_pixels": int(conn_u.sum())})
                _ml = {"connected_mask": conn_l} if debug_dir is not None else {}
                lower_diag.append({"_masks": _ml, "connected_pixels": int(conn_l.sum())})

            upper_levels.append(upper_latched)
            lower_levels.append(lower_latched)
            upper_rolling_levels.append(upper_latched)
            lower_rolling_levels.append(lower_latched)

            if debug:
                debug_levels[f"level_{level_idx + 1}"] = {
                    "upper": upper_diag,
                    "lower": lower_diag,
                }

            if debug_dir is not None:
                _save_edge_multi_debug_panels(
                    debug_dir=debug_dir,
                    frame_indices=frame_indices,
                    processed_frames=_debug_processed,
                    baselines=_debug_baselines,
                    upper_results=_debug_upper_results,
                    lower_results=_debug_lower_results,
                    upper_latched=upper_latched,
                    lower_latched=lower_latched,
                    upper_diag=upper_diag,
                    lower_diag=lower_diag,
                    split_rows=split_rows,
                    level_idx=level_idx,
                    sec_processed_frames=_debug_sec_processed,
                    sec_upper_results=_debug_sec_upper_results,
                    sec_lower_results=_debug_sec_lower_results,
                    upper_rolling_frames=upper_rolling_frames,
                    lower_rolling_frames=lower_rolling_frames,
                )

        result_keys = self._build_interface_result_keys(interface_list)
        display_colors = _resolve_multi_interface_colors(interface_list)
        inclusive_masks: Dict[str, Dict[str, np.ndarray]] = {key: {} for key in result_keys}
        exclusive_masks: Dict[str, Dict[str, np.ndarray]] = {key: {} for key in result_keys}
        frame_level_maps: Dict[str, np.ndarray] = {}

        for frame_pos, frame_idx in enumerate(frame_indices):
            frame_key = f"frame_{frame_idx:04d}"
            shape = frame_shapes[frame_pos]
            split_row = split_rows[frame_pos]

            frame_level = np.zeros(shape, dtype=np.uint8)
            for level_idx, key in enumerate(result_keys, start=1):
                full_mask = self._assemble_full_mask(
                    shape=shape,
                    split_row=split_row,
                    upper_mask=upper_levels[level_idx - 1][frame_pos],
                    lower_mask_flipped=lower_levels[level_idx - 1][frame_pos],
                )
                inclusive_masks[key][frame_key] = full_mask
                frame_level[full_mask] = np.uint8(level_idx)

            frame_level_maps[frame_key] = frame_level
            for level_idx, key in enumerate(result_keys, start=1):
                exclusive_masks[key][frame_key] = frame_level == np.uint8(level_idx)

        paths: Dict[str, Any] = {
            "inclusive_masks": {},
            "exclusive_masks": {},
            "overlays": None,
        }

        if save_masks:
            masks_root = self.owner.specimen.results_dir(overlay_dirname, "edge_multi", masks_dirname)
            for level_idx, interface in enumerate(interface_list):
                key = result_keys[level_idx]
                inclusive_path = save_mask_bundle(inclusive_masks[key], masks_root / f"{key}_inclusive.npz")
                exclusive_path = save_mask_bundle(exclusive_masks[key], masks_root / f"{key}_exclusive.npz")
                paths["inclusive_masks"][key] = str(inclusive_path)
                paths["exclusive_masks"][key] = str(exclusive_path)
                store_interface_masks(
                    interface,
                    primary_path=inclusive_path,
                    secondary_path=exclusive_path,
                )

        if save_overlays:
            overlay_dir = self.owner.specimen.results_dir(overlay_dirname, "edge_multi", "overlays")
            labels = [
                _interface_legend_label(self.owner.specimen, interface)
                for interface in interface_list
            ]
            if raw_stack is None:
                raise ValueError("Cannot save overlays without a full raw image stack.")
            for frame_pos, frame_idx in enumerate(frame_indices):
                frame_key = f"frame_{frame_idx:04d}"
                raw_frame = _ensure_uint8(raw_stack[frame_idx])
                frame_masks = [exclusive_masks[key][frame_key] for key in result_keys]
                save_path = overlay_dir / f"edge_multi_overlay_{frame_idx:04d}.png"
                _save_multi_level_overlay(
                    raw_frame=raw_frame,
                    level_masks=frame_masks,
                    labels=labels,
                    colors=display_colors,
                    save_path=save_path,
                )
            paths["overlays"] = str(overlay_dir)

        result: Dict[str, Any] = {
            "interfaces": [
                {
                    "key": result_keys[idx],
                    "name": interface.name,
                    "label": _interface_legend_label(self.owner.specimen, interface),
                    "color_rgba": display_colors[idx],
                }
                for idx, interface in enumerate(interface_list)
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

    def _resolve_multi_params(self, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge and validate multi-interface attribution parameters."""
        resolved = {
            "secondary_similarity_threshold": 0.6,
            "min_primary_frac_for_secondary": 0.0,
            # Optional[int]: 0-based position in the sampled stack at or after which
            # secondary accumulation begins. Frames at position < this value produce zero
            # secondary output. None means no gate (secondary runs from the first frame).
            # Callers are responsible for converting external frame IDs to sample positions.
            "secondary_start_frame": None,
        }
        if params:
            for key in resolved:
                if key in params:
                    resolved[key] = params[key]

        resolved["secondary_similarity_threshold"] = float(resolved["secondary_similarity_threshold"])
        resolved["min_primary_frac_for_secondary"] = float(resolved["min_primary_frac_for_secondary"])
        if resolved["secondary_start_frame"] is not None:
            resolved["secondary_start_frame"] = int(resolved["secondary_start_frame"])
        return resolved

    @staticmethod
    def _assemble_full_mask(
        *,
        shape: Tuple[int, int],
        split_row: int,
        upper_mask: np.ndarray,
        lower_mask_flipped: np.ndarray,
    ) -> np.ndarray:
        """Combine upper/lower half masks into one full-frame mask."""
        full = np.zeros(shape, dtype=bool)
        full[:split_row, :] = np.asarray(upper_mask, dtype=bool)
        full[split_row:, :] = np.flipud(np.asarray(lower_mask_flipped, dtype=bool))
        return full

    @staticmethod
    def _build_interface_result_keys(interfaces: Sequence[Interface]) -> List[str]:
        """Build unique filesystem-safe result keys from interface names."""
        seen: Dict[str, int] = {}
        keys: List[str] = []
        for idx, interface in enumerate(interfaces):
            raw_base = str(interface.name).strip() or f"interface_{idx + 1}"
            base = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in raw_base)
            base = base.strip("_") or f"interface_{idx + 1}"
            count = seen.get(base, 0)
            seen[base] = count + 1
            if count == 0:
                keys.append(base)
            else:
                keys.append(f"{base}_{count + 1}")
        return keys

    def _resolve_primary_params(self, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge and validate primary edge detector parameters.

        ``hard_floor`` is expressed as an intensity fraction in normalized space
        (``0.0``-``1.0``). The default ``0.90`` is chosen after internal
        delamination tuning; for reference, Glud/Bender-style crack workflows are
        often configured around ``0.96``. Percentile-based scaling is the default
        normalization path; fixed ``scale_min``/``scale_max`` values override it
        when either bound is provided explicitly.
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

        window_edge = _resolve_pair(resolved["window_edge"], name="window_edge", caster=int)
        gaussian_filters = _resolve_pair(resolved["gaussian_filters"], name="gaussian_filters", caster=float)

        seed_ratio = float(resolved["seed_ratio"])
        if seed_ratio <= 0:
            raise ValueError("seed_ratio must be > 0.")

        connectivity_mode = str(resolved.get("connectivity_mode", "directional")).strip().lower()
        if connectivity_mode == "legacy_flood":
            raise ValueError(
                "connectivity_mode='legacy_flood' has been removed; "
                "use connectivity_mode='directional' or 'columnwise'."
            )
        if connectivity_mode not in {"directional", "columnwise"}:
            raise ValueError("connectivity_mode must be 'directional' or 'columnwise'.")

        resolved["window_edge"] = window_edge
        resolved["gaussian_filters"] = gaussian_filters
        resolved["scale_min"] = _resolve_optional_float(resolved["scale_min"])
        resolved["scale_max"] = _resolve_optional_float(resolved["scale_max"])
        resolved["scale_min_percentile"] = _resolve_optional_float(resolved.get("scale_min_percentile"))
        resolved["scale_max_percentile"] = _resolve_optional_float(resolved.get("scale_max_percentile"))
        resolved["seed_ratio"] = seed_ratio
        resolved["connectivity_mode"] = connectivity_mode
        lateral_px = resolved.get("directional_lateral_drift_px")
        resolved["directional_lateral_drift_px"] = None if lateral_px is None else max(0, int(lateral_px))
        resolved["directional_lateral_drift_scale"] = max(
            0.0,
            float(resolved.get("directional_lateral_drift_scale", 0.25)),
        )
        hard_floor = resolved.get("hard_floor")
        resolved["hard_floor"] = _resolve_hard_floor_ratio(hard_floor)
        resolved["post_threshold_closing_px"] = max(0, int(resolved.get("post_threshold_closing_px", 4)))
        resolved["post_threshold_closing_scale"] = _resolve_pos_scale(resolved.get("post_threshold_closing_scale"))
        resolved["post_threshold_closing_radius"] = (
            None
            if resolved.get("post_threshold_closing_radius") is None
            else int(resolved["post_threshold_closing_radius"])
        )
        resolved["pre_threshold_closing_radius"] = (
            None
            if resolved.get("pre_threshold_closing_radius") is None
            else int(resolved["pre_threshold_closing_radius"])
        )
        resolved["min_object_px"] = max(0, int(resolved.get("min_object_px", 0)))
        return resolved

    def _process_edge_slice(
        self,
        slice_img: np.ndarray,
        *,
        prev_latched: Optional[np.ndarray],
        params: Dict[str, Any],
        avg_crack_width_px: float,
    ) -> Dict[str, Any]:
        """Run edge segmentation, reconstruction, and latching on one slice."""
        slice_uint8 = _ensure_uint8(slice_img)

        wy = max(1, int(params["window_edge"][0]))
        wx = max(1, int(params["window_edge"][1]))
        filtered_max = ndi.maximum_filter(slice_uint8, size=(wy, wx), mode="reflect")
        filtered_min = ndi.minimum_filter(filtered_max, size=(wy, wx), mode="reflect")
        sharpened = unsharp_mask(
            filtered_min,
            radius=float(avg_crack_width_px),
            amount=2.0,
            preserve_range=True,
        )

        smoothed = ndi.gaussian_filter(sharpened, params["gaussian_filters"])

        scale_min = params.get("scale_min")
        scale_max = params.get("scale_max")
        pct_min = params.get("scale_min_percentile")
        pct_max = params.get("scale_max_percentile")
        if scale_min is None and scale_max is None and pct_min is not None and pct_max is not None:
            p_min = float(np.percentile(smoothed, float(pct_min)))
            p_max = float(np.percentile(smoothed, float(pct_max)))
            if np.isfinite(p_min) and np.isfinite(p_max) and p_max > p_min:
                scale_min, scale_max = p_min, p_max
        if scale_min is None:
            scale_min = 150.0
        if scale_max is None:
            scale_max = 255.0
        scale_min = float(scale_min)
        scale_max = float(scale_max)
        if scale_max > scale_min:
            constant_scaled = np.clip((smoothed.astype(np.float32) - scale_min) / (scale_max - scale_min), 0.0, 1.0)
        else:
            constant_scaled = np.zeros_like(smoothed, dtype=np.float32)

        closed = constant_scaled

        fallback = self.owner._images_threshold(closed, params["window_edge"])
        if params["threshold_strategy"] == "kmeans":
            thresh = self.owner._kmeans_threshold(closed, fallback)
        else:
            thresh = fallback

        hard_floor = params.get("hard_floor")
        hard_floor_eff: Optional[float]
        if hard_floor is None:
            hard_floor_eff = None
            floor_mask = np.ones_like(smoothed, dtype=bool)
        else:
            hard_floor_eff = float(hard_floor)
            smoothed_norm = smoothed.astype(np.float32) / 255.0
            floor_mask = smoothed_norm <= hard_floor_eff

        binary = (closed < thresh) & floor_mask
        radius_override = params.get("post_threshold_closing_radius")
        if radius_override is None:
            radius_override = params.get("post_threshold_closing_px")
        if radius_override is None:
            radius_override = params.get("pre_threshold_closing_radius")
        if radius_override is not None:
            close_radius = int(radius_override)
        else:
            close_scale = params.get("post_threshold_closing_scale")
            if close_scale is None:
                close_radius = 4
            else:
                close_radius = int(round(float(close_scale) * avg_crack_width_px))

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

        height = combined_upper.shape[0]
        seed_depth = max(1, int(round(float(params["seed_ratio"]) * height)))
        connectivity_mode = str(params.get("connectivity_mode", "directional"))
        lateral_drift_px_override = params.get("directional_lateral_drift_px")
        if lateral_drift_px_override is None:
            lateral_drift_px = max(
                1,
                int(round(float(params.get("directional_lateral_drift_scale", 0.25)) * float(avg_crack_width_px))),
            )
        else:
            lateral_drift_px = max(0, int(lateral_drift_px_override))

        primary_seed = np.zeros_like(combined_upper, dtype=np.uint8)
        primary_seed[:seed_depth, :] = combined_upper[:seed_depth, :].astype(np.uint8)
        if connectivity_mode == "columnwise":
            primary_edge_snapshot = _rebuild_edge_connected_columnwise(
                combined_upper,
                seed_depth=seed_depth,
            )
        else:
            primary_edge_snapshot = _rebuild_edge_connected_directional(
                combined_upper,
                seed_depth=seed_depth,
                lateral_drift_px=lateral_drift_px,
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
            "hard_floor_eff": None if hard_floor_eff is None else float(hard_floor_eff),
            "binary": binary,
            "binary_closed": binary_closed,
            "close_radius": int(close_radius),
            "min_object_px": int(min_object_px),
            "mask": mask,
            "combined_upper": combined_upper,
            "primary_seed": primary_seed,
            "primary_edge_snapshot": primary_edge_snapshot,
            "primary_latched": primary_latched,
            "connectivity_mode": connectivity_mode,
            "directional_lateral_drift_px": int(lateral_drift_px),
        }
