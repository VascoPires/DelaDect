"""Delamination detection workflows.

This module provides a class-based API centred on
:class:`DelaminationDetector`, with edge and diffuse detection exposed as two
peer sub-detectors: :class:`EdgeDetector` (``detector.edge``) and
:class:`DiffuseDetector` (``detector.diffuse``). Shared infrastructure
(preprocessing, caching, combined arbitration) lives directly on
:class:`DelaminationDetector`.

The implementation is intentionally stateful: frame-to-frame latching,
preprocess cache reuse, and debug exports are coordinated by detector
instances rather than stateless helper functions.
"""

from __future__ import annotations

from collections import deque
import json
import logging
from pathlib import Path
import warnings
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, unsharp_mask
from skimage.morphology import closing, disk, reconstruction

from deladect.io.delamination import (
    save_interface_metrics,
    save_mask_bundle,
    store_interface_delamination_results,
    store_interface_masks,
)
from deladect.specimen import DEFAULT_PRIMARY_DELAMINATION_COLOR, Interface, Specimen

logger = logging.getLogger(__name__)

EDGE_OVERLAY_RGBA = (1.0, 0.0, 0.0, 0.35)
DIFFUSE_OVERLAY_RGBA = (0.0, 1.0, 0.0, 0.35)
MULTI_INTERFACE_DEFAULT_COLORS: Tuple[Tuple[float, float, float, float], ...] = (
    (0.89, 0.10, 0.11, 0.35),
    (0.12, 0.47, 0.71, 0.35),
    (0.20, 0.63, 0.17, 0.35),
    (1.00, 0.50, 0.05, 0.35),
    (0.58, 0.40, 0.74, 0.35),
    (0.55, 0.34, 0.29, 0.35),
    (0.89, 0.47, 0.76, 0.35),
    (0.50, 0.50, 0.50, 0.35),
)

PROGRESS_MILESTONES: Tuple[int, ...] = (25, 50, 75, 90)
PREPROCESS_MANIFEST_FILENAME = "preprocess_manifest.json"
DIFFUSE_CRACK_FRAME_POLICIES: Tuple[str, ...] = ("current", "reference_latest", "reference_midpoint")
CRACK_OVERLAY_RGBA: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.95)


def _progress_init(stage: str, total_frames: int, enabled: bool) -> Optional[Dict[int, bool]]:
    """Initialize milestone tracking and print start banner."""
    if not enabled:
        return None
    total = max(0, int(total_frames))
    print(f"[progress] {stage}: start ({total} frames)", flush=True)
    return {milestone: False for milestone in PROGRESS_MILESTONES}


def _progress_update(
    stage: str,
    completed_frames: int,
    total_frames: int,
    state: Optional[Dict[int, bool]],
) -> None:
    """Emit milestone progress updates at 25/50/75/90%."""
    if state is None:
        return
    total = max(1, int(total_frames))
    completed = max(0, int(completed_frames))
    percent = 100.0 * float(completed) / float(total)

    for milestone in PROGRESS_MILESTONES:
        already = bool(state.get(milestone, False))
        if not already and percent >= float(milestone):
            print(f"[progress] {stage}: {milestone}% ({completed}/{total})", flush=True)
            state[milestone] = True


def _progress_done(stage: str, total_frames: int, enabled: bool) -> None:
    """Emit completion banner for a progress-tracked stage."""
    if not enabled:
        return
    total = max(0, int(total_frames))
    print(f"[progress] {stage}: done ({total}/{total})", flush=True)


def _preprocess_manifest_path(cache_dir: Path) -> Path:
    """Return manifest path for one preprocess cache directory."""
    return cache_dir / PREPROCESS_MANIFEST_FILENAME


def _write_preprocess_manifest(cache_dir: Path, manifest: Dict[str, Any]) -> None:
    """Persist cache-level preprocessing metadata to JSON."""
    manifest_path = _preprocess_manifest_path(cache_dir)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _as_scalar(value: Any, default: Any = None) -> Any:
    """Convert numpy scalar/array values to Python scalars safely."""
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        if value.shape == ():
            return value.item()
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
    """Return [start, end) indices used as normalization reference for one frame."""
    idx = max(0, int(frame_idx))
    mode = str(reference_mode)
    window = max(1, int(reference_window))
    skip = max(0, int(reference_skip))

    if mode == "rolling_median":
        end_idx = max(0, idx - skip)
        start_idx = max(0, end_idx - window)
        return int(start_idx), int(end_idx)

    if mode == "static":
        if idx < skip:
            return idx, idx + 1
        return skip, skip + 1

    return idx, idx + 1


def _reference_anchor_index(
    frame_idx: int,
    *,
    start_idx: int,
    end_idx: int,
    policy: str,
) -> int:
    """Select one crack anchor index from a [start, end) reference window."""
    idx = max(0, int(frame_idx))
    start = max(0, int(start_idx))
    end = max(start, int(end_idx))
    policy_name = str(policy)

    if policy_name == "current":
        return idx
    if end <= start:
        return idx
    if policy_name == "reference_latest":
        return end - 1
    if policy_name == "reference_midpoint":
        return start + (end - start - 1) // 2
    return idx


def _build_frame_reference_metadata(
    frame_idx: int,
    *,
    reference_mode: str,
    reference_window: int,
    reference_skip: int,
) -> Dict[str, Any]:
    """Build per-frame preprocessing metadata used for crack-frame alignment."""
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
        policy="reference_midpoint",
    )
    return {
        "ref_start_idx": int(start_idx),
        "ref_end_idx": int(end_idx),
        "ref_anchor_idx": int(anchor_idx),
        "reference_mode": str(reference_mode),
        "reference_window": int(reference_window),
        "reference_skip": int(reference_skip),
    }


def _extract_preprocess_frame_metadata(payload: Any, frame_idx: int) -> Dict[str, Any]:
    """Read per-frame preprocess metadata from cached payload with safe fallbacks."""
    default_meta = _build_frame_reference_metadata(
        frame_idx,
        reference_mode="static",
        reference_window=1,
        reference_skip=0,
    )

    try:
        reference_mode = str(_as_scalar(payload["reference_mode"], default_meta["reference_mode"]))
    except Exception:
        reference_mode = str(default_meta["reference_mode"])
    try:
        reference_window = int(_as_scalar(payload["reference_window"], default_meta["reference_window"]))
    except Exception:
        reference_window = int(default_meta["reference_window"])
    try:
        reference_skip = int(_as_scalar(payload["reference_skip"], default_meta["reference_skip"]))
    except Exception:
        reference_skip = int(default_meta["reference_skip"])

    base_meta = _build_frame_reference_metadata(
        frame_idx,
        reference_mode=reference_mode,
        reference_window=reference_window,
        reference_skip=reference_skip,
    )

    for key in ("ref_start_idx", "ref_end_idx", "ref_anchor_idx"):
        try:
            base_meta[key] = int(_as_scalar(payload[key], base_meta[key]))
        except Exception:
            pass
    return base_meta


def _reference_settings_from_cache_paths(
    processed_cache_paths: Optional[Sequence[Path]],
) -> Dict[str, Any]:
    """Best-effort extraction of reference settings from cached preprocess payloads."""
    defaults = {
        "reference_mode": "static",
        "reference_window": 10,
        "reference_skip": 0,
    }
    if not processed_cache_paths:
        return defaults

    first_path = Path(processed_cache_paths[0])
    if not first_path.exists():
        return defaults

    try:
        with np.load(first_path, allow_pickle=False) as payload:
            meta = _extract_preprocess_frame_metadata(payload, 0)
    except Exception:
        return defaults

    return {
        "reference_mode": str(meta.get("reference_mode", defaults["reference_mode"])),
        "reference_window": max(1, int(meta.get("reference_window", defaults["reference_window"]))),
        "reference_skip": max(0, int(meta.get("reference_skip", defaults["reference_skip"]))),
    }


def _coerce_cracks_by_frame(cracks: Any, frame_count: int) -> List[Any]:
    """Return CrackDect-style crack output as a frame-indexed Python list.

    CrackDect outputs may be Python sequences, object arrays (for ragged frame
    results), or dense numeric arrays with shape ``(frames, cracks, 2, 2)``.
    A ``(cracks, 2, 2)`` array is also accepted for a single-frame stack.
    """
    if isinstance(cracks, np.ndarray):
        if cracks.ndim == 4 and cracks.shape[-2:] == (2, 2):
            frame_cracks = [cracks[idx] for idx in range(cracks.shape[0])]
        elif cracks.ndim == 3 and cracks.shape[-2:] == (2, 2):
            if frame_count == 1:
                frame_cracks = [cracks]
            elif cracks.shape[0] == frame_count:
                frame_cracks = [cracks[idx : idx + 1] for idx in range(frame_count)]
            else:
                raise ValueError(
                    "A (cracks, 2, 2) array is only unambiguous for one frame; "
                    "for multiple frames provide a per-frame sequence or an "
                    "array shaped (frames, cracks, 2, 2)."
                )
        elif cracks.ndim == 1 and cracks.dtype == object:
            frame_cracks = list(cracks)
        else:
            raise ValueError(
                "Unsupported cracks array shape. Expected an object array by frame, "
                "(frames, cracks, 2, 2), or (cracks, 2, 2) for one frame."
            )
    else:
        try:
            frame_cracks = list(cracks)
        except TypeError as exc:
            raise TypeError("cracks must be a per-frame sequence or NumPy array.") from exc

    if len(frame_cracks) > frame_count:
        frame_cracks = frame_cracks[:frame_count]
    elif len(frame_cracks) < frame_count:
        frame_cracks.extend([[] for _ in range(frame_count - len(frame_cracks))])
    return frame_cracks


class DelaminationDetector:
    """Main delamination API for a specimen/interface pair.

    Parameters
    ----------
    specimen:
        Specimen containing image stacks, geometric metadata, and output paths.
    interface:
        Interface currently analysed (used for colors, names, and persistence keys).
    history_clamp:
        If ``True``, run minimum-history clamping before reference normalization.
    save_preprocess_outputs:
        If ``True``, save the raw/baseline/processed preview triplet while preprocessing.
    preprocess_outputs_dirname:
        Output folder name under the specimen results root for preprocess previews.
    """

    def __init__(
        self,
        specimen: Specimen,
        interface: Interface,
        *,
        history_clamp: bool = True,
        save_preprocess_outputs: bool = False,
        preprocess_outputs_dirname: str = "Preprocessor_outputs",
    ) -> None:
        """Create a detector bound to one specimen/interface pair."""
        self.specimen = specimen
        self.interface = interface
        self.history_clamp = bool(history_clamp)
        self.save_preprocess_outputs = bool(save_preprocess_outputs)
        self.preprocess_outputs_dirname = str(preprocess_outputs_dirname)
        self._stack_override = self._resolve_stack_override()
        self._notice_flags: Dict[str, bool] = {}

        self.edge = EdgeDetector(self)
        self.diffuse = DiffuseDetector(self)


    def save_delamination_overlay(
        self,
        *,
        frame_idx: int,
        overlay_type: str,
        overlay_dirname: str = "delamination",
        masks_dirname: str = "masks",
        edge_exclusion_px: int = 5,
        save_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Save one overlay view for a specific frame.

        The helper reads previously saved mask bundles from
        ``<results>/<overlay_dirname>/both/<masks_dirname>`` and renders one
        of the supported views.

        Parameters
        ----------
        frame_idx:
            Frame index in the raw stack.
        overlay_type:
            One of ``"diffuse"``, ``"edge"``, ``"both"``, or ``"total_dela"``.
        overlay_dirname:
            Base output folder used by detection workflows.
        masks_dirname:
            Mask bundle subfolder name.
        edge_exclusion_px:
            Fallback dilation radius when exclusion masks are not present on disk.
        save_path:
            Optional explicit output path. If omitted, a default folder/filename is used.

        Returns
        -------
        dict[str, Any]
            ``{"path": pathlib.Path}`` — path of the written image.
        """
        overlay_type = str(overlay_type).lower()
        if overlay_type not in {"diffuse", "edge", "both", "total_dela"}:
            raise ValueError("overlay_type must be one of: 'diffuse', 'edge', 'both', 'total_dela'.")

        stacks = self._select_stacks()
        raw_stack = getattr(self.specimen, "image_stack_full", None) or stacks.get("full")
        if raw_stack is None:
            raise ValueError("Cannot save overlays without a full raw image stack.")

        raw_frame = _ensure_uint8(raw_stack[frame_idx])
        frame_key = f"frame_{frame_idx:04d}"
        masks_root = self.specimen.results_dir(overlay_dirname, "both", masks_dirname)
        edge_raw_path = masks_root / "edge_raw.npz"
        edge_exclusion_path = masks_root / "edge_exclusion.npz"
        diffuse_final_path = masks_root / "diffuse_final.npz"
        diffuse_raw_path = masks_root / "diffuse_raw.npz"
        combined_path = masks_root / "combined.npz"

        edge_raw = _load_mask_frame(edge_raw_path, frame_key)
        if edge_raw is None:
            raise ValueError("Edge masks are missing. Run detect_both_delaminations with save_masks=True.")

        edge_exclusion = _load_mask_frame(edge_exclusion_path, frame_key)
        if edge_exclusion is None:
            edge_exclusion = _dilate_edge_mask(edge_raw, max(0, int(edge_exclusion_px)))

        diffuse_final = _load_mask_frame(diffuse_final_path, frame_key)
        if diffuse_final is None:
            diffuse_final = _load_mask_frame(diffuse_raw_path, frame_key)

        if overlay_type in {"diffuse", "both"} and diffuse_final is None:
            raise ValueError("Diffuse masks are missing. Run detect_both_delaminations with save_masks=True.")

        combined = _load_mask_frame(combined_path, frame_key)
        if combined is None and diffuse_final is not None:
            combined = edge_exclusion | diffuse_final

        if save_path is None:
            if overlay_type == "edge":
                save_dir = self.specimen.results_dir(overlay_dirname, "edge", "overlays")
                save_path = save_dir / f"edge_overlay_{frame_idx:04d}.png"
            elif overlay_type == "diffuse":
                save_dir = self.specimen.results_dir(overlay_dirname, "diffuse", "overlays")
                save_path = save_dir / f"diffuse_overlay_{frame_idx:04d}.png"
            elif overlay_type == "both":
                save_dir = self.specimen.results_dir(overlay_dirname, "both", "overlays")
                save_path = save_dir / f"combined_overlay_{frame_idx:04d}.png"
            else:
                save_dir = self.specimen.results_dir(overlay_dirname, "total", "overlays")
                save_path = save_dir / f"total_overlay_{frame_idx:04d}.png"

        if overlay_type == "edge":
            _save_edge_overlay(raw_frame, edge_exclusion, save_path, view="mask")
        elif overlay_type == "diffuse":
            _save_diffuse_overlay(raw_frame, diffuse_final, save_path)
        elif overlay_type == "both":
            _save_combined_overlay(
                raw_frame,
                edge_mask=edge_exclusion,
                diffuse_mask=diffuse_final,
                save_path=save_path,
                view="classified",
                edge_color=EDGE_OVERLAY_RGBA,
                diffuse_color=DIFFUSE_OVERLAY_RGBA,
                union_color=self.interface.delamination_color_rgba,
            )
        else:
            if combined is None:
                raise ValueError("Combined masks are missing. Run detect_both_delaminations with save_masks=True.")
            _save_single_overlay(raw_frame, combined, save_path, self.interface.delamination_color_rgba)

        return {"path": save_path}

    def detect_both_delaminations(
        self,
        *,
        cracks: Optional[Sequence[np.ndarray]] = None,
        avg_crack_width_px: float,
        processed_cache_paths: Optional[List[Path]] = None,
        processed_stack: Optional[List[np.ndarray]] = None,
        save_overlays: bool = True,
        overlay_dirname: str = "delamination",
        overlay_view: str = "union",
        save_component_overlays: bool = False,
        edge_overlay_view: str = "both",
        edge_exclusion_px: int = 5,
        save_masks: bool = True,
        masks_dirname: str = "masks",
        save_metrics: bool = True,
        metrics_filename: str = "frame_metrics.csv",
        max_frames: Optional[int] = None,
        edge_params: Optional[Dict[str, Any]] = None,
        diffuse_params: Optional[Dict[str, Any]] = None,
        max_center_px: Optional[float] = None,
        max_angle_deg: float = 15.0,
        max_cost: float = 1.8,
        return_masks: bool = False,
        return_intermediates: bool = False,
        debug: bool = False,
        save_edge_debug: bool = False,
        progress: bool = False,
    ) -> Dict[str, Any]:
        """Detect edge and crack-tracking diffuse delamination, then resolve overlap.

        Overlaps are resolved in favour of edge delamination. An optional
        edge-exclusion halo can be applied before arbitration so excluded pixels are
        counted as edge damage.  The diffuse step always uses
        :meth:`diffuse_crack_tracking`.

        Parameters
        ----------
        cracks:
            Per-frame cracks required for diffuse ROI construction.
        avg_crack_width_px:
            Average crack width in pixels; forwarded to :meth:`diffuse_crack_tracking`.
        processed_cache_paths, processed_stack:
            Optional preprocessed source.  If omitted, static-reference preprocessing
            is executed automatically.  When providing pre-computed frames, they must
            have been produced with ``reference_mode="static"``; rolling-median
            preprocessed frames are only appropriate for :meth:`detect_edge_multi`.
        save_overlays:
            If ``True``, save combined overlays per frame.
        overlay_dirname:
            Base output folder used under specimen results.
        overlay_view:
            ``"union"`` for one-color union masks or ``"classified"`` for edge/diffuse colors.
        save_component_overlays:
            If ``True``, also save component overlays generated in this combined run.
        edge_overlay_view:
            View mode for optional edge component overlays (``mask``, ``line``, ``both``).
        edge_exclusion_px:
            Edge dilation radius used for overlap arbitration and edge metrics.
        save_masks:
            If ``True``, persist edge/diffuse/combined bundles as ``.npz``.
        masks_dirname:
            Mask bundle subfolder name.
        save_metrics:
            If ``True``, write per-frame metrics CSV.
        metrics_filename:
            Filename for metrics CSV.
        max_frames:
            Optional cap on processed frames.
        edge_params, diffuse_params:
            Optional parameter overrides passed to component detectors.
        max_center_px, max_angle_deg, max_cost:
            Track-assignment thresholds forwarded to :meth:`diffuse_crack_tracking`.
        return_masks:
            If ``True``, include masks in the returned dictionary.
        debug:
            If ``True``, include edge/diffuse debug payloads.

        Returns
        -------
        dict[str, Any]
            Result dictionary containing metrics, output paths, parameters, and
            optional masks/debug payloads.
        """
        if cracks is None:
            raise ValueError("Diffuse delamination requires `cracks` to be provided.")
        if processed_cache_paths and processed_stack:
            raise ValueError("Provide either processed_cache_paths or processed_stack, not both.")
        if overlay_view not in {"union", "classified"}:
            raise ValueError("overlay_view must be one of: 'union', 'classified'.")

        stacks = self._select_stacks()
        raw_stack = getattr(self.specimen, "image_stack_full", None) or stacks.get("full")
        if save_overlays and raw_stack is None:
            raise ValueError("Cannot save overlays without a full raw image stack.")

        if processed_cache_paths is None and processed_stack is None:
            stack = getattr(self.specimen, "image_stack_full", None) or stacks.get("full")
            if stack is None:
                raise ValueError("Specimen has no full image stack to preprocess.")
            auto_key = f"both_auto_{self.interface.name}"
            processed_cache_paths = self.preprocess_stack_to_disk(
                stack,
                key=auto_key,
                max_frames=max_frames,
                cache_dirname="Preprocessor_cache",
                reference_mode="static",
                progress=progress,
            )["cache_paths"]

        edge_result = self.edge.detect_primary(
            save_overlays=save_component_overlays,
            overlay_dirname=overlay_dirname,
            overlay_view=edge_overlay_view,
            max_frames=max_frames,
            params=edge_params,
            debug=debug,
            save_debug_outputs=save_edge_debug,
            progress=progress,
        )
        edge_masks, edge_debug = edge_result["masks"], edge_result["debug"]
        from deladect.detection.crack_tracking import normalize_detections as _normalize_det

        if processed_stack is not None:
            proc_frames_list = list(processed_stack)[:max_frames] if max_frames else list(processed_stack)
            selected_indices_list = list(range(len(proc_frames_list)))
        else:
            assert processed_cache_paths is not None
            paths_to_load = processed_cache_paths[:max_frames] if max_frames else processed_cache_paths
            loaded = list(self.iter_preprocessed_cache(paths_to_load))
            proc_frames_list = [f for _, f in loaded]
            selected_indices_list = list(range(len(proc_frames_list)))

        cracks_by_frame = _coerce_cracks_by_frame(cracks, len(proc_frames_list))
        crack_frames_normalized = [
            _normalize_det(cracks_by_frame[i])
            for i in selected_indices_list
        ]

        crack_tracking_result: Dict[str, Any] = self.diffuse.diffuse_crack_tracking(
            proc_frames_list,
            crack_frames_normalized,
            selected_indices_list,
            avg_crack_width_px=avg_crack_width_px,
            diffuse_params=self.diffuse._resolve_diffuse_params(diffuse_params),
            max_center_px=max_center_px,
            max_angle_deg=max_angle_deg,
            max_cost=max_cost,
            return_intermediates=return_intermediates,
        )
        ct_frame_masks = crack_tracking_result["frame_masks"]
        _ref_shape = proc_frames_list[0].shape[:2] if proc_frames_list else (1, 1)
        diffuse_masks: Dict[str, np.ndarray] = {
            f"frame_{i:04d}": ct_frame_masks.get(i, np.zeros(_ref_shape, dtype=bool))
            for i in selected_indices_list
        }

        # For region-override specimens (upper/lower/middle splits), diffuse_crack_tracking
        # ran on the full-frame preprocessed stack and may have produced detections in the
        # upper and lower rows that belong exclusively to edge detection. Zero those rows out
        # before latching so the bad pixels never accumulate.
        if self._uses_stack_overrides():
            _ov_stacks = self._select_stacks()
            _upper_s = _ov_stacks.get("upper")
            _lower_s = _ov_stacks.get("lower")
            if _upper_s is not None and _lower_s is not None:
                _uh = int(np.asarray(_ensure_uint8(_upper_s[0])).shape[0])
                _lh = int(np.asarray(_ensure_uint8(_lower_s[0])).shape[0])
                for _fk in list(diffuse_masks.keys()):
                    _m = diffuse_masks[_fk].copy()
                    _m[:_uh, :] = False
                    if _lh > 0:
                        _m[-_lh:, :] = False
                    diffuse_masks[_fk] = _m

        frame_keys = sorted(set(edge_masks.keys()) & set(diffuse_masks.keys()))
        if not frame_keys:
            raise ValueError("No overlapping frame keys between edge and diffuse masks.")

        # Latch both masks: once a pixel is flagged as damaged it stays flagged.
        _cum_edge: Optional[np.ndarray] = None
        for _fk in frame_keys:
            _m = np.asarray(edge_masks[_fk], dtype=bool)
            _cum_edge = _m if _cum_edge is None else (_cum_edge | _m)
            edge_masks[_fk] = _cum_edge.copy()

        _cum_diffuse: Optional[np.ndarray] = None
        for _fk in frame_keys:
            _m = np.asarray(diffuse_masks[_fk], dtype=bool)
            _cum_diffuse = _m if _cum_diffuse is None else (_cum_diffuse | _m)
            diffuse_masks[_fk] = _cum_diffuse.copy()

        metrics_rows: List[Dict[str, Any]] = []
        diffuse_final_masks: Dict[str, np.ndarray] = {}
        combined_masks: Dict[str, np.ndarray] = {}
        overlap_masks: Dict[str, np.ndarray] = {}
        edge_exclusion_masks: Dict[str, np.ndarray] = {}
        exclusion_radius = max(0, int(edge_exclusion_px))
        progress_state = _progress_init("combined_delamination", len(frame_keys), progress)

        for idx, frame_key in enumerate(frame_keys):
            edge_raw = np.asarray(edge_masks[frame_key], dtype=bool)
            edge_exclusion = _dilate_edge_mask(edge_raw, exclusion_radius)
            diffuse_raw = np.asarray(diffuse_masks[frame_key], dtype=bool)
            diffuse_final, combined, overlap = _apply_edge_precedence(edge_exclusion, diffuse_raw)
            diffuse_final_masks[frame_key] = diffuse_final
            combined_masks[frame_key] = combined
            overlap_masks[frame_key] = overlap
            edge_exclusion_masks[frame_key] = edge_exclusion

            frame_idx = _frame_index_from_key(frame_key)
            if frame_idx is not None:
                frame_pixels = int(edge_exclusion.size)
                metrics_rows.append(
                    _build_metrics_row(
                        frame_idx=frame_idx,
                        frame_pixels=frame_pixels,
                        edge_mask=edge_exclusion,
                        diffuse_raw=diffuse_raw,
                        diffuse_final=diffuse_final,
                        overlap_mask=overlap,
                        combined_mask=combined,
                    )
                )

            if save_component_overlays and raw_stack is not None and frame_idx is not None:
                raw_frame = _ensure_uint8(raw_stack[frame_idx])
                diffuse_overlay_dir = self.specimen.results_dir(overlay_dirname, "diffuse", "overlays")
                diffuse_overlay_path = diffuse_overlay_dir / f"diffuse_overlay_{frame_idx:04d}.png"
                frame_cracks = cracks_by_frame[frame_idx] if frame_idx < len(cracks_by_frame) else None
                if self._uses_stack_overrides():
                    upper_h = int(np.asarray(_ensure_uint8(getattr(self.specimen, "image_stack_upper")[frame_idx])).shape[0])
                    middle_h = int(np.asarray(_ensure_uint8(getattr(self.specimen, "image_stack_middle")[frame_idx])).shape[0])
                    frame_cracks = self._cracks_for_full_overlay(
                        frame_cracks,
                        upper_height=upper_h,
                        middle_height=middle_h,
                        full_height=int(raw_frame.shape[0]),
                    )
                _save_diffuse_overlay(raw_frame, diffuse_final, diffuse_overlay_path, cracks=frame_cracks)

            if save_overlays and raw_stack is not None:
                if frame_idx is None:
                    continue
                raw_frame = _ensure_uint8(raw_stack[frame_idx])
                overlay_dir = self.specimen.results_dir(overlay_dirname, "both", "overlays")
                overlay_path = overlay_dir / f"combined_overlay_{frame_idx:04d}.png"
                _save_combined_overlay(
                    raw_frame,
                    edge_mask=edge_exclusion,
                    diffuse_mask=diffuse_final,
                    save_path=overlay_path,
                    view=overlay_view,
                    edge_color=EDGE_OVERLAY_RGBA,
                    diffuse_color=DIFFUSE_OVERLAY_RGBA,
                    union_color=self.interface.delamination_color_rgba,
                    cracks=(
                        self._cracks_for_full_overlay(
                            cracks_by_frame[frame_idx] if frame_idx < len(cracks_by_frame) else None,
                            upper_height=int(np.asarray(_ensure_uint8(getattr(self.specimen, "image_stack_upper")[frame_idx])).shape[0])
                            if self._uses_stack_overrides()
                            else 0,
                            middle_height=int(np.asarray(_ensure_uint8(getattr(self.specimen, "image_stack_middle")[frame_idx])).shape[0])
                            if self._uses_stack_overrides()
                            else int(raw_frame.shape[0]),
                            full_height=int(raw_frame.shape[0]),
                        )
                        if self._uses_stack_overrides()
                        else (cracks_by_frame[frame_idx] if frame_idx < len(cracks_by_frame) else None)
                    ),
                )

            _progress_update("combined_delamination", idx + 1, len(frame_keys), progress_state)

        _progress_done("combined_delamination", len(frame_keys), progress)

        metrics_df = pd.DataFrame(metrics_rows)

        paths: Dict[str, Optional[str]] = {
            "edge_raw_masks": None,
            "edge_exclusion_masks": None,
            "diffuse_raw_masks": None,
            "diffuse_masks": None,
            "combined_masks": None,
            "metrics": None,
            "combined_overlays": None if not save_overlays else str(
                self.specimen.results_dir(overlay_dirname, "both", "overlays")
            ),
        }

        if save_masks:
            masks_root = self.specimen.results_dir(overlay_dirname, "both", masks_dirname)
            paths["edge_raw_masks"] = str(save_mask_bundle(edge_masks, masks_root / "edge_raw.npz"))
            paths["edge_exclusion_masks"] = str(
                save_mask_bundle(edge_exclusion_masks, masks_root / "edge_exclusion.npz")
            )
            paths["diffuse_raw_masks"] = str(save_mask_bundle(diffuse_masks, masks_root / "diffuse_raw.npz"))
            paths["diffuse_masks"] = str(save_mask_bundle(diffuse_final_masks, masks_root / "diffuse_final.npz"))
            paths["combined_masks"] = str(save_mask_bundle(combined_masks, masks_root / "combined.npz"))

        if save_metrics:
            metrics_dir = self.specimen.results_dir(overlay_dirname, "both", "metrics")
            metrics_path = save_interface_metrics(metrics_df, metrics_dir / metrics_filename)
            paths["metrics"] = str(metrics_path)

        store_interface_delamination_results(
            self.interface,
            diffuse_raw_path=Path(paths["diffuse_raw_masks"]) if paths["diffuse_raw_masks"] else None,
            diffuse_path=Path(paths["diffuse_masks"]) if paths["diffuse_masks"] else None,
            combined_path=Path(paths["combined_masks"]) if paths["combined_masks"] else None,
            metrics_path=Path(paths["metrics"]) if paths["metrics"] else None,
        )

        result: Dict[str, Any] = {
            "metrics": metrics_df,
            "paths": paths,
            "params": {
                "edge_exclusion_px": exclusion_radius,
            },
        }
        if return_masks:
            result["masks"] = {
                "edge_raw": edge_masks,
                "edge_exclusion": edge_exclusion_masks,
                "diffuse_raw": diffuse_masks,
                "diffuse": diffuse_final_masks,
                "combined": combined_masks,
                "overlap": overlap_masks,
            }
        if debug:
            result["debug"] = {
                "edge": edge_debug,
            }
        result["crack_tracking"] = crack_tracking_result
        if return_intermediates:
            result["_debug_internals"] = {
                "proc_frames": proc_frames_list,
                "selected_indices": selected_indices_list,
                "crack_frames_normalized": crack_frames_normalized,
            }

        return result

    def _resolve_stack_override(self) -> Dict[str, bool]:
        """Return availability flags for explicit upper/lower/middle paths."""
        return {
            "upper": self.specimen.path_upper_border is not None,
            "lower": self.specimen.path_lower_border is not None,
            "middle": self.specimen.path_middle is not None,
        }

    def _uses_stack_overrides(self) -> bool:
        """Return ``True`` when all region overrides are available."""
        return all(self._stack_override.values())

    def _select_stacks(self) -> Dict[str, Optional[List[np.ndarray]]]:
        """Select either override stacks or default full-stack analysis."""
        if self._uses_stack_overrides():
            return {
                "upper": getattr(self.specimen, "image_stack_upper", None),
                "lower": getattr(self.specimen, "image_stack_lower", None),
                "middle": getattr(self.specimen, "image_stack_middle", None),
                "full": None,
            }
        return {
            "upper": None,
            "lower": None,
            "middle": None,
            "full": getattr(self.specimen, "image_stack_full", None),
        }

    @staticmethod
    def _cracks_for_full_overlay(
        cracks: Optional[Sequence[np.ndarray]],
        *,
        upper_height: int,
        middle_height: int,
        full_height: int,
    ) -> Optional[List[np.ndarray]]:
        """Shift middle-stack crack coordinates into full-frame coordinates for overlays.

        Crack detection in region mode runs on the middle stack, so crack ``y`` values
        are in ``[0, middle_height)``. Full-frame overlays require ``+ upper_height``.
        If cracks already appear to be in full-frame coordinates, they are left unchanged.
        """
        if cracks is None:
            return None

        if upper_height <= 0 or middle_height <= 0 or full_height <= middle_height:
            return [np.asarray(segment, dtype=float).reshape(-1, 2) for segment in cracks]

        prepared: List[np.ndarray] = []
        y_values: List[float] = []
        for segment in cracks:
            try:
                arr = np.asarray(segment, dtype=float).reshape(-1, 2)
            except Exception:
                continue
            if arr.shape[0] < 2:
                continue
            prepared.append(arr)
            y_values.extend(arr[:, 0].tolist())

        if not prepared:
            return []

        y_min = min(y_values)
        y_max = max(y_values)

        # Heuristic: middle-stack coordinates sit near [0, middle_height).
        if y_min >= -1.0 and y_max <= float(middle_height) + 1.0:
            shifted: List[np.ndarray] = []
            for arr in prepared:
                arr_shift = arr.copy()
                arr_shift[:, 0] += float(upper_height)
                shifted.append(arr_shift)
            return shifted

        return prepared

    def apply_minimum_history(
        self,
        stack: Optional[List[np.ndarray]],
        *,
        key: str,
        history_buffers: Dict[str, Any],
        mode: str = "running",
        window_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Apply minimum history clamp across an entire stack.

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

        buffer = history_buffers.get(key)
        if mode == "rolling" and (buffer is None or not isinstance(buffer, deque)):
            buffer = deque(maxlen=effective_window)

        processed: List[np.ndarray] = []
        for frame in stack:
            if mode == "running":
                history = history_buffers.get(key)
                if history is None:
                    history = frame.copy()
                else:
                    history = np.minimum(history, frame)
                history_buffers[key] = history
                processed.append(np.minimum(frame, history))
            else:
                if buffer is None:
                    buffer = deque(maxlen=effective_window)
                buffer.append(frame)
                history_buffers[key] = buffer
                processed.append(np.minimum.reduce(list(buffer)))

        return {"frames": processed}

    def apply_reference_normalization(
        self,
        stack: Optional[List[np.ndarray]],
        *,
        reference_mode: str = "static",
        reference_window: int = 10,
        reference_skip: int = 0,
        output_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Apply reference normalisation and return processed stacks.

        Returns
        -------
        dict[str, Any]
            ``{"processed_frames": list[np.ndarray], "baseline_frames": list[np.ndarray | None]}``.
        """
        if stack is None:
            raise ValueError("A valid image stack is required for reference normalisation.")

        raw_stack = [_ensure_uint8(frame) for frame in stack]
        processed_frames: List[np.ndarray] = []
        baseline_frames: List[Optional[np.ndarray]] = []

        plot_state = None
        output_dir = self._resolve_preprocess_output_dir(output_key)
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            image_shape = raw_stack[0].shape if raw_stack else None
            plot_state = _prepare_preprocess_figure(image_shape)

        reference_window = max(1, int(reference_window))
        reference_skip = max(0, int(reference_skip))

        baseline_static_float: Optional[np.ndarray] = None

        for idx, frame in enumerate(raw_stack):
            frame_float = _frame_to_float(frame)
            baseline_float: Optional[np.ndarray] = None

            if reference_mode == "rolling_median":
                end_idx = max(0, idx - reference_skip)
                start_idx = max(0, end_idx - reference_window)
                window_frames: List[np.ndarray] = []
                for ref_idx in range(start_idx, end_idx):
                    if ref_idx < 0 or ref_idx >= len(raw_stack):
                        continue
                    window_frames.append(_frame_to_float(raw_stack[ref_idx]))
                baseline_float = np.median(np.stack(window_frames, axis=0), axis=0) if window_frames else frame_float
            elif reference_mode == "static":
                if baseline_static_float is None and idx >= reference_skip:
                    baseline_static_float = frame_float
                baseline_float = baseline_static_float if baseline_static_float is not None else frame_float

            processed, baseline_uint8 = self._normalize_reference_frame(
                frame_uint8=frame,
                frame_float=frame_float,
                baseline_float=baseline_float,
            )

            processed_frames.append(processed)
            baseline_frames.append(baseline_uint8)

            self._maybe_save_preprocess_plot(
                plot_state=plot_state,
                output_dir=output_dir,
                frame_idx=idx,
                raw_frame=frame,
                frame_float=frame_float,
                baseline_float=baseline_float,
                processed=processed,
            )

        if plot_state is not None:
            _close_preprocess_figure(plot_state)

        return {"processed_frames": processed_frames, "baseline_frames": baseline_frames}

    def normalize_delamination_stack(
        self,
        stack: Optional[List[np.ndarray]],
        *,
        key: str,
        history_buffers: Dict[str, Any],
        history_mode: str = "running",
        history_window_size: Optional[int] = None,
        reference_mode: str = "static",
        reference_window: int = 10,
        reference_skip: int = 0,
    ) -> Dict[str, Any]:
        """Apply minimum history then reference normalization, returning processed frames only.

        Returns
        -------
        dict[str, Any]
            ``{"frames": list[np.ndarray]}``.
        """
        if stack is None:
            raise ValueError("A valid image stack is required for delamination normalization.")

        history_stack = (
            self.apply_minimum_history(
                stack,
                key=key,
                history_buffers=history_buffers,
                mode=history_mode,
                window_size=history_window_size,
            )["frames"]
            if self.history_clamp
            else [_ensure_uint8(frame) for frame in stack]
        )

        normalization_result = self.apply_reference_normalization(
            history_stack,
            reference_mode=reference_mode,
            reference_window=reference_window,
            reference_skip=reference_skip,
            output_key=key,
        )
        return {"frames": normalization_result["processed_frames"]}

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
        """Preprocess a stack and persist processed/baseline frames to ``.npz``.

        This method runs the same normalization stack used by detectors, but stores
        each frame payload on disk so downstream detection can be repeated without
        re-running preprocessing.

        Parameters
        ----------
        stack:
            Raw grayscale-compatible frames.
        key:
            Cache key used to build ``<results>/<cache_dirname>/<key>/``.
        max_frames:
            Optional cap on number of frames written.
        history_mode:
            ``"running"`` minimum history or ``"rolling"`` minimum history.
        history_window_size:
            Rolling history window size when ``history_mode="rolling"``.
        reference_mode:
            Reference normalization mode.  Two options are supported:

            ``"static"``
                Use a fixed early-frame baseline for the entire stack.  This is the
                standard mode for all general delamination detection
                (:meth:`detect_primary`, :meth:`detect_both_delaminations`,
                :meth:`~DelaminationDetector.detect_diffuse_delamination`).

            ``"rolling_median"``
                Use a rolling median of recent frames as the reference.  Reserved
                for multi-interface edge detection (:meth:`detect_edge_multi`),
                where the propagating damage front requires an adaptive baseline.
                Do not use this mode with single-interface detection methods.

        reference_window:
            Reference window size for rolling median.

            Tip: ``reference_mode="rolling_median"`` with ``reference_window=1``
            behaves like a single-frame rolling reference (legacy ``rolling``-style
            baseline): after warmup, frame ``n`` is normalized against roughly
            ``n - (reference_skip + 1)``.
        reference_skip:
            Number of newest prior frames skipped from rolling reference.

            With ``reference_window=1`` this becomes an explicit lag control:

            - ``reference_skip=0`` -> previous frame (``n-1``)
            - ``reference_skip=1`` -> two frames behind (``n-2``)
            - ``reference_skip=2`` -> three frames behind (``n-3``)

            Early frames with insufficient history fall back to the current frame.
        cache_dirname:
            Root cache folder under specimen results.

        Returns
        -------
        dict[str, Any]
            ``{"cache_paths": list[pathlib.Path]}`` — ordered list of written ``.npz`` paths.
        """
        if stack is None:
            raise ValueError("A valid image stack is required for preprocessing.")
        if history_mode not in {"running", "rolling"}:
            raise ValueError("history_mode must be 'running' or 'rolling'.")

        cache_dir = self._resolve_preprocess_cache_dir(cache_dirname, key)
        cache_dir.mkdir(parents=True, exist_ok=True)

        output_dir = self._resolve_preprocess_output_dir(key)
        plot_state = None

        stack_seq: Sequence[np.ndarray]
        if hasattr(stack, "__len__") and hasattr(stack, "__getitem__"):
            stack_seq = cast(Sequence[np.ndarray], stack)
        else:
            stack_seq = list(stack)
        total_frames = len(stack_seq)
        limit = total_frames if max_frames is None else min(max_frames, total_frames)
        progress_state = _progress_init("preprocess_stack", limit, progress)

        history: Optional[np.ndarray] = None
        history_buffer: Optional[Deque[np.ndarray]] = None

        reference_window = max(1, int(reference_window))
        reference_skip = max(0, int(reference_skip))
        ref_buffer: Deque[np.ndarray] = deque(maxlen=reference_window + reference_skip + 1)
        baseline_static_float: Optional[np.ndarray] = None

        cache_paths: List[Path] = []

        for idx in range(limit):
            raw = _ensure_uint8(stack_seq[idx])
            if self.history_clamp:
                if history_mode == "running":
                    history = raw if history is None else np.minimum(history, raw)
                    history_frame = np.minimum(raw, history)
                else:
                    if history_buffer is None:
                        history_buffer = deque(maxlen=history_window_size or 10)
                    history_buffer.append(raw)
                    history_frame = np.minimum.reduce(list(history_buffer))
            else:
                history_frame = raw

            frame_float = _frame_to_float(history_frame)
            baseline_float: Optional[np.ndarray] = None
            if reference_mode == "rolling_median":
                buffer_list = list(ref_buffer)
                end = max(0, len(buffer_list) - reference_skip)
                start = max(0, end - reference_window)
                window_frames = buffer_list[start:end]
                baseline_float = np.median(np.stack(window_frames, axis=0), axis=0) if window_frames else frame_float
            elif reference_mode == "static":
                if baseline_static_float is None and idx >= reference_skip:
                    baseline_static_float = frame_float
                baseline_float = baseline_static_float if baseline_static_float is not None else frame_float

            processed, baseline_uint8 = self._normalize_reference_frame(
                frame_uint8=history_frame,
                frame_float=frame_float,
                baseline_float=baseline_float,
            )

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

            if output_dir is not None and plot_state is None:
                plot_state = _prepare_preprocess_figure(raw.shape)
            self._maybe_save_preprocess_plot(
                plot_state=plot_state,
                output_dir=output_dir,
                frame_idx=idx,
                raw_frame=raw,
                frame_float=frame_float,
                baseline_float=baseline_float,
                processed=processed,
            )

            if reference_mode == "rolling_median":
                ref_buffer.append(frame_float)

            _progress_update("preprocess_stack", idx + 1, limit, progress_state)

        if plot_state is not None:
            _close_preprocess_figure(plot_state)

        manifest = {
            "version": 1,
            "frame_count": int(limit),
            "history_mode": str(history_mode),
            "history_window_size": None if history_window_size is None else int(history_window_size),
            "reference_mode": str(reference_mode),
            "reference_window": int(reference_window),
            "reference_skip": int(reference_skip),
        }
        _write_preprocess_manifest(cache_dir, manifest)

        _progress_done("preprocess_stack", limit, progress)

        return {"cache_paths": cache_paths}

    def iter_preprocessed_cache(self, cache_paths: List[Path]):
        """Yield (index, processed_frame) from cached preprocess frames."""
        for idx, path in enumerate(cache_paths):
            with np.load(path, allow_pickle=False) as payload:
                processed = payload["processed"]
            yield idx, processed

    def iter_preprocessed_cache_with_metadata(self, cache_paths: List[Path]):
        """Yield (index, processed_frame, metadata) from cached preprocess frames."""
        for idx, path in enumerate(cache_paths):
            with np.load(path, allow_pickle=False) as payload:
                processed = payload["processed"]
                metadata = _extract_preprocess_frame_metadata(payload, idx)
            yield idx, processed, metadata

    def _images_threshold(self, image: np.ndarray, window_edge: Tuple[int, int]) -> float:
        """Compute fallback Otsu threshold after directional min/max filtering."""
        wy, wx = max(1, int(window_edge[0])), max(1, int(window_edge[1]))
        filtered_max = ndi.maximum_filter(image, size=(wy, wx))
        filtered_min = ndi.minimum_filter(filtered_max, size=(wy, wx))
        return float(threshold_otsu(filtered_min))

    def _kmeans_threshold(
        self,
        image: np.ndarray,
        fallback: float,
        *,
        max_iter: int = 20,
        tol: float = 1e-2,
    ) -> float:
        """Estimate a two-cluster threshold with safe fallback behaviour."""
        data = np.asarray(image, dtype=np.float32).reshape(-1)
        if data.size == 0:
            return float(fallback)
        mask = np.isfinite(data)
        values = data[mask]
        if values.size == 0:
            return float(fallback)
        v_min = float(values.min())
        v_max = float(values.max())
        if v_max - v_min < 1e-3:
            return float(fallback)

        centroids = np.array([v_min, v_max], dtype=np.float32)
        for _ in range(max_iter):
            distances = np.abs(values[:, None] - centroids[None, :])
            labels = np.argmin(distances, axis=1)
            new_centroids = centroids.copy()
            updated = False
            for idx in (0, 1):
                cluster_vals = values[labels == idx]
                if cluster_vals.size == 0:
                    new_centroids[idx] = v_min if idx == 0 else v_max
                    continue
                candidate = float(cluster_vals.mean())
                if abs(candidate - centroids[idx]) > tol:
                    updated = True
                new_centroids[idx] = candidate
            centroids = new_centroids
            if not updated:
                break

        dark, bright = sorted(float(value) for value in centroids)
        if abs(bright - dark) < 1e-3:
            return float(fallback)
        return float(0.5 * (dark + bright))

    def _resolve_preprocess_cache_dir(self, cache_dirname: str, key: str) -> Path:
        """Resolve cache output directory for a preprocessing run key."""
        return self.specimen.results_dir(cache_dirname, key)

    def _resolve_preprocess_output_dir(self, output_key: Optional[str]) -> Optional[Path]:
        """Return preprocess preview directory when preview export is enabled."""
        if not self.save_preprocess_outputs:
            return None
        parts = [self.preprocess_outputs_dirname]
        if output_key:
            parts.append(str(output_key))
        return self.specimen.results_dir(*parts)

    def _normalize_reference_frame(
        self,
        *,
        frame_uint8: np.ndarray,
        frame_float: np.ndarray,
        baseline_float: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply ratio-based normalization against a baseline reference frame."""
        if baseline_float is None:
            return frame_uint8, None
        denominator = np.maximum(baseline_float, 1e-3)
        ratio = np.clip(frame_float / denominator, 0.0, 1.0)
        processed = (ratio * 255.0).astype(np.uint8)
        baseline_uint8 = (baseline_float * 255.0).astype(np.uint8)
        return processed, baseline_uint8

    def _maybe_save_preprocess_plot(
        self,
        *,
        plot_state,
        output_dir: Optional[Path],
        frame_idx: int,
        raw_frame: np.ndarray,
        frame_float: np.ndarray,
        baseline_float: Optional[np.ndarray],
        processed: np.ndarray,
    ) -> None:
        """Save one preprocessing preview panel when plotting is enabled."""
        if plot_state is None or output_dir is None:
            return
        baseline_display = baseline_float if baseline_float is not None else frame_float
        save_path = output_dir / f"preprocess_{frame_idx:04d}.png"
        _update_preprocess_figure(
            plot_state,
            raw_frame,
            baseline_display,
            processed,
            frame_idx,
            save_path,
        )


class DiffuseDetector:
    """Diffuse-focused delamination workflows.

    The class exposes:

    - :meth:`diffuse_delamination` for crack-guided diffuse detection.
    - :meth:`diffuse_crack_tracking` for track-based diffuse detection used by
      :meth:`DelaminationDetector.detect_both_delaminations`.
    """

    def __init__(self, owner: DelaminationDetector) -> None:
        """Create a diffuse detector bound to its parent delamination detector."""
        self.owner = owner

    def diffuse_delamination(
        self,
        *,
        cracks: Optional[Sequence[np.ndarray]] = None,
        processed_cache_paths: Optional[List[Path]] = None,
        processed_stack: Optional[List[np.ndarray]] = None,
        save_overlays: bool = False,
        overlay_dirname: str = "delamination",
        max_frames: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        progress: bool = False,
    ) -> Dict[str, Any]:
        """Detect diffuse delamination masks using crack-guided ROIs.

        This workflow builds one threshold per frame from the union of ROI values,
        then applies that threshold inside each crack-guided ROI after edge-style
        filtering. Frame masks are latched over time (logical OR).

        Parameters
        ----------
        cracks:
            Per-frame crack segments. Required. The ROI builder uses these segments
            to define local diffuse search windows.
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
            )

        stacks = self.owner._select_stacks()
        raw_stack = getattr(self.owner.specimen, "image_stack_full", None) or stacks.get("full")
        if save_overlays and raw_stack is None:
            raise ValueError("Cannot save overlays without a full raw image stack.")

        cracks_list = list(cracks)
        if processed_cache_paths is None and processed_stack is None:
            stack = getattr(self.owner.specimen, "image_stack_full", None) or stacks.get("full")
            if stack is None:
                raise ValueError("Specimen has no full image stack to preprocess.")
            restore_preprocess_outputs = None
            if save_overlays and not self.owner.save_preprocess_outputs:
                restore_preprocess_outputs = self.owner.save_preprocess_outputs
                self.owner.save_preprocess_outputs = True
            try:
                auto_key = f"diffuse_auto_{self.owner.interface.name}"
                processed_cache_paths = self.owner.preprocess_stack_to_disk(
                    stack,
                    key=auto_key,
                    max_frames=max_frames,
                    cache_dirname="Preprocessor_cache",
                    reference_mode="static",
                    progress=progress,
                )["cache_paths"]
            finally:
                if restore_preprocess_outputs is not None:
                    self.owner.save_preprocess_outputs = restore_preprocess_outputs

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
        cracks: Sequence[np.ndarray],
        processed_cache_paths: Optional[List[Path]] = None,
        save_overlays: bool = False,
        overlay_dirname: str = "delamination",
        max_frames: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        progress: bool = False,
    ) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, Any]]]:
        """Diffuse detection path that prioritizes explicit upper/middle/lower stacks."""
        stacks = self.owner._select_stacks()
        upper_stack = stacks.get("upper")
        middle_stack = stacks.get("middle")
        lower_stack = stacks.get("lower")
        raw_stack = getattr(self.owner.specimen, "image_stack_full", None)

        if upper_stack is None or middle_stack is None or lower_stack is None:
            raise ValueError(
                "Region override mode requires upper/middle/lower stacks to be available."
            )

        cracks_list = list(cracks)
        if not cracks_list:
            raise ValueError("Diffuse delamination requires at least one crack frame.")

        diffuse_params = self._resolve_diffuse_params(params)
        if diffuse_params.get("reference_mode") is None:
            ref_from_cache = _reference_settings_from_cache_paths(processed_cache_paths)
            diffuse_params["reference_mode"] = ref_from_cache["reference_mode"]
            diffuse_params["reference_window"] = ref_from_cache["reference_window"]
            diffuse_params["reference_skip"] = ref_from_cache["reference_skip"]

        total_frames = min(len(middle_stack), len(upper_stack), len(lower_stack), len(cracks_list))
        if max_frames is not None:
            total_frames = min(total_frames, max(0, int(max_frames)))

        if total_frames <= 0:
            raise ValueError("No frames available for region-overridden diffuse detection.")

        middle_key = f"diffuse_middle_auto_{self.owner.interface.name}"
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
                raw_frame = None
                if raw_stack is not None and idx < len(raw_stack):
                    raw_candidate = _ensure_uint8(raw_stack[idx])
                    if raw_candidate.shape[:2] == mask_full.shape[:2]:
                        raw_frame = raw_candidate
                if raw_frame is None:
                    raw_frame = np.vstack(
                        [
                            _ensure_uint8(upper_stack[idx]),
                            _ensure_uint8(middle_stack[idx]),
                            _ensure_uint8(lower_stack[idx]),
                        ]
                    )
                overlay_cracks = self.owner._cracks_for_full_overlay(
                    frame_cracks,
                    upper_height=upper_h,
                    middle_height=middle_h,
                    full_height=int(raw_frame.shape[0]),
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
        window_diffuse = tuple(resolved["window_diffuse"])
        gaussian_filters = tuple(resolved["gaussian_filters"])
        if len(window_diffuse) != 2:
            raise ValueError("window_diffuse must be a tuple/list with 2 values.")
        if len(gaussian_filters) != 2:
            raise ValueError("gaussian_filters must be a tuple/list with 2 values.")
        resolved["window_diffuse"] = (int(window_diffuse[0]), int(window_diffuse[1]))
        resolved["gaussian_filters"] = (float(gaussian_filters[0]), float(gaussian_filters[1]))
        resolved["scale_min"] = float(resolved["scale_min"])
        resolved["scale_max"] = float(resolved["scale_max"])
        resolved["scale_min_percentile"] = (
            None if resolved["scale_min_percentile"] is None else float(resolved["scale_min_percentile"])
        )
        resolved["scale_max_percentile"] = (
            None if resolved["scale_max_percentile"] is None else float(resolved["scale_max_percentile"])
        )
        hard_floor = resolved.get("hard_floor")
        resolved["hard_floor"] = _resolve_hard_floor_ratio(hard_floor)
        resolved["post_threshold_closing_px"] = max(0, int(resolved.get("post_threshold_closing_px", 4)))
        scale_val = resolved.get("post_threshold_closing_scale")
        resolved["post_threshold_closing_scale"] = None if scale_val is None else float(scale_val)
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

    def _diffuse_mask_for_crack(
        self,
        image: np.ndarray,
        crack: np.ndarray,
        *,
        threshold: float,
        dx: float,
        dy: float,
        params: Dict[str, Any],
        avg_crack_width_px: float,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Build diffuse mask and bounds for one crack segment ROI."""
        geom = self._diffuse_roi_geometry(image, crack, dx=dx, dy=dy)
        if geom is None:
            bounds = (0, 0, 0, 0)
            return np.zeros((0, 0), dtype=bool), bounds

        preprocessed = self._diffuse_prethreshold_image(
            geom["patch"],
            params=params,
            avg_crack_width_px=avg_crack_width_px,
        )
        return self._diffuse_mask_from_preprocessed(
            geom=geom,
            closed=preprocessed["closed"],
            floor_mask=preprocessed["floor_mask"],
            threshold=threshold,
            params=params,
            avg_crack_width_px=avg_crack_width_px,
        )


class EdgeDetector:
    """Edge-focused delamination workflows.

    The class exposes:

    - :meth:`detect_primary` for single-interface edge tracking.
    - :meth:`detect_edge_multi` for hierarchical multi-interface promotion.
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
            stack = getattr(self.owner.specimen, "image_stack_full", None) or stacks.get("full")
            if stack is None:
                raise ValueError("Specimen has no full image stack to preprocess.")
            restore_preprocess_outputs = None
            if save_overlays and not self.owner.save_preprocess_outputs:
                restore_preprocess_outputs = self.owner.save_preprocess_outputs
                self.owner.save_preprocess_outputs = True
            try:
                auto_key = f"edge_primary_auto_{self.owner.interface.name}"
                processed_cache_paths = self.owner.preprocess_stack_to_disk(
                    stack,
                    key=auto_key,
                    max_frames=max_frames,
                    cache_dirname="Preprocessor_cache",
                    reference_mode="static",
                    progress=progress,
                )["cache_paths"]
            finally:
                if restore_preprocess_outputs is not None:
                    self.owner.save_preprocess_outputs = restore_preprocess_outputs

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
                debug_payloads[frame_key] = {
                    "processed": processed.copy(),
                    "upper": {
                        "smoothed": upper_result["smoothed"],
                        "constant_scaled": upper_result["constant_scaled"],
                        "closed": upper_result["closed"],
                        "threshold": upper_result["threshold"],
                        "hard_floor_eff": upper_result["hard_floor_eff"],
                        "close_radius": upper_result["close_radius"],
                        "min_object_px": upper_result["min_object_px"],
                        "binary": upper_result["binary"],
                        "binary_closed": upper_result["binary_closed"],
                        "mask": upper_result["mask"],
                        "primary_edge_snapshot": upper_result["primary_edge_snapshot"],
                        "status": upper_result["status"],
                    },
                    "lower": {
                        "smoothed": lower_result["smoothed"],
                        "constant_scaled": lower_result["constant_scaled"],
                        "closed": lower_result["closed"],
                        "threshold": lower_result["threshold"],
                        "hard_floor_eff": lower_result["hard_floor_eff"],
                        "close_radius": lower_result["close_radius"],
                        "min_object_px": lower_result["min_object_px"],
                        "binary": lower_result["binary"],
                        "binary_closed": lower_result["binary_closed"],
                        "mask": lower_result["mask"],
                        "primary_edge_snapshot": lower_result["primary_edge_snapshot"],
                        "status": lower_result["status"],
                    },
                }

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
        stacks = self.owner._select_stacks()
        upper_stack = stacks.get("upper")
        middle_stack = stacks.get("middle")
        lower_stack = stacks.get("lower")
        raw_stack = getattr(self.owner.specimen, "image_stack_full", None)

        if upper_stack is None or middle_stack is None or lower_stack is None:
            raise ValueError(
                "Region override mode requires upper/middle/lower stacks to be available."
            )

        total_frames = min(len(upper_stack), len(middle_stack), len(lower_stack))
        if max_frames is not None:
            total_frames = min(total_frames, max(0, int(max_frames)))
        if total_frames <= 0:
            raise ValueError("No frames available for region-overridden edge detection.")

        edge_params = self._resolve_primary_params(params)
        reference_defaults = _reference_settings_from_cache_paths(processed_cache_paths)
        reference_mode = str((params or {}).get("reference_mode", reference_defaults["reference_mode"]))
        reference_window = int((params or {}).get("reference_window", reference_defaults["reference_window"]))
        reference_skip = int((params or {}).get("reference_skip", reference_defaults["reference_skip"]))

        upper_cache_paths = self.owner.preprocess_stack_to_disk(
            upper_stack,
            key=f"edge_upper_auto_{self.owner.interface.name}",
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
            key=f"edge_lower_auto_{self.owner.interface.name}",
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
                raw_frame = None
                if raw_stack is not None and idx < len(raw_stack):
                    raw_candidate = _ensure_uint8(raw_stack[idx])
                    if raw_candidate.shape[:2] == primary_full.shape[:2]:
                        raw_frame = raw_candidate
                if raw_frame is None:
                    raw_frame = np.vstack(
                        [
                            _ensure_uint8(upper_stack[idx]),
                            middle_raw,
                            _ensure_uint8(lower_stack[idx]),
                        ]
                    )
                overlay_dir = self.owner.specimen.results_dir(overlay_dirname, "edge", "overlays")
                overlay_path = overlay_dir / f"edge_overlay_{idx:04d}.png"
                _save_edge_overlay(raw_frame, primary_full, overlay_path, view=overlay_view)

            if debug_root is not None:
                raw_frame = None
                if raw_stack is not None and idx < len(raw_stack):
                    raw_candidate = _ensure_uint8(raw_stack[idx])
                    if raw_candidate.shape[:2] == primary_full.shape[:2]:
                        raw_frame = raw_candidate
                if raw_frame is None:
                    raw_frame = np.vstack(
                        [
                            _ensure_uint8(upper_stack[idx]),
                            middle_raw,
                            _ensure_uint8(lower_stack[idx]),
                        ]
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
                debug_payloads[frame_key] = {
                    "processed": np.vstack(
                        [
                            upper_processed,
                            np.zeros((middle_h, width), dtype=np.uint8),
                            lower_processed,
                        ]
                    ),
                    "upper": {
                        "smoothed": upper_result["smoothed"],
                        "constant_scaled": upper_result["constant_scaled"],
                        "closed": upper_result["closed"],
                        "threshold": upper_result["threshold"],
                        "hard_floor_eff": upper_result["hard_floor_eff"],
                        "close_radius": upper_result["close_radius"],
                        "min_object_px": upper_result["min_object_px"],
                        "binary": upper_result["binary"],
                        "binary_closed": upper_result["binary_closed"],
                        "mask": upper_result["mask"],
                        "primary_edge_snapshot": upper_result["primary_edge_snapshot"],
                        "status": upper_result["status"],
                    },
                    "lower": {
                        "smoothed": lower_result["smoothed"],
                        "constant_scaled": lower_result["constant_scaled"],
                        "closed": lower_result["closed"],
                        "threshold": lower_result["threshold"],
                        "hard_floor_eff": lower_result["hard_floor_eff"],
                        "close_radius": lower_result["close_radius"],
                        "min_object_px": lower_result["min_object_px"],
                        "binary": lower_result["binary"],
                        "binary_closed": lower_result["binary_closed"],
                        "mask": lower_result["mask"],
                        "primary_edge_snapshot": lower_result["primary_edge_snapshot"],
                        "status": lower_result["status"],
                    },
                }

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

        The first level uses primary edge detection. Each deeper level is promoted from
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
            binary and mask fed into ``_compute_promoted_snapshots``, while
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
            computing binary/mask for the promotion logic.  Ignored if
            ``secondary_cache_paths`` is not set.
        secondary_params:
            Promotion parameters for deeper levels: secondary_similarity_threshold.
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
            auto_key = f"edge_multi_auto_{interfaces[0].name if interfaces else 'i0'}"
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
            parent_upper = upper_rolling_levels[level_idx - 1]
            parent_lower = lower_rolling_levels[level_idx - 1]

            upper_latched: List[np.ndarray] = []
            lower_latched: List[np.ndarray] = []
            upper_diag: List[Dict[str, Any]] = []
            lower_diag: List[Dict[str, Any]] = []
            acc_u = np.zeros_like(upper_mask_frames[0], dtype=bool)
            acc_l = np.zeros_like(lower_mask_frames[0], dtype=bool)

            # Use established primary (delayed by ~rolling-median window) so the growing
            # primary front is excluded: only ROLLING mask pixels inside settled primary
            # area accumulate as secondary, catching the interior darkening event.
            primary_upper = upper_levels[0]
            primary_lower = lower_levels[0]
            sec_ref_window = int(_sec_edge_params.get("reference_window", 7))
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
        """Merge and validate multi-interface promotion parameters."""
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

    def _compute_promoted_snapshots(
        self,
        *,
        parent_frames: Sequence[np.ndarray],
        binary_frames: Sequence[np.ndarray],
        mask_frames: Sequence[np.ndarray],
        similarity_threshold: float,
        seed_ratio: float,
        avg_crack_width_px: float,
        return_masks: bool = False,
        active_frames: Optional[Sequence[bool]] = None,
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """Generate per-frame promotion snapshots for the next interface level."""
        snapshots: List[np.ndarray] = []
        diagnostics: List[Dict[str, Any]] = []
        width = int(max(1, round(avg_crack_width_px)))
        lateral_drift_px = max(1, int(round(0.25 * float(avg_crack_width_px))))
        vert_struct = np.ones((width, 1), dtype=bool)

        for idx, parent_curr in enumerate(parent_frames):
            combined_bool = np.asarray(parent_curr, dtype=bool)

            if active_frames is not None and not active_frames[idx]:
                snapshots.append(np.zeros(combined_bool.shape, dtype=bool))
                diag: Dict[str, Any] = {
                    "status": "primary_not_established",
                    "similarity_bin": None,
                    "similarity": None,
                    "candidate_pixels": 0,
                    "snapshot_pixels": 0,
                    "directional_lateral_drift_px": lateral_drift_px,
                }
                if return_masks:
                    diag["_masks"] = {}
                diagnostics.append(diag)
                continue

            current_binary_bool = np.asarray(binary_frames[idx], dtype=bool)
            binary_mask_bool = np.asarray(mask_frames[idx], dtype=bool)

            front_band = np.zeros_like(combined_bool, dtype=bool)
            growth = np.zeros_like(combined_bool, dtype=bool)
            difference = np.zeros_like(combined_bool, dtype=bool)
            accumulated_v = np.zeros_like(combined_bool, dtype=bool)
            candidate_track = np.zeros_like(combined_bool, dtype=bool)
            candidate = np.zeros_like(combined_bool, dtype=bool)

            status = "no_prev_parent"
            similarity_bin: Optional[float] = None
            similarity: Optional[float] = None

            prev_parent = np.asarray(parent_frames[idx - 1], dtype=bool) if idx > 0 else None
            if prev_parent is not None:
                eroded = ndi.binary_erosion(combined_bool, structure=vert_struct, border_value=1)
                front_band = combined_bool & ~eroded
                growth = combined_bool & ~prev_parent

                difference = combined_bool & ~(binary_mask_bool & prev_parent & ~front_band)
                accumulated_v = (~difference) & current_binary_bool
                candidate_track = accumulated_v & combined_bool

                total_pixels = int(combined_bool.sum())
                overlap_pixels_0 = int((current_binary_bool & combined_bool).sum())
                similarity_bin = overlap_pixels_0 / max(1, total_pixels)

                if similarity_bin >= similarity_threshold:
                    status = "parent_overlap"
                else:
                    candidate = candidate_track.copy()
                    overlap_pixels = int(((candidate | growth) & combined_bool).sum())
                    similarity = overlap_pixels / max(1, total_pixels)
                    if similarity == 0.0:
                        candidate[:] = False
                        status = "no_overlap"
                    elif similarity > similarity_threshold:
                        candidate[:] = False
                        status = "too_similar"
                    else:
                        status = "candidate"

            height = candidate.shape[0]
            seed_depth = max(1, int(round(float(seed_ratio) * height)))
            edge_seed = np.zeros_like(candidate, dtype=np.uint8)
            edge_seed[:seed_depth, :] = candidate[:seed_depth, :].astype(np.uint8)
            snapshot = _rebuild_edge_connected_directional(
                candidate,
                seed_depth=seed_depth,
                lateral_drift_px=lateral_drift_px,
            )
            snapshot &= combined_bool

            snapshots.append(snapshot)
            diag: Dict[str, Any] = {
                "status": status,
                "similarity_bin": similarity_bin,
                "similarity": similarity,
                "candidate_pixels": int(np.count_nonzero(candidate)),
                "snapshot_pixels": int(np.count_nonzero(snapshot)),
                "directional_lateral_drift_px": int(lateral_drift_px),
            }
            if return_masks:
                diag["_masks"] = {
                    "front_band": front_band,
                    "growth": growth,
                    "difference": difference,
                    "accumulated_v": accumulated_v,
                    "candidate_track": candidate_track,
                    "candidate": candidate,
                    "snapshot": snapshot,
                }
            diagnostics.append(diag)

        return snapshots, diagnostics

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

        window_edge = tuple(resolved["window_edge"])
        gaussian_filters = tuple(resolved["gaussian_filters"])
        if len(window_edge) != 2:
            raise ValueError("window_edge must be a tuple/list with 2 values.")
        if len(gaussian_filters) != 2:
            raise ValueError("gaussian_filters must be a tuple/list with 2 values.")

        seed_ratio = float(resolved["seed_ratio"])
        if seed_ratio <= 0:
            raise ValueError("seed_ratio must be > 0.")

        connectivity_mode = str(resolved.get("connectivity_mode", "directional")).strip().lower()
        if connectivity_mode not in {"directional", "legacy_flood"}:
            raise ValueError("connectivity_mode must be 'directional' or 'legacy_flood'.")

        resolved["window_edge"] = (int(window_edge[0]), int(window_edge[1]))
        resolved["gaussian_filters"] = (float(gaussian_filters[0]), float(gaussian_filters[1]))
        resolved["scale_min"] = None if resolved["scale_min"] is None else float(resolved["scale_min"])
        resolved["scale_max"] = None if resolved["scale_max"] is None else float(resolved["scale_max"])
        resolved["scale_min_percentile"] = (
            None if resolved.get("scale_min_percentile") is None else float(resolved["scale_min_percentile"])
        )
        resolved["scale_max_percentile"] = (
            None if resolved.get("scale_max_percentile") is None else float(resolved["scale_max_percentile"])
        )
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
        scale_val = resolved.get("post_threshold_closing_scale")
        resolved["post_threshold_closing_scale"] = None if scale_val is None else max(0.0, float(scale_val))
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
        if connectivity_mode == "legacy_flood":
            primary_edge_snapshot = reconstruction(
                seed=primary_seed,
                mask=combined_upper.astype(np.uint8),
                method="dilation",
            ).astype(bool)
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
def _ensure_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert input frame to single-channel ``uint8``."""
    frame_float = frame.astype(np.float32)
    if frame_float.ndim == 3:
        frame_float = frame_float.mean(axis=2)
    if frame_float.max() <= 1.0:
        frame_float = frame_float * 255.0
    return np.clip(frame_float, 0, 255).astype(np.uint8)


def _frame_to_float(frame: np.ndarray) -> np.ndarray:
    """Convert input frame to single-channel ``float32`` in ``[0, 1]``."""
    frame_float = frame.astype(np.float32)
    if frame_float.ndim == 3:
        frame_float = frame_float.mean(axis=2)
    if frame_float.max() > 1.0:
        frame_float = frame_float / 255.0
    return frame_float


def save_crack_numbering(
    out_path: Any,
    raw_frame: np.ndarray,
    segments: List[np.ndarray],
    track_ids: List[Optional[int]],
    *,
    frame_abs: int,
    dpi: int = 150,
) -> None:
    """Draw crack segments on the raw frame labelled by track ID and save to disk.

    Segments belonging to a known track are colour-coded by ``track_id % 10``.
    Unmatched detections (``track_id=None``) are drawn in red with a ``?`` label.

    *segments* should be a list of arrays of shape ``(2, 2)``: ``[[y0, x0], [y1, x1]]``.
    """
    import matplotlib.pyplot as plt
    raw = raw_frame[..., 0] if raw_frame.ndim == 3 else raw_frame
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    ax.imshow(raw, cmap="gray")
    for seg, tid in zip(segments, track_ids):
        seg = np.asarray(seg, dtype=np.float32).reshape(2, 2)
        (y0, x0), (y1, x1) = seg
        cy, cx = float((y0 + y1) / 2), float((x0 + x1) / 2)
        color = "tab:red" if tid is None else f"C{int(tid) % 10}"
        label = "?" if tid is None else str(tid)
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=2.0)
        ax.text(
            cx, cy, label,
            color="white", fontsize=8,
            bbox=dict(facecolor=color, alpha=0.7, edgecolor="none", pad=1.0),
        )
    ax.set_title(f"crack numbering | frame {frame_abs}")
    ax.axis("off")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


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


def _resolve_hard_floor_ratio(value: Any) -> Optional[float]:
    """Normalize ``hard_floor`` to ratio space.

    The preferred input is a normalized ratio (for example ``0.90``). Legacy
    8-bit values greater than ``1.0`` are interpreted as pixel intensities and
    converted via ``value / 255``.
    """
    if value is None:
        return None
    hard_floor = float(value)
    if hard_floor > 1.0:
        warnings.warn(
            "hard_floor > 1.0 detected; interpreting as 8-bit intensity and "
            "converting to ratio (value / 255). Use ratio values such as 0.90.",
            DeprecationWarning,
            stacklevel=3,
        )
        hard_floor = hard_floor / 255.0
    return float(hard_floor)


def _rebuild_edge_connected_directional(
    mask: np.ndarray,
    *,
    seed_depth: int,
    lateral_drift_px: int,
) -> np.ndarray:
    """Rebuild edge-connected mask using vertical-biased propagation.

    Growth is causal from top to bottom: each row can only activate pixels that
    are present in ``mask`` and have support from the previous activated row
    within ``lateral_drift_px`` columns.
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


def _apply_edge_precedence(
    edge_mask: np.ndarray,
    diffuse_raw_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve overlap by assigning shared pixels to edge damage."""
    overlap = edge_mask & diffuse_raw_mask
    diffuse_final = diffuse_raw_mask & ~edge_mask
    combined = edge_mask | diffuse_final
    return diffuse_final, combined, overlap


def _dilate_edge_mask(edge_mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Dilate edge mask by ``radius_px`` pixels using a disk footprint."""
    if radius_px <= 0:
        return np.asarray(edge_mask, dtype=bool)
    structure = disk(int(radius_px))
    return ndi.binary_dilation(edge_mask, structure=structure)


def _frame_index_from_key(frame_key: str) -> Optional[int]:
    """Parse a ``frame_XXXX`` key into an integer index."""
    parts = frame_key.split("_")
    if len(parts) != 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _mask_px(mask: np.ndarray) -> int:
    """Count non-zero pixels in a boolean-like mask."""
    return int(np.count_nonzero(mask))


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


def _build_metrics_row(
    *,
    frame_idx: int,
    frame_pixels: int,
    edge_mask: np.ndarray,
    diffuse_raw: np.ndarray,
    diffuse_final: np.ndarray,
    overlap_mask: np.ndarray,
    combined_mask: np.ndarray,
) -> Dict[str, Any]:
    """Build one per-frame metrics record for combined delamination exports."""
    edge_px = _mask_px(edge_mask)
    diffuse_raw_px = _mask_px(diffuse_raw)
    overlap_px = _mask_px(overlap_mask)
    diffuse_px = _mask_px(diffuse_final)
    combined_px = _mask_px(combined_mask)
    total = float(frame_pixels) if frame_pixels > 0 else 1.0

    return {
        "frame": frame_idx,
        "frame_pixels": frame_pixels,
        "edge_px": edge_px,
        "diffuse_raw_px": diffuse_raw_px,
        "overlap_px": overlap_px,
        "diffuse_px": diffuse_px,
        "combined_px": combined_px,
        "edge_frac": edge_px / total,
        "diffuse_raw_frac": diffuse_raw_px / total,
        "overlap_frac": overlap_px / total,
        "diffuse_frac": diffuse_px / total,
        "combined_frac": combined_px / total,
    }


def _accumulate_latched_masks(snapshots: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Return cumulative OR masks frame by frame."""
    accumulated: List[np.ndarray] = []
    state: Optional[np.ndarray] = None
    for snapshot in snapshots:
        curr = np.asarray(snapshot, dtype=bool)
        if state is None:
            state = curr.copy()
        else:
            state = state | curr
        if state is None:
            continue
        accumulated.append(state.copy())
    return accumulated


def _filter_edge_connected(mask: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return mask pixels whose connected component touches the reference area."""
    mask = np.asarray(mask, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    if not np.any(mask) or not np.any(reference):
        return np.zeros_like(mask, dtype=bool)
    labeled, _ = ndi.label(mask | reference)
    ref_labels = set(np.unique(labeled[reference])) - {0}
    if not ref_labels:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(labeled, list(ref_labels)) & mask


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



def _interface_legend_label(specimen: Specimen, interface: Interface) -> str:
    """Build a plot label combining interface name and ply pair."""

    def _ply_name(idx: Optional[int]) -> Optional[str]:
        if idx is None:
            return None
        if idx < 0 or idx >= len(specimen.plies):
            return None
        return specimen.plies[idx].name

    upper = _ply_name(interface.upper_ply_index)
    lower = _ply_name(interface.lower_ply_index)
    if upper and lower:
        return f"{interface.name}: {upper}/{lower}"
    if upper:
        return f"{interface.name}: {upper}/?"
    if lower:
        return f"{interface.name}: ?/{lower}"
    return interface.name


def _normalize_rgba(color: Sequence[float], *, default_alpha: float = 0.35) -> Tuple[float, float, float, float]:
    """Normalize arbitrary color input to a clipped RGBA tuple."""
    values = [float(v) for v in color]
    if len(values) == 3:
        values.append(float(default_alpha))
    if len(values) != 4:
        return (1.0, 0.0, 0.0, float(default_alpha))
    return (
        float(np.clip(values[0], 0.0, 1.0)),
        float(np.clip(values[1], 0.0, 1.0)),
        float(np.clip(values[2], 0.0, 1.0)),
        float(np.clip(values[3], 0.0, 1.0)),
    )


def _rgba_close(
    left: Sequence[float],
    right: Sequence[float],
    *,
    tolerance: float = 1e-6,
) -> bool:
    """Return ``True`` when two RGBA colors are numerically close."""
    if len(left) != 4 or len(right) != 4:
        return False
    return all(abs(float(left[idx]) - float(right[idx])) <= tolerance for idx in range(4))


def _resolve_multi_interface_colors(interfaces: Sequence[Interface]) -> List[Tuple[float, float, float, float]]:
    """Use interface colors, but replace repeated default color with a level palette."""
    default_rgba = _normalize_rgba(DEFAULT_PRIMARY_DELAMINATION_COLOR, default_alpha=0.9)
    resolved: List[Tuple[float, float, float, float]] = []
    for idx, interface in enumerate(interfaces):
        interface_color = _normalize_rgba(interface.delamination_color_rgba)
        if _rgba_close(interface_color, default_rgba):
            resolved.append(MULTI_INTERFACE_DEFAULT_COLORS[idx % len(MULTI_INTERFACE_DEFAULT_COLORS)])
        else:
            resolved.append(interface_color)
    return resolved


def _save_multi_level_overlay(
    *,
    raw_frame: np.ndarray,
    level_masks: Sequence[np.ndarray],
    labels: Sequence[str],
    colors: Sequence[Sequence[float]],
    save_path: Path,
) -> None:
    """Save a classified multi-interface overlay with an external legend."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(raw_frame, cmap="gray")

    handles: List[Patch] = []
    for idx, mask in enumerate(level_masks):
        label = labels[idx] if idx < len(labels) else f"level_{idx + 1}"
        color = colors[idx] if idx < len(colors) else (1.0, 0.0, 0.0, 0.35)
        rgba = _normalize_rgba(color)
        mask_bool = np.asarray(mask, dtype=bool)
        if np.any(mask_bool):
            _overlay_mask(ax, mask_bool, rgba)
        handles.append(Patch(facecolor=rgba, edgecolor="none", label=label))

    if handles:
        legend = ax.legend(
            handles,
            [handle.get_label() for handle in handles],
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            title=r"Interface legend",
            frameon=True,
            fontsize=8,
            title_fontsize=9,
        )
        legend.get_frame().set_linewidth(0.6)
    ax.axis("off")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def _edge_limit_rows(mask_bool: np.ndarray, *, side: str) -> Optional[np.ndarray]:
    """Compute per-column edge lines used by line-only edge overlays."""
    if mask_bool.size == 0 or not np.any(mask_bool):
        return None
    h, _ = mask_bool.shape
    row_idx = np.arange(h, dtype=np.int32).reshape(-1, 1)
    if side == "bottom":
        rows = np.where(mask_bool, row_idx, -1).max(axis=0).astype(float)
        rows[rows < 0] = np.nan
    else:
        rows = np.where(mask_bool, row_idx, h).min(axis=0).astype(float)
        rows[rows >= h] = np.nan
    return rows


def _save_edge_overlay(
    raw_frame: np.ndarray,
    primary_mask: np.ndarray,
    save_path: Path,
    *,
    view: str = "mask",
    mask_color: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.35),
) -> None:
    """Save edge overlay in ``mask``, ``line``, or ``both`` modes."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.imshow(raw_frame, cmap="gray")

    if view in {"mask", "both"}:
        _overlay_mask(ax, primary_mask, mask_color)

    if view in {"line", "both"}:
        split_row = primary_mask.shape[0] // 2
        upper = primary_mask[:split_row, :]
        lower_unflipped = primary_mask[split_row:, :]
        rows_u = _edge_limit_rows(upper, side="bottom")
        if rows_u is not None:
            ax.plot(np.arange(upper.shape[1]), rows_u, color="red", ls='-', linewidth=0.6)
        rows_l = _edge_limit_rows(lower_unflipped, side="top")
        if rows_l is not None:
            ax.plot(np.arange(lower_unflipped.shape[1]), split_row + rows_l, color="red",  ls='-', linewidth=0.6)

    ax.axis("off")
    fig.savefig(save_path)
    plt.close(fig)


def _save_diffuse_overlay(
    raw_frame: np.ndarray,
    diffuse_mask: np.ndarray,
    save_path: Path,
    mask_color: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.35),
    *,
    cracks: Optional[Sequence[np.ndarray]] = None,
    crack_color: Tuple[float, float, float, float] = CRACK_OVERLAY_RGBA,
) -> None:
    """Save diffuse mask overlay over a grayscale raw frame."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.imshow(raw_frame, cmap="gray")

    _overlay_mask(ax, diffuse_mask, mask_color)
    _overlay_cracks(ax, cracks, color=crack_color)

    ax.axis("off")
    fig.savefig(save_path)
    plt.close(fig)


def _save_combined_overlay(
    raw_frame: np.ndarray,
    *,
    edge_mask: np.ndarray,
    diffuse_mask: np.ndarray,
    save_path: Path,
    view: str = "union",
    edge_color: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.35),
    diffuse_color: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.35),
    union_color: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.35),
    cracks: Optional[Sequence[np.ndarray]] = None,
    crack_color: Tuple[float, float, float, float] = CRACK_OVERLAY_RGBA,
) -> None:
    """Save combined edge/diffuse overlay in union or classified view."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.imshow(raw_frame, cmap="gray")

    if view == "classified":
        _overlay_mask(ax, diffuse_mask, diffuse_color)
        _overlay_mask(ax, edge_mask, edge_color)
    else:
        combined = edge_mask | diffuse_mask
        _overlay_mask(ax, combined, union_color)

    _overlay_cracks(ax, cracks, color=crack_color)

    ax.axis("off")
    fig.savefig(save_path)
    plt.close(fig)


def _overlay_mask(
    ax,
    mask: np.ndarray,
    color: Tuple[float, float, float, float],
) -> None:
    """Render one RGBA mask layer onto an existing axes."""
    overlay = np.zeros((*mask.shape, 4), dtype=float)
    overlay[mask] = color
    ax.imshow(overlay)


def _overlay_cracks(
    ax,
    cracks: Optional[Sequence[np.ndarray]],
    *,
    color: Tuple[float, float, float, float] = CRACK_OVERLAY_RGBA,
    linewidth: float = 0.8,
) -> None:
    """Render crack segments on top of an existing axes."""
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


def _save_single_overlay(
    raw_frame: np.ndarray,
    mask: np.ndarray,
    save_path: Path,
    color: Tuple[float, float, float, float],
) -> None:
    """Save one-color overlay for a precomputed mask."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.imshow(raw_frame, cmap="gray")
    _overlay_mask(ax, mask, color)
    ax.axis("off")
    fig.savefig(save_path)
    plt.close(fig)


def _load_mask_frame(path: Path, frame_key: str) -> Optional[np.ndarray]:
    """Load one frame mask from an NPZ bundle, returning ``None`` if absent."""
    if not path.exists():
        return None
    payload = np.load(path, allow_pickle=False)
    if frame_key not in payload:
        return None
    return np.asarray(payload[frame_key], dtype=bool)


def _save_edge_multi_debug_panels(
    *,
    debug_dir: Path,
    frame_indices: List[int],
    processed_frames: List[np.ndarray],
    baselines: List[np.ndarray],
    upper_results: List[Dict[str, Any]],
    lower_results: List[Dict[str, Any]],
    upper_latched: List[np.ndarray],
    lower_latched: List[np.ndarray],
    upper_diag: List[Dict[str, Any]],
    lower_diag: List[Dict[str, Any]],
    split_rows: List[int],
    level_idx: int,
    sec_processed_frames: Optional[List[np.ndarray]] = None,
    sec_upper_results: Optional[List[Dict[str, Any]]] = None,
    sec_lower_results: Optional[List[Dict[str, Any]]] = None,
    upper_rolling_frames: Optional[List[np.ndarray]] = None,
    lower_rolling_frames: Optional[List[np.ndarray]] = None,
) -> None:
    """Save per-frame debug panels for detect_edge_multi — secondary detection focus.

    Layout (6 rows × 5 cols, 3 rows per half):

    UPPER:
      Row 0  ROLLING input  — rolling processed · binary · binary_closed · mask · rolling_parent_latched
      Row 1  PROMOTION      — front_band · growth · candidate_track · candidate · secondary_latched (result)
      Row 2  REFERENCE      — primary_latched (static, for spatial reference only)  +  difference
    [separator]
    LOWER: same 3 rows
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_rolling = (
        sec_processed_frames is not None
        and sec_upper_results is not None
        and sec_lower_results is not None
        and upper_rolling_frames is not None
        and lower_rolling_frames is not None
    )

    panels_dir = debug_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    _BG_ROLLING   = "#ddeeff"   # blue  → rolling_median detection steps
    _BG_PROMOTION = "#fff3dd"   # amber → promotion / candidate logic
    _BG_REF       = "#eeeeee"   # grey  → reference images (primary, for context only)

    def _show(ax, img, title, cmap="gray", vmin=None, vmax=None, bg=None):
        if bg is not None:
            ax.set_facecolor(bg)
        if img is None or (hasattr(img, "size") and img.size == 0):
            ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="#aaaaaa")
        else:
            arr = np.asarray(img)
            if arr.dtype == bool:
                arr = arr.astype(np.float32)
            ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=8, pad=3, fontweight="bold")
        ax.axis("off")

    def _sep(axes_row, label):
        for ax in axes_row:
            ax.axis("off")
        axes_row[2].text(0.5, 0.5, label, ha="center", va="center",
                         transform=axes_row[2].transAxes, fontsize=10,
                         color="#444444", fontweight="bold")

    COLS = 5
    ROWS = 7  # 3 rows upper + sep + 3 rows lower

    for i, frame_idx in enumerate(frame_indices):
        if i >= len(upper_results):
            break

        proc  = processed_frames[i] if i < len(processed_frames) else None
        split = split_rows[i] if i < len(split_rows) else (proc.shape[0] // 2 if proc is not None else 0)

        ur = upper_results[i]
        lr = lower_results[i]
        ul = np.asarray(upper_latched[i], dtype=bool) if i < len(upper_latched) else None
        ll = np.asarray(lower_latched[i], dtype=bool) if i < len(lower_latched) else None

        ud = upper_diag[i] if i < len(upper_diag) else {}
        ld = lower_diag[i] if i < len(lower_diag) else {}
        u_masks = ud.get("_masks", {})
        l_masks = ld.get("_masks", {})

        sec_proc = sec_processed_frames[i] if has_rolling and i < len(sec_processed_frames) else None
        su = sec_upper_results[i] if has_rolling and i < len(sec_upper_results) else {}
        sl = sec_lower_results[i] if has_rolling and i < len(sec_lower_results) else {}
        urf = np.asarray(upper_rolling_frames[i], dtype=bool) if has_rolling and i < len(upper_rolling_frames) else None
        lrf = np.asarray(lower_rolling_frames[i], dtype=bool) if has_rolling and i < len(lower_rolling_frames) else None

        urp = sec_proc[:split, :]            if sec_proc is not None else None
        lrp = np.flipud(sec_proc[split:, :]) if sec_proc is not None else None

        # Primary latched (static) for spatial reference.
        u_primary_ref = ur.get("primary_latched")
        l_primary_ref = lr.get("primary_latched")

        conn_u    = ud.get("connected_pixels", 0)
        conn_l    = ld.get("connected_pixels", 0)

        fig, axes = plt.subplots(ROWS, COLS, figsize=(22, 18))
        fig.patch.set_facecolor("#f9f9f9")
        fig.suptitle(
            f"Frame {i}  (abs {frame_idx})  ·  level {level_idx + 1}\n"
            f"UPPER  connected_px={conn_u}\n"
            f"LOWER  connected_px={conn_l}",
            fontsize=10, y=0.998,
        )

        # ── UPPER ────────────────────────────────────────────────────────────
        # Row 0: rolling_median detection chain
        _show(axes[0, 0], urp,                    "ROLLING processed\n(frame ÷ local median)", vmin=0, vmax=255, bg=_BG_ROLLING)
        _show(axes[0, 1], su.get("binary"),        "ROLLING binary",                           bg=_BG_ROLLING)
        _show(axes[0, 2], su.get("binary_closed"), "ROLLING binary closed",                    bg=_BG_ROLLING)
        _show(axes[0, 3], su.get("mask"),          "ROLLING mask",                             bg=_BG_ROLLING)
        _show(axes[0, 4], urf,                     "ROLLING parent latched\n(sim-gate reference)", bg=_BG_ROLLING)

        # Row 1: connectivity filter → secondary result
        _show(axes[1, 0], None,                               "n/a",                                        bg=_BG_PROMOTION)
        _show(axes[1, 1], None,                               "n/a",                                        bg=_BG_PROMOTION)
        _show(axes[1, 2], u_masks.get("connected_mask"),      "ROLLING in settled primary\n(delayed ref)",  bg=_BG_PROMOTION)
        _show(axes[1, 3], None,                               "n/a",                                        bg=_BG_PROMOTION)
        _show(axes[1, 4], ul,                                 "RESULT secondary latched",                   bg=_BG_PROMOTION)

        # Row 2: reference (static primary) + difference — grey, labelled as context only
        _show(axes[2, 0], u_primary_ref,              "REF primary latched\n(static — context only)", bg=_BG_REF)
        _show(axes[2, 1], u_masks.get("difference"),  "REF difference mask",                          bg=_BG_REF)
        for col in range(2, COLS):
            axes[2, col].axis("off")
            axes[2, col].set_facecolor(_BG_REF)

        # ── SEPARATOR ────────────────────────────────────────────────────────
        _sep(axes[3], "─────  LOWER HALF  ─────")

        # ── LOWER ────────────────────────────────────────────────────────────
        _show(axes[4, 0], lrp,                    "ROLLING processed\n(frame ÷ local median)", vmin=0, vmax=255, bg=_BG_ROLLING)
        _show(axes[4, 1], sl.get("binary"),        "ROLLING binary",                           bg=_BG_ROLLING)
        _show(axes[4, 2], sl.get("binary_closed"), "ROLLING binary closed",                    bg=_BG_ROLLING)
        _show(axes[4, 3], sl.get("mask"),          "ROLLING mask",                             bg=_BG_ROLLING)
        _show(axes[4, 4], lrf,                     "ROLLING parent latched\n(sim-gate reference)", bg=_BG_ROLLING)

        _show(axes[5, 0], None,                               "n/a",                                        bg=_BG_PROMOTION)
        _show(axes[5, 1], None,                               "n/a",                                        bg=_BG_PROMOTION)
        _show(axes[5, 2], l_masks.get("connected_mask"),      "ROLLING in settled primary\n(delayed ref)",  bg=_BG_PROMOTION)
        _show(axes[5, 3], None,                               "n/a",                                        bg=_BG_PROMOTION)
        _show(axes[5, 4], ll,                                 "RESULT secondary latched",                   bg=_BG_PROMOTION)

        _show(axes[6, 0], l_primary_ref,              "REF primary latched\n(static — context only)", bg=_BG_REF)
        _show(axes[6, 1], l_masks.get("difference"),  "REF difference mask",                          bg=_BG_REF)
        for col in range(2, COLS):
            axes[6, col].axis("off")
            axes[6, col].set_facecolor(_BG_REF)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = panels_dir / f"frame_{i:04d}_abs{frame_idx:04d}.png"
        fig.savefig(out, dpi=100, bbox_inches="tight")
        plt.close(fig)


def _save_edge_debug_frame(
    *,
    frame_dir: Path,
    raw_frame: np.ndarray,
    processed: np.ndarray,
    upper_slice: np.ndarray,
    lower_slice: np.ndarray,
    upper_result: Dict[str, Any],
    lower_result: Dict[str, Any],
    lower_latched_unflipped: Optional[np.ndarray],
    full_latched: np.ndarray,
) -> None:
    """Persist intermediate edge-processing arrays for one debug frame."""
    import matplotlib.pyplot as plt

    def _save_gray(name: str, image: np.ndarray, vmin: float, vmax: float) -> None:
        plt.imsave(frame_dir / name, image, cmap="gray", vmin=vmin, vmax=vmax)

    _save_gray("raw.png", raw_frame, 0, 255)
    _save_gray("processed.png", processed, 0, 255)

    _save_gray("upper_edge_slice.png", upper_slice, 0, 255)
    _save_gray("upper_filtered_max.png", upper_result["filtered_max"], 0, 255)
    _save_gray("upper_filtered_min.png", upper_result["filtered_min"], 0, 255)
    _save_gray("upper_sharpened.png", upper_result["sharpened"], 0, 255)
    _save_gray("upper_smoothed.png", upper_result["smoothed"], 0, 255)
    _save_gray("upper_constant_scaled.png", upper_result["constant_scaled"], 0, 1)
    _save_gray("upper_closed.png", upper_result["closed"], 0, 1)
    _save_gray("upper_binary.png", upper_result["binary"].astype(float), 0, 1)
    _save_gray("upper_binary_closed.png", upper_result["binary_closed"].astype(float), 0, 1)
    _save_gray("upper_mask.png", upper_result["mask"].astype(float), 0, 1)
    _save_gray("upper_combined.png", upper_result["combined_upper"].astype(float), 0, 1)
    _save_gray("upper_primary_seed.png", upper_result["primary_seed"].astype(float), 0, 1)
    _save_gray("upper_primary_edge_snapshot.png", upper_result["primary_edge_snapshot"].astype(float), 0, 1)
    _save_gray("upper_primary_latched_accum.png", upper_result["primary_latched"].astype(float), 0, 1)

    _save_gray("lower_edge_slice_processed.png", lower_slice, 0, 255)
    _save_gray("lower_filtered_max.png", lower_result["filtered_max"], 0, 255)
    _save_gray("lower_filtered_min.png", lower_result["filtered_min"], 0, 255)
    _save_gray("lower_sharpened.png", lower_result["sharpened"], 0, 255)
    _save_gray("lower_smoothed.png", lower_result["smoothed"], 0, 255)
    _save_gray("lower_constant_scaled.png", lower_result["constant_scaled"], 0, 1)
    _save_gray("lower_closed.png", lower_result["closed"], 0, 1)
    _save_gray("lower_binary.png", lower_result["binary"].astype(float), 0, 1)
    _save_gray("lower_binary_closed.png", lower_result["binary_closed"].astype(float), 0, 1)
    _save_gray("lower_mask.png", lower_result["mask"].astype(float), 0, 1)
    _save_gray("lower_combined.png", lower_result["combined_upper"].astype(float), 0, 1)
    _save_gray("lower_primary_seed.png", lower_result["primary_seed"].astype(float), 0, 1)
    _save_gray("lower_primary_edge_snapshot.png", lower_result["primary_edge_snapshot"].astype(float), 0, 1)
    _save_gray("lower_primary_latched_accum.png", lower_result["primary_latched"].astype(float), 0, 1)

    if lower_latched_unflipped is not None:
        _save_gray("lower_primary_latched_accum_unflipped.png", lower_latched_unflipped.astype(float), 0, 1)
    _save_gray("full_primary_latched_accum.png", full_latched.astype(float), 0, 1)


def _prepare_preprocess_figure(image_shape: Optional[Tuple[int, int]]):
    """Create reusable matplotlib artists for preprocess triplet previews."""
    import matplotlib.pyplot as plt

    _DPI = 100
    if image_shape is not None and len(image_shape) >= 2:
        height, width = image_shape[:2]
        placeholder_shape = (int(height), int(width))
        figsize = (3 * int(width) / _DPI, int(height) / _DPI)
    else:
        placeholder_shape = (2, 2)
        figsize = plt.rcParams["figure.figsize"]

    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=_DPI, constrained_layout=True)
    axes = list(axes)

    im_raw = axes[0].imshow(np.zeros(placeholder_shape), cmap="gray", vmin=0, vmax=255, aspect="equal")
    axes[0].set_title("Raw")
    im_base = axes[1].imshow(np.zeros(placeholder_shape), cmap="gray", vmin=0.0, vmax=1.0, aspect="equal")
    axes[1].set_title("Rolling median baseline")
    im_proc = axes[2].imshow(np.zeros(placeholder_shape), cmap="gray", vmin=0, vmax=255, aspect="equal")
    axes[2].set_title("Processed")
    for ax in axes:
        ax.axis("off")

    return (fig, axes, {"raw": im_raw, "baseline": im_base, "processed": im_proc})


def _update_preprocess_figure(
    plot_state,
    raw: np.ndarray,
    baseline: np.ndarray,
    processed: np.ndarray,
    frame_idx: int,
    save_path,
) -> None:
    """Update and save one preprocess preview figure."""
    fig, axes, artists = plot_state
    artists["raw"].set_data(raw)
    artists["baseline"].set_data(baseline)
    artists["processed"].set_data(processed)

    for key, frame in ("raw", raw), ("baseline", baseline), ("processed", processed):
        height, width = frame.shape[:2]
        ax = artists[key].axes
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)

    axes[0].set_xlabel(f"idx={frame_idx}")
    axes[1].set_xlabel("baseline")
    axes[2].set_xlabel("processed")
    fig.suptitle(f"Preprocessing - frame {frame_idx}", fontsize=12)
    fig.savefig(save_path, dpi=fig.get_dpi())


def _close_preprocess_figure(plot_state) -> None:
    """Close preprocess preview figure and release matplotlib resources."""
    import matplotlib.pyplot as plt

    fig, _, _ = plot_state
    plt.close(fig)


__all__ = [
    "DelaminationDetector",
    "EdgeDetector",
    "DiffuseDetector",
]
