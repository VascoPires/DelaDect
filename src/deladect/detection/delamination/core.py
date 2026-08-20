"""Combined delamination orchestration: :class:`DelaminationDetector`.

Composes :class:`~deladect.detection.delamination.edge.EdgeDetector` and
:class:`~deladect.detection.delamination.diffuse.DiffuseDetector` (owner
pattern -- ``self.edge``/``self.diffuse``) and mixes in
:class:`~deladect.detection.delamination.PreprocessingMixin`.
"""


from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import disk

from deladect.io.delamination import (
    save_interface_metrics,
    save_mask_bundle,
    store_interface_delamination_results,
)
from deladect.specimen import (
    Interface,
    Specimen,
)

logger = logging.getLogger(__name__)

from ._common import (
    DIFFUSE_OVERLAY_RGBA,
    EDGE_OVERLAY_RGBA,
    CrackInput,
    _coerce_cracks_by_frame,
    _ensure_uint8,
    _mask_px,
    _progress_done,
    _progress_init,
    _progress_update,
    _result_key_token,
)
from ._overlays import (
    _overlay_mask,
    _save_combined_overlay,
    _save_diffuse_overlay,
    _save_edge_overlay,
)
from ._preprocess import PreprocessingMixin
from .diffuse import DiffuseDetector
from .edge import EdgeDetector


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



class DelaminationDetector(PreprocessingMixin):
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
        cracks: Optional[CrackInput] = None,
        processed_cache_paths: Optional[List[Path]] = None,
        processed_stack: Optional[List[np.ndarray]] = None,
        save_overlays: bool = True,
        overlay_dirname: str = "delamination",
        overlay_view: str = "classified",
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
        track_cracks: bool = False,
        max_center_px: Optional[float] = None,
        max_angle_deg: float = 15.0,
        max_cost: float = 1.8,
        return_masks: bool = False,
        return_intermediates: bool = False,
        debug: bool = False,
        save_edge_debug: bool = False,
        progress: bool = False,
        crack_coordinate_space: str = "middle",
    ) -> Dict[str, Any]:
        """Detect edge and diffuse delamination, then resolve overlap.

        Overlaps are resolved in favour of edge delamination. An optional
        edge-exclusion halo can be applied before arbitration so excluded pixels are
        counted as edge damage. By default, diffuse ROIs are evaluated independently
        per frame; crack tracking can be enabled with ``track_cracks=True``.

        Parameters
        ----------
        cracks:
            Per-frame cracks or the orientation-keyed result returned by
            :func:`deladect.detection.crack_analysis`. Every orientation present
            in a structured analysis result is merged for diffuse ROI construction.
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
        track_cracks:
            If ``True``, associate cracks between frames before diffuse detection.
            If ``False``, evaluate the crack-guided diffuse ROIs independently in
            each frame and latch the resulting masks over time. Defaults to ``False``.
        max_center_px, max_angle_deg, max_cost:
            Track-assignment thresholds forwarded to :meth:`diffuse_crack_tracking`.
        return_masks:
            If ``True``, include masks in the returned dictionary.
        debug:
            If ``True``, include edge/diffuse debug payloads.
        crack_coordinate_space:
            Coordinate space of ``cracks``, used for full-frame overlays in
            region-override mode: ``"middle"`` (default) if crack detection
            ran on the middle-region stack, or ``"full"`` if it was forced
            onto the full-frame stack (e.g. via ``use_full_stack=True`` in
            :func:`deladect.detection.crack_analysis`). This is never
            inferred from the coordinate values themselves.

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
        if crack_coordinate_space not in {"middle", "full"}:
            raise ValueError("crack_coordinate_space must be one of: 'middle', 'full'.")
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
            auto_key = f"both_auto_{_result_key_token(self.interface.name)}"
            processed_cache_paths = self.preprocess_stack_to_disk(
                stack,
                key=auto_key,
                max_frames=max_frames,
                cache_dirname="Preprocessor_cache",
                reference_mode="static",
                progress=progress,
            )["cache_paths"]

        edge_result = self.edge.detect_primary(
            processed_cache_paths=processed_cache_paths,
            processed_stack=processed_stack,
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
        proc_frames_list: List[np.ndarray] = []
        selected_indices_list: List[int] = []
        crack_frames_normalized: List[List[Any]] = []
        cracks_by_frame: List[Any] = []
        crack_tracking_result: Optional[Dict[str, Any]] = None

        if track_cracks:
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

            crack_tracking_result = self.diffuse.diffuse_crack_tracking(
                proc_frames_list,
                crack_frames_normalized,
                selected_indices_list,
                avg_crack_width_px=self.specimen.avg_crack_width_px,
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
        else:
            diffuse_result = self.diffuse.diffuse_delamination(
                cracks=cracks,
                processed_cache_paths=processed_cache_paths,
                processed_stack=processed_stack,
                save_overlays=False,
                overlay_dirname=overlay_dirname,
                max_frames=max_frames,
                params=diffuse_params,
                debug=debug,
                progress=progress,
            )
            diffuse_masks = diffuse_result["masks"]
            cracks_by_frame = _coerce_cracks_by_frame(cracks, len(diffuse_masks))

        # In tracked region-override runs, diffuse detection uses the full preprocessed
        # stack and may produce detections in rows reserved for edge detection. Zero
        # those rows before latching so they never accumulate.
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
                    frame_cracks = self._cracks_for_full_overlay(
                        frame_cracks,
                        shift=(crack_coordinate_space == "middle"),
                        upper_height=upper_h,
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
                            shift=(crack_coordinate_space == "middle"),
                            upper_height=int(
                                np.asarray(_ensure_uint8(getattr(self.specimen, "image_stack_upper")[frame_idx])).shape[0]
                            ),
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
                "track_cracks": bool(track_cracks),
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
        if return_intermediates and track_cracks:
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
        shift: bool,
        upper_height: int,
    ) -> Optional[List[np.ndarray]]:
        """Reshape crack segments for a full-frame overlay.

        ``shift`` must be supplied by the caller based on which stack crack
        detection actually ran on -- it is never inferred from the coordinate
        values themselves. A full-frame crack near the top of the image is
        numerically indistinguishable from a middle-region crack near
        ``y=0``, so guessing from value ranges previously produced a real
        overlay bug where cracks appeared shifted into the wrong region.
        When ``shift`` is ``True``, coordinates are translated by
        ``upper_height`` so middle-region ``y=0`` always maps to full-frame
        ``y=upper_height``, regardless of crack length or distribution.
        """
        if cracks is None:
            return None

        prepared: List[np.ndarray] = []
        for segment in cracks:
            try:
                arr = np.asarray(segment, dtype=float).reshape(-1, 2)
            except Exception:
                continue
            if arr.shape[0] < 2:
                continue
            prepared.append(arr)

        if shift and upper_height > 0:
            offset = np.array([float(upper_height), 0.0])
            return [arr + offset for arr in prepared]
        return prepared

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
