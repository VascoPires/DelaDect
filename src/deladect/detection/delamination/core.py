"""Combined edge and diffuse delamination detection: :class:`DelaminationDetector`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.morphology import disk

from deladect.io.delamination import (
    save_interface_metrics,
    save_mask_bundle,
    store_interface_delamination_results,
)
from deladect.specimen import Interface, Specimen

from ._common import (
    CrackInput,
    _Progress,
    _auto_preprocess_cache_paths,
    _coerce_cracks_by_frame,
    _ensure_uint8,
    _kmeans_split,
    _minmax_otsu_threshold,
)
from ._overlays import (
    DIFFUSE_OVERLAY_RGBA,
    EDGE_OVERLAY_RGBA,
    _save_combined_overlay,
    _save_diffuse_overlay,
    _save_edge_overlay,
    _save_single_overlay,
)
from ._preprocess import PreprocessingMixin
from .diffuse import DiffuseDetector
from .edge import EdgeDetector

# overlay_type -> (results subfolder, file prefix)
_OVERLAY_OUTPUTS = {
    "edge": ("edge", "edge_overlay"),
    "diffuse": ("diffuse", "diffuse_overlay"),
    "both": ("both", "combined_overlay"),
    "total_dela": ("total", "total_overlay"),
}


def _dilate_edge_mask(edge_mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Grow the edge mask by ``radius_px`` pixels."""
    if radius_px <= 0:
        return np.asarray(edge_mask, dtype=bool)
    return ndi.binary_dilation(edge_mask, structure=disk(int(radius_px)))


def _latch_masks(masks: Dict[str, np.ndarray], frame_keys: Sequence[str]) -> None:
    """Replace each mask by the union of it and all earlier masks, so damage never disappears."""
    accumulated: Optional[np.ndarray] = None
    for frame_key in frame_keys:
        frame_mask = np.asarray(masks[frame_key], dtype=bool)
        accumulated = frame_mask if accumulated is None else accumulated | frame_mask
        masks[frame_key] = accumulated.copy()


def _build_metrics_row(
    *,
    frame_idx: int,
    edge_mask: np.ndarray,
    diffuse_raw: np.ndarray,
    diffuse_final: np.ndarray,
    overlap_mask: np.ndarray,
    combined_mask: np.ndarray,
) -> Dict[str, Any]:
    frame_pixels = int(edge_mask.size)
    counts = {
        "edge_px": int(np.count_nonzero(edge_mask)),
        "diffuse_raw_px": int(np.count_nonzero(diffuse_raw)),
        "overlap_px": int(np.count_nonzero(overlap_mask)),
        "diffuse_px": int(np.count_nonzero(diffuse_final)),
        "combined_px": int(np.count_nonzero(combined_mask)),
    }
    total = float(frame_pixels) if frame_pixels > 0 else 1.0
    fractions = {name.replace("_px", "_frac"): count / total for name, count in counts.items()}
    return {"frame": frame_idx, "frame_pixels": frame_pixels, **counts, **fractions}


def _load_mask_frame(path: Path, frame_key: str) -> Optional[np.ndarray]:
    """Load one frame of a saved mask file, or ``None`` if the file or frame is missing."""
    if not path.exists():
        return None
    payload = np.load(path, allow_pickle=False)
    if frame_key not in payload:
        return None
    return np.asarray(payload[frame_key], dtype=bool)


class DelaminationDetector(PreprocessingMixin):
    """Detect delamination at one interface of a specimen.

    Edge and diffuse detection are available as ``detector.edge`` and
    ``detector.diffuse``; :meth:`detect_both_delaminations` runs both and
    combines them.

    Parameters
    ----------
    specimen:
        Specimen with the image stacks.
    interface:
        Interface being analyzed. Its name is used for cache folders and
        its color in overlays; saved results are recorded on it.
    history_clamp:
        Clamp each frame to its minimum history before normalizing.
    save_preprocess_outputs:
        Save raw / baseline / processed previews while preprocessing.
    preprocess_outputs_dirname:
        Folder for those previews under the specimen results.
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
        """Create a detector for ``interface`` of ``specimen``."""
        self.specimen = specimen
        self.interface = interface
        self.history_clamp = bool(history_clamp)
        self.save_preprocess_outputs = bool(save_preprocess_outputs)
        self.preprocess_outputs_dirname = str(preprocess_outputs_dirname)
        self._region_mode = all(
            path is not None
            for path in (specimen.path_upper_border, specimen.path_lower_border, specimen.path_middle)
        )
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
        """Save an overlay of one frame from masks saved by :meth:`detect_both_delaminations`.

        Masks are read from ``<results>/<overlay_dirname>/both/<masks_dirname>``.

        Parameters
        ----------
        frame_idx:
            Frame to draw.
        overlay_type:
            ``"edge"``, ``"diffuse"``, ``"both"`` (edge and diffuse in two
            colors) or ``"total_dela"`` (their union in the interface color).
        overlay_dirname, masks_dirname:
            Folders used when the masks were saved.
        edge_exclusion_px:
            Edge dilation to use if the dilated edge masks weren't saved.
        save_path:
            Output file. Defaults to the overlays folder of the chosen type.

        Returns
        -------
        dict[str, Any]
            ``{"path": pathlib.Path}`` of the saved image.
        """
        overlay_type = str(overlay_type).lower()
        if overlay_type not in _OVERLAY_OUTPUTS:
            raise ValueError("overlay_type must be one of: 'diffuse', 'edge', 'both', 'total_dela'.")

        raw_stack = getattr(self.specimen, "image_stack_full", None)
        if raw_stack is None:
            raise ValueError("Cannot save overlays without a full raw image stack.")

        raw_frame = _ensure_uint8(raw_stack[frame_idx])
        frame_key = f"frame_{frame_idx:04d}"
        masks_root = self.specimen.results_dir(overlay_dirname, "both", masks_dirname)

        edge_raw = _load_mask_frame(masks_root / "edge_raw.npz", frame_key)
        if edge_raw is None:
            raise ValueError("Edge masks are missing. Run detect_both_delaminations with save_masks=True.")

        edge_exclusion = _load_mask_frame(masks_root / "edge_exclusion.npz", frame_key)
        if edge_exclusion is None:
            edge_exclusion = _dilate_edge_mask(edge_raw, max(0, int(edge_exclusion_px)))

        diffuse_final = _load_mask_frame(masks_root / "diffuse_final.npz", frame_key)
        if diffuse_final is None:
            diffuse_final = _load_mask_frame(masks_root / "diffuse_raw.npz", frame_key)
        if overlay_type in {"diffuse", "both"} and diffuse_final is None:
            raise ValueError("Diffuse masks are missing. Run detect_both_delaminations with save_masks=True.")

        combined = _load_mask_frame(masks_root / "combined.npz", frame_key)
        if combined is None and diffuse_final is not None:
            combined = edge_exclusion | diffuse_final

        if save_path is None:
            subfolder, prefix = _OVERLAY_OUTPUTS[overlay_type]
            save_path = self.specimen.results_dir(overlay_dirname, subfolder, "overlays") / f"{prefix}_{frame_idx:04d}.png"

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
        """Detect edge and diffuse delamination and combine them.

        Edge masks are dilated by ``edge_exclusion_px``; where the two
        overlap, the pixel counts as edge delamination. Both masks are
        latched over time.

        Parameters
        ----------
        cracks:
            Cracks per frame, or the result of
            :func:`~deladect.detection.crack_analysis` (all orientations are
            used). Diffuse detection looks around these cracks.
        processed_cache_paths, processed_stack:
            Preprocessed frames, as cache files or arrays. If neither is
            given, the full stack is preprocessed with
            ``reference_mode="static"``. Frames you pass should also use a
            static reference.
        save_overlays:
            Save a combined overlay per frame.
        overlay_dirname:
            Output folder under the specimen results.
        overlay_view:
            ``"classified"`` (edge and diffuse in two colors) or ``"union"``
            (one color).
        save_component_overlays:
            Also save separate edge and diffuse overlays.
        edge_overlay_view:
            ``"mask"``, ``"line"`` or ``"both"`` for the edge overlays.
        edge_exclusion_px:
            Dilation of the edge mask before resolving the overlap.
        save_masks:
            Save all masks to ``.npz``.
        masks_dirname:
            Folder for the mask files.
        save_metrics:
            Save the per-frame metrics to CSV.
        metrics_filename:
            Name of that CSV.
        max_frames:
            Process only the first ``max_frames`` frames.
        edge_params, diffuse_params:
            Parameter overrides for edge and diffuse detection.
        track_cracks:
            Follow cracks from frame to frame and look for diffuse damage
            around tracked and vanished cracks, instead of around each
            frame's cracks independently.
        max_center_px, max_angle_deg, max_cost:
            Crack matching limits used with ``track_cracks``.
        return_masks:
            Include all masks in the result.
        return_intermediates:
            With ``track_cracks``, include the tracking inputs and large
            intermediate arrays in the result.
        debug:
            Include edge detection debug data in the result.
        save_edge_debug:
            Save every intermediate edge detection image to disk.
        progress:
            Print progress.
        crack_coordinate_space:
            ``"middle"`` if the cracks were detected on the middle region
            stack, ``"full"`` if on the full frames. Used to place the cracks
            on overlays in region mode.

        Returns
        -------
        dict[str, Any]
            ``metrics`` (DataFrame), ``paths``, ``params`` and
            ``crack_tracking``, plus ``masks``, ``debug`` and
            ``_debug_internals`` when requested.
        """
        if cracks is None:
            raise ValueError("Diffuse delamination requires `cracks` to be provided.")
        if processed_cache_paths and processed_stack:
            raise ValueError("Provide either processed_cache_paths or processed_stack, not both.")
        if crack_coordinate_space not in {"middle", "full"}:
            raise ValueError("crack_coordinate_space must be one of: 'middle', 'full'.")
        if overlay_view not in {"union", "classified"}:
            raise ValueError("overlay_view must be one of: 'union', 'classified'.")

        raw_stack = getattr(self.specimen, "image_stack_full", None)
        if save_overlays and raw_stack is None:
            raise ValueError("Cannot save overlays without a full raw image stack.")

        if processed_cache_paths is None and processed_stack is None:
            processed_cache_paths = _auto_preprocess_cache_paths(
                self, save_overlays=False, max_frames=max_frames, progress=progress, key_prefix="both_auto"
            )

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
        edge_masks = edge_result["masks"]

        crack_tracking_result: Optional[Dict[str, Any]] = None
        if track_cracks:
            diffuse_masks, cracks_by_frame, crack_tracking_result, tracking_internals = self._tracked_diffuse_masks(
                cracks=cracks,
                processed_cache_paths=processed_cache_paths,
                processed_stack=processed_stack,
                max_frames=max_frames,
                diffuse_params=diffuse_params,
                max_center_px=max_center_px,
                max_angle_deg=max_angle_deg,
                max_cost=max_cost,
                return_intermediates=return_intermediates,
            )
        else:
            diffuse_masks = self.diffuse.diffuse_delamination(
                cracks=cracks,
                processed_cache_paths=processed_cache_paths,
                processed_stack=processed_stack,
                save_overlays=False,
                overlay_dirname=overlay_dirname,
                max_frames=max_frames,
                params=diffuse_params,
                debug=debug,
                progress=progress,
            )["masks"]
            cracks_by_frame = _coerce_cracks_by_frame(cracks, len(diffuse_masks))

        if self._uses_stack_overrides():
            self._clear_edge_region_rows(diffuse_masks)

        frame_keys = sorted(set(edge_masks) & set(diffuse_masks))
        if not frame_keys:
            raise ValueError("No overlapping frame keys between edge and diffuse masks.")
        _latch_masks(edge_masks, frame_keys)
        _latch_masks(diffuse_masks, frame_keys)

        exclusion_radius = max(0, int(edge_exclusion_px))
        edge_exclusion_masks: Dict[str, np.ndarray] = {}
        diffuse_final_masks: Dict[str, np.ndarray] = {}
        combined_masks: Dict[str, np.ndarray] = {}
        overlap_masks: Dict[str, np.ndarray] = {}
        metrics_rows: List[Dict[str, Any]] = []

        combined_overlay_dir = self.specimen.results_dir(overlay_dirname, "both", "overlays") if save_overlays else None
        diffuse_overlay_dir = None
        if save_component_overlays and raw_stack is not None:
            diffuse_overlay_dir = self.specimen.results_dir(overlay_dirname, "diffuse", "overlays")
        progress_log = _Progress("combined_delamination", len(frame_keys), progress)

        for idx, frame_key in enumerate(frame_keys):
            frame_idx = int(frame_key.split("_")[1])
            # Pixels claimed by both detectors (after dilating the edge mask) count as edge damage.
            edge_exclusion = _dilate_edge_mask(np.asarray(edge_masks[frame_key], dtype=bool), exclusion_radius)
            diffuse_raw = np.asarray(diffuse_masks[frame_key], dtype=bool)
            overlap = edge_exclusion & diffuse_raw
            diffuse_final = diffuse_raw & ~edge_exclusion
            combined = edge_exclusion | diffuse_final

            edge_exclusion_masks[frame_key] = edge_exclusion
            diffuse_final_masks[frame_key] = diffuse_final
            combined_masks[frame_key] = combined
            overlap_masks[frame_key] = overlap
            metrics_rows.append(
                _build_metrics_row(
                    frame_idx=frame_idx,
                    edge_mask=edge_exclusion,
                    diffuse_raw=diffuse_raw,
                    diffuse_final=diffuse_final,
                    overlap_mask=overlap,
                    combined_mask=combined,
                )
            )

            if diffuse_overlay_dir is not None or combined_overlay_dir is not None:
                raw_frame = _ensure_uint8(raw_stack[frame_idx])
                frame_cracks = self._overlay_cracks(cracks_by_frame, frame_idx, crack_coordinate_space)
            if diffuse_overlay_dir is not None:
                _save_diffuse_overlay(
                    raw_frame,
                    diffuse_final,
                    diffuse_overlay_dir / f"diffuse_overlay_{frame_idx:04d}.png",
                    cracks=frame_cracks,
                )
            if combined_overlay_dir is not None:
                _save_combined_overlay(
                    raw_frame,
                    edge_mask=edge_exclusion,
                    diffuse_mask=diffuse_final,
                    save_path=combined_overlay_dir / f"combined_overlay_{frame_idx:04d}.png",
                    view=overlay_view,
                    edge_color=EDGE_OVERLAY_RGBA,
                    diffuse_color=DIFFUSE_OVERLAY_RGBA,
                    union_color=self.interface.delamination_color_rgba,
                    cracks=frame_cracks,
                )

            progress_log.update(idx + 1)

        progress_log.done()
        metrics_df = pd.DataFrame(metrics_rows)

        paths: Dict[str, Optional[str]] = {
            "edge_raw_masks": None,
            "edge_exclusion_masks": None,
            "diffuse_raw_masks": None,
            "diffuse_masks": None,
            "combined_masks": None,
            "metrics": None,
            "combined_overlays": None if combined_overlay_dir is None else str(combined_overlay_dir),
        }

        if save_masks:
            masks_root = self.specimen.results_dir(overlay_dirname, "both", masks_dirname)
            for path_key, masks, filename in (
                ("edge_raw_masks", edge_masks, "edge_raw.npz"),
                ("edge_exclusion_masks", edge_exclusion_masks, "edge_exclusion.npz"),
                ("diffuse_raw_masks", diffuse_masks, "diffuse_raw.npz"),
                ("diffuse_masks", diffuse_final_masks, "diffuse_final.npz"),
                ("combined_masks", combined_masks, "combined.npz"),
            ):
                paths[path_key] = str(save_mask_bundle(masks, masks_root / filename))

        if save_metrics:
            metrics_dir = self.specimen.results_dir(overlay_dirname, "both", "metrics")
            paths["metrics"] = str(save_interface_metrics(metrics_df, metrics_dir / metrics_filename))

        def as_path(key: str) -> Optional[Path]:
            return Path(paths[key]) if paths[key] else None

        store_interface_delamination_results(
            self.interface,
            diffuse_raw_path=as_path("diffuse_raw_masks"),
            diffuse_path=as_path("diffuse_masks"),
            combined_path=as_path("combined_masks"),
            metrics_path=as_path("metrics"),
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
            result["debug"] = {"edge": edge_result["debug"]}
        result["crack_tracking"] = crack_tracking_result
        if return_intermediates and track_cracks:
            result["_debug_internals"] = tracking_internals
        return result

    def _tracked_diffuse_masks(
        self,
        *,
        cracks: CrackInput,
        processed_cache_paths: Optional[List[Path]],
        processed_stack: Optional[List[np.ndarray]],
        max_frames: Optional[int],
        diffuse_params: Optional[Dict[str, Any]],
        max_center_px: Optional[float],
        max_angle_deg: float,
        max_cost: float,
        return_intermediates: bool,
    ) -> Tuple[Dict[str, np.ndarray], List[Any], Dict[str, Any], Dict[str, Any]]:
        """Diffuse masks from crack tracking.

        Returns ``(masks, cracks per frame, tracking result, intermediates)``.
        """
        from deladect.detection.crack_tracking import normalize_detections

        if processed_stack is not None:
            proc_frames = list(processed_stack)[:max_frames] if max_frames else list(processed_stack)
        else:
            cache_paths = processed_cache_paths[:max_frames] if max_frames else processed_cache_paths
            proc_frames = [frame for _, frame in self.iter_preprocessed_cache(cache_paths)]
        selected_indices = list(range(len(proc_frames)))

        cracks_by_frame = _coerce_cracks_by_frame(cracks, len(proc_frames))
        crack_detections = [normalize_detections(cracks_by_frame[i]) for i in selected_indices]

        tracking = self.diffuse.diffuse_crack_tracking(
            proc_frames,
            crack_detections,
            selected_indices,
            avg_crack_width_px=self.specimen.avg_crack_width_px,
            diffuse_params=self.diffuse._resolve_diffuse_params(diffuse_params),
            max_center_px=max_center_px,
            max_angle_deg=max_angle_deg,
            max_cost=max_cost,
            return_intermediates=return_intermediates,
        )
        frame_shape = proc_frames[0].shape[:2] if proc_frames else (1, 1)
        masks = {
            f"frame_{i:04d}": tracking["frame_masks"].get(i, np.zeros(frame_shape, dtype=bool))
            for i in selected_indices
        }
        internals = {
            "proc_frames": proc_frames,
            "selected_indices": selected_indices,
            "crack_frames_normalized": crack_detections,
        }
        return masks, cracks_by_frame, tracking, internals

    def _clear_edge_region_rows(self, diffuse_masks: Dict[str, np.ndarray]) -> None:
        """Clear the upper and lower edge region rows of the diffuse masks (region mode)."""
        stacks = self._select_stacks()
        upper_stack, lower_stack = stacks.get("upper"), stacks.get("lower")
        if upper_stack is None or lower_stack is None:
            return
        upper_height = np.asarray(upper_stack[0]).shape[0]
        lower_height = np.asarray(lower_stack[0]).shape[0]
        for frame_key, mask in diffuse_masks.items():
            mask = mask.copy()
            mask[:upper_height, :] = False
            if lower_height > 0:
                mask[-lower_height:, :] = False
            diffuse_masks[frame_key] = mask

    def _overlay_cracks(
        self,
        cracks_by_frame: Sequence[Any],
        frame_idx: int,
        crack_coordinate_space: str,
    ) -> Optional[List[np.ndarray]]:
        """Cracks of frame ``frame_idx`` in full-frame coordinates, for overlays."""
        cracks = cracks_by_frame[frame_idx] if frame_idx < len(cracks_by_frame) else None
        if not self._uses_stack_overrides():
            return cracks
        return self._cracks_for_full_overlay(
            cracks,
            shift=(crack_coordinate_space == "middle"),
            upper_height=int(np.asarray(self.specimen.image_stack_upper[frame_idx]).shape[0]),
        )

    def _uses_stack_overrides(self) -> bool:
        """``True`` when upper, lower and middle region stacks were all given."""
        return self._region_mode

    def _select_stacks(self) -> Dict[str, Optional[List[np.ndarray]]]:
        """The region stacks in region mode, otherwise only the full stack."""
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
        """Crack segments as ``(n, 2)`` arrays, moved down by ``upper_height`` if ``shift``.

        ``shift`` has to come from the caller: a middle-region crack near
        ``y=0`` and a full-frame crack near the top have the same
        coordinates.
        """
        if cracks is None:
            return None

        prepared: List[np.ndarray] = []
        for segment in cracks:
            try:
                arr = np.asarray(segment, dtype=float).reshape(-1, 2)
            except Exception:
                continue
            if arr.shape[0] >= 2:
                prepared.append(arr)

        if shift and upper_height > 0:
            offset = np.array([float(upper_height), 0.0])
            return [arr + offset for arr in prepared]
        return prepared

    def _images_threshold(self, image: np.ndarray, window_edge: Tuple[int, int]) -> float:
        """Otsu threshold after a max and a min filter over ``window_edge``."""
        return _minmax_otsu_threshold(image, window_edge)

    def _kmeans_threshold(
        self,
        image: np.ndarray,
        fallback: float,
        *,
        max_iter: int = 20,
        tol: float = 1e-2,
    ) -> float:
        """Two-cluster k-means threshold of ``image``, or ``fallback`` if it can't be split."""
        threshold = _kmeans_split(image, max_iter=max_iter, tol=tol)
        return float(fallback) if threshold is None else threshold
