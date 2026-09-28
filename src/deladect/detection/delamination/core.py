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
    store_interface_masks,
)
from deladect.io import layout as layout_io
from deladect.specimen import Interface, Specimen

from ._common import (
    CrackInput,
    _Progress,
    _cracks_by_frame,
    _ensure_uint8,
)
from ._overlays import (
    DIFFUSE_OVERLAY_RGBA,
    EDGE_OVERLAY_RGBA,
    _save_combined_overlay,
    _save_diffuse_overlay,
    _save_edge_overlay,
    _save_single_overlay,
)
from ._frames import PreprocessedFrames, resolve_frames
from ._preprocess import PreprocessingMixin
from ._regions import RegionLayout
from .diffuse import DiffuseDetector
from .edge import EdgeDetector

_SAVED_OVERLAY_TYPES = ("edge", "diffuse", "both", "total_dela")


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
        self.layout = RegionLayout.from_specimen(specimen)
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
        if overlay_type not in _SAVED_OVERLAY_TYPES:
            raise ValueError("overlay_type must be one of: 'diffuse', 'edge', 'both', 'total_dela'.")

        raw_stack = getattr(self.specimen, "image_stack_full", None)
        if raw_stack is None:
            raise ValueError("Cannot save overlays without a full raw image stack.")

        raw_frame = _ensure_uint8(raw_stack[frame_idx])
        frame_key = f"frame_{frame_idx:04d}"
        masks_root = layout_io.combined_masks_dir(self.specimen, overlay_dirname, masks_dirname)

        def load(name: str) -> Optional[np.ndarray]:
            return _load_mask_frame(masks_root / layout_io.COMBINED_MASK_FILES[name], frame_key)

        edge_raw = load("edge_raw")
        if edge_raw is None:
            raise ValueError("Edge masks are missing. Run detect_both_delaminations with save_masks=True.")

        edge_exclusion = load("edge_exclusion")
        if edge_exclusion is None:
            edge_exclusion = _dilate_edge_mask(edge_raw, max(0, int(edge_exclusion_px)))

        diffuse_final = load("diffuse")
        if diffuse_final is None:
            diffuse_final = load("diffuse_raw")
        if overlay_type in {"diffuse", "both"} and diffuse_final is None:
            raise ValueError("Diffuse masks are missing. Run detect_both_delaminations with save_masks=True.")

        combined = load("combined")
        if combined is None and diffuse_final is not None:
            combined = edge_exclusion | diffuse_final

        if save_path is None:
            save_path = layout_io.overlay_dir(self.specimen, overlay_dirname, overlay_type) / layout_io.overlay_name(
                overlay_type, frame_idx
            )

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
            used). Diffuse detection looks around these cracks. There must be
            one entry per frame.
        processed_cache_paths, processed_stack:
            Preprocessed frames, as cache files or arrays. If neither is
            given, the full stack is preprocessed with
            ``reference_mode="static"``. Frames you pass should also use a
            static reference. In region mode the regions are cut from these
            full frames.
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
            Save all masks to ``.npz`` and record the files on the interface
            (the edge masks as its primary masks).
        masks_dirname:
            Folder for the mask files.
        save_metrics:
            Save the per-frame metrics to CSV.
        metrics_filename:
            Name of that CSV.
        max_frames:
            Process only the first ``max_frames`` frames (at least 1), of
            both the images and the cracks.
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
            stack, ``"full"`` if on the full frames. Only used to place the
            cracks on overlays in region mode. Detection in region mode needs
            cracks from the middle region stack; if the last frame has cracks
            outside the middle region, a ``ValueError`` is raised.

        Returns
        -------
        dict[str, Any]
            ``metrics`` (DataFrame), ``paths``, ``params`` and
            ``crack_tracking``, plus ``masks``, ``debug`` and
            ``_debug_internals`` when requested.
        """
        if cracks is None:
            raise ValueError("Diffuse delamination requires `cracks` to be provided.")
        if crack_coordinate_space not in {"middle", "full"}:
            raise ValueError("crack_coordinate_space must be one of: 'middle', 'full'.")
        if overlay_view not in {"union", "classified"}:
            raise ValueError("overlay_view must be one of: 'union', 'classified'.")
        if edge_overlay_view not in {"mask", "line", "both"}:
            raise ValueError("edge_overlay_view must be one of: 'mask', 'line', 'both'.")

        raw_stack = getattr(self.specimen, "image_stack_full", None)
        if (save_overlays or save_component_overlays) and raw_stack is None:
            raise ValueError("Cannot save overlays without a full raw image stack.")

        frames = resolve_frames(
            self,
            processed_cache_paths=processed_cache_paths,
            processed_stack=processed_stack,
            max_frames=max_frames,
            auto_key="both_auto",
            progress=progress,
        )
        cracks_by_frame = _cracks_by_frame(cracks, len(frames), max_frames)
        self.layout.check_cracks_in_middle(cracks_by_frame)

        edge_result = self.edge._detect_primary(
            frames,
            edge_params=self.edge._resolve_primary_params(edge_params),
            save_overlays=save_component_overlays,
            overlay_dirname=overlay_dirname,
            overlay_view=edge_overlay_view,
            debug=debug,
            progress=progress,
            debug_root=self.specimen.results_dir("edge_accumulation_debug") if save_edge_debug else None,
        )
        edge_masks = edge_result["masks"]

        resolved_diffuse = self.diffuse._resolve_diffuse_params(diffuse_params)
        crack_tracking_result: Optional[Dict[str, Any]] = None
        if track_cracks:
            diffuse_masks, crack_tracking_result, tracking_internals = self._tracked_diffuse_masks(
                frames,
                cracks_by_frame,
                diffuse_params=resolved_diffuse,
                max_center_px=max_center_px,
                max_angle_deg=max_angle_deg,
                max_cost=max_cost,
                return_intermediates=return_intermediates,
            )
        else:
            diffuse_masks = self.diffuse._detect(
                frames,
                cracks_by_frame,
                params=resolved_diffuse,
                save_overlays=False,
                overlay_dirname=overlay_dirname,
                debug=debug,
                progress=progress,
                crack_coordinate_space=crack_coordinate_space,
            )["masks"]

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

        combined_overlay_dir = layout_io.overlay_dir(self.specimen, overlay_dirname, "both") if save_overlays else None
        diffuse_overlay_dir = None
        if save_component_overlays and raw_stack is not None:
            diffuse_overlay_dir = layout_io.overlay_dir(self.specimen, overlay_dirname, "diffuse")
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
                frame_cracks = self.layout.cracks_for_display(cracks_by_frame[frame_idx], crack_coordinate_space)
            if diffuse_overlay_dir is not None:
                _save_diffuse_overlay(
                    raw_frame,
                    diffuse_final,
                    diffuse_overlay_dir / layout_io.overlay_name("diffuse", frame_idx),
                    cracks=frame_cracks,
                )
            if combined_overlay_dir is not None:
                _save_combined_overlay(
                    raw_frame,
                    edge_mask=edge_exclusion,
                    diffuse_mask=diffuse_final,
                    save_path=combined_overlay_dir / layout_io.overlay_name("both", frame_idx),
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
            masks_root = layout_io.combined_masks_dir(self.specimen, overlay_dirname, masks_dirname)
            for name, masks in (
                ("edge_raw", edge_masks),
                ("edge_exclusion", edge_exclusion_masks),
                ("diffuse_raw", diffuse_masks),
                ("diffuse", diffuse_final_masks),
                ("combined", combined_masks),
            ):
                paths[f"{name}_masks"] = str(save_mask_bundle(masks, masks_root / layout_io.COMBINED_MASK_FILES[name]))

        if save_metrics:
            metrics_dir = layout_io.combined_metrics_dir(self.specimen, overlay_dirname)
            paths["metrics"] = str(save_interface_metrics(metrics_df, metrics_dir / metrics_filename))

        def as_path(key: str) -> Optional[Path]:
            return Path(paths[key]) if paths[key] else None

        store_interface_masks(self.interface, primary_path=as_path("edge_raw_masks"))
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
        frames: PreprocessedFrames,
        cracks_by_frame: List[Any],
        *,
        diffuse_params: Dict[str, Any],
        max_center_px: Optional[float],
        max_angle_deg: float,
        max_cost: float,
        return_intermediates: bool,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], Dict[str, Any]]:
        """Diffuse masks from crack tracking, on the diffuse rows of each frame.

        Returns ``(masks, tracking result, intermediates)``.
        """
        from deladect.detection.crack_tracking import normalize_detections

        rows = self.layout.diffuse_rows()
        proc_frames = (frames if rows is None else frames.rows(*rows)).to_list()
        selected_indices = list(range(len(proc_frames)))
        crack_detections = [normalize_detections(cracks_by_frame[i]) for i in selected_indices]

        tracking = self.diffuse.diffuse_crack_tracking(
            proc_frames,
            crack_detections,
            selected_indices,
            avg_crack_width_px=self.specimen.avg_crack_width_px,
            diffuse_params=diffuse_params,
            max_center_px=max_center_px,
            max_angle_deg=max_angle_deg,
            max_cost=max_cost,
            return_intermediates=return_intermediates,
        )
        frame_shape = proc_frames[0].shape[:2] if proc_frames else (1, 1)
        masks = {
            f"frame_{i:04d}": self.layout.diffuse_to_full(
                tracking["frame_masks"].get(i, np.zeros(frame_shape, dtype=bool))
            )
            for i in selected_indices
        }
        internals = {
            "proc_frames": proc_frames,
            "selected_indices": selected_indices,
            "crack_frames_normalized": crack_detections,
        }
        return masks, tracking, internals
