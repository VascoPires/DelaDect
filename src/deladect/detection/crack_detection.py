"""Minimal crack-detection helpers layered on top of the legacy CrackDect pipeline.

Entry points:

* :func:`crack_eval` - run CrackDect for one ply orientation.
* :func:`crack_eval_by_orientation` - run CrackDect for a set of orientations on a specimen.
* :func:`crack_eval_crossply` - convenience for 0/90 cross-ply laminates.
* :func:`crack_eval_plus_minus` - convenience for +/- theta laminates.
* :func:`plot_cracks` - small Matplotlib helper to visualise detected segments.

Coordinate convention
---------------------
Crack segments are represented as ``[[row0, col0], [row1, col1]]`` (``[y, x]``).
When plotting, columns map to the x-axis and rows map to the y-axis.

Reference: CrackDect publication, https://doi.org/10.1016/j.softx.2021.100832
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from skimage.io import imread

from deladect.io.cracks import (
    crack_results_subdir,
    save_cracks as save_crack_bundle,
)
from deladect.specimen import Ply, Specimen

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency during tests
    from crackdect import detect_cracks_bender as _detect_cracks_bender
except Exception:  # pragma: no cover
    _detect_cracks_bender = None  # type: ignore[assignment]


def crack_eval(
    specimen: Specimen,
    *,
    crack_width_px: Optional[float] = None,
    min_crack_size_px: Optional[float] = None,
    export_images: bool = False,
    background: bool = False,
    comparison: bool = False,
    save_cracks: bool = False,
    ply: Optional[Ply] = None,
    results_dir: Optional[str] = None,
    use_full_stack: Optional[bool] = None,
    color_cracks: str = "red",
    frame_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run CrackDect on one ply orientation.

    The ply orientation (laminate convention, commonly in ``[-90, 90]``) is mapped
    into CrackDect's detection angle by

    ``theta_fd = (90 - ply.orientation_deg) % 180``.

    Parameters
    ----------
    specimen:
        Specimen containing the image stacks and output root.
    crack_width_px, min_crack_size_px:
        Optional overrides for CrackDect filter and length thresholds.
        If omitted, ply-level defaults are used.
    export_images:
        If ``True``, write per-frame crack overlays.
    background:
        If ``True``, draw overlays on top of grayscale frames.
    comparison:
        If ``True``, duplicate the background horizontally in plots.
    save_cracks:
        If ``True``, persist crack segments as ``.npz``.
    ply:
        Ply being analysed. Required.
    results_dir:
        Optional result-root override.
    use_full_stack:
        If ``True``, force full-stack detection. If ``False``, prefer middle-stack.
        If ``None``, choose automatically.
    color_cracks:
        Matplotlib color for exported overlays.
    Returns
    -------
    dict[str, Any]
        Dictionary containing cracks, per-frame metrics, output paths, and parameters.
    """
    if _detect_cracks_bender is None:
        raise ImportError("crackdect is required to run crack detection.")
    if ply is None:
        raise ValueError("`ply` must be provided for crack detection.")

    stack = _select_stack(specimen, use_full_stack)
    theta = _theta_from_ply(ply)
    crack_width_source = (
        crack_width_px
        if crack_width_px is not None
        else (ply.avg_crack_width_px or specimen.avg_crack_width_px or 10.0)
    )
    crack_width = int(round(float(crack_width_source)))
    min_size_source = (
        min_crack_size_px
        if min_crack_size_px is not None
        else (ply.min_crack_length_px if ply.min_crack_length_px is not None else max(crack_width * 2.0, crack_width))
    )
    min_size = int(round(float(min_size_source)))

    densities, cracks, thresholds = _detect_cracks_bender(
        stack,
        theta=theta,
        crack_width=crack_width,
        min_size=min_size,
    )

    cracks_list = list(cracks)
    densities_list = [float(value) for value in densities]
    thresholds_list = [float(value) for value in thresholds]

    plots_path: Optional[str] = None
    crack_bundle_path: Optional[str] = None

    if export_images:
        plots_dir = crack_results_subdir(
            specimen,
            ply,
            "plots",
            results_root=results_dir,
        )
        plots_path = str(plots_dir)
        for idx, crack in enumerate(cracks_list):
            frame = stack[idx]
            fig, ax = plot_cracks(
                frame,
                crack,
                background_flag=background,
                color=color_cracks,
                comparison=comparison,
            )
            ax.set_xlabel("x [Px]")
            ax.set_ylabel("y [Px]")
            label = frame_labels[idx] if frame_labels is not None else f"{idx:04d}"
            fig.savefig(str(plots_dir / f"cracks_{label}.png"))
            plt.close(fig)

    if save_cracks:
        saved = save_crack_bundle(
            specimen,
            ply,
            cracks_list,
            results_root=results_dir,
        )
        crack_bundle_path = str(saved)

    metrics = _build_crack_metrics_table(cracks_list, densities_list, thresholds_list)
    return {
        "cracks": cracks_list,
        "densities": densities_list,
        "thresholds": thresholds_list,
        "metrics": metrics,
        "paths": {
            "plots": plots_path,
            "cracks": crack_bundle_path,
        },
        "params": {
            "theta_fd": theta,
            "crack_width_px": crack_width,
            "min_crack_size_px": min_size,
        },
        "orientation_deg": float(ply.orientation_deg),
        "ply": ply,
    }


def crack_eval_by_orientation(
    specimen: Specimen,
    *,
    orientations: Optional[Sequence[float]] = None,
    tolerance: float = 1e-3,
    crack_width_px: Optional[float] = None,
    min_crack_size_px: Optional[float] = None,
    export_images: bool = False,
    background: bool = False,
    comparison: bool = False,
    save_cracks: bool = False,
    results_dir: Optional[str] = None,
    use_full_stack: Optional[bool] = None,
    color_cracks: str = "red",
    frame_labels: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run crack detection once per unique ply orientation.

    Plies are grouped by orientation (within ``tolerance``). One representative ply
    from each orientation group is used for detection and returned together with
    metadata and raw CrackDect outputs.

    Each orientation entry includes ``cracks``, ``densities``, ``thresholds``,
    ``metrics``, ``paths``, and ``params``.
    """
    groups = _group_plies_by_orientation(specimen, tolerance=tolerance)
    target_orientations = list(orientations) if orientations is not None else None
    results: Dict[str, Dict[str, Any]] = {}

    def _matches_target(angle: float) -> bool:
        if target_orientations is None:
            return True
        return any(abs(angle - target) <= tolerance for target in target_orientations)

    for angle, plies in groups:
        if not _matches_target(angle):
            continue
        if not plies:
            continue
        primary = plies[0]
        if len(plies) > 1:
            duplicate_names = ", ".join(ply.name for ply in plies[1:])
            logger.warning(
                "Multiple plies found at %.3f°; using '%s' and merging %d duplicates (%s).",
                angle,
                primary.name,
                len(plies) - 1,
                duplicate_names,
            )
            primary_signature = (primary.avg_crack_width_px, primary.min_crack_length_px)
            for candidate in plies[1:]:
                candidate_signature = (candidate.avg_crack_width_px, candidate.min_crack_length_px)
                if candidate_signature != primary_signature:
                    logger.warning(
                        "Duplicate ply '%s' at %.3f° has different crack settings; "
                        "using '%s' defaults.",
                        candidate.name,
                        angle,
                        primary.name,
                    )
                    break

        structured = crack_eval(
            specimen,
            ply=primary,
            crack_width_px=crack_width_px,
            min_crack_size_px=min_crack_size_px,
            export_images=export_images,
            background=background,
            comparison=comparison,
            save_cracks=save_cracks,
            results_dir=results_dir,
            use_full_stack=use_full_stack,
            color_cracks=color_cracks,
            frame_labels=frame_labels,
        )
        payload: Dict[str, Any] = {
            "orientation_deg": angle,
            "ply": primary,
            "plies": plies,
            "cracks": structured.get("cracks", []),
            "densities": structured.get("densities", []),
            "thresholds": structured.get("thresholds", []),
            "metrics": structured.get("metrics"),
            "paths": structured.get("paths", {}),
            "params": structured.get("params", {}),
        }

        label = _orientation_label(angle)
        results[label] = payload

    if target_orientations is not None:
        missing = [
            target
            for target in target_orientations
            if not any(abs(angle - target) <= tolerance for angle, _ in groups)
        ]
        if missing:
            logger.warning(
                "Requested orientations not found in specimen '%s': %s",
                specimen.name,
                ", ".join(str(value) for value in missing),
            )

    return results


def crack_eval_crossply(
    specimen: Specimen,
    *,
    crack_width_px: Optional[float] = None,
    min_crack_size_px: Optional[float] = None,
    export_images: bool = False,
    background: bool = False,
    comparison: bool = False,
    save_cracks: bool = False,
    results_dir: Optional[str] = None,
    use_full_stack: Optional[bool] = None,
    color_cracks: str = "red",
    tolerance: float = 1e-3,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate a cross-ply laminate at 0 and 90 degrees.

    This is a convenience wrapper around :func:`crack_eval_by_orientation`.
    """
    return crack_eval_by_orientation(
        specimen,
        orientations=[0.0, 90.0],
        tolerance=tolerance,
        crack_width_px=crack_width_px,
        min_crack_size_px=min_crack_size_px,
        export_images=export_images,
        background=background,
        comparison=comparison,
        save_cracks=save_cracks,
        results_dir=results_dir,
        use_full_stack=use_full_stack,
        color_cracks=color_cracks,
    )


def crack_eval_plus_minus(
    specimen: Specimen,
    theta: float,
    *,
    transverse_layer: bool = False,
    crack_width_px: Optional[float] = None,
    min_crack_size_px: Optional[float] = None,
    export_images: bool = False,
    background: bool = False,
    comparison: bool = False,
    save_cracks: bool = False,
    results_dir: Optional[str] = None,
    use_full_stack: Optional[bool] = None,
    color_cracks: str = "red",
    tolerance: float = 1e-3,
    frame_labels: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate a plus/minus laminate and optionally a transverse layer.

    Parameters
    ----------
    theta:
        Positive laminate angle. Detection is performed for ``+theta`` and ``-theta``.
    transverse_layer:
        If ``True``, also evaluate 90-degree cracks.
    """
    orientations: List[float] = [float(theta), float(-theta)]
    if transverse_layer:
        orientations.append(90.0)
    return crack_eval_by_orientation(
        specimen,
        orientations=orientations,
        tolerance=tolerance,
        crack_width_px=crack_width_px,
        min_crack_size_px=min_crack_size_px,
        export_images=export_images,
        background=background,
        comparison=comparison,
        save_cracks=save_cracks,
        results_dir=results_dir,
        use_full_stack=use_full_stack,
        color_cracks=color_cracks,
        frame_labels=frame_labels,
    )


def plot_cracks(
    image: np.ndarray,
    cracks: Sequence[np.ndarray],
    *,
    linewidth: float = 1.0,
    color: str = "red",
    background_flag: bool = False,
    comparison: bool = False,
):
    """Plot crack segments with an optional grayscale background.

    This helper mirrors the visual style used in CrackDect examples and tests.

    Parameters
    ----------
    image:
        Background image array.
    cracks:
        Iterable of crack segments shaped ``(n, 2, 2)`` with ``(y, x)`` endpoints.
    linewidth:
        Width of crack lines.
    color:
        Matplotlib color for crack segments.
    background_flag:
        If ``True``, draw ``image`` before plotting cracks.
    comparison:
        If ``True``, duplicate the frame horizontally for side-by-side comparisons.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Figure/axes containing the rendered crack overlay.
    """
    fig, ax = plt.subplots()
    frame = image
    if comparison:
        frame = np.hstack((image, image))
    if background_flag:
        vmin, vmax = (0, np.iinfo(frame.dtype).max) if np.issubdtype(frame.dtype, np.integer) else (None, None)
        ax.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax)
    if cracks is not None:
        for segment in cracks:
            if len(segment) != 2:
                continue
            (y0, x0), (y1, x1) = segment
            ax.plot((x0, x1), (y0, y1), color=color, linewidth=linewidth, linestyle="-")
    ax.set_ylim(frame.shape[0], 0)
    ax.set_xlim(0, frame.shape[1])
    ax.set_aspect("equal")
    ax.tick_params(axis="both", which="both", length=0)
    ax.grid(False)
    return fig, ax


def _group_plies_by_orientation(
    specimen: Specimen,
    *,
    tolerance: float = 1e-3,
) -> List[Tuple[float, List[Ply]]]:
    """Group specimen plies by orientation within ``tolerance`` degrees."""
    groups: List[Tuple[float, List[Ply]]] = []
    for ply in specimen.plies:
        matched = False
        for idx, (angle, plies) in enumerate(groups):
            if abs(ply.orientation_deg - angle) <= tolerance:
                plies.append(ply)
                matched = True
                break
        if not matched:
            groups.append((float(ply.orientation_deg), [ply]))
    return groups


def _orientation_label(angle: float) -> str:
    """Return a stable string key for an orientation angle."""
    if abs(angle - round(angle)) <= 1e-6:
        return str(int(round(angle)))
    return f"{angle:g}"


def order_cracks(
    crack_list: np.ndarray,
    *,
    delimiter: bool = True,
    image_height: int = 1,
    image_width: int = 1,
) -> np.ndarray:
    """Order cracks by their minimum column (x-axis) coordinate.

    Crack segments are represented as ``[row, col]``. Sorting by minimum column
    orders cracks left-to-right, which is the spacing direction for near-vertical
    crack families.
    """
    if len(crack_list) == 0:
        return crack_list
    min_col = np.minimum(crack_list[:, 0, 1], crack_list[:, 1, 1])
    ordered = crack_list[np.argsort(min_col)]
    if delimiter:
        left_boundary = np.array([[[0.0, 0.0], [float(image_height), 0.0]]])
        right_boundary = np.array([[[0.0, float(image_width)], [float(image_height), float(image_width)]]])
        ordered = np.vstack((left_boundary, ordered, right_boundary))
    return ordered


def crack_grouping(
    ordered_cracks: np.ndarray,
    *,
    threshold: float = 5.0,
    generate_vertical_crack: bool = True,
    group_within_crack_width: bool = True,
    avg_crack_width_px: float = 10.0,
) -> np.ndarray:
    """Group cracks that lie within ``threshold`` pixels of each other.

    Grouping is performed along the column axis (left-right distance) while keeping
    output cracks in ``[row, col]`` format.
    """
    if len(ordered_cracks) == 0:
        return ordered_cracks

    grouped = ordered_cracks
    if group_within_crack_width:
        span = max(avg_crack_width_px * 2.0, 1.0)
        merged: List[np.ndarray] = []
        idx = 0
        while idx < len(grouped):
            band = [grouped[idx]]
            col_ref = np.mean(grouped[idx][:, 1])
            j = idx + 1
            while j < len(grouped) and abs(np.mean(grouped[j][:, 1]) - col_ref) <= span:
                band.append(grouped[j])
                j += 1
            if len(band) > 1:
                rows = [pt[0] for crack in band for pt in crack]
                cols = [pt[1] for crack in band for pt in crack]
                merged.append(np.array([[min(rows), np.mean(cols)], [max(rows), np.mean(cols)]]))
            else:
                merged.append(band[0])
            idx = j
        grouped = np.array(merged)

    updated: List[np.ndarray] = []
    i = 0
    while i < len(grouped) - 1:
        current = grouped[i]
        nxt = grouped[i + 1]
        d1 = np.linalg.norm(current[1] - nxt[0])
        d2 = np.linalg.norm(nxt[1] - current[0])
        if min(d1, d2) <= threshold:
            if generate_vertical_crack:
                rows = [pt[0] for pt in np.vstack((current, nxt))]
                cols = [pt[1] for pt in np.vstack((current, nxt))]
                col_mid = (min(cols) + max(cols)) / 2
                updated.append(np.array([[min(rows), col_mid], [max(rows), col_mid]]))
            else:
                updated.append(np.array([current[0], nxt[1]]))
            i += 2
        else:
            updated.append(current)
            i += 1
    if i == len(grouped) - 1:
        updated.append(grouped[-1])
    return np.array(updated)


def crack_filter(crack_list: List[np.ndarray], *, length_threshold: float) -> List[np.ndarray]:
    """Filter cracks shorter than ``length_threshold``."""
    from deladect.utils import crack_length

    return [crack for crack in crack_list if crack_length(crack) >= length_threshold]


def compute_crack_spacing(crack_list: List[np.ndarray]) -> Tuple[List[float], float, float]:
    """Compute crack spacing statistics from segment midpoints.

    Spacing is measured along the column axis (x-axis / left-right distance).
    Crack coordinates are expected in ``[row, col]`` format.
    """
    if not crack_list:
        return [], 0.0, 0.0
    from deladect.utils import crack_mid_point

    raw_midpoints = [crack_mid_point(crack) for crack in crack_list]
    midpoints: List[Tuple[float, float]] = []
    for row_val, col_val in raw_midpoints:
        if row_val is None or col_val is None:
            continue
        midpoints.append((float(cast(float, row_val)), float(cast(float, col_val))))
    midpoints = sorted(midpoints, key=lambda pt: pt[1])
    spacings = [
        midpoints[idx + 1][1] - midpoints[idx][1]
        for idx in range(len(midpoints) - 1)
    ]
    avg = float(np.mean(spacings)) if spacings else 0.0
    std = float(np.std(spacings)) if spacings else 0.0
    return spacings, avg, std


def crack_filtering_postprocessing(
    specimen: Specimen,
    cracks: List[np.ndarray],
    *,
    avg_crack_grouping_th_px: float = 10.0,
    crack_length_th: float = 5.0,
    export_images: bool = False,
    background: bool = False,
    remove_outliers: bool = True,
    grouping: bool = False,
    results_dir: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[List[np.ndarray]]]:
    """Post-process crack frames and compute spacing statistics.

    The routine can order cracks, apply length filtering, optionally group nearby
    segments, compute spacing statistics, remove outliers (IQR rule), and optionally
    export overlay plots.

    Returns
    -------
    tuple[list[dict[str, Any]], list[list[np.ndarray]]]
        ``(records, filtered_frames)`` where ``records`` contains per-frame spacing
        metrics and ``filtered_frames`` stores the filtered crack lists.
    """
    if not cracks:
        return [], []

    reference_paths = specimen.path_middle_list or specimen.path_full_list
    if not reference_paths:
        raise ValueError("Specimen has no image paths to reference.")
    image = imread(reference_paths[0])
    height, width = image.shape[:2]
    records: List[Dict[str, Any]] = []
    filtered_frames: List[List[np.ndarray]] = []

    plots_dir = None
    if export_images:
        plots_dir = specimen.results_dir(
            "plots",
            "filtered_cracks",
            results_root=results_dir,
        )

    for idx, frame_cracks in enumerate(cracks):
        ordered = order_cracks(
            frame_cracks,
            delimiter=True,
            image_height=height,
            image_width=width,
        )
        filtered = crack_filter(list(ordered), length_threshold=crack_length_th)
        grouped = (
            crack_grouping(
                np.asarray(filtered),
                threshold=avg_crack_grouping_th_px,
                generate_vertical_crack=True,
                group_within_crack_width=True,
                avg_crack_width_px=specimen.avg_crack_width_px,
            )
            if grouping
            else np.asarray(filtered)
        )

        spacing, avg_spacing, std_spacing = compute_crack_spacing(list(grouped))

        if remove_outliers and spacing:
            spacing_array = np.asarray(spacing)
            q1, q3 = np.percentile(spacing_array, [25, 75])
            iqr = q3 - q1
            mask = (spacing_array >= q1 - 1.5 * iqr) & (spacing_array <= q3 + 1.5 * iqr)
            filtered_spacing = spacing_array[mask]
            if filtered_spacing.size:
                avg_spacing = float(np.mean(filtered_spacing))
                std_spacing = float(np.std(filtered_spacing))

        records.append(
            {
                "Picture": idx,
                "Avg_spacing": avg_spacing / specimen.scale_px_mm,
                "Std_spacing": std_spacing / specimen.scale_px_mm,
            }
        )
        filtered_frames.append(filtered)

        if export_images and plots_dir is not None:
            fig, ax = plot_cracks(
                image,
                filtered,
                color="black",
                background_flag=background,
            )
            ax.set_xlabel("x [Px]")
            ax.set_ylabel("y [Px]")
            fig.savefig(str(plots_dir / f"filtered_{idx:04d}.png"))
            plt.close(fig)

    return records, filtered_frames


def pixels_to_length(input_data: List[Any], *, scale_px_mm: float) -> List[Any]:
    """Convert crack-related data from pixels to millimetres.

    ``input_data`` can be either:

    - a list of numeric values (for direct scaling), or
    - a list of spacing dictionaries with ``Avg_spacing`` and ``Std_spacing`` keys.
    """
    if all(isinstance(value, (int, float)) for value in input_data):
        return [value / scale_px_mm for value in input_data]
    scaled: List[Dict[str, Any]] = []
    for entry in input_data:
        if not isinstance(entry, dict):
            raise ValueError("Input must be rho list or processed crack-spacing data.")
        scaled.append(
            {
                "Picture": entry.get("Picture"),
                "Avg_spacing": entry.get("Avg_spacing", 0.0) / scale_px_mm,
                "Std_spacing": entry.get("Std_spacing", 0.0) / scale_px_mm,
            }
        )
    return scaled


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_crack_metrics_table(
    cracks: List[Sequence[np.ndarray]],
    densities: List[float],
    thresholds: List[float],
) -> pd.DataFrame:
    """Build a per-frame crack metrics table for structured outputs."""
    rows: List[Dict[str, Any]] = []
    frame_count = max(len(cracks), len(densities), len(thresholds))
    for frame_idx in range(frame_count):
        frame_cracks = cracks[frame_idx] if frame_idx < len(cracks) else []
        rows.append(
            {
                "frame": frame_idx,
                "crack_count": int(len(frame_cracks)),
                "rho": float(densities[frame_idx]) if frame_idx < len(densities) else 0.0,
                "threshold_rho": float(thresholds[frame_idx]) if frame_idx < len(thresholds) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _select_stack(specimen: Specimen, use_full_stack: Optional[bool]):
    """Select middle/full image stack with deterministic fallback warnings."""
    middle_stack = getattr(specimen, "image_stack_middle", None)
    full_stack = getattr(specimen, "image_stack_full", None)

    if use_full_stack is None:
        # auto mode: prefer already-loaded middle stack if available
        if middle_stack is not None:
            use_full = False
        elif full_stack is not None:
            use_full = True
        else:
            raise ValueError("Specimen has no image stack for region 'path_full' or 'path_middle'.")
    else:
        use_full = use_full_stack

    if specimen.path_middle is None and (specimen.path_upper_border or specimen.path_lower_border):
        warnings.warn(
            "Upper/lower stacks were provided without a middle stack. Attempting detection in the whole picture stack.",
            RuntimeWarning,
            stacklevel=3,
        )

    if not use_full:
        if middle_stack is not None:
            return middle_stack
        if full_stack is not None:
            warnings.warn(
                "Middle stack requested but no middle stack is available; using full stack instead.",
                RuntimeWarning,
                stacklevel=3,
            )
            return full_stack
        raise ValueError("Specimen has no image stack for region 'path_middle' or 'path_full'.")

    if full_stack is not None:
        return full_stack
    if middle_stack is not None:
        warnings.warn(
            "Full stack requested but no full stack is available; using middle stack instead.",
            RuntimeWarning,
            stacklevel=3,
        )
        return middle_stack
    raise ValueError("Specimen has no image stack for region 'path_full' or 'path_middle'.")


def _theta_from_ply(ply: Ply) -> int:
    """Map laminate ply angle to CrackDect detection angle."""
    angle = float(ply.orientation_deg)
    theta_fd = (90.0 - angle) % 180.0
    return int(round(theta_fd))


__all__ = [
    "crack_eval",
    "crack_eval_by_orientation",
    "crack_eval_crossply",
    "crack_eval_plus_minus",
    "plot_cracks",
    "order_cracks",
    "crack_grouping",
    "crack_filter",
    "compute_crack_spacing",
    "crack_filtering_postprocessing",
    "pixels_to_length",
]
