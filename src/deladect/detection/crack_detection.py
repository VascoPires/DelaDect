"""Crack detection with CrackDect.

:func:`crack_analysis` runs CrackDect once per ply orientation,
:func:`crack_eval` runs it for one ply and :func:`plot_cracks` draws
the result. The other functions compute crack spacing.

Cracks are segments ``[[row0, col0], [row1, col1]]``, i.e. ``[y, x]``.

CrackDect: https://doi.org/10.1016/j.softx.2021.100832
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.io import imread

from deladect.io.cracks import crack_results_subdir, save_cracks as save_crack_bundle
from deladect.specimen import Ply, Specimen
from deladect.utils import crack_length, crack_mid_point, draw_crack_segments

logger = logging.getLogger(__name__)

try:
    from crackdect import detect_cracks_bender as _detect_cracks_bender
except Exception:
    _detect_cracks_bender = None


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
    """Detect the cracks of one ply with CrackDect.

    The ply angle is converted to CrackDect's detection angle as
    ``(90 - ply.orientation_deg) % 180``.

    Parameters
    ----------
    specimen:
        Specimen with the image stacks.
    crack_width_px, min_crack_size_px:
        Override the ply's crack width and minimum crack length.
    export_images:
        Save a crack plot per frame.
    background:
        Draw the frame behind the cracks in those plots.
    comparison:
        Show the frame twice side by side in those plots.
    save_cracks:
        Save the cracks to ``.npz`` and record the file on the ply.
    ply:
        Ply to analyze (required).
    results_dir:
        Write results under this root instead of the specimen's.
    use_full_stack:
        ``True`` uses the full frames, ``False`` the middle region, and
        ``None`` the middle region when it was given.
    color_cracks:
        Crack color in the plots.
    frame_labels:
        Labels used in the plot file names instead of the frame index.

    Returns
    -------
    dict[str, Any]
        ``cracks``, ``densities``, ``thresholds``, a per-frame ``metrics``
        table, output ``paths``, the ``params`` used, ``orientation_deg``
        and ``ply``.
    """
    if _detect_cracks_bender is None:
        raise ImportError("crackdect is required to run crack detection.")
    if ply is None:
        raise ValueError("`ply` must be provided for crack detection.")

    stack = _select_stack(specimen, use_full_stack)
    theta = _theta_from_ply(ply)
    if crack_width_px is None:
        crack_width_px = ply.avg_crack_width_px or specimen.avg_crack_width_px or 10.0
    crack_width = int(round(float(crack_width_px)))
    if min_crack_size_px is None:
        min_crack_size_px = (
            ply.min_crack_length_px if ply.min_crack_length_px is not None else max(crack_width * 2.0, crack_width)
        )
    min_size = int(round(float(min_crack_size_px)))

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
        plots_dir = crack_results_subdir(specimen, ply, "plots", results_root=results_dir)
        plots_path = str(plots_dir)
        for idx, crack in enumerate(cracks_list):
            label = frame_labels[idx] if frame_labels is not None else f"{idx:04d}"
            _save_crack_plot(
                stack[idx],
                crack,
                plots_dir / f"cracks_{label}.png",
                background_flag=background,
                color=color_cracks,
                comparison=comparison,
            )

    if save_cracks:
        crack_bundle_path = str(save_crack_bundle(specimen, ply, cracks_list, results_root=results_dir))

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


def _resolve_requested_plies(
    specimen: Specimen, plies: Sequence[Any]
) -> List[Ply]:
    """Turn a list of ply names and :class:`Ply` objects into plies, raising for unknown names."""
    resolved: List[Ply] = []
    missing: List[str] = []
    for entry in plies:
        if isinstance(entry, Ply):
            resolved.append(entry)
            continue
        ply = specimen.get_ply_by_name(str(entry))
        if ply is None:
            missing.append(str(entry))
        else:
            resolved.append(ply)
    if missing:
        raise ValueError(
            f"Requested plies not found in specimen '{specimen.name}': "
            f"{', '.join(missing)}"
        )
    return resolved


def _build_crack_analysis_payload(
    structured: Dict[str, Any],
    *,
    orientation_deg: float,
    ply: Ply,
    plies: List[Ply],
) -> Dict[str, Any]:
    """Turn a :func:`crack_eval` result into one :func:`crack_analysis` entry."""
    return {
        "orientation_deg": orientation_deg,
        "ply": ply,
        "plies": plies,
        **{key: structured[key] for key in ("cracks", "densities", "thresholds", "metrics", "paths", "params")},
    }


def crack_analysis(
    specimen: Specimen,
    *,
    orientations: Optional[Sequence[float]] = None,
    plies: Optional[Sequence[Any]] = None,
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
    """Detect cracks once per ply orientation, or once per requested ply.

    By default plies are grouped by orientation (within ``tolerance``)
    and CrackDect runs once per group, using the group's first ply. A
    warning is logged when a group has more than one ply.

    ``orientations`` limits detection to those angles, e.g.
    ``[0.0, 90.0]``. ``plies`` instead runs detection on each given ply
    (name or :class:`Ply`), which lets you pick a specific ply when
    several share an orientation. The two can't be combined.

    The other arguments are passed to :func:`crack_eval`.

    Returns
    -------
    dict[str, dict]
        Keyed by orientation (e.g. ``"0"``, ``"90"``) or by ply name when
        ``plies`` is given. Each entry holds ``cracks``, ``densities``,
        ``thresholds``, ``metrics``, ``paths``, ``params``,
        ``orientation_deg``, ``ply`` and ``plies``.
    """
    if orientations is not None and plies is not None:
        raise ValueError("Pass only one of `orientations` or `plies`, not both.")

    results: Dict[str, Dict[str, Any]] = {}
    common_eval_kwargs: Dict[str, Any] = dict(
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

    if plies is not None:
        requested_plies = _resolve_requested_plies(specimen, plies)
        for ply in requested_plies:
            structured = crack_eval(specimen, ply=ply, **common_eval_kwargs)
            results[ply.name] = _build_crack_analysis_payload(
                structured,
                orientation_deg=float(ply.orientation_deg),
                ply=ply,
                plies=[ply],
            )
        return results

    groups = _group_plies_by_orientation(specimen, tolerance=tolerance)
    target_orientations = list(orientations) if orientations is not None else None

    def matches_target(angle: float) -> bool:
        return target_orientations is None or any(abs(angle - target) <= tolerance for target in target_orientations)

    for angle, group_plies in groups:
        if not matches_target(angle):
            continue
        primary = group_plies[0]
        if len(group_plies) > 1:
            duplicate_names = ", ".join(ply.name for ply in group_plies[1:])
            logger.warning(
                "Multiple plies found at %.3f°; using '%s' and merging %d duplicates (%s). "
                "Pass `plies=[...]` to select specific plies instead.",
                angle,
                primary.name,
                len(group_plies) - 1,
                duplicate_names,
            )
            primary_settings = (primary.avg_crack_width_px, primary.min_crack_length_px)
            for candidate in group_plies[1:]:
                if (candidate.avg_crack_width_px, candidate.min_crack_length_px) != primary_settings:
                    logger.warning(
                        "Duplicate ply '%s' at %.3f° has different crack settings; "
                        "using '%s' defaults.",
                        candidate.name,
                        angle,
                        primary.name,
                    )
                    break

        structured = crack_eval(specimen, ply=primary, **common_eval_kwargs)
        results[_orientation_label(angle)] = _build_crack_analysis_payload(
            structured,
            orientation_deg=angle,
            ply=primary,
            plies=group_plies,
        )

    if target_orientations is not None:
        missing_orientations = [
            target
            for target in target_orientations
            if not any(abs(angle - target) <= tolerance for angle, _ in groups)
        ]
        if missing_orientations:
            logger.warning(
                "Requested orientations not found in specimen '%s': %s",
                specimen.name,
                ", ".join(str(value) for value in missing_orientations),
            )

    return results


def plot_cracks(
    image: np.ndarray,
    cracks: Sequence[np.ndarray],
    *,
    linewidth: float = 1.0,
    color: str = "red",
    background_flag: bool = False,
    comparison: bool = False,
):
    """Plot crack segments, optionally over the image.

    Parameters
    ----------
    image:
        Frame the cracks were detected on.
    cracks:
        Segments of shape ``(n, 2, 2)`` with ``(y, x)`` end points.
    linewidth, color:
        Line style of the cracks.
    background_flag:
        Draw ``image`` behind the cracks.
    comparison:
        Show the image twice side by side.

    Returns
    -------
    dict[str, Any]
        ``{"figure": Figure, "axes": Axes}``.
    """
    fig, ax = plt.subplots()
    frame = image
    if comparison:
        frame = np.hstack((image, image))
    if background_flag:
        vmin, vmax = (0, np.iinfo(frame.dtype).max) if np.issubdtype(frame.dtype, np.integer) else (None, None)
        ax.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax)
    draw_crack_segments(ax, cracks, color=color, linewidth=linewidth)
    ax.set_ylim(frame.shape[0], 0)
    ax.set_xlim(0, frame.shape[1])
    ax.set_aspect("equal")
    ax.tick_params(axis="both", which="both", length=0)
    ax.grid(False)
    return {"figure": fig, "axes": ax}


def _save_crack_plot(image: np.ndarray, cracks: Sequence[np.ndarray], save_path, **plot_kwargs: Any) -> None:
    plot_result = plot_cracks(image, cracks, **plot_kwargs)
    fig, ax = plot_result["figure"], plot_result["axes"]
    ax.set_xlabel("x [Px]")
    ax.set_ylabel("y [Px]")
    fig.savefig(str(save_path))
    plt.close(fig)


def _group_plies_by_orientation(
    specimen: Specimen,
    *,
    tolerance: float = 1e-3,
) -> List[Tuple[float, List[Ply]]]:
    """Group the plies whose angle is within ``tolerance`` of the first ply of a group."""
    groups: List[Tuple[float, List[Ply]]] = []
    for ply in specimen.plies:
        for angle, plies in groups:
            if abs(ply.orientation_deg - angle) <= tolerance:
                plies.append(ply)
                break
        else:
            groups.append((float(ply.orientation_deg), [ply]))
    return groups


def _orientation_label(angle: float) -> str:
    """Key for an angle in the results dict: ``"90"`` for whole degrees, else e.g. ``"22.5"``."""
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
    """Sort cracks left to right by their smallest column.

    With ``delimiter``, vertical segments at the left and right image
    borders are added, so spacing is also measured to the edges.
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
    """Merge cracks that are close to each other along the columns.

    With ``group_within_crack_width``, cracks whose mean columns are
    within twice the crack width become one vertical segment. Then
    neighbors whose end points are within ``threshold`` pixels are
    joined, as a vertical segment if ``generate_vertical_crack``.
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
    """Drop cracks shorter than ``length_threshold``."""
    return [crack for crack in crack_list if crack_length(crack) >= length_threshold]


def compute_crack_spacing(crack_list: List[np.ndarray]) -> Dict[str, Any]:
    """Column distance between the midpoints of neighboring cracks.

    Returns
    -------
    dict[str, Any]
        ``{"spacing": list[float], "avg_spacing": float, "std_spacing": float}``.
    """
    if not crack_list:
        return {"spacing": [], "avg_spacing": 0.0, "std_spacing": 0.0}

    mid_cols = sorted(col for _, col in map(crack_mid_point, crack_list) if col is not None)
    spacings = [right - left for left, right in zip(mid_cols, mid_cols[1:])]
    avg = float(np.mean(spacings)) if spacings else 0.0
    std = float(np.std(spacings)) if spacings else 0.0
    return {"spacing": spacings, "avg_spacing": avg, "std_spacing": std}


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
) -> Dict[str, Any]:
    """Filter the cracks of every frame and compute the crack spacing.

    Each frame's cracks are sorted, filtered by length, optionally
    grouped, and their spacing computed. With ``remove_outliers`` the
    mean and standard deviation ignore spacings outside 1.5 IQR.
    Spacings are converted to millimeters with ``specimen.scale_px_mm``.

    Returns
    -------
    dict[str, Any]
        ``{"records": [...], "filtered_frames": [...]}``: one spacing
        record and one filtered crack list per frame.
    """
    if not cracks:
        return {"records": [], "filtered_frames": []}

    reference_paths = specimen.path_middle_list or specimen.path_full_list
    if not reference_paths:
        raise ValueError("Specimen has no image paths to reference.")
    image = imread(reference_paths[0])
    height, width = image.shape[:2]
    records: List[Dict[str, Any]] = []
    filtered_frames: List[List[np.ndarray]] = []

    plots_dir = specimen.results_dir("plots", "filtered_cracks", results_root=results_dir) if export_images else None

    for idx, frame_cracks in enumerate(cracks):
        ordered = order_cracks(frame_cracks, delimiter=True, image_height=height, image_width=width)
        filtered = crack_filter(list(ordered), length_threshold=crack_length_th)
        grouped = np.asarray(filtered)
        if grouping:
            grouped = crack_grouping(
                grouped,
                threshold=avg_crack_grouping_th_px,
                generate_vertical_crack=True,
                group_within_crack_width=True,
                avg_crack_width_px=specimen.avg_crack_width_px,
            )

        spacing_result = compute_crack_spacing(list(grouped))
        spacing = spacing_result["spacing"]
        avg_spacing = spacing_result["avg_spacing"]
        std_spacing = spacing_result["std_spacing"]

        if remove_outliers and spacing:
            spacing_array = np.asarray(spacing)
            q1, q3 = np.percentile(spacing_array, [25, 75])
            iqr = q3 - q1
            inliers = spacing_array[(spacing_array >= q1 - 1.5 * iqr) & (spacing_array <= q3 + 1.5 * iqr)]
            if inliers.size:
                avg_spacing = float(np.mean(inliers))
                std_spacing = float(np.std(inliers))

        records.append(
            {
                "Picture": idx,
                "Avg_spacing": avg_spacing / specimen.scale_px_mm,
                "Std_spacing": std_spacing / specimen.scale_px_mm,
            }
        )
        filtered_frames.append(filtered)

        if plots_dir is not None:
            _save_crack_plot(
                image, filtered, plots_dir / f"filtered_{idx:04d}.png", color="black", background_flag=background
            )

    return {"records": records, "filtered_frames": filtered_frames}


def pixels_to_length(input_data: List[Any], *, scale_px_mm: float) -> Dict[str, Any]:
    """Divide pixel values by ``scale_px_mm``.

    ``input_data`` is either a list of numbers or a list of spacing
    records with ``Avg_spacing`` and ``Std_spacing`` keys.

    Returns
    -------
    dict[str, Any]
        ``{"values": [...]}`` in the same form as ``input_data``.
    """
    if all(isinstance(value, (int, float)) for value in input_data):
        return {"values": [value / scale_px_mm for value in input_data]}
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
    return {"values": scaled}


def _build_crack_metrics_table(
    cracks: List[Sequence[np.ndarray]],
    densities: List[float],
    thresholds: List[float],
) -> pd.DataFrame:
    """Table with one row per frame: crack count, crack density ``rho`` and its threshold."""
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
    """Stack to detect cracks on: the middle region by default, else the full frames."""
    stacks = {
        "middle": getattr(specimen, "image_stack_middle", None),
        "full": getattr(specimen, "image_stack_full", None),
    }
    if use_full_stack is None:
        if stacks["middle"] is None and stacks["full"] is None:
            raise ValueError("Specimen has no image stack for region 'path_full' or 'path_middle'.")
        use_full_stack = stacks["middle"] is None

    if specimen.path_middle is None and (specimen.path_upper_border or specimen.path_lower_border):
        warnings.warn(
            "Upper/lower stacks were provided without a middle stack. Attempting detection in the whole picture stack.",
            RuntimeWarning,
            stacklevel=3,
        )

    wanted, fallback = ("full", "middle") if use_full_stack else ("middle", "full")
    if stacks[wanted] is not None:
        return stacks[wanted]
    if stacks[fallback] is not None:
        warnings.warn(
            f"{wanted.capitalize()} stack requested but no {wanted} stack is available; using {fallback} stack instead.",
            RuntimeWarning,
            stacklevel=3,
        )
        return stacks[fallback]
    raise ValueError(f"Specimen has no image stack for region 'path_{wanted}' or 'path_{fallback}'.")


def _theta_from_ply(ply: Ply) -> int:
    """CrackDect detection angle for a ply: ``(90 - angle) % 180``."""
    return int(round((90.0 - float(ply.orientation_deg)) % 180.0))


__all__ = [
    "crack_eval",
    "crack_analysis",
    "plot_cracks",
    "order_cracks",
    "crack_grouping",
    "crack_filter",
    "compute_crack_spacing",
    "crack_filtering_postprocessing",
    "pixels_to_length",
]
