"""Overlay images and debug figures for delamination results."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from deladect.specimen import DEFAULT_PRIMARY_DELAMINATION_COLOR, Interface, Specimen
from deladect.utils import draw_crack_segments

RGBA = Tuple[float, float, float, float]

EDGE_OVERLAY_RGBA: RGBA = (1.0, 0.0, 0.0, 0.35)
DIFFUSE_OVERLAY_RGBA: RGBA = (0.0, 1.0, 0.0, 0.35)
CRACK_OVERLAY_RGBA: RGBA = (0.0, 0.0, 1.0, 0.95)
MULTI_INTERFACE_DEFAULT_COLORS: Tuple[RGBA, ...] = (
    (0.89, 0.10, 0.11, 0.35),
    (0.12, 0.47, 0.71, 0.35),
    (0.20, 0.63, 0.17, 0.35),
    (1.00, 0.50, 0.05, 0.35),
    (0.58, 0.40, 0.74, 0.35),
    (0.55, 0.34, 0.29, 0.35),
    (0.89, 0.47, 0.76, 0.35),
    (0.50, 0.50, 0.50, 0.35),
)


def _normalize_rgba(color: Sequence[float], *, default_alpha: float = 0.35) -> RGBA:
    """Clip a 3- or 4-value color to an RGBA tuple; anything else becomes red."""
    values = [float(v) for v in color]
    if len(values) == 3:
        values.append(float(default_alpha))
    if len(values) != 4:
        return (1.0, 0.0, 0.0, float(default_alpha))
    r, g, b, a = (float(np.clip(v, 0.0, 1.0)) for v in values)
    return (r, g, b, a)


def _rgba_close(left: Sequence[float], right: Sequence[float], *, tolerance: float = 1e-6) -> bool:
    if len(left) != 4 or len(right) != 4:
        return False
    return all(abs(float(a) - float(b)) <= tolerance for a, b in zip(left, right))


def _interface_legend_label(specimen: Specimen, interface: Interface) -> str:
    """Legend label ``"<interface>: <upper ply>/<lower ply>"``, with ``?`` for unknown plies."""

    def ply_name(idx: Optional[int]) -> Optional[str]:
        if idx is None or not 0 <= idx < len(specimen.plies):
            return None
        return specimen.plies[idx].name

    upper = ply_name(interface.upper_ply_index)
    lower = ply_name(interface.lower_ply_index)
    if not upper and not lower:
        return interface.name
    return f"{interface.name}: {upper or '?'}/{lower or '?'}"


def _resolve_multi_interface_colors(interfaces: Sequence[Interface]) -> List[RGBA]:
    """Colors for the interfaces; those left at the default color get one from a palette instead."""
    default_rgba = _normalize_rgba(DEFAULT_PRIMARY_DELAMINATION_COLOR, default_alpha=0.9)
    resolved: List[RGBA] = []
    for idx, interface in enumerate(interfaces):
        interface_color = _normalize_rgba(interface.delamination_color_rgba)
        if _rgba_close(interface_color, default_rgba):
            resolved.append(MULTI_INTERFACE_DEFAULT_COLORS[idx % len(MULTI_INTERFACE_DEFAULT_COLORS)])
        else:
            resolved.append(interface_color)
    return resolved


def _overlay_mask(ax, mask: np.ndarray, color: RGBA) -> None:
    """Draw ``mask`` on ``ax`` in ``color``."""
    overlay = np.zeros((*mask.shape, 4), dtype=float)
    overlay[mask] = color
    ax.imshow(overlay)


@contextmanager
def _overlay_figure(raw_frame: np.ndarray, save_path: Path) -> Iterator[Any]:
    """Figure with ``raw_frame`` in gray; saved to ``save_path`` and closed when the block ends."""
    fig, ax = plt.subplots()
    ax.imshow(raw_frame, cmap="gray")
    yield ax
    ax.axis("off")
    fig.savefig(save_path)
    plt.close(fig)


def _edge_limit_rows(mask_bool: np.ndarray, *, side: str) -> Optional[np.ndarray]:
    """Per column, the lowest (``side="bottom"``) or highest (``"top"``) masked row, NaN where empty."""
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
    mask_color: RGBA = EDGE_OVERLAY_RGBA,
) -> None:
    """Save an edge overlay as a filled ``"mask"``, the front as a ``"line"``, or ``"both"``."""
    with _overlay_figure(raw_frame, save_path) as ax:
        if view in {"mask", "both"}:
            _overlay_mask(ax, primary_mask, mask_color)

        if view in {"line", "both"}:
            split_row = primary_mask.shape[0] // 2
            upper = primary_mask[:split_row, :]
            lower = primary_mask[split_row:, :]
            rows_upper = _edge_limit_rows(upper, side="bottom")
            if rows_upper is not None:
                ax.plot(np.arange(upper.shape[1]), rows_upper, color="red", ls="-", linewidth=0.6)
            rows_lower = _edge_limit_rows(lower, side="top")
            if rows_lower is not None:
                ax.plot(np.arange(lower.shape[1]), split_row + rows_lower, color="red", ls="-", linewidth=0.6)


def _save_diffuse_overlay(
    raw_frame: np.ndarray,
    diffuse_mask: np.ndarray,
    save_path: Path,
    mask_color: RGBA = DIFFUSE_OVERLAY_RGBA,
    *,
    cracks: Optional[Sequence[np.ndarray]] = None,
    crack_color: RGBA = CRACK_OVERLAY_RGBA,
) -> None:
    with _overlay_figure(raw_frame, save_path) as ax:
        _overlay_mask(ax, diffuse_mask, mask_color)
        draw_crack_segments(ax, cracks, color=crack_color, linewidth=0.8)


def _save_combined_overlay(
    raw_frame: np.ndarray,
    *,
    edge_mask: np.ndarray,
    diffuse_mask: np.ndarray,
    save_path: Path,
    view: str = "union",
    edge_color: RGBA = EDGE_OVERLAY_RGBA,
    diffuse_color: RGBA = DIFFUSE_OVERLAY_RGBA,
    union_color: RGBA = EDGE_OVERLAY_RGBA,
    cracks: Optional[Sequence[np.ndarray]] = None,
    crack_color: RGBA = CRACK_OVERLAY_RGBA,
) -> None:
    """Save edge and diffuse masks in two colors (``"classified"``) or one (``"union"``)."""
    with _overlay_figure(raw_frame, save_path) as ax:
        if view == "classified":
            _overlay_mask(ax, diffuse_mask, diffuse_color)
            _overlay_mask(ax, edge_mask, edge_color)
        else:
            _overlay_mask(ax, edge_mask | diffuse_mask, union_color)
        draw_crack_segments(ax, cracks, color=crack_color, linewidth=0.8)


def _save_single_overlay(raw_frame: np.ndarray, mask: np.ndarray, save_path: Path, color: RGBA) -> None:
    with _overlay_figure(raw_frame, save_path) as ax:
        _overlay_mask(ax, mask, color)


def _save_multi_level_overlay(
    *,
    raw_frame: np.ndarray,
    level_masks: Sequence[np.ndarray],
    labels: Sequence[str],
    colors: Sequence[Sequence[float]],
    save_path: Path,
) -> None:
    """Save a multi-interface overlay with one color per interface and a legend."""
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(raw_frame, cmap="gray")

    handles: List[Patch] = []
    for idx, mask in enumerate(level_masks):
        label = labels[idx] if idx < len(labels) else f"level_{idx + 1}"
        rgba = _normalize_rgba(colors[idx] if idx < len(colors) else EDGE_OVERLAY_RGBA)
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


_PANEL_BG_ROLLING = "#ddeeff"  # rolling-median detection steps
_PANEL_BG_PROMOTION = "#fff3dd"  # attribution to the deeper interface
_PANEL_BG_REFERENCE = "#eeeeee"  # static primary, shown for context only


def _show_panel(ax, img, title: str, *, bg: str, vmin=None, vmax=None) -> None:
    ax.set_facecolor(bg)
    if img is None or (hasattr(img, "size") and img.size == 0):
        ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes, fontsize=8, color="#aaaaaa")
    else:
        arr = np.asarray(img)
        if arr.dtype == bool:
            arr = arr.astype(np.float32)
        ax.imshow(arr, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=8, pad=3, fontweight="bold")
    ax.axis("off")


def _draw_half_panels(
    axes,
    first_row: int,
    *,
    rolling_processed: Optional[np.ndarray],
    rolling_result: Dict[str, Any],
    rolling_latched: Optional[np.ndarray],
    connected_mask: Optional[np.ndarray],
    latched: Optional[np.ndarray],
    primary_latched: Optional[np.ndarray],
) -> None:
    """Draw the three debug rows (rolling-median input, attribution, reference) of one specimen half."""
    rolling_row, promotion_row, reference_row = axes[first_row], axes[first_row + 1], axes[first_row + 2]

    _show_panel(rolling_row[0], rolling_processed, "ROLLING processed\n(frame ÷ local median)",
                vmin=0, vmax=255, bg=_PANEL_BG_ROLLING)
    _show_panel(rolling_row[1], rolling_result.get("binary"), "ROLLING binary", bg=_PANEL_BG_ROLLING)
    _show_panel(rolling_row[2], rolling_result.get("binary_closed"), "ROLLING binary closed", bg=_PANEL_BG_ROLLING)
    _show_panel(rolling_row[3], rolling_result.get("mask"), "ROLLING mask", bg=_PANEL_BG_ROLLING)
    _show_panel(rolling_row[4], rolling_latched, "ROLLING parent latched\n(sim-gate reference)", bg=_PANEL_BG_ROLLING)

    _show_panel(promotion_row[0], None, "n/a", bg=_PANEL_BG_PROMOTION)
    _show_panel(promotion_row[1], None, "n/a", bg=_PANEL_BG_PROMOTION)
    _show_panel(promotion_row[2], connected_mask, "ROLLING in settled primary\n(delayed ref)", bg=_PANEL_BG_PROMOTION)
    _show_panel(promotion_row[3], None, "n/a", bg=_PANEL_BG_PROMOTION)
    _show_panel(promotion_row[4], latched, "RESULT secondary latched", bg=_PANEL_BG_PROMOTION)

    _show_panel(reference_row[0], primary_latched, "REF primary latched\n(static — context only)", bg=_PANEL_BG_REFERENCE)
    _show_panel(reference_row[1], None, "REF difference mask", bg=_PANEL_BG_REFERENCE)
    for ax in reference_row[2:]:
        ax.axis("off")
        ax.set_facecolor(_PANEL_BG_REFERENCE)


def _save_edge_multi_debug_panels(
    *,
    debug_dir: Path,
    frame_indices: List[int],
    processed_frames: List[np.ndarray],
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
    """Save one debug figure per frame for a deeper interface of ``detect_edge_multi``.

    Each half of the specimen gets three rows: the rolling-median
    detection, what was attributed to this interface, and the static
    primary mask for reference.
    """
    import matplotlib

    matplotlib.use("Agg")

    has_rolling = all(
        frames is not None
        for frames in (sec_processed_frames, sec_upper_results, sec_lower_results,
                       upper_rolling_frames, lower_rolling_frames)
    )

    def item(frames: Optional[Sequence[Any]], i: int, default: Any = None) -> Any:
        return frames[i] if has_rolling and i < len(frames) else default

    def as_bool(frames: Optional[Sequence[np.ndarray]], i: int, rolling_only: bool = False) -> Optional[np.ndarray]:
        if rolling_only and not has_rolling:
            return None
        return np.asarray(frames[i], dtype=bool) if i < len(frames) else None

    panels_dir = debug_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    for i, frame_idx in enumerate(frame_indices):
        if i >= len(upper_results):
            break

        proc = processed_frames[i] if i < len(processed_frames) else None
        split = split_rows[i] if i < len(split_rows) else (proc.shape[0] // 2 if proc is not None else 0)
        upper_diag_i = upper_diag[i] if i < len(upper_diag) else {}
        lower_diag_i = lower_diag[i] if i < len(lower_diag) else {}
        sec_proc = item(sec_processed_frames, i)

        fig, axes = plt.subplots(7, 5, figsize=(22, 18))
        fig.patch.set_facecolor("#f9f9f9")
        fig.suptitle(
            f"Frame {i}  (abs {frame_idx})  ·  level {level_idx + 1}\n"
            f"UPPER  connected_px={upper_diag_i.get('connected_pixels', 0)}\n"
            f"LOWER  connected_px={lower_diag_i.get('connected_pixels', 0)}",
            fontsize=10, y=0.998,
        )

        _draw_half_panels(
            axes, 0,
            rolling_processed=sec_proc[:split, :] if sec_proc is not None else None,
            rolling_result=item(sec_upper_results, i, {}),
            rolling_latched=as_bool(upper_rolling_frames, i, rolling_only=True),
            connected_mask=upper_diag_i.get("_masks", {}).get("connected_mask"),
            latched=as_bool(upper_latched, i),
            primary_latched=upper_results[i].get("primary_latched"),
        )

        for ax in axes[3]:
            ax.axis("off")
        axes[3][2].text(0.5, 0.5, "─────  LOWER HALF  ─────", ha="center", va="center",
                        transform=axes[3][2].transAxes, fontsize=10, color="#444444", fontweight="bold")

        _draw_half_panels(
            axes, 4,
            rolling_processed=np.flipud(sec_proc[split:, :]) if sec_proc is not None else None,
            rolling_result=item(sec_lower_results, i, {}),
            rolling_latched=as_bool(lower_rolling_frames, i, rolling_only=True),
            connected_mask=lower_diag_i.get("_masks", {}).get("connected_mask"),
            latched=as_bool(lower_latched, i),
            primary_latched=lower_results[i].get("primary_latched"),
        )

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(panels_dir / f"frame_{i:04d}_abs{frame_idx:04d}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)


# (result key, file suffix, vmax, cast to float)
_EDGE_DEBUG_LAYERS = (
    ("filtered_max", "filtered_max", 255, False),
    ("filtered_min", "filtered_min", 255, False),
    ("sharpened", "sharpened", 255, False),
    ("smoothed", "smoothed", 255, False),
    ("constant_scaled", "constant_scaled", 1, False),
    ("closed", "closed", 1, False),
    ("binary", "binary", 1, True),
    ("binary_closed", "binary_closed", 1, True),
    ("mask", "mask", 1, True),
    ("combined_upper", "combined", 1, True),
    ("primary_seed", "primary_seed", 1, True),
    ("primary_edge_snapshot", "primary_edge_snapshot", 1, True),
    ("primary_latched", "primary_latched_accum", 1, True),
)


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
    """Save every intermediate image of edge detection for one frame as a PNG."""

    def save_gray(name: str, image: np.ndarray, vmax: float) -> None:
        plt.imsave(frame_dir / name, image, cmap="gray", vmin=0, vmax=vmax)

    save_gray("raw.png", raw_frame, 255)
    save_gray("processed.png", processed, 255)

    for side, edge_slice, slice_name, result in (
        ("upper", upper_slice, "upper_edge_slice.png", upper_result),
        ("lower", lower_slice, "lower_edge_slice_processed.png", lower_result),
    ):
        save_gray(slice_name, edge_slice, 255)
        for key, suffix, vmax, as_float in _EDGE_DEBUG_LAYERS:
            image = result[key].astype(float) if as_float else result[key]
            save_gray(f"{side}_{suffix}.png", image, vmax)

    if lower_latched_unflipped is not None:
        save_gray("lower_primary_latched_accum_unflipped.png", lower_latched_unflipped.astype(float), 1)
    save_gray("full_primary_latched_accum.png", full_latched.astype(float), 1)
