"""Matplotlib overlay rendering: edge/diffuse/combined overlays and debug panels.

Depends on :mod:`._common` and on the shared crack-drawing primitive in
:mod:`deladect.utils`.
"""


from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from deladect.specimen import (
    DEFAULT_PRIMARY_DELAMINATION_COLOR,
    Interface,
    Specimen,
)
from deladect.utils import draw_crack_segments

logger = logging.getLogger(__name__)

from ._common import (
    CRACK_OVERLAY_RGBA,
    MULTI_INTERFACE_DEFAULT_COLORS,
    _normalize_rgba,
    _rgba_close,
)


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
    draw_crack_segments(ax, cracks, color=color, linewidth=linewidth)


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
