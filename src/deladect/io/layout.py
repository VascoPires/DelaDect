"""Folder and file names of the results, under ``<results_root>/<specimen name>``.

Detection code writes, and :meth:`~deladect.detection.DelaminationDetector.save_delamination_overlay`
reads back, through these functions, so each name is defined once::

    cracks/ply_<ply>/data/<specimen>_<ply>_cracks.npz
    cracks/ply_<ply>/plots/, cracks/ply_<ply>/metrics/
    <dirname>/edge/overlays/edge_overlay_0003.png
    <dirname>/diffuse/overlays/diffuse_overlay_0003.png
    <dirname>/both/overlays/combined_overlay_0003.png
    <dirname>/both/masks/edge_raw.npz, edge_exclusion.npz, diffuse_raw.npz, diffuse_final.npz, combined.npz
    <dirname>/both/metrics/frame_metrics.csv
    <dirname>/total/overlays/total_overlay_0003.png
    <dirname>/edge_multi/masks/<interface>_inclusive.npz, <interface>_exclusive.npz
    <dirname>/edge_multi/overlays/edge_multi_overlay_0003.png

``<dirname>`` is the ``overlay_dirname`` argument (default
``"delamination"``). Ply and interface names are made safe with
:func:`~deladect.specimen.sanitize_path_token`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

from deladect.specimen import Ply, Specimen, sanitize_path_token

# overlay kind -> (subfolder, file prefix)
OVERLAYS: Dict[str, tuple] = {
    "edge": ("edge", "edge_overlay"),
    "diffuse": ("diffuse", "diffuse_overlay"),
    "both": ("both", "combined_overlay"),
    "total_dela": ("total", "total_overlay"),
    "edge_multi": ("edge_multi", "edge_multi_overlay"),
}

# mask name -> file saved by detect_both_delaminations
COMBINED_MASK_FILES: Dict[str, str] = {
    "edge_raw": "edge_raw.npz",
    "edge_exclusion": "edge_exclusion.npz",
    "diffuse_raw": "diffuse_raw.npz",
    "diffuse": "diffuse_final.npz",
    "combined": "combined.npz",
}


def overlay_dir(specimen: Specimen, dirname: str, kind: str) -> Path:
    """Folder of the overlays of ``kind`` (a key of :data:`OVERLAYS`), created if needed."""
    return specimen.results_dir(dirname, OVERLAYS[kind][0], "overlays")


def overlay_name(kind: str, frame_idx: int) -> str:
    """File name of the ``kind`` overlay of one frame, e.g. ``edge_overlay_0003.png``."""
    return f"{OVERLAYS[kind][1]}_{frame_idx:04d}.png"


def combined_masks_dir(specimen: Specimen, dirname: str, masks_dirname: str = "masks") -> Path:
    """Folder of the masks saved by ``detect_both_delaminations``."""
    return specimen.results_dir(dirname, "both", masks_dirname)


def combined_metrics_dir(specimen: Specimen, dirname: str) -> Path:
    """Folder of the per-frame metrics saved by ``detect_both_delaminations``."""
    return specimen.results_dir(dirname, "both", "metrics")


def multi_masks_dir(specimen: Specimen, dirname: str, masks_dirname: str = "masks") -> Path:
    """Folder of the masks saved by ``detect_edge_multi``."""
    return specimen.results_dir(dirname, "edge_multi", masks_dirname)


def ply_dir(specimen: Specimen, ply: Ply, *parts: str, results_root: Optional[str] = None) -> Path:
    """Crack results folder of ``ply`` (or a subfolder of it), created if needed."""
    return specimen.results_dir(
        "cracks", f"ply_{sanitize_path_token(ply.name, fallback='ply')}", *parts, results_root=results_root
    )


def crack_file_name(specimen: Specimen, ply: Ply) -> str:
    """Default file name of a ply's saved cracks."""
    return f"{specimen.name}_{sanitize_path_token(ply.name, fallback='ply')}_cracks.npz".lstrip("_-")


def unique_names(names: Iterable[str], *, fallback: str = "interface") -> List[str]:
    """Safe, unique file names: ``["a", "a"]`` gives ``["a", "a_2"]``.

    An empty name becomes ``<fallback>_<position>``.
    """
    seen: Dict[str, int] = {}
    result: List[str] = []
    for idx, name in enumerate(names):
        base = sanitize_path_token(str(name).strip() or f"{fallback}_{idx + 1}", fallback=f"{fallback}_{idx + 1}")
        count = seen.get(base, 0)
        seen[base] = count + 1
        result.append(base if count == 0 else f"{base}_{count + 1}")
    return result


__all__ = [
    "COMBINED_MASK_FILES",
    "OVERLAYS",
    "combined_masks_dir",
    "combined_metrics_dir",
    "crack_file_name",
    "multi_masks_dir",
    "overlay_dir",
    "overlay_name",
    "ply_dir",
    "unique_names",
]
