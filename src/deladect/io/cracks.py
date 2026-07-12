"""Crack-related IO helpers with ply-scoped defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from deladect.specimen import Ply, Specimen
from .bundles import load_npz_bundle, save_npz_bundle

PLY_CRACK_RESULTS_KEY = "crack_results_path"


def _sanitize_name(name: str) -> str:
    """Normalize a ply name for filesystem-safe folder names."""
    # Names are user-facing labels and may contain whitespace, slashes, or
    # other path syntax.  Keep a readable identifier without allowing one
    # label to introduce another directory level.
    cleaned = "_".join(name.strip().split())
    return cleaned.replace("/", "_").replace("\\", "_").replace(":", "_") or "ply"


def _ply_base_dir(specimen: Specimen, ply: Ply, results_root: Optional[str]) -> Path:
    """Return the per-ply base directory used by crack exports."""
    return specimen.results_dir("cracks", f"ply_{_sanitize_name(ply.name)}", results_root=results_root)


def crack_results_dir(specimen: Specimen, ply: Ply, *, results_root: Optional[str] = None) -> Path:
    """Return the canonical crack-results directory for ``ply``."""
    return _ply_base_dir(specimen, ply, results_root)


def crack_results_subdir(
    specimen: Specimen,
    ply: Ply,
    name: str,
    *,
    results_root: Optional[str] = None,
) -> Path:
    """Return/create a standard crack subdirectory under a ply root."""
    return specimen.results_dir(
        "cracks",
        f"ply_{_sanitize_name(ply.name)}",
        name,
        results_root=results_root,
    )


def _resolve_folder(
    specimen: Specimen,
    ply: Ply,
    folder_name: Optional[str],
    *,
    default_subdir: str,
    results_root: Optional[str],
) -> Path:
    """Resolve an export folder from optional override/default settings."""
    subdir = folder_name or default_subdir
    return specimen.results_dir(
        "cracks",
        f"ply_{_sanitize_name(ply.name)}",
        subdir,
        results_root=results_root,
    )


def store_ply_crack_results(ply: Ply, data: Dict[str, np.ndarray], path: Path) -> Path:
    """Persist crack bundles for a ply and record the NPZ path in metadata."""
    saved = save_npz_bundle(data, path)
    ply.metadata[PLY_CRACK_RESULTS_KEY] = str(saved)
    return saved


def load_ply_crack_results(ply: Ply) -> Dict[str, np.ndarray]:
    """Reload crack data referenced in ``ply.metadata``."""
    path = ply.metadata.get(PLY_CRACK_RESULTS_KEY)
    if not path:
        raise ValueError(f"ply '{ply.name}' has no stored crack results.")
    return load_npz_bundle(Path(path))


def export_rho(
    specimen: Specimen,
    ply: Ply,
    *rho_lists: List[float],
    folder_name: Optional[str] = None,
    file_name: str = "rho_data.csv",
    rho_names: Optional[List[str]] = None,
    results_root: Optional[str] = None,
) -> Path:
    """Export rho sequences (and optional strain) to a ply-scoped metrics folder."""
    if not rho_lists or any(len(rho) == 0 for rho in rho_lists):
        raise ValueError("No rho data to export.")
    length = len(rho_lists[0])
    if not all(len(rho) == length for rho in rho_lists):
        raise ValueError("All rho lists must have the same length.")
    labels = rho_names or [f"rho_{idx+1}" for idx in range(len(rho_lists))]
    frame_ids = list(range(length))
    payload: Dict[str, Any] = {"frame_id": frame_ids}
    for label, rho in zip(labels, rho_lists):
        payload[label] = rho
    df = pd.DataFrame(payload)
    if specimen.experimental_data is not None and "strain_y" in specimen.experimental_data.columns:
        strain = specimen.experimental_data["strain_y"].reset_index(drop=True)
        df.insert(1, "strain_y", strain)
    target = _resolve_folder(
        specimen,
        ply,
        folder_name,
        default_subdir="metrics",
        results_root=results_root,
    )
    file_path = target / file_name
    df.to_csv(file_path, index=False)
    return file_path


def export_crack_spacing(
    specimen: Specimen,
    ply: Ply,
    processed_data: List[Dict[str, Any]],
    *,
    folder_name: Optional[str] = None,
    file_name: str = "crack_spacing.csv",
    results_root: Optional[str] = None,
) -> Path:
    """Export crack-spacing metrics to a ply-scoped metrics folder."""
    if not processed_data:
        raise ValueError("No data to export.")
    df = pd.DataFrame(processed_data)
    if specimen.experimental_data is not None and "strain_y" in specimen.experimental_data.columns:
        strain = specimen.experimental_data["strain_y"].reset_index(drop=True)
        df.insert(1, "strain_y", strain)
    target = _resolve_folder(
        specimen,
        ply,
        folder_name,
        default_subdir="metrics",
        results_root=results_root,
    )
    file_path = target / file_name
    df.to_csv(file_path, index=False)
    return file_path


def save_cracks(
    specimen: Specimen,
    ply: Ply,
    cracks: List[Sequence[Sequence[float]]],
    *,
    folder_name: Optional[str] = None,
    file_name: Optional[str] = None,
    results_root: Optional[str] = None,
) -> Path:
    """Persist cracks to NPZ and record the path in ply metadata."""
    target = _resolve_folder(
        specimen,
        ply,
        folder_name,
        default_subdir="data",
        results_root=results_root,
    )
    resolved = (file_name or f"{specimen.name}_{_sanitize_name(ply.name)}_cracks.npz").lstrip("_-")
    if Path(resolved).suffix == "":
        resolved = f"{resolved}.npz"
    file_path = target / resolved
    payload: Dict[str, Any] = {}
    for idx, crack in enumerate(cracks):
        payload[f"frame_{idx:04d}"] = np.asarray(crack, dtype=np.float32)
    return store_ply_crack_results(ply, payload, file_path)


def load_cracks(ply: Ply) -> List[np.ndarray]:
    """Load cracks from ply metadata and return ordered frame arrays."""
    bundle = load_ply_crack_results(ply)
    frame_keys = sorted(bundle.keys())
    return [np.asarray(bundle[key]) for key in frame_keys]


__all__ = [
    "crack_results_dir",
    "crack_results_subdir",
    "PLY_CRACK_RESULTS_KEY",
    "export_crack_spacing",
    "export_rho",
    "load_cracks",
    "load_ply_crack_results",
    "save_cracks",
    "store_ply_crack_results",
]
