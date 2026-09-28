"""Save and load crack results. Each ply gets its own folder under ``cracks/``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from deladect.specimen import Ply, Specimen
from .bundles import load_npz_bundle, save_npz_bundle
from .layout import crack_file_name, ply_dir as _ply_dir

PLY_CRACK_RESULTS_KEY = "crack_results_path"


def crack_results_dir(specimen: Specimen, ply: Ply, *, results_root: Optional[str] = None) -> Path:
    """Return (and create) the crack results folder of ``ply``."""
    return _ply_dir(specimen, ply, results_root=results_root)


def crack_results_subdir(
    specimen: Specimen,
    ply: Ply,
    name: str,
    *,
    results_root: Optional[str] = None,
) -> Path:
    """Return (and create) the subfolder ``name`` in the crack results folder of ``ply``."""
    return _ply_dir(specimen, ply, name, results_root=results_root)


def store_ply_crack_results(ply: Ply, data: Dict[str, np.ndarray], path: Path) -> Path:
    """Save crack arrays to ``path`` and record the file in ``ply.metadata``."""
    saved = save_npz_bundle(data, path)
    ply.metadata[PLY_CRACK_RESULTS_KEY] = str(saved)
    return saved


def load_ply_crack_results(ply: Ply) -> Dict[str, np.ndarray]:
    """Load the crack arrays recorded in ``ply.metadata``."""
    path = ply.metadata.get(PLY_CRACK_RESULTS_KEY)
    if not path:
        raise ValueError(f"ply '{ply.name}' has no stored crack results.")
    return load_npz_bundle(Path(path))


def _write_ply_metrics(
    specimen: Specimen,
    ply: Ply,
    df: pd.DataFrame,
    *,
    folder_name: Optional[str],
    file_name: str,
    results_root: Optional[str],
) -> Path:
    """Write ``df`` to the ply's metrics folder, adding the specimen's strain column if it has one."""
    if specimen.experimental_data is not None and "strain_y" in specimen.experimental_data.columns:
        df.insert(1, "strain_y", specimen.experimental_data["strain_y"].reset_index(drop=True))
    file_path = _ply_dir(specimen, ply, folder_name or "metrics", results_root=results_root) / file_name
    df.to_csv(file_path, index=False)
    return file_path


def export_rho(
    specimen: Specimen,
    ply: Ply,
    *rho_lists: List[float],
    folder_name: Optional[str] = None,
    file_name: str = "rho_data.csv",
    rho_names: Optional[List[str]] = None,
    results_root: Optional[str] = None,
) -> Path:
    """Write crack density sequences to CSV, one column per sequence.

    Columns are named ``rho_names`` (default ``rho_1``, ``rho_2``, ...).
    The specimen's ``strain_y`` is added when experimental data is loaded.
    """
    if not rho_lists or any(len(rho) == 0 for rho in rho_lists):
        raise ValueError("No rho data to export.")
    length = len(rho_lists[0])
    if not all(len(rho) == length for rho in rho_lists):
        raise ValueError("All rho lists must have the same length.")
    labels = rho_names or [f"rho_{idx + 1}" for idx in range(len(rho_lists))]
    payload: Dict[str, Any] = {"frame_id": list(range(length))}
    payload.update(zip(labels, rho_lists))
    return _write_ply_metrics(
        specimen, ply, pd.DataFrame(payload),
        folder_name=folder_name, file_name=file_name, results_root=results_root,
    )


def export_crack_spacing(
    specimen: Specimen,
    ply: Ply,
    processed_data: List[Dict[str, Any]],
    *,
    folder_name: Optional[str] = None,
    file_name: str = "crack_spacing.csv",
    results_root: Optional[str] = None,
) -> Path:
    """Write per-frame crack spacing records to CSV, with ``strain_y`` if available."""
    if not processed_data:
        raise ValueError("No data to export.")
    return _write_ply_metrics(
        specimen, ply, pd.DataFrame(processed_data),
        folder_name=folder_name, file_name=file_name, results_root=results_root,
    )


def save_cracks(
    specimen: Specimen,
    ply: Ply,
    cracks: List[Sequence[Sequence[float]]],
    *,
    folder_name: Optional[str] = None,
    file_name: Optional[str] = None,
    results_root: Optional[str] = None,
) -> Path:
    """Save one crack array per frame to ``.npz`` and record the file in ``ply.metadata``."""
    target = _ply_dir(specimen, ply, folder_name or "data", results_root=results_root)
    resolved = (file_name or crack_file_name(specimen, ply)).lstrip("_-")
    if Path(resolved).suffix == "":
        resolved = f"{resolved}.npz"
    payload = {f"frame_{idx:04d}": np.asarray(crack, dtype=np.float32) for idx, crack in enumerate(cracks)}
    return store_ply_crack_results(ply, payload, target / resolved)


def load_cracks(ply: Ply) -> List[np.ndarray]:
    """Load the cracks saved by :func:`save_cracks`, one array per frame."""
    bundle = load_ply_crack_results(ply)
    return [np.asarray(bundle[key]) for key in sorted(bundle)]


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
