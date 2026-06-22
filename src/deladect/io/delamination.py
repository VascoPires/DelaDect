"""Delamination artefact storage and reload helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from deladect.specimen import Interface
from .bundles import load_npz_bundle, save_npz_bundle

INTERFACE_PRIMARY_MASKS_KEY = "primary_masks_path"
INTERFACE_SECONDARY_MASKS_KEY = "secondary_masks_path"
INTERFACE_DIFFUSE_RAW_MASKS_KEY = "diffuse_raw_masks_path"
INTERFACE_DIFFUSE_MASKS_KEY = "diffuse_masks_path"
INTERFACE_COMBINED_MASKS_KEY = "combined_masks_path"
INTERFACE_METRICS_KEY = "delamination_metrics_path"


def save_mask_bundle(data: Dict[str, np.ndarray], path: Path) -> Path:
    """Persist a bundle of masks to NPZ."""
    return save_npz_bundle(data, path)


def save_interface_metrics(metrics: pd.DataFrame, path: Path) -> Path:
    """Persist delamination metrics to CSV and return the resolved path."""
    target = Path(path)
    if target.suffix.lower() != ".csv":
        target = target.with_suffix(".csv") if target.suffix == "" else target.with_suffix(target.suffix + ".csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(target, index=False)
    return target


def store_interface_masks(
    interface: Interface,
    *,
    primary_masks: Optional[Dict[str, np.ndarray]] = None,
    primary_path: Optional[Path] = None,
    secondary_masks: Optional[Dict[str, np.ndarray]] = None,
    secondary_path: Optional[Path] = None,
) -> None:
    """Persist interface primary/secondary masks and update metadata paths."""
    if primary_masks is not None:
        if primary_path is None:
            raise ValueError("primary_path must be provided when primary_masks are supplied.")
        saved_primary = save_npz_bundle(primary_masks, primary_path)
        interface.metadata[INTERFACE_PRIMARY_MASKS_KEY] = str(saved_primary)
    elif primary_path is not None:
        interface.metadata[INTERFACE_PRIMARY_MASKS_KEY] = str(Path(primary_path))

    if secondary_masks is not None:
        if secondary_path is None:
            raise ValueError("secondary_path must be provided when secondary_masks are supplied.")
        saved_secondary = save_npz_bundle(secondary_masks, secondary_path)
        interface.metadata[INTERFACE_SECONDARY_MASKS_KEY] = str(saved_secondary)
    elif secondary_path is not None:
        interface.metadata[INTERFACE_SECONDARY_MASKS_KEY] = str(Path(secondary_path))


def store_interface_delamination_results(
    interface: Interface,
    *,
    diffuse_raw_masks: Optional[Dict[str, np.ndarray]] = None,
    diffuse_raw_path: Optional[Path] = None,
    diffuse_masks: Optional[Dict[str, np.ndarray]] = None,
    diffuse_path: Optional[Path] = None,
    combined_masks: Optional[Dict[str, np.ndarray]] = None,
    combined_path: Optional[Path] = None,
    metrics_path: Optional[Path] = None,
) -> None:
    """Persist diffuse/combined outputs and record paths in interface metadata."""
    if diffuse_raw_masks is not None:
        if diffuse_raw_path is None:
            raise ValueError("diffuse_raw_path must be provided when diffuse_raw_masks are supplied.")
        saved_raw = save_npz_bundle(diffuse_raw_masks, diffuse_raw_path)
        interface.metadata[INTERFACE_DIFFUSE_RAW_MASKS_KEY] = str(saved_raw)
    elif diffuse_raw_path is not None:
        interface.metadata[INTERFACE_DIFFUSE_RAW_MASKS_KEY] = str(Path(diffuse_raw_path))

    if diffuse_masks is not None:
        if diffuse_path is None:
            raise ValueError("diffuse_path must be provided when diffuse_masks are supplied.")
        saved_diffuse = save_npz_bundle(diffuse_masks, diffuse_path)
        interface.metadata[INTERFACE_DIFFUSE_MASKS_KEY] = str(saved_diffuse)
    elif diffuse_path is not None:
        interface.metadata[INTERFACE_DIFFUSE_MASKS_KEY] = str(Path(diffuse_path))

    if combined_masks is not None:
        if combined_path is None:
            raise ValueError("combined_path must be provided when combined_masks are supplied.")
        saved_combined = save_npz_bundle(combined_masks, combined_path)
        interface.metadata[INTERFACE_COMBINED_MASKS_KEY] = str(saved_combined)
    elif combined_path is not None:
        interface.metadata[INTERFACE_COMBINED_MASKS_KEY] = str(Path(combined_path))

    if metrics_path is not None:
        interface.metadata[INTERFACE_METRICS_KEY] = str(Path(metrics_path))


def load_interface_primary_masks(interface: Interface) -> Dict[str, np.ndarray]:
    """Load primary masks linked to ``interface`` metadata."""
    path = interface.metadata.get(INTERFACE_PRIMARY_MASKS_KEY)
    if not path:
        raise ValueError(f"interface '{interface.name}' has no stored primary masks.")
    return load_npz_bundle(Path(path))


def load_interface_secondary_masks(interface: Interface) -> Dict[str, np.ndarray]:
    """Load secondary masks linked to ``interface`` metadata."""
    path = interface.metadata.get(INTERFACE_SECONDARY_MASKS_KEY)
    if not path:
        raise ValueError(f"interface '{interface.name}' has no stored secondary masks.")
    return load_npz_bundle(Path(path))


def load_interface_diffuse_raw_masks(interface: Interface) -> Dict[str, np.ndarray]:
    """Load diffuse raw masks linked to ``interface`` metadata."""
    path = interface.metadata.get(INTERFACE_DIFFUSE_RAW_MASKS_KEY)
    if not path:
        raise ValueError(f"interface '{interface.name}' has no stored diffuse raw masks.")
    return load_npz_bundle(Path(path))


def load_interface_diffuse_masks(interface: Interface) -> Dict[str, np.ndarray]:
    """Load diffuse masks linked to ``interface`` metadata."""
    path = interface.metadata.get(INTERFACE_DIFFUSE_MASKS_KEY)
    if not path:
        raise ValueError(f"interface '{interface.name}' has no stored diffuse masks.")
    return load_npz_bundle(Path(path))


def load_interface_combined_masks(interface: Interface) -> Dict[str, np.ndarray]:
    """Load combined masks linked to ``interface`` metadata."""
    path = interface.metadata.get(INTERFACE_COMBINED_MASKS_KEY)
    if not path:
        raise ValueError(f"interface '{interface.name}' has no stored combined masks.")
    return load_npz_bundle(Path(path))


def load_interface_metrics(interface: Interface) -> pd.DataFrame:
    """Load metrics CSV linked to ``interface`` metadata."""
    path = interface.metadata.get(INTERFACE_METRICS_KEY)
    if not path:
        raise ValueError(f"interface '{interface.name}' has no stored metrics CSV.")
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return pd.read_csv(resolved)


__all__ = [
    "INTERFACE_COMBINED_MASKS_KEY",
    "INTERFACE_DIFFUSE_MASKS_KEY",
    "INTERFACE_DIFFUSE_RAW_MASKS_KEY",
    "INTERFACE_METRICS_KEY",
    "INTERFACE_PRIMARY_MASKS_KEY",
    "INTERFACE_SECONDARY_MASKS_KEY",
    "load_interface_combined_masks",
    "load_interface_diffuse_masks",
    "load_interface_diffuse_raw_masks",
    "load_interface_metrics",
    "load_interface_primary_masks",
    "load_interface_secondary_masks",
    "save_interface_metrics",
    "save_mask_bundle",
    "store_interface_delamination_results",
    "store_interface_masks",
]
