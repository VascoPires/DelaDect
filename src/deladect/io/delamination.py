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


def _store_masks_field(
    interface: Interface,
    *,
    masks: Optional[Dict[str, np.ndarray]],
    path: Optional[Path],
    metadata_key: str,
    label: str,
) -> None:
    """Save one masks/path field and record it in ``interface.metadata``.

    Shared by ``store_interface_masks`` and
    ``store_interface_delamination_results``: if ``masks`` are given, persist
    them to ``path`` (required in that case) and record the saved path;
    otherwise, if a bare ``path`` was given, just record it as-is.
    """
    if masks is not None:
        if path is None:
            raise ValueError(f"{label}_path must be provided when {label}_masks are supplied.")
        saved = save_npz_bundle(masks, path)
        interface.metadata[metadata_key] = str(saved)
    elif path is not None:
        interface.metadata[metadata_key] = str(Path(path))


def store_interface_masks(
    interface: Interface,
    *,
    primary_masks: Optional[Dict[str, np.ndarray]] = None,
    primary_path: Optional[Path] = None,
    secondary_masks: Optional[Dict[str, np.ndarray]] = None,
    secondary_path: Optional[Path] = None,
) -> None:
    """Persist interface primary/secondary masks and update metadata paths."""
    _store_masks_field(
        interface, masks=primary_masks, path=primary_path,
        metadata_key=INTERFACE_PRIMARY_MASKS_KEY, label="primary",
    )
    _store_masks_field(
        interface, masks=secondary_masks, path=secondary_path,
        metadata_key=INTERFACE_SECONDARY_MASKS_KEY, label="secondary",
    )


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
    _store_masks_field(
        interface, masks=diffuse_raw_masks, path=diffuse_raw_path,
        metadata_key=INTERFACE_DIFFUSE_RAW_MASKS_KEY, label="diffuse_raw",
    )
    _store_masks_field(
        interface, masks=diffuse_masks, path=diffuse_path,
        metadata_key=INTERFACE_DIFFUSE_MASKS_KEY, label="diffuse",
    )
    _store_masks_field(
        interface, masks=combined_masks, path=combined_path,
        metadata_key=INTERFACE_COMBINED_MASKS_KEY, label="combined",
    )

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
