"""Low-level NPZ bundle helpers used by IO modules."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


def save_npz_bundle(data: Dict[str, np.ndarray], path: Path) -> Path:
    """Write a non-empty key-to-array payload to compressed NPZ."""
    if not data:
        raise ValueError("Refusing to store an empty data bundle.")
    resolved = Path(path)
    if resolved.suffix.lower() != ".npz":
        resolved = (
            resolved.with_suffix(resolved.suffix + ".npz")
            if resolved.suffix
            else resolved.with_suffix(".npz")
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(resolved, **data)
    return resolved


def load_npz_bundle(path: Path) -> Dict[str, np.ndarray]:
    """Load a compressed NPZ bundle into a plain dictionary."""
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    payload = np.load(resolved, allow_pickle=False)
    return {key: payload[key] for key in payload.files}


__all__ = ["load_npz_bundle", "save_npz_bundle"]
