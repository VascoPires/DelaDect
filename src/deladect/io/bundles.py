"""Read and write dictionaries of arrays as compressed ``.npz`` files."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


def with_extension(path: Path, extension: str) -> Path:
    """Add ``extension`` (e.g. ``".npz"``) to ``path`` unless it already has it."""
    path = Path(path)
    if path.suffix.lower() != extension:
        path = path.with_suffix(path.suffix + extension)
    return path


def save_npz_bundle(data: Dict[str, np.ndarray], path: Path) -> Path:
    """Save ``data`` to a compressed ``.npz`` file and return the path written."""
    if not data:
        raise ValueError("Refusing to store an empty data bundle.")
    resolved = with_extension(path, ".npz")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(resolved, **data)
    return resolved


def load_npz_bundle(path: Path) -> Dict[str, np.ndarray]:
    """Load an ``.npz`` file into a plain dict."""
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    payload = np.load(resolved, allow_pickle=False)
    return {key: payload[key] for key in payload.files}


__all__ = ["load_npz_bundle", "save_npz_bundle"]
