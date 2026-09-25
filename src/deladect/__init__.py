"""DelaDect: crack and delamination detection in translucent composites."""

import os
from importlib.metadata import PackageNotFoundError, version


def _configure_numba_defaults() -> None:
    """Turn off numba JIT unless ``DELADECT_ENABLE_NUMBA_JIT`` is set.

    crackdect's jitted helpers (``find_cracks``, ``_find_crack_end``) no
    longer compile with recent numba, so they run as plain Python instead.
    """
    if os.environ.get("NUMBA_DISABLE_JIT") is not None:
        return
    if os.environ.get("DELADECT_ENABLE_NUMBA_JIT", "").strip().lower() in {"1", "true", "yes", "on"}:
        return
    os.environ["NUMBA_DISABLE_JIT"] = "1"


_configure_numba_defaults()

try:
    __version__ = version("deladect")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .specimen import Specimen  # noqa: E402
from .detection import DelaminationDetector, crack_analysis, plot_cracks  # noqa: E402

__all__ = [
    "__version__",
    "Specimen",
    "DelaminationDetector",
    "crack_analysis",
    "plot_cracks",
]
