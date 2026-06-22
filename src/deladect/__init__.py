"""Top-level package for the rebuilt DelaDect stack."""

import os
from importlib.metadata import PackageNotFoundError, version


def _configure_numba_defaults() -> None:
    """Default to non-JIT crackdect execution unless the user opted in."""
    if os.environ.get("NUMBA_DISABLE_JIT") is not None:
        return
    if os.environ.get("DELADECT_ENABLE_NUMBA_JIT", "").strip().lower() in {"1", "true", "yes", "on"}:
        return
    os.environ["NUMBA_DISABLE_JIT"] = "1"


_configure_numba_defaults()

try:  # pragma: no cover - metadata not available during editable installs
    __version__ = version("deladect")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
