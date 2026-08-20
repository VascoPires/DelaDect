"""Delamination detection workflows.

This package provides a class-based API centred on
:class:`~deladect.detection.delamination.core.DelaminationDetector`, with
edge and diffuse detection exposed as two peer sub-detectors:
:class:`~deladect.detection.delamination.edge.EdgeDetector`
(``detector.edge``) and
:class:`~deladect.detection.delamination.diffuse.DiffuseDetector`
(``detector.diffuse``). Shared infrastructure (preprocessing, caching,
combined arbitration) lives directly on ``DelaminationDetector``.

The implementation is intentionally stateful: frame-to-frame latching,
preprocess cache reuse, and debug exports are coordinated by detector
instances rather than stateless helper functions.
"""

from .core import DelaminationDetector
from .diffuse import DiffuseDetector
from .edge import EdgeDetector

# Keep the public import path stable on the classes themselves (repr, pickling,
# Sphinx autodoc) even though they're implemented in private submodules.
for _cls in (DelaminationDetector, EdgeDetector, DiffuseDetector):
    _cls.__module__ = __name__
del _cls

__all__ = ["DelaminationDetector", "EdgeDetector", "DiffuseDetector"]
