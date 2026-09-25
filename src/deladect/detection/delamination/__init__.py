"""Delamination detection.

:class:`DelaminationDetector` preprocesses the frames and combines two
sub-detectors, :class:`EdgeDetector` (``detector.edge``) and
:class:`DiffuseDetector` (``detector.diffuse``). Masks are latched from
frame to frame: once a pixel is delaminated it stays delaminated.
"""

from .core import DelaminationDetector
from .diffuse import DiffuseDetector
from .edge import EdgeDetector

# Report the public import path in repr, pickling and the API docs.
for _cls in (DelaminationDetector, EdgeDetector, DiffuseDetector):
    _cls.__module__ = __name__
del _cls

__all__ = ["DelaminationDetector", "EdgeDetector", "DiffuseDetector"]
