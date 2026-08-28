Detection API Reference
=======================

The public detection API is exposed through :mod:`deladect.detection`. This
page is intentionally compact: signatures, summaries, classes, methods, and
functions below are generated from the source docstrings. For algorithm
explanations and complete workflows, see :doc:`methodology`; for tunable
settings, see :doc:`parameter_reference`.

The package has three public areas:

- :mod:`deladect.detection.crack_detection` provides crack-analysis functions.
- :mod:`deladect.detection.crack_tracking` provides crack descriptors and
  matching helpers used by diffuse delamination workflows.
- :mod:`deladect.detection.delamination` provides the class-based edge,
  diffuse, combined, and multi-interface delamination API.

.. currentmodule:: deladect.detection

Classes
-------

.. autosummary::
   :toctree: generated

   DelaminationDetector
   EdgeDetector
   DiffuseDetector
   CrackDetection
   CrackTrack

Functions
---------

.. autosummary::
   :toctree: generated

   crack_analysis
   crack_eval
   plot_cracks
   normalize_detections
   match_tracks

Modules
-------

The module pages provide automatically generated class/function inventories,
matching the organization of the source package.

.. autosummary::
   :toctree: generated

   crack_detection
   crack_tracking
   delamination

Coordinate convention
---------------------

Crack segments use ``[row, col]`` ordering, equivalent to ``[y, x]``:

- in memory: ``[[row0, col0], [row1, col1]]``
- in plots: ``col`` maps to the x-axis and ``row`` maps to the y-axis

Keep this convention when implementing custom spacing, grouping, or tracking
logic.

Related documentation
---------------------

- :doc:`methodology` explains the complete delamination workflow.
- :doc:`edge_delamination` and :doc:`diffuse_delamination` explain the two
  detection modes.
- :doc:`parameter_reference` lists every detector setting.
- :doc:`examples/getting_started` provides a complete runnable analysis.
