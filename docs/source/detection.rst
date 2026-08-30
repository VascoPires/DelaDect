Detection API Reference
=======================

The public detection API is exposed through :mod:`deladect.detection`. In
this page all the accessible classes, methods, functions are accessible along
with all the default values of each function.

The package has two public areas:

- :mod:`deladect.detection.crack_detection` provides crack-analysis functions.
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

The module pages provide automatically generated class and function
inventories, matching the organization of the source package.

.. autosummary::
   :toctree: generated

   crack_detection
   crack_tracking
   delamination

Related documentation
---------------------

- :doc:`methodology` explains the complete delamination workflow.
- :doc:`edge_delamination` and :doc:`diffuse_delamination` explain the two
  detection modes.
- :doc:`examples/getting_started` provides a complete runnable analysis.
