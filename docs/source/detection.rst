Detection API Reference
=======================

This page lists the main public classes and functions of
``deladect.detection``, including their arguments and default values.

- :class:`~deladect.specimen.Specimen` holds the images, plies and
  interfaces of a test specimen, and is passed to every detector.
- The delamination detectors find edge, diffuse, combined, and
  multi-interface delamination.
- The crack functions run CrackDect on each ply and plot the result.

.. currentmodule:: deladect.specimen

Specimen
--------

.. autosummary::
   :toctree: generated

   Specimen

.. currentmodule:: deladect.detection

Classes
-------

.. autosummary::
   :toctree: generated

   DelaminationDetector
   EdgeDetector
   DiffuseDetector

Functions
---------

.. autosummary::
   :toctree: generated

   crack_analysis
   crack_eval
   plot_cracks

Related documentation
---------------------

For more details on how to use the detection methods, see:

- :doc:`methodology` for the complete delamination workflow.
- :doc:`edge_delamination` and :doc:`diffuse_delamination` for each detection method.
- :doc:`examples/getting_started` for a complete analysis example.
