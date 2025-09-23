Utility Helpers
===============

Overview
--------
The :mod:`deladect.utils` module groups together the small helpers that keep the rest of
DelaDect tidy. Geometry helpers give quick access to per-crack measurements, while the
filesystem helpers make it easy to organise output folders and save intermediate results.

Crack geometry helpers
----------------------
- :func:`~deladect.utils.crack_mid_point` returns the midpoint of a crack segment and is
  handy when computing spacing between cracks.
- :func:`~deladect.utils.crack_length` calculates the segment length in pixels so you can
  filter out short artefacts.
- :func:`~deladect.utils.crack_px_mm` converts crack coordinates from pixels to millimetres
  using the calibration factor stored on :class:`~deladect.detection.Specimen`.

Folder and serialisation helpers
--------------------------------
``DataProcessor`` is the go-to utility for creating folder structures and persisting crack
catalogues. The snippet below mirrors the patterns used in the automated tests:

.. code-block:: python

    from deladect.utils import DataProcessor

    # Create a reporting structure
    results_dir = DataProcessor.generate_folder("reports", "sample-1")

    # Persist detected cracks and reload them later
    DataProcessor.save_cracks_to_file(cracks, results_dir, "detected_cracks.pkl")
    reloaded = DataProcessor.load_cracks(results_dir + "/detected_cracks.pkl")

    # Join crack catalogues from different angles
    combined = DataProcessor.join_cracks(cracks_theta_plus, cracks_theta_minus)

Working with ``Specimen``
-------------------------
Most utilities surface through ``Specimen`` methods. For example,
:meth:`deladect.detection.Specimen.save_cracks` delegates to
:func:`~deladect.utils.DataProcessor.save_cracks_to_file`, and
:meth:`deladect.detection.Specimen.pixels_to_length` builds on
:func:`~deladect.utils.crack_px_mm`. Keeping these helpers in ``utils`` means you can reuse
them directly in custom scripts or notebooks.

Reference
---------
.. automodule:: deladect.utils
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
