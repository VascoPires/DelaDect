DelaDect Detection API
======================

Overview
--------
DelaDect's detection layer turns aligned image sequences into crack-density curves,
spacing statistics, and ready-to-share figures. The
:class:`~deladect.detection.Specimen` class keeps specimen metadata, image stacks, and
analysis helpers together so you can run complete evaluations without wiring everything by
hand.

Key capabilities
----------------
- Flexible image stack backends (in-memory or SQL) that switch automatically based on your
  ``stack_backend`` choice and memory budget.
- Convenience wrappers for common laminate scenarios (standard, cross-ply, and plus/minus
  layups) with optional caching and post-processing steps.
- Visualisation helpers that overlay detected cracks on top of the original or processed
  frames for quick sanity checks.
- Export tools for rho/theta series, crack spacing summaries, and pickled crack data for
  follow-up analysis.

Typical workflow
----------------
The snippet below mirrors the layout of ``tests/test_DelaDect_crack_detection.py`` and shows
how to run a minimal crack-detection pass on the bundled sample dataset.

.. code-block:: python

    from pathlib import Path
    from deladect.detection import Specimen

    data_root = Path("example_images") / "sample-1"
    specimen = Specimen(
        name="sample-1",
        dimensions={"width": 20.13, "thickness": 2.27},
        scale_px_mm=41.033,
        path_cut=data_root / "cut",
        path_upper_border=data_root / "upper",
        path_lower_border=data_root / "lower",
        path_middle=data_root / "middle",
        sorting_key="_sc",
        image_types=["png"],
    )

    cracks, rho, theta = specimen.crack_eval(
        theta_fd=0,
        background=True,
        export_images=True,
        save_cracks=True,
        output_dir="reports/sample-1",
    )

    specimen.upload_experimental_data(data_root / "experimental_data.csv")
    rho_mm = specimen.pixels_to_length(rho)
    specimen.export_rho(rho_mm, folder_name="reports/sample-1")

Advanced scenarios
------------------
``Specimen`` includes helpers for multi-angle laminates and post-processing:

- :meth:`~deladect.detection.Specimen.crack_eval_crossply` detects and overlays cracks at
  0-degree and 90-degree for cross-ply configurations while optionally caching intermediate
  results.
- :meth:`~deladect.detection.Specimen.crack_eval_plus_minus` handles plus/minus laminates and
  can add an extra transverse layer for post-processing when required.
- :meth:`~deladect.detection.Specimen.crack_filtering_postprocessing` combines ordering,
  grouping, filtering, and crack-spacing estimation (with outlier rejection) so you can tidy
  the crack catalogue before exporting.

See also
--------
- :doc:`examples/getting_started` for a narrated first run through the sample dataset.
- :doc:`examples/crack_detection` for practical testing flows, plots, and reporting ideas.

API reference
-------------
The sections above highlight the most common routines. The full API is documented below for
quick lookup.

.. automodule:: deladect.detection
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
