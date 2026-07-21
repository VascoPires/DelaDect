Results Storage
================

Overview
--------
Everything DelaDect saves lives under a per-specimen results root,
resolved by :meth:`~deladect.specimen.Specimen.results_dir`:

.. code-block:: text

   results/<specimen.name>/

(``results/`` by default, relative; override with ``specimen.results_root``.)
Each call to ``results_dir(*parts)`` joins and creates subdirectories under
that root; every part must be a single safe path component.

Config
------
.. code-block:: text

   results/<name>/config/<name>_config.json

Written by ``save_specimen``.

Cracks
------
.. code-block:: text

   results/<name>/cracks/ply_<sanitized_ply_name>/
     data/       <specimen>_<sanitized_ply_name>_cracks.npz
     metrics/    rho_data.csv, crack_spacing.csv
     plots/      cracks_<label>.png   (per-frame overlays)

The NPZ file stores one array per frame, keyed ``frame_0000``,
``frame_0001``, and so on. Its path is recorded back onto
``ply.metadata["crack_results_path"]`` so it can be found again without
re-deriving the folder name.

Delamination masks
-------------------
.. code-block:: text

   results/<name>/<overlay_dirname>/<edge|diffuse|both|total>/<masks_dirname>/
     edge_raw.npz
     edge_exclusion.npz
     diffuse_raw.npz
     diffuse_final.npz
     combined.npz

   results/<name>/<overlay_dirname>/<edge|diffuse|both|total>/overlays/
     edge_overlay_<idx:04d>.png
     diffuse_overlay_<idx:04d>.png
     combined_overlay_<idx:04d>.png
     total_overlay_<idx:04d>.png

   results/<name>/<overlay_dirname>/both/metrics/
     frame_metrics.csv

Defaults: ``overlay_dirname="delamination"``, ``masks_dirname="masks"``.
Every mask NPZ uses the same per-frame key convention: ``frame_<idx:04d>``.

The corresponding paths are written onto ``interface.metadata`` under these
keys: ``primary_masks_path``, ``secondary_masks_path``,
``diffuse_raw_masks_path``, ``diffuse_masks_path``, ``combined_masks_path``,
``delamination_metrics_path``.

Multi-interface edge progression uses the same pattern under
``edge_multi/masks/``, with one ``<interface>_inclusive.npz`` and
``<interface>_exclusive.npz`` pair per interface (see :doc:`methodology`).

Preprocessing cache
--------------------
.. code-block:: text

   results/<name>/<cache_dirname>/<key>/preprocess_%04d.npz

Default ``cache_dirname="Preprocessor_cache"``. See :doc:`Image_pre_processing`
for what each cached frame contains and how ``key`` is chosen.

Generic NPZ helper
-------------------
All of the above ultimately go through ``save_npz_bundle(data, path)``
(``deladect.io.bundles``): it refuses an empty dict, coerces the filename to
end in ``.npz``, and creates parent directories as needed.
``load_npz_bundle(path)`` raises if the file is missing rather than
returning an empty result.

Related pages
-------------
- :doc:`detection` and :doc:`delamination` for the detectors that produce
  these files.
- :doc:`Image_pre_processing` for the preprocessing cache format.
- :doc:`examples/save_reload_results` for a full save/reload walkthrough.
