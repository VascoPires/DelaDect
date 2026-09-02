How to store results
=====================

Overview
--------
Everything DelaDect saves lives under a per-specimen results root.

.. code-block:: text

   results/<specimen.name>/

(``results/`` by default, relative; override with ``specimen.results_root``.)
Each call to ``results_dir(*parts)`` joins and creates subdirectories under
that root; every part must be a single safe path component.

Config
------
.. code-block:: text

   results/<name>/config/<name>_config.json

``specimen.save_config()`` writes the manifest here (the path is also
available as ``specimen.config_path()``, without saving). Mutations like
``add_ply`` and ``add_interface`` are not persisted until you call it.
``save_specimen(specimen, path)`` does the same thing at an arbitrary path
of your choosing, as used in :doc:`examples/delamination_multi_interface`.

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


Save, reload, and experimental data
------------------------------------

You can save a specimen manifest and later reload the specimen together with
its stored crack and delamination results, to work on a previously analyzed
specimen without re-running detection.

Pass ``save_specimen`` the path where the manifest should live, for example
the ``manifest`` path built in :doc:`examples/delamination_multi_interface`:

.. code-block:: python

   from deladect.io import save_specimen

   manifest = specimen.results_dir("config") / "specimen.json"
   save_specimen(specimen, manifest)

Reload the same ``manifest`` path later, in the same session or a fresh one,
to get the specimen back together with everything under its results root:

.. code-block:: python

   from deladect.io import load_specimen, load_stored_results

   specimen = load_specimen(
       manifest,
       auto_init_stacks=False,
       load_results=True,
       verbose=True,
       strict=True,
   )
   bundles = load_stored_results(specimen, strict=True, verbose=True)

``strict=True`` fails fast when a metadata path points to a missing file,
rather than silently skipping it.

``strain_csv`` round-trips through the manifest like any other constructor
argument, so ``specimen.experimental_data`` is populated again as soon as
the specimen is reloaded. No separate step needed.

