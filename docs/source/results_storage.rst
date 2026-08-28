How to store results
=====================

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

Save, reload, and experimental data
------------------------------------

You can persist a specimen manifest and later reload the specimen together
with its stored crack and delamination artefacts. Experimental data is also
restored because ``strain_csv`` is part of the saved manifest. This supports
reproducible workflows where analysis and later inspection happen in
different Python sessions.

Run ``python examples/01_getting_started.py`` first, followed by:

.. code-block:: bash

   python examples/03_save_reload_results.py

``01_getting_started.py`` builds its specimen from ``example_images/sample-1``
with ``strain_csv=".../sample-1/experimental_data.csv"``, then saves the
manifest via ``save_specimen``. The second example only reloads:

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

Experimental data on reload
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``strain_csv`` round-trips through ``to_dict``/``from_dict`` like any other
constructor argument, so ``specimen.experimental_data`` is populated again as
soon as the specimen is reconstructed -- no separate reload step is needed:

.. code-block:: python

   print(specimen.experimental_data)

Expected console messages
~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``verbose=True`` in ``load_specimen(..., load_results=True)``, messages
are printed for discovered artefacts, for example:

- ``Found cracks for ply 'ply_0' (...)``
- ``Found edge/diffuse delamination artefacts for interface 'i0': ...``

Strict mode
~~~~~~~~~~~

Use ``strict=True`` to fail fast when a metadata path points to a missing
file, rather than silently skipping it.

Related pages
-------------
- :doc:`detection` for the detectors that produce these files.
- :doc:`Image_pre_processing` for the preprocessing cache format.
