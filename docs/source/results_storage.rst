Results Storage Schema
======================

Overview
--------
DelaDect stores most crack and delamination artifacts as compressed NPZ bundles
and records their paths in ply and interface metadata.

Frame-key convention
--------------------
Mask and crack bundles use frame keys in the form:

- ``frame_0000``
- ``frame_0001``
- ...

Keys are zero-padded to 4 digits.

Crack storage
-------------
Ply crack bundles are stored in NPZ files and linked through:

- metadata key: ``crack_results_path``
- constant: ``PLY_CRACK_RESULTS_KEY``

Typical output layout:

- ``results/<specimen>/cracks/ply_<ply_name>/data/<specimen>_<ply_name>_cracks.npz``

Delamination storage (single interface)
---------------------------------------
Combined edge/diffuse workflows use these interface metadata keys:

- ``primary_masks_path``
- ``secondary_masks_path``
- ``diffuse_raw_masks_path``
- ``diffuse_masks_path``
- ``combined_masks_path``
- ``delamination_metrics_path``

Constants are exported from :mod:`deladect.io.delamination` and
:mod:`deladect.io.cracks` (and re-exported via :mod:`deladect.io`).

Combined workflow bundles
~~~~~~~~~~~~~~~~~~~~~~~~~
When ``detect_both_delaminations`` runs with mask saving enabled:

- ``edge_raw.npz``
- ``edge_exclusion.npz``
- ``diffuse_raw.npz``
- ``diffuse_final.npz``
- ``combined.npz``

Metrics are typically stored at:

- ``results/<specimen>/<overlay_dirname>/both/metrics/frame_metrics.csv``

Multi-interface edge bundles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When ``detect_edge_multi`` runs with mask saving enabled, each interface gets:

- ``<interface_key>_inclusive.npz``
- ``<interface_key>_exclusive.npz``

These paths are linked in interface metadata using ``primary_masks_path``
(inclusive) and ``secondary_masks_path`` (exclusive) for backward-compatible
storage access.

Programmatic helpers
--------------------
Use :mod:`deladect.io` helpers (or the domain-specific modules
:mod:`deladect.io.cracks` and :mod:`deladect.io.delamination`) to avoid manual path handling:

- ``save_mask_bundle``
- ``store_interface_masks``
- ``store_interface_delamination_results``
- ``load_interface_primary_masks``
- ``load_interface_secondary_masks``
- ``load_interface_diffuse_raw_masks``
- ``load_interface_diffuse_masks``
- ``load_interface_combined_masks``
- ``save_interface_metrics``
- ``load_interface_metrics``
- ``load_ply_crack_results``
- ``load_specimen``
- ``load_stored_results``

Reloading specimen + stored artefacts
-------------------------------------
You can restore a specimen definition from JSON and optionally trigger eager
loading of referenced artefacts (cracks/masks) with status messages.

.. code-block:: python

    from deladect.io import load_specimen, load_stored_results

    specimen = load_specimen(
        "results/sample-1/config/sample-1.json",
        auto_init_stacks=False,
        load_results=True,
        verbose=True,
    )

    # Optional: obtain all loaded bundles as a nested dictionary
    bundles = load_stored_results(specimen, strict=False, verbose=False)
    print(bundles.keys())  # dict_keys(['plies', 'interfaces', 'summary'])

Examples
--------

Load crack bundle from ply metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from deladect.io import load_ply_crack_results

    bundle = load_ply_crack_results(ply)
    cracks_f0 = bundle["frame_0000"]

Load combined masks from interface metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from deladect.io import load_interface_combined_masks

    combined = load_interface_combined_masks(interface)
    mask_f10 = combined["frame_0010"]

Register a precomputed bundle path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pathlib import Path
    from deladect.io import store_interface_masks

    store_interface_masks(
        interface,
        primary_path=Path("results/demo/edge_primary.npz"),
        secondary_path=Path("results/demo/edge_secondary.npz"),
    )
