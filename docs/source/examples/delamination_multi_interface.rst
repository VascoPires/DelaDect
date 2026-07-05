Multi-Interface Delamination (Sample-5)
=======================================

This example documents the notebook flow used for sample-5 multi-interface
edge delamination:

- notebook: ``notebooks/edge_multi_interface_sample5_validation.ipynb``
- data: ``example_images/sample-5/cut_images``

Goal
----
Detect hierarchical edge delamination across an ordered interface list
(``i0 -> i1 -> i2``), save masks and overlays, and inspect promotion behavior.

Step 1 - Build specimen and interfaces
--------------------------------------

.. code-block:: python

    from pathlib import Path

    from deladect.detection import DelaminationDetector
    from deladect.specimen import Specimen
    from skimage.io import imread

    specimen = Specimen(
        name="sample-5-edge-multi",
        scale_px_mm=41.03328366,
        path_full="example_images/sample-5/cut_images",
        sorting_key="_sc",
        image_types=["png"],
        auto_init_stacks=False,
        results_root="results",
    )

    specimen.path_full_list = [str(p) for p in sorted(Path(specimen.path_full).glob("*.png"))]
    specimen.image_stack_full = [imread(path) for path in specimen.path_full_list]

    specimen.add_ply(name="ply_0", orientation_deg=0.0, avg_crack_width_px=8.0, min_crack_length_px=20.0)
    specimen.add_ply(name="ply_1", orientation_deg=90.0, avg_crack_width_px=8.0, min_crack_length_px=20.0)
    specimen.add_ply(name="ply_2", orientation_deg=0.0, avg_crack_width_px=8.0, min_crack_length_px=20.0)
    specimen.add_ply(name="ply_3", orientation_deg=90.0, avg_crack_width_px=8.0, min_crack_length_px=20.0)

    specimen.add_interface(name="i0", upper_ply_index=0, lower_ply_index=1)
    specimen.add_interface(name="i1", upper_ply_index=1, lower_ply_index=2)
    specimen.add_interface(name="i2", upper_ply_index=2, lower_ply_index=3)

Step 2 - Preprocess with rolling reference
------------------------------------------

.. code-block:: python

    detector = DelaminationDetector(specimen, specimen.interfaces[0], save_preprocess_outputs=True)

    cache_paths = detector.preprocess_stack_to_disk(
        specimen.image_stack_full,
        key="sample5_multi_edge",
        reference_mode="rolling_median",
        reference_window=7,
        reference_skip=2,
        cache_dirname="Preprocessor_cache",
    )

Step 3 - Run multi-interface edge detection
-------------------------------------------

.. code-block:: python

    result = detector.edge.detect_edge_multi(
        interfaces=specimen.interfaces,
        processed_cache_paths=cache_paths,
        save_masks=True,
        save_overlays=True,
        overlay_dirname="delamination",
        params={
            "secondary_similarity_threshold": 0.6,
            "promotion_persistence_frames": 3,
            "onset_policy": "confirm_then_backfill",
            "allow_terminal_promotion": True,
        },
    )

    print(result["paths"]["overlays"])

Where outputs are written
-------------------------
- ``results/sample-5-edge-multi/Preprocessor_cache/sample5_multi_edge/``
- ``results/sample-5-edge-multi/delamination/edge_multi/masks/``
- ``results/sample-5-edge-multi/delamination/edge_multi/overlays/``

Suggested screenshots for this page
-----------------------------------
Add these to ``docs/source/_static/delamination/`` and embed with ``.. figure::``:

1. ``sample5_multi_overlay_example.png``
   - one representative ``edge_multi_overlay_XXXX.png`` frame
2. ``sample5_multi_masks_browser.png``
   - file browser view of generated mask bundles
3. ``sample5_multi_terminal_promotion.png``
   - frame sequence highlighting terminal promotion in the tail frames

Notes
-----
- ``detect_edge_multi`` requires preprocessed input by design.
- The function logs a reminder recommending rolling-median preprocessing
  with ``reference_skip >= 1``.

Troubleshooting
---------------
- Empty deeper levels (i1/i2)
  - lower ``secondary_similarity_threshold`` slightly (e.g. 0.6 -> 0.5) and inspect overlays.
- Immediate over-promotion
  - increase ``promotion_persistence_frames`` and/or ``reference_skip``.
- Edge case at sequence tail
  - keep ``allow_terminal_promotion=True`` to include late-arriving damage.
