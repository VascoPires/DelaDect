03 - Multi-Interface Edge Delamination
======================================

Goal
----

Distinguish primary edge delamination at ``i0`` from a later promoted region at
the deeper ``i1`` interface. This is an edge-only workflow; it does not require
crack detection.

The example is useful with two interfaces. The included sample validates a
primary and one secondary level, but it does not justify a third interface.

Run it
------

.. code-block:: bash

   python examples/03_multi_interface_delamination.py

The script uses the ten frames in ``example_images/sample-4`` and writes below
``results/03-multi-interface``.

Why two preprocessing caches are used
-------------------------------------

The primary edge mask is accumulated against a static reference. A separate
rolling-median cache highlights changes inside the established primary region
and drives promotion to ``i1``.

.. code-block:: python

   primary_cache = detector.preprocess_stack_to_disk(
       specimen.image_stack_full,
       key="primary_static",
       reference_mode="static",
   )["cache_paths"]

   secondary_cache = detector.preprocess_stack_to_disk(
       specimen.image_stack_full,
       key="secondary_rolling",
       reference_mode="rolling_median",
       reference_window=7,
       reference_skip=2,
   )["cache_paths"]

   result = detector.edge.detect_edge_multi(
       interfaces=specimen.interfaces,
       processed_cache_paths=primary_cache,
       secondary_cache_paths=secondary_cache,
       save_masks=True,
       save_overlays=True,
   )

Observed result
---------------

On the included frames, ``i0`` begins in the third sampled frame and grows to
1,997,433 pixels. ``i1`` appears in the final sampled frame with 284,227 pixels.
These values are useful smoke-test expectations, not universal thresholds.

Inspect ``results/03-multi-interface/delamination/edge_multi/overlays`` and the
inclusive/exclusive bundles in the adjacent ``masks`` directory.

Input limitation
----------------

This method currently requires a full-height stack (or full-height preprocessed
frames). Explicit upper, middle, and lower region stacks are not supported by
the multi-interface path.
