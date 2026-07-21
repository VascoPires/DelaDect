02 - Advanced Options: Split-Region Analysis
=============================================

Goal
----

Show how to evaluate a specimen when the user supplies separate upper-border,
middle, and lower-border image stacks in addition to the full frames.

This is a useful advanced example because each region has a distinct purpose:

- crack detection automatically uses the middle stack;
- edge delamination detection uses the upper and lower stacks;
- output masks are reassembled to the dimensions of the full image;
- full images remain available as the background for overlays.

Run it
------

.. code-block:: bash

   python examples/02_advanced_options.py

The script uses ``example_images/sample-2`` and writes below
``results/02-advanced-options``. The full frames are 791 pixels high; the
explicit regions are 70 pixels upper, 651 pixels middle, and 70 pixels lower.
The generated masks are therefore reassembled to ``(791, 2878)``.

Configuration
-------------

.. code-block:: python

   specimen = Specimen.from_cross_ply(
       name="02-advanced-options",
       scale_px_mm=41.03328366,
       path_full="example_images/sample-2/full",
       path_upper_border="example_images/sample-2/upper",
       path_middle="example_images/sample-2/middle",
       path_lower_border="example_images/sample-2/lower",
       strain_csv="example_images/sample-2/experimental_data.csv",
       image_types=["png"],
       results_root="results",
   )

Interpretation
--------------

The example successfully demonstrates region routing and edge evaluation. With
the four sparsely sampled frames supplied in the repository, the final diffuse
mask is empty. That is a valid result, but it means this dataset should not be
used as evidence that diffuse damage was detected. Use a denser time sequence
when evaluating diffuse progression.

The three region paths must be supplied together. If any one is missing,
DelaDect falls back to the full-stack analysis path.

Scope limitation
----------------

Split-region input is implemented for single-interface edge and diffuse
workflows. :meth:`deladect.detection.EdgeDetector.detect_edge_multi` currently
expects a full-height processed stack and splits it into upper and lower halves;
it does not consume the explicit upper/middle/lower stacks.
