02 - Multi-Interface Edge Delamination
======================================

In the first example, the fundamentals of DelaDect were covered. For that example,
the specimen had a single interface where delamination detection was performed. However,
DelaDect also offers the possibility to perform delamination detection across multiple interfaces.

This functionality is only available for edge delamination detection for specimens such as the one
shown in Sample-3 from the examples provided. Since from a single image we only obtain a distribution
of intensity, it is not possible to distinguish between delamination at different interfaces. However,
if we have a sequence of images, it is possible to detect delamination at one interface and then
if additional delamination is detected in the same region in subsequent images, we can detect
delamination on a different layer.

.. note::

   This approach is built on the assumption that any additional darkening in
   the same region is due to delamination at a different interface, and it
   is only valid when delamination shows up sequentially during a
   mechanical test with one interface dominant in the initial stages. This
   is often the case for laminates such as :math:`[\pm \theta /90^\circ]_s`,
   but it is an assumption, not something the algorithm verifies.

This example is divided into three parts. The first part shows how different normalization
of the images in a sequence can be performed, here we will see the differences between using the static
reference and the rolling median reference. The second part shows how to perform delamination detection
on a single interface. Finally, the third part shows how to perform delamination detection on multiple
interfaces.

A `Binder <https://mybinder.org/v2/gh/vascodcpires/deladect/main?labpath=notebooks/multi_interface_edge_delamination.ipynb>`_
notebook that serves as a companion to this example is available in the
repository and can be run without installation.

.. grid:: 2 2 4 4
   :gutter: 2

   .. grid-item-card:: Build the specimen
      :link: multi-interface-build
      :link-type: ref

      Three plies, two interfaces, and the primary/secondary distinction.

   .. grid-item-card:: Standalone edge
      :link: multi-interface-standalone
      :link-type: ref

      ``detect_primary`` on one interface, no promotion involved.

   .. grid-item-card:: Multi-interface promotion
      :link: multi-interface-promotion
      :link-type: ref

      ``detect_edge_multi``, static vs. rolling-median caches, and the
      observed result.

   .. grid-item-card:: Out30-p1, in 3D
      :link: multi-interface-3d
      :link-type: ref

      An interactive, real-data 3D render of the promoted interfaces.

.. _multi-interface-build:

Building the specimen
----------------------

Multi-interface promotion needs at least two interfaces, so this specimen has
three ply directions: 90, -30, 30 (symmetric) and two interfaces: ``90/-30`` between the first two
plies, and ``-30/30`` between the last two. ``90/-30`` is what we call the primary interface, since it
is the first interface to show delamination, while ``-30/30`` is the secondary interface.
As seen before for the :doc:`getting started <getting_started>` example, the specimen can be built by:

.. code-block:: python

   from pathlib import Path

   from deladect.detection import DelaminationDetector
   from deladect.io import save_specimen
   from deladect.io.delamination import save_mask_bundle
   from deladect.specimen import Specimen

   specimen = Specimen(
       name="02-multi-interface-edge",
       scale_px_mm=41.03328366,
       path_full=str(data_root),
       sorting_key="_sc",
       image_types=["png"],
       avg_crack_width_px=8.0,
   )
   specimen.add_ply(
       name="ply_0",
       orientation_deg=90.0,
       avg_crack_width_px=8.0,
       min_crack_length_px=20.0,
   )
   specimen.add_ply(
       name="ply_1",
       orientation_deg=-30.0,
       avg_crack_width_px=8.0,
       min_crack_length_px=20.0,
   )
   specimen.add_ply(
       name="ply_2",
       orientation_deg=30.0,
       avg_crack_width_px=8.0,
       min_crack_length_px=20.0,
   )
   specimen.add_interface(name="90/-30", upper_ply=0, lower_ply=1)
   specimen.add_interface(name="-30/30", upper_ply=1, lower_ply=2)

   detector = DelaminationDetector(specimen, specimen.interfaces[0], save_preprocess_outputs=True)

The script uses the ten frames in ``example_images/sample-3`` and writes below
``results/02-multi-interface-edge``:

.. code-block:: bash

   python examples/02_multi_interface_edge_delamination.py

.. _multi-interface-standalone:

1. Standalone edge delamination
--------------------------------

:meth:`~deladect.detection.delamination.EdgeDetector.detect_primary` runs
entirely on its own -- no crack catalogue, no diffuse pipeline, and no second
interface required. This is the same edge algorithm used inside
``detect_both_delaminations``, just called directly on ``90/-30``.

.. code-block:: python

   primary_only = detector.edge.detect_primary(
       save_overlays=True,
       overlay_dirname="edge_only",
       params={
           "window_edge": (1, 130),
           "gaussian_filters": (0.5, 15.0),
           "hard_floor": 0.90,
           "scale_min_percentile": 10,
           "scale_max_percentile": 95,
           "seed_ratio": 0.01,
           "post_threshold_closing_px": 20,
       },
   )
   primary_only_masks_path = save_mask_bundle(
       primary_only["masks"],
       specimen.results_dir("edge_only", "edge", "masks") / "primary.npz",
   )

``detect_primary`` doesn't save masks by itself (only overlays, when
requested); the snippet above saves the returned masks explicitly with
``save_mask_bundle`` so they persist alongside the overlays.

.. _multi-interface-promotion:

2. Multi-interface promotion
------------------------------

:meth:`~deladect.detection.delamination.EdgeDetector.detect_edge_multi` adds
hierarchical promotion to a deeper interface. It needs two separate
preprocessing caches: a *static*-reference cache drives the primary
accumulation at ``90/-30``, while a *rolling-median*-reference cache drives the
promotion check at ``-30/30`` -- it must stay sensitive to change happening
inside a region already flagged as damaged, which a static reference would no
longer highlight.

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

   multi_result = detector.edge.detect_edge_multi(
       interfaces=specimen.interfaces,
       processed_cache_paths=primary_cache,
       secondary_cache_paths=secondary_cache,
       save_masks=True,
       save_overlays=True,
       primary_params={
           "window_edge": (1, 130),
           "gaussian_filters": (0.5, 15.0),
           "hard_floor": 0.90,
           "scale_min_percentile": 10,
           "scale_max_percentile": 95,
           "seed_ratio": 0.01,
           "post_threshold_closing_px": 20,
       },
       secondary_edge_params={
           "window_edge": (1, 30),
           "gaussian_filters": (0.5, 15.0),
           "hard_floor": 0.90,
           "scale_min_percentile": 10,
           "scale_max_percentile": 95,
           "seed_ratio": 0.01,
           "post_threshold_closing_px": 10,
       },
       secondary_params={
           "secondary_similarity_threshold": 0.80,
           "min_primary_frac_for_secondary": 0.10,
           "secondary_start_frame": 2,  # frame 195, closest sample-3 frame to id 181
       },
   )

   manifest = specimen.results_dir("config") / "specimen.json"
   save_specimen(specimen, manifest)

For the full promotion mechanics -- how a candidate becomes promoted, and
what each parameter in ``secondary_params`` actually does -- see
:doc:`../edge_delamination`. For the conceptual *why* behind static vs.
rolling-median references, with schematic diagrams, see
:doc:`../Image_pre_processing`.

Static vs. rolling-median preprocessing, on this data
""""""""""""""""""""""""""""""""""""""""""""""""""""""

A quick, concrete look at what the two caches created above actually produce,
straight from this run: the *baseline* each reference mode computes, and the
resulting *processed* frame, at the last sampled frame (where they've
diverged the most).

.. code-block:: python

   import numpy as np

   frame_idx = 9  # last sampled frame, where the two references have diverged the most
   cache_root = specimen.results_dir("Preprocessor_cache")

   def load_frame(key: str) -> dict:
       path = cache_root / key / f"preprocess_{frame_idx:04d}.npz"
       with np.load(path, allow_pickle=False) as payload:
           return {"baseline": payload["baseline"], "processed": payload["processed"]}

   static = load_frame("primary_static")
   rolling = load_frame("secondary_rolling")

.. figure:: ../_static/examples/static_vs_rolling_median_preprocessing.png
   :alt: Static vs rolling-median reference preprocessing compared on the same frame of Sample-3
   :width: 100%
   :align: center

   Sample-3, frame 272. **(a)** the static baseline is fixed to an early
   reference frame, so it never absorbs the delamination front; the
   normalized frame shows it as one strong, high-contrast band. **(b)** the
   rolling-median baseline (window=7, skip=2) tracks recent frames, so it
   partially absorbs the already-established front into the baseline itself
   -- the normalized band is fainter and narrower, but the reference stays
   sensitive to *new* change happening on top of it.

Observed result
----------------

.. figure:: ../_static/examples/multi_interface_detection_outputs.png
   :alt: Standalone single-interface edge delamination compared with multi-interface promotion, frame 272
   :width: 100%
   :align: center

   **(a)** Standalone ``detect_primary`` on ``90/-30`` alone. **(b)** The same
   frame from ``detect_edge_multi``: ``90/-30`` unchanged, plus ``-30/30``
   promoted wherever the rolling-median pass found further change inside the
   settled ``90/-30`` region.

``90/-30`` begins growing from the third sampled frame onward and reaches
2,015,119 pixels by the final frame. ``-30/30`` stays at zero until the final
sampled frame, where it appears with 318,015 pixels -- visible as the blue
regions in panel (b) above, concentrated near the top and bottom edges rather
than spread along ``90/-30``'s full length.

.. figure:: ../_static/examples/multi_interface_area_plot.png
   :alt: Detected area in pixels for interfaces 90/-30 and -30/30 plotted against frame number
   :width: 100%
   :align: center

   Detected area per interface across the sampled frames. ``-30/30`` only
   registers non-zero area once promotion condition is met in the last frame;
   these values are useful smoke-test expectations for this dataset, not
   universal thresholds.

Inspect ``results/02-multi-interface-edge/delamination/edge_multi/overlays``
and the inclusive/exclusive bundles in the adjacent ``masks`` directory. The
single-interface run from step 1 is written separately, under
``results/02-multi-interface-edge/edge_only/edge``.

.. _multi-interface-3d:

Out30-p1, in 3D
------------------------

Unlike the illustrative laminate above and the sample-3 walkthrough on the
rest of this page, this scene is result-backed with a *different*, real
specimen: ``Out30-p1`` from the EMB90 study, the same ``[+30/-30/90]_s``
laminate, at frame 217, rendered directly from the crack and interface-mask
artefacts its own ``detect_edge_multi`` run produced (using the same
DelaDect API this page walks through, just on the real EMB90 study data
rather than sample-3) -- real ply geometry, real detected cracks, real
delamination masks, no exaggeration.

.. raw:: html

   <label style="display: inline-flex; align-items: center; gap: 0.4em; margin-bottom: 0.5em;">
     <input type="checkbox" id="sample3-3d-cracks-toggle">
     Show cracks
   </label>
   <div style="position: relative; width: 100%; height: 520px;">
     <iframe id="sample3-3d-with-cracks" src="../_static/examples/Out30-p1_3d_view.html"
             width="100%" height="520"
             style="position: absolute; top: 0; left: 0; border: 1px solid #ccc; border-radius: 4px; visibility: hidden;"
             loading="lazy">
     </iframe>
     <iframe id="sample3-3d-no-cracks" src="../_static/examples/Out30-p1_3d_view_no_cracks.html"
             width="100%" height="520"
             style="position: absolute; top: 0; left: 0; border: 1px solid #ccc; border-radius: 4px;"
             loading="lazy">
     </iframe>
   </div>
   <script>
     document.getElementById("sample3-3d-cracks-toggle").addEventListener("change", function (event) {
       var withCracks = document.getElementById("sample3-3d-with-cracks");
       var noCracks = document.getElementById("sample3-3d-no-cracks");
       if (event.target.checked) {
         withCracks.style.visibility = "visible";
         noCracks.style.visibility = "hidden";
       } else {
         withCracks.style.visibility = "hidden";
         noCracks.style.visibility = "visible";
       }
     });
   </script>

Drag to rotate, scroll to zoom.

.. dropdown:: How the crack-toggle mechanism works
   :icon: gear
   :color: secondary

   Same ``Plotter.export_html()`` mechanism as above, rendered with a white
   background to match the docs page. Both crack states are pre-rendered
   from the same camera angle and swapped client-side, so toggling is
   instant; rotating one view and then flipping the checkbox resets to that
   shared starting angle rather than carrying your rotation over.

.. _multi-interface-input-limitation:

Input limitation
------------------

.. warning::

   Multi-interface promotion currently requires a full-height stack (or
   full-height preprocessed frames). Explicit upper, middle, and lower
   region stacks, as used in :doc:`getting_started`, are **not** supported
   by this path.
