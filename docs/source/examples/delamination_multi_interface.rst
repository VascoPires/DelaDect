02 - Multi-Interface Edge Delamination
======================================

This example shows edge delamination on its own: first on a single interface,
then across two interfaces with hierarchical promotion. It is an edge-only
workflow -- no crack detection and no diffuse delamination are involved.

A `Binder <https://mybinder.org/v2/gh/vascodcpires/deladect/main?labpath=notebooks/multi_interface_edge_delamination.ipynb>`_
notebook that serves as a companion to this example is available in the
repository and can be run without installation.

Building the specimen
----------------------

Multi-interface promotion needs at least two interfaces, so this specimen has
three plies (``[0, 90, 0]``) and two interfaces: ``i0`` between the first two
plies, and ``i1`` between the last two. ``i0`` is the primary interface;
``i1`` is the deeper interface that can be *promoted* once evidence of
delamination persists beneath it. Crack detection isn't needed for this
workflow, so no ply-level crack parameters are set beyond the defaults.

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
   for index, orientation in enumerate((0.0, 90.0, 0.0)):
       specimen.add_ply(
           name=f"ply_{index}",
           orientation_deg=orientation,
           avg_crack_width_px=8.0,
           min_crack_length_px=20.0,
       )
   for index in range(2):
       specimen.add_interface(name=f"i{index}", upper_ply=index, lower_ply=index + 1)

   detector = DelaminationDetector(specimen, specimen.interfaces[0], save_preprocess_outputs=True)

The script uses the ten frames in ``example_images/sample-3`` and writes below
``results/02-multi-interface-edge``:

.. code-block:: bash

   python examples/02_multi_interface_edge_delamination.py

1. Standalone edge delamination
--------------------------------

:meth:`~deladect.detection.delamination.EdgeDetector.detect_primary` runs
entirely on its own -- no crack catalogue, no diffuse pipeline, and no second
interface required. This is the same edge algorithm used inside
``detect_both_delaminations``, just called directly on ``i0``.

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

2. Multi-interface promotion
------------------------------

:meth:`~deladect.detection.delamination.EdgeDetector.detect_edge_multi` adds
hierarchical promotion to a deeper interface. It needs two separate
preprocessing caches: a *static*-reference cache drives the primary
accumulation at ``i0``, while a *rolling-median*-reference cache drives the
promotion check at ``i1`` -- it must stay sensitive to change happening
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
:doc:`../edge_delamination`.

Observed result
----------------

.. figure:: ../_static/examples/multi_interface_detection_outputs.png
   :alt: Standalone single-interface edge delamination compared with multi-interface promotion, frame 272
   :width: 100%
   :align: center

   **(a)** Standalone ``detect_primary`` on ``i0`` alone. **(b)** The same
   frame from ``detect_edge_multi``: ``i0`` unchanged, plus ``i1`` promoted
   wherever the rolling-median pass found further change inside the settled
   ``i0`` region.

``i0`` begins growing from the third sampled frame onward and reaches
2,015,119 pixels by the final frame. ``i1`` stays at zero until the final
sampled frame, where it appears with 318,015 pixels -- visible as the blue
regions in panel (b) above, concentrated near the top and bottom edges rather
than spread along ``i0``'s full length.

.. figure:: ../_static/examples/multi_interface_area_plot.png
   :alt: Detected area in pixels for i0 and i1 plotted against frame number
   :width: 70%
   :align: center

   Detected area per interface across the sampled frames. ``i1`` only
   registers non-zero area once promotion condition is met in the last frame;
   these values are useful smoke-test expectations for this dataset, not
   universal thresholds.

Inspect ``results/02-multi-interface-edge/delamination/edge_multi/overlays``
and the inclusive/exclusive bundles in the adjacent ``masks`` directory. The
single-interface run from step 1 is written separately, under
``results/02-multi-interface-edge/edge_only/edge``.

Cracks and delamination together, in 3D
-----------------------------------------

The scene below is a **different specimen** from the rest of this page: a
three-ply ``[90, -30, 30]`` plus/minus/transverse laminate, rather than
sample-3's edge-only ``[0, 90, 0]`` workflow. It's included here because
sample-3 never runs crack detection, so it can't show cracks and
delamination together -- this laminate has both, at its last analyzed
frame. Each ply is drawn at an intentionally exaggerated 8 mm thickness with
a 30 mm gap (real plies are much thinner) so the three layers stay visually
distinct; the dark prisms are individual cracks and the translucent
red/blue planes are the primary/secondary delamination masks.

.. raw:: html

   <iframe src="../_static/examples/full_laminate_3d_view.html"
           width="100%" height="520" style="border: 1px solid #ccc; border-radius: 4px;"
           loading="lazy">
   </iframe>

Drag to rotate, scroll to zoom. This is a self-contained ``vtk.js`` export
(the same mechanism PyVista's own documentation uses for its interactive
examples: ``Plotter.export_html()`` serializes the actual VTK scene to run
natively in the browser, so transparency and lighting carry over correctly).
There is no PyVista -- or any other 3D library -- dependency anywhere in
DelaDect itself; the scene was generated once, offline, and only the
resulting static HTML file is shipped with the docs.

Sample-3 itself, in 3D
------------------------

Unlike the illustrative laminate above, this scene is result-backed: it's
specimen ``Out30-p1`` at frame 217, rendered directly from the crack and
interface-mask artefacts this page's ``detect_edge_multi`` walkthrough
produced -- real ply geometry, real detected cracks, real delamination
masks, no exaggeration.

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

Drag to rotate, scroll to zoom. Same ``Plotter.export_html()`` mechanism as
above, rendered with a white background to match the docs page. Both crack
states are pre-rendered from the same camera angle and swapped client-side,
so toggling is instant; rotating one view and then flipping the checkbox
resets to that shared starting angle rather than carrying your rotation over.

Input limitation
------------------

Multi-interface promotion currently requires a full-height stack (or
full-height preprocessed frames). Explicit upper, middle, and lower region
stacks, as used in :doc:`getting_started`, are not supported by this path.
