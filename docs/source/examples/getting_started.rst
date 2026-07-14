Getting Started
===============

This first example walkthrough is designed to get you up and running with a minimal set of steps.
A fully runnable version of this walkthrough (plus a bit more exploration) 
is available as a Jupyter notebook at ``notebooks/getting_started.ipynb`` and is also hosted
on binder.

1) Specimen setup
--------------------

.. code-block:: python

    from pathlib import Path

    from deladect.detection import crack_eval_crossply
    from deladect.specimen import Specimen
    from skimage.io import imread


    # Running the sample-1 example
    data_root = Path("example_images") / "sample-1" / "full"
    frame_paths = sorted(data_root.glob("*.png"))

    # Create a specimen object from the cross-ply images
    specimen = Specimen.from_cross_ply(
        name="sample-1-quickstart",
        scale_px_mm=41.03328366,
        path_full=str(data_root),
        sorting_key="_sc",
        image_types=["png"],
        results_root="results",
        avg_crack_width_px=8.0,
    )

``Specimen.from_cross_ply`` is convenience function: it builds a ``Specimen``
and automatically adds two plies at ``[0, 90]`` and one ``"0/90"`` interface
between them. For more information on the Specimen class and how to create
custom layups, see :doc:`specimen`.


2) First crack run 
-----------------------------------

For diffuse delamination detection, we need to run crack detection first. The
``crack_eval_crossply`` is a convenience function that runs crack detection on 
two directions, 0° and 90°. The output of this function is a dictionary with
the keys ``"0"`` and ``"90"``, each containing a dictionary with the detected
cracks.

.. code-block:: python

   crack_results = crack_eval_crossply(
       specimen,
       export_images=True,
       save_cracks=True,
       results_dir="results",
   )

   print(crack_results.keys())

3) First delamination run
--------------------------

For this example, "Sample-1" we are performing both diffuse and edge delamination detection. 


- ``detector.edge.detect_primary(...)`` - edge-only detection
- ``detector.diffuse.diffuse_delamination(...)`` - diffuse-only detection (crack-guided ROIs)
- ``detector.detect_both_delaminations(...)`` - combined edge + diffuse



.. code-block:: python

   from deladect.detection import DelaminationDetector

   # from_cross_ply already added a single "0/90" interface between the two plies
   interface = specimen.interfaces[0]
   detector = DelaminationDetector(specimen, interface, save_preprocess_outputs=True)

   # Use both cross-ply crack families as diffuse ROI input
   cracks = Specimen.join_cracks(
       crack_results["0"]["cracks"],
       crack_results["90"]["cracks"],
   )

   cache_paths = detector.preprocess_stack_to_disk(
       specimen.image_stack_full,
       key="sample1_quickstart",
       reference_mode="rolling_median",
       reference_window=3,
       reference_skip=1,
   )["cache_paths"]

For edge-only or diffuse-only results:

.. code-block:: python

   edge_result = detector.edge.detect_primary(processed_cache_paths=cache_paths, debug=True)
   diffuse_result = detector.diffuse.diffuse_delamination(cracks=cracks, processed_cache_paths=cache_paths, debug=True)

For the combined workflow used in most analyses:

.. code-block:: python

   result = detector.detect_both_delaminations(
       cracks=cracks,
       avg_crack_width_px=8.0,
       processed_cache_paths=cache_paths,
       diffuse_params={
           "window_diffuse": (60, 60),
       },
       save_overlays=True,
       overlay_view="classified",
       save_masks=True,
       save_metrics=True,
       edge_exclusion_px=5,
       return_masks=False,
   )

   print(result["paths"])

4) Save and reload specimen + artefact references
-------------------------------------------------

.. code-block:: python

   from deladect.io import load_specimen, save_specimen

   manifest = specimen.results_dir("config") / "sample-1-quickstart.json"
   save_specimen(specimen, manifest)

   specimen_reloaded = load_specimen(
       manifest,
       auto_init_stacks=False,
       load_results=True,
       verbose=True,
   )

   print(specimen_reloaded.name)

5) Building a specimen manually
--------------------------------

``Specimen.from_cross_ply`` is sugar over three calls: build a plain
``Specimen``, then call ``add_ply`` once per orientation. You can do this
yourself - useful for laminates that aren't a simple ``[0, 90]`` cross-ply,
or when you want explicit control over ply/interface naming:

.. code-block:: python

   specimen_manual = Specimen(
       name="sample-1-manual",
       scale_px_mm=41.03328366,
       path_full=str(data_root),
       sorting_key="_sc",
       image_types=["png"],
       auto_init_stacks=False,
       results_root="results",
       avg_crack_width_px=8.0,
   )
   specimen_manual.path_full_list = [str(path) for path in frame_paths]
   specimen_manual.image_stack_full = [imread(str(path)) for path in frame_paths]

   specimen_manual.add_ply(name="ply_0", orientation_deg=0.0, avg_crack_width_px=8.0, min_crack_length_px=20.0)
   specimen_manual.add_ply(name="ply_90", orientation_deg=90.0, avg_crack_width_px=8.0, min_crack_length_px=20.0)

   interface_manual = specimen_manual.add_interface(name="i0", upper_ply_index=0, lower_ply_index=1)

From here, ``crack_eval_crossply``, ``DelaminationDetector``, and everything
else in steps 2-4 work identically on ``specimen_manual`` /
``interface_manual``.

What you should see
-------------------
- Crack overlays and NPZ bundles under ``results/<specimen>/cracks/``.
- Delamination overlays, masks, and metrics under
  ``results/<specimen>/delamination/both/``.
- Optional preprocess preview panels under
   ``results/<specimen>/Preprocessor_outputs/``.
- A specimen JSON manifest under ``results/<specimen>/config/``.

Quick validation checks
-----------------------
- Crack output should contain both ``"0"`` and ``"90"`` orientation keys.
- ``frame_metrics.csv`` should include ``edge_frac``, ``diffuse_frac``, and
  ``combined_frac``.
- If diffuse appears too broad, increase ``reference_skip`` and reduce
  ``post_threshold_closing_scale`` in ``diffuse_params``.

Next steps
----------
- ``notebooks/getting_started.ipynb`` for a fully runnable version of this walkthrough.
- :doc:`save_reload_results` for manifest-based reload workflows.
- :doc:`crack_detection` for full crack workflows and post-processing.
- :doc:`delamination_multi_interface` for sample-5 multi-interface edge progression.
- :doc:`../results_storage` for file/bundle schema details.
