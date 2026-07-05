Getting Started
===============

This walkthrough takes you from a fresh clone to a first crack and
delamination result using bundled sample data.

1) Environment setup
--------------------

.. code-block:: bash

   python -m venv .venv
   .venv\Scripts\activate      # Windows
   # source .venv/bin/activate  # macOS/Linux
   pip install --upgrade pip
   pip install -e .[dev]

2) First crack run (sample images)
-----------------------------------

.. code-block:: python

   from pathlib import Path

   from deladect.detection import crack_eval_crossply
   from deladect.specimen import Specimen
   from skimage.io import imread

   data_root = Path("example_images") / "sample-1" / "full"
   frame_paths = sorted(data_root.glob("*.png"))

   specimen = Specimen.from_cross_ply(
       name="sample-1-quickstart",
       scale_px_mm=41.03328366,
       path_full=str(data_root),
       sorting_key="_sc",
       image_types=["png"],
       auto_init_stacks=False,
       results_root="results",
       avg_crack_width_px=8.0,
   )

   specimen.path_full_list = [str(path) for path in frame_paths]
   specimen.image_stack_full = [imread(str(path)) for path in frame_paths]

   crack_results = crack_eval_crossply(
       specimen,
       export_images=True,
       save_cracks=True,
       results_dir="results",
   )

   print(crack_results.keys())

4) First delamination run (combined edge + diffuse)
---------------------------------------------------

.. code-block:: python

   from deladect.detection import DelaminationDetector

   interface = specimen.add_interface(name="i0", upper_ply_index=0, lower_ply_index=1)
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
    )

   result = detector.detect_both_delaminations(
       cracks=cracks,
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

5) Save and reload specimen + artefact references
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
- :doc:`save_reload_results` for manifest-based reload workflows.
- :doc:`crack_detection` for full crack workflows and post-processing.
- :doc:`delamination_multi_interface` for sample-5 multi-interface edge progression.
- :doc:`../results_storage` for file/bundle schema details.
