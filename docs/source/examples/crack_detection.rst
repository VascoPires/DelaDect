Crack Detection
===============

This page covers the function-level crack API.

Coordinate convention: crack endpoints are stored as ``[row, col]`` (``[y, x]``).
When plotted, columns map to the horizontal axis.

Input expectations
------------------
- The image stack should be shift-corrected (or otherwise stable frame-to-frame).
- Each target orientation should have at least one configured ply.
- Ply metadata should include realistic crack width/length defaults.
- Use ``results_dir`` for output roots.

Basic single-orientation run
----------------------------

.. code-block:: python

    from pathlib import Path

    from deladect.detection import crack_eval
    from deladect.specimen import Specimen
    from skimage.io import imread

    data_root = Path("my_specimen_images") / "sample-1"

    specimen = Specimen(
        name="sample-1",
        scale_px_mm=41.033,
        path_full=str(data_root),
        sorting_key="_sc",
        image_types=["png"],
        auto_init_stacks=False,
        results_root="results",
    )
    specimen.path_full_list = [str(p) for p in sorted(data_root.glob("*.png"))]
    specimen.image_stack_full = [imread(path) for path in specimen.path_full_list]

    ply = specimen.add_ply(
        name="ply_0",
        orientation_deg=0.0,
        avg_crack_width_px=8.0,
        min_crack_length_px=20.0,
    )

    result = crack_eval(
        specimen,
        ply=ply,
        export_images=True,
        save_cracks=True,
        results_dir="results",
    )
    cracks = result["cracks"]
    rho = result["densities"]
    th = result["thresholds"]
    print(result["metrics"].head())

Plus/minus convenience wrapper
------------------------------

.. code-block:: python

    from deladect.detection import crack_eval_plus_minus

    specimen.add_ply(name="ply_p45", orientation_deg=45.0, avg_crack_width_px=8.0, min_crack_length_px=20.0)
    specimen.add_ply(name="ply_m45", orientation_deg=-45.0, avg_crack_width_px=8.0, min_crack_length_px=20.0)

    pm = crack_eval_plus_minus(
        specimen,
        theta=45.0,
        transverse_layer=True,
        export_images=False,
        save_cracks=True,
    )

    print(pm.keys())  # e.g. dict_keys(['45', '-45', '90'])

Cross-ply convenience wrapper
-----------------------------

.. code-block:: python

    from deladect.detection import crack_eval_crossply

    # add the 90-degree ply before running this convenience wrapper
    specimen.add_ply(
        name="ply_90",
        orientation_deg=90.0,
        avg_crack_width_px=8.0,
        min_crack_length_px=20.0,
    )

    crossply = crack_eval_crossply(
        specimen,
        export_images=True,
        save_cracks=True,
        results_dir="results",
    )

    cracks_0 = crossply["0"]["cracks"]
    cracks_90 = crossply["90"]["cracks"]

Post-processing and spacing
---------------------------
Post-processing utilities are available in the crack submodule.

.. code-block:: python

    from deladect.detection.crack_detection import (
        crack_filtering_postprocessing,
        pixels_to_length,
    )

    postprocess_result = crack_filtering_postprocessing(
        specimen,
        cracks_90,
        avg_crack_grouping_th_px=50,
        crack_length_th=20,
        remove_outliers=True,
        grouping=True,
    )
    spacing_data_px = postprocess_result["records"]
    filtered_cracks = postprocess_result["filtered_frames"]

    spacing_data_mm = pixels_to_length(spacing_data_px, scale_px_mm=specimen.scale_px_mm)["values"]

Visual sanity check
-------------------

.. code-block:: python

    from deladect.detection import plot_cracks

    plot_result = plot_cracks(
        image=specimen.image_stack_full[-1],
        cracks=cracks_90[-1],
        color="red",
        background_flag=True,
    )
    plot_result["figure"].savefig("results/sample-1_last_frame_cracks.png")

What to inspect after a run
---------------------------
- ``results/<specimen>/cracks/ply_*/plots/`` for per-frame overlays.
- ``results/<specimen>/cracks/ply_*/data/`` for persisted crack bundles.
- crack-density trends (``rho``) for monotonicity and sudden anomalies.

Tip for diffuse-delamination workflows
--------------------------------------
When using cracks as diffuse ROI input, merge both orientation families for
cross-ply stacks:

.. code-block:: python

    cracks_all = Specimen.join_cracks(crossply["0"]["cracks"], crossply["90"]["cracks"])

Suggested figure placeholders
-----------------------------
Create these images under ``docs/source/_static/cracks/`` and add with ``.. figure::``:

1. ``sample1_cracks_overlay.png`` - representative crack overlay frame
2. ``sample1_density_curve.png`` - crack-density vs frame index plot
3. ``sample1_spacing_hist.png`` - spacing distribution after post-processing
