Parameter List
==============

This page is an index into the parameter documentation that lives next to
each algorithm, plus the crack-detection parameters (not documented
elsewhere yet). It doesn't duplicate those sections — follow the links for
full detail.

Delamination detection
------------------------
See :doc:`delamination`'s "Key parameter groups" section for:

- Edge primary parameters (``detect_primary(params=...)``) — ``window_edge``,
  ``threshold_strategy``, ``gaussian_filters``, ``scale_min``/``scale_max``,
  ``seed_ratio``, ``connectivity_mode``, ``hard_floor``,
  ``post_threshold_closing_scale``, ``min_object_px``.
- Diffuse parameters (``diffuse_delamination(params=...)``) — ``diffuse_dx``/``diffuse_dy``,
  ``crack_frame_policy``, threshold sampling controls, ``hard_floor``,
  ``post_threshold_closing_scale``.
- Multi-interface promotion parameters (``detect_edge_multi(params=...)``) —
  ``secondary_similarity_threshold``, ``min_primary_frac_for_secondary``, and
  ``secondary_start_frame``.

Preprocessing
-------------
See :doc:`Image_pre_processing` for ``history_clamp``, ``history_mode``,
``history_window_size``, ``reference_mode``, ``reference_window``, and
``reference_skip``.

Crack detection
----------------
.. currentmodule:: deladect.detection.crack_detection

These aren't yet covered elsewhere, so they're documented in full here.

``crack_eval``
^^^^^^^^^^^^^^
.. code-block:: python

   crack_eval(
       specimen, *,
       crack_width_px=None, min_crack_size_px=None,
       export_images=False, background=None, comparison=False,
       save_cracks=False, ply=None, results_dir=None,
       use_full_stack=False, color_cracks=None, frame_labels=None,
   )

- ``crack_width_px`` / ``min_crack_size_px`` — override the ply's/specimen's
  own defaults for expected crack width and the minimum size to keep.
- ``export_images`` — write per-frame overlay PNGs.
- ``save_cracks`` — persist results as an NPZ bundle (see
  :doc:`results_storage`).
- ``use_full_stack`` — evaluate against the full image stack instead of
  the middle-region stack.
- Orientation ``theta`` used internally is derived as
  ``(90 - ply.orientation_deg) % 180``.

``crack_eval_by_orientation``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

   crack_eval_by_orientation(specimen, *, orientations, tolerance=1e-3, ...)

Groups plies by orientation (within ``tolerance``) and runs ``crack_eval``
once per group; if a group has duplicate plies, only the first is used
(logged). Returns a dict keyed by orientation label.

``crack_eval_crossply``
^^^^^^^^^^^^^^^^^^^^^^^^
Thin wrapper over ``crack_eval_by_orientation`` that fixes
``orientations=[0.0, 90.0]`` — the common cross-ply case.

``crack_eval_plus_minus``
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

   crack_eval_plus_minus(specimen, theta, *, transverse_layer=False, ...)

Evaluates ``+theta`` and ``-theta``; also evaluates 90° if
``transverse_layer=True``.

``plot_cracks``
^^^^^^^^^^^^^^^^
.. code-block:: python

   plot_cracks(image, cracks, *, linewidth=1.0, color="red",
               background_flag=False, comparison=False)

Plotting helper used by the ``export_images``/``comparison`` paths above.

Related pages
-------------
- :doc:`detection` for the crack/delamination API overview.
- :doc:`delamination` for the full delamination parameter tables.
- :doc:`Image_pre_processing` for normalization parameters.
