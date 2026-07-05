Parameter List
==============

This page is the single source of truth for core delamination parameters,
their defaults, and where each parameter is applied in the pipeline.

Edge (primary)
--------------

.. list-table:: Edge parameter map
   :header-rows: 1
   :widths: 24 14 28 34

   * - Parameter
     - Default
     - Applied at stage
     - Notes
   * - ``window_edge``
     - ``(0, 60)``
     - max/min morphology prefilter
     - Controls candidate smoothing footprint.
   * - ``gaussian_filters``
     - ``(0.5, 15.0)``
     - Gaussian smoothing before scaling
     - Feeds threshold + hard-floor gate.
   * - ``scale_min``, ``scale_max``
     - ``150.0``, ``255.0``
     - Linear scaling to ``closed`` signal
     - Used for frame thresholding.
   * - ``threshold_strategy``
     - ``"kmeans"``
     - Threshold selection on ``closed``
     - ``"images"`` fallback via image-threshold helper.
   * - ``hard_floor``
     - ``0.90``
     - Weak-signal gate on normalized ``smoothed``
     - Keep pixels where ``smoothed/255 <= hard_floor``.
   * - ``post_threshold_closing_scale``
     - ``2.5``
     - Binary closing after thresholding
     - Bridges small gaps.
   * - ``post_threshold_closing_radius``
     - ``None``
     - Optional explicit closing override
     - ``0`` disables closing.
   * - ``min_object_px``
     - ``0``
     - Connected-component cleanup
     - Removes small isolated detections.
   * - ``seed_ratio``
     - ``0.01``
     - Edge-connected reconstruction seeding
     - Top-band seed depth fraction.
   * - ``connectivity_mode``
     - ``"directional"``
     - Edge-connected reconstruction mode
     - Optional ``"legacy_flood"`` compatibility.
   * - ``directional_lateral_drift_px``
     - ``None``
     - Directional row-to-row support
     - Explicit horizontal drift in px.
   * - ``directional_lateral_drift_scale``
     - ``0.25``
     - Directional row-to-row support
     - Used when ``*_px`` is ``None``.


Diffuse
-------

.. list-table:: Diffuse parameter map
   :header-rows: 1
   :widths: 24 14 28 34

   * - Parameter
     - Default
     - Applied at stage
     - Notes
   * - ``diffuse_dx``, ``diffuse_dy``
     - ``20.0``, ``20.0``
     - Crack ROI geometry
     - ROI half-width/half-height around crack segments.
   * - ``window_diffuse``
     - ``(0, 60)``
     - max/min morphology prefilter
     - Controls ROI candidate smoothing footprint.
   * - ``gaussian_filters``
     - ``(0.5, 15.0)``
     - Gaussian smoothing before scaling
     - Feeds threshold + hard-floor gate.
   * - ``scale_min``, ``scale_max``
     - ``150.0``, ``255.0``
     - Fixed linear scaling bounds
     - Used when percentile override is not active.
   * - ``scale_min_percentile``, ``scale_max_percentile``
     - ``10.0``, ``99.0``
     - Optional per-ROI adaptive scaling
     - Active only when both are set.
   * - ``hard_floor``
     - ``0.90``
     - Weak-signal gate on normalized ROI ``smoothed``
     - Keep pixels where ``smoothed/255 <= hard_floor``.
   * - ``threshold_max_samples``
     - ``200000``
     - Frame threshold estimation
     - Caps sampled pixels from union of ROI values.
   * - ``threshold_downsample``
     - ``2``
     - Frame threshold estimation
     - Spatial decimation before sampling.
   * - ``post_threshold_closing_scale``
     - ``2.5``
     - Binary closing after thresholding
     - Bridges narrow gaps inside ROI masks.
   * - ``crack_frame_policy``
     - ``"reference_midpoint"``
     - Crack-frame alignment to preprocessing metadata
     - ``{"current", "reference_latest", "reference_midpoint"}``.


Notes
-----

- ``hard_floor`` accepts normalized values in ``[0, 1]``. Legacy values above
  ``1`` are interpreted as 8-bit intensity and converted via ``value / 255``.
- Glud/Bender crack pipelines commonly use strict thresholds near ``0.96`` in
  their processed domain; delamination defaults here are tuned separately.
