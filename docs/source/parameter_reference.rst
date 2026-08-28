Parameter Reference
=====================

This page collects every tunable parameter for edge detection, diffuse
detection, and multi-interface promotion in one place. For what the
algorithms actually do, see :doc:`methodology`, :doc:`edge_delamination`,
and :doc:`diffuse_delamination`. For the callable signatures these
parameters are passed into, see :doc:`detection`.

.. currentmodule:: deladect.detection.delamination

Edge primary parameters (``detect_primary(params=...)``)
-----------------------------------------------------------
- ``window_edge=(0, 60)``
- ``threshold_strategy="kmeans"``
- ``gaussian_filters=(0.5, 15.0)``
- ``scale_min=None``, ``scale_max=None`` (percentile-based scaling is the
  default; set both explicitly to switch to fixed-bound scaling)
- ``seed_ratio=0.01``
- ``connectivity_mode="directional"`` (also supports strict ``"columnwise"``;
  the former ``"legacy_flood"`` mode was removed)
- ``directional_lateral_drift_px=None`` (explicit horizontal drift per row)
- ``directional_lateral_drift_scale=0.25`` (used when ``*_px`` is ``None``)
- ``hard_floor=0.90`` (normalized gate on smoothed image; tweak per specimen)
- ``post_threshold_closing_px=4`` (pixel-radius closing; this is the active
  default -- takes precedence over ``post_threshold_closing_scale`` whenever set)
- ``post_threshold_closing_scale=None`` (optional size-relative closing,
  scaled by ``avg_crack_width_px``; only used when ``post_threshold_closing_px``,
  ``post_threshold_closing_radius``, and ``pre_threshold_closing_radius`` are
  all ``None``)
- ``post_threshold_closing_radius`` (optional explicit override; ``0`` disables closing)
- ``pre_threshold_closing_radius`` (legacy alias for explicit closing radius)
- ``min_object_px=0`` (remove small connected components after closing)

Diffuse parameters (``detector.diffuse.diffuse_delamination(params=...)``)
-------------------------------------------------------------------------------
- ``diffuse_dx=20.0``, ``diffuse_dy=20.0``
- ``crack_frame_policy in {"current", "reference_latest", "reference_midpoint"}``
- ``threshold_max_samples=200000``
- ``threshold_downsample=2``
- ``window_diffuse=(0, 60)``
- ``gaussian_filters=(0.5, 15.0)``
- ``scale_min=150.0``, ``scale_max=255.0`` (used as fixed scaling bounds)
- ``scale_min_percentile=10.0``, ``scale_max_percentile=99.0``
  (when both are set, per-ROI percentiles override fixed bounds)
- ``hard_floor=0.90`` (normalized gate on diffuse-smoothed ROI; tweak per specimen)
- ``post_threshold_closing_px=4`` (pixel-radius closing; this is the active
  default -- takes precedence over ``post_threshold_closing_scale`` whenever set)
- ``post_threshold_closing_scale=None`` (optional size-relative closing,
  scaled by ``avg_crack_width_px``; only used when ``post_threshold_closing_px``
  is ``None``)

How the filtering parameters work
-----------------------------------
``window_edge`` and ``window_diffuse`` are ``(wy, wx)`` sizes for a rectangular
neighbourhood. For each frame, the image is passed through a maximum filter
followed by a minimum filter, both using that same window -- a grayscale
morphological *closing*. This bridges small gaps between nearby bright pixels
and smooths the detected front, without growing its overall extent the way a
maximum filter alone would.

Making one dimension of the window much larger than the other (the default
``(0, 60)``, or ``(1, 60)`` as used in :doc:`examples/getting_started`) makes
the closing directional: gaps are bridged along the wide axis while the
narrow axis is left almost untouched. This reconnects a broken delamination
front running along that axis without merging separate, unrelated features
stacked in the other direction. A square window instead closes gaps evenly in
both directions, which suits more localized damage rather than an elongated
front. Larger windows bridge bigger gaps and produce a smoother, more
connected result, but can also merge genuinely separate damage regions
together and blur their boundaries; smaller windows preserve fine detail but
may leave a real, continuous front fragmented into disconnected pieces.

``diffuse_dx`` and ``diffuse_dy`` set the half-extent, in pixels, of the
rectangular region of interest built around each crack segment for diffuse
detection. A smaller ROI restricts the search to damage immediately adjacent
to a crack; a larger ROI captures diffuse damage further from the crack but
raises the chance of picking up unrelated background or neighbouring cracks'
damage.

``seed_ratio`` is the fraction of the *split-half* slice height (not the
whole-image pixel count) trusted as the initial seed region for edge-connected
reconstruction: ``seed_depth = round(seed_ratio * height)`` rows starting from
the specimen edge. Detections in that shallow seed band are assumed genuine
and used to grow the rest of the connected edge-damage region frame to frame.
A ratio of ``0.01`` seeds from the first 1% of rows in each half; raising it
trusts a deeper region as ground truth (useful if the true edge front starts
further in), while lowering it restricts seeding to a thinner, more
conservative band.

Hard-floor notes
------------------
- Glud/Bender-style crack pipelines commonly use a strict threshold near ``0.96`` in their processed domain.
- For delamination segmentation in this repository, ``0.90`` is the current practical default in recent tuning.
- You can override ``hard_floor`` independently for edge and diffuse in your study registry or per-run params.

For a lagged single-frame rolling reference, use preprocessing
``reference_mode="rolling_median"`` with ``reference_window=1`` and tune
``reference_skip`` for lag depth. See :doc:`Image_pre_processing` for the
full normalization reference.

Multi-interface promotion parameters (``detect_edge_multi(params=...)``)
-----------------------------------------------------------------------------
- ``secondary_similarity_threshold=0.6``
- ``min_primary_frac_for_secondary=0.0``
- ``secondary_start_frame=None``

Use ``processed_cache_paths`` for static-reference primary detection and
``secondary_cache_paths`` for rolling-median secondary detection.
