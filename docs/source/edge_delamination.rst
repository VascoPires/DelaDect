Edge Delamination
=================

Edge delamination is damage that remains connected to a specimen free edge.
The upper and lower specimen halves are processed independently; the lower
half is flipped so that the relevant free edge is row zero in both cases.

Detection sequence
------------------

For each frame, :meth:`deladect.detection.delamination.EdgeDetector.detect_primary`
performs the following operations:

1. directional maximum and minimum filtering;
2. unsharp masking and directional Gaussian smoothing;
3. constant scaling and thresholding;
4. morphological closing of the binary candidate mask;
5. directional reconstruction from a shallow free-edge seed; and
6. frame-to-frame latching.

The pixel-scale versions of these operations are shown on the
:doc:`image_operations` page.

Free-edge reconstruction
------------------------

Thresholding alone can produce isolated dark regions away from the edge.
Both reconstruction modes reject isolated regions. ``"directional"`` accepts
a candidate when the preceding row contains an accepted pixel within the
configured horizontal drift range. The drift is a tolerance: a value of three
permits support from any accepted pixel within ``+-3`` columns.
``"columnwise"`` is stricter: only the pixel directly above in the same column
can provide support. Empty rows cannot be jumped in either mode. The removed
``legacy_flood`` mode is no longer available.

The animations below use the same real threshold-candidate crop and seed row,
so the effect of the connectivity rule can be compared directly. Pixel
coordinates follow the image convention ``[row, col] = [y, x]``: columns and
``x`` run horizontally, while rows and ``y`` run vertically.

.. figure:: _static/edge_delamination/seed_ratio_directional.gif
   :alt: Directional free-edge reconstruction with horizontal lateral support
   :width: 960
   :align: center

   **Directional connectivity.** Growth proceeds row by row, and an accepted
   pixel in the preceding row may provide support within the displayed
   horizontal :math:`\Delta x` tolerance.

.. figure:: _static/edge_delamination/seed_ratio_columnwise.gif
   :alt: Columnwise free-edge reconstruction with same-column support
   :width: 960
   :align: center

   **Columnwise connectivity.** Growth still proceeds row by row, but support
   must come from the pixel directly above in the same column
   (:math:`\Delta x = 0`).

Frame-to-frame latching
-----------------------

The accepted mask is combined with the previous mask using a logical OR.
Previously detected edge damage is therefore retained while newly connected
damage is added. Where edge and diffuse classifications overlap, the combined
workflow assigns the shared pixels to edge delamination.

Key parameters
--------------

- ``seed_ratio`` controls the depth of the initial free-edge seed.
- ``connectivity_mode`` supports ``"directional"`` and ``"columnwise"``.
- ``directional_lateral_drift_px`` explicitly sets horizontal drift per row.
- ``directional_lateral_drift_scale`` derives drift from average crack width
  when no explicit pixel value is supplied.
- ``post_threshold_closing_radius`` controls binary closing.
- ``hard_floor`` provides an additional normalized intensity gate.

See :doc:`delamination` for the full API and parameter listing.

Multi-interface promotion
--------------------------

:meth:`~deladect.detection.delamination.EdgeDetector.detect_edge_multi` extends
the same primary edge algorithm above to a hierarchy of interfaces, ordered
shallow to deep. The first (primary) interface accumulates exactly as
described above. Each deeper interface is *promoted* from its parent: a pixel
only becomes secondary damage once it is both (a) classified in a secondary
binary pass and (b) already covered by the parent interface's established
mask.

Why two preprocessing caches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``detect_edge_multi`` accepts ``processed_cache_paths`` for the primary
accumulation and an optional, separate ``secondary_cache_paths`` for the
promotion step:

- The primary pass should use a **static**-reference cache, matching
  :meth:`~deladect.detection.delamination.EdgeDetector.detect_primary` and
  ``detect_both_delaminations``, so the shallow interface's result is
  identical across the two entry points.
- The promotion pass needs a **rolling-median**-reference cache instead. A
  static reference stops highlighting change once a region has already
  darkened, but promotion specifically needs to detect *further* change
  happening inside an area the primary pass has already flagged. A rolling
  reference stays sensitive to that interior change.

If ``secondary_cache_paths`` is omitted, the primary pass's own binary/mask
output is reused for promotion, which works but is less sensitive to damage
that develops after the primary front has already passed over a region.

How a candidate becomes promoted
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each frame and each deeper level, the algorithm:

1. Takes the secondary binary mask for that frame (from the rolling-median
   cache pass, or the primary pass if no separate cache was given).
2. Intersects it with the *parent* interface's latched mask, but read back
   ``reference_window`` frames earlier (the same window used for the
   rolling-median reference) -- an already-settled primary region rather than
   its still-growing edge.
3. Keeps only pixels still connected to the free edge.
4. OR-accumulates the result into that level's running mask, the same
   frame-to-frame latching used by the primary pass.

``secondary_start_frame`` gates this off entirely: frames at or before the
given index produce no secondary output for that level, which is useful when
a specimen has a known dwell period before deeper damage can physically
occur.

``secondary_similarity_threshold`` and ``min_primary_frac_for_secondary`` are
accepted and validated but are not currently consulted by the promotion
computation above -- ``secondary_similarity_threshold`` is only echoed back
in the returned ``params`` dict, and the primary-area-fraction gate implied
by ``min_primary_frac_for_secondary`` is computed but not yet wired into the
per-frame decision. Do not rely on either to change output; the effective
controls are ``secondary_start_frame`` and the reference-window delay
described above.

See :doc:`examples/delamination_multi_interface` for a runnable script and
notebook, and :doc:`delamination` for the full API.
