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
Directional reconstruction rejects those regions. Starting from the shallow
seed strip at row zero, a candidate pixel is accepted only when the preceding
row contains an accepted pixel within the configured horizontal drift range.

The drift value is a tolerance, not a required number of supporting pixels.
For example, a drift of three means that one accepted pixel anywhere within
``±3`` columns in the preceding row is sufficient. Empty rows cannot be
jumped.

Frame-to-frame latching
-----------------------

The accepted mask is combined with the previous mask using a logical OR.
Previously detected edge damage is therefore retained while newly connected
damage is added. Where edge and diffuse classifications overlap, the combined
workflow assigns the shared pixels to edge delamination.

Key parameters
--------------

- ``seed_ratio`` controls the depth of the initial free-edge seed.
- ``directional_lateral_drift_px`` explicitly sets horizontal drift per row.
- ``directional_lateral_drift_scale`` derives drift from average crack width
  when no explicit pixel value is supplied.
- ``post_threshold_closing_radius`` controls binary closing.
- ``hard_floor`` provides an additional normalized intensity gate.

See :doc:`delamination` for the full API and parameter listing.

