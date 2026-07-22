Diffuse Delamination
====================

Diffuse delamination is sought locally around tracked transverse cracks. The
track supplies two things that an isolated crack detection cannot provide: a
persistent identity and a baseline image from when that crack was first seen.
The later neighborhood can then be compared with its own initial state.

.. image:: _static/delamination/diffuse_crack_tracking.gif
   :alt: Crack tracking baseline normalization and diffuse delamination detection
   :width: 760
   :align: center

Why crack tracking is needed
----------------------------

Crack detections are generated independently in every frame. Their ordering
can change, endpoints can move slightly, and cracks may grow or disappear.
The tracker assigns a persistent ID using four geometric cues:

- center-to-center distance;
- crack-angle difference;
- bounding-box overlap; and
- change in crack length.

Candidate assignments outside the distance or angle gates are rejected. The
remaining assignments are sorted by cost and accepted one-to-one, so one
detection cannot update several tracks. An unmatched detection starts a new
track; an unmatched active track is terminated.

The first detection of a track stores its baseline frame, segment, length, and
bounding box. Later matched detections retain that original baseline while
updating the track's latest geometry and history.

Removing the initial crack state
--------------------------------

For a matched track, the algorithm constructs a crack-aligned rectangular
region around the **baseline segment**. ``diffuse_dx`` controls its width
perpendicular to the crack and ``diffuse_dy`` extends it beyond the crack ends.
Exactly the same affine geometry samples both the baseline and current frames.

The normalized region is

.. math::

   R = \operatorname{clip}\left(\frac{I_{current}}
   {\max(I_{baseline}, 10^{-3})}, 0, 1\right).

This division is the initial-state removal:

- an unchanged crack pixel might give ``75 / 75 = 1`` and disappears from the
  dark-change image;
- unchanged background similarly remains near ``1``;
- a newly darkened neighborhood might give ``140 / 235 = 0.60`` and remains
  available for diffuse classification.

The ratio is then passed through the diffuse max/min, sharpening, smoothing,
scaling, thresholding, and binary-closing operations. Values in the dark tail
become the local diffuse mask, which is projected from crack-aligned
coordinates back into the full frame.

Vanishing cracks
----------------

When a previously matched crack is no longer detected, its track terminates.
The final segment still defines a neighborhood, and the current frame is
compared with the stored baseline. This allows diffuse damage to be evaluated
even when the visible crack itself has vanished into the damaged region.

Why the method works
--------------------

The method is sensitive to *change around a particular crack*, rather than to
the crack's absolute darkness. It works when image shift correction keeps the
same physical pixels aligned and when the tracker preserves the correct crack
identity. The geometric gates reduce accidental identity swaps, while the
local ROI prevents unrelated dark regions elsewhere in the specimen from
driving the classification.

See :doc:`delamination` for the complete API and :doc:`image_operations` for
the individual filtering operations.

