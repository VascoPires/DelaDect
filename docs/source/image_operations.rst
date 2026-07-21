Image Operations
================

The animations below use one synthetic image with oversized pixels.  The same
delamination band is carried through every operation so that each change can
be followed directly.

Directional max/min filtering
-----------------------------

The horizontal ``1 × 5`` window removes narrow vertical crack artefacts while
retaining the longer horizontal delamination band.  The maximum pass is
followed by the minimum pass.

.. image:: _static/image_operations/max_min_cleanup.gif
   :alt: Directional maximum and minimum filtering of an oversized-pixel delamination band
   :width: 760
   :align: center

Sharpening and Gaussian smoothing
---------------------------------

The cleaned max/min output is sharpened and then smoothed directionally.  The
horizontal Gaussian width is larger than the vertical width, following the
orientation of the free edge.

.. image:: _static/image_operations/sharpen_and_smooth.gif
   :alt: Unsharp masking followed by directional Gaussian smoothing
   :width: 760
   :align: center

Constant scaling
----------------

The smoothed intensities are mapped to ``[0, 1]`` using fixed lower and upper
bounds.  Values outside the bounds are clipped.

.. image:: _static/image_operations/constant_scaling.png
   :alt: Constant scaling of the smoothed image to zero through one
   :width: 760
   :align: center

Thresholding
------------

Pixels below the threshold become the binary delamination candidate mask.

.. image:: _static/image_operations/thresholding.gif
   :alt: Thresholding the scaled image into a binary candidate mask
   :width: 760
   :align: center

Morphological closing
---------------------

Closing is a dilation followed by an erosion.  Dilation first expands the
candidate mask so that nearby regions meet and small spaces disappear.  The
erosion then removes a layer from the outside, returning the band roughly to
its original thickness.  The newly bridged spaces remain filled because they
are now inside the connected region.  The animation displays the ``3 × 3``
discrete disk with radius one pixel; production uses the configured closing
radius in the same way.

.. image:: _static/image_operations/morphological_closing.gif
   :alt: Morphological closing with the disk footprint and newly filled pixels
   :width: 760
   :align: center

Frame-to-frame latching
-----------------------

The current detection is combined with the previous latched mask using a
cumulative OR.  Blue pixels were already detected, yellow pixels occur in
both masks, and red pixels are newly added.  Consequently, previously
detected delamination is retained even if it is absent from a later frame.

.. image:: _static/image_operations/frame_latching.gif
   :alt: Frame-to-frame latching with previous overlap and new delamination pixels
   :width: 760
   :align: center
