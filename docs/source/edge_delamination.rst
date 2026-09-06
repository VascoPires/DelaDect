Edge Delamination
=================

Edge delamination is damage that remains connected to a specimen free edge.
For this, the image is split into upper and lower specimen halves (or those regions
are provided by the user) and each one is processed independently. 

Detection sequence
------------------

For each frame, :meth:`deladect.detection.delamination.EdgeDetector.detect_primary`
applies filters, unsharp masking, directional Gaussian
smoothing, constant scaling, thresholding, and morphological closing (see
the workflow figure in :doc:`methodology`). After those operations the free edge
reconstruction is done which ensures that the delamination front is always
connected to the edge.

Free-edge reconstruction
------------------------

Thresholding can produce masks that are not
connected to the edge. Both reconstruction modes remove these
regions. In `"directional"` mode, a pixel is accepted if the
previous row contains an accepted pixel within the specified
horizontal drift range. For example, a drift of three allows
a shift of up to `+-3` columns. In `"columnwise"` mode, the
connection must be in the same column. Neither mode can skip
empty rows.

The animations below use the same real candidate mask and seed row.

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

See :doc:`detection` for the full API reference, including default values.

Detecting damage at more than one interface, in laminates with more than two
plies, is covered separately in :doc:`multi_interface_delamination`.
