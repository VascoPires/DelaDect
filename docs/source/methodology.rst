Delamination Detection Methodology
==================================

Edge and diffuse delamination are both reached through a single
:class:`~deladect.detection.delamination.DelaminationDetector` per
``(specimen, interface)`` pair. Then two other detectors are available inside
the same object, for the two delamination modes:
``detector.edge`` and
``detector.diffuse``.

Detection modes
---------------

DelaDect distinguishes three delamination modes, each documented on its own page:

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Edge delamination
      :link: edge_delamination
      :link-type: doc

      Performs delamination detection and ensures 
      edge-connectivity.

   .. grid-item-card:: Diffuse delamination
      :link: diffuse_delamination
      :link-type: doc

      Delamination performed in ROI around cracks.

   .. grid-item-card:: Multi-interface delamination
      :link: multi_interface_delamination
      :link-type: doc

      The delamination methodology is performed
      in multiple interfaces.

.. toctree::
   :maxdepth: 1
   :hidden:

   edge_delamination
   diffuse_delamination
   multi_interface_delamination

Detection sequence
-------------------

Edge and diffuse delamination target different features (delamination
starting on the edge and delamination around cracks), but both are built on the same common
methodology, applied per frame:

1. **Minimum history**: a cumulative minimum over the stack so the image
   stack only gets darker. See :doc:`Image_pre_processing`.
2. **Normalization**: division by a reference frame. See
   :doc:`Image_pre_processing`.
3. **Max/min filtering**: morphological closing (a max filter then a min
   filter) with the window size, which suppresses thin 
   structures like cracks while preserving diffuse and broad delamination.
4. **Sharpening and Gaussian smoothing**: unsharp masking widens the
   contrast between delaminated and intact regions, then Gaussian
   smoothing for noise supressing.
5. **Constant scaling**: maps intensities to a fixed range so thresholding
   behaves consistently across frames.
6. **Thresholding**: k-means (k=2) turns the processed frame into a binary
   candidate mask, with Otsu as a fallback if k-means does not converge.
7. **Morphological closing**: fills small holes and bridges narrow gaps
   left by thresholding.
8. **Accumulation**: union with the previous frame's mask, so detected
   damage only grows and single-frame flicker is rejected.

.. figure:: _static/methodology/workflow_edge.png
   :alt: Eight-step delamination detection workflow illustrated on an edge-delamination example, from minimum history through accumulation
   :width: 320
   :align: center

   The eight-step procedure applied to an edge-delamination example. 
   Steps 1–5 filter the image, steps 6–7 create and 
   clean the binary mask, and step 8 compares it with the 
   previous frame.

Combining the two
------------------

:meth:`~deladect.detection.delamination.DelaminationDetector.detect_both_delaminations`
runs both pipelines together and resolves any overlap
between the two modes favouring the edge delamination. This means that
if diffuse and edge delamination is found in the same place, that region
is classified as edge delamination.


See also
--------

- :doc:`detection` for the callable API and default parameter values.
- :doc:`image_operations` for the pixel-scale filtering steps behind edge detection.
