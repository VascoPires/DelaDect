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

DelaDect distinguishes two delamination modes, each documented on its own page:

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Edge delamination
      :link: edge_delamination
      :link-type: doc

      Damage connected to a specimen free edge: directional reconstruction
      and frame-to-frame latching.

   .. grid-item-card:: Diffuse delamination
      :link: diffuse_delamination
      :link-type: doc

      Damage sought locally around tracked transverse cracks, using a
      per-crack baseline to isolate new darkening.

   .. grid-item-card:: Multi-interface delamination
      :link: multi_interface_delamination
      :link-type: doc

      Attributing later damage to the correct, deeper interface in
      laminates with more than two plies. Edge-only.

.. toctree::
   :maxdepth: 1
   :hidden:

   edge_delamination
   diffuse_delamination
   multi_interface_delamination

Combining the two
------------------

:meth:`~deladect.detection.delamination.DelaminationDetector.detect_both_delaminations`
runs both pipelines together and resolves any overlap
between the two modes favouring the edge delamination.

Known limitation: connected edge regions in a full-image run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section describes a separate, intentionally unconstrained full-image
experiment; it is not output from the split-region :doc:`examples/getting_started`
run. In that comparison, frame 0003 illustrates a classification limitation
that occurs when edge detection is run on the full image rather than
constrained to explicit upper and lower regions. Delamination growing inward
from the upper and lower specimen boundaries has connected into one edge-mask
component. That component touches both boundaries and also occupies part of
the specimen middle, where diffuse delamination may physically be present.

The combined workflow resolves overlap with edge precedence: a diffuse
candidate is classified exclusively as edge wherever the masks overlap. In
the unconstrained comparison, 26,032 of 26,058 diffuse-candidate pixels
(99.90 percent) overlap the edge-exclusion mask. Only 26 diffuse pixels
survive in the complete frame. By contrast, the verified split-region
Getting Started run produces 662,041 diffuse candidates, 5,327 overlapping
pixels (0.80 percent), and 656,714 surviving diffuse pixels for frame 0003.
The square-cell diagram below shows the unconstrained mask relationship over
the full specimen height in a representative 600-pixel-wide region.

.. figure:: _static/examples/connected_edge_square_masks.svg
   :alt: Connected edge delamination limitation in Sample-1 frame 0003
   :width: 100%
   :align: center

   Sample-1 frame 0003, shown as 30-by-30-pixel square cells over the full
   specimen height. Panel 1 isolates the edge component that touches both the
   upper and lower boundaries. Panel 2 shows diffuse candidates as green
   vertical stripes over the pale-red edge mask. In panel 3, those stripes are
   red because edge precedence overwrites the overlapping diffuse label.

This demonstrates ambiguity in the classification rule, not proof of the
physical damage class at every pixel. Once edge regions connect through the
middle, DelaDect cannot represent coexisting edge and diffuse labels there.
The explicit region configuration used by the Getting Started example avoids
that specific failure mode by preventing the edge detector from evaluating
the middle rows.

See also
--------

- :doc:`detection` for the callable API and default parameter values.
