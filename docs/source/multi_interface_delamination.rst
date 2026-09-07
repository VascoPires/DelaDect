Multi-Interface Delamination
=============================

Everything on the :doc:`edge_delamination` and :doc:`diffuse_delamination`
pages detects damage on a single interface. In a laminate with more than two
plies, delamination can also develop at more than one interface over the
course of a tensile test. Deladect is able to detect delamination in multiple
interfaces. 

Consider a symmetric laminate such as :math:`[\pm \theta /90^\circ]_s`. From a single
frame, DelaDect only sees a distribution of pixel intensity. There is no 
way to tell, from one image alone, which physical interface a dark region
belongs to. What carries that information is the sequence of frames: if
damage tends to appear at one interface first and only later develops in
another one, then watching *where new darkening shows up relative to what is
already damaged* is enough to tell the interfaces apart. This method is
ideal for specimens such as the "Sample-3" available in the examples folder.

That is the idea behind multi-interface delamination. Interfaces are given an
explicit order, the first (primary) interface is detected
exactly as in :doc:`edge_delamination`, and later darkening is checked inside
an already established primary region.

.. figure:: _static/multi_interface_delamination/interface_order.png
   :alt: Ordered three-ply cross-section showing the primary 90/-theta interface before the secondary -theta/+theta interface
   :width: 100%
   :align: center

   The interface order for a :math:`[\pm \theta /90^\circ]_s` laminate.

.. note::

   This whole method relies on the assumption that damage
   at a given interface generally appears before damage at the next interface
   in the supplied order. This is often the case for laminates such as
   :math:`[\pm \theta /90^\circ]_s` with one interface dominant in the
   initial stages of the test. Therefore it comes to user judgement to
   decide if the results are valid or not. 

:meth:`~deladect.detection.delamination.EdgeDetector.detect_edge_multi`
extends the same primary edge algorithm from :doc:`edge_delamination` to an
ordered interface list of any length. The first (primary) interface
accumulates exactly as described on that page. The additional interfaces are
attributed recursively. That is, a pixel only becomes damage at interface *n* 
once it is both (a) classified in the secondary pass and (b) already covered 
by the mask established at interface *n-1*. For example, for a specimen 
with three interfaces, the first detected delamination is classified 
as primary. For example, for a specimen with three interfaces, 
the first detected delamination is classified as primary. 
Then used to detect further delamination. 
If this new delamination occurs in a region where primary delamination 
has already been detected, it is associated with the second interface. 
The same procedure is repeated for the third interface: further 
delamination is associated with the third interface if it occurs 
in a region where secondary delamination has already been detected.



Normalization for multi-interface detection
------------------------------

In order to destinguish damage between interfaces, a relative normalization
using a somewhat recent frame is required. Otherwise, the same damage would
be detected and incorrectly categorized. Hence, if the pre-processed images
are not ptovided to ``detect_edge_multi``, the rolling-median normalization
is performed as the default behaviour. For more information about the
normalization, go to :doc:`Image_pre_processing`.

- The primary pass uses a **static**-reference normalization, matching
  :meth:`~deladect.detection.delamination.EdgeDetector.detect_primary`.
- The additional pass need a **rolling-median**-reference cache
  instead. A static reference stops highlighting change once a region has
  already darkened, but this check specifically needs to detect *further*
  change happening inside an area the primary pass has already flagged. A
  rolling reference stays sensitive to that change. 


Candidate to detection
------------------------------------------------------

For each frame and each interface beyond the primary, the algorithm:

1. Thresholds the rolling-median-preprocessed frame to generate a binary
   candidate mask, using the same thresholding procedure as the primary pass.
   The difference is that the rolling-median reference shows relative new damage
   in darkening in regions where primary delamination has already formed.

2. Intersects this candidate mask with the **primary interface's
   mask**, read `reference_window` frames earlier. This uses an already
   settled parent region rather than the still-growing edge of the current
   parent mask.

3. Keeps only pixels that remain connected to the free edge.

4. Accumulates the accepted pixels into the running mask for that interface.

.. figure:: _static/multi_interface_delamination/secondary_attribution.png
:width: 100%
:align: center

Example of how delamination is attributed to different interfaces. 
A rolling-reference candidate is eligible for a
given interface only inside the settled mask of the previous interface and
the intersection must remain connected to the specimen edge and is then 
latched over time similar to before.


See :doc:`examples/delamination_multi_interface` for a runnable script and
notebook, and :doc:`detection` for the full API.
