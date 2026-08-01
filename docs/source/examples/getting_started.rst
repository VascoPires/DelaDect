01 - Getting Started
====================

This first example aims to show a complete DelaDect workflow, including
crack detection, diffuse delamination detection, and edge delamination detection. 

A `Binder <https://mybinder.org/v2/gh/vascodcpires/deladect/main?labpath=notebooks/getting_started.ipynb>`_,
notebook which serves as a companion to this example is available in the repository and can be run 
without installation.

The first step of any analysis in DelaDect is to create a specimen object. The specimen 
object serves as a container for all the relevant information about the analysis.
The specimen is built by constructing a 
:class:`~deladect.specimen.Specimen` object and calling
:meth:`~deladect.specimen.Specimen.add_ply` to add plies and calling 
:meth:`~deladect.specimen.Specimen.add_interface` for the interfaces. 
Here plies are associated with the crack detection and in which orientation
the cracks are aligned and interfaces are associated with the delamination detection.
For a diffuse-edge delamination detection, defining an interface is not mandatory, however
it is mandatory for multi-interface delamination detection (see :doc:`delamination_multi_interface`).

There are two ways of supplying the specimen images to DelaDect: either by providing a single full-frame 
stack or by providing three separate stacks for the upper, middle, and lower regions of the specimen (here,
assuming that the specimen is oriented horizontally).

.. figure:: ../_static/examples/split_image_vs_full_image.png
   :alt: Full-frame image stack compared with separate upper, middle, and lower image stacks
   :width: 100%
   :align: center

The benefit of providing separate stacks is that the edge detection 
can be constrained to the upper and lower regions and diffuse delamination (and crack detection) 
can be constrained to the middle region. This can be useful when edge and diffuse delamination 
end up connected in a given specimen and edge delamination takes over (see how edge delamination is
computed in :doc:`../delamination`).

In the following snippet, a specimen object is created with three separate stacks and the plies and inerface
are added. As a reminder, an assumption of the tool are that cracks are oriented in the same direction as the 
plies. More details about the crack detection can be found in :doc:`../detection`.

.. code-block:: python

   # imports
   from pathlib import Path

   from deladect.detection import DelaminationDetector, crack_analysis
   from deladect.io import save_specimen
   from deladect.specimen import Specimen
   
   # specimen object
   specimen = Specimen(
       name="01-getting-started",
       scale_px_mm=41.03328366,
       path_full=str(data_root / "full"),
       path_upper_border=str(data_root / "upper"),
       path_middle=str(data_root / "middle"),
       path_lower_border=str(data_root / "lower"),
       image_types=["png"],
       avg_crack_width_px=14.0,
   )
   
   # add plies to the specimen and defines orientation of the cracks
   ply90 = specimen.add_ply(name="ply_90", orientation_deg=90.0, avg_crack_width_px=14.0, min_crack_length_px=90.0)
   ply0 = specimen.add_ply(name="ply_0", orientation_deg=0.0, avg_crack_width_px=14.0, min_crack_length_px=90.0)

   interface = specimen.add_interface(name="i0", upper_ply_index=ply90, lower_ply_index=ply0)

After the specimen is created, the crack detection can be run with :func:`crack_analysis`. This step is only required
if diffuse delamination is required for the analysis.

.. code-block:: python

   crack_results = crack_analysis(
       specimen,
       export_images=True,
       background=True,
       save_cracks=True,
   )

For the crack analysis, the main parameters used are ``avg_crack_width_px`` and ``min_crack_length_px``,
and those are defined in the ply object and are inherited from
`CrackDect <https://crackdect.readthedocs.io/en/latest/index.html>`_ [Drvoderic2022]_.

The output of the function is a structured dictionary with the crack detection results for each orientation. 
The output can be used directly for diffuse delamination, as shown in the following code snippet.

.. code-block:: python

   detector = DelaminationDetector(
       specimen,
       interface,
       save_preprocess_outputs=True,
   )

  # diffuse delamination parameters 
  diffuse_params={
      "window_diffuse": (30, 30),
      "diffuse_dx": 20.0,
      "diffuse_dy": 20.0,
  }

  
  # edge delamination parameters
  edge_params={
      "window_edge": (1, 60),
      "seed_ratio": 0.01,
  }


   # performs both edge and diffuse delamination
   result = detector.detect_both_delaminations(
       cracks=crack_results,
       save_overlays=True,
       save_metrics=True,
       diffuse_params=diffuse_params,
       edge_params=edge_params
   )

The suggested parameters for the delamination detection for this example are the ones presented in the code snippet above:

- ``window_edge`` and ``window_diffuse`` are the sizes of the sliding windows used for the edge and diffuse
  detection, respectively. For a window of size ``(wy, wx)``, the image is passed through a maximum filter
  followed by a minimum filter, both using that window as a rectangular neighbourhood. In practical terms, the
  window size controls how cracks and noise are filtered out from the delamination detection.
- ``diffuse_dx`` and ``diffuse_dy`` are the dimensions of the region of interest (ROI) around each crack used to
  compute the diffuse delamination.
- ``seed_ratio`` is the fraction of rows, starting from the specimen edge, trusted as the initial seed region for
  edge-connected reconstruction. A ratio of 0.01 seeds from the first 1% of rows in each split half.

More information about these and other parameters can be found in :doc:`../delamination`.
The outputs of the delamination detection are shown below.

.. image:: ../_static/examples/getting_started_detection_outputs.png
   :alt: Crack detection and classified edge and diffuse delamination outputs for the Getting Started example
   :width: 100%
   :align: center  




.. image:: ../_static/examples/getting_started_metric_plots.png
   :alt: Crack density and detected delamination plotted against frame number
   :width: 100%
   :align: center

The complete orientation-keyed result from :func:`crack_analysis` can be passed
directly to combined or standalone diffuse detection. DelaDect validates that
the orientation outputs have equal frame counts and merges every orientation
present in the result. To use only selected crack families, filter them during
analysis:

.. code-block:: python

   crack_results = crack_analysis(specimen, orientations=[90.0])

Static-reference preprocessing is selected automatically for this combined,
single-interface workflow. Rolling-median preprocessing is reserved for the
multi-interface edge example. Diffuse regions are evaluated independently for
each frame by default and then latched over time. Pass ``track_cracks=True`` to
enable cross-frame crack association and the tracked diffuse workflow.

Why the full stack is still supplied
------------------------------------

``path_full`` is currently required by :class:`deladect.specimen.Specimen`.
DelaDect does not automatically create ``image_stack_full`` by vertically
joining the upper, middle, and lower stacks during specimen initialization.
If ``path_full`` is omitted, automatic stack initialization raises
``ValueError: Specimen requires a valid path_full to initialise the image stack.``

When all three region paths are supplied, they override where the component
algorithms evaluate damage: cracks use ``middle`` and primary edge detection
uses ``upper`` plus ``lower``. The full stack is still used for combined
preprocessing and as the background for full-frame overlays. Some individual
region detector paths can internally stack the three regions as an overlay
fallback, but this is not a general replacement for ``path_full`` and does not
make the combined Getting Started workflow work without it.

Results to inspect
------------------

- ``results/01-getting-started/cracks/`` contains crack overlays and bundles.
- ``results/01-getting-started/delamination/both/overlays/`` contains the
  classified edge/diffuse overlays.
- ``results/01-getting-started/delamination/both/metrics/frame_metrics.csv``
  contains edge, diffuse, overlap, and combined fractions.
- ``results/01-getting-started/config/specimen.json`` stores the reloadable
  specimen definition and result references.

Full-image comparison: connected edge regions
---------------------------------------------

This section describes a separate, intentionally unconstrained full-image
experiment; it is not output from the split-region Getting Started run above.
In that comparison, frame 0003 illustrates a classification limitation that
occurs when edge detection is run on the full image rather than constrained to
the explicit upper and lower regions. Delamination growing inward from the
upper and lower specimen boundaries has connected into one edge-mask component.
That component touches both boundaries and also occupies part of the specimen
middle, where diffuse delamination may physically be present.

The combined workflow resolves overlap with edge precedence:

.. math::

   M_{\mathrm{diffuse,final}} =
   M_{\mathrm{diffuse,raw}} \cap \neg M_{\mathrm{edge,exclusion}}

Consequently, a diffuse candidate is classified exclusively as edge wherever
the masks overlap. In the unconstrained comparison, 26,032 of 26,058
diffuse-candidate pixels (99.90 percent) overlap the edge-exclusion mask. Only
26 diffuse pixels survive in the complete frame. By contrast, the verified
split-region Getting Started run produces 662,041 diffuse candidates, 5,327
overlapping pixels (0.80 percent), and 656,714 surviving diffuse pixels for
frame 0003. The square-cell diagram below shows the unconstrained mask
relationship over the full specimen height in a representative 600-pixel-wide
region.

.. figure:: ../_static/examples/connected_edge_square_masks.svg
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
The explicit region configuration used by this Getting Started example avoids
that specific failure mode by preventing the edge detector from evaluating the
middle rows.

Continue with :doc:`advanced_options` to control the image regions explicitly.

References
----------

.. [Drvoderic2022] Drvoderic, M., Bender, J. J., Pletz, M., & Schuecker, C. (2022).
   Version 0.2 - CrackDect: Detecting crack densities in images of fiber-reinforced
   polymers. *SoftwareX*, 19, 101198.
   `<https://doi.org/10.1016/j.softx.2022.101198>`_
