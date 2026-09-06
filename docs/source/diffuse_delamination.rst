Diffuse Delamination
====================

Diffuse delamination is assembled crack by crack. For each crack
in the specimen, DelaDect creates a local delamination mask around that crack
and places the mask back at the crack's position in the full image.

.. figure:: _static/delamination/diffuse_mask_assembly.png
   :width: 100%
   :align: center

   Assembly of the diffuse delamination masks. Each ROI around 
   a detected crack is evaluated independently.

The full mask is the logical union of the individual crack masks.

Running diffuse detection
-------------------------

Diffuse detection needs a configured specimen and interface, together with
the cracks detected in each frame. The detector returns one assembled,
full-frame mask per frame.

.. code-block:: python

   from deladect.detection import DelaminationDetector, crack_analysis

   cracks = crack_analysis(specimen, save_cracks=True)
   detector = DelaminationDetector(specimen, interface)

   diffuse_result = detector.diffuse.diffuse_delamination(
       cracks=cracks,
       save_overlays=True,
       params={
           "diffuse_dx": 40.0,
           "diffuse_dy": 10.0,
           "window_diffuse": (30, 30),
       },
       progress=True,
   )

   diffuse_masks = diffuse_result["masks"]

The parameter values are measured in pixels and should be matched to the
image scale:

- ``diffuse_dx`` is the half-width of the local region perpendicular to a
  crack.
- ``diffuse_dy`` extends the local region beyond both crack ends.

