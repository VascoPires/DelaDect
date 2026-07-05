Shift Correction GUI
====================

Accurate crack detection starts with well-aligned image sequences. The auxiliary
script in ``aux_scripts/shift_correction`` provides a lightweight GUI for
correcting marker-based shift distortion before frames reach DelaDect.

When to use it
--------------
Use the GUI whenever raw footage shows translational drift or camera jitter that
would otherwise produce false crack detections. The tool addresses limitations
in the shift correction bundled with CrackDect for static experiments.

Dependencies
------------
Create a virtual environment and install the libraries listed below (versions
match the setup used by the authors in production):

- ``matplotlib`` 3.7+
- ``numpy`` 1.24+
- ``Pillow`` 9.4+
- ``scipy`` 1.11+
- ``scikit-image`` 0.20+

All images for a run must live in a single folder and contain visible markers.

Using the GUI
-------------
1. Launch the tool from the script directory:

   .. code-block:: bash

      python shift_correction.py

   or via installed console entry point:

   .. code-block:: bash

      shift_correction

2. Open the first frame via ``File -> Open First Image``.
3. Choose the destination for the corrected frames with ``File -> Save Images In``.
4. Mark each reference marker using ``Ctrl + Left Click`` (``Command`` on macOS). Remove a
   point with ``Shift + Left Click``.
5. Trigger ``File -> Perform Shift Correction`` and monitor the console for progress.

Fine-tuning
-----------
The GUI exposes several processing parameters:

- **Step size (``n``)** - skip frames when set to values greater than 1.
- **Threshold value** - adjusts marker detection sensitivity (start around 30-50).
- **Gaussian filter** - smooths noisy images so the marker centroid is easier to track.
- **Median filter** - reduces local inconsistencies inside each marker.

Outputs
-------
The application writes shift-corrected frames to your target folder and can
optionally generate diagnostic plots for marker tracking (see
``aux_scripts/shift_correction/media`` for examples).

Typical output structure:

- ``<output>/<specimen>/shift_corrected/*.bmp``
- ``<output>/<specimen>/plots/*.png`` (when plotting is enabled)
- ``<output>/<specimen>/strain_data.csv`` (when strain evaluation is enabled)

Integrating with DelaDect
-------------------------
Point your :class:`~deladect.specimen.Specimen` paths (``path_full``/``path_middle`` etc.) to
these corrected folders and proceed with the workflows described in
:doc:`../examples/getting_started` and :doc:`../detection`.
