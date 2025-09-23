Shift Correction GUI
====================

Accurate crack detection starts with well-aligned image sequences. The auxiliary script in
``aux_scripts/shift_correction`` provides a lightweight GUI for correcting marker-based
shift distortion before the frames ever reach DelaDect.

When to use it
--------------
Use the GUI whenever the raw footage exhibits translational drift or camera jitter that
would otherwise lead to false crack detections. The tool was designed to overcome the
limitations of the shift correction bundled with CrackDect for static experiments.

Dependencies
------------
Create a small virtual environment and install the libraries listed below (versions are the
ones the authors use in production):

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
The application writes shift-corrected frames into your target folder and optionally
produces diagnostic plots for marker tracking (see ``aux_scripts/shift_correction/media`` for
examples).

Integrating with DelaDect
-------------------------
Point your :class:`~deladect.detection.Specimen` paths (``path_cut``/``path_middle`` etc.) to
these corrected folders and proceed with the workflows described in
:doc:`../examples/getting_started` and :doc:`../detection`.
