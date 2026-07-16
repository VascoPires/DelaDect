Shift Correction GUI
====================

Introduction
------------
The first step in any evaluation, whether for crack or delamination detection, 
is to ensure that the image sequence is properly aligned. A properly aligned 
sequence means that the same physical point on the specimen appears at the same 
pixel coordinates in every frame.

During mechanical testing, local deformation and strain cause the specimen to move relative to the camera. 
As a result, cracks or delaminations may appear to shift between consecutive images, 
even though they remain at the same physical location on the specimen. 
If left uncorrected, this apparent motion can lead to false detections.

.. image:: _static/shift_correction/l1_cut_images_sequence.gif
   :alt: Raw cut-image sequence from the L1 specimen
   :width: 960
   :align: center

.. image:: _static/shift_correction/l1_shift_correction.gif
   :alt: Real shift-correction example with tracked markers and mesh
   :width: 960
   :align: center

DelaDect ships a lightweight GUI, installed as the
``shift_correction`` console command, that performs shift
distortion correction for a sequence of frames.

Required:

- Specimen image needs to be oriented horizontally with the load direction also horizontal.
- Figures need to be already cut, with the markers clearly visible.
- All the figures for a given run must live in the same folder.

.. warning::
   All images in a run **must share the same dimensions**. Loading a folder
   with mismatched image sizes (width and height) raises an error before any correction is
   attempted. 

   Marker detection assumes **dark markers on a lighter background**. If your markers are
   lighther than the background no points will be detected.


How to Use
----------
1. **Prepare the images.** Make sure all pictures for the sequence are in a
   single folder;

2. **Launch the GUI.** From an environment where DelaDect is installed, run:

   .. code-block:: bash

      shift_correction

   Alternatively, run the module directly from the source tree:

   .. code-block:: bash

      python -m deladect.cli.shift_correction

3. **Open the first image.** With the Shift Correction GUI open, go to
   ``File -> Open First Image`` and select the first frame of the series:

   .. image:: _static/shift_correction/app.png
      :alt: Shift Correction GUI main window
      :width: 720
      :align: center

4. **Set the output folder.** Choose where the shift-corrected images should
   be written via ``File -> Save Images In``.

5. **Mark the points.** Click each marker using ``Ctrl + Left Click``
   (``Command + Left Click`` on macOS). Aim for the center of each marker; a
   misplaced point can be removed with ``Shift + Left Click``:

   .. image:: _static/shift_correction/selection.png
      :alt: Marker point selection in the GUI
      :width: 720
      :align: center

   .. warning::
      Use **at least 4 markers**, and avoid placing any marker within a few
      pixels of the image border.

   .. note::
      A marker that goes undetected or unmatched in a frame is dropped for
      that frame only, not permanently: on the next frame it is re-guessed
      from how the *other* markers moved, and re-attached if a real point is
      found near that guess. The longer a marker stays missing, the less
      reliable that guess becomes, since it is no longer based on the
      marker's own motion. If results look off, check the tracking output
      (see :ref:`shift_correction_outputs`) to confirm every marker was
      tracked consistently across the whole sequence.

   .. warning::
      Between two consecutive processed frames, a marker is only matched if
      it moved less than a fixed search radius (10 px by default). If the specimen
      moves quickly relative to the frame rate, the matching marker can be outside
      of the search radius. In this case, the search radius can be increased in the GUI's settings.

6. **Run the correction.** Trigger ``File -> Perform Shift Correction`` and
   monitor the console for progress.

   .. warning::
      Re-running into the same output folder **overwrites** any existing
      ``shift_corrected``/``tracking`` images and ``strain_data.csv`` without
      confirmation. Use a fresh output folder if you want to keep a previous
      run's results.

Commands
--------
The shortcuts depend on the operating system, but most actions are shared:

**Windows/Linux**

- Add a point: ``Ctrl + Left Click``

**macOS**

- Add a point: ``Command + Left Click``

**Common commands**

- Pan the figure: ``Left Click``
- Zoom in or out: ``Mouse Wheel``
- Delete a point: ``Shift + Left Click``

.. _shift_correction_outputs:

Strain evaluation
-------
The tool also includes a very simple strain evaluation, which is only available 
when **exactly 4 or 8 markers** are selected. 
The strain is computed from the tracked marker coordinates 
and written to a CSV file in the output folder. 

The strain values are expressed in the local coordinate system defined by the markers, 
with the origin at the top-left marker.


Outputs
-------
The application writes shift-corrected frames to the selected output folder.
It also creates a tracking folder with the detected marker coordinates for each
frame. Here it is possible to check if the marker was properly detected. If the marker
is improperly detected, it is likely that the shift correction was unsuccessful. 

.. image:: _static/shift_correction/0105_dic_test.png
   :alt: Tracked marker points overlaid on a sample frame
   :width: 720
   :align: center

More information about the shift correction can be found
`here <https://crackdect.readthedocs.io/en/latest/shift_correction.html>`_.
The current implementation estimates a smooth displacement field from the
tracked marker coordinates using :class:`scipy.interpolate.RBFInterpolator`
with a thin-plate-spline kernel. Concretely, for marker positions
:math:`\mathbf{x}_i^{(0)}` in the reference image and
:math:`\mathbf{x}_i^{(n)}` in a later frame, the interpolator constructs a
warp :math:`T(\mathbf{x})` such that
:math:`T(\mathbf{x}_i^{(0)}) \approx \mathbf{x}_i^{(n)}` while remaining smooth
between markers. The corrected frame is then obtained by evaluating the
inverse mapping and resampling the image intensities onto the reference grid,
so all frames are expressed in the same coordinate system. The first animation
below shows the raw image sequence, and the second shows the corresponding
shift-correction idea on a real sequence.



Typical output structure:

- ``<output>/<specimen>/shift_corrected/*.bmp``
- ``<output>/<specimen>/plots/*.png`` (when plotting is enabled)
- ``<output>/<specimen>/strain_data.csv`` (when strain evaluation is enabled)

.. warning::
   Strain evaluation only supports exactly **4 or 8** marked points, placed
   in a fixed counter-clockwise order:

   - **4 points:** top-left, bottom-left, bottom-right, top-right.
   - **8 points:** top-left, middle-left, bottom-left, bottom-middle,
     bottom-right, middle-right, top-right, top-middle.

   Any other point count raises an error, but the point *order* is **not**
   validated — marking the right number of points in the wrong order will
   silently produce incorrect strain values.

.. _shift_correction_fine_tuning:

Fine-tuning
-----------
The GUI exposes several processing parameters to adjust marker detection and
tracking:

- **Step size (``n``)**: number of images to skip during evaluation. With
  ``n = 1`` every image is processed; with ``n = 2`` only every second image
  is considered, and so on.
- **Threshold value**: the maximum pixel intensity treated as "black" in the
  grayscale image, which drives marker detection. Lower values focus on
  darker pixels but may fail to detect markers if set too low; higher values
  include more pixels but risk detecting spurious points if set too high. A
  good starting point is **10-30** for very dark markers, though lighting
  conditions can shift this considerably.
- **Gaussian filter**: smooths the image, which helps when markers are
  poorly defined and multiple points are detected per marker. A recommended
  range is **1-5**.
- **Median filter**: averages out local inconsistencies inside a marker,
  improving point detection accuracy.

These suggested values work well for the example images above, but should be
tuned for your own data.

Integrating with DelaDect
--------------------------
Point your :class:`~deladect.specimen.Specimen` paths (``path_full``,
``path_middle``, etc.) to the shift-corrected output folders and continue
with the workflows described in :doc:`examples/getting_started` and
:doc:`detection`.

The math behind the warp
-------------------------
The marker-based correction above is a concrete case of a more general
tool: fitting a smooth spatial transform :math:`T(\mathbf{x})` from a
handful of point correspondences, then resampling an image through it with
:class:`scipy.interpolate.RBFInterpolator`. The animation below applies the
same thin-plate-spline machinery to a synthetic image and a much larger,
sheared/rotated displacement field, to make the warp itself easier to see.

.. image:: _static/shift_correction/rbf_interpolator_warp.gif
   :alt: RBFInterpolator warping a synthetic image from an initial to a final point layout
   :width: 960
   :align: center
