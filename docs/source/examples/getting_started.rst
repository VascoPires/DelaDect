Getting Started
===============

This walkthrough takes you from a fresh clone of DelaDect to your first crack-detection
results using the sample dataset bundled with the repository.

1. Set up a Python environment (Python 3.9 or newer) and install DelaDect in editable mode
   so scripts and notebooks pick up local changes:

   .. code-block:: bash

      python -m venv .venv
      .venv\\Scripts\\activate      # On macOS/Linux use: source .venv/bin/activate
      pip install --upgrade pip
      pip install -e .[dev]

2. Grab the example images (already shift-corrected) located in
   ``example_images/sample-1``. The folder layout matches what the
   :class:`~deladect.detection.Specimen` constructor expects (``cut``, ``middle``, ``upper``,
   ``lower`` and an optional ``experimental_data.csv``).

3. Run the smoke test in ``tests/test_DelaDect_crack_detection.py`` to generate reports in
   ``tests/test_results``. The test suite covers cross-ply analysis, post-processing, and
   plotting in one go:

   .. code-block:: bash

      pytest tests/test_DelaDect_crack_detection.py -k "basic_crack_detection"

4. Open the generated artefacts. Each run creates timestamped subfolders with crack
   overlays, CSV exports, and pickled crack catalogues that you can reuse in notebooks or
   comparison plots.

Where to next?
--------------
- :doc:`../detection` introduces every high-level method on
  :class:`~deladect.detection.Specimen`.
- :doc:`shift_correction` shows how to prepare aligned image stacks with the auxiliary GUI.
- :doc:`crack_detection` expands on the testing workflow and demonstrates how to reuse the
  output for custom reporting.
