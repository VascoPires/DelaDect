# DelaDect

Delamination and crack detection toolkit for transparent composite laminates. DelaDect
extends [CrackDect](https://github.com/mattdrvo/CrackDect) with higher-level workflows,
report-ready exports, and a shift-correction GUI tailored for static and fatigue tests.

## Highlights
- End-to-end automation for crack density, spacing, and visual overlays via the
  `deladect.detection.Specimen` workflow.
- Flexible image-stack backends (in-memory or SQL) selected automatically from your memory
  budget.
- Rich utilities for organising output folders, serialising intermediate crack catalogues,
  and post-processing results.
- Optional GUI-based shift correction (`aux_scripts/shift_correction`) for preparing aligned
  image sequences.
- Comprehensive example suite in `tests/test_DelaDect_crack_detection.py` that doubles as a
  playbook for cross-ply and +/- theta laminates.

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate      # On macOS/Linux use: source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
```

The editable install keeps the package in sync while you iterate on notebooks, scripts, and
documentation. See the Sphinx guide in `docs/` (or the published site once built) for
additional platform-specific instructions.

## Quick start
1. Ensure the example dataset under `example_images/sample-1` is available (images are
   already shift-corrected).
2. Run the smoke tests to produce crack overlays, CSV exports, and pickled crack data in
   `tests/test_results/<timestamp>`:
   ```bash
   pytest tests/test_DelaDect_crack_detection.py
   ```
3. Explore the generated artefacts in your favourite viewer or notebook. They provide
   concrete references for `Specimen.crack_eval`, `crack_eval_crossply`, and the
   `crack_filtering_postprocessing` pipeline.

## Documentation
- Build the HTML documentation locally:
  ```bash
  sphinx-build -b html docs/source docs/build/html
  ```
- Key entry points:
  - `docs/source/detection.rst` - API guide and workflow overview.
  - `docs/source/utils.rst` - geometry helpers and folder utilities.
  - `docs/source/examples/` - narrated examples (getting started, shift correction, crack
    detection, image handling).

## Shift correction GUI
The auxiliary GUI that accompanies the paper lives under `aux_scripts/shift_correction`.
Launch it with:
```bash
python shift_correction.py
```
Follow the on-screen instructions to mark reference points, tweak thresholds, and export
aligned frames ready for DelaDect.

## Repository layout
```
deladect/                 Core package (detection workflows and utilities)
aux_scripts/              Supporting tools (shift correction, image prep)
docs/                     Sphinx documentation sources
example_images/           Sample dataset used in the tests and tutorials
tests/                    Pytest suite demonstrating end-to-end scenarios
```

## Contributing
Issues and pull requests are welcome. Before submitting changes, run the test suite and
HTML doc build commands above to ensure everything stays green.

## License
Licensed under the MIT License. See `LICENSE` for details.
