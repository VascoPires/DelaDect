<p align="center">
  <img src="docs/source/deladect_logo.svg" alt="DelaDect logo" width="340">
</p>

<h1 align="center">DelaDect</h1>

<p align="center">
  Optical crack and delamination detection for translucent fiber-reinforced composites.
</p>

<p align="center">
  Built for quantitative damage analysis, reproducible exports, and presentation-ready visual outputs.
</p>

<p align="center">
  <a href="https://deladect.readthedocs.io">Documentation</a>
  ·
  <a href="https://github.com/VascoPires/DelaDect">Repository</a>
</p>

![DelaDect sample sequence](sample5_sequence.gif)

DelaDect is a Python package for quantitative damage analysis in optical image stacks of fiber-reinforced polymers. It extends [CrackDect](https://github.com/mattdrvo/CrackDect) with higher-level workflows for crack detection, edge and diffuse delamination detection, reusable preprocessing caches, specimen persistence, and shift-corrected image preparation.

## Overview

DelaDect is organized around a practical two-stage workflow:

- Stage I: crack detection for cross-ply and angle-ply laminates.
- Stage II: edge and diffuse delamination detection with reproducible mask, overlay, and metric exports.
- Shift correction and strain evaluation for preparing stable image sequences before damage analysis.
- Save/reload utilities for preserving specimen metadata and result references across runs.

The package is designed for research workflows where the same dataset needs to support both method development and clean visual communication in reports, papers, and conference presentations.

## Why DelaDect

- Quantifies crack density and delamination progression from image sequences.
- Supports reusable preprocessing caches to speed up iterative tuning.
- Exports report-friendly artefacts including overlays, NPZ masks, and CSV metrics.
- Includes a `shift_correction` GUI/CLI workflow for static and fatigue image stacks.
- Provides documentation and example workflows for first-time and advanced users.

## Installation

DelaDect supports `Python >= 3.9`, with `Python 3.10` recommended.

Install from PyPI:

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux
pip install --upgrade pip
pip install deladect
```

Or install from source:

```bash
git clone https://github.com/VascoPires/DelaDect.git
cd DelaDect
pip install -e .[dev]
```

## Quick Start

```python
from pathlib import Path

from deladect.detection import crack_eval_crossply
from deladect.specimen import Specimen

data_root = Path("my_specimen_images")

specimen = Specimen.from_cross_ply(
    name="my-specimen",
    scale_px_mm=41.03,
    path_full=str(data_root),
    sorting_key="_sc",
    image_types=["png"],
    auto_init_stacks=True,
    results_root="results",
    avg_crack_width_px=8.0,
)

crack_results = crack_eval_crossply(
    specimen,
    export_images=True,
    save_cracks=True,
    results_dir="results",
)

print(crack_results.keys())
```

For a full first run including delamination detection and save/reload flows, start with the [Getting Started guide](https://deladect.readthedocs.io/en/latest/examples/getting_started.html).

## Shift Correction

The package exposes a dedicated shift-correction entry point for preparing aligned image stacks before crack or delamination analysis.

```bash
shift_correction --help
```

This workflow is especially useful for static and fatigue experiments where small acquisition drift would otherwise degrade downstream detection quality.

## Documentation Roadmap

- Start with [Getting Started](https://deladect.readthedocs.io/en/latest/examples/getting_started.html) for a first full workflow.
- Use [Methodology](https://deladect.readthedocs.io/en/latest/methodology.html) for the processing logic and assumptions.
- Use [Detection](https://deladect.readthedocs.io/en/latest/detection.html) and [Delamination](https://deladect.readthedocs.io/en/latest/delamination.html) for API and algorithm details.
- Use [Results Storage](https://deladect.readthedocs.io/en/latest/results_storage.html) when integrating outputs into downstream scripts.

To build the docs locally:

```bash
pip install sphinx sphinx-rtd-theme
sphinx-build -b html docs/source docs/build/html
```

## Repository Layout

```text
src/deladect/         Core package
  cli/                shift_correction entry point
  detection/          Crack and delamination detection
  io/                 Save/load utilities
  specimen.py         Specimen and laminate metadata model
  utils.py            Image and geometry helpers
docs/                 Sphinx documentation
sample5_sequence.gif  Sample animated sequence preview
pyproject.toml        Packaging and dependency configuration
```

## Authors And License

DelaDect is maintained by Vasco D. C. Pires.

Licensed under the MIT License. See `LICENSE` for details.
