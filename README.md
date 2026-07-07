<p align="center">
  <img src="docs/source/deladect_logo_white.svg" alt="DelaDect logo" width="340">
</p>

<p align="center">
[![Documentation](https://img.shields.io/badge/docs-online-blue?logo=readthedocs)](https://deladect.readthedocs.io)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/VascoPires/DelaDect/HEAD)
[![License](https://img.shields.io/github/license/VascoPires/DelaDect)](LICENSE)
</p>

<p align="center">
  <img src="docs/source/_static/sample5_sequence.gif" alt="DelaDect visualization" width="720">
</p>

DelaDect is a Python package for quantitative damage analysis in fiber-reinforced polymers. Out of the box it is able to perform crack and delamination detection based on image sequences.

## Installation

DelaDect was coded in in `Python 3.10` with the Prerequisites defined in [Prerequisites](#prerequisites). Therefore, it is highly recomended for a new Python environment to be created.

To install from source:

```bash
git clone https://github.com/VascoPires/DelaDect.git
cd DelaDect
pip install
```

## Prerequisites

Dependencies are installed automatically via `pip`. The tool uses the following libraries:

- [CrackDect](https://github.com/mattdrvo/CrackDect) 0.2
- scikit-image 0.18+
- NumPy >= 1.19
- SciPy 1.6+
- Matplotlib >= 3.3
- SQLAlchemy 1.3+
- Numba 0.52+
- psutil 5.8+

## Quick Start

For a full first run including delamination detection, start with the [Getting Started guide](https://deladect.readthedocs.io/en/latest/examples/getting_started.html). Furthermore, several examples and explanations about the methodology are also available in the [documentation](https://deladect.readthedocs.io).

## Shift Correction

The package also included a shift-distortion correction for the image stack preparation before any analysis. It is assumed that the images provided to DelaDect have already been properly shift-corrected, as this represents a fundamental step of the methodology. 

```bash
shift_correction --help
```

## Repository Layout

```text
src/deladect/         Core package
  cli/                shift_correction entry point
  detection/          Crack and delamination detection
  io/                 IO utilities
  specimen.py         Specimen and laminate classes
  utils.py            Image and geometry helpers
docs/                 Documentation
pyproject.toml        Packaging and dependency configuration
```

## Authors And License

DelaDect is maintained by Vasco D. C. Pires.

Licensed under the AGPL-3.0 License. See `LICENSE` for details.
