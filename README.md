<p align="center">
  <img src="docs/source/deladect_logo_white.svg" alt="DelaDect logo" width="600">
</p>


<p align="center">
  <a href="https://deladect.readthedocs.io">
    <img src="https://img.shields.io/badge/docs-online-blue?logo=readthedocs" alt="Documentation">
  </a>
  <a href="https://mybinder.org/v2/gh/VascoPires/DelaDect/HEAD">
    <img src="https://mybinder.org/badge_logo.svg" alt="Binder">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/github/license/VascoPires/DelaDect" alt="License">
  </a>
</p>


<p align="center">
  <img src="docs/source/_static/sample5_sequence.gif" alt="DelaDect visualization" width="720">
</p>

DelaDect is a Python package for quantitative damage analysis in fiber-reinforced polymers. Out of the box it is able to perform crack and delamination detection based on image sequences.

## Installation

DelaDect was coded in `Python 3.10` with the Prerequisites defined in [Prerequisites](#prerequisites). A fresh Python environment (e.g. `conda create -n deladect python=3.10`) is strongly recommended, since DelaDect pins NumPy below 2.0 and an environment with a newer NumPy already installed can cause import errors (e.g. `No module named 'numpy.core'`).

To install from source:

```bash
git clone https://github.com/VascoPires/DelaDect.git
cd DelaDect
conda create -n deladect python=3.10
conda activate deladect
pip install -e .
```

## Prerequisites

Dependencies are installed automatically via `pip`, pinned in `pyproject.toml`. The tool uses the following libraries:

- [CrackDect](https://github.com/mattdrvo/CrackDect) 0.2
- scikit-image >= 0.18.1, < 0.23
- NumPy >= 1.23.5, < 2 (CrackDect 0.2 relies on a NumPy API removed in 2.0)
- SciPy >= 1.10.0, < 1.13
- Pandas >= 1.3.5, < 2.2
- Matplotlib >= 3.7.5
- Numba (via CrackDect)
- psutil (via CrackDect)

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
