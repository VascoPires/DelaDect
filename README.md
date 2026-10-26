<p align="center">
  <img src="https://raw.githubusercontent.com/VascoPires/DelaDect/main/docs/source/deladect_logo_white.svg" alt="DelaDect logo" width="600">
</p>

<h3 align="center">Quantitative crack and delamination detection for fiber-reinforced polymers</h3>

<p align="center">
  <a href="https://deladect.readthedocs.io">
    <img src="https://img.shields.io/badge/docs-online-blue?logo=readthedocs" alt="Documentation">
  </a>
  <a href="https://mybinder.org/v2/gh/VascoPires/DelaDect/HEAD">
    <img src="https://mybinder.org/badge_logo.svg" alt="Binder">
  </a>
  <a href="https://pypi.org/project/deladect/">
    <img src="https://img.shields.io/pypi/v/deladect?logo=pypi&logoColor=white" alt="PyPI">
  </a>
  <a href="https://doi.org/10.2139/ssrn.7404666">
    <img src="https://img.shields.io/badge/DOI-10.2139%2Fssrn.7404666-blue?logo=doi" alt="Preprint DOI">
  </a>
  <a href="https://github.com/VascoPires/DelaDect/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/VascoPires/DelaDect" alt="License">
  </a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/VascoPires/DelaDect/main/docs/source/_static/sample5_sequence.gif" alt="DelaDect visualization" width="720">
</p>

DelaDect is a Python package for quantitative damage analysis in fiber-reinforced polymers. Out of the box it is able to perform crack and delamination detection based on image sequences.

## Highlights

- Detects matrix cracks in image sequences.
- Separates delamination connected to a free edge from diffuse damage around tracked cracks.
- Retains detected damage between frames and reports its area over time.
- Supports the multi-interface edge-delamination.

## Installation

DelaDect requires **Python 3.10** (any 3.10.x release). Use a fresh environment so its dependencies remain separate from other projects. NumPy is restricted to versions below 2.0 for compatibility with CrackDect.

Until DelaDect is published on PyPI, install it from source:

```bash
conda create -n deladect python=3.10 pip
conda activate deladect
git clone https://github.com/VascoPires/DelaDect.git
cd DelaDect
python -m pip install .
```

For development, use `python -m pip install -e ".[dev]"` instead. Check the installation with:

```bash
python --version
python -c "import deladect; print(deladect.__version__)"
```

## Prerequisites

Python dependencies are installed automatically via `pip`, with version constraints defined in `pyproject.toml`:

- [CrackDect](https://github.com/mattdrvo/CrackDect) >= 0.2
- scikit-image >= 0.18.1, < 0.23
- NumPy >= 1.23.5, < 2 
- SciPy >= 1.10.0, < 1.13
- Pandas >= 1.3.5, < 2.2
- Matplotlib >= 3.7.5
- Pillow >= 8.4.0

The shift-correction GUI also requires Tk and a graphical desktop. If Tk is missing from the Conda environment, install it with `conda install tk`.

## Quick Start

For a full first run including delamination detection, start with the [Getting Started guide](https://deladect.readthedocs.io/en/latest/examples/getting_started.html). Furthermore, several examples and explanations about the methodology are also available in the [documentation](https://deladect.readthedocs.io).

## Shift Correction

The package includes a GUI for correcting image shifts before analysis. Launch it from the activated environment:

```bash
shift_correction
```

See the [Shift Correction guide](https://deladect.readthedocs.io/en/latest/shift_correction.html) for image preparation and GUI instructions.

## Repository Layout

```text
src/deladect/         Core package
  cli/                shift_correction entry point
  detection/          Crack and delamination detection
  io/                 IO utilities
  specimen.py         Specimen and laminate classes
  utils.py            Image and geometry helpers
docs/                 Documentation
example_images/       Sample images and annotations used by the examples
notebooks/            Runnable analysis notebooks
pyproject.toml        Packaging and dependency configuration
```

Generated analysis outputs are written to `results/`, which is excluded from
version control. Notebook outputs are cleared before committing; run the
notebooks to regenerate their figures. See [docs/README.md](docs/README.md)
for local documentation build instructions.

## Citing DelaDect

If you use DelaDect in your work, please cite the preprint:

```text
Pires, V. D. C., Rettl, M., Pletz, M., Schuecker, C. DelaDect: Optical
delamination detection in fiber-reinforced polymers. Available at SSRN:
https://ssrn.com/abstract=7404666, http://dx.doi.org/10.2139/ssrn.7404666
```

```bibtex
@misc{pires_deladect,
  author       = {Pires, Vasco D. C. and Rettl, Matthias and Pletz, Martin and Schuecker, Clara},
  title        = {{DelaDect}: Optical delamination detection in fiber-reinforced polymers},
  year         = {2026},
  howpublished = {SSRN preprint},
  url          = {https://ssrn.com/abstract=7404666},
  doi          = {10.2139/ssrn.7404666}
}
```

## Authors And License

DelaDect is maintained by Vasco D. C. Pires.

Licensed under the AGPL-3.0 License. See `LICENSE` for details.
