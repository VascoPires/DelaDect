DelaDect: Optical Delamination Detection in Fiber-Reinforced Polymers
=====================================================================

.. image:: _static/logo.png
   :alt: DelaDect Logo
   :align: center

DelaDect is a Python package for detecting delaminations and cracks in
fiber-reinforced polymers using optical methods.  It builds on the CrackDect project and
adds higher-level workflows, improved shift-correction tooling, and richer reporting.

Installation
------------

Create an isolated environment, install the dependencies, and add DelaDect in editable
mode while you explore the examples:

.. code-block:: bash

   python -m venv .venv
   .venv\\Scripts\\activate      # On macOS/Linux use: source .venv/bin/activate
   pip install --upgrade pip
   pip install -e .[dev]

The :doc:examples/getting_started guide expands on the commands above and walks through
your first analysis run.

Prerequisites
-------------

- CrackDect 0.2
- scikit-image 0.18+
- numpy >= 1.19
- scipy 1.6+
- matplotlib >= 3.3
- sqlalchemy 1.3+
- numba 0.52+
- psutil 5.8+


------------------

.. toctree::
   :maxdepth: 2
   :caption: User Documentation

   image_pre_processing
   detection
   utils
   delamination


--------

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/getting_started
   image_handling
   examples/shift_correction
   examples/crack_detection

Project Information
-------------------

Authors
~~~~~~~
- Vasco D. C. Pires

License
~~~~~~~
This project is licensed under the MIT License.
