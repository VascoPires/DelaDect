DelaDect: Optical Delamination Detection in Fiber-Reinforced Polymers
=====================================================================

DelaDect is a Python package for quantitative damage analysis in 
fiber-reinforced polymers. Out of the box, the tool is able to perform the detection of cracks
and delamination from a sequence of images. 

The tool is capable of destinguishing between diffuse and edge delamination 
and can also destinguish delamination between multiple interfaces under certain conditions (see :doc:`methodology` for details). 

.. image:: _static/sample5_sequence.gif
   :alt: DelaDect visualization
   :width: 720
   :align: center

If you are new here, start with :doc:`examples/getting_started` or :doc:`methodology` after the instalation steps below.


Quick Start
-----------

It is recommended to use DelaDect in an isolated environment (venv or Conda) so dependencies remain
reproducible and separate from the system's Python.

Installation
~~~~~~~~~~~~
Supported Python is ``>=3.10``.
You can check your current Python version:

.. code-block:: bash

   $ python --version

Create and activate a fresh environment:

.. code-block:: bash

   # Option A: Conda
   $ conda create -n deladect python=3.10 -y
   $ conda activate deladect

   # Option B: venv
   $ python -m venv .venv
   $ .venv\Scripts\activate        # Windows
   # $ source .venv/bin/activate    # Linux/macOS

Then, install DelaDect and dependencies:

.. code-block:: bash

   $ pip install deladect


.. _prerequisites:

Prerequisites
-------------

DelaDect dependencies are installed automatically. 

- `CrackDect ≥ 0.2 <https://pypi.org/project/crackdect/>`_
- `NumPy ≥ 1.23.5 <https://numpy.org/>`_
- `SciPy ≥ 1.10.0 <https://scipy.org/>`_
- `Pandas ≥ 1.3.5 <https://pandas.pydata.org/>`_
- `Matplotlib ≥ 3.7.5 <https://matplotlib.org/>`_
- `scikit-image ≥ 0.18.1 <https://scikit-image.org/>`_
- `Pillow ≥ 8.4.0 <https://python-pillow.org/>`_

Documentation index
---------------------
- In :doc:`examples/getting_started` you will find some step-by-step examples to get you started with DelaDect.
- In :doc:`methodology` you will find a detailed description of the algorithms and methods used in DelaDect.

- Use :doc:`detection` and :doc:`delamination` for algorithm and API details.
- Use :doc:`parameter_list` for a full parameter list and default values.
- Use :doc:`Image_pre_processing` for normalization strategy tuning.
.


.. toctree::
   :maxdepth: 1
   :caption: User Guide
   :hidden:

   quick_start
   prerequisites
   methodology
   shift_correction
   image_handling
   Image_pre_processing
   parameter_list
   results_storage

.. toctree::
   :maxdepth: 1
   :caption: Examples
   :hidden:

   examples/getting_started
   examples/crack_detection
   examples/delamination_multi_interface
   examples/save_reload_results
  

.. toctree::
   :maxdepth: 1
   :caption: Project Information
   :hidden:



Project Information
-------------------

Authors
~~~~~~~
The current code base was developed by
`Vasco D. C. Pires <www.vascodcpires.com/>`_ with affiliation to the
`Institute Designing Plastics and Composite Materials (TU Leoben) <https://www.kunststofftechnik.at/en/konstruieren>`_.

License
~~~~~~~
This project is licensed under the AGPL-3.0 License.
