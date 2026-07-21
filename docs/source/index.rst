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

If you are new here, start with :doc:`examples/getting_started` after the instalation steps below.


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

Create and activate a new environment (either conda or venv). For example, using conda:

.. code-block:: bash

   $ conda create -n deladect_env python=3.10 -y
   $ conda activate deladect_env

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

Documentation overview
----------------------

In the left sidebar, the documentation is divided into two main sections: Examples and User Guide. In the examples,
you will find step-by-step walkthroughs of the main functionalities of DelaDect. In the User Guide, 
you will find a detailed description of the algorithms and methods used in DelaDect.

There is also a binder notebooks available in this repository to run the examples without installing DelaDect. 
You can access it at
`Binder <https://mybinder.org/v2/gh/vascodcpires/deladect/main?labpath=notebooks/getting_started.ipynb>`_.

.. toctree::
   :maxdepth: 1
   :caption: Examples
   :hidden:

   examples/getting_started
   examples/advanced_options
   examples/delamination_multi_interface
   examples/save_reload_results

.. toctree::
   :maxdepth: 1
   :caption: User Guide
   :hidden:

   methodology
   image_operations
   shift_correction
   image_handling
   Image_pre_processing
   detection
   delamination
   parameter_list
   results_storage

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
