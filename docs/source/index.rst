DelaDect: Optical Delamination Detection in Fiber-Reinforced Polymers
=====================================================================

.. image:: _static/logo.png
   :alt: DelaDect Logo
   :align: center

DelaDect is a Python package for detecting delaminations in fiber-reinforced polymers using optical methods.  
It provides tools for image processing and feature extraction to identify and classify delaminations from sequences of images.  
On top of CrackDect, DelaDect adds additional functionality and quality-of-life improvements to simplify the delamination detection workflow.

Getting Started
---------------

The first step is to install the required dependencies. The full list is available in :ref:`Prerequisites`.  
DelaDect can be installed directly from PyPI:

.. code-block:: bash

   $ pip install deladec

Alternatively, you can clone the repository and install it locally:

.. code-block:: bash

   $ git clone https://github.com/<your-repo>/deladec.git
   $ cd deladec
   $ pip install .

Dependencies are installed automatically, but note that some versions are pinned for compatibility reasons.  
If you prefer using the latest package versions, be aware that certain features may not work as expected.  
We recommend creating a virtual environment to keep things isolated. With conda:

.. code-block:: bash

   $ conda create -n deladec_env python=3.8
   $ conda activate deladec_env

Prerequisites
-------------

- CrackDect 0.2
- scikit-image 0.18.1
- numpy >= 1.19.5
- scipy 1.6.0
- matplotlib >= 3.3.4
- sqlalchemy 1.3.23
- numba 0.52.0
- psutil 5.8.0

Authors
-------

- Vasco D.C. Pires

License
-------

This project is licensed under the MIT License.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   image_pre_processing
   image_handling
   
