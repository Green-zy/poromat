.. poromat documentation master file

Documentation for poromat
====================================

**poromat** is a Python package for predicting stress–strain curves of porous titanium alloys under various testing conditions.

It supports three regression methods:
- Physics-informed interpolation
- LightGBM
- Meta-learning with uncertainty estimates 

It is particularly suited for applications with sparse mechanical test data, such as in materials science and engineering research.

Installation & Quick Start
--------------------------

1. **Install via PyPI** (Python 3.10 required):

   .. code-block:: bash

      pip install poromat

2. **Download models and data**:

   .. code-block:: python

      import poromat
      poromat.download_all_models()
      poromat.download_data()

3. **Plot a stress–strain curve**:

   .. code-block:: python

      poromat.plot(16, 300, 3300, step=0.002, method="meta")

Navigation
----------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   changelog

About
-----

Developed by **Yun Zhou**  
Background in Mechanical Engineering and Applied Data Science 
GitHub: https://github.com/Green-zy/poromat
