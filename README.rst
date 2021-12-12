evodcinv
========

|License| |Stars| |Pyversions| |Version| |Downloads| |Code style: black| |Codacy Badge| |Codecov| |Build| |Travis| |DOI|

**evodcinv** is a Python library to invert surface wave dispersion data (e.g., phase velocity dispersion curves) for an isotropic layered velocity model using Evolutionary Algorithms. It relies on `stochopy <https://github.com/keurfonluu/stochopy>`__ for the evolutionary optimizers while forward modeling is heavy-lifted by `disba <https://github.com/keurfonluu/disba>`__.

.. figure:: https://raw.githubusercontent.com/keurfonluu/evodcinv/master/.github/sample.png
   :alt: sample
   :width: 100%
   :align: center

   Inversion of phase velocity dispersion curve (fundamental mode).

Features
--------

Invertible data curves:

-  Love-wave phase and/or group velocity dispersion curves,
-  Rayleigh-wave phase and/or group velocity dispersion curves,
-  Rayleigh-wave ellipticity (experimental).

Installation
------------

The recommended way to install **evodcinv** and all its dependencies is through the Python Package Index:

.. code:: bash

   pip install evodcinv --user

Otherwise, clone and extract the package, then run from the package location:

.. code:: bash

   pip install . --user

To test the integrity of the installed package, check out this repository and run:

.. code:: bash

   pytest

Documentation
-------------

Refer to the online `documentation <https://keurfonluu.github.io/evodcinv/>`__ for detailed description of the API and examples.

Alternatively, the documentation can be built using `Sphinx <https://www.sphinx-doc.org/en/master/>`__:

.. code:: bash

   pip install -r doc/requirements.txt
   sphinx-build -b html doc/source doc/build

Usage
-----

The following example inverts a Rayleigh-wave phase velocity dispersion curve (fundamental mode).

.. code:: python

    from evodcinv import EarthModel, Layer, Curve

    # Initialize model
    model = EarthModel()

    # Build model search boundaries from top to bottom
    # First argument is the bounds of layer's thickness [km]
    # Second argument is the bounds of layer's S-wave velocity [km/s]
    model.add(Layer([0.001, 0.1], [0.1, 3.0]))
    model.add(Layer([0.001, 0.1], [0.1, 3.0]))

    # Configure model
    model.configure(
        optimizer="cpso",  # Evolutionary algorithm
        misfit="rmse",  # Misfit function type
        optimizer_args={
            "popsize": 10,  # Population size
            "maxiter": 100,  # Number of iterations
            "workers": -1,  # Number of cores
            "seed": 0,
        },
    )

    # Define the dispersion curves to invert
    # period and velocity are assumed to be data arrays
    curves = [Curve(period, velocity, 0, "rayleigh", "phase")]

    # Run inversion
    res = model.invert(curves)
    print(res)

Expected output:

.. code-block::

    --------------------------------------------------------------------------------
    Best model out of 1000 models (1 run)

    Velocity model                                    Model parameters
    ----------------------------------------          ------------------------------
             d        vp        vs       rho                   d        vs        nu
          [km]    [km/s]    [km/s]   [g/cm3]                [km]    [km/s]       [-]
    ----------------------------------------          ------------------------------
        0.0296    0.5033    0.2055    2.0000              0.0296    0.2055    0.4000
        1.0000    1.8191    1.0080    2.0000                   -    1.0080    0.2785
    ----------------------------------------          ------------------------------

    Number of layers: 2
    Number of parameters: 5
    Best model misfit: 0.0153
    --------------------------------------------------------------------------------

Contributing
------------

Please refer to the `Contributing
Guidelines <https://github.com/keurfonluu/evodcinv/blob/master/CONTRIBUTING.rst>`__ to see how you can help. This project is released with a `Code of Conduct <https://github.com/keurfonluu/evodcinv/blob/master/CODE_OF_CONDUCT.rst>`__ which you agree to abide by when contributing.

.. |License| image:: https://img.shields.io/github/license/keurfonluu/evodcinv
   :target: https://github.com/keurfonluu/evodcinv/blob/master/LICENSE

.. |Stars| image:: https://img.shields.io/github/stars/keurfonluu/evodcinv?logo=github
   :target: https://github.com/keurfonluu/evodcinv

.. |Pyversions| image:: https://img.shields.io/pypi/pyversions/evodcinv.svg?style=flat
   :target: https://pypi.org/pypi/evodcinv/

.. |Version| image:: https://img.shields.io/pypi/v/evodcinv.svg?style=flat
   :target: https://pypi.org/project/evodcinv

.. |Downloads| image:: https://pepy.tech/badge/evodcinv
   :target: https://pepy.tech/project/evodcinv

.. |Code style: black| image:: https://img.shields.io/badge/code%20style-black-000000.svg?style=flat
   :target: https://github.com/psf/black

.. |Codacy Badge| image:: https://img.shields.io/codacy/grade/bd53f27ac85d419d996c434353f08760.svg?style=flat
   :target: https://www.codacy.com/gh/keurfonluu/evodcinv/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=keurfonluu/evodcinv&amp;utm_campaign=Badge_Grade

.. |Codecov| image:: https://img.shields.io/codecov/c/github/keurfonluu/evodcinv.svg?style=flat
   :target: https://codecov.io/gh/keurfonluu/evodcinv

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5775193.svg?style=flat
   :target: https://doi.org/10.5281/zenodo.5775193

.. |Build| image:: https://img.shields.io/github/workflow/status/keurfonluu/evodcinv/Python%20package
   :target: https://github.com/keurfonluu/evodcinv

.. |Travis| image:: https://img.shields.io/travis/com/keurfonluu/evodcinv/master?label=docs
   :target: https://keurfonluu.github.io/evodcinv/