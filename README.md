[![Build Status](https://travis-ci.com/mmoriniere/SiegPy.svg?branch=master)](https://travis-ci.com/mmoriniere/SiegPy)
[![codecov](https://codecov.io/gh/mmoriniere/SiegPy/branch/master/graph/badge.svg)](https://codecov.io/gh/mmoriniere/SiegPy)

# SiegPy: "Siegert states with Python"

This module provides the tools to find the numerical Siegert (or resonant)
states of 1D potentials with compact support, as well as the analytical ones of
the 1D Square-Well Potential:

```
>>> from siegpy import SWPBasisSet, SWPotential
>>> pot = SWPotential(5, 5)
>>> siegerts = SWPBasisSet.find_Siegert_states(pot, re_k_max=5, im_k_max=2, step_re=3)
>>> len(siegerts)
18
>>> siegerts.plot_energies()
>>> siegerts.plot_wavefunctions()
```

A basis set made of such states is discrete and can later be compared to the 
exact results (*i.e.* using bound and continuum states) for:

* its completeness, compared to a basis set using the usual continuum states,
* its ability to reproduce the response function (*i.e.*, to provide an 
approximation of the Green's function with discrete states),
* its ability to reproduce the time-propagation of an initial wavepacket.

```
>>> from siegpy import Rectangular
>>> r = Rectangular(-1.5, 1.5)
>>> siegerts.plot_completeness_convergence(r)
>>> time_grid = [0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0]
>>> import numpy as np
>>> xgrid = np.linspace(-3, 3, 201)
>>> siegerts.grid = xgrid
>>> siegerts.Siegert_propagation(r, time_grid)
```


## Installation

Currently, `git clone` this repository and run `pip install .` in the `SiegPy` 
folder.

This module requires Python3.4+ and the following packages, which will be
installed while running the previous command if necessary:
* numpy
* scipy
* matplotlib
* mpmath


## Documentation

The documentation can be found [here](https://siegpy.readthedocs.io/en/latest/).

Ipython notebooks are stored in the 'doc/notebooks' folder and are part of the 
documentation as tutorials for the SiegPy module. They can also be considered as
an introduction to the Siegert states. They are supposed to be read in a
specific order, which can be is given in the `doc/notebooks.rst` file.

The whole documentation can be compiled manually. This requires the
installation of the some extra packages (such as sphinx, nbsphinx,
sphinx_rtd_theme, ...). This can be done by running `pip install .[doc]`. 
You can then go in the `doc` folder and running the `make html` command (or
alternatively `python3 -m sphinx . build -jN`, where N is the number of
precessors used for the compilation). This command creates the code
documentation and runs all the notebooks in order to create a tutorial.

## For developpers

For developpers:
* Run `pip3 install -e .[test]` to get all the packages required for testing. 
Running `pytest` in the SiegPy folder will then launch all the tests.
    * Unit tests are performed using pytest. 
    * pytest-cov shows which part of the code is currently not covered by
    unit tests (this is performed by running the `pytest` command).
    * flake8 is the tool used to check for PEP8 compliance (run `flake8` in the 
    `SiegPy` folder).
* If you modify notebooks, install and use nbstripout in order to clean the 
ouput cells before each `git commit` to save memory on the repository. See 
https://github.com/kynan/nbstripout for details.
* Documentation is created using sphinx (http://www.sphinx-doc.org/). The use
of restructured text in the docstrings is therefore recommended.
