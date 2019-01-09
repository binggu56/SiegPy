# -*- coding: utf-8 -*-
"""
File containing the tests for the eigenstate.py file.
"""

import pytest
import numpy as np
from siegpy import Eigenstate

# Variables for the tests of Eigenstate
l = 4.0
step = 1.0
xgrid = np.arange(-l, l + 10 ** (-14), step)
wrong_grid = np.arange(-l / 2, l / 2 + 10 ** (-14), step)
wf_expected = np.ones_like(xgrid)
wf_even = Eigenstate([-1, 0, 1], [1, 0, 1], 2.0)
wf_odd = Eigenstate([-1, 0, 1], [1, 0, -1], 2.0)
wf_not_even = Eigenstate([1, 2, 3], [1, 0, 1], 2.0)
wf_not_odd = Eigenstate([1, 2, 3], [1, 0, -1], 2.0)
wf_complex = Eigenstate([1, 2, 3], [1 + 1.0j, 2.0j, 1 + 2.0j], 2.0)
wf_complex_conj = Eigenstate([1, 2, 3], [1 - 1.0j, -2.0j, 1 - 2.0j], 2.0)


class TestEigenstate:
    def test_init_raises_ValueError(self):
        # grid and wf do not have the same length
        with pytest.raises(ValueError):
            Eigenstate(wrong_grid, wf_expected, 1.0)
        # the grid is not real
        cplx_grid = np.array([1.0j] * len(wf_expected))
        with pytest.raises(ValueError):
            Eigenstate(cplx_grid, wf_expected, 1.0)
        # The grid is real, this works fine
        cplx_grid = np.array([0.0j] * len(wf_expected))
        Eigenstate(cplx_grid, wf_expected, 1.0)
        # the grid is not real
        cplx_grid[0] = 1.0j  # Now it will raise an error
        with pytest.raises(ValueError):
            Eigenstate(cplx_grid, wf_expected, 1.0)

    def test_add_raises_ValueError(self):
        wf1 = Eigenstate([1, 2, 3], [1, 1, 1], 1.0)
        wf3 = Eigenstate([1, 2, 3, 4], [0, -1, 0, 1], 1.0)
        with pytest.raises(ValueError):
            wf1 + wf3

    def test_scal_prod(self):
        assert wf_complex.scal_prod(wf_complex) == 7.5 + 0j
        assert wf_complex.scal_prod(wf_complex, xlim=(1, 3)) == 7.5 + 0j

    def test_scal_prod_raises_ValueError(self):
        # different grids
        wf = Eigenstate(xgrid, wf_expected, 1.0)
        with pytest.raises(ValueError):
            wf_complex.scal_prod(wf)
        # wrong xlim
        with pytest.raises(ValueError):
            wf_complex.scal_prod(wf_complex, xlim=(-1, 1))
        with pytest.raises(ValueError):
            wf_complex.scal_prod(wf_complex, xlim=(2, 4))
        with pytest.raises(ValueError):
            wf_complex.scal_prod(wf_complex, xlim=(4, 6))

    def test_norm(self):
        assert wf_complex.norm() == wf_complex.scal_prod(wf_complex)

    def test_wavenumber(self):
        assert wf_even.wavenumber == 2
