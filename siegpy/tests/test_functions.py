# -*- coding: utf-8-*-
"""
File containing the tests for the functions.py file.
"""

import pytest
import numpy as np
from siegpy import Eigenstate, SWPotential, Potential, Gaussian, Rectangular


# Global variables
l = 4.
step = 1.
xgrid = np.arange(-l, l+10**(-14), step)
wf_even = Eigenstate([-1, 0, 1], [1, 0, 1], 1.0)

# Variables for Gaussian tests
h = 2.0
sigma = 0.5
xc = 0.0
k0 = 2.
g = Gaussian(sigma, xc, h=h)
g_grid = Gaussian(sigma, xc, h=h, grid=xgrid)
gauss_expected = np.array(
    [2.53283311e-14+0.j, 3.04599595e-08+0.j, 6.70925256e-04+0.j,
     2.70670566e-01+0.j, 2.00000000e+00+0.j, 2.70670566e-01+0.j,
     6.70925256e-04+0.j, 3.04599595e-08+0.j, 2.53283311e-14+0.j])
g_momentum = Gaussian(sigma, xc, h=h, k0=k0)
g_grid_momentum = Gaussian(sigma, xc, h=h, k0=k0, grid=xgrid)
gauss_momentum_expected = np.array(
    [-3.68527303e-15-2.50587932e-14j, 2.92467480e-08+8.51098476e-09j,
     -4.38546014e-04+5.07757908e-04j, -1.12638700e-01-2.46120050e-01j,
     2.00000000e+00+0.00000000e+00j, -1.12638700e-01+2.46120050e-01j,
     -4.38546014e-04-5.07757908e-04j, 2.92467480e-08-8.51098476e-09j,
     -3.68527303e-15+2.50587932e-14j])
# Variables for the tests of Rectangular
a = 2.0
r = Rectangular(-a/2., a/2., h=h)
r_grid = Rectangular(-a/2., a/2., h=h, grid=xgrid)
r_grid_expected = np.array([0., 0., 0., 2., 2., 2., 0., 0., 0.], dtype=complex)
r_momentum = Rectangular(-a/2., a/2., k0=k0, h=h)
r_grid_momentum = Rectangular(-a/2., a/2., k0=k0, h=h, grid=xgrid)
r_grid_momentum_expected = np.array(
    [0.j, 0.j, 0.j, -0.83229367-1.81859485j, 2.0+0.j, -0.83229367+1.81859485j,
     0.j, 0.j, 0.j], dtype=complex)


class TestAnalyticWavefunction():

    def test_add(self):
        f1 = r + g_grid
        f2 = g_grid + r
        assert f1 == f2
        assert r.grid is None and r.values is None


class TestGaussian():

    @pytest.mark.parametrize("to_evaluate", [
        "Gaussian(0, 0, h=1)",  # sigma = 0
        "Gaussian(-2, -2, h=1)",  # sigma < 0
    ])
    def test_init_raises_ValueError(self, to_evaluate):
        with pytest.raises(ValueError):
            eval(to_evaluate)

    def test_eq(self):
        g != wf_even

    def test_is_odd(self):
        assert not g.is_odd

    def test_inside_potential(self):
        l0 = 4.
        V0 = 10
        pot = SWPotential(l0, V0)
        g_thin = Gaussian(sigma/2., xc)
        g_large = Gaussian(l0, xc)
        assert g_thin.is_inside(pot)
        assert not g_large.is_inside(pot)

    def test_inside_potential_raises_TypeError(self):
        pot = Potential(g_grid.grid, g_grid.values)
        with pytest.raises(TypeError):
            g.is_inside(pot)

    def test_norm(self):
        assert g.norm() == g_momentum.norm()
        assert np.isclose(g.norm(), 3.5449077018110318)

    def test_conjugate(self):
        assert g_momentum.conjugate() == Gaussian(sigma, xc, k0=-k0, h=h)


class TestRectangular():

    @pytest.mark.parametrize("to_evaluate", [
        "Rectangular(1, 2, h=0)",  # zero height
        "Rectangular(2, -2, h=0)",  # zero height, width < 0
    ])
    def test_init_raises_ValueError(self, to_evaluate):
        with pytest.raises(ValueError):
            eval(to_evaluate)

    def test_eq(self):
        assert r_grid == r

    @pytest.mark.parametrize("to_compare", [
        Rectangular.from_center_and_width(xc, a+2., h=h),
        Rectangular.from_center_and_width(xc+1.0, a, h=h),
        Rectangular.from_center_and_width(xc, a),
        wf_even
    ])
    def test_not_eq(self, to_compare):
        assert r != to_compare

    def test_is_odd(self):
        assert not r.is_odd

    def test_norm(self):
        assert r.norm() == r_momentum.norm()

    def test_conjugate(self):
        assert r_momentum.conjugate() == \
            Rectangular(xc-a/2., xc+a/2, k0=-k0, h=h)

    def test_split(self):
        # Rectangular function spreading over all regions
        xl = -10
        xr = 5
        r_large = Rectangular(xl, xr)
        l0 = 5
        V0 = 10
        pot = SWPotential(l0, V0)
        r_I, r_II, r_III = r_large.split(pot)
        assert r_I == Rectangular(xl, -l0/2)
        assert r_II == Rectangular(-l0/2, l0/2)
        assert r_III == Rectangular(l0/2, xr)
        # Rectangular function spreading over region I
        r_regI = Rectangular(xl, -l0)
        r_I, r_II, r_III = r_regI.split(pot)
        assert r_I == r_regI
        assert r_II is None
        assert r_III is None
        # Rectangular function spreading over region II and III
        r_regII_to_III = Rectangular(0, xr)
        r_I, r_II, r_III = r_regII_to_III.split(pot)
        assert r_I is None
        assert r_II == Rectangular(0, l0/2)
        assert r_III == Rectangular(l0/2, xr)
        # Rectangular function spreading over region II
        r_regII = Rectangular(-l0/4, l0/4)
        r_I, r_II, r_III = r_regII.split(pot)
        assert r_I is None
        assert r_II == r_regII
        assert r_III is None

    def test_split_raises_TypeError(self):
        with pytest.raises(TypeError):
            r.split(r)
