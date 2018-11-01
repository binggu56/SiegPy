# -*- coding: utf-8 -*-
"""
This file tests the symbolicpotentials.py file
"""

import pytest
import numpy as np
import sympy
from sympy.abc import x
from siegpy import (Potential, SWPotential, SymbolicPotential,
                    WoodsSaxonPotential, TwoGaussianPotential,
                    FourGaussianPotential, Gaussian)

# Initial variables
l = 4
V0 = 10
npts = 5
xgrid = np.linspace(-l, l, npts)
# Some potentials
wsp = WoodsSaxonPotential(1, 10, 4)
tgp = TwoGaussianPotential(1, 10, 4, 1, -10, 4)
tgp_fg = TwoGaussianPotential.from_Gaussians(Gaussian(1, 10, h=4),
                                             Gaussian(1, -10, h=4))
fgp = FourGaussianPotential(1, 20, 0, 1, 10, 4, 1, -10, 4, 1, -20, 0)
fgp_fg = FourGaussianPotential.from_Gaussians(Gaussian(1, 20, h=0),
                                              Gaussian(1, 10, h=4),
                                              Gaussian(1, -10, h=4),
                                              Gaussian(1, -20, h=0))


class TestSymbolicPotential():

    def test_add(self):
        # The addition of two symbolic potentials gives another
        # symbolic potential
        sym_func = sympy.Piecewise((0, x < -l/2), (-V0/2, x <= l/2),
                                   (0, x > l/2))
        pot_half = SymbolicPotential(sym_func, grid=xgrid)
        pot = pot_half + pot_half
        assert pot.symbolic == sympy.Piecewise((0, x < -l/2), (-V0, x <= l/2),
                                               (0, x > l/2))
        # Addition of a Potential
        pot_1 = Potential(xgrid, np.ones_like(xgrid))
        other = pot + pot_1
        expected = np.array([1, -9, -9, -9, 1])
        np.testing.assert_array_almost_equal(other.values, expected)
        # Addition of a SWPotential
        pot_2 = SWPotential(l, V0, grid=xgrid)
        other = pot + pot_2
        expected = np.array([0, -2*V0, -2*V0, -2*V0, 0])
        np.testing.assert_array_almost_equal(other.values, expected)

    def test_add_raise_ValueError(self):
        # ValueError because the potentials have no grid
        sym_func = sympy.Piecewise((0, x < -l/2), (-V0/2, x <= l/2),
                                   (0, x > l/2))
        pot_half = SymbolicPotential(sym_func, grid=xgrid)
        pot = SWPotential(l, V0/2)
        with pytest.raises(ValueError):
            pot_half += pot


class TestWoodsSaxonPotential():

    @pytest.mark.parametrize("to_evaluate", [
        "WoodsSaxonPotential(0, 10, 4)", "WoodsSaxonPotential(1, 10, 0)"
    ])
    def test_init_raises_ValueError(self, to_evaluate):
        with pytest.raises(ValueError):
            eval(to_evaluate)

    @pytest.mark.parametrize("method, value", [
        ("width", 1), ("depth", 10), ("sharpness", 4)
    ])
    def test_properties(self, method, value):
        assert eval("wsp.{}".format(method)) == value


class TestGaussianPotentials():

    @pytest.mark.parametrize("obj1, obj2", [
        (tgp, fgp), (tgp, tgp_fg), (fgp, fgp_fg)
    ])
    def test_init(self, obj1, obj2):
        assert obj1 == obj2

    # tgp = TwoGaussianPotential(1, 10, 4, 1, -10, 4)
    # fgp = FourGaussianPotential(1, 20, 0, 1, 10, 4, 1, -10, 4, 1, -20, 0)
    @pytest.mark.parametrize("method, value", [
        ("sigmas", [1, 1]),
        ("centers", [10, -10]),
        ("amplitudes", [4, 4])
    ])
    def test_TwoGaussianPotential_properties(self, method, value):
        assert eval("tgp.{}".format(method)) == value

    @pytest.mark.parametrize("method, value", [
        ("sigmas", [1]*4),
        ("centers", [20, 10, -10, -20]),
        ("amplitudes", [0, 4, 4, 0])
    ])
    def test_FourGaussianPotential_properties(self, method, value):
        assert eval("fgp.{}".format(method)) == value
