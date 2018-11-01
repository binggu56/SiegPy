# -*- coding: utf-8 -*-
"""
File containing the tests for the swpotential.py file.
"""

import pytest
import numpy as np
from siegpy import SWPotential
from siegpy.functions import Function


# Some initial values useful for the following tests
l = 4.
V0 = 10.
xgrid = np.linspace(-l, l, 9)
pot = SWPotential(l, V0)
pot_grid = SWPotential(l, V0, grid=xgrid)


class TestSWPotential():

    @pytest.mark.parametrize("value, expected", [
        (pot.width, l), (pot.depth, V0), (pot.grid, None), (pot.values, None)
    ])
    def test_init(self, value, expected):
        assert value == expected

    @pytest.mark.parametrize("value, expected", [
        (pot_grid.grid, xgrid),
        (pot_grid.values,
         np.array([0., 0., -10., -10., -10., -10., -10., 0., 0]))
    ])
    def test_init_with_grid(self, value, expected):
        np.testing.assert_array_equal(value, expected)

    @pytest.mark.parametrize("to_evaluate", [
        "SWPotential(-l, V0)", "SWPotential(l, -V0)", "SWPotential(-l, -V0)"
    ])
    def test_init_raise_ValueError(self, to_evaluate):
        with pytest.raises(ValueError):
            eval(to_evaluate)

    def test_eq(self):
        assert pot == pot_grid

    def test_add_raises_TypeError(self):
        # A SWPotential instance being a potential, only another
        # potential can be added to it:
        pot_with_grid = SWPotential(l, V0, grid=xgrid)
        with pytest.raises(TypeError):
            pot_with_grid += Function(xgrid, np.ones_like(xgrid))
