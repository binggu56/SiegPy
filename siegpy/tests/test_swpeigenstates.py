# -*- coding: utf-8 -*-
"""
File containing the tests for the swpeigenstate.py file.
"""

import numpy as np
import pytest
from siegpy import (Eigenstate, SWPContinuum, SWPSiegert,
                    Rectangular, Gaussian, Potential)
from siegpy.swputils import q
from siegpy.swpbasisset import SWPBasisSet
from siegpy.swpeigenstates import ParityError
from siegpy.analyticeigenstates import WavenumberError

# Variables for Siegert states
filename = "doc/notebooks/siegerts.dat"
siegerts = SWPBasisSet.from_file(filename, nres=5)
res1 = siegerts.resonants[0]
res2 = siegerts.resonants[1]
l = 4.
xgrid = np.linspace(-l, l, 9)
siegerts_with_grid = SWPBasisSet.from_file(filename, grid=xgrid, nres=5)
siegerts_na_with_grid = SWPBasisSet.from_file(filename, nres=5,
                                              grid=xgrid, analytic=False)
bnd1 = siegerts_with_grid.bounds[0]
bnd2 = siegerts_with_grid.bounds[1]
res_na_1 = siegerts_na_with_grid.resonants[0]
res_na_2 = siegerts_na_with_grid.resonants[1]
h = 2.
pot = bnd1.potential

# Variables for continuum states
k = 1.
qq = q(k, pot.depth)
c_e = SWPContinuum(k, 'e', pot)
c_o = SWPContinuum(k, 'o', pot)
c_e_with_grid = SWPContinuum(k, 'e', pot, grid=xgrid)
c_e_expected = np.array(
    [-0.360072+0.434201j, -0.188472+0.227273j, 0.107958-0.130183j,
     0.014460-0.017436j, -0.111701+0.134697j, 0.014460-0.017436j,
     0.107958-0.130183j, -0.188472+0.227273j, -0.360072+0.434201j])
c_o_with_grid = SWPContinuum(k, 'o', pot, grid=xgrid)
c_o_expected = np.array(
    [-0.466284+0.224128j, -0.422628+0.203144j, 0.038320-0.018419j,
     -0.148014+0.071146j, 0.000000+0.j, 0.148014-0.071146j,
     -0.038320+0.018419j, 0.422628-0.203144j, 0.466284-0.224128j])
c_e_na = SWPContinuum(k, 'e', pot, grid=xgrid, analytic=False)
c_o_na = SWPContinuum(k, 'o', pot, grid=xgrid, analytic=False)

# Test functions
r1 = Rectangular(-10., 10., h=h)
r1p = Rectangular(-10., 10., k0=1., h=2.)
r1m = Rectangular(-10., 10., k0=-1., h=2.)
r2 = Rectangular(-1.8, 2.2, h=h)
r2p = Rectangular(-1.8, 2.2, h=h, k0=qq)
r2m = Rectangular(-1.8, 2.2, h=h, k0=-qq)
g1 = Gaussian(0.25, 0.0, h=h)
g2 = Gaussian(0.25, 0.2, h=h)
r_na_1 = Rectangular(-2., 2., h=h, grid=xgrid)
r_na_k0_1 = Rectangular(-2., 2., k0=2., h=h, grid=xgrid)
r_na_2 = Rectangular(0.5, 4.5, h=h, grid=xgrid)
r_na_k0_2 = Rectangular(0.5, 4.5, k0=2., h=h, grid=xgrid)
r_na_3 = Rectangular.from_width_and_center(4., 2.5, h=2., grid=xgrid)
r_na_k0_3 = Rectangular.from_width_and_center(4., 2.5, k0=2., h=2., grid=xgrid)
g_na_1 = Gaussian(0.25, 0.0, h=h, grid=xgrid)
g_na_k0_1 = Gaussian(0.25, 0.0, k0=2., h=h, grid=xgrid)
g_na_2 = Gaussian(0.25, 0.2, h=h, grid=xgrid)
g_na_k0_2 = Gaussian(0.25, 0.2, k0=2., h=h, grid=xgrid)


class TestSWPSiegert():

    def test_init(self):
        for sieg, sieg_with_grid in zip(siegerts, siegerts_with_grid):
            assert sieg == sieg_with_grid
        np.testing.assert_array_equal(siegerts_with_grid.grid, xgrid)

    def test_init_raises_ValueError(self):
        # wrong grid: complex grid
        cplx_grid = 0.j * np.zeros(len(xgrid))
        cplx_grid[0] = 1.j  # Now it will raise an error
        with pytest.raises(ValueError):
            SWPBasisSet.from_file(filename, grid=cplx_grid)
        # wrong analyticity
        with pytest.raises(ValueError):
            SWPBasisSet.from_file(filename, analytic=1)

    def test_init_raises_TypeError(self):
        # The potential is not a 1D SWP
        not_swpot = Potential([-1, 0, 1], [1, 1, 1])
        not_swpot.width = pot.width
        not_swpot.depth = pot.depth
        kbnd1 = bnd1.wavenumber
        with pytest.raises(TypeError):
            SWPSiegert(kbnd1, 'e', not_swpot)
            # SWPContinuum(1, 'e', not_swpot)

    def test_eq(self):
        # Most of it is already tested by test_SWPSiegert
        wf = Eigenstate(bnd1.grid, bnd1.values, bnd1.energy)
        assert bnd1 != wf

    @pytest.mark.parametrize("i, expected", [
        (0, np.array([
            3.50166702e-05+0.j, 2.92662372e-03+0.j, 1.80532848e-01+0.j,
            5.11858466e-01+0.j, 6.39217095e-01+0.j, 5.11858466e-01+0.j,
            1.80532848e-01+0.j, 2.92662372e-03+0.j, 3.50166702e-05+0.j])),
        (1, np.array([
            -8.98800192e-05+0.j, -6.51954764e-03+0.j, -3.47195120e-01+0.j,
            -6.12030918e-01+0.j, 0.00000000e+00+0.j, 6.12030918e-01-0.j,
            3.47195120e-01-0.j, 6.51954764e-03-0.j, 8.98800192e-05-0.j])),
        (2, np.array([
            -2.07754635e-04+0.j, -1.17863573e-02+0.j, -4.86247485e-01-0.j,
            -2.18587366e-01-0.j, 6.36404847e-01+0.j, -2.18587366e-01-0.j,
            -4.86247485e-01-0.j, -1.17863573e-02+0.j, -2.07754635e-04+0.j]))
    ])
    def test_compute_wavefunction(self, i, expected):
        np.testing.assert_array_almost_equal(
            siegerts_with_grid.bounds[i].values, expected)

    @pytest.mark.parametrize("value, expected", [
        (bnd1.parity, 'e'), (bnd2.parity, 'o')
    ])
    def test_parity(self, value, expected):
        assert value == expected

    def test_parity_raises_WavenumberError(self):
        # Wavenumbers that do not correspond to a Siegert state
        with pytest.raises(WavenumberError):
            SWPSiegert(-1-1.j, 'e', pot)

    @pytest.mark.parametrize("wrong_parity", [1, 'toto', None, 'o'])
    def test_parity_raises_ParityError(self, wrong_parity):
        kbnd1 = bnd1.wavenumber
        # All other value than the correct parity (here, 'e') raise a
        # ParityError.
        with pytest.raises(ParityError):
            SWPSiegert(kbnd1, wrong_parity, pot)

    @pytest.mark.parametrize("value, expected", [
        (bnd1.is_even, True), (bnd2.is_even, False),
        (bnd1.is_odd, False), (bnd2.is_odd, True)
    ])
    def test_is_parity(self, value, expected):
        assert value == expected

    def test_Siegert_type(self):
        for bnd in siegerts.bounds:
            assert bnd.Siegert_type == 'b'
        for abnd in siegerts.antibounds:
            assert abnd.Siegert_type == 'ab'
        for res in siegerts.resonants:
            assert res.Siegert_type == 'r'
        for ares in siegerts.antiresonants:
            assert ares.Siegert_type == 'ar'

    @pytest.mark.parametrize("ks", [1+1.j, -1+1.j, 0.j])
    def test_Siegert_type_raises_WavenumberError(self, ks):
        # Wavenumbers in the wrong quadrant
        with pytest.raises(WavenumberError):
            SWPSiegert(ks, 'e', pot)

    @pytest.mark.parametrize("value, expected", [
        (res1.scal_prod(r1), 0.j),
        (res2.scal_prod(r1), 57.075936675111286-3.1687957339529444j),
        (res1.scal_prod(r2), -0.17101876295932378+0.19370353333533974j),
        (res2.scal_prod(r2), -0.24119510662101751-0.063348512069781412j),
        (res1.scal_prod(g1), 0.j),
        (res2.scal_prod(g1), 0.3163788232996931+0.014746496869473297j),
        (res1.scal_prod(g2), 0.333747443464633-0.024123971996539793j),
        (res2.scal_prod(g2), 0.1364133481914243+0.02457912519237248j)
    ])
    def test_analytical_scal_prod(self, value, expected):
        decimal = 12
        np.testing.assert_almost_equal(value, expected, decimal=decimal)

    @pytest.mark.parametrize("value, expected", [
        (res_na_1.scal_prod(r_na_1), 0.j),
        (res_na_2.scal_prod(r_na_1), 4.1743949112633345-2.607164369836998j),
        (res_na_1.scal_prod(r_na_k0_1),
         0.70717092799511272-1.7207880711569299j),
        (res_na_1.scal_prod(r_na_2),
         -0.47198718301390397-0.83427379754553099j),
        (res_na_2.scal_prod(r_na_2), 1.3505045842605954-0.49831687689635684j),
        (res_na_1.scal_prod(r_na_k0_2),
         -0.47211403485904602-1.7626664365405986j),
        (res_na_1.scal_prod(g_na_1), 0.j),
        (res_na_2.scal_prod(g_na_1), 1.3473605337042931-0.088174608547941097j),
        (res_na_1.scal_prod(g_na_k0_1),
         -5.6534278452805584e-05-0.00082932286712860553j),
        (res_na_1.scal_prod(g_na_2),
         -0.008110237745499925+0.00055286844217250616j),
        (res_na_2.scal_prod(g_na_2),
         0.98439751748650228-0.065913082743066342j),
        (res_na_1.scal_prod(g_na_k0_2),
         0.0028706545528717434-0.0076292402507863915j)
    ])
    def test_numerical_scal_prod(self, value, expected):
        decimal = 12
        np.testing.assert_almost_equal(value, expected, decimal=decimal)

    def test_scal_prod_raises_TypeError(self):
        # Scalar product with continuum1DSWP not implemented
        c = SWPContinuum(1., 'e', bnd1.potential)
        with pytest.raises(TypeError):
            bnd1.scal_prod(c)
        # Scalar product not implemented for SWPSiegert, even if
        # bound state
        # (TODO: make the bound states scalar product possible ?)
        with pytest.raises(TypeError):
            bnd1.scal_prod(bnd1)

    def test_scal_prod_raises_ValueError(self):
        # Different grid for both functions in the numerical scalar product
        grid_fake = np.linspace(-(l+1), l+1, 11)
        bnd1_fake = SWPSiegert(
            bnd1.wavenumber, 'e', pot, grid=grid_fake, analytic=False)
        g = Gaussian(0.25, 0.0, h=h, grid=xgrid)
        with pytest.raises(ValueError):
            bnd1_fake.scal_prod(g)

    def test_MLE_strength_function(self):
        r = Rectangular.from_width_and_center(4., 0.2, h=h)
        kgrid = np.arange(0.1, 10.2, 1.)
        # Bound state
        bnd1_MLE_rf = bnd1.MLE_strength_function(r, kgrid)
        bnd1_MLE_rf_expected = np.array([
            0.005265, 0.054574, 0.090293, 0.109551, 0.116229, 0.115406,
            0.110816, 0.104659, 0.098098, 0.091697, 0.085703
        ])
        np.testing.assert_array_almost_equal(bnd1_MLE_rf, bnd1_MLE_rf_expected)
        # Resonant state
        res_1 = siegerts.resonants[0]
        r1_MLE_rf = res_1.MLE_strength_function(r, kgrid)
        r1_MLE_rf_expected = np.array([
            -0.004673, -0.007866, 0.003747, 0.00831, 0.004756, 0.003259,
            0.002468, 0.001983, 0.001656, 0.001422, 0.001245
        ])
        np.testing.assert_array_almost_equal(r1_MLE_rf, r1_MLE_rf_expected)


class TestSWPContinuum():

    def test_init_raises_WavenumberError(self):
        # Due to a complex wavenumber
        with pytest.raises(WavenumberError):
            SWPContinuum(1j, 'o', pot)

    @pytest.mark.parametrize("value, expected", [
        (c_e == SWPContinuum(k, 'e', pot, grid=xgrid), True),
        (c_e == c_o, False), (c_e == Rectangular(-2., 2.), False),
        (c_e != 1, True)
    ])
    def test_eq(self, value, expected):
        assert value == expected

    @pytest.mark.parametrize("wrong_parity", [1, 'toto', None])
    def test_parity_raises_ParityError(self, wrong_parity):
        # All other value than the correct parity (here, 'e') raise a
        # ParityError.
        with pytest.raises(ParityError):
            SWPContinuum(1, wrong_parity, pot)

    @pytest.mark.parametrize("value, expected", [
        (c_e.is_even, True), (c_o.is_even, False),
        (c_e.is_odd, False), (c_o.is_odd, True)
    ])
    def test_is_parity(self, value, expected):
        assert value == expected

    @pytest.mark.parametrize("state, expected", [
        (c_e_with_grid, c_e_expected), (c_o_with_grid, c_o_expected)
    ])
    def test_compute_wavefunction(self, state, expected):
        np.testing.assert_array_almost_equal(state.values, expected)

    def test_compute_wavefunction_raises_ValueError(self):
        # wrong grid: complex grid
        cplx_grid = 0.j * np.zeros(len(xgrid))
        cplx_grid[0] = 1.j  # Now it will raise an error
        with pytest.raises(ValueError):
            SWPContinuum(k, 'e', pot, grid=cplx_grid)

    @pytest.mark.parametrize("value, expected", [
        (c_e.scal_prod(r1), -0.90635923273823826-1.0929535715333105j),
        (c_o.scal_prod(r1), 0.j),
        (c_e.scal_prod(r1p), 8.1518896862942469+0.90334965979254944j),
        (c_o.scal_prod(r1p), 0.044058371125370144-0.091660394131671374j),
        (c_e.scal_prod(r1m), -0.62073133482877063+8.1782662389604024j),
        (c_o.scal_prod(r1m), -0.044058371125370144+0.091660394131671374j),
        (c_e.scal_prod(r2p), -0.46402741449617413-0.53088816154466156j),
        (c_o.scal_prod(r2p), 0.30059494527010172-0.58538480308246721j),
        (c_e.scal_prod(r2m), -0.43585286304346693-0.55425261681753712j),
        (c_o.scal_prod(r2m), -0.26937227999135543+0.60039259120519894j),
        (c_e.scal_prod(g1), -0.072629271280681904-0.087581632731056958j),
        (c_o.scal_prod(g1), 0.j),
        (c_e.scal_prod(g2), -0.044201378011213856-0.053301221214571481j),
        (c_o.scal_prod(g2), -0.077013178199935931-0.037017898720824186j)])
    def test_analytical_scal_prod(self, value, expected):
        decimal = 14
        np.testing.assert_almost_equal(value, expected, decimal=decimal)

    @pytest.mark.parametrize("value, expected", [
        (c_e_na.scal_prod(r_na_1), 0.26626637332669501+0.32108326720237679j),
        (c_o_na.scal_prod(r_na_1), 0.j),
        (c_e_na.scal_prod(r_na_k0_1),
         -0.52973455183891471-0.63879226854430993j),
        (c_e_na.scal_prod(r_na_3), -0.49218097140147571-0.59350744285884549j),
        (c_o_na.scal_prod(r_na_3), 1.5309260929446813+0.7358697353142144j),
        (c_e_na.scal_prod(r_na_k0_3),
         0.0052031824473088156-0.94598903180912952j),
        (c_e_na.scal_prod(g_na_1), -0.22338282414061131-0.26937117937892679j),
        (c_o_na.scal_prod(g_na_1), 0.j),
        (c_e_na.scal_prod(g_na_k0_1),
         -0.22341030109728754-0.26940431308226365j),
        (c_e_na.scal_prod(g_na_2), -0.16205020291442584-0.19541186501508359j),
        (c_o_na.scal_prod(g_na_2), 0.0017661284228762+0.0008489243609226155j),
        (c_e_na.scal_prod(g_na_k0_2),
         -0.16248453420586526-0.19555059652396745j)])
    def test_numerical_scal_prod(self, value, expected):
        decimal = 14
        np.testing.assert_almost_equal(value, expected, decimal=decimal)

    def test_scal_prod_raises_TypeError(self):
        # Scalar product not implemented for SWPSiegert,
        # even if it is known that it is zero for bound state.(TODO?)
        with pytest.raises(TypeError):
            c_e.scal_prod(bnd1)
