# -*- coding: utf-8 -*
import pytest
import numpy as np
from siegpy.coordinatemappings import (
    ErfKGCoordMap,
    TanhKGCoordMap,
    ErfSimonCoordMap,
    TanhSimonCoordMap,
    UniformCoordMap,
)

# Uncomment the next lines to compare siegpy outputs with the analytic
# results (beware: this makes the tests slower, due to sympy usage)
# import scipy.special
# import sympy
# from siegpy.smoothfunctions import ErfSmoothFunction, TanhSmoothFunction
#
#
# x, x0, tta, lbda = sympy.symbols("x x_0 theta lambda")
# mods = ['numpy', {'erf': scipy.special.erf}]
# substitutions = [(x0, x00), (lbda, lbda0), (tta, tta0)]


x00 = 0.5
lbda0 = 1
tta0 = 0.4
npts = 5
xgrid = np.linspace(-2 * x00, 2 * x00, npts)
# Uniform coordinate mapping
cm_u = UniformCoordMap(tta0)
cm_u_grid = UniformCoordMap(tta0, grid=xgrid)
cm_u_no_GCVT_grid = UniformCoordMap(tta0, GCVT=False, grid=xgrid)
# Erf KG Coordinate mapping
cm_eKG = ErfKGCoordMap(tta0, x00, lbda0, GCVT=False, grid=xgrid)
cm_eKG_GCVT = ErfKGCoordMap(tta0, x00, lbda0, GCVT=True, grid=xgrid)
# Tanh KG Coordinate mapping
cm_tKG = TanhKGCoordMap(tta0, x00, lbda0, GCVT=False, grid=xgrid)
cm_tKG_GCVT = TanhKGCoordMap(tta0, x00, lbda0, GCVT=True, grid=xgrid)
# Erf Simon Coordinate mapping
cm_eS = ErfSimonCoordMap(tta0, x00, lbda0, GCVT=False, grid=xgrid)
cm_eS_GCVT = ErfSimonCoordMap(tta0, x00, lbda0, GCVT=True, grid=xgrid)
cm_eS_GCVT_no_grid = ErfSimonCoordMap(tta0, x00, lbda0, GCVT=True)
# Tanh Simon Coordinate mapping
cm_tS = TanhSimonCoordMap(tta0, x00, lbda0, GCVT=False, grid=xgrid)
cm_tS_GCVT = TanhSimonCoordMap(tta0, x00, lbda0, GCVT=True, grid=xgrid)
cm_tS_GCVT_no_grid = TanhSimonCoordMap(tta0, x00, lbda0, GCVT=True)


class TestUniformCoordMap:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (cm_u.theta, tta0),
            (cm_u.grid, None),
            (cm_u.values, None),
            (cm_u.f_values, None),
            (cm_u.f_dx_values, None),
            (cm_u.f_dx2_values, None),
            (cm_u.dxi_values, {"theta": None}),
            (cm_u.f_dxi_values, {"theta": None}),
            (cm_u.U0_values, None),
            (cm_u.U1_values, None),
            (cm_u.U11_values, None),
            (cm_u.U2_values, None),
            (cm_u.V0_values, None),
            (cm_u.V1_values, None),
            (cm_u.V2_values, None),
        ],
    )
    def test_init_without_grid_attributes(self, value, expected):
        assert value == expected

    @pytest.mark.parametrize(
        "values, expected",
        [
            (cm_u_grid.grid, xgrid),
            (
                cm_u_grid.values,
                np.array(
                    [
                        -0.92106099 - 0.38941834j,
                        -0.46053050 - 0.19470917j,
                        0.00000000 + 0.0j,
                        0.46053050 + 0.19470917j,
                        0.92106099 + 0.38941834j,
                    ]
                ),
            ),
            (
                cm_u_grid.f_values,
                np.array(
                    [
                        0.92106099 + 0.38941834j,
                        0.92106099 + 0.38941834j,
                        0.92106099 + 0.38941834j,
                        0.92106099 + 0.38941834j,
                        0.92106099 + 0.38941834j,
                    ]
                ),
            ),
            (cm_u_grid.f_dx_values, 0),
            (cm_u_grid.f_dx2_values, 0),
        ],
    )
    def test_init_with_grid(self, values, expected):
        np.testing.assert_array_almost_equal(values, expected)


# Uncomment the next lines to compare siegpy outputs with the analytic
# results (beware: this makes the tests slower, due to sympy usage)
#    def test_init(self):
#        cm_u = UniformCoordMap(tta0)
#        assert cm.theta == tta0 and cm.grid is None
#        cm.grid = xgrid
#        np.testing.assert_array_almost_equal(cm.grid, xgrid)
#        F = x * sympy.exp(sympy.I*tta)
#        Fl = sympy.lambdify(x, F.subs(tta, tta0))
#        np.testing.assert_array_almost_equal(cm.values, Fl(xgrid))


class TestErfKGCoordMap:
    @pytest.mark.parametrize(
        "value, expected",
        [(cm_eKG.theta, tta0), (cm_eKG.x0, x00), (cm_eKG.lbda, lbda0)],
    )
    def test_init_GCVT_False_attributes(self, value, expected):
        assert value == expected

    @pytest.mark.parametrize(
        "values, expected",
        [
            (cm_eKG.grid, xgrid),
            (
                cm_eKG.dxi_values["theta"],
                np.array(
                    [
                        0.23774123 - 0.73994247j,
                        0.06637073 - 0.28160925j,
                        0.00000000 + 0.0j,
                        -0.06637073 + 0.28160925j,
                        -0.23774123 + 0.73994247j,
                    ]
                ),
            ),
            (
                cm_eKG.f_dxi_values["theta"],
                np.array(
                    [
                        -0.46640835 + 1.06552723j,
                        -0.21382013 + 0.72731305j,
                        -0.09140531 + 0.47070738j,
                        -0.21382013 + 0.72731305j,
                        -0.46640835 + 1.06552723j,
                    ]
                ),
            ),
            (
                cm_eKG.dxi_values["x0"],
                np.array(
                    [
                        -0.06103919 + 0.18997751j,
                        -0.03540738 + 0.15023258j,
                        -0.00000000 + 0.0j,
                        0.03540738 - 0.15023258j,
                        0.06103919 - 0.18997751j,
                    ]
                ),
            ),
            (
                cm_eKG.f_dxi_values["x0"],
                np.array(
                    [
                        0.01431887 + 0.0545679j,
                        0.07320107 - 0.21460662j,
                        0.06700761 - 0.34506723j,
                        0.07320107 - 0.21460662j,
                        0.01431887 + 0.0545679j,
                    ]
                ),
            ),
            (
                cm_eKG.dxi_values["lbda"],
                np.array(
                    [
                        0.01596750 - 0.04969701j,
                        -0.00952251 + 0.04040376j,
                        -0.00000000 + 0.0j,
                        0.00952251 - 0.04040376j,
                        -0.01596750 + 0.04969701j,
                    ]
                ),
            ),
            (
                cm_eKG.f_dxi_values["lbda"],
                np.array(
                    [
                        -0.07586772 + 0.2101968j,
                        -0.01059859 + 0.07078348j,
                        0.03350380 - 0.17253361j,
                        -0.01059859 + 0.07078348j,
                        -0.07586772 + 0.2101968j,
                    ]
                ),
            ),
            (
                cm_eKG.U0_values["theta"],
                np.array(
                    [
                        0.05462472 - 0.17879462j,
                        -0.07179869 - 0.0299599j,
                        -0.11722563 + 0.04731146j,
                        -0.07179869 - 0.0299599j,
                        0.05462472 - 0.17879462j,
                    ]
                ),
            ),
            (
                cm_eKG.U1_values["theta"],
                np.array(
                    [
                        -0.05954141 + 0.09439845j,
                        -0.10874063 + 0.0853842j,
                        -0.00000000 + 0.0j,
                        0.10874063 - 0.0853842j,
                        0.05954141 - 0.09439845j,
                    ]
                ),
            ),
            (
                cm_eKG.U2_values["theta"],
                np.array(
                    [
                        0.46436385 + 0.31653741j,
                        0.21920179 + 0.30570517j,
                        0.08972918 + 0.22232581j,
                        0.21920179 + 0.30570517j,
                        0.46436385 + 0.31653741j,
                    ]
                ),
            ),
            (
                cm_eKG.U11_values["theta"],
                np.array(
                    [
                        -0.46436385 - 0.31653741j,
                        -0.21920179 - 0.30570517j,
                        -0.08972918 - 0.22232581j,
                        -0.21920179 - 0.30570517j,
                        -0.46436385 - 0.31653741j,
                    ]
                ),
            ),
            (
                cm_eKG.U0_values["x0"],
                np.array(
                    [
                        -0.00330192 - 0.00844518j,
                        0.02107222 + 0.0098715j,
                        0.08593603 - 0.03468319j,
                        0.02107222 + 0.0098715j,
                        -0.00330192 - 0.00844518j,
                    ]
                ),
            ),
            (
                cm_eKG.U1_values["x0"],
                np.array(
                    [
                        0.00057542 + 0.00538261j,
                        0.03358728 - 0.02412403j,
                        0.00000000 + 0.0j,
                        -0.03358728 + 0.02412403j,
                        -0.00057542 - 0.00538261j,
                    ]
                ),
            ),
            (
                cm_eKG.U2_values["x0"],
                np.array(
                    [
                        0.02718944 - 0.00193122j,
                        -0.06159261 - 0.0941581j,
                        -0.06577887 - 0.16298311j,
                        -0.06159261 - 0.0941581j,
                        0.02718944 - 0.00193122j,
                    ]
                ),
            ),
            (
                cm_eKG.U11_values["x0"],
                np.array(
                    [
                        -0.02718944 + 0.00193122j,
                        0.06159261 + 0.0941581j,
                        0.06577887 + 0.16298311j,
                        0.06159261 + 0.0941581j,
                        -0.02718944 + 0.00193122j,
                    ]
                ),
            ),
            (
                cm_eKG.U0_values["lbda"],
                np.array(
                    [
                        0.00819896 - 0.03497036j,
                        -0.00710197 - 0.00187414j,
                        0.04296801 - 0.0173416j,
                        -0.00710197 - 0.00187414j,
                        0.00819896 - 0.03497036j,
                    ]
                ),
            ),
            (
                cm_eKG.U1_values["lbda"],
                np.array(
                    [
                        -0.01021439 + 0.01885363j,
                        -0.00906639 + 0.00939055j,
                        0.00000000 + 0.0j,
                        0.00906639 - 0.00939055j,
                        0.01021439 - 0.01885363j,
                    ]
                ),
            ),
            (
                cm_eKG.U2_values["lbda"],
                np.array(
                    [
                        0.09304514 + 0.05477884j,
                        0.02445078 + 0.02575787j,
                        -0.03288943 - 0.08149155j,
                        0.02445078 + 0.02575787j,
                        0.09304514 + 0.05477884j,
                    ]
                ),
            ),
            (
                cm_eKG.U11_values["lbda"],
                np.array(
                    [
                        -0.09304514 - 0.05477884j,
                        -0.02445078 - 0.02575787j,
                        0.03288943 + 0.08149155j,
                        -0.02445078 - 0.02575787j,
                        -0.09304514 - 0.05477884j,
                    ]
                ),
            ),
        ],
    )
    def test_init_GCVT_False(self, values, expected):
        np.testing.assert_array_almost_equal(values, expected)

    # Do the same functions for the case GCVT=True
    @pytest.mark.parametrize(
        "value, expected",
        [
            (cm_eKG_GCVT.theta, tta0),
            (cm_eKG_GCVT.x0, x00),
            (cm_eKG_GCVT.lbda, lbda0),
            (cm_eKG_GCVT.U2_values, None),
            (cm_eKG_GCVT.U11_values, None),
            (cm_eKG_GCVT.dxi_values["theta"], None),
            (cm_eKG_GCVT.f_dxi_values["theta"], None),
            (cm_eKG_GCVT.dxi_values["x0"], None),
            (cm_eKG_GCVT.f_dxi_values["x0"], None),
            (cm_eKG_GCVT.dxi_values["lbda"], None),
            (cm_eKG_GCVT.f_dxi_values["lbda"], None),
        ],
    )
    def test_init_GCVT_True_attributes(self, value, expected):
        assert value == expected

    @pytest.mark.parametrize(
        "values, expected",
        [
            (cm_eKG_GCVT.grid, xgrid),
            (cm_eKG_GCVT.values, cm_eKG.values),
            (cm_eKG_GCVT.f_values, cm_eKG.f_values),
            (cm_eKG_GCVT.f_dx_values, cm_eKG.f_dx_values),
            (cm_eKG_GCVT.f_dx2_values, cm_eKG.f_dx2_values),
            (cm_eKG_GCVT.V0_values, cm_eKG.V0_values),
            (cm_eKG_GCVT.V1_values, cm_eKG.V1_values),
            (cm_eKG_GCVT.V2_values, cm_eKG.V2_values),
            (
                cm_eKG_GCVT.U0_values,
                np.array(
                    [
                        0.96361263 - 0.19293889j,
                        0.97900392 - 0.18209633j,
                        1.00000000 + 0.0j,
                        0.97900392 - 0.18209633j,
                        0.96361263 - 0.19293889j,
                    ]
                ),
            ),
            (
                cm_eKG_GCVT.U1_values,
                np.array(
                    [
                        -0.97742632 + 0.14853993j,
                        -0.49746909 + 0.03548306j,
                        0.00000000 + 0.0j,
                        0.49746909 - 0.03548306j,
                        0.97742632 - 0.14853993j,
                    ]
                ),
            ),
        ],
    )
    def test_init_GCVT_True(self, values, expected):
        np.testing.assert_array_almost_equal(values, expected)

    # Test that some errors are raised
    @pytest.mark.parametrize(
        "to_evaluate", ["ErfKGCoordMap(0.4, -0.5, 1)", "ErfKGCoordMap(0.4, 0.5, -1)"]
    )
    def test_init_raises_ValueError(self, to_evaluate):
        with pytest.raises(ValueError):
            eval(to_evaluate)


# Uncomment the next lines to compare siegpy outputs with the analytic
# results (beware: this makes the tests slower, due to sympy usage)
#     def test_init(self):
#         cm = ErfKGCoordMap(tta0, x00, lbda0, GCVT=False)
#         assert cm.theta == tta0 and cm.lbda == lbda0 and cm.x0 == x00
#         cm.grid = xgrid
#         np.testing.assert_array_almost_equal(cm.grid, xgrid)
#         sf = cm.smooth_func
#         assert isinstance(sf, ErfSmoothFunction)
#         np.testing.assert_array_almost_equal(sf.grid, xgrid)
#         # Test siegpy gives the same values as the analytic result
#         # 1- Create the analytic expressions
#         g = 1 + (sympy.erf(lbda*(x-x0)) - sympy.erf(lbda*(x+x0)))/2
#         F = x * sympy.exp(sympy.I*tta*g)
#         f = F.diff(x)
#         f_dx = f.diff(x)
#         f_dx2 = f.diff(x, 2)
#         F_dth = F.diff(tta)
#         f_dth = f.diff(tta)
#         F_dx0 = F.diff(x0)
#         f_dx0 = f.diff(x0)
#         F_dl = F.diff(lbda)
#         f_dl = f.diff(lbda)
#         V0 = 1/4 * f_dx2/f**3 - 5/8*f_dx**2/f**4
#         V1 = f_dx/f**3
#         V2 = 1/2 * (1 - 1/f**2)
#         U0_factor = 1/2 * (f_dx**2/f**5 + 1/2*f_dx2/f**4)
#         U1_factor = - 1/2 * f_dx/f**4
#         U2_factor = 1 / (2*f**3)
#         U11_factor = - 1 / (2*f**3)
#         # 2- Lambdify them to evaluate the analytic results over a grid
#         Fl = sympy.lambdify(x, F.subs(substitutions), modules=mods)
#         fl = sympy.lambdify(x, f.subs(substitutions), modules=mods)
#         f_dxl = sympy.lambdify(x, f_dx.subs(substitutions), modules=mods)
#         f_dx2l = sympy.lambdify(x, f_dx2.subs(substitutions), modules=mods)
#         F_dthl = sympy.lambdify(x, F_dth.subs(substitutions), modules=mods)
#         f_dthl = sympy.lambdify(x, f_dth.subs(substitutions), modules=mods)
#         F_dx0l = sympy.lambdify(x, F_dx0.subs(substitutions), modules=mods)
#         f_dx0l = sympy.lambdify(x, f_dx0.subs(substitutions), modules=mods)
#         F_dll = sympy.lambdify(x, F_dl.subs(substitutions), modules=mods)
#         f_dll = sympy.lambdify(x, f_dl.subs(substitutions), modules=mods)
#         V0l = sympy.lambdify(x, V0.subs(substitutions), modules=mods)
#         V1l = sympy.lambdify(x, V1.subs(substitutions), modules=mods)
#         V2l = sympy.lambdify(x, V2.subs(substitutions), modules=mods)
#         U0_thl = sympy.lambdify(x, (f_dth*U0_factor).subs(substitutions), modules=mods)  # noqa
#         U1_thl = sympy.lambdify(x, (f_dth*U1_factor).subs(substitutions), modules=mods)  # noqa
#         U2_thl = sympy.lambdify(x, (f_dth*U2_factor).subs(substitutions), modules=mods)  # noqa
#         U11_thl = sympy.lambdify(x, (f_dth*U11_factor).subs(substitutions), modules=mods)  # noqa
#         U0_ll = sympy.lambdify(x, (f_dl*U0_factor).subs(substitutions), modules=mods)  # noqa
#         U1_ll = sympy.lambdify(x, (f_dl*U1_factor).subs(substitutions), modules=mods)  # noqa
#         U2_ll = sympy.lambdify(x, (f_dl*U2_factor).subs(substitutions), modules=mods)  # noqa
#         U11_ll = sympy.lambdify(x, (f_dl*U11_factor).subs(substitutions), modules=mods)  # noqa
#         U0_x0l = sympy.lambdify(x, (f_dx0*U0_factor).subs(substitutions), modules=mods)  # noqa
#         U1_x0l = sympy.lambdify(x, (f_dx0*U1_factor).subs(substitutions), modules=mods)  # noqa
#         U2_x0l = sympy.lambdify(x, (f_dx0*U2_factor).subs(substitutions), modules=mods)  # noqa
#         U11_x0l = sympy.lambdify(x, (f_dx0*U11_factor).subs(substitutions), modules=mods)  # noqa
#         # 3- Compare siegpy results and analytic results
#         np.testing.assert_array_almost_equal(cm.values, Fl(xgrid))
#         np.testing.assert_array_almost_equal(cm.f_values, fl(xgrid))
#         np.testing.assert_array_almost_equal(cm.f_dx_values, f_dxl(xgrid))
#         np.testing.assert_array_almost_equal(cm.f_dx2_values, f_dx2l(xgrid))
#         np.testing.assert_array_almost_equal(cm.dxi_values['theta'], F_dthl(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.f_dxi_values['theta'], f_dthl(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.dxi_values['x0'], F_dx0l(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.f_dxi_values['x0'], f_dx0l(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.dxi_values['lbda'], F_dll(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.f_dxi_values['lbda'], f_dll(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.V0_values, V0l(xgrid))
#         np.testing.assert_array_almost_equal(cm.V1_values, V1l(xgrid))
#         np.testing.assert_array_almost_equal(cm.V2_values, V2l(xgrid))
#         np.testing.assert_array_almost_equal(cm.U0_values['theta'], U0_thl(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.U1_values['theta'], U1_thl(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.U2_values['theta'], U2_thl(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.U11_values['theta'], U11_thl(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.U0_values['lbda'], U0_ll(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.U1_values['lbda'], U1_ll(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.U2_values['lbda'], U2_ll(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.U11_values['lbda'], U11_ll(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.U0_values['x0'], U0_x0l(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.U1_values['x0'], U1_x0l(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.U2_values['x0'], U2_x0l(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.U11_values['x0'], U11_x0l(xgrid))  # noqa
#         # Check the virial potentials if GCVT=True
#         cm = ErfKGCoordMap(tta0, x00, lbda0, GCVT=True, grid=xgrid)
#         assert cm.U2_values is None and cm.U11_values is None
#         U0 = 1 - F*f_dx/f**2
#         U1 = F/f
#         U0l = sympy.lambdify(x, U0.subs(substitutions), modules=mods)
#         U1l = sympy.lambdify(x, U1.subs(substitutions), modules=mods)
#         np.testing.assert_array_almost_equal(cm.U0_values, U0l(xgrid))
#         np.testing.assert_array_almost_equal(cm.U1_values, U1l(xgrid))
#
#     def test_init_raises_ValueError(self):
#         with pytest.raises(ValueError):
#             cm = ErfKGCoordMap(tta0, -x00, lbda0)
#         with pytest.raises(ValueError):
#             cm = ErfKGCoordMap(tta0, x00, -lbda0)


class TestTanhKGCoordMap:
    @pytest.mark.parametrize(
        "value, expected",
        [(cm_tKG.theta, tta0), (cm_tKG.x0, x00), (cm_tKG.lbda, lbda0)],
    )
    def test_init_GCVT_False_attributes(self, value, expected):
        assert value == expected

    @pytest.mark.parametrize(
        "values, expected",
        [
            (cm_tKG.grid, xgrid),
            (
                cm_tKG.dxi_values["theta"],
                np.array(
                    [
                        0.23851649 - 0.74104516j,
                        0.07590083 - 0.30015351j,
                        0.00000000 + 0.0j,
                        -0.07590083 + 0.30015351j,
                        -0.23851649 + 0.74104516j,
                    ]
                ),
            ),
            (
                cm_tKG.f_dxi_values["theta"],
                np.array(
                    [
                        -0.42108799 + 1.00045404j,
                        -0.22217027 + 0.73208346j,
                        -0.11483639 + 0.52548126j,
                        -0.22217027 + 0.73208346j,
                        -0.42108799 + 1.00045404j,
                    ]
                ),
            ),
            (
                cm_tKG.dxi_values["x0"],
                np.array(
                    [
                        -0.05926445 + 0.18412829j,
                        -0.03481160 + 0.13766417j,
                        -0.00000000 + 0.0j,
                        0.03481160 - 0.13766417j,
                        0.05926445 - 0.18412829j,
                    ]
                ),
            ),
            (
                cm_tKG.f_dxi_values["x0"],
                np.array(
                    [
                        0.01698550 + 0.02371255j,
                        0.06991029 - 0.20927216j,
                        0.06716170 - 0.30732607j,
                        0.06991029 - 0.20927216j,
                        0.01698550 + 0.02371255j,
                    ]
                ),
            ),
            (
                cm_tKG.dxi_values["lbda"],
                np.array(
                    [
                        0.00748585 - 0.02325775j,
                        -0.01029595 + 0.04071582j,
                        -0.00000000 + 0.0j,
                        0.01029595 - 0.04071582j,
                        -0.00748585 + 0.02325775j,
                    ]
                ),
            ),
            (
                cm_tKG.f_dxi_values["lbda"],
                np.array(
                    [
                        -0.05522018 + 0.1619022j,
                        -0.00458724 + 0.03801313j,
                        0.03358085 - 0.15366303j,
                        -0.00458724 + 0.03801313j,
                        -0.05522018 + 0.1619022j,
                    ]
                ),
            ),
            (
                cm_tKG.U0_values["theta"],
                np.array(
                    [
                        0.04892529 - 0.11326268j,
                        -0.04656896 - 0.02039551j,
                        -0.10659759 + 0.04892742j,
                        -0.04656896 - 0.02039551j,
                        0.04892529 - 0.11326268j,
                    ]
                ),
            ),
            (
                cm_tKG.U1_values["theta"],
                np.array(
                    [
                        -0.05151859 + 0.06877471j,
                        -0.08904484 + 0.06874095j,
                        -0.00000000 + 0.0j,
                        0.08904484 - 0.06874095j,
                        0.05151859 - 0.06877471j,
                    ]
                ),
            ),
            (
                cm_tKG.U2_values["theta"],
                np.array(
                    [
                        0.41510900 + 0.33112303j,
                        0.22182936 + 0.30927531j,
                        0.11218869 + 0.24442419j,
                        0.22182936 + 0.30927531j,
                        0.41510900 + 0.33112303j,
                    ]
                ),
            ),
            (
                cm_tKG.U11_values["theta"],
                np.array(
                    [
                        -0.41510900 - 0.33112303j,
                        -0.22182936 - 0.30927531j,
                        -0.11218869 - 0.24442419j,
                        -0.22182936 - 0.30927531j,
                        -0.41510900 - 0.33112303j,
                    ]
                ),
            ),
            (
                cm_tKG.U0_values["x0"],
                np.array(
                    [
                        -0.00190532 - 0.00271323j,
                        0.01326198 + 0.00625262j,
                        0.06234327 - 0.02861505j,
                        0.01326198 + 0.00625262j,
                        -0.00190532 - 0.00271323j,
                    ]
                ),
            ),
            (
                cm_tKG.U1_values["x0"],
                np.array(
                    [
                        0.00085019 + 0.00214691j,
                        0.02622092 - 0.01910428j,
                        0.00000000 + 0.0j,
                        -0.02622092 + 0.01910428j,
                        -0.00085019 - 0.00214691j,
                    ]
                ),
            ),
            (
                cm_tKG.U2_values["x0"],
                np.array(
                    [
                        0.01342010 - 0.0048479j,
                        -0.06147461 - 0.0909364j,
                        -0.06561320 - 0.14295072j,
                        -0.06147461 - 0.0909364j,
                        0.01342010 - 0.0048479j,
                    ]
                ),
            ),
            (
                cm_tKG.U11_values["x0"],
                np.array(
                    [
                        -0.01342010 + 0.0048479j,
                        0.06147461 + 0.0909364j,
                        0.06561320 + 0.14295072j,
                        0.06147461 + 0.0909364j,
                        -0.01342010 + 0.0048479j,
                    ]
                ),
            ),
            (
                cm_tKG.U0_values["lbda"],
                np.array(
                    [
                        0.00644859 - 0.01834291j,
                        -0.00247251 - 0.00060048j,
                        0.03117163 - 0.01430752j,
                        -0.00247251 - 0.00060048j,
                        0.00644859 - 0.01834291j,
                    ]
                ),
            ),
            (
                cm_tKG.U1_values["lbda"],
                np.array(
                    [
                        -0.00734449 + 0.01137742j,
                        -0.00379129 + 0.00416196j,
                        0.00000000 + 0.0j,
                        0.00379129 - 0.00416196j,
                        0.00734449 - 0.01137742j,
                    ]
                ),
            ),
            (
                cm_tKG.U2_values["lbda"],
                np.array(
                    [
                        0.06889295 + 0.04750038j,
                        0.01362134 + 0.01331522j,
                        -0.03280660 - 0.07147536j,
                        0.01362134 + 0.01331522j,
                        0.06889295 + 0.04750038j,
                    ]
                ),
            ),
            (
                cm_tKG.U11_values["lbda"],
                np.array(
                    [
                        -0.06889295 - 0.04750038j,
                        -0.01362134 - 0.01331522j,
                        0.03280660 + 0.07147536j,
                        -0.01362134 - 0.01331522j,
                        -0.06889295 - 0.04750038j,
                    ]
                ),
            ),
        ],
    )
    def test_init_GCVT_False(self, values, expected):
        np.testing.assert_array_almost_equal(values, expected)

    # Do the same functions for the case GCVT=True
    @pytest.mark.parametrize(
        "value, expected",
        [
            (cm_tKG_GCVT.theta, tta0),
            (cm_tKG_GCVT.x0, x00),
            (cm_tKG_GCVT.lbda, lbda0),
            (cm_tKG_GCVT.U2_values, None),
            (cm_tKG_GCVT.U11_values, None),
            (cm_tKG_GCVT.dxi_values["theta"], None),
            (cm_tKG_GCVT.f_dxi_values["theta"], None),
            (cm_tKG_GCVT.dxi_values["x0"], None),
            (cm_tKG_GCVT.f_dxi_values["x0"], None),
            (cm_tKG_GCVT.dxi_values["lbda"], None),
            (cm_tKG_GCVT.f_dxi_values["lbda"], None),
        ],
    )
    def test_init_GCVT_True_attributes(self, value, expected):
        assert value == expected

    @pytest.mark.parametrize(
        "values, expected",
        [
            (cm_tKG_GCVT.grid, xgrid),
            (cm_tKG_GCVT.values, cm_tKG.values),
            (cm_tKG_GCVT.f_values, cm_tKG.f_values),
            (cm_tKG_GCVT.f_dx_values, cm_tKG.f_dx_values),
            (cm_tKG_GCVT.f_dx2_values, cm_tKG.f_dx2_values),
            (cm_tKG_GCVT.V0_values, cm_tKG.V0_values),
            (cm_tKG_GCVT.V1_values, cm_tKG.V1_values),
            (cm_tKG_GCVT.V2_values, cm_tKG.V2_values),
            (
                cm_tKG_GCVT.U0_values,
                np.array(
                    [
                        0.97583894 - 0.15882752j,
                        0.98627785 - 0.14689248j,
                        1.00000000 + 0.0j,
                        0.98627785 - 0.14689248j,
                        0.97583894 - 0.15882752j,
                    ]
                ),
            ),
            (
                cm_tKG_GCVT.U1_values,
                np.array(
                    [
                        -0.98553540 + 0.11939586j,
                        -0.49832349 + 0.02890404j,
                        0.00000000 + 0.0j,
                        0.49832349 - 0.02890404j,
                        0.98553540 - 0.11939586j,
                    ]
                ),
            ),
        ],
    )
    def test_init_GCVT_True(self, values, expected):
        np.testing.assert_array_almost_equal(values, expected)

    # Test that some errors are raised
    @pytest.mark.parametrize(
        "to_evaluate", ["TanhKGCoordMap(0.4, -0.5, 1)", "TanhKGCoordMap(0.4, 0.5, -1)"]
    )
    def test_init_raises_ValueError(self, to_evaluate):
        with pytest.raises(ValueError):
            eval(to_evaluate)


# Uncomment the next lines to compare siegpy outputs with the analytic
# results (beware: this makes the tests slower, due to sympy usage)
#     def test_init(self):
#         cm = TanhKGCoordMap(tta0, x00, lbda0, grid=xgrid, GCVT=False)
#         assert cm.theta == tta0 and cm.lbda == lbda0 and cm.x0 == x00
#         np.testing.assert_array_almost_equal(cm.grid, xgrid)
#         sf = cm.smooth_func
#         assert isinstance(sf, TanhSmoothFunction)
#         np.testing.assert_array_almost_equal(sf.grid, xgrid)
#         # Test siegpy gives the same values as the analytic result
#         # 1- Create the analytic expressions
#         g = 1 + (sympy.tanh(lbda*(x-x0)) - sympy.tanh(lbda*(x+x0)))/2
#         F = x * sympy.exp(sympy.I*tta*g)
#         f = F.diff(x)
#         f_dx = f.diff(x)
#         f_dx2 = f.diff(x, 2)
#         F_dth = F.diff(tta)
#         f_dth = f.diff(tta)
#         F_dx0 = F.diff(x0)
#         f_dx0 = f.diff(x0)
#         F_dl = F.diff(lbda)
#         f_dl = f.diff(lbda)
#         V0 = 1/4 * f_dx2/f**3 - 5/8*f_dx**2/f**4
#         V1 = f_dx/f**3
#         V2 = 1/2 * (1 - 1/f**2)
#         U0_factor = 1/2 * (f_dx**2/f**5 + 1/2*f_dx2/f**4)
#         U1_factor = - 1/2 * f_dx/f**4
#         U2_factor = 1 / (2*f**3)
#         U11_factor = - 1 / (2*f**3)
#         # 2- Lambdify them to evaluate the analytic results over a grid
#         Fl = sympy.lambdify(x, F.subs(substitutions))
#         fl = sympy.lambdify(x, f.subs(substitutions))
#         f_dxl = sympy.lambdify(x, f_dx.subs(substitutions))
#         f_dx2l = sympy.lambdify(x, f_dx2.subs(substitutions))
#         F_dthl = sympy.lambdify(x, F_dth.subs(substitutions))
#         f_dthl = sympy.lambdify(x, f_dth.subs(substitutions))
#         F_dx0l = sympy.lambdify(x, F_dx0.subs(substitutions))
#         f_dx0l = sympy.lambdify(x, f_dx0.subs(substitutions))
#         F_dll = sympy.lambdify(x, F_dl.subs(substitutions))
#         f_dll = sympy.lambdify(x, f_dl.subs(substitutions))
#         V0l = sympy.lambdify(x, V0.subs(substitutions))
#         V1l = sympy.lambdify(x, V1.subs(substitutions))
#         V2l = sympy.lambdify(x, V2.subs(substitutions))
#         U0_thl = sympy.lambdify(x, (f_dth*U0_factor).subs(substitutions))
#         U1_thl = sympy.lambdify(x, (f_dth*U1_factor).subs(substitutions))
#         U2_thl = sympy.lambdify(x, (f_dth*U2_factor).subs(substitutions))
#         U11_thl = sympy.lambdify(x, (f_dth*U11_factor).subs(substitutions))
#         U0_ll = sympy.lambdify(x, (f_dl*U0_factor).subs(substitutions))
#         U1_ll = sympy.lambdify(x, (f_dl*U1_factor).subs(substitutions))
#         U2_ll = sympy.lambdify(x, (f_dl*U2_factor).subs(substitutions))
#         U11_ll = sympy.lambdify(x, (f_dl*U11_factor).subs(substitutions))
#         U0_x0l = sympy.lambdify(x, (f_dx0*U0_factor).subs(substitutions))
#         U1_x0l = sympy.lambdify(x, (f_dx0*U1_factor).subs(substitutions))
#         U2_x0l = sympy.lambdify(x, (f_dx0*U2_factor).subs(substitutions))
#         U11_x0l = sympy.lambdify(x, (f_dx0*U11_factor).subs(substitutions))
#         # 3- Compare siegpy results and analytic results
#         np.testing.assert_array_almost_equal(cm.values, Fl(xgrid))
#         np.testing.assert_array_almost_equal(cm.f_values, fl(xgrid))
#         np.testing.assert_array_almost_equal(cm.f_dx_values, f_dxl(xgrid))
#         np.testing.assert_array_almost_equal(cm.f_dx2_values, f_dx2l(xgrid))
#         np.testing.assert_array_almost_equal(cm.dxi_values['theta'], F_dthl(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.f_dxi_values['theta'], f_dthl(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.dxi_values['x0'], F_dx0l(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.f_dxi_values['x0'], f_dx0l(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.dxi_values['lbda'], F_dll(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.f_dxi_values['lbda'], f_dll(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.V0_values, V0l(xgrid))
#         np.testing.assert_array_almost_equal(cm.V1_values, V1l(xgrid))
#         np.testing.assert_array_almost_equal(cm.V2_values, V2l(xgrid))
#         np.testing.assert_array_almost_equal(cm.U0_values['theta'], U0_thl(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.U1_values['theta'], U1_thl(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.U2_values['theta'], U2_thl(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.U11_values['theta'], U11_thl(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.U0_values['lbda'], U0_ll(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.U1_values['lbda'], U1_ll(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.U2_values['lbda'], U2_ll(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.U11_values['lbda'], U11_ll(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.U0_values['x0'], U0_x0l(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.U1_values['x0'], U1_x0l(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.U2_values['x0'], U2_x0l(xgrid))  # noqa
#         np.testing.assert_array_almost_equal(cm.U11_values['x0'], U11_x0l(xgrid))  # noqa
#         # Check the virial potentials if GCVT=True
#         cm = TanhKGCoordMap(tta0, x00, lbda0, GCVT=True, grid=xgrid)
#         assert cm.U2_values is None and cm.U11_values is None
#         U0 = 1 - F*f_dx/f**2
#         U1 = F/f
#         U0l = sympy.lambdify(x, U0.subs(substitutions), modules=mods)
#         U1l = sympy.lambdify(x, U1.subs(substitutions), modules=mods)
#         np.testing.assert_array_almost_equal(cm.U0_values, U0l(xgrid))
#         np.testing.assert_array_almost_equal(cm.U1_values, U1l(xgrid))


class TestErfSimonCoordMap:
    @pytest.mark.parametrize(
        "value, expected", [(cm_eS.theta, tta0), (cm_eS.x0, x00), (cm_eS.lbda, lbda0)]
    )
    def test_init_GCVT_False_attributes(self, value, expected):
        assert value == expected

    @pytest.mark.parametrize(
        "values, expected",
        [
            (cm_eS.grid, xgrid),
            (
                cm_eS.dxi_values["theta"],
                np.array(
                    [
                        0.23190220 - 0.54850028j,
                        0.10006787 - 0.23668276j,
                        -0.00000000 + 0.0j,
                        -0.10006787 + 0.23668276j,
                        -0.23190220 + 0.54850028j,
                    ]
                ),
            ),
            (
                cm_eS.f_dxi_values["theta"],
                np.array(
                    [
                        -0.30265491 + 0.71584618j,
                        -0.22533677 + 0.53297158j,
                        -0.18672614 + 0.44164886j,
                        -0.22533677 + 0.53297158j,
                        -0.30265491 + 0.71584618j,
                    ]
                ),
            ),
            (
                cm_eS.dxi_values["x0"],
                np.array(
                    [
                        -0.05867556 + 0.28945563j,
                        -0.03326098 + 0.16408157j,
                        0.00000000 - 0.0j,
                        0.03326098 - 0.16408157j,
                        0.05867556 - 0.28945563j,
                    ]
                ),
            ),
            (
                cm_eS.f_dxi_values["x0"],
                np.array(
                    [
                        0.03937923 - 0.19426385j,
                        0.06092065 - 0.30053101j,
                        0.06937022 - 0.34221406j,
                        0.06092065 - 0.30053101j,
                        0.03937923 - 0.19426385j,
                    ]
                ),
            ),
            (
                cm_eS.dxi_values["lbda"],
                np.array(
                    [
                        -0.01499550 + 0.0739751j,
                        -0.01407624 + 0.06944027j,
                        -0.00000000 + 0.0j,
                        0.01407624 - 0.06944027j,
                        0.01499550 - 0.0739751j,
                    ]
                ),
            ),
            (
                cm_eS.f_dxi_values["lbda"],
                np.array(
                    [
                        -0.01030138 + 0.05081829j,
                        0.01638409 - 0.08082524j,
                        0.03468511 - 0.17110703j,
                        0.01638409 - 0.08082524j,
                        -0.01030138 + 0.05081829j,
                    ]
                ),
            ),
            (
                cm_eS.U0_values["theta"],
                np.array(
                    [
                        0.01044981 - 0.01973181j,
                        -0.02669119 + 0.00230284j,
                        -0.04473239 + 0.00752934j,
                        -0.02669119 + 0.00230284j,
                        0.01044981 - 0.01973181j,
                    ]
                ),
            ),
            (
                cm_eS.U1_values["theta"],
                np.array(
                    [
                        -0.04945059 + 0.0374093j,
                        -0.04199013 + 0.01426553j,
                        0.00000000 + 0.0j,
                        0.04199013 - 0.01426553j,
                        0.04945059 - 0.0374093j,
                    ]
                ),
            ),
            (
                cm_eS.U2_values["theta"],
                np.array(
                    [
                        0.20679115 + 0.34833997j,
                        0.08940363 + 0.29355965j,
                        0.04435387 + 0.25075954j,
                        0.08940363 + 0.29355965j,
                        0.20679115 + 0.34833997j,
                    ]
                ),
            ),
            (
                cm_eS.U11_values["theta"],
                np.array(
                    [
                        -0.20679115 - 0.34833997j,
                        -0.08940363 - 0.29355965j,
                        -0.04435387 - 0.25075954j,
                        -0.08940363 - 0.29355965j,
                        -0.20679115 - 0.34833997j,
                    ]
                ),
            ),
            (
                cm_eS.U0_values["x0"],
                np.array(
                    [
                        -0.00161220 + 0.00546153j,
                        0.01362006 - 0.00400608j,
                        0.03083574 - 0.01184513j,
                        0.01362006 - 0.00400608j,
                        -0.00161220 + 0.00546153j,
                    ]
                ),
            ),
            (
                cm_eS.U1_values["x0"],
                np.array(
                    [
                        0.01046492 - 0.01185619j,
                        0.02030637 - 0.01182979j,
                        0.00000000 - 0.0j,
                        -0.02030637 + 0.01182979j,
                        -0.01046492 + 0.01185619j,
                    ]
                ),
            ),
            (
                cm_eS.U2_values["x0"],
                np.array(
                    [
                        -0.06933812 - 0.07659134j,
                        -0.07733939 - 0.14305246j,
                        -0.06793275 - 0.17254761j,
                        -0.07733939 - 0.14305246j,
                        -0.06933812 - 0.07659134j,
                    ]
                ),
            ),
            (
                cm_eS.U11_values["x0"],
                np.array(
                    [
                        0.06933812 + 0.07659134j,
                        0.07733939 + 0.14305246j,
                        0.06793275 + 0.17254761j,
                        0.07733939 + 0.14305246j,
                        0.06933812 + 0.07659134j,
                    ]
                ),
            ),
            (
                cm_eS.U0_values["lbda"],
                np.array(
                    [
                        0.00042174 - 0.0014287j,
                        0.00366300 - 0.0010774j,
                        0.01541787 - 0.00592257j,
                        0.00366300 - 0.0010774j,
                        0.00042174 - 0.0014287j,
                    ]
                ),
            ),
            (
                cm_eS.U1_values["lbda"],
                np.array(
                    [
                        -0.00273756 + 0.00310151j,
                        0.00546122 - 0.00318152j,
                        0.00000000 - 0.0j,
                        -0.00546122 + 0.00318152j,
                        0.00273756 - 0.00310151j,
                    ]
                ),
            ),
            (
                cm_eS.U2_values["lbda"],
                np.array(
                    [
                        0.01813845 + 0.02003585j,
                        -0.02079977 - 0.03847273j,
                        -0.03396638 - 0.0862738j,
                        -0.02079977 - 0.03847273j,
                        0.01813845 + 0.02003585j,
                    ]
                ),
            ),
            (
                cm_eS.U11_values["lbda"],
                np.array(
                    [
                        -0.01813845 - 0.02003585j,
                        0.02079977 + 0.03847273j,
                        0.03396638 + 0.0862738j,
                        0.02079977 + 0.03847273j,
                        -0.01813845 - 0.02003585j,
                    ]
                ),
            ),
        ],
    )
    def test_init_GCVT_False(self, values, expected):
        np.testing.assert_array_almost_equal(values, expected)

    # Do the same functions for the case GCVT=True
    @pytest.mark.parametrize(
        "value, expected",
        [
            (cm_eS_GCVT.theta, tta0),
            (cm_eS_GCVT.x0, x00),
            (cm_eS_GCVT.lbda, lbda0),
            (cm_eS_GCVT_no_grid.grid, None),
            (cm_eS_GCVT_no_grid.values, None),
            (cm_eS_GCVT_no_grid.f_values, None),
            (cm_eS_GCVT_no_grid.f_dx_values, None),
            (cm_eS_GCVT_no_grid.f_dx2_values, None),
            (cm_eS_GCVT.U2_values, None),
            (cm_eS_GCVT.U11_values, None),
            (cm_eS_GCVT.dxi_values["theta"], None),
            (cm_eS_GCVT.f_dxi_values["theta"], None),
            (cm_eS_GCVT.dxi_values["x0"], None),
            (cm_eS_GCVT.f_dxi_values["x0"], None),
            (cm_eS_GCVT.dxi_values["lbda"], None),
            (cm_eS_GCVT.f_dxi_values["lbda"], None),
        ],
    )
    def test_init_GCVT_True_attributes(self, value, expected):
        assert value == expected

    @pytest.mark.parametrize(
        "values, expected",
        [
            (cm_eS_GCVT.grid, xgrid),
            (cm_eS_GCVT.values, cm_eS.values),
            (cm_eS_GCVT.f_values, cm_eS.f_values),
            (cm_eS_GCVT.f_dx_values, cm_eS.f_dx_values),
            (cm_eS_GCVT.f_dx2_values, cm_eS.f_dx2_values),
            (cm_eS_GCVT.V0_values, cm_eS.V0_values),
            (cm_eS_GCVT.V1_values, cm_eS.V1_values),
            (cm_eS_GCVT.V2_values, cm_eS.V2_values),
            (
                cm_eS_GCVT.U0_values,
                np.array(
                    [
                        0.97198098 - 0.14962171j,
                        0.99580621 - 0.07209952j,
                        1.00000000 + 0.0j,
                        0.99580621 - 0.07209952j,
                        0.97198098 - 0.14962171j,
                    ]
                ),
            ),
            (
                cm_eS_GCVT.U1_values,
                np.array(
                    [
                        -0.99182523 + 0.07274134j,
                        -0.49958213 + 0.01310497j,
                        0.00000000 + 0.0j,
                        0.49958213 - 0.01310497j,
                        0.99182523 - 0.07274134j,
                    ]
                ),
            ),
        ],
    )
    def test_init_GCVT_True(self, values, expected):
        np.testing.assert_array_almost_equal(values, expected)

    # Test that some errors are raised
    @pytest.mark.parametrize(
        "to_evaluate",
        ["ErfSimonCoordMap(0.4, -0.5, 1)", "ErfSimonCoordMap(0.4, 0.5, -1)"],
    )
    def test_init_raises_ValueError(self, to_evaluate):
        with pytest.raises(ValueError):
            eval(to_evaluate)


# Uncomment the next lines to compare siegpy outputs with the analytic
# results (beware: this makes the tests slower, due to sympy usage)
#    def test_init(self):
#        cm = ErfSimonCoordMap(tta0, x00, lbda0, GCVT=False)
#        assert cm.theta == tta0 and cm.lbda == lbda0 and cm.x0 == x00
#        cm.grid = xgrid
#        np.testing.assert_array_almost_equal(cm.grid, xgrid)
#        sf = cm.smooth_func
#        assert isinstance(sf, ErfSmoothFunction)
#        np.testing.assert_array_almost_equal(sf.grid, xgrid)
#        # Test siegpy gives the same values as the analytic result
#        # 1- Create the analytic expressions
#        F = x + (sympy.exp(sympy.I*tta) - 1) * (x + 1/(2*lbda) *
#            (lbda*(x-x0)*sympy.erf(lbda*(x-x0))
#             - lbda*(x+x0)*sympy.erf(lbda*(x+x0))
#             + (sympy.exp(-(lbda*(x-x0))**2)
#                - sympy.exp(-(lbda*(x+x0))**2))/sympy.sqrt(sympy.pi)))
#        f = F.diff(x)
#        f_dx = f.diff(x)
#        f_dx2 = f.diff(x, 2)
#        F_dth = F.diff(tta)
#        f_dth = f.diff(tta)
#        F_dx0 = F.diff(x0)
#        f_dx0 = f.diff(x0)
#        F_dl = F.diff(lbda)
#        f_dl = f.diff(lbda)
#        # 2- Lambdify them to evaluate the analytic results over a grid
#        Fl = sympy.lambdify(x, F.subs(substitutions), modules=mods)
#        fl = sympy.lambdify(x, f.subs(substitutions), modules=mods)
#        f_dxl = sympy.lambdify(x, f_dx.subs(substitutions), modules=mods)
#        f_dx2l = sympy.lambdify(x, f_dx2.subs(substitutions), modules=mods)
#        F_dthl = sympy.lambdify(x, F_dth.subs(substitutions), modules=mods)
#        f_dthl = sympy.lambdify(x, f_dth.subs(substitutions), modules=mods)
#        F_dx0l = sympy.lambdify(x, F_dx0.subs(substitutions), modules=mods)
#        f_dx0l = sympy.lambdify(x, f_dx0.subs(substitutions), modules=mods)
#        F_dll = sympy.lambdify(x, F_dl.subs(substitutions), modules=mods)
#        f_dll = sympy.lambdify(x, f_dl.subs(substitutions), modules=mods)
#        # 3- Compare siegpy results and analytic results
#        np.testing.assert_array_almost_equal(cm.values, Fl(xgrid))
#        np.testing.assert_array_almost_equal(cm.f_values, fl(xgrid))
#        np.testing.assert_array_almost_equal(cm.f_dx_values, f_dxl(xgrid))
#        np.testing.assert_array_almost_equal(cm.f_dx2_values, f_dx2l(xgrid))
#        np.testing.assert_array_almost_equal(cm.dxi_values['theta'], F_dthl(xgrid))  # noqa
#        np.testing.assert_array_almost_equal(cm.f_dxi_values['theta'], f_dthl(xgrid))  # noqa
#        np.testing.assert_array_almost_equal(cm.dxi_values['x0'], F_dx0l(xgrid))  # noqa
#        np.testing.assert_array_almost_equal(cm.f_dxi_values['x0'], f_dx0l(xgrid))  # noqa
#        np.testing.assert_array_almost_equal(cm.dxi_values['lbda'], F_dll(xgrid))  # noqa
#        np.testing.assert_array_almost_equal(cm.f_dxi_values['lbda'], f_dll(xgrid))  # noqa
#        # Extra tests when GCVT=True
#        cm = ErfSimonCoordMap(tta0, x00, lbda0, grid=xgrid, GCVT=True)
#        for v in cm.dxi_values.values():
#            assert v is None
#        for v in cm.f_dxi_values.values():
#            assert v is None


class TestTanhSimonCoordMap:
    @pytest.mark.parametrize(
        "value, expected", [(cm_tS.theta, tta0), (cm_tS.x0, x00), (cm_tS.lbda, lbda0)]
    )
    def test_init_GCVT_False_attributes(self, value, expected):
        assert value == expected

    @pytest.mark.parametrize(
        "values, expected",
        [
            (cm_tS.grid, xgrid),
            (
                cm_tS.dxi_values["theta"],
                np.array(
                    [
                        0.24624369 - 0.5824211j,
                        0.11024807 - 0.2607612j,
                        -0.00000000 + 0.0j,
                        -0.11024807 + 0.2607612j,
                        -0.24624369 + 0.5824211j,
                    ]
                ),
            ),
            (
                cm_tS.f_dxi_values["theta"],
                np.array(
                    [
                        -0.30315612 + 0.71703166j,
                        -0.24112898 + 0.57032366j,
                        -0.20946144 + 0.49542291j,
                        -0.24112898 + 0.57032366j,
                        -0.30315612 + 0.71703166j,
                    ]
                ),
            ),
            (
                cm_tS.dxi_values["x0"],
                np.array(
                    [
                        -0.05396529 + 0.26621911j,
                        -0.03005974 + 0.14828937j,
                        0.00000000 - 0.0j,
                        0.03005974 - 0.14828937j,
                        0.05396529 - 0.26621911j,
                    ]
                ),
            ),
            (
                cm_tS.f_dxi_values["x0"],
                np.array(
                    [
                        0.03817310 - 0.18831383j,
                        0.05604568 - 0.27648203j,
                        0.06208140 - 0.30625717j,
                        0.05604568 - 0.27648203j,
                        0.03817310 - 0.18831383j,
                    ]
                ),
            ),
            (
                cm_tS.dxi_values["lbda"],
                np.array(
                    [
                        -0.01544592 + 0.07619712j,
                        -0.01293863 + 0.06382826j,
                        -0.00000000 + 0.0j,
                        0.01293863 - 0.06382826j,
                        0.01544592 - 0.07619712j,
                    ]
                ),
            ),
            (
                cm_tS.f_dxi_values["lbda"],
                np.array(
                    [
                        -0.00482175 + 0.02378643j,
                        0.01657618 - 0.08177286j,
                        0.03104070 - 0.15312859j,
                        0.01657618 - 0.08177286j,
                        -0.00482175 + 0.02378643j,
                    ]
                ),
            ),
            (
                cm_tS.U0_values["theta"],
                np.array(
                    [
                        0.00883490 - 0.01424963j,
                        -0.02153609 + 0.00404439j,
                        -0.04064435 + 0.01087449j,
                        -0.02153609 + 0.00404439j,
                        0.00883490 - 0.01424963j,
                    ]
                ),
            ),
            (
                cm_tS.U1_values["theta"],
                np.array(
                    [
                        -0.03941563 + 0.0299458j,
                        -0.03555209 + 0.01474531j,
                        0.00000000 + 0.0j,
                        0.03555209 - 0.01474531j,
                        0.03941563 - 0.0299458j,
                    ]
                ),
            ),
            (
                cm_tS.U2_values["theta"],
                np.array(
                    [
                        0.20763657 + 0.34853554j,
                        0.11079691 + 0.30843344j,
                        0.06956766 + 0.27698349j,
                        0.11079691 + 0.30843344j,
                        0.20763657 + 0.34853554j,
                    ]
                ),
            ),
            (
                cm_tS.U11_values["theta"],
                np.array(
                    [
                        -0.20763657 - 0.34853554j,
                        -0.11079691 - 0.30843344j,
                        -0.06956766 - 0.27698349j,
                        -0.11079691 - 0.30843344j,
                        -0.20763657 - 0.34853554j,
                    ]
                ),
            ),
            (
                cm_tS.U0_values["x0"],
                np.array(
                    [
                        -0.00143841 + 0.00388018j,
                        0.00925007 - 0.00375516j,
                        0.02188677 - 0.01088276j,
                        0.00925007 - 0.00375516j,
                        -0.00143841 + 0.00388018j,
                    ]
                ),
            ),
            (
                cm_tS.U1_values["x0"],
                np.array(
                    [
                        0.00806617 - 0.00917658j,
                        0.01453982 - 0.00980188j,
                        0.00000000 - 0.0j,
                        -0.01453982 + 0.00980188j,
                        -0.00806617 + 0.00917658j,
                    ]
                ),
            ),
            (
                cm_tS.U2_values["x0"],
                np.array(
                    [
                        -0.06731736 - 0.07412856j,
                        -0.07738936 - 0.12769096j,
                        -0.07157900 - 0.14967818j,
                        -0.07738936 - 0.12769096j,
                        -0.06731736 - 0.07412856j,
                    ]
                ),
            ),
            (
                cm_tS.U11_values["x0"],
                np.array(
                    [
                        0.06731736 + 0.07412856j,
                        0.07738936 + 0.12769096j,
                        0.07157900 + 0.14967818j,
                        0.07738936 + 0.12769096j,
                        0.06731736 + 0.07412856j,
                    ]
                ),
            ),
            (
                cm_tS.U0_values["lbda"],
                np.array(
                    [
                        0.00018169 - 0.00049012j,
                        0.00273582 - 0.00111063j,
                        0.01094339 - 0.00544138j,
                        0.00273582 - 0.00111063j,
                        0.00018169 - 0.00049012j,
                    ]
                ),
            ),
            (
                cm_tS.U1_values["lbda"],
                np.array(
                    [
                        -0.00101886 + 0.00115912j,
                        0.00430033 - 0.00289902j,
                        0.00000000 - 0.0j,
                        -0.00430033 + 0.00289902j,
                        0.00101886 - 0.00115912j,
                    ]
                ),
            ),
            (
                cm_tS.U2_values["lbda"],
                np.array(
                    [
                        0.00850304 + 0.00936338j,
                        -0.02288883 - 0.03776612j,
                        -0.03578950 - 0.07483909j,
                        -0.02288883 - 0.03776612j,
                        0.00850304 + 0.00936338j,
                    ]
                ),
            ),
            (
                cm_tS.U11_values["lbda"],
                np.array(
                    [
                        -0.00850304 - 0.00936338j,
                        0.02288883 + 0.03776612j,
                        0.03578950 + 0.07483909j,
                        0.02288883 + 0.03776612j,
                        -0.00850304 - 0.00936338j,
                    ]
                ),
            ),
        ],
    )
    def test_init_GCVT_False(self, values, expected):
        np.testing.assert_array_almost_equal(values, expected)

    # Do the same functions for the case GCVT=True
    @pytest.mark.parametrize(
        "value, expected",
        [
            (cm_tS_GCVT.theta, tta0),
            (cm_tS_GCVT.x0, x00),
            (cm_tS_GCVT.lbda, lbda0),
            (cm_tS_GCVT_no_grid.grid, None),
            (cm_tS_GCVT_no_grid.values, None),
            (cm_tS_GCVT_no_grid.f_values, None),
            (cm_tS_GCVT_no_grid.f_dx_values, None),
            (cm_tS_GCVT_no_grid.f_dx2_values, None),
            (cm_tS_GCVT.U2_values, None),
            (cm_tS_GCVT.U11_values, None),
            (cm_tS_GCVT.dxi_values["theta"], None),
            (cm_tS_GCVT.f_dxi_values["theta"], None),
            (cm_tS_GCVT.dxi_values["x0"], None),
            (cm_tS_GCVT.f_dxi_values["x0"], None),
            (cm_tS_GCVT.dxi_values["lbda"], None),
            (cm_tS_GCVT.f_dxi_values["lbda"], None),
        ],
    )
    def test_init_GCVT_True_attributes(self, value, expected):
        assert value == expected

    @pytest.mark.parametrize(
        "values, expected",
        [
            (cm_tS_GCVT.grid, xgrid),
            (cm_tS_GCVT.values, cm_tS.values),
            (cm_tS_GCVT.f_values, cm_tS.f_values),
            (cm_tS_GCVT.f_dx_values, cm_tS.f_dx_values),
            (cm_tS_GCVT.f_dx2_values, cm_tS.f_dx2_values),
            (cm_tS_GCVT.V0_values, cm_tS.V0_values),
            (cm_tS_GCVT.V1_values, cm_tS.V1_values),
            (cm_tS_GCVT.V2_values, cm_tS.V2_values),
            (
                cm_tS_GCVT.U0_values,
                np.array(
                    [
                        0.97930813 - 0.11964254j,
                        0.99591153 - 0.05853004j,
                        1.00000000 + 0.0j,
                        0.99591153 - 0.05853004j,
                        0.97930813 - 0.11964254j,
                    ]
                ),
            ),
            (
                cm_tS_GCVT.U1_values,
                np.array(
                    [
                        -0.99339457 + 0.05850527j,
                        -0.49948216 + 0.01071531j,
                        0.00000000 + 0.0j,
                        0.49948216 - 0.01071531j,
                        0.99339457 - 0.05850527j,
                    ]
                ),
            ),
        ],
    )
    def test_init_GCVT_True(self, values, expected):
        np.testing.assert_array_almost_equal(values, expected)

    # Test that some errors are raised
    @pytest.mark.parametrize(
        "to_evaluate",
        ["TanhSimonCoordMap(0.4, -0.5, 1)", "TanhSimonCoordMap(0.4, 0.5, -1)"],
    )
    def test_init_raises_ValueError(self, to_evaluate):
        with pytest.raises(ValueError):
            eval(to_evaluate)


# Uncomment the next lines to compare siegpy outputs with the analytic
# results (beware: this makes the tests slower, due to sympy usage)
#    def test_init(self):
#        cm = TanhSimonCoordMap(tta0, x00, lbda0, GCVT=False)
#        assert cm.theta == tta0 and cm.lbda == lbda0 and cm.x0 == x00
#        cm.grid = xgrid
#        np.testing.assert_array_almost_equal(cm.grid, xgrid)
#        sf = cm.smooth_func
#        assert isinstance(sf, TanhSmoothFunction)
#        np.testing.assert_array_almost_equal(sf.grid, xgrid)
#        # Test siegpy gives the same values as the analytic result
#        # 1- Create the analytic expressions
#        F = x + (sympy.exp(sympy.I*tta) - 1) * (x + 1/(2*lbda) *
#            sympy.log(sympy.cosh(lbda*(x-x0))/sympy.cosh(lbda*(x+x0))))
#        f = F.diff(x)
#        f_dx = f.diff(x)
#        f_dx2 = f.diff(x, 2)
#        F_dth = F.diff(tta)
#        f_dth = f.diff(tta)
#        F_dx0 = F.diff(x0)
#        f_dx0 = f.diff(x0)
#        F_dl = F.diff(lbda)
#        f_dl = f.diff(lbda)
#        # 2- Lambdify them to evaluate the analytic results over a grid
#        Fl = sympy.lambdify(x, F.subs(substitutions))
#        fl = sympy.lambdify(x, f.subs(substitutions))
#        f_dxl = sympy.lambdify(x, f_dx.subs(substitutions))
#        f_dx2l = sympy.lambdify(x, f_dx2.subs(substitutions))
#        F_dthl = sympy.lambdify(x, F_dth.subs(substitutions))
#        f_dthl = sympy.lambdify(x, f_dth.subs(substitutions))
#        F_dx0l = sympy.lambdify(x, F_dx0.subs(substitutions))
#        f_dx0l = sympy.lambdify(x, f_dx0.subs(substitutions))
#        F_dll = sympy.lambdify(x, F_dl.subs(substitutions))
#        f_dll = sympy.lambdify(x, f_dl.subs(substitutions))
#        # 3- Compare siegpy results and analytic results
#        np.testing.assert_array_almost_equal(cm.values, Fl(xgrid))
#        np.testing.assert_array_almost_equal(cm.f_values, fl(xgrid))
#        np.testing.assert_array_almost_equal(cm.f_dx_values, f_dxl(xgrid))
#        np.testing.assert_array_almost_equal(cm.f_dx2_values, f_dx2l(xgrid))
#        np.testing.assert_array_almost_equal(cm.dxi_values['theta'], F_dthl(xgrid))  # noqa
#        np.testing.assert_array_almost_equal(cm.f_dxi_values['theta'], f_dthl(xgrid))  # noqa
#        np.testing.assert_array_almost_equal(cm.dxi_values['x0'], F_dx0l(xgrid))  # noqa
#        np.testing.assert_array_almost_equal(cm.f_dxi_values['x0'], f_dx0l(xgrid))  # noqa
#        np.testing.assert_array_almost_equal(cm.dxi_values['lbda'], F_dll(xgrid))  # noqa
#        np.testing.assert_array_almost_equal(cm.f_dxi_values['lbda'], f_dll(xgrid))  # noqa
#        # Extra tests when GCVT=True
#        cm = TanhSimonCoordMap(tta0, x00, lbda0, grid=xgrid, GCVT=True)
#        for v in cm.dxi_values.values():
#            assert v is None
#        for v in cm.f_dxi_values.values():
#            assert v is None
