# -*- coding: utf-8 -*
import numpy as np
import pytest
from siegpy.smoothfunctions import ErfSmoothFunction, TanhSmoothFunction
# Uncomment the next lines to compare siegpy outputs with the analytic
# results (beware: this makes the tests slower, due to sympy usage)
# import scipy.special
# import sympy

x00 = 1
lbda0 = 1
sfe_no_grid = ErfSmoothFunction(x00, lbda0)
sft_no_grid = TanhSmoothFunction(x00, lbda0)
npts = 5
xgrid = np.linspace(-2*x00, 2*x00, npts)
sfe_grid = ErfSmoothFunction(x00, lbda0, grid=xgrid)
sft_grid = TanhSmoothFunction(x00, lbda0, grid=xgrid)


class TestErfSmoothFunction():

    @pytest.mark.parametrize("value, expected", [
        (sfe_no_grid.x0, x00), (sfe_no_grid.lbda, lbda0),
        (sfe_no_grid.grid, None), (sfe_no_grid.values, None),
        (sfe_no_grid.dx_values, None), (sfe_no_grid.dx2_values, None),
        (sfe_no_grid.dx3_values, None),
        (sfe_no_grid.dxi_values['x0'], None),
        (sfe_no_grid.dx_dxi_values['x0'], None),
        (sfe_no_grid.dxi_values['lbda'], None),
        (sfe_no_grid.dx_dxi_values['lbda'], None)
    ])
    def test_init_without_grid(self, value, expected):
        assert value == expected

    @pytest.mark.parametrize("values, expected", [
        (sfe_grid.grid, xgrid),
        (sfe_grid.values,
         np.array([0.92136144, 0.50233887, 0.15729921,
                   0.50233887, 0.92136144])),
        (sfe_grid.dx_values,
         np.array([-0.20748412, -0.55385609, 0., 0.55385609, 0.20748412])),
        (sfe_grid.dx2_values,
         np.array([-0.41468974, 0.04133397, 0.83021499,
                   0.04133397, -0.41468974])),
        (sfe_grid.dx3_values,
         np.array([-0.4127402, 1.27304806, 0., -1.27304806, 0.4127402])),
        (sfe_grid.dxi_values['x0'],
         np.array([-0.20762338, -0.57452308, -0.4151075,
                   -0.57452308, -0.20762338])),
        (sfe_grid.dx_dxi_values['x0'],
         np.array([-0.41552526, -0.04133397, -0., 0.04133397, 0.41552526])),
        (sfe_grid.dxi_values['lbda'],
         np.array([0.20734487, -0.02066699, -0.4151075,
                   -0.02066699, 0.20734487])),
        (sfe_grid.dx_dxi_values['lbda'],
         np.array([0.2063701, -0.63652403, 0., 0.63652403, -0.2063701]))
    ])
    def test_init_with_grid(self, values, expected):
        np.testing.assert_array_almost_equal(values, expected)

    @pytest.mark.parametrize("to_evaluate", [
        "ErfSmoothFunction(-0.5, 1)", "ErfSmoothFunction(0.5, -1)"
    ])
    def test_init_raises_ValueError(self, to_evaluate):
        with pytest.raises(ValueError):
            eval(to_evaluate)


class TestTanhSmoothFunction():

    @pytest.mark.parametrize("value, expected", [
        (sft_no_grid.x0, x00), (sft_no_grid.lbda, lbda0),
        (sft_no_grid.grid, None), (sft_no_grid.values, None),
        (sft_no_grid.dx_values, None), (sft_no_grid.dx2_values, None),
        (sft_no_grid.dx3_values, None),
        (sft_no_grid.dxi_values['x0'], None),
        (sft_no_grid.dx_dxi_values['x0'], None),
        (sft_no_grid.dxi_values['lbda'], None),
        (sft_no_grid.dx_dxi_values['lbda'], None)])
    def test_init_without_grid(self, value, expected):
        assert value == expected

    @pytest.mark.parametrize("values, expected", [
        (sft_grid.grid, xgrid),
        (sft_grid.values,
         np.array([0.8832697, 0.51798621, 0.23840584, 0.51798621, 0.8832697])),
        (sft_grid.dx_values,
         np.array([-0.20505415, -0.46467459, 0., 0.46467459, 0.20505415])),
        (sft_grid.dx2_values,
         np.array([-0.31003276, 0.06810934, 0.63970001, 0.06810934,
                   -0.31003276])),
        (sft_grid.dx3_values,
         np.array([-0.29137328, 1.12632703, 0., -1.12632703, 0.29137328])),
        (sft_grid.dxi_values['x0'],
         np.array([-0.21492019, -0.53532541, -0.41997434, -0.53532541,
                   -0.21492019])),
        (sft_grid.dx_dxi_values['x0'],
         np.array([-0.32966725, -0.06810934, -0., 0.06810934, 0.32966725])),
        (sft_grid.dxi_values['lbda'],
         np.array([0.19518812, -0.07065082, -0.41997434, -0.07065082,
                   0.19518812])),
        (sft_grid.dx_dxi_values['lbda'],
         np.array([0.08534411, -0.60089328, 0., 0.60089328, -0.08534411]))])
    def test_init_with_grid(self, values, expected):
        np.testing.assert_array_almost_equal(values, expected)

    @pytest.mark.parametrize("to_evaluate", [
        "TanhSmoothFunction(-0.5, 1)", "TanhSmoothFunction(0.5, -1)"
    ])
    def test_init_raises_ValueError(self, to_evaluate):
        with pytest.raises(ValueError):
            eval(to_evaluate)


# Uncomment the next lines to compare siegpy outputs with the analytic
# results (beware: this makes the tests slower, due to sympy usage)
# class TestSmoothFunction():
#
#     def test_init(self):
#         # All values are set to None if no grid is passed
#         x00 = 1
#         lbda0 = 1
#         sfe = ErfSmoothFunction(x00, lbda0)
#         assert sfe.x0 == x00 and sfe.lbda == lbda0
#         assert sfe.values is None
#         assert sfe.dx_values is None
#         assert sfe.dx2_values is None
#         assert sfe.dx3_values is None
#         assert sfe.dxi_values['x0'] is None
#         assert sfe.dx_dxi_values['x0'] is None
#         assert sfe.dxi_values['lbda'] is None
#         assert sfe.dx_dxi_values['lbda'] is None
#         # When the grid is updated, all the previous attributes also
#         # are updated. Make sure it compares to the analytic results.
#         # 1- Create all the symbolic expressions:
#         x, x0, lbda = sympy.symbols("x x_0 lambda")
#         erf_sf = 1 + (sympy.erf(lbda*(x-x0)) - sympy.erf(lbda*(x+x0)))/2
#         erf_sf_dx = erf_sf.diff(x)
#         erf_sf_dx2 = erf_sf.diff(x, 2)
#         erf_sf_dx3 = erf_sf.diff(x, 3)
#         erf_sf_dx0 = erf_sf.diff(x0)
#         erf_sf_dx_dx0 = erf_sf_dx.diff(x0)
#         erf_sf_dl = erf_sf.diff(lbda)
#         erf_sf_dx_dl = erf_sf_dx.diff(lbda)
#         # 2- lambdify them in order to evaluate them for a given grid
#         substitutions = [(x0, x00), (lbda, lbda0)]
#         mods = ['numpy', {'erf': scipy.special.erf}]
#         erf_sfl = sympy.lambdify(x, erf_sf.subs(substitutions), modules=mods)
#         erf_sf_dxl = sympy.lambdify(x, erf_sf_dx.subs(substitutions),
#                                     modules=mods)
#         erf_sf_dx2l = sympy.lambdify(x, erf_sf_dx2.subs(substitutions),
#                                      modules=mods)
#         erf_sf_dx3l = sympy.lambdify(x, erf_sf_dx3.subs(substitutions),
#                                      modules=mods)
#         erf_sf_dx0l = sympy.lambdify(x, erf_sf_dx0.subs(substitutions),
#                                      modules=mods)
#         erf_sf_dx_dx0l = sympy.lambdify(x, erf_sf_dx_dx0.subs(substitutions),
#                                         modules=mods)
#         erf_sf_dll = sympy.lambdify(x, erf_sf_dl.subs(substitutions),
#                                     modules=mods)
#         erf_sf_dx_dll = sympy.lambdify(x, erf_sf_dx_dl.subs(substitutions),
#                                        modules=mods)
#         # 3- Compare siegpy results and analytic results
#         grid = np.linspace(-2*x00, 2*x00, 31)
#         sfe.grid = grid
#         np.testing.assert_array_almost_equal(sfe.values, erf_sfl(grid))
#         np.testing.assert_array_almost_equal(sfe.dx_values, erf_sf_dxl(grid))
#         np.testing.assert_array_almost_equal(sfe.dx2_values, erf_sf_dx2l(grid))  # noqa
#         np.testing.assert_array_almost_equal(sfe.dx3_values, erf_sf_dx3l(grid))  # noqa
#         np.testing.assert_array_almost_equal(sfe.dxi_values['x0'],
#                                              erf_sf_dx0l(grid))
#         np.testing.assert_array_almost_equal(sfe.dx_dxi_values['x0'],
#                                              erf_sf_dx_dx0l(grid))
#         np.testing.assert_array_almost_equal(sfe.dxi_values['lbda'],
#                                              erf_sf_dll(grid))
#         np.testing.assert_array_almost_equal(sfe.dx_dxi_values['lbda'],
#                                              erf_sf_dx_dll(grid))
#         # The same type of test can be performed for the other type of
#         # smooth function:
#         # 1- Create all the symbolic expressions:
#         tanh_sf = 1 + (sympy.tanh(lbda*(x-x0)) - sympy.tanh(lbda*(x+x0)))/2
#         tanh_sf_dx = tanh_sf.diff(x)
#         tanh_sf_dx2 = tanh_sf.diff(x, 2)
#         tanh_sf_dx3 = tanh_sf.diff(x, 3)
#         tanh_sf_dx0 = tanh_sf.diff(x0)
#         tanh_sf_dx_dx0 = tanh_sf_dx.diff(x0)
#         tanh_sf_dl = tanh_sf.diff(lbda)
#         tanh_sf_dx_dl = tanh_sf_dx.diff(lbda)
#         # 2- lambdify them in order to evaluate them for a given grid
#         tanh_sfl = sympy.lambdify(x, tanh_sf.subs(substitutions))
#         tanh_sf_dxl = sympy.lambdify(x, tanh_sf_dx.subs(substitutions))
#         tanh_sf_dx2l = sympy.lambdify(x, tanh_sf_dx2.subs(substitutions))
#         tanh_sf_dx3l = sympy.lambdify(x, tanh_sf_dx3.subs(substitutions))
#         tanh_sf_dx0l = sympy.lambdify(x, tanh_sf_dx0.subs(substitutions))
#         tanh_sf_dx_dx0l = sympy.lambdify(x, tanh_sf_dx_dx0.subs(substitutions))  # noqa
#         tanh_sf_dll = sympy.lambdify(x, tanh_sf_dl.subs(substitutions))
#         tanh_sf_dx_dll = sympy.lambdify(x, tanh_sf_dx_dl.subs(substitutions))
#         # 3- Compare siegpy results and analytic results
#         sft = TanhSmoothFunction(x00, lbda0, grid=grid)
#         np.testing.assert_array_almost_equal(sft.values, tanh_sfl(grid))
#         np.testing.assert_array_almost_equal(sft.dx_values, tanh_sf_dxl(grid))  # noqa
#         np.testing.assert_array_almost_equal(sft.dx2_values,
#                                              tanh_sf_dx2l(grid))
#         np.testing.assert_array_almost_equal(sft.dx3_values,
#                                              tanh_sf_dx3l(grid))
#         np.testing.assert_array_almost_equal(sft.dxi_values['x0'],
#                                              tanh_sf_dx0l(grid))
#         np.testing.assert_array_almost_equal(sft.dx_dxi_values['x0'],
#                                              tanh_sf_dx_dx0l(grid))
#         np.testing.assert_array_almost_equal(sft.dxi_values['lbda'],
#                                              tanh_sf_dll(grid))
#         np.testing.assert_array_almost_equal(sft.dx_dxi_values['lbda'],
#                                              tanh_sf_dx_dll(grid))
