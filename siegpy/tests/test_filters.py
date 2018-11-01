# -*- coding: utf-8 -*
import numpy as np
import pytest
# Test that all the filters are imported:
from siegpy.filters import FD2_filters, FD8_filters, Sym8_filters


class TestFilters():

    @pytest.mark.parametrize("values, expected", [
        (FD8_filters.grad_filter,
         [1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280]),
        (FD8_filters.laplac_filter,
         [-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560]),
        (Sym8_filters.magic_filter,
         [0.0e0, 2.727344929119796596577e-6,
          -0.00005185986881173432922849e0, 0.0004944322768868991919228e0,
          -0.003441281444934938572809e0, 0.01337263414854794752733e0,
          -0.02103025160930381434955e0, -0.06048952891969835160028e0,
          0.9940415697834003993179e0, 0.06126258958312079821954e0,
          0.02373821463724942397566e0, -0.009420470302010803859227e0,
          0.001747237136729939034494e0, -0.0003015803813269046316716e0,
          0.00008762984476210559564689e0, -0.00001290557201342060969517e0,
          8.433424733352934109473e-7])
    ])
    def test_init(self, values, expected):
        np.testing.assert_array_almost_equal(values, expected)

    @pytest.mark.parametrize("values, expected", [
        (FD2_filters.fill_gradient_matrix(3),
         np.array([[0, 0.5, 0], [-0.5, 0, 0.5], [0, -0.5, 0]])),
        (FD2_filters.fill_laplacian_matrix(3),
         np.array([[-2, 1, 0], [1, -2, 1], [0, 1, -2]]))
    ])
    def test_fill_matrices(self, values, expected):
        np.testing.assert_array_almost_equal(values, expected)

    @pytest.mark.parametrize("to_evaluate", [
        "Sym8_filters.fill_magic_matrix(3)",
        "FD8_filters.fill_gradient_matrix(3)"
    ])
    def test_fill_matrices_raises_ValueError(self, to_evaluate):
        with pytest.raises(ValueError):
            eval(to_evaluate)
