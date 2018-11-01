# -*- coding: utf-8 -*
r"""
The :class:`Filters` class is defined below.

The :class:`WaveletFilters` class inherits from the previous class and
is specific to Daubechies wavelets, where a so-called magic filter has
to be defined, in addition to the gradient and laplacian filters.

Both classes are used to define some families of filters:

* FD2_filters, for the Finite Difference filters of order 2,
* FD8_filters, for the Finite Difference filters of order 8,
* Sym8_filters, for a family of Daubechies wavelets filters.

These three families of filters are easily available:

>>> from siegpy import FD2_filters, FD8_filters, Sym8_filters
"""

import numpy as np


class Filters():
    r"""
    This class specifies a family of filters that are useful to describe
    numerical Hamiltonians. It allows the definition of the gradient and
    Laplacian operators in matrix form by means of filters and a
    specified number of grid points.
    """

    def __init__(self, grad_filter, laplac_filter):
        r"""
        Parameters
        ----------
        grad_filter: numpy array or list
            Gradient filter.
        laplac_filter: numpy array or list
            Laplacian filter.
        """
        self._grad_filter = np.array(grad_filter)
        self._laplac_filter = np.array(laplac_filter)

    @property
    def grad_filter(self):
        r"""
        Returns
        -------
        numpy array
            Gradient filter.
        """
        return self._grad_filter

    @property
    def laplac_filter(self):
        r"""
        Returns
        -------
        numpy array
            Laplacian filter.
        """
        return self._laplac_filter

    def fill_gradient_matrix(self, npts):
        r"""
        Fill a gradient matrix of dimension ``npts``.

        Parameters
        ----------
        npts: int
            Number of grid points.

        Returns
        -------
        2D numpy array
            Gradient matrix.
        """
        return self._fill_matrix(self.grad_filter, npts)

    def fill_laplacian_matrix(self, npts):
        r"""
        Fill a Laplacian matrix of dimension ``npts``.

        Parameters
        ----------
        npts: int
            Number of grid points.

        Returns
        -------
        2D numpy array
            Laplacian matrix.
        """
        return self._fill_matrix(self.laplac_filter, npts)

    @staticmethod
    def _fill_matrix(_filter, npts):
        r"""
        This method creates a matrix operator, given a filter and a
        number of grid points.

        Parameters
        ----------
        _filter: numpy array
            Filter of an operator.
        npts: int
            Number of grid points.

        Returns
        -------
        numpy array
            Operator in matrix form, filled thanks to a filter.

        Raises
        ------
        ValueError
            If the filter is larger than the discretization grid.
        """
        # Check that input values are consistent
        if len(_filter) > npts:
            raise ValueError("The filter is larger than the number of grid "
                             "points ({} > {}). Increase npts."
                             .format(len(_filter), npts))
        # Fill the operator matrix via diagonal matrices with offset
        operator = np.zeros((npts, npts))
        half_len = len(_filter) // 2
        for i, val in enumerate(_filter):
            operator += val * np.eye(npts, k=i-half_len)
        return operator


class WaveletFilters(Filters):
    r"""
    This class defines the methods specific to the Daubechies wavelets
    filers.
    """

    def __init__(self, grad_filter, laplac_filter, magic_filter):
        r"""
        The main difference with respect to the parent class is the
        requirement of a so-called magic filter.

        Parameters
        ----------
        grad_filter: numpy array or list
            Gradient filter.
        laplac_filter: numpy array or list
            Laplacian filter.
        magic_filter: numpy array or list
            Magic filter.
        """
        self._magic_filter = np.array(magic_filter)
        super().__init__(grad_filter, laplac_filter)

    @property
    def magic_filter(self):
        r"""
        Returns
        -------
        numpy array
            Magic filter.
        """
        return self._magic_filter

    def fill_magic_matrix(self, npts):
        r"""
        Fill a magic filter matrix of dimension ``npts``.

        Parameters
        ----------
        npts: int
            Number of grid points.

        Returns
        -------
        2D numpy array
            Magic filter matrix.
        """
        return self._fill_matrix(self.magic_filter, npts)


# Initialization of the family of Finite Difference filters of order 2
FD2_grad = [-0.5, 0, 0.5]
FD2_laplac = [1, -2, 1]
FD2_filters = Filters(FD2_grad, FD2_laplac)


# Initialization of the family of Finite Difference filters of order 8
FD8_grad = [1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280]
FD8_laplac = [-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560]
FD8_filters = Filters(FD8_grad, FD8_laplac)


# Initialization of a family of Daubechies wavelets filters
Sym8_grad = [-1.58546475167710251017e-19, 1.240078536096648535363e-14,
             -7.252206916665149851159e-13, -9.697184925637300947553e-10,
             -7.207948238588481597102e-8, 3.993810456408053712134e-8,
             2.451992111053665419192e-7, 0.00007667706908380351933902e0,
             -0.001031530213375445369098e0, 0.00695837911645070749502e0,
             -0.03129014783948023634382e0, 0.1063640682894442760935e0,
             -0.3032593514765938346888e0, 0.8834460460908270942786e0, 0.0e0,
             -0.8834460460908270942786e0, 0.3032593514765938346888e0,
             -0.1063640682894442760935e0, 0.03129014783948023634382e0,
             -0.00695837911645070749502e0, 0.001031530213375445369098e0,
             -0.00007667706908380351933902e0, -2.451992111053665419192e-7,
             -3.993810456408053712134e-8, 7.207948238588481597102e-8,
             9.697184925637300947553e-10, 7.252206916665149851159e-13,
             -1.240078536096648535363e-14, 1.58546475167710251017e-19]
Sym8_grad.reverse()
Sym8_laplac = [-6.924474940639200179928e-18, 2.708004936263194382774e-13,
               -5.81387983028254054796e-11, -1.058570554967414703735e-8,
               -3.723076304736927584879e-7, 2.090423495292036595792e-6,
               -0.00002398228524507599670406e0, 0.0004516792028750223534948e0,
               -0.004097656893426338238993e0, 0.0220702918848225552379e0,
               -0.08226639997421233409877e0, 0.2371780582153805636239e0,
               -0.6156141465570069496315e0, 2.219146593891116389879e0,
               -3.55369228991319019413e0, 2.219146593891116389879e0,
               -0.6156141465570069496315e0, 0.2371780582153805636239e0,
               -0.08226639997421233409877e0, 0.0220702918848225552379e0,
               -0.004097656893426338238993e0, 0.0004516792028750223534948e0,
               -0.00002398228524507599670406e0, 2.090423495292036595792e-6,
               -3.723076304736927584879e-7, -1.058570554967414703735e-8,
               -5.81387983028254054796e-11, 2.708004936263194382774e-13,
               -6.924474940639200179928e-18]
Sym8_magic = [0.0e0, 2.727344929119796596577e-6,
              -0.00005185986881173432922849e0, 0.0004944322768868991919228e0,
              -0.003441281444934938572809e0, 0.01337263414854794752733e0,
              -0.02103025160930381434955e0, -0.06048952891969835160028e0,
              0.9940415697834003993179e0, 0.06126258958312079821954e0,
              0.02373821463724942397566e0, -0.009420470302010803859227e0,
              0.001747237136729939034494e0, -0.0003015803813269046316716e0,
              0.00008762984476210559564689e0, -0.00001290557201342060969517e0,
              8.433424733352934109473e-7]
Sym8_filters = WaveletFilters(Sym8_grad, Sym8_laplac, Sym8_magic)
