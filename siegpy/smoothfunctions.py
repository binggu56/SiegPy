# -*- coding: utf-8 -*
r"""
The :class:`SmoothFunction` class and its methods are defined hereafter.

It is used as the base class of two more specific smooth functions:

* the :class:`TanhSmoothFunction` class,
* and the :class:`ErfSmoothFunction` class.
"""

from abc import abstractmethod, ABCMeta
import numpy as np
from scipy.special import erf


class SmoothFunction(metaclass=ABCMeta):
    r"""
    .. note:: This is an abstract class.

    Smooth functions are used when a smooth complex scaling is applied
    to the potential. The aim of this class is to easily update the
    values of the smooth functions (and its derivatives) when the
    Refection-Free Complex Absorbing Potentials and the Virial operator
    are defined.
    """

    PARAMETERS = ["x0", "lbda"]

    def __init__(self, x0, lbda, c0=1, cp=1, cm=-1, grid=None):
        r"""
        The smooth functions :math:`q` used here are of the form:

        .. math::

            q(x) = c_0 + c_+ r(\lambda (x - x_0)) - c_- r(\lambda (x + x_0))

        The smooth function should be 0 on a large part of the
        :math:`[-x_0, x_0]` range of the grid, while tending to 1 at
        both infinities.

        The initialization of a smooth function requires the value of
        :math:`x_0` and of the parameter :math:`\lambda`, indicating
        how smoothly the function goes from 0 to 1 (the larger, the
        sharper).

        Parameters
        ----------
        x0: float
            Inflection point.
        lbda: float
            Sharpness parameter (the function is smoother for
            smaller values).
        c0: float
            Constant term :math:`c_0` of the smooth function defintion
            (default to 1).
        cp: float
            Constant term :math:`c_+` of the smooth function defintion
            (default to 1).
        cm: float
            Constant term :math:`c_-` of the smooth function defintion
            (default to -1).
        numpy array
            Discretization grid of the smooth function (optional).

        Raises
        ------
        ValueError
            If ``x0`` or ``lbda`` are not strictly positive.
        """
        # Check the value of x0 and lbda before initializing the
        # corresponding attributes.
        if x0 <= 0:
            raise ValueError("The inflection point must be positive.")
        if lbda <= 0:
            raise ValueError("The sharpness parameter must be positive.")
        self._x0 = x0
        self._lbda = lbda
        # Set initial values for the smooth function
        self._c0 = c0
        self._cp = cp
        self._cm = cm
        # Use the grid setter
        if grid is not None:
            self.grid = np.array(grid)
        else:
            self.grid = grid

    @property
    def x0(self):
        r"""
        Returns
        -------
        float
            Inflection point.
        """
        return self._x0

    @property
    def lbda(self):
        r"""
        Returns
        -------
        float
            Sharpness parameter.
        """
        return self._lbda

    @property
    def grid(self):
        r"""
        Returns
        -------
        numpy array
            Discretization grid of the smooth function.
        """
        return self._grid

    @grid.setter
    def grid(self, new_grid):
        r"""
        Setter of the :attr:`grid` attribute.

        Parameters
        ----------
        new_grid: numpy array
            New discretization grid.
        """
        if self._to_be_updated(new_grid):
            self._grid = new_grid
            self._update_all_values()

    def _to_be_updated(self, new_grid):
        r"""
        Returns
        -------
        bool
            ``True`` if the grid has to be updated.
        """
        return not (hasattr(self, "grid") and np.array_equal(self.grid, new_grid))

    def _update_all_values(self):
        r"""
        Update the values of the test functions and all of its
        derivatives.
        """
        grid = self.grid
        lbda = self.lbda
        if grid is not None:
            cp = self._cp
            cm = self._cm
            grid_p = lbda * (grid - self.x0)
            grid_m = lbda * (grid + self.x0)
            r_p = self._get_r_values(grid_p)
            r_m = self._get_r_values(grid_m)
            r_dx_p = self._get_r_dx_values(grid_p)
            r_dx_m = self._get_r_dx_values(grid_m)
            r_dx2_p = self._get_r_dx2_values(grid_p)
            r_dx2_m = self._get_r_dx2_values(grid_m)
            r_dx3_p = self._get_r_dx3_values(grid_p)
            r_dx3_m = self._get_r_dx3_values(grid_m)
            self._values = self._c0 + (cp * r_p + cm * r_m) / 2
            self._dx_values = lbda * (cp * r_dx_p + cm * r_dx_m) / 2
            self._dx2_values = lbda ** 2 * (cp * r_dx2_p + cm * r_dx2_m) / 2
            self._dx3_values = lbda ** 3 * (cp * r_dx3_p + cm * r_dx3_m) / 2
            self._dxi_values = {}
            self._dx_dxi_values = {}
            self._dxi_values["x0"] = -lbda * (cp * r_dx_p - cm * r_dx_m) / 2
            self._dx_dxi_values["x0"] = -lbda ** 2 * (cp * r_dx2_p - cm * r_dx2_m) / 2
            self._dxi_values["lbda"] = (cp * grid_p * r_dx_p + cm * grid_m * r_dx_m) / (
                2 * lbda
            )
            self._dx_dxi_values["lbda"] = (
                cp * (grid_p * r_dx2_p + r_dx_p) + cm * (grid_m * r_dx2_m + r_dx_m)
            ) / 2
        else:
            self._values = None
            self._dx_values = None
            self._dx2_values = None
            self._dx3_values = None
            self._dxi_values = {param: None for param in self.PARAMETERS}
            self._dx_dxi_values = {param: None for param in self.PARAMETERS}

    @property
    def values(self):
        r"""
        Returns
        -------
        numpy array
            Values of the smooth function.
        """
        return self._values

    @property
    def dx_values(self):
        r"""
        Returns
        -------
        numpy array
            Values of the first derivative of the smooth function.
        """
        return self._dx_values

    @property
    def dx2_values(self):
        r"""
        Returns
        -------
        numpy array
            Values of the second derivative of the smooth
            function.
        """
        return self._dx2_values

    @property
    def dx3_values(self):
        r"""
        Returns
        -------
        numpy array
            Values of the third derivative of the smooth function.
        """
        return self._dx3_values

    @property
    def dxi_values(self):
        r"""
        Returns
        -------
        numpy array
            Values of the first derivative of the smooth function
            with respect to both parameters :math:`x_0` and
            :math:`\lambda`.
        """
        return self._dxi_values

    @property
    def dx_dxi_values(self):
        r"""
        Returns
        -------
        numpy array
            Values of the first derivative with respect to both
            parameters :math:`x_0` and :math:`\lambda` of the first
            derivative of the smooth function.
        """
        return self._dx_dxi_values

    @abstractmethod
    def _get_r_values(self, grid):  # pragma: no cover
        r"""
        .. note:: This is an asbtract method.

        Parameters
        ----------
        grid: numpy array
            Discretization grid.

        Returns
        -------
        numpy array
            Values of the function :math:`r`, evaluated on a
            given set of grid points.
        """
        pass

    @abstractmethod
    def _get_r_dx_values(self, grid):  # pragma: no cover
        r"""
        .. note:: This is an asbtract method.

        Parameters
        ----------
        grid: numpy array
            Discretization grid.

        Returns
        -------
            Values of first derivative of the function :math:`r`,
            evaluated on a given set of grid points.
        """
        pass

    @abstractmethod
    def _get_r_dx2_values(self, grid):  # pragma: no cover
        r"""
        .. note:: This is an asbtract method.

        Parameters
        ----------
        grid: numpy array
            Discretization grid.

        Returns
        -------
            Values of second derivative of the function :math:`r`,
            evaluated on a given set of grid points.
        """
        pass

    @abstractmethod
    def _get_r_dx3_values(self, grid):  # pragma: no cover
        r"""
        .. note:: This is an asbtract method.

        Parameters
        ----------
        grid: numpy array
            Discretization grid.

        Returns
        -------
            Values of third derivative of the function :math:`r`,
            evaluated on a given set of grid points.
        """
        pass


class ErfSmoothFunction(SmoothFunction):
    r"""
    In this case, the function :math:`r` corresponds to the error
    function :math:`\text{erf}`.
    """

    def _get_r_values(self, grid):
        r"""
        Parameters
        ----------
        grid: numpy array
            Discretization grid.

        Returns
        -------
            Values of the function :math:`\text{erf}`, evaluated on a
            given set of grid points.
        """
        return erf(grid)

    def _get_r_dx_values(self, grid):
        r"""
        Parameters
        ----------
        grid: numpy array
            Discretization grid.

        Returns
        -------
            Values of first derivative of the function
            :math:`\text{erf}`, evaluated on a given set of grid points.
        """
        return 2 / np.sqrt(np.pi) * np.exp(-grid ** 2)

    def _get_r_dx2_values(self, grid):
        r"""
        Parameters
        ----------
        grid: numpy array
            Discretization grid.

        Returns
        -------
            Values of second derivative of the function
            :math:`\text{erf}`, evaluated on a given set of grid points.
        """
        return -2 * grid * self._get_r_dx_values(grid)

    def _get_r_dx3_values(self, grid):
        r"""
        Parameters
        ----------
        grid: numpy array
            Discretization grid.

        Returns
        -------
            Values of third derivative of the function
            :math:`\text{erf}`, evaluated on a given set of grid points.
        """
        return -2 * (self._get_r_dx_values(grid) + grid * self._get_r_dx2_values(grid))


class TanhSmoothFunction(SmoothFunction):
    r"""
    In this case, the function :math:`r` corresponds to the hyperbolic
    tangent :math:`\tanh`.
    """

    def _get_r_values(self, grid):
        r"""
        Parameters
        ----------
        grid: numpy array
            Discretization grid.

        Returns
        -------
            Values of the function :math:`\tanh`, evaluated on a
            given set of grid points.
        """
        return np.tanh(grid)

    def _get_r_dx_values(self, grid):
        r"""
        Parameters
        ----------
        grid: numpy array
            Discretization grid.

        Returns
        -------
            Values of first derivative of the function :math:`\tanh`,
            evaluated on a given set of grid points.
        """
        return 1 / np.cosh(grid) ** 2

    def _get_r_dx2_values(self, grid):
        r"""
        Parameters
        ----------
        grid: numpy array
            Discretization grid.

        Returns
        -------
            Values of second derivative of the function :math:`\tanh`,
            evaluated on a given set of grid points.
        """
        return -2 * np.tanh(grid) / np.cosh(grid) ** 2

    def _get_r_dx3_values(self, grid):
        r"""
        Parameters
        ----------
        grid: numpy array
            Discretization grid.

        Returns
        -------
            Values of third derivative of the function :math:`\tanh`,
            evaluated on a given set of grid points.
        """
        r = self._get_r_values(grid)
        r_dx = self._get_r_dx_values(grid)
        r_dx2 = self._get_r_dx2_values(grid)
        return -2 * (r_dx ** 2 + r * r_dx2)
