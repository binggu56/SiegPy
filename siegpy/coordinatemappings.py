# -*- coding: utf-8 -*
r"""
Various classes are defined to represent different **coordinate mappings**.
They define the coordinate transformation (and its derivatives) so that
the Reflection-Free Complex Absorbing Potentials and the virial operator
can be easily discretized over the discretization grid.

The :class:`CoordMap` class is the most basic one. It is an abstract
class, requiring only a complex scaling angle :math:`\theta` and a
discretization grid.

The **Uniform Complex Scaling** (UCS) transformation
:math:`F_{UCS}: x \mapsto x e^{i \theta}` is easily derived from it in
the :class:`UniformCoordMap` class.

Another well-known transformation is the **Exterior Complex Scaling**
(ECS), that leaves the potential unscaled inside a region
:math:`[-x_0, x_0]` (*i.e.*, :math:`F_{ECS}: x \mapsto x`), while
amounting to the UCS outside (*i.e.* :math:`(x - x_0) e^{i \theta}` for
:math:`x > x_0`). This has the advantage of leaving the innermost
potential unscaled.

However, the most efficient coordinate transformations discussed in the
literature (when it comes to finding Siegert states numerically) are
know as **Smooth Exterior Complex Scaling** (SECS). Contrary to the
usual ECS, there are smooth transitions between both regimes, hence
their name. The sharpness of these transitions is then controlled by
another parameter, :math:`\lambda`. The abstract base class
:class:`SmoothExtCoordMap` allows for the representation of such
coordinate transformations.

A SECS generally relies on a function :math:`q` that smoothly goes from
0 to 1 around :math:`\pm x_0`. This is why a :class:`SmoothFuncCoordMap`
class is also defined as an abstract base class deriving from the
:class:`SmoothExtCoordMap` class. In practice, all the implemented SECS
implemented in SiegPy derive from the :class:`SmoothFuncCoordMap` class.

There are two main possibilites to define a smooth coordinate
transformations by using a smooth function :math:`q`:

* :math:`F_{KG}: x \mapsto x e^{i \theta q(x)}`, that will be called the
  Kalita-Gupta (KG) coordinate transformation,
* :math:`F_{S}: x \mapsto F_S(x)` such that its derivative with respect to
  :math:`x` is :math:`F_S^\prime = f_S: x \mapsto 1 + (e^{i \theta} - 1) q(x)`.
  This will be labeled as the Simon coordinate transformation.

This is the reason why two other abstract base classes were defined:

* the :class:`KGCoordMap` class,
* and the :class:`SimonCoordMap` class.

They both require a smooth function :math:`q` as parameters.

Two main types of smooth functions are implemented in SiegPy: one based
on the error function :math:`\text{erf}`, the other on :math:`\tanh`,
hence giving rise to four classes that can readily be used to find
Siegert states numerically:

* the :class:`ErfKGCoordMap` class,
* the :class:`ErfSimonCoordMap` class,
* the :class:`TanhKGCoordMap` class,
* the :class:`TanhSimonCoordMap` class.

See :class:`~siegpy.smoothfunctions` for more details on the smooth
functions.
"""

from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.special import erf
from .smoothfunctions import ErfSmoothFunction, TanhSmoothFunction


class CoordMap(metaclass=ABCMeta):
    r"""
    .. note:: This is an abstract class.

    Base class of all the other coordinate mappings.
    """

    PARAMETERS = ["theta"]

    def __init__(self, theta, grid, GCVT):
        r"""
        Parameters
        ----------
        theta: float
            Complex scaling angle.
        grid: numpy array or None
            Discretization grid.
        GCVT: bool
            Stands for Generalized Complex Virial Theorem. If it is set
            to ``True``, then only one virial value is computed, else
            there are one per coordinate mapping parameter.
        """
        self._theta = theta
        self._GCVT = GCVT
        if grid is not None:
            self.grid = np.array(grid)
        else:
            self.grid = grid

    @property
    def theta(self):
        r"""
        Returns
        -------
        float
            Complex scaling angle.
        """
        return self._theta

    @property
    def GCVT(self):
        r"""
        Returns
        -------
        bool
            Parameter stating which virial operator(s) have to be used.
        """
        return self._GCVT

    @property
    def grid(self):
        r"""
        Returns
        -------
        numpy array
            Discretization grid.
        """
        return self._grid

    @grid.setter
    def grid(self, new_grid):
        r"""
        Setter of the grid, updating the values of the coordinate
        transformation, its derivatives, the Reflection-Free Complex
        Absorbing Potentials (RF-CAP) and the potentials of the virial
        operator.

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
        Parameters
        ----------
        new_grid: numpy array
            New discretization grid.

        Returns
        -------
        bool
            ``True`` if the grid has to be updated.
        """
        if hasattr(self, "_grid") and np.array_equal(self._grid, new_grid):
            return False
        return True

    @property
    def values(self):
        r"""
        Returns
        -------
        numpy array
            Values of the coordinate transorfmation.
        """
        return self._values

    @property
    def f_values(self):
        r"""
        Returns
        -------
        numpy array
            Values of the first derivative of the coordinate
            transorfmation.
        """
        return self._f_values

    @property
    def f_dx_values(self):
        r"""
        Returns
        -------
        numpy array
            Values of the second derivative of the coordinate
            transorfmation.
        """
        return self._f_dx_values

    @property
    def f_dx2_values(self):
        r"""
        Returns
        -------
        numpy array
            Values of the third derivative of the coordinate
            transorfmation.
        """
        return self._f_dx2_values

    @property
    def dxi_values(self):
        r"""
        Returns
        -------
        numpy array
            Values of the first derivative with respect to all the
            parameters of the coordinate transorfmation.
        """
        return self._dxi_values

    @property
    def f_dxi_values(self):
        r"""
        Returns
        -------
        numpy array
            Values of the first derivative with respect to all the
            parameters of the first derivative of the coordinate
            transorfmation.
        """
        return self._f_dxi_values

    @property
    def V0_values(self):
        r"""
        Returns
        -------
        numpy array
            Values of the first additional RF-CAP.
        """
        return self._V0_values

    @property
    def V1_values(self):
        r"""
        Returns
        -------
        numpy array
            Values of the second additional RF-CAP.
        """
        return self._V1_values

    @property
    def V2_values(self):
        r"""
        Returns
        -------
        numpy array
            Values of the third additional RF-CAP.
        """
        return self._V2_values

    @property
    def U0_values(self):
        r"""
        Returns
        -------
        numpy array
            Values of the first additional virial operator potential.
        """
        return self._U0_values

    @property
    def U1_values(self):
        r"""
        Returns
        -------
        numpy array
            Values of the second additional virial operator potential.
        """
        return self._U1_values

    @property
    def U2_values(self):
        r"""
        Returns
        -------
        numpy array
            Values of the third additional virial operator potential.
        """
        return self._U2_values

    @property
    def U11_values(self):
        r"""
        Returns
        -------
        numpy array
            Values of the fourth additional virial operator potential.
        """
        return self._U11_values

    def _update_all_values(self):
        r"""
        Update the values of the coordinate mapping and its derivatives
        with respect to :math:`x` and with respect to the coordinate
        mapping parameters, as well as related quantities such as the
        Reflection-Free Complex Absorbing Potential and the virial
        operator.
        """
        self._update_x_deriv_values()
        self._update_param_deriv_values()
        self._update_RFCAP_values()
        self._update_virial_values()

    def _update_x_deriv_values(self):
        r"""
        Update the values of the coordinate mapping and its three first
        derivatives with respect to :math:`x`.
        """
        if self.grid is not None:
            self._update_x_deriv_from_grid()
        else:
            self._values = None
            self._f_values = None
            self._f_dx_values = None
            self._f_dx2_values = None

    def _update_x_deriv_from_grid(self):
        r"""
        Update the values of the coordinate mapping and its three first
        derivatives with respect to :math:`x`, if the grid is not
        ``None``.
        """
        self._values = self._get_values()  # pylint: disable=E1111
        self._f_values = self._get_f_values()
        self._f_dx_values = self._get_f_dx_values()
        self._f_dx2_values = self._get_f_dx2_values()

    def _update_param_deriv_values(self):
        r"""
        Update the values of the derivative with respect to the
        coordinate mapping parameters of the coordinate mapping and its
        first derivative with respect to :math:`x`.
        """
        if self.grid is not None and not self.GCVT:
            self._update_param_deriv_from_grid()
        else:
            self._dxi_values = {param: None for param in self.PARAMETERS}
            self._f_dxi_values = {param: None for param in self.PARAMETERS}

    def _update_param_deriv_from_grid(self):
        r"""
        Update the values of the derivative with respect to the
        coordinate mapping parameters of the coordinate mapping and its
        first derivative with respect to :math:`x`, if the grid is not
        ``None``.
        """
        self._dxi_values = self._get_dxi_values()
        self._f_dxi_values = self._get_f_dxi_values()

    @abstractmethod
    def _get_values(self, *args):  # pragma: no cover
        r"""
        .. note:: This is an abstract method.

        Evaluate the values of the coordinate mapping with respect to
        :math:`x`.
        """
        pass

    @abstractmethod
    def _get_f_values(self, *args):  # pragma: no cover
        r"""
        .. note:: This is an abstract method.

        Evaluate the values of the first derivative of the coordinate
        mapping with respect to :math:`x`.
        """
        pass

    @abstractmethod
    def _get_f_dx_values(self, *args):  # pragma: no cover
        r"""
        .. note:: This is an abstract method.

        Evaluate the values of the second derivative of the coordinate
        mapping with respect to :math:`x`.
        """
        pass

    @abstractmethod
    def _get_f_dx2_values(self, *args):  # pragma: no cover
        r"""
        .. note:: This is an abstract method.

        Evaluate the values of the third derivative of the coordinate
        mapping with respect to :math:`x`.
        """
        pass

    @abstractmethod
    def _get_dxi_values(self, *args):  # pragma: no cover
        r"""
        .. note:: This is an abstract method.

        Evaluate the values of the first derivative of the coordinate
        mapping with respect to the coordinate mapping parameters.
        """
        pass

    @abstractmethod
    def _get_f_dxi_values(self, *args):  # pragma: no cover
        r"""
        .. note:: This is an abstract method.

        Evaluate the values of the first derivative with respect to the
        coordinate mapping parameters of the first derivative with
        respect to :math:`x` of the coordinate mapping.
        """
        pass

    def _update_RFCAP_values(self):
        r"""
        Update the various potentials used to define the Reflection-Free
        Complex Absorbing Potentials
        """
        if self.grid is not None:
            f = self.f_values
            f_dx = self.f_dx_values
            f_dx2 = self.f_dx2_values
            self._V0_values = 1 / 4 * f_dx2 / f ** 3 - 5 / 8 * f_dx ** 2 / f ** 4
            self._V1_values = f_dx / f ** 3
            self._V2_values = 1 / 2 * (1 - 1 / f ** 2)
        else:
            self._V0_values = None
            self._V1_values = None
            self._V2_values = None

    def _update_virial_values(self):
        r"""
        Update the values of the different potentials used to define the
        various virial operators.
        """
        # Initialize some variables for readability
        F = self.values
        f = self.f_values
        f_dx = self.f_dx_values
        f_dx2 = self.f_dx2_values
        if self.GCVT:
            # Initialize the GCVT potentials if required
            if self.grid is not None:
                # The virial operator potentials are numpy arrays
                self._U0_values = 1 - F * f_dx / f ** 2
                self._U1_values = F / f
            else:
                self._U0_values = None
                self._U1_values = None
            self._U2_values = None
            self._U11_values = None
        else:
            # Else initialize the usual virial potentials for each
            # parameter. They are all stored in a dictionary, whose keys
            # are the labels of the parameters.
            if self.grid is not None:
                U0_factor = 1 / 2 * (f_dx ** 2 / f ** 5 + 1 / 2 * f_dx2 / f ** 4)
                U1_factor = -1 / 2 * f_dx / f ** 4
                U2_factor = 1 / (2 * f ** 3)
                U11_factor = -1 / (2 * f ** 3)
                self._U0_values = {
                    param: U0_factor * self.f_dxi_values[param]
                    for param in self.PARAMETERS
                }
                self._U1_values = {
                    param: U1_factor * self.f_dxi_values[param]
                    for param in self.PARAMETERS
                }
                self._U2_values = {
                    param: U2_factor * self.f_dxi_values[param]
                    for param in self.PARAMETERS
                }
                self._U11_values = {
                    param: U11_factor * self.f_dxi_values[param]
                    for param in self.PARAMETERS
                }
            else:
                self._U0_values = {param: None for param in self.PARAMETERS}
                self._U1_values = {param: None for param in self.PARAMETERS}
                self._U2_values = {param: None for param in self.PARAMETERS}
                self._U11_values = {param: None for param in self.PARAMETERS}


class UniformCoordMap(CoordMap):
    r"""
    The uniform coordinate transformation corresponds to:
    :math:`x \mapsto x e^{i \theta}`.

    """

    def __init__(self, theta, GCVT=True, grid=None):
        r"""
        Parameters
        ----------
        theta: float
            Complex scaling angle.
        GCVT: bool
            Stands for Generalized Complex Virial Theorem. If it is set
            to ``True``, then only one virial value is computed, else
            there are one per parameter (here, :attr:`theta`). Defaults
            to ``True``.
        grid: numpy array
            Discretization grid (optional).
        """
        super().__init__(theta, grid, GCVT)

    def _get_values(self):
        r"""
        Returns
        -------
        numpy array
            Values of the coordinate mapping with respect to :math:`x`.
        """
        return self.grid * np.exp(1j * self.theta)

    def _get_f_values(self):
        r"""
        Returns
        -------
        numpy array
            Values of the first derivative of the coordinate mapping
            with respect to :math:`x`.
        """
        return np.exp(1j * self.theta) * np.ones_like(self.grid)

    def _get_f_dx_values(self):
        r"""
        Returns
        -------
        numpy array
            Values of the second derivative of the coordinate mapping
            with respect to :math:`x`.
        """
        return 0

    def _get_f_dx2_values(self):
        r"""
        Returns
        -------
        numpy array
            Values of the third derivative of the coordinate mapping
            with respect to :math:`x`.
        """
        return 0

    def _get_dxi_values(self):
        r"""
        Returns
        -------
        numpy array
            Values of the first derivative of the coordinate mapping
            with respect to the coordinate mapping parameters.
        """
        return {"theta": 1j * self.values}

    def _get_f_dxi_values(self):
        r"""
        Returns
        -------
        numpy array
            Values of the first derivative with respect to the
            coordinate mapping parameters of the first derivative with
            respect to :math:`x` of the coordinate mapping.
        """
        return {"theta": 1j * self.f_values}


class SmoothExtCoordMap(CoordMap, metaclass=ABCMeta):
    r"""
    .. note:: This is an abstract class.

    This is the base class for all the other classes implementing a
    particular type of smooth exterior complex scaling.

    .. warning::

        This class must be used to create child classes if no smooth
        function is actually required.
    """

    PARAMETERS = CoordMap.PARAMETERS + ["x0", "lbda"]

    def __init__(self, theta, x0, lbda, GCVT, grid):
        r"""
        Parameters
        ----------
        theta: float
            Complex scaling angle.
        x0: float
            Inflection point.
        lbda: float
            Sharpness parameter
        GCVT: bool
            Stands for Generalized Complex Virial Theorem. If it is set
            to ``True``, then only one virial value is computed, else
            there are three, one for each parameter (here,
            :attr:`theta`, :attr:`x0` and :attr:`lbda`).
        grid: numpy array or None
            Discretization grid.

        Raises
        ------
        ValueError
            If the ``x0`` or ``lbda`` is not positive.
        """
        # LEAVE THIS COMMENT
        # # The values must be checked if there is a child class that
        # # does not require a smooth function.
        # if x0 <= 0:
        #     raise ValueError("The inflection point must be positive.")
        # if lbda <= 0:
        #     raise ValueError("The sharpness parameter must be positive.")
        self._x0 = x0
        self._lbda = lbda
        super().__init__(theta, grid, GCVT)

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
            Sharpness paramter.
        """
        return self._lbda


class SmoothFuncCoordMap(SmoothExtCoordMap, metaclass=ABCMeta):
    r"""
    .. note:: This is an abstract class.

    Class used to define methods that may not be generally shared by any
    type of :class:`~siegpy.coordinatemappings.SmoothExtCoordMap` child
    classes, because a Smooth Exterior Coordinate Mapping could be
    defined without an explicit use of a smooth function.

    .. warning::

        This class must be used to create child classes if a smooth
        function is required.
    """

    def __init__(self, theta, smooth_func, GCVT, grid):
        r"""
        Parameters
        ----------
        theta: float
            Complex scaling angle.
        smooth_func: SmoothFunction
            Smooth function :math:`q`.
        GCVT: bool
            Stands for Generalized Complex Virial Theorem. If it is set
            to ``True``, then only one virial value is computed, else
            there are one per coordinate mapping parameter (including
            :attr:`theta` here).
        grid: numpy array or None
            Discretization grid.
        """
        self._smooth_func = smooth_func
        super().__init__(theta, smooth_func.x0, smooth_func.lbda, GCVT, grid)

    @property
    def smooth_func(self):
        r"""
        Returns
        -------
        SmoothFunction
            Smooth function :math:`q`.
        """
        return self._smooth_func

    def _update_all_values(self):
        r"""
        Update the grid and values of the smooth function before using
        the parent method updating all other values.
        """
        self.smooth_func.grid = self.grid
        super()._update_all_values()

    def _get_gp_and_gm(self):
        r"""
        .. note::

            This function only avoids code repetition in child classes.

        Returns
        -------
        tuple made of two numpy arrays
            Two numpy arrays that are often used by the child
            classes, while only being based on properties.
        """
        grid = self.grid
        lbda = self.lbda
        x0 = self.x0
        gp = lbda * (grid - x0)
        gm = lbda * (grid + x0)
        return gp, gm


class SimonCoordMap(SmoothFuncCoordMap, metaclass=ABCMeta):
    r"""
    .. note:: This is an abstract class.

    This class allows the representation of the smooth exterior
    coordinate mapping :math:`F: x \mapsto F(x)` such that its
    derivative with respect to :math:`x` is equal to
    :math:`1 + (e^{i \theta} - 1) q(x)`, where :math:`q` is a smooth
    function.
    """

    def _update_x_deriv_from_grid(self):
        r"""
        Update the values of the coordinate mapping and its three first
        derivatives with respect to :math:`x`, if the grid is not
        ``None``.
        """
        # Set some variables for readability
        gp, gm = self._get_gp_and_gm()
        m = self._get_m_values(gp, gm)  # pylint: disable=E1120
        # Set the attributes
        self._values = self._get_values(m)
        self._f_values = self._get_f_values()
        self._f_dx_values = self._get_f_dx_values()
        self._f_dx2_values = self._get_f_dx2_values()

    def _update_param_deriv_from_grid(self):
        r"""
        Update the values of the derivative with respect to the
        coordinate mapping parameters of the coordinate mapping and its
        first derivative with respect to :math:`x`.
        """
        # Set some variables for readability
        gp, gm = self._get_gp_and_gm()
        m = self._get_m_values(gp, gm)  # pylint: disable=E1120
        m_dx0 = self._get_m_dx0_values(gp, gm)  # pylint: disable=E1120
        m_dl = self._get_m_dl_values(gp, gm)  # pylint: disable=E1120
        # Set the attributes
        self._dxi_values = self._get_dxi_values(m, m_dx0, m_dl)
        self._f_dxi_values = self._get_f_dxi_values()

    def _get_values(self, m):
        r"""
        Parameters
        ----------
        m: numpy array
            Values of the intermediary function which is the factor of
            :math:`(e^{i \theta} - 1)` term in the coordinate mapping
            function.

        Returns
        -------
        numpy array
            Values of the second derivative of the coordinate
            mapping with respect to :math:`x`.
        """
        return self.grid + (np.exp(1j * self.theta) - 1) * m

    def _get_f_values(self):
        r"""
        Returns
        -------
        numpy array
            Values of the first derivative of the coordinate
            mapping with respect to :math:`x`.
        """
        return 1 + (np.exp(1j * self.theta) - 1) * self.smooth_func.values

    def _get_f_dx_values(self):
        r"""
        Returns
        -------
        numpy array
            Values of the second derivative of the coordinate
            mapping with respect to :math:`x`.
        """
        return (np.exp(1j * self.theta) - 1) * self.smooth_func.dx_values

    def _get_f_dx2_values(self):
        r"""
        Returns
        -------
        numpy array
            Values of the third derivative of the coordinate
            mapping with respect to :math:`x`.
        """
        return (np.exp(1j * self.theta) - 1) * self.smooth_func.dx2_values

    def _get_dxi_values(self, m, m_dx0, m_dl):
        r"""
        Parameters
        ----------
        m: numpy array
            Values of the intermadiary function.
        m_dx0: numpy array
            First derivative of the intermediary function with respect
            to :attr:`x0`.
        m_dl: numpy array
            First derivative of the intermediary function with respect
            to the sharpness parameter.

        Returns
        -------
        numpy array
            Values of the first derivative of the coordinate
            mapping with respect to the coordinate mapping parameters.
        """
        F_dth = 1j * np.exp(1j * self.theta) * m
        F_dx0 = (np.exp(1j * self.theta) - 1) * m_dx0
        F_dl = (np.exp(1j * self.theta) - 1) * m_dl
        return {"theta": F_dth, "x0": F_dx0, "lbda": F_dl}

    def _get_f_dxi_values(self):
        r"""
        Returns
        -------
        numpy array
            Values of the first derivative with respect to the
            coordinate mapping parameters of the first derivative with
            respect to :math:`x` of the coordinate mapping.
        """
        sf = self.smooth_func
        f_dth = 1j * np.exp(1j * self.theta) * sf.values
        f_dx0 = (np.exp(1j * self.theta) - 1) * sf.dxi_values["x0"]
        f_dl = (np.exp(1j * self.theta) - 1) * sf.dxi_values["lbda"]
        return {"theta": f_dth, "x0": f_dx0, "lbda": f_dl}

    @abstractmethod
    def _get_m_values(self, grid, lbda, gp, gm):  # pragma: no cover
        r"""
        .. note:: This is an abstract method.

        Parameters
        ----------
        grid: numpy array
            Discretization grid.
        lbda: float
            Sharpness parameter
        gp: numpy array
            Intermadiary value.
        gm: numpy array
            Intermadiary value.

        Returns
        -------
        numpy array
            The values of an intermediate function.
        """
        pass

    @abstractmethod
    def _get_m_dx0_values(self, grid, lbda, gp, gm):  # pragma: no cover
        r"""
        .. note:: This is an abstract method.

        Parameters
        ----------
        grid: numpy array
            Discretization grid.
        lbda: float
            Sharpness parameter
        gp: numpy array
            Intermadiary value.
        gm: numpy array
            Intermadiary value.

        Returns
        -------
        numpy array
            The values of the first derivative with respect to
            :attr:`x0` of an intermediate function.
        """
        pass

    @abstractmethod
    def _get_m_dl_values(self, grid, lbda, gp, gm):  # pragma: no cover
        r"""
        .. note:: This is an abstract method.

        Parameters
        ----------
        grid: numpy array
            Discretization grid.
        lbda: float
            Sharpness parameter
        gp: numpy array
            Intermadiary value.
        gm: numpy array
            Intermadiary value.

        Returns
        -------
        numpy array
            The values of the first derivative with respect to
            the sharpness parameter :attr:`\lambda` of an intermediate
            function.
        """
        pass


class TanhSimonCoordMap(SimonCoordMap):
    r"""
    This class defines the smooth exterior coordinate mapping of the
    Simon type using the smooth function based on :math:`\tanh`.
    """

    def __init__(self, theta, x0, lbda, GCVT=True, grid=None):
        r"""
        Parameters
        ----------
        theta: float
            Complex scaling angle.
        x0: float
            Inflection point.
        lbda: float
            Sharpness parameter
        GCVT: bool
            Stands for Generalized Complex Virial Theorem. If it is set
            to ``True``, then only one virial value is computed, else
            there are one per coordinate mapping parameter (here,
            :attr:`theta`, :attr:`x0` and :attr:`lbda`). Defaults to
            ``True``.
        grid: numpy array or None
            Discretization grid (optional).
        """
        smooth_func = TanhSmoothFunction(x0, lbda, grid=grid)
        super().__init__(theta, smooth_func, GCVT, grid)

    def _get_m_values(self, gp, gm):
        r"""
        Parameters
        ----------
        gp: numpy array
            Intermadiary value.
        gm: numpy array
            Intermadiary value.

        Returns
        -------
        numpy array
            The values of an intermediate function.
        """
        return self.grid + np.log(np.cosh(gp) / np.cosh(gm)) / (2 * self.lbda)

    def _get_m_dx0_values(self, gp, gm):
        r"""
        Parameters
        ----------
        gp: numpy array
            Intermadiary value.
        gm: numpy array
            Intermadiary value.

        Returns
        -------
        numpy array
            The values of the first derivative with respect to
            :attr:`x0` of an intermediate function.
        """
        return -(np.tanh(gp) + np.tanh(gm)) / 2

    def _get_m_dl_values(self, gp, gm):
        r"""
        Parameters
        ----------
        gp: numpy array
            Intermadiary value.
        gm: numpy array
            Intermadiary value.

        Returns
        -------
        numpy array
            The values of the first derivative with respect to
            the sharpness parameter :attr:`\lambda` of an intermediate
            function.
        """
        return (
            gp * np.tanh(gp) - gm * np.tanh(gm) - np.log(np.cosh(gp) / np.cosh(gm))
        ) / (2 * self.lbda ** 2)


class ErfSimonCoordMap(SimonCoordMap):
    r"""
    This class defines the smooth exterior coordinate mapping of the
    Simon type using the smooth function based on :math:`\text{erf}`.
    """

    def __init__(self, theta, x0, lbda, GCVT=True, grid=None):
        r"""
        Parameters
        ----------
        theta: float
            Complex scaling angle.
        x0: float
            Inflection point.
        lbda: float
            Sharpness parameter
        GCVT: bool
            Stands for Generalized Complex Virial Theorem. If it is set
            to ``True``, then only one virial value is computed, else
            there are one per coordinate mapping parameter (here,
            :attr:`theta`, :attr:`x0` and :attr:`lbda`). Defaults to
            ``True``.
        grid: numpy array or None
            Discretization grid (optional).
        """
        smooth_func = ErfSmoothFunction(x0, lbda, grid=grid)
        super().__init__(theta, smooth_func, GCVT, grid)

    def _get_m_values(self, gp, gm):
        r"""
        Parameters
        ----------
        gp: numpy array
            Intermadiary value.
        gm: numpy array
            Intermadiary value.

        Returns
        -------
        numpy array
            The values of an intermediate function.
        """
        term1 = gp * erf(gp) - gm * erf(gm)
        term2 = (np.exp(-gp ** 2) - np.exp(-gm ** 2)) / np.sqrt(np.pi)
        return self.grid + (term1 + term2) / (2 * self.lbda)

    def _get_m_dx0_values(self, gp, gm):
        r"""
        Parameters
        ----------
        gp: numpy array
            Intermadiary value.
        gm: numpy array
            Intermadiary value.

        Returns
        -------
        numpy array
            The values of the first derivative with respect to
            :attr:`x0` of an intermediate function.
        """
        return -(erf(gp) + erf(gm)) / 2

    def _get_m_dl_values(self, gp, gm):
        r"""
        Parameters
        ----------
        gp: numpy array
            Intermadiary value.
        gm: numpy array
            Intermadiary value.

        Returns
        -------
        numpy array
            The values of the first derivative with respect to
            the sharpness parameter :attr:`\lambda` of an intermediate
            function.
        """
        return (np.exp(-gm ** 2) - np.exp(-gp ** 2)) / (2 * np.sqrt(np.pi) * self.lbda)


class KGCoordMap(SmoothFuncCoordMap):
    r"""
    .. note:: This is an abstract class.

    This class allows the representation of the smooth exterior
    coordinate mapping :math:`F: x \mapsto x e^{i \theta q(x)}`, that we
    will call the Kalita-Gupta (KG) coordinate mapping.
    :math:`q` represents the smooth function.
    """

    def _update_x_deriv_from_grid(self):
        r"""
        Update the values of the coordinate mapping and its three first
        derivatives with respect to :math:`x`, if the grid is not
        ``None``.
        """
        # Get the values "recursively"
        F = self._get_values()
        f = self._get_f_values(F)
        f_dx = self._get_f_dx_values(F, f)
        f_dx2 = self._get_f_dx2_values(F, f, f_dx)
        # Set the properties
        self._values = F
        self._f_values = f
        self._f_dx_values = f_dx
        self._f_dx2_values = f_dx2

    def _update_param_deriv_from_grid(self):
        r"""
        Update the values of the derivative with respect to the
        coordinate mapping parameters of the coordinate mapping and its
        first derivative with respect to :math:`x`.
        """
        # Set some variables for readability
        a = self.smooth_func.values
        a_dx0 = self.smooth_func.dxi_values["x0"]
        a_dl = self.smooth_func.dxi_values["lbda"]
        F = self.values
        # Set the properties
        F_dth = 1j * a * F
        F_dx0 = 1j * self.theta * F * a_dx0
        F_dl = 1j * self.theta * F * a_dl
        self._dxi_values = {"theta": F_dth, "x0": F_dx0, "lbda": F_dl}
        self._f_dxi_values = self._get_f_dxi_values(F, F_dth, F_dx0, F_dl)

    def _get_values(self):
        r"""
        Returns
        -------
        numpy array
            Values of the second derivative of the coordinate
            mapping with respect to :math:`x`.
        """
        a = self.smooth_func.values
        return self.grid * np.exp(1j * self.theta * a)

    def _get_f_values(self, F):
        r"""
        Parameters
        ----------
        F: numpy array
            Values of the coordinate mapping.

        Returns
        -------
        numpy array
            Values of the first derivative of the coordinate
            mapping with respect to :math:`x`.
        """
        a = self.smooth_func.values
        a_dx = self.smooth_func.dx_values
        return np.exp(1j * self.theta * a) + 1j * self.theta * a_dx * F

    def _get_f_dx_values(self, F, f):
        r"""
        Parameters
        ----------
        F: numpy array
            Values of the coordinate mapping.
        f: numpy array
            Values of the first derivative of the coordinate mapping
            with respect to :math:`x`.

        Returns
        -------
        numpy array
            Values of the second derivative of the coordinate
            mapping with respect to :math:`x`.
        """
        a = self.smooth_func.values
        a_dx = self.smooth_func.dx_values
        a_dx2 = self.smooth_func.dx2_values
        # Evaluate the derivative
        factor = np.exp(1j * self.theta * a) + f
        return 1j * self.theta * (a_dx * factor + a_dx2 * F)

    def _get_f_dx2_values(self, F, f, f_dx):
        r"""
        Parameters
        ----------
        F: numpy array
            Values of the coordinate mapping.
        f: numpy array
            Values of the first derivative of the coordinate mapping
            with respect to :math:`x`.
        f_dx: numpy array
            Values of the second derivative of the coordinate mapping
            with respect to :math:`x`.

        Returns
        -------
        numpy array
            Values of the third derivative of the coordinate mapping
            with respect to :math:`x`.
        """
        a = self.smooth_func.values
        a_dx = self.smooth_func.dx_values
        a_dx2 = self.smooth_func.dx2_values
        a_dx3 = self.smooth_func.dx3_values
        # Evaluate the derivative
        factor = a_dx2 + 1j * self.theta * a_dx ** 2
        return (
            1j
            * self.theta
            * (
                a_dx * f_dx
                + 2 * a_dx2 * f
                + a_dx3 * F
                + factor * np.exp(1j * self.theta * a)
            )
        )

    def _get_dxi_values(self, *args):  # pragma: no cover
        r"""
        Does nothing, as its role is already performed by the
        :meth:`_update_param_deriv_from_grid` method
        """
        pass

    def _get_f_dxi_values(self, F, F_dth, F_dx0, F_dl):
        r"""
        Parameters
        ----------
        F: numpy array
            Values of the coordinate mapping.
        F_dth: numpy array
            Values of the first derivative of the coordinate mapping
            with respect to the complex scaling angle.
        F_dx0: numpy array
            Values of the second derivative of the coordinate mapping
            with respect to :attr:`x0`.
        F_dl: numpy array
            Values of the first derivative of the coordinate mapping
            with respect to the sharpness parameter.

        Returns
        -------
        numpy array
            Values of the first derivative with respect to the
            coordinate mapping parameters of the first derivative with
            respect to :math:`x` of the coordinate mapping.
        """
        theta = self.theta
        sf = self.smooth_func
        a = sf.values
        a_dx = sf.dx_values
        a_dx0 = sf.dxi_values["x0"]
        a_dl = sf.dxi_values["lbda"]
        a_dx_dx0 = sf.dx_dxi_values["x0"]
        a_dx_dl = sf.dx_dxi_values["lbda"]
        # Evaluate the derivatives
        f_dth = 1j * (a * np.exp(1j * theta * a) + a_dx * (F + theta * F_dth))
        f_dx0 = (
            1j * theta * (a_dx0 * np.exp(1j * theta * a) + a_dx_dx0 * F + a_dx * F_dx0)
        )
        f_dl = 1j * theta * (a_dl * np.exp(1j * theta * a) + a_dx_dl * F + a_dx * F_dl)
        return {"theta": f_dth, "x0": f_dx0, "lbda": f_dl}


class TanhKGCoordMap(KGCoordMap):
    r"""
    This class defines the smooth exterior coordinate mapping of the
    Kalita-Gupta type using the smooth function based on :math:`\tanh`.
    """

    def __init__(self, theta, x0, lbda, GCVT=True, grid=None):
        r"""
        Parameters
        ----------
        theta: float
            Complex scaling angle.
        x0: float
            Inflection point.
        lbda: float
            Sharpness parameter
        GCVT: bool
            Stands for Generalized Complex Virial Theorem. If it is set
            to ``True``, then only one virial value is computed, else
            there are one per coordinate mapping parameter (here,
            :attr:`theta`, :attr:`x0` and :attr:`lbda`). Defaults to
            ``True``.
        grid: numpy array or None
            Discretization grid (optional).
        """
        smooth_func = TanhSmoothFunction(x0, lbda, grid=grid)
        super().__init__(theta, smooth_func, GCVT=GCVT, grid=grid)


class ErfKGCoordMap(KGCoordMap):
    r"""
    This class defines the smooth exterior coordinate mapping of the
    Kalita-Gupta type using the smooth function based on
    :math:`\text{erf}`.
    """

    def __init__(self, theta, x0, lbda, GCVT=True, grid=None):
        r"""
        Parameters
        ----------
        theta: float
            Complex scaling angle.
        x0: float
            Inflection point.
        lbda: float
            Sharpness parameter
        GCVT: bool
            Stands for Generalized Complex Virial Theorem. If it is set
            to ``True``, then only one virial value is computed, else
            there are one per coordinate mapping parameter (here,
            :attr:`theta`, :attr:`x0` and :attr:`lbda`). Defaults to
            ``True``.
        grid: numpy array
            Discretization grid (optional).
        """
        smooth_func = ErfSmoothFunction(x0, lbda, grid=grid)
        super().__init__(theta, smooth_func, GCVT=GCVT, grid=grid)
