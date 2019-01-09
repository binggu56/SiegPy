# -*- coding: utf-8 -*-
# I do not want pylint to tell me I do not call the __init__ of base
# class for the Symbolicpotential class; that is on purpose.
# pylint: disable=W0231
r"""
The :class:`SymbolicPotential` class is defined below, along with some
child classes:

* the :class:`WoodsSaxonPotential` class,
* the :class:`TwoGaussianPotential` class,
* the :class:`FourGaussianPotential` class.
"""

from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.special
import sympy
from sympy.abc import x
from .potential import Potential
from .functions import Gaussian


class SymbolicPotential(Potential):
    r"""
    A Symbolic potential is defined by a symbolic function using the
    sympy package. The symbolic function can be encoded in a string.
    The function must be a function of ``x`` only (no other parameters).
    """

    def __init__(self, sym_func, grid=None):
        r"""
        Parameters
        ----------
        sym_func: str or sympy symbolic function
           Symbolic function of the potential.
        grid: list or numpy array or NoneType
            Discretization grid of the potential (optional).

        Raises
        ------
        ValueError
            If a parameter of the symbolic function is not :math:`x`.


        Examples

        The symbolic function can be a string:

        >>> f = "1 / (x**2 + 1)"
        >>> pot = SymbolicPotential(f)

        Updating the grid automatically updates the values of the
        potential:

        >>> xgrid = [-1, 0, 1]
        >>> pot.grid = xgrid
        >>> pot.values
        array([ 0.5,  1. ,  0.5])

        The symbolic function must be a function of x only:

        >>> from sympy.abc import a, b, x
        >>> f = a / (x**2 + b)
        >>> SymbolicPotential(f)
        Traceback (most recent call last):
        ValueError: The only variable of the analytic function should be x.


        This means you must assign values to the parameters beforehand:

        >>> pot = SymbolicPotential(f.subs([(a, 1), (b, 1)]), grid=xgrid)
        >>> pot.values
        array([ 0.5,  1. ,  0.5])
        """
        # Make sure to convert the symbolic function in a sympy format
        sym_func = sympy.sympify(sym_func)
        # Check that x is the only variable
        if any([s != x for s in sym_func.atoms(sympy.Symbol)]):
            raise ValueError("The only variable of the analytic function should be x.")
        # Set the symbolic function
        self._symbolic = sym_func
        # Set the grid and the values according to the grid (both are
        # updated by the grid setter)
        self.grid = grid

    @property
    def symbolic(self):
        r"""
        Returns
        -------
        Sympy symbolic function
            Symbolic function of the potential.
        """
        return self._symbolic

    @property
    def grid(self):
        r"""
        Returns
        -------
        NoneType or numpy array
            Values of the grid.
        """
        return self._grid

    @grid.setter
    def grid(self, grid):
        r"""
        Setter of the grid, updating the values of the potential at the
        same time.

        Parameters
        ----------
        grid: NoneType or numpy array
            Discretization grid of the potential.
        """
        if grid is not None:
            self._grid = np.array(grid)
            self._values = self._compute_values(self.grid)
        else:
            self._grid = None
            self._values = None

    def _compute_values(self, grid):
        r"""
        Evaluate the values of the potential for a given grid.

        Parameters
        ----------
        grid: list or numpy array
            Discretization grid or complex scaled grid.

        Returns
        -------
        numpy array
            Values of the potential according to its grid.
        """
        grid = np.array(grid)
        mods = ["numpy", {"erf": scipy.special.erf}]
        return sympy.lambdify(x, self.symbolic, modules=mods)(grid)

    def complex_scaled_values(self, coord_map):
        r"""
        Evaluate the complex scaled potential for a given coordinate
        mapping.

        Parameters
        ----------
        coord_map: CoordMap
            Coordinate mapping.

        Returns
        -------
        numpy array
            Complex scaled values of the potential.
        """
        # Update the grid and the values of the coordinate mapping
        coord_map.grid = self.grid
        return self._compute_values(coord_map.values)

    def __add__(self, other):
        r"""
        Parameters
        ----------
        other: Potential
            Another potential.

        Returns
        -------
        SymbolicPotential or Potential
            Sum of both potentials.

        Raises
        ------
        ValueError
            If the other potential is not symbolic and both potentials
            have no grid.
        """
        if isinstance(other, SymbolicPotential):
            sym_func = sympy.simplify(self.symbolic + other.symbolic)
            return SymbolicPotential(sym_func, grid=self.grid)
        elif self.grid is not None and other.grid is not None:
            return super().__add__(other)
        else:
            raise ValueError(
                "Cannot add potentials that are not discretized over a grid."
            )


class WoodsSaxonPotential(SymbolicPotential):
    r"""
    This class defines a symmetric and smooth Woods-Saxon potential of
    the form:

    .. math::

        V(x) = V_0 \left( \frac{1}{1 + e^{\lambda(x+l/2)}}
        - \frac{1}{1 + e^{\lambda(x-l/2)}} \right)

    where :math:`V_0` is the potential depth, :math:`l` is the potential
    characteristic width and :math:`\lambda` is the sharpness
    parameter.
    """

    def __init__(self, l, V0, lbda, grid=None):
        r"""
        Parameters
        ----------
        l: float
            Characteristic width of the potential.
        V0: float
            Potential depth (if negative, it corresponds to a potential
            barrier).
        lbda: float
            Sharpness of the potential.
        grid: numpy array
            Discretization grid of the potential (optional).

        Raises
        ------
        ValueError
            If the width or the sharpness parameters is strictly
            negative.
        """
        # Check the parameters
        if l <= 0:
            raise ValueError("The width of the potential must be positive.")
        if lbda <= 0:
            raise ValueError("The sharpness of the potential must be positive.")
        # Initialize the attributes
        self._width = l
        self._sharpness = lbda
        self._depth = V0
        # Initialize the symbolic function
        func = "+ {} / (1 + exp({}*(x - {}/2)))"
        sym_func = func.format(V0, lbda, -l) + func.format(-V0, lbda, l)
        # Initialize the potential as a SymbolicPotential
        super().__init__(sym_func, grid=grid)

    @property
    def width(self):
        r"""
        Returns
        -------
        float
            Width of the potential.
        """
        return self._width

    @property
    def depth(self):
        r"""
        Returns
        -------
        float
            Depth of the potential.
        """
        return self._depth

    @property
    def sharpness(self):
        r"""
        Returns
        -------
        float
            Sharpness of the potential.
        """
        return self._sharpness


class MultipleGaussianPotential(SymbolicPotential, metaclass=ABCMeta):
    r"""
    This class avoids some code repetition inside the classes
    :class:`TwoGaussianPotential` and :class:`FourGaussianPotential`.
    """

    @classmethod
    @abstractmethod
    def from_Gaussians(cls, *args, **kwargs):  # pragma: no cover
        r"""
        .. note:: This is an asbtract class method.

        Initalization of the potential from Gaussian functions and not
        from the parameters allowing to define these Gaussian functions.

        Returns
        -------
        MultipleGaussianPotential
            Potential initialized from multiple Gaussian functions.
        """
        pass

    @property
    @abstractmethod
    def gaussians(self):  # pragma: no cover
        r"""
        .. note:: This is an asbtract property.

        Returns
        -------
        list
            All the Gaussian functions used to create the potential.
        """
        pass

    @property
    def sigmas(self):
        r"""
        Returns
        -------
        list
            Sigma of the Gaussian functions of the potential.
        """
        return [g.sigma for g in self.gaussians]

    @property
    def centers(self):
        r"""
        Returns
        -------
        list
            Center of the Gaussian functions of the potential.
        """
        return [g.center for g in self.gaussians]

    @property
    def amplitudes(self):
        r"""
        Returns
        -------
        list
            Amplitude of the Gaussian functions of the potential.
        """
        return [g.amplitude for g in self.gaussians]


class TwoGaussianPotential(MultipleGaussianPotential):
    r"""
    This class defines a potential made of the sum of two Gaussian
    functions.
    """

    def __init__(self, sigma1, xc1, h1, sigma2, xc2, h2, grid=None):
        r"""
        Parameters
        ----------
        sigma1: float
            Sigma of the first Gaussian function.
        xc1: float
            Center of the first Gaussian function.
        h1: float
            Amplitude of the first Gaussian function.
        sigma2: float
            Sigma of the second Gaussian function.
        xc2: float
            Center of the second Gaussian function.
        h2: float
            Amplitude of the second Gaussian function.
        grid: numpy array
            Discretization grid of the potential (optional).
        """
        # Intialize the two Gaussian functions
        self._gaussian1 = Gaussian(sigma1, xc1, h=h1, grid=grid)
        self._gaussian2 = Gaussian(sigma2, xc2, h=h2, grid=grid)
        # Initialize the symbolic function of the potential
        func = "+ {} * exp(- (x - {})**2 / (2*{}**2))"
        g1 = func.format(h1, xc1, sigma1)
        g2 = func.format(h2, xc2, sigma2)
        sym_func = g1 + g2
        # Inittialize the symbolic functions
        super().__init__(sym_func, grid=grid)

    @classmethod
    def from_Gaussians(cls, gauss1, gauss2, grid=None):
        r"""
        Initialization of a TwoGaussianPotential instance from two
        Gaussian functions.

        Parameters
        ----------
        gauss1: Gaussian
            First Gaussian function.
        gauss2: Gaussian
            Second Gaussian function.
        grid: numpy array
            Discretization grid of the potential (optional).

        Returns
        -------
        TwoGaussianPotential
            Potential initialized from two Gaussian functions.
        """
        sigma1, xc1, h1 = gauss1._params
        sigma2, xc2, h2 = gauss2._params
        return cls(sigma1, xc1, h1, sigma2, xc2, h2, grid=grid)

    @property
    def gaussian1(self):
        r"""
        Returns
        -------
        Gaussian
            First Gaussian function of the potential.
        """
        return self._gaussian1

    @property
    def gaussian2(self):
        r"""
        Returns
        -------
        Gaussian
            Second Gaussian function of the potential.
        """
        return self._gaussian2

    @property
    def gaussians(self):
        r"""
        Returns
        -------
        list
            Both Gaussian functions of the potential.
        """
        return [self.gaussian1, self.gaussian2]


class FourGaussianPotential(MultipleGaussianPotential):
    r"""
    This class defines a potential made of the sum of four Gaussian
    functions.
    """

    def __init__(
        self,
        sigma1,
        xc1,
        h1,
        sigma2,
        xc2,
        h2,
        sigma3,
        xc3,
        h3,
        sigma4,
        xc4,
        h4,
        grid=None,
    ):
        r"""
        Parameters
        ----------
        sigma1: float
            Sigma of the first Gaussian function.
        xc1: float
            Center of the first Gaussian function.
        h1: float
            Amplitude of the first Gaussian function.
        sigma2: float
            Sigma of the second Gaussian function.
        xc2: float
            Center of the second Gaussian function.
        h2: float
            Amplitude of the second Gaussian function.
        sigma3: float
            Sigma of the third Gaussian function.
        xc3: float
            Center of the third Gaussian function.
        h3: float
            Amplitude of the third Gaussian function.
        sigma4: float
            Sigma of the fouth Gaussian function.
        xc4: float
            Center of the fouth Gaussian function.
        h4: float
            Amplitude of the fouth Gaussian function.
        grid: numpy array
            Discretization grid of the potential (optional).
        """
        # Initialize the four Gaussian functions
        self._gaussian1 = Gaussian(sigma1, xc1, h=h1, grid=grid)
        self._gaussian2 = Gaussian(sigma2, xc2, h=h2, grid=grid)
        self._gaussian3 = Gaussian(sigma3, xc3, h=h3, grid=grid)
        self._gaussian4 = Gaussian(sigma4, xc4, h=h4, grid=grid)
        # Initialize the symbolic function
        tg1 = TwoGaussianPotential(sigma1, xc1, h1, sigma2, xc2, h2).symbolic
        tg2 = TwoGaussianPotential(sigma3, xc3, h3, sigma4, xc4, h4).symbolic
        sym_func = tg1 + tg2
        # Initialize the symbolic potential
        super().__init__(sym_func, grid=grid)

    @classmethod
    def from_Gaussians(cls, gauss1, gauss2, gauss3, gauss4, grid=None):
        r"""
        Initialization of a FourGaussianPotential instance from four
        Gaussian functions.

        Parameters
        ----------
        gauss1: Gaussian
            First Gaussian function.
        gauss2: Gaussian
            Second Gaussian function.
        gauss3: Gaussian
            Third Gaussian function.
        gauss4: Gaussian
            Fourth Gaussian function.
        grid: numpy array
            Discretization grid of the potential (optional).

        Returns
        -------
        FourGaussianPotential
            Potential initialized from four Gaussian functions.
        """
        sigma1, xc1, h1 = gauss1._params
        sigma2, xc2, h2 = gauss2._params
        sigma3, xc3, h3 = gauss3._params
        sigma4, xc4, h4 = gauss4._params
        return cls(
            sigma1,
            xc1,
            h1,
            sigma2,
            xc2,
            h2,
            sigma3,
            xc3,
            h3,
            sigma4,
            xc4,
            h4,
            grid=grid,
        )

    @property
    def gaussian1(self):
        r"""
        Returns
        -------
        Gaussian
            First Gaussian function of the potential.
        """
        return self._gaussian1

    @property
    def gaussian2(self):
        r"""
        Returns
        -------
        Gaussian
            Second Gaussian function of the potential.
        """
        return self._gaussian2

    @property
    def gaussian3(self):
        r"""
        Returns
        -------
        Gaussian
            Third Gaussian function of the potential.
        """
        return self._gaussian3

    @property
    def gaussian4(self):
        r"""
        Returns
        -------
        Gaussian
            Fouth Gaussian function of the potential.
        """
        return self._gaussian4

    @property
    def gaussians(self):
        r"""
        Returns
        -------
        list
            Four Gaussian functions of the potential.
        """
        return [self.gaussian1, self.gaussian2, self.gaussian3, self.gaussian4]
