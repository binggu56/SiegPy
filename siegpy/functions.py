# -*- coding: utf-8 -*-
r"""
The class :class:`Function` is the base class for various other
classes of interest, which are:

* the :class:`~siegpy.functions.AnalyticFunction` class, to represent
  analytic functions,
* the :class:`~siegpy.functions.potential.Potential` class, to
  represent a potential.
* the :class:`~siegpy.functions.eigenstate.Eigenstate` class, to
  represent eigenstates of a potential.

Two classes used to represent two particular types of analytic
functions are also defined here:

1. :class:`~siegpy.functions.Gaussian`, to represent a Gaussian
   function,
2. :class:`~siegpy.functions.Rectangular`, to represent a rectangular
   function.
"""


from copy import deepcopy
from abc import ABCMeta, abstractmethod
import numpy as np
from siegpy.utils import init_plot, finalize_plot


class Function():
    r"""
    Class defining a 1-dimendional (1D) function from its grid and
    the corresponding values.

    A function can be plotted, and different types of operations can
    be performed (*e.g.*, scalar product with another function,
    addition of another function). It is also possible to compute its
    norm, and return its conjugate and its absolute value as another
    function.
    """

    def __init__(self, grid, values):
        r"""
        Parameters
        ----------
        grid: list or set or numpy array
            Discretization grid.
        values: list or set or numpy array
            Values of the function evaluated on the grid points.

        Raises
        ------
        ValueError
            If the grid is not made of reals or if the ``grid`` and
            ``values`` arrays have incoherent lengths.

        Example

        Note that both ``grid`` and ``values`` are converted to numpy
        arrays:

        >>> f = Function([-1, 0, 1], [1, 2, 3])
        >>> f.grid
        array([-1,  0,  1])
        >>> f.values
        array([1, 2, 3])
        """
        # Check that the grid and values have consistent lengths
        if len(grid) != len(values):
            msg = "Both grid and values must have the same length ({} != {})"\
                  .format(len(grid), len(values))
            raise ValueError(msg)
        # Check that the grid is not made of complex numbers
        if np.any(np.iscomplex(grid)):
            raise ValueError("The grid has to be made of reals.")
        self._values = np.array(values)
        self._grid = np.array(grid)

    @property
    def grid(self):
        r"""
        Returns
        -------
        numpy array
            Grid of a Function instance.

        .. warning::

            The grid of a Function instance cannot be modified:

            >>> f = Function([-1, 0, 1], [1, 2, 3])
            >>> f.grid = [-1, 1]
            Traceback (most recent call last):
            AttributeError: can't set attribute
        """
        return self._grid

    @property
    def values(self):
        r"""
        Returns
        -------
        numpy array
            Values of a Function instance.

        .. warning::

            The values of a Function instance cannot be modified:

            >>> f = Function([-1, 0, 1], [1, 2, 3])
            >>> f.values = [3, 0, 1]
            Traceback (most recent call last):
            AttributeError: can't set attribute
        """
        return self._values

    def __eq__(self, other):
        r"""
        Two functions are equal if they have the same grid and values.

        Parameters
        ----------
        other: object
            Another object.

        Returns
        -------
        bool
            ``True`` if ``other`` is a Function instance with the same
            :attr:`grid` and :attr:`values`.

        Examples

        >>> f = Function([1, 2, 3], [1, 1, 1])
        >>> Function([1, 2, 3], [1, 1, 1]) == f
        True
        >>> Function([1, 2, 3], [-1, 0, 1]) == f
        False
        >>> Function([-1, 0, 1], [1, 1, 1]) == f
        False
        >>> f == 1
        False
        """
        return isinstance(other, Function) and (
            np.array_equal(self.grid, other.grid) and
            np.array_equal(self.values, other.values))

    def __add__(self, other):
        r"""
        Add two functions, if both grids are the same.

        Parameters
        ----------
        other: Function
            Another function.

        Returns
        -------
        Function
            Sum of both functions.

        Raises
        ------
        ValueError
            If both the :attr:`grid` of both functions differ.

        Examples

        Two functions can be added:

        >>> f1 = Function([1, 2, 3], [1, 1, 1])
        >>> f2 = Function([1, 2, 3], [0, -1, 0])
        >>> f = f1 + f2

        The grid of the new function is the same, and its values are
        the sum of both values:

        >>> f.grid
        array([1, 2, 3])
        >>> f.values
        array([1, 0, 1])

        It leaves the other functions unchanged:

        >>> f1.values
        array([1, 1, 1])
        >>> f2.values
        array([ 0, -1,  0])
        """
        if np.array_equal(self.grid, other.grid):
            return Function(self.grid, self.values + other.values)
        else:
            raise ValueError(
                "Both Functions must be discretized on the same grid.")

    @property
    def is_even(self):
        r"""
        Returns
        -------
        bool
            ``True`` if the function is even, ``False`` if not.

        Examples

        >>> Function([1, 2, 3], [1j, 1j, 1j]).is_even
        False
        >>> Function([-1, 0, 1], [1j, 1j, 1j]).is_even
        True
        """
        npts = len(self._values)
        half = int(npts/2)
        return (np.array_equal(self.grid[:half], -self.grid[npts-half:][::-1])
                and np.array_equal(self._values[:half],
                                   self._values[npts-half:][::-1]))

    @property
    def is_odd(self):
        r"""
        Returns
        -------
        bool
            ``True`` if the function is odd, ``False`` if not.

        Examples

        >>> Function([-1, 0, 1], [-1j, 0j, 1j]).is_odd
        True
        >>> Function([1, 2, 3], [-1j, 0j, 1j]).is_odd
        False
        """
        npts = len(self._values)
        half = int(npts/2)
        return (np.array_equal(self.grid[:half], -self.grid[npts-half:][::-1])
                and np.array_equal(self.values[:half],
                                   -self.values[npts-half:][::-1]))

    def plot(self, xlim=None, ylim=None, title=None, file_save=None):  # pragma: no cover  # noqa
        r"""
        Plot the real and imaginary parts of the function.

        Parameters
        ----------
        xlim: tuple(float or int, float or int)
            Range of the x axis of the plot (optional).
        ylim: tuple(float or int, float or int)
            Range of the y axis of the plot (optional).
        title: str
            Title for the plot.
        file_save: str
            Filename of the plot to be saved (optional).
        """
        # Object-oriented plots
        fig, ax = init_plot()
        # Define the plot
        ax.plot(self.grid, np.real(self.values), color='blue', label='Re')
        ax.plot(self.grid, np.imag(self.values), color='red', label='Im')
        # Finalize the plot
        ax.legend()
        finalize_plot(fig, ax, xlim=xlim, ylim=ylim, title=title,
                      file_save=file_save, xlabel="$x$")

    def abs(self):
        r"""
        Returns
        -------
        Function
            Absolute value of the function.

        Example

        Applying the :meth:`abs` method to a :class:`Function` instance
        returns a new :class:`Function` instance (*i.e.*, the initial
        one is unchanged):

        >>> f = Function([1, 2, 3], [1j, 1j, 1j])
        >>> f.abs().values
        array([ 1.,  1.,  1.])
        >>> f.values
        array([ 0.+1.j,  0.+1.j,  0.+1.j])
        """
        return Function(self.grid, np.abs(self._values))

    def conjugate(self):
        r"""
        Returns
        -------
        Function
            Conjugate of the function.

        Example

        Applying the :meth:`conjugate` method to a :class:`Function`
        instance returns a new :class:`Function` instance (*i.e.* the
        initial one is unchanged):

        >>> f = Function([1, 2, 3], [1j, 1j, 1j])
        >>> f.conjugate().values
        array([ 0.-1.j,  0.-1.j,  0.-1.j])
        >>> f.values
        array([ 0.+1.j,  0.+1.j,  0.+1.j])
        """
        return Function(self.grid, np.conjugate(self._values))

    def scal_prod(self, other, xlim=None):
        r"""
        Evaluate the usual scalar product of two functions f and g:

        :math:`\langle f | g \rangle = \int f^*(x)  g(x) \text{d}x`

        where `*` represents the conjugation.

        .. note::

            * The trapezoidal integration rule is used.

        Parameters
        ----------
        other: Function
            Another function.
        xlim: tuple(float or int, float or int)
            Range of the x-axis for the integration (optional).

        Returns
        -------
        float
            Value of the scalar product

        Raises
        ------
        ValueError
            If the grid of both functions are different or if the
            interval given by ``xlim`` is not inside the one defined by
            the discretization grid.

        Example

        >>> grid = [-1, 0, 1]
        >>> f = Function(grid, [1, 0, 1])
        >>> g = Function(grid, np.ones_like(grid))
        >>> f.scal_prod(g)
        1.0
        """
        # Check that both have the same grid
        if not np.array_equal(self.grid, other.grid):
            raise ValueError("Both grids are different.")
        # Reduce the interval where the scalar product is computed,
        # if desired
        if xlim is not None:
            # Check the limits are well-defined
            xl, xr = xlim
            if xl < min(self.grid) or xr > max(self.grid):
                raise ValueError(
                    "The interval defined by xlim must be inside the grid.")
            # Define the contracted grid and functions
            where = np.logical_and(xl <= self.grid, xr >= self.grid)
            grid = self.grid[where]
            f1 = self.values[where]
            f2 = other.values[where]
        else:
            grid = self.grid
            f1 = self.values
            f2 = other.values
        # Use the trapezoidal integration rule for the integration.
        return np.trapz(np.conjugate(f1)*f2, x=grid)

    def norm(self):
        r"""
        Returns
        -------
        float
            Norm of the function.

        Example

        >>> Function([-1, 0, 1], [2j, 0, 2j]).norm()
        (4+0j)
        """
        # This is nothing but the scalar product of the function with
        # itself.
        return self.scal_prod(self)


class AnalyticFunction(Function, metaclass=ABCMeta):
    r"""
    .. note::

        **This is an abstract class.** A child class must implement the
        :class:`~siegpy.functions.AnalyticFunction._compute_values`
        class. Examples of such child classes are:

        * the :class:`~siegpy.functions.Rectangular` class,
        * the :class:`~siegpy.functions.Gaussian` class.

    The main change with respect to the
    :class:`siegpy.functions.Function` class is that the grid is an
    optional parameter, so that, if it is modified, then the values of
    the function are updated accordingly.
    """

    def __init__(self, grid=None):
        r"""
        Parameters
        ----------
        grid: list or set or numpy array
            Discretization grid (optional).
        """
        if grid is not None:
            values = self.evaluate(grid)
            Function.__init__(self, grid, values)
        else:
            self._grid = None
            self._values = None

    def evaluate(self, grid):
        r"""
        Wrapper for the
        :func:`~siegpy.functions.AnalyticFunction._compute_values` to
        evaluate the analytic function either for a grid of points or a
        single point in the 1-dimensional space.

        Parameters
        ----------
        grid: int or float or numpy array
            Discretization grid.

        Returns
        -------
        float or complex or numpy array
            Values of the function for all the grid points.
        """
        try:
            # Convert the grid to a numpy array before computing the values
            len(grid)
            grid = np.array(grid)
            return self._compute_values(grid)
        except TypeError:
            # The grid is made of one element: return a single element
            grid = np.array([grid])
            return self._compute_values(grid)[0]

    @abstractmethod
    def _compute_values(self, grid):  # pragma: no cover
        r"""
        .. note:: This is an abstract method.

        Compute the values of the function given a discretization grid
        that has been converted to a numpy array by the
        :meth:`~siegpy.functions.AnalyticFunction.evaluate` method.

        Parameters
        ----------
        grid: numpy array
            Discretization grid.

        Returns
        -------
        numpy array
            Values of the analytic function over the provided ``grid``.
        """
        pass

    @property
    def grid(self):
        r"""
        :attr:`grid` still is an attribute, but it is more powerful than
        for a :class:`~siegpy.functions.Function` instance: when the
        grid of an :class:`AnalyticFunction` instance is updated, so are
        its values.
        """
        return super().grid

    @grid.setter
    def grid(self, new_grid):
        r"""
        Setter of the :attr:`grid` attribute, updating the values of
        the function at the same time.

        Parameters
        ----------
        new_grid: list or set or numpy array
            New discretization grid.
        """
        if new_grid is not None:
            new_values = self.evaluate(new_grid)
            Function.__init__(self, new_grid, new_values)
        else:
            AnalyticFunction.__init__(self)

    def __add__(self, other):
        r"""
        Parameters
        ----------
        other: Function, or any class inheriting from it
            Another Function.

        Returns
        -------
        Function
            Sum of an :class:`AnalyticFunction` instance with
            another function. It requires that at least one of both
            functions has a grid that is not ``None``.

        Raises
        ------
        ValueError
            If both functions have grids set to ``None``.
        """
        if self.grid is None:
            if other.grid is None:
                raise ValueError(
                    "Two analytic functions not discretized cannot be added.")
            else:
                # Perform the addition by adding a grid to a copy of
                # self (in order not to modify it).
                new = deepcopy(self)
                new.grid = other.grid
                return new + other
        elif other.grid is None:
            # Perform the addition by adding a grid to a copy of other
            # (in order not to modify it).
            new = deepcopy(other)
            new.grid = self.grid
            return self + new
        else:
            # Both functions have a discretization grid: perform the
            # usual addition of two functions
            return super().__add__(other)


class Gaussian(AnalyticFunction):
    r"""
    A Gaussian function is characterized by:

    * a width ``sigma``
    * a center ``xc``
    * an initial momentum ``k0``
    * an amplitude ``h``

    In addition to the behaviour of an analytic function:

    * the norm is computed analytically,
    * the attribute :attr:`~siegpy.functions.Gaussian.is_even` returns
      ``True`` if the Gaussian function is centered, even if the
      discretization grid is not centered,
    * the equality and addition of two Gaussian functions are also
      defined.
    """

    def __init__(self, sigma, xc, k0=0., h=1., grid=None):
        r"""
        Parameters
        ----------
        sigma: strictly positive float
            Width of the Gaussian function.
        xc: float
            Center of the Gaussian function.
        k0: float
            Initial momentum of the Gaussian function (default to 0).
        h: float
            Maximal amplitude of the Gaussian function (default to 1).
        grid: list or set or numpy array
            Discretization grid (optional).

        Raises
        ------
        ValueError
            If ``sigma`` is negative.

        Examples

        A Gaussian function is characterized by various attributes:

        >>> g = Gaussian(4, 2, k0=5)
        >>> g.sigma
        4
        >>> g.center
        2
        >>> g.momentum
        5
        >>> g.amplitude
        1.0

        If no discretization grid is passed, then the atrributes
        :attr:`~siegpy.functions.AnalyticFunction.grid` and
        :attr:`~siegpy.functions.AnalyticFunction.values` are set to
        ``None``:

        >>> g.grid is None and g.values is None
        True

        If a grid is given, the Gaussian is discretized:

        >>> g1 = Gaussian(4, 2, k0=5, grid=[-1, 0, 1])
        >>> g2 = Gaussian(4, 2, k0=5, h=2, grid=[-1, 0, 1])
        >>> np.array_equal(g2.values, 2*g1.values)
        True

        .. note::

            The only way to modify the values of a Gaussian is by
            setting its grid:

            >>> g.grid = [-1, 0, 1]
            >>> assert np.array_equal(g.grid, g1.grid)
            >>> assert np.array_equal(g.values, g1.values)
            >>> g.values = [2, 1, 2]
            Traceback (most recent call last):
            AttributeError: can't set attribute

        .. warning::

            Finally, a Gaussian function must have a strictly positive
            sigma:

            >>> Gaussian(-1, 1)
            Traceback (most recent call last):
            ValueError: The Gaussian must have a strictly positive sigma.
            >>> Gaussian(0, 1)
            Traceback (most recent call last):
            ValueError: The Gaussian must have a strictly positive sigma.
        """
        # Check the initial values
        if sigma <= 0.0:
            raise ValueError(
                "The Gaussian must have a strictly positive sigma.")
        # Set the attirbutes
        self._sigma = sigma
        self._center = xc
        self._momentum = k0
        self._amplitude = h
        super().__init__(grid)

    @property
    def sigma(self):
        r"""
        Returns
        -------
        float
            Width of the Gaussian function.
        """
        return self._sigma

    @property
    def center(self):
        r"""
        Returns
        -------
        float
            Center of the Gaussian function.
        """
        return self._center

    @property
    def momentum(self):
        r"""
        Returns
        -------
        float
            Momentum of the Gaussian function.
        """
        return self._momentum

    @property
    def amplitude(self):
        r"""
        Returns
        -------
        float
            Amplitude of the Gaussian function.
        """
        return self._amplitude

    @property
    def _params(self):
        r"""
        Shortcut to get the main parameters of a Gaussian function.

        Returns
        -------
        tuple
            sigma, center and amplitude of the Gaussian function.
        """
        return self.sigma, self.center, self.amplitude

    def __eq__(self, other):
        r"""
        Parameters
        ----------
        other: object
            Another object.

        Returns
        -------
        bool
            ``True`` if both Gaussian functions have the same sigma,
            center, momentum and amplitude.

        Examples

        Two Gaussian functions with different amplitudes are different:

        >>> g1 = Gaussian(4, 2, k0=5, grid=[-1, 0, 1])
        >>> g2 = Gaussian(4, 2, k0=5, h=2, grid=[-1, 0, 1])
        >>> g1 == g2
        False

        If only the grid differs, then the two Gaussian functions are
        the same:

        >>> g3 = Gaussian(4, 2, k0=5, grid=[-1, -0.5, 0, 0.5, 1])
        >>> g1 == g3
        True
        """
        if isinstance(other, Gaussian):
            return ((self.sigma == other.sigma) and
                    (self.center == other.center) and
                    (self.momentum == other.momentum) and
                    (self.amplitude == other.amplitude))
        else:
            return False  # as it is definietly not a Gaussian

    def __add__(self, other):
        r"""
        Add another function to a Gaussian function.

        Parameters
        ----------
        other: Function
            Another function.

        Returns
        -------
        Gaussian or Function.
            Sum of a Gaussian function with another function.

        Example

        Two Gaussian functions differing only by the amplitude can be
        added to give another Gaussian function:

        >>> g1 = Gaussian(1, 0)
        >>> g2 = Gaussian(1, 0, h=3)
        >>> g = g1 + g2
        >>> assert g.amplitude == 4
        >>> assert g.center == g1.center
        >>> assert g.sigma == g1.sigma
        >>> assert g.momentum == g1.momentum

        Two different Gaussian functions that were not discretized
        cannot be added:

        >>> g1 + Gaussian(1, 1)
        Traceback (most recent call last):
        ValueError: Two analytic functions not discretized cannot be added.

        Finally, if the Gaussian function is not discretized over a
        grid, but the other function is, then the addition can be
        performed:

        >>> g = g1 + Function([-1, 0, 1], [1, 2, 3])
        >>> g.grid
        array([-1,  0,  1])
        >>> g.values
        array([ 1.60653066+0.j,  3.00000000+0.j,  3.60653066+0.j])
        """
        # If the other function is the same Gaussian function,
        # except for the amplitude, then return a Gaussian instance
        # with the correct amplitude
        if isinstance(other, Gaussian):
            k0 = self.momentum
            if self.center == other.center and self.sigma == other.sigma \
                    and k0 == other.momentum:
                h1 = self.amplitude
                h2 = other.amplitude
                return Gaussian(self.sigma, self.center, k0=k0, h=h1+h2,
                                grid=self.grid)
        # Otherwise, proceed as expected from an AnalyticFunction
        return super().__add__(other)

    def __repr__(self):
        r"""
        Returns
        -------
        str
            Representation of a Gaussian instance.
        """
        return ("Gaussian function of width {:.2f} and centered in {:.2f}"
                .format(self.sigma, self.center))

    @property
    def is_even(self):
        r"""
        Returns
        -------
        bool
            ``True`` if the Gaussian function is even.

        Examples

        A centered Gaussian with no initial momentum is even, even if
        its grid is not centered:

        >>> Gaussian(2, 0, grid=[-2, -1, 0]).is_even
        True

        A non-centered Gaussian is not even:

        >>> Gaussian(2, 2).is_even
        False

        A centered Gaussian with a non-zero initial momentum is
        not even:

        >>> Gaussian(2, 0, k0=1).is_even
        False
        """
        return self.momentum == 0. and self.center == 0.

    @property
    def is_odd(self):
        r"""
        Returns
        -------
        bool
            ``False``, as a Gaussian function can't be odd.
        """
        return False

    def is_inside(self, sw_pot, tol=10**(-5)):
        r"""
        Check if the Gaussian function can be considered as inside the
        1D Square-Well potential, given a tolerance value that must be
        larger than the values of the Gaussian function at the border of
        the potential.

        Parameters
        ----------
        sw_pot: SWPotential
            1D potential.
        tol: float
            Tolerance value (default to :math:`10^{-5}`).

        Returns
        -------
        bool
            ``True`` if the Gaussian function is inside the 1D SWP.

        Raises
        ------
        TypeError
            If ``sw_pot`` is not a
            :class:`~siegpy.swpotential.SWPotential` instance.
        """
        from siegpy import SWPotential
        if not isinstance(sw_pot, SWPotential):
            raise TypeError("potential is not a SWPotential instance")
        l = sw_pot.width
        return (abs(self.evaluate(-l/2)) <= tol) and \
               (abs(self.evaluate(+l/2)) <= tol)

    def abs(self):
        r"""
        Returns
        -------
        Gaussian
            Absolute value of the Gaussian function.

        Example

        The absolute value of a Gaussian function is nothing but
        the same Gaussian function without initial momentum:

        >>> g = Gaussian(5, -1, h=4, k0=-6)
        >>> assert g.abs() == Gaussian(5, -1, h=4)
        """
        h = self.amplitude
        return Gaussian(self.sigma, self.center, h=h, grid=self.grid)

    def conjugate(self):
        r"""
        Returns
        -------
        Gaussian
            Conjugate of the Gaussian function.

        Example

        The conjugate of a Gaussian is the same Gaussian function
        with a negative momentum:

        >>> g = Gaussian(5, -1, h=4, k0=-6)
        >>> assert g.conjugate() == Gaussian(5, -1, h=4, k0=6)
        """
        h = self.amplitude
        k_0 = self.momentum
        return Gaussian(self.sigma, self.center, h=h, k0=-k_0, grid=self.grid)

    def norm(self):
        r"""
        Returns
        -------
        float
            Analytic norm of the Gaussian function.

        Example

        The norm of a Gaussian function does not depend on the
        discretization grid:

        >>> g1 = Gaussian(2, 0)
        >>> g2 = Gaussian(2, 0, grid=[-1, 0, 1])
        >>> assert g1.norm() == g2.norm()
        """
        return np.sqrt(np.pi) * self.sigma * self.amplitude**2

    def _compute_values(self, grid):
        r"""
        Evaluate the Gaussian for each grid point :math:`x_0`:
        :math:`a e^{-\frac{(x_0-xc)^2}{2 sigma^2}} * e^{i k_0 x}`

        Parameters
        ----------
        grid: numpy array
            Discretization grid.

        Returns
        -------
        float or complex
            Value of the Gaussian function over the grid.

        Example

        >>> g = Gaussian(5, -1, h=-3.5)
        >>> g.evaluate(-1) == -3.5
        True
        """
        return (self.amplitude *
                np.exp(- (grid-self.center)**2 / (2*self.sigma**2)
                       + 1.j * self.momentum * grid))


class Rectangular(AnalyticFunction):
    r"""
    A rectangular function is characterized by:

    * a left and right border ``xl`` and ``xr`` (or, alternatively, by
      a width :math:`a = xr - xl` and a center ``xc``; see
      :meth:`~siegpy.functions.Rectangular.from_center_and_width` or
      :meth:`~siegpy.functions.Rectangular.from_width_and_center`),
    * an initial momentum ``k0``,
    * an amplitude ``h``.

    In addition to the behaviour of an analytic function:

    * the norm is computed analytically,
    * the attribute :attr:`~siegpy.functions.Rectangular.is_even`
      returns ``True`` if the rectangular function is centered, even if
      the discretization grid is not centered,
    * the equality and addition of two rectangular functions are also
      defined.
    """

    def __init__(self, xl, xr, k0=0.0, h=1., grid=None):
        r"""
        Parameters
        ----------
        xl: float
            Left border of the rectangular function.
        xr: float
            Right border of the rectangular function.
        k0: float
            Initial momentum of the rectangular function (default to 0).
        h: float
            Maximal amplitude of the rectangular function (default to
            1).
        grid: list or set or numpy array
            Discretization grid (optional).

        Raises
        ------
        ValueError
            If the amplitude ``h`` is zero or if the width is negative.

        Examples

        A rectangular function has several attributes:

        >>> r = Rectangular(-4, 2, k0=5)
        >>> r.xl
        -4
        >>> r.xr
        2
        >>> r.width
        6
        >>> r.center
        -1.0
        >>> r.momentum
        5
        >>> r.amplitude
        1.0

        If no discretization grid is passed, then the atrributes
        :attr:`~siegpy.functions.AnalyticFunction.grid` and
        :attr:`~siegpy.functions.AnalyticFunction.values` are set to
        ``None``:

        >>> r.grid is None and r.values is None
        True

        If a grid is passed, then the rectangular function is
        discretized (meaning its attribute
        :attr:`~siegpy.functions.AnalyticFunction.values` is not
        ``None``):

        >>> r1 = Rectangular(-4, 2, k0=5, grid=[-1, 0, 1])
        >>> r2 = Rectangular(-4, 2, k0=5, h=2, grid=[-1, 0, 1])
        >>> np.array_equal(r2.values, 2*r1.values)
        True

        .. note::
            The only way of modifying the values of a Rectangular instance
            is by setting a new grid:

            >>> r.grid = [-1, 0, 1]
            >>> assert np.array_equal(r.grid, r1.grid)
            >>> assert np.array_equal(r.values, r1.values)
            >>> r.values = [2, 1, 2]
            Traceback (most recent call last):
            AttributeError: can't set attribute

        .. warning::

            A rectangular function must have a strictly positive width:

            >>> Rectangular(1, -1)
            Traceback (most recent call last):
            ValueError: The width must be strictly positive.
            >>> Rectangular(1, 1)
            Traceback (most recent call last):
            ValueError: The width must be strictly positive.
        """
        # Check the width and amplitude
        if xl >= xr:
            raise ValueError("The width must be strictly positive.")
        if h == 0:
            raise ValueError(
                "The amplitude of the rectangular function must not be 0")
        # Initialize the attributes
        self._xl = xl
        self._xr = xr
        self._amplitude = h
        self._width = xr - xl
        self._center = (xl + xr) / 2.
        self._momentum = k0
        super().__init__(grid)

    @classmethod
    def from_center_and_width(cls, xc, width, k0=0.0, h=1.0, grid=None):
        r"""
        Initialization of a rectangular function centered in ``xc``, of
        width ``a``, with an amplitude ``h`` and with an initial
        momentum ``k0``.

        Parameters
        ----------
        xc: float
            Center of the rectangular function.
        width: float
            Width of the rectangular function.
        k0: float
            Initial momentum of the rectangular function (default to 0).
        h: float
            Maximal amplitude of the rectangular function (default
            to 1.).
        grid: list or set or numpy array
            Discretization grid (optional).

        Returns
        -------
        Rectangular
            An initialized rectangular function.

        Example

        >>> r = Rectangular.from_center_and_width(4, 2)
        >>> r.xl
        3.0
        >>> r.xr
        5.0
        """
        return cls(xc-width/2, xc+width/2, k0=k0, h=h, grid=grid)

    @classmethod
    def from_width_and_center(cls, width, xc, k0=0.0, h=1.0, grid=None):
        r"""
        Similar to the class method
        :meth:`~siegpy.functions.Rectangular.from_center_and_width`.

        Returns
        -------
        Rectangular
            An initialized rectangular function.

        Example

        >>> r = Rectangular.from_width_and_center(4, 2)
        >>> r.xl
        0.0
        >>> r.xr
        4.0
        """
        return cls.from_center_and_width(xc, width, k0=k0, h=h, grid=grid)

    @property
    def xl(self):
        r"""
        Returns
        -------
        float
            Left border of the rectangular function.
        """
        return self._xl

    @property
    def xr(self):
        r"""
        Returns
        -------
        float
            Right border of the rectangular function.
        """
        return self._xr

    @property
    def amplitude(self):
        r"""
        Returns
        -------
        float
            Amplitude of the rectangular function.
        """
        return self._amplitude

    @property
    def width(self):
        r"""
        Returns
        -------
        float
            Width of the rectangular function.
        """
        return self._width

    @property
    def center(self):
        r"""
        Returns
        -------
        float
            Center of the rectangular function.
        """
        return self._center

    @property
    def momentum(self):
        r"""
        Returns
        -------
        float
            Momentum of the rectangular function.
        """
        return self._momentum

    def __eq__(self, other):
        r"""
        Parameters
        ----------
            Another object
        other: object

        Returns
        -------
        bool
            ``True`` if both objects are Rectangular functions with the
            same amplitude, width, center and momentum.

        Examples

        Two rectangular functions with different amplitudes are
        different:

        >>> r1 = Rectangular(-4, 2, k0=5, grid=[-1, 0, 1])
        >>> r2 = Rectangular(-4, 2, k0=5, h=2, grid=[-1, 0, 1])
        >>> r1 == r2
        False

        Two rectangular functions that are identical, except for the
        grid, are considered to be the same:

        >>> r3 = Rectangular(-4, 2, k0=5, grid=[-1, -0.5, 0, 0.5, 1])
        >>> r1 == r3
        True
        """
        if isinstance(other, Rectangular):
            return ((self.width == other.width) and
                    (self.center == other.center) and
                    (self.momentum == other.momentum) and
                    (self.amplitude == other.amplitude))
        else:
            return False

    def __add__(self, other):
        r"""
        Add another function to a rectangular function.

        Parameters
        ----------
        other: Function
            Another function.

        Returns
        -------
        Rectangular or Function.
            Sum of a rectangular function with another function.

        Example

        Two rectangular functions differing only by the amplitude can
        be added to give another rectangular function:

        >>> r1 = Rectangular(-1, 1)
        >>> r2 = Rectangular(-1, 1, h=3)
        >>> r = r1 + r2
        >>> assert r.amplitude == 4
        >>> assert r.center == r1.center
        >>> assert r.width == r1.width
        >>> assert r.momentum == r1.momentum

        Two different rectangular functions that were not discretized
        cannot be added:

        >>> r1 + Rectangular(-2, 2)
        Traceback (most recent call last):
        ValueError: Two analytic functions not discretized cannot be added.

        Finally, if the rectangular function is not discretized over a
        grid, but the other function is, then the addition can be
        performed:

        >>> r = r1 + Function([-1, 0, 1], [1, 2, 3])
        >>> r.grid
        array([-1,  0,  1])
        >>> r.values
        array([ 2.+0.j,  3.+0.j,  4.+0.j])
        """
        # If the other function is the same rectangular function,
        # except for the amplitude, then return a Rectangular instance
        # with the correct amplitude
        if isinstance(other, Rectangular):
            k0 = self.momentum
            if self.center == other.center and self.width == other.width \
               and k0 == other.momentum:
                h1 = self.amplitude
                h2 = other.amplitude
                return Rectangular.from_width_and_center(
                    self.width, self.center, k0=k0, h=h1+h2, grid=self.grid)
        # Otherwise, proceed as expected from an AnalyticFunction
        return super().__add__(other)

    def __repr__(self):
        r"""
        Returns
        -------
        str
            Representation of a Rectangular instance.
        """
        return ("Rectangular function of width {s.width:.2f} and centered "
                "in {s.center:.2f}".format(s=self))

    @property
    def is_even(self):
        r"""
        Returns
        -------
        bool
            ``True`` if the rectangular function is even.

        Examples

        A centered rectangular function is even, even if t grid is
        not centered:

        >>> Rectangular(-2, 2, grid=[-2, -1, 0]).is_even
        True

        A non-centered rectangular function cannot be even:

        >>> Rectangular(-2, 0).is_even
        False

        A centered rectangular function with an initial momentum
        cannot be even:

        >>> Rectangular(-2, 2, k0=1).is_even
        False
        """
        return self.momentum == 0. and self.center == 0.

    @property
    def is_odd(self):
        r"""
        Returns
        -------
        bool
            ``False``, as a rectangular function can't be odd.
        """
        return False

    def _compute_values(self, grid):
        r"""
        Evaluation of the Rectangular function for each grid point.

        Parameters
        ----------
        grid: numpy array
            Discretization grid.

        Returns
        -------
        float
            Value of the Rectangular function over the grid.

        Examples

        >>> r = Rectangular(-5, 1, h=-3.5)
        >>> r.evaluate(-1) == -3.5
        True
        >>> r.evaluate(-10) == 0
        True
        """
        # Use the numpy vectorization to divide the grid in three
        where_1, = np.where(grid < self.xl)
        where_2 = np.logical_and(grid >= self.xl, grid <= self.xr)
        where_3, = np.where(grid > self.xr)
        grid_1 = grid[where_1]
        grid_2 = grid[where_2]
        grid_3 = grid[where_3]
        # Evaluate the recytangular function and return it
        values_1 = np.zeros_like(grid_1)
        values_2 = self.amplitude * np.exp(1.j * self.momentum * grid_2)
        values_3 = np.zeros_like(grid_3)
        return np.concatenate((values_1, values_2, values_3))

    def abs(self):
        r"""
        Returns
        -------
        Rectangular
            Absolute value of the Rectangular function.

        Example

        The absolute value of a Rectangular function is nothing but
        the same Rectangular function without initial momentum:

        >>> r = Rectangular(-5, 1, h=4, k0=-6)
        >>> assert r.abs() == Rectangular(-5, 1, h=4)
        """
        return Rectangular(
            self.xl, self.xr, h=self.amplitude, grid=self.grid)

    def conjugate(self):
        r"""
        Returns
        -------
        Rectangular
            Conjugate of the Rectangular function.

        Example

        The conjugate of a Rectangular is the same Rectangular
        function with a negative momentum:

        >>> r = Rectangular(-5, 1, h=4, k0=-6)
        >>> assert r.conjugate() == Rectangular(-5, 1, h=4, k0=6)
        """
        h = self.amplitude
        k_0 = self.momentum
        return Rectangular(self.xl, self.xr, h=h, k0=-k_0, grid=self.grid)

    def norm(self):
        r"""
        Returns
        -------
        float
            Analytic norm of a rectangular function, that is equal to
            its width times its amplitude squared.

        Examples

        >>> r = Rectangular(-1, 1, h=3)
        >>> r.norm()
        18

        The norm does not depend on the discretization grid:

        >>> r.norm() == Rectangular(-1, 1, h=3, grid=[-1, 0, 1]).norm()
        True
        """
        return self.width * self.amplitude**2

    def split(self, sw_pot):
        r"""
        Split the rectangular function into three other rectangular
        functions, each one spreading over only one of the three regions
        defined by a 1D Square-Well potential SWP. If the original
        rectangular function does not spread over one particular region,
        the returned value for this region is ``None``.

        Parameters
        ----------
        sw_pot: SWPotential
            1D Square-Well Potential

        Returns
        -------
        tuple of length 3
            Three rectangular functions.

        Raises
        ------
        TypeError
            If ``sw_pot`` is not a
            :class:`~siegpy.swpotential.SWPotential` instance.
        """
        # Check that the argument sw_pot is a SWPotential instance
        from siegpy import SWPotential
        if not isinstance(sw_pot, SWPotential):
            raise TypeError("The argument must be a SWPotential.")
        # Initial parameters of the rectangular function
        l = sw_pot.width
        xl = self.xl
        xr = self.xr
        h = self.amplitude
        k0 = self.momentum
        grid = self.grid
        # Rectangular function in region I
        if xl < -l/2:
            r_1 = Rectangular(xl, min(xr, -l/2), h=h, k0=k0, grid=grid)
        else:
            r_1 = None
        # Rectangular function in region III
        if xr > l/2:
            r_3 = Rectangular(max(xl, l/2), xr, h=h, k0=k0, grid=grid)
        else:
            r_3 = None
        # Rectangular function in region II
        if xl >= l/2 or xr <= -l/2:
            r_2 = None
        else:
            if xl <= -l/2:
                xl_2 = -l/2
            else:
                xl_2 = xl
            if xr >= l/2:
                xr_2 = l/2
            else:
                xr_2 = xr
            r_2 = Rectangular(xl_2, xr_2, h=h, k0=k0, grid=grid)
        return r_1, r_2, r_3
