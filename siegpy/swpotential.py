# -*- coding: utf-8 -*-
"""
The :class:`SWPotential` class representing a 1D Square-Well Potential
(1DSWP) is defined below.
"""

from siegpy import Potential, Rectangular


class SWPotential(Rectangular, Potential):
    """
    A 1D SW potential is nothing but a rectangular function that is
    considered as a potential. This class therefore inherits from and
    extends the :class:`~siegpy.functions.Rectangular` and
    :class:`~siegpy.potential.Potential` classes.
    """

    def __init__(self, l, V0, grid=None):
        """
        A 1D SWP is initialized from a width :math:`l` and a depth
        :math:`V_0`. If a grid is given, the potential is discretized
        over this grid.

        Parameters
        ----------
        l: float
            Width of the potential.
        V0: float
            Depth of the potential.
        grid: numpy array or list or set
            Discretization grid of the potential (optional).

        Raises
        ------
        ValueError
            If the depth and width of the potential are not strictly
            positive.


        Examples

        A SWPotential instance represents nothing but a centered
        rectangular function with a negative amplitude:

        >>> xgrid = [-4, -2, 0, 2, 4]
        >>> pot = SWPotential(5, 10, grid=xgrid)
        >>> pot.width == 5 and pot.amplitude == -10
        True
        >>> pot.center == 0 and pot.momentum == 0
        True
        >>> pot.values
        array([  0.+0.j, -10.+0.j, -10.+0.j, -10.+0.j,   0.+0.j])

        Rather than dealing with a negative amplitude, the positive
        depth of the SW potential is mainly used:

        >>> pot.depth
        10
        """
        # Check that V0 is positive
        if V0 <= 0.0:
            raise ValueError("The depth the potential must be strictly positive.")
        # Initialization of the 1DSWP, that has a 'depth' attribute
        super().__init__(-l / 2, l / 2, h=-V0, grid=grid)
        self._depth = V0

    @classmethod
    def from_center_and_width(cls, xc, a, k0=0.0, h=1.0, grid=None):
        r"""
        Class method overriding the inherited
        :func:`siegpy.functions.Rectangular.from_center_and_width`
        class method.

        Parameters
        ----------
        xc: float
            Center of the potential (must be 0).
        a: float
            Width of the potential.
        k0: float
            Initial momentum of the potential (must be 0).
        h: float
            Depth of the potential (default to 1).
        grid: list or numpy array
            Discretization grid (optional).

        Returns
        -------
        SWPotential
            An initialized 1D SW potential.

        Raises
        ------
        ValueError
            If the initial momentum or the center is not zero.


        Examples

        >>> pot = SWPotential.from_center_and_width(0, 5, h=-10)
        >>> pot.width == 5 and pot.depth == 10
        True

        A Potential cannot have a non-zero center nor an initial
        momentum:

        >>> SWPotential.from_center_and_width(2, 5)
        Traceback (most recent call last):
        ValueError: A SWPotential must be centered.
        >>> SWPotential.from_center_and_width(0, 5, k0=1)
        Traceback (most recent call last):
        ValueError: A SWPotential cannot have an initial momentum.

        The same is valid for the :meth:`from_width_and_center` class
        method:

        >>> SWPotential.from_width_and_center(5, 2)
        Traceback (most recent call last):
        ValueError: A SWPotential must be centered.
        >>> SWPotential.from_width_and_center(5, 0, k0=1)
        Traceback (most recent call last):
        ValueError: A SWPotential cannot have an initial momentum.
        """
        # Check the values of the parameters
        if xc != 0:
            raise ValueError("A SWPotential must be centered.")
        if k0 != 0:
            raise ValueError("A SWPotential cannot have an initial momentum.")
        # Return a SWPotential instance
        return cls(a, -h, grid=grid)

    @property
    def depth(self):
        """
        Returns
        -------
        float
            Depth of the 1D SW potential.
        """
        return self._depth

    def __eq__(self, other):
        """
        Parameters
        ----------
        other: object
            Any other object.

        Returns
        -------
        bool
            ``True`` if two SW potentials have the same width and depth.


        Examples

        >>> SWPotential(5, 10) == SWPotential(5, 10, grid=[-1, 0, 1])
        True
        >>> SWPotential(5, 10) == Potential([-1, 0, 1], [-10, -10, -10])
        False
        """
        return isinstance(other, SWPotential) and (
            self.width == other.width and self.depth == other.depth
        )

    def __repr__(self):
        """
        Returns
        -------
        str
            Representation of the square-well potential.
        """
        return (
            "1D Square-Well Potential of width {0.width:.2f} "
            "and depth {0.depth:.2f}".format(self)
        )

    def __add__(self, other):
        """
        Override of the addition, taking care of the case of the
        addition of two SW potentials with the same width.

        Parameters
        ----------
        other: Potential
            Another potential

        Returns
        -------
        Potential or Function
            The sum of both potentials.


        Examples

        The addition of two Square-Well potentials of the same width
        keeps the analyticity:

        >>> xgrid = [-4, -2, 0, 2, 4]
        >>> pot = SWPotential(5, 10, grid=xgrid)
        >>> pot += SWPotential(5, 5)
        >>> isinstance(pot, SWPotential)
        True
        >>> pot.amplitude == -15 and pot.depth == 15 and pot.width == 5
        True
        >>> pot.values
        array([  0.+0.j, -15.+0.j, -15.+0.j, -15.+0.j,   0.+0.j])

        In any other case, the analyticity is lost, but the result
        still is a potential:

        >>> pot += Potential(xgrid, [5, 5, 5, 5, 5])
        >>> isinstance(pot, SWPotential)
        False
        >>> isinstance(pot, Potential)
        True
        >>> pot.values
        array([  5.+0.j, -10.+0.j, -10.+0.j, -10.+0.j,   5.+0.j])
        """
        if isinstance(other, SWPotential) and self.width == other.width:
            return SWPotential(self.width, self.depth + other.depth, grid=self.grid)
        return super().__add__(other)

    def complex_scaled_values(self, coord_map):
        r"""
        Evaluates the complex scaled SW potential. It actually amounts
        to the potential without complex scaling, as the Square-Well
        Potential is a piecewise function (the complex scaling has no
        effect on it).

        Parameters
        ----------
        coord_map: CoordMap
            Coordinate mapping.

        Returns
        -------
        numpy array
            Complex scaled values of the potential.
        """
        return self.values
