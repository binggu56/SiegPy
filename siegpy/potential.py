# -*- coding: utf-8 -*-
"""
The :class:`Potential` class is defined below.
"""

from .functions import Function


class Potential(Function):
    r"""
    Class defining a generic 1D potential.


    Examples

    A Potential is a function:

    >>> pot = Potential([-1, 0, 1], [-3, 2, 0])
    >>> pot.grid
    array([-1,  0,  1])
    >>> pot.values
    array([-3,  2,  0])

    The main difference is that only Potential instances can be added
    to a potential:

    >>> pot += Function([1, 2, 3], [0, -1, 0])
    Traceback (most recent call last):
    TypeError: Cannot add a <class 'siegpy.functions.Function'> to a Potential
    """

    def __add__(self, other):
        r"""
        Add two potentials.

        Parameters
        ----------
        other: Potential
            Another potential.

        Returns
        -------
        Potential
            Sum of both potentials.

        Raises
        ------
        TypeError
            If ``other`` is not a :class:`Potential` instance.


        Examples

        Two potentials can be added:

        >>> pot1 = Potential([1, 2, 3], [1, 1, 1])
        >>> pot2 = Potential([1, 2, 3], [0, -1, 0])
        >>> pot = pot1 + pot2
        >>> pot.grid
        array([1, 2, 3])
        >>> pot.values
        array([1, 0, 1])

        The previous potentials are unchanged:

        >>> pot1.values
        array([1, 1, 1])
        >>> pot2.values
        array([ 0, -1,  0])
        """
        if not isinstance(other, Potential):
            raise TypeError(
                "Cannot add a {} to a Potential".format(type(other)))
        f = super().__add__(other)
        return Potential(f.grid, f.values)

    def complex_scaled_values(self, coord_map):
        r"""
        Evaluate the complex scaled value of the potential given a
        coordinate mapping.

        Parameters
        ----------
        coord_map: CoordMap
            Coordinate mapping used.

        Raises
        ------
        NotImplementedError
        """
        # Not yet implemented for an arbitrary potential, only for
        # analytic potentials.
        raise NotImplementedError()
