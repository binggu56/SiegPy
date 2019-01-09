# -*- coding: utf-8 -*-
r"""
The classes defining the analytic eigenstates of a generic 1D potential
with compact support are presented below:
* :class:`AnalyticEigenstate` is the base class of the other two,
* :class:`AnalyticSiegert` is meant to represent analytic Siegert states,
* :class:`AnalyticContinuum` is meant to represent analytic continuum
  states

They all are abstract base classes:

A new exception is also defined:
:class:`~siegpy/analyticeigenstates/WavenumberError`.
"""

from abc import ABCMeta, abstractmethod
import numpy as np
from .eigenstates import Eigenstate
from .functions import AnalyticFunction


class AnalyticEigenstate(AnalyticFunction, Eigenstate, metaclass=ABCMeta):
    r"""
    Class gathering the attributes and methods necessary to define an
    analytic eigenstate of any analytical problem (such as the 1D
    Square-Well Potential case).

    .. note::

        To consider a case as analytic, the wavefunctions of the
        Siegert and continuum states should be known analytically. The
        analytic scalar product of both types of eigenstates with at
        least one type of test function should also be defined.
    """

    # You should be able to define the type of an analytical eigenstate
    SIEGERT_TYPES = {
        "b": "bound",
        "ab": "anti-bound",
        "r": "resonant",
        "ar": "anti-resonant",
        None: "continuum",
    }

    def __init__(self, k, potential, grid, analytic):
        r"""
        Parameters
        ----------
        k: complex
            Wavenumber of the eigenstate.
        potential: Potential
            Potential for which the analytic eigenstate is known.
        grid: list or set or numpy array
            Discretization grid.
        analytic: bool
            If ``True``, the scalar products must be computed
            analytically.

        Raises
        ------
        WavenumberError
            If the inferred Siegert type is unknown.
        """
        S_type = self._find_Siegert_type(k)
        if S_type in self.SIEGERT_TYPES:
            self._Siegert_type = S_type
        else:
            # Error only if bad implementation, hence no coverage
            raise WavenumberError()  # pragma: no cover
        self._wavenumber = k
        self._energy = k ** 2 / 2
        self._potential = potential
        self.analytic = analytic
        super().__init__(grid)

    @property
    def potential(self):
        r"""
        Returns
        -------
        complex or float
            Potential for which the eigenstate is a solution.
        """
        return self._potential

    @property
    def wavenumber(self):
        r"""
        Returns
        -------
        complex or float
            Wavenumber of the eigenstate.
        """
        return self._wavenumber

    @property
    def energy(self):
        r"""
        Returns
        -------
        complex or float
            Energy of the Eigenstate
        """
        return self._energy

    @abstractmethod
    def _find_Siegert_type(self, k):  # pragma: no cover
        r"""
        .. note:: This is an asbtract method.

        Parameters
        ----------
        k: float or complex
            Wavenumber of the eigenstate

        Returns
        -------
        str or NoneType
            Type of the eigenstate from its wavenumber k (``'b'`` for
            bound states, ``'ab'`` for anti-bound states, ``'r'`` for
            resonant states, ``'ar'`` for anti-resonant states, and
            ``None`` for continuum states).
        """
        pass

    @property
    def analytic(self):
        r"""
        Returns
        -------
        bool
            Analyticity of the scalar products.
        """
        return self._analytic

    @analytic.setter
    def analytic(self, new_value):
        r"""
        Setter of the analyticity of the eigenstate.

        Parameters
        ----------
        new_value: bool
            New value of the analyticity of the scalar products.

        Raises
        ------
        ValueError
            If ``new_value`` is not a Boolean.
        """
        if isinstance(new_value, bool):
            self._analytic = new_value
        else:
            raise ValueError(
                "analytic must be a Boolean (cannot be {})".format(new_value)
            )

    def __eq__(self, other):
        r"""
        Two analytic eigenstates are the same if they share the same
        wavenumber, potential and Siegert_type.

        Parameters
        ----------
        other: object
            Another object

        Returns
        -------
        bool
            ``True`` if both eigenstates are the same.
        """
        if isinstance(other, AnalyticEigenstate):
            return (
                np.isclose(self.wavenumber, other.wavenumber)
                and (self.potential == other.potential)
                and (self.Siegert_type == other.Siegert_type)
            )
        # Not any else or elif covered, since there is only one
        # analytical case actually implemented (SWP case)
        else:
            return False  # pragma: no cover

    def __repr__(self):
        r"""
        Returns
        -------
        str
            Representation of an analytic eigenstate.
        """
        Siegert_type = self.SIEGERT_TYPES[self.Siegert_type].capitalize()
        return "{} eigenstate of energy {:.3f}".format(Siegert_type, self.energy)

    @abstractmethod
    def scal_prod(self, other, xlim=None):
        r"""
        .. note:: This is an asbtract method.

        Evaluate the scalar product between the state and a test
        function ``other``. It ensures that the numerical scalar product
        is computed when the analytical scalar product with test
        functions is not implemented in the child class.

        Parameters
        ----------
        other: Function
            Test function.
        xlim: tuple(float, float)
            Bounds of the space grid defining the interval over which
            the scalar product must be computed.

        Returns
        -------
        float or complex
            Value of the scalar product of the state with a test
            function.
        """
        return super().scal_prod(other, xlim=xlim)


class AnalyticSiegert(AnalyticEigenstate, metaclass=ABCMeta):
    r"""
    Class specifying the abstract method allowing to find the type of
    the analytic eigenstate, in the case it is a Siegert state.
    """

    def _find_Siegert_type(self, k):
        r"""
        Parameters
        ----------
        k: complex
            Wavenumber of the Siegert state.

        Returns
        -------
        str
            The type of the Siegert state, namely:

            * ``'b'`` for a bound state
            * ``'ab'`` for an anti-bound state
            * ``'r'`` for a resonant state
            * ``'ar'`` for an anti-resonant state

        Raises
        ------
        WavenumberError
            If the wavenumber is equal to zero.
        """
        # If its real part is zero, then it's either a bound
        # or an antibound state
        if k.real == 0.0:
            if k.imag > 0.0:
                S_type = "b"
            elif k.imag < 0.0:
                S_type = "ab"
            else:
                # This is a very rare case, hence no coverage
                raise WavenumberError(
                    "The wavenumber is zero: impossible to " "define the parity."
                )  # pragma: no cover
        # Else, if its real part is positive...
        elif k.real > 0.0:
            # ... if it has a negative imaginary part,
            # then it's a resonant state
            if k.imag < 0.0:
                S_type = "r"
        # Finally, if the real part is negative...
        elif k.real < 0.0:
            # ... if it has a negative imaginary part,
            # then it's an anti-resonant state
            if k.imag < 0.0:
                S_type = "ar"
        return S_type


class AnalyticContinuum(AnalyticEigenstate, metaclass=ABCMeta):
    r"""
    Class specifying the abstract method allowing to find the type of
    the analytic eigenstate, in the case it is a continuum state.
    """

    def _find_Siegert_type(self, k):
        r"""
        Parameters
        ----------
        k: float
            Wavenumber of the continuum state.

        Returns
        -------
        NoneType
            ``None``, given that a continuum state is not a Siegert
            state.

        Raises
        ------
        WavenumberError
            If the wavenumber is imaginary.
        """
        if not np.isclose(k.imag, 0.0):
            raise WavenumberError(
                "A continuum state cannot have an imaginary wavenumber"
            )
        else:
            return None


class WavenumberError(Exception):
    r"""
    Error thrown if the wavenumber of an eigenstate is incorrect.
    """
    pass
