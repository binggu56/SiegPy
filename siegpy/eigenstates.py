# -*- coding: utf-8 -*-
r"""
The class defining the eigenstates of a generic 1D potential with
compact support is presented below.
"""

import numpy as np
from .functions import Function


class Eigenstate(Function):
    r"""
    Class defining an eigenstate. The main change with the
    :class:`~siegpy.functions.Function` class is that an Eigenstate
    instance is associated to an energy, and possibly a Siegert type and
    a virial value.
    """

    # You should be able to define the type of an analytical eigenstate
    SIEGERT_TYPES = {"b": "bound", "r": "resonant", None: "continuum", "U": "unknown"}

    def __init__(self, grid, values, energy, Siegert_type="U", virial=None):
        r"""
        Parameters
        ----------
        grid: list or set or numpy array
            Discretization grid.
        values: list or set or numpy array
            Function evaluated on the grid points.
        energy: float or complex
            Energy of the eigenstate.
        Siegert_type: str
            Type of the Eigenstate (default to 'U' for unknown).
        virial: float or complex
            Value of the virial theorem for the eigenstate (optional).


        Examples

        An Eigenstate instance has several attributes:

        >>> wf = Eigenstate([0, 1, 2], [1, 2, 3], 1.0)
        >>> wf.grid
        array([0, 1, 2])
        >>> wf.values
        array([1, 2, 3])
        >>> wf.energy
        1.0
        """
        self._energy = energy
        self._wavenumber = np.sqrt(2 * energy)
        self._Siegert_type = Siegert_type
        self._virial = virial
        super().__init__(grid, values)

    @property
    def energy(self):
        r"""
        Returns
        -------
        float or complex
            Energy of the eigenstate.
        """
        return self._energy

    @property
    def wavenumber(self):
        r"""
        Returns
        -------
        complex
            Wavenumber of the eigenstate
        """
        return self._wavenumber

    @property
    def virial(self):
        r"""
        Returns
        -------
        float or complex
            Virial of the eigenstate.
        """
        return self._virial

    @property
    def Siegert_type(self):
        r"""
        Returns
        -------
        str or NoneType
            The type of the state.
        """
        return self._Siegert_type

    def __repr__(self):
        r"""
        Returns
        -------
        str
            Representation of the Eigenstate instance.
        """
        Siegert_type = self.SIEGERT_TYPES[self.Siegert_type].capitalize()
        return "{} eigenstate of energy {:.3f}".format(Siegert_type, self.energy)

    def scal_prod(self, other, xlim=None):
        r"""
        Override of the :meth:`siegpy.functions.Function.scal_prod`
        method to take into account the c-product for resonant and
        anti-resonant states.

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
        """
        if self.Siegert_type in ["r", "ar"]:
            return self.conjugate().scal_prod(other, xlim=xlim)
        else:
            return super().scal_prod(other, xlim=xlim)
