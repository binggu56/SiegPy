#  -*- coding: utf-8 -*-
"""
Here are defined the classes allowing one to create and use the
eigenstates of the 1-dimendional (1D) Square Well Potential (SWP)
case.

The two classes are:

* :class:`~siegpy.swpeigenstates.SWPSiegert`, the class of Siegert
  states of the 1DSWP
* :class:`~siegpy.swpeigenstates.SWPContinuum`, the class of Continuum
  states

Both classes derive from the
:class:`~siegpy.swpeigenstates.SWPEigenstate` abstract class, that
forces to redefine the scalar product so that it is computed
analytically when possible.
"""

from abc import ABCMeta, abstractmethod
import math
import cmath
import numpy as np
from scipy.special import erf
from siegpy import Gaussian, Rectangular, SWPotential
from .analyticeigenstates import AnalyticEigenstate, AnalyticSiegert, AnalyticContinuum
from .swputils import q, fem, fep, fom, fop, find_parity


class SWPEigenstate(AnalyticEigenstate, metaclass=ABCMeta):
    r"""
    This is the base class for any eigenstate of the 1D Square-Well
    Potential (1DSWP). It defines some generic methods, while leaving
    some others to be defined by its subclasses, that are:

    * the :class:`~siegpy.swpeigenstates.SWPSiegert` class, defining the
      Siegert states of the Hamiltonian defined by the 1D SWP,
    * the :class:`~siegpy.swpeigenstates.SWPContinuum` class, defining
      the continuum states of the same problem.
    """

    PARITIES = {"e": "Even ", "o": "Odd "}

    def __init__(self, k, parity, potential, grid, analytic):
        r"""
        In addition to those of an
        :class:`~siegpy.analyticeigenstates.AnalyticEigenstate`, one
        of the main characteristics of a SWPEigenstate is its parity
        (the eigenstate must be an even or odd function).

        Parameters
        ----------
        k: complex
            Wavenumber of the state.
        parity: str
            Parity of the state (``'e'`` for even, ``'o'`` for odd).
        potential: SWPotential
            1D Square-Well Potential giving rise to this eigenstate.
        grid: list or set or numpy array
            Discretization grid.
        analytic: bool
            If ``True``, the scalar products must be computed
            analytically.

        Raises
        ------
        TypeError
            If the potential is not a :class:`SWPotential` instance.
        """
        # Check that the potential is a SWPotential
        if not isinstance(potential, SWPotential):
            raise TypeError("potential must be a SWPotential instance")
        # Initialization of the attributes specific to the 1DSWP case.
        self._parity = parity
        super().__init__(k, potential, grid, analytic)

    @property
    def parity(self):
        r"""
        Returns
        -------
        str
            Parity of the eigenstate (``'e'`` if it is even or ``'o'``
            if it is odd).
        """
        return self._parity

    def __eq__(self, other):
        r"""
        Two eigenstates of the 1D SWP are the same if they have the
        same wavenumber, parity, potential and Siegert_type.

        Parameters
        ----------
        other: object
            Another object

        Returns
        -------
        bool
            ``True`` if both eigenstates are the same.
        """
        return isinstance(other, SWPEigenstate) and (
            self.parity == other.parity and super().__eq__(other)
        )

    def __repr__(self):
        r"""
        Returns
        -------
        str
            Representation of an eigenstate of the 1D SWP case.
        """
        return self.PARITIES[self.parity] + super().__repr__().lower()

    @property
    def is_even(self):
        r"""
        Returns
        -------
        bool
            ``True`` if the eigenstate is even.
        """
        return self.parity == "e"

    @property
    def is_odd(self):
        r"""
        Returns
        -------
        bool
            ``True`` if the eigenstate is odd.
        """
        return self.parity == "o"

    def _compute_values(self, grid):
        r"""
        Evaluate the wavefunction of the eigenstate discretized over the
        whole grid.

        Parameters
        ----------
        grid: list or set or numpy array
            Discretization grid.

        Returns
        -------
        numpy array
            Wavefunction discretized over the grid.
        """
        # Separate the grid in two in the first half of the space grid
        # (region I and first half of region II)
        grid = np.array(grid)
        xr = self.potential.xr  # Limit between region II and III
        xl = self.potential.xl  # Limit between region I and II
        # Define the location of the grid points in region I and II for
        # negative values
        where_1, = np.where(grid < xl)
        where_2 = np.logical_and(grid >= xl, grid <= xr)
        where_3, = np.where(grid > xr)
        # Set both grids according to the locations and return them
        grid_1 = grid[where_1]
        grid_2 = grid[where_2]
        grid_3 = grid[where_3]
        # Compute the eigenstate in the three regions of space. This
        # hugely depends on the parity of the state. We use the parity
        # to compute it only in the first half of the grid points,
        # wf_inf. The other half, wf_sup, is the reverse of the array
        # wf_inf, with a minus sign if the state is odd.
        if self.is_even:
            wf_1 = self._even_wf_1(grid_1)
            wf_2 = self._even_wf_2(grid_2)
            wf_3 = self._even_wf_1(-grid_3)
        elif self.is_odd:
            wf_1 = self._odd_wf_1(grid_1)
            wf_2 = self._odd_wf_2(grid_2)
            wf_3 = -self._odd_wf_1(-grid_3)
        # Return the whole numpy array
        return np.concatenate((wf_1, wf_2, wf_3))

    @abstractmethod
    def _even_wf_1(self, grid_1):  # pragma: no cover
        r"""
        .. note:: This is an asbtract method.

        Evaluate the even eigenstate wavefunction over the grid points
        in region *I*.

        Parameters
        ----------
        grid_1: numpy array
            Grid in region *I*.

        Returns
        -------
        numpy array
            Even eigenstate wavefunction discretized over the grid in
            region *I*.
        """
        pass

    @abstractmethod
    def _even_wf_2(self, grid_2):  # pragma: no cover
        r"""
        .. note:: This is an asbtract method.

        Evaluate the even eigenstate wavefunction over the grid points
        in region *II*.

        Parameters
        ----------
        grid_2: numpy array
            Grid in region *II*.

        Returns
        -------
        numpy array
            Even eigenstate wavefunction discretized over the grid in
            region *II*.
        """
        pass

    @abstractmethod
    def _odd_wf_1(self, grid_1):  # pragma: no cover
        r"""
        .. note:: This is an asbtract method.

        Evaluate the odd eigenstate wavefunction over the grid points
        in region *I*.

        Parameters
        ----------
        grid_1: numpy array
            Grid in region *I*.

        Returns
        -------
        numpy array
            Odd eigenstate wavefunction discretized over the grid in
            region *I*.
        """
        pass

    @abstractmethod
    def _odd_wf_2(self, grid_2):  # pragma: no cover
        r"""
        .. note:: This is an asbtract method.

        Evaluate the odd eigenstate wavefunction over the grid points
        in region *II*.

        Parameters
        ----------
        grid_2: numpy array
            Grid in region *II*.

        Returns
        -------
        numpy array
            Odd eigenstate wavefunction discretized over the grid in
            region *II*.
        """
        pass

    def scal_prod(self, other, xlim=None):
        r"""
        Evaluate the scalar product of an eigenstate with another
        function. It can be computed analytically or not.

        .. note::

            If it has to be computed analytically, a ``TypeError``
            may be raised if the analytic scalar product with the test
            function `other` is analytically unknown (at present, only
            scalar products with Gaussian and rectangular functions are
            possible).

        Parameters
        ----------
        other: Function
            Another Function.
        xlim: tuple(float or int, float or int)
            Range of the x axis for the integration (optional).

        Returns
        -------
        complex
            Result of the scalar product.

        Raises
        ------
        TypeError
            If the value of the analytical scalar product with the other
            function is unknown.
        """
        # Analytical scalar product
        if self.analytic:
            if isinstance(other, Gaussian):
                return self._scal_prod_with_Gaussian(other)
            elif isinstance(other, Rectangular):
                return self._scal_prod_with_Rectangular(other)
            # #TODO: define the next functions?
            # #elif isinstance(other, SWPContinuum):
            # #    sp = self._scal_prod_with_SWPContinuum(other)
            # #elif isinstance(other, SWPSiegert):
            # #    sp = self._scal_prod_with_SWPSiegert(other)
            else:
                raise TypeError(
                    "You are trying to analytically compute a scalar product "
                    "between a SWPSiegert and an object of type {}.".format(type(other))
                    + "\nThis can only be done with a "
                    "Gaussian or Rectangular function"
                )
        # Numerical scalar product
        else:
            if isinstance(self, AnalyticSiegert):
                return self.conjugate().scal_prod(other, xlim=xlim)
            else:
                return super().scal_prod(other, xlim=xlim)

    def _scal_prod_with_Gaussian(self, gaussian):
        r"""
        Parameters
        ----------
        gaussian: Gaussian
            Gaussian function.

        Returns
        -------
        complex or float
            Value of the analytic scalar product of an eigenstate with a
            Gaussian test function.
        """
        # Useful values for readability
        k = self.wavenumber
        V0 = self.potential.depth
        l = self.potential.width
        qq = q(k, V0)
        xc = gaussian.center
        sigma = gaussian.sigma
        k0 = gaussian.momentum
        # Exponentials for regions I and III
        expkp = cmath.exp(1.0j * xc * (k0 + k) - (sigma * (k0 + k)) ** 2 / 2)
        expkm = cmath.exp(1.0j * xc * (k0 - k) - (sigma * (k0 - k)) ** 2 / 2)
        # Expontentials for region II
        expqp = cmath.exp(1.0j * xc * (k0 + qq) - (sigma * (k0 + qq)) ** 2 / 2)
        expqm = cmath.exp(1.0j * xc * (k0 - qq) - (sigma * (k0 - qq)) ** 2 / 2)
        # erf arguments for regions I and III
        zkpp = (xc + l / 2 + 1.0j * sigma ** 2 * (k0 + k)) / (sigma * math.sqrt(2))
        zkmp = (xc - l / 2 + 1.0j * sigma ** 2 * (k0 + k)) / (sigma * math.sqrt(2))
        zkpm = (xc + l / 2 + 1.0j * sigma ** 2 * (k0 - k)) / (sigma * math.sqrt(2))
        zkmm = (xc - l / 2 + 1.0j * sigma ** 2 * (k0 - k)) / (sigma * math.sqrt(2))
        # erf arguments for regions II
        zqpp = (xc + l / 2 + 1.0j * sigma ** 2 * (k0 + qq)) / (sigma * math.sqrt(2))
        zqmp = (xc - l / 2 + 1.0j * sigma ** 2 * (k0 + qq)) / (sigma * math.sqrt(2))
        zqpm = (xc + l / 2 + 1.0j * sigma ** 2 * (k0 - qq)) / (sigma * math.sqrt(2))
        zqmm = (xc - l / 2 + 1.0j * sigma ** 2 * (k0 - qq)) / (sigma * math.sqrt(2))
        # The value of the scalar product depends on the parity of the
        # continuum Siegert state.
        if self.is_odd:
            # If the Gaussian function is centered and if the continuum
            # state is odd, then the scalar product is equal to 0.0.
            if gaussian.is_even:
                return 0.0
            else:
                return self._sp_odd_gauss(
                    gaussian,
                    expkp,
                    expkm,
                    expqp,
                    expqm,
                    zkpp,
                    zkmp,
                    zkpm,
                    zkmm,
                    zqpp,
                    zqmp,
                    zqpm,
                    zqmm,
                )
        else:
            return self._sp_even_gauss(
                gaussian,
                expkp,
                expkm,
                expqp,
                expqm,
                zkpp,
                zkmp,
                zkpm,
                zkmm,
                zqpp,
                zqmp,
                zqpm,
                zqmm,
            )

    @abstractmethod
    def _sp_odd_gauss(
        self,
        gaussian,
        expkp,
        expkm,
        expqp,
        expqm,
        zkpp,
        zkmp,
        zkpm,
        zkmm,
        zqpp,
        zqmp,
        zqpm,
        zqmm,
    ):  # pragma: no cover
        pass

    @abstractmethod
    def _sp_even_gauss(
        self,
        gaussian,
        expkp,
        expkm,
        expqp,
        expqm,
        zkpp,
        zkmp,
        zkpm,
        zkmm,
        zqpp,
        zqmp,
        zqpm,
        zqmm,
    ):  # pragma: no cover
        pass

    def _scal_prod_with_Rectangular(self, rect):
        r"""
        Parameters
        ----------
        rect: Rectangular
            Rectangular function.

        Returns
        -------
        complex or float
            Value of the analytic scalar product of an eigenstate with a
            Rectangular test function.
        """
        # Split the rectangular function into three (one for each
        # region of space)
        r_1, r_2, r_3 = rect.split(self.potential)
        # Compute the scalar product accordingly
        scal_prod = 0.0j
        if r_1 is not None:
            scal_prod += self._sp_R_1(r_1)
        if r_2 is not None:
            scal_prod += self._sp_R_2(r_2)
        if r_3 is not None:
            scal_prod += self._sp_R_3(r_3)
        return scal_prod

    @abstractmethod
    def _sp_R_1(self, rect):  # pragma: no cover
        r"""
        .. note:: This is an asbtract method.

        Parameters
        ----------
        rect: Rectangular
            Rectangular function.

        Returns
        -------
        float or complex
            Value of the analytic scalar product between an eigenstate
            and a rectangular function spreading over region *I*.
        """
        pass

    def _sp_R_2(self, rect):
        r"""
        Parameters
        ----------
        rect: Rectangular
            Rectangular function.

        Returns
        -------
        float or complex
            Value of the analytic scalar product between an eigenstate
            and a rectangular function spreading over region *II*.
        """
        if self.is_odd and rect.is_even:
            return 0
        else:
            return self._sp_R_2_other_cases(rect)

    @abstractmethod
    def _sp_R_2_other_cases(self, rect):  # pragma: no cover
        r"""
        .. note:: This is an asbtract method.

        Parameters
        ----------
        rect: Rectangular
            Rectangular function.

        Returns
        -------
        complex or float
            Value of the analytic scalar product of an eigenstate with a
            rectangular test function in region *II* when the result is
            not obviously 0.
        """
        pass

    def _sp_R_3(self, rect):
        r"""
        Parameters
        ----------
        rect: Rectangular
            Rectangular function.

        Returns
        -------
        complex or float
            Value of the analytic scalar product of an eigenstate with a
            rectangular test function in region *III*.
        """
        # Given the parity of the states, the scalar product in region
        # III is closely related to the scalar product in region I.
        k0 = rect.momentum
        h = rect.amplitude
        sp = self._sp_R_1(Rectangular(-rect.xr, -rect.xl, k0=k0, h=h))
        if self.is_even:
            return sp
        else:
            return -sp


class SWPSiegert(SWPEigenstate, AnalyticSiegert):
    r"""
    This class defines a Siegert state in the case of the 1D SWP.
    """

    def __init__(self, ks, parity, potential, grid=None, analytic=True):
        r"""
        Parameters
        ----------
        ks: complex
            Wavenumber of the Siegert state.
        parity: str
            Parity of the Siegert state (``'e'`` for even, ``'o'`` for
            odd).
        potential: SWPotential
            1D Square-Well Potential giving rise to this eigenstate.
        grid: list or set or numpy array
            Discretization grid (optional).
        analytic: bool
            If ``True``, the scalar products must be computed
            analytically (default to ``True``).

        Raises
        ------
        ParityError
            If the parity is inconsistent with the Siegert state
            wavenumber.
        """
        # Check that the parity is consistent with the wavenumber
        if parity != find_parity(ks, potential):
            raise ParityError("Parity inconsistent with the wavenumber")
        # Set the _factor attribute (used for the discretization over a
        # grid)
        l = potential.width
        self._factor = cmath.sqrt(-1.0j * ks) / cmath.sqrt(1.0 - 1.0j * ks * l / 2)
        # Find the type of the Siegert state from its wavenumber
        super().__init__(ks, parity, potential, grid, analytic)

    def _even_wf_1(self, grid_1):
        r"""
        Evaluate the even Siegert state wavefunction over the grid
        points in region *I*.

        Parameters
        ----------
        grid_1: numpy array
            Grid in region *I*.

        Returns
        -------
        numpy array
            Even Siegert state wavefunction discretized over the grid in
            region *I*.
        """
        ks = self.wavenumber
        qs = q(ks, self.potential.depth)
        l = self.potential.width
        return (
            self._factor
            * cmath.cos(qs * l / 2)
            * (np.exp(-1.0j * ks * (grid_1 + l / 2)))
        )

    def _even_wf_2(self, grid_2):
        r"""
        Evaluate the even Siegert state wavefunction over the grid
        points in region *II*.

        Parameters
        ----------
        grid_2: numpy array
            Grid in region *II*.

        Returns
        -------
        numpy array
            Even Siegert state wavefunction discretized over the grid in
            region *II*.
        """
        qs = q(self.wavenumber, self.potential.depth)
        return self._factor * np.cos(qs * grid_2)

    def _odd_wf_1(self, grid_1):
        r"""
        Evaluate the odd Siegert state wavefunction over the grid points
        in region *I*.

        Parameters
        ----------
        grid_1: numpy array
            Grid in region *I*.

        Returns
        -------
        numpy array
            Odd Siegert state wavefunction discretized over the grid in
            region *I*.
        """
        ks = self.wavenumber
        qs = q(ks, self.potential.depth)
        l = self.potential.width
        return (
            -self._factor
            * cmath.sin(qs * l / 2)
            * (np.exp(-1.0j * ks * (grid_1 + l / 2)))
        )

    def _odd_wf_2(self, grid_2):
        r"""
        Evaluate the odd Siegert state wavefunction over the grid points
        in region *II*.

        Parameters
        ----------
        grid_2: numpy array
            Grid in region *II*.

        Returns
        -------
        numpy array
            Odd Siegert state wavefunction discretized over the grid in
            region *II*.
        """
        qs = q(self.wavenumber, self.potential.depth)
        return self._factor * np.sin(qs * grid_2)

    def _sp_even_gauss(
        self,
        gaussian,
        expkp,
        expkm,
        expqp,
        expqm,
        zkpp,
        zkmp,
        zkpm,
        zkmm,
        zqpp,
        zqmp,
        zqpm,
        zqmm,
    ):
        r"""
        Parameters
        ----------
        gaussian: Gaussian
            Gaussian function.

        Returns
        -------
        float or complex
            Analytical value of the c-product between an even Siegert
            state and a Gaussian test function.
        """
        # Useful values for readability
        ks = self.wavenumber
        qs = q(ks, self.potential.depth)
        l = self.potential.width
        sigma = gaussian.sigma
        h = gaussian.amplitude
        factor = cmath.sqrt(math.pi * ks / (2.0j + ks * l))
        factor1 = sigma * h * factor * cmath.exp(-1.0j * ks * l / 2)
        factor2 = sigma * h * factor / 2
        # Term for region I and III
        term1 = (
            factor1
            * cmath.cos(qs * l / 2)
            * (expkp * (erf(zkmp) + 1) - expkm * (erf(zkpm) - 1))
        )
        # Term for region II
        term2 = factor2 * (
            expqp * (erf(zqpp) - erf(zqmp)) - expqm * (erf(zqmm) - erf(zqpm))
        )
        return term1 + term2

    def _sp_odd_gauss(
        self,
        gaussian,
        expkp,
        expkm,
        expqp,
        expqm,
        zkpp,
        zkmp,
        zkpm,
        zkmm,
        zqpp,
        zqmp,
        zqpm,
        zqmm,
    ):
        r"""
        Parameters
        ----------
        gaussian: Gaussian
            Gaussian function.

        Returns
        -------
        float or complex
            Analytical value of the c-product between an odd Siegert
            state and a Gaussian test function.
        """
        # Useful values for readability
        ks = self.wavenumber
        qs = q(ks, self.potential.depth)
        l = self.potential.width
        sigma = gaussian.sigma
        h = gaussian.amplitude
        factor = cmath.sqrt(cmath.pi * ks / (2.0j + ks * l))
        factor1 = sigma * h * factor * cmath.exp(-1.0j * ks * l / 2)
        factor2 = sigma * h * factor / 2
        # Term for region I and III
        term1 = (
            factor1
            * cmath.sin(qs * l / 2)
            * (expkp * (erf(zkmp) + 1) + expkm * (erf(zkpm) - 1))
        )
        # Term for region II
        term2 = (
            -1.0j
            * factor2
            * (expqp * (erf(zqpp) - erf(zqmp)) + expqm * (erf(zqmm) - erf(zqpm)))
        )
        return term1 + term2

    def _sp_R_1(self, rect):
        r"""
        Parameters
        ----------
        rect: Rectangular
            Rectangular function.

        Returns
        -------
        complex or float
            Value of the analytic scalar product of a Siegert state with
            a rectangular function spreading over region *I*.
        """
        # Initial variables
        ks = self.wavenumber
        qs = q(ks, self.potential.depth)
        l = self.potential.width
        h = rect.amplitude
        k0 = rect.momentum
        xc = rect.center
        a = rect.width
        factor1 = 2.0 * h * cmath.exp(-1.0j * ks * l / 2)
        dk = ks - k0
        factor2 = cmath.exp(-1.0j * dk * xc) * cmath.sin(dk * a / 2) / dk
        norm = cmath.sqrt(ks / (1.0j + ks * l / 2))
        # Compute the scalar product depending on the parity of the
        # Siegert state.
        if self.is_odd:
            parity_term = -cmath.sin(qs * l / 2)
        else:
            parity_term = cmath.cos(qs * l / 2)
        scal_prod = norm * factor1 * factor2 * parity_term
        return scal_prod

    def _sp_R_2_other_cases(self, rect):
        r"""
        Parameters
        ----------
        rect: Rectangular
            Rectangular function.

        Returns
        -------
        complex or float
            Value of the analytic scalar product of a Siegert state with
            a rectangular function spreading over region *II*.
        """
        # Initial variables
        ks = self.wavenumber
        qs = q(ks, self.potential.depth)
        l = self.potential.width
        a = rect.width
        xc = rect.center
        h = rect.amplitude
        k0 = rect.momentum
        factor = cmath.sqrt(ks / (1.0j + ks * l / 2))
        dkp = qs + k0
        dkm = qs - k0
        # Compute the scalar product depending on the parity of the
        # Siegert state.
        term1 = cmath.sin(dkp * a / 2) / dkp * cmath.exp(+1.0j * dkp * xc)
        term2 = cmath.sin(dkm * a / 2) / dkm * cmath.exp(-1.0j * dkm * xc)
        if self.is_even:
            return h * factor * (term1 + term2)
        else:
            return -1.0j * h * factor * (term1 - term2)

    def MLE_strength_function(self, test, kgrid):
        r"""
        Evaluate the contribution of a Siegert state to the
        Mittag-Leffler expansion of the strength function, for a given
        test function discretized on a grid of wavenumbers ``kgrid``.

        Parameters
        ----------
        test: Wavefunction
            Test function.
        kgrid: numpy array
            Discretization grid of the wavenumber.

        Returns
        -------
        numpy array
            Contribution of the Siegert state to the strength function.
        """
        # Set initial variables
        ks = self.wavenumber
        sp = self.scal_prod(test)
        # Evaluate the strength function and return its imaginary part.
        strength_func = -1.0 / math.pi * sp ** 2 / (ks * (kgrid - ks))
        return strength_func.imag


class SWPContinuum(SWPEigenstate, AnalyticContinuum):
    r"""
    Class defining a continuum state of the 1D Square-Well potential.
    """

    def __init__(self, k, parity, potential, grid=None, analytic=True):
        r"""
        Parameters
        ----------
        k: complex
            Wavenumber of the continuum state.
        parity: str
            Parity of the continuum state (``'e'`` for even, ``'o'`` for
            odd).
        potential: SWPotential
            1D Square-Well Potential giving rise to this eigenstate.
        grid: list or set or numpy array
            Discretization grid (optional).
        analytic: bool
            If ``True``, the scalar products must be computed
            analytically (default to ``True``).

        Raises
        ------
        ParityError
            If the parity is not ``'e'`` (even) or ``'o'`` (odd).
        """
        # Check the parity
        if parity not in self.PARITIES:
            raise ParityError("The value of parity must be 'e' or 'o'.")
        # Initialize the values of the Jost functions of the same
        # parity as the state and of a factor, to be used later when
        # evaluating the wavefunction or some analytical scalar products
        l = potential.width
        qq = q(k, potential.depth)
        if parity == "e":
            self._jostm = fem(k, l, k2=qq)
            self._jostp = fep(k, l, k2=qq)
        if parity == "o":
            self._jostm = fom(k, l, k2=qq)
            self._jostp = fop(k, l, k2=qq)
        self._factor = 1.0 / (2.0 * math.sqrt(math.pi) * self._jostp)
        # Initialize all the other attributes
        super().__init__(k, parity, potential, grid, analytic)

    def _even_wf_1(self, grid_1):
        r"""
        Evaluate the even eigenstate wavefunction over the grid points
        in region *I*.

        Parameters
        ----------
        grid_1: numpy array
            Grid in region *I*.

        Returns
        -------
        numpy array
            Even eigenstate wavefunction discretized over the grid in
            region *I*.
        """
        k = self.wavenumber
        return self._factor * (
            self._jostp * np.exp(1.0j * k * grid_1)
            + self._jostm * np.exp(-1.0j * k * grid_1)
        )

    def _even_wf_2(self, grid_2):
        r"""
        Evaluate the even Siegert state wavefunction over the grid
        points in region *II*.

        Parameters
        ----------
        grid_2: numpy array
            Grid in region *II*.

        Returns
        -------
        numpy array
            Even Siegert state wavefunction discretized over the grid in
            region *II*.
        """
        qq = q(self.wavenumber, self.potential.depth)
        return self._factor * np.cos(qq * grid_2)

    def _odd_wf_1(self, grid_1):
        r"""
        Evaluate the odd eigenstate wavefunction over the grid points
        in region *I*.

        Parameters
        ----------
        grid_1: numpy array
            Grid in region *I*.

        Returns
        -------
        numpy array
            Odd eigenstate wavefunction discretized over the grid in
            region *I*.
        """
        # This can be done, because parity is taken care of in __init__
        # for this particular case (only calls to attributes _jostm,
        # _jostp and _factors in _even_wf_1)
        return self._even_wf_1(grid_1)

    def _odd_wf_2(self, grid_2):
        r"""
        Evaluate the odd Siegert state wavefunction over the grid
        points in region *II*.

        Parameters
        ----------
        grid_2: numpy array
            Grid in region *II*.

        Returns
        -------
        numpy array
            Odd Siegert state wavefunction discretized over the grid in
            region *II*.
        """
        qq = q(self.wavenumber, self.potential.depth)
        return -self._factor * np.sin(qq * grid_2)

    def _sp_even_gauss(
        self,
        gaussian,
        expkp,
        expkm,
        expqp,
        expqm,
        zkpp,
        zkmp,
        zkpm,
        zkmm,
        zqpp,
        zqmp,
        zqpm,
        zqmm,
    ):
        r"""
        Parameters
        ----------
        gaussian: Gaussian
            Gaussian function.

        Returns
        -------
        float or complex
            Analytical value of the c-product between an even continuum
            state and a Gaussian test function.
        """
        # Useful values for readability
        k = self.wavenumber
        qq = q(k, self.potential.depth)
        l = self.potential.width
        fp = fep(k, l, k2=qq)
        fm = fem(k, l, k2=qq)
        sigma = gaussian.sigma
        h = gaussian.amplitude
        factor = sigma * h / (2 * math.sqrt(2.0))
        factor1 = -1.0 * factor
        factor2 = factor / 2.0
        # Scalar product evaluation:
        # - in region I and III
        termk1 = expkp * (fp * (erf(zkpp) - 1.0) - fm * (erf(zkmp) + 1.0))
        termk2 = expkm * (fm * (erf(zkpm) - 1.0) - fp * (erf(zkmm) + 1.0))
        term1 = factor1 * (termk1 + termk2)
        # - in region II
        termq1 = expqp * (erf(zqpp) - erf(zqmp))
        termq2 = expqm * (erf(zqpm) - erf(zqmm))
        term2 = factor2 * (termq1 + termq2)
        return (term1 + term2) / fm

    def _sp_odd_gauss(
        self,
        gaussian,
        expkp,
        expkm,
        expqp,
        expqm,
        zkpp,
        zkmp,
        zkpm,
        zkmm,
        zqpp,
        zqmp,
        zqpm,
        zqmm,
    ):
        r"""
        Parameters
        ----------
        gaussian: Gaussian
            Gaussian function.

        Returns
        -------
        float or complex
            Analytical value of the c-product between an odd continuum
            state and a Gaussian test function.
        """
        # Useful values for readability
        k = self.wavenumber
        qq = q(k, self.potential.depth)
        l = self.potential.width
        fp = fop(k, l, k2=qq)
        fm = fom(k, l, k2=qq)
        sigma = gaussian.sigma
        h = gaussian.amplitude
        factor = sigma * h / (2 * math.sqrt(2.0))
        factor1 = -1.0 * factor
        factor2 = factor / 2.0
        # Scalar product evaluation:
        # - in region I and III
        termk1 = expkp * (fp * (erf(zkpp) - 1.0) + fm * (erf(zkmp) + 1.0))
        termk2 = expkm * (fm * (erf(zkpm) - 1.0) + fp * (erf(zkmm) + 1.0))
        term1 = factor1 * (termk1 + termk2)
        # - in region II
        termq1 = expqp * (erf(zqpp) - erf(zqmp))
        termq2 = expqm * (erf(zqmm) - erf(zqpm))
        term2 = 1.0j * factor2 * (termq1 + termq2)
        return (term1 + term2) / fm

    def _sp_R_1(self, rect):
        r"""
        Parameters
        ----------
        rect: Rectangular
            Rectangular function.

        Returns
        -------
        complex or float
            Value of the analytic scalar product of a continuum state
            with a Rectangular function spreading over region *I*.
        """
        # Initial variables
        k = self.wavenumber
        xc = rect.center
        a = rect.width
        h = rect.amplitude
        k0 = rect.momentum
        dkp = k + k0
        dkm = k - k0
        # Initial variables depending on the parity of the state
        jostp = self._jostp
        jostm = self._jostm
        # Scalar product in the general case
        if k0 not in [k, -k]:
            factor = h / (math.sqrt(math.pi) * jostm)
            term1 = jostp * cmath.exp(1.0j * dkp * xc) * math.sin(dkp * a / 2) / dkp
            term2 = jostm * cmath.exp(-1.0j * dkm * xc) * math.sin(dkm * a / 2) / dkm
            return factor * (term1 + term2)
        # Scalar product if the initial momentum would lead to a
        # division by zero in the general case
        else:
            factor = h / (2 * math.sqrt(math.pi))
            if k0 == k:
                sign = 1.0
                f1 = 1.0
                f2 = jostp / jostm
            else:
                sign = -1.0
                f1 = jostp / jostm
                f2 = 1.0
            return factor * (
                a * f1 + cmath.exp(sign * 2.0j * k * xc) * math.sin(k * a) / k * f2
            )

    def _sp_R_2_other_cases(self, rect):
        r"""
        Parameters
        ----------
        rect: Rectangular
            Rectangular function.

        Returns
        -------
        complex or float
            Value of the analytic scalar product of an even continuum
            state with a Rectangular function spreading over region
            *II*.
        """
        # Useful values for readability
        k = self.wavenumber
        qq = q(k, self.potential.depth)
        a = rect.width
        xc = rect.center
        h = rect.amplitude
        k0 = rect.momentum
        if self.is_even:
            jost = fem(k, self.potential.width, k2=qq)
            psign = +1.0
            im = 1.0
        else:
            jost = fom(k, self.potential.width, k2=qq)
            psign = -1.0
            im = 1.0j
        # Scalar product in the general case
        if k0 not in [qq, -qq]:
            factor = h / (2 * im * math.sqrt(math.pi) * jost)
            dkp = qq + k0
            dkm = qq - k0
            term1 = math.sin(dkp * a / 2) / dkp * cmath.exp(+1.0j * dkp * xc)
            term2 = math.sin(dkm * a / 2) / dkm * cmath.exp(-1.0j * dkm * xc)
            return psign * factor * (term1 + psign * term2)
        # Scalar product if the initial momentum would lead to a
        # division by zero in the general case
        else:
            factor = psign * h / (4 * im * math.sqrt(math.pi))
            term = math.sin(qq * a) / qq * cmath.exp(2.0j * qq * xc) + psign * a
            if k0 == qq:
                return factor * term / jost
            else:
                return np.conjugate(factor * term) / jost


class ParityError(Exception):
    r"""
    Error thrown if the parity of an eigenstate is incorrect.
    """
    pass
