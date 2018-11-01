# -*- coding: utf-8 -*-
r"""
The :class:`SWPBasisSet` class, representing a basis set made of
eigenstates of a 1D Square-Well Potential (SWP), is defined below.
"""

import numpy as np
from mpmath import findroot
from siegpy import SWPSiegert, SWPContinuum, SWPotential  # noqa
from .swputils import dep, dop
from .swpeigenstates import SWPEigenstate
from .basisset import BasisSetError
from .analyticbasisset import AnalyticBasisSet, _continuum_contributions_to_CR
from .analyticeigenstates import WavenumberError


__all__ = ["SWPBasisSet"]


class SWPBasisSet(AnalyticBasisSet):
    r"""
    Class representing a basis set in the case of 1D Square Well
    Potential. It mostly implements the abstract methods of its parent
    class.
    """

    def __init__(self, states=None):
        r"""
        Parameters
        ----------
        states: list of SWPEigenstate instances
            Eigenstates of a 1D SW potential.

        Raises
        ------
        ValueError
            If all states are not
            :class:`~siegpy.swpeigenstates.SWPEigenstate` instances.


        Example

        A basis set is initialized by a list of states:

        >>> pot = SWPotential(4.442882938158366, 10)
        >>> bnd1 = SWPSiegert(4.42578048382546j, 'e', pot)
        >>> bnd2 = SWPSiegert(4.284084610255061j, 'o', pot)
        >>> bs = SWPBasisSet(states=[bnd1, bnd2])
        >>> assert bs[0] == bnd1
        >>> assert bs[1] == bnd2

        The basis set can be empty:

        >>> bs = SWPBasisSet()
        >>> assert bs.is_empty
        """
        if states is not None:
            # Check that all states are SWPEigenstates
            if not all([isinstance(s, SWPEigenstate) for s in states]):
                raise ValueError(
                    "All the states states must be SWPEigenstate instances")
        # Use the parent class to initialize the basis set
        super().__init__(states=states)

    @classmethod
    def find_Siegert_states(cls, pot, re_kmax, re_hk, im_kmax, im_hk=None,
                            analytic=True, grid=None, bounds_only=False):
        r"""
        The Siegert states wavenumbers are found using the Muller
        scheme of the `mpmath findroot method`_. This allows one to find
        a complex root of a function, starting from a wavenumber as
        input guess.

        To find the Siegert states in a given portion of the
        wavenumber complex plane, a grid of input guess wavenumbers is
        therefore required. The parameters specifying this grid are
        listed below:

        Parameters
        ----------
        pot: SWPotential
            1D Square-Well Potential for which we look for the continuum
            states.
        re_kmax: float
            Maximal value for the real part of the resonant states.
        re_hk: float
            Real part of the grid step for initial roots.
        im_kmax: float
            (Absolute) maximal value for the imaginary part of the
            resonant and anti-resonant states.
        im_hk: float
            Imaginary part of the grid step for the initial roots
            (optional, except in the cases where the imaginary part of
            the resonant states becomes bigger (in absolute value) than
            the imaginary part of the first bound state).
        analytic: bool
            If ``True``, scalar products with the Siegert states will be
            computed analytically (default to ``True``).
        grid: numpy array or list or set
            Discretization grid of the wavefunctions of the Siegert
            states (optional).
        bounds_only: bool
            If ``True``, only the bounds states have to be found
            (default to ``False``).

        Returns
        -------
        SWPBasisSet
            Sorted basis set with all the Siegerts found in the
            user-defined range.


        Examples

        Read a basis set from a file as a reference:

        >>> from siegpy.swpbasisset import SWPBasisSet
        >>> bs_1 = SWPBasisSet.from_file("doc/notebooks/siegerts.dat", nres=3)
        >>> len(bs_1.resonants)
        3

        To find the Siegert states of a given potential, proceed as
        follows:

        >>> pot = bs_1.potential
        >>> bs = SWPBasisSet.find_Siegert_states(pot, 4.5, 1.5, 1.0, im_hk=1.0)
        >>> bs == bs_1
        True

        The previous test shows that all the Siegert states of the
        reference are recovered, and this includes the resonant states
        whose wavenumber can have a real part up to 4.5 and an
        imaginary part as low as -1.0.

        .. warning::

            It is not ensured that all Siegert states in the defined
            range are found: you may want to check the grid_step
            values.

        For instance, if the grid step along the real wavenumber axis
        is too large, the reference results are not recovered:

        >>> bs = SWPBasisSet.find_Siegert_states(pot, 4.5, 4.5, 1.0, im_hk=1.0)
        >>> bs == bs_1
        False

        .. _`mpmath findroot method`: https://bit.ly/2BSVCz8
        """
        # Define the functions to be passed to findroot
        dep_k, dop_k = _get_wrapper_functions(pot)
        # We first look for the wavenumbers of the bound and
        # anti-bound states along the imaginary axis of wavenumbers.
        kgrid = _bnd_abnd_input_guess(pot, im_hk, bounds_only)
        wavenumbers, parities = _find_bnd_abnd_states(
            kgrid, dep_k, dop_k, pot.depth, bounds_only)
        # If required, we search for the wavenumbers of the resonant and
        # anti-resonant states.
        if not bounds_only:
            kgrid = _res_ares_input_guess(re_kmax, re_hk, im_kmax, im_hk)
            wavenumbers, parities = _find_res_ares_states(
                kgrid, dep_k, dop_k, wavenumbers, parities)
        # The basis set made of the Siegert states just found is created
        siegerts = [SWPSiegert(k_s, parity, pot, grid=grid, analytic=analytic)
                    for k_s, parity in zip(wavenumbers, parities)]
        return cls(states=siegerts)

    @classmethod
    def find_continuum_states(cls, pot, kmax, hk, kmin=None, even_only=False,
                              analytic=True, grid=None):
        r"""
        Initialize a BasisSet instance made of SWPContinuum instances.
        The basis set has :math:`2*n_k` elements if ``even_only=False``,
        :math:`n_k` elements otherwise (where :math:`n_k` is the number
        of continuum states defined by the grid step ``hk`` and the
        minimal and maximal values of the wavenumber grid ``kmin`` and
        ``kmax``).

        Parameters
        ----------
        pot: SWPotential
            1D Square-Well Potential for which we look for the continuum
            states.
        kmax: float
            Wavenumber of the last continuum state.
        hk: float
            Grid step of the wavenumber grid.
        kmin: float
            Wavenumber of the first continuum state (optional)
        even_only: bool
            If ``True``, only even continuum states are created (default
            to ``False``)
        analytic: bool
            If ``True``, the scalar products will be computed
            analytically (default to ``True``).
        grid: numpy array or list or set
            Discretization grid of the wavefunctions of the continuum
            states (optional).

        Returns
        -------
        SWPBasisSet
            Basis set of all continuum states defined by the grid of
            wavenumbers.

        Raises
        ------
        WavenumberError
            If ``hk``, ``kmin`` or ```kmax`` is not strictly positive.


        Examples

        Let us start by defining a potential:

        >>> from siegpy.swpbasisset import SWPBasisSet
        >>> bs_ref = SWPBasisSet.from_file("doc/notebooks/siegerts.dat")
        >>> pot = bs_ref.potential

        The continuum states are found, given a potential and a grid
        of initial wavenumbers (note that the minimal and maximal
        wavenumber cannot be 0)

        >>> hk = 1; kmax = 3
        >>> bs = SWPBasisSet.find_continuum_states(pot, kmax, hk)
        >>> bs.wavenumbers
        [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]

        It is possible to find only the even continuum states:

        >>> p = pot
        >>> bs = SWPBasisSet.find_continuum_states(p, kmax, hk, even_only=True)
        >>> bs.wavenumbers
        [1.0, 2.0, 3.0]
        >>> assert len(bs.even) == 3 and bs.odd.is_empty

        The minimal wavenumber can set:

        >>> bs = SWPBasisSet.find_continuum_states(pot, kmax, hk, kmin=3)
        >>> bs.wavenumbers
        [3.0, 3.0]
        """
        # Check the given values
        if kmax <= 0.0:
            raise WavenumberError(
                "The maximal wavenumber must be strictly positive.")
        if hk <= 0.0:
            raise WavenumberError(
                "The wavenumber grid step must be striclty positive")
        if kmin is None:
            kmin = hk
        elif kmin <= 0.0:
            raise WavenumberError(
                "The minimal wavenumber must be strictly positive.")
        # Initialize the grid of wavenumbers
        kgrid = np.arange(kmin, kmax+hk/2, hk)
        # Initialize the basis set with the even states
        cont = [SWPContinuum(k, 'e', pot, grid=grid, analytic=analytic)
                for k in kgrid]
        # Add the odd continuum states to the basis set, if required
        if not even_only:
            cont += [SWPContinuum(k, 'o', pot, grid=grid, analytic=analytic)
                     for k in kgrid]
        # Return a basis set made of the continuum states
        return cls(states=cont)

    @property
    def parity(self):
        r"""
        Returns
        -------
        None or str
            ``None`` if both parities are present, ``'e'`` or ``'o'`` if
            all the states are even or odd, respectively.
        """
        if all([state.is_even for state in self]) or \
           all([state.is_odd for state in self]):
            return self[0].parity
        else:
            return None

    def __add__(self, states):
        r"""
        Add a list of states or the states of another SWPBasisSet
        instance to the current SWPBasisSet instance.

        Parameters
        ----------
        states: list
            Eigenstates of a Hamiltonian.

        Returns
        -------
            A new basis set.
        SWPBasisSet

        Raises
        ------
        TypeError
            If the added states are not :class:`SWPEigenstate`
            instances.


        Examples

        >>> pot = SWPotential(4.442882938158366, 10)
        >>> bnd1 = SWPSiegert(4.42578048382546j, 'e', pot)
        >>> bnd2 = SWPSiegert(4.284084610255061j, 'o', pot)
        >>> bs1 = SWPBasisSet(states=[bnd1])
        >>> bs1 += bnd2
        >>> bs2 = SWPBasisSet() + bnd1 + bnd2
        >>> assert bs1 == bs2
        """
        # If states is a basis set, then convert it to a list of states
        if isinstance(states, SWPBasisSet):
            states = states.states
        # If states actually is a single state, make it iterable.
        elif not isinstance(states, list):
            states = [states]
        # Make sure that each state is a Function
        if not all([isinstance(s, SWPEigenstate) for s in states]):
            raise TypeError("Only wavefunctions of the SWP case can be added "
                            "to a SWPBasisSet")
        # Return a new basis set
        return SWPBasisSet(states=self.states+states)

    @property
    def even(self):
        r"""
        Returns
        -------
        SWPBasisSet
            All the even states of the basis set.
        """
        return SWPBasisSet(states=[state for state in self if state.is_even])

    @property
    def odd(self):
        r"""
        Returns
        -------
        SWPBasisSet
            All the odd states of the basis set.
        """
        return SWPBasisSet(states=[state for state in self if state.is_odd])

    def plot_wavefunctions(self, nres=None, xlim=None, ylim=None, title=None,
                           file_save=None):  # pragma: no cover
        r"""
        Plot the bound, resonant and anti-resonant wavefunctions of the
        basis set along with the potential. The continuum and anti-bound
        states, if any are present in the basis set, are not plotted.

        The wavefunctions are translated along the y-axis by their
        energy (for bound states) or absolute value of their energy (for
        resonant and anti-resonant states).

        Parameters
        ----------
        nres: int
            Number of resonant and antiresonant wavefunctions to plot.
        xlim: tuple(float or int, float or int)
            Range of the x axis of the plot (optional)
        ylim: tuple(float or int, float or int)
            Range of the y axis of the plot (optional)
        title: str
            Plot title (optional).
        file_save: str
            Filename of the plot to be saved (optional)
        """
        # Set the range of the plot on the x-axis (xlim)
        if xlim is None:
            xlim = (-self.potential.width, self.potential.width)
        # Set the range of the plot on the y-axis (ylim)
        if ylim is None:
            ymin = - 1.1 * self.potential.depth
            # Define the maximum of the y-axis
            if nres == 0:
                ymax = abs(ymin) / 10
            else:
                ymax = abs(self.resonants[:nres][-1].energy)+2
            ylim = (ymin, ymax)
        super().plot_wavefunctions(nres=nres, xlim=xlim, ylim=ylim,
                                   title=title, file_save=file_save)

    def continuum_contributions_to_CR(self, test, hk=None, kmax=None):
        r"""
        Evaluate the continuum contributions to the completeness
        relation.

        This is an overriding of the inherited
        :meth:`siegpy.analyticbasisset.AnalyticBasisSet.continuum_contributions_to_CR`
        method in order to account for the parity of the continuum
        states of the Square-Well potential.

        Parameters
        ----------
        test: Function
            Test function.
        hk: float
            Grid step for the wavenumbers of the "on-the-fly" continuum
            basis sets (optional).
        kmax: float
            Maximal wavenumber of the on-the-fly continuum basis set
            (optional).

        Returns
        -------
        numpy array
            Contribution of each continuum state of the basis set to the
            exact completeness relation.

        Raises
        ------
        BasisSetError
            If the basis set has less odd continuum states than even
            continuum states when the test function is not even.
        """
        # Boolean defining if the continuum has to be defined
        # on-the-fly, given the input parameters
        OTF = (hk is not None) and (kmax is not None)
        # 1- Get the continuum states
        if not OTF:
            # Use the continuum states in the basis set
            even_cont = self.continuum.even
            odd_cont = self.continuum.odd
            if (len(odd_cont) != len(even_cont)) and not test.is_even:
                raise BasisSetError(
                    "Not enough odd continuum states in the basis set, since"
                    "the test function is not even.")
        else:
            # Define the continuum states on-the-fly
            pot = self.potential
            cont = SWPBasisSet.find_continuum_states(pot, kmax, hk,
                                                     even_only=test.is_even)
            even_cont = cont.even
            odd_cont = cont.odd
        # 2- Sum the contribution of even and odd continuum states
        #    having the same wavenumbers
        k_grid, cont_contribs = \
            _continuum_contributions_to_CR(even_cont, test)
        if not test.is_even:
            k_grid, odd_contribs = \
                _continuum_contributions_to_CR(odd_cont, test)
            cont_contribs += odd_contribs
        return k_grid, cont_contribs

    @staticmethod
    def _evaluate_integrand(q, k, test, eta, potential):
        r"""
        Evaluate the integrand used to compute the strength function
        "on-the-fly."

        Parameters
        ----------
        q: float
            Wavenumber of the continuum state considered.
        k: float
            Wavenumber for which the strength function is evaluated.
        test: Function
            Test function.
        eta: float
            Infinitesimal for integration (if ``None``, default to 10
            times the value of the grid-step of the continuum basis
            set).
        potential: Potential
            Potential of the currently studied analytical case.

        Returns
        -------
        complex
            Value of the integrand.
        """
        # Even states contribution
        c_p = SWPContinuum(q, 'e', potential)
        sp_p = c_p.scal_prod(test)
        # Odd states contribution
        if not test.is_even:
            c_m = SWPContinuum(q, 'o', potential)
            sp_m = c_m.scal_prod(test)
        else:
            sp_m = 0.
        # Add both contributions
        cont_contrib = abs(sp_p)**2 + abs(sp_m)**2
        # Return the evaluation of the integrand
        return eta * cont_contrib / ((q**2/2. - k**2/2.)**2 + eta**2)

    def _propagate(self, test, time_grid, weights=None):
        r"""
        Evaluate the time-propagation of a test wavepacket as the matrix
        product of two matrices: one to account for the time dependance
        of the propagation of the wavepacket (mat_time), the other for
        its space dependance (mat_space).

        This is an overriding of the inherited
        :meth:`siegpy.basisset.BasisSet._propagate` method to take into
        account the parity of the eigenstates in the 1D SWP case.

        Parameters
        ----------
        test: Function
            Test function.
        time_grid: numpy array or list of positive numbers
            Times for which the propagation is evaluated. It must
            contain positive numbers only.
        weights: dict
            Dictionary of the weights to use for the time-propagation.
            Keys correspond to a type of Siegert states ('ab' for
            anti-bounds, 'b' for bounds, 'r' for resonants and 'ar' for
            anti-resonants) and the corresponding value is the weight to
            use for all the states of the given type (optional).

        Returns
        -------
        2D numpy array
            Propagated wavepacket for the different times of
            ``time_grid``.

        Raises
        ------
        BasisSetError
            If the basis set has less odd continuum states than even
            continuum states when the test function is not even.
        """
        # Check that there is the same number of even and odd continuum
        # states in the basis set if the test function is not even.
        # If not, some (a priori non-negligible) contributions to the
        # propagation would be missing.
        if (not test.is_even) and (len(self.continuum.even) >
                                   len(self.continuum.odd)):
            raise BasisSetError(
                "Not enough odd continuum states in the basis set, "
                "since the test function is not even.")
        # Compute the even states contributions and then add the odd
        # states contribution if needed
        time_prop = super(SWPBasisSet, self.even)._propagate(
            test, time_grid, weights=weights)
        if not test.is_even:
            time_prop += super(SWPBasisSet, self.odd)._propagate(
                test, time_grid, weights=weights)
        return time_prop

    def _add_one_continuum_state(self):
        r"""
        Add two continuum state to the basis set, depending on the
        already existing continuum states.

        This is an overriding of the inherited
        :meth:`siegpy.analyticbasisset.AnalyticBasisSet._add_one_continuum_state`
        method to account for the parity of the continuum states in the
        1D SWP case.


        Returns
        -------
        SWPBasisSet
            The same basis set with one more continuum state.
        """
        cont = self.continuum
        kmax = cont[-1].wavenumber
        hk = cont[1].wavenumber - cont[0].wavenumber
        # Add a continuum state to the basis set
        self += SWPContinuum(kmax+hk, cont.parity, self.potential,
                             grid=self.grid, analytic=self.analytic)
        return self


def _get_wrapper_functions(pot):
    r"""
    Wrappers for the functions "nullified" by the application of
    the Siegert boundary conditions. It makes them one-parameter
    functions suitable for findroot (from mpmath).
    findroot is self-consistent and returns a different type of
    complex numbers than numpy, thus requiring a conversion to
    a numpy-compatible format by using the "complex" method.

    Parameters
    ----------
    pot: SWPotential
        Potential considered

    Returns
    -------
    tuple
        Two functions.
    """
    l = pot.width
    V0 = pot.depth

    def dep_k(k):
        r"""
        Wraps one of the four delta functions of the 1DSWP case.
        See :function:`oneDSWP_utils.dep` for details.
        """
        return dep(complex(k), l, V0=V0)

    def dop_k(k):
        r"""
        Wraps one of the four delta functions of the 1DSWP case.
        See :function:`oneDSWP_utils.dop` for details.
        """
        return dop(complex(k), l, V0=V0)

    return dep_k, dop_k


def _bnd_abnd_input_guess(pot, im_hk, bounds_only):
    r"""
    Define the grid of input guesses passed to the findroot method of
    the mpmath package, that allows to find the bound and possibly the
    anti-boud states (if ``bounds_only`` is set to ``False``).

    This grid is defined via the following points:
    1- The energy of the bound states must be between 0 and the
      bottom of the potential, -V0.
    2- The same is true of the antibound states.
    3- The wavenumber of the bound states has a positive
       imaginary part (negative for the anti-bound states).
       Both types have purely imaginary wavenumbers.
    4- The number of bound states can be inferred from the
       properties of the potential (width and depth).
    5- There is at least one more bound states than anti-bound
       states.

    Parameters
    ----------
    pot: SWPotential
        SWP considered
    hk: float
        Wavenumber grid step between to points of the input guess grid.
    bounds_only: bool
        If ``True``, only the bounds states have to be found (default to
        ``False``).

    Returns
    -------
    numpy array
        Grid of input guesses in order to find the bound and possibly
        the anti-bound states.
    """
    # - grid extension (from points 1-3):
    V0 = pot.depth
    im_max = np.sqrt(2*V0)
    if bounds_only:
        im_min = 0.0
    else:
        im_min = -im_max
    # - grid step (from points 4 and 5):
    l = pot.width
    n_bounds = int(l/np.pi*np.sqrt(2.*V0) + 1)
    step = float(im_max / (n_bounds*3))
    if im_hk is not None:
        step = min(step, im_hk)
    # Return the whole grid of input guesses
    return 1.j * np.arange(im_min, im_max, step)


def _find_bnd_abnd_states(kgrid, dep_k, dop_k, V0, bounds_only):
    r"""
    The wavenumbers of the bound and anti-bound states (if
    ``bounds_only = False``) are searched along a finite portion of the
    imaginary axis.

    Parameters
    ----------
    kgrid: numpy array
        Grid of input guesses.
    dep_k: function
        Function whose zeros are even Siegert states wavenumbers.
    dop_k: function
        Function whose zeros are odd Siegert states wavenumbers.
    V0: float
        Depth of the potential
    bounds_only: bool
        If ``True``, only the bounds states have to be found (default to
        ``False``).

    Returns
    -------
    tuple(list of complex, list of str)
        List of wavenumbers found along with the parity of the state.
    """
    # Loop over two functions whose zeros are the Siegert states
    # wavenumbers k_s
    wavenumbers = []
    parities = []
    # A numerical tolerance ensures that the solutions are close
    # enough to the imaginary axis to be bound or antibound states.
    # It also defines how close two Siegert states wavenumbers can
    # be along the imaginary axis.
    tol = 10**(-14)
    for func, parity in ((dep_k, 'e'), (dop_k, 'o')):
        bnds_abnds = _bnd_abnd_wavenumbers(kgrid, func, tol, V0, bounds_only)
        for k_s in bnds_abnds:
            # Check that the solution was not already known
            if not np.any(np.isclose(k_s, wavenumbers, rtol=tol)):
                # Make it purely imaginary
                k_s = np.imag(k_s)*1.j
                # Add it to the list of wavenumbers
                wavenumbers.append(k_s)
                parities.append(parity)
    return wavenumbers, parities


def _bnd_abnd_wavenumbers(kgrid, func, tol, V0, bounds_only):
    r"""
    Generator of solutions of the Jost delta functions along the
    imaginary axis. A solution of these functions is generated for
    each point of the wavenumber grid, given that it has no real part.

    If ``bounds_only`` is set to ``True``, only the bound states are
    returned. findroot may also give artificial solutions whose
    wavenumbers correspond to the bottom of the 1D SWP: they are
    discarded.

    Parameters
    ----------
    kgrid: numpy array
        Grid of input guesses
    func: function
        Function whose zeros are Siegert states wavenumbers.
    tol: float
        Tolerance on the value of the real part of the wavenumbers
        found.
    V0: float
        Potential depth.
    bounds_only: bool
        If ``True``, only the bounds states have to be found (default to
        ``False``).

    Yields
    ------
    complex
        Valid bound or anti-bound wavenumber.
    """
    for k_0 in kgrid:
        step = kgrid[1] - kgrid[0]
        # A solution is searched around each initial
        # wavenumber k0 in an initial interval [k_dn, k_up]
        k_dn = k_0 - step/2.
        k_up = k_0 + step/2.
        # Try to find a solution k_s from this starting point
        try:
            input_guess = [k_0, k_dn, k_up]
            k_s = findroot(func, input_guess, solver='muller', tol=1.e-17)
            # Do not yield a wavenumber if the solution is not
            # valid for any reason.
            not_bound = np.imag(k_s) <= 0
            has_real_part = abs(np.real(k_s)) > tol
            is_potential_minimum = k_s**2/2. == -complex(V0)  # artifact
            if not ((bounds_only and not_bound) or
                    has_real_part or is_potential_minimum):
                yield complex(k_s)
        # If findroot does not converge, it throws a ValueError.
        # Catch it and do nothing.
        except ValueError:
            pass


def _res_ares_input_guess(re_kmax, re_hk, im_kmax, im_hk):
    r"""
    Define the grid of input guesses passed to the findroot method of
    the mpmath package, that allows to find the resonant and
    anti-resonant states.

    Parameters
    ----------
    re_kmax: float
        Maximal real part of the resonant and anti-resonant states.
    re_hk: float
        Wavenumber grid step along the real axis.
    im_kmax: float
        (Absolute) maximal value for the imaginary part of the resonant
        and anti-resonant states.
    im_hk: float
        Wavenumber grid step along the imaginary axis.

    Returns
    -------
    numpy array
        Grid of input guesses in order to find the resonant and
        anti-resonant states.
    """
    # grid extension along the real axis
    re_min = 0.0
    re_max = re_kmax * (1.+re_hk*0.001)
    # grid extension along the imaginary axis
    im_min = -abs(im_kmax)
    im_max = 0.
    # grid step along the imaginary axis
    if im_hk is None:
        im_hk = float(im_kmax / 2)
    # Create the input guess
    re_kgrid = np.arange(re_min, re_max, re_hk)
    im_kgrid = 1.j*np.arange(im_min, im_max, im_hk)
    re, im = np.meshgrid(re_kgrid, im_kgrid)
    return (re + im).flatten()


def _find_res_ares_states(kgrid, dep_k, dop_k, wavenumbers, parities):
    r"""
    The resonant states are searched in one quadrant of the complex
    wavenumber plane (:math:`\text{Im}(k_s) < 0` and
    :math:`\text{Re}(k_s) > 0`), starting from an input guess grid.

    Some computation time is saved by only looking for the resonant
    states, as it is easy to find the related anti-resonant states via
    :math:`k_{ares} = - {k_{res}}^*`, where :math:`*` represents the
    complex conjugation.

    Parameters
    ----------
    kgrid: numpy array
        Grid of input guesses.
    dep_k: function
        Function whose zeros are even Siegert states wavenumbers.
    dop_k: function
        Function whose zeros are odd Siegert states wavenumbers.
    wavenumbers: list of complex
        Siegert states wavenumbers already found.
    parities: list of str
        Parities of the Siegert states already found.

    Returns
    -------
    tuple(list of complex, list of str)
        List of wavenumbers found along with the parity of the state.
    """
    # Tolerance on the real and imaginary parts for two Siegert
    # states to be considered as different.
    tol = 10**(-12)
    for func, parity in ((dep_k, 'e'), (dop_k, 'o')):
        res_ares = _res_ares_wavenumbers(kgrid, func)
        for k_s in res_ares:
            # Check that the solution was not already known
            if not np.any(np.isclose(k_s, wavenumbers, rtol=tol)):
                # Add it to the list of wavenumbers and create a
                # new resonant state and its anti-resonant
                # counterpart
                for k in (k_s, -k_s.conjugate()):
                    wavenumbers.append(k)
                    parities.append(parity)
    return wavenumbers, parities


def _res_ares_wavenumbers(kgrid, func):
    r"""
    Generator of solutions of the Jost delta functions in the fourth
    quadrant of the complex wavenumber plane. A solution of these
    functions is generated for each point of the wavenumber grid,
    given that the constraints on the location in the complex
    wavenumber plane are satisfied.

    Parameters
    ----------
    kgrid: numpy array
        Grid of input guesses.
    dep_k: function
        Function whose zeros are Siegert states wavenumbers.

    Yields
    ------
    complex
        Valid resonant state wavenumber.
    """
    # Unpack some important values to check the location of the
    # Siegert state wavenumber found
    re_min = np.real(kgrid[0])
    im_min = np.imag(kgrid[0])
    re_max = np.real(kgrid[-1])
    im_max = 0
    # Loop over the flatten grid of input guesses
    for k_0 in kgrid:
        # Try to find a solution k_s from this starting point
        try:
            k_s = findroot(func, k_0, solver='muller', tol=1.e-17)
            # The wavenumber must be inside the user-defined area
            if (re_min <= np.real(k_s) <= re_max and
                    im_min <= np.imag(k_s) <= im_max):
                yield complex(k_s)
        # If findroot does not converge, it throws a ValueError.
        # Catch it and do nothing.
        except ValueError:  # pragma: no cover
            pass
