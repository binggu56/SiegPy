# -*- coding: utf-8 -*
r"""
The :class:`AnalyticBasisSet` class and its methods are defined hereafter.
"""
# TODO: Add a create_exact_basis_set method (for AnalyticBasisSet)?


from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.special import wofz
from siegpy import BasisSet, BasisSetError
from siegpy.utils import init_plot, finalize_plot
from siegpy.basisset import _set_mat_time, _set_mat_space


__all__ = ["AnalyticBasisSet"]


class AnalyticBasisSet(BasisSet, metaclass=ABCMeta):
    r"""
    This should be the base class for the specific basis set classes
    of any analytic cases available in SiegPy.

    Its methods allow to compute most the main quantities and forces
    the child classes to implement the other ones.

    For instance, any relevant Siegert state expansion (such as the
    Mittag-Leffler Expansion (MLE), which uses all Siegert states with a
    weight 1/2, or the Berggren expansion, which uses only bound and
    resonant states with a weight 1) of the quantities of interest (such
    as the completeness relation (CR), zero operator, strength function,
    strength function and time-propagation must therefore be defined
    here, as they should be valid for any analytical case.

    On the other hand, all the methods used to define the analytical
    eigenstates must be implemented by the child classes.
    """

    def __init__(self, states=None):
        r"""
        An AnalyticBasisSet instance is initialized from a list of
        :class:`~siegpy.analyticeigenstates.AnalyticEigenstate`
        instances.

        Parameters
        ----------
        states: list
            Analytic eigenstates of a Hamiltonian. If ``None``, it means
            that the BasisSet instance is empty.

        Raises
        ------
        ValueError
            If any state comes from a different potential than the
            others.
        """
        # Sort the list of states
        new_states = []
        if states:
            # Check that all states come from the same potential
            pot = states[0].potential
            if any([s.potential != pot for s in states]):
                raise ValueError(
                    "All the eigenstates must come from the same potential")
            # Loop over each type of states
            for S_type in self.STATES_TYPES:
                # Sort each type of state by increasing real energy
                tmp = [s for s in states if s.Siegert_type == S_type]
                tmp.sort(key=lambda x: x.energy.real)
                # Update the list of states
                new_states += tmp
        super().__init__(states=new_states)

    @classmethod
    def from_file(cls, filename, grid=None, analytic=True, nres=None,
                  kmax=None, bounds_only=False):
        r"""
        Read a basis set from a binary file.

        Parameters
        ----------
        filename: str
            Name of a file containing a basis set.
        grid: numpy array or list or set
            Discretization grid of the wavefunctions of the Siegert
            states (optional).
        analytic: bool
            If ``True``, the scalar products will be computed
            analytically.
        nres: int
            Number of resonant couples in the Siegert basis set created
            (optional).
        kmax: float
            Value of the maximal continuum wavenumber in the returned
            basis set, also containing the bound states, if there were
            any in the file (optional).
        bounds_only: bool
            If ``True``, the basis set returned contains only bound
            states.

        Returns
        -------
        AnalyticBasisSet
            The basis set read from the file.


        Examples

        All the examples given below are given for the SWP analytical
        case, but could be easily adapted for any other analytical
        cases.

        A basis set is read from a file in the following manner:

        >>> from siegpy import SWPBasisSet
        >>> filename = "doc/notebooks/siegerts.dat"
        >>> bs = SWPBasisSet.from_file(filename)

        It contains a certain number of states, with a given
        discretization grid (here, the grid is ``None``) and
        analyticity:

        >>> len(bs)
        566
        >>> bs.grid
        >>> assert bs.analytic == True

        It is possible to update the grid (and the values of the
        eigenstates at the same time) in the 1D SW potential case:

        >>> bs = SWPBasisSet.from_file(filename, grid=[-2, -1, 0, 1, 2])
        >>> bs.grid
        array([-2, -1,  0,  1,  2])
        >>> print(bs[0].values)
        [ 0.18053285+0.j  0.51185847+0.j  0.63921710-0.j  0.51185847-0.j
          0.18053285-0.j]

        The analyticity of the states can also be updated:

        >>> bs = SWPBasisSet.from_file(filename, analytic=False)
        >>> assert bs.analytic == False

        If required, only the bound states are read:

        >>> bounds = SWPBasisSet.from_file(filename, bounds_only=True)
        >>> assert len(bounds) == len(bs.bounds)

        The number of resonant and antiresonant states can also be
        chosen:

        >>> siegerts = SWPBasisSet.from_file(filename, nres=5)
        >>> assert len(siegerts.bounds) == len(bs.bounds)
        >>> assert len(siegerts.antibounds) == len(bs.antibounds)
        >>> assert len(siegerts.resonants) == 5
        >>> assert len(siegerts.antiresonants) == 5

        If the basis set contains continuum states, it is also
        possible to keep only the states whose wavenumber is
        smaller than kmax (in addition to the bound states).
        """
        # Read the basis set from the file
        basis = super().from_file(filename)
        # Keep only the required states
        if bounds_only:
            basis = basis.bounds
        elif nres is not None:
            basis = (basis.bounds + basis.antibounds +
                     basis.resonants[:nres] + basis.antiresonants[:nres])
        elif kmax is not None:
            cont = [c for c in basis.continuum if c.wavenumber <= kmax]
            basis = basis.bounds + cont
        # Update the analyticity and the grid of each state in the
        # basis set before returning it
        basis.analytic = analytic
        basis.grid = grid
        return basis

    @classmethod
    @abstractmethod
    def find_Siegert_states(cls):  # pragma: no cover
        r"""
        .. note:: This is an asbtract class method.

        Initialize a basis made of Siegert states found analytically.

        Returns
        -------
        AnalyticBasisSet
            A basis set made of
            :class:`~siegpy.analyticeigenstates.AnalyticSiegert`
            instances.
        """
        pass

    @classmethod
    @abstractmethod
    def find_continuum_states(cls, *args, **kwargs):  # pragma: no cover
        r"""
        .. note:: This is an asbtract class method.

        Initialize a basis made of continuum states found analytically.

        Returns
        -------
        AnalyticBasisSet
            A basis set made of
            :class:`~siegpy.analyticeigenstates.AnalyticContinuum`
            instances.
        """
        pass

    @property
    def grid(self):
        r"""
        Returns
        -------
        numpy array or None
            Value of the discretization grid of all the states in the
            basis set.

        Raises
        ------
        ValueError
            If at least one state has a different grid than the others
            or if the basis set is empty.


        Examples

        In the following example, the states have no grid:

        >>> from siegpy import SWPBasisSet
        >>> siegerts = SWPBasisSet.from_file("doc/notebooks/siegerts.dat")
        >>> siegerts.grid is None
        True

        You can also use the
        :attr:`~siegpy.analyticbasisset.AnalyticBasisSet.grid`
        attribute to update the grid (and therefore all the values of
        the wavefunctions) of the states in the analytic basis set at
        the same time:

        >>> xgrid = [-1, 0, 1]
        >>> siegerts.grid = xgrid
        >>> siegerts[-1].grid
        array([-1,  0,  1])
        """
        if self.is_not_empty:
            xgrid = self[0].grid
            if xgrid is None:
                if any([state.grid is not None for state in self]):
                    raise ValueError(
                        "Some states have a different value for 'grid'")
                else:
                    return None
            else:
                if any([state.grid is None for state in self]) or \
                        any([not np.allclose(state.grid, xgrid)
                             for state in self]):
                    raise ValueError(
                        "Some states have a different value for 'grid'")
                else:
                    return xgrid
        else:
            raise ValueError("Empy basis set, it has no 'grid' attribute")

    @grid.setter
    def grid(self, new_grid):
        r"""
        Setter for the :attr:`grid` attribute.

        Parameters
        ----------
        new_grid: list or numpy array
            New grid to apply to all the states.
        """
        for state in self:
            state.grid = new_grid

    @property
    def analytic(self):
        r"""
        Returns
        -------
        bool
            Value of the :attr:`analytic` attribute of all the states
            in the basis set.

        Raises
        ------
        ValueError
            If some states have different value for :attr:`analytic` or
            if the basis set is empty.


        Examples

        In the following example, all the states require analytic scalar
        products:

        >>> from siegpy import SWPBasisSet
        >>> siegerts = SWPBasisSet.from_file("doc/notebooks/siegerts.dat")
        >>> siegerts.analytic
        True

        You can also use the
        :attr:`~siegpy.analyticbasisset.AnalyticBasisSet.analytic`
        attribute to update the values for all the states in the basis
        set at the same time:

        >>> siegerts.analytic = False
        >>> all([state.analytic == False for state in siegerts])
        True
        """
        if self.is_not_empty:
            analytic = self[0].analytic
            if all([state.analytic == analytic for state in self]):
                return analytic
            else:
                raise ValueError(
                    "Some states have a different value for 'analytic'")
        else:
            raise ValueError("Empy basis set, it has no 'analytic' attribute")

    @analytic.setter
    def analytic(self, value):
        r"""
        Setter for the :attr:`analytic` attribute.

        Parameters
        ----------
        value: bool
            New value to apply to all the states.
        """
        for state in self:
            state.analytic = value

    @property
    def potential(self):
        r"""
        Returns
        -------
        Potential
            Potential of all the states in the basis set.

        Raises
        ------
        ValueError
            If some states come from a different potential or if the
            basis set is empty.


        Example

        >>> from siegpy import SWPBasisSet
        >>> siegerts = SWPBasisSet.from_file("doc/notebooks/siegerts.dat")
        >>> siegerts.potential
        1D Square-Well Potential of width 4.44 and depth 10.00
        """
        if self.is_not_empty:
            pot = self[0].potential
            if all([state.potential == pot for state in self]):
                return pot
            else:
                raise ValueError("Some states come from a different potential")
        else:
            raise ValueError("Empy basis set, it has no 'potential' attribute")

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

        Raises
        ------
        ValueError
            If the minimum of the potential cannot be found.
        """
        # Object-oriented plots
        fig, ax = init_plot()
        # Set useful variables
        potential = self.potential
        bnds = self.bounds
        res = self.resonants
        ares = self.antiresonants
        # Plot only part of the resonances, if asked by the user
        if nres is not None:
            res = res[:nres]
            ares = ares[:nres]
        siegerts_to_plot = bnds + res + ares  # List of Siegert states
        # Loop over the Siegert states
        for i, wf in enumerate(siegerts_to_plot):
            # Define the value of its energy given its type
            if wf.Siegert_type in ('r', 'ar'):
                energy = abs(wf.energy)
            elif wf.Siegert_type in ('b', 'ab'):
                energy = - abs(wf.energy)
            # Define the labels
            label_re, label_im, label_en = None, None, None
            if i == 0:
                label_re, label_im, label_en = 'Re[WF]', 'Im[WF]', 'Energy'
            # Add the wf (real and imaginary parts) in the plot
            ax.plot(wf.grid, np.real(wf.values)+energy, 'b', label=label_re)
            ax.plot(wf.grid, np.imag(wf.values)+energy, 'r', label=label_im)
            # Also plot the value of its ("translation") energy.
            ax.plot(wf.grid, np.array([energy]*len(wf.grid)), 'k--',
                    label=label_en)
        # Plot the potential as well
        ax.plot(potential.grid, np.real(potential.values), 'k-',
                label='Re[Potential]')
        ax.plot(potential.grid, np.imag(potential.values), c='k', ls='dotted',
                ms=1, label='Im[Potential]')
        # Finalize the plot
        # Set the range of the plot on the y-axis (ylim)
        if ylim is None:
            # Define the minimum of the y-axis:
            if hasattr(potential, 'values'):
                ymin = 1.1 * np.min(potential.values).real
            else:
                raise ValueError("Cannot find the minimum of the potential")
            # Define the maximum of the y-axis
            if nres == 0:
                ymax = abs(ymin) / 10
            else:
                ymax = abs(res[-1].energy)+2
            ylim = (ymin, ymax)
        finalize_plot(fig, ax, xlim=xlim, ylim=ylim, title=title,
                      file_save=file_save, xlabel="$x$", ylabel="Energy",
                      leg_loc=6, leg_bbox_to_anchor=(1, 0.5))

    def plot_wavenumbers(self, xlim=None, ylim=None, title=None,
                         file_save=None):  # pragma: no cover
        r"""
        Plot the wavenumbers of the Siegert states in the basis set.

        Parameters
        ----------
        xlim: tuple(float or int, float or int)
            Range of the x axis of the plot (optional).
        ylim: tuple(float or int, float or int)
            Range of the y axis of the plot (optional).
        title: str
            Plot title (optional).
        file_save: str
            Filename of the plot to be saved (optional).
        """
        _complex_plane_plot(
            self.states, 'wavenumber', xlim, ylim, title, file_save)

    def plot_energies(self, xlim=None, ylim=None, title=None,
                      file_save=None):  # pragma: no cover
        r"""
        Plot the energies of the Siegert states in the basis set.

        Parameters
        ----------
        xlim: tuple(float or int, float or int)
            Range of the x axis of the plot (optional).
        ylim: tuple(float or int, float or int)
            Range of the y axis of the plot (optional).
        title: str
            Plot title (optional).
        file_save: str
            Filename of the plot to be saved (optional).
        """
        _complex_plane_plot(
            self.states, 'energy', xlim, ylim, title, file_save)

    def MLE_contributions_to_CR(self, test):
        r"""
        Evaluate the contribution of each state of the basis set to the
        completeness relation, according to the Mittag-Leffler
        Expansion.

        Each element of the array is defined by:
        .. math::

            \frac{ \left\langle test | \varphi_S \right)
            \left( \varphi_S | test \right\rangle }
            {2 \left\langle test | test \right\rangle}

        where :math:`\varphi_S` is a Siegert state of the basis set.

        Parameters
        ----------
        test: Function
            Test function

        Returns
        -------
        numpy array
            Contribution of each state of the basis set to the
            completeness relation acording to the Mittag-Leffler
            Expansion of the completeness relation.
        """
        # Check if the test function needs to be explicitely
        # conjugated:
        # TODO: add the case of a complex test function if not analytic
        #       scalar products
        try:
            conjugate_test = test.momentum != 0
        except AttributeError:
            conjugate_test = False
        # Contribution of each Siegert state to the MLE of the CR
        siegerts = self.siegerts
        scal_prods = siegerts.scal_prod(test)
        if conjugate_test:
            scal_prods_l = siegerts.scal_prod(test.conjugate())
            MLE_contribs = scal_prods * scal_prods_l
        else:
            MLE_contribs = scal_prods**2
        return np.array(MLE_contribs / (2 * test.norm()))

    def MLE_contributions_to_zero_operator(self, test):
        r"""
        Evaluate the contribution of each state of the basis set to the
        zero operator, according to the Mittag-Leffler Expansion.

        Each element of the returned array is defined by:

        .. math::

            \frac{ \left\langle test | \varphi_S \right)
            \left( \varphi_S | test \right\rangle }
            {2 k_S \left\langle test | test \right\rangle}

        where :math:`k_s` is the wavenumber of the Siegert state
        :math:`\varphi_S`

        Parameters
        ----------
        test: Function
            Test function.

        Returns
        -------
        numpy array
            Contribution of each state of the basis set to the zero
            operator acording to the Mittag-Leffler Expansion of the
            zero operator.
        """
        # One just needs to compute the contributions of each state to
        # the MLE of the CR and then divide each contribution by the
        # wavenumber of the corresponding state.
        # Thanks to numpy, this can be done in one line.
        return self.MLE_contributions_to_CR(test) / self.wavenumbers

    def MLE_completeness(self, test, nres=None):
        r"""
        Evaluate the value of the Mittag-Leffler Expansion of the
        completeness relation using all Siegert states in the basis set
        for a given test function.

        Returns the result of the following sum over all Siegert states
        :math:`varphi_S`:

        .. math::

            \sum_S \frac{ \left\langle test | \varphi_S \right)
            \left( \varphi_S | test \right\rangle }
            {2 \left\langle test | test \right\rangle}

        Parameters
        ----------
        test: Function
            Test function.
        nres: int
            Number of (anti-)resonant states to use (optional).

        Returns
        -------
        float
            Evaluation of the completeness of the basis set using the
            Mittag-Leffler Expansion.
        """
        return _MLE(self.siegerts, 'CR', test, nres=nres)

    def MLE_zero_operator(self, test, nres=None):
        r"""
        Evaluate the value of the Mittag-Leffler Expansion of the zero
        operator using all Siegert states in the basis set for a given
        test function.

        Returns the result of the following sum over all Siegert states:

        .. math::

            \sum_S \frac{ \left\langle test | \varphi_S \right)
            \left( \varphi_S | test \right\rangle }
            { 2 k_S \left\langle test | test \right\rangle }

        where :math:`k_s` is the wavenumber of the Siegert state
        :math:`\varphi_S`

        Parameters
        ----------
        test: Function
            Test function.
        nres: int
            Number of (anti-)resonant states to use (optional).

        Returns
        -------
        float
            Evaluation of the zero operator of the basis set using the
            Mittag-Leffler Expansion.
        """
        return _MLE(self.siegerts, 'Zero', test, nres=nres)

    def exact_completeness(self, test, hk=None, kmax=None):
        r"""
        Evaluate the exact completeness relation using the bound states
        of the basis set and the continuum states defined by either
        `hk` and `kmax` (respectively the grid-step and
        maximum wavenumber of the grid of continuum states) or all the
        continuum states in the basis set.

        Parameters
        ----------
        test: Function
            Test function.
        hk: float
            Grid step for the wavenumbers of the "on-the-fly" continuum
            basis sets (optional).
        kmax: float
            Maximal wavenumber of the "on-the-fly" continuum basis set
            (optional).

        Returns
        -------
        float
            Value of the exact completeness relation, using bound and
            continuum states.
        """
        # Bound states contribution to the exact CR
        bnds = self.bounds
        bnds_contrib = 2 * bnds.MLE_completeness(test)
        # Continuum states contribution to the CR: this is computed
        # through an integration of the sum of even and odd continuum
        # states contributions for each wavenumber of the grid.
        kgrid, integrand = self.continuum_contributions_to_CR(test, hk=hk,
                                                              kmax=kmax)
        if hk is None:
            hk = kgrid[1] - kgrid[0]
        cont_contribs = np.trapz(integrand, dx=hk)
        return bnds_contrib + cont_contribs

    @abstractmethod
    def continuum_contributions_to_CR(self, test, hk=None, kmax=None):  # pragma: no cover  # noqa
        r"""
        .. note:: This is an asbtract class method.

        Evaluate the continuum contributions to the completeness
        relation.

        Parameters
        ----------
        test: Function
            Test function.
        hk: float
            Grid step for the wavenumbers of the "on-the-fly" continuum
            basis sets (optional).
        kmax: float
            Maximal wavenumber of the "on-the-fly" continuum basis set
            (optional).

        Returns
        -------
        numpy array
            Contribution of each continuum state of the basis set to the
            exact completeness relation.
        r"""
        pass

    def MLE_completeness_convergence(self, test, nres=None):
        r"""
        Evaluate the convergence of the Mittag-Leffler Expansion of the
        completeness relation for the basis set, given a test
        function.

        Parameters
        ----------
        test: Function
            Test function.
        nres: int
            Number of (anti-)resonant states to use (default: use all of
            them).

        Returns
        -------
        tuple(numpy array, numpy array)
            Two arrays of the same length. The first one is made of the
            absolute value of the resonant wavenumbers, while the second
            one is made of the values of the convergence of the MLE of
            the completeness relation.
        """
        return _MLE_convergence(self.siegerts, 'CR', test, nres=nres)

    def Berggren_completeness_convergence(self, test, nres=None):
        r"""
        Evaluate the convergence of the CR using the Berggren expansion.

        Parameters
        ----------
        test: Function
            Test function.
        nres: int
            Number of resonant states to use (default: use all of them).

        Returns
        -------
        tuple(numpy array, numpy array)
            Two arrays of the same length. The first one is made of the
            absolute value of the resonant wavenumbers, while the second
            one is made of the values of the convergence of the Berggren
            expansion of the completeness relation.
        """
        # Keep only the required number of resonant states
        res = self.resonants
        if nres is not None:
            res._states = res._states[:nres]
        # Find the bound and resonant states contributions to the CR
        bnds_contrib = 2 * self.bounds.MLE_contributions_to_CR(test)
        res_contrib = 2 * res.MLE_contributions_to_CR(test)
        # Create the output values as usual
        kgrid = np.insert(np.abs(res.wavenumbers), 0, 0)
        CR_conv = np.sum(bnds_contrib)+np.insert(np.cumsum(res_contrib), 0, 0)
        return kgrid, CR_conv

    def MLE_zero_operator_convergence(self, test, nres=None):
        r"""
        Evaluate the convergence of the Mittag-Leffler Expansion of the
        zero operator for the basis set, given a test function.

        Parameters
        ----------
        test: Function
            Test function.
        nres: int
            Number of (anti-)resonant states to use (default: use all of
            them).

        Returns
        -------
        tuple(numpy array, numpy array)
            Two arrays of the same length. The first one is made of the
            absolute value of the resonant wavenumbers, while the second
            one is made of the values of the convergence of the MLE of
            the zero operator.
        """
        return _MLE_convergence(self.siegerts, 'Zero', test, nres=nres)

    def exact_completeness_convergence(self, test, hk=None, kmax=None):
        r"""
        Evaluate the convergence of the exact completeness relation
        (using the continuum and bound states of the basis set) for a
        given test function.

        A set of continuum states is defined on-the-fly if the values
        of both ``hk`` and ``kmax`` are not ``None`` (otherwise,
        the continuum states of the basis set, if any, are used).

        Parameters
        ----------
        test: Function
            Test function.
        hk: float
            Grid step for the wavenumbers of the "on-the-fly" continuum
            basis sets (optional).
        kmax: float
            Maximal wavenumber of the "on-the-fly" continuum basis set
            (optional).

        Returns
        -------
        tuple(numpy array, numpy array)
            Two arrays of the same length. The first one is made of the
            continuum wavenumbers, while the second is made of the
            values of the convergence of the exact completeness
            relation.
        """
        # Bound states contribution to the exact CR
        bnds = self.bounds
        bnds_contrib = 2 * bnds.MLE_completeness(test)
        # Convergence of the continuum contribution to the CR
        kgrid, cont_contribs = (
            self.continuum_contributions_to_CR(test, hk=hk, kmax=kmax))
        if hk is None:
            hk = kgrid[1] - kgrid[0]
        conv_cont_contribs = hk * (np.cumsum(cont_contribs) - cont_contribs/2.)
        # Sum both bound and continuum states contributions to the exact CR
        conv_CR = np.real(conv_cont_contribs + bnds_contrib)
        # Return the convergence of the completeness relation
        return kgrid, conv_CR

    def plot_completeness_convergence(self, test, hk=None, kmax=None,
                                      nres=None, exact=True, MLE=True,
                                      title=None, file_save=None):  # pragma: no cover  # noqa
        r"""
        Plot the convergence of both the Mittag-Leffler Expansion and
        the exact completeness relation for a given test function.

        The exact convergence is computed on-the-fly (*i.e.* without
        the need to have continuum states in the basis set):

        * if ``hk`` and ``kmax`` are not ``None``,

        * if ``exact`` and ``MLE`` are set to ``True`` and if
          the basis set does not contain enough continuum states to
          reach the absolute value of the last resonant wavenumber used
          (``kmax`` is therefore defined by the wavenumber of the
          last resonant state used in the MLE of the CR, and ``hk``
          is set to a default value)

        Parameters
        ----------
        test: Function
            Test function.
        hk: float
            Grid step for the wavenumbers of the "on-the-fly" continuum
            basis sets (optional).
        kmax: float
            Maximal wavenumber of the "on-the-fly" continuum basis set
            (optional).
        nres: int
            Number of resonant couples contributions to be plotted
            (optional).
        exact: bool
            If ``True``, allows the plot of the exact strength function.
        MLE: bool
            If ``True``, allows the plot of the Mittag-Leffler Expansion
            of the strength function.
        title: str
            Plot title (optional).
        file_save: str
            Filename of the plot to be saved (optional).
        """
        # Object-oriented plots
        fig, ax = init_plot()
        # Plot the expected value of 1
        ax.axhline(1, color='black', lw=1.5)
        # Set some initial values
        if MLE:
            # Evaluate the convergence of the MLE of the CR
            abs_kres, MLE_CR = self.MLE_completeness_convergence(test,
                                                                 nres=nres)
            # Define if the parameters for the on-the-fly creation of
            # the continuum states have to be initialized
            if exact:
                # OTF is a Boolean stating if the continuum has to be
                # defined on-the-fly, given the input parameters
                OTF = (hk is not None) and (kmax is not None)
                if not OTF and (self.continuum.is_empty or
                                self.continuum[-1].wavenumber < abs_kres[-1]):
                    if kmax is None:
                        kmax = abs_kres[-1]
                    if hk is None:
                        hk = 0.1
        if exact:
            # Evaluate and plot the convergence of the exact CR
            kgrid, exact_CR = \
                self.exact_completeness_convergence(test, hk=hk, kmax=kmax)
            ax.plot(kgrid, exact_CR, color='#d73027', label='Exact')
        # Plot the convergence of the MLE of the CR above the
        # convergence of the exact CR
        if MLE:
            ax.plot(abs_kres, np.real(MLE_CR), color='#4575b4', label='MLE',
                    ls='', marker='.', ms=10)
        # Finalize the plot
        ax.legend()
        if MLE and not exact:
            ylabel = "$CR_{MLE}$"
        else:
            ylabel = "$CR$"
        finalize_plot(fig, ax, title=title, file_save=file_save,
                      xlabel="$k$", ylabel=ylabel)

    def MLE_strength_function(self, test, kgrid):
        r"""
        Evaluate the Mittag-Leffler expansion of the strength function
        for a given test function. The test function is discretized on
        a grid of wavenumbers ``kgrid``.

        Parameters
        ----------
        test: Function
            Test function.
        kgrid: numpy array
            Wavenumbers for which the strength function is evaluated.

        Returns
        -------
        numpy array
            MLE of the strength function evaluated over the wavenumber
            grid ``kgrid``.
        """
        # Initialize stren_func
        stren_func = np.zeros_like(kgrid)
        # Loop over the siegert states to update the MLE of the RF
        for state in self.siegerts:
            stren_func += state.MLE_strength_function(test, kgrid)
        return stren_func

    def exact_strength_function(self, test, kgrid, eta=None):
        r"""
        Evaluate the exact strength function over a given wavenumber
        grid ``kgrid`` for a given a test function.

        ``eta`` is an infinitesimal used to make the integration over
        the continuum states possible (the poles along the real axis
        being avoided).

        Parameters
        ----------
        test: Function
            Test function.
        kgrid: numpy array
            Wavenumbers for which the strength function is evaluated.
        eta: float
            Infinitesimal for integration (if ``None``, default to 10
            times the value of the grid-step of the continuum basis
            set).

        Returns
        -------
        numpy array
            Exact strength function evaluated on the grid of wavenumbers
            ``kgrid``.
        """
        # Applying the method continuum_contributions_to_CR gives:
        # sum_p |<test|varphi_p>|**2 / <test|test>
        # for each continuum wavenumber. We actually want:
        # sum_p |<test|varphi_p>|**2
        # so we need to multiply by the norm <test|test>.
        qgrid, cont_values = self.continuum_contributions_to_CR(test)
        cont_values *= test.norm()
        # Grid-step for integration
        h_q = qgrid[1] - qgrid[0]
        # Define the value of eta
        if eta is None:
            eta = 10 * h_q
        # Integrand for each k in kgrid (the imaginary part is already
        # taken into account, and eta is a parameter used to avoid
        # poles during the integration over the real wavenumber axis).
        integrands = [eta * cont_values / ((qgrid**2/2.-k**2/2.)**2 + eta**2)
                      for k in kgrid]
        # strength function (after integration of each integrand)
        stren_func = np.array([np.trapz(rf, dx=h_q) for rf in integrands])
        stren_func /= np.pi
        return stren_func

    def exact_strength_function_OTF(self, test, kgrid, hk=10**(-4), eta=None,
                                    tol=10**(-4)):
        r"""
        Evaluate the exact strength function "on-the-fly" over a given
        wavenumber grid ``kgrid`` for a given a test function.

        It is not necessary to have continuum states in the basis set to
        compute the exact strength function, because they can be
        computed "on-the-fly" (hence OTF). This even leads to a quicker
        evaluation of the strength function because the integral, to be
        compute for each point of ``kgrid``, actually has an integrand
        that peaks around each particular value of ``kgrid`` and quickly
        vanishes around its maximum.

        Such a minimal approximated integrand is constructed by using
        only the continuum states aroud the point of the ``kgrid``
        considered, given the tolerance parameter tol.
        It allows to reach smaller values of ``hk`` and ``eta``, and
        therefore more precise results, in a shorter amount of time.

        Parameters
        ----------
        test: Function
            Test function.
        kgrid: numpy array
            Wavenumbers for which the strength function is evaluated.
        hk: float
            Grid step for the wavenumbers of the "on-the-fly" continuum
            basis sets (optional).
        eta: float
            Infinitesimal for integration (if ``None``, default to 10
            times the value of the grid-step of the continuum basis
            set).
        tol: float
            Tolerance value to truncate the integrand function (ratio of
            the value of a possible new point of the integrand to the
            maximal value of the integrand).

        Returns
        -------
        numpy array
            Approximate exact strength function evaluated on the grid of
            wavenumbers ``kgrid``.
        """
        # Set initial variables
        stren_func = []
        if eta is None:
            eta = 10 * hk
        pot = self.potential
        # Loop over points in kgrid
        for k in kgrid:
            # Build the integrand:
            # - initialize
            integrand_m = []
            integrand_p = []
            first = self._evaluate_integrand(k, k, test, eta, pot)
            q_m = k
            q_p = k
            new_int_m = first
            new_int_p = first
            # - find new terms while necessary
            while new_int_m / first > tol and q_m > 0.:
                q_m = q_m - hk
                new_int_m = self._evaluate_integrand(q_m, k, test, eta, pot)
                integrand_m.insert(0, new_int_m)
            while new_int_p / first > tol:
                q_p = q_p + hk
                new_int_p = self._evaluate_integrand(q_p, k, test, eta, pot)
                integrand_p.append(new_int_p)
            # - assemble the whole integrand function
            integrand = integrand_m + [first] + integrand_p
            # Do the integral, and append the result to stren_func
            stren_func.append(np.trapz(integrand, dx=hk))
        # Return the correctly normalized strength function as an array
        stren_func = np.array(stren_func) / np.pi
        return stren_func

    @staticmethod
    @abstractmethod
    def _evaluate_integrand(q, k, test, eta, potential):  # pragma: no cover
        r"""
        .. note:: This is an asbtract class method.

        Evaluate the integrand used to compute the strength function
        "on-the-fly". It must be implemented in the child class.

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
        """
        pass

    def plot_strength_function(self, test, kgrid, nres=None, exact=True,
                               MLE=True, bnds_abnds=False, title=None,
                               file_save=None):  # pragma: no cover
        r"""
        Plot the convergence of both the Mittag-Leffler Expansion and
        the exact strength function for a given test function.

        The exact convergence is computed on-the-fly (without the need
        to have continuum states in the BasisSet self).

        Parameters
        ----------
        test: Function
            Test function.
        kgrid: numpy array
            Wavenumbers for which the strength function is evaluated.
        nres: int
            Number of resonant couples contributions to be plotted
            (default to None, meaning that none are plotted; if
            ``nres=0``, then only the sum of the bound and anti-bound
            states contributions is plotted).
        exact: bool
            If ``True``, allows the plot of the exact strength function.
        MLE: bool
            If ``True``, allows the plot of the Mittag-Leffler Expansion
            of the strength function.
        bnds_abnds: bool
            If ``True``, allows to plot the individual contributions of
            the bound and anti-bound states.
        title: str
            Plot title (optional).
        file_save: str
            Filename of the plot to be saved (optional).
        """
        # Object-oriented plots
        fig, ax = init_plot()
        # set initial values
        if exact:
            # Evaluate and plot the convergence of the exact strength
            # function, continuum states being computed on-the-fly.
            exact_SF = self.exact_strength_function_OTF(test, kgrid)
            ax.plot(kgrid, exact_SF, color='#d73027', lw=4, label='Exact')
        if MLE:
            # Evaluate and plot the MLE of the SF
            MLE_SF = self.MLE_strength_function(test, kgrid)
            ax.plot(kgrid, MLE_SF, color='#000000', ls='--', label='MLE')
        if nres is not None:
            # Plot the contribution of the bound and anti-bound states
            # to the strength function
            basis = self.bounds + self.antibounds
            if basis.is_not_empty:
                sf = basis.MLE_strength_function(test, kgrid)
                ax.plot(kgrid, sf, color='#b2182b', lw=1.5, label="b+ab")
            # Plot the contribution of the nres first resonance couples
            # to the strength function
            res = self.resonants[:nres]
            ares = self.antiresonants[:nres]
            for i in range(nres):
                sf = (res[i].MLE_strength_function(test, kgrid) +
                      ares[i].MLE_strength_function(test, kgrid))
                label = "$N_{res} = $"+"{}".format(i+1)
                ax.plot(kgrid, sf, lw=1.5, label=label)
        # Plot the contribution of each bound and antibound
        # states to the MLE of the strength function.
        if bnds_abnds:
            bnds = self.bounds
            for state in bnds:
                sf = state.MLE_strength_function(test, kgrid)
                ax.plot(kgrid, sf, color='#FF0000')
            abnds = self.antibounds
            for state in abnds:
                sf = state.MLE_strength_function(test, kgrid)
                ax.plot(kgrid, sf, color='#660000')
        # Finalize the plot
        if MLE or exact:
            # If resonant contributions, place the legends on the right
            if nres is not None and nres > 0:
                leg_loc, leg_bbox_to_anchor = 6, (1, 0.5)
            # Else, place the legends on the default location
            else:
                ax.legend()
                leg_loc, leg_bbox_to_anchor = None, None
        finalize_plot(fig, ax, title=title, file_save=file_save,
                      leg_loc=leg_loc, leg_bbox_to_anchor=leg_bbox_to_anchor,
                      xlabel="$k$", ylabel="$S(k)$")

    def exact_propagation(self, test, time_grid):
        r"""
        Evaluate the exact time-propagation of a wavepacket ``test``
        over a given time grid (made of positive numbers only).

        Parameters
        ----------
        test: Function
            Test function.
        time_grid: numpy array or list of positive numbers
            Times for which the propagation is evaluated. It must
            contain positive numbers only.

        Returns
        -------
        2D numpy array
            Exact propagated wavepacket for the different times of
            ``time_grid``.
        """
        # Evaluate separately the contribution of the continuum states
        # and of the bound states to the time propagation of the
        # wavepacket and finally sum them.
        cont_contrib = self.continuum._propagate(test, time_grid)
        bnds_contrib = self.bounds.Siegert_propagation(test, time_grid,
                                                       weights={'b': 1.0})
        return cont_contrib + bnds_contrib

    def exact_propagation_OTF(self, test, time_grid, kmax, hk):
        r"""
        Evaluate the exact time-propagation of a given test function
        for a given time grid after an on-the-fly creation of the
        continuum states given the values of ``kmax`` and ``hk``.

        Parameters
        ----------
        test: Function
            Test function.
        time_grid: numpy array or list of positive numbers
            Times for which the propagation is evaluated. It must
            contain positive numbers only.
        hk: float
            Grid step for the wavenumbers of the "on-the-fly" continuum
            basis sets (optional).
        kmax: float
            Maximal wavenumber of the "on-the-fly" continuum basis set
            (optional).

        Returns
        -------
        :returns: Exact propagated wavepacket for the different times
            of ``time_grid``.
        :rtype: 2D numpy array
        """
        pot = self.potential
        xgrid = self.grid
        cont = self.__class__.find_continuum_states(pot, kmax, hk, grid=xgrid)
        exact = self.bounds + cont
        return exact.exact_propagation(test, time_grid)

    def MLE_propagation(self, test, time_grid):
        r"""
        Evaluate the Mittag-Leffler Expansion of the time-propagation of
        a test wavepacket over a given time grid.

        Parameters
        ----------
        test: Function
            Test function.
        time_grid: numpy array or list of positive numbers
            Times for which the propagation is evaluated. It must
            contain positive numbers only.

        Returns
        -------
        2D numpy array
            MLE of the propagated wavepacket for the different times of
            ``time_grid``.
        """
        # The Mittag-Leffler expansions consists in using all Siegert
        # states with a 0.5 weight
        return self.Siegert_propagation(test, time_grid,
                                        weights={'b': 0.5, 'ab': 0.5,
                                                 'r': 0.5, 'ar': 0.5})

    def Berggren_propagation(self, test, time_grid):
        r"""
        Evaluate the Berggren Expansion of the time-propagation of a
        test wavepacket over a given time grid.

        Parameters
        ----------
        test: Function
            Test function.
        time_grid: numpy array or list of positive numbers
            Times for which the propagation is evaluated. It must
            contain positive numbers only.

        Returns
        -------
        2D numpy array
            Berggren expansion of the propagated wavepacket for the
            different times of ``time_grid``.
        """
        # The Berggren expansion consists in using the bound and
        # resonant states only (with a weight 1).
        return self.Siegert_propagation(test, time_grid,
                                        weights={'b': 1.0, 'r': 1.0})

    def Siegert_propagation(self, test, time_grid, weights=None):
        r"""
        Evaluate a user-defined expansion over the Siegert states of the
        time-propagation of a test wavepacket over a given time grid.

        The user chooses the weight of each type of Siegert states of
        the basis set. If no weights are passed, then exact Siegert
        states expansion is performed.

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
            Siegert states expansion of the propagated wavepacket for
            the different times of ``time_grid``.

        Raises
        ------
        KeyError
            If an invalid key is found in ``weights``.
        """
        # If a weight dictionary is given, then propagate accordingly,
        # without using the Faddeeva function
        if weights is not None:
            # Check that weights has valid keys
            for S_type in weights.keys():
                if S_type not in self.SIEGERT_STATES_TYPES:
                    raise KeyError("Invalid key {} in the dictionary of "
                                   "weights. The keys must be a subset of "
                                   "('ab', 'b', 'r', 'ar').".format(S_type))
            # Create 0 values for the missing Siegert types
            for S_type in self.SIEGERT_STATES_TYPES:
                if S_type not in weights:
                    weights[S_type] = 0.0
            # Propagate the wavepacket according to the weights
            return self.siegerts._propagate(test, time_grid, weights=weights)
        # Else, if no weights are given, then perform the exact Siegert
        # expansion, using the Faddeeva function.
        else:
            return self.exact_Siegert_propagation(test, time_grid)

    def exact_Siegert_propagation(self, test, time_grid):
        r"""
        Evaluate the exact Siegert propagation of the initial wavepacket
        test for the times of time_grid. In contrast with all other
        expansions for the propagation, the time-evolution of the
        Siegert states is not only due to an exponential, but also to
        state- and time-dependent weights (see eq. 69 of Santra et al.,
        PRA 71 (2005)).

        There is no need to define any weight, since the correct weight
        is known analytically.

        Parameters
        ----------
        test: Function
            Test function.
        time_grid: numpy array or list of positive numbers
            Times for which the propagation is evaluated. It must
            contain positive numbers only.

        Returns
        -------
        2D numpy array
            Exact Siegert expansion of the propagated wavepacket for the
            different times of ``time_grid``.

        Raises
        ------
        BasisSetError
            If the basis set contains no Siegert states.
        """
        # Check that the basis set is not empty.
        if self.siegerts.is_empty:
            raise BasisSetError("The basis set must contain Siegert states")
        # Make sure that time_grid is made of positive numbers
        # and that t=0 is part of the time_grid
        time_grid = np.array(time_grid) - min(time_grid)
        # Prepare the calculation by separating the states needing an
        # exponential term in the time propagation:
        # - states with exponential terms
        basis_1 = self.bounds + self.resonants
        # - states without exponential terms
        basis_2 = self.antibounds + self.antiresonants
        # Prepare the weights accordingly
        weights_1 = {'b': 1, 'r': 1}
        weights_2 = {'ab': 1, 'ar': 1}
        # Evaluate the contribution of the states with the exponential
        # terms in the time_propagation
        if basis_1.is_not_empty:
            mat_space_1 = _set_mat_space_S(basis_1, test, weights_1)
            mat_time_1 = _set_mat_time_S(basis_1, time_grid, with_exp=True)
            mat_prop = mat_time_1.dot(mat_space_1)
        else:
            mat_prop = 0
        # Evaluate the other contributions to the exact Siegert
        # propagation and add them to the first matrix
        if basis_2.is_not_empty:
            mat_space_2 = _set_mat_space_S(basis_2, test, weights_2)
            mat_time_2 = _set_mat_time_S(basis_2, time_grid, with_exp=False)
            mat_prop += mat_time_2.dot(mat_space_2)
        # Return the exact
        return mat_prop

    def _propagate(self, test, time_grid, weights=None):
        r"""
        Evaluate the time-propagation of a test wavepacket as the matrix
        product of two matrices: one to account for the time dependance
        of the propagation of the wavepacket (mat_time), the other for
        its space dependance (mat_space).

        The contribution of the continuum states to the time
        propagation of the wavepacket requires a numerical integration
        that is performed using the composite Simpson's rule.

        The contribution of the Siegert states to the time propagation
        of the wavepacket is computed through a discrete sum over all
        the Siegert states, with a user-defined weight for each type
        of Siegert states.

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
            If the basis set does not contain Siegert nor continuum
            states.
        """
        # Check that there are states to be used
        if (self.siegerts+self.continuum).is_empty:
            raise BasisSetError("The basis set must not be empty")
        # Make sure that time_grid is made of positive numbers
        # and that t=0 is part of the time_grid
        time_grid = np.array(time_grid) - min(time_grid)
        # Find the Siegert states contribution to the time propagation
        mat_prop_S = _propagate_over_Siegerts(self.siegerts, test, time_grid,
                                              weights)
        # Find the continuum states contribution to the time propagation
        mat_prop_c = _propagate_over_continuum(self.continuum, test, time_grid)
        # Add both contributions
        return mat_prop_S + mat_prop_c
        # return prop #TODO: list of functions (with time as attribute?)

    @abstractmethod
    def _add_one_continuum_state(self):  # pragma: no cover
        r"""
        .. note:: This is an asbtract class method.

        Add a continuum state to the basis set, depending on the already
        existing continuum states.

        Returns
        -------
        AnalyticBasisSet
            Instance of the child class with one more continuum state.
        """
        pass

    def plot_propagation(self, test, time_grid, exact=True, exact_Siegert=True,
                         MLE=False, Berggren=False, xlim=None, ylim=None,
                         title=None, file_save=None):  # pragma: no cover
        r"""
        Plot the time propagation of a wavepacket using either the exact
        expansion (using bound and continuum states), the exact Siegert
        states expansion, the Mittag-Leffler Expansion or the Berggren
        expansion over a given time grid.

        Any of these expansions can be turned on or off by setting the
        corresponding optional arguments (``exact``, ``exact_Siegert``,
        ``MLE`` and ``Berggren`` respectively) to ``True`` or ``False``.
        By default, both exact expansions are plotted.

        Parameters
        ----------
        test: Function
            Test function.
        time_grid: numpy array or list of positive numbers
            Times for which the propagation is evaluated. It must
            contain positive numbers only.
        exact: bool
            If ``True``, allows the plot of the exact time-propagation
            (using bound and continuum states).
        exact_Siegert: bool
            If ``True``, allows the plot of the exact time-propagation
            (using all Siegert states).
        MLE: bool
            If ``True``, allows the plot of the Mittag-Leffler Expansion
            of the time-propagation.
        Berggren: bool
            If ``True``, allows the plot of the Berggren expansion of
            the time-propagation.
        xlim: tuple(float or int, float or int)
            Range of the x axis of the plot (optional).
        xlim: tuple(float or int, float or int)
            Range of the y axis of the plot (optional).
        title: str
            Plot title (optional).
        file_save: str
            Base name of the files where the plots are to be saved
            (optional).
        """
        # Evaluate the various time-propagation expansions required
        if exact:
            # Evaluate and plot the convergence of the exact
            # time-propagation.
            if self.continuum.is_empty:
                # Create continuum states on-the-fly
                kmax = abs(self.resonants.wavenumbers[-1])
                exact_tp = self.exact_propagation_OTF(test, time_grid,
                                                      kmax, hk=0.05)
            else:
                exact_tp = self.exact_propagation(test, time_grid)
        else:
            exact_tp = None
        if exact_Siegert:
            # Evaluate and plot the convergence of the exact
            # Siegert time-propagation.
            exact_S_tp = self.exact_Siegert_propagation(test, time_grid)
        else:
            exact_S_tp = None
        if MLE:
            # Evaluate and plot the MLE of the time-propagation
            MLE_tp = self.MLE_propagation(test, time_grid)
        else:
            MLE_tp = None
        if Berggren:
            # Evaluate the Berggren expansion of the time-propagation
            Ber_tp = self.Berggren_propagation(test, time_grid)
        else:
            Ber_tp = None
        _generate_time_plots(self.grid, time_grid, exact_tp, exact_S_tp,
                             MLE_tp, Ber_tp, xlim, ylim, title, file_save)


def _complex_plane_plot(states, attr, xlim, ylim, title, file_save):  # pragma: no cover  # noqa
    r"""
    Plot either the wavenumber or the energy of each Siegert state in
    the basis set.

    Parameters
    ----------
    states: list
        States of the basis set to be plotted.
    attr: str
        Name of the eigenstate attribute to look for, defining the type
        of plot to be done.
    xlim: tuple(float or int, float or int)
        Range of the x axis of the plot (optional).
    ylim: tuple(float or int, float or int)
        Range of the y axis of the plot (optional).
    title: str
        Plot title (optional).
    file_save: str
        Filename of the plot to be saved (optional).
    """
    # Object-oriented plots
    fig, ax = init_plot()
    # Add the real and imaginary axes
    ax.axhline(0, color='black', lw=1)  # black line y=0
    ax.axvline(0, color='black', lw=1)  # black line x=0
    # Get the data to be plotted thanks to the name of the attribute
    plot_data = {}
    for S_type in BasisSet.STATES_TYPES:
        plot_data[S_type] = [getattr(state, attr) for state in states
                             if state.Siegert_type == S_type]
    # Plot the data for each type of states
    labels = {'ab': 'Anti-bounds', 'b': 'Bounds', 'r': 'Resonants',
              'ar': 'Anti-resonants', None: 'Continuum', 'U': 'Unknown'}
    markers = {'ab': 's', 'b': 'o', 'r': 'd', 'ar': '^', None: 'o',
               'U': 'x'}
    colors = {'ab': '#660000', 'b': '#FF0000', 'r': '#0000FF',
              'ar': '#000066', None: 'k', 'U': 'k'}
    for S_type in BasisSet.STATES_TYPES:
        if plot_data[S_type] != []:
            ax.plot(np.real(plot_data[S_type]), np.imag(plot_data[S_type]),
                    color=colors[S_type], marker=markers[S_type],
                    linestyle='None', label=labels[S_type])
    # Set the base name of the x and y axis of the plot
    if attr == 'wavenumber':
        data_label = 'k'
    elif attr == 'energy':
        data_label = 'E'
    xlabel = "Re[${}$]".format(data_label)
    ylabel = "Im[${}$]".format(data_label)
    # Finalize the plot
    finalize_plot(fig, ax, xlim=xlim, ylim=ylim, title=title, leg_loc=2,
                  file_save=file_save, xlabel=xlabel, ylabel=ylabel)


def _MLE(siegerts, operator, test, nres=None):
    r"""
    Evaluate the Mittag-Leffler Expansion of an operator over all the
    Siegert states in the basis set as a sum of the contribution of each
    state.

    Parameters
    ----------
    siegerts: AnalyticBasisSet
        Basis set made of Siegert states only.
    operator: str
        Name of the operator.
    test: Function
        Test function.
    nres: int
        Number of (anti-)resonant states to use.

    Returns
    -------
    float
        Evaluation of the operator on the whole the basis set using the
        Mittag-Leffler Expansion.

    Raises
    ------
    ValueError
        If the name of the operator is not 'CR' nor 'Zero'.
    """
    # Keep the required siegert states
    if nres is not None:
        siegerts = siegerts.bounds + siegerts.antibounds \
            + siegerts.resonants[:nres] + siegerts.antiresonants[:nres]
    # Define which MLE to obtain
    if operator == 'CR':  # Completeness relation
        contribs = siegerts.MLE_contributions_to_CR(test)
    elif operator == 'Zero':  # Zero operator
        contribs = siegerts.MLE_contributions_to_zero_operator(test)
    else:
        raise ValueError('Unknown operator {}'.format(operator))
    # Return the cumulative sum of the values
    return np.sum(contribs)


def _MLE_convergence(siegerts, operator, test, nres=None):
    r"""
    Evaluate the convergence of the Mittag-Leffler Expansion of an
    operator as a cumulative sum of the contributions of each
    resonant couple in the basis set.

    An initial value corresponding to the bound and anti-bound
    states contributions is added to all the resonant couples.
    It also corresponds to the first value of the second returned
    array.

    .. warning::

       It returns a tuple made of two arrays:

       * the first one corresponds to the wavenumbers where the
         convergence is evaluated (with 0 as the first element),

       * the second one corresponds to the convergence of the
         operator (the first element corresponding to the bound and
         anti-bound states contributions).

    Parameters
    ----------
    siegerts: AnalyticBasisSet
        Basis set made of Siegert states only.
    operator: str
        Name of the operator.
    test: Function
        Test function.
    nres: int
        Number of (anti-)resonant states to use (default: use all of
        them).

    Returns
    -------
    tuple(numpy array, numpy array)
        Wavenumbers and convergence of the MLE of the operator on
        the whole the basis set.
    """
    # The initial value is the sum of the contribution
    # of bound and anti-bound states.
    bnd_abnd = siegerts.bounds + siegerts.antibounds
    init = _MLE(bnd_abnd, operator, test)
    # Evaluate the cumulative sum of the contribution of each
    # resonant/anti-resonant couples contribution to the
    # completenes relation
    res = siegerts.resonants
    ares = siegerts.antiresonants
    if nres is not None:
        res._states = res._states[:nres]
        ares._states = ares._states[:nres]
    if operator == 'CR':
        res_contribs = res.MLE_contributions_to_CR(test)
        ares_contribs = ares.MLE_contributions_to_CR(test)
    elif operator == 'Zero':
        res_contribs = res.MLE_contributions_to_zero_operator(test)
        ares_contribs = ares.MLE_contributions_to_zero_operator(test)
    cum_sum = np.cumsum(res_contribs + ares_contribs)
    # Add a zero at the beginning in order to get the bound and
    # anti-bound states contributions as the first element of the
    # completeness relation array.
    cum_sum = np.insert(cum_sum, 0, 0.)
    # Create the completeness relation evolution list by adding the
    # initial value to all elements of cum_sum.
    conv = cum_sum + init
    # Get the absolute value of resonant wavenumbers.
    # They will be used as the abscissa of the plot.
    kgrid = np.insert(np.abs(res.wavenumbers), 0, 0)
    # Return the array of values to plot (grid of wavenumbers and
    # convergence of the MLE of the operator)
    return kgrid, conv


def _continuum_contributions_to_CR(continuum, test):
    r"""
    Evaluate the contribution of each continuum state of the basis set
    to the exact completeness relation.

    Note that the length of the returned array is equal to
    len(continuum) + 1, in order to account for the 0 contribution of
    the continuum state at k=0.

    Each element of the array is defined by:
    :math:`\sum_{p=+/-}
    \frac{| \left\langle test | \varphi_p \right\rangle |^2}
    {\left\langle test | test \right\rangle}`
    where :math:`\varphi` is a wavefunction of the basis set.

    Parameters
    ----------
    test: Function
        Test function.
    continuum: BasisSet
        Basis set of continuum states (optional).

    Returns
    -------
    numpy array
        Contribution of each continuum state of the basis set to the
        exact completeness relation.

    Raises
    ------
    BasisSetError
        If there are not enough continuum states.
    """
    # 1- Check that there are enough continuum states available
    if len(continuum) <= 1:
        raise BasisSetError(
            "There are not enough continuum states in the basis set.")
    # 2- Contribution of each continuum states
    cont_contribs = np.abs(continuum.scal_prod(test))**2
    cont_contribs = np.insert(cont_contribs, 0, 0.) / test.norm()
    # 3- Get the grid of corresponding wavenumbers
    kgrid = [0.] + continuum.wavenumbers
    return np.array(kgrid), cont_contribs


def _propagate_over_continuum(continuum, test, time_grid):
    r"""
    Evaluate the continuum states contributions to the time propagation
    of ``test`` for all the times in ``time_grid``.

    Parameters
    ----------
    continuum: AnalyticBasisSet
        Basis set made of continuum states only.
    test: Function
        Test function.
    time_grid: numpy array or list of positive numbers
        Times for which the propagation is evaluated. It must
        contain positive numbers only.

    Returns
    -------
    2D numpy array
        Continuum states contributions to the time propagation of
        ``test`` for all the times in ``time_grid``.
    """
    # If there are no continuum states, then they have no
    # contribution to the time propagation
    if continuum.is_empty:
        mat_prop = 0
    # Else, make sure that there are an odd number of continuum
    # states and compute their contribution to the time propagation
    else:
        if len(continuum) % 2 == 0:
            continuum = continuum._add_one_continuum_state()
        mat_space = _set_mat_space_c(continuum, test)
        mat_time = _set_mat_time(continuum, time_grid)
        mat_prop = mat_time.dot(mat_space)
    return mat_prop


def _propagate_over_Siegerts(siegerts, test, time_grid, weights):
    r"""
    Evaluate the Siegert states contributions to the time
    propagation of ``test`` for all the times in ``time_grid``.

    Parameters
    ----------
    siegerts: AnalyticBasisSet
        Basis set made of Siegert states only.
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
        Siegert states contributions to the time propagation of
        ``test`` for all the times in ``time_grid``.

    Raises
    ------
    ValueError
        If no weights are passed.
    """
    # If there are no Siegert states, then they have no contribution
    # to the time propagation
    if siegerts.is_empty:
        mat_prop = 0
    # Else, compute their contribution to the time propagation
    else:
        if weights is None:
            raise ValueError("A dictionary of weights must be provided.")
        mat_space = _set_mat_space_S(siegerts, test, weights)
        mat_time = _set_mat_time(siegerts, time_grid)
        mat_prop = mat_time.dot(mat_space)
    return mat_prop


def _set_mat_time_S(siegerts, time_grid, with_exp=True):
    r"""
    Evaluate the mat_time matrix based on the Siegert states. This
    matrix can then be used to compute the time propagation of an
    initial wavepacket ``test``. It takes the Faddeva function into
    account to compute the exact Siegert time propagation. The usual
    exponential terms may be taken into account, if ``with_exp`` is
    set to ``True``.

    Parameters
    ----------
    siegerts: AnalyticBasisSet
        Basis set made of Siegert states only.
    time_grid: numpy array or list of positive numbers
        Times for which the propagation is evaluated. It must
        contain positive numbers only.
    with_exp: bool
        Sets if the usual exponential term must be used.

    Returns
    -------
    2D numpy array
        Matrix mat_time based on the Siegert states of the basis set.
    """
    # Add the expontial terms only if required and define the sign
    # in front of the Faddeeva function
    if with_exp:
        mat_time = _set_mat_time(siegerts, time_grid)
        sign = -1.
    else:
        mat_time = 0.
        sign = 1.
    # Make the matrix with the appropriate Feddeeva functions
    K, T = np.meshgrid(siegerts.wavenumbers, time_grid)
    mat_time += sign * wofz(
        - sign * np.exp(1.j*np.pi/4) * np.sqrt(T/2.) * K) / 2
    return mat_time


def _set_mat_space_S(siegerts, test, weights):
    r"""
    Evaluate the mat_space matrix based on the Siegert states. This
    matrix can then be used to compute the time propagation of an
    initial wavepacket ``test``. The dictionary of weights gives the
    type of Siegert expansion to be used.

    Parameters
    ----------
    siegerts: AnalyticBasisSet
        Basis set made of Siegert states only.
    test: Function
        Test function.
    weights: dict
        Dictionary of the weights to use for the time-propagation.
        Keys correspond to a type of Siegert states ('ab' for
        anti-bounds, 'b' for bounds, 'r' for resonants and 'ar' for
        anti-resonants) and the corresponding value is the weight to
        use for all the states of the given type (optional).

    Returns
    -------
    2D numpy array
        Matrix mat_space based on the Siegert states of the basis set.

    Raises
    ------
    BasisSetError
        If there are not enough Siegert states.
    """
    # Check that the basis contain at least one Siegert state
    if siegerts.is_empty:
        raise BasisSetError(
            "The basis set must contain at least one Siegert state.")
    # Build the matrix mat_space_S by looping over the desired
    # types of Siegert states, using the desired weight.
    mat_space_S = []
    for states in (siegerts.bounds, siegerts.antibounds, siegerts.resonants,
                   siegerts.antiresonants):
        if states.is_not_empty:
            S_type = states[0].Siegert_type
            contribs = _set_mat_space(states, test, weights[S_type])
            for contrib in contribs:
                mat_space_S.append(contrib)
    return np.array(mat_space_S)


def _set_mat_space_c(continuum, test):
    r"""
    Evaluate the mat_space matrix based on the continuum states.
    This matrix can then be used to compute the time propagation of
    an initial wavepacket ``test``. The integration over the
    continuum states is performed thanks to a vector accounting for
    the composite Simpson's rule (which is the integration scheme
    used here).

    Parameters
    ----------
    continuum: AnalyticBasisSet
        Basis set made of continuum states only.
    test: Function
        Test function.

    Returns
    -------
    2D numpy array
        Matrix mat_time based on the continuum states of the basis set.

    Raises
    ------
    BasisSetError
        If there are not enough continuum states.
    """
    # Check that the basis contain at least two continuum states
    # (to define the wavenumber grid step)
    if len(continuum) <= 1:
        raise BasisSetError("Too few continuum states in the basis set.")
    # Prepare the integration over the continuum states with
    # the composite Simpson's rule (CSR):
    alpha = np.array([4. if (ik % 2 == 1) else 2.
                      for ik in range(1, len(continuum)-1)])
    alpha = np.insert(alpha, 0, 1)
    alpha = np.append(alpha, 1)
    hk = continuum[1].wavenumber - continuum[0].wavenumber
    alpha *= hk / 3.
    # Initialize mat_space
    mat_space_c = _set_mat_space(continuum, test)
    # Mutliply by the integration vector before before returning
    return (alpha * mat_space_c.T).T


def _generate_time_plots(xgrid, time_grid, exact_tp, exact_S_tp, MLE_tp,
                         Ber_tp, xlim, ylim, title, file_save):  # pragma: no cover  # noqa
    r"""
    Parameters
    ----------
    xgrid: numpy array
        Space grid used to discretize the time-propagated wavepacket.
    time_grid: numpy array or list of positive numbers
        Times for which the propagation is evaluated. It must
        contain positive numbers only.
    exact_tp: numpy array
        Exact time-propagation (using bound and continuum states).
    exact_S_tp: bool
        Exact Siegert states expansion of the time-propagation (using
        all Siegert states).
    MLE_tp: bool
        Mittag-Leffler Expansion of the time-propagation (using all
        Siegert states).
    Berggren_tp: bool
        Berggren expansion of the time-propagation (using bound and
        resonant states).
    xlim: tuple(float or int, float or int)
        Range of the x axis of the plot (optional).
    xlim: tuple(float or int, float or int)
        Range of the y axis of the plot (optional).
    title: str
        Plot title (optional).
    file_save: str
        Base name of the files where the plots are to be saved
        (optional).
    """
    # Loop to create the plots for each time in time_grid
    for i, time in enumerate(time_grid):
        # Object-oriented plots
        fig, ax = init_plot()
        if exact_tp is not None:
            ax.plot(xgrid, np.real(exact_tp[i]), color='k', lw=5,
                    label='Re[$f_{exact}(t)$]')
            ax.plot(xgrid, np.imag(exact_tp[i]), color='grey', lw=5,
                    label='Im[$f_{exact}(t)$]')
        if exact_S_tp is not None:
            ax.plot(xgrid, np.real(exact_S_tp[i]), color='#276419',
                    label='Re[$f_{S, exact}(t)$]')
            ax.plot(xgrid, np.imag(exact_S_tp[i]), color='#b8e186',
                    label='Im[$f_{S, exact}(t)$]')
        if MLE_tp is not None:
            ax.plot(xgrid, np.real(MLE_tp[i]), color='#2166ac',
                    label='Re[$f_{MLE}(t)$]')
            ax.plot(xgrid, np.imag(MLE_tp[i]), color='#d1e5f0',
                    label='Im[$f_{MLE}(t)$]')
        if Ber_tp is not None:
            ax.plot(xgrid, np.real(Ber_tp[i]), color='#d73027',
                    label='Re[$f_{Ber}(t)$]')
            ax.plot(xgrid, np.imag(Ber_tp[i]), color='#fc8d59',
                    label='Im[$f_{Ber}(t)$]')
        # Set the title
        time_str = "t = {:.3f}".format(time)
        if title is not None:
            new_title = title + "; " + time_str
        else:
            new_title = time_str
        # Set the output file name
        if file_save is not None:
            file_save += "_t_{:.3f}".format(time)
        # Finalize the plot
        finalize_plot(fig, ax, xlim=xlim, ylim=ylim, title=new_title,
                      file_save=file_save, xlabel="$x$", ylabel="$f(x, t)$",
                      leg_loc=6, leg_bbox_to_anchor=(1, 0.5))
