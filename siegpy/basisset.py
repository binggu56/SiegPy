# -*- coding: utf-8 -*
r"""
The BasisSet class and its methods are defined hereafter.
"""
# TODO: It might be less confusing to store the eigenstates in an
#      actual set, while allowing to remove the append method.
# TODO: Add a create_exact_basis_set method (for AnalyticBasisSet)?


import warnings
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import cm
import numpy as np
from siegpy import Eigenstate
from siegpy.utils import init_plot, finalize_plot


__all__ = ["BasisSet", "BasisSetError"]


class BasisSet:
    r"""
    This class is arguably the most important one in the whole module
    since it defines all the main methods allowing to study the Siegert
    states of a given Hamiltonian (and compare their relevance with the
    traditional continuum states).
    """

    SIEGERT_STATES_TYPES = ["b", "ab", "r", "ar"]
    STATES_TYPES = SIEGERT_STATES_TYPES + [None, "U"]

    def __init__(self, states=None, potential=None, coord_map=None, max_virial=0):
        r"""
        Parameters
        ----------
        states: list of Eigenstate instances or Eigenstate or None
            Eigenstates of a Hamiltonian. If ``None``, it means that the
            BasisSet instance will be empty.
        potential: Potential or None
            Potential leading to the eigenstates.
        coord_map: CoordMap or None
            Coordinate mapping used to find the eigenstates.
        max_virial: float or None
            Maximal virial used to discriminate between Siegert states
            and other eigenstates.

        Raises
        ------
        ValueError
            If the value of states is invalid.
        """
        self._potential = potential
        self._coord_map = coord_map
        # If no states are passed, initialize an empty basis set
        if states is None:
            self._states = []
        # Else, if a list of states is passed, use it
        elif isinstance(states, list) and all(
            [isinstance(state, Eigenstate) for state in states]
        ):
            self._states = states
        # Else, if an only state is passed, use it
        elif isinstance(states, Eigenstate):
            self._states = [states]
        # Otherwise, states has an incorrect value
        else:
            raise ValueError("The basis set cannot be initialized: wrong states.")
        if self.coord_map is not None:
            self.max_virial = max_virial

    @property
    def potential(self):
        r"""
        Returns
        -------
        Potential
            Potential used to find the eigenstates.
        """
        return self._potential

    @property
    def coord_map(self):
        r"""
        Returns
        -------
        CoordMap
            Coordinate mapping used to find the eigenstates.
        """
        return self._coord_map

    @property
    def max_virial(self):
        r"""
        If updated, :attr:`max_virial` updates the :attr:`Siegert_type`
        attribute of the Siegert states in the basis set.

        Returns
        -------
        float
            Maximal virial for a state to be considered as a Siegert
            state.
        """
        return self._max_virial

    @max_virial.setter
    def max_virial(self, new_value):
        r"""
        The setter updates of the value of the :attr:`Siegert_type`
        attribute of the states in the basis set.

        Parameters
        ----------
        new_value: float
            New maximal virial value.
        """
        self._max_virial = abs(new_value)
        # Update the Siegert type of all the eigenstates
        for state in self.states:
            # If there is no coordinate mapping, then there are only
            # bound and continuum states
            if self.coord_map.theta == 0:
                if state.energy < 0:
                    new_type = "b"
                else:
                    new_type = None
            # If there is a coordinate mapping, then there are bound,
            # resonant and unknown states
            else:
                if abs(state.virial) <= self.max_virial:
                    en = state.energy
                    if en.real < 0:
                        new_type = "b"
                    elif en.real > 0:
                        new_type = "r"
                else:
                    new_type = "U"
            # Update the type of the considered eigenstate
            state._Siegert_type = new_type

    @property
    def states(self):
        r"""
        Returns
        -------
        list
            States of a BasisSet instance.


        Example

        >>> BasisSet().states
        []
        """
        return self._states

    def write(self, filename):
        r"""
        Write the basis set in a binary file (using pickle).

        Parameters
        ----------
        filename: str
            Name of the file to be written.


        Example

        >>> BasisSet().write("tmp.dat")
        """
        # Remove a previously existing file
        if os.path.exists(filename):
            os.remove(filename)
        # Write the states in the basis set in different binary files
        # using pickle
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_file(cls, filename):
        r"""
        Initialize a basis set from a binary file.

        Parameters
        ----------
        filename: str
            Name of a file containing a basis set.

        Raises
        ------
        NameError
            If a the file does not exist.
        """
        # Check the file exists
        if not os.path.exists(filename):
            raise NameError("The required file {} does not exist.".format(filename))
        # Read the whole file and return it
        with open(filename, "rb") as f:
            basis = pickle.load(f)
        return basis

    def __getitem__(self, index):
        r"""
        Parameters
        ----------
        index: int
            Index of an eigenstate.

        Returns
        -------
        Eigenstate
            The eigenstate of given ``index``.
        """
        return self.states[index]

    def __add__(self, states):
        r"""
        Add a list of states or the states of another BasisSet instance
        to a BasisSet instance.

        Parameters
        ----------
        states: list of Eigenstate instances or BasisSet
            Eigenstates of a Hamiltonian or another basis set.

        Returns
        -------
        BasisSet
            A new basis set.

        Raises
        ------
        TypeError
            If ``states`` is not a basis set or a list of eigenstates.
        """
        # If states is a basis set, then convert it to a list of states
        if isinstance(states, BasisSet):
            states = states.states
        # If states actually is a single state, make it iterable.
        elif not isinstance(states, list):
            states = [states]
        # Make sure that each state is a Function
        if not all([isinstance(s, Eigenstate) for s in states]):
            raise TypeError("Only wavefunctions can be added to a basis set")
        # Return a new basis set
        return self.__class__(states=self.states + states)

    def __repr__(self):
        r"""
        Returns
        -------
        str
            Representation of the list of states.
        """
        return repr(self.states)

    def __len__(self):
        r"""
        Returns
        -------
        int
            Length of the list of states
        """
        return len(self.states)

    def __eq__(self, other):
        r"""
        Returns
        -------
        bool
            ``True`` if both basis sets contain the same states.
        """
        return (
            isinstance(other, BasisSet)
            and len(self) == len(other)
            and all([state in self for state in other])
        )

    @property
    def is_empty(self):
        r"""
        Returns
        -------
        bool
            ``True`` if the basis set is empty.
        """
        return len(self) == 0

    @property
    def is_not_empty(self):
        r"""
        Returns
        -------
        bool
            ``True`` if the basis set is not empty.
        """
        return not self.is_empty

    @property
    def bounds(self):
        r"""
        Returns
        -------
        BasisSet
            Basis set made of all the bound states of the current basis
            set.
        """
        return self.__class__(
            states=[state for state in self if state.Siegert_type == "b"]
        )

    @property
    def antibounds(self):
        r"""
        Returns
        -------
        BasisSet
            Basis set made of all the anti-bound states of the current
            basis set.
        """
        return self.__class__(
            states=[state for state in self if state.Siegert_type == "ab"]
        )

    @property
    def resonants(self):
        r"""
        Returns
        -------
        BasisSet
            Basis set made of all the resonant states of the current
            basis set.
        """
        return self.__class__(
            states=[state for state in self if state.Siegert_type == "r"]
        )

    @property
    def antiresonants(self):
        r"""
        Returns
        -------
        BasisSet
            Basis set made of all the anti-resonant states of the
            current basis set.
        """
        return self.__class__(
            states=[state for state in self if state.Siegert_type == "ar"]
        )

    @property
    def siegerts(self):
        r"""
        Returns
        -------
        BasisSet
            Basis set made of all the Siegert states of the current
            basis set.
        """
        return self.bounds + self.antibounds + self.resonants + self.antiresonants

    @property
    def continuum(self):
        r"""
        Returns
        -------
        BasisSet
            Basis set made of all the continuum states of the current
            basis set.
        """
        return self.__class__(
            states=[state for state in self if state.Siegert_type is None]
        )

    @property
    def unknown(self):
        r"""
        Returns
        -------
        BasisSet
            Basis set made of all the states of unknown type of the
            current basis set.
        """
        return BasisSet(states=[state for state in self if state.Siegert_type == "U"])

    def plot_wavefunctions(
        self, nres=None, nstates=None, xlim=None, ylim=None, title=None, file_save=None
    ):  # pragma: no cover
        r"""
        Plot the bound, resonant and anti-resonant wavefunctions of the
        basis set along with the potential. The continuum states, if any
        in the basis set, are not plotted.

        The wavefunctions are translated along the y-axis by their
        energy (for bound states) or absolute value of their
        energy (for resonant and anti-resonant states).

        Parameters
        ----------
        nres: int
            Number of resonant wavefunctions to plot (optional).
        nstates: int
            Number of wavefunctions to plot (optional).
        xlim: tuple(float or int, float or int)
            Range of the x axis of the plot (optional).
        xlim: tuple(float or int, float or int)
            Range of the y axis of the plot (optional).
        title: str
            Plot title (optional).
        file_save: str
            Filename of the plot to be saved (optional).
        """
        # Object oriented plot
        fig, ax = init_plot()
        # Plot only part of the states, if asked by the user
        if nres is not None:
            states = self.bounds + self.resonants[:nres]
        elif nstates is not None:
            states = self.states[:nstates]
        else:
            msg = "There may be too many wavefunctions to plot for readability"
            states = self.states
            warnings.warn(msg)
        # Loop over the Siegert states
        for i, wf in enumerate(states):
            # Define the value of its energy given its type
            energy = np.real(wf.energy)
            # Define the labels
            label_re, label_im, label_en = None, None, None
            if i == 0:
                label_re, label_im, label_en = "Re[WF]", "Im[WF]", "Energy"
            # Add the wf (real and imaginary parts) in the plot
            ax.plot(wf.grid, np.real(wf.values) + energy, "b", label=label_re)
            ax.plot(wf.grid, np.imag(wf.values) + energy, "r", label=label_im)
            # Also plot the value of its ("translation") energy.
            ax.plot(wf.grid, np.array([energy] * len(wf.grid)), "k--", label=label_en)
        # Plot the potential as well
        pot = self.potential
        if pot is not None:
            ax.plot(pot.grid, np.real(pot.values), "k-", label="Re[Potential]")
            ax.plot(
                pot.grid,
                np.imag(pot.values),
                c="k",
                ls="dotted",
                ms=1,
                label="Im[Potential]",
            )
        finalize_plot(
            fig,
            ax,
            xlim=xlim,
            ylim=ylim,
            title=title,
            file_save=file_save,
            leg_loc=6,
            leg_bbox_to_anchor=(1, 0.5),
            xlabel="$x$",
            ylabel="Energy",
        )

    @property
    def energies(self):
        r"""
        Returns
        -------
        list
            The energies of the states in the current basis set.
        """
        return [state.energy for state in self]

    @property
    def wavenumbers(self):
        r"""
        Returns
        -------
        list
            The wavenumbers of the states in the curent basis set.
        """
        return [state.wavenumber for state in self]

    @property
    def virials(self):
        r"""
        Returns
        -------
        list
            Virial values of the states in the current basis set.
        """
        return [state.virial for state in self]

    @property
    def no_coord_map(self):
        r"""
        Returns
        -------
        bool
            ``True`` if the coordinate mapping is such that
            :math:`x \mapsto x`.
        """
        if self.coord_map is not None:
            return self.coord_map.theta == 0
        else:
            return False

    def plot_wavenumbers(
        self, xlim=None, ylim=None, title=None, file_save=None, show_unknown=True
    ):  # pragma: no cover # noqa
        r"""
        Plot the wavenumbers of the Siegert states in the basis set.

        Parameters
        ----------
        xlim: tuple(float or int, float or int)
            Range of the x axis of the plot (optional).
        xlim: tuple(float or int, float or int)
            Range of the y axis of the plot (optional).
        title: str
            Plot title (optional).
        file_save: str
            Filename of the plot to be saved (optional).
        show_unknown: bool
            If ``True``, plot the data of the states with an unknown
            type.
        """
        _complex_plane_plot(
            self, "wavenumber", xlim, ylim, title, file_save, show_unknown
        )

    def plot_energies(
        self, xlim=None, ylim=None, title=None, file_save=None, show_unknown=True
    ):  # pragma: no cover
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
        show_unknown: bool
            If ``True``, plot the data of the states with an unknown
            type.
        """
        _complex_plane_plot(self, "energy", xlim, ylim, title, file_save, show_unknown)

    def scal_prod(self, test):
        r"""
        Method computing the scalar product
        :math:`\left\langle \varphi | test \right\rangle`
        for each state :math:`\varphi` in the basis set.

        Parameters
        ----------
        test: Function
            Test function.

        Returns
        -------
        numpy array
            Scalar product of all the states in the basis set with the
            test function.
        """
        return np.array([state.scal_prod(test) for state in self])

    def completeness_convergence(self, test, klim=None):
        r"""
        Evaluate the convergence of the completeness relation for the
        current basis set using a given test function.

        Parameters
        ----------
        test: Function
            Test function.
        klim: tuple(float or int, float or int)
            Wavenumber range where the completeness convergence must be
            computed (optional).

        Returns
        -------
        tuple(numpy array, numpy array)
            A tuple made of the array of the states wavenumbers and the
            array of the convergence of the completeness relation using
            all the states (both have the same length).

        Raises
        ------
        ValueError
            If the wavenumber ``klim`` range covers negative values.
        """
        init = self.bounds + self.unknown
        if self.no_coord_map:
            # Usual scalar product
            sp_init = init.scal_prod(test)
            init_val = np.sum(np.abs(sp_init) ** 2) / test.norm()
            basis = self.continuum
            basis.states.sort(key=lambda x: x.energy.real)
            scal_prods = basis.scal_prod(test)
            CR_conv = np.cumsum(np.abs(scal_prods) ** 2) / test.norm()
        else:
            # c-product because some states have complex energies
            sp_init_1 = np.conjugate(init.scal_prod(test.conjugate()))
            sp_init_2 = np.conjugate(init.scal_prod(test))
            init_val = np.sum(sp_init_1 * sp_init_2) / test.norm()
            basis = self.resonants
            basis.states.sort(key=lambda x: x.energy.real)
            scal_prods_1 = np.conjugate(basis.scal_prod(test.conjugate()))
            scal_prods_2 = np.conjugate(basis.scal_prod(test))
            CR_conv = np.cumsum(scal_prods_1 * scal_prods_2) / test.norm()
        CR_conv = np.insert(CR_conv, 0, 0) + init_val
        # Keep only the results in the desired wavenumber range
        real_en = np.insert(np.real(basis.energies), 0, 0)
        if klim is not None:
            # The range is defined by xlim
            if klim[0] < 0 or klim[1] < 0:
                raise ValueError("The limits should be positive.")
            where = np.logical_and(
                real_en >= klim[0] ** 2 / 2, real_en <= klim[1] ** 2 / 2
            )
            CR_conv = CR_conv[where]
        else:
            # The x-axis range starts from the 0 energy/wavenumber
            where = np.where(real_en >= 0)
            CR_conv = CR_conv[where]
        kgrid = np.insert(basis.wavenumbers, 0, 0)
        kgrid = np.abs(kgrid[where])
        return kgrid, CR_conv

    def Berggren_completeness_convergence(self, test, klim=None):
        r"""
        Method evaluating the convergence of the CR using the Berggren
        expansion.

        Parameters
        ----------
        test: Function
            Test function.
        xlim: tuple(float or int, float or int)
            Wavenumber range where the completeness convergence must be
            computed (optional).

        Returns
        -------
        tuple(numpy array, numpy array)
            A tuple made of the array of the states wavenumbers
            and the array of the convergence of the Berggren
            completeness relation (both have the same length).
        """
        basis = self.bounds + self.resonants
        kgrid, CR_conv = basis.completeness_convergence(test, klim=klim)
        return kgrid, CR_conv

    def plot_completeness_convergence(
        self, test, klim=None, title=None, file_save=None
    ):  # pragma: no cover
        r"""
        Plot the convergence of the completeness relation using all or
        the fiest ``nstates`` in the basis set.

        Parameters
        ----------
        test: Function
            Test function.
        klim: tuple(float or int, float or int)
            Wavenumber range where the completeness convergence must be
            computed and range of the x axis of the plot (optional).
        title: str
            Plot title (optional).
        file_save: str
            Filename of the plot to be saved (optional).
        """
        # Object-oriented plots
        fig, ax = init_plot()
        # Plot the expected value of 1
        ax.axhline(1, color="black", lw=1.5)
        kgrid, CR_conv = self.completeness_convergence(test, klim=klim)
        ax.plot(kgrid, np.real(CR_conv), color="#d73027", ls="", marker=".", ms=10)
        # Finalize the plot
        if klim is None:
            klim = (0, kgrid[-1])
        finalize_plot(
            fig,
            ax,
            xlim=klim,
            title=title,
            file_save=file_save,
            xlabel="$k$",
            ylabel="CR",
        )

    def MLE_strength_function(self, test, kgrid):
        r"""
        .. warning::

            Only the peaks due to the resonant couples can be produced
            at the moment. Numerical anti-bound states are required for
            the true MLE of the strength function to be computed from a
            numerical basis set.

        Evaluate the Mittag-Leffler expansion strength function of the
        basis set for a given test function, discretized on a grid of
        wavenumbers kgrid.

        Parameters
        ----------
        test: Function
            Test function.
        kgrid: numpy array
            Wavenumbers for which the strength function is evaluated.

        Returns
        -------
        numpy array
            MLE of the strength function evaluated on the kgrid.
        """
        # Initialize strength_func
        strength_func = 1.0j * np.zeros_like(kgrid)
        # Loop over the siegert states to update the MLE of the RF
        for state in self.resonants:
            # Values for the resonant state contribution
            k_r = state.wavenumber
            sp_r = state.scal_prod(test)
            # Values for the corresponding anti-resonant state contribution
            ares = Eigenstate(
                state.grid,
                np.conjugate(state.values),
                np.conjugate(state.energy),
                Siegert_type="ar",
            )
            ares._wavenumber = -np.conjugate(k_r)
            k_ar = ares.wavenumber
            sp_ar = ares.scal_prod(test)
            # Add the contributions of the resonant couple
            for k, sp in [(k_r, sp_r), (k_ar, sp_ar)]:
                strength_func += -1.0 / np.pi * sp ** 2 / (k * (kgrid - k))
        return np.imag(strength_func)

    def plot_strength_function(
        self, test, kgrid, nres=None, title=None, file_save=None
    ):  # pragma: no cover
        r"""
        Plot the Mittag-Leffler Expansion of the strength function for a
        given test function.

        The MLE of the strength function evaluated using 1, ...,
        ``nres`` resonant couples can also be plotted.

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
        title: str
            Plot title (optional).
        file_save: str
            Filename of the plot to be saved (optional).
        """
        # Object-oriented plots
        fig, ax = init_plot()
        # Evaluate and plot the MLE of the RF
        MLE_rf = self.MLE_strength_function(test, kgrid)
        ax.plot(kgrid, MLE_rf, color="#000000", ls="--", label="MLE")
        # Plot the contribution of the nres first resonance couples
        # to the strength function
        if nres is not None:
            res = self.resonants
            res.states.sort(key=lambda x: x.energy.real)
            res = res[:nres]
            for i, state in enumerate(res):
                rf = BasisSet(state).MLE_strength_function(test, kgrid)
                label = "$N_{res} = $" + "{}".format(i + 1)
                ax.plot(kgrid, rf, lw=1.5, label=label)
        # Finalize the plot
        if nres is not None:
            leg_loc = 6
            leg_bbox_to_anchor = (1, 0.5)
        else:
            leg_loc = None
            leg_bbox_to_anchor = None
        finalize_plot(
            fig,
            ax,
            title=title,
            file_save=file_save,
            leg_loc=leg_loc,
            leg_bbox_to_anchor=leg_bbox_to_anchor,
            xlabel="$k$",
            ylabel="$S(k)$",
        )

    def Berggren_propagation(self, test, time_grid):
        r"""
        Evaluate the Berggren Expansion of the time-propagation of a
        test wavepacket over a given time grid. Only bound and resonant
        states are used, with a weight 1.

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
        # Given that the bound and resonant states are the only Siegert
        # states that can be found numerically, it is sufficient to use
        # all the Siegert states to compute the time propagation
        return self.siegerts._propagate(test, time_grid)

    def _propagate(self, test, time_grid):
        r"""
        Evaluate the contribution of all the states to the time
        propagation of the initial wavepacket ``test`` for all the times
        in ``time_grid``.

        .. note::

            The same default weight is used for all the states.

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
            Propagated wavepacket for the different times of ``time_grid``.
        """
        mat_time = _set_mat_time(self, time_grid)
        mat_space = _set_mat_space(self, test)
        return mat_time.dot(mat_space)


def _complex_plane_plot(
    basis, attr, xlim, ylim, title, file_save, show_unknown
):  # pragma: no cover
    r"""
    Plot either the wavenumber or the energy of each bound, resonant
    continuum and unknown states in the basis set (the latter if
    ``show_unknown`` is set to ``True``).

    Parameters
    ----------
    basis: BasisSet
        Basis set made of the states to be plotted.
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
    show_unknown: bool
        If ``True``, plot the data of the states with an unknown
        type.
    """
    # Object-oriented plots
    fig, ax = init_plot()
    # Add the real and imaginary axes
    ax.axhline(0, color="black", lw=1)  # black line y=0
    ax.axvline(0, color="black", lw=1)  # black line x=0
    # Select which states have to be plotted
    to_plot = basis.bounds + basis.resonants + basis.continuum
    if show_unknown is True or len(basis.unknown) == len(basis):
        to_plot += basis.unknown
    # Store the data to plot (energy or wavenumber) and their virial
    plot_data = [getattr(state, attr) for state in to_plot][::-1]
    virials = np.log10(to_plot.virials[::-1])
    # Set the min and max values of the virial colors
    vmin = np.min(np.log10(basis.virials))
    vmax = np.max(np.log10(basis.virials))
    # Normalize the colors used accordingly
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    # Set the colorbar and its title
    colorbar = cm.ScalarMappable(cmap=plt.cm.CMRmap, norm=norm)  # pylint: disable=E1101
    colorbar.set_array([])
    plt.colorbar(colorbar, label="$log_{10}$(|virial|)")
    # Plot all the data
    if plot_data != []:
        color = plt.cm.CMRmap(norm(virials))  # pylint: disable=E1101
        ax.scatter(np.real(plot_data), np.imag(plot_data), c=color, cmap="CMRmap")
    # Set the base name of the x and y axis of the plot
    if attr == "wavenumber":
        data_label = "k"
    elif attr == "energy":
        data_label = "E"
    xlabel = "Re[${}$]".format(data_label)
    ylabel = "Im[${}$]".format(data_label)
    # Finalize the plot
    finalize_plot(
        fig,
        ax,
        xlim=xlim,
        ylim=ylim,
        title=title,
        file_save=file_save,
        xlabel=xlabel,
        ylabel=ylabel,
    )


def _set_mat_time(basis, time_grid):
    r"""
    Evaluate a matrix mat_time of size :math:`n_t * n_k` where
    :math:`n_k` is the number of states in the basis set ``basis`` and
    :math:`n_t` is the length of the array ``time_grid``.

    Each element of the matrix is defined by:

    .. math::

       \text{mat_time}[i_t][i_k] = e^{-i \text{energy}[i_k]
       \text{time_grid}[i_t]}

    where :math:`i_k` is a counter for the states of the basis set
    and :math:`i_t` is a counter for the points of the time grid.

    This matrix can be used to compute the time-propagation of an
    initial wavepacket.

    Parameters
    ----------
    basis: BasisSet
        Basis set.
    time_grid: numpy array or list of positive numbers
        Times for which the propagation is evaluated. It must
        contain positive numbers only.

    Returns
    -------
    2D numpy array
        Matrix mat_time based on the states of the basis set.
    """
    # Build the matrix mat_time
    E, T = np.meshgrid(basis.energies, time_grid)
    mat_time = np.exp(-1.0j * E * T)
    return mat_time
    # return np.array(mat_time)


def _set_mat_space(basis, test, weight=1.0):
    r"""
    Evaluate a matrix mat_space of size :math:`n_k * n_x` where
    :math:`n_k` is the number of states in the basis set ``basis`` and
    :math:`n_x` is the length of the space grid on which the states of
    the basis set are discretized.

    Each element of the matrix is defined by:

    .. math::

       \text{mat_space}[i_k][i_x] = w \left\langle phi_{i_k} |
       \text{test} \right\rangle phi_{i_k}[i_x]

    where :math:`w` is the weight, :math:`i_k` is a counter for the
    states of the basis set and :math:`i_x` is a counter for the
    space grid-points.

    This matrix can then be used to compute the time-propagation of an
    initial wavepacket.

    Parameters
    ----------
    basis: BasisSet
        Basis set.
    test: Function
        Test function.
    weigth: float or int or complex
        Mutliplying factor used for all the states (default to 1.0).

    Returns
    -------
    2D numpy array
        Matrix mat_space based on the states of the basis set.

    Raises
    ------
    BasisSetError
        If the basis set is empty.
    """
    # Check that the basis set is not empty
    if basis.is_empty:
        raise BasisSetError("The basis set must not be empty")
    # Build the matrix mat_space
    if weight == 0:
        # If the weigth is zero, the result is obviously a matrix
        # made of zeros
        shape = (len(basis), len(basis[0].values))
        return np.zeros(shape)
    else:
        mat_space = [weight * state.scal_prod(test) * state.values for state in basis]
        return np.array(mat_space)


class BasisSetError(Exception):
    r"""
    Error thrown in the case of an ill-defined BasisSet instance.
    """
    pass
