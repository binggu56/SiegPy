# -*- coding: utf-8 -*
r"""
The Hamiltonian class and its methods are defined hereafter.

The main use of this class is to create a basis set made of the
eigenstates of a numerical Hamiltonian, when given a potential and a
coordinate mapping.
"""
import numpy as np
import scipy.linalg as LA
from siegpy import BasisSet, Eigenstate, UniformCoordMap, Sym8_filters


class Hamiltonian():
    r"""
    A Hamiltonian has to be defined when one is interested in finding
    numerically the Siegert states of a potential when the eigenstates
    are not known analytically.
    """

    def __init__(self, potential, coord_map, filters=Sym8_filters):
        r"""
        A Hamiltonian is defined by a potential and a coordinate
        mapping, which gives rise to extra potentials to be added, known
        as the Reflection-Free Complex Absorbing Potentials (RF-CAP).

        Filters allowing to define the gradient and laplacian operators
        are also required. The default value corresponds to Daubechies
        Wavelet filters.

        Parameters
        -----------
        potential: Potential
            Potential studied.
        coord_map: CoordMap
            Coordinate mapping used.
        filters: Filters
            Filters used to define the Hamiltonian matrix.
        """
        self._potential = potential
        # Make sure that the coordinate mapping uses the correct grid
        grid = potential.grid
        coord_map.grid = grid
        self._coord_map = coord_map
        # Define the filters according to the grid
        self._filters = filters
        npts = len(grid)
        self._gradient_matrix = filters.fill_gradient_matrix(npts)
        self._laplacian_matrix = filters.fill_laplacian_matrix(npts)
        if hasattr(filters, 'magic_filter'):
            self._magic_matrix = filters.fill_magic_matrix(npts)
        else:
            self._magic_matrix = None
        # Set the Hamiltonian and virial operator matrices
        self._matrix = self._find_hamiltonian_matrix()
        self._virial_matrix = self._find_virial_matrix()

    @property
    def potential(self):
        r"""
        Returns
        -------
        Potential
            Potential considered.
        """
        return self._potential

    @property
    def coord_map(self):
        r"""
        Returns
        -------
        CoordMap
            Complex scaling considered.
        """
        return self._coord_map

    @property
    def filters(self):
        r"""
        Returns
        -------
        Filters
            Filters used to define the matrices.
        """
        return self._filters

    @property
    def gradient_matrix(self):
        r"""
        Returns
        -------
        2D numpy array
            Gradient matrix used to define the matrices.
        """
        return self._gradient_matrix

    @property
    def laplacian_matrix(self):
        r"""
        Returns
        -------
        2D numpy array
            Laplacian matrix used to define the matrices.
        """
        return self._laplacian_matrix

    @property
    def magic_matrix(self):
        r"""
        Returns
        -------
        2D numpy array
            Magic filter matrix used to define the matrices.
        """
        return self._magic_matrix

    @property
    def matrix(self):
        r"""
        Returns
        -------
        2D numpy array
            Hamiltonian matrix.
        """
        return self._matrix

    @property
    def virial_matrix(self):
        r"""
        Returns
        -------
        2D numpy array
            Virial operator matrix.
        """
        return self._virial_matrix

    def _find_hamiltonian_matrix(self):
        r"""
        Evaluate the Hamiltonian matrix.

        Returns
        -------
        2D numpy array
            Hamiltonian matrix.
        """
        # Get initial values for the operators
        grad = self.gradient_matrix
        laplac = self.laplacian_matrix
        # Get the grid spacing
        grid = self.potential.grid
        h = grid[1] - grid[0]
        # Set the values of the RF-CAP
        cm = self.coord_map
        if isinstance(cm, UniformCoordMap):
            VF = self.potential.complex_scaled_values(cm)
            ham = self._pot_to_mat(VF)
            tmp = - np.exp(-2j*cm.theta) * np.eye(len(VF)) / 2
            ham += np.dot(tmp, laplac/h**2)
        else:
            # Use of a smooth exterior complex scaling
            VF = self.potential.complex_scaled_values(cm)
            V0 = cm.V0_values
            V1 = cm.V1_values
            V2 = cm.V2_values
            # Create the Hamiltonian by converting the potential to matrices
            # and multiplying them by the appropriate operator
            tmp = self._pot_to_mat(V2 - 1/2)  # add the kinetic term to V2
            ham = np.dot(tmp, laplac/h**2)
            tmp = self._pot_to_mat(V1)
            ham += np.dot(tmp, grad/h)
            # The operator applied to VF and V0 should be the identity,
            # hence the matrices are already correct
            ham += self._pot_to_mat(V0+VF)
        return ham

    def _find_virial_matrix(self):
        r"""
        Evaluate the virial operator matrix.

        Returns
        -------
        2D numpy array
            Virial operator matrix.
        """
        cm = self.coord_map
        # Set the values of the virial potentials
        U0 = cm.U0_values
        U1 = cm.U1_values
        U2 = cm.U2_values
        U11 = cm.U11_values
        # Create the virial matrix
        if cm.GCVT:
            vir = self._build_virial_matrix(U0, U1, U2, U11)
        else:
            # In this case, there is one virial operator per parameter
            vir = {xi: self._build_virial_matrix(U0[xi], U1[xi],
                                                 U2[xi], U11[xi]) for xi in U0}
            # Add the potential term to U0 (i.e.: V'(F)*dF/dxi)
            pot = self.potential
            for xi in vir:
                vir[xi] += pot.complex_scaled_values(cm) * cm.dxi_values[xi]
        return vir

    def _build_virial_matrix(self, U0, U1, U2, U11):
        r"""
        Build the matrix operator given a set of virial potentials.

        Parameters
        -----------
        U0: numpy array
            First virial potential.
        U1: numpy array
            Second virial potential.
        U2: numpy array
            Thirs virial potential.
        U11: numpy array
            Fourth virial potential.

        Returns
        -------
        2D numpy array
            Virial operator as a matrix.
        """
        # Get initial values for the operators
        grad = self.gradient_matrix
        laplac = self.laplacian_matrix
        # Get the grid spacing h
        grid = self.potential.grid
        h = grid[1] - grid[0]
        # Create the virial matrix by converting the potentials to
        # matrices and then multiplying them by the appropriate operator
        vir = self._pot_to_mat(U0)
        tmp = self._pot_to_mat(U1)
        vir += np.dot(tmp, grad/h)
        if self.coord_map.GCVT:
            vir += np.dot(grad.T/h, tmp)
        else:
            tmp = self._pot_to_mat(U11)
            vir += np.dot(grad.T/h, np.dot(tmp, grad/h))
            tmp = self._pot_to_mat(U2)
            vir = np.dot(tmp, laplac/h**2)
        return vir

    def solve(self, max_virial=None):
        r"""
        Find the eigenstates of the potential and evaluate the virial
        for each of them. This is the main method of the Hamiltonian
        class.

        Parameters
        -----------
        max_virial
            Maximal virial value for a state to be considered as a
            Siegert state.

        Returns
        -------
        BasisSet
            Basis set made of the eigenstates of the Hamiltonian.
        """
        # Set initial values
        grid = self.potential.grid
        h = grid[1] - grid[0]
        # The maximal virial value must be positive
        if max_virial is None:
            max_virial = 0
        # Solve the Hamiltonian
        energies, vr = LA.eig(self.matrix)
        # energies, vl, vr = LA.eig(self.matrix, left=True)
        # Initialize a list (to be made of Eigenstate instances)
        # ultimately used to initialize a BasisSet instance
        states = []
        # Loop over the eigenstates
        for i, energy in enumerate(energies):
            # Normalize the right eigenvector
            vr_i = vr[:, i]
            nrmr = np.dot(vr_i, vr_i)
            vr_i /= np.sqrt(nrmr)
            # vl_i = vl[:, i]
            # nrml = np.dot(vl_i, vl_i)
            # vl_i /= np.sqrt(nrml)
            # Compute the virial associated to the eigenvector
            if self.coord_map.GCVT:
                virial = np.abs(np.dot(vr_i, np.dot(self.virial_matrix, vr_i)))
            else:
                matrices = self.virial_matrix
                virials = {xi: np.dot(vr_i, np.dot(matrices[xi], vr_i))
                           for xi in matrices}
                virial_values = np.array([virials[xi] for xi in virials])
                virial = np.sqrt(np.sum(np.abs(virial_values**2)))
                # print(energy, virial)
                # virial = np.dot(vl_i, np.dot(self.virial_matrix, vr_i))
            # Add the eigenstate to the list of states
            states.append(
                Eigenstate(
                    grid, vr_i/np.sqrt(h), energy, 'U', virial=virial))
        # Sort the list of states by increasing virial values
        states.sort(key=lambda x: np.log10(x.virial))
        return BasisSet(states=states, potential=self.potential,
                        coord_map=self.coord_map, max_virial=max_virial)

    def _pot_to_mat(self, potential_values):
        r"""
        Convert the potential values to a diagonal matrix.

        Parameters
        -----------
        potential_values: numpy array
            Values of a potential.

        Returns
        -------
        2D numpy array
            Potential as a diagonal matrix.
        """
        # Initialize the potential matrix as a diagonal matrix
        npts = len(potential_values)
        mat = potential_values * np.eye(npts)
        # Apply the magic filter, if necessary
        magic = self.magic_matrix
        if magic is not None:
            tmp = np.dot(mat, magic.T)
            mat = np.dot(magic, tmp)
        return mat
