r"""
The main classes and objects of the SiegPy module are made available via
a simple from siegpy import.

Also, some default parameters are set for the rendering of matplotlib.
"""
import matplotlib as mpl
from .functions import Gaussian, Rectangular
from .potential import Potential
from .symbolicpotentials import (SymbolicPotential, WoodsSaxonPotential,
                                 TwoGaussianPotential,
                                 FourGaussianPotential)
from .swpotential import SWPotential
from .eigenstates import Eigenstate
from .swpeigenstates import SWPSiegert, SWPContinuum
from .basisset import BasisSet, BasisSetError
from .swpbasisset import SWPBasisSet
from .filters import FD2_filters, FD8_filters, Sym8_filters
from .coordinatemappings import (ErfKGCoordMap, TanhKGCoordMap,
                                 ErfSimonCoordMap, TanhSimonCoordMap,
                                 UniformCoordMap)
from .hamiltonian import Hamiltonian

# Set default values for matplotlib
mpl.rcParams['lines.linewidth'] = 2.5  # line width in points
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.labelsize'] = 20  # fontsize of the x any y labels
mpl.rcParams['xtick.labelsize'] = 14  # fontsize of the tick labels
mpl.rcParams['ytick.labelsize'] = 14  # fontsize of the tick labels
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['axes.titlesize'] = 24  # size of the figure title
mpl.rcParams['savefig.bbox'] = 'tight'  # size of the figure title
