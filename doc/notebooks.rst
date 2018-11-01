.. _notebooks:

Tutorial
--------

This tutorial is made of notebooks that are available in the
`Siegpy/doc/notebooks` folder after cloning the project. The aim of these
series of notebooks is to present the most relevant capabilities offered by
the SiegPy module while giving an overview of the Siegert states and how
useful they can be in order to reproduce results obtained using the
cumbersome continuum of states.

1D Square-Well Potential: the analytical case
=============================================

.. toctree::
   :maxdepth: 1
   :caption: First steps

   notebooks/find_Siegerts_1DSWP


.. toctree::
   :maxdepth: 1
   :caption: Completeness relation in the 1D Square-well Potential case

   notebooks/completeness_relation
   notebooks/completeness_relation_problematic_cases
   notebooks/completeness_relation_multiple_test_functions


.. toctree::
   :maxdepth: 1
   :caption: Strength function

   notebooks/strength_function
   notebooks/convergence_MLE_strength_function


.. toctree::
   :maxdepth: 1
   :caption: Time propagation of a wavepacket

   notebooks/time_propagation
   notebooks/time_propagation_initial_momentum
   notebooks/time_propagation_error_estimation



1D Square-Well Potential: the numerical case
============================================

.. toctree::
   :maxdepth: 1
   :caption: Numerical bound and continuum states

   notebooks/find_numerical_bound_and_continuum.ipynb
   notebooks/completeness_relation_numerical_bound_and_continuum.ipynb


.. toctree::
   :maxdepth: 1
   :caption: Numerical Siegert states

   notebooks/find_numerical_Siegert.ipynb
   notebooks/completeness_relation_numerical_Siegert.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Influence of some parameters on the numerical basis sets

   notebooks/influence_virials_siegerts.ipynb
   notebooks/influence_filters_siegerts.ipynb
   notebooks/influence_coordMap_siegerts_1.ipynb
   notebooks/influence_grid_continuum.ipynb
   notebooks/influence_grid_siegerts.ipynb



Other numerical potentials
==========================

.. toctree::
   :maxdepth: 1

   notebooks/woods-saxon_potential.ipynb
   notebooks/gaussian_potentials.ipynb
   notebooks/other_symbolic_potentials.ipynb
   notebooks/numerical_strength_function.ipynb
