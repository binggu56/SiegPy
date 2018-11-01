.. SiegPy documentation master file, created by
   sphinx-quickstart on Thu Aug 17 16:05:10 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


SiegPy: Siegert states with Python
==================================

This Python module aims at providing the tools to show how Siegert (or 
resonant) states can conveniently replace the usual continuum of unoccupied
states in some numerical problems in quantum mechanics (*e.g.*, for LR-TDDFT
or in the GW approximation in many-body perturbation theory).

.. note:: It currently focuses on the 1D Square-Well Potential (SWP) case. 
          It will be generalized to 1D potentials with compact support in 
          future releases.

.. warning:: It requires python>=3.4.

For an overview of what is possible with SiegPy, see the tutorial made of 
notebooks :ref:`here <notebooks>`. For a detailed documentation of the 
classes and their associated methods, go to :ref:`api`.


Table of contents
=================

.. toctree::
   :maxdepth: 2

   getting_started
   notebooks
   api_objects


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
