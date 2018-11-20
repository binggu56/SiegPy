Getting started
---------------

Installation
^^^^^^^^^^^^

To install SiegPy, run the following commands:

* ``git clone https://github.com/mmoriniere/SiegPy.git``
* ``cd SiegPy``;
* ``pip install .``

.. warning:: 

    * Make sure that ``pip`` corresponds to a python3.4+ version.
      If not, you may use ``pip3`` instead, or create an alias with bash.

Usage
^^^^^

SiegPy provides the tools to create basis sets made of Siegert states.
It is then interesting to see how they compares to an exact basis set 
(made of bound and continuum states) when it comes to compute the following
quantities of physical interest:

* the **completeness relation**, showing that both types of basis sets can be
  considered as complete. This is the probably the most fundamental quantity.
* the **response function**, *i.e.* an approximation of the Green's function
  with discrete states),
* the **time-propagation** of an initial wavepacket.

You can find some case studies :ref:`here <notebooks>`.


Documentation
^^^^^^^^^^^^^

The present documentation can be compiled locally on your computer:

* You must have ipython installed, with python3 as a kernel 
  (run ``ipython3 kernel install`` for the latter point).
* Run ``pip install .[doc]`` in the top folder. This will install the 
  required packages for the documentation to be compiled.
* Go to the `doc` folder and run ``make html`` 
  (or alternatively ``python -m sphinx . build -jN``, where N is
  the number of precessors used for the compilation). The root `index.html` of
  the documentation will then be located in the `build` folder.

.. warning:: Also make sure that ``python`` corresponds to a python3.4+ version.


For developpers
^^^^^^^^^^^^^^^

SiegPy should always be fully tested, with a code coverage around 100%.
pytest and some related packages are used to that end. To ensure that you
can run the tests and build the documentation, run the command 
``pip install -e .[test,doc]``.
