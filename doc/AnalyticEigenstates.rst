AnalyticEigenstates
--------------------

The classes documented below are the base classes you should consider
subclassing before adding a new analytical case to the SiegPy module.

The :class:`~siegpy.analyticeigenstates.AnalyticEigenstate` class is the base
class for two other classes:

* the :class:`~siegpy.analyticeigenstates.AnalyticSiegert` class,
* and the :class:`~siegpy.analyticeigenstates.AnalyticContinuum` class.

If your analytic case is already well defined by these last classes, you may
only subclass these two. However, if you need to add some methods or
attributes to both classes, you may also subclass
:class:`~siegpy.analyticeigenstates.AnalyticEigenstate`.

For instance, the parity of the analytic eigenstates in the 1D Square-Well
Potential case made it mandatory.


.. autoclass:: siegpy.analyticeigenstates.AnalyticEigenstate
    :special-members: __eq__

.. autoclass:: siegpy.analyticeigenstates.AnalyticSiegert
    :private-members:

.. autoclass:: siegpy.analyticeigenstates.AnalyticContinuum
    :private-members:

