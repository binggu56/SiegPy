AnalyticBasisSet
----------------

This abstract base class is meant to implement the main methods available to
all basis set classes that are specific to an analytical case. This ensures
that all analytical cases have the same basic API, while avoiding code
repetition.

As an example, it is used to define the
:class:`~siegpy.swpbasisset.SWPBasisSet` class, which is the specific basis set
class for the analytical 1D Square-Well potential case. You might notice that
only a few methods were implemented there.

.. autoclass:: siegpy.analyticbasisset.AnalyticBasisSet
