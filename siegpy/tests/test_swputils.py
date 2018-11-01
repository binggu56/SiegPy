# -*- coding: utf-8 -*-
"""
File containing the tests for the swputils.py file.
"""

import pytest
from siegpy.swputils import q, dep, dop, dem, dom, fep, fop, fem, fom

# Initial values for the tests
k = 5+1.j
l = 3.
V_0 = 10.
qq = q(k, V_0)


def test_q():
    assert q(k, V_0) == complex(6.6754047327002546+0.7490182543549625j)


@pytest.mark.parametrize("value1, value2", [
    (dem(k, l, k2=qq), dem(k, l, V0=V_0)),
    (dep(k, l, k2=qq), dep(k, l, k2=qq)),
    (dom(k, l, k2=qq), dom(k, l, V0=V_0)),
    (dop(k, l, k2=qq), dop(k, l, V0=V_0))
])
def test_deltas(value1, value2):
    assert value1 == value2


@pytest.mark.parametrize("to_evaluate", [
    "dem(k, l)", "dom(k, l)", "dep(k, l)", "dop(k, l)"
])
def test_deltas_raise_ValueError(to_evaluate):
    with pytest.raises(ValueError):
        eval(to_evaluate)


@pytest.mark.parametrize("value1, value2", [
    (fem(k, l, k2=qq), fem(k, l, V0=V_0)),
    (fep(k, l, k2=qq), fep(k, l, k2=qq)),
    (fom(k, l, k2=qq), fom(k, l, V0=V_0)),
    (fop(k, l, k2=qq), fop(k, l, V0=V_0))
])
def test_Jost(value1, value2):
    assert value1 == value2
