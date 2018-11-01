# -*- coding: utf-8 -*-
"""
This file contains utilitary functions concerning the
One-Dimensional Square Well Potential (1DSWP) case.
"""

import numpy as np
from .analyticeigenstates import WavenumberError


__all__ = ["q", "dep", "dem", "dop", "dom", "fep", "fem", "fop", "fom",
           "find_parity"]


def q(k, V0):
    r"""
    Evaluate the wavenumber in region *II*, given by:
    :math:`q = \sqrt{k**2 + 2 V_0}`, where :math:`k` is the wavenumber
    in region *I* or *III* and :math:`V_0` is the depth of the
    potential.

    Parameters
    ----------
    k: complex or float
        Wavenumber (in region *I* or *III*) of the state considered.
    V0: float
        Depth of the potential.

    Returns
    -------
    complex or float
        Wavenumber in region *II* of the state considered.
    """
    return np.sqrt(k**2 + 2. * V0)


def dem(k, l, k2=None, V0=None):
    r"""
    Evaluate one of the four delta functions found in the 1DSWP case for
    a given wavenumber ``k``. ``dem`` stands for :math:`\delta_e^-` (see
    paper).

    Parameters
    ----------
    k: complex or float
        Wavenumber (in region *I* or *III*) of the state considered.
    l: float
        Width of the potential.
    k2: complex or float
        Wavenumber in region *II* of the state considered.
    V0: float
        Depth of the potential (optional, set to None).

    Returns
    -------
    complex
        Value of :math:`\delta_e^-`.

    Raises
    ------
    ValueError
        If the wavenumber in region *II* cannot be evaluated.
    """
    if V0:
        qq = q(k, V0)
    elif k2:
        qq = k2
    else:
        raise ValueError("qq cannot be computed.")
    return - k * np.cos(qq * l / 2.) - 1.j * qq * np.sin(qq * l / 2.)


def dep(k, l, k2=None, V0=None):
    r"""
    Evaluate one of the four delta functions found in the 1DSWP case for
    a given wavenumber ``k``. ``dep`` stands for :math:`\delta_e^+` (see
    paper).

    See :function:`dem` for more details about the input variables.

    Returns
    -------
    complex
        Value of :math:`\delta_e^+`.
    """
    return dem(-complex(k), l, k2, V0)


def dom(k, l, k2=None, V0=None):
    r"""
    Evaluate one of the four delta functions found in the 1DSWP case for
    a given wavenumber ``k``. ``dom`` stands for :math:`\delta_o^-` (see
    paper).

    See :function:`dem` for more details about the input variables.

    Returns
    -------
    complex
        Value of :math:`\delta_o^-`.

    Raises
    ------
    ValueError
        If the wavenumber in region *II* cannot be evaluated.
    """
    if V0:
        qq = q(k, V0)
    elif k2:
        qq = k2
    else:
        raise ValueError("qq cannot be computed.")
    return - k * np.sin(qq * l / 2.) + 1.j * qq * np.cos(qq * l / 2.)


def dop(k, l, k2=None, V0=None):
    r"""
    Evaluate one of the four delta functions found in the 1DSWP case for
    a given wavenumber ``k``. ``dop`` stands for :math:`\delta_o^+` (see
    paper).

    See :function:`dem` for more details about the input variables.

    Returns
    -------
    complex
        Value of :math:`\delta_o^+`.
    """
    return dom(-complex(k), l, k2, V0)


def fep(k, l, k2=None, V0=None):
    r"""
    Evaluate one of the four Jost functions found in the 1DSWP case for
    a given wavenumber ``k``. ``fep`` stands for :math:`f_e^+` (see
    paper).

    See :function:`dem` for more details about the input variables.

    Returns
    -------
    complex
        Value of :math:`f_e^+`.
    """
    return dep(k, l, k2, V0) * np.exp(1.j * k * l / 2.) / (2. * k)


def fop(k, l, k2=None, V0=None):
    r"""
    Evaluate one of the four Jost functions found in the 1DSWP case for
    a given wavenumber ``k``. ``fop`` stands for :math:`f_o^+` (see
    paper).

    See :function:`dem` for more details about the input variables.

    Returns
    -------
    complex
        Value of :math:`f_o^+`.
    """
    return dop(k, l, k2, V0) * np.exp(1.j * k * l / 2.) / (2. * k)


def fem(k, l, k2=None, V0=None):
    r"""
    Evaluate one of the four Jost functions found in the 1DSWP case for
    a given wavenumber ``k``. ``fem`` stands for :math:`f_e^-` (see
    paper).

    See :function:`dem` for more details about the input variables.

    Returns
    -------
    complex
        Value of :math:`f_e^+`.
    """
    return fep(-k, l, k2, V0)


def fom(k, l, k2=None, V0=None):
    r"""
    Evaluate one of the four Jost functions found in the 1DSWP case for
    a given wavenumber ``k``. ``fom`` stands for :math:`f_o^-` (see
    paper).

    See :function:`dem` for more details about the input variables.

    Returns
    -------
    complex
        Value of :math:`f_o^+`.
    """
    return fop(-k, l, k2, V0)


def find_parity(ks, sw_pot):
    r"""
    Find the parity of the Siegert state from its wavenumber and the 1D
    Square-Well Potential.

    A Siegert state satisfies some boundary conditions. In the 1D SWP
    case, to apply those boundary conditions give rise to two
    equalities: :math:`\delta_e^+(k) = 0` and :math:`\delta_o^+(k) = 0`.
    A Siegert state satisfying the first one is even (resp. odd for the
    second).

    Parameters
    ----------
    ks: complex
        Wavenumber of the Siegert state.
    sw_pot: SWPotential
        1D Square-Well Potential giving rise to this eigenstate.

    Returns
    -------
    str
        ``'e'`` for an even state, ``'o'`` for an odd state.

    Raises
    ------
    WavenumberError
        If the wavenumber does not correspond to a Siegert state
    """
    l = sw_pot.width
    V0 = sw_pot.depth
    # Check if it is an even Siegert state
    abs_dep = abs(dep(ks, l, V0=V0))
    if np.isclose(abs_dep, 0.0):
        return 'e'
    # Check if it is an odd Siegert state
    abs_dop = abs(dop(ks, l, V0=V0))
    if np.isclose(abs_dop, 0.0):
        return 'o'
    # If none of the above return are reached, raise an error
    raise WavenumberError(
        "The wavenumber does not correspond to a Siegert state.")
