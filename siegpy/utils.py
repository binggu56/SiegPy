# -*- coding: utf-8 -*-
"""
This file contains utilitary functions for the SiegPy module.
It mainly allows to remove the need to import matplotlib.pyplot as plt
in the rest of the SiegPy module.
"""

import matplotlib.pyplot as plt


def init_plot():  # pragma: no cover
    r"""
    Initialize the object-oriented plot using matplotlib.

    Returns
    -------
    tuple
        The figures and the axes of the plot.
    """
    fig, ax = plt.subplots()
    return fig, ax


def finalize_plot(fig, ax, xlim=None, ylim=None, title=None, file_save=None,
                  leg_loc=None, leg_bbox_to_anchor=None, xlabel=None,
                  ylabel=None):  # pragma: no cover
    r"""
    Finalize the plot using matplotlib.

    Parameters
    ----------
    fig: matplotlib.figure.Figure
        Figure.
    ax: matplotlib.axes._subplots.AxesSubplot
        Axes.
    xlim: tuple(float or int, float or int)
        Range of the x axis of the plot (optional)
    xlim: tuple(float or int, float or int)
        Range of the y axis of the plot (optional)
    title: str
        Plot title (optional).
    file_save: str
        Base name of the files where the plots are to be saved
        (optional).
    leg_loc: int
        Location of the legend (optional).
    leg_bbox_to_anchor: tuple
        Localization of the legend box (optional).
    xlabel: str
        Label of the x axis (optional).
    ylabel: str
        Label of the y axis (optional).
    """
    # Set the range of the plot on the x- and y-axis
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    # Set ticks and axis label
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    # Add the legend
    if leg_loc is not None and leg_bbox_to_anchor is not None:
        ax.legend(loc=leg_loc, bbox_to_anchor=leg_bbox_to_anchor)
    elif leg_loc is not None:
        ax.legend(loc=leg_loc)
    # Add the title
    if title is not None:
        ax.set_title(title)
    # Save the plot if required
    if file_save is not None:
        fig.savefig(file_save)
    # Finally, show it
    plt.show()
