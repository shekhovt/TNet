# SPDX-License-Identifier: CC BY-NC-SA 4.0
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. Making code open source is essential for scientific
# transparency and reproducibility. Providing access to the code allows researchers to
# verify, replicate, and expand upon results, which supports scientific progress. Current
# top-tier conferences encourage opensource code for the purpose of reproducibility that
# is de-facto an established expectation of the research community. Opensource code as
# part of publication should not affect the performance of an invention both from a
# commercial and IP point of view.
""" matplotlib drawing to a pdf setup """

import matplotlib

#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox


figsize = (6.0, 6.0 * 3 / 4)

def save_pdf(file_name):
    plt.savefig(file_name, bbox_inches='tight', dpi=199, pad_inches=0)


def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels()
    #    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])
    
    return bbox.expanded(pad, pad-1)


def save_axis(ax, file_name):
    # ax = plt.gca()
    fig = ax.figure
    fig.canvas.draw()
    # extent = full_extent(ax).transformed(fig.dpi_scale_trans.inverted())
    extent = ax.get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(file_name, bbox_inches=extent, pad_inches=0)
