# SPDX-License-Identifier: CC BY-NC-SA 4.0
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. Making code open source is essential for scientific
# transparency and reproducibility. Providing access to the code allows researchers to
# verify, replicate, and expand upon results, which supports scientific progress. Current
# top-tier conferences encourage opensource code for the purpose of reproducibility that
# is de-facto an established expectation of the research community. Opensource code as
# part of publication should not affect the performance of an invention both from a
# commercial and IP point of view.
import matplotlib.pyplot as plt
import matplotlib

from ..tools import *


SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the figure title

prop_cycle = plt.rcParams['axes.prop_cycle']
cc = prop_cycle.by_key()['color']

import matplotlib.colors as mcolors
cc1 =  list(mcolors.TABLEAU_COLORS)
cc1 = cc1 + ['red','green','blue']
cc1 = cc1 * 4
cc = cc1

markers = 'ov^<>sp*ox+dsX123'
markers = markers * 4

def savefig(outf):
    print(outf)
    force_path(outf)
    plt.savefig(outf, bbox_inches='tight', pad_inches=0.0)
    