# SPDX-License-Identifier: CC BY-NC-SA 4.0
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. Making code open source is essential for scientific
# transparency and reproducibility. Providing access to the code allows researchers to
# verify, replicate, and expand upon results, which supports scientific progress. Current
# top-tier conferences encourage opensource code for the purpose of reproducibility that
# is de-facto an established expectation of the research community. Opensource code as
# part of publication should not affect the performance of an invention both from a
# commercial and IP point of view.
 # %%
import os, sys
import relimport

# %%
# Imports
from builtins import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
# %matplotlib inline
#__________________________________
try:
    from IPython import get_ipython
    ipy = get_ipython()
    if ipy is not None:
        ipy.run_line_magic("load_ext", "autoreload")
        ipy.run_line_magic("autoreload", "2")
except Exception:
    print("Not in IPython")
    pass  # IPython not available, skip autoreload
import importlib
#__________________________________

from ..utilities.drawing import *
from ..train import *
