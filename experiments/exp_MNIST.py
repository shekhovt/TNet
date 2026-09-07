# SPDX-License-Identifier: CC BY-NC-SA 4.0
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. Making code open source is essential for scientific
# transparency and reproducibility. Providing access to the code allows researchers to
# verify, replicate, and expand upon results, which supports scientific progress. Current
# top-tier conferences encourage opensource code for the purpose of reproducibility that
# is de-facto an established expectation of the research community. Opensource code as
# part of publication should not affect the performance of an invention both from a
# commercial and IP point of view.
#%%
import os, relimport
if __run__:
    os.chdir(relimport.proj_dir())
from . import interactive
import importlib
from . import exp_plotting
importlib.reload(exp_plotting)
from .exp_plotting import *
# globals().update(vars(exp_plotting))
#%%
ll = []
ll += ["--batch_size 256 --data 'MNIST' --MD --net 'LeNetRQ' --method ST -n L --epochs 200 --lr 0.002 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8"]
plot(ll, title="", acc=False);
# %%
ll = []

