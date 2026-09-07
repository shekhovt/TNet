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

if __run__:
    os.chdir(relimport.proj_dir())

# %%
# Imports
from builtins import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
# %matplotlib inline
#__________________________________

from ..utilities.drawing import *
from ..train import *
from ..layers import Quant

# %%
K = 3
l = Quant(Quant.Options(q_noise_type='logistic', q_noise_sigma=1/6, K=K))
x = torch.linspace(-1,5, 500)
def f1(x):
    return x.clamp(min=0), 'ReLU'
def f2(x):
    return x.clamp(min=0, max = K-1), 'Clamp'
def f3(x):
    return l.mean_embedding(x), 'Mean'
def f4(x):
    return l.quantize(x), 'Quant'
def f5(x):
    n = l.sample_noise(x, CN=0)
    return l.quantize(x+n), 'Stochastic Quant'
for f in [f1, f2, f3, f4, f5]:
    plt.figure(figsize=(3,3))
    y, name = f(x)
    if f == f5:
        plt.plot(x, y, '.')
    else:
        plt.plot(x, y, '-')
    plt.plot(np.arange(K), np.arange(K)*0, 'ok')
    plt.grid()
    plt.axis('equal')
    plt.xlim(-0.5,K-0.5)
    plt.ylim(-0.1,K-1)
    plt.yticks(np.arange(K))
    plt.xticks(np.arange(K))
    ax = plt.gca()
    # set the x-spine
    ax.spines['left'].set_position('zero')

    # turn off the right spine/ticks
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()

    # set the y-spine
    ax.spines['bottom'].set_position('zero')

    # turn off the top spine/ticks
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()

    plt.title(name)
    plt.draw()

    path = '../Quant-GCPR25/exp/propmethods/' + name + '.pdf'
    force_path(path)
    savefig(path)
    plt.show()
# %%
