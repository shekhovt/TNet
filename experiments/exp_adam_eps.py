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

# baseline Adam eps=1e-5
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --fp16 --distil -v 'T0.25'"]
# baseline Adam eps=1e-4
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --fp16 --distil -v 'T0.25-Adam1e-4'"]
# baseline Adam eps=1e-8
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --fp16 --distil -v 'T0.25-Adam1e-8'"]
# CAdam delta = 0.01
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --fp16 --distil -v 'T0.25-CAdam-d=0.01'"]
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --fp16 --distil -v 'T0.25-CAdam-d=0.1'"]
# CAdam delta-eps-nK
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --fp16 --distil -v 'T0.25-CAdam-d=0.1-eps=nK'"]

rr = plot(ll,title='Adam-eps experiment');

# %%

#%%
# Increase problem complexity by using a higher temperature
ll = []
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --fp16 --distil -T 1 --optimizer Adam --Adam_eps 1e-5"]
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --fp16 --distil -T 1 --optimizer Adam --Adam_eps 1e-6"]
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --fp16 --distil -T 1 --optimizer Adam --Adam_eps 1e-4"]
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --fp16 --distil -T 1 --optimizer CAdam --Adam_eps 1e-8 --CAdam_d 0.1"]
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --fp16 --distil -T 1 --optimizer CAdam --Adam_eps 1e-8 --CAdam_d 0.5"]
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --fp16 --distil -T 1 --optimizer CAdam --Adam_eps 1e-5 --CAdam_d 0.0"]
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --fp16 --distil -T 1 --optimizer CAdam --Adam_eps 1e-5 --CAdam_d 0.5"]
# ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --fp16 --distil -T 1 --optimizer CAdam --Adam_eps 1e-8 --CAdam_d 0.01"]

rr = plot(ll,title='Adam-eps experiment');

# %%
