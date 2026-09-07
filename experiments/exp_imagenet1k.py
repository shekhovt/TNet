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
os.chdir(relimport.proj_dir())
from . import interactive
import importlib
from . import exp_plotting
importlib.reload(exp_plotting)
from .exp_plotting import *

# assert(False)

# %% Width Scaling law
ll = []
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 4 -W 4 --cudagraph --fp16 --distil -v 'T0.25'"]
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --seed=2"]
# ll += ["--batch_size 128 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s,m=1.5)' --method ST-det -n U --epochs 200 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8"]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s,m=0.75)' --method ST-det -n U --epochs 200 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8"]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 4 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s,m=1.5)' --method ST-det -n U --epochs 200 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8"]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s,m=2)' --method ST-det -n U --epochs 200 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8"]
# >>
#ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s,m=2)' --method ST-det -n U --epochs 200 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --seed 2"]
# >>
#
rr = plot(ll, title="", experiment="Quant-GCPR25/exp/width", acc=None);
val_summary(rr)
#%% New Dilation results
ll =[]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s,Dilation=True)' --method ST-det -n U --epochs 200 --lr 0.003 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s,Dilation=True)' --method ST-det -n U --epochs 200 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s,Dilation=True)' --method ST-det -n U --epochs 200 --lr 0.003 --compile -A 8 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s,Dilation=True)' --method ST-det -n U --epochs 200 --lr 0.003 --compile -A 4 -W 4 --cudagraph --fp16 --distil -v 'T0.25'"]

# # Old Dilation results
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s,Dilation=True)' --method ST --epochs 200 --lr 0.0025 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s,Dilation=True)' --method ST --epochs 200 --lr 0.0025 --compile -A 4 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s,Dilation=True)' --method ST --epochs 200 --lr 0.0025 --compile -A 8 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s,Dilation=True)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 4 -W 4 --cudagraph --fp16 --distil -v 'T0.25'"]

rr = plot(ll, title="", experiment="Quant-GCPR25/exp/distill-new", acc=None);
val_summary(rr)
#%%

# globals().update(vars(exp_plotting))
# %% vs Real
ll =[]
ll += [("--batch_size 256 --data 'imagenet(res=512)' --MD --net 'resnet18()' --method ReLU --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'", "resnet18")]
ll += [("--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ReLU --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-5 --fp16", "TNet ReLU")]
# ll += [("--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method Clamp --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'", "TNet Clamp")]
ll += [("--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 16 -W 16 --cudagraph --fp16 --distil -v 'T0.25'", "TNet A4 W4")]
ll += [("--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --fp16 --distil -v 'T0.25'", "TNet A3 W3")]
ll += [("--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 4 -W 4 --cudagraph --fp16 --distil -v 'T0.25'", "TNet A2 W2")]
ll += [("--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'", "TNet A1-2 W1")]
rr = plot(ll, title="", experiment="Quant-GCPR25/exp/convergence", acc=None);
val_summary(rr)
# %% vs Real, new Setup
ll = []
ll += [("--batch_size 256 --data 'imagenet(res=512)' --MD --net 'resnet18()' --method ReLU --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'", "resnet18")]
ll += [("--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ReLU --epochs 200 --lr 0.003 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16", "TNet ReLU")]
# ll += [("--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method Clamp --epochs 200 --lr 0.003 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16", "TNet Clamp")]
ll += [("--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.003 --compile -A 16 -W 16 --cudagraph --fp16 --distil --Adam_eps 1e-8", "TNet A4 W4")]
ll += [("--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.003 --compile -A 8 -W 8 --cudagraph --fp16 --distil --Adam_eps 1e-8", "TNet A3 W3")]
ll += [("--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.003 --compile -A 4 -W 4 --cudagraph --fp16 --distil --Adam_eps 1e-8", "TNet A2 W2")]
ll += [("--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.003 --compile -A 2 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8", "TNet A1-2 W1")]
rr = plot(ll, title="", experiment="Quant-GCPR25/exp/convergence", acc=None);
val_summary(rr)
#%%
ll = []

# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --fp16"]
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --fp16 --distil"]
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8)' --method ST --epochs 200 --lr 0.01 --compile -A 2 -W 2 --cudagraph --fp16"] # terminated, poor
# ll += ["--batch_size 512 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --fp16"]
# Tower8 distill 0.25
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]    
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8)' --method ST --epochs 200 --lr 0.0025 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
# todo: Dilation T/F * ((A=2, W2), (A=2-4, W2), (A=4, W2), (A4, W4), (A=8, W2), (Clamp), (ReLU) )
# No dilation
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method Clamp --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
# Optimizes setup
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
# extra, checking fp16 vs fp32
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distil -v 'T0.25-fp32'"]

ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 4 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --fp16 --distil -v 'T0.25'"]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 16 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 16 -W 16 --cudagraph --fp16 --distil -v 'T0.25'"]
# # Dilation results
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s,Dilation=True)' --method ST --epochs 200 --lr 0.0025 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s,Dilation=True)' --method ST --epochs 200 --lr 0.0025 --compile -A 4 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s,Dilation=True)' --method ST --epochs 200 --lr 0.0025 --compile -A 8 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
# # extra
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 4 -W 4 --cudagraph --fp16 --distil -v 'T0.25'"]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s,Dilation=True)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 4 -W 4 --cudagraph --fp16 --distil -v 'T0.25'"]
# # extra, checking fp16 vs fp32
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --distil -v 'T0.25-v32'"]
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --distil -v 'T0.25-v32-gradscalar'"]
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --distil -v 'T0.25-v32-wscale'"]
# # more debug configs
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --distil -v 'T0.25-fp16-old' --fp16"]
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --distil -v 'T0.25-v32-old'"]
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --distil -v 'T0.25-v32-old-gradscaler'"]
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --distil -v 'T0.25-v32-old-gradscaler-AC'"]
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --distil -v 'T0.25-v32'"]
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --distil -v 'T0.25-fp16-new-constGS' --fp16"]
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --distil -v 'T0.25-fp32-new-constGS'"]
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --distil -v 'T0.25-fp32-Adam_eps0.5'"]
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --distil -v 'T0.25-fp32-Adam_eps1e-4'"]

# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 2 -W 3 --cudagraph --fp16 --distil -v 'T0.25'"]

# %%
# BiNeal
ll = []
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'BiNealNet(m=1)' --method ST-det -n U --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --distil -v 'T0.25-fp32' --compile"]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'BiNealNet(m=1)' --method ST-det -n U --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --distil -v 'T0.25-fp16' --fp16 --compile"]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'BiNealNet(m=1.5)' --method ST-det -n U --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --distil -v 'T0.25-fp32' --compile"]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'BiNealNet(m=1.5)' --method ST-det -n U --epochs 200 --lr 0.002 -A 2 -W 2 --cudagraph --distil -T 0.25 --fp16 --compile"]
# resnet
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'resnet18()' --method ReLU --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]

# New setup
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'BiNealNet(m=1.5)' --method ST-det -n U --epochs 200 --lr 0.002 -A 2 -W 2 --cudagraph --distil -T 0.25 --fp16 --compile"]
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil -T 0.25"]


# plot(ll, title="", experiment="Quant-GCPR25/exp/scaling", loc=2)
rr = plot(ll, title="")
val_summary(rr)
# %%
r.o.keys()


# %%
#%%
ll = []
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --fp16 --distil -v 'T0.25'"]
# BiNeal
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'BiNealNet(m=1)' --method ST-det -n U --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --distil -v 'T0.25-fp32' --compile"]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'BiNealNet(m=1)' --method ST-det -n U --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --distil -v 'T0.25-fp16' --fp16 --compile"]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'BiNealNet(m=1.5)' --method ST-det -n U --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --distil -v 'T0.25-fp32' --compile"]

# extra, checking fp16 vs fp32
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --distil -v 'T0.25-v32'"]
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --distil -v 'T0.25-v32-gradscalar'"]
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --distil -v 'T0.25-v32-wscale'"]
# more debug configs
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --distil -v 'T0.25-fp16-old' --fp16"]
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --distil -v 'T0.25-v32-old'"]
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --distil -v 'T0.25-v32-old-gradscaler'"]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --distil -v 'T0.25-v32-old-gradscaler-AC'"]
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --distil -v 'T0.25-v32'"]
# # ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --distil -v 'T0.25-fp16-new-constGS' --fp16"]
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --distil -v 'T0.25-fp32-new-constGS'"]
# # ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --distil -v 'T0.25-fp32-Adam_eps0.5'"]


rr = plot(ll, title="",loc=4)



# %%
#%%
ll = []
# new setup, trying BiNeal and binary with smaller lr and eps 1e-8
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'BiNealNet(m=1.5)' --method ST-det -n U --epochs 200 --lr 0.002 -A 2 -W 2 --cudagraph --distil -T 0.25 --fp16 --compile"]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil -T 0.25"]

rr = plot(ll, title="",loc=4)

#%%
ll = []
# trash
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --fp16 --distil -T 1 --optimizer CAdam --Adam_eps 1e-5 --CAdam_d 0.5"]
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --fp16 --distil -T 0.25 --optimizer CAdam --Adam_eps 1e-5 --CAdam_d 0.5"]
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --fp16 --distil -T 0.25 --optimizer CAdam --Adam_eps 1e-5 --CAdam_d 0.5 -v 'moproj'"]
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 4 -W 2 --cudagraph --fp16 --distil -T 0.25 --optimizer CAdam --Adam_eps 1e-5 --CAdam_d 0.5"]
# %%
# CAdam, eps
ll = []
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 8 -W 8 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16"]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --fp16 --distil -T 0.25 --optimizer CAdam --Adam_eps 1e-8 --CAdam_d 0.5"]
ll += [("--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --fp16 --distil -v 'T0.25'", "eps=1e-5")]
ll += [("--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --distil -v 'T0.25-v32-old-gradscaler-AC'", "eps=1e-8")]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 4 -W 2 --cudagraph --fp16 --distil -T 0.25 --optimizer CAdam --Adam_eps 1e-5 --CAdam_d 0.5"]

# eps = 1e-4
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --fp16 --distil -T 0.25 --optimizer Adam --Adam_eps 1e-4"]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --distil -v 'T0.25-fp32-Adam_eps1e-4'"]

ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --fp16 --distil -T 0.25 --optimizer CAdam --Adam_eps 1e-5 --CAdam_d 0.5"]


rr = plot(ll, title="",loc=4)
# %%
for (i,r) in enumerate(rr):
    Wb = int(np.log2(r.o.W))
    Ab = int(np.log2(r.o.A))
    dilation = 'Dilation' in r.o.args_str
    acc = np.floor((r.hist['val_A1'].max())*1000)/10
    running = len(r.hist['val_A1']) < 21
    # print(f'Dilation={dilation} Fb={Ab}, Wb={Wb}, Acc={acc}{"+" if running else ""}')
    # print(f'{ll[i][1]} Dilation={dilation} & {Ab} & {Wb} & {acc}{"+" if running else ""}')
    print(f'{r.o.net}  {r.o.method} & {Ab} & {Wb} & {acc}{"+" if running else ""}')

# %%

gr_hist(rr[-1])

# %%
# !!!Image size 256 -> input image size 128 (same amount of weights, smaller feature maps)
# Image size 128
ll = []
ll += ["--batch_size 256 --data 'imagenet(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --fp16 --distil -T 0.25 --optimizer Adam --Adam_eps 1e-8"]
ll += ["--batch_size 256 --data 'imagenet(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --fp16 --distil -T 0.25 --optimizer Adam --Adam_eps 1e-5"]
ll += ["--batch_size 256 --data 'imagenet(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --fp16 --distil -T 0.25 --optimizer Adam --Adam_eps 1e-4"]
ll += ["--batch_size 256 --data 'imagenet(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --fp16 --distil -T 0.25 --optimizer CAdam --Adam_eps 1e-8 --CAdam_d 0.03"]
# ll += ["--batch_size 256 --data 'imagenet(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --fp16 --distil -T 0.25 --optimizer Adam --Adam_eps 1e-5 --beta1 0.9605 --beta2 0.996"]
rr = plot(ll, title="",loc=4)
# %%

#%%
# Learning Rate
ll = []
ll += [("--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --fp16 --distil -v 'T0.25'","lr 0.005 eps=1e-5")]
ll += [("--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --fp16 --distil -T 0.25 --optimizer CAdam --Adam_eps 1e-8 --CAdam_d 0.5","lr 0.005 CAdam(delta=0.5, eps=1e-8)")]
ll += [("--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 8 -W 8 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16","lr 0.002 eps=1e-8")]
rr = plot(ll, title="",loc=4)
# %%

#%%
# Epochs
ll = []
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 100 --lr 0.002 --compile -A 8 -W 8 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16"]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 8 -W 8 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16"]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 300 --lr 0.002 --compile -A 8 -W 8 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16"]
# rr = plot(ll, title="",loc=4, experiment="Quant-GCPR25/exp/epochs", Abottom=60)
rr = plot(ll, title="",loc=4, Abottom=60)
val_summary(rr)
# %%
gr_hist(rr[-3])
# %%
# Distillation Temperature / Teacher / Classifier / no-wproj
ll =[]
ll += [("--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 8 -W 8 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16", "T=0.25 distill")]
ll += [("--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 8 -W 8 --cudagraph --distil -T 0.1 --Adam_eps 1e-8 --fp16 --seed 2", "T=0.1 distill")]
# error ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 8 -W 8 --cudagraph --distil -T 0.1 --Adam_eps 1e-8 --fp16"]
ll += [("--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 8 -W 8 --cudagraph --distil -T 1 --Adam_eps 1e-8 --fp16", "T=1 distill")]
ll += [("--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 8 -W 8 --cudagraph --Adam_eps 1e-8 --fp16", "No distillation")]
rr = plot(ll, title="", experiment="Quant-GCPR25/exp/distill",loc=4, Abottom=55)
# %%
# %%
# Modifications A8 W8
ll =[]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 8 --cudagraph --fp16 --distil -v 'T0.25'"]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 8 -W 8 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16"]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 8 -W 8 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16 -v 'teacher-79'"]
# ##>>
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 8 -W 8 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16 -v 'no-wproj'"]
# ##>>
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 8 -W 8 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16 -v 'c5'"]
# no effect
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 8 -W 8 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16 -v 'warmup-2-epochs'"]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=GenRes8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 8 -W 8 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16 -v 'rec'"]
# verification of higher learning rate without Adam eps: less stable, inferiror
rr = plot(ll, title="",loc=4, Abottom=55)
val_summary(rr)

# %%
# Modifications A2 W2
ll = []
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16"]

ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16"]
ll += [("--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=GenRes8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16 -v 'rec-no-alpha'", 'bw')]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=GenRes8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16 -v 'rec'"]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=GenRes8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16 -v 'rec-no-alpha1'"]
# classifier modification -- poor
# ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16 -v 'c5'"]

# >>
# >>
# >>
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=GenRes8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16 --unlock"]
# crashed several times, not promissing
#
# freeze-in_sb -- poor
# not expressive enough / not enough parameters?
# fu>> 
# fu>> 
# marr>> out of mem
rr = plot(ll, title="",loc=4, Abottom=55)
val_summary(rr)
# gr_hist(rr[4])
# %%
gr_hist(rr[-1], class_name = 'ScaleBias', param_name='bias')
# gr_hist(rr[4], class_name = 'BatchNorm2d', param_name=['weight', 'bias'])
# gr_hist(rr[-1], class_name = 'QConv2d', param_name='weight')
# gr_hist(rr[-3], class_name = 'GenResBlock', param_name='w')
# gr_hist(rr[-2], class_name = 'GenResBlock', param_name='w')

# %%
# Centering (1K). Outcome: no effect
ll =[]
ll += [("--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 8 -W 8 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16", "mean")]
ll += [("--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 8 -W 8 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16 -v 'mean-learned'","mean+learned")]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 8 -W 8 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16 -v 'K-center'"]
rr = plot(ll, title="",loc=4)

# %%
# Modifications A2 W2
ll =[]
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
# for A2 W2 degradation of results becasue of the lr=0.002
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16"]
ll += [("--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=GenRes8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16 -v 'rec-no-alpha'", 'bw')]
#
# free linear weights for fusion - unstable training
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=GenRes8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16 -v 'rec'"]
# no extra weights, fusion just by quantized conv -- stable training, lower training ACC => need a convex comb weights with softmax
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=GenRes8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16 -v 'rec-no-alpha1'"]
#
ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --distil -T 0.25 --Adam_eps 1e-8 --fp16 -v 'c5'"]
rr = plot(ll, title="",loc=4, Abottom=55)
val_summary(rr)
# %%

