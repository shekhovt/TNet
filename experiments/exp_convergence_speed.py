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
# %% Can we reproduce resnet18 (training) performance?
# >> Add resnet18 implemented with our layers so that ReLU variant would coincide with standard resnet18
ll = []
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'resnet18()' --method ReLU --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8","resnet18")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'ResNet18()' --method ReLU --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil","ResNet18 ReLU")]
# @ fu
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'ResNet18()' --method ReLU --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil  --Adam_eps 1e-8","ResNet18 ReLU")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'ResNet18()' --method ReLU --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil  --Adam_eps 1e-8 --v 'c1'","ResNet18 ReLU c1")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'ResNet18()' --method ReLU --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil  --Adam_eps 1e-8 --v 'c1-K2'","ResNet18 ReLU c1-K2")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'ResNet18()' --method ReLU --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil  --Adam_eps 1e-8 --v 'c1-K2-b0'","ResNet18 ReLU c1-K2-b0")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'ResNet18()' --method Clamp --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil  --Adam_eps 1e-8 --v 'c1-K2-b0'","ResNet18 Clamp c1-K2-b0")]
# TODO: Isolate [constrained weights / gradinet projection / block0] 
plot(ll, title="", loc=0, acc=False);
#%% Old parameter setup (still default --Adam_eps 1e-5)
ll = []
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'resnet18()' --method ReLU --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'","ResNet18")]
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ReLU --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'","TNet ReLU")]
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'", "TNet 1-2/1b")]
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 8 -W 8 --cudagraph --fp16 --distil -v 'T0.25'", "TNet 3bit")]
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 16 -W 16 --cudagraph --fp16 --distil -v 'T0.25'", "TNet 4bit")]
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 16 -W 16 --cudagraph --fp16 --distil --seed 2", "TNet 4bit")]
#ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
# 

#%% Comparison of Convergence with the actual parameter setup (--Adam_eps 1e-8)
ll = []
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'resnet18()' --method ReLU --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8","resnet18")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ReLU --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8","TNet ReLU")]
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method Clamp --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8","TNet Clamp")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 16 -W 16 --cudagraph --fp16 --distil --Adam_eps 1e-8", "TNet A4 W4")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 8 -W 8 --cudagraph --fp16 --distil --Adam_eps 1e-8", "TNet A3 W3")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 4 -W 4 --cudagraph --fp16 --distil --Adam_eps 1e-8", "TNet A2 W2")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8", "TNet A1-2 W1")]
# Note: ReLU has the same training loss and accuracy curve as 3/4 bit but generalizes better...
# rr = plot(ll);
plot(ll, title="", experiment="Quant-GCPR25/exp/convergence-100", loc=0, acc=False, Abottom=50);


# %% Old vs new parameter setup
ll = []
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8", "TNet A1-2 W1")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'", "TNet 1-2/1b Adam1e-5")]
#
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 8 -W 8 --cudagraph --fp16 --distil --Adam_eps 1e-8", "TNet A3 W3")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 8 -W 8 --cudagraph --fp16 --distil -v 'T0.25'", "TNet 3bit Adam1e-5")]
#
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 16 -W 16 --cudagraph --fp16 --distil --Adam_eps 1e-8", "TNet A4 W4")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 16 -W 16 --cudagraph --fp16 --distil --seed 2", "TNet 4bit Adam1e-5")]
#
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ReLU --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8","TNet ReLU")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ReLU --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'","TNet ReLU  Adam1e-5")]
plot(ll, title="", experiment=None, loc=0, acc=False, Abottom=50);

# %%

#%% With / Wo MD
ll =[]
# ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --net 'QResNet18(gate=Tower8s)' --method ReLU --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --net 'QResNet18(gate=Tower8s)' --method ReLU --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil","TNet ReLU, no MD")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --net 'QResNet18(gate=Tower8s)' --method ReLU --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --MD","TNet ReLU MD")]
# reference
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'", "TNet 1-2/1b")]
#@ fu
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil --noise_sigma_W 0.33", "TNet 1b, W_noise_std=0.33, no MD")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil --noise_sigma_W 0.33 --MD", "TNet 1b, W_noise_std=0.33, MD")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'", "TNet 1b, W_noise_std=0, MD")]

# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 8 -W 8 --cudagraph --fp16 --distil --noise_sigma_W 0.33", "TNet 3bit")]
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 8 -W 8 --cudagraph --fp16 --distil --noise_sigma_W 0.33 --MD", "TNet 3bit MD")]
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 8 -W 8 --cudagraph --fp16 --distil -v 'T0.25'", "TNet 3bit MD-det")]
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 8 -W 8 --cudagraph --fp16 --distil --Adam_eps 1e-8", "TNet 3bit Adam1e-8")]
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 8 -W 8 --cudagraph --fp16 --distil -v 'T0.25'", "TNet 3bit MD")]
# @ marr
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8", "TNet 1-2/1b")]
# ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 8 -W 8 --cudagraph --fp16 --distil --noise_sigma_W 0.33 --MD"]
# >> repeat everything with --Adam_eps 1e-8
# rr = plot(ll);
plot(ll, title="", experiment="Quant-GCPR25/exp/MD", loc=0, acc=False);
# %%
