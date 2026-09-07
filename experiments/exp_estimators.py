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
    # U-noise, ST-GS
# can test here again: pre-training strategies and lrep stratrgy
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ZGR --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
# ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method Mean --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
# add MeanSample (erorrs) and GS-ST(t=1)
# ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method MeanSample --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method 'GS-ST(t=1.0)' --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
# U noise
# ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
# ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST -n U --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
# performs very poorly, why?
# ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ZGR -n U --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]        
# T noise
# ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST -n T --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
# ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ZGR -n T --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
# lrep
# ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST --lrep 4 --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile"]
# pretraining
# ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ZGR --pm 'ST(epochs=150)' --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
# ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ZGR --pm 'ST(epochs=100)' --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
# ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST --pm 'Mean(epochs=50)' --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
# ST-WX
# ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method 'ST(WX=True)' --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
# No MD ablation
# ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --net 'QResNet18(gate=Tower8s)' --method ST --epochs 200 --lr 0.0025 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]

rr = plot(ll,title='Estimators');
test_variant = 'best_val_A1-test.pkl'
print(load_test_results(ll[0], test_variant=test_variant))
print(load_test_results(ll[1], test_variant=test_variant))

# %%

#%%
ll = []
# ST, ST-det, ZGR
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST --epochs 200 --lr 0.002 -A 2 -W 2 --cudagraph --fp16 --distil --compile"]
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det --epochs 200 --lr 0.002 -A 2 -W 2 --cudagraph --fp16 --distil --compile"]
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ZGR --epochs 200 --lr 0.002 -A 2 -W 2 --cudagraph --fp16 --distil --compile"]
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method 'GS-ST(t=1.0)' --epochs 200 --lr 0.002 -A 2 -W 2 --cudagraph --fp16 --distil --compile"]
# ST-WX
# ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method 'ST(WX=True)' --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
# No MD ablation
# ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --net 'QResNet18(gate=Tower8s)' --method ST --epochs 200 --lr 0.0025 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]

# rr = plot(ll,title='Estimators');
plot(ll, title="", experiment="estimators", loc=4);
test_variant = 'best_val_A1-test.pkl'
print(load_test_results(ll[0], test_variant=test_variant))
print(load_test_results(ll[1], test_variant=test_variant))

# %%


#%%
ll = []
# Mean pretraining 
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil --compile"]
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST --pm 'Mean(epochs=50)' --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil  --compile"]
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST --pm 'Mean(epochs=100)' --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil  --compile"]
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST --pm 'Mean(epochs=150)' --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil  --compile"]
#
# ST-WX
# ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method 'ST(WX=True)' --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
# No MD ablation
# ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --net 'QResNet18(gate=Tower8s)' --method ST --epochs 200 --lr 0.0025 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]

# rr = plot(ll,title='Mean Pretraining');
plot(ll, title="Mean Pretraining", experiment="preatraining-mean", loc=4);

# %%
# Clamp pretraining 
ll = []
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil --compile"]
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST --pm 'Clamp(epochs=50)' --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil  --compile"]
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST --pm 'Clamp(epochs=100)' --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil  --compile"]
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST --pm 'Clamp(epochs=150)' --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil  --compile"]
# rr = plot(ll,title='Clamp Pretraining');
plot(ll, title="Clamp Pretraining", experiment="preatraining-clamp", loc=4);


# %%
ll = []
ll += [("--batch_size 256 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'a0-b1-c128-Kw-bn-mw0-projmax1-in_sb-xbn1'", "no distillation")]
ll += [("--batch_size 256 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax1-bsbT0.25'","T=0.25")]
ll += [("--batch_size 256 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax1-bsb'", "T=1")]
rr = plot(ll,title='Distillaiton');
# %%

#%%
# ll = []
# # ST, ZGR, STQ..
ll = []
# ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-Q --epochs 200 --lr 0.002 -A 8 -W 256 --cudagraph --fp16 --distil --compile"]
# ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST --epochs 200 --lr 0.002 -A 8 -W 256 --cudagraph --fp16 --distil --compile"]
# ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ZGR --epochs 200 --lr 0.002 -A 8 -W 256 --cudagraph --fp16 --distil --compile"]
# ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method 'GS-ST(t=1.0)' --epochs 200 --lr 0.002 -A 8 -W 256 --cudagraph --fp16 --distil --compile"]
# plot(ll, title="");
# ll = []
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST -n L --epochs 200 --lr 0.002 --compile -A 8 -W 256 --cudagraph --fp16 --distil --Adam_eps 1e-8"]
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ZGR -n L --epochs 200 --lr 0.002 --compile -A 8 -W 256 --cudagraph --fp16 --distil --Adam_eps 1e-8"]
# ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-Q -n L --epochs 200 --lr 0.002 --compile -A 8 -W 256 --cudagraph --fp16 --distil --Adam_eps 1e-8","MVE")]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-Q -n L --epochs 200 --lr 0.002 --compile -A 8 -W 256 --cudagraph --fp16 --distil --Adam_eps 1e-8 -v 'stab2'","MVE")]
plot(ll, title="",experiment="res/fig/plots/mvu",acc=False,Abottom=55);

# %%
ll = []
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST -n L --epochs 200 --lr 0.002 --compile -A 4 -W 256 --cudagraph --fp16 --distil --Adam_eps 1e-8"]
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ZGR -n L --epochs 200 --lr 0.002 --compile -A 4 -W 256 --cudagraph --fp16 --distil --Adam_eps 1e-8"]
ll += [("--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-Q -n L --epochs 200 --lr 0.002 --compile -A 4 -W 256 --cudagraph --fp16 --distil --Adam_eps 1e-8 -v 'stab2'","MVE")]
# >>
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ZGR -n L --CN 1 --epochs 200 --lr 0.002 --compile -A 4 -W 256 --cudagraph --fp16 --distil --Adam_eps 1e-8"]
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST -n L --CN 1 --epochs 200 --lr 0.002 --compile -A 4 -W 256 --cudagraph --fp16 --distil --Adam_eps 1e-8"]
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n L --epochs 200 --lr 0.002 --compile -A 4 -W 256 --cudagraph --fp16 --distil --Adam_eps 1e-8"]
#
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method MeanSample -n L --epochs 200 --lr 0.002 --compile -A 4 -W 256 --cudagraph --fp16 --distil --Adam_eps 1e-8"]
ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method MeanSample -n L --epochs 200 --lr 0.002 --compile -A 4 -W 256 --cudagraph --fp16 --distil --Adam_eps 1e-8 -v 'var_eps=1e-3'"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s)' --method MeanSample -n L --epochs 200 --lr 0.002 --compile -A 4 -W 256 --cudagraph --fp16 --distil --Adam_eps 1e-8"]
plot(ll, title="",experiment="res/fig/plots/mvu-2b",acc=False,Abottom=55);

# #%%
# # rr = plot(ll,title='Estimators');
# plot(ll, title="", experiment="estimators", loc=4);
# test_variant = 'best_val_A1-test.pkl'
# print(load_test_results(ll[0], test_variant=test_variant))
# print(load_test_results(ll[1], test_variant=test_variant))

# %%
ll = []
ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s)' --method ST -n L --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8"]
ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s)' --method MeanSample -n L --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s)' --method MeanSample -n L --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 -v 'w-var'"]
# # ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s)' --method MeanSample -n L --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 -v 'sigma_learn'"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s)' --method MeanSample -n L --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 -v 'w-var-0.01'"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s)' --method MeanSample -n L --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 -v 'reg=1e-6'"]
ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s)' --method MeanSample -n L --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 -v 'reg=1e-7'"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s)' --method ST -n L --epochs 400 --lr 0.001 -A 2 -W 2 --fp16 --distil --Adam_eps 1e-8"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s)' --method MeanSample -n L --epochs 400 --lr 0.001 -A 2 -W 2 --fp16 --distil --Adam_eps 1e-8"]
ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s)' --method ST -n L --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 -v 'reg=1e-7'"]
ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n L --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 -v 'reg=1e-7'"]
# plot(ll, title="", experiment="res/fig/plots/DotST", acc=False, Abottom=55);
plot(ll, title="", acc=False, Abottom=55);
# %% DotST evaluation on smaller size models
ll = []
# BiNeal Arch
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'BiNealNet(m=0.5)' --method ST -n L --epochs 200 --lr 0.002 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'BiNealNet(m=0.5)' --method MeanSample -n L --epochs 200 --lr 0.002 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'BiNealNet(m=0.1)' --method ST -n L --epochs 200 --lr 0.002 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'BiNealNet(m=0.1)' --method MeanSample -n L --epochs 200 --lr 0.002 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'BiNealNet(m=0.1)' --method ST -n L --epochs 200 --lr 0.002 -A 4 -W 2 --fp16 --distil --Adam_eps 1e-8 --pathversion 6"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'BiNealNet(m=0.1)' --method MeanSample -n L --epochs 200 --lr 0.002 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 -v 'eps=1e-4'"]
# 
# TNet arch
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.1)' --method ST -n L --epochs 200 --lr 0.002 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.1)' --method MeanSample -n L --epochs 200 --lr 0.002 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.1)' --method ST -n L --epochs 200 --lr 0.001 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.1)' --method MeanSample -n L --epochs 200 --lr 0.001 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.1)' --method ST-Q -n L --epochs 200 --lr 0.001 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8"]
# 0.03
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.1)' --method ST -n L --epochs 200 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.1)' --method MeanSample -n L --epochs 200 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.1)' --method ST-Q -n L --epochs 200 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8"]
# 400 epochs
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.1)' --method ST -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.1)' --method MeanSample -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.1)' --method ST-Q -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8"]
#
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.1)' --method ST -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --seed 2"]
# # ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.1)' --method MeanSample -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --seed 2"]
# # ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.1)' --method ST-Q -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --seed 2"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.1)' --method ST -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --seed 2 -v 'sched1'"]
# #
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.1)' --method ST-Q -n L --epochs 800 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --seed 2"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.1)' --method ST -n L --epochs 800 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --seed 2"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.1)' --method ST -n L --epochs 800 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --seed 2 -v 'sched1'"]

# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.1,ch_grow=1)' --method ST -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --seed 2"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.1,ch_grow=1)' --method ST-Q -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --seed 2"]

# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.1,ch_grow=1,pre_cls_dim=100)' --method ST -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --seed 2"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.1,ch_grow=1,pre_cls_dim=100)' --method ST-Q -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --seed 2"]
# BN eps=1-e5
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.1,ch_grow=1,pre_cls_dim=100)' --method ST-det -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --seed 2"]

# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.1,ch_grow=1,pre_cls_dim=100)' --method ST -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --seed 2 -v 'bn_ep=1e-5'"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.1,ch_grow=1,pre_cls_dim=100)' --method ST-Q -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --seed 2 -v 'bn_ep=1e-5'"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.1,ch_grow=1,pre_cls_dim=100)' --method ST -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --CN 1 --seed 2"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.1,ch_grow=1,pre_cls_dim=100)' --method ST-Q -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --CN 1 --seed 2 -v 'bn_ep=1e-5'"]

# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.1,ch_grow=1,pre_cls_dim=100)' --method MeanSample -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --CN 1 --seed 2 -v 'bn_ep=1e-5'"]

# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.05,ch_grow=1,pre_cls_dim=200)' --method ST -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --seed 2"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.05,ch_grow=1,pre_cls_dim=200)' --method ST-Q -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --seed 2"]

# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.05,ch_grow=1,pre_cls_dim=200)' --method ST -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --CN 1 --seed 2"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.05,ch_grow=1,pre_cls_dim=200)' --method ST-Q -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --CN 1 --seed 2 -v 'bn_ep=1e-5'"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.05,ch_grow=1,pre_cls_dim=200)' --method MeanSample -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --CN 1 --seed 2 -v 'bn_ep=1e-5'"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.05,ch_grow=1,pre_cls_dim=200)' --method ST -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --CN 1 --seed 2 -v 'bn_ep=1e-5'"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.05,ch_grow=1,pre_cls_dim=200)' --method ST-Q -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --CN 1 --seed 3 -v 'bn_ep=1e-5'"]

# MeanSample: when generating the sample, CN is not taken into account -- fixed
# Maybe BN interferes -- -xBN

# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.05,ch_grow=1,pre_cls_dim=200)' --method ST -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --xBN --seed 2 -v 'bn_ep=1e-5'"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.05,ch_grow=1,pre_cls_dim=200)' --method ST -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --xBN --seed 1 -v 'bn_ep=1e-5'"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.05,ch_grow=1,pre_cls_dim=200)' --method ST-Q -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --xBN --seed 2 -v 'bn_ep=1e-5'"]

# Maybe projection issue, why the loss remains stochastic? Disabled projection. Loss stochastic e.g. becasue of augmentation
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.05,ch_grow=1,pre_cls_dim=200)' --method ST -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --xBN --seed 1 -v 'bn_ep=1e-5-no_proj'"]
# # NaN issues: becasue of all centered weights = 0 -> constant centering -> substantially worse
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.05,ch_grow=1,pre_cls_dim=200)' --method ST -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --xBN --seed 2 -v 'bn_ep=1e-5-no_proj'"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.05,ch_grow=1,pre_cls_dim=200)' --method ST-Q -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --xBN --seed 1 -v 'bn_ep=1e-5-no_proj'"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.05,ch_grow=1,pre_cls_dim=200)' --method MeanSample -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --xBN --seed 2 -v 'bn_ep=1e-5-no-proj'"]
# # Const centering + learned -- not as good
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.05,ch_grow=1,pre_cls_dim=200)' --method ST -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --xBN --seed 2 -v 'bn_ep=1e-5-no_proj-ccent'"]
# Returned mean centering
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.05,ch_grow=1,pre_cls_dim=200)' --method MeanSample -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --xBN --seed 1 -v 'bn_ep=1e-5-no-proj'"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.05,ch_grow=1,pre_cls_dim=200)' --method ST -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --xBN --seed 3 -v 'bn_ep=1e-5-no-proj'"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.05,ch_grow=1,pre_cls_dim=200)' --method ST-Q -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --xBN --seed 3 -v 'bn_ep=1e-5-no-proj'"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.05,ch_grow=1,pre_cls_dim=200)' --method ST-Q -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --xBN --seed 4 -v 'bn_ep=1e-5-no-proj'"]
# # ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.05,ch_grow=1,pre_cls_dim=200)' --method ST-Q -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --xBN --seed 5 -v 'bn_ep=1e-5-no-proj'"]

# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.05,ch_grow=1,pre_cls_dim=200)' --method ST -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --xBN --seed 5 -v 'bn_ep=1e-5-no-proj'"]

# # Activation regularization
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.05,ch_grow=1,pre_cls_dim=200)' --method ST -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --xBN --seed 5 -v 'bn_ep=1e-5-no-proj-areg1e-5'"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.05,ch_grow=1,pre_cls_dim=200)' --method ST-Q -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --xBN --seed 5 -v 'bn_ep=1e-5-no-proj-areg1e-5'"]

# # m = 0.5
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.5,ch_grow=1,pre_cls_dim=200)' --method ST -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --xBN --seed 5 -v 'bn_ep=1e-5-no-proj-areg1e-5'"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.5,ch_grow=1,pre_cls_dim=200)' --method ST-Q -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --xBN --seed 5 -v 'bn_ep=1e-5-no-proj-areg1e-5'"]

# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.5,ch_grow=1,pre_cls_dim=200)' --method ST -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --xBN --seed 5 -v 'bn_ep=1e-5-no-proj-areg1e-7'"]
# ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.5,ch_grow=1,pre_cls_dim=200)' --method ST-Q -n L --epochs 400 --lr 0.003 --compile -A 4 -W 2 --cudagraph --fp16 --distil --Adam_eps 1e-8 --xBN --seed 5 -v 'bn_ep=1e-5-no-proj-areg1e-7'"]

# enabling more integer states, esp. for weights
ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.5,ch_grow=1,pre_cls_dim=200)' --method ST -n L --epochs 400 --lr 0.003 --compile -A 8 -W 64 --cudagraph --fp16 --distil --Adam_eps 1e-8 --xBN --seed 5 -v 'bn_ep=1e-5-no-proj-areg1e-8'"]
ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.5,ch_grow=1,pre_cls_dim=200)' --method ST-Q -n L --epochs 400 --lr 0.003 --compile -A 8 -W 64 --cudagraph --fp16 --distil --Adam_eps 1e-8 --xBN --seed 5 -v 'bn_ep=1e-5-no-proj-areg1e-8'"]

# enabled BN
ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.5,ch_grow=1,pre_cls_dim=200)' --method ST -n L --epochs 400 --lr 0.003 --compile -A 8 -W 64 --cudagraph --fp16 --distil --Adam_eps 1e-8 --seed 5 -v 'bn_ep=1e-5-no-proj-areg1e-8'"]
ll += ["--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=Tower8s,m=0.5,ch_grow=1,pre_cls_dim=200)' --method ST-Q -n L --epochs 400 --lr 0.003 --compile -A 8 -W 64 --cudagraph --fp16 --distil --Adam_eps 1e-8 --seed 5 -v 'bn_ep=1e-5-no-proj-areg1e-8'"]


# Maybe still 12x2x9 approx 200 is many input units?

# try lr=0.002 epochs=400 t see the tail / stall regimes
# try a yet smaller model, without channel doubling

plot(ll, title="", acc=False);
# %%
ll = []

