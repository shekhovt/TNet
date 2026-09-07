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
try:
    from IPython import get_ipython
    ipy = get_ipython()
    if ipy is not None:
        ipy.run_line_magic("load_ext", "autoreload")
        ipy.run_line_magic("autoreload", "2")
except Exception:
    pass  # IPython not available, skip autoreload
import importlib
#__________________________________

from .drawing import *
from ..train import *

# from .. import train
# importlib.reload(train)
# from ..train import *
# globals().update(vars(train))



# %%

def load_hist(l):
    res = dotdict()
    o = o_from_str(l)
    r = find_root(o)
    print(r)
    rf = r + 'out/hist.pkl'
    res.o = o
    res.hist = load_object(rf)
    return res,o

def get_xy(res, what, method):
    if 'epoch_' in what:
        if '_A1' in what:
            y = res.hist.A1
        else: #"_L1"
            y = res.hist.L1
        x = res.hist.epoch
    else: # 'val_'
        if what == 'val_L1':
            if method in ['Mean', 'ReLU', 'Clamp']: # method is real
                y = res.hist.v_real_L
            else:
                y = res.hist.v_det_L # method is det
        else:
            y = res.hist[what]
        x = res.hist.val_epoch

    # if what == 'epoch_L1':
    #     x = res.hist.epoch
    #     y = res.hist.L1
    # elif what == 'epoch_A1':
    #     x = res.hist.epoch
    #     y = res.hist.A1
    # elif what == 'val_A1':
    #     y = res.hist.val_A1
    #     x = np.arange(len(y))
    # elif what == 'tr_det_L':
    #     y = res.hist.tr_det_L
    #     x = np.arange(len(y))
    # elif what == 'tr_mult_L':
    #     y = res.hist.tr_mult_L
    #     x = np.arange(len(y))
    # elif what == 'v_det_L':
    #     y = res.hist.v_det_L
    #     x = np.arange(len(y))
    # elif what == 'v_det_A':
    #     y = res.hist.v_det_A
    #     x = np.arange(len(y))
    # elif what == 'tr_det_A':
    #     y = res.hist.tr_det_A
    #     x = np.arange(len(y))        
    # else:
    #     raise AttributeError("unknown case")
    return x,y

#%%

dataset = 'imagenet-1000'

# exp = 'mw_ablation'
# exp = 'distill'
# exp = 'sb'
# exp = 'A123'
# exp = 'arch'
# exp = 'distillT'
# exp = 'hard-100'
exp = 'estimators'
# exp = 'normal-depth'

ll = []

if dataset == 'imagenet-1000':

    # ll += ["--batch_size 256 --data imagenet-1000 --net QResNet18(gate=none) --method ReLU -W 2 -A 2 -n L --MD --unlock --cudagraph --data_seed 3 --epochs 200 --lr 0.005 --compile --tag FFCV-RAND-BN-Fix"]

    # # ll += ["--batch_size 128 --data imagenet-1000 --net QResNet18(gate=none) --method ReLU -W 2 -A 2 -n L --MD --unlock --cudagraph --data_seed 3 --epochs 200 --lr 0.005 --compile"]

    # ll += ["--batch_size 256 --data imagenet-1000 --net 'QResNet18(gate=none)' --method ST -W 2 -A 2 -n L --MD --unlock --cudagraph --epochs 200 --lr 0.005 --compile --pm 'Mean(epochs=20)'"]

    # # ll += ["--batch_size 128 --data imagenet-1000 --net QResNet18(gate=none) --method ST -W 2 -A 2 -n L --MD --unlock --cudagraph --data_seed 3 --epochs 200 --lr 0.005 --compile --pm Mean(epochs=20)"]

    # ll += ["--batch_size 256 --data imagenet-1000 --net 'QResNet18(gate=none)' --method Mean -W 2 -A 2 -n L --MD --unlock --cudagraph --epochs 200 --lr 0.005 --compile"]

    # # ll += ["--batch_size 256 --data imagenet-1000 --net QResNet18(gate=none) --method ST -W 2 -A 2 -n L --MD --unlock --cudagraph --epochs 200 --lr 0.005 --compile --winit res/imagenet-1000-QResNet18(gate=none)-v4-c26/W=2\ A=2/ST/n=L\ 0.3333333333333333\ BS256\ MD\ pm=Mean(epochs=20)/o=Adam\ lr=0.005\ wd=0/last.pkl"]

    # # ll +=["--batch_size 128 --data imagenet-1000 --net QResNet18(gate=none) --method ST -W 2 -A 2 -n L --MD --unlock --cudagraph --epochs 200 --lr 0.0005 --compile --winit res/imagenet-1000-QResNet18(gate=none)-v4-c26/W=2\ A=2/ST/n=L\ 0.3333333333333333\ BS256\ MD\ pm=Mean(epochs=20)/o=Adam\ lr=0.005\ wd=0/final.pkl"]

    # # ll += ["--batch_size 128 --data imagenet-1000 --net 'QResNet18(gate=none)' --method Mean -W 2 -A 2 -n L --MD --unlock --cudagraph --data_seed 3 --epochs 200 --lr 0.005 --compile --test"]
    # # ll += ["--batch_size 128 --data imagenet-1000 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 4 -W 2 --cudagraph -v 'augment3'"]
    # # ll += ["--batch_size 256 --data imagenet-1000 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 4 -W 2 --cudagraph -v 'augment3'"]
    # # ll += ["--batch_size 128 --data imagenet-1000 --MD --net 'QResNet18(gate=Tower3)' --method Mean --epochs 200 --lr 0.005 --compile -A 4 -W 2 --cudagraph -v 'augment3-layers-0-0-1'"]
    # # ll += ["--batch_size 128 --data imagenet-1000 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'augment3-layers-0-0-1'"]

    # # ll += ["--batch_size 128 --data imagenet-1000 --MD --net 'QResNet18(gate=Tower3,Dilation=True)' --method ReLU --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --compile"]
    # ll += ["--batch_size 128 --data imagenet-1000 --MD --net 'QResNet18(gate=Tower3,Dilation=True)' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --compile"]    
    # ll += ["--batch_size 128 --data imagenet-1000 --MD --net 'QResNet18(gate=Tower3,Dilation=True)' --method ST --epochs 200 --lr 0.005 -A 8 -W 2 --cudagraph --compile"]

    # # ll += ["--batch_size 128 --data 'imagenet-1000' --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128'"]
    # # ll += ["--batch_size 128 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower5a)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128'"]
    # ll += ["--batch_size 128 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower5a)' --method ST --epochs 200 --lr 0.01 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-fp16-gscaling1' --fp16"]
    # ll += ["--batch_size 128 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower5a)' --method ST --epochs 200 --lr 0.02 --compile -A 2 -W 2 --cudagraph -v 'fp16-gscaling1-T0.25' --fp16 --distil"]
    # #
    ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --fp16"]
    ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --fp16 --distil"]
    # ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8)' --method ST --epochs 200 --lr 0.01 --compile -A 2 -W 2 --cudagraph --fp16"] # terminated, poor
    ll += ["--batch_size 512 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --fp16"]
    # Tower8 distill 0.25
    ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]    
    ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8)' --method ST --epochs 200 --lr 0.0025 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
    ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s,Dilation=True)' --method ST --epochs 200 --lr 0.0025 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
    # A123
    ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s,Dilation=True)' --method ST --epochs 200 --lr 0.0025 --compile -A 4 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
    ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s,Dilation=True)' --method ST --epochs 200 --lr 0.0025 --compile -A 8 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
    # todo: Dilation T/F * ((A=2, W2), (A=2-4, W2), (A=4, W2), (A4, W4), (A=8, W2), (Clamp), (ReLU) )
    # No dilation
    ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method Clamp --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
    # Optimizes setup
    ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
    ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 4 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
    ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
    ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 16 -W 16 --cudagraph --fp16 --distil -v 'T0.25'"]
    ll += ["--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 16 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
    # check Clamp and Relu do GD with MD correctly

elif dataset == 'imagenet-100':
    if exp == 'distill': # mean_weight ablation
        # # Tower 2,3,4
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower2)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'augment3'"]
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'augment3'"]
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower4)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'augment3'"]
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower20)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph"]

        # # ll += ["--batch_size 128 --data imagenet-100 --net resnet18 --method ReLU -W 2 -A 2 -n L --MD --unlock --cudagraph --data_seed 3 --epochs 200 --lr 0.005 --compile"]
        # # # Bits
        # # # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower4,DB=False)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2"]
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net QResNet18(gate=Tower4) --method ReLU --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v augment-3"]
        # # ll += ["--batch_size 128 --data imagenet-100 --MD --net QResNet18(gate=Tower4) --method Mean --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v augment-3"]
        # # ll += ["--batch_size 128 --data imagenet-100 --MD --net QResNet18(gate=Tower4,DB=False) --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v augment3"]
        # # ll += ["--batch_size 128 --data imagenet-100 --MD --net QResNet18(gate=Tower4) --method ST --epochs 200 --lr 0.005 --compile -A 4 -W 2 --cudagraph -v augment3"]
        # # ll += ["--batch_size 128 --data imagenet-100 --MD --net QResNet18(gate=Tower4) --method ST --epochs 200 --lr 0.005 --compile -A 8 -W 2 --cudagraph -v augment3"]

        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'augment3'"]
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net QResNet18(gate=Tower4) --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v augment3"]

        ll += ["--batch_size 128 --data imagenet-100 --MD --net 'resnet18' --method ReLU --epochs 200 --lr 0.005 --distill"] 
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ReLU --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'as1'"]
        ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ReLU --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128'"]

        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method Mean --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph"]
        ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph"]
        # # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ST --lrep 5 --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph"]
        # # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ST --lrep 10 --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph"]

        # # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method Mean --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'Adam-mean-K'"]
        # # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'Adam-mean-K'"]
        # # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ST --lrep 5 --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'Adam-mean-K'"]
        # # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ZGR --lrep 5 --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'Adam-mean-K'"]
        # # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'c1'"]
        # # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'Adam-c1'"]
        # # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'Adam-c0'"]
        # # # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'Adam-c0-n'"]
        # # # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'Adam-c0-n' --seed 2"]
        # # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'Adam-c2-n'"]
        # # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'Adam-c3'"]
        # # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'Adam-c3-n'"]
        # # # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'Adam-c3-n-d0'"]
        # # # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'Adam-c3-n-d03'"]
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ReLU --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill"]
        ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill"]
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --t_stage 4"]
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill --t_stage 4"]
        
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ReLU --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'a2'"]
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ReLU --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a2'"]
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'a2'"]
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 400 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'as1'"]
        # # augmentation with schedule and MixUp
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'as1'"]
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 400 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'as1'"]
        # # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill --t_stage 4 -v 'as1-mix'"]

        ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0'"]
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0' --t_stage 4"]
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-randhead' --t_stage 4"]
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b0'"]
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b0-l1'"]
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower4)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b0'"] 
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-l222'"]
        ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1'"]
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1' --mixup 0.1"]    
        # 
        ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128'"]
        # invalidated (higher resolution)
        # ll += ["--batch_size 128 --data 'imagenet-100' --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 4 -W 2 --cudagraph --distill -v 'a0-b1-c128'"]
        # 

        # ll += ["--batch_size 128 --data 'imagenet-100(res=512)' --MD --net 'resnet18' --method ReLU --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128'"]
        # ll += ["--batch_size 128 --data 'imagenet-100(res=512)' --MD --net 'QResNet18(gate=Tower5)' --method ReLU --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128'"]
        ll += ["--batch_size 128 --data 'imagenet-100(res=512)' --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128'"]    
        
        ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5a)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128'"]
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5b)' --method ST --epochs 200 --lr 0.01 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-gscaling1' --fp16"]
        # ll += ["--batch_size 128 --data 'imagenet-100' --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.01 --compile -A 2 -W 2 --cudagraph --distill --fp16 -v 'fp16-gscale1'"]
        # invalidated (reason: used mean_weight factor 1.0, proj = no proj)
        #ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ReLU --epochs 200 --lr 0.001 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-gs2'"]
        #ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.001 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-gs2'"]
        #ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.001 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-gs2-proj' --fp16"]
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.001 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-gs2-porj'"]
        #ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.001 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-gs2-proj'"]
        #ll += ["--batch_size 128 --d`ata imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.01 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-gs2'"]
        #ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.001 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-gs2'"]
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-c4'"]

    elif exp == 'arch':
        # T5
        ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax'"]
        ll += ["--batch_size 128 --data 'imagenet-100' --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.01 --compile -A 2 -W 2 --cudagraph --distill --fp16 -v 'fp16-gscale1'"]
        # T5a
        ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5a)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128'"]
        ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5a)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax1'"]
        # T5b
        ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5b)' --method ST --epochs 200 --lr 0.01 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-gscaling1' --fp16"]
        ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5b)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax1'"]
        # T8
        ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower8)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax1' --fp16"]
        ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower8s)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax1' --fp16"]
        ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower8s2)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax1' --fp16"]
        ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower8s1)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax1' --fp16"]
        # b0/b1
        ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower8s)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b0-c128-Kw-bn-mw0-projmax1' --fp16"]
        # lr
        ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower8s)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax1T0.25' --fp16"]
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower8s)' --method ST --epochs 400 --lr 0.0025 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax1T0.25' --fp16"]
        # dilation
        ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower8s,Dilation=True)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill --fp16"]

    elif exp == 'mw_ablation': # mean_weight ablation
        ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128'"]
        ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw'"]
        ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn'"]
        ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-prev'"]        
        ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-new'"]
        ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0'"]
        ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.01 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0'"]
        ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax'"]
        # failed proj variants
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-proj'"]
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.0005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-gs2'"]
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.01 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-proj'"]
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-proj2'"]        
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-proj2'"]
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.001 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-proj2'"]
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-proj3'"]
        #

    elif exp == 'fp16': # mean_weight ablation
        ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax1'"]
        ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax1-fp16'"]
        ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax1-fp16.2'"]
    #

    elif exp == 'sb': # mean_weight ablation
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax1-fp16.2'"]
        ll += ["--batch_size 256 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax1'"]
        ll += ["--batch_size 256 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax1-in_sb'"]
        ll += ["--batch_size 256 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax1-in_sb2'"]
        ll += ["--batch_size 256 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax1-in_sb2-neg'"]
        ll += ["--batch_size 256 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax1-in_sb-xbn1'"]
        #
        #
        # bsb
        ll += ["--batch_size 256 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax1-bsb'"]
        ll += ["--batch_size 256 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.01 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax1-bsb'"]
        ll += ["--batch_size 256 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.01 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw/2-bn-mw0-projmax1-bsb'"]
        ll += ["--batch_size 256 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.0025 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw2-bn-mw0-projmax1-bsb'"]
        # 2x scale lr
        ll += ["--batch_size 256 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax1-in_sb-xbn1-sblr2'"]

    elif exp == 'distillT': # mean_weight ablation
        # Temperature vs no distilation 
         # T=0 (no distill)
        ll += ["--batch_size 256 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'a0-b1-c128-Kw-bn-mw0-projmax1-in_sb-xbn1'"]
        # T = 0.25
        ll += ["--batch_size 256 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax1-bsbT0.25'"]
        # T = 1
        ll += ["--batch_size 256 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax1-bsb'"]

        # 

    elif exp == 'A123': # mean_weight ablation
        #
        ll += ["--batch_size 256 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax1'"]
        ll += ["--batch_size 256 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax1-in_sb-xbn1'"]
        ll += ["--batch_size 256 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5,DB=False)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax1'"]
        ll += ["--batch_size 256 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5,DB=False)' --method ST --epochs 200 --lr 0.005 --compile -A 3 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax1'"]
        ll += ["--batch_size 256 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5,DB=False)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 3 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax1'"]
        # ll += ["--batch_size 256 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 4 -W 4 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax1' --fp16"]        
        # negate half of activations
        # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax1-inhibit' --fp16"]
        # double half of activations
        # ll += ["--batch_size 256 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-Kw-bn-mw0-projmax1-x2' --fp16"]


        # invalid (input 224) # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128' --mixup 0.1"]
        # invalid (input 224) # ll += ["--batch_size 128 --data 'imagenet-100' --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 4 -W 2 --cudagraph --distill -v 'a0-b1-c128'"]

        
        # ll += ["--batch_size 128 --data 'imagenet-100' --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.02 --compile -A 2 -W 2 --cudagraph -v 'fp16-gscaling1-T0.25' --fp16 --distil"]
        # ll += ["--batch_size 128 --data 'imagenet-100' --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.02 --compile -A 2 -W 2 --cudagraph -v 'fp16-gscaling1-T0.25-BCE' --fp16 --distil"]
        # ll += ["--batch_size 128 --data 'imagenet-100' --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.02 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-c128-fp16-T1' --fp16"]
        # ll += ["--batch_size 128 --data 'imagenet-100' --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.02 --compile -A 2 -W 2 --cudagraph -v 'fp16-gscaling1-T1-BCE' --fp16 --distil"]
        # ll += ["--batch_size 128 --data 'imagenet-100' --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.05 --compile -A 2 -W 2 --cudagraph -v 'fp16-gscaling1-T1-BCE' --fp16 --distil"]

    elif exp == 'hard-100':
        # lr experiment
        ll += ["--batch_size 256 --data 'imagenet-100(res=256,set=LA)' --MD --net 'QResNet18(gate=Tower8)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
        ll += ["--batch_size 256 --data 'imagenet-100(res=256,set=LA)' --MD --net 'QResNet18(gate=Tower8)' --method ST --epochs 200 --lr 0.0025 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]

    elif exp == 'estimators':
        # U-noise, ST-GS
        # can test here again: pre-training strategies and lrep stratrgy
        ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
        ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
        ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ZGR --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
        ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method Mean --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
        # add MeanSample (erorrs) and GS-ST(t=1)
        # ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method MeanSample --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
        ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method 'GS-ST(t=1.0)' --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
        # U noise
        ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
        ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST -n U --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
        # performs very poorly, why?
        ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ZGR -n U --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]        
        # T noise
        # ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST -n T --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
        # ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ZGR -n T --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
        # lrep
        ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST --lrep 4 --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile"]
        # pretraining
        ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ZGR --pm 'ST(epochs=150)' --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
        ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ZGR --pm 'ST(epochs=100)' --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
        ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method ST --pm 'Mean(epochs=50)' --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
        # ST-WX
        ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s)' --method 'ST(WX=True)' --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]
        # No MD ablation
        ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --net 'QResNet18(gate=Tower8s)' --method ST --epochs 200 --lr 0.0025 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"]


    elif exp == 'normal-depth':
        # plain networks
        ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Normal,DB=False,layers=[0,0,0,0])' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile"]
        # ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Normal,DB=False,layers=(1,1,1,1))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile"]
        # ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Normal,DB=False,layers=(2,2,2,2))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile"]
        # ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Normal,DB=False,layers=(3,3,3,3))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile"]
        # perhaps recompute with no extra layers after the last reduction
        ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Normal,DB=False,layers=(1,1,1,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile"]
        ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Normal,DB=False,layers=(2,2,2,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile"]
        ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Normal,DB=False,layers=(3,3,3,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile"]        
        # tower networks
        # T2, T3, T5
        ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=2,layers=(0,0,0,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile"]
        ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=2,layers=(1,1,1,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile"]
        ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=2,layers=(2,2,2,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile"]        
        # ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=2,layers=(1,1,1,1))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile"]
        # ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=2,layers=(2,2,2,2))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile"]
        # ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=3,layers=(0,0,0,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile"]
        # ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=3,layers=(1,1,1,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile"]
        #         
        ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=4,layers=(0,0,0,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile"]
        ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=4,layers=(1,1,1,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile"]
        ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=8,layers=(0,0,0,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile"]
        ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower,DB=False,order=8,layers=(1,1,1,0))' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile"]
        # !! not a fair comparison -- wideer basechannels
        # ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s,DB=False)' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile"]
        ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'QResNet18(gate=Tower8s64,DB=False)' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile"]
        # BiNeal
        ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'BiNealNet(m=1)' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile"]
        ll += ["--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'BiNealNet(m=1)' --method ST-det -n U --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile"]

        #





#imagenet-100-QResNet18(gate=Tower5)-v4-c26/W=2 A=2/ST/n=L 0.3333333333333333 MD fp16-gscaling1-T0.25-BCE distill/o=Adam lr=0.02 wd=0
#imagenet-100-QResNet18(gate=Tower5)-v4-c26/W=2 A=2/ST/n=L 0.3333333333333333 MD fp16-gscaling1-T0.25-BCE distill/o=Adam lr=0.02 wd=0/

elif dataset == 'imagenet-100' and False:
    # implementation before weight init updata
    ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=none)' --method ST --cudagraph --epochs 200 --lr 0.005 --compile -A 2 -W 2 -v 'augment-less'"]
    # new implementation of weight inint + local learning rate
    # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=none)' --method ST --cudagraph --epochs 200 --lr 0.005 --compile -A 2 -W 2 -v 'init-b-mw0'"]
    # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=none)' --method ST --cudagraph --epochs 200 --lr 0.005 --compile -A 2 -W 2 -v 'init-b-mw0.33'"]

    # Tower3 models
    #
    # ll += ["--batch_size 128 --data imagenet-100 --net 'QResNet18(gate=Tower3)' --method ST -W 2 -A 2 --MD --cudagraph --epochs 200 --lr 0.005 --compile"]


    ll += ["--batch_size 128 --data imagenet-100 --net 'QResNet18(gate=Tower3)' --method ST -W 2 -A 2 --MD --cudagraph --epochs 200 --lr 0.005 --compile -v 'old-init'"]

    # ll += ["--batch_size 128 --data imagenet-100 --net 'QResNet18(gate=Tower3)' --method ST -W 2 -A 2 --MD --cudagraph --epochs 200 --lr 0.005 --compile -v '+end-tower2'"]

    # #ll += ["--batch_size 128 --data imagenet-100 --net 'QResNet18(gate=Tower3,DB=False)' --method ST -W 2 -A 2 --MD --cudagraph --epochs 200 --lr 0.005 --compile -v '+end-tower2'"]    

    ll += ["--batch_size 128 --data imagenet-100 --net 'QResNet18(gate=Tower3)' --method ST -W 2 -A 2 --MD --cudagraph --epochs 200 --lr 0.005 --compile -v 'stage-1-normal'"]

    ll += ["--batch_size 128 --data imagenet-100 --net 'QResNet18(gate=Tower4)' --method ST -W 2 -A 2 --MD --cudagraph --epochs 200 --lr 0.005 --compile -v 'stage-1-normal'"]

    # ll += ["--batch_size 128 --data imagenet-100 --net 'QResNet18(gate=Tower4)' --method ST -W 2 -A 4 --MD --cudagraph --epochs 200 --lr 0.005 --compile -v 'stage-1-normal'"]

    # ll += ["--batch_size 128 --data imagenet-100 --net 'QResNet18(gate=Tower4)' --method ReLU -W 2 -A 2 --MD --cudagraph --epochs 200 --lr 0.005 --compile -v 'stage-1-normal'"]

    # ll += ["--batch_size 128 --data imagenet-100 --net 'QResNet18(gate=Tower3,DB=False)' --method ST -W 2 -A 2 --MD --cudagraph --epochs 200 --lr 0.005 --compile -v '+end-tower2'"]    

    # Tower4-Block0  w/wo endT3
    ll += ["--batch_size 128 --data imagenet-100 --net 'QResNet18(gate=Tower4)' --method ST -W 2 -A 2 --MD --cudagraph --epochs 200 --lr 0.005 --compile -v 'Block0'"]
    # ll += ["--batch_size 128 --data imagenet-100 --net 'QResNet18(gate=Tower4)' --method ST -W 2 -A 2 --MD --cudagraph --epochs 200 --lr 0.005 --compile -v 'Block0-end-T3'"]

    # 200-400-800
    # ll += ["--batch_size 128 --data imagenet-100 --net 'QResNet18(gate=Tower4)' --method ST -W 2 -A 2 --MD --cudagraph --epochs 200 --lr 0.005 --compile -v 'Block0'"]
    # ll += ["--batch_size 128 --data imagenet-100 --net 'QResNet18(gate=Tower4)' --method ST -W 2 -A 2 --MD --cudagraph --epochs 400 --lr 0.005 --compile -v 'Block0'"]
    # ll += ["--batch_size 128 --data imagenet-100 --net 'QResNet18(gate=Tower4)' --method ST -W 2 -A 2 --MD --cudagraph --epochs 800 --lr 0.005 --compile -v 'Block0'"]

    ll += ["--batch_size 128 --data imagenet-100 --net 'QResNet18(gate=Tower4)' --method ST -W 2 -A 2 --MD --cudagraph --epochs 200 --lr 0.005 --compile -v 'Block0' --xBN"]
    ll += ["--batch_size 128 --data imagenet-100 --net 'QResNet18(gate=Tower4)' --method ST -W 2 -A 2 --MD --cudagraph --epochs 200 --lr 0.005 --compile -v 'Block0-WN'"]

    ll += ["--batch_size 128 --data imagenet-100 --net 'QResNet18(gate=Tower4)' --method ST -W 2 -A 2 --MD --cudagraph --epochs 200 --lr 0.005 --compile -v 'Block0-WNEx' --xBN"]

    # Block0 optimization
    ll += ["--batch_size 128 --data imagenet-100 --net 'QResNet18(gate=Tower4)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --MD --cudagraph -v 'Block0--S2-Init'"]
    ll += ["--batch_size 128 --data imagenet-100 --net 'QResNet18(gate=Tower4)' --method ST -W 2 -A 2 --MD --cudagraph --epochs 200 --lr 0.005 --compile -v 'Block0_baseline'"]

    ll += ["--batch_size 128 --data imagenet-100 --net 'QResNet18(gate=Tower4)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --MD --cudagraph -v 'Block0--S2-Init+Conv'"]

    # data augmentation
    ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower4)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'augment3'"]
    ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower4)' --method ST --epochs 200 --lr 0.005 --compile -A 4 -W 2 --cudagraph -v 'augment-3'"]
    ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower4)' --method ReLU --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'augment-3'"]
    ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower4)' --method Mean --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'augment-3'"]    
    ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower4)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'augment-3-T-4-4-4'"]
    #
    ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower4)' --method ST-Mean --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'augment-3'"]
    ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Tower4)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'augment3' --tag 'T-6-4-6'"]

elif dataset == 'imagenet-100' and False: #Bineal
    #
    # BiNeal is the same as Cat, with or without convex comb
    #
    # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=none)' --method ST --cudagraph --epochs 200 --lr 0.005 --compile -A 2 -W 2 -v 'init-b-mw0.33'"]
    # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=BiNeal)' --method ST --cudagraph --epochs 200 --lr 0.005 --compile -A 2 -W 2"]
    # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=BiNeal)' --method ST --cudagraph --epochs 200 --lr 0.005 --compile -A 2 -W 2 -v 'learn-alpha'"]

    #
    # BiNeal deep vs shallow path
    #

    ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=BiNeal)' --method ST --cudagraph --epochs 200 --lr 0.005 --compile -A 2 -W 2"]
    ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Normal)' --method ST --cudagraph --epochs 200 --lr 0.005 --compile -A 2 -W 2"]
    # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=BiNeal)' --method ST --cudagraph --epochs 200 --lr 0.005 --compile -A 2 -W 2 -v '1x1-eta0'"]

    ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Normal)' --method ST --cudagraph --epochs 200 --lr 0.005 --compile -A 2 -W 2 -v 'shallow'"]

    # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=BiNeal)' --method ST --cudagraph --epochs 200 --lr 0.005 --compile -A 2 -W 2 -v 'sin'"]

elif dataset == 'imagenet-100' and False:
#     # QResNet requantize:
#     # block2 improves training but not validation
#     # sin schedule does not matter
#     # learned skip improves training slightly
    # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet(gate=none)' --method ST --cudagraph --epochs 200 --lr 0.005 --compile -A 2 -W 2"]
    ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet(gate=requantize)' --method ST --cudagraph --epochs 200 --lr 0.005 --compile -A 2 -W 2"]
 #   ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet(gate=requantize)' --method ST --cudagraph --epochs 200 --lr 0.005 --compile -A 2 -W 2 -v 'lin-schedule'"]

    # block2 improves training but not validation
    ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet(gate=none)' --method ST --cudagraph --epochs 200 --lr 0.005 --compile -A 2 -W 2"]
    # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet(gate=none)' --method ST --cudagraph --epochs 200 --lr 0.005 --compile -A 2 -W 2 -v 'block2'"]


    # Learned skip improves training slightly
    ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet(gate=none)' --method ST --cudagraph --epochs 200 --lr 0.005 --compile -A 2 -W 2 -v 'block2'"]
    ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet(gate=requantize)' --method ST --cudagraph --epochs 200 --lr 0.005 --compile -A 2 -W 2 -v 'block2-learn'"]
    # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet(gate=requantize)' --method ST --cudagraph --epochs 200 --lr 0.005 --compile -A 2 -W 2 -v 'with-SB-sin-schedule'"]

    # ll += ["--batch_size 128 --data imagenet-100 --MD --net 'QResNet18(gate=Normal)' --method ST --cudagraph --epochs 200 --lr 0.005 --compile -A 2 -W 2 -v 'shallow'"]


elif dataset == 'imagenet-10':
    # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=none)' --method ST --cudagraph --epochs 200 --lr 0.005 --compile -A 2 -W 2"]

    # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=none)' --method ST --cudagraph --epochs 200 --lr 0.005 --compile -A 2 -W 2 -v llr1"]

    # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=none)' --method ST --cudagraph --epochs 200 --lr 0.005 --compile -A 2 -W 2 -v llr2"]

    # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=none)' --method ST --cudagraph --epochs 200 --lr 0.005 --compile -A 2 -W 2 -v llr3"]

    # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=none)' --method ST --cudagraph --epochs 200 --lr 0.005 --compile -A 2 -W 2 -v llrw-2"]

    # # W4 Cat gate
    # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=none)' --method ST --cudagraph --epochs 200 --lr 0.005 --compile -A 2 -W 4 -v llrw-2"]
    # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=none)' --method ST --cudagraph --epochs 200 --lr 0.005 --compile -A 2 -W 4 -v llrw-1"]

    # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet(gate=none)' --method ST --cudagraph --epochs 200 --lr 0.005 --compile -A 2 -W 2"]

    # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet(gate=none)' --method ST --cudagraph --epochs 200 --lr 0.01 --compile -A 2 -W 2 -v 'KW/2'"]

    # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet(gate=none)' --method ST --cudagraph --epochs 200 --lr 0.05 --compile -A 2 -W 2 -v 'KW/10'"]

    # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet(gate=none)' --method ST --cudagraph --epochs 200 --lr 0.005 --compile -A 2 -W 2 -v 'old-init'"]

    # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower4)' --method ST --epochs 200 --lr 0.05 --compile -A 2 -W 2 --cudagraph"]

    # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet(gate=none)' --method ST --cudagraph --epochs 200 --lr 0.005 --compile -A 2 -W 2 --seed 2"]

    if False:
        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower4)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'KW1'"]

        # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower4)' --method ST --epochs 200 --lr 0.001 --compile -A 2 -W 2 --cudagraph -v 'KW5'"]
        
        # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower4)' --method ST --epochs 200 --lr 0.01 --compile -A 2 -W 2 --cudagraph -v 'KW/2'"]

        # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower4)' --method ST --epochs 200 --lr 0.02 --compile -A 2 -W 2 --cudagraph -v 'KW/4'"]

        # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower4)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'C256'"]

        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower4)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'Block0--S2'"]

        # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower4)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'Block0--S2' --seed 2"] 

        # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower4)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --seed 2"] 

        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower4)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'Block0--S2-xBN'"] 

        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower4)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'Block0--S2-xBN' --seed 2"]

        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower4)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'Block0--S2-Init'"]

        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower4)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'Block0_baseline'"]

        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower4)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'Block0--S2-xBN+conv'"]

        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower4)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'Block0--S2-Initx2'"]

        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower4)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'x-last-BN'"]

        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower4)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 -v 'augment-1'"]
        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower4)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 -v 'augment-0'"]
        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower4)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 -v 'augment-0-filter0'"]
        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower4)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 -v 'augment-2'"]    
        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower4)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 -v 'beta-2'"]
        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower4)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 -v 'augment-3'"]
        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower4)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 -v 'jitter-0.5'"]
        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower4)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 -v 'jitter-1'"]
        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower4)' --method ST --epochs 100 --lr 0.005 --compile -A 2 -W 2 -v 'jitter+sharp'"]
        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower4)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 -v 'jitter+sharp'"]
        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower4)' --method ST --epochs 100 --lr 0.005 --compile -A 2 -W 2 -v 'all'"]
        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower4)' --method ST-Mean --epochs 100 --lr 0.005 --compile -A 2 -W 2"]

    if False:
        # #### ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph"]
        # # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v '/20'"]
        #     # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method Mean --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph"]
        #     # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --lrep 5 --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph"]
        #     # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --lrep 10 --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph"]
        #     # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ZGR --lrep 5 --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph"]
        # #### ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v '/20'"]
        #         # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v '/10'"]
        #         # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ZGR --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v '/10'"]
        #         # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v '/10-cls'"]
        # ##### ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v '/10-cls' --seed 2"]
        # ##### ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 8 --cudagraph -v '/10'"]
        # # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 256 --cudagraph -v '/10-cls'"]
        # # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 256 --cudagraph -v 'old-cls'"]
        # # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 256 --cudagraph -v '/10-cls-K'"]    
        # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v '/10-cls-K'"]
        # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v '/10-cls-K' --seed 2"]
        # # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'Adam-mean-nK-100'"]
        # # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'Adam-mean-nK-256'"]
        # # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'Adam-mean-nK-1024'"]
        # # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'Adam-mean-K'"]
        # # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'Adam-mean-K*5'"]
        # # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'Adam-mean-K-5'"]
        # # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'Adam-mean-K*16'"]

        # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph"]
        # # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --t_stage 4"]
        # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --t_stage 4 -v 'adaptor' "]
        # # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --t_stage 4 -v 'adaptor-fc' "]
        # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --t_stage 4 -v 'adaptor' --distill"]
        # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --t_stage 4 -v 'adaptor-distill50' --distill"]
        # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill"]
        # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ReLU --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill"]
        # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'n0'"]
        # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'augment-RRC'"]
        # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'augment-B0.25'"]
        # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'scale=1'"]
        # # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 400 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill"]
        # # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 400 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'as1'"]
        # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'as1'"]
        # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'as1-mix'"]
        # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 400 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'as1-mix'"]
        # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ReLU --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'as1-mix'"]
        # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.01 --compile -A 2 -W 2 --cudagraph --distill -v 'as1-mix'"]
        # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'as1-RRC'"]

        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0'"]
        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-mix'"]
        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ReLU --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0'"]
        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ReLU --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-mix'"]
        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1'"]
        # ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --distill -v 'a0-b1-X' --mixup 0.1"]
        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 -A 2 -W 2  -v 'a0-b1-l222' --distill --compile --cudagraph"]
        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 -A 2 -W 2  -v 'a0-b1-c128' --distill --compile --cudagraph"]
        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 -A 2 -W 2  -v 'a0-b1-c128' --distill --compile --cudagraph --mixup 0.1"]

    if True:
        ll += ["--batch_size 128 --data 'imagenet-10(set=FP)' --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 -A 2 -W 2  -v 'a0-b1-c128' --distill --compile --cudagraph"]
        ll += ["--batch_size 128 --data 'imagenet-10(set=FP)' --MD --net 'QResNet18(gate=Tower5)' --method ReLU --epochs 200 --lr 0.005 -A 2 -W 2  -v 'a0-b1-c128' --distill --compile --cudagraph"]
        # ll += ["--batch_size 128 --data 'imagenet-10(set=FP)' --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'fp16' --fp16"]
        ll += ["--batch_size 128 --data 'imagenet-10(set=FP)' --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'fp16' --fp16 --distill"]
        ll += ["--batch_size 128 --data 'imagenet-10(set=FP)' --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'fp16-gscaling1' --fp16 --distill"]
        ll += ["--batch_size 128 --data 'imagenet-10(set=FP)' --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.01 --compile -A 2 -W 2 --cudagraph -v 'fp16-gradscaling1' --fp16 --distill"]
        ll += ["--batch_size 128 --data 'imagenet-10(set=FP)' --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.01 --compile -A 2 -W 2 --cudagraph -v 'fp16-gscaling1-BN16' --fp16 --distill"]
        ll += ["--batch_size 128 --data 'imagenet-10(set=FP)' --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.01 --compile -A 2 -W 2 --cudagraph -v 'fp16-gscaling1-xsq' --fp16 --distill"]
        ll += ["--batch_size 128 --data 'imagenet-10(set=FP)' --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.01 --compile -A 2 -W 2 --cudagraph -v 'fp16-gscaling1' --fp16 --distill -n U"]
        # ll += ["--batch_size 128 --data 'imagenet-10(set=FP)' --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.01 --compile -A 2 -W 2 --cudagraph -v 'fp16-gscaling1-T4' --fp16 --distill"]
        ll += ["--batch_size 128 --data 'imagenet-10(set=FP)' --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.01 --compile -A 2 -W 2 --cudagraph -v 'fp16-gscaling1-T1-BCE' --fp16 --distill"]
        ll += ["--batch_size 128 --data 'imagenet-10(set=FP)' --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.01 --compile -A 2 -W 2 --cudagraph -v 'fp16-gscaling1-T0.25-BCE' --fp16 --distill"]
        ll += ["--batch_size 128 --data 'imagenet-10(set=FP)' --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.02 --compile -A 2 -W 2 --cudagraph -v 'fp16-gscaling1-T1-BCE' --fp16 --distill"]
        ll += ["--batch_size 128 --data 'imagenet-10(set=FP)' --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.02 --compile -A 2 -W 2 --cudagraph -v 'fp16-gscaling1-T0.25' --fp16 --distill"]

    if False:
        ll += ["--batch_size 128 --data 'imagenet-10(set=LA)' --MD --net 'QResNet18(gate=Tower5)' --method ST --epochs 200 --lr 0.005 -A 2 -W 2  -v 'a0-b1-c128' --distill --compile --cudagraph"]
        ll += ["--batch_size 128 --data 'imagenet-10(set=LA)' --MD --net 'QResNet18(gate=Tower5)' --method ReLU --epochs 200 --lr 0.005 -A 2 -W 2  -v 'a0-b1-c128' --distill --compile --cudagraph"]

        
    if False:
        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph -v 'Adam-mean-K*5'"]

        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 256 --cudagraph -v '/10-cls'"]
        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 256 --cudagraph -v 'old-cls'"]
        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 256 --cudagraph -v '/10-cls-K'"]    
        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 256 --cudagraph -v 'Adam-mean-K'"]
        ll += ["--batch_size 128 --data imagenet-10 --MD --net 'QResNet18(gate=Tower3)' --method ST --epochs 200 --lr 0.005 --compile -A 2 -W 256 --cudagraph -v 'Adam-mean-K*16'"]

# # ['epoch_L1', 'tr_det_L', 'v_det_L', 'v_det_A','tr_det_A']

# %%

def sort_dicts_by_commonality(dict_list):
    
    from collections import Counter
    # Step 1: Count key occurrences
    key_counter = Counter()
    for d in dict_list:
        # print(d.keys())
        key_counter.update(d.keys())
    # Step 2: Sort the keys in each dict by frequency (descending)
    print(key_counter)
    def sort_dict_by_key_frequency(d, freq_map):
        sorted_keys = sorted(d.keys(), key=lambda k: (-freq_map[k], k))
        # print(sorted_keys)
        return {k: d[k] for k in sorted_keys}
    # Step 3: Apply to each dict
    sorted_dicts = [sort_dict_by_key_frequency(d, key_counter) for d in dict_list]
    return sorted_dicts



# def plot(ll):
unimportant = {'args_str', 'data_seed', 'test', 'unlock', 'seed', 'compile', 'cudagraph', 'MD'} #,'variant'}
important = {'method' ,'lr'}
rr = []
oo = []
od_default = dict(o_from_str('')) # default
for k in important:
    od_default.pop(k, None) # remove important from default

for (i,l) in enumerate(ll):
    res, o = load_hist(l)
    rr += [res]
    od = dict(o)
    for k in unimportant:
        if k in od.keys():
            del od[k]
    oo += [set(od.items())]

common = set.intersection(*oo)
print(f'common options: ', common)

# compute common non-default
common_nondef = common - set(dict(od_default).items())
print(f'common non-default: ', common_nondef)

uniqs = []
for i in range(len(rr)):
            res = rr[i]
            o = oo[i]
            # remove common and default
            unique = o - common - set(dict(od_default).items())
            unique = dict(unique)
            # remove unimportant
            unique = {k: v for k, v in unique.items() if k not in unimportant}
            uniqs.append(unique)

uniqs = sort_dicts_by_commonality(uniqs)

# for what in ['epoch_L1', 'epoch_A1', 'v_det_A', 'tr_det_A']:
for what in ['epoch_L1', 'epoch_A1', 'tr_det_A', 'val_A1']:
# for what in ['epoch_L1', 'epoch_A1', 'tr_det_A', 'v_det_A']:
# for what in ['L1', 'A1']:
    # plt.figure(figsize=(16,10))
    plt.figure(figsize=(10,7))
    # for p_what in ['epoch_', 'val_']:
    a = plt.gca()
    for i in range(len(rr)):
        res = rr[i]
        o = oo[i]
        unique = uniqs[i]
        o = rr[i].o
        x,y = get_xy(res, what, o.method)
        if 'epoch_' in what:
            st = '-'
        else:
            st = '-'
        if 'epoch_' in what:
            markevery = np.arange(9,len(x),10)
            marker_indices = np.arange(9, len(x), 10)  # every 10 points
        else:
            markevery = 1
            marker_indices = np.arange(0, len(x), 1)  # every 1 point
            
        if '_L1' in what:
            y_units = 1
        else:
            y_units = 100

        label = ' '
        for (k,v) in unique.items():
            # print(k, end=' ')
            if k=='winit' and v != '':
                label += f'{k} '
            else:
                if v == True:
                    label += f'{k} '
                elif v != '' and v is not None:
                    if type(v) == str:
                        if 'QResNet18' in v:
                            n, dv = parse_expression(v)
                            for k1,v1 in dv.items():
                                if type(v1) == str and v1 not in ['True', 'False','None']:
                                    label += f'{v1} ' # abbreviate keys
                                else:
                                    label += f'{k1}={v1} ' # abbreviate keys
                        else:
                            label += f'{v} '
                    else:
                        label += f'{k}={v} '
            
        if len(rr) > 3 and i < len(rr) - 3:
            lw = 0.5
            ms = 2
        else:
            lw = 2
            ms = 2
        # if st == '-':
        # a.plot(x,y*100,st, color = cc[i], marker= markers[i], label=label,linewidth=lw, markevery=markevery, markersize=ms)
        a.plot(x[0:2],y[0:2]*y_units,st, color = cc[i], marker= markers[i], label=label,linewidth=lw, markevery=100, markersize=ms*4)
        a.plot(x,y*y_units,st, color = cc[i], label=None,linewidth=lw)
        z_marker = np.random.randint(1, 5, size=len(marker_indices))
        for z in np.unique(z_marker):
            mask = z_marker == z
            # plt.scatter(x[mask], y[mask], zorder=z)
            a.scatter(x[marker_indices[mask]], y[marker_indices[mask]]*y_units, marker= markers[i], s=ms*16, color=cc[i], zorder=z, label=None)  # suppress duplicate legend

    if not '_L1' in what:
        a.set_ylim(bottom=40)
        plt.ylabel('accuracy')
        a.legend(loc=0)
    else:
        a.set_yscale('log')

    # plt.xlim(0,100)
    # plt.ylim(0.5,0.85)
    a.grid()
    title = "".join(f"{k}={v}, " for k,v in dict(common_nondef).items())
    if len(title)>2:
        title = title[:-2]
    plt.title(what + " ("+title+")")
    plt.xlabel('epochs')
    plt.draw()
    plt.show()
# plot(ll)
# %%

# %%
# %%
def load_net(s:str):
    o = o_from_str(s)
    print(o.args_str)
    current_setup = setup(o)
    o1 = copy.deepcopy(o)
    o1.batch_size = 64
    datas = current_setup.create_data(o1)
    net = current_setup.create_net(o1, print_info=True)
    state = load_state(o, net, 'best_val_A1')
    return net, datas, o1

# Load model and data and check learned mean_weights
l = ll[-1]
print(l)
net, datas, o = load_net(l)
for n,m in net.named_modules():
    if isinstance(m, QConv2d):
        qw = m.quantizer.forward(m.weight, method = o.m_eval_t_det)
        mw = qw.mean(dim=(1,2,3), keepdim=False)
        lw = m.centering.mean_weight #* 0.33
        ew = mw - lw
        print(n)
        print(f'\t mean weight  : {mw.min().item():f} {mw.max().item():f}')
        print(f'\t l_mean weight: {lw.min().item():f} {lw.max().item():f}')
        print(f'\t e_mean weight: {ew.min().item():f} {ew.max().item():f}')



# %%
if False:
    # Activation Statistics
    lvc = LoggedVarsConfiguration()
    lvc.append (LoggedVarSpecs ('Quant', 'Quant output (A)', log_forward_output = True, is_activation = 'layer.is_activation', condition = 'layer.is_activation', discretization_K = 'layer.K', logged_format = LoggedFormat.PER_CHANNEL_HISTOGRAM))
    lvc.groups = []
    DataProfiler.clear()
    DataProfiler.configure (varConfig = lvc, loggingFrequency = 0, net = net, logFirstRun = True)
    DataProfiler.enabled = True; 
    DataProfiler.logNextForwardPass()
    for (data,targets) in datas.val_loader:
        data = data.to(dev)
        targets = targets.to(dev)
        net.forward(data,method=o.m_eval_t_det)
        break

    ent = []
    print(len(DataProfiler._loggedData.items()))
    for k, v in DataProfiler._loggedData.items():
        if (len (v) > 0):
            print(k, len(v))
            h = torch.stack(v[0]['data'][2])
        

# %%

# %%
# Activation Statistics

def filter(layer) ->bool:
    return isinstance(layer, Quant) and layer.is_activation

def logPostForward(layer, method, x, y, *args, **kwargs):
    if filter(layer):
        if not hasattr(layer, 'fw_y') or layer.fw_y is None:
            layer.fw_y = y.detach()
        else:
            layer.fw_y = torch.cat([layer.fw_y, y.detach()], dim=0) # on batch dim

method = o.m_eval_t_det
method.post_hook = logPostForward

for i, (data,targets) in enumerate(datas.val_loader):
    data = data.to(dev)
    targets = targets.to(dev)
    net.forward(data,method=method)
    if i>5:
        break

# %%
names = list()
me = dict()
name2layer = dict()
ents = dict()
dead = dict()
for (n, m) in net.named_modules():
    if filter(m):
        print(n)
        y = m.fw_y.long()
        h = torch.nn.functional.one_hot(y, m.K).sum(dim=(0,2,3)) # [C K]
        # print(h)
        H = h.sum(dim=(0)) # [K]
        h = h / h.sum(dim=-1, keepdim=True)
        ent = -(h*torch.log2(h+1e-6)).sum(dim=-1) # [C]
        ents[n],_ = torch.sort(ent.cpu())
        meanEnt = ent.mean()
        me[n] = meanEnt.item()
        name2layer[n] = m
        dead[n] = sum(ent <0.01).cpu()
        H = H / H.sum()
        names += [n]
        print(H)
        # break
        # 
plt.figure()
plt.barh(me.keys(), me.values(), align='center')
plt.xlabel('average entropy [bits / neuron]')
plt.draw()
plt.show()
# %% dead units
plt.figure()
plt.barh(dead.keys(), [len(v) for v in ents.values()], align='center', edgecolor='blue', fill=False)
plt.barh(dead.keys(), dead.values(), align='center', color='red')
plt.xlabel('dead chanels')
plt.draw()
plt.show()

# %% entropy per channel
plt.figure(figsize=(8,8))
for k,v in ents.items():
    plt.plot(v, label=k)
    plt.text(len(v), v[-1], k[:3])
plt.ylabel('average entropy [bits / neuron]')
plt.xlabel('channel')
plt.title('Entropy per channel')
plt.draw()
plt.show()


# %%
for n,m in net.named_modules():
    if isinstance(m, Quant):
        print(n, m.K, m.is_activation)
# %%

#

data_sweep_accumulate_BN(net, datas.train_loader_test, o.m_eval_t_real, max_batches=100)
# %%

# %%
L1, A1 = evaluate_m(net, datas.val_loader, o.m_eval_t_real)
print(f'Val (det)      L: {L1:.4g}   A: {A1 * 100:4.2f}%')

# %%
print(net)
# %%
print(net[0].fine_scale[0].weight.norm()*0.01)
print(net[0].coarse_scale[0].weight.norm())
# %%
