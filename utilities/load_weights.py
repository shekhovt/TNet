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

# %load_ext autoreload
# %autoreload 2

#__________________________________

from ..train import *


# %%
def load_net(path:str):
    state = load_object(path)
    o = state.o
    current_setup = setup(o)
    net = current_setup.create_net(o, print_info=True)
    remove_BN(net, init_only = not o.xBN)
    net.load_state_dict(state.net)
    return net, o

# Point this at a checkpoint .pkl saved by train.py under res/<experiment>/...
net, ol = load_net('res/<experiment>/best_val_A1.pkl')
net_setup = setup(ol)

print(net)
if False: # BN correction
    current_setup = setup(ol)
    datas = current_setup.create_data(ol)
    data_sweep_accumulate_BN(net, datas.train_loader_test, o.m_eval_t_det, max_batches=100)
    state = dotdict()
    state.net = net.state_dict()
    state.method = ol.method
    state.o = ol
    # save_object(state, 'res/<experiment>/best_val_A1-BN-corrected.pkl')

#%%
# now create some new training config, for data only
o = o_from_str("--data imagenet-100 --net QResNet18(gate=none) -W 2 -A 2 -m ST --batch_size 128")
current_setup = setup(o)
datas = current_setup.create_data(o)

if False: # TODO save which classes we trained for, if not 1K, API to prune in the arch class
    # prune network classifier
    for p in [net[-4].weight, net[-4].bias, net[-4].running_mean, net[-4].running_var, net[-5].weight, net[-5].centering.mean_weight]:
        p.data = p[datas.subclasses]

# %%
remove_BN(net,init_only=False)
L1, A1 = evaluate_m(net, datas.val_loader, o.m_eval_t_det)
print(f'Val (det)      L: {L1:.4g}   A: {A1 * 100:4.2f}%')


# %%
