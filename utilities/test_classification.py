# SPDX-License-Identifier: CC BY-NC-SA 4.0
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. Making code open source is essential for scientific
# transparency and reproducibility. Providing access to the code allows researchers to
# verify, replicate, and expand upon results, which supports scientific progress. Current
# top-tier conferences encourage opensource code for the purpose of reproducibility that
# is de-facto an established expectation of the research community. Opensource code as
# part of publication should not affect the performance of an invention both from a
# commercial and IP point of view.
import os, sys
import relimport

import torch
import torchvision.models as models
import gc
import pickle
from collections import OrderedDict
from typing import Dict, Callable, Union
import logging

#________________Quant________________________
from .tools import *
from .train import setup_o, o_from_str
from .arch_imagenet import create_net, TowerBlock, CatnMerge, BiNealBlock
from .methods import compile_args
from .layers import QReLU, ScaleBias, QConv2d, EClassificationNet, ESequential
#from .op_counter import OpCounter
from collections import defaultdict
from dataclasses import dataclass

# def FW(net, x, **kwargs):
#     y = net(x, **kwargs)
#     return y

    # o = o_from_str("--net resnet18")
    # o.num_classes = num_classes
    # setup_o(o)
    # net = create_net(o)
    # net.o = o
    # yield net, o.net

    # o = o_from_str("--net Net --method ReLU")
    # setup_o(o)
    # net = Net(o)
    # net.o = o
    # yield net, "Conv+MP+SB"

    # o = o_from_str("--net Net --method ReLU --compile")
    # setup_o(o)
    # net = Net(o)
    # net = torch.compile(net, **compile_args)
    # net.o = o
    # yield net, "Conv+MP+SB-compiled"

num_classes = 1000

o = o_from_str("--net 'QResNet18(gate=Tower8s)' --method ST -A 4 -W 2")
setup_o(o)
o.num_classes = num_classes
net = create_net(o)
net.o = o
net.to(dev)
net.eval ()

batch_size = 1 # 128
image_size = 224
x = torch.rand (batch_size, 3, image_size, image_size).to(dev)

kwargs = dict(method = net.o.m_eval_t_det)
y = net (x, **kwargs)
