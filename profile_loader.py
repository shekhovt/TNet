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

if __run__:
    os.chdir(relimport.proj_dir())


import torch
import torchvision.models as models
import gc
from collections import OrderedDict
from typing import Dict, Callable
import logging

#________________Quant________________________
from .tools import *

import time
from .train import o_from_str, setup_o
from .setup_imagenet import *


# dev = torch.device('cuda:0')


if __run__:
    o = o_from_str("--batch_size 64 --data imagenette-160 --net resnet50")
    setup_o(o)
    
    datas = create_data(o)

    loader = datas.train_loader
    epochs = 5
    loader_time = 0.0
    start_e = torch.cuda.Event(enable_timing=True)
    end_e = torch.cuda.Event(enable_timing=True)
    n = len(loader)
    
    for epoch in range(epochs):
        print(f'epoch {epoch}')
        dataiter = iter(loader)
        while True:
            # t1 = time()
            start_e.record()
            data = next(dataiter, None)
            if data is None:
                break
            (x,y) = data
            x = x.to(dev)
            y= y.to(dev)
            end_e.record()
            torch.cuda.synchronize()
            time.sleep(3/n)
            t = start_e.elapsed_time(end_e)
            loader_time += t/1000
            # loader_time += time() - t1


    # Printing loader times:
    # print(f"loader accumulated time: {loader_time:.3f}")
    print(f"loader time per epoch: {loader_time/epochs:.3f}s")
