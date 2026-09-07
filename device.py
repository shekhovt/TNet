# SPDX-License-Identifier: CC BY-NC-SA 4.0
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. Making code open source is essential for scientific
# transparency and reproducibility. Providing access to the code allows researchers to
# verify, replicate, and expand upon results, which supports scientific progress. Current
# top-tier conferences encourage opensource code for the purpose of reproducibility that
# is de-facto an established expectation of the research community. Opensource code as
# part of publication should not affect the performance of an invention both from a
# commercial and IP point of view.
import os
import torch

global dev
dev = torch.device('cpu')  # default

def select_device():
    global dev
    if dev != torch.device('cpu'):
        return dev

    # Detect torchrun/DDP/SLURM process
    if os.environ.get('RANK') is not None or os.environ.get('LOCAL_RANK') is not None or os.environ.get('SLURM_PROCID') is not None:
        # Do NOT set CUDA_VISIBLE_DEVICES under DDP; let train.py bind per-rank
        dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        return dev

    # Single-process fallback: pick a single most-free GPU and restrict visibility
    from nvitop import select_devices
    CUDA_VISIBLE_DEVICES = ','.join(select_devices(format='uuid', min_count=1, max_count=1,
                                                   min_free_memory='10GiB', max_gpu_utilization=50))
    if CUDA_VISIBLE_DEVICES:
        print('Setting CUDA_VISIBLE_DEVICES=', CUDA_VISIBLE_DEVICES)
        os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    t = torch.tensor([], device=dev)
    dev = t.device
    print('Selected device:', dev)
    return dev

select_device()
