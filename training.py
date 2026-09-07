# SPDX-License-Identifier: CC BY-NC-SA 4.0
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. Making code open source is essential for scientific
# transparency and reproducibility. Providing access to the code allows researchers to
# verify, replicate, and expand upon results, which supports scientific progress. Current
# top-tier conferences encourage opensource code for the purpose of reproducibility that
# is de-facto an established expectation of the research community. Opensource code as
# part of publication should not affect the performance of an invention both from a
# commercial and IP point of view.
from argparse import ArgumentParser
from doctest import DebugRunner

from scipy import io as sio
import numpy as np
import itertools

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
import torch.utils
from torch.amp import autocast, GradScaler

from typing import Callable
from itertools import chain
import copy
import shlex
import pyparsing as pp

from types import SimpleNamespace

#________________Quant________________________ 
from .tools import *
from . import setup_mnist
from . import setup_cifar
from . import setup_imagenet
from .layers import ScaleBias, QConv2d, ESequential
from .methods import MethodDet
from . import device
from . import functional
#_____________________________________________

# gloabal
global current_setup
current_setup = None

# def record(**kwargs):
#     return SimpleNamespace(**kwargs)

class SafeIter:
    def __init__(self, loader):
        self.loader = loader
        self.it = None

    def __enter__(self):
        self.it = iter(self.loader)
        return self.it
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.it is not None:
            if hasattr(self.it, 'close'):
                self.it.close()
            torch.cuda.empty_cache()

class EArgumentParser(ArgumentParser):
    def oa_from_str(self, s, **overwrite):
        """ parse options and args from str"""
        ops, args = self.parse_known_args(shlex.split(s)) 
        o = dotdict(dict(**vars(ops)), **overwrite)
        o.args_str = s
        for (k, v) in overwrite.items():
            o.args_str += f" --{k}={v}"
        return o, args


    def o_from_str(self, s, **overwrite):
        """ parse only options from str"""
        return self.oa_from_str(s, **overwrite)[0]


    def o_from_cmd(self, **overwrite):
        args_str = ' '.join(sys.argv[1:])
        o = self.o_from_str(args_str, **overwrite)
        return o


def split_brackets(s: str) -> Tuple[str, str]:
    # grammar = Word(pp.alphanums) + pp.Optional(nestedExpr('(', ')'))
    LBRACE, RBRACE, EQ = map(pp.Suppress, "()=")
    identifier = pp.pyparsing_common.identifier
    key = identifier()
    #key = pp.Word(pp.alphanums + '-')
    # value = pp.pyparsing_common.number | pp.Word(pp.printables) | pp.Word(pp.alphanums + '_') | identifier()
    value = pp.Word(pp.alphanums + '_.')
    obj_item = pp.Forward()
    key_with_value = pp.Group(
        key("key") + EQ + value("value"))
    kvs = key_with_value + pp.Optional(pp.OneOrMore(pp.Suppress(',') + key_with_value))
    params = LBRACE + kvs + RBRACE
    name = pp.Word(pp.alphanums + '_-')

    # define an overall expression, with surrounding ()'s
    params = LBRACE + kvs + RBRACE

    # value = pp.quotedString | pp.OneOrMore(pp.Word(pp.printables, excludeChars="(),") | pp.nestedExpr())
    # params = LBRACE + pp.originalTextFor(pp.delimitedList(value)) + RBRACE

    grammar = name + pp.Optional(params)
    r = grammar.parseString(s)
    if len(r) == 1:
        return r[0], dotdict()
    else:
        # print(r[1:])
        d = dotdict(dict(r[1:]))
        # d = eval("dotdict(" + r[1] + ")")
        return r[0], d


def setup_data(o):
    global current_setup
    o.data_base = o.data
    if '-D' in o.data:
        o.dynamic_binarization = True
        o.data_base = o.data_base.replace('-D', '')
    else:
        o.dynamic_binarization = False

    if '-B' in o.data:
        o.static_binarization = True
        o.data_base = o.data_base.replace('-B', '')
    else:
        o.static_binarization = False
           
    # if '-28' in o.data:
    #     o.data_rescale_to = (28,28)
    #     o.data_base = o.data_base.replace('-28', '')
    # else:
    #     o.data_rescale_to = None

    if o.data_base == 'MNIST' or o.data_base == 'FMNIST' or o.data_base == 'Omniglot' or o.data_base == 'Omniglot-28':
        current_setup = setup_mnist
    elif o.data_base == 'CIFAR':
        current_setup = setup_cifar
    elif 'imagenet' in o.data:
        o.data_base = 'imagenet'
        current_setup = setup_imagenet
    # elif 'imagenet' in o.data:
    #     if o.data == 'imagenet':
    #         o.data = 'imagenet-1000'
    #         o.data_base = o.data
    #     current_setup = setup_imagenet
    #     if 'imagenet-' in o.data:
    #         name,cls = o.data_base.split('-')
    #         o.data_base = 'imagenet'
    #         o.data_args = int(cls)
    #         # net_name, net_args = split_brackets(o.net)
    #         # o.dataset_name = net_name
    #         # o.dataset_name = net_args
    else:
        raise NotImplementedError(f'Unknown dataset {o.data}')
    # configure network parameters from the method


def ops_to_str(opsd):
    s = ''
    store_true_keys = ['MD', 'xBN', 'xsquash']
    for k, v in opsd.items():
        if k in store_true_keys:  # store_true keys, avoid writing --MD False
            if v == True:
                s += f' --{k}'
        elif v is not None:
            s += f' --{k} {v}'
    return s


def new_optimizer(net, o, epoch = 0):
    #
    lr_batch_factor = (o.batch_size / 128)**0.5
    print(f'lr_batch_factor: {lr_batch_factor}')
    lr0 = o.lr * lr_batch_factor
    all = set(net.parameters())
    has_local_lr = {p for p in all if hasattr(p, 'local_lr')}
    llr = {p.local_lr for p in has_local_lr} # set of unique values
    global_lr = all - has_local_lr
    param_groups = [{'params': list(global_lr)}]
    for local_factor in llr:
        param_groups += [{'params': [p for p in has_local_lr if p.local_lr == local_factor], 'lr': local_factor*lr0}]

    if o.optimizer == 'Adam':
        opt = torch.optim.Adam(param_groups, lr=lr0, betas=(o.beta1, o.beta2), weight_decay=o.wd, eps = o.Adam_eps)
    elif o.optimizer == 'CAdam':
        opt = CustomAdam(param_groups, lr=o.lr, betas=(o.beta1, o.beta2), weight_decay=o.wd, eps = o.Adam_eps, delta = o.CAdam_d)
    elif o.optimizer == 'SGD':
        opt = torch.optim.SGD(param_groups, lr=lr0, momentum=0.9, nesterov=True, weight_decay=o.wd)
    else:
        raise NotImplemented(f'Do not know optimizer {o.optimizer}')
            
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[o.epochs/2+1, o.epochs*3/4+1], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[o.epochs*0.9], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, o.epochs+10)

    scheduler = get_cosine_schedule_with_warmup(opt, o.n_batches, o.n_batches * (o.epochs+10)) # 1-epoch warmup should be Ok to get us started
    # scheduler = get_cosine_schedule_with_warmup(opt, o.n_batches, o.n_batches * (o.epochs+1)) # gets negative lr at the end?

    # torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    opt.step() # to suppress warning that scheduler is stepped ahead of optimizer (maybe better to save scheduler state?)
    for e in range(epoch):
        for b in range(o.n_batches):
            scheduler.step()
    return opt, scheduler


# def new_optimizer(net, o):
#     if o.optimizer == 'Adam':
#         opt = torch.optim.Adam(net.parameters(), lr=o.lr, betas=(
#             o.beta1, o.beta2), weight_decay=o.wd)
#     elif o.optimizer == 'SGD':
#         opt = torch.optim.SGD(net.parameters(), lr=o.lr,
#                               momentum=0.9, nesterov=True, weight_decay=o.wd)
#     else:
#         raise NotImplemented(f'Do not know optimizer {o.optimizer}')
#     # assert (o.epochs >= 200)
#     # scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[o.epochs/2, o.epochs*3/4], gamma=0.1)
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[], gamma=1.0)
#     return opt, scheduler

# def data_sweep_accumulate_BN(net, loader, method):
#     net.train()
#     for m in net.modules():
#         if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
#             m.momentum = None
#             m.reset_running_stats()
#     with torch.no_grad():
#         for i, (data, target) in enumerate(loader):
#             # forward pass
#             # dd = samples_to_batch(data, 5)
#             scores = net.forward(data, method)

def data_sweep_accumulate_BN(net, loader, method, max_batches=100):
    net.train()
    for m in net.modules():
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            m.momentum = None
            m.track_running_stats = True
            m.reset_running_stats()
    with torch.no_grad(), SafeIter(loader) as it:
        for i, (data, target) in enumerate(it):
            data = data.to(device.dev)
            target = target.to(device.dev)
            # forward pass
            # dd = samples_to_batch(data, 5)
            scores = net.forward(data, method=method)
            if max_batches is not None and i>max_batches:
                break
    """ Disabling running_stats:
        it is unnecessary if we do a sweep to adopt to each method and causes problems in training cuda graph
    """
    for m in net.modules():
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False

# def data_sweep_accumulate_BN(net, loader, method):
#     BNs = [m for m in net.modules() if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d)]
#     net.train() 
#     # we use train becasue we wan to use current batch statistics. but if the data is not shuffled, they will be wrong. 
#     # the network performs deep transforms depending on these statistics to calculate deeper statistics,
#     # we cannot fix this by accumulating globally over the dataset
#     # would be ok if they were not used by the network itself, maybe can do two passes?
#     # it is an impossible problem we are trying to pick one value that approximates true BN stats over a shuffled training batch
#     for m in BNs:
#         m.momentum = None
#         m.track_running_stats = True
#         m.reset_running_stats()
#         m.t_mean = torch.zeros_like(m.running_mean, dtype=torch.float64)
#         m.t_M2 = torch.zeros_like(m.running_mean, dtype=torch.float64)
#     nd = 0
#     with torch.no_grad():
#         for batches, (data, target) in enumerate(loader):
#             data = data.to(device.dev)
#             target = target.to(device.dev)
#             scores = net.forward(data, method)
#             bs = data.shape[0]
#             for m in BNs:
#                 m.t_mean = m.t_mean + m.running_mean*bs
#                 m.t_M2 = m.t_M2 + (m.running_var.to(dtype=torch.float64) + m.running_mean**2)*bs
#                 m.momentum = None                
#                 m.track_running_stats = True
#                 m.reset_running_stats()
#             nd += bs
#         print(nd)
#         for i,m in enumerate(BNs):
#             mean = m.t_mean / nd
#             m.running_mean = mean.float()
#             m.running_var = (m.t_M2 / nd - mean**2).float()
#             del m.t_mean
#             del m.t_M2
#         print(m.running_mean[0].item(), m.running_var[0].item())

#     """ Disabling running_stats:
#         it is unnecessary if we do a sweep to adopt to each method and causes problems in training cuda graph
#     """
#     for m in BNs:
#         m.track_running_stats = False



def remove_BN(net: Union[nn.ModuleList, nn.Sequential, nn.Module], init_only=True):
    def BN_must_replace(M:nn.BatchNorm2d):
        return  (not init_only or (hasattr(M,"init_only") and M.init_only))
        
    def BN_replacement(M:nn.BatchNorm2d):
        mu = M.running_mean
        v = M.running_var
        std = ((v + M.eps) ** 0.5)
        if M.affine:
            s = M.weight / std
            b = M.bias - mu * s
        else:
            s = 1 / std
            b = - mu * s
        # test time BN(x) equals to  x * s + b
        l = ScaleBias(channels=M.num_features, learnable=M.affine, is_activation=True) # if BN was learnable this will be learnable
        l.set_affine(s, b)
        return l

    # remove in lists
    if isinstance(net, nn.Sequential) or isinstance(net, nn.ModuleList):
        for (i, M) in (enumerate(net)): # remove in Sequntial / List
            if (isinstance(M, nn.BatchNorm1d) or isinstance(M, nn.BatchNorm2d)):
                if BN_must_replace(M):
                    l = BN_replacement(M)
                    net[i] = l
                    print(f'+1 removed')
            else:
                remove_BN(M, init_only=init_only) # remove in sublist / children
    
    # remove in submodules
    for name, M in net.named_children():
        if (isinstance(M, nn.BatchNorm1d) or isinstance(M, nn.BatchNorm2d)):
            if BN_must_replace(M):
                l = BN_replacement(M)
                net.__setattr__(name,l)
                print(f'+1 removed')
        else:
            remove_BN(M, init_only=init_only)



def BN_tracking(net, on=True):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = on


def print_weight_utilization(net):
    for n, l in net.named_modules():
        if isinstance(l, QConv2d):
            print('layer', n.ljust(2), end=': ')
            K = l.quantizer.quant.K
            try:
                del l.quantizer.out_sb
            except:
                pass
            qw = l.quantizer.forward(l.weight, method=MethodDet(dotdict(MD=False)))
            N = qw.view(-1).shape[0]
            hh = [(qw == k).sum().item()/N for k in range(K)]
            print('state utilization=', ' '.join([f'{v*100:3.2f}%' for v in hh]))


import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


# def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs):
#     def lr_lambda(current_epoch):
#         if current_epoch < warmup_epochs:
#             return float(current_epoch) / float(max(1, warmup_epochs))
#         progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
#         return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine decay

#     return LambdaLR(optimizer, lr_lambda)

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

class CustomAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay = 0, delta = 0.1):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(CustomAdam, self).__init__(params, defaults)
        self.delta = delta

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('CustomAdam does not support sparse gradients.')

                state = self.state[p]

                # Initialize state on first use
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Update moving averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # gsq = (grad*grad).mean(dim = [*range(1, grad.dim())], keepdim=True)
                # exp_avg_sq.mul_(beta2).add_(gsq, alpha=1 - beta2)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias-corrected moving averages
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # denom = (exp_avg_sq/ bias_correction2).sqrt().add_(group['eps'])
                delta = self.delta
                if hasattr(p, 'eps'): # Conv weight kernel
                    eps = p.eps
                else:
                    eps = group['eps']
                
                # denom = (exp_avg_sq*(1-delta) + exp_avg_sq.mean(dim = [*range(1, grad.dim())], keepdim=True) * delta).sqrt() + eps
                denom = ((exp_avg_sq*(1-delta) + exp_avg_sq.mean() * delta)/bias_correction2).sqrt() + eps

                step_size = group['lr'] / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
    

def mixup_data(x, y=None, delta = 0.1):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if delta > 0:
        lam = 1 - np.random.uniform()*delta
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    if y is None:
        return mixed_x
    else:
        mixed_y = lam * y + (1-lam) * y[index]
        return mixed_x, mixed_y
    
from torch.ao.quantization import fuse_modules


def fuse_resnet(model):
    # Fuse stem
    fuse_modules(model, ['conv1', 'bn1', 'relu'], inplace=True)

    for module_name, module in model.named_children():
        if module_name.startswith('layer'):
            for bottleneck in module:
                # Only fuse conv + bn in Bottleneck, not relu
                fuse_modules(bottleneck, ['conv1', 'bn1'], inplace=True)
                fuse_modules(bottleneck, ['conv2', 'bn2'], inplace=True)
                fuse_modules(bottleneck, ['conv3', 'bn3'], inplace=True)

                # Downsample path
                if hasattr(bottleneck, 'downsample') and isinstance(bottleneck.downsample, nn.Sequential):
                    modules = [name for name, _ in bottleneck.downsample.named_children()]
                    if '0' in modules and '1' in modules:
                        fuse_modules(bottleneck.downsample, ['0', '1'], inplace=True)

    return model

def fuse_BN(model):
    # Basic blocks
    for module_name, module in model.named_children():
        if isinstance(module, torch.nn.Sequential):
            for i, block in module.named_children():
                if hasattr(block, "conv1") and hasattr(block, "bn1"):
                    fuse_modules(block, ['conv1', 'bn1'], inplace=True)
                if hasattr(block, "conv2") and hasattr(block, "bn2"):
                    fuse_modules(block, ['conv2', 'bn2'], inplace=True)
                if hasattr(block, "conv3") and hasattr(block, "bn3"):
                    fuse_modules(block, ['conv3', 'bn3'], inplace=True)
        elif hasattr(module, "conv1") and hasattr(module, "bn1"):
            fuse_modules(module, ['conv1', 'bn1'], inplace=True)
    return model


import torch
import torch.nn as nn
from contextlib import contextmanager
from collections import defaultdict

class AutocastDtypeTracker:
    def __init__(self, model):
        self.model = model
        self.dtype_map = defaultdict(list)

    def _hook(self, name):
        def wrapper(module, input, output):
            dtype = output.dtype if isinstance(output, torch.Tensor) else type(output)
            self.dtype_map[dtype].append((name, module.__class__.__name__))
        return wrapper

    @contextmanager
    def track(self):
        hooks = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # leaf modules only
                hooks.append(module.register_forward_hook(self._hook(name)))
        try:
            yield
        finally:
            for h in hooks:
                h.remove()

    def summary(self):
        print("\n=== Autocast Dtype Summary ===")
        for dtype, layers in self.dtype_map.items():
            print(f"\n{dtype} ({len(layers)} layers):")
            for name, modname in layers:
                print(f"  - {name}: {modname}")

def prof_model(model):
    # turch performance profiler
    model = model.half().cuda()  # model is in fp16
    tracker = AutocastDtypeTracker(model)
    example_input = torch.randn(1, 3, 224, 224).cuda()
    with tracker.track():
        with torch.amp.autocast():
            output = model(example_input)
    tracker.summary()

    with torch.profiler.profile(with_stack=True, profile_memory=True) as prof:
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(input)

    print(prof.key_averages().table(sort_by="cuda_time_total"))


class Graphed_FWBW:
    def copy_inputs(self, args):
        for i, a in enumerate(args):
            self.capture_args[i].copy_(a)

    def get_autocast(self):
        return torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_enabled)

    def __init__(self, net, net_loss, args, kwargs={}, graph_it:bool=True, dev = None, fp16 = True, scaler = None, warmup_batches:int=10, amp_dtype=None, amp_enabled=None, sync_after_replay: bool=False):
        if dev is None:
            assert(isinstance(args[0], torch.Tensor))
            self.dev = args[0].device
        else:
            self.dev = dev
        self.capture_args = copy.deepcopy(args)
        self.capture_kwargs = copy.deepcopy(kwargs)
        self.capture_loss = None
        self.capture_outputs = None
        self.graph_it = graph_it
        self.fp16 = fp16
        # Autocast config: enable only if dtype is provided or explicitly enabled
        self.amp_dtype = amp_dtype
        self.amp_enabled = (bool(amp_enabled) if amp_enabled is not None else (amp_dtype is not None))
        # Synchronization behavior after graph replay
        self.sync_after_replay = sync_after_replay

        # Simple helpers for debugging data feeding
        def _tensor_checksum(t: torch.Tensor, n: int = 1024):
            try:
                flat = t.view(-1)
                n = min(n, flat.numel())
                return flat[:n].float().sum().item()
            except Exception:
                return None

        self.input_checksum = lambda: [
            _tensor_checksum(t) if isinstance(t, torch.Tensor) else None for t in self.capture_args
        ]

        def scaled_wf_bw(*args, **kwargs):
            with self.get_autocast():
                self.capture_outputs = net_loss(*args, **kwargs)
                if isinstance(self.capture_outputs, tuple) or isinstance(self.capture_outputs, list):
                    self.capture_loss = self.capture_outputs[0]
                else:
                    self.capture_loss = self.capture_outputs
            if scaler is None:
                self.capture_loss.mean().backward()
            else:
                scaler.scale(self.capture_loss.mean()).backward()


        # WARMUP and CAPTURE
        net.train()
        s = torch.cuda.Stream(device=self.dev)
        torch.cuda.synchronize()
        s.wait_stream(torch.cuda.current_stream())

        with torch.cuda.device(self.dev):
            with torch.cuda.stream(s):  # warmup on a side stream, never understood why
                start_e = torch.cuda.Event(enable_timing=True)
                end_e = torch.cuda.Event(enable_timing=True)
                start_e.record()

                # Reset BN stats
                for m in net.modules():
                    if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                        # m.momentum = None
                        m.track_running_stats = False
                        # m.reset_running_stats()

                # WARMUP
                for i in range(warmup_batches):
                    scaled_wf_bw(*self.capture_args, **self.capture_kwargs)
                    if i ==0: 
                        for n,p in net.named_parameters():
                            if p.requires_grad:
                                if p.grad is None:
                                    print(f'{n}.grad = None')
                                else:
                                    pass
                # BN enable tracking
                for m in net.modules():
                    if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                        m.track_running_stats = True
                        m.reset_running_stats()


            end_e.record()
            torch.cuda.synchronize()
            t = start_e.elapsed_time(end_e)
            print(f'Warmup/Compile time: {t/1000:3.2f}s')

            # Capture CUDA Graph    
            torch.cuda.synchronize()
            torch.cuda.current_stream().wait_stream(s)

            functional.all_checks = False
            if graph_it:
                self.G = torch.cuda.CUDAGraph()
                torch.compiler.cudagraph_mark_step_begin()
                torch.cuda.synchronize()
                with torch.cuda.graph(self.G):
                    scaled_wf_bw(*self.capture_args, **self.capture_kwargs)
                torch.cuda.synchronize()
                s.wait_stream(torch.cuda.current_stream())
                torch.cuda.current_stream().wait_stream(s)

                def net_FWBW(*args, **kwargs):
                    # check kwargs are matching to those used at capture time
                    assert kwargs == self.capture_kwargs
                    self.copy_inputs(args)
                    # Replay the captured CUDA graph (forward + backward).
                    # The underlying storage of capture tensors (incl. loss) updates on replay.
                    self.G.replay()
                    if self.sync_after_replay:
                        torch.cuda.current_stream().synchronize()
                    # if isinstance(self.capture_outputs, tuple) or isinstance(self.capture_outputs, list):
                    #     self.capture_loss = self.capture_outputs[0]
                    # else:
                    #     self.capture_loss = self.capture_outputs
                    return self.capture_loss.detach()
            else:
                def net_FWBW(*args, **kwargs):
                    scaled_wf_bw(*args, **kwargs)
                    return self.capture_loss.detach()
        self.net_FWBW = net_FWBW
       
    def __call__(self, *args, **kwargs):
        return self.net_FWBW(*args, **kwargs)