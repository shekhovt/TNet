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
import relimport  # sets __package__, sys.path, __run__

if __run__:
    os.chdir(relimport.proj_dir())

_in_venv = sys.prefix != getattr(sys, 'base_prefix', sys.prefix)
if _in_venv:
    sys.path = sorted(sys.path, key=lambda x: x.find(sys.prefix) >= 0, reverse=True)    # move the virtual env to the top of the module search paths --> venv packages will take precedence over system ones
# os.environ['LD_LIBRARY_PATH'] += ':'+'/usr/lib/x86_64-linux-gnu/'
    

from argparse import ArgumentParser
from doctest import DebugRunner

from scipy import io as sio
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
import torch.amp
from torch.optim.lr_scheduler import LRScheduler
import torch.nn.functional as F
import torch.utils
from typing import Callable
# from itertools import chain
import copy
import shlex
import pyparsing as pp
# import itertools
import glob
import logging
# import warnings
import multiprocessing
# warnings.filterwarnings("error")

#________________Quant________________________
from . import device
from .tools import *
from .methods import *
from . import functional
from .training import *
from . import training
from .data_profiler import *
from .arch_imagenet import *
try:
    from .transformer import *   # vision transformer: working-repo only, absent from the public release
except ImportError:
    pass
#_____________________________________________

# DEBUG: todo disable checks with python -O
# torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.benchmark = True # benchmark multiple convolution algorithms and select the fastest
# torch._logging.set_logs(dynamo=logging.DEBUG)
# torch._logging.set_logs(
#     dynamo=logging.DEBUG,
#     inductor=logging.INFO,
#     guards=True,        # show guard logs
#     output_code=True    # log generated backend code
# )
# torch._dynamo.config.verbose = True
# torch._logging.set_logs(recompiles=True, recompiles_verbose = True)
# torch.autograd.detect_anomaly()
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True 

op = ArgumentParser()
op.set_defaults(pathversion=6)
op.add_argument("-B", "--batch_size", type=int, default=128, help="batch size")
op.add_argument('--epochs', type=int, default=200, help="how long to train before bias correction phase")
op.add_argument("--lr", type=float, default=0.01, help="learning rate")
op.add_argument("--data", type=str, default='imagenet', help="imagenet(cls=,res=,set=LA/HA/MC/LC/FP)")
op.add_argument("--workers", type=int, default=2, help="dataloader worker processes (setup_imagenet.py). Default of 2 matches a minimal Slurm CPU allocation; raise it only once a job has been submitted with more CPUs")
op.add_argument("--net", type=str, default='', help="QResNet(gate=['none', 'requantize', 'Qmask'])")
op.add_argument("-W", "--W", type=int, default=2, help="quantization levels per weight (0-continuous)")
op.add_argument("-A", "--A", type=int, default=2, help="quantization levels per activation (0-continuous)")
op.add_argument("--seed", type=int, default=1, help="network initialization seed")
op.add_argument("--data_seed", type=int, default=2, help="network initialization seed")
op.add_argument("-m", "--method", type=str, default='ST', help="method to use: ReLU, Clamp, GS(t=0.1), GS-ST(t=), ST(WX=False), ST-R, ST-det, ST-Q, ST2, ZGR, RF(M=), Mean, MeanSample, ReinMax")
# Compile
op.add_argument("--compile", action="store_true", default=False, help="Use torch.compile on some pieces")
op.add_argument("--cudagraph", action="store_true", default=False, help="use CUDA Graph")
op.add_argument("--fp16", action="store_true", default=False, help="use fp16 / mixed precision")
# Auxiliary
op.add_argument("--load", type=str, default='', help="load 'best' / 'last' / 'final' checkpoint from the experiment")
op.add_argument("--test", action="store_true", default=False, help="Test only")
op.add_argument("--unlock", action="store_true", default=False, help="remove the lock file if present")
op.add_argument("--tag", type=str, default='', help="text tag to identify running experiment")
op.add_argument("-v", "--variant", type=str, default='', help="text logged to path to differentiate modifications")
op.add_argument("-dp", "--data_profiler", type=str, nargs='?', default=None, const='', help="Log data during training for post analysis. Optionally specify a path to a configuration JSON")
op.add_argument("--pathversion", type=int, help="version of find_root")
# Data Profiling
op.add_argument("--winit", type=str, default='', help="path to a previously saved model to initialize weights from")
# Fixed Setup
op.add_argument("--optimizer", type=str, default='Adam', help="optimizer: Adam, SGD, CAdam")
op.add_argument("--Adam_eps", type=float, default=1e-5, help="Adam eps")
op.add_argument("-n", "--noise", type=str, default='L', help="noise type for relaxed quantization: L - logistic, U - uniform, T - triangular, N -- gaussian. Only logistic and normal noise has learnable sigma")
op.add_argument("--noise_sigma", type=float, default=1/3, help="quantization noise sigma for Logistic / Gaussian noise")
op.add_argument("-nW", "--noise_type_W", type=str, default='L', help="noise type for relaxed quantization: L - logistic, U - uniform, T - triangular, N -- gaussian. Only logistic and normal noise has learnable sigma")
op.add_argument("--noise_sigma_W", type=float, default=0.01, help="quantization noise sigma for Logistic / Gaussian noise")
op.add_argument("--MD", action="store_true", default=False, help="use mirror descent for weights")
# Experimental
op.add_argument('--pm', type=str, default = None, help="pretraining method(method params, epochs)")
op.add_argument("--n_samples", type=int, default=20, help="number of samples for multi-sample test")
op.add_argument("--lrep", type=int, default=1, help="local sample repeats --- in infinite limit appraches the mean propagation")
op.add_argument("--xBN", action="store_true", default=False, help="do not use BN during training")
op.add_argument("--wd", type=float, default=0.0, help="weight decay") # need to make it meaningful, e.g. (w-(K-1)/2)^2
op.add_argument("--CN", type=int, default=0, help="Corellated noise kind: 0 -- no adjustment")
op.add_argument("--NS", type=float, default=1.0, help="noise scale") # some heuristic from FouST
op.add_argument("--t_stage", type=int, default=0, help="distill quantized net from teacher at the given layer, if > 0")
op.add_argument("--distill", action="store_true", default=False, help="use teacher logits as target")
op.add_argument("--mixup", type=float, default=0.0, help="mixup data augmentation parameter")
# op.add_argument("-d", type=str, default="", help="dictionary of experimental parameters")
op.add_argument("-T", type=float, default=0.25, help="teacher temperature")
op.add_argument("--CAdam_d", type=float, default=0.01, help="CAdam delta")
op.add_argument("--beta1", type=float, default=None, help="Adam beta1")
op.add_argument("--beta2", type=float, default=None, help="Adam beta2")
# Depricated
op.add_argument("--codeversion", type=int, default=26, help="version of the code")
op.add_argument("--dynamic_dims", type=bool, default=True, help="version of the code")


def get_autocast():
    return torch.amp.autocast(device_type="cuda", enabled=o.fp16)

def get_grad_scaler():
    return torch.amp.GradScaler(init_scale = 2.0**10, growth_interval=20000, enabled = o.fp16)

# def get_grad_scaler():
    # return torch.amp.GradScaler(init_scale = 2.0**10, enabled = o.fp16, growth_factor=1.01, backoff_factor=1.0-1e-10, growth_interval=int(1e10)) # constant GradScaler

# def get_grad_scaler():
    # return torch.amp.GradScaler(init_scale = 2.0**10, enabled = False)

# def get_autocast():
#     return torch.amp.autocast(device_type="cuda", dtype=torch.float32)

# def get_grad_scaler():
#     return torch.amp.GradScaler(init_scale = 2.0**10)

def oa_from_str(s, **overwrite):
    global op
    ops, args = op.parse_known_args(shlex.split(s))
    o = dotdict(dict(**vars(ops)), **overwrite)
    o.args_str = s
    for (k, v) in overwrite.items():
        o.args_str += f" --{k}={v}"
    return o, args


def o_from_str(s, **overwrite):
    global op
    ops = op.parse_args(shlex.split(s))
    o = dotdict(dict(**vars(ops)), **overwrite)
    o.args_str = s
    for (k, v) in overwrite.items():
        o.args_str += f" --{k}={v}"
    return o
    # return oa_from_str(s, **overwrite)[0]


def str_non_default(o):
    o1 = dict(o_from_str(o.args_str))
    od_default = dict(o_from_str(''))
    if o.pathversion < 6:
        omit = {'data', 'net', 'pathversion', 'codeversion', 'method', 'W', 'A', 'data_seed', 'seed', 'optimizer', 'lr', 'wd', 'args_str', 'load', 'tag'} 
    elif o.pathversion == 6:
        omit = {'data', 'net', 'pathversion', 'codeversion', 'method', 'W', 'A', 'data_seed', 'seed', 'optimizer', 'lr', 'wd', 'args_str', 'load', 'tag','compile','cudagraph'} 
    selected = set(o1.items()) - set(od_default.items()) # demove also defaults
    oo = {k: v for k,v in selected if k not in omit} # make it a dict again
    k_sorted = sorted(oo.keys())
    s = ''
    for k in k_sorted:
        if type(oo[k]) == bool:
            s += f' {k}'
        else:
            s += f' {k}={oo[k]}'
    return s


def find_root_e(o, path=None):
    if path is None:
        path = f'res/{o.data}'
        if o.net != '':
            path += f'-{o.net}'
        path += f'-v{o.pathversion}'
        if o.codeversion > 1:
            path += f'-c{o.codeversion}'
    method_real = o.method in ('ReLU', 'Clamp', 'Mean')
    if method_real:
        exp_group = f'Real'
    else:
        exp_group = f'W={o.W:d} A={o.A:d}'
    if o.data_seed != op.get_default('data_seed'):
        exp_group += f' data_seed={o.data_seed}'
    method = f'{o.method}'
    optimizer = f'o={o.optimizer} lr={o.lr:.2g} wd={o.wd:.2g}'
    if o.seed != op.get_default('seed'):
        optimizer += f' seed={o.seed}'
    if o.pathversion <5:
        method_v = f'n={o.noise}'
        if o.noise == 'L':
            method_v += f' {o.noise_sigma}'
        if o.batch_size != 128:
            method_v += f' BS{o.batch_size}'
        if o.MD:
            method_v += f' MD'
        if o.xBN:
            method_v += f' no-BN'
        if o.CN != 0:
            method_v += f' CN{o.CN}'
        if o.NS != 1.0:
            method_v += f' NS{o.NS}'
        if o.variant != '':
            method_v += f' {o.variant}'
        if o.lrep != 1:
            method_v += f' rep{o.lrep}'
        if o.pathversion >= 2:
            if o.epochs != 200:
                method_v += f' epochs={o.epochs}'
            if o.pm is not None:
                method_v += f' pm={o.pm}'
            if o.winit != '':
                method_v += f' winit'
            if o.t_stage != 0:
                method_v += f' t_stage={o.t_stage}'
            if o.distill:
                method_v += f' distill'
            if o.mixup > 0.0:
                method_v += f' mixup={o.mixup:.2g}'
            # if o.lrstep != 100:
            #     method_v += f' lrstep={o.epochs}'
    else:
        method_v = str_non_default(o)

    root_dir = path + '/' + exp_group + '/' + \
        method + '/' + method_v + '/' + optimizer + '/'
    return root_dir, path + '/'


def find_root(o, path=None):
    return find_root_e(o, path=path)[0]


import ast
# def parse_expression(expr):
#     # Parse the expression into an AST node
#     node = ast.parse(expr, mode='eval').body

#     if not isinstance(node, ast.Call):
#         raise ValueError("Expression must be a function or class call like 'A(arg1=1)'")

#     # Extract the function/class name
#     if isinstance(node.func, ast.Name):
#         func_name = node.func.id
#     else:
#         raise ValueError("Only simple function or class names are supported")

#     # Convert keyword arguments into a dictionary
#     args_dict = {}
#     for kw in node.keywords:
#         key = kw.arg
#         value = ast.literal_eval(kw.value)
#         args_dict[key] = value

#     return func_name, args_dict

def parse_expression(expr):
    """
    Parse a string like 'A(arg1=1,arg2=2)' into ('A', {'arg1': 1, 'arg2': 2}),
    treating bare names like Normal as strings ("Normal").
    """
    node = ast.parse(expr, mode='eval').body

    if not isinstance(node, ast.Call):
        raise ValueError("Expression must be a function or class call")

    if not isinstance(node.func, ast.Name):
        raise ValueError("Only simple function/class names supported")

    func_name = node.func.id
    args_dict = {}

    for kw in node.keywords:
        key = kw.arg
        value_node = kw.value

        if isinstance(value_node, ast.Name):
            # Convert bare names to strings
            value = value_node.id
        else:
            # Use literal_eval for lists, dicts, numbers, strings, etc.
            value = ast.literal_eval(value_node)

        args_dict[key] = value

    return func_name, args_dict

def split_brackets(s: str) -> Tuple[str, dotdict]:
    # todo: interprete with python 
    LBRACE, RBRACE, EQ = map(pp.Suppress, "()=")
    identifier = pp.pyparsing_common.identifier
    key = identifier()
    value = pp.Word(pp.alphanums + '_.')
    obj_item = pp.Forward()
    key_with_value = pp.Group(
        key("key") + EQ + value("value"))
    kvs = key_with_value + pp.Optional(pp.OneOrMore(pp.Suppress(',') + key_with_value))
    params = LBRACE + kvs + RBRACE
    name = pp.Word(pp.alphanums + '_-')
    params = LBRACE + kvs + RBRACE
    grammar = name + pp.Optional(params)
    r = grammar.parseString(s)
    if len(r) == 1:
        return r[0], dotdict()
    else:
        d = dotdict(dict(r[1:]))
        return r[0], d

def p_method(m_name, m_args, o):
    if m_name == 'ReLU':  # method_real
        m = MethodReal(dotdict(MD=o.MD, q_squash = 'relu', **m_args))
    elif m_name == 'Clamp':
        m = MethodReal(dotdict(MD=o.MD, q_squash = 'clamp', **m_args))
    elif m_name == 'Mean':
        m = MethodMean(dotdict(MD=o.MD, **m_args))
    elif m_name == 'MeanSample':
        m = MethodMeanSample(dotdict(MD=o.MD, **m_args))
    elif m_name == 'ST': 
        m = MethodST(dotdict(MD=o.MD, detST=False, CN=False, temp=None, reweighting=False, **m_args))
    else:
        raise AttributeError(f"Unimplemented {m_name}")
    m.o.compile = o.compile
    return m

def setup_o(o:dotdict) ->None:
    m_name, m_args = split_brackets(o.method)
#    print(m_args)

    # network
    o.net_name, o.net_args = parse_expression(o.net)

    # o.d_str = o.d
    # o.d = parse_expression(o.d)

    # dataset
    o.data_name, o.data_args = split_brackets(o.data)

    method_real = m_name in ('ReLU', 'Clamp', 'Mean')
    o.method_real = method_real
    o.m_name = m_name
    # o.q_noise_sigma_learnable = None
    # o.q_noise_sigma = None
    noise_remap = dict(L='logistic', U='uniform', T='triangular', N='normal')
    
    assert (o.W >= 2 and o.A >= 2)
    if o.noise == 'L' or o.noise == 'N':
        o.q_noise_sigma = o.noise_sigma
        # o.q_noise_sigma_learnable = True
        o.q_noise_sigma_learnable = False
    else: # noise type U/T
        o.q_noise_sigma = 1  # currently no effect, inplementation requires specific scale
        o.q_noise_sigma_learnable = False
    
    o.QReLU_A = QReLU.Options(is_activation=True, K = o.A, from_o=o).replace(q_noise_type = noise_remap[o.noise])
    o.QReLU_W = QReLU.Options(is_activation=False, K = o.W, from_o=o).replace(q_noise_type = noise_remap[o.noise_type_W], q_noise_sigma = o.noise_sigma_W, q_noise_sigma_learnable = False)

    # pretraining method
    if o.pm is not None:
        pm_name, pm_args = split_brackets(o.pm)
        o.p_epochs = int(pm_args.epochs)
        o.m_pretrain = p_method(pm_name, pm_args, o)
    else:
        pm_name = None
        o.p_epochs = 0
        o.m_pretrain = None
                
    if m_name == 'ReLU':  # method_real
        # this sets it for weights as well, which we want if we wish to quantize post-training
        o.m_train = MethodReal(dotdict(MD=o.MD, q_squash = 'relu'))
    elif m_name == 'Clamp':
        o.m_train = MethodReal(dotdict(MD=o.MD, q_squash = 'clamp'))
    elif m_name == 'Mean':
        o.m_train = MethodMean(dotdict(MD=o.MD))
    elif m_name == 'MeanSample':
        o.m_train = MethodMeanSample(dotdict(MD=o.MD, detST=False, CN=o.CN, temp=None, reweighting=False, WX=False))
    elif m_name == 'GS' or m_name == 'GS-ST':
        o.m_train = MethodGS(dotdict(temp=float(m_args.t), STGS=( m_name == 'GS-ST'), MD=o.MD, CN=o.CN))
    elif m_name == 'ST' or m_name == 'ST-R' or m_name == 'ST-det' or m_name == 'ST-Mean':
        if m_name == 'ST-det':
            o.q_noise_sigma_learnable = False  # no effect
        temp = float(m_args.t) if hasattr(m_args, 't') else None
        WX = bool(m_args.WX) if hasattr(m_args, 'WX') else False
        o.m_train = MethodST(dotdict(temp=temp, MD=o.MD, reweighting=m_name == 'ST-R', detST=m_name == 'ST-det', CN=o.CN, WX=WX))
        if m_name == 'ST-Mean':
            switch_layer = int(m_args.switch) if hasattr(m_args, 'switch') else 2
            m2 = MethodMean(dotdict(MD=o.MD))
            o.m_train = MethodCompose(o.m_train, m2, switch_layer)

    elif m_name == 'SR':
        assert(o.noise == 'U')
        assert(o.MD == False)
        o.q_noise_sigma_learnable = False
        o.m_train = MethodST(dotdict(MD=False))
    elif m_name == 'GR':
        # o.m_train = MethodGR(dotdict(t=float(m_args.t), K=int(m_args.K), MD=o.MD, CN=o.CN))
        o.m_train = MethodGR(dotdict(GR_temp=float(m_args.t), GR_samples=int(m_args.K), MD=o.MD, CN=o.CN))
    elif m_name == 'ST2':
        o.m_train = MethodST2(dotdict(MD=o.MD))
    elif m_name == 'RF':
        o.m_train = MethodRF1(dotdict(M=int(m_args.M)))
    elif m_name == 'ARSM':
        o.m_train = MethodARSM(dotdict())
    elif m_name == 'ZGR':
        temp = float(m_args.t) if hasattr(m_args, 't') else None
        o.m_train = MethodZGR(dotdict(MD=o.MD, CN=o.CN, temp=temp, alpha=None, det=False))
    elif m_name == 'ZGR-det':
        temp = float(m_args.t) if hasattr(m_args, 't') else None
        o.m_train = MethodZGR(dotdict(MD=o.MD, CN=o.CN, det=True, temp=temp, alpha=None))
    elif m_name == 'ST-Q':
        o.m_train = MethodSTQ(dotdict(MD=o.MD, CN=o.CN))
    elif m_name == 'ReinMax':
        temp = float(m_args.t) if hasattr(m_args, 't') else None
        o.m_train = MethodReinMax(dotdict(MD=o.MD, CN=o.CN, temp=temp))
    else:
        raise AttributeError("unknown training method")
    #
    if o.lrep > 1:
        o.m_train = MethodRepeat(o.m_train, samples = o.lrep)

    o.m_train.o.compile = o.compile
    #
    test_arg = dict(compile = False, MD=False, CN=None)
    o.m_eval_t_det = MethodDet(dotdict(**test_arg))
    o.m_eval_t_sample = MethodSample(dotdict(**test_arg))
    o.m_eval_t_mult = MethodMultiSample(dotdict(n_samples=o.n_samples, **test_arg))

    if method_real:
        o.m_eval_t_real = copy.deepcopy(o.m_train)
        o.m_eval_t_real.o.compile = False

    if m_name == 'ST-Mean':
        m2 = MethodMean(dotdict(MD=o.MD))
        o.m_eval_t_det = MethodCompose(o.m_eval_t_det, m2, o.m_train.switch_layer)


    

def setup(o: dotdict):
    global current_setup
    setup_data(o)
    # print(current_setup)
    # print(training.current_setup)
    current_setup = training.current_setup
    # o.coherent_noise = (o.CN == 1)
    setup_o(o)
    # o.lock()
    return current_setup


# def new_optimizer(net, o, epoch = 0):
#     if o.optimizer == 'Adam':
#         opt = torch.optim.Adam(net.parameters(), lr=o.lr, betas=(o.beta1, o.beta2), weight_decay=o.wd)
#     elif o.optimizer == 'SGD':
#         opt = torch.optim.SGD(net.parameters(), lr=o.lr,momentum=0.9, nesterov=True, weight_decay=o.wd)
#     else:
#         raise NotImplemented(f'Do not know optimizer {o.optimizer}')
#     # assert (o.epochs >= 200)
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[o.epochs/2, o.epochs*3/4], gamma=0.1)
#     for e in range(epoch):
#         scheduler.step()
#     return opt, scheduler


# def loss(x, y, method):
    # return nn.CrossEntropyLoss(reduction='none')(x, y)


# def teacher_targets(data, targets):
#     if o.distill:
#         with torch.no_grad():
#             return teacher(data)
#     else:
#         return targets
#     # return nn.CrossEntropyLoss(reduction='none')(x, y)


def evaluate_m(net, loader, method, BN_online=False):
    if BN_online:
        net.train()  # uses BN batch statistics instead of running averages
        BN_tracking(net, False)
    else:
        net.eval()
    n_samples = 0
    L1 = 0.0
    A1 = 0.0
    with torch.no_grad():
        with SafeIter(loader) as it:
            for i, (data, targets) in enumerate(it):
                data = data.to(dev)
                targets = targets.to(dev)
                t_targets = F.one_hot(targets.to(dev), o.num_classes).float()
                assert (data.device == dev)
                # data = data.to(dev)
                # target = target.to(dev)
                scores = net.forward(data, method=method)
                L1 += net.loss(scores, t_targets).sum().item()
                # L1 += loss(scores, target, method=method).sum().item()
                # loss, scores = net.net_loss(data, target, method)
                # L1 += loss.sum().item()
                A1 += (torch.argmax(scores, dim=-1) == targets).cpu().sum().item()
                n_samples += data.size(0)
    L1 /= n_samples
    A1 /= n_samples
    #
    if BN_online:
        BN_tracking(net, True)
    return L1, A1


# def evaluate(net, loader, o, BN_online=False):
#     L1, A1 = evaluate_m(net, loader, o.m_eval1, BN_online)
#     L2, A2 = evaluate_m(net, loader, o.m_eval2, BN_online)
#     return L1, A1, L2, A2

def set_lr(opt:torch.optim.Optimizer, lr=None):
    for g in opt.param_groups:
        if lr is None:
            g["lr"] = g["saved_lr"]
        else:
            g["saved_lr"] = g["lr"]
            g["lr"] = lr

def continue_training(net, datas, opt, scheduler:LRScheduler, hist, root_dir, m_train, start_epoch, end_epoch, final_epoch, scaler:torch.amp.GradScaler | None = None):
    print('Method:', m_train.__class__.__name__)
    
    capture_input = 0
    capture_targets = 0
    capture_scores = 0
    capture_loss = 0

    def clean_dp():
        for f in glob.glob(find_root(o) + "training_*.pkl"):
            os.remove(f)
        for f in glob.glob(find_root(o) + "eval_val_*.pkl"):
            os.remove(f)

    def data_profiling(method:Method, epoch):
        # DataProfiler
        m = copy.deepcopy(method)
        m.o.compile = False
        DataProfiler.enabled = True; DataProfiler.loggingFrequency = 0; DataProfiler.logNextForwardPass ()  
        with SafeIter(datas.train_loader) as it:
            for i, (data, targets) in enumerate(it):
                data = data.to(dev)
                targets = targets.to(dev)
                net.net_loss(data, targets, method=m)
                break
        DataProfiler.enabled = False        
        if (DataProfiler.hasDataToSave ()): 
            DataProfiler.saveCheckpoint (find_root(o) + "training_", comment = f"Epoch {epoch}")

    def warmup(net, opt, method, loader, graph_it:bool=True, max_batches=10):
        """
        New function for warmup, to be used in the beginnig of each epoch
        Input: method
        graph_it -- if true will record a CUDA graph
        Return:
        a callable for forward-backward of (graphed) network with method
        """
        nonlocal capture_input
        nonlocal capture_targets
        nonlocal capture_scores
        nonlocal capture_loss        
        net.train()
        set_lr(opt, 1e-10)
        s = torch.cuda.Stream(device=dev)
        torch.cuda.synchronize()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.device(dev):
            with torch.cuda.stream(s):  # warmup on a side stream, never understood why
                # print('Warmup/Compile')
                start_e = torch.cuda.Event(enable_timing=True)
                end_e = torch.cuda.Event(enable_timing=True)
                start_e.record()
                capture_input = None
                capture_targets = None
                for m in net.modules():
                    if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                        m.momentum = None
                        m.track_running_stats = True
                        m.reset_running_stats()

                # def prewarmed_FWBW(data, targets):
                #     capture_targets.copy_(targets)
                #     capture_input.copy_(data)
                #     net.zero_grad()
                #     with get_autocast():
                #         capture_loss, capture_scores = net.net_loss(capture_input, capture_targets, method)
                #     if scaler is None:
                #         capture_loss.mean().backward()
                #         opt.step() # warmup optimizer with 0 learning rate
                #     else:
                #         scaler.scale(capture_loss.mean()).backward()
                #         scaler.step(opt)
                #         scaler.update()

                if True:
                    for i in range(max_batches):
                # with SafeIter(loader) as it:
                #     for i, (data, targets) in enumerate(it):
                #         data = data.to(dev)
                #         targets = F.one_hot(targets.to(dev), o.num_classes).float()
                        data = torch.rand([o.batch_size] + o.input_shape, dtype=torch.float32, device = dev)
                        targets = F.one_hot(torch.zeros([o.batch_size], device=dev, dtype = int), o.num_classes).float()
                        #
                        if capture_input is None:
                            capture_input = data
                        else:
                            capture_input.copy_(data)
                        if capture_targets is None:
                            capture_targets = targets
                        else:
                            capture_targets.copy_(targets)
                        # net.zero_grad(set_to_none=False)
                        net.zero_grad()
                        with get_autocast():
                            capture_loss, capture_scores = net.net_loss(capture_input, capture_targets, method)
                        if scaler is None:
                            capture_loss.mean().backward()
                        else:
                            scaler.scale(capture_loss.mean()).backward()

                        if i ==0: 
                            for n,p in net.named_parameters():
                                if p.requires_grad:
                                    if p.grad is None:
                                        print(f'{n}.grad = None')
                                    else:
                                        pass
                                        # check_real(p.grad) # disabled becasue of GradScaler?
                                    # print(f'{n}.grad.abs().max() = {p.grad.abs().max()}')
                        # if scaler is None:
                        #     opt.step() # warmup optimizer with 0 learning rate
                        # else:
                        #     scaler.step(opt)
                        #     scaler.update()
                        # if i >= max_batches:
                        #     break

                for m in net.modules():
                    if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                        m.track_running_stats = False


            end_e.record()
            torch.cuda.synchronize()
            t = start_e.elapsed_time(end_e)
            print(f'Warmup/Compile time: {t/1000:3.2f}s')
            torch.cuda.current_stream().wait_stream(s)
            #
            # TODO: How to make sure the loader is not doing CUDA operations in other processes while we are capturing the graph?
            #
            torch.cuda.synchronize()
            torch.cuda.current_stream().wait_stream(s)
            functional.all_checks = False
            if graph_it:
                G = torch.cuda.CUDAGraph()
                torch.compiler.cudagraph_mark_step_begin()
                torch.cuda.synchronize()
                with torch.cuda.graph(G):
                    net.zero_grad()
                    with get_autocast():
                        capture_loss, capture_scores = net.net_loss(capture_input, capture_targets, method)
                    if scaler is None:
                        capture_loss.mean().backward()
                    else:
                        scaler.scale(capture_loss.mean()).backward()
                torch.cuda.synchronize()
                s.wait_stream(torch.cuda.current_stream())
                torch.cuda.current_stream().wait_stream(s)
                #
                def net_FWBW(data, targets):
                    capture_input.copy_(data)
                    capture_targets.copy_(targets)
                    G.replay()
                    return capture_loss.detach(), capture_scores.detach()
            else:
                def net_FWBW(data, targets):
                    net.zero_grad()
                    with get_autocast():
                        capture_loss, capture_scores = net.net_loss(data, targets, method)
                    if scaler is None:
                        capture_loss.mean().backward()
                    else:
                        scaler.scale(capture_loss.mean()).backward()
                    return capture_loss.detach(), capture_scores.detach()
        set_lr(opt)
        return net_FWBW


    # def capture_graph(method): # in fact warmup+compile+capture graph
    #     nonlocal capture_input
    #     nonlocal capture_targets
    #     nonlocal capture_scores
    #     nonlocal capture_loss
    #     s = torch.cuda.Stream(device=dev)
    #     torch.cuda.synchronize()
    #     s.wait_stream(torch.cuda.current_stream())
    #     net.train()
    #     set_lr(opt, 1e-5)
    #     # WARMUP / Compile
    #     torch._dynamo.reset() # clear the cache of compiled functions (e.g. when swapping methods)
    #     with torch.cuda.device(dev):
    #         with torch.cuda.stream(s):  # warmup on a side stream, why?
    #             capture_input = None
    #             capture_targets = None
    #             # warmup
    #             # print('Warmup/Compile')
    #             start_e = torch.cuda.Event(enable_timing=True)
    #             end_e = torch.cuda.Event(enable_timing=True)
    #             start_e.record()
    #             for i, (data, targets) in enumerate(datas.train_loader):
    #                 data = data.to(dev)
    #                 targets = targets.to(dev)
    #                 #
    #                 if capture_input is None:
    #                     capture_input = data
    #                 else:
    #                     capture_input.copy_(data)
    #                 if capture_targets is None:
    #                     capture_targets = targets
    #                 else:
    #                     capture_targets.copy_(targets)
    #                 # net.zero_grad(set_to_none=False)
    #                 net.zero_grad()
    #                 capture_loss, capture_scores = net.net_loss(capture_input, capture_targets, method)
    #                 capture_loss.mean().backward()
    #                 opt.step()
    #                 if i > 10:
    #                     break

    #         end_e.record()
    #         torch.cuda.synchronize()
    #         t = start_e.elapsed_time(end_e)
    #         print(f'Warmup/Compile time: {t/1000:3.2f}s')
    #         torch.cuda.current_stream().wait_stream(s)
    #         functional.all_checks = False
    #         G = torch.cuda.CUDAGraph()
    #         torch.compiler.cudagraph_mark_step_begin()
    #         torch.cuda.current_stream().wait_stream(s)
    #         torch.cuda.synchronize()
    #         with torch.cuda.graph(G):
    #             net.zero_grad()
    #             capture_loss, capture_scores = net.net_loss(capture_input, capture_targets, method)
    #             capture_loss.mean().backward()
    #         torch.cuda.synchronize()
    #         s.wait_stream(torch.cuda.current_stream())
    #     set_lr(opt)
    #     return G

    def soft_teacher_targets(teacher, data, subclasses):
        with torch.no_grad():
            teacher.eval()
            with get_autocast():
                logits = teacher(data)
                # T = 1
                T = o.T
                soft_targets = torch.nn.functional.softmax(logits/T, dim=-1) # renormalized teacher to our set of subclasses -- Ok
            # if subclasses is not None:
            #     return soft_targets[subclasses]
            # else:
            return soft_targets

    def iter_mod_parameters(model: nn.Module):
        for module_name, module in model.named_modules():
            class_name = module.__class__.__name__
            for param_name, param in module.named_parameters(recurse=False):
                id = module_name + ':' + param_name
                yield id, record(module_name=module_name, class_name=class_name, param_name=param_name), param, module

    class GradSatsRecord:
        def __init__(self, net):
            self.net = net
            self.start_epoch()
            
        def start_epoch(self):
            self.d = {}
            self.N = 0 
            for id, r, p, m in iter_mod_parameters(self.net):
                self.d[id] = r
                self.d[id].grad_sq = 0
                self.d[id].numel = p.numel()
                self.d[id].shape = p.shape
                if isinstance(m, QAnyLinear):
                    self.d[id].K = m.quantizer.quant.K
                # print(id)
            # print(self.d)
    
        def add_record(self, scale):
            for id, r, p, m in iter_mod_parameters(self.net):
                if p.grad is not None:
                    self.d[id].grad_sq += (p.grad.float()**2).mean().item()*(scale**2)
            self.N += 1

        def end_epoch(self):
            for id, r, p, m in iter_mod_parameters(self.net):
                self.d[id].grad_sq /= self.N
            # print(self.d)
            return self.d


    def train_epochs(FWBW, start_epoch, end_epoch):
        te = [torch.cuda.Event(enable_timing=True) for i in range(3)]
        start_epoch_e = torch.cuda.Event(enable_timing=True)
        end_epoch_e = torch.cuda.Event(enable_timing=True)
        start_ld_e = torch.cuda.Event(enable_timing=True)
        end_ld_e = torch.cuda.Event(enable_timing=True)
        start_opt_e = torch.cuda.Event(enable_timing=True)
        end_opt_e = torch.cuda.Event(enable_timing=True)
        print(f'path={root_dir} tag={o.tag}')
        #
        print(f'lr = {scheduler.get_last_lr()}')
        net.train()
        
        # datas.train_loader.pipelines['image'][0].scale = (1.0, 1.0)
        # datas.train_loader.pipeline_specs['image'].__dict__['decoder'].scale = (1.0, 1.1)

        for epoch in range(start_epoch, end_epoch):
            torch.manual_seed(epoch) # data shuffling, independent of whether warmup, etc used before
            #
            #
            L1 = float(0.0)
            A1 = 0
            n_data = 0
            fw_time = 0
            ld_time = 0
            opt_time = 0
            gr = GradSatsRecord(net)
            # schedulers
            sched_alpha = np.sin((epoch / final_epoch)*math.pi/2)
            if datas.tr_scheduler is not None:
                datas.tr_scheduler.schedule(sched_alpha)
                datas.train_loader.recompile = True
            #
            start_epoch_e.record()
            with SafeIter(datas.train_loader) as it:
                dataiter = enumerate(it)
                # loop fetching a mini-batch of data at each iteration
                # for i, (data, targets) in enumerate(datas.train_loader):                
                while True:
                    try:
                        start_ld_e.record()
                        i, (data, targets) = next(dataiter)
                        data = data.to(dev)
                        targets = targets.to(dev)
                        end_ld_e.record()
                    except StopIteration:
                        break
                    
                    if o.mixup>0:
                        data = mixup_data(data, None, delta = o.mixup) # this will regularize towards linearity in target predictions of the teacher

                    if o.distill:
                        t_targets = soft_teacher_targets(teacher, data, datas.subclasses) # this uses augmented data, but without mixup
                    else:
                        t_targets = F.one_hot(targets,o.num_classes).float()

                    te[0].record()
                    loss_train, scores = FWBW(data, t_targets)
                    te[1].record()
                    L1 += loss_train.sum().item()
                    #
                    assert (L1 == L1)  # test for nan
                    A1 += (torch.argmax(scores, dim=-1) == targets).sum().cpu().item()
                    n_data += data.shape[0]
                    # check grad
                    if False:
                        for (n, p) in net.named_parameters():
                            if p.grad is None:
                                raise RuntimeError(f'{n} -- grad is None')
                            check_real(p.grad)
                    # make the optimization step
                    gr.add_record(scale = 1.0/scaler.get_scale() if scaler is not None else 1)
                    start_opt_e.record()
                    if scaler is None:
                        opt.step()
                    else:
                        scaler.step(opt)
                        scaler.update()
                    # project weights
                    for m in net.modules():
                        if isinstance(m, QAnyLinear):
                            m.w_project()

                    scheduler.step() # per-iteration scheduler
                    end_opt_e.record()
                    torch.cuda.synchronize()
                    fw_time += te[0].elapsed_time(te[1])
                    ld_time += start_ld_e.elapsed_time(end_ld_e)
                    opt_time += start_opt_e.elapsed_time(end_opt_e)

            end_epoch_e.record()
            torch.cuda.synchronize()
            epoch_time = start_epoch_e.elapsed_time(end_epoch_e)
            epoch_str = "{:03d}".format(epoch)
            L1 /= n_data
            A1 /= n_data
            # nonlocal A1_train
            # A1_train = A1
            
            print(f'Epoch: {epoch_str} L1: {L1:.4g}  A1: {A1 * 100:4.2f}%  time/epoch: total:{epoch_time/1000:3.2f}s ld:{(ld_time)/1000:3.2f}s fwbw:{(fw_time)/1000:3.2f}s opt:{(opt_time)/1000:3.2f}s')
            hist.L1 = np.append(hist.L1, [L1])
            hist.A1 = np.append(hist.A1, [A1])
            hist.epoch = np.append(hist.epoch, [epoch])
            hist.method = np.append(hist.method, [o.method])
            if not hasattr(hist,'gr'):
                hist.gr = []
            hist.gr.append(gr.end_epoch())

        
    epochs_chunk = 10
    print_gates(net)

    if o.data_profiler is not None:
        clean_dp()
        data_profiling(m_train, epoch=0)

    # if o.cudagraph:
    #     G = capture_graph(m_train)    
    # warmup/compile + cuda graph capture
    FWBW = warmup(net, opt, m_train, datas.train_loader, graph_it = o.cudagraph)
    ##
    for xepoch in range(start_epoch, end_epoch, epochs_chunk):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # if (not o.compile and o.data_profiler is not None): DataProfiler.enabled = True; DataProfiler.loggingFrequency = 0; DataProfiler.logNextForwardPass ()  # data logging is not available in compiled mode
        train_epochs(FWBW, xepoch, xepoch+epochs_chunk)
        # if (not o.compile and o.data_profiler is not None): DataProfiler.enabled = False
        epoch = xepoch+epochs_chunk-1
        #
        if o.data_profiler is not None:
            data_profiling(m_train, epoch)
        # evaluation on the training set
        print_gates(net)
        if o.method_real:
            data_sweep_accumulate_BN(net, datas.train_loader_test, o.m_eval_t_real)
            # evaluate as real
            L1, A1 = evaluate_m(net, datas.val_loader, o.m_eval_t_real)
            print(f'Val (real)      L: {L1:.4g}   A: {A1 * 100:4.2f}%')
            hist.v_real_L = np.append(hist.v_real_L, L1)
            hist.v_real_A = np.append(hist.v_real_A, A1)
        # evaluate as quantized

        # set BN statistics to det method
        # if (o.compile and o.data_profiler is not None): DataProfiler.enabled = True; DataProfiler.loggingFrequency = 0; DataProfiler.logNextForwardPass ()  # when compiled, the logging is not available during training. So the data are captured here instead. The notable difference is that the very first 'forward', ie. the weight initialization, is not logged
        data_sweep_accumulate_BN(net, datas.train_loader_test, o.m_eval_t_det)
        # if (DataProfiler.enabled):
        #     DataProfiler.enabled = False
        #     if (DataProfiler.hasDataToSave ()): DataProfiler.saveCheckpoint (find_root(o) + "training_", comment = f"Epoch {xepoch}")


        # measure on training set
        L1, A1 = evaluate_m(net, datas.train_loader_test, o.m_eval_t_det)
        print(f'Train (det)    L: {L1:.4g}   A: {A1 * 100:4.2f}%')
        hist.tr_det_L = np.append(hist.tr_det_L, L1)
        hist.tr_det_A = np.append(hist.tr_det_A, A1)
        
        # measure on val set
        # L1, A1 = evaluate_m(net, datas.test_loader, o.m_eval_t_det)
        # print(f'Test (det)     L: {L1:.4g}   A: {A1 * 100:4.2f}%')
        if (o.data_profiler is not None): DataProfiler.enabled = True; DataProfiler.loggingFrequency = 0; DataProfiler.logNextForwardPass ()
        L1, A1 = evaluate_m(net, datas.val_loader, o.m_eval_t_det)
        if (o.data_profiler is not None):
            DataProfiler.enabled = False
            if (DataProfiler.hasDataToSave ()): DataProfiler.saveCheckpoint (find_root(o) + "eval_val_", comment = f"Eval/val {xepoch}")
            
        print(f'Val (det)      L: {L1:.4g}   A: {A1 * 100:4.2f}%')
        hist.v_det_L = np.append(hist.v_det_L, L1)
        hist.v_det_A = np.append(hist.v_det_A, A1)
        
        # additionally evaluate noisy and multi-sample metrics
        if not o.method_real: # additional evaluations: sampling mode and mltisample mode
            # adjust BN stats for sampling
            data_sweep_accumulate_BN(net, datas.train_loader_test, o.m_eval_t_sample)
            #
            L1, A1 = evaluate_m(net, datas.train_loader_test, o.m_eval_t_sample)
            print(f'Train (sample) L: {L1:.4g}   A: {A1 * 100:4.2f}%')
            hist.tr_sample_L = np.append(hist.tr_sample_L, L1)
            hist.tr_sample_A = np.append(hist.tr_sample_A, A1)

            L1, A1 = evaluate_m(net, datas.val_loader, o.m_eval_t_sample)
            print(f'Val (sample)   L: {L1:.4g}   A: {A1 * 100:4.2f}%')
            hist.v_sample_L = np.append(hist.v_sample_L, L1)
            hist.v_sample_A = np.append(hist.v_sample_A, A1)
            #
            if False:
                L1, A1 = evaluate_m(net, datas.train_loader, o.m_eval_t_mult)
                print(f'Train (mult)   L: {L1:.4g}   A: {A1 * 100:4.2f}%')
                hist.tr_mult_L = np.append(hist.tr_mult_L, L1)
                hist.tr_mult_A = np.append(hist.tr_mult_A, A1)

                L1, A1 = evaluate_m(net, datas.val_loader, o.m_eval_t_mult)
                print(f'Val   (mult)   L: {L1:.4g}   A: {A1 * 100:4.2f}%')
                hist.v_mult_L = np.append(hist.v_mult_L, L1)
                hist.v_mult_A = np.append(hist.v_mult_A, A1)

        if o.method_real:
            A1 = hist.v_real_A[-1]
            A2 = hist.v_det_A[-1]
        else:
            A1 = hist.v_det_A[-1]
            A2 = 0 # hist.v_mult_A[-1]
        # # evaluation on the validation set
        # L1, A1, L2, A2 = evaluate(net, datas.val_loader, o, BN_online=True)
        # if o.method_real:
        #     print(f'Val L1(real): {L1:.4g}   A1(real): {A1 * 100:4.2f}%  L2(det): {L2:.4g}   A2(det): {A2 * 100:4.2f}%')
        # else:
        #     print(f'Val L1(det): {L1:.4g}   A1(det): {A1 * 100:4.2f}%  L2(mult): {L2:.4g}   A2(mult): {A2 * 100:4.2f}%')
        hist.val_epoch = np.append(hist.val_epoch, epoch)
        hist.val_L1 = np.append(hist.val_L1, L1) # depricated not used
        hist.val_A1 = np.append(hist.val_A1, A1)
        # hist.val_L2 = np.append(hist.val_L2, L2)
        # hist.val_A2 = np.append(hist.val_A2, A2)
        fname = root_dir + '/out/hist.pkl'
        force_path(fname)
        save_object(hist, fname)

        if epoch > 0:
            state = dotdict()
            state.net = net.state_dict()
            state.method = o.method
            state.epoch = epoch
            state.hist = hist
            state.o = o
            
            #save model checkpoint
            fname = root_dir + 'last.pkl'
            force_path(fname)
            save_object(state, fname)

            if A1 > hist.best_A1:
                print(f' ==== Saving best A1: {A1 * 100:4.2f}% ==== ')
                fname = root_dir + '/best_val_A1.pkl'
                force_path(fname)
                save_object(state, fname)
                hist.best_A1 = A1

            if A2 > hist.best_A2:
                print(f' ==== Saving best A2: {A2 * 100:4.2f}% ==== ')
                fname = root_dir + '/best_val_A2.pkl'
                force_path(fname)
                save_object(state, fname)
                hist.best_A2 = A2

            if epoch+epochs_chunk > final_epoch:
                fname = root_dir + 'final.pkl'
                force_path(fname)
                save_object(state, fname)
        sys.stdout.flush()
    return epoch


def run_train(o):
    root_dir = find_root(o)
    ofile = root_dir + 'args.txt'
    with open(ofile, mode='wt') as f:
        f.write(o.args_str)
        
    datas = current_setup.create_data(o)
    o.subclasses = datas.subclasses
    net = current_setup.create_net(o, print_info=True)

    if o.distill:
        global teacher        
        import torchvision.models as models
        print('Distill Soft Targets')
        # teacher = models.resnet18(pretrained=True)
        teacher = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # teacher = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1)
        for param in teacher.parameters():
            param.requires_grad = False
        teacher.eval()
        fuse_resnet(teacher)
        if o.fp16:
            teacher.half()
        teacher.to(dev)
        if datas.subclasses is not None:
            teacher.fc.weight.data = teacher.fc.weight[datas.subclasses]
            teacher.fc.bias.data = teacher.fc.bias[datas.subclasses]
            teacher.fc.out_features = len(datas.subclasses)

        
    # print_net_info(o, net)
    # m_train = o.m_train
    # m_eval1 = o.m_eval1
    # m_eval2 = o.m_eval2

    if (o.data_profiler is not None):       # if -dp argument was specified
        lvc = LoggedVarsConfiguration()
        lvc.setDefaults()
        if len (o.data_profiler) > 0:       # if the optional JSON path was given
            lvc.load (o.data_profiler)
        lvc.save (root_dir + 'loggedVarsConfiguration.json')    
        DataProfiler.configure (varConfig = lvc, loggingFrequency = 0, net = net, logFirstRun = True)

    if o.load != '':
        # start_epoch = o.load + 1
        if o.load == 'best':
            fname = 'best_val_A1'
        else:
            fname = o.load
        state = load_state(o, net, fname)
        start_epoch = state.epoch + 1
        print(f'start_epoch = {start_epoch}')
        fname = root_dir + '/out/hist.pkl'
        hist = load_object(fname)
        print(hist.keys())
        hist.L1 = hist.L1[0:start_epoch]
        hist.A1 = hist.A1[0:start_epoch]
        hist.epoch = hist.epoch[0:start_epoch]
        hist.method = hist.method[0:start_epoch]
        hist.val_L1 = hist.val_L1[hist.val_epoch < start_epoch]
        if len(hist.val_L2) > 0:
            hist.val_L2 = hist.val_L2[hist.val_epoch < start_epoch]
            hist.val_A2 = hist.val_A2[hist.val_epoch < start_epoch]
        hist.val_A1 = hist.val_A1[hist.val_epoch < start_epoch]
        hist.val_epoch = hist.val_epoch[hist.val_epoch < start_epoch]
    else:
        if o.winit != '':
            state = load_object(o.winit)
            net.load_state_dict(state.net)

        hist = dotdict()
        hist.L1 = np.array([])
        hist.A1 = np.array([])
        hist.epoch = np.array([])
        hist.method = np.array([])
        hist.val_epoch = np.array([])
        hist.val_L1 = np.array([])
        hist.val_L2 = np.array([])
        hist.val_A1 = np.array([])
        hist.val_A2 = np.array([])
        hist.train_NLL = np.array([])
        start_epoch = 1
        hist.best_A1 = 0  # accuracy
        hist.best_A2 = 0
        hist.tr_det_L = np.array([])
        hist.tr_det_A = np.array([])
        hist.tr_sample_L = np.array([])
        hist.tr_sample_A = np.array([])
        hist.tr_mult_L = np.array([])
        hist.tr_mult_A = np.array([])
        hist.v_real_L = np.array([])
        hist.v_real_A = np.array([])
        hist.v_det_L = np.array([])
        hist.v_det_A = np.array([])
        hist.v_sample_L = np.array([])
        hist.v_sample_A = np.array([])
        hist.v_mult_L = np.array([])
        hist.v_mult_A = np.array([])
        #

    # n_batches = 0
    # for i, (data, target) in enumerate(datas.train_loader):
        # n_batches += 1
    n_batches = len(datas.train_loader)
    if n_batches < 1000:
        beta2 = 1 - 2 / (n_batches + 1)
        beta1 = 1 - 2 / (n_batches / 10 + 1)
    else:
        beta1 = 0.9
        beta2 = 0.999
    o.n_batches = n_batches
    
    if o.beta1 is None:
        o.beta1 = beta1
    if o.beta2 is None:
        o.beta2 = beta2
    
    print(f'n_batches={n_batches} selecting beta1={o.beta1:.4g} beta2={o.beta2:.4g}')

    # end_epoch = start_epoch + o.epochs + 1
    end_epoch = o.epochs + 2
  
    if True: # remov BN with "init_only=True" or all BN if o.xBN
        # continue_training(o.m_train, 0, 1, final_epoch=0)
        # torch.manual_seed(3)
        # print('BN_accumulate t_sample')
        # data_sweep_accumulate_BN(net, datas.train_loader, o.m_eval_t_sample)
        # torch.manual_seed(3)
        # L1, A1 = evaluate_m(net, datas.train_loader, o.m_eval_t_sample)
        # print(f'Train (m_eval_t_sample) L: {L1:.4g}   A: {A1 * 100:4.2f}%')
        # torch.manual_seed(3)
        # L1, A1 = evaluate_m(net, datas.train_loader, o.m_train) #o.m_eval_t_sample)
        # print(f'Train (m_train) L: {L1:.4g}   A: {A1 * 100:4.2f}%')
        #
        torch.manual_seed(3)
        print('BN_accumulate train')

        if True: # test FW-BW pass executes before compiling
            s = torch.cuda.Stream(device=dev)
            torch.cuda.synchronize()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.device(dev):
                with torch.cuda.stream(s):
                    data, target = next(iter(datas.train_loader))
                    data = data.to(dev)
                    target = target.to(dev)
                    scores = net.forward(data, method=o.m_train)
                    scores.sum().backward()

        data_sweep_accumulate_BN(net, datas.train_loader, o.m_train, max_batches = 5)
        # torch.manual_seed(3)
        # L1, A1 = evaluate_m(net, datas.train_loader, o.m_train) #o.m_eval_t_sample)
        # print(f'Train (m_train) L: {L1:.4g}   A: {A1 * 100:4.2f}%')
        # torch.manual_seed(3)
        # L1, A1 = evaluate_m(net, datas.train_loader, o.m_eval_t_sample)
        # print(f'Train (m_eval_t_sample) L: {L1:.4g}   A: {A1 * 100:4.2f}%')
        # #
        mm = [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d) and ((hasattr(m,'init_only') and m.init_only) or o.xBN)]
        if o.xBN:
            print(f"Removing All BN (Replacing with ScaleBias) -- {len(mm)} instances")
        else:
            print(f"Removing BN with init_only flag -- {len(mm)} instances")

        remove_BN(net, init_only = not o.xBN)
        # torch.manual_seed(3)
        # L1, A1 = evaluate_m(net, datas.train_loader, o.m_eval_t_sample)
        # print(f'Train (t_sample) L: {L1:.4g}   A: {A1 * 100:4.2f}%')
        # torch.manual_seed(3)
        # L1, A1 = evaluate_m(net, datas.train_loader, o.m_train)
        # print(f'Train (train) L: {L1:.4g}   A: {A1 * 100:4.2f}%')
        if o.xBN:
            print(net)
        
    scaler = get_grad_scaler()
    # scaler = torch.amp.GradScaler(init_scale = 2.0**10)
    # scaler = None
    opt = None
        
    if o.compile and False: # first check correctness without compile
        o.m_train.o.compile = False
        data, target = next(iter(datas.train_loader))
        data = data.to(dev)
        target = target.to(dev)
        scores = net.forward(data, o.m_train)
        scores.sum().backward()
        o.m_train.o.compile = True

    torch.manual_seed(o.seed)
    if o.m_pretrain is not None and start_epoch == 1:
        print(f'Pretraininig using {o.pm}')
        opt, scheduler = new_optimizer(net, o, start_epoch)
        torch.manual_seed(o.seed)
        continue_training(net, datas, opt, scheduler, hist, root_dir, o.m_pretrain, start_epoch, o.p_epochs+1, final_epoch = end_epoch-1, scaler=scaler)
        start_epoch = o.p_epochs+1
    else:
        print(f'starting from epoch {start_epoch}, scheduler for {o.epochs+10} epochs')
        opt, scheduler = new_optimizer(net, o, start_epoch)
    # opt, scheduler = new_optimizer(net, o, start_epoch)    
    # calibrate_lr(net, datas.train_loader, o.m_train)
        
    print(f'Trainig ({start_epoch}, {end_epoch})')
    continue_training(net, datas, opt, scheduler, hist, root_dir, o.m_train, start_epoch, end_epoch, final_epoch = end_epoch-1, scaler=scaler)
    print(f'learning finished')


def save_state(o, state, variant):
    root_dir = find_root(o)
    fname = root_dir + f'/{variant}.pkl'
    force_path(fname)
    save_object(state, fname)


def load_state(o, net, variant):
    root_dir = find_root(o)
    fname = root_dir + f'/{variant}.pkl'
    state = load_object(fname)
    try:
        net.load_state_dict(state.net)
    except RuntimeError:
        remove_BN(net, init_only=True)
        try:
            net.load_state_dict(state.net)
        except RuntimeError:
            remove_BN(net, init_only=False)
            net.load_state_dict(state.net)
    print('Loaded state at epoch=', state.epoch)
    return state


class LoadError(RuntimeError):
    pass


def print_gates(net):
    for (n,l) in net.named_modules():
        if isinstance(l, QMask):
            print(n, end='\t')
            C = l.eta.shape[0]
            shape = (64, C, 2048//C, 2048//C)
            g, p = l.forward(shape, method = o.m_train)
            open = (g>0.5).sum(dim=(0,2,3))
            T = shape[0]*shape[2]*shape[3]
            fo = open/T # fraction of open gates per channel
            print(f'{fo.mean():3.2f}', end='  ')
            q1 = (fo<=0.25).float().mean()
            q2 = ((fo>0.25).logical_and(fo<=0.5)).float().mean()
            q3 = ((fo>0.5).logical_and(fo<=0.75)).float().mean()
            q4 = ((fo>0.75)).float().mean()
            print(f'closed<=|{q1*100:3.0f}% |{q2*100:3.0f}% |{q3*100:3.0f}% |{q4*100:3.0f}% |=>open ')

    i = 0
    for (n,l) in net.named_modules():
        if isinstance(l, QConv2d) and l.convex_comb:
            print(n, end='\t')
            print(l)
            i = i +1

def run_test(data, o, variant):
    net = current_setup.create_net(o)
    try:
        state = load_state(o, net, variant)
        print(variant)
    except FileNotFoundError as e:
        print('File not found')
        print(e)
        return
    # except RuntimeError as e:
    #     raise e
    # rec = np.array([]).reshape(0, 4)
    root_dir = find_root(o)
    variant_t = variant + '-test'
    # print_gates(net)
    res = dotdict()
    with Tee(root_dir + f'{variant_t}.txt'):
        # test as binary:
        data_sweep_accumulate_BN(net, data.train_loader_test, o.m_eval_t_det) # flat average should not depend on order
        L1, A1 = evaluate_m(net, data.test_loader, method=o.m_eval_t_det)
        print(f'Det Test:\n\tL1: {L1:.4g}   A: {A1 * 100:4.2f}%')
        res.det = dotdict(L1=L1, A1=A1)
        if o.method_real:
            data_sweep_accumulate_BN(net, data.train_loader_test, o.m_eval_t_real)
            L1, A1 = evaluate_m(net, data.test_loader, method=o.m_eval_t_real)
            print(f'Real Test\n\tL1: {L1:.4g}   A: {A1 * 100:4.2f}%')
            res.real = dotdict(L1=L1, A1=A1)
        else:
            # multi-sample test
            data_sweep_accumulate_BN(net, data.train_loader_test, o.m_eval_t_mult)
            L1, A1 = evaluate_m(net, data.test_loader, method=o.m_eval_t_mult)
            print(
                f'MultiSample Test({o.n_samples}):\n\tL1: {L1:.4g}   A: {A1 * 100:4.2f}%')
            res.mult = dotdict(L1=L1, A1=A1)
            #
            #
            #
            # res.mult_BN = dotdict(L1=L1, A1=A1)
            # # update BN
            # print('Removing BN')
            # data_sweep_accumulate_BN(net, data.train_loader, o.m_eval1)
            # net.eval()
            # remove_BN(net)
            # L1, A1 = evaluate_m(net, data.test_loader, method=o.m_eval1)
            # print(f'Det Test BN updated:\n\tL1: {L1:.4g}   A: {A1 * 100:4.2f}%')
            # res.det_BN = dotdict(L1=L1, A1=A1)
            # data_sweep_accumulate_BN(net, data.train_loader, o.m_eval2)

    fname = root_dir + f'/{variant_t}.pkl'
    save_object(res, fname)


# def get_data(o):
#     data = dotdict()
#     data.train, data.train_loader, data.val_loader, data.test, data.test_loader = current_setup.create_data(o)
#     return data


def run_tests(o, data=None):
    print('Running Tests')
    if data is None:
        data = current_setup.create_data(o)
    run_test(data, o, 'best_val_A1')
    run_test(data, o, 'best_val_A2')
    run_test(data, o, 'final')


def one_experiment(o):
    setup(o)
    # print(current_setup)
    root_dir = find_root(o)
    force_path(root_dir)
    if o.test:
        mode = 'ta'
    else:
        mode = 'tw'
    with Tee(root_dir + 'log.txt', mode):
        # print(o)
        with torch.cuda.device(dev):
            print(root_dir)
            if not o.test:
                run_train(o)
            run_tests(o)


def one_experiment_safe(o):
    r = find_root(o)
    force_path(r)
    lockfile = r + 'lock.txt'
    try:
        lock = os.open(lockfile, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        print(f"Locked: {lockfile}")
    except FileExistsError as e:
        print(f'Locking failed: {lockfile}')
        # print(e.what())
        if o.unlock:
            print(f'Openinig non-exclusive')
            lock = os.open(lockfile, os.O_CREAT | os.O_RDONLY)
        else:
            print(f'Exiting')
            return
    try:
        one_experiment(o)
    except:  # any exception including interrupt
        # release lock
        os.close(lock) # DEBUG
        os.unlink(lockfile) # DEBUG
        raise
    # release lock
    os.close(lock)
    os.unlink(lockfile)
    print(f"Completed: {lockfile}")


if __run__:
    global dev
    dev = device.select_device()
    # print(dev)
    multiprocessing.freeze_support()
    # args = op.parse_args()
    # o = dotdict(**vars(args), args_str=args_str)
    # print(ops_to_str(vars(args)))
    args_str = ' '.join(sys.argv[1:])
    o = o_from_str(args_str)
    one_experiment_safe(o)
