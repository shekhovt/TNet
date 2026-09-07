# SPDX-License-Identifier: CC BY-NC-SA 4.0
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. Making code open source is essential for scientific
# transparency and reproducibility. Providing access to the code allows researchers to
# verify, replicate, and expand upon results, which supports scientific progress. Current
# top-tier conferences encourage opensource code for the purpose of reproducibility that
# is de-facto an established expectation of the research community. Opensource code as
# part of publication should not affect the performance of an invention both from a
# commercial and IP point of view.

# --- What this file is -----------------------------------------------------------------------
# The original energy script, kept UNCHANGED as the reference implementation -- the file name is
# historical and describes it poorly. Two things in it are load-bearing: fuse_model(), which
# walks a constructed network into the list of fused convolutions this whole package is defined
# on and which the native entry path still calls; and net_calc_energy_stats(), which prints
# energy totals and is the oracle tests/test_reference.py checks the current cost model against.

import os, sys
import relimport

if __run__:
    os.chdir(relimport.proj_dir())
    
import torch
import torchvision.models as models
import gc
import pickle
from collections import OrderedDict
from typing import Dict, Callable, Union
import logging

#________________Quant________________________
from ..tools import *
from ..train import setup_o, o_from_str
from ..arch_imagenet import create_net, TowerBlock, CatnMerge, BiNealBlock
from ..methods import compile_args
from ..layers import QReLU, ScaleBias, QConv2d, EClassificationNet, ESequential
from ..layers import *
#from ..op_counter import OpCounter
from collections import defaultdict
from dataclasses import dataclass
    
#________________DEBUG________________________
# TORCH_LOGS=guards
# torch._logging.set_logs(dynamo=logging.DEBUG)
# torch._logging.set_logs(dynamo=logging.INFO)
# torch._logging.set_logs(guards=True)
# torch._logging.set_logs(recompiles=True)
# torch._logging.set_logs(recompiles=True, recompiles_verbose = True, fusion = True)
# torch._logging.set_logs(recompiles=True, recompiles_verbose = True)
# dev = torch.device('cuda:0')

Mb = 1024**2


class FusedConv:#(nn.Module):
    def __init__(self, conv: nn.Conv2d | nn.Linear, affine:ScaleBias, pooling: nn.MaxPool2d | nn.AvgPool2d | nn.AdaptiveAvgPool2d | None = None):
#        super().__init__()
        self.conv = conv
        self.affine = affine
        self.pooling = pooling
        self.K_out = 0
        self.K_in = 0
        self.K_W = 0
        self.in_size = tuple()
        self.conv_out_size = tuple()
        self.weights_size = tuple ()

    def __repr__(self):
        return f'in_size={self.in_size} K_in={self.K_in} @ (W_shape={tuple(self.conv.weight.shape) if hasattr (self, "conv") else self.weights_size} K_W= {self.K_W}) -> conv_out_size={self.conv_out_size} -> reduction={self.pooling}, K_out={self.K_out}'

def flatten_model_list(M):
    result = []
    if isinstance(M, ESequential) and not isinstance(M, QReLU):
        for mod in M:
            result.extend(flatten_model_list(mod))
    elif isinstance(M, TowerBlock):
        if M.residual is None:
            result.extend(flatten_model_list(M.normal))
        else:
            result.extend(flatten_model_list(M.residual))
            result.extend(flatten_model_list(M.merge))
    elif isinstance(M, CatnMerge):
        result.extend(flatten_model_list(M.merge))
    elif isinstance(M, BiNealBlock):
        M.normal[-2].K_out = 2**4
        M.skip[-2].K_out = 2**4
        result.extend(flatten_model_list(M.normal))
        result.extend(flatten_model_list(M.skip))
    else:
        result.append(M)
    return result

def fuse_model(model:EClassificationNet):
    ll = flatten_model_list(model)
    print(ll)
    rr = []
    fc = None
    in_K = 8   
    if model.o.net.startswith ('MobileNet'):
        in_K = 256
    for l in ll:
        if isinstance(l, QConv2d):
            if fc is not None: # append last fused conv
                if fc.K_out == 0:
                    fc.K_out = 2**8
                rr.append(fc)
            fc = FusedConv(l, ScaleBias(l.weight.shape[0])) # new conv
            fc.K_in = in_K
            fc.K_W = l.quantizer.quant.K
            fc.in_size = l.in_size
            fc.conv_out_size = l.out_size
            if hasattr(l, 'K_out'):
                fc.K_out = l.K_out
        elif isinstance(l, nn.MaxPool2d) or isinstance(l, nn.AvgPool2d) or isinstance(l, nn.AdaptiveAvgPool2d):
            assert(fc is not None)
            fc.pooling = l
        elif isinstance(l, QReLU):
            assert(fc is not None)
            if fc.K_out == 0:
                fc.K_out = l.quant.K
            in_K = l.quant.K
        elif isinstance(l, nn.Conv2d) or isinstance(l, nn.Linear):
            if fc is not None: # append last fused conv
                rr.append(fc)            
            fc = FusedConv(l, ScaleBias(l.weight.shape[0])) # new conv
            fc.K_in = in_K
            fc.K_W = 8 
            if model.o.net.startswith ('MobileNet'):
                fc.K_W = 256 # 2**32
            fc.in_size = l.in_size
            fc.conv_out_size = l.out_size

    if fc is not None:
        if fc.K_out == 0:
            fc.K_out = 2**8
        rr.append(fc)
    return rr


class Net(nn.Module):
    def __init__(self, o, *args, **kwargs):
        super().__init__()
        channels = 256
        self.conv = nn.Conv2d(3,channels,3,1,3//2)
        self.mp = nn.MaxPool2d(kernel_size=2,stride=2)
        self.weight = torch.nn.Parameter(torch.rand(channels))
        self.bias = torch.nn.Parameter(torch.rand(channels))

    def forward(self, x):
        x = self.conv(x)
        x = self.mp(x)
        x = x*self.weight.view([1,-1,1,1]) + self.bias.view([1,-1,1,1])
        return x


def mem_usage():
    method = 3
    if method == 0:
        return torch.cuda.memory_allocated(0)
    elif method == 1:
        stats = torch.cuda.memory_stats(0)
        f = stats['allocated_bytes.all.freed']
        t = stats['allocated_bytes.all.allocated']
        c = stats['allocated_bytes.all.current']
        return t-f
        # return stats.active_bytes.all.current
    elif method == 2:
        return torch.cuda.max_memory_allocated()
    elif method == 3: # not a good method, it is for GPU not for the process
        (f, t) = torch.cuda.mem_get_info()
        mem = t-f # total - free
        return mem

def prof_models():
    num_classes = 1000
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

    # o = o_from_str("--net 'Net()' --method ST-det")
    # setup_o(o)
    # net.o = o
    # yield net, "Net"

    # o = o_from_str("--net 'Net()' --method ST-det --compile")
    # setup_o(o)
    # # net = torch.compile(net, **compile_args)
    # net.o = o
    # yield net, "Net-compiled"

    # o = o_from_str("--net 'QResNet18(gate=Tower8s)' --method ST -A 4 -W 2")
    o = o_from_str("--net 'ResNet18()' --method ReLU -A 2 -W 2")
    setup_o(o)
    o.num_classes = num_classes
    net = create_net(o)
    net.o = o
    yield net, "ResNet18"

    # o = o_from_str("--batch_size 256 --data 'imagenet-10' --MD --net 'QResNet18(gate=GenResSA)' --method ST-det -n U --epochs 200 --lr 0.002 --compile -A 2 -W 2 --cudagraph --Adam_eps 1e-8")
    # setup_o(o)
    # o.num_classes = num_classes
    # net = create_net(o)
    # net.o = o
    # yield net, "GenResSA"

    return

    # method = 'ST'
    method = 'MeanSample'
    # net_arg = "QResNet(gate=requantize,DB=True,DC=False)"
    net_arg = "QResNet18(gate=none)"
    o = o_from_str(f"--method {method} -W 2 -A 2 -n L --MD --net {net_arg}")
    o.num_classes = num_classes
    setup_o(o)
    net = create_net(o)
    net.o = o
    yield net, o.net
    
    o = o_from_str(f"--method {method} -W 2 -A 2 -n L --MD --net {net_arg} --compile")
    o.num_classes = num_classes
    setup_o(o)
    net = create_net(o)
    net.o = o
    yield net, o.net + '-compiled'

def print_shape(module, args, output):
    for name, m in module.net[0].named_modules():
        if m is module:
            print(name + ': ' + str(module))
    mem = np.prod(output.shape)*4 / (1024**2) # assuming floats

    print('inputs: ' + ', '.join(str(list(a.shape)) for a in args))
    print(f'weights: {np.prod(module.weight.shape)/1000000.0:3.2f}M')
    print(f'output f shape= {list(output.shape)} mem={mem:4.0f}Mb')
    module.in_size = tuple(args[0].shape)
    module.out_size = tuple(output.shape)

def add_hooks(net):
    for m in net.modules():
        if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear): # or isinstance(m,nn.BatchNorm2d):
            m.register_forward_hook(print_shape)
            m.net = [net]
            
def remove_hooks(net):
    for m in net.modules():
        if hasattr(m, "_forward_hooks"):
            m._forward_hooks: Dict[int, Callable] = OrderedDict()


def FW(net, x, **kwargs):
    y = net(x, **kwargs)
    return y
   
def compute(net, x, **kwargs):
    y = net(x, **kwargs)
    (y**2).sum().backward()

def profile_net(net, x, **kwargs):
    gc.collect()
    torch.cuda.empty_cache()
    start_e = torch.cuda.Event(enable_timing=True)
    end_e = torch.cuda.Event(enable_timing=True)
    s = torch.cuda.Stream(device=dev)
    s.wait_stream(torch.cuda.current_stream())
    opt = torch.optim.Adam(net.parameters())
    
    def benchmark_FW(T, BW=False):
        opt.zero_grad()
        gc.collect()
        torch.cuda.empty_cache()
        mem0 = mem_usage() # total - free, model and saved grads mem
        tt = 0
        for j in range(T):
            start_e.record()                
            opt.zero_grad()
            if BW:
                compute(net,x, **kwargs)
            else:
                FW(net,x, **kwargs)
            end_e.record()
            torch.cuda.synchronize()
            t = start_e.elapsed_time(end_e)
            tt += t
        pre = 'FW' if not BW else 'FW-BW'
        print(f'{pre} Time: {tt/T:.1f} ms')
        mem = mem_usage()
        print(f'{pre} mem: {(mem - mem0)/Mb:.1f}')
    T = 20 # repeats
    with torch.cuda.device(dev):
        with torch.cuda.stream(s):  # warmup on a side stream, according to examples
            # check correctness
            # compute(net,x, **kwargs)
            # warmup iterations
            for i in range(4):
                if i==0:
                    print(f"_______________Compile/Warmup Iteration_________________________")
                    print(f'id(True)={id(True)}')
                    print(f'id(False)={id(False)}')
                    benchmark_FW(1, False)
                    remove_hooks(net)                    
                else:
                    print(f"_______________Benchmark Iteration {i} (not graphed)____________")
                    benchmark_FW(T, False)
                    benchmark_FW(T, True)
        torch.cuda.current_stream().wait_stream(s)
        opt.zero_grad()
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        mem0 = mem_usage()
        G = torch.cuda.CUDAGraph()
        with torch.cuda.graph(G): # graph forward-backward
            opt.zero_grad()
            compute(net,x, **kwargs)
        # benchmark graphed
        print(f"_______________Benchmark graphed__________________________")
        tt = 0
        for j in range(T):
            start_e.record()
            G.replay()
            end_e.record()
            torch.cuda.synchronize()
            t = start_e.elapsed_time(end_e)
            tt += t
        print(f'FW-BW time: {tt/T:.1f} ms')
        # print(x.grad.mean())
        mem = mem_usage()
        print(f'FW-BW mem: {(mem- mem0)/Mb:.1f}')
        print(f'Total mem: {(mem)/Mb:.1f} + moments in optimizer')


def profile_all():
    netiter = iter(prof_models())
    while True:
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        mem0 = mem_usage()
        try:
            net, nname = next(netiter)
            print(net)
        except StopIteration:
            break
        with Tee('prof_' + nname + '.txt'):
            print(net)
            total_params = sum(p.numel() for p in net.parameters())
            print(f'Parameters: {total_params/10**6:3.2f}M')
            mem = mem_usage()
            print(f'Model mem: {(mem-mem0)/Mb:.1f}')

            net.to(dev)
            batch_size = 128
            image_size = 224
            # batch_size = 2
            # image_size = 768
            
            add_hooks(net)
            
            kwargs = dict(method = net.o.m_train)
            x = torch.rand(batch_size, 3, image_size, image_size).to(dev)
            profile_net(net, x, **kwargs)
            
            # opt = torch.optim.Adam(net.parameters())

            # n = 0
            # tt = 0
            # for i in range(100):
            #     x = torch.rand(batch_size, 3, image_size, image_size).to('cuda')
            #     start_e = torch.cuda.Event(enable_timing=True)
            #     end_e = torch.cuda.Event(enable_timing=True)
            #     start_e.record()
            #     opt.zero_grad()
            #     if isinstance(net, ESequential):
            #         y = net.forward(x, method = net.o.m_train)
            #     else:
            #         y = net(x)
            #     a = torch.cuda.memory_allocated(0)
            #     r = torch.cuda.memory_reserved(0)    
            #     y.sum().backward()
            #     opt.step()
            #     end_e.record()
            #     torch.cuda.synchronize()
            #     t = start_e.elapsed_time(end_e)
            #     if i==0:
            #         remove_hooks(net)
            #     if i>5:
            #         tt += t
            #         n += 1

            # Mb = 1024**2
            # print(f'FW-BW time: {tt/n/1000:3.3f}s')
            # # print(f'FW-BW Memory: allocated {a/Mb:3.2f} reserved {r/Mb:3.2f}')
            # with torch.cuda.device('cuda'):
            #     (f, t) = torch.cuda.mem_get_info() # used as in nvidia-smi
            # print(f'mem_get_info used: {(t-f)/Mb} (matches nvidia-smi)')
            # # from pynvml import *
            # # nvmlInit()
            # # h = nvmlDeviceGetHandleByIndex(0)
            # # info = nvmlDeviceGetMemoryInfo(h)
            # # print(f'total    : {info.total/Mb}')
            # # print(f'free     : {info.free/Mb}')
            # # print(f'used     : {info.used/Mb}')
            # # print(f'used+free: {(info.used+info.free)/Mb}')


#def dotProdEnergy (n, b1, b2, gamma_0):
#    return b1 * n * (b2+2)


def k_function1(b2):
    if b2 == 1:
        return 4
    elif b2 == 2:
        return 3
    elif b2 == 3 or b2 == 4:
        return 2
    else:
        return 1
k_function = np.vectorize(k_function1)

def compute_dot_energy (b1, b2, n, E1, gamma_0):
    if (b1 > b2): b1, b2 = b2, b1
    K = math.ceil (math.log2 (n))
    k = k_function(b2)
    first_term = b1 * n * (b2 + 2) * E1
    second_term = gamma_0 * (2 ** (K - k)) * (b2 + 2 + k) * E1
    third_term = b1*(b2 + b1 + K)*E1 + b1*gamma_0*(b2 + b1 + K)*E1 # summation reduction in the end
    # third_term = b1*(b2 + b1 + K)*E1 + 2*gamma_0*(b2 + b1 + K)*E1 # (shift adder to run in 2 cycles, but no difference)
    return first_term + second_term + third_term

def compute_add_energy (b, n, E1):
    return n * b * E1;

def compute_shift_energy (b, n, E1):
    return n * b * E1;

def compute_mult_energy (b1, b2, n, E1):
    return n * b1 * b2 * E1;

@dataclass
class OpCounters:
    nAdds : int = 0;
    nMults : int = 0;
    nCmps : int = 0;
    nShifts : int = 0;
    
    nMemReads : int = 0;       # in bits
    nMemReadsA : int = 0;       # in bits
    nMemReadsW : int = 0;       # in bits
    nMemWrites : int = 0;      # in bits 
    
    computeEnergy : int = 0;

# cfg_string = "--net 'QResNet18(gate=Tower8s)' --method ST -A 2 -W 2"
# cfg_string = "--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"
# cfg_string = "--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 4 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"
# cfg_string = "--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 8 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"
# cfg_string = "--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 16 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"
# cfg_string = "--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s)' --method ST-det -n U --epochs 200 --lr 0.005 --compile -A 16 -W 16 --cudagraph --fp16 --distil -v 'T0.25'"
# # Dilation results
# cfg_string = "--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s,Dilation=True)' --method ST --epochs 200 --lr 0.0025 --compile -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"
# cfg_string = "--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s,Dilation=True)' --method ST --epochs 200 --lr 0.0025 --compile -A 4 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"
# cfg_string = "--batch_size 256 --data 'imagenet(res=512)' --MD --net 'QResNet18(gate=Tower8s,Dilation=True)' --method ST --epochs 200 --lr 0.0025 --compile -A 8 -W 2 --cudagraph --fp16 --distil -v 'T0.25'"


# cfg_string = "--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'BiNealNet()' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile"
# cfg_string = "--batch_size 256 --data 'imagenet-100(res=256)' --MD --net 'BiNealNet(m=1.5)' --method ST --epochs 200 --lr 0.005 -A 2 -W 2 --cudagraph --fp16 --distil -v 'T0.25' --compile"

def net_calc_energy_stats (cfg_string, rr: list):
    # [print(r) for r in rr]
    stats = defaultdict (OpCounters)
    K_after_conv = 256       # first quantization, after convolution, before pooling & affine
    K_affine_coeffs = 256    # resolution of affine coefficients

    considerSeparableconvolutionFused = True

    maxMemReadA : int = 0
    maxMemWrittenA : int = 0

    # E1 = 0.1/32.0            # cost of 1-bit adder, pJ, 45nm
    E1 = 0.03/32.0           # cost of 1-bit adder, pJ, 7nm
    #fJ = 0.001 # pJ
    #gamma_0 = 3.0*fJ / E1   # relative cost of register bits
    gamma_0 = 1

    # E_mem_per_bit = 1300 / 64 # pJ, DDR4
    E_mem_per_bit_on_chip = 150 #6         # pJ, L1 cache type
    E_mem_per_bit_ddr = 150           # pJ, main memory

    for r in rr:
        print (r);
        weights_size = r.conv.weight.shape if hasattr (r, "conv") else r.weights_size
        nWeights = weights_size [0] * weights_size [1] * weights_size [2] * weights_size [3]
        nAddBits = math.ceil (math.log2 (max (1, r.K_in * r.K_W * r.in_size [1] * weights_size [2] * weights_size [3])))
        nOutEl = r.conv_out_size [1] * r.conv_out_size [2] * r.conv_out_size [3];
        nOutChannels = r.conv_out_size [1];
        nDotProdEl = weights_size [2] * weights_size [3] * r.in_size [1];
        b_in = math.ceil (math.log2 (r.K_in))
        b_W = math.ceil (math.log2 (r.K_W))
        b_out = math.ceil (math.log2 (r.K_out)) if r.K_out > 0 else 0
        b_after_conv = math.ceil (math.log2 (K_after_conv))
        b_affine_coeffs = math.ceil (math.log2 (K_affine_coeffs))
    # conv2d
        stats [(r.K_in, r.K_W)].nMults += r.conv_out_size [2] * r.conv_out_size [3] * nWeights;
        stats [2**nAddBits].nAdds += r.conv_out_size [2] * r.conv_out_size [3] * nWeights;
        mra = b_in * r.in_size [1] * r.in_size [2] * r.in_size [3];
        stats [r.K_in].nMemReads += mra
        stats [r.K_in].nMemReadsA += mra
        if mra > maxMemReadA: maxMemReadA = mra
        mrw = b_W * nWeights;
        stats [r.K_W].nMemReads += mrw
        stats [r.K_W].nMemReadsW += mrw
        stats [(r.K_in, r.K_W)].computeEnergy += nOutEl * compute_dot_energy (b_in, b_W, nDotProdEl // (r.conv.groups if hasattr (r.conv, 'groups') else 1), E1, gamma_0)
    # pooling
        if (isinstance (r.pooling, nn.MaxPool2d)):
            stats [K_after_conv].nCmps += nOutEl * r.pooling.kernel_size * r.pooling.kernel_size
            stats [K_after_conv].computeEnergy += nOutEl * compute_add_energy (b_after_conv, r.pooling.kernel_size * r.pooling.kernel_size, E1)
            nOutEl = r.conv_out_size [1] * r.conv_out_size [2]/r.pooling.stride * r.conv_out_size [3]/r.pooling.stride;
        if (isinstance (r.pooling, nn.AvgPool2d)):
            stats [K_after_conv].nAdds += nOutEl * r.pooling.kernel_size * r.pooling.kernel_size
            stats [K_after_conv].nMults += nOutEl
            stats [K_after_conv].computeEnergy += nOutEl * compute_add_energy (b_after_conv, r.pooling.kernel_size * r.pooling.kernel_size, E1) + nOutEl * compute_mult_energy (b_after_conv, math.ceil (math.log2 (r.pooling.kernel_size * r.pooling.kernel_size)), 1, E1)
            nOutEl = r.conv_out_size [1] * r.conv_out_size [2]/r.pooling.stride * r.conv_out_size [3]/r.pooling.stride;
        if (isinstance (r.pooling, nn.AdaptiveAvgPool2d)):
            kw = (r.conv_out_size [2] // (r.pooling.output_size if isinstance (r.pooling.output_size, int) else r.pooling.output_size [0]))
            kh = (r.conv_out_size [3] // (r.pooling.output_size if isinstance (r.pooling.output_size, int) else r.pooling.output_size [1]))
            stats [K_after_conv].nAdds += nOutEl * kw * kh
            stats [K_after_conv].nMults += nOutEl
            stats [K_after_conv].computeEnergy += nOutEl * compute_add_energy (b_after_conv, kw * kh, E1) + nOutEl * compute_mult_energy (b_after_conv, math.ceil (math.log2 (kw * kh)), 1, E1)
            nOutEl = r.conv_out_size [1] * ((r.pooling.output_size + r.pooling.output_size) if isinstance (r.pooling.output_size, int) else (r.pooling.output_size [0] * r.pooling.output_size [1]));
    # affine
        stats [K_after_conv].nMults += nOutEl;
        stats [K_after_conv**2].nAdds += nOutEl;
        mrw = b_affine_coeffs * nOutChannels * 2;
        stats [K_affine_coeffs].nMemReads += mrw
        stats [K_affine_coeffs].nMemReadsW += mrw
        stats [K_after_conv].computeEnergy += nOutEl * compute_mult_energy (b_after_conv, b_affine_coeffs, 1, E1);
        stats [K_after_conv*K_affine_coeffs].computeEnergy += nOutEl * compute_add_energy (b_after_conv + b_affine_coeffs, 1, E1);
    # quantization
        stats [2**nAddBits].nShifts += nOutEl;
        stats [K_after_conv**2].nShifts += nOutEl;
        stats [2**nAddBits].computeEnergy += nOutEl * compute_shift_energy (nAddBits, 1, E1);
        stats [K_after_conv**2].computeEnergy += nOutEl * compute_shift_energy (b_after_conv * 2, 1, E1);
    # out
        if (r.K_out > 0):
            mwa = b_out * nOutEl;
            if considerSeparableconvolutionFused and hasattr (r.conv, 'groups') and r.conv.groups > 1:
                stats [r.K_out].nMemReadsA -= mwa   # HACK - remove now the activations read in the next pointwise convolution
                mwa = 0
            stats [r.K_out].nMemWrites += mwa
            if mwa > maxMemWrittenA: maxMemWrittenA = mwa
        else:
            mwa = 0
        print (f"  Activations in: {mra} b ({mra/8/1024/1024:.2f} MB), activations out: {mwa} b ({mwa/8/1024/1024:.2f} MB)")

    totalComputeEnergy : int = 0
    totalMemWritten : int = 0
    totalMemRead : int = 0
    totalMemReadA : int = 0
    totalMemReadW : int = 0

    for k, v in stats.items ():
        print (f"Quantization {k} | {[ math.log2 (i) for i in k ] if isinstance (k, tuple) else math.log2 (k)} bits:")
        if v.nMults > 0: print (f'    nMults: {v.nMults}, {v.nMults/1000/1000:.2f} M');
        if v.nAdds > 0: print (f'    nAdds: {v.nAdds}, {v.nAdds/1000/1000:.2f} M');
        if v.nCmps > 0: print (f'    nCmps: {v.nCmps}, {v.nCmps/1000/1000:.2f} M');
        if v.nShifts > 0: print (f'    nShifts: {v.nShifts}, {v.nShifts/1000/1000:.2f} M');
        if v.nMemWrites > 0: print (f'    nMemWrites: {v.nMemWrites} b, {v.nMemWrites/8/1024/1024:.2f} MB');
        if v.nMemReads > 0: print (f'    nMemReads: {v.nMemReads} b, {v.nMemReads/8/1024/1024:.2f} MB');
        if v.nMemReadsA > 0: print (f'    nMemReadsA: {v.nMemReadsA} b, {v.nMemReadsA/8/1024/1024:.2f} MB');
        if v.nMemReadsW > 0: print (f'    nMemReadsW: {v.nMemReadsW} b, {v.nMemReadsW/8/1024/1024:.2f} MB');
        print (f'    energy: {v.computeEnergy}, {v.computeEnergy/1000/1000:.2f}M');
        totalComputeEnergy += v.computeEnergy
        totalMemRead += v.nMemReads
        totalMemWritten += v.nMemWrites
        totalMemReadA += v.nMemReadsA
        totalMemReadW += v.nMemReadsW

    print (cfg_string)
    print (f"Total compute energy: {int (totalComputeEnergy)}, {totalComputeEnergy/1000/1000:.2f} uJ")
    print (f"Total mem written: {int (totalMemWritten)} b, {totalMemWritten/8/1024/1024:.2f} MB, {totalMemWritten*E_mem_per_bit_on_chip/1000/1000:.2f} uJ")
    print (f"Total mem read: {int (totalMemRead)} b, {totalMemRead/8/1024/1024:.2f} MB")
    print (f"  Total mem read weights: {int (totalMemReadW)} b, {totalMemReadW/8/1024/1024:.2f} MB, {totalMemReadW*E_mem_per_bit_ddr/1000/1000:.2f} uJ")
    print (f"  Total mem read activations: {int (totalMemReadA)} b, {totalMemReadA/8/1024/1024:.2f} MB, {totalMemReadA*E_mem_per_bit_on_chip/1000/1000:.2f} uJ")
    totalMemActivations = int (totalMemWritten + totalMemReadA);
    print (f"Total mem activations: {totalMemActivations} b, {totalMemActivations/8/1024/1024:.2f} MB, {totalMemActivations*E_mem_per_bit_on_chip/1000/1000:.2f} uJ")
    totalEnergy = (totalMemActivations*E_mem_per_bit_on_chip + totalMemReadW*E_mem_per_bit_ddr + totalComputeEnergy);
    print (f"Total energy: {totalEnergy/1000/1000:.2f} uJ")
    print (f"Max activations in: {int (maxMemReadA)} b, {maxMemReadA/8/1024/1024:.2f} MB, {maxMemReadA/8/1024:.2f} kB")
    print (f"Max activations out: {int (maxMemWrittenA)} b, {maxMemWrittenA/8/1024/1024:.2f} MB, {maxMemWrittenA/8/1024:.2f} kB")

    print (f" / {totalMemReadW/8/1024/1024:.2f} / {totalMemReadW*E_mem_per_bit_ddr/1000/1000:.0f} & {totalMemActivations/8/1024/1024:.2f} / {totalMemActivations*E_mem_per_bit_on_chip/1000/1000:.0f} & {totalComputeEnergy/1000/1000:.0f} & {totalEnergy/1000/1000:.0f} \\\\")

def net_energy(cfg_string:str, num_classes=1000, image_size=224):
    o = o_from_str(cfg_string)
    setup_o(o)
    o.num_classes = num_classes
    net = create_net(o)
    net.o = o
    net.to('cpu')
    net.eval()
    print(net)
    batch_size = 1
    image_size = image_size
    add_hooks(net)
    kwargs = dict(method = net.o.m_train)
    x = torch.rand(batch_size, 3, image_size, image_size)
    FW(net,x, **kwargs)
    rr = fuse_model(net)
    net_calc_energy_stats (cfg_string, rr)
    print (f"N of parameters: {sum (p.numel () for p in net.parameters ())}")
    print (f"N of trainable parameters: { sum (p.numel () for p in net.parameters () if p.requires_grad) }")



def profile_energies():
    # cfg_string = "--net 'QResNet18(gate=Tower8s)' --method ST -A 2 -W 2"
    # cfg_string = "--net 'QResNet18(gate=Tower8s,Dilation=True)' --method ST -A 4 -W 4"
    # cfg_string = "--net 'BiNealNet(m=1)' --method ST -A 2 -W 2"
    # cfg_string = "--net 'BiNealNet(m=1.5)' --method ST -A 2 -W 2"
    # cfg_string = "--net 'BOLDNet(m=1)' --method ST -A 2 -W 2"
    # cfg_string = "--net 'BOLDNet(m=3)' --method ST -A 2 -W 2"
    # cfg_string = "--net 'BOLDNet(m=4)' --method ST -A 2 -W 2"
    # cfg_string =  "--net 'QResNet18(gate=Tower8s)' --method ST-det -n U -A 4 -W 4 --fp16"
    # cfg_string =  "--net 'QResNet18(gate=Tower8s)' --method ST-det -n U -A 256 -W 256 --fp16"
    # cfg_string =  "--net 'BiNealNet(m=1)' --method ST-det -n U -A 2 -W 2"
    # cfg_string =  "--net 'MobileNetv1(m=2)' --method ST-det -n U -A 65536 -W 256"
    # cfg_string =  "--net 'QResNet18(gate=Tower8s,m=0.75)' --method ST-det -A 4 -W 2"
    # cfg_string =  "--net 'QResNet18(gate=Tower8s,m=0.75)' --method ST-det -A 4 -W 2"
    cfg_string = "--net 'QResNet18(gate=Tower8s,Dilation=True)' --method ST-det -A 4 -W 4"
    net_energy(cfg_string)

    # with open ("data/profile_reactnet.pickle", "rb") as handle:
    #     rr = pickle.load (handle)
    # net_calc_energy_stats ("ReactNet", rr)

    # with open ("data/profile_birealnet.pickle", "rb") as handle:
    #     rr = pickle.load (handle)
    # net_calc_energy_stats ("BiRealNet", rr)


def teest_net():
    # import device
    # global dev
    # select_device()
    # dev = device.dev
    cfg_string = "--net 'QResNet18(gate=GenResSE)' --method ST -A 2 -W 2 --fp16"
    net_energy(cfg_string)

# teest_net()

# profile_all()

if __run__:
    profile_energies()
