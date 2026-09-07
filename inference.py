# SPDX-License-Identifier: CC BY-NC-SA 4.0
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. Making code open source is essential for scientific
# transparency and reproducibility. Providing access to the code allows researchers to
# verify, replicate, and expand upon results, which supports scientific progress. Current
# top-tier conferences encourage opensource code for the purpose of reproducibility that
# is de-facto an established expectation of the research community. Opensource code as
# part of publication should not affect the performance of an invention both from a
# commercial and IP point of view.
# %% compact network representation for inference mode
import os, sys
import relimport

if __run__:
    os.chdir(relimport.proj_dir())
    
import torch
import torchvision.models as models

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from dataclasses import asdict, dataclass
from collections import defaultdict
import numpy as np

# %%

@dataclass
class ConvOptions:
    in_channels:int
    out_channels:int
    kernel_size:int
    stride:int=1 # default
    dilation:int=1 # default
    groups:int=1 # default
    padding:int | str = 'same' # default
    padding_mode:int | str = 'replicate', # default
    bias:bool = False
    # def __post_init__(self):
    #     if self.padding == 'same':
    #         self.padding = self.kernel_size // 2

@dataclass
class PoolOptions:
    operation: str | None = None # 'max' or 'avg' or None # None / max is currently used
    kernel_size:int = 2
    stride:int=1 # default
    dilation:int=1 # default
    padding:int = 0 # default

class ElemetaryBlock(nn.Module):
    def __init__(self, KA:int, KW:int, co: ConvOptions, po:PoolOptions | None = None):
        super().__init__()
        self.KA = KA
        self.KW = KW
        self.co = co
        if po is not None:
            pd = asdict(po)
            pd.pop('operation')
            if po.operation == 'max':
                self.pool = nn.MaxPool2d(**pd)
            elif po.operation == 'avg':
                pd.pop ('dilation')
                self.pool = nn.AvgPool2d(**pd)
            else:
                self.pool = None
        else:
            self.pool = None

    def forward(self, x:Tensor, w:Tensor, s1:Tensor, s2:Tensor, b: Tensor):
        """
        Equation (9) of the paper:
        y = Q(w@x * s1 + 1@ x * s2 + b)
        #
        x: input tensor, shape (B,C,H,W), integer
        w: conv kernel weights, shape (C_out, C_in / groups, K, K), integer
        s1: [C_out]
        s2: [C_out]
        b: [C_out]
        """

        try:
            # integer conv
            self.y = F.conv2d(x.to (torch.float32), w.to (torch.float32), bias=None, stride=self.co.stride, padding=self.co.padding, dilation=self.co.dilation, groups=self.co.groups) # padding_mode = self.co.padding_mode, 
            # sum over the same shape as the kernel -- separable
            self.x1 = x.sum(dim=1, keepdim = True)
            self.w1 = torch.ones(size=(1,1,self.co.kernel_size,self.co.kernel_size), device=x.device, dtype=self.x1.dtype)
            self.y1 = F.conv2d(self.x1, self.w1, bias=None, stride=self.co.stride, padding=self.co.padding, dilation=self.co.dilation) # padding_mode = self.co.padding_mode
            self.z1 = self.y * s1.view([1,-1,1,1]) + self.y1 * s2.view([1,-1,1,1]) + b.view([1,-1,1,1])
            if self.pool is not None:
                self.z2 = self.pool(self.z1)
            else:
                self.z2 = self.z1
            self.z = self.z2.clamp(0, self.KA-1).to(torch.uint8)
            return self.z
        except Exception as e:
            print ('Exception: ', e)
        return None

def test():
    torch.manual_seed(1)
    B,C,H,W = 1,3,32,32
    C_out = 6
    KA = 16
    KW = 4
    x = torch.randint(0,KA,(B,C,H,W), dtype=torch.uint8)
    w = torch.randint(0,KW,(C_out,C,3,3), dtype=torch.uint8)
    s1 = torch.rand(C_out)*0.08
    s2 = -torch.rand(C_out)*0.06
    b = -torch.rand(C_out)*4
    co = ConvOptions(in_channels=C, out_channels=C_out, kernel_size=3, stride=1)
    po = PoolOptions(operation='max', kernel_size=2, stride=2)
    block = ElemetaryBlock(KA=KA, KW=KW, co=co, po=po)
    z = block(x,w,s1,s2,b)
    print(z.shape, z.dtype, z.min(), z.max())
    print(z[0,1])


def tensorToFile (f, x):
    npx = x.numpy ()
    np.array (npx.dtype.kind).tofile (f)
    np.array (npx.dtype.itemsize).astype ('uint32').tofile (f)
    np.array (len (npx.shape)).astype ('uint32').tofile (f)
    np.array (npx.shape).astype ('uint32').tofile (f)
    npx.tofile (f)


def dumpData (p, x, w, s1, s2, b, z, block):
    fn = 'data/'
    for [k, v] in p.items ():
        fn = fn + str (v) + '_'
    fn = fn [:-1]

    cfgStr = ''
    for [k, v] in p.items (): 
        cfgStr = cfgStr + str (k) + '=' + ('"' if isinstance (v, str) else '') + str (v) + ('"' if isinstance (v, str) else '') + os.linesep
    cfgStr = cfgStr + 'dataPath="' + fn + '.dat"'
    f = open (fn + '.cfg', 'w+t')
    f.write (cfgStr)
    f.close ()
    
    f = open (fn + '.dat', 'w+b')
# inputs    
    tensorToFile (f, x)
    tensorToFile (f, w)
    tensorToFile (f, s1)
    tensorToFile (f, s2)
    tensorToFile (f, b)
#output
    tensorToFile (f, z)
#intermediates
    tensorToFile (f, block.y)
    tensorToFile (f, block.x1)
    tensorToFile (f, block.w1)
    tensorToFile (f, block.y1)
    tensorToFile (f, block.z1)
    tensorToFile (f, block.z2)

    f.close ()



def gen_test_data ():

    inputSizes = [ 32 ]                                     # [ 4, 16, 32, 37, 64, 240 ]
    inputChannels = [ 1, 3, 16, 256, 512 ]                     # [ 3, 4, 16, 256]
    outputChannels = [ 1, 16, 256, 512 ]                       # [ 4, 6, 8, 16, 256 ]
    KAs = [ 2, 4, 16, 256 ]                                 # [ 2, 3, 4, 8, 16, 256 ]
    KWs = [ 2, 4, 16, 256 ]                                 # [ 2, 3, 4, 8, 16, 256 ]
    convPaddingModes = [ 'zeros' ]                          # [ 'zeros', 'replicate' ]  # replicate is not supported by functional.conv2d
    convKernelSizes = [ 1, 3 ]                              # [ 1, 3, 4, 5, 7 ]
    convStrides = [ 1, 2 ]                                  # [ 1, 2, 3, 4 ]
    poolOperations = [ 'none', 'max', 'avg' ]                      # [ 'none', 'max', 'avg' ]
    poolKernelSizes = [ 2 ]                                 # [ 2, 3, 4 ]
    poolStrides = [ 1, 2 ]                                  # [ 1, 2, 3, 4 ]

    p = defaultdict ()
    p ['b'] = 1
    for p ['h'] in inputSizes:
        p ['w'] = p ['h']
        for p ['cIn'] in inputChannels:
            for p ['cOut'] in outputChannels:
                for p ['kA'] in KAs:
                    for p ['kW'] in KWs:
                        for p ['poolOperation'] in poolOperations:
                            for p ['poolKernelSize'] in poolKernelSizes:
                                for p ['poolStride'] in poolStrides:
                                    for p ['convKernelSize'] in convKernelSizes:
                                        for p ['convStride'] in convStrides:
                                            for p ['convPaddingMode'] in convPaddingModes:
                                                p ['convPadding'] = p ['convKernelSize'] // 2 #'same'
                                
                                                torch.manual_seed(1)
                                                x = torch.randint (0, p ['kA'], (p ['b'], p ['cIn'], p ['h'], p ['w']), dtype=torch.uint8)
                                                w = torch.randint (0, p ['kW'], (p ['cOut'], p ['cIn'], 3, 3), dtype=torch.uint8)
                                                s1 = torch.rand (p ['cOut'])*0.08
                                                s2 = -torch.rand (p ['cOut'])*0.06
                                                b = -torch.rand (p ['cOut'])*4
                                                co = ConvOptions (in_channels = p ['cIn'], out_channels = p ['cOut'], kernel_size = p ['convKernelSize'], stride = p ['convStride'], padding = p ['convPadding'], padding_mode = p ['convPaddingMode'])
                                                po = PoolOptions (operation = p ['poolOperation'], kernel_size = p ['poolKernelSize'], stride = p ['poolStride'])
                                                block = ElemetaryBlock(KA = p ['kA'], KW = p ['kW'], co=co, po=po)
                                                z = block (x, w, s1, s2, b)
                                                print ([ p ['w'], p ['h'], p ['cIn'], p ['cOut'], p ['kA'], p ['kW'], p ['convPaddingMode'], p ['convPadding'], p ['convKernelSize'], p ['convStride'], p ['poolOperation'], p ['poolKernelSize'], p ['poolStride'] ])

                                                if z is not None:
                                                    dumpData (p, x, w, s1, s2, b, z, block)


# %%
if __run__:
#    test()
    gen_test_data ()
# %%
