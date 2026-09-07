# SPDX-License-Identifier: CC BY-NC-SA 4.0
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. Making code open source is essential for scientific
# transparency and reproducibility. Providing access to the code allows researchers to
# verify, replicate, and expand upon results, which supports scientific progress. Current
# top-tier conferences encourage opensource code for the purpose of reproducibility that
# is de-facto an established expectation of the research community. Opensource code as
# part of publication should not affect the performance of an invention both from a
# commercial and IP point of view.
from __future__ import annotations

# Absolute imports
from typing import TYPE_CHECKING, OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
import torch.utils
import math
import abc
from torch.nn.modules.batchnorm import _BatchNorm
from collections.abc import Iterable
from types import SimpleNamespace
import itertools

import copy
from dataclasses import dataclass, InitVar, asdict
import dataclasses
from scipy.special import logit

# __________________Quant__________________________
# Relative (our) imports
from .tools import dotdict, soft_plus, soft_minus
from .utils import samples_to_batch, batch_to_samples
from .random_variable import *
from .functional import *
from .layer_base import *
from .categorical.cat_layers import Categorical
from .inductor import *

# _______________________distributions________________________________
class ShiftScaleDiscreteDistribution:
    def __init__(self, p, quant, scale=1.0, shift=0.0):
        self.p = p
        self.quant = quant
        self.scale = scale        
        self.shift = shift
    
    def __add__(self, x:Tensor):
        self.shift = self.shift + x
        return self

    def __mul__(self, s:Tensor):
        self.scale = self.scale*s
        self.shift = self.shift*s
        return self
    
    @staticmethod
    def cat(xx, dim=None):
        p = torch.cat([x.p for x in xx], dim= dim)
        if isinstance(xx[0].scale, Tensor):
            scale = torch.cat([x.scale for x in xx], dim= dim)
        else:
            scale = xx[0].scale
        if isinstance(xx[0].shift, Tensor):
            shift = torch.cat([x.shift for x in xx], dim= dim)
        else:
            shift = xx[0].shift
        r = xx[0].__class__(p, xx[0].quant, scale=scale, shift=shift)
        return r
    

class CatDistribution(ShiftScaleDiscreteDistribution):
    def as_RV(self):
        m = self.quant.expectation(self.p)
        v = (self.quant.second_moment(self.p) - m**2).clamp(min=1e-6)
        m = m * self.scale + self.shift
        v = v * self.scale**2
        return RandomVar(m,v)
    
    def sample(self):
        q = self.quant.sample(self.p)
        return q * self.scale + self.shift

class BernoulliDistribution(ShiftScaleDiscreteDistribution):
    def as_RV(self):
        m = self.p
        v = (m*(1-m)).clamp(min=1e-6)
        m = m * self.scale + self.shift
        v = v * self.scale**2
        return RandomVar(m,v)


class QuantProxy(KWLayer):
    def __init__(self, quant:Quant, eta:Tensor, scale=1.0, shift=0.0, sample=None, method=None):
        super().__init__()
        self.quant = quant
        self.eta = eta
        self.method = method
        # if sample is None:
            # self._sample = method.forward_Quant(quant, self.eta)
            # self._sample = self.quant.quantize(self.eta + self.quant.sample_noise(self.eta, 0))
        # else:
        self._sample = sample
        self.scale = scale
        self.shift = shift

    @staticmethod
    def cat(xx, dim=None):
        eta = torch.cat([x.eta for x in xx], dim= dim)
        sample = torch.cat([x._sample for x in xx], dim= dim)
        if isinstance(xx[0].scale, Tensor):
            scale = torch.cat([x.scale for x in xx], dim= dim)
        else:
            scale = xx[0].scale
        if isinstance(xx[0].shift, Tensor):
            shift = torch.cat([x.shift for x in xx], dim= dim)
        else:
            shift = xx[0].shift
        r = xx[0].__class__(xx[0].quant, eta, scale=scale, shift=shift, sample= sample)
        return r
    
    def mean_embedding(self) -> Tensor:
        return self.quant.mean_embedding(self.eta)

    def moments(self):
        p = self.quant.cat_distribution(self.eta)
        m1 = self.quant.expectation(p)
        m2 = self.quant.second_moment(p)
        return (m1,m2)
    
    def sample(self):
        return self._sample

class Distribution2ST(KWLayer):
    def forward(self, xx, **kwargs):
        if isinstance(xx,CatDistribution):
            r = xx.as_RV()
            q = xx.sample()
            m = r.mean
            return q + (m - m.detach())
        elif isinstance(xx,QuantProxy):
            q = xx.sample()
            m = xx.mean_embedding()
            return q + (m - m.detach())
        else:
            return xx


# _______________________help functions_______________________________

class EWA_BN(torch.nn.BatchNorm2d):
    def __init__(self, num_features, momentum = 0.1, **kwargs):
        super().__init__(num_features=num_features, momentum=momentum, **kwargs)
        self.register_buffer('EWA_m', torch.zeros(num_features))
        self.register_buffer('EWA_v', torch.zeros(num_features))
        self.EWA_q = 0.5
        self.EWA_t = 0
    
    def forward(self, x:Tensor):

        # v, m = torch.var_mean(x, dim=(0,2,3), unbiased=False, keepdim=False)
        # self.EWA_t += 1
        # q = self.EWA_q / (1 -(1-self.EWA_q)**float(self.EWA_t))
        # em = self.EWA_m* (1-q) + m*q
        # ev = self.EWA_v* (1-q) + v*q

        # self.EWA_m.data = em.detach()
        # self.EWA_v.data = ev.detach()
        s = [1,-1,1,1]

        if self.training:
            # compute m, v statistics
            # v, m = torch.var_mean(x, dim=(0,2,3), unbiased=False, keepdim=False)
            # # !!!! v can turn out to be negative
            # # v = v.clip(min=1e-5)
            # vm = v.mean()
            # v = (v*0.9 + vm*0.1)
            self.EWA_t += 1
            # q = self.EWA_q / (1 -(1-self.EWA_q)**float(self.EWA_t))
            # # em = self.EWA_m* (1-q) + m*q
            # # ev = self.EWA_v* (1-q) + v*q
            # # self.EWA_m.data = em.detach()
            # # self.EWA_v.data = ev.detach()

            # em = m
            # ev = v
            n = x.shape[0]
            C = x.shape[1]
            M = torch.mean(x, dim=(2,3), keepdim=False)
            M2 = torch.mean(x**2, dim=(2,3), keepdim=False)
            Mf = (M.sum(dim=0,keepdim=True) - M)/(n-1)
            # Mf = ((M.sum(dim=0,keepdim=True) - M).detach() + M)/n
            M2f = (M2.sum(dim=0,keepdim=True) - M2)/(n-1)
            # M2f = ((M2.sum(dim=0,keepdim=True) - M2).detach() + M2)/n
            Vf = M2f - Mf**2
            Vf = Vf.clamp(min = 1e-5)

            # V, M = torch.var_mean(x, dim=(2,3), keepdim=False)
            # em = M.mean(dim=0, keepdim=False) # for running stat
            # M = M.mean(dim=0,keepdim=True) - M/x.shape[0] # use all other samples but the current one
            # ev = V.mean(dim=0, keepdim=False) + M.var(dim=0, keepdim=False)# average var, for running stat
            # ev = (ev*0.9 + ev.mean()*0.1)
            # V = V.mean(dim=0,keepdim=True) - V/x.shape[0] # use all other samples but the current one

            em = Mf.view([n, C, 1,1])
            ev = Vf.view([n, C, 1,1])

            # m = M.mean(dim=0)
            # v = M2.mean(dim=0) - m**2
            m = em.mean(dim=0).flatten()
            v = ev.mean(dim=0).flatten()

            if self.track_running_stats:
                if self.EWA_t ==1:
                    self.running_mean.fill_(0)
                    self.running_var.fill_(1.0)        
                if self.momentum is None:
                    q = 1.0/float(self.EWA_t)
                else:
                    raise RuntimeError("not supposed use case")
                    q = self.EWA_q / (1 -(1-self.EWA_q)**float(self.EWA_t))
                self.running_mean = self.running_mean * (1-q) + m.detach()*q
                self.running_var = self.running_var * (1-q) + v.detach()*q
        else:
            em = self.running_mean.detach().view(s)
            ev = self.running_var.detach().view(s)
        
        y = (x - em)/((ev + self.eps)**0.5)
        if self.affine:
            y = y * self.weight.view(s) + self.bias.view(s)

        return y
        
    def reset_running_stats(self):
        super().reset_running_stats()
        self.EWA_t = 0
        if hasattr(self, "EWA_m"):
            self.EWA_m.data.fill_(0)
            self.EWA_v.data.fill_(1e-5)

# Norm = EWA_BN
Norm = nn.BatchNorm2d

def make_norm(channels, learnable=True, weight = 1.0, bias = 0.0, init_only = False, eps=0.0, Norm=Norm): # DEBUG (eps=0 was the default, if it fails hard we need a better fix)
    if True: # BN
        # Norm = nn.BatchNorm2d
        if learnable:
            # can split off learnable ScaleBias as a separate Layer
            norm = Norm(channels, affine=True, eps=eps)
            norm.weight.data.fill_(weight)
            norm.bias.data.fill_(bias)
            r = norm
        else: # not learnable
            if weight ==1.0 and bias == 0.0: # no ScaleBias needed
                norm = Norm(channels, affine=False, eps = eps)
                r = norm
            else: # with ScaleBias
                norm = Norm(channels, affine=False, eps = eps)
                r = ESequential([norm, ScaleBias(weight=weight, bias=bias, learnable=False)])
        norm.init_only = init_only
        return r
    # return Identity()
    # return LayerNorm(channels, eps=1e-6, data_format="channels_first") # slower training, costly at inference
    # return ESequential([nn.BatchNorm2d(channels, affine=False), ConstantScaleBias(0.25,0.5)])


""" A tool to save the computation in the expression y - y.detach() used for substituting derivative """
class GradOnly(torch.autograd.Function):
    @staticmethod
    # so this propagates forward only zeros
    def forward(ctx, x):
        t = torch.zeros_like(x, requires_grad=True)
        return t

    @staticmethod
    # for back prop it lets the ongoing grad flow just pass without any change...
    def backward(ctx, grad_output):
        return grad_output


# get some tensor and apply the selected changes -> zero it out for forward prop, pass grad for back prop without change
def grad_only(x):
    return GradOnly.apply(x)


class SubstGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        return x  # y is not used on the forward

    @staticmethod
    def backward(ctx, grad_output):
        # gradient goes to y -> to the input which was ignored for forward prop
        return None, grad_output


def subst_grad(x, y):
    # return x.detach() + (y - y.detach())
    return SubstGrad.apply(x, y)

def sigmoid_MD(x):
    return subst_grad(x.sigmoid(), x)

def Phi_approx_MD(x):
    return subst_grad(Phi_approx(x), x)

def tanh_MD(x):
    return subst_grad(torch.tanh(x), x)

class GClip(torch.autograd.Function): # TODO: revise, what is it for?
    @staticmethod
    def forward(ctx, x, min, max):
        inside = torch.logical_and(x >= min, x < max).to(x)
        ctx.save_for_backward(inside)
        return torch.clip(x, min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        inside, = ctx.saved_tensors
        # print(grad_output.shape)
        # print(inside.shape)
        r = grad_output * inside
        return r, None, None  # gradient goes to y


def gclip(x:Tensor, min, max) ->Tensor:
    # return x.detach() + (y - y.detach())
    return GClip.apply(x, min, max)
# ____ end clipped STE trick


""" 
Now we define basic Quantization for activations and two methods: Default and Relaxed Quantization
Intended architecture of a network is to use blocks: (QLinear -> BN -> QReLU). 
ReLU is not needed ever. BN can be replaced with a ScaleBias.
"""

# _________________Quant___________________

# class QO(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.q_noise_type='uniform'
#         self.q_noise_sigma=None
#         self.q_noise_sigma_learnable=False
#         self.NS=1.0

# class CompilableModule:
#     def register_buffer(self, name, val):
#         self.__dict__['name'] = val
#         pass


class Quant(KWLayer):
    @dataclass(frozen=True)
    class Options():
        q_noise_type:str = 'uniform'
        q_noise_sigma:float | None = None
        q_noise_sigma_learnable:bool = False
        NS:float = 1.0
        is_activation:bool = True
        K:int = 2
        from_o:InitVar[dict|dotdict|None] = None
        #
        def __post_init__(self, from_o):
            if from_o is not None:
                #look up for self fields in kwargs
                self_d = dataclasses.asdict(self)
                for k in self_d.keys():
                    if k in from_o.keys():
                        object.__setattr__(self, k, from_o[k])
                return self
        #
        def replace(self, **kwargs):
            """
            assumes only valid kwargs, raise if there is a name missmatch
            """
            return dataclasses.replace(self, **kwargs) # creates a new object rather than mutates
        
    
    def __init__(self, o:Quant.Options):
        """
        Quantize the input to nearest integer and clamp to the range [0, K-1]
        """
        super().__init__()
        self.o = o
        # self.o = SimpleNamespace(q_noise_type=o.q_noise_type, q_noise_sigma=o.q_noise_sigma, q_noise_sigma_learnable=o.q_noise_sigma_learnable, NS=o.NS)
        # o = self.o
        
        # self.o = dotdict(q_noise_type=None, q_noise_sigma = None, q_noise_sigma_learnable = None, CN = 0, NS = 1.0)
        # self.o.set(o)
        
        # self.o = SimpleNamespace(q_noise_type='uniform', q_noise_sigma=None, q_noise_sigma_learnable=False, NS=1.0) # DBUG TODO: use correct options
        # o = self.o
        
        # self.o = QO()
        # create the methods work area for plausible values
        K = o.K
        is_activation = o.is_activation
        #
        self.range = (0, K - 1)
        self.K = K
        self.is_activation = is_activation
        self.cat = Categorical(C=K, embedding='integer')
        if o.q_noise_sigma_learnable:
            # trainable noise standard deviation, with respect to integer grid size 1, initialized to 1/3
            self.sigma_ = Parameter(soft_minus(torch.tensor([o.q_noise_sigma])))
        elif o.q_noise_sigma is not None:
            self.register_buffer('sigma_', soft_minus(torch.tensor([o.q_noise_sigma])))

        left = torch.tensor(np.arange(K)) - 0.5
        left[0] = -10000
        right = torch.tensor(np.arange(K)) + 0.5
        right[K - 1] = 10000
        # threshold points between bins
        thresholds = torch.tensor([-100000, *range(self.K - 1), 100000]) + 0.5
        grid = torch.tensor(np.arange(K))
        # .register_buffer(name, tensor) -> it is not considered a parameter for which grad is computed
        self.register_buffer('left', left)
        self.register_buffer('right', right)
        self.register_buffer('thresholds', thresholds)
        self.register_buffer('grid', grid)

    # def __eq__(self, other:Quant):
    #     return hash(self) == hash(other)
    
    # def __hash__(self):
    #     return hash((self.K,self.is_activation))

    def get_sigma(self):
        return soft_plus(self.sigma_) + 1e-10  # smooth approx to ReLu

    def logistic_scale(self):
        # standard logistic distribution has variance pi^2/3
        # the scale factor s such that sigmoid(x/s) is the cdf of logistic distribution with the desired std sigma
        return self.get_sigma() * math.sqrt(3) / math.pi

    def noise_cdf(self, x:Tensor, temp=None) -> Tensor:  # zero mean noise is assumed
        eps = 1e-10
        if temp is None:
            temp = 1
        if self.o.q_noise_type == 'logistic':
            return torch.sigmoid(x / (self.logistic_scale() * temp))
        if self.o.q_noise_type == 'normal':
            return gauss_cdf(x / (self.get_sigma() * temp))
        elif self.o.q_noise_type == 'uniform':
            # return gclip(x, min=-0.5 - eps, max=0.5) + 0.5
            return gclip(x / temp, min=-0.5, max=0.5) + 0.5
        elif self.o.q_noise_type == 'triangular':
            x = x / temp
            y = (x + x ** 2 / 2 + 1 / 2) * torch.logical_and(x >= -1, x < 0).to(x)
            y += (x - x ** 2 / 2 + 1 / 2) * torch.logical_and(x >= 0, x <= 1).to(x)
            y += (x > 1).to(x)
            return y

    def noise_pdf(self, x, temp=None) ->Tensor:
        if temp is None:
            temp = 1
        if self.o.q_noise_type == 'logistic':
            scale = self.logistic_scale() * temp
            P = torch.sigmoid(x / scale)
            p = (P) * (1 - P) / scale
            return p
        if self.o.q_noise_type == 'normal':
            scale = self.get_sigma() * temp
            return gauss_pdf(x/scale) / scale
        elif self.o.q_noise_type == 'uniform':
            p = torch.logical_and(x / temp >= -0.5, x / temp < 0.5).to(x) / temp
            return p
        elif self.o.q_noise_type == 'triangular':
            p = torch.clamp(1 - torch.abs(x/temp), min=0.0) / temp
            return p

    # @torch.compiler.disable()
    def sample_noise(self, x, CN):
        """
        CN -- correlated noise: 0 -- uncorrelated, 1 -- correlated over the batch for activations
        """
        eps = 1e-10
        shape = list(x.shape)
        if self.is_activation and CN in [1, 2]:
            shape[0] = 1  # same noise for all samples in the batch
        if self.o.q_noise_type == 'logistic':
            # method: invert logistic cdf
            n = torch.empty(shape, dtype=x.dtype, device=x.device)
            # fills it with random values from [eps ; 1-eps ]
            n.uniform_(eps, 1.0 - eps)
            # not sure, how does sampling log. noise work, but I guess this is it...
            n = self.logistic_scale() * torch.logit(n)
        elif self.o.q_noise_type == 'normal':
            n = torch.empty(shape, dtype=x.dtype, device=x.device)
            n.normal_()
            n = self.get_sigma() * n
        elif self.o.q_noise_type == 'uniform':
            n = torch.empty(shape, dtype=x.dtype, device=x.device)
            n.uniform_(-0.5 + eps, 0.5 - eps)
        elif self.o.q_noise_type == 'triangular':
            # method: triangular noise density is a convolution of two uniform noise densities
            n1 = torch.empty(shape, dtype=x.dtype, device=x.device)
            n2 = torch.empty(shape, dtype=x.dtype, device=x.device)
            n1.uniform_(-0.5 + eps, 0.5 - eps)
            n2.uniform_(-0.5 + eps, 0.5 - eps)
            n = n1 + n2
            # if n.dim() == 4 and n.shape[3] > 5:  # debug
            # print(n.shape)
            # print("noise=",n[0,0,:,:])
        else:
            raise AttributeError(f" Unknown noise type '{self.o.q_noise_type}'")

        if self.o.NS != 1.0: 
            n = n * self.o.NS
        return n

    def quantize(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            return torch.clamp(torch.round(x), min=self.range[0], max=self.range[1])

    def cat_probability(self, x, qx, temp=None):
        """ probability of obtaining the given quantized result with the input x """
        qx = qx.long()              # torch to torch.int64
        F1 = self.noise_cdf(self.left[qx] - x, temp)
        F2 = self.noise_cdf(self.right[qx] - x, temp)
        p = F2 - F1
        # make sure non-negative # TODO: need a stable implementation of log_p
        p = torch.clamp(p, min=1e-20)
        return p

    # cdf evaluation for each bin in regard to the input x
    def cat_distribution(self, x, temp=None):
        """
        :param x: [*] any shape
        :return: p [*, K] -- categorical distribution for each x
        """
        dims = x.dim()
        v = self.thresholds.view((1,) * dims + (-1,)) - x.view(x.shape + (1,))  # [*, K+1]
        # check_real(v)
        P = self.noise_cdf(v, temp)  # cdf values for all bins
        # check_real(P)
        p = torch.diff(P, dim=-1)  # [*, K] probabilities of all bins
        p = torch.clamp(p, min=1e-20)  # make sure non-negative
        return p

    def d_distribution(self, x):
        """
        :param x: [*] any shape
        :return: p [*, K] -- categorical distribution for each x
        derivative of the distribution?
        """
        dims = x.dim()
        v = self.thresholds.view((1,) * dims + (-1,)) - \
            x.view(x.shape + (1,))  # [*, K+1]
        P = -self.noise_pdf(v)  # derivative
        p = torch.diff(P, dim=-1)  # [*, K] probability derivatives of all bins
        return p

    # compute the output expected value over the whole grid
    def expectation(self, p):
        dims = p.dim() - 1
        z = (p * self.grid.view((1,) * dims + (-1,))).sum(dim=-1, keepdim=False)
        return z

    def sample(self, p):
        idx = torch.distributions.Categorical(probs=p,  validate_args = False).sample()
        y = idx.squeeze(-1)
        return y

    def mean_embedding(self, eta: Tensor, temp=None) -> Tensor:
        if self.K==2:
            return self.noise_cdf(eta-0.5, temp) # verified
            # p = self.cat_distribution(eta, temp)
            # return p[...,1]
        
        p = self.cat_distribution(eta, temp)
        return self.expectation(p)
        # TODO: shortcut for triangular and uniform noises
    
    def second_moment(self, p:Tensor):
        dims = p.dim() - 1
        z = (p * (self.grid**2).view((1,) * dims + (-1,))).sum(dim=-1, keepdim=False)
        return z

    # some noises have predicatable cdf over the majority of the grid -> evaluate faster
    def d_mean_embedding(self, eta: Tensor, temp = None):
        if temp is not None and temp != 1:
          raise NotImplementedError('Requires verification')
        with torch.no_grad():
            if self.o.q_noise_type == 'uniform':
                return torch.logical_and(eta >= self.range[0], eta <= self.range[1]).to(eta)
            if self.o.q_noise_type == 'triangular':
                y = torch.clamp(self.K / 2 - torch.abs(eta - (self.K - 1) / 2), min=0, max=1)
                return y
            else:  # logistic, normal, etc where have no simplifying expressin
                y = 0
                for k in range(1, self.K): # TODO: vectorize or use autodiff of mean_embedding
                    y += self.noise_pdf(eta - (k - 0.5))
                return y

# _________________ConstantScale___________________

# class ConstantScaleBias(Mod):
#     """
#     Constant Scale ans Bias, same for all channels.
#     Implementation nevertheless stores values in a tensor, which ensures compiling one code for different values.
#     """
#     def __init__(self, scale = 1.0, bias = 0.0):
#         super().__init__()
#         if bias is None:
#             bias = 0.0
#         wb = torch.tensor((scale, bias), dtype = torch.float32)
#         self.register_buffer('wb', wb)
        
#     def set_to_map(self, x: tuple, y: tuple):
#         """x*s + b = y"""
#         self.wb.data[0] = (y[1] - y[0])/(x[1] - x[0])
#         self.wb.data[1] = y[0] - x[0]*self.wb.data[0]
#         return self
        

#     @property
#     def weight(self):
#         return self.wb[0]
    
#     @property
#     def bias(self):
#         return self.wb[1]
        
#     def forward(self, x, **kwargs):
#         y = x*self.wb[0] + self.wb[1]
#         return y


# _________________ScaleBias___________________

class ScaleBias(Mod):
    """
    applies per-channel scale and bias y(n,c,w,h) = x(n,c,w,h) * w(c) + b(c)
    """

    def __init__(self, channels=1, weight = 1.0, bias:float|None = 0.0, learnable = True, is_activation=False):
        super().__init__()
        if bias is not None:
            self.bias = Parameter(Tensor(channels))
            # self.local_lr = 5
        else:
            self.bias = None
        self.weight = Parameter(Tensor(channels))
        # self.local_lr = 5
        self.reset_parameters(weight = weight, bias=bias)
        if not learnable:
            w = self.weight.data
            del self.weight
            self.register_buffer('weight', w)
            if self.bias is not None:
                b = self.bias.data
                del self.bias
                self.register_buffer('bias', b)
        self.channels = channels
        self.is_activation = is_activation
        #TODO: network init pass to get the shape for scaling

    @property
    def learnable(self):
        return self.weight.requires_grad
    
    @learnable.setter
    def learnable(self, learnable:bool):
        if learnable and not self.learnable:
            self.weight = Parameter(self.weight.data)
            if self.bias is not None:
                self.bias = Parameter(self.bias.data)
        elif not learnable and self.learnable:
            w = self.weight.data
            del self.weight
            self.register_buffer('weight', w) # deletes / overwrites self.weight, but registerbuffer will complai
            if self.bias is not None:
                b = self.bias.data
                del self.bias
                self.register_buffer('bias', b)  # deletes / overwrites self.weight
        return self
    def reset_parameters(self, weight, bias):
        # this is ment to be for renormalization, not random
        if self.bias is not None:
            self.bias.data.fill_(bias)
        self.weight.data.fill_(weight)

    def get_affine(self):
        """ get the affine transform in the form f(x) = x*weight + bias"""
        return self.weight, self.bias

    def set_affine(self, weight, bias):
        """ set to affine f(x) = x*weight + bias"""
        if isinstance(weight, Tensor) and weight.numel()>1:
            self.weight.data = weight
            self.bias.data = bias
        else:
            self.weight.data.fill_(weight)
            self.bias.data.fill_(bias)
        return self

    def align_shape(self, sh):
        # align dimensions to input for broadcasting
        if len(sh) == 4:
            shape = [1, -1, 1, 1]
        elif len(sh) == 2:
            shape = [1, -1]
        elif len(sh) == 3:
            shape = [1, 1, -1]
        else:
            raise ValueError("don\'t know how to treat such input dimension")
        w = self.weight.view(shape)
        b = self.bias.view(shape) if self.bias is not None else None
        return w, b

    def forward(self, x):
        assert isinstance(x, Tensor)
        # ss = np.prod(np.array(x.shape[2:]))
        w, b = self.align_shape(x.shape)
        if b is not None:
            return x * w + b
        else:
            return x * w
        
    # def set_to_map(self, x: tuple, y: tuple):
    #     """x*s + b = y"""
    #     s = (y[1] - y[0])/(x[1] - x[0])
    #     b = y[0] - x[0]*s
    #     if isinstance(s, Tensor):
    #         self.weight.data = s
    #         self.bias.data = b
    #     else:
    #         self.weight.data.fill_(s)
    #         self.bias.data.fill_(b)
    #     return self
    
    def set_to_map(self, x: tuple, y: tuple):
        """x*s + b = y"""
        s = (y[1] - y[0])/(x[1] - x[0])
        b = y[0] - x[0]*s
        self.set_affine(s,b)
        return self


    def __repr__(self):
        if self.channels>1:
            bm = self.bias.min().item() if self.bias is not None else 0
            bM = self.bias.max().item() if self.bias is not None else 0
            tmpstr = 'ScaleBias (bias: {:.2g}-{:.2g} scale: {:.2g}-{:.2g})'.format(bm, bM, self.weight.min().item(), self.weight.max().item())
        else:
            tmpstr = f'ScaleBias (bias: {self.bias.item() if self.bias is not None else 0:.2g} scale: {self.weight.item():.2g} learnable:{self.weight.requires_grad})'
        return tmpstr

# ____________________________________________

class BoundedScaleBias(ScaleBias):
    """ This class is designated for adjusting the distribution statistics after BN,
    It implements a scale and bias but restricted so that the transformed distribution has some prescribed minimal mass in the region of interest, i.e., the distribution is not allowed to be shifted too far. This actually doe not occur so much in practice anyway -- it would result in dead units (having constant response). But the hope is that by using BoundedScaleBias we could more safely use larger learning rates, and keep the state "trainable".
    """
    def __init__(self, channels=1, weight = 1.0, bias = 0.0, learnable = True, is_activation=False, range1 = 1):
        Mod.__init__(self)
        if not learnable:
            raise AttributeError("Please use common ScaleBias for non-learnable")
        if bias is not None:
            self.latent_bias = Parameter(Tensor(channels))
        else:
            self.latent_bias = None
        self.latent_weight = Parameter(Tensor(channels))
        self.delta = 0.1
        self.l_delta = logit(1 - self.delta) # minimal mass of distribution in the quantizatino range
        self.range1 = range1
        self.s_max = range1 * 2
        # super().__init__(channels=channels, weight = weight, bias = bias, learnable = learnable, is_activation=is_activation)
        self.reset_parameters(weight = weight, bias=bias)
        self.channels = channels
        self.is_activation = is_activation
    
    def mu_min(self):
        return -self.weight/(V_S**0.5) * self.l_delta

    def mu_max(self):
        return self.range1 + self.weight/(V_S**0.5) * self.l_delta

    @property
    def bias(self):
        if self.latent_bias is not None:
            eta = sigmoid_MD(self.latent_bias)
            mu = (1-eta) * self.mu_min() + eta * self.mu_max() 
            return mu
        else:
            return None
    
    @bias.setter
    def bias(self, val):
        eta = (val - self.mu_min()) / (self.mu_max() - self.mu_min())
        self.latent_bias.data = torch.logit(eta)

    @property
    def weight(self):
        eta = sigmoid_MD(self.latent_weight)
        return eta * self.s_max # maximum scale factor

    @weight.setter
    def weight(self, val):
        # bias is dependent on the weight, so we first get the current bias
        b = self.bias
        self.latent_weight.data = torch.logit(val/self.s_max)
        # and set the desired bias back
        self.bias = b


    def reset_parameters(self, weight, bias):
        # this is ment to be for renormalization, not random
        if self.latent_bias is not None:
            w = self.weight
            w.data.fill_(weight)
            self.weight = w
            #
            b = self.bias
            b.data.fill_(bias)
            self.bias = b

    def set_affine(self, weight, bias):
        """ set to affine f(x) = x*weight + bias"""
        if isinstance(weight, Tensor) and weight.numel()>1:
            self.weight = weight.data
            self.bias = bias.data
        else:
            self.reset_parameters(self, weight, bias)
        
        check_real(self.weight)
        check_real(self.bias)
        return self



# _________________RandMask___________________

def ashape(shape):
    # align dimensions to input for broadcasting
    if len(shape) == 4:
        ashape = [1, -1, 1, 1]
    elif len(shape) == 2:
        ashape = [1, -1]
    else:
        raise ValueError("don\'t know how to trat such input dimension")
    return ashape


class QMask(Mod):
    """
    """
    def __init__(self, in_channels, o:Quant.Options, eta = 0.0):
        super().__init__()
        self.eta = Parameter(Tensor(in_channels))
        # self.eta = Parameter(Tensor(1))
        # self.register_buffer('eta',Tensor(in_channels))
        # self.eta.data.fill_(eta)
        S = 1
        self.eta.data.uniform_(-S, 0)
        self.sb = ScaleBias(learnable=False).set_to_map((-S,S),(0,1))
        # torch._dynamo.mark_dynamic(self.eta, 0)
        # mark_unbacked(self.eta, 0)
        # if eta>0.5:
        #     self.eta.data.uniform_(1-eta, eta)
        # else:
        #     self.eta.data.fill_(eta)
        self.quant = Quant(o)

    def forward(self, shape, **kwargs):
        if isinstance(shape, Tensor):
            shape = shape.shape
        shape = list(shape)
        eta = self.eta.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        eta = kwargs['method'].dispatch(self.sb, eta, **kwargs)
        p = self.quant.cat_distribution(eta)[...,1] # probability of 1
        # shape[1] = self.eta.shape[0]
        # x = self.eta.view(ashape(shape)).expand(shape)
        if False:
            qx = self.quant.forward(eta,**kwargs)
            qx = qx.expand(shape)
        else:
            eta = eta.expand(shape)
            qx = self.quant.forward(eta,**kwargs)
        return qx, p

# _________________QReLU___________________


class QReLU(ESequential):
    """
        QReLU: input scaling + quantization
    """
    Options = Quant.Options
    def __init__(self, o: QReLU.Options, channels = None):
        quant = Quant(o)
        min, max = quant.range
        super().__init__()
        self.o = o
        self.is_activation = o.is_activation
        # by default, for activations upscale the input first, so assume it was in the range [0,1], which is not fully accurate input to activaitons
        # in_sb = ConstantScaleBias().set_to_map((0, 1), quant.range)
        if o.is_activation and channels is not None and channels >1: # generic activations: add learnable per-channel randomized scale-bias
            # in_sb = ScaleBias(1, learnable=True).set_affine(weight=in_sb.weight, bias=in_sb.bias)
            # in_range = (0, 1)
            # randomize in_range
            if False:
                # left = torch.empty(channels).uniform_()*0.3 # uniform in [0, 0.3]
                # right = 1 + torch.empty(channels).uniform_() # uniform in [1,2]
                left = torch.empty(channels).uniform_()-0.5
                right = torch.ones(channels)*2
                mask = (torch.ones(channels)*0.5).bernoulli() > 0.5
                lm = left[mask].clone()
                rm = right[mask].clone()
                left[mask] = -rm
                right[mask] = -lm
                in_sb = ScaleBias(channels=channels, learnable=True, is_activation = o.is_activation).set_to_map((left, right), quant.range)
            else:
                in_range = (0.2, 2)
                # in_range = (-1,2)
                inv_range = (-in_range[1], -in_range[0]) # actually this is not negating the output, why
                # inv_range = (-in_range[0], -in_range[1])
                in_sb = ScaleBias(channels=channels, learnable=True, is_activation = o.is_activation).set_to_map(in_range, quant.range) # Learnable Scale-Bias per channel
                if True: 
                    # set half the channels to opposit assymetric pref
                    mask = (torch.ones(channels)*0.5).bernoulli() > 0.5
                    neg_sb = ScaleBias(channels=1, learnable=False).set_to_map(inv_range, quant.range) # this is to create data for half of activaitons, disposed
                    in_sb.weight.data[mask] = neg_sb.weight
                    in_sb.bias.data[mask] = neg_sb.bias
                    # in_sb.weight.local_lr = 2
                    # in_sb.bias.local_lr = 2
                if False:
                    # Bounded ScaleBias
                    in_sb0 = copy.deepcopy(in_sb)
                    in_sb = BoundedScaleBias(channels=channels, learnable=True, is_activation = o.is_activation, range1 = quant.range[1]).set_affine(in_sb0.weight, in_sb0.bias)
            if False: # add a global learnable param (can improve training dynamics becase its gradient is less stochastic)
                sb = ScaleBias(learnable=True, weight = 1.0, bias=None, is_activation = o.is_activation)
                ll = [sb, in_sb, quant]
            else:
                # in_sb.weight.local_lr = 2
                ll = [in_sb, quant]
        elif o.is_activation: # channels = None or channels = 1 -- global (same for all channles) scale-bias
            in_sb = ScaleBias(learnable=True, is_activation = o.is_activation).set_to_map((0, 1), quant.range)
            ll = [in_sb, quant]
        else: # Not an activation, we initialize, currently is unused (deleted in QAnyLinear)
            in_sb = ScaleBias(learnable=False, is_activation = o.is_activation).set_to_map((0, 1), quant.range)
            ll = [in_sb, quant]
        self.extend(ll)
        self._in_sb = [in_sb]
        self._quant = [quant]
    
    @property
    def in_sb(self) -> ScaleBias:
        return self._in_sb[0]
        
    @property
    def quant(self) -> Quant:
        return self._quant[0]
    
    
    def __repr__(self):
        return f"QReLU(K={self.quant.K}, is_activation={self.quant.is_activation})"
    
    @property
    def max(self):
        return self.quant.range[1]

# _________________Centering_________________
class WeightCentering(Mod):
    """
    Idea taken from: "High-Performance Large-Scale Image Recognition Without Normalization" / 
    CHARACTERIZING SIGNAL PROPAGATION TO CLOSE THE PERFORMANCE GAP IN UNNORMALIZED RESNETS

    """
    def __init__(self, channels, const = None):
        super().__init__()
        self.const = const
        if self.const is not None:
            self.mean_weight = Parameter(torch.zeros(channels))
        
    def forward(self, x:Tensor) -> Tensor:
        """
        x [c_out c_in,...]
        """       
        if x.dim()==4:
            dims = (1,2,3)
        else:
            dims = (1,)
        if self.const is not None:
            # y = x - self.const
            y = x - self.const + self.mean_weight.view([-1] + [1]*len(dims))*0.3
        else:
            # raise RuntimeError("WTF")
            mean = x.mean(dim=dims, keepdim=True) # ch_in, W, H
            # y = x - mean + self.mean_weight.view([-1] + [1]*len(dims))*0.3
            y = x - mean
        return y

# _________________QAnyLinear___________________


class QAnyLinear:
    """ This class abstracts weight quantization in any Linear or Conv layer
        Assumes that in the derived class self.bias and self.weight will exist
        Requires:
        o -- passed to QReLU
    """
    
    # skip_rel_lr = 0.01
    @staticmethod
    def muM(K):
        mu = (K-1)/2
        M = (K-1)*(2*K-1)/6
        return mu,M
    
    @staticmethod
    def varK(K):
        return (K-1)*(K+1)/12

    def xw_var_init(self, Kx, Kw):
        mu1, M1 = self.muM(Kw)
        mu2, M2 = self.muM(Kx)
        var = M1*M2 - (mu1*mu2)**2
        return var

    def __init__(self, o: dotdict, out_channels: int, norm = True, convex_comb=False, centering = 'mean', **kwargs):
        """ parent constructor disabled because nn.Module gets initialized by the other parent class
        convex_comb was applicable in BiNeal-like architectures, introducing convex combination with learnable coefficient instead of just sum of two branches, currently replaced with concatenated input features
        centering: 'mean' -- subtract mean over all weights for each output channel
                    'const' -- subtract (K-1)/2 from all weights, so that they are in the range [-(K-1)/2, (K-1)/2]
                    None -- no centering
        """
        #
        self.out_channels = out_channels
        quantizer = QReLU(o.QReLU_W)
        del quantizer[0] # no need input scaling and shifting of latent weights

        if centering == 'mean':
            w_centering = WeightCentering(out_channels)           
        elif centering == 'const':
            w_centering = WeightCentering(out_channels, const = (quantizer.quant.K-1)/2 )
            raise RuntimeError('unexpected')
        else: # None or 'none'
            w_centering = Identity()
        #
        self.w_pipeline = ESequential([quantizer, w_centering])
        if norm:
             # DEBUG (eps=0 was the default, if it fails hard we need a better fix)
            self.norm = Norm(out_channels, affine=False, eps=1e-5) # Why affine=False? QReLU has leearnable ScaleBias
            self.norm.init_only = False
        else:
            self.norm = Identity()

        self.convex_comb=convex_comb
        if convex_comb:
            # self.convex_w = torch.nn.Parameter(torch.ones(out_channels)*0.5)
            self.convex_w = torch.nn.Parameter(torch.tensor(1.0)) # goes into methods AnyLinear alpha= sigmoid, for convolutions created in CatnMerge, where the formula is (1-alpha)*skip + alpah*residual. So sigmoid(1) favours deep connections

        if self.bias is not None:
            self.bias.data.zero_()
        W = self.weight
        """
        When loss is invariant to the choice of weight init scale, the init scale controlls only the relative learning rate
        We can set either: the initial scale or the local learning rate, we choose to set the scale at convenience according
        the number of states, and corresponding local_lr
        The standard He init would set w ~ U[+-sqrt(3/n_in) ] and lr = 1.
        This is equivalent to w ~[0, 1] and lr = sqrt(n_in/3)/2
        This is equivalent to w ~[-0.5, Kw-0.5] and lr = Kw*sqrt(n_in/3)/2
        We do not want it to vary with n_in and can set some nominal n_in for the network
        """
        Kw = quantizer.quant.K
        # Kx = o.A # activation quantization
        W.data.uniform_(-0.5, Kw - 0.5) # same mass to all bins. E.g for Kw = 2 uniform on [-0.5, 1.5], will be quiantized to {0, 1}
        n_in = math.prod(W.shape[1:]) # do not want to use this dynamically
        n_in = 3*3*128
        # W.local_lr = Kw * math.sqrt(n_in/3)/2 # factor of about 9.79
        W.local_lr = Kw
        W.eps = (math.sqrt(256*9)*8)/(math.sqrt(n_in)*Kw)*o.Adam_eps
        # self.norm_0 = self.wnorm() # weihgt norm at initialization

    # def wnorm(self):
    #     # n_in = math.prod(self.weight.shape[1:]) # do not want to use this dynamically
    #     norm = (self.weight.view([self.weight.shape[0], -1]).std(dim = 1, keepdim=True) + 0.1)
    #     return norm.view([self.weight.shape[0]] + [1]*(self.weight.dim()-1))

    # # Actually all these interfere with forward and GD averaging (projected SGD is problematic?)
    # # proj2
    # def w_project(self):
    #     W = self.weight
    #     W.data -= W.mean(dim=(1,2,3),keepdim=True)
    #     W.data *= (self.norm_0.to(W) / self.wnorm())
    #     W.data += (self.quantizer.quant.K-1)/2

    # # projmax
    # def w_project(self):
    #     W = self.weight
    #     W.data.clamp_(min = -1, max = self.quantizer.quant.K)

    # projmax1
    def w_project(self):
        # DEBUG:
        # return
        W = self.weight
        K = self.quantizer.quant.K
        W.data.clamp_(min = -2, max = K+1)

    # # proj3
    # def w_project(self):
    #     W = self.weight
    #     W.data -= W.mean(dim=(1,2,3),keepdim=True)
    #     W.data *= torch.clamp((self.norm_0.to(W) / self.wnorm()), max = 1.0) # ! Only shrink down if expanding, not inflate
    #     W.data += (self.quantizer.quant.K-1)/2


    @property
    def quantizer(self) -> QReLU:
        return self.w_pipeline[0]
    
    @property
    def centering(self):
        return self.w_pipeline[1]

    def fan_in(self):
        s = self.weight.shape
        return s[1]*s[2]*s[3]

    def reg(self):
        Kw = self.quantizer.quant.K
        return (((self.weight+0.5)/Kw - 0.5)**2).sum()

    def __repr__(self):
        # s = nn.Conv2d.__repr__(self)
        if self.weight.dim() == 2:
            s = f'QLinear({self.weight.shape[1]}->{self.weight.shape[0]}'
        else:
            s = f'QConv2d({self.weight.shape[2]}x{self.weight.shape[3]}, {self.weight.shape[1]}->{self.weight.shape[0]}'
            if self.stride not in {1, (1,1)}:
                s += f', stride={self.stride}'
            if self.dilation not in {1, (1,1)}:
                s+= f', dilation = {self.dilation}'
            if self.groups != 1:
                s+= f', groups = {self.groups}'
        s += f', KW={self.quantizer.quant.K}'
        if self.convex_comb: 
            alpha = self.convex_w.sigmoid().detach().cpu().item()
            s += f', alpha={alpha:3.2f}'
        if self.norm.__class__ is not Identity:
            s += f', {self.norm.__class__.__name__}'
        s += ')'
        # s+= '\n'
        # s += f'\t norm: {str(self.norm)}'
        return s

    def out_SB(self):
        """
        """
        raise DeprecationWarning('')
        W = self.weight
        dims = np.arange(1, W.dim() + 1) # assime first dimension is output_channels
        n_in = np.prod(W.shape[dims])
        K = self.quantizer.quant.K
        Eu = (K-1)/2
        Eu = 1/2
        S_out = Eu/n_in # controlls the magnitude of updates
        sb = ConstantScaleBias()
        sb.set_to_map(self.quantizer.quant.range, (-S_out, S_out)) # weight norm will remove the shift in any case
        sb.wb[1] = 0 # drop the bias
        return sb

    def forward(self, x: Tensor, method: 'Method' = None, **kwargs):
        return method.dispatch(self, x, **kwargs)

    @abc.abstractmethod
    def forward_WB(self, x, weight, bias) -> Tensor:
        """ to be implemented by specific realizations, must do the normal forward propagation with the given weight and bias
          preserving all the stride, etc settings """
        return


class Cat(KWLayer):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def default_forward(self, xx:tuple|list):
        if isinstance(xx[0], tuple):
            x0 = torch.cat([x[0] for x in xx], dim = self.dim)
            x1 = torch.cat([x[1] for x in xx], dim = self.dim)
            return (x0, x1)
        else:
            return torch.cat(xx, dim = self.dim)

    

# _________________QLinear___________________


class QLinear(QAnyLinear, nn.Linear):
    def __init__(self, o: dotdict, in_channels, out_channels, *args, norm=True, convex_comb=False, centerin:str = 'mean', **kwargs):
        nn.Linear.__init__(self, in_channels, out_channels, *args, **kwargs)
        QAnyLinear.__init__(self, o, out_channels, norm=norm, convex_comb=convex_comb, centerin=centerin, **kwargs)


    def forward_WB(self, x, weight, bias):
        return F.linear(x, weight, bias)

# _________________QConv2d___________________

@dataclass
class ConvOptions:
    in_channels:int
    out_channels:int
    kernel_size:int
    stride:int=1
    dilation:int=1
    groups:int=1
    padding:int|None = None
    padding_mode = 'replicate',
    bias:bool = False

    def __post_init__(self):
        if self.padding is None:
            self.padding = self.kernel_size // 2
        assert self.padding is not None  # tells Pylance it's int

class QConv2d(QAnyLinear, nn.Conv2d):
    def __init__(self, o: dotdict, in_channels, out_channels, *args, norm=True, convex_comb=False, centerin:str = 'mean', **kwargs):
        nn.Conv2d.__init__(self, in_channels, out_channels, *args, **kwargs)
        QAnyLinear.__init__(self, o, out_channels, norm=norm, convex_comb=convex_comb, centerin=centerin, **kwargs)

    def forward_WB(self, x, weight, bias):
        return self._conv_forward(x, weight, bias)


#_________________Reshaping, etc Layers_____________________________

class ELoss(Mod):
    def __init__(self, gt: Tensor, loss_fn: Callable[[Tensor, Tensor], Tensor]):
        super().__init__()
        self.gt = gt.detach()
        self.loss_fn = loss_fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.loss_fn(x, self.gt)


class FuncLayer(nn.Module):
    """ Wrapper to create a layer out of a function """

    def __init__(self, func: Callable[..., Tensor], *args) -> None:
        super().__init__()
        self.args = args
        self.func = func

    def forward(self, x: Tensor) -> Tensor:
        y = self.func(x, *self.args)
        return y


class KWFuncLayer(KWLayer):
    """ Wrapper to create a layer out of a function """

    def __init__(self, func: Callable[..., Tensor], *args) -> None:
        super().__init__()
        self.args = args
        self.func = func

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        y = self.func(x, *self.args, **kwargs)
        return y



def split_dim(x: Tensor, d1: int, size: int) -> Tensor:
    s = list(x.shape)
    return x.view(s[:d1] + [-1, size] + s[d1+1:])


def merge_dims(x: Tensor, d1: int, d2: int) -> Tensor:
    # merge two dimensions d1 and d2 and put them to d1
    assert(d1 >= 0)
    x = x.unsqueeze(d1+1)  # make 1 at d1+1
    if d2>d1:
        d2 = d2 + 1  # position of d2 has moved (will fail for d2 being negative and before d1 effectively)
    x = x.swapdims(d1+1, d2)  # swap d2 with 1 at d1+1
    x = x.squeeze(d2)  # remove 1 at d2
    s = list(x.shape)
    x = x.view(s[:d1] + [-1] + s[d1+2:])
    return x


# _________________ResNet Layers___________________

class Identity(KWLayer):
    def forward(self, x, **kwargs):
        return x

# _________________Special Output Layer___________________

class Parallel(ESequential):
    """ Propagation method will call all layers in the list on the same input and perform a reduciton of the outputs """
    def __init__(self, *args: nn.Module, reduction = 'sum') -> None:
        super().__init__(*args)
        self.reduction = reduction
    

# _________________Aux Experimental Hacking Layers___________________

class Tuple0(KWLayer):
    def forward(self, xx, **kwargs):
        if isinstance(xx, tuple):
            return xx[0] # 
        else:
            return xx

#________________________Network Functions_______________________________________

# class EClassificationNet(ESequential):
class EClassificationNet(KWLayer):
    def loss(self, x, y):
        # return F.binary_cross_entropy_with_logits(x, y.float(),reduction='none')
        return nn.CrossEntropyLoss(reduction='none')(x, y)

    def loss_network(self, targets):
        # return ESequential(self, ELoss(targets, nn.CrossEntropyLoss(reduction='none').forward))
        return ESequential(self, ELoss(targets, self.loss))

    def net_loss(self, input, targets, method):
        # x = self.forward(input, method=method)
        # return x
        # return ELoss(targets, nn.CrossEntropyLoss(reduction='none'))(x)
        return self.loss_network(targets).forward(input, method=method)


def init_net(net: ESequential, bn_weight=1.0, bn_bias=0.0):
    """Initializes BN scales """
    for m in net.modules():
        if isinstance(m, _BatchNorm):
            if m.affine:
                m.weight.data.fill_(bn_weight)
                m.bias.data.fill_(bn_bias)



class AdaptiveLSEPool2d(Mod):
    def __init__(self, output_size, **kwargs):
        super().__init__(**kwargs)
        self.output_size = output_size
        assert (output_size in {1, (1,1)})

    def forward(self, x, **kwargs):
        return x.view(list(x.shape[0:2]) + [-1] ).logsumexp(dim=-1, keepdim=True).unsqueeze(-1)
    

class QLinearQuantized(KWLayer):
    def __init__(self, o: dotdict, in_channels, out_channels, bias=True, centering = 'mean'):
        super().__init__()
        self.linear = QConv2d(o, in_channels, out_channels, kernel_size=1, bias=bias, centerin=centering, norm=True)
        self.quant = QReLU(o.QReLU_A, channels=out_channels) # per-channel quantization of activations

    def forward(self, x: Tensor, **kwargs):
        x = self.linear(x, **kwargs)
        x = self.quant(x, **kwargs)
        return x


def unfold_output_shape(input_shape, kernel_size, stride=1, padding=0, dilation=1):
    """
    input_shape: (N, C, H, W)
    Returns: H_out, W_out
    """
    N, C, H, W = input_shape

    assert isinstance(kernel_size, int)
    assert isinstance(stride, int)
    assert isinstance(padding, int)
    assert isinstance(dilation, int)

    H_out = math.floor((H + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
    W_out = math.floor((W + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)

    return (H_out, W_out)


# NOTE: 2D self-attention (SelftAttention2d, GlobalSelftAttention2d) and their
# helpers (local_attention, local_attention_agg) are future/experimental work,
# moved to arch_experiments.py (working repo only, not in the public release).

