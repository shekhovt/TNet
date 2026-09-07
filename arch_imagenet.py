# SPDX-License-Identifier: CC BY-NC-SA 4.0
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. Making code open source is essential for scientific
# transparency and reproducibility. Providing access to the code allows researchers to
# verify, replicate, and expand upon results, which supports scientific progress. Current
# top-tier conferences encourage opensource code for the purpose of reproducibility that
# is de-facto an established expectation of the research community. Opensource code as
# part of publication should not affect the performance of an invention both from a
# commercial and IP point of view.
import torchvision.models as models
import contextlib
from dataclasses import asdict

# -----------------
from .tools import *
from .methods import *
from .layers import *
import copy

# we are using now integer inputs {0,.., K-1} to conv, both padding modes are legal, replicate seems a bit better
padding_mode = 'replicate'
# padding_mode0 = 'replicate'

# ── Gate registry ────────────────────────────────────────────────────────────
# Experimental / "future" gates (Squeeze-Excitation, attention, …) register
# themselves here from `arch_experiments.py`, which is kept in the working repo
# only and is NOT part of the public release. Paper gates are handled directly
# by the if/elif ladder in `QResNet18.construct`. A builder receives a `dotdict`
# context and returns a dict of construct overrides (`Block`, `block_args`,
# `layers_stage`, `tower_order`, optionally `regtower_order`, `base_channels`,
# `chmult_stage`).
GATE_BUILDERS = {}

def register_gate(name):
    def deco(fn):
        GATE_BUILDERS[name] = fn
        return fn
    return deco

# Net registry, used the same way for whole networks that live outside this file
# (e.g. the vision transformer in `transformer.py`, which is also working-repo
# only). `create_net` consults it before falling back to this module's globals.
NET_BUILDERS = {}

def register_net(name):
    def deco(cls):
        NET_BUILDERS[name] = cls
        return cls
    return deco

_current_verbosity = 5

def get_verbosity():
    return _current_verbosity

def set_verbosity(level):
    global _current_verbosity
    _current_verbosity = level

    # @property
    # def args(self):
    #     return self.in_channels, self.out_channels, self.kernel_size
    
    # @property
    # def args(self):
    #     return self.in_channels, self.out_channels, self.kernel_size



@contextlib.contextmanager
def verbosity(level):
    global _current_verbosity
    prev_level = _current_verbosity
    _current_verbosity = level
    try:
        yield
    finally:
        _current_verbosity = prev_level

#_________________________________________________________
class Wrapped(EClassificationNet, ESequential):
    def __init__(self, net):
        super().__init__()
        self += [net]
        



   
def forward_binary_gate(qmask:QMask, x, y, **kwargs):
    g, q = qmask.forward(y.shape, **kwargs)
    # p = p.clamp(min=0.1, max=0.9)
    # r = y*g/p + x * (1-g)/(1-p)
    rq = x * (1-g) + y*g
    r = x * (1-q) + y*q
    # r = x + y
    r = subst_grad(rq, r)
    # where y=x can pass without gate
    # both = (x + y) / 2
    # gated = y*g + x * (1-g)
    # r = torch.where((x - y).abs() < 1e-3, both, gated)
    return r

@torch.compile(**compile_args)
def forward_binary_gate_compiled(qmask:QMask, x, y, **kwargs):
    return forward_binary_gate(qmask, x, y, **kwargs)
    
def forward_cat_gate(cat:Categorical, logits, x, y, **kwargs):
    C = logits.shape[-1]
    L = logits.view([1, -1, 1, 1, C]).expand(list(y.shape) + [C]) # [N C H W Cat]
    G = cat.forward(L, **kwargs) # [N C H W Cat] one-hot
    # R = G.new_empty(G.shape)
    # R[...,0] = x
    # R[...,1] = y
    # R[...,2] = torch.min(x,y)
    # R[...,3] = torch.max(x,y)
    # R[...,4] = torch.clamp(x-y,min=0)
    # R[...,5] = torch.clamp(y-x,min=0)
    # R[...,6] = torch.max(y-x,x-y)
    # r = (R*G).sum(dim=-1)            
    r = G[...,0]*x + G[...,1]*y + G[...,2]*torch.min(x,y) + G[...,3]*torch.max(x,y) + G[...,4]*torch.max(y-x,x-y)
    return r

@torch.compile(**compile_args)
def forward_cat_gate_compiled(cat:Categorical, logits, x, y, **kwargs):
    return forward_cat_gate(cat, logits, x, y, **kwargs)


    
#__________________________QResNet_____________________________________________________________

def normal_block(in_channels, out_channels, o, stride=1, dilation = 1, groups = 1, kernel_size = 3, convex_comb=False, padding_mode=padding_mode, centering = 'mean'):
    ll = []
    padding = 'same' if stride == 1 else (kernel_size*dilation//2)
    ll += [QConv2d(o, in_channels, out_channels, kernel_size=kernel_size, stride= stride, dilation = dilation, padding=padding, groups=groups, padding_mode=padding_mode, bias=False, convex_comb=convex_comb, centerin=centering)]
    if groups >1:
        ll += [nn.ChannelShuffle(groups=groups)]
    # ll += [make_norm(out_features, learnable=False)] # QReLU has a learnable sb_in (local and global)
    ll += [QReLU(o.QReLU_A, channels=out_channels)]
    return ESequential(*ll)



class GenResBlock(KWLayer):
    def fan_in(self, i):
        in_nodes = {pair[0] for pair in self.pairs if pair[1] == i}
        return in_nodes
    
    def __init__(self, in_features, out_features, o, next_o = None, stride=1, dilation=1, groups = 1, kernel_size = 3, order = 3, **kwargs):
        super().__init__()
        self.order = order
        if next_o is None:
            next_o = o
        # variant = 'fw'
        variant = 'rec'
        if variant =='fw': # baseline all fw:
            pairs = {(0, i) for i in range(1, order + 1)}
        if variant =='bw': # baseline all fw:
            pairs = {(i, order) for i in range(order)}
        elif variant =='fw-sparse': # variant 1: sparse
            pairs = {(0, order), (0, order//2)}
        elif variant =='rec':
            # variant 2: recurrent
            def rec_pairs(p):
                pairs = set()
                if p[1] > p[0] + 1:
                    med = p[0] + (p[1]-p[0])//2
                    pairs = pairs | {p} | rec_pairs((p[0], med)) | rec_pairs((med, p[1]))
                return pairs
            pairs = rec_pairs((0, order))
        else:
            raise RuntimeError('variant not set')
        
        print(pairs)

        for i in range(1, order + 1):
            pairs.add((i-1, i)) # add the backbone sequential
            pairs = pairs - {(i, i)} # remove self-loops if present
        
        self.pairs = pairs
        ll = []
        # create fused convolutions
        for i in range(1, order + 1):
            in_nodes = self.fan_in(i)
            for j in in_nodes:
                assert(j < i)
            if i<order: # not the last one
                ll += [normal_block(in_features*len(in_nodes), in_features, o, stride=1, dilation = dilation, groups=groups, kernel_size=kernel_size)]
            else: # last one -- reduction
                ll += [normal_block(in_features*len(in_nodes), out_features, next_o, stride=stride, dilation = dilation, groups=groups, kernel_size=kernel_size)]
        self.blocks = ESequential(ll)
        # create weight coeffs
        self.w = torch.nn.Parameter(torch.ones(order+1, order+1)*(-10)) # init all as inactive
        for i in range(1,order+1):
            for j in self.fan_in(i):
                self.w.data[j,i] = (i-j)*0.2 # init with a hihgher score for deep paths
        # self.pair2ind = {p:i for (i,p) in enumerate(pairs)}

    def forward(self, x, **kwargs):
        # alpha = self.convex_w.sigmoid()
        # alpha = torch.sigmoid(self.w) + self.w - self.w.detach()
        alpha = self.w.softmax(dim=0)
        outs = [x]
        for i in range(1,self.order + 1):
            in_nodes = self.fan_in(i)
            # print(i, in_nodes)
            # xx = torch.cat([outs[j]*alpha[j,i] for j in in_nodes], dim=1)
            xx = torch.cat([outs[j] for j in in_nodes], dim=1)
            y = self.blocks[i-1].forward(xx, **kwargs)
            outs.append(y)
        return y
        
    def __repr__(self):
        if get_verbosity() == 1: # brief
            s = f"GenResBlock({self.order})"
        else:
            s = super().__repr__()
        return s


# NOTE: Squeeze-Excitation blocks (SE_Decoder, SE_Encoder, SEBlock, GenResBlockSE)
# and their gates (GenResSE / GenResSA / GenResSGA) are future/experimental work,
# kept in arch_experiments.py (working repo only, not in the public release).
# arch_experiments.py registers those gates into GATE_BUILDERS on import (see end of file).


class ReductionBlock(KWLayer):
    def __init__(self, in_features, out_features, o, stride=2, dilation = 1, groups = 1, kernel_size = 3, gate_kind='none'):
        super().__init__()
        self.normal = normal_block(in_features, out_features, o, stride=stride, dilation = dilation, groups=groups, kernel_size=kernel_size)
        self.eta = Parameter(torch.ones(out_features)*2)
        self.quantizer = QReLU(o.QReLU_A)
        del self.quantizer[0] # drop in_sb part, assuming input is quantization-ready
        self.pool = nn.AvgPool2d(kernel_size=kernel_size,padding='same',stride=stride)
        self.in_features = in_features
        self.out_features = out_features
        self.gate_kind = gate_kind

    def forward(self, x, **kwargs):
        y = self.normal.forward(x, **kwargs)
        if self.gate_kind == 'none':
            return y
        else:
            mx = self.pool(x)
            if self.out_features > self.in_features:
                mx = torch.cat([mx,mx], dim=1) # duplicate along channels
            alpha = torch.sigmoid(self.eta)
            alpha = subst_grad(alpha, self.eta)
            alpha = alpha.view([1,-1,1,1])
            r = (1-alpha)*mx + alpha* y
            qr = self.quantizer.forward(r, **kwargs)
            return qr


def residual_block(in_features, out_features, o, stride=1, groups = 1, Q=True):
    ll = []
    ll += [QConv2d(o, in_features, out_features, kernel_size=3, stride= stride, padding=1, groups=groups, padding_mode=padding_mode, bias=False)]
    if groups >1:
        ll += [nn.ChannelShuffle(groups=groups)]    
    if Q: # quantizing here, used in reduction layers
        # ll += [make_norm(out_features, learnable=False)] # QReLU has a learnable sb_in
        ll += [QReLU(o.QReLU_A, channels=out_features)]
    else: # this will be the output to be added to the skip branch
        # ll += [make_norm(out_features, learnable=False)] # QReLU has a learnable sb_in
        # ll += [make_norm(out_features, learnable=False)]
        ll += [QReLU(o.QReLU_A, channels=out_features)]
        del ll[-1][-1] # delete quantizer
        # ll += [ScaleBias(channels=out_features, weight=(o.A-1), bias=True, learnable=True)]
        # ll += [ScaleBias(channels=out_features, weight=1.0, bias=True, learnable=True)] # if we have ReLU should enable bias
        # ll += [nn.ReLU()]        
    return ESequential(*ll)

def residual_block2(in_features, out_features, o, stride=1, groups = 1, Q=True):
    # ll = [normal_block(in_features, out_features, o, groups=1, kernel_size=1)]
    ll = [normal_block(in_features, out_features, o, groups=groups)]
    ll += [residual_block(in_features, out_features, o, stride=stride, groups=groups, Q=Q)]
    return ESequential(*ll)

"""Reproducing BiNeal-Net. In our implementation can use Tower2 instead """
class BiNealBlock(KWLayer):
    def __init__(self, in_features, out_features, o, next_o=None, stride=2, groups = 1, kernel_size = 3, skip_kernel_size=1, gate_kind='none', dilation = 1, **kwargs):
        super().__init__()
        if next_o is None:
            next_o = o
        self.normal = ESequential([normal_block(in_features, out_features, o, stride = stride, groups=groups, kernel_size=kernel_size),
                                  QConv2d(next_o, out_features, out_features, kernel_size=kernel_size, stride= 1, padding=kernel_size//2, groups=groups, padding_mode=padding_mode, bias=False, norm=False),
                                  make_norm(out_features, learnable=True, init_only=False),
                                  ])
        kernel_size = 1
        self.skip =  ESequential([QConv2d(next_o, in_features, out_features, kernel_size=skip_kernel_size, stride = stride, padding=skip_kernel_size//2, groups=groups, padding_mode=padding_mode, bias=False, norm=False),
                                 make_norm(out_features, learnable=True, init_only=False),
                                 ])
        # self.eta = Parameter(torch.ones(out_features)*(0))
        self.quantizer = ESequential([make_norm(out_features, learnable=True), QReLU(next_o.QReLU_A, channels=out_features)])

    def forward(self, x, **kwargs):
        y = self.normal.forward(x, **kwargs)
        x = self.skip.forward(x, **kwargs)
        # alpha = torch.sigmoid(self.eta)
        # alpha = subst_grad(alpha, self.eta)
        # alpha = alpha.view([1,-1,1,1])
        # alpha = 0.5
        # r = (1-alpha)*x + alpha* y
        r = x + y
        # r = x
        assert(kwargs['method'] is not None)
        qr = self.quantizer.forward(r, **kwargs)
        return qr


class CatnMerge(KWLayer):
    def __init__(self, o, in_features, out_features, stride=1, dilation = 1, groups =1 , kernel_size=1,  convex_comb = True, padding_mode=padding_mode):
        super().__init__()
        self.cat = Cat(dim=1)
        self.merge = normal_block(in_features*2, out_features, o, stride=stride, dilation = dilation, groups=groups, kernel_size=kernel_size, convex_comb=convex_comb, padding_mode=padding_mode)

    def forward(self, x, y, **kwargs):
        y = self.cat.forward([x,y], **kwargs)
        r = self.merge.forward(y, **kwargs)
        return r


class CatBlock(KWLayer):
    def __init__(self, in_features, out_features, o, next_o = None, stride=1, dilation = 1, groups = 1, kernel_size = 3, **kwargs):
        super().__init__()
        if next_o is None:
            next_o = o
        self.residual = normal_block(in_features, in_features, o, stride=1, dilation = dilation, groups=groups, kernel_size=kernel_size)
        self.cat = Cat(dim=1)
        self.merge = normal_block(in_features*2, out_features, next_o, stride=stride, dilation = dilation, groups=groups, kernel_size=kernel_size)

    def forward(self, x, **kwargs):
        y = self.residual.forward(x, **kwargs)
        # y = torch.cat([x,y], dim=1) # cat along channles
        y = self.cat.forward([x,y], **kwargs)
        r = self.merge.forward(y, **kwargs)
        return r

class TowerBlock(KWLayer):
    def __init__(self, in_features, out_features, o, next_o = None, stride=1, dilation=1, groups = 1, kernel_size = 3, order = 3, reduction_first= False):
        super().__init__()
        self.order = order
        if next_o is None:
            next_o = o
        if order == 1:
            self.residual = None
        elif order == 2:
            if reduction_first:
                self.residual = normal_block(in_features, out_features, o, stride=stride, dilation = dilation, groups=groups, kernel_size=kernel_size)
            else:
                self.residual = normal_block(in_features, in_features, o, stride=1, dilation = dilation, groups=groups, kernel_size=kernel_size)
        else: # order >2
            if reduction_first:
                self.residual = TowerBlock(in_features, out_features, o, stride=stride, dilation = dilation, groups = groups, kernel_size = kernel_size, order = order-1, reduction_first=True)
            else:
                self.residual = TowerBlock(in_features, in_features, o, stride=1, dilation = dilation, groups = groups, kernel_size = kernel_size, order = order-1)
        if self.residual is None:
            self.normal = normal_block(in_features, out_features, next_o, stride=stride, dilation = dilation, groups=groups, kernel_size=kernel_size)
        else:
            if reduction_first:
                self.merge = CatnMerge(next_o, out_features, out_features,stride=1, dilation = dilation, groups=groups, kernel_size=kernel_size)
            else:
                self.merge = CatnMerge(next_o,in_features=in_features,out_features=out_features,stride=stride, dilation = dilation, groups=groups, kernel_size=kernel_size)
        #     in_features = in_features *2
        #     self.cat = Cat(dim=1)
        # self.merge = normal_block(in_features, out_features, next_o, stride=stride, groups=groups, kernel_size=kernel_size)

    def forward(self, x, **kwargs):
        if self.residual is None:
            return self.normal.forward(x, **kwargs)
        else:
            y = self.residual.forward(x, **kwargs)
            return self.merge.forward(x, y, **kwargs)
        
    def __repr__(self):
        if get_verbosity() == 1: # brief
            s = f"Tower({self.order})"
        else:
            s = super().__repr__()
        return s
    

class FusedResBlock(KWLayer):
    def __init__(self, in_features, out_features, o, next_o = None, stride=1, dilation=1, groups = 1, kernel_size = 3, order = 3, reduction_first= False):
        super().__init__()
        self.order = order
        if next_o is None:
            next_o = o
        ll = []
        for i in range(order - 1):
            ll += [normal_block(in_features, in_features, o, stride=1, dilation = dilation, groups=groups, kernel_size=kernel_size)]
        self.blocks = ESequential(ll)
        self.fusion = normal_block(in_features*order, out_features, next_o, stride=stride, dilation = dilation, groups=groups, kernel_size=kernel_size)

    def forward(self, x, **kwargs):
        # outs = torch.empty([x.shape[0], self.order] + list(x.shape[1:]) , dtype=x.dtype, device = x.device) # B x Order x Channels x W x H 
        # outs[:,0] = x
        outs = [x]
        for i in range(self.order - 1):
            # outs[:,i+1]  = self.blocks[i].forward(outs[:,i], **kwargs)
            outs.append(self.blocks[i].forward(outs[i], **kwargs))
        outs = torch.cat(outs, dim=1)
        y = self.fusion.forward(outs, **kwargs)
        # y = self.fusion.forward(outs.flatten(start_dim=1,end_dim=2), **kwargs)
        return y
        
    def __repr__(self):
        if get_verbosity() == 1: # brief
            s = f"Fusion({self.order})"
        else:
            s = super().__repr__()
        return s





# class Block0_b0(ESequential):
#     def __init__(self, o, next_o, out_features):
#         """ reduce resolution to 1/4 while encoding with (next_o, out_features)
#         o is used for weights and activations in input convs
#         next_o is used for weights and activatinos in merge conv
#         """
#         in_features = 3
#         kernel_size = 7        
#         ll = []
#         ll += normal_block(in_features, out_features, o, stride=2, kernel_size=kernel_size)
#         ll += normal_block(out_features, out_features, next_o, stride=2, kernel_size=3)
#         super().__init__(*ll)

#     def __repr__(self):
#         if get_verbosity() == 1:
#             s = self.__class__.__name__ + ', '.join(l.__class__.__name__ for l in self) + ')'
#         else:
#             s = super().__repr__()
#         return s

class Block0_b1(ESequential):
    def __init__(self, o, next_o, out_features):
        """ reduce resolution to 1/4 while encoding with (next_o, out_features)
        o is used for weights and activations in input convs
        """
        ll = []
        in_features = 3
        kernel_size = 7
        # ll += [QConv2d(o, in_features, out_features, kernel_size=kernel_size, stride= 2, padding=kernel_size//2, padding_mode=padding_mode, bias=False, norm = False)]
        ll += [QConv2d(o, in_features, out_features, kernel_size=kernel_size, stride= 2, padding=0, bias=False, norm = False)]
        ll += [nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)]
        # ll += [make_norm(channels = out_features, learnable=True)] # don't need learnable before QRELU
        ll += [make_norm(channels = out_features, learnable=False)] # don't need learnable before QRELU (it has learnable input_sb)
        ll += [QReLU(o.QReLU_A, channels=out_features)]
        super().__init__(*ll)

class Block0_resnet(ESequential):
    def __init__(self, o, next_o, out_features):
        """ reduce resolution to 1/4 while encoding with (next_o, out_features)
        o is used for weights and activations in input convs
        """
        ll = []
        in_features = 3
        kernel_size = 7
        ll += [QConv2d(o, in_features, out_features, kernel_size=kernel_size, stride= 2, padding=kernel_size//2, padding_mode=padding_mode, bias=False, norm = False)]
        ll += [make_norm(channels = out_features, learnable=True)]
        ll += [QReLU(o.QReLU_A, channels=out_features)]
        ll += [nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)]
        super().__init__(*ll)


# class Block0_b1_fp(ESequential):
#     def __init__(self, o, next_o, out_features):
#         """ reduce resolution to 1/4 while encoding with (next_o, out_features)
#         o is used for weights and activations in input convs
#         next_o is used for weights and activatinos in merge conv
#         """
#         ll = []
#         in_features = 3
#         kernel_size = 7
#         ll += [nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride= 2, padding=kernel_size//2, padding_mode=padding_mode, bias=False)]
#         ll += [nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)]
#         ll += [make_norm(channels = out_features, learnable=False)] # don't need learnable before QRELU
#         ll += [QReLU(o.QReLU_A, channels=out_features)]
#         super().__init__(*ll)

# Block0 = Block0_b1

# class Block0_baseline(ESequential):
#     """
#     flat 2-layer architecture
#     """
#     def __init__(self, o, next_o, out_features):
#         ll = []
#         kernel_size = 5
#         n_in = 3*kernel_size**2 # 3 for input channels
#         w_var = QAnyLinear.varK(o.QReLU_W.K)
#         scale = (3/ (n_in * w_var)) **0.5
#         ll += [QConv2d(o, 3, out_features, kernel_size=kernel_size, stride=2, padding='same', padding_mode=padding_mode, bias=False, norm=False)]
#         quant = QReLU(o.QReLU_A, channels=out_features)
#         quant[0].weight.data *= scale
#         ll += [quant]
#         ll += normal_block(out_features, out_features, next_o, stride=2, groups=1, kernel_size=3)
#         super().__init__(*ll)


class NormalBlock(KWLayer):
    def __init__(self, in_features, out_features, o, next_o = None, stride=1, dilation = 1, groups = 1, kernel_size = 3, **kwargs):
        super().__init__()
        if next_o is None:
            next_o = o
        # self.block = ESequential([normal_block(in_features, in_features, o, stride=1, groups=groups, kernel_size=kernel_size),
        #                           normal_block(in_features, out_features, next_o, kernel_size=kernel_size, stride= stride, groups=groups)])

        self.block = normal_block(in_features, out_features, o, stride=stride, dilation = dilation, groups=groups, kernel_size=kernel_size)

    def forward(self, x, **kwargs):
        return self.block.forward(x, **kwargs)


class Gate(KWLayer):
    def __init__(self, layer, skip, out_features, o, gate_kind = None):
        super().__init__()
        self.layer = layer
        self.skip = skip
        self.gate_kind = gate_kind
        if gate_kind == 'QMask':
            gate = QMask(out_features, o.QReLU_W.replace(K=2), eta = 0) # eta <0.5 -- higher chance for skip connection, eta > 0.5 -- higher chance for res connection
            self.gate = gate    
        elif gate_kind == 'none' or gate_kind is None:
            pass
        elif gate_kind == 'requantize':
            # self.eta = Parameter(torch.ones((out_features))*(-1))
            self.eta = Parameter(torch.zeros(out_features))
            # self.quantizer = QReLU(o.QReLU_A, channels=None)
            # del self.quantizer[0] # drop in_sb part, assuming input is quantization-ready
            # replace ScaleBias, asssuming input is quantization-ready
            # self.quantizer[0] = ScaleBias(channels=out_features,weight=1.0, bias = 0.0, learnable=True, is_activation=True)
            self.quantizer = Quant(o.QReLU_A)
        else:
            raise AttributeError(f'Unknown gate kind {gate_kind}')
        
        
    def forward(self, x_in, **kwargs):
        y = self.layer.forward(x_in, **kwargs)
        if self.gate_kind == 'none':
            return y
        x = self.skip.forward(x_in, **kwargs)
        if self.gate_kind == 'QMask':
            if kwargs['method'].o.compile:
                # if hasattr(x, '_dynamo_dynamic_indices'):
                #     del x._dynamo_dynamic_indices
                # if hasattr(y, '_dynamo_dynamic_indices'):
                #     del y._dynamo_dynamic_indices
                return forward_binary_gate_compiled(self.gate, x, y, **kwargs)
            return forward_binary_gate(self.gate, x, y, **kwargs)
        elif self.gate_kind == 'requantize':
            alpha = torch.sigmoid(self.eta)
            # alpha = torch.tanh(self.eta)
            alpha = subst_grad(alpha, self.eta)
            alpha = alpha.view([1,-1,1,1])
            # r = x + y*alpha # assume x is integer and y will be the learned increment by residual branch, stochastically requantized
            # r = x + y # assume x is integer and y will be the learned increment by residual branch, stochastically requantized
            r = x * (1-alpha) + y * alpha #
            qr = self.quantizer.forward(r, **kwargs)
            return qr
        else:
            raise AttributeError('Unknown gate kind')

def res_layer(channels, groups, o, gate_kind = "requintize"):
    skip = Identity()
    res_b = residual_block2
    if gate_kind == "requantize":
        normal = res_b(channels, channels, o, groups = groups, Q = False)
        return Gate(normal, skip, out_features = channels, o=o, gate_kind='requantize')
    elif gate_kind == "QMask":
        normal = res_b(channels, channels, o, groups = groups, Q = True)
        return Gate(normal, skip, out_features = channels, o=o, gate_kind='QMask')
    elif gate_kind == "none":
        return res_b(channels, channels, o, groups = groups, Q = True) # plain non-residual block

"""Older Version, suppors some variants of gates: requantize, categorical, etc """        
class QResNet(EClassificationNet):
    def __init__(self, o, gate:str = 'none', DB=True, DC=False, print_info=False, layers=None, **kwargs):
        if isinstance(DB,str):
            DB = DB=='True'
        if isinstance(DC,str):
            DC = DC=='True'
        super().__init__()
        o0 = o
        o = copy.deepcopy(o)
        o1 = copy.deepcopy(o) # first layer
        ll = []
        o1.QReLU_W = o1.QReLU_W.replace(K=16)
        # o1.Wlr = 0.087
        # o1.Wlr = 1
        stages = 5 # stages of spatial resolution reduction
        base_channels = 64
        channels = 3        
        layers_stage = np.array([0,0,0,1,2], dtype=int)
        # layers_stage = np.array([0,0,3,5,4], dtype=int)
        chmult_stage = np.array([1,1,2,4,8], dtype=int) # stage 1 replaces maxpool
        groups_stage = np.array([1,1,1,1,1], dtype=int)
        bits_stage   = (1,1,1,1,1)
        if DB: # double bits
            bits_stage = (1,1,2,2,2)
        if DC: # double channel
            chmult_stage[2:] *=2
            groups_stage[2:] *=2
        #
        for i,stage in enumerate(range(stages)):
            groups = groups_stage[i]
            next_channels = base_channels * chmult_stage[i]
            bits = bits_stage[i]
            if print_info:
                print(f'stage {stage}: ({channels}->{next_channels})\t groups {groups}\t bits {bits}')
            if o0.A == 2: # applies in case we start with binary
                o.A = 2**bits
                o.QReLU_A = o.QReLU_A.replace(K=o.A) # Activation states
            if stage==0: # first reduction from image to qunatized
                ll += [normal_block(channels, next_channels, o1, stride=2, groups = groups, kernel_size=7)] # reduction block
            else:
                ll += [normal_block(channels, next_channels, o, stride=2, groups = groups)] # reduction block
                # ll += [ReductionBlock(channels, next_channels, o, stride=2, groups = groups, gate_kind='requantize')] # reduction block
            channels = next_channels
            for l in range(layers_stage[stage]):
                if print_info:
                    print(f'\t res_layer ({channels})')
                ll += [res_layer(channels, groups, o, gate_kind=gate)]
                # ll += [CatBlock(channels, channels, o, groups=groups,stride=1,kernel_size=3)]
                #
        if print_info:
            print(f'classifier ({channels}->{o.num_classes})')
        # Classifer: [Conv 1x1, AvgPool] is the same as [AvgPool, Conv 1x1] but the second version requires less ops, even considering bits
        if False:
            ll += [nn.AdaptiveAvgPool2d((1,1))] # "requires adaptive number of bits, i.e. 6 bits to represent sum of 7x7 window with binary activations"
            ll += [QConv2d(o1, channels, o.num_classes, kernel_size=(1, 1), stride=1, bias=False)] # not a bottleneck compared to prev layer
            ll += [make_norm(o.num_classes, learnable=True)]
            ll += [ScaleBias(weight = 0.1, bias=None, learnable=False)]
            ll += [nn.Flatten()]
        else:
        # binary weights should be ok here -- lots of inputs
            ll += [QConv2d(o, channels, o.num_classes, kernel_size=(1, 1), stride=1, bias=False)]
            ll += [make_norm(o.num_classes, learnable=True)]
            ll += [ScaleBias(weight = 0.1, bias=None, learnable=False)]
            ll += [nn.AdaptiveAvgPool2d((1,1))]
            ll += [nn.Flatten()]
        super().__init__(*ll)

class BiNealNet(EClassificationNet, ESequential):
    def __init__(self, o, m=1, print_info=False, Dilation=False, skip_kernel=1, **kwargs):
        super().__init__()
        o = copy.deepcopy(o)
        o1 = copy.deepcopy(o) # first layer
        ll = []
        o1.QReLU_W = o1.QReLU_W.replace(K=256)
        groups = 1
        dilation = 1
        if Dilation:
            raise AttributeError('Not implemented yet')
            stride_reduce = 1
        else:
            stride_reduce = 2
        # stage 0
        ll += [Block0_b1(o1, o, int(64*m))]
        # layer1 of ResNet
        ll += [BiNealBlock(int(64*m), int(64*m), o, stride=1,  dilation = dilation, groups=groups, skip_kernel_size=skip_kernel)]
        ll += [BiNealBlock(int(64*m), int(64*m), o, stride=1,  dilation = dilation, groups=groups, skip_kernel_size=skip_kernel)]
        # Steg 1
        ll += [BiNealBlock(int(64*m), int(128*m), o, stride=2, dilation = dilation, groups=groups, skip_kernel_size=skip_kernel)] # expand on tage start
        ll += [BiNealBlock(int(128*m), int(128*m), o, stride=1, dilation = dilation, groups=groups, skip_kernel_size=skip_kernel)]
        # Steg 2
        ll += [BiNealBlock(int(128*m), int(256*m), o, stride=2, dilation = dilation, groups=groups, skip_kernel_size=skip_kernel)] # expand on tage start
        ll += [BiNealBlock(int(256*m), int(256*m), o, stride=1, dilation = dilation, groups=groups, skip_kernel_size=skip_kernel)]
        # Steg 3
        ll += [BiNealBlock(int(256*m), int(512*m), o, stride=2, dilation = dilation, groups=groups, skip_kernel_size=skip_kernel)] # expand on tage start
        ll += [BiNealBlock(int(512*m), int(512*m), o, stride=1, dilation = dilation, groups=groups, skip_kernel_size=skip_kernel)]
        # Classifer:
        if True: # ResNet classifier
            ll += [Distribution2ST()] # hacking to allow MeanSample
            ll += [nn.AdaptiveAvgPool2d((1,1))] # "requires adaptive number of bits, i.e. 6 bits to represent sum of 7x7 window with binary activations"
            # ll += [make_norm(int(512*m), learnable=True, init_only=False)] # learnable affine
            ll += [nn.Conv2d(in_channels=int(512*m), out_channels=o.num_classes, kernel_size=1, bias=True)]
            ll += [make_norm(o.num_classes, learnable=True)] # learnable affine
            ll += [nn.Flatten()]
        else: # our classifier
            channels = int(512*m)
            ll += [normal_block(channels, 2048, o, kernel_size=1)] # expand channels, actually don't have to quantize before pooling, it can be fused
            ll += [Tuple0()] # hacking to allow ST-WX
            ll += [nn.AdaptiveAvgPool2d((1,1))] # requires adaptive number of bits, i.e. 6 bits to represent sum of 7x7 window with binary activations -- but can swap after training
            ll += [QConv2d(o1, 2048, o.num_classes, kernel_size=(1, 1), stride=1, bias=False, norm = False)] # not a bottleneck compared to prev layer
            ll += [make_norm(o.num_classes, learnable=True)] # learnable affine
            ll += [ScaleBias(weight = 0.1, bias=None, learnable=False)] # scalar SB
            ll += [nn.Flatten()]
        super().__init__(*ll)
        # print(self)

class BOLDNet(BiNealNet):
    def __init__(self, o, m=1, skip_kernel=3, **kwargs):
        super().__init__(o,m, skip_kernel=skip_kernel, **kwargs)


class QResNet18(EClassificationNet, ESequential):
    def __init__(self, o, gate:str = 'none', DB=True, DC=False, Dilation = False, print_info=False, layers=None, m=1, **kwargs):
        super().__init__()
        self.construct(o, gate, DB, DC, Dilation, print_info, layers=layers,m=m, **kwargs)


    def construct(self, o, gate:str = 'none', DB=True, DC=False, Dilation = False, print_info=False, DW=False, layers=None, order=None, m=1, ch_grow=2, pre_cls_dim=2048, **kwargs):
        Block0 = Block0_b1
        print(gate)
        print(layers)
        if isinstance(DB,str):
            DB = DB=='True'
        if isinstance(DC,str):
            DC = DC=='True'
        if isinstance(DW,str):
            DW = DW=='True'
        o0 = o
        o = copy.deepcopy(o)
        next_o = copy.deepcopy(o)
        o1 = copy.deepcopy(o) # first layer
        ll = []
        o1.QReLU_W = o1.QReLU_W.replace(K=16)

        stages = 4 # stages of spatial resolution reduction
        base_channels = int(64*m)
        channels = 3        
        # layers_stage = np.array([2,1,1,1], dtype=int) # ResNet18 layout
        chmult_stage = np.array([1,2,4,8], dtype=int) # stage 1 replaces maxpool
        groups_stage = np.array([1,1,1,1], dtype=int)
        bits_stage   = (1,1,1,1)
        regtower_order = np.array([0,3,3,3], dtype=int)
        if DB: # double bits
            bits_stage = (1,2,2,2)
        if DC: # double channel
            chmult_stage[1:] *=2
            groups_stage[1:] *=2

        if gate == 'Cat' or gate == 'none':
            Block = CatBlock
            block_args = [dict()] * 4
        elif gate == 'BiNeal':
            Block = BiNealBlock
            block_args = [dict()] * 4
        elif gate == 'Normal':
            Block = NormalBlock
            block_args = [dict()] * 4
            if layers is not None:
                layers_stage = np.array(layers, dtype=int) # these are extra layers per stages (1,2,3,4): (B0 + l0) + (N + l1) + (N + l2) + (N + l3) + classifier
                tower_order = np.array([0,0,0,0], dtype=int)
        elif gate == 'Tower':
            Block = TowerBlock
            block_args = [dict()] * 4
            if layers is not None:
                layers_stage = np.array(layers, dtype=int) # these are extra layers per stages (1,2,3,4): (B0 + l0) + (N + l1) + (N + l2) + (N + l3) + classifier
            tower_order = np.array([order,order,order,order], dtype=int)
            regtower_order = np.array([order,order,order,order], dtype=int)
            print(f'layers_stage ={layers_stage}')
        elif gate == 'Tower2':
            Block = TowerBlock
            layers_stage = np.array([0,0,0,0], dtype=int)
            tower_order = np.array([0,2,2,2], dtype=int)
        elif gate == 'Tower3':
            Block = TowerBlock
            layers_stage = np.array([1,1,1,0], dtype=int) # standard Tower3 for all common experiments (extra inner layers)
            # layers_stage = np.array([0,0,0,0], dtype=int) # alternatives
            # layers_stage = np.array([1,1,1,1], dtype=int) # 
            tower_order = np.array([0,3,3,3], dtype=int)
        elif gate == 'Tower4':
            Block = TowerBlock
            layers_stage = np.array([0,1,1,0], dtype=int)
            # layers_stage = np.array([0,0,0,0], dtype=int)
            tower_order = np.array([0,4,4,4], dtype=int)
        elif gate == 'Tower5':
            Block = TowerBlock
            layers_stage = np.array([0,0,0,0], dtype=int)
            tower_order = np.array([0,5,5,3], dtype=int)
            base_channels = int(128*m)
            chmult_stage = np.array([1,1,2,4], dtype=int) # stage 1 replaces maxpool

        elif gate == 'Tower5a':
            Block = TowerBlock
            layers_stage = np.array([0,1,1,0], dtype=int)
            tower_order = np.array([0,5,5,3], dtype=int)
            base_channels = int(128*m)
            chmult_stage = np.array([1,1,2,4], dtype=int) # stage 1 replaces maxpool

        elif gate == 'Tower5b': # 24M params -- too heavy layer at 512 features
            Block = TowerBlock
            layers_stage = np.array([0,1,1,1], dtype=int)
            tower_order = np.array([0,5,5,3], dtype=int)
            regtower_order = np.array([0,3,3,1], dtype=int)
            base_channels = int(128*m)
            chmult_stage = np.array([1,1,2,4], dtype=int) # stage 1 replaces maxpool

        elif gate == 'Tower8':
            Block = TowerBlock
            layers_stage = np.array([0,0,0,0], dtype=int)
            tower_order = np.array([0,5,8,6], dtype=int)
            regtower_order = np.array([0,0,0,1], dtype=int)
            base_channels = int(128*m)
            chmult_stage = np.array([1,1,2,4], dtype=int) # stage 1 replaces maxpool
        elif gate == 'Tower8s64': # Use 64 base channels, no post layer
            Block = TowerBlock
            layers_stage = np.array([0,0,0,0], dtype=int)
            tower_order = np.array([0,3,8,6], dtype=int)
            regtower_order = np.array([0,0,0,0], dtype=int)
            base_channels = int(64*m)
            chmult_stage = np.array([1,2,4,8], dtype=int) # stage 1 replaces maxpool            
        elif gate == 'Tower8s': # Almost the same as T8
            Block = TowerBlock
            block_args = [dict()] * 4
            layers_stage = np.array([0,0,0,0], dtype=int)
            tower_order = np.array([0,3,8,6], dtype=int)
            regtower_order = np.array([0,0,0,1], dtype=int)
            base_channels = int(128*m)
            # chmult_stage = np.array([1,1,2,4], dtype=int) # stage 1 replaces maxpool
            chmult_stage = np.array([1,1,ch_grow,ch_grow**2], dtype=float) # stage 1 replaces maxpool
            # Block0 = Block0_b0 # no improvemnt but costs one more conv and a feature map
        elif gate == 'Tower8s2': # 11.91M params. Just slightly worse than T8s
            Block = TowerBlock
            layers_stage = np.array([0,0,0,0], dtype=int)
            tower_order = np.array([0,2,8,6], dtype=int)
            regtower_order = np.array([0,0,0,1], dtype=int)
            base_channels = int(128*m)
            chmult_stage = np.array([1,1,2,4], dtype=int) # stage 1 replaces maxpool

        elif gate == 'Tower8s1': # This is noticibly worse than Tower82
            Block = TowerBlock
            layers_stage = np.array([0,0,0,0], dtype=int)
            tower_order = np.array([0,1,8,6], dtype=int)
            regtower_order = np.array([0,0,0,1], dtype=int)
            base_channels = int(128*m)
            chmult_stage = np.array([1,1,2,4], dtype=int) # stage 1 replaces maxpool

        elif gate == 'Tower20':
            Block = TowerBlock
            order = 20
            base_channels = int(128*m)
            layers_stage = np.array([0,0,0,0], dtype=int)
            tower_order = np.array([0,order,order,order], dtype=int)

        elif gate == 'Fusion8s': # Almost the same as T8
            Block = FusedResBlock
            block_args = [dict()] * 4
            layers_stage = np.array([0,0,0,0], dtype=int)
            tower_order = np.array([0,3,8,6], dtype=int)
            regtower_order = np.array([0,0,0,1], dtype=int)
            base_channels = int(128*m)
            chmult_stage = np.array([1,1,2,4], dtype=int) # stage 1 replaces maxpool

        elif gate == 'GenRes8s': # Almost the same as T8
            Block = GenResBlock
            block_args = [dict()] * 4
            layers_stage = np.array([0,0,0,0], dtype=int)
            tower_order = np.array([0,3,8,6], dtype=int)
            regtower_order = np.array([0,0,0,1], dtype=int)
            base_channels = int(128*m)
            chmult_stage = np.array([1,1,ch_grow,ch_grow**2], dtype=float) # stage 1 replaces maxpool

        elif gate in GATE_BUILDERS: # experimental / future gates registered from arch_experiments.py
            _cfg = GATE_BUILDERS[gate](dotdict(o=o, m=m, order=order, ch_grow=ch_grow,
                                               base_channels=base_channels,
                                               chmult_stage=chmult_stage,
                                               regtower_order=regtower_order))
            Block         = _cfg['Block']
            block_args    = _cfg.get('block_args', [dict()] * 4)
            layers_stage  = _cfg['layers_stage']
            tower_order   = _cfg['tower_order']
            regtower_order = _cfg.get('regtower_order', regtower_order)
            base_channels = _cfg.get('base_channels', base_channels)
            chmult_stage  = _cfg.get('chmult_stage', chmult_stage)

        else:
            raise ValueError(f"unknown gate {gate!r}; available paper gates plus registered: {sorted(GATE_BUILDERS)}")


        dilation = 1

        ll = []
        stage_in_bits = None
        for stage in range(stages):
            stage_in_channels = channels
            stage_ll = []            
            groups = groups_stage[stage]
            # order = tower_order[i]
            next_channels = int(base_channels * chmult_stage[stage])
            if o0.A == 2: # applies in case we start with binary
                bits = bits_stage[stage]
                next_o.A = 2**bits
                next_o.QReLU_A = next_o.QReLU_A.replace(K=next_o.A) # Activation states
            else:
                bits = math.log2(o0.A)
            if stage==0: # first reduction from image to qunatized resolution x 1/4
                stage_ll += [Block0(o1, next_o, next_channels)]
            else:
                stride_to_dilation = (stage > 1 and Dilation)
                stage_ll += [Block(channels, next_channels, o, next_o=next_o, stride=1 if stride_to_dilation else 2, dilation=dilation, groups = groups, order=tower_order[stage], **(block_args[stage]))] # reduction block
                if stride_to_dilation:
                    dilation *= 2
            channels = next_channels
            o = copy.deepcopy(next_o)
            # add a number of regular blocks after the reduction block
            # print(f"Adding {stage}:{layers_stage[stage]} order= {regtower_order[stage]}")
            # DW (double width): inner (stride-1) blocks of a stage run at 2x channels.
            # The first inner block expands C->2C, the rest stay 2C->2C, so the stage outputs 2C;
            # the next stage's reduction conv (kept at standard output) compresses 2C->C_next.
            inner_channels = channels * 2 if DW else channels
            for l in range(layers_stage[stage]):
                stage_ll += [Block(channels, inner_channels, o, stride=1, dilation = dilation, groups=groups, order = regtower_order[stage], **block_args[stage])]
                channels = inner_channels
                #
            l = ESequential(*stage_ll)
            l.info = dotdict(stage=stage, in_channels = stage_in_channels, out_channels = channels, in_bits = stage_in_bits, out_bits = bits, dilation = dilation)
            stage_in_bits = bits
            ll += [l]
        # if print_info:
        #     print(f'classifier ({channels}->{o.num_classes})')
        # Classifer: [Conv 1x1, AvgPool] is the same as [AvgPool, Conv 1x1] but the second version requires less ops, even considering bits
        stage_in_channels = channels
        classifier_ll = []
        classifier = 'c3'
        pre_cls_channels = int(m*pre_cls_dim)
        # classifier = 'c5-max'
        if classifier == 'c2': # c2 n0 (no norm, init_olny norm for scaling, learnable scaling)
            classifier_ll += [nn.AdaptiveAvgPool2d((1,1))] # requires adaptive number of bits, i.e. 6 bits to represent sum of 7x7 window with binary activations -- but can swap after training
            classifier_ll += [QConv2d(o1, channels, o.num_classes, kernel_size=(1, 1), stride=1, bias=False, norm = False)] # not a bottleneck compared to prev layer
            # classifier_ll += [make_norm(o.num_classes, learnable=True)]
            classifier_ll += [make_norm(o.num_classes, learnable=True, init_only=True)]
            classifier_ll += [ScaleBias(weight = 0.1, bias=None, learnable=False)]

        elif classifier == 'c3': # '-c3-n' (with active BN) '-c3' passive norm
            classifier_ll += [Tuple0()] # hacking to allow SE model
            classifier_ll += [normal_block(channels, pre_cls_channels, o, kernel_size=1)] # expand channels, actually don't have to quantize before pooling, it can be fused
            classifier_ll += [Tuple0()] # hacking to allow ST-WX
            classifier_ll += [Distribution2ST()] # hacking to allow MeanSample
            classifier_ll += [nn.AdaptiveAvgPool2d((1,1))] # requires adaptive number of bits, i.e. 6 bits to represent sum of 7x7 window with binary activations -- but can swap after training
            classifier_ll += [QConv2d(o1, pre_cls_channels, o.num_classes, kernel_size=(1, 1), stride=1, bias=False, norm = False)] # not a bottleneck compared to prev layer
            classifier_ll += [make_norm(o.num_classes, learnable=True)] # learnable affine TODO: this is odd, to made BN channelwise on 1x1 resolution, even not all classes are represented...
            classifier_ll += [ScaleBias(weight = 0.1, bias=None, learnable=False)] # scalar SB

        elif classifier == 'c3-n': # '-c3-n' (with active BN) '-c3' passive norm
            classifier_ll += [Tuple0()] # hacking to allow SE model
            classifier_ll += [normal_block(channels, pre_cls_channels, o, kernel_size=1)] # expand channels, actually don't have to quantize before pooling, it can be fused
            classifier_ll += [Tuple0()] # hacking to allow ST-WX
            classifier_ll += [Distribution2ST()] # hacking to allow MeanSample
            classifier_ll += [nn.AdaptiveAvgPool2d((1,1))] # requires adaptive number of bits, i.e. 6 bits to represent sum of 7x7 window with binary activations -- but can swap after training
            classifier_ll += [QConv2d(o1, pre_cls_channels, o.num_classes, kernel_size=(1, 1), stride=1, bias=False, norm = False)] # not a bottleneck compared to prev layer
            classifier_ll += [make_norm(o.num_classes, learnable=True, init_only=True)]
            classifier_ll += [ScaleBias(weight = 0.1, bias=None, learnable=False)] # scalar SB

        elif classifier == 'c5': # 'c5' # experimental
            classifier_ll += [QConv2d(o, channels, pre_cls_channels, kernel_size=1, groups=groups, padding_mode=padding_mode, bias=False, convex_comb=False, norm=False)]
            classifier_ll += [Tuple0()] # hacking to allow ST-WX
            classifier_ll += [nn.AdaptiveAvgPool2d((1,1))] # requires adaptive number of bits, i.e. 6 bits to represent sum of 7x7 window with binary activations -- but can swap after training
            classifier_ll += [make_norm(pre_cls_channels, learnable=True)]
            classifier_ll += [QReLU(o.QReLU_A.replace(K=16), channels=pre_cls_channels)]
            classifier_ll += [QConv2d(o1, pre_cls_channels, o.num_classes, kernel_size=(1, 1), stride=1, bias=False, norm = False)] # not a bottleneck compared to prev layer
            classifier_ll += [make_norm(o.num_classes, learnable=True)] # learnable affine
            classifier_ll += [ScaleBias(weight = 0.1, bias=None, learnable=False)] # scalar SB

        elif classifier == 'c5-relu':
            classifier_ll += [QConv2d(o, channels, pre_cls_channels, kernel_size=1, groups=groups, padding_mode=padding_mode, bias=False, convex_comb=False, norm=False)]
            classifier_ll += [make_norm(2048, learnable=True)]
            classifier_ll += [nn.ReLU()]
            classifier_ll += [nn.AdaptiveAvgPool2d((1,1))]
            classifier_ll += [make_norm(pre_cls_channels, learnable=True, init_only=True)] # init_olny was necessary to prevent NaN
            classifier_ll += [QReLU(o.QReLU_A.replace(K=16), channels=pre_cls_channels)]
            classifier_ll += [QConv2d(o1, pre_cls_channels, o.num_classes, kernel_size=(1, 1), stride=1, bias=False, norm = False)] # not a bottleneck compared to prev layer
            classifier_ll += [make_norm(o.num_classes, learnable=True)] # learnable affine
            classifier_ll += [ScaleBias(weight = 0.1, bias=None, learnable=False)] # scalar SB

        elif classifier == 'c5-relu2':
            classifier_ll += [QConv2d(o, channels, pre_cls_channels, kernel_size=1, groups=groups, padding_mode=padding_mode, bias=False, convex_comb=False, norm=False)]
            classifier_ll += [make_norm(pre_cls_channels, learnable=True)]
            classifier_ll += [nn.ReLU()]
            classifier_ll += [nn.AdaptiveAvgPool2d((1,1))]
            classifier_ll += [make_norm(pre_cls_channels, learnable=True, init_only=True)]
            classifier_ll += [nn.ReLU()]
            classifier_ll += [QConv2d(o1, pre_cls_channels, o.num_classes, kernel_size=(1, 1), stride=1, bias=False, norm = False)] # not a bottleneck compared to prev layer
            classifier_ll += [make_norm(o.num_classes, learnable=True)] # learnable affine
            classifier_ll += [ScaleBias(weight = 0.1, bias=None, learnable=False)] # scalar SB

        elif classifier == 'c5-relu1':
            classifier_ll += [QConv2d(o, channels, pre_cls_channels, kernel_size=1, groups=groups, padding_mode=padding_mode, bias=False, convex_comb=False, norm=False)]
            classifier_ll += [make_norm(2048, learnable=True)]
            classifier_ll += [nn.ReLU()]
            classifier_ll += [nn.AdaptiveAvgPool2d((1,1))]
            classifier_ll += [QConv2d(o1, pre_cls_channels, o.num_classes, kernel_size=(1, 1), stride=1, bias=False, norm = False)] # not a bottleneck compared to prev layer
            classifier_ll += [make_norm(o.num_classes, learnable=True)] # learnable affine
            classifier_ll += [ScaleBias(weight = 0.1, bias=None, learnable=False)] # scalar SB

        elif classifier == 'c5-max': # 'c5' # experimental
            classifier_ll += [QConv2d(o, channels, pre_cls_channels, kernel_size=1, groups=groups, padding_mode=padding_mode, bias=False, convex_comb=False, norm=False)]
            classifier_ll += [Tuple0()] # hacking to allow ST-WX
            classifier_ll += [nn.AdaptiveMaxPool2d((1,1))] # requires adaptive number of bits, i.e. 6 bits to represent sum of 7x7 window with binary activations -- but can swap after training
            classifier_ll += [make_norm(pre_cls_channels, learnable=True)]
            classifier_ll += [QReLU(o.QReLU_A.replace(K=16), channels=pre_cls_channels)]
            classifier_ll += [QConv2d(o1, pre_cls_channels, o.num_classes, kernel_size=(1, 1), stride=1, bias=False, norm = False)] # not a bottleneck compared to prev layer
            classifier_ll += [make_norm(o.num_classes, learnable=True)] # learnable affine
            classifier_ll += [ScaleBias(weight = 0.1, bias=None, learnable=False)] # scalar SB                        

            # # c4
            # classifier_ll += [normal_block(channels, 2048, o, kernel_size=1)] # expand channels
            # classifier_ll += [nn.AdaptiveAvgPool2d((1,1))] # requires adaptive number of bits, i.e. 6 bits to represent sum of 7x7 window with binary activations -- but can swap after training
            # classifier_ll += [QConv2d(o1, 2048, o.num_classes, kernel_size=(1, 1), stride=1, bias=False, norm = False)] # not a bottleneck compared to prev layer
            # classifier_ll += [make_norm(o.num_classes, learnable=False, init_only=True)]
            # classifier_ll += [ScaleBias(channels=o.num_classes, weight = 0.1, bias=0.0, learnable=True)]

        elif False: # c0 this was with BN inside QConv2d, unintentionally but appears to work better
            # binary weights should be ok here -- lots of inputs
            classifier_ll += [QConv2d(o, channels, o.num_classes, kernel_size=(1, 1), stride=1, bias=False, padding = 0, norm = True)] # todo: this has unintentinal norm
            classifier_ll += [make_norm(o.num_classes, learnable=True, init_only=True)]
            classifier_ll += [ScaleBias(weight = 0.1, bias=None, learnable=False)]
            classifier_ll += [nn.AdaptiveAvgPool2d((1,1))]
        else: # c1
            classifier_ll += [normal_block(channels, pre_cls_channels, o, kernel_size=1)] # expand channels
            classifier_ll += [QConv2d(o, pre_cls_channels, o.num_classes, kernel_size=(1, 1), stride=1, bias=False, padding = 0, norm=False)]
            classifier_ll += [make_norm(o.num_classes, learnable=True, init_only=True)]
            classifier_ll += [ScaleBias(weight = 0.1, bias=None, learnable=False)]
            classifier_ll += [nn.AdaptiveAvgPool2d((1,1))]
        classifier_ll += [nn.Flatten()]
        l = ESequential(*classifier_ll)
        l.info = dotdict(stage=stages, in_channels = stage_in_channels, out_channels = o.num_classes)
        ll += [l]
        super().__init__(*ll)
        self.name = str(self.__class__.__name__) + str(dict(gate=gate, DB=DB, DC=DC))
        # if print_info:
        #     total_params = sum(p.numel() for p in self.parameters())
        #     print(f'Parameters: {total_params/10**6:3.2f}M")
        # init convex comb coefficients according to depth (naive approach, in the order of cinstruction)
        mm = [(n,l) for (n,l) in self.named_modules() if isinstance(l, QConv2d) and l.convex_comb]
        for i,(n,l) in enumerate(mm):
            depth = i / (len(mm)-1)
            alpha = 0.3 #0.3*(1-depth) + 0.8*depth
            l.convex_w.data = torch.logit(torch.tensor(alpha)) 
            # if print_info:
            #     print(n, end='\t')
            #     print(i, l)
            i = i +1

        if print_info:
            self.print_info()
            # print(self)

    def print_info(self):
        with verbosity(1):
            print(self.name)
            for (stage,l) in enumerate(self):
                if stage < len(self)-1:
                    print(f'stage {stage}: channels ({l.info.in_channels}->{l.info.out_channels})\t out_bits {l.info.out_bits} dilation = {l.info.dilation}')
                    # print(f'\t' + ','.join(r.__class__.__name__ for r in l))
                    print(f'\t' + ','.join(str(r) for r in l))
                else:
                    print(f'stage {stage} -- classifier: channels ({l.info.in_channels}->{l.info.out_channels})')
            total_params = sum(p.numel() for p in self.parameters())
            print(f'Parameters: {total_params/10**6:3.2f}M')

    # def net_loss(self, input, targets, method):
    #     loss, scores = super().net_loss(input,targets, method)
    #     reg_loss = 0
    #     for m in self.modules():
    #         # if isinstance(m, QConv2d):
    #         #     reg_loss = reg_loss + m.reg() * 1e-7
    #         if isinstance(m, Quant) and m.is_activation:
    #             reg_loss = reg_loss + m.reg_loss * 1e-8
    #     loss = loss + reg_loss
    #     return loss, scores
    

    # def forward_features(self, x:Tensor, stages = (0, -2), method = None, **kwargs):
    #     """
    #     Propagate forward and compute features at the output of the specified stages:
    #     stage = 0 is a resolutino reduction to 1/2^2
    #     stage = -2 is the last stage spatially arranged features
    #         - 1/2^5 resolution if using stride
    #         - 1/2^3 resolution if using stride + (False, True, True) dilation config
    #     stage = -1 -- scores of the classifer
    #     """
    #     features = []
    #     stages +=  tuple(s + len(self) for s in stages)
    #     for stage, layer in enumerate(self):
    #         x = method.dispatch(layer, x , **kwargs)
    #         if stage in stages:
    #             features += [x]
    #     return features
    
    # def net_loss(self, input, targets, method):
    #     f0, f1, scores = self.forward_features(input, stages= (0,-1,-2,), method =method)
    #     # print(f0.shape, f1.shape)
    #     loss = nn.CrossEntropyLoss(reduction='none').forward(scores, target=targets)
    #     return loss, scores


# def teacher_head(stage):
#     net = models.resnet18(pretrained=True)
#     for param in net.parameters():
#         param.requires_grad = False
#     classifier = torch.nn.Sequential([net.avgpool, nn.Flatten(), net.fc])
#     stage0 = torch.nn.Sequential([net.conv1, net.bn1, net.relu, net.maxpool])
#     net_seq = torch.nn.Sequential([stage0, net.layer1, net.layer2, net.layer3, net.layer4, classifier])
#     head = net_seq[4:]
#     return head



class ResNet18(EClassificationNet):
    class Block(KWLayer):
        def __init__(self, in_features, out_features, o, next_o=None, stride=2, groups = 1, kernel_size = 3, skip_kernel_size=1, gate_kind='none', dilation = 1):
            super().__init__()
            if next_o is None:
                next_o = o
            self.normal = ESequential([normal_block(in_features, out_features, o, stride = stride, groups=groups, kernel_size=kernel_size),
                                    QConv2d(next_o, out_features, out_features, kernel_size=kernel_size, stride= 1, padding=kernel_size//2, groups=groups, padding_mode=padding_mode, bias=False, norm=False),
                                    make_norm(out_features, learnable=True, init_only=False), # like normal block without the activation
                                    ])
            kernel_size = 1
            if out_features == in_features and stride ==1:
                self.skip = Identity()
            else:
                self.skip =  ESequential([QConv2d(next_o, in_features, out_features, kernel_size=skip_kernel_size, stride = stride, padding=skip_kernel_size//2, groups=groups, padding_mode=padding_mode, bias=False, norm=False),
                                    make_norm(out_features, learnable=True, init_only=False),
                                    ])
            self.quantizer = ESequential([QReLU(next_o.QReLU_A, channels=out_features)]) # could add one more BN make_norm(out_features, learnable=True)

        def forward(self, x, **kwargs):
            y = self.normal.forward(x, **kwargs)
            x = self.skip.forward(x, **kwargs)
            r = x + y
            assert(kwargs['method'] is not None)
            qr = self.quantizer.forward(r, **kwargs)
            return qr

    def __init__(self, o, m=1, print_info=False, Dilation=False, skip_kernel=1, **kwargs):
        super().__init__()
        o = copy.deepcopy(o)
        o1 = copy.deepcopy(o) # first layer
        ll = []
        # o1.QReLU_W = o1.QReLU_W.replace(K=256)
        groups = 1
        dilation = 1
        if Dilation:
            raise AttributeError('Not implemented yet')
            stride_reduce = 1
        else:
            stride_reduce = 2
        Block = ResNet18.Block
        # stage 0
        ll += [Block0_resnet(o1, o, int(64*m))]
        # layer1 of ResNet
        ll += [Block(int(64*m), int(64*m), o, stride=1,  dilation = dilation, groups=groups, skip_kernel_size=skip_kernel)]
        ll += [Block(int(64*m), int(64*m), o, stride=1,  dilation = dilation, groups=groups, skip_kernel_size=skip_kernel)]
        # Steg 1
        ll += [Block(int(64*m), int(128*m), o, stride=2, dilation = dilation, groups=groups, skip_kernel_size=skip_kernel)] # expand on tage start
        ll += [Block(int(128*m), int(128*m), o, stride=1, dilation = dilation, groups=groups, skip_kernel_size=skip_kernel)]
        # Steg 2
        ll += [Block(int(128*m), int(256*m), o, stride=2, dilation = dilation, groups=groups, skip_kernel_size=skip_kernel)] # expand on tage start
        ll += [Block(int(256*m), int(256*m), o, stride=1, dilation = dilation, groups=groups, skip_kernel_size=skip_kernel)]
        # Steg 3
        ll += [Block(int(256*m), int(512*m), o, stride=2, dilation = dilation, groups=groups, skip_kernel_size=skip_kernel)] # expand on tage start
        ll += [Block(int(512*m), int(512*m), o, stride=1, dilation = dilation, groups=groups, skip_kernel_size=skip_kernel)]
        # Classifer:
        if True: # ResNet classifier
            ll += [nn.AdaptiveAvgPool2d((1,1))] # "requires adaptive number of bits, i.e. 6 bits to represent sum of 7x7 window with binary activations"
            # ll += [make_norm(int(512*m), learnable=True, init_only=False)] # learnable affine
            ll += [nn.Conv2d(in_channels=int(512*m), out_channels=o.num_classes, kernel_size=1, bias=True)]
            # ll += [make_norm(o.num_classes, learnable=True)] # learnable affine
            ll += [nn.Flatten()]
        super().__init__(*ll)
        # print(self)


class MobileNetv1(EClassificationNet, ESequential):

    @classmethod
    def block(cls, in_channels:int, out_channels:int, o, next_o=None, stride=2, kernel_size = 3, **kwargs):
        o1 = copy.deepcopy(o)
        o1.QReLU_W = o1.QReLU_W.replace(K=256)
        conv_dw = QConv2d(o1,in_channels=in_channels,out_channels=in_channels,stride=stride, kernel_size=kernel_size, groups=in_channels,padding=kernel_size//2)
        o1 = copy.deepcopy(o)
        o1.QReLU_W = o1.QReLU_W.replace(K=16)
        conv = QConv2d(o1,in_channels=in_channels,out_channels=out_channels,stride=1, groups=1,kernel_size=1)
        return ESequential([conv_dw, QReLU(o.QReLU_A, channels=in_channels), conv, QReLU(o.QReLU_A, channels=out_channels)])

    def __init__(self, o, m=1, print_info=False, Dilation=False, skip_kernel=1, **kwargs):
        super().__init__()
        # stage 0: 224x224
        ll = []
        ll += [QConv2d(o, 3, int(32*m), kernel_size=3, stride= 2, padding=3//2, padding_mode=padding_mode, bias=False, norm = False)]
        # layer1: 112x112
        ll += [MobileNetv1.block(int(32*m), int(64*m),o, stride=1)]
        ll += [MobileNetv1.block(int(64*m), int(128*m),o, stride=2)]
        # Steg 1: 56x56
        ll += [MobileNetv1.block(int(128*m), int(128*m),o, stride=1)]
        ll += [MobileNetv1.block(int(128*m), int(256*m),o, stride=2)]
        # Steg 2: 28x28
        ll += [MobileNetv1.block(int(256*m), int(256*m),o, stride=1)]
        ll += [MobileNetv1.block(int(256*m), int(512*m),o, stride=2)]
        # Steg 3: 14x14
        for l in range(5):
            ll += [MobileNetv1.block(int(512*m), int(512*m),o, stride=1)]
        ll += [MobileNetv1.block(int(512*m), int(1024*m),o, stride=2)]
        #: 7x7
        ll += [MobileNetv1.block(int(1024*m), int(1024*m),o, stride=1)]
        ll += [MobileNetv1.block(int(1024*m), int(1024*m),o, stride=1)]
        # Classifer: 7x7
        ll += [nn.AdaptiveAvgPool2d((1,1))] # "requires adaptive number of bits, i.e. 6 bits to represent sum of 7x7 window with binary activations"
        ll += [nn.Conv2d(in_channels=int(1024*m), out_channels=o.num_classes, kernel_size=1, bias=True)]
        ll += [nn.Flatten()]
        super().__init__(*ll)
        # print(self)

def freeze_net(net):
    for param in net.parameters():
        param.requires_grad = False
    net.eval()

def seqence_net(net):
    if isinstance(net, models.ResNet):
        # classifier = ESequential([net.avgpool, nn.Flatten(), net.fc, make_norm(net.fc.weight.shape[0], learnable=True)])
        classifier = ESequential([net.avgpool, nn.Flatten(), net.fc])
        stage0 = ESequential([net.conv1, net.bn1, net.relu, net.maxpool, net.layer1])
        net_seq = ESequential([stage0, net.layer2, net.layer3, net.layer4, classifier])
        return net_seq
    else:
        raise RuntimeError("do not know how")

class EvalESequential(ESequential):
    def train(self, mode=True):
        ESequential.train(self, False)

def cross_net(net1, net2, stage, pretrained = True):
    # find the first linear layer in net2[stage] and make it trainable
    for m in net2[stage].modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            print('Making layer learnable:', m)
            d = m.weight.shape[1] # 
            # make trainable and re-inint
            if False:
                m.weight.requires_grad = True
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.requires_grad = True
                    nn.init.zeros_(m.bias)
                adaptor = []
                # TODO: issue: the skip connection may be downsampling and in this case it also needs to be learnable
            else:
                adaptor = [ESequential([torch.nn.Conv2d(d, d, 3, padding=1, bias=False), make_norm(num_features=d, learnable=True), torch.nn.LeakyReLU()])]
            break # stop after finding the first linear layer!
    if False: # make classifier learnable and reinit
        fc = net2[-1][-1]
        fc.weight.requires_grad = True
        fc.bias.requires_grad = True
        nn.init.kaiming_uniform_(fc.weight, mode='fan_in', nonlinearity='relu')
        # fc.weight.data *= 0.1
        nn.init.zeros_(fc.bias)
    if pretrained:
        head = EvalESequential([net2[k] for k in range(stage, len(net2))])
        net = EClassificationNet([net1[k] for k in range(stage)] + adaptor + [head])
    else:
        head = ESequential([net2[k] for k in range(stage, len(net2))])
        net = EClassificationNet([net1[k] for k in range(stage)] + [head]) # don't need adaptor
    return net

def teacher_head(net, stage = 4, subclasses = None, pretrained = True):
    if pretrained:
        teacher = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        freeze_net(teacher)
    else:
        teacher = models.resnet18()
    if subclasses is not None:
        teacher.fc.weight.data = teacher.fc.weight[subclasses]
        teacher.fc.bias.data = teacher.fc.bias[subclasses]
        teacher.fc.out_features = len(subclasses)
    teacher = seqence_net(teacher)
    cn = cross_net(net, teacher, stage=stage, pretrained = pretrained)
    cn.to(dev)
    return cn

def create_net(o, print_info=False):
    # print(o)
    print(o.net_name, o.net_args)
    torch.manual_seed(o.seed) # network initialization seed
    if o.net_name == 'resnet18':
        net = Wrapped(models.resnet18(weights=None, num_classes = o.num_classes)).to(dev)
    elif o.net_name == 'resnet50':
        net = Wrapped(models.resnet50(weights=None, num_classes = o.num_classes)).to(dev)
    elif o.net_name == 'QResNet':
        net = QResNet(o, **o.net_args, print_info=print_info).to(dev)
    elif o.net_name == 'BiNealNet':
        net = BiNealNet(o, **o.net_args, print_info=print_info).to(dev)        
    elif o.net_name == 'QResNet18':
        net = QResNet18(o, **o.net_args, print_info=print_info).to(dev)
    elif o.net_name in NET_BUILDERS:
        net = NET_BUILDERS[o.net_name](o, **o.net_args, print_info=print_info).to(dev)
    else:
        cls = globals()[o.net_name]
        net = cls(o, **o.net_args, print_info=print_info).to(dev)
    # print(net)
    if o.t_stage > 0:
        net = teacher_head(net, o.t_stage-1, subclasses = o.subclasses, pretrained=False)
        print(net)

    # unique set of module types
    mm = {}
    params = {}
    for m in net.modules():
        t = type(m)
        if t not in mm:
            mm[t] = m
            params[t] = {}
        for n,p in m.named_parameters(recurse=False):
            if n not in params[t]:
                params[t][n] = p

    for t,m in mm.items():
        if params[t] != {}:
            print(f'{t.__name__}:')
            for n,p in params[t].items():
                print(f'\t{n}')
        
        

    # if o.fp16:
    #     pass
    #     net.half()
    #     # for m in net.modules():
    #     #     if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
    #     #         m.float()

    return net


# ── Auto-load experimental architectures (working repo only) ──────────────────
# `arch_experiments.py` registers the future/experimental gates (Squeeze-
# Excitation, attention) into GATE_BUILDERS. It is intentionally absent from the
# public release, so this import is optional: the paper architectures work
# without it; only the experimental gates become unavailable.
try:
    from . import arch_experiments  # noqa: F401  (registers extra gates on import)
except ImportError:
    pass
