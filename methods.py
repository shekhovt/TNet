# SPDX-License-Identifier: CC BY-NC-SA 4.0
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. Making code open source is essential for scientific
# transparency and reproducibility. Providing access to the code allows researchers to
# verify, replicate, and expand upon results, which supports scientific progress. Current
# top-tier conferences encourage opensource code for the purpose of reproducibility that
# is de-facto an established expectation of the research community. Opensource code as
# part of publication should not affect the performance of an invention both from a
# commercial and IP point of view.
import relimport

from torch.nn.modules.batchnorm import _BatchNorm
#_______________________________

# from . import func_PP as F_PP
from .layers import *
from .categorical import Categorical
from . import categorical
from .functional import *
from . import functional
from .data_profiler import DataProfiler
from .inductor import *

#____________________Base Method Class_________________



# @torch.compile(**compile_args)
# def compiled_QReLU(method, layer, x, **kwargs):
#     return method.forward_QReLU(layer, x, **kwargs)

def dataprofiler_pre_hook(layer, method, inputs, **kwargs):
    if (DataProfiler.enabled):
        return DataProfiler.logPreForward (layer, method, inputs[0], *inputs, **kwargs)

def dataprofiler_post_hook(layer, method, inputs, outputs, **kwargs):
    if (DataProfiler.enabled):
        return DataProfiler.logPostForward (layer, method, inputs[0], outputs, *inputs, **kwargs)

class Method():
    """ Base class for different propagation methods"""
    # class Options(dotdict):
    #     def __init__(self, **kwargs):
    #         self.q_noise_type='uniform'
    #         self.q_noise_sigma=None
    #         self.q_noise_sigma_learnable=False
    #         self.NS=1.0
    #         for (k,v) in kwargs:
    #             if k in self.keys():
    #                 self[k] = v
    #             else:
    #                 raise AttributeError(f'Unrecognized option {k}={v}')
    
    def __init__(self, o: dotdict = None):
        """ different methods might have different options passed as dotdict"""
        if o == None:
            self.o = dotdict()
        else:
            self.o = o
        if not hasattr(self.o,'compile'):
            self.o.compile = False
        if not hasattr(self.o,'dynamic_dims'):
            self.o.dynamic_dims = True
        if not hasattr(self.o,'MD'):
            self.o.MD = False
        if not hasattr(self.o,'WX'):
            self.o.WX = False
        self.pre_hook = None
        self.post_hook = None
        self.set_default_hooks()
        # if True:
            # self.compiled_QRELU = torch.compile(self.forward_QReLU, fullgraph=True, dynamic=True, backend = "inductor", options={'group_fusion':True, 'force_same_precision':True, 'disable_cpp_codegen':False, 'trace.graph_diagram':True, "triton.cudagraphs": False})
        # else:
            # self.compiled_QRELU = self.forward_QReLU
            
    def set_default_hooks(self):
        self.pre_hook = dataprofiler_pre_hook # set the method but don't bind the DataProfiler object
        self.post_hook = dataprofiler_post_hook

    def dispatch(self, layer, *args, **kwargs):
        # dispatch the forward to a specialized implementation by the class name

        res = None
        if self.pre_hook is not None:
            self.pre_hook(layer, self, args, **kwargs)
        # if (DataProfiler.enabled):
        #     DataProfiler.logPreForward (layer, self, args [0], args, kwargs)

        if isinstance(layer, Identity):
            res = self.forward_Identity(layer, *args, **kwargs)
        elif isinstance(layer,Quant):
            res = self.forward_Quant(layer, *args, **kwargs)
        elif isinstance(layer,Categorical):
            res = self.forward_Categorical(layer, *args, **kwargs)        
        elif isinstance(layer, QReLU):
            res = self.forward_QReLU(layer, *args, **kwargs)
            # return self.forward_QLinear(layer, *args, **kwargs)
        elif isinstance(layer, QLinear):
            res = self.forward_QLinear(layer, *args, **kwargs)
        elif isinstance(layer, QAnyLinear):
            res = self.forward_QAnyLinear(layer, *args, **kwargs)
        elif isinstance(layer, WeightCentering):
            res = self.forward_WeightCentering(layer, *args, **kwargs)
        elif isinstance(layer, Cat):
            res=  self.forward_Cat(layer, *args, **kwargs)
        elif isinstance(layer, Parallel):
            res = self.forward_Parallel(layer, *args, **kwargs)
        elif isinstance(layer, ESequential):
            res = self.forward_ESequential(layer, *args, **kwargs)
        elif isinstance(layer, KWLayer):
            res = layer.forward(*args, method=self, **kwargs)  # call the extended forward with additional **kwargs
        elif isinstance(layer, nn.Module):
            res = layer(*args)  # the call method calls the default forward and forward_hooks
        else:
            res = layer.forward(*args)
        # if (res == 0).all():
        #     raise RuntimeError(f"0 in forward after {str(layer)}?")
        # if not (res == res).all():
        #     raise RuntimeError(f"NAN in forward after {str(layer)}")
        # if (DataProfiler.enabled):
        #     DataProfiler.logPostForward (layer, self, args [0], res, args, kwargs)
        if self.post_hook is not None:
            self.post_hook(layer, self, args, res, **kwargs)
        return res

    def forward_KWLayer(self, layer: KWLayer, x: Tensor, **kwargs) -> Tensor:
        y = layer.forward(x, **kwargs)
        return y

    # @torch.compile(fullgraph=True, dynamic=True, backend = "inductor", options={'group_fusion':True, 'force_same_precision':True, 'disable_cpp_codegen':False, 'trace.graph_diagram':True, "triton.cudagraphs": False})
    # def compiled_QRELU(self, l, x, **kwargs):
        # return self.dispatch(l, x, **kwargs)
    
    @torch.compile(**compile_args)
    def forward_QReLU_compiled(self, layer: QReLU, x: Tensor, **kwargs):
        return self.forward_ESequential(layer, x , **kwargs)

    def forward_QReLU(self, layer: QReLU, x: Tensor, **kwargs):
        if self.o.compile:
            if self.o.dynamic_dims:
                for d in [1,2,3]:
                    torch._dynamo.mark_dynamic(x, d)
                x._dynamo_dynamic_indices = {1,2,3}
            functional.all_checks = False
            # torch._dynamo.mark_dynamic(layer.in_sb.weight,0)
            # torch._dynamo.mark_dynamic(layer.in_sb.bias,0)
            return self.forward_QReLU_compiled(layer, x, **kwargs)
        else:
            return self.forward_ESequential(layer, x , **kwargs)
        # for l in layer:
        #     x = self.dispatch(l, x, **kwargs)
        # return x

    def forward_ESequential(self, layer: ESequential, x: Tensor, **kwargs):
        k = 0
        while k < len(layer):
            l = layer[k]
            # if self.o.compile and isinstance(l, _BatchNorm) and isinstance(layer[k+1], QReLU) and False:
            #     @torch.compile(**compile_args)
            #     def compiled_BNQReLU(x, **kwargs):
            #         x = self.dispatch(l, x, **kwargs)
            #         # x = self.dispatch(layer[k+1], x, **kwargs)
            #         x = self.forward_ESequential(layer[k+1],x , **kwargs)
            #         return x
            #     for d in [1,2,3]:
            #         torch._dynamo.mark_dynamic(x, d)
            #     y = compiled_BNQReLU(x, **kwargs)
            #     k += 2
            # elif self.o.compile and isinstance(l, QReLU):
            #     # @torch.compile(fullgraph=True, dynamic = False, backend = "inductor", options={'group_fusion':True, 'force_same_precision':True, 'disable_cpp_codegen':False, 'trace.graph_diagram':True, "triton.cudagraphs": False})
            #     # def compiled_1(x, **kwargs):
            #         # return self.dispatch(l, x, **kwargs)
            #     y = self.compiled_QRELU(l, x, **kwargs)
            #     k += 1
            # else:
            if isinstance(l, KWLayer) or isinstance(l, QAnyLinear):
                y = l(x, method=self, **kwargs) # use call, allows hooks 
            else:
                y = self.dispatch(l, x, **kwargs) # process without hooks? cannot use call, standard Modules will complain about mehod
            k += 1
            if not self.o.compile and isinstance(x,Tensor):
                if  (x==0).all():
                    raise RuntimeError(f'All zeros after {str(l)}')
                if  not (x==x).all():
                    raise RuntimeError(f'NaN after {str(l)}')
            if isinstance(l, ELoss):
                scores = x  # [M, B]
                return y, scores
            x = y
        return x
    
    # def forward_QReLU(self, layer: QReLU, x: Tensor, **kwargs):
        # for l in (layer):
            # x = self.dispatch(l, x, **kwargs)
        # return x

    def forward_Parallel(self, layer: Parallel, x: Tensor, **kwargs):
        r = 0
        for l in layer:  # now we ar edoing this sequentially, but can do in parallel
            y = self.dispatch(l, x, **kwargs)
            # parse tuple (loss, scores)
            if isinstance(y, tuple):
                y = y[0]
            if layer.reduction == 'sum':
                r += y
        return r

    def forward_Identity(self, layer: Identity, x: Tensor, **kwargs):
        return x

    def forward_Quant(self, layer: Quant, x: Tensor, **kwargs) -> Union[Tensor, RandomVar]:
        """This is to be implemented by specific methods, no common default"""
        raise NotImplementedError()
        

    def forward_QConv2d(self, layer: QConv2d, x: Tensor, **kwargs) -> Union[Tensor, RandomVar]:
        return self.forward_QAnyLinear(layer, x, **kwargs)

    def forward_QLinear(self, layer: QConv2d, x: Tensor, **kwargs) -> Union[Tensor, RandomVar]:
        return self.forward_QAnyLinear(layer, x, **kwargs)
    
    def xw_var(w, w_quantizer:QReLU, K):
        """
        w [c_out, c_in, wy, wx]
        return:
        out_var [c_out, 1, 1, 1] -- vaiance of the output assuming uniform categorical input in [0,..K], per out channel
        """
        eta = w_quantizer[:-1].forward(w) # [sb_in]
        quant = w_quantizer.quant # quant layer
        p = quant.cat_distribution(eta)
        muK2 = ((K-1)/2)**2 # mean activation 
        MK = (K-1)*(2*K-1)/6 # second moment activation
        # becasue activation stats are the same, can sum the weight stats inplace        
        mu2 = (quant.expectation(p)**2).sum(dim=(1,2,3), keepdim=True) # squared mean, summed over input dims
        M = quant.second_moment(p).sum(dim=(1,2,3), keepdim=True) # second moment, summed over input dims
        var = M*MK - mu2*muK2 # formula for variance of product
        return var
      
    def forward_QAnyLinear(self, layer: QAnyLinear, xx, **kwargs) -> Union[Tensor, RandomVar]:
        """Common implementation for determinisitc and Sampling-Based methods"""
        w = layer.weight.view(layer.weight.shape) # view as Tensor, needed for torch.compile to set dynamic shape (workaround, maybe fixed now)
        if self.o.WX and isinstance(xx, tuple):
            x, mx = xx  # WX ST
        else:
            x = mx = xx # input layer

        # Experimental: 
        # C = x.shape[1]
        # # x[:,:C] *= -1 # these are going to be "inhibiting"
        # x[:,:C] *= 2 # make more significant than other
        #            
        assert isinstance(x, Tensor)
        if self.o.compile and self.o.dynamic_dims:
            torch._dynamo.mark_dynamic(w, (0,1,2,3))
        with torch.amp.autocast(device_type="cuda",enabled=False):
            qweight = layer.w_pipeline.forward(w, method=self, **kwargs)
            if self.o.WX:
                mweight = layer.w_pipeline.forward(w, method=MethodMean(self.o), **kwargs)  # WX ST
            if False:  # Pre-BN
                Vx = x.var(dim=0).mean(dim=(-1,-2)) # [C]
                Vx = x.new_ones(x.shape[1]) # [C]
                Ex = x.mean(dim=(0,2,3)) # [C]
                x = x - Ex.view([1,-1,1,1]) # batch centering
                SMy = (qweight**2).sum(dim=(2,3)) # [O, C]
                Vxy = SMy @ Vx # [O]
                qweight = qweight / ((Vxy + 1e-5)**0.5).view([-1, 1, 1, 1])  # now it is weight norm
            shape = qweight.shape
            n_in = np.prod(np.array(shape[1:]))
            """scaling is for a better numerical range with fp16, does not affect FW/BW provided that we have a normalization with eps=0 afterwards"""
            scale = 1/(n_in) ** 0.5 / (layer.w_pipeline[0].quant.K-1)
            # scale = 1.0
            if layer.convex_comb:
                alpha = layer.convex_w.sigmoid()
                alpha = alpha + (layer.convex_w - layer.convex_w.detach())
                alpha = torch.stack([1-alpha, alpha], dim=0).view([-1,2,1] + [1]*(len(shape)-2))
                alpha = alpha * scale
                shape2 = [shape[0], 2, shape[1]//2] + list(shape[2:])
                qweight = (qweight.view(shape2) * alpha).view(shape)
                if self.o.WX:
                    mweight = (mweight.view(shape2) * alpha).view(shape)
            else:
                qweight *= scale
                if self.o.WX:
                    mweight *= scale
        #
        y = layer.forward_WB(x, qweight, layer.bias)
        if self.o.WX:
            my = layer.forward_WB(mx, mweight, layer.bias)  # WX ST
            y = subst_grad(y, my) # WX ST
        # y = y - y.mean(dim=(0,2,3), keepdim=True)
        # y = y / (y.std(dim=(0,2,3), keepdim=True) + 1e-5)
        # if y.mean(dim=(0,2,3)).abs().max()>0.05:
        #     print(y.mean(dim=(0,2,3)).abs().max().item(), end = ' ')
        #     print(qweight.abs().mean().item(), end = ' ')
        #     print(y.std(dim=(0,2,3)).max().item())
        return self.dispatch(layer.norm, y)
    
    def forward_WeightCentering(self, layer:WeightCentering, x):
        return layer.forward(x)
    
    def forward_Cat(self, layer:Cat, x:list|tuple , **kwargs):
        return layer.default_forward(x)


# _________________Deterministic Methods___________________

class MethodDet(Method):
    short_name = 'det'
    """ deterministic quantization"""

    def forward_Quant(self, quant: Quant, x: Tensor, **kwargs):
        return quant.quantize(x)
    
    def forward_Categorical(self, cat: Categorical, x: Tensor, **kwargs):
        return cat.embed_argmax(x)


class MethodReal(Method):
    def forward_Quant(self, quant: Quant, x: Tensor, **kwargs):
        """ no quantization, Squash is performing the clipping on the min value for 'ReLU' and both min and max for 'Clamp' """
        if self.o.q_squash == 'relu':
            y = torch.clamp(x, min = quant.range[0])
        elif self.o.q_squash == 'clamp':
            y = torch.clamp(x, min=quant.range[0], max=quant.range[1])
        else:
            raise RuntimeError(f"can't recognize method type {self.o.q_squash}")
        if self.o.MD and not quant.is_activation:
            return subst_grad(y, x) # MD through Clamp / ReLU
        return y
    
class MethodMean(Method):
    def forward_Quant(self, quant: Quant, x: Tensor, **kwargs):
        y = quant.mean_embedding(x)
        if self.o.MD and not quant.is_activation:
            return subst_grad(y, x)
        return y
    

  

# _________________Sampling-Based Methods___________________


class MethodSample(Method):

    def forward_Quant(self, quant: Quant, x: Tensor, samples = 1, **kwargs):
        # add noise, quantize = round + clamp
        # if multiple samples, return the average -- experimental use cases
        # if hasattr(self.o,"reg_eps") and self.o.reg_eps > 0 and 
        if quant.is_activation:
            r = ((x - quant.range[1]/2)**2).sum()
            quant.reg_loss = r
        if samples == 1:
            n = quant.sample_noise(x, self.o.CN)
            qx = quant.quantize(x + n)
        else:
            xe = x.unsqueeze(-1).expand([*x.shape, samples]) # expand to samples
            ne = quant.sample_noise(xe, self.o.CN)
            qx = quant.quantize(xe + ne).mean(dim=-1)
        return qx

    def forward_Categorical(self, layer: Categorical, logits: Tensor, ex_outs=None, **kwargs):
        """
        logits - [*,C] -- logits of categorical variables
        """
        # same as the dafault layer forward, the owerride does not have effect, but the dafault behaviour of Categorical may change
        index = layer.sample(logits)
        y = layer.embed(index)  # [*, embedding_size]
        if ex_outs is not None:
            ex_outs.index = index
        return y


class MethodMultiSample(Method):
    def forward_ESequential(self, layers: ESequential, x: Tensor, **kwargs):
        """ This works only if the network outputs logits with classes along the last dimension
        """
        p = 0
        n_samples = self.o.n_samples
        for i in range(n_samples):
            s = MethodSample(self.o).forward_ESequential(layers, x, **kwargs)
            p = p + torch.softmax(s, dim=-1)
        p /= n_samples
        score = torch.log(p)
        return score

    # def before_begin(self, x):
    #     # this is invalid, need to sample weights as well
    #     return samples_to_batch(x, self.o.n_samples)

    # def after_end(self, x):
    #     S = batch_to_samples(x, self.o.n_samples)
    #     P = torch.softmax(S, dim=-1)
    #     p = P.mean(dim=0, keepdim=False)
    #     score = torch.log(p)
    #     return score

#_______________________________________________________________


class MethodGS(Method):
    """
    Relaxed Quantization method of Louizos et al.
    Requires:
     o.temp:float - temperature parameter for the relaxation
     o.STGS:bool -- whether to use the ST-GS variant
    """

    def forward_Quant(self, quant: Quant, x: Tensor, **kwargs):
        """
        Overrides Method.forward_Quant
        :param layer: QuantRQ
        :param x: [*] -- Tensor of input activations to quantize
        :return:  [*] -- Tensor of quantized activations
        """
        MD = self.o.MD and not quant.is_activation
        xg = x.detach() if MD else x
        p = quant.cat_distribution(xg)
        logp = torch.log(p)
        # relaxed GS sample in the one_hot encoding -- another categorical distribution
        y = F.gumbel_softmax(logp, tau=self.o.temp, hard=self.o.STGS)
        # relaxed or hard sample, differentiable in noise and in x if not MD
        z = quant.expectation(y)
        if MD:
            # now gradient in x bypasses the (noisy) GS Jacobian
            z = z + (x - x.detach())
        return z

    def forward_Categorical(self, layer: Categorical, logits: Tensor, **kwargs):
        pi = F.gumbel_softmax(logits, tau=self.o.temp, hard=self.o.STGS)
        y = layer.expectation(probs=pi)
        return y


#_______________________________________________________________


class MethodST(MethodSample):
    """
    """
    @staticmethod
    def compute_dx(x, qx, quant: Quant, o):
        if quant.o.q_noise_sigma_learnable or (o.temp is not None and o.temp != 1):
            dx = quant.mean_embedding(x, o.temp)  # differentiable in x and noise std
        else:  # for non-parametric noises can use explicit derivative
            J = quant.d_mean_embedding(x, o.temp).detach()
            dx = J * x
            # dx = quant.mean_embedding(x) # debug
        if o.reweighting:
            # raise DeprecationWarning()
            # assert (not o.detST)
            m = torch.logical_or(x <= 0.5, x >= quant.K - 1.5)
            # dx_zgr = MethodZGR.compute_dx(x[m], qx[m], quant, alpha=1, temp=1)
            p = quant.cat_distribution(x)
            phi = quant.expectation(p)
            dx_ST = phi
            index = qx
            py = p.gather(-1, index.to(torch.long).unsqueeze(-1)).squeeze(-1)
            dx_RE = (index - phi.detach()) * torch.log(py)
            dx_zgr = (dx_ST + dx_RE)/2
            # dx[m] = dx_zgr[m] #GUDA GRAPH problem
            m = m.to(x)
            dx = dx * (1-m) + dx_zgr*m
        return dx

    def forward_Quant(self, quant: Quant, x: Tensor, samples = 1, **kwargs):
        """
        Overrides Method.forward_Quant
        :param layer: QuantRQ
        :param x: [*] -- Tensor of input activations to quantize
        :return:  [*] -- Tensor of quantized activations
        """
        MD = self.o.MD and not quant.is_activation
        if self.o.detST:  # determenistic forward, no injected noise
            qx = quant.quantize(x)
        else:  # stochastic rounding for the forward
            qx = MethodSample.forward_Quant(self, quant, x, samples=samples)
        if not MD:
            dx = self.compute_dx(x, qx, quant, self.o)
        else:  # MD
            if quant.o.q_noise_sigma_learnable:  # logisitc noise needs gradinet in std
                dx = self.compute_dx(x.detach(), qx, quant, self.o) + x
            else:  # fixed noise std
                dx = x

        if quant.is_activation and self.o.WX:
            return subst_grad(qx, dx), quant.mean_embedding(x) # WX-ST
        else:
            return subst_grad(qx, dx)
    

    # def forward_QReLU(self, layer: QReLU, x: Tensor, **kwargs):
    #     if not layer.quant.is_activation:
    #         return self.forward_ESequential(layer, x , **kwargs)
    #     else:
    #         p = layer.in_sb.bias
    #         object.__setattr__(layer.in_sb, 'bias', p.detach())
    #         y = self.forward_ESequential(layer, x , **kwargs)
    #         object.__setattr__(layer.in_sb, 'bias', p)
    #         y = y + (p - p.detach()).view([1,-1,1,1])
    #         return y


    # @torch.compile(**compile_args)
    # def forward_QReLU_compiled(self, layer: QReLU, x: Tensor, **kwargs):
    #     if not layer.quant.is_activation:
    #         return self.forward_ESequential(layer, x , **kwargs)
    #     else:
    #         p = layer.in_sb.bias
    #         del layer.in_sb.bias
    #         layer.in_sb.bias = p.detach()
    #         y = self.forward_ESequential(layer, x , **kwargs)
    #         layer.in_sb.bias = p
    #         y = y + (p - p.detach()).view([1,-1,1,1])
    #         return y

        

    def forward_Categorical(self, layer: Categorical, logits: Tensor, ex_outs = None, **kwargs):
        """
        logits - [*,C] -- logits of categorical variables
        """
        index = layer.sample(logits)
        y = layer.embed(index)  # [*, embedding_size]
        if self.o.temp is not None:
            my = layer.expectation(logits=logits/self.o.temp)  # [*, embedding_size]
        else:
            my = layer.expectation(logits=logits)  # [*, embedding_size]
        assert (y.shape == my.shape)
        if ex_outs is not None:
            ex_outs.index = index
        return subst_grad(y, my)


class MethodRepeat(Method):
    def __init__(self, method:Method, samples = 5):
        super().__init__(method.o)
        self.method = method
        self.samples = samples

    def forward_Quant(self, quant: Quant, x: Tensor, **kwargs):
        """
        Overrides Method.forward_Quant
        :param layer: QuantRQ
        :param x: [*] -- Tensor of input activations to quantize
        :return:  [*] -- Tensor of quantized activations
        """
        if quant.is_activation:
            if isinstance(self.method, MethodST):
                qx = self.method.forward_Quant(quant, x, samples = self.samples, **kwargs) # Can accelerate for many methods as e.g. mean embedding is the same for all samples
            else:
                x = x.unsqueeze(0).expand([self.samples, *x.shape])
                qx = self.method.forward_Quant(quant, x,  **kwargs)
                qx = qx.mean(dim=0)
        else:
            qx = self.method.forward_Quant(quant, x, **kwargs)
        return qx
        

#_______________________________________________________________


class MethodZGR(MethodSample):
    @staticmethod
    def compute_dx(x, index, quant: Quant, alpha, temp):
        # for efficinecy reasons we will compute (1-alpha)*ST(t) + alpha*ZGR(t)
        # = (1-alpha)*ST(t) + alpha*(ST(t) + DARN(t))/2 = (1-alpha/2)*ST(t) + alpha/2*DARN(t)
        # for t=1 it is ST(1)--ZGR family
        p = quant.cat_distribution(x, temp)
        phi = quant.expectation(p)
        dx_ST = phi
        assert((index == index).all())
        assert((index < p.shape[-1]).all())
        assert((index >= 0).all())
        py = p.gather(-1, index.to(torch.long).unsqueeze(-1)).squeeze(-1)
        dx_RE = (index - phi.detach()) * torch.log(py)
        dx = (1-alpha/2)*dx_ST + alpha/2*dx_RE
        return dx

    def forward_Quant(self, quant: Quant, x: Tensor, **kwargs):
        MD = self.o.MD and not quant.is_activation
        temp = self.o.temp if self.o.temp is not None else 1
        alpha = self.o.alpha if self.o.alpha is not None else 1
        beta = alpha**0.2
        temp = temp*(1 - beta) + 1.0*(beta) # anneal temperature down to 1, faster than transiting to ZGR
        #
        if self.o.det:
            qx = quant.quantize(x)
        else:
            qx = MethodSample.forward_Quant(self,quant,x)
        if not MD:
            dx = MethodZGR.compute_dx(x, qx, quant, alpha, temp)
        else:  # MD
            if quant.o.q_noise_sigma_learnable:
                dx = MethodZGR.compute_dx(x.detach(), qx, quant, alpha, temp) + x
            else:
                dx = x

        return subst_grad(qx, dx)

    def forward_Categorical(self, cat: Categorical, logits: Tensor, ex_outs=None, **kwargs):
        if False: # implementation using 1-hot embedding
            # get 1-hot categorical with ZGR gradient
            y = categorical.ZGR(logits)
            # convert to embeddig
            z = cat.expectation(probs=y)
            return z
        else: # implementation using the target embedding directly
            # sample categorical
            temp = self.o.temp if self.o.temp is not None else 1
            alpha = self.o.alpha if self.o.alpha is not None else 1
            temp = temp*(1 - alpha) + 1.0*(alpha)
            #
            p = logits.softmax(dim=-1)
            index = torch.distributions.categorical.Categorical(probs=p, validate_args=False).sample()
            y = cat.embed(index)
            logits = logits / temp
            logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
            p = logp.exp()
            phi = cat.expectation(probs=p)  # mean embedding
            dx_ST = phi
            logpx = logp.gather(-1, index.unsqueeze(-1))
            dx_RE = (y - phi.detach()) * logpx
            dx = (1-alpha/2)*dx_ST + alpha/2*dx_RE
            # else:
                # dx = (dx_ST + dx_RE) / 2
            # if self.o.alpha is not None:
                # dx = self.o.alpha * dx_ST + (1-self.o.alpha)*dx
            if ex_outs is not None:
                ex_outs.index = index
            return subst_grad(y, dx)


# #_______________________________________________________________

# class MethodZGR_Cat(Method):
#     def forward_Quant(self, quant: Quant, x: Tensor, **kwargs):
#         logp = quant.cat_distribution(x).log_()
#         # get 1-hot GR categorical
#         y = categorical.ZGR(logp)
#         # convert to ordinal
#         z = quant.expectation(y)
#         return z


#_______________________________________________________________

class MethodGR(Method):
    def forward_Quant(self, quant: Quant, x: Tensor, **kwargs):
        # p = quant.cat_distribution(x)
        # logp = p.log()
        # # get 1-hot GR categorical
        # y = categorical.GR(logp, k=self.o.K, temp=self.o.t)
        # # convert to ordinal
        # z = quant.expectation(y)
        # return z
        MD = self.o.MD and not quant.is_activation
        xg = x.detach() if MD else x
        p = quant.cat_distribution(xg)
        logp = torch.log(p)
        # relaxed GS sample in the one_hot encoding -- another categorical distribution
        # y = categorical.GR(logp, k=self.o.K, temp=self.o.t)
        y = categorical.GR(logp, k=self.o.GR_samples, temp=self.o.GR_temp)
        # relaxed hard sample, differentiable in noise and in x if not MD
        z = quant.expectation(y)
        if MD:
            # gradient in x bypasses the (noisy) Jacobian
            z = z + (x - x.detach())
        return z

    def forward_Categorical(self, cat: Categorical, logits: Tensor, ex_outs=None, **kwargs):
        I = torch.distributions.categorical.Categorical(logits=logits, validate_args=False).sample()
        # get 1-hot GR categorical
        y = categorical.GR(logits, k=self.o.GR_samples, temp=self.o.GR_temp, I = I)
        # convert to embedding
        z = cat.expectation(probs=y)
        if ex_outs is not None:
            ex_outs.index = I
        return z


class MethodST2(Method):
    """
    Categorical Quadratically unbiased minimum variance estimatro
    """
    def forward_Quant(self, quant: Quant, x: Tensor, **kwargs):
        # p = quant.cat_distribution(x)
        # logp = p.log()
        # # get 1-hot GR categorical
        # y = categorical.GR(logp, k=self.o.K, temp=self.o.t)
        # # convert to ordinal
        # z = quant.expectation(y)
        # return z
        MD = self.o.MD and not quant.is_activation
        xg = x.detach() if MD else x
        p = quant.cat_distribution(xg)
        logp = torch.log(p)
        # relaxed GS sample in the one_hot encoding -- another categorical distribution
        y = categorical.ST2(logp)
        # relaxed hard sample, differentiable in noise and in x if not MD
        z = quant.expectation(y)
        if MD:
            # gradient in x bypasses the (noisy) Jacobian
            z = z + (x - x.detach())
        return z

    def forward_Categorical(self, cat: Categorical, logits: Tensor, **kwargs):
        # get 1-hot GR categorical
        y = categorical.ST2(logits)
        # convert to embedding
        z = cat.expectation(probs=y)
        return z


#_______________________________________________________________


class MethodARSM(Method):
    def surrogate(self, quant: Quant, x: Tensor, fun=None, **kwargs):
        """ DEPRICATED, for test case only"""
        p = quant.cat_distribution(x)
        logits = p.log()
        loss = categorical.ARSM(logits, fun)
        return loss

    def forward_ESequential(self, layer: ESequential, x: Tensor, **kwargs) -> Tensor:
        for (i, l) in enumerate(layer):
            if isinstance(l, Categorical):
                tail = layer[i+1:]  # nn.Sequantial tail of the network
                return self.forward_Categorical(l, x, tail=tail, **kwargs)  # categorical will execute the tail
            if isinstance(l, Quant):
                tail = layer[i+1:]  # nn.Sequantial tail of the network
                return self.forward_Quant(l, x, tail=tail, **kwargs)  # categorical will execute the tail
            # kwargs.pop("tail", None)
            x = self.dispatch(l, x, **kwargs)
        return x

    def forward_Categorical(self, layer: Categorical, x: Tensor, tail=None, **kwargs) -> Tensor:
        assert (tail is not None), "Must have access to the tail of the network till the loss function"
        # convention: categorical outputs the embedding
        # but ARSM expects function of the index

        def fun(index) -> Tensor:
            y = layer.embed(index)
            loss = tail.forward(y, method=self)
            return loss
        return categorical.ARSM(x, fun)

    def forward_Quant(self, quant: Quant, x: Tensor, tail=None, **kwargs):
        p = quant.cat_distribution(x)
        logits = p.log()
        # use the categorical implementation
        cat = Categorical(C=quant.K, embedding='integer').to(x)
        y = self.forward_Categorical(cat, logits, tail=tail)
        return y

#_______________________________________________________________


class MethodRF(Method):
    """
     Multi-sample REINFORCE with built-in baseline
     [Eq. 23, Kool et al. 2020 "Estimating Gradients for Discrete Random Variables by Sampling without Replacement"]
    """
    def forward_ESequential(self, layer: ESequential, x: Tensor, net_ctx=None, **kwargs) -> Tensor:
        
        def dispatch_split_samples(l, x):
            X = batch_to_samples(x, self.o.M)
            Y = []
            for s in range(X.shape[0]):
                y = self.dispatch(l, X[s], net_ctx=net_ctx, **kwargs)
                Y.append(y)
            y = torch.cat(Y, dim=0)
            y.multisample = True
            return y
        
        # for the whole network: propagate all deterministic layers, once see a Categorical layer go to multisample mode
        outermost = False
        scores = None
        if net_ctx is None:
            net_ctx = dotdict()
            net_ctx.net = layer
            outermost = True # We ar eprocessing the whole network and not a Sequential block inside the network
        for l in layer:
            if isinstance(l, Categorical) or isinstance(l, Quant) or isinstance(l, QAnyLinear): # copy-split the input just before Categorical / Quant/ QLinear if not split yet
                if not hasattr(x, 'multisample'):
                    # split
                    x = samples_to_batch(x, self.o.M)
                    x.multisample = True
            # continue
            if isinstance(l, ELoss):
                scores = batch_to_samples(x, self.o.M).mean(dim=0)  # [M, B]
            
            if (isinstance(l, nn.BatchNorm2d) or isinstance(l, nn.BatchNorm1d) or isinstance(l, ELoss)) and hasattr(x, 'multisample'): # sensetive layers
                y = dispatch_split_samples(l, x)
            else:
                y = self.dispatch(l, x, net_ctx=net_ctx, **kwargs)
                if hasattr(x, 'multisample'):
                    y.multisample = True
            x = y
        if outermost:
            xx = batch_to_samples(x, self.o.M)  # [M, B]
            net_ctx.results = xx.detach() # save for backward
            x = xx.mean(dim=0)
            # need to hook the gradient of the loss in x, because it coulb be a sum or mean reduction, not known to the network forward
            class Hook(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x
                @staticmethod
                def backward(ctx, J):
                    net_ctx.loss_J = J
                    return J
            return Hook.apply(x), scores
        else:
            return x
           
    def forward_Parallel(self, layer: Parallel, x: Tensor, **kwargs):
        ## hacky solution to do reduction when one branch was expanded with samples and the other one was not
        r = None
        for l in layer:  # now we are doing this sequentially, but can do in parallel
            y = self.dispatch(l, x, **kwargs)
            if layer.reduction == 'sum':
                if r is None:
                    r = y
                else:
                    if r.shape[0] <= y.shape[0]:
                        r = r.repeat((y.shape[0]//r.shape[0],) + tuple(r.shape[1:])) + y
                    else:
                        r = y.repeat((r.shape[0]//y.shape[0],) + tuple(y.shape[1:])) + r
        return r
    
    # def forward_LayerKW(self, layer: FuncLayer, x: Tensor, **kwargs) -> Tensor:
    #     y = layer.func(x, *layer.args, **kwargs)
    #     return y

    def forward_Categorical(self, layer: Categorical, logits: Tensor, net_ctx, is_activation = True, ex_outs=None, **kwargs) -> Tensor:
        # 
        # save for backward network context, on the backward will have results
        M = self.o.M
        logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        y = layer.sample(logits=logp) # [M*B, *]
        if ex_outs is not None:
            ex_outs.index = y
        logpy = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)  # [M*B, *]
        shape = logpy.shape

        class RFaux(torch.autograd.Function):
            @staticmethod
            def forward(ctx, logpy, y): # we will define gradient in logpy
                # input:
                # logpy [M*B, *]
                # y [M*B, *]
                ye = layer.embed(y)
                ye.requires_grad = True
                return ye

            @staticmethod
            def backward(ctx, J):
                ff = net_ctx.results  # [M, B]
                loss_J = net_ctx.loss_J  # [B]
                df = (ff - ff.mean(dim=0, keepdim=True))/(M-1)*loss_J.view([1, -1])  # [M, B]
                if is_activation: # first dimension of the input tensor is batch size
                    # assert (ff.shape[1]*M == shape[0])
                    g = df.view([shape[0]] + [1]*(len(shape)-1)).expand(shape)
                else:  # weight case: first dimension of the input tensor is weight shape
                    # assert (ff.shape[1]*M != shape[0])
                    df = df.sum(dim=-1)  # [M]
                    # g neds to be [M*C_in, *]
                    df = df.repeat(shape[0]//df.shape[0])
                    g = df.view([-1] + [1]*(len(shape)-1)).expand(shape)
                return g, None       
        return RFaux.apply(logpy, y)
               
    def forward_Quant(self, quant: Quant, x: Tensor, net_ctx, **kwargs):
        p = quant.cat_distribution(x)
        logits = p.log()
        # use the categorical implementation
        if not hasattr(quant, 'cat') or quant.cat is None:
            # need a cached Cat layer, that remembers embedding
            quant.cat = Categorical(C=quant.K, embedding='integer').to(x) # todo: optimize out
        y = self.forward_Categorical(quant.cat, logits, net_ctx, is_activation = quant.is_activation)
        return y
    
    def forward_QAnyLinear(self, layer: QAnyLinear, x: Tensor, **kwargs) -> Union[Tensor, RandomVar]:
        assert isinstance(x, Tensor)
        assert(x.multisample)
        # quantizer is a ESequential with Quant in the end, it will split the logits before Quant
        qweight = layer.quantizer.forward(layer.weight, method=self, **kwargs) # [S*C_out, C_in, *]
        assert(qweight.multisample)
        #
        X = batch_to_samples(x, self.o.M) # [S, B, *]
        W = batch_to_samples(qweight, self.o.M)  # [S, C_out, C_in, *]
        Y = []
        # Loop over samples to multiply independnetly
        for s in range(self.o.M):
            xs = X[s]
            ws = W[s]
            # ws = ws + layer.skip_scale.view((-1,) + (1,) * (ws.ndim - 1))
            ys = layer.forward_WB(xs, ws, layer.bias)
            Y.append(ys)
        y = torch.cat(Y, dim=0) # [B*S, ]
        y.multisample = True
        return y
        # return layer.forward()    

# _________________________________________________________________________

class MethodRF1(Method):
    """
     Multi-sample REINFORCE with built-in baseline
     [Eq. 8, Kool et al. 2020 "Buy 4 REINFORCE Samples, Get a Baseline for Free!"]
    """
    def __init__(self, o):
        super().__init__(o)
        self.net_ctx = dotdict()
    
    def forward_ESequential(self, layer: ESequential, x: Tensor, net_ctx=None, s = 0, **kwargs) -> Tensor:
        # what we do for the whole network
        if net_ctx is None:
            net_ctx = self.net_ctx
            # net_ctx = dotdict()
            # net_ctx.net = layer
            # self-baseline (more memory)
            ll = [] #torch.zeros(self.o.M,x.shape[0], device = x.device) # [M x B]
            scores = [] # [M, B, K]
            for s in range(self.o.M):
                l, score = super(MethodRF1, self).forward_ESequential(layer, x, net_ctx=net_ctx, s=s, **kwargs) # losses per sample
                # net_ctx.results[s], score = MethodST().dispatch(layer, x, net_ctx=net_ctx, s=s, **kwargs) # losses per sample
                ll.append(l)
                scores.append(score)
            ll = torch.stack(ll)
            l_mean = ll.mean(dim=0) # [B]
            net_ctx.results = ll.detach()
            s_mean = torch.stack(scores).mean(dim=0) # [B, K]
            # before returning need to hook the gradient of the loss in x, because it coulb be a sum or mean reduction in mini-batch, not known to the network forward
            class Hook(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x
                @staticmethod
                def backward(ctx, J):
                    net_ctx.loss_J = J
                    return J
            return Hook.apply(l_mean), s_mean
        else:
            return super(MethodRF1, self).forward_ESequential(layer, x, net_ctx=net_ctx, s= s, **kwargs) # single sample loss

    def forward_Categorical(self, layer: Categorical, logits: Tensor, net_ctx, is_activation = True, ex_outs=None, s=0, **kwargs) -> Tensor:
        # 
        # save for backward network context, on the backward will have results
        logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        # first dimension of the logits tensor is either batch size or C_in for weights. Denote it as D
        y = layer.sample(logits=logp) # [*] -- logits.shape()[:-1]
        if ex_outs is not None:
            ex_outs.index = y
        logpy = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)  # [*]
        shape = logpy.shape
        M = self.o.M
        class RFaux(torch.autograd.Function):
            @staticmethod
            def forward(ctx, logpy, y): # we will define gradient in logpy
                # input:
                # logpy [*]
                # y [*]
                ye = layer.embed(y)
                ye.requires_grad = True
                return ye

            @staticmethod
            def backward(ctx, J):
                ff = net_ctx.results  # [M, B]
                loss_J = net_ctx.loss_J  # [B]
                df = (ff[s] - ff.mean(dim=0, keepdim=False))/(M-1)*loss_J  # [B]
                if is_activation: # first dimension of the input tensor is batch size
                    g = df.view([shape[0]] + [1]*(len(shape)-1)).expand(shape)
                else:  # weight case: first dimension of the input tensor is weight shape
                    df = df.sum(dim=-1)  # [] -- scalar
                    # g neds to be [C_in, ...]
                    g = df.expand(shape)
                return g, None       
        return RFaux.apply(logpy, y)
               
    def forward_Quant(self, quant: Quant, x: Tensor, net_ctx, s=0, **kwargs):
        #todo: Quant via Categorical gives diffenret results -- DEBUG
        if False:
            p = quant.cat_distribution(x)
            logits = p.log()
            # use the categorical implementation
            if not hasattr(quant, 'cat') or quant.cat is None:
                # need a cached Cat layer, that remembers embedding
                quant.cat = Categorical(C=quant.K, embedding='integer').to(x) # todo: optimize out
            y = self.forward_Categorical(quant.cat, logits, net_ctx, is_activation = quant.is_activation)
            return y
        M = self.o.M
        n = quant.sample_noise(x, self.o.CN)
        y = quant.quantize(x + n).detach()
        py = quant.cat_probability(x, y)
        logpy = torch.log(py)
        shape = y.shape
        
        # return y + logpy - logpy.detach()
        
        class RFaux(torch.autograd.Function):
            @staticmethod
            def forward(ctx, logpy, y): # we will define gradient in logpy
                y.requires_grad = True
                return y

            @staticmethod
            def backward(ctx, J):
                ff = net_ctx.results  # [M, B]
                loss_J = net_ctx.loss_J  # [B]
                df = (ff[s] - ff.mean(dim=0, keepdim=False))/(M-1)*loss_J  # [B]
                if quant.is_activation: # first dimension of the input tensor is batch size
                    g = df.view([shape[0]] + [1]*(len(shape)-1)).expand(shape)
                else:  # weight case: first dimension of the input tensor is weight shape
                    df = df.sum(dim=-1)  # [] -- scalar
                    # g neds to be [C_in, ...]
                    g = df.expand(shape)
                return g, None       
        return RFaux.apply(logpy, y)

# _________________________________________________________________________


# @torch.jit.script
def STQ_dx(qx:Tensor, p:Tensor, xx:Tensor) ->Tensor:
    Mu1 = (xx * p).sum(dim=-1)  # mean embedding
    Mu2 = (xx**2 * p).sum(dim=-1)  # mean squared embedding
    # compute a,b,c
    with torch.no_grad():
        mu1 = Mu1.detach()
        mu2 = Mu2.detach()
        # v = torch.clamp(a*c - b**2, min=1e-10)  #  variance of embedding under distribution
        # v = mu2 - mu1**2 + 0.01 #1e-10
        v = mu2 - mu1**2
        # v = torch.clamp(v, min=0.1) #1e-10
        a1 = (mu2 - qx*mu1)/v
        a2 = (qx - mu1)/v*0.5
    #
    F = a1 * Mu1 + a2*Mu2
    return F

class MethodSTQ(MethodSample):
    def compute_dx(self, eta, qx, quant:Quant):

        p = quant.cat_distribution(eta)  # [* K]
        # dp = quant.d_distribution(eta) # [* K]
        xx = quant.grid.view((1,)*eta.dim() + (-1,))  # [* K]
        return STQ_dx(qx, p, xx)
        # compute mu1, mu2 -- expected gradient for linear and qudratic functions
        # mu1 = (xx * dp).sum(dim=-1)  # [*]
        # # mu1 = quant.d_mean_embedding(eta)
        # mu2 = 1/2*(xx**2 * dp).sum(dim=-1)  # [*]
        # compute lambda1, labmda2
        # lambda1 = (c * mu1 - b * mu2)/v
        # lambda2 = (a * mu2 - b * mu1)/v
        # # compute transfer coefficient to eta
        # f = (lambda1 + qx * lambda2)
        # # what will be the variance of the estimator?
        # # m1 = quant.mean_embedding(eta)
        # # m2 = (xx**2 * p).sum(dim=-1)
        # # var = lambda1**2 + 2*m1*lambda1*lambda2 + m2*lambda2**2
        # # mask = var > 2
        # # f[mask] = mu1[mask] # give up quadratic unbiased
        # d = f.detach()*eta

        # ok, f is linear in mu1, mu2; a, b, c, v depend on distribution but do not need Jacobian
        Mu1 = (xx * p).sum(dim=-1)  # mean embedding
        Mu2 = (xx**2 * p).sum(dim=-1)  # mean squared embedding
        # compute a,b,c
        with torch.no_grad():
            mu1 = Mu1.detach()
            mu2 = Mu2.detach()
            # v = torch.clamp(a*c - b**2, min=1e-10)  #  variance of embedding under distribution
            v = mu2 - mu1**2 + 1e-10
            a1 = (mu2 - qx*mu1)/v
            a2 = (qx - mu1)/v*0.5
        #
        F = a1 * Mu1 + a2*Mu2
        return F

    def forward_Quant(self, quant: Quant, x: Tensor, **kwargs):
        MD = self.o.MD and not quant.is_activation
        qx = MethodSample.forward_Quant(self, quant,x, **kwargs)
        # n = quant.sample_noise(x, self.o.CN)
        # qx = quant.quantize(x + n)
        if not MD:
            dx = self.compute_dx(x, qx, quant)
        else:  # MD
            if quant.o.q_noise_sigma_learnable:
                dx = self.compute_dx(x.detach(), qx, quant) + x
            else:
                dx = x
        return subst_grad(qx, dx)


# _________________ ReinMax ________________________
class MethodReinMax(MethodSample):
    def compute_qx(self, eta, quant:Quant):
        p = quant.cat_distribution(eta)  # [* K]
        logp = torch.log(p)
        y, _ = categorical.reinmax(logp, self.o.temp) # one_hot
        z = quant.expectation(y)
        return z

    def forward_Quant(self, quant: Quant, x: Tensor, **kwargs):
        MD = self.o.MD and not quant.is_activation
        if MD:  # MD
            raise NotImplementedError()
        qx = self.compute_qx(x, quant)        
        return qx

# _________________ Local Reparam ___________________


class MethodMeanSample(MethodST):
    short_name = 'MeanSample'
       
    def forward_Quant(self, quant: Quant, x: Tensor, **kwargs):
        # return MethodST.forward_Quant(self, quant, x, **kwargs)

        if not quant.is_activation:
            return MethodST.forward_Quant(self, quant, x, **kwargs)
        # quant is activation
        # create a proxy object
        sample = MethodST.forward_Quant(self, quant, x, **kwargs) # TODO: better could be MethodSample.forward_quant, or explicit
        return QuantProxy(quant, x, sample = sample, method= self)
        if False:
            assert(isinstance(x,Tensor))
            m = quant.mean_embedding(x) # x is latent weight
            if self.o.MD:
                return subst_grad(m, x) # shouldn't we sample /quantize the weight?
            else:
                return m
        else:
            if isinstance(x, Tensor):
                if quant.K==2:
                    p = quant.mean_embedding(x)
                    return BernoulliDistribution(p, quant)
                else:
                    p = quant.cat_distribution(x)
                    return CatDistribution(p, quant)
                # requantizing x assuming it is Normally distributed is not a good idea.
                # If we had a cat distribution of x we could update it correctly
                x = x.mean # hack: ignoring the variance
            elif quant.K > 2: # update distribution (for requantize)
                if isinstance(x, BernoulliDistribution):
                    p = torch.stack(1-x.p, x.p, dim=-1) # [B C H W 2]
                elif isinstance(x, CatDistribution):
                    p = x.p # [B C H W K]
                y = x.shift.unsqueeze(-1) + x.quant.grid.view([1]*x.shift.dim() + [-1])*(x.scale.unsqueeze(-1))
                Q = quant.cat_distribution(y) # [K1, K2] transition probabilities for each shifted grid point
                p = torch.einsum('...k, ...kl -> ...l', x.p, Q)
                return CatDistribution(p, quant)
            elif isinstance(x, BernoulliDistribution) and quant.K==2:
                p = x.p # [B C H W K]
                # assert(quant.K == x.quant.K)
                q0 = quant.mean_embedding(0 + x.shift) # conditional distribution assuming x=0
                q1 = quant.mean_embedding(1*x.scale + x.shift)  # conditional distribution assuming x=1
                p = q1 * x.p + q0 * (1-x.p)
                return BernoulliDistribution(p, quant)
            else:
                raise RuntimeError('Unimplemented')
                
                # p = quant.cat_distribution(x) # for binary this is the same as mean
                # m = quant.expectation(p)
                # V = (quant.second_moment(p) - m**2).clamp(min=1e-6)
                # return RandomVar(m,V)

    def forward_Cat(self, layer:Cat, x:list|tuple, **kwargs):
        if isinstance(x[0], Tensor):
            return layer.default_forward(x)
        else:
            return x[0].cat(x, dim=layer.dim)
    
    def forward_WeightCentering(self, centering: WeightCentering, x:Union[RandomVar, Tensor], **kwargs):
        if isinstance(x,Tensor):
            return centering.forward(x) # center sample
        elif isinstance(x, RandomVar):
            # center only mean, keep variance the same
            m = centering.forward(x.mean) # center mean
            return RandomVar(m, x.var)
        else:
            raise NotImplementedError()
       
    def forward_QAnyLinear(self, layer: QAnyLinear, x:Union[Tensor, CatDistribution, BernoulliDistribution], **kwargs) -> Tensor:
        # DEBUG
        # if isinstance(x, QuantProxy):
        #     qx = x.sample()
        #     m = x.mean_embedding()
        #     x = qx + (m - m.detach())
        # return MethodST.forward_QAnyLinear(self, layer, x, **kwargs)
        # DEBUG
        # """Common implementation for determinisitc and Sampling-Based methods"""
        # w = layer.weight.view(layer.weight.shape) # view as Tensor, needed for torch.compile to set dynamic shape (workaround, maybe fixed now)
        # Experimental: 
        # C = x.shape[1]
        # # x[:,:C] *= -1 # these are going to be "inhibiting"
        # x[:,:C] *= 2 # make more significant than other
        #            
        # assert isinstance(x, Tensor)
        # torch._dynamo.mark_dynamic(w, (0,1,2,3))
        # with torch.amp.autocast(device_type="cuda",enabled=False):
        #     qweight = layer.w_pipeline.forward(w, method=self, **kwargs)
        #     shape = qweight.shape
        #     n_in = np.prod(np.array(shape[1:]))
        #     """scaling is for a better numerical range with fp16, does not affect FW/BW provided that we have a normalization with eps=0 afterwards"""
        #     scale = 1/(n_in) ** 0.5
        #     # scale = 1.0
        #     if layer.convex_comb:
        #         alpha = layer.convex_w.sigmoid()
        #         alpha = alpha + (layer.convex_w - layer.convex_w.detach())
        #         alpha = torch.stack([1-alpha, alpha], dim=0).view([-1,2,1] + [1]*(len(shape)-2))
        #         alpha = alpha * scale
        #         shape2 = [shape[0], 2, shape[1]//2] + list(shape[2:])
        #         qweight = (qweight.view(shape2) * alpha).view(shape)
        #     else:
        #         qweight *= scale
        # #
        # y = layer.forward_WB(x, qweight, layer.bias)
        # return self.dispatch(layer.norm, y)
        # #
                
        """
        For det input compute det output
        For RV input compute mean and variance and sample
        """
        w = layer.weight.view(layer.weight.shape) # view as Tensor, needed for torch.compile to set dynamic shape (workaround, maybe fixed now)
        torch._dynamo.mark_dynamic(w, (0,1,2,3))
        with torch.amp.autocast(device_type="cuda",enabled=False):
            qw = layer.w_pipeline.forward(w, method=self, **kwargs) # currently, just sample weights
            assert(isinstance(qw, Tensor))
            shape = qw.shape
            n_in = np.prod(np.array(shape[1:]))
            """scaling is for a better numerical range with fp16, does not affect FW/BW provided that we have a normalization with eps=0 afterwards"""
            # Kw = layer.w_pipeline[0].quant.K
            scale = 1/(n_in) ** 0.5
            if layer.convex_comb:
                alpha = layer.convex_w.sigmoid()
                alpha = alpha + (layer.convex_w - layer.convex_w.detach())
                alpha = torch.stack([1-alpha, alpha], dim=0).view([-1,2,1] + [1]*(len(shape)-2))
                alpha = alpha * scale
                shape2 = [shape[0], 2, shape[1]//2] + list(shape[2:])
                qw = (qw.view(shape2) * alpha).view(shape)
            else:
                qw *= scale
            # qw *= scale # for better numerical range, cancelled by normalization below

            if isinstance(x,Tensor): # unquantized network input
                rsample = layer.forward_WB(x, qw, layer.bias)
            elif isinstance(x, QuantProxy):
                if False: # DEBUG
                    qx = x.sample()
                    m = x.mean_embedding()
                    rx = qx + (m - m.detach())
                    rsample = layer.forward_WB(rx, qw, layer.bias)
                else:
                    # x is qunatized
                    m1_x, m2_x = x.moments()
                    v_x = torch.clamp(m2_x - m1_x**2, min = 0.1) # for numerical stability
                    # y mean and var
                    m1_y = layer.forward_WB(m1_x, qw, None)
                    v_y = layer.forward_WB(v_x, qw**2, None) + 1e-3 # qw can be all zeros? yes if we use dynamic centering
                    # v_y += layer.forward_WB(m2_x, torch.ones_like(qw)*(scale**2)*0.01, None)
                    # m2_y = layer.forward_WB(m2_x, qw**2, None)
                    # v_y = torch.clamp(m2_y - m1_y**2, min = 1e-8)
                    s_y = v_y**0.5
                    with torch.no_grad():
                            qx = x.sample()
                            qy = layer.forward_WB(qx, qw, None) # discrete sample output
                            n = (qy - m1_y) / s_y # think of it as N(0,1)
                    rsample = m1_y + n * s_y # value of qy, gradient goes thorugh m1_y and s_y
                    if layer.bias is not None:
                        raise RuntimeError("Normalization below cancels bias")

                # var = layer.forward_WB(rx.var, qw**2, None) + 1e-8
                # if False: # Common Local Reparameterization Trick
                #     n = torch.empty_like(mean).normal_()
                #     rsample = n * (var **0.5) + mean
                # else: # Discrete Local Reparameterization Trick
                #     qx = rx.sample()
                #     qy = layer.forward_WB(qx, qw, layer.bias) # discrete sample output
                #     s = var**0.5
                #     with torch.no_grad():
                #         n = (qy - mean) / (s) # think of it as N(0,1)
                #     rsample = mean + n * s
                # y = rsample
            
            return self.dispatch(layer.norm, rsample)



# ______________________________________________________
class MethodCompose(Method):
    def __init__(self, m1, m2, switch_layer):
        super().__init__(m1.o)
        self.m1 = m1
        self.m2 = m2
        self.switch_layer:int = switch_layer

    def dispatch(self, layer, *args, **kwargs):
        self.m1.o.compile = self.o.compile
        self.m2.o.compile = self.o.compile
        if isinstance(layer, EClassificationNet):
            net1 = ESequential([layer[k] for k in range(0,self.switch_layer)])
            net2 = ESequential([layer[k] for k in range(self.switch_layer, len(layer))])
            x = self.m1.dispatch(net1, *args, **kwargs)
            x = self.m2.dispatch(net2, x, **kwargs)
            return x
        else:
            return super().dispatch(layer, *args, **kwargs)

# ______________________________________________________
# ______________________________________________________

def test1():
    print("Testing layers with ST")
    o = dotdict(A=3, W=3, q_noise_type='logistic', q_noise_sigma=1, NS=1.0)
    m = MethodST(o)
    #
    x = torch.zeros((2, 10, 7, 7))
    x.requires_grad = True
    # Test fully connected netwokr
    net = ESequential(nn.Flatten(), nn.Linear(10*7*7, 10), QReLU(o), QLinear(o, 10, 8*3), FuncLayer(split_dim, 1, 8),  Categorical(C=8))
    y = net.forward(x, method=m)
    l = ((y - 1)**2).sum()
    l.backward()
    print(x.grad.sum())
    # Test Convolutional network
    net = ESequential(nn.Conv2d(10, 10, 3), QReLU(o), QConv2d(o, 10, 8, 3), FuncLayer(split_dim, 1, 8), FuncLayer(torch.swapdims, 2, -1), Categorical(C=8), FuncLayer(merge_dims, 1, -1))
    y = net.forward(x, method=m)
    print(y.shape)
    l = ((y - 1)**2).sum()
    l.backward()
    print(x.grad.sum())
    print("Test passed")
    print("Testing all Methods")
    # mm = [MethodDS(dotdict(temp=1.0))]
    # for m in mm:
    #     y = net.forward(x, method=m)
    #     print(y.shape)
    #     l = ((y - 1)**2).sum()
    #     l.backward()
    #     print(x.grad.sum())


def test_vae():
    o = dotdict(A=3, W=3, q_noise_type='logistic', q_noise_sigma=1, NS=1.0, M = 1000)
    m0 = MethodZGR(o)
    # m0 = MethodARSM(o)
    m = MethodRF(o)
    #
    torch.manual_seed(0)
    x = torch.zeros((1, 8))
    x.requires_grad = True
    net = ESequential(nn.Linear(8, 4*8), FuncLayer(split_dim, 1, 8), Categorical(C=8), nn.Flatten(), nn.Linear(4*3, 8))

    def fun(y, gt, samples = None, **kwargs):
        X = gt.detach()
        if samples is not None and samples > 1:
            X = samples_to_batch(X, samples)
        return ((X-y)**2).sum(dim=-1)
    
    x.grad = None
    max_samples_batch = 1000
    X = samples_to_batch(x,max_samples_batch)
    loss_net = ESequential(*net.children(), KWFuncLayer(fun, X.detach()))
    l = loss_net.forward(X, method=m0)
    l.mean().backward()
    g = x.grad
    print(g)
    #
    loss_net = ESequential(*net.children(), KWFuncLayer(fun, x))
    x.grad = None
    l = loss_net.forward(x, method=m)
    l.mean().backward()
    print(x.grad)

if __run__:
    test_vae()
