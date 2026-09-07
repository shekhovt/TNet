# SPDX-License-Identifier: CC BY-NC-SA 4.0
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. Making code open source is essential for scientific
# transparency and reproducibility. Providing access to the code allows researchers to
# verify, replicate, and expand upon results, which supports scientific progress. Current
# top-tier conferences encourage opensource code for the purpose of reproducibility that
# is de-facto an established expectation of the research community. Opensource code as
# part of publication should not affect the performance of an invention both from a
# commercial and IP point of view.
"""
ZGR gradient estimator
"""

import torch
from torch import Tensor
import torch.nn.functional as F


def ZGR_categorical(logits, x=None):
    """Returns a categorical sample from Categorical softmax(logits) (over axis=-1) as a
    one-hot vector, with ZGR gradient.
    
    Input: 
    logits [*, C], where C is the number of categories
    x: (optional) categorical sample to use instead of drawing a new sample. [*]
    
    Output: categorical samples with ZGR gradient [*,C] encoded as one_hot
    """
    # return ZGR_Function().apply(logits, x)
    # using surrogate loss
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    p = logp.exp()  # [*, C]
    dx_ST = p  # [*,C]
    index = torch.distributions.categorical.Categorical(probs=p, validate_args=False).sample()  # [*]
    num_classes = logits.shape[-1]
    y = F.one_hot(index, num_classes=num_classes).to(p)  # [*,C], same dtype as p
    logpx = logp.gather(-1, index.unsqueeze(-1)) # [*,1] -- log probability of drawn sample
    dx_RE = (y - p.detach()) * logpx
    dx = (dx_ST + dx_RE) / 2
    return y + (dx - dx.detach())


def ZGR_binary(logits:Tensor, x:Tensor=None)->Tensor:
    """Returns a Bernoulli sample for given logits with ZGR = DARN(1/2) gradient
    Input: logits [*]
    x: (optional) binary sample to use instead of drawing a new sample. [*]
    Output: binary samples with ZGR gradient [*], dtype as logits
    """
    p = torch.sigmoid(logits)
    if x is None:
        x = p.bernoulli()
    J = (x * (1-p) + (1-x)*p )/2
    return x + J.detach()*(logits - logits.detach()) # value of x with J on backprop to logits



if __name__ == '__main__':
    def fun(b):
        return torch.sum((b-1.0)**2)
    """Checking that ZGR_categorical with 2 categories matches with ZGR_binary """
    torch.manual_seed(2)
    logits = torch.arange(12).reshape(-1,2).float()
    logits.requires_grad = True
    c = ZGR_categorical(logits)
    b = c[:,0]
    loss = fun(b)
    loss.backward()
    g = logits.grad
    print(g)
    logits.grad = None
    blogits = (logits[:,0] - logits[:,1])
    b1 = ZGR_binary(blogits, b.detach())
    loss = fun(b1)
    loss.backward()
    g1 = logits.grad
    print(g1)
    assert((g1 -g).max().abs()<1e-6)
    print('Passed')
    