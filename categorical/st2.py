# SPDX-License-Identifier: CC BY-NC-SA 4.0
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. Making code open source is essential for scientific
# transparency and reproducibility. Providing access to the code allows researchers to
# verify, replicate, and expand upon results, which supports scientific progress. Current
# top-tier conferences encourage opensource code for the purpose of reproducibility that
# is de-facto an established expectation of the research community. Opensource code as
# part of publication should not affect the performance of an invention both from a
# commercial and IP point of view.
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
import torch.utils
import math
import copy

from ..functional import *


def estimator(logits: Tensor) -> Tensor:
    """
    Input logits: Tensor [*,K]
    Output categorical one_hot vector [*,K]
    """
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    p = logp.exp()  # [*, C]
    index = torch.distributions.categorical.Categorical(probs=p, validate_args=False).sample()  # [*]
    num_classes = logits.shape[-1]
    logpx = logp.gather(-1, index.unsqueeze(-1))  # [*,1] -- log probability of drawn sample
    px = p.gather(-1, index.unsqueeze(-1))  # [*,1] -- probability of drawn sample

    class Aux(torch.autograd.Function):
        @staticmethod
        def forward(ctx, p, logpx):  # we will define gradient in p  and logpx
            # input:
            # p [*, K]
            # logpx [*,1]
            y = F.one_hot(index, num_classes=num_classes).to(p)  # [*,C], same dtype as p
            y.requires_grad = True
            return y

        @staticmethod
        def backward(ctx, J):
            # J [*, K]
            #
            p1 = p.unsqueeze(-1)  # [*, K, 1]
            p2 = p.unsqueeze(-2)  # [*, 1, K]
            G = p1*p2/(p1 + p2 + 1e-10)  # [*, K, K]
            d = G.sum(axis=-1)  # [*, K]
            D = torch.diag_embed(d, dim1=G.dim()-2, dim2=G.dim()-1)
            I = torch.diag_embed(torch.ones_like(d), dim1=G.dim()-2, dim2=G.dim()-1)
            L = (D - G)  # [*, K, K]
            if False:  # Neumann series
                M = I - L
                IL = I + M
                for k in range(5):
                    M = M@M
                    IL = IL@(I + M)
                # IL is the inverse of L
                # now apply it
                IL -= IL.flatten(start_dim=p.dim()-1).mean(dim=-1).unsqueeze(-1).unsqueeze(-1)
                IL1 = torch.linalg.pinv(L)
                assert (IL - IL1).norm(dim=(-1, -2)).max() < 0.001
            else:
                IL1 = torch.linalg.pinv(L)
                IL = IL1
            ILx = IL.gather(-2, unsqueeze_expand(index.unsqueeze(-1), -1, num_classes))   # [*, 1, K]
            Jp = (J * p/(p + px + 1e-10)*0.5).unsqueeze(-2) # [*, 1, K]
            J_p = (Jp @  (IL - ILx)).squeeze(-2)  # [*, K]
            J_loxpx = J.gather(-1, index.unsqueeze(-1))*0.5  # [*, 1]
            return J_p, J_loxpx

    return Aux.apply(p, logpx)


if __name__ == '__main__':
    torch.manual_seed(0)
    logits = torch.arange(12).reshape(2, -1).float()
    logits.requires_grad = True
    c = estimator(logits)
    loss = torch.sum((c-1.0)**2)
    loss.backward()
    g = logits.grad
    print(g)
    print(g.sum(dim=-1))
