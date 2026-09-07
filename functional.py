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
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
import code
import math

all_checks = True


def log1p_exp(x: Tensor) -> Tensor:
    """
    compute log(1+exp(a)) = log(exp(0)+exp(a))
    """
    m = torch.clamp(x, min=0).detach()  # max(x,0)
    return m + torch.log(torch.exp(x - m) + torch.exp(-m))


def soft_plus(x: Tensor, beta=1):
    return F.softplus(x, beta)


def soft_minus(x: Tensor, beta=1):
    """ inverse of soft_plus """
    return torch.log(torch.exp(x * beta) - 1) / beta


def log_sum_exp2(x: Tensor, y: Tensor) -> Tensor:
    """
    compute log(exp(x) + exp(y))
    """
    m = torch.max(x, y).detach()
    return m + torch.log(torch.exp(x - m) + torch.exp(y - m))


def log_sub_exp2(x: Tensor, y: Tensor) -> Tensor:
    """
    compute log(exp(x) - exp(y))
    """
    m = torch.max(x, y).detach()
    return m + torch.log(torch.exp(x - m) - torch.exp(y - m))


def soft_clamp(x, min, max, beta=6):
    """soft clamp to the range min, max, with sharpness beta (larger is sharper)
       the slope at the middle of the range is always 1,
       beta = 4-6 and range (min=0, max=1) is similar to sigmoid(4*(x-0.5)), in particular at x=0.5 has value 0.5 -- the only threshold for quantization with grid of {0,1}, thus not affecting the forward deterministic path
       """
    # y = min + F.softplus(x - min, beta) - F.softplus(x - max, beta)
    if beta>20:
        y = x.clamp(min = min, max = max)
    y1 = min + F.softplus(x - min, beta)
    y2 = max - F.softplus(max - x, beta)
    y = torch.where(x < (max-min)/2, y1, y2)
    return y


def gauss_pdf(x):
    return 1 / math.sqrt(2 * math.pi) * torch.exp(-(x * x) / 2)


def gauss_cdf(x):
    return 0.5 + 0.5 * torch.erf(x / math.sqrt(2))

V_S = (math.pi ** 2) / 3  # variance of standard logistic distribution
def Phi_approx(x):
    """
    Approximate narmal_cdf with logistic
    """
    return torch.sigmoid(x * math.sqrt(V_S)) # by matching variance

def Phi_approx_inv(x):
    return torch.logit(x) / math.sqrt(V_S)


def unsqueeze_expand(input, dim, size_dim=1):
    """
    unsqueezes a dimension and expands this dimension to size size_dim
    """
    if dim < 0:
        dim = dim + input.dim() + 1
    s = list(input.shape)
    return input.unsqueeze(dim).expand(s[:dim] + [size_dim] + s[dim:])


class NumericalProblem(BaseException):
    pass


def isnan(x):
    return (x != x)

def check_var(x):
    """
    Check variance values are elligible
    """
    # print(torch.min(x.data))
    if all_checks:
        if not (x.data >= 0).all():
            raise NumericalProblem('variance is negative or nan')
            # code.interact(local=locals())
            #


def check_real(x):
    """
       Check for NaN / Inf
    """
    if all_checks:
        assert ((x == x).all()), "NaN encountered"
        assert (x.abs() < float('inf')).all(), "Inf encountered"
        # if not (x == x).all():
        #     # code.interact(local=locals())
        #     raise NumericalProblem('NaN encountered')
        # if not (x.abs() < float('inf')).all():
        #     # code.interact(local=locals())
        #     raise NumericalProblem('+-Inf encountered')
