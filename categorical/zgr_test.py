# SPDX-License-Identifier: CC BY-NC-SA 4.0
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. Making code open source is essential for scientific
# transparency and reproducibility. Providing access to the code allows researchers to
# verify, replicate, and expand upon results, which supports scientific progress. Current
# top-tier conferences encourage opensource code for the purpose of reproducibility that
# is de-facto an established expectation of the research community. Opensource code as
# part of publication should not affect the performance of an invention both from a
# commercial and IP point of view.
# %%
import torch
from torch import Tensor
import torch.nn.functional as F

import matplotlib.pyplot as plt

def ZGR_binary(logits:Tensor, x:Tensor=None, MD = False)->Tensor:
    """Returns a Bernoulli sample for given logits with ZGR = DARN(1/2) gradient
    Input: logits [*]
    x: (optional) binary sample to use instead of drawing a new sample. [*]
    MD = True: the Jacobian dp/dlogit is skipped, good for Mirror Descent
    Output: binary samples with ZGR gradient [*], dtype as logits
    """
    p = torch.sigmoid(logits)
    if MD:
        p = p.detach() + logits - logits.detach()
    if x is None:
        x = p.bernoulli()
    J = (x * (1-p) + (1-x)*p )/2
    return x + J.detach()*(logits - logits.detach()) # value of x with J on backprop to logits


import itertools

def iterate_binary_masks(n):
  """
  Iterates over all binary masks of length n.

  Args:
    n: The length of the binary masks.

  Yields:
    A tuple representing a binary mask.
  """
  for mask in itertools.product([0, 1], repeat=n):
    yield torch.tensor(mask, dtype=torch.float32)


def ZGR_binary_vector(logits:Tensor, x:Tensor=None, vector=False)->Tensor:
    """Returns a Bernoulli sample for given logits with ZGR = DARN(1/2) gradient
    Input: logits [B, *] -- B - batch size
    x: (optional) binary sample to use instead of drawing a new sample. [*]
    Output: binary samples with ZGR gradient [B, *], dtype as logits
    """
    p = torch.sigmoid(logits)
    if x is None:
        x = p.bernoulli().detach()
    dx_ST = p
    # log probability of x
    logpx = torch.log(p) * x + torch.log(1-p)*(1-x)
    if vector:
        assert(logits.dim()>1)
        B = x.shape[0]
        logpv = logpx.view([B,-1]).sum(dim=-1)
        logpv = logpv.view([B] + [1]*(x.dim()-1)) # log-probability of the whole vector
        dx_RE = (x - p.detach()) * logpv
    else:
        dx_RE = (x - p.detach()) * logpx # log-probability of each coordinate independently
    dx = (dx_ST + dx_RE) / 2
    return x + (dx - dx.detach())


def ZGR_categorical(logits, index=None):
    """Returns a categorical sample from Categorical softmax(logits) (over axis=-1) as a
    one-hot vector, with ZGR gradient.
    
    Input: 
    logits [*, C], where C is the number of categories
    index: (optional) categorical sample to use instead of drawing a new sample. [*]
    
    Output: categorical samples with ZGR gradient [*,C] encoded as one_hot
    """
    # using surrogate loss
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    p = logp.exp()  # [*, C]
    dx_ST = p  # [*,C]
    if index is None:
        index = torch.distributions.categorical.Categorical(probs=p, validate_args=False).sample()  # [*]
    else:
        index = index
    num_classes = logits.shape[-1]
    y = F.one_hot(index, num_classes=num_classes).to(p)  # [*,C], same dtype as p
    # logpx = logp.gather(-1, index.unsqueeze(-1)) # [*,1] -- log probability of drawn sample
    logpx = ((logp * y).sum(dim=-1)).unsqueeze(-1) # [*, 1] -- log probability of drawn sample
    dx_RE = (y - p.detach()) * logpx
    dx = (dx_ST + dx_RE) / 2
    return y + (dx - dx.detach())


def test1():
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
test1()

#%%

n = 10
# make a quadratic loss function in R^n
torch.manual_seed(0)
L = torch.randn(n,n)
Q = L@L.T # symmetric positive defininte matrix
m = (torch.rand(n)>0).float() # optimum will be at a binary point m
def loss(x):
    # quadratic loss function
    l = torch.einsum('...i,ij,...j->...',x-m,Q,x-m)
    # l = torch.einsum('...i,ij->...',x-m, Q) # linear
    return l

def reinforce(eta, x, loss):
    eta.grad.zero_()
    logits = eta
    p = torch.sigmoid(logits)
    # log probability of x
    logpx = torch.log(p* x + (1-p)*(1-x))
    if x is None:
        x = p.bernoulli()
    l = loss(x)
    gf = l * logpx
    f = l + (gf - gf.detach())
    return f


def E_loss(eta):
    B = 1000
    s = eta.shape
    eta = eta.unsqueeze(0).expand([B] + list(s))
    El = loss(ZGR_binary(eta)).mean()
    return El

def var(eta, vector = False):
    B = 1000
    s = eta.shape
    eta = eta.unsqueeze(0).expand([B] + list(s)).clone().detach().requires_grad_(True)
    l = loss(ZGR_binary(eta, vector = vector)).mean()
    l.backward()
    g = eta.grad
    return g.var()

eta = torch.zeros(n, requires_grad=True) # Bernoulli logits

print(var(eta, False))
print(var(eta, True))

opt = torch.optim.Adam([eta], lr=1e-1)

epochs = 1000
for it in range(epochs):
    opt.zero_grad()
    l = loss(ZGR_binary(eta.unsqueeze(0), vector=True))
    l.backward()
    opt.step()
    if it %100 == 0:
        El = E_loss(eta)
        print(f'L={El:3.2f}')


# %%

X = torch.stack([*iterate_binary_masks(n)])
print(X.shape)

PhiT = X

def loss_cat(y):
    return loss(y @ PhiT)

def E_loss1(eta):
    p = torch.sigmoid(eta).unsqueeze(0)
    logpX = torch.log(p* X + (1-p)*(1-X)).sum(dim=-1)
    l = loss(X).detach()
    El = (l * logpX.exp()).sum()
    return El

def grad_GT(eta):
    eta.grad = None
    l = E_loss1(eta)
    l.backward()
    return eta.grad

# # grad at x

# opt.zero_grad()
# p = torch.sigmoid(eta)
# x = p.bernoulli().detach()
# l = loss(ZGR_binary(eta.unsqueeze(0), x=x, vector=False))
# l.backward()
# print(eta.grad)

# opt.zero_grad()
# l = loss(ZGR_binary(eta.unsqueeze(0), x=x, vector=True))
# l.backward()
# print(eta.grad)

def eta2P(eta):
    p = torch.sigmoid(eta).unsqueeze(0)
    logpX = torch.log(p* X + (1-p)*(1-X)).sum(dim=-1)
    P = logpX.exp()
    return P

eta.grad = None
P = eta2P(eta)
L = loss(X)
EL = (L*P).sum()
EL.backward()
print('E grad GT:', eta.grad)

eta.grad.zero_()
P = eta2P(eta)
L = loss(ZGR_binary(eta.unsqueeze(0), x = X, vector=False))
(L*P.detach()).sum().backward()
print('E grad ZGR:', eta.grad)

eta.grad.zero_()
P = eta2P(eta)
L = loss(ZGR_binary(eta.unsqueeze(0), x = X, vector=True))
(L*P.detach()).sum().backward()
print('E grad ZGR-v:', eta.grad)

index = torch.arange(X.shape[0])
eta.grad.zero_()
P = eta2P(eta)
Y = ZGR_categorical(P.log().unsqueeze(0),index=index)
L = loss_cat(Y) # [2**n]
# print(L)
EL = (L*P.detach()).sum()
# print(EL.item())
EL.backward()
print('E grad ZGR-cat:', eta.grad)

# # %%
# EG = 0
# for index in range(2**n):
#     eta.grad.zero_()
#     P = eta2P(eta)
#     y = ZGR_categorical(P.log(), index=torch.tensor(index))
#     py = P[index]
#     l = loss_cat(y)
#     l.backward()
#     print(eta.grad)
#     EG += (eta.grad * py).detach()

# print(EG)


# # %%
# EG = 0
# Ep = 0
# for x in iterate_binary_masks(n):
#     eta.grad.zero_()
#     p = eta.sigmoid()
#     y = ZGR_binary(eta.unsqueeze(0), x = x.unsqueeze(0), vector=True)
#     assert((x - y).max().abs() < 1e-6)
#     py = (p * x + (1-p)*(1-x)).prod(dim=-1)
#     l = loss(y)
#     l.backward()
#     print(eta.grad)
#     EG += (eta.grad * py).detach()
#     Ep += py

# print(EG)
# print(Ep)

# %%
