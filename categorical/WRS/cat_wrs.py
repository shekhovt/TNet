# SPDX-License-Identifier: CC BY-NC-SA 4.0
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. Making code open source is essential for scientific
# transparency and reproducibility. Providing access to the code allows researchers to
# verify, replicate, and expand upon results, which supports scientific progress. Current
# top-tier conferences encourage opensource code for the purpose of reproducibility that
# is de-facto an established expectation of the research community. Opensource code as
# part of publication should not affect the performance of an invention both from a
# commercial and IP point of view.
""" Weighted Random Sampling, Categorical Distribution Model
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
import torch.utils
import math
from typing import Callable
import numpy as np
import itertools
import scipy.stats

def bitfield(n, bits):
    return [n >> i & 1 for i in range(bits)]


class WRS:
    def __init__(self, k: int, device: torch.device):
        # super().__init__()
        self.k = k
        # precompute all possible binary masks for subsets of set {1,..,k}
        A = []
        for i in range(2**k):
            a = np.array(bitfield(i, k))
            A.append(a)
        A = np.vstack(A)
        A = torch.tensor(A).to(device=device, dtype=torch.bool)  # mask
        self.subsets_not = torch.logical_not(A)  # [2^k, k]
        self.signs = (-1)**(A.sum(dim=1))  # [2^k]

    def sample_indices(self, log_p: Tensor, n_samples: int) -> Tensor:
        """ 
        ### Draw n_samples of k subsets without replacement from Categorical
        p [N] -- cat probability

        Sampling scheme:
         top-k(log(p) + G), G~Gumbel(0,1), F_G(x) = exp(-exp(-x)) 
        =top-k( log(p) -log(-log(U)) ) # G~-log(-log(U))
        =top-k( p/-log(U) ) # exp is monotone
        =top-k( log(U)/p ) # inverse negative is monotone
        =top-k( log(U^{1/p}) )
        =top-k( U^{1/p} )
        """
        p = log_p.exp()
        # G = torch.distributions.gumbel.Gumbel(torch.zeros_like(p), torch.ones_like(p)).sample((n_samples,))
        # keys = p.log() + G
        U = p.new_empty(n_samples, p.shape[0]).uniform_()
        # G = -torch.log(-torch.log(U))
        # keys = p/(-torch.log(U))
        # keys = torch.log(U)/p
        # keys = torch.log(U**(1/p))
        keys = U**(1/p)
        # sort keys by descent
        (_, ii) = torch.topk(keys, k=self.k, dim=-1, largest=True)  # [n_samples, k]
        
        # DEBUG: draw-by-draw
        # N = p.shape[0]
        # cp = p.view(1, N).expand(n_samples, N) # [n_samples, N]
        # ii = []
        # p_sample = torch.ones(n_samples)
        # for k in range(self.k):
        #     i = torch.distributions.categorical.Categorical(probs=cp).sample()
        #     p_sample *= cp.gather(1,i.unsqueeze(-1)).flatten()
        #     ii += [i]
        #     vii = torch.vstack(ii).T
        #     cp = cp.scatter(1, i.view(-1, 1).expand(n_samples, N), torch.zeros(n_samples, N))
        #     cp = cp/cp.sum(dim=1,keepdim=True)
        # ii = vii
        return ii

    def ordered_log_P(self, cat_logits: Tensor) -> Tensor:
        """ Compute probability of ordered sample
            input:
            cat_logits [n_samples, k] -- logits of base categorical log probabilities of the sample
            output:
            log_P [n_samples] -- log probability of drawing such ordered sample
        """
        cat_p = cat_logits.exp()
        n_samples = cat_logits.shape[0]
        K = cat_logits.shape[1]
        Z = cat_logits.new_ones(n_samples)
        # logZ = cat_logits.new_zeros(n_samples) # todo: logZ, vectorize
        log_P = cat_logits.sum(dim=1)
        for k in range(K):
            log_P = log_P - torch.log(Z)
            Z = Z - cat_p[:, k]
            # Z = log_sub_exp(Z, )
        return log_P

    def unordered_log_P(self, cat_logits: Tensor) -> Tensor:
        """ Compute probability of unordered sample
            input:
            cat_logits [n_samples, k] -- base categorical log probabilities of the sample
            output:
            log_P [n_samples] -- log probability of drawing such unordered sample
        """
        K = cat_logits.shape[1]
        cat_p = cat_logits.exp()  # [n_sample, k]
        # Probability of the unordered sample
        Sum_S = cat_p.sum(dim=-1)  # [n_samples]
        # num = (cat_p[None, :, :]*self.subsets[:, None, :]).sum(dim=-1)  # [2**k, n_samples]
        Sum_notC = (cat_p[None, :, :]*self.subsets_not[:, None, :]).sum(dim=-1)  # [2**k, n_samples]
        ratio = self.signs[:, None]/(1 - Sum_notC)  # [2**k, n_samples]
        P = ratio.sum(dim=0)*(1-Sum_S)  # [n_samples]
        return torch.log(P)

    def RLOO_surrogate(self, sample: Tensor, f: Callable, log_P: Tensor) -> Tensor:
        """
        sample [n_samples, k, d] -- minimal samples
        f -- objective fucntion to evaluate for each drawn sample with the specs:
            f(points)->loss:
            points [n_samples, k, D] -- n_samples of k points in D dimensions
            loss [n_samples] -- loss for each sample
        log_P [n_samples] log probabilitiy of sample
        """
        # call the function with the sample
        n_samples = sample.shape[0]
        ff = f(sample)  # [n_samples]
        assert ff.dim() == 1
        assert ff.shape[0] == n_samples
        # form the RLOO gradient
        ffmean = ff.mean(dim=0, keepdim=True)
        df = (ff - ffmean)/(n_samples-1)  # [n_samples]
        loss = ffmean + (df.detach()*log_P).sum()  # ffmean is differentiable in point cordinates, log_P is differentiable in logits
        return loss

    def RLOO_ordered(self, points: Tensor, logits: Tensor,  n_samples: int, f: Callable[[Tensor], Tensor]) -> Tensor:
        """
        Weighted Random Sampling without replacement, with REINFORCE Leave One Out gradinet estimator
        Input:
        points [N, D] -- N correspondances in D dimensions
        logits [N] -- logits of importance weights
        k -- minimal sample size
        n_sample: number of samples to draw, must be >=2
        f -- objective fucntion to evaluate for each drawn sample with the specs:
            f(points)->loss:
            points [n_samples, k, D] -- n_samples of k points in D dimensions
            loss [n_samples] -- loss for each sample
        """
        assert n_samples >= 2
        assert points.shape[0] == logits.shape[0]
        N = points.shape[0]
        # normalize logits
        logits = logits - torch.logsumexp(logits, dim=-1)
        # draw n_samples of k subsets without replacement
        ii = self.sample_indices(logits, n_samples)
        # select points
        sample = points[ii, :]  # [n_samples, k, D]
        # base logits of the sample:
        l_sample = logits[ii]  # [n_sample, k]
        # Compute log-probability of the sample
        log_P = self.ordered_log_P(l_sample)
        return self.RLOO_surrogate(sample, f, log_P)

    def RLOO_unordered(self, points: Tensor, logits: Tensor,  n_samples: int, f: Callable[[Tensor], Tensor]) -> Tensor:
        """
        Weighted Random Sampling without replacement, with REINFORCE Leave One Out gradinet estimator
        Input:
        points [N, D] -- N correspondances in D dimensions
        logits [N] -- logits of importance weights
        n_sample: number of samples to draw, must be >=2
        f -- objective fucntion to evaluate for each drawn sample with the specs:
            f(points)->loss:
            points [n_samples, k, D] -- n_samples of k points in D dimensions
            loss [n_samples] -- loss for each sample
            !!! f must be permutation-invariant
        """
        assert n_samples >= 2
        assert points.shape[0] == logits.shape[0]
        N = points.shape[0]
        # normalize logits
        logits = logits - torch.logsumexp(logits, dim=-1)
        # draw n_samples of k subsets without replacement
        ii = self.sample_indices(logits, n_samples)
        # select points
        sample = points[ii, :]  # [n_samples, k, D]
        # base logits of the sample:
        l_sample = logits[ii]  # [n_sample, k]
        log_P = self.unordered_log_P(l_sample)
        return self.RLOO_surrogate(sample, f, log_P)

    def exhaustive_unordered(self, points, logits, f):
        """ gradient by exhaustive combinations
        debug purpose only
        !!! f must be permutation-invariant
        """
        N = points.shape[0]
        # normalize logits
        logits = logits - torch.logsumexp(logits, dim=-1)
        # all possible minimal samples
        ii = torch.tensor(np.stack(list(itertools.combinations(np.arange(N), self.k))))  # [n_sample, k]
        sample = points[ii, :]  # [n_samples, k, D]
        l_sample = logits[ii]  # [n_sample, k]
        log_P = self.unordered_log_P(l_sample)  # [n_sample]
        ff = f(sample)
        F = (ff*torch.exp(log_P)).sum()
        return F
        

# unit tests

def test_probs():
    np.random.seed(0)
    torch.manual_seed(1)
    #
    num_points = 6
    sample_size = 4
    logits = torch.rand([num_points])*2-1
    # check prob of unordered draw
    logits = logits - logits.logsumexp(dim=0)
    sampler = WRS(sample_size, logits.device)
    #
    for t in range(1):
        ii = sampler.sample_indices(logits, 1)
        # ii = torch.tensor([0, 2]).view(1, -1)
        # emperical probs
        n_samples = 100000
        mult_ii = sampler.sample_indices(logits, n_samples)
        # check how often ii is in mult_ii, unordered:
        ordered_freq = torch.all(mult_ii == ii, dim=-1).float().mean()
        sort_ii, _ = torch.sort(ii)
        sort_mult_ii, _ = torch.sort(mult_ii, dim=1)
        unordered_freq = torch.all(sort_mult_ii == sort_ii, dim=-1).float().mean()
        # analytic
        cat_logits = logits[ii]
        log_P1 = sampler.unordered_log_P(cat_logits)
        pii = torch.vstack([torch.tensor(p) for p in itertools.permutations(ii.flatten().cpu().detach().numpy())])
        plogits = logits[pii]
        log_P2 = sampler.ordered_log_P(plogits)
        print('ordered freq:', ordered_freq.item())
        print('ordered comp:', log_P2[0].exp().item())
        #
        print('unordered freq  :', unordered_freq.item())
        log_P2 = torch.logsumexp(log_P2, dim=0)
        print('sum over ordered:', log_P2.exp().item())
        print('unordered comput:', log_P1.exp().item())
    print("Test0 completed")


def test_grad_exists():
    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    t = torch.tensor([], device=dev)
    dev = t.device  # elects the current cuda device
    
    num_samples = 10
    num_points = 3
    sample_size = 2
    coordinates = 4

    matches = torch.rand([num_points, coordinates], device=dev)
    matches.requires_grad = True
    logits = torch.rand([num_points], device=dev)*4-2
    logits.requires_grad = True
    target = torch.rand([coordinates], device=dev)  # fake target for grad testing

    def loss_f(min_matches):
        return torch.norm(min_matches.mean(dim=1)-target[None, :], dim=(1))  # min_matches

    sampler = WRS(sample_size, matches.device)
    loss = sampler.RLOO_unordered(matches, logits, num_samples, loss_f)
    #
    loss.backward()
    assert matches.grad is not None
    assert logits.grad is not None
    print(logits.grad.shape)
    assert (logits.grad.dim() == 1)
    assert (logits.grad.shape[0] == num_points)
    print("Test1 passed")


def test_grad_correct():
    np.random.seed(0)
    torch.manual_seed(1)
    # dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dev = torch.device('cpu')
    t = torch.tensor([], device=dev)
    dev = t.device  # elects the current cuda device

    num_samples = 3
    num_points = 25
    sample_size = 3
    coordinates = 4

    matches = torch.rand([num_points, coordinates], device=dev)
    matches.requires_grad = True
    logits = torch.rand([num_points], device=dev)-0.5
    logits.requires_grad = True

    target = torch.rand([coordinates], device=dev)

    def loss_f(min_matches):
        return torch.norm(min_matches.mean(dim=1)-target[None, :], dim=(1))  # min_matches

    sampler = WRS(sample_size, matches.device)
    # torch.autograd.set_detect_anomaly(True)
    
    print("=======Test2=======")
    
    def grad0():
        logits.grad = None
        loss = sampler.exhaustive_unordered(matches, logits, loss_f)
        loss.backward()
        return logits.grad.detach(), loss.mean().item()

    def grad1():
        logits.grad = None
        loss = sampler.RLOO_ordered(matches, logits, num_samples, loss_f)
        loss.backward()
        return logits.grad.detach(), loss.mean().item()

    def grad2():
        logits.grad = None
        loss = sampler.RLOO_unordered(matches, logits, num_samples, loss_f)
        loss.backward()
        return logits.grad.detach(), loss.mean().item()

    g0, l0 = grad0()

    T = 10000
    torch.manual_seed(6)
    G1 = []
    G2 = []
    L1 = []
    L2 = []
    for t in range(T):
        g1, l1 = grad1()
        g2, l2 = grad2()
        G1 += [g1]
        G2 += [g2]
        L1 += [l1]
        L2 += [l2]
    G1 = torch.vstack(G1)
    G2 = torch.vstack(G2)
    L1 = torch.tensor(L1)
    L2 = torch.tensor(L2)

    # print(G1.shape)

    m1 = G1.mean(dim=0)
    m2 = G2.mean(dim=0)
    v1 = G1.var(dim=0)
    v2 = G2.var(dim=0)

    # squared bias estimate
    b1 = ((m1 - g0)**2) - v1/T
    b2 = ((m2 - g0)**2) - v2/T
    b0 = ((m1 - m2)**2) - v1/T - v2/T

    # print('grad difference', (m1-m2).cpu().numpy())
    # print('squared bias:', b.mean().cpu().numpy())
    print('grad1   sbias:', b1.mean().cpu().numpy())
    print('grad2   sbias:', b2.mean().cpu().numpy())
    print('grad1-2 sbias:', b0.mean().cpu().numpy())
    print('grad1  std:', (v1.mean()**0.5).cpu().numpy())
    print('grad2  std:', (v2.mean()**0.5).cpu().numpy())
    print('grad1  var/n_sampes:', (v1.mean()/T).cpu().numpy())
    print('grad2  var/n_sampes:', (v2.mean()/T).cpu().numpy())

    # def statistic(sample):
    #     return (((sample.mean() - g0)**2) - sample.var()/num_samples).mean()
    # scipy.stats.bootstrap((G1,), statistic)

    print("Test2 completed")


if __name__ == "__main__":
    # test_probs()
    test_grad_exists()
    test_grad_correct()
