# SPDX-License-Identifier: CC BY-NC-SA 4.0
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. Making code open source is essential for scientific
# transparency and reproducibility. Providing access to the code allows researchers to
# verify, replicate, and expand upon results, which supports scientific progress. Current
# top-tier conferences encourage opensource code for the purpose of reproducibility that
# is de-facto an established expectation of the research community. Opensource code as
# part of publication should not affect the performance of an invention both from a
# commercial and IP point of view.
import numpy as np
import torch.nn.functional as F
import math

from ..layer_base import *


def bitfield(n, bits):
    return [n >> i & 1 for i in range(bits)]


def signbits(C, zbits):
    """
    this is needed to encode a categorical sample as -1/1
    """
    A = []
    for c in range(C):
        a = np.array(bitfield(c, zbits))
        A.append(a)
    A = np.vstack(A)
    A = A*2 - 1
    return torch.tensor(A).float()


class Categorical(KWLayer):
    """
    Categorical discrete variable layer
    Input: logits
    Output: sample represented using the chosen embedding
    Methods should implement own forward in order to attach gradient estimaotrs
    """

    def __init__(self, C: int, embedding='signbits') -> None:
        """
        C -- number of classes
        """
        super().__init__()
        self.embedding_type = embedding
        self.C = C
        if embedding == 'onehot':
            self.embedding = None
        else:
            if embedding == 'signbits':
                zbits = math.log2(C)
                assert (zbits == math.floor(zbits)), "For signbits embedding the number of classes must be a power of two"
                A = signbits(C, int(zbits))
            elif embedding == 'integer':
                A = torch.arange(C).view(-1, 1).to(torch.float32)  # encode as 0,1,..C-1 integers
            # self.embedding = A # [C, embed_size]
            #self.embedding.requires_grad = False
            self.register_buffer('embedding', A, persistent=True)  # [C, embed_size]

    def embedding_size(self):
        if self.embedding_type == 'onehot':
            return self.C
        else:
            return self.embedding.shape[1]

    def embed(self, index: Tensor) -> Tensor:
        """ index [*] -- integer tensor to be used as index
        """
        # ST methods must override to make differentiable
        if self.embedding_type == 'onehot':
            y = F.one_hot(index, num_classes=self.C).to(dtype=torch.float32)  # [*,C], same dtype as p
        else:
            y = self.embedding[index, :]  # [*, embedding_size, embidding_dim]
            if self.embedding_type == 'integer':
                y = y.squeeze(-1)  # [*, embedding_size]
        return y

    def embed_argmax(self, logits: Tensor) -> Tensor:
        ii = torch.argmax(logits, dim=-1)
        return self.embed(ii)

    def sample(self, logits: Tensor) -> Tensor:
        """ logits [*, C] -- tensor of logits
        return: indices
        by default we return a non-diffenrentiable sample of indices
        """
        if True: # fallback, our implementation
            logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
            log_p = logits
            p = log_p.exp()
            U = p.new_empty(p.shape).uniform_()
            keys = U**(1/p)
            ii = torch.argmax(keys, dim=-1)
            return ii
        else:
            return torch.distributions.Categorical(logits=logits,  validate_args = False).sample()

    def expectation(self, *, logits=None, probs=None) -> Tensor: # * is to eat positional arguments
        """ x [*, C] -- tensor of logits
        output: mean embedding = \sum_{x} p(x)embedding(x)  [*, d]
        """
        assert (probs is not None or logits is not None)
        if probs is None:
            probs = torch.softmax(logits, dim=-1)  # [*, C]
        if self.embedding_type =='onehot':
            y = probs
        else:
            y = (probs.unsqueeze(-1) * self.embedding.view([1]*(probs.dim()-1) + list(self.embedding.shape))).sum(dim=-2)  # [*, embedding_size]
        return y

    # def forward(self, x: Tensor) -> Tensor:
    #     """ x [*, C] -- tensor of logits
    #     """
    #     # Methods must override to make differentiable
    #     index = self.sample(x)
    #     y = self.embed(index)
    #     return y
