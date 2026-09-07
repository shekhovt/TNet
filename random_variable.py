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
import numbers
from typing import Union, Dict, Callable

from torch import Tensor
import torch.nn.functional as F

all_checks = False
# all_checks = True


# some workarounds for Tensor and Tensor conversions
def to_tensor(x: Union[Tensor, numbers.Number]) -> Union[Tensor, None]:
    if x is None or torch.is_tensor(x):
        return x
    else:
        return torch.tensor(x)

def to_variable(x: Union[Tensor, numbers.Number]) -> Tensor:
    return to_tensor(x)


class TensorCat:
    """
        Holds a tuple of Tensors of equal sizes
    """

    @property
    def first(self) -> Tensor:
        return self.list[0]

    @first.setter
    def first(self, value):
        self.list[0] = value

    @property
    def second(self) -> Tensor:
        return self.list[1]

    @second.setter
    def second(self, value):
        self.list[1] = value

    def __init__(self, tensors):
        # self.tensor = torch.cat([t.view([1] + list(t.size())) for t in tensors], dim = 0)
        # self.list = [self.tensor.select(dim=0, index=i) for i in range(len(tensors))]
        self.list = list(tensors)

    # shape, concatentation, slicing, resizing
    def size(self, *args):
        return self.list[0].size(*args)

    @property
    def device(self):
        return self.list[0].device

    def stride(self, *args):
        return self.list[0].stride(*args)

    def dim(self) -> int:
        if hasattr(self.list[0], 'dim'):
            return self.list[0].dim()
        else:
            return 0

    @property
    def ndim(self) -> int:
        return self.dim()


    def cat(self, other, dim) -> 'TensorCat':
        return self.__class__([torch.cat(self.list[i], self.list[i], dim) for i in range(len(self.list))])
    
    def __getitem__(self, key):
        return self.__class__([self.list[i].__getitem__(key) for i in range(len(self.list))])

    def expand(self, sz) -> 'TensorCat':
        return self.__class__([self.list[i].expand(sz) for i in range(len(self.list))])

    def flatten(self, **kwargs) -> 'TensorCat':
        return self.__class__([x.flatten(**kwargs) for x in self.list])

    def clone(self) -> 'TensorCat':
        return self.__class__([self.list[i].clone() for i in range(len(self.list))])
    
    def new_zeros(self, sz) -> 'TensorCat':
        return self.__class__([self.list[i].new_zeros(sz) for i in range(len(self.list))])

    def new_ones(self, sz) -> 'TensorCat':
        return self.__class__([self.list[i].new_ones(sz) for i in range(len(self.list))])

    def zeros_like(self) -> 'TensorCat':
        return self.new_zeros(self.size())

    def view(self, sz) -> 'TensorCat':
        return self.__class__([self.list[i].view(sz) for i in range(len(self.list))])

    def contiguous(self) -> 'TensorCat':
        return self.__class__([self.list[i].contiguous() for i in range(len(self.list))])

    def detach(self) -> 'TensorCat':
        return self.__class__([self.list[i].detach() for i in range(len(self.list))])

    # statement arithmetics (+=)
    def __iadd__(self, b) -> 'TensorCat':
        if isinstance(b, TensorCat):
            [self.list[i].__iadd__(b.list[i]) for i in range(len(self.list))]
        else:  # Tensor, constant, etc
            [self.list[i].__iadd__(b) for i in range(len(self.list))]
        return self
            
    def __isub__(self, b) -> 'TensorCat':
        if isinstance(b, TensorCat):
            [self.list[i].__isub__(b.list[i]) for i in range(len(self.list))]
        else:  # Tensor, constant, etc
            [self.list[i].__isub__(b) for i in range(len(self.list))]
        return self
            
    def __imul__(self, b) -> 'TensorCat':
        if isinstance(b, TensorCat):
            [self.list[i].__imul__(b.list[i]) for i in range(len(self.list))]
        else:  # Tensor, constant, etc
            [self.list[i].__imul__(b) for i in range(len(self.list))]
        return self

    # arithmetics
    def __neg__(self):
        return self.__class__([self.list[i].__neg__() for i in range(len(self.list))])
    
    def __add__(self, b) -> 'TensorCat':
        if isinstance(b, TensorCat):
            return self.__class__([self.list[i].__add__(b.list[i]) for i in range(len(self.list))])
        else:  # Tensor, constant, etc
            return self.__class__([self.list[i].__add__(b) for i in range(len(self.list))])

    def __sub__(self, b) -> 'TensorCat':
        if isinstance(b, TensorCat):
            return self.__class__([self.list[i].__sub__(b.list[i]) for i in range(len(self.list))])
        else:  # Tensor, constant, etc
            return self.__class__([self.list[i].__sub__(b) for i in range(len(self.list))])

    def __mul__(self, b) -> 'TensorCat':
        if isinstance(b, TensorCat):
            return self.__class__([self.list[i].__mul__(b.list[i]) for i in range(len(self.list))])
        else:  # Tensor, constant, etc
            return self.__class__([self.list[i].__mul__(b) for i in range(len(self.list))])
        
    def __rmul__(self, b):
        if isinstance(b, TensorCat):
            return self.__class__([self.list[i].__rmul__(b.list[i]) for i in range(len(self.list))])
        else:  # Tensor, constant, etc
            return self.__class__([self.list[i].__rmul__(b) for i in range(len(self.list))])

    def __truediv__(self, b) -> 'TensorCat':
        if isinstance(b, TensorCat):
            return self.__class__([self.list[i].__truediv__(b.list[i]) for i in range(len(self.list))])
        else:  # Tensor, constant, etc
            return self.__class__([self.list[i].__truediv__(b) for i in range(len(self.list))])
        
    def __rtruediv__(self, b) -> 'TensorCat':
        if isinstance(b, TensorCat):
            return self.__class__([self.list[i].__rtruediv__(b.list[i]) for i in range(len(self.list))])
        else:  # Tensor, constant, etc
            return self.__class__([self.list[i].__rtruediv__(b) for i in range(len(self.list))])

    def fill_(self, val):
        for t in self.list:
            t.fill_(val)



class RandomVar(TensorCat):
    """
    Holds a pair of mean and variance
    """
    
    # @property
    # @abstractmethod
    @property
    def mean(self) -> Tensor:
        return self.list[0]

    @mean.setter
    def mean(self, value):
        self.list[0] = value

    @property
    def var(self) -> Tensor:
        return self.list[1]

    @var.setter
    def var(self, value):
        self.list[1] = value

    @property
    def std(self) -> Tensor:
        return self.var.sqrt()
        
    def __init__(self, mean=None, var=None):
        if isinstance(mean, (list, tuple)):
            TensorCat.__init__(self, mean)
        else:
            if var is None and mean is not None:
                var = torch.zeros_like(mean)
            TensorCat.__init__(self, [mean, var])
    
    # arithmetics
    def __add__(self, b) -> 'RandomVar':
        if isinstance(b, TensorCat):
            return self.__class__(self.mean + b.mean, self.var + b.var)
        else:  # Tensor, constant, etc
            return self.__class__(self.mean + b, self.var)
    
    def __sub__(self, b) -> 'RandomVar':
        if isinstance(b, TensorCat):
            return self.__class__(self.mean - b.mean, self.var + b.var)
        else:  # Tensor, constant, etc
            return self.__class__(self.mean - b, self.var)
    
    def __mul__(self, b) -> 'RandomVar':
        if isinstance(b, RandomVar):
            return self.__class__(self.mean * b.mean, self.var * b.var + self.var * square(b.mean) + square(self.mean) * b.var)
        elif isinstance(b, TensorCat):
            raise ValueError('Operation not defined')
        else:  # Tensor, constant, etc
            return self.__class__(self.mean * b, self.var * b * b)
    
    def __truediv__(self, s) -> 'RandomVar':
        if isinstance(s, TensorCat):
            raise ValueError('Operation not defined')
        else:  # Tensor, constant, etc
            return self.__class__(self.mean / s, self.var / (s * s))
    
    # sampling as Gaussian
    def sample(self, bounded=False) -> Tensor:
        """
        sample from Normal distribution (self.mean, self.var), differentiable in mean, var
        """
        n = torch.empty_like(self.mean)
        n.normal_()
        if bounded:
            n = torch.clamp(n, min=-3, max=3)
        if all_checks:
            if not (self.var.data >= 0).all():
                print(self.var.data.min())
            assert (self.var.data >= 0).all()
            assert (self.var.data == self.var.data).all()
            assert (self.var.data != float('inf')).all()
        y = self.mean + n * torch.sqrt(self.var + 1e-16)
        return y
    
       
    
def upgrade_to_RV(x) -> RandomVar:
    if isinstance(x, RandomVar):
        return RandomVar(x.mean, x.var)  # to avoid confusion whether RandomVar is constructed or pointer taken
    elif isinstance(x, Tensor) or isinstance(x, Tensor):
        return RandomVar(mean=to_variable(x), var=None)


def square(x):
    return x * x


def cat(seq, dim=0):
    if isinstance(seq[0], torch.Tensor):
        return torch.cat(seq, dim)
    elif isinstance(seq[0], RandomVar):
        return RandomVar(torch.cat([x.mean for x in seq], dim), torch.cat([x.var for x in seq], dim))
    else:
        raise ValueError


def shallow_copy(x):
    if isinstance(x, torch.Tensor):
        return x.view(x.size())
    elif isinstance(x, RandomVar):
        return RandomVar(shallow_copy(x.mean), shallow_copy(x.var))
    else:
        raise ValueError
