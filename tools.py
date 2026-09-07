# SPDX-License-Identifier: CC BY-NC-SA 4.0
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. Making code open source is essential for scientific
# transparency and reproducibility. Providing access to the code allows researchers to
# verify, replicate, and expand upon results, which supports scientific progress. Current
# top-tier conferences encourage opensource code for the purpose of reproducibility that
# is de-facto an established expectation of the research community. Opensource code as
# part of publication should not affect the performance of an invention both from a
# commercial and IP point of view.
import scipy
from scipy.stats import norm
from scipy.special import expit
from scipy.stats import special_ortho_group
import scipy.stats
from typing import Tuple, Callable, Any, Union
from types import SimpleNamespace
from torch.utils.data import DataLoader as Loader

import math
import copy
import numpy as np
import torchvision
import pickle

from dataclasses import dataclass, InitVar
import dataclasses

import os
import sys

# import torch
# from torch.utils.data.sampler import SubsetRandomSampler
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch import Tensor

from .functional import *
from . import device

global dev
dev = device.dev

# from nvitop import select_devices

# CUDA_VISIBLE_DEVICES = ','.join(select_devices(format='uuid', min_count=1, max_count=1, min_free_memory='10GiB', max_gpu_utilization=100))
# print('Settting CUDA_VISIBLE_DEVICES=', CUDA_VISIBLE_DEVICES)
# os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

# # query if we have GPU
# dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# t = torch.tensor([], device=dev)
# dev = t.device  # selects the current cuda device


def record(**kwargs):
    return SimpleNamespace(**kwargs)


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        torch.save(obj, output)
    # with open(filename, 'wb') as output:
    # pickle.dump(obj, output, pickle.DEFAULT_PROTOCOL)


def load_object(filename):
    try:
        res = torch.load(open(filename, "rb"), map_location=device.dev, weights_only=False)
    except RuntimeError as e:
        res = pickle.load(open(filename, "rb"))
    return res


def mkdir_recursive(path, wpublic:bool = False):
    sub_path = os.path.dirname(path)
    if len(sub_path) > 0 and not os.path.exists(sub_path):
        mkdir_recursive(sub_path, wpublic=wpublic)
    if not os.path.exists(path):
        try:
            os.mkdir(path)
            os.chmod(path, 0o777) # (rwx for all)
        except FileExistsError:  # it could have been created in the meantime by a parallel process
            pass


def force_path(file_name, wpublic = False):
    mkdir_recursive(os.path.dirname(file_name), wpublic=wpublic)


class dotdict(SimpleNamespace):
    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            assert(isinstance(args[0],dict))
            self.__dict__.update(args[0])
        else:
            assert(len(args) == 0)
        self.__dict__.update(kwargs)
        
    def keys(self):
        return self.__dict__.keys()
    
    def __eq__(self, other):
        return hash(self) == hash(other)
    
    def __hash__(self):
        return hash(str(self))
    
    def update(self, other:SimpleNamespace):
        for k in self.keys():
            if k in other.__dict__:
                self.__dict__[k] = other.__dict__[k]
    
    def __len__(self):
        return len(self.__dict__)
    
    def __getitem__(self, key):
        return self.__dict__[key]
    
    def items(self):
        return self.__dict__.items()
                
    @classmethod
    def from_dict(cls, other:SimpleNamespace):
        R = cls()
        R.update(other)
        return R

# class dotdict(dict):
#     """dot.notation access to dictionary attributes"""
#     """ For a more elaborate solution take a look at the EasyDict package https://pypi.org/project/easydict/ """

#     # def __init__(self):
#     #     super.__init__()
#     #     self.__dict__.locked = False
#     #
#     # def __getattr__(self, attr):
#     #     r = dict.get(self, attr)
#     #     if r is None and
#     __getattr__ = dict.get
#     __getitem__ = dict.get
#     __setattr__ = dict.__setitem__
#     __delattr__ = dict.__delitem__

#     def __getstate__(self): return self.__dict__

#     def __setstate__(self, d): self.__dict__.update(d)

#     # def lock(self, locked=True):
#     #     self.__dict__.locked = locked


class DataXY(torch.utils.data.Dataset):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, transform=None):
        # self.X = torch.Tensor(X).to('float').to(dev)
        # self.Y = torch.Tensor(Y).to(dev)
        self.X = X.to(device.dev)
        self.Y = Y.to(device.dev)
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return self.X.size(0)

    def set_transform(self, transform: torch.nn.Module):
        self.transform = transform

    def split(dataset, valid_size=0.1, random_seed=0):
        """
        creates a random split into train and validation
        """
        assert ((valid_size >= 0) and (valid_size <= 1)), "[!] valid_size should be in the range [0, 1]."
        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.ceil(valid_size * num_train))
        np.random.seed(random_seed)
        np.random.shuffle(indices)

        train_idx, val_idx = indices[split:], indices[:split]

        train_set = DataXY(dataset.X[train_idx], dataset.Y[train_idx], dataset.transform)
        val_set = DataXY(dataset.X[val_idx], dataset.Y[val_idx], dataset.transform)

        return train_set, val_set

    def fraction(self, fraction=0.1, random_seed=0):
        """
        Take a random fraction of the dataset containing a given fractino of all class examples
        """
        np.random.seed(random_seed)
        #
        K = self.Y.max().item() + 1
        X = []
        Y = []
        for k in range(K):
            mask = self.Y == k
            kX = self.X[mask]
            kY = self.Y[mask]
            num = kY.size(0)
            indices = list(range(num))
            split = int(np.ceil(fraction * num))
            np.random.shuffle(indices)
            idx = indices[:split]
            X += [kX[idx]]
            Y += [kY[idx]]
        return DataXY(torch.cat(X), torch.cat(Y), self.transform)
    
    def binarize(self):
        self.X = (self.X > 0.5).to(self.X)
        
def Dataset_to_XY(dataset):
    # pump through loader to get the transforms on the dataset applied
    loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)
    X = []
    Y = []
    for data, target in loader:
        X += [data]
        Y += [target]
    X = torch.cat(X)
    Y = torch.cat(Y)
    return DataXY(X, Y)

def to_tensor(x):
    if isinstance(x, Tensor):
        return x
    else:
        return torch.tensor(x)

class ImageFolderGPU(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        x = x.to(device.dev)
        y = to_tensor(y).to(device.dev)
        return x, y
    
# class loaderGPU(torch.utils.data.DataLoader):
    # def __next__(self):
        

class ScheduledArray:
    def __init__(self, min_ar, max_ar, target = None):
        self.min_ar = np.array(min_ar)
        self.max_ar = np.array(max_ar)
        if target is not None:
            self.current = target
        else:
            self.current = copy.deepcopy(self.min_ar)

    def schedule(self, alpha):
        """alpha in [0,1]"""
        a = self.min_ar*(1-alpha) + self.max_ar * alpha
        for i in range(len(self.current)):
            self.current[i] = a[i]

class ScheduledAttribute:
    def __init__(self, object, attribute, min_ar, max_ar):
        self.min_ar = np.array(copy.deepcopy(min_ar))
        self.max_ar = np.array(copy.deepcopy(max_ar))
        self.object = object
        self.attribute = attribute
        self.schedule(0.0)

    def schedule(self, alpha):
        """alpha in [0,1]"""
        a = (self.min_ar*(1-alpha) + self.max_ar * alpha)
        if isinstance(self.object.__dict__[self.attribute], (int, float)):
            self.object.__dict__[self.attribute] = a.item()
        elif isinstance(self.object.__dict__[self.attribute], tuple):
            self.object.__dict__[self.attribute] = tuple(a)
        else:
            for i in range(len(self.object.__dict__[self.attribute])):
                self.object.__dict__[self.attribute][i] = a[i]


class AScheduler:
    def __init__(self):
        self.register = []
    
    def new_array(self, min_ar, max_ar) -> np.ndarray:
        sa = ScheduledArray(min_ar, max_ar)
        self.register.append(sa)
        return sa.current
    
    def link_to(self, object, attribute, min_ar, max_ar):
        sa = ScheduledAttribute(object, attribute, min_ar, max_ar)
        self.register.append(sa)
    
    def schedule(self, alpha):
        for a in self.register:
            a.schedule(alpha)


@dataclass
class Datas:
    train_set:torch.utils.data.Dataset
    train_loader:Loader
    val_set:torch.utils.data.Dataset
    val_loader:Loader
    test_set:torch.utils.data.Dataset
    test_loader:Loader
    train_loader_test:Loader
    subclasses:np.ndarray | None = None
    tr_scheduler: AScheduler | None = None

# class Datas():
#     def __init__(self, train_set: DataXY, train_loader: Loader, val_set: DataXY, val_loader: Loader, test_set: DataXY, test_loader: Loader, train_loader_test: Loader):
#         self.train_set = train_set
#         self.train_loader = train_loader
#         self.val_set = val_set
#         self.val_loader = val_loader
#         self.test_set = test_set
#         self.test_loader = test_loader
#         self.train_loader_test = train_loader_test


class Tee(object):
    def __init__(self, name, mode='wt'):
        self.fname = name
        self.mode = mode

    def __enter__(self):
        self.file = open(self.fname, self.mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, *args):
        self.flush()
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()
        
    def isatty(self):
        return self.stdout.isatty()

def insert_BN(net: nn.ModuleList):
    for (i, l) in reversed(list(enumerate(net))):
        print(l.__class__.__name__)
        if isinstance(l, nn.Linear):  # or isinstance(l, nn.Conv2d):
            # print(l.weight.size())
            o_channels = l.weight.size(0)
            bn = nn.BatchNorm1d(num_features=o_channels, momentum=None)
            bn.to(l.weight)
            bn.weight.data.fill_(1.0)
            bn.bias.data.fill_(0)
            bn.train(True)
            net.insert(i + 1, bn)


# def remove_BN(net: nn.ModuleList):
#     for (i, bn) in reversed(list(enumerate(net))):
#         if isinstance(bn, nn.BatchNorm1d):  # or isinstance(l, nn.Conv2d):
#             l = net[i - 1]
#             assert (isinstance(l, nn.Linear))
#             mu = bn.running_mean
#             v = bn.running_var
#             std = ((v + bn.eps) ** 0.5)
#             s = (std * bn.weight)
#             b = (- mu / std + bn.bias)
#             # now BN(Wx + a) = (Wx + a)*s + b = W*s*x + a*s + b
#             l.weight.data = l.weight * s.view([-1, 1])  # multiply output dimensitons
#             l.bias.data = l.bias * s + b
#             del net[i]
