# SPDX-License-Identifier: CC BY-NC-SA 4.0
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. Making code open source is essential for scientific
# transparency and reproducibility. Providing access to the code allows researchers to
# verify, replicate, and expand upon results, which supports scientific progress. Current
# top-tier conferences encourage opensource code for the purpose of reproducibility that
# is de-facto an established expectation of the research community. Opensource code as
# part of publication should not affect the performance of an invention both from a
# commercial and IP point of view.
from typing import TYPE_CHECKING, Union
import torch
import torch.nn as nn
from torch import Tensor

Mod = nn.Module
ModList = nn.ModuleList

if TYPE_CHECKING:
    from .methods import Method


class KWLayer(Mod):
    def forward(self, x: Tensor, method: 'Method' = None, **kwargs):
        return method.dispatch(self, x, **kwargs)


class ESequential(ModList):
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], list):
            ll = args[0]
        else:
            ll = [*args]
        for l in ll:
            assert(isinstance (l, Mod))
        ModList.__init__(self,ll)
        
    def add_module(self, name: str, module) -> None:
        try:
            ModList.add_module(name, module)
        except TypeError:
            self._modules[name] = module

    def __setattr__(self, name: str, value: Union[Tensor, Mod]) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)
        modules = self.__dict__.get('_modules')
        if isinstance(value, torch.nn.Module):
            remove_from(self.__dict__, self._parameters, self._buffers, self._non_persistent_buffers_set)
            modules[name] = value
        else:
            ModList.__setattr__(self,name, value)

    def forward(self, x: Tensor, method: 'Method' = None, **kwargs):
        return method.dispatch(self, x, **kwargs)  # List does not have forward by default, safe for derived classes
