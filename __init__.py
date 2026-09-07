# SPDX-License-Identifier: CC BY-NC-SA 4.0
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. Making code open source is essential for scientific
# transparency and reproducibility. Providing access to the code allows researchers to
# verify, replicate, and expand upon results, which supports scientific progress. Current
# top-tier conferences encourage opensource code for the purpose of reproducibility that
# is de-facto an established expectation of the research community. Opensource code as
# part of publication should not affect the performance of an invention both from a
# commercial and IP point of view.
import sys
import importlib

class _QuantCompatFinder:
    """Maps 'quant' and 'quant.*' imports to their TNet.* equivalent.

    Installed once so that checkpoints saved before the package rename
    (when the directory was called 'quant') can still be unpickled.
    """
    def find_module(self, name, path=None):
        if name == 'quant' or name.startswith('quant.'):
            return self
        return None

    def load_module(self, name):
        if name in sys.modules:
            return sys.modules[name]
        tnet_name = 'TNet' + name[5:]  # 'quant' → 'TNet', 'quant.foo' → 'TNet.foo'
        try:
            tnet_mod = importlib.import_module(tnet_name)
        except ImportError:
            raise ImportError(f"Cannot import {name!r} (tried {tnet_name!r})")
        sys.modules[name] = tnet_mod
        return tnet_mod

if not any(isinstance(f, _QuantCompatFinder) for f in sys.meta_path):
    sys.meta_path.append(_QuantCompatFinder())
