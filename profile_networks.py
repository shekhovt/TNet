# SPDX-License-Identifier: CC BY-NC-SA 4.0
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. Making code open source is essential for scientific
# transparency and reproducibility. Providing access to the code allows researchers to
# verify, replicate, and expand upon results, which supports scientific progress. Current
# top-tier conferences encourage opensource code for the purpose of reproducibility that
# is de-facto an established expectation of the research community. Opensource code as
# part of publication should not affect the performance of an invention both from a
# commercial and IP point of view.
"""Profile one network's compute, memory and energy cost -- no data, no GPU, no checkpoint.

This is the front door to :mod:`TNet.energy_ECML`, the energy model published with the ECML-PKDD
2026 paper.  It builds the network from a command line, describes it as layer geometry plus
quantizer widths, and prices it::

    python profile_networks.py                                      # the default configuration
    python profile_networks.py --net 'BiNealNet(m=1)' --method ST -A 2 -W 2
    python profile_networks.py --layers --net 'QResNet18(gate=Tower8s)' --method ST -A 2 -W 2

Every argument except ``--layers`` is passed through to the training command-line parser, so any
configuration that trains can be profiled.  For the whole comparison table -- every method, the
plots, and the diff against the published table -- use the package's own report instead::

    python -m TNet.energy_ECML.report --all

This file was once a second, drifting copy of ``energy_ECML/evaluate_networks.py``.  That
file is now the frozen reference the model is tested against, and nothing duplicates it; see
``energy_ECML/README.md``.
"""

import relimport

import shlex
import sys

from .energy_ECML import evaluate
from .energy_ECML.adapters.native import specs_from_cfg

DEFAULT_CFG = "--net 'QResNet18(gate=Tower8s,Dilation=True)' --method ST-det -A 4 -W 4"


def profile(cfg_string: str = DEFAULT_CFG, per_layer: bool = False):
    """Print the energy breakdown of one configuration and return it."""
    specs = specs_from_cfg(cfg_string)
    e = evaluate(specs, name=cfg_string)
    r = e.as_row()
    h, w = specs[0].in_shape[2], specs[0].in_shape[3]

    print(cfg_string)
    print(f"  {len(specs)} layers, input {h}x{w}")
    print(f"  weights    {r['MB_weights']:8.2f} MB   {r['uJ_weights']:10.0f} uJ")
    print(f"  features   {r['MB_features']:8.2f} MB   {r['uJ_features']:10.0f} uJ")
    print(f"  compute    {'':8}      {r['uJ_CEE']:10.0f} uJ")
    print(f"  total      {'':8}      {r['uJ_TEE']:10.0f} uJ")

    if per_layer:
        print()
        print(f"  {'layer':<24} {'in shape':>18} {'out shape':>18} {'uJ':>10}")
        for l in e.layers:
            i, o = l.spec.in_shape, l.spec.out_shape
            print(f"  {l.name[:24]:<24} {str(tuple(i)):>18} {str(tuple(o)):>18} "
                  f"{l.TEE / 1e6:10.2f}")
    return e


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    per_layer = False
    if "--layers" in argv:
        argv.remove("--layers")
        per_layer = True
    cfg = " ".join(shlex.quote(a) for a in argv) if argv else DEFAULT_CFG
    profile(cfg, per_layer=per_layer)


if __run__:
    main()
