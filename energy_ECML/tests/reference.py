# SPDX-License-Identifier: CC BY-NC-SA 4.0
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. Making code open source is essential for scientific
# transparency and reproducibility. Providing access to the code allows researchers to
# verify, replicate, and expand upon results, which supports scientific progress. Current
# top-tier conferences encourage opensource code for the purpose of reproducibility that
# is de-facto an established expectation of the research community. Opensource code as
# part of publication should not affect the performance of an invention both from a
# commercial and IP point of view.

# --- What this file is -----------------------------------------------------------------------
# Test helper. Runs the untouched reference implementation over a given layer list and parses
# its printed totals back into exact integers, which is the only way to compare against it
# without editing it.

"""Run the untouched reference model and parse its printed totals.

``energy_ECML/evaluate_networks.py:net_calc_energy_stats`` prints and returns nothing, so the only
way to compare against it without editing it is to capture stdout.  This module does that, and
converts a :class:`~TNet.energy_ECML.spec.LayerSpec` list into the ``FusedConv`` list the reference
expects.

**Precision.** The reference prints each total twice: once as ``int(...)`` and once rounded to
two decimals.  We parse the integer form, so compute energy is compared at 1 pJ resolution --
about 1e-8 relative on a 150 uJ figure -- and every memory figure is compared exactly, being an
integer number of bits.
"""

from __future__ import annotations

import contextlib
import io
import re

import torch
import torch.nn as nn

from ..spec import LayerSpec

__all__ = ["fused_from_specs", "run_reference", "REFERENCE_KEYS"]

_PAT = {
    "compute_pJ": re.compile(r"^Total compute energy: (\d+),"),
    "bits_written": re.compile(r"^Total mem written: (\d+) b"),
    "bits_read_weight": re.compile(r"^  Total mem read weights: (\d+) b"),
    "bits_read_act": re.compile(r"^  Total mem read activations: (\d+) b"),
    "bits_activations": re.compile(r"^Total mem activations: (\d+) b"),
    "max_read_act": re.compile(r"^Max activations in: (\d+) b"),
    "max_written_act": re.compile(r"^Max activations out: (\d+) b"),
}

REFERENCE_KEYS = tuple(_PAT)


def fused_from_specs(specs: list[LayerSpec]):
    """Build the ``FusedConv`` list the reference prices from a ``LayerSpec`` list.

    The reference has a single ``K_in``/``K_W`` per layer, so the *operand* widths are used --
    that is the field the reference derives both its memory and its compute from, and comparing
    under ``K_in_stored == K_in_operand`` is the only comparison that is well posed.
    """
    from ..evaluate_networks import FusedConv
    from ...layers import ScaleBias

    rr = []
    for s in specs:
        if s.K_in_stored != s.K_in_operand or s.K_W_stored != s.K_W_operand:
            raise ValueError(
                f"{s.name}: the reference cannot express independent stored/operand widths; "
                "see tests/test_bitwidths.py for how that case is checked instead"
            )
        conv = nn.Conv2d(
            s.in_shape[1], s.weight_shape[0],
            (s.weight_shape[2], s.weight_shape[3]),
            stride=s.stride, padding=0, bias=False, groups=s.groups,
        )
        conv.weight = nn.Parameter(torch.zeros(*s.weight_shape))
        pool = None
        p = s.pooling
        if p is not None:
            if p.kind == "max":
                pool = nn.MaxPool2d(kernel_size=p.kernel_size, stride=p.stride)
            elif p.kind == "avg":
                pool = nn.AvgPool2d(kernel_size=p.kernel_size, stride=p.stride)
            else:
                pool = nn.AdaptiveAvgPool2d(
                    p.output_size[0] if p.output_size_was_int else tuple(p.output_size)
                )
        f = FusedConv(conv, ScaleBias(s.weight_shape[0]), pool)
        f.in_size = tuple(s.in_shape)
        f.conv_out_size = tuple(s.out_shape)
        f.K_in, f.K_W, f.K_out = s.K_in_operand, s.K_W_operand, s.K_out_stored
        rr.append(f)
    return rr


def run_reference(specs_or_fused, name: str = "oracle") -> dict:
    """Return the reference's totals as a dict of exact integers."""
    from ..evaluate_networks import net_calc_energy_stats

    rr = specs_or_fused
    if rr and isinstance(rr[0], LayerSpec):
        rr = fused_from_specs(rr)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        net_calc_energy_stats(name, rr)
    out = {}
    for line in buf.getvalue().splitlines():
        for k, p in _PAT.items():
            if k not in out:
                m = p.match(line)
                if m:
                    out[k] = int(m.group(1))
    missing = set(_PAT) - set(out)
    if missing:
        raise AssertionError(f"reference output did not contain {sorted(missing)}")
    return out
