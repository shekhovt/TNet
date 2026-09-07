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
# The oracle test: the cost model must agree with the untouched reference implementation on 20
# configurations -- every memory count bit-identical, every compute total equal at the 1 pJ
# resolution the reference prints.

"""The oracle test: the new cost model must agree with ``evaluate_networks.py``.

This is what licenses eventually retiring the old path, and it was written before the model it
judges.  ``energy_ECML/evaluate_networks.py`` is not edited by this package; it is the reference.

Two families of configuration are covered:

* **native** -- our own networks, built from a command line by
  :func:`TNet.energy_ECML.adapters.native.specs_from_cfg`.  Both sides then see the same layer list,
  so a disagreement is in the cost model and nowhere else.
* **hand-written** -- the baseline geometries of
  :mod:`TNet.energy_ECML.adapters.handwritten`, ported from the descriptors the published
  baseline rows were produced from.

The comparison uses ``Assumptions(separable_fusion="groups_gt_1")``, because that is the rule the
reference implements and therefore what the published numbers contain.  The default policy,
``"explicit"``, is checked separately in ``test_terms.py``.
"""

from __future__ import annotations

import pytest

from ..spec import Assumptions, PAPER_TECHNOLOGY, Technology
from ..model import evaluate
from . import geometry as hw
from .reference import run_reference

LEGACY = Assumptions(separable_fusion="groups_gt_1")

# Configurations of our own networks that the current code can run.  Verified 2026-09-07: the
# torchvision spellings (`--net 'resnet18()'`) construct but `fuse_model` returns an empty list,
# and `arch_imagenet.ResNet18` / `MobileNetv1` do not construct at all (their base list lacks
# `ESequential`), so none of the three is a usable oracle case.
NATIVE_CFGS = [
    "--net 'QResNet18(gate=Tower8s)' --method ST -A 2 -W 2",
    "--net 'QResNet18(gate=Tower8s)' --method ST -A 4 -W 2",
    "--net 'QResNet18(gate=Tower8s)' --method ST -A 8 -W 2",
    "--net 'QResNet18(gate=Tower8s)' --method ST -A 16 -W 16",
    "--net 'QResNet18(gate=Tower8s)' --method ST -A 256 -W 256",
    "--net 'QResNet18(gate=Tower8s,Dilation=True)' --method ST -A 4 -W 4",
    "--net 'QResNet18(gate=Tower8s,m=0.75)' --method ST-det -A 4 -W 2",
    "--net 'BiNealNet(m=1)' --method ST -A 2 -W 2",
    "--net 'BiNealNet(m=1.5)' --method ST -A 2 -W 2",
    "--net 'BOLDNet(m=1)' --method ST -A 2 -W 2",
    "--net 'BOLDNet(m=3)' --method ST -A 2 -W 2",
]

HAND_CASES = {
    "resnet18-8-8": lambda: hw.resnet18(256, 256),
    "resnet18-f32": lambda: hw.resnet18(hw.F32, hw.F32, K_first_last=hw.F32,
                                        K_in_first=hw.F32, K_stored=hw.F32),
    "reactnet-bireal18": lambda: hw.resnet18(2, 2, K_ds_W=256, K_ds_A=256,
                                             bireal_downsample=True),
    "ewgs-w2a2": lambda: hw.resnet18(4, 4),
    "ewgs-w1a1": lambda: hw.resnet18(2, 2),
    "resnet50-8-8": lambda: hw.resnet50(256, 256),
    "resnet50-f32": lambda: hw.resnet50(hw.F32, hw.F32, K_first_last=hw.F32,
                                        K_in_first=hw.F32, K_stored=hw.F32),
}


def _compare(specs, name):
    ref = run_reference(specs, name)
    # PAPER_TECHNOLOGY, not the default: the oracle hardcodes the published 150 pJ/bit,
    # so this comparison must be made at the constants the paper was priced with.
    e = evaluate(specs, PAPER_TECHNOLOGY, LEGACY, name=name)

    # Memory is an exact integer number of bits on both sides.
    assert int(e.counts.bits_read_weight) == ref["bits_read_weight"]
    assert int(e.counts.bits_read_act) == ref["bits_read_act"]
    assert int(e.counts.bits_written_act) == ref["bits_written"]
    assert int(e.counts.bits_written_act + e.counts.bits_read_act) == ref["bits_activations"]
    assert int(e.max_bits_read_act) == ref["max_read_act"]
    assert int(e.max_bits_written_act) == ref["max_written_act"]

    # Compute energy is compared at the 1 pJ resolution the reference prints.
    assert int(e.CEE) == ref["compute_pJ"], (
        f"{name}: compute {int(e.CEE)} pJ vs reference {ref['compute_pJ']} pJ"
    )


@pytest.mark.parametrize("key", sorted(HAND_CASES))
def test_handwritten_matches_reference(key):
    specs = HAND_CASES[key]()
    for s in specs:                 # the reference has one K_in; compare on the operand width
        s.K_in_stored = s.K_in_operand
        s.K_W_stored = s.K_W_operand
    _compare(specs, key)


@pytest.mark.parametrize("cfg", NATIVE_CFGS)
def test_native_matches_reference(cfg):
    from ..adapters.native import specs_from_cfg
    specs = specs_from_cfg(cfg)
    assert specs, f"{cfg}: fuse_model produced no layers"
    _compare(specs, cfg)


def test_totals_are_the_sum_of_the_layers():
    """Per-layer, per-stage and total rows must be consistent.

    They are separate summations of the same float terms, so they are compared with a tight
    relative tolerance rather than for equality: floating-point addition is not associative.
    """
    specs = hw.resnet18(4, 4)
    e = evaluate(specs, PAPER_TECHNOLOGY, assumptions=LEGACY)
    by_layer = sum(l.CEE for l in e.layers)
    by_stage = sum(s.CEE for s in e.stages.values())
    assert abs(by_layer - e.CEE) <= 1e-9 * e.CEE
    assert abs(by_stage - e.CEE) <= 1e-9 * e.CEE
    assert sum(l.counts.bits_read_weight for l in e.layers) == e.counts.bits_read_weight


def test_skip_reads_are_not_a_reference_case():
    """The appendix's residual read has no counterpart in the reference implementation.

    Appendix: *"In ResNet, we added memory reads whenever there is a skip connection around two
    convolutional blocks."*  The script that produced the published ResNet rows has no term for
    it, and neither does ``net_calc_energy_stats``.  So enabling
    ``skip_reads`` cannot agree with the reference -- and the size of the disagreement is the
    thing worth pinning, because it is the size of the gap between the paper's sentence and the
    paper's numbers.
    """
    plain = evaluate(hw.resnet18(256, 256), PAPER_TECHNOLOGY, assumptions=LEGACY)
    skips = evaluate(hw.resnet18(256, 256, skip_reads=True), PAPER_TECHNOLOGY, assumptions=LEGACY)
    extra = skips.counts.bits_read_act - plain.counts.bits_read_act
    assert extra == 7426048, extra
    # Measured on ResNet-18 8/8: +42.5 % activation reads, +23.0 % feature memory,
    # +5.9 % on the total (weights dominate this row, so the effect on TEE is modest).
    assert 0.425 == round(extra / plain.counts.bits_read_act, 3)
    assert 1.230 == round(skips.MMEE_features / plain.MMEE_features, 3)
    assert 1.059 == round(skips.TEE / plain.TEE, 3)
