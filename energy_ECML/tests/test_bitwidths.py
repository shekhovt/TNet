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
# Checks that a layer's stored width and its operand width are genuinely independent, and that
# one evaluation of such a layer list equals the two-run composition the single-width reference
# model used to need.

"""Stored width and operand width are independent -- and that retires ``compose()``.

``energy_ECML/SOTA/README.md`` describes a two-run convention: because ``net_calc_energy_stats``
derives both the activation memory read and the convolution compute cost from a single ``K_in``,
a row under the storage assumption (8-bit stored features, published operand widths) had to be
composed from two runs of the model -- compute from the run with ``K_in`` = operand width,
activation memory from the run with ``K_in = 256``.

With five independent fields a row is one object and one evaluation.  These tests assert the
composition identity: the split-field evaluation must equal what ``compose()`` assembled.
"""

from __future__ import annotations

import pytest

from ..spec import PAPER_TECHNOLOGY, Assumptions, LayerSpec, Technology
from ..model import evaluate
from . import geometry as hw

LEGACY = Assumptions(separable_fusion="groups_gt_1")


def _as_single_width(specs, K_in):
    """The same geometry with one ``K_in`` everywhere the storage assumption would vary it."""
    out = []
    for s in specs:
        t = LayerSpec(**{k: getattr(s, k) for k in s.__dataclass_fields__})
        if t.K_in_stored != t.K_in_operand:      # a quantized layer, not the stem or classifier
            t.K_in_stored = t.K_in_operand = K_in
        else:
            t.K_in_stored = t.K_in_operand = t.K_in_operand
        out.append(t)
    return out


@pytest.mark.parametrize("K_A,K_W", [(2, 2), (4, 4), (8, 8), (16, 16), (2, 4)])
def test_split_widths_equal_the_two_run_composition(K_A, K_W):
    """One evaluation of the split spec == compose(compute run, memory run)."""
    split = hw.resnet18(K_A, K_W)
    compute_run = _as_single_width(split, K_A)      # K_in = the operand width
    memory_run = _as_single_width(split, 256)       # K_in = the stored width

    one = evaluate(split, PAPER_TECHNOLOGY, assumptions=LEGACY)
    a = evaluate(compute_run, PAPER_TECHNOLOGY, assumptions=LEGACY)
    b = evaluate(memory_run, PAPER_TECHNOLOGY, assumptions=LEGACY)

    # compute (and the accumulator width it carries) comes from the operand run
    assert one.CEE == pytest.approx(a.CEE, rel=1e-12)
    # activation memory comes from the stored run
    assert one.counts.bits_read_act == b.counts.bits_read_act
    assert one.counts.bits_written_act == b.counts.bits_written_act
    # weight memory is identical in both, which is what compose() asserted
    assert a.counts.bits_read_weight == b.counts.bits_read_weight == one.counts.bits_read_weight
    # and therefore the composed total
    assert one.TEE == pytest.approx(
        b.MMEE_features + a.MMEE_weights + a.CEE, rel=1e-12)


def test_reactnet_bireal18_convention_in_one_run():
    """The row the validation gate of ``energy_ECML/SOTA/README.md`` is written against.

    ``1/8`` -- one-bit convolution operands, 8-bit stored feature maps, 8-bit first convolution,
    classifier and downsample convolutions, Bi-Real average-pool downsample.
    """
    specs = hw.resnet18(2, 2, K_ds_W=256, K_ds_A=256, bireal_downsample=True)
    e = evaluate(specs, PAPER_TECHNOLOGY, assumptions=LEGACY)
    # published row: 11.7M / 2.0 / 2494 | 3.6 / 4440 | 43 | 6977
    assert round(e.MB_weights, 1) == 2.0
    assert round(e.MMEE_weights / 1e6) == 2494
    assert round(e.MB_features, 1) == 3.6
    # features 4533 uJ against the published 4440 -- the 2.1 % gap the gate already recorded
    assert round(e.MMEE_features / 1e6) == 4533


def test_stored_width_alone_moves_memory_and_not_compute():
    a = evaluate(hw.resnet18(4, 4), PAPER_TECHNOLOGY, assumptions=LEGACY)
    b = evaluate(hw.resnet18(4, 4, K_stored=16), PAPER_TECHNOLOGY, assumptions=LEGACY)
    assert b.counts.bits_read_act < a.counts.bits_read_act
    assert b.counts.bits_written_act < a.counts.bits_written_act
    assert b.counts.bits_read_weight == a.counts.bits_read_weight


def test_operand_width_alone_moves_compute_and_not_activation_memory():
    a = evaluate(hw.resnet18(4, 4), PAPER_TECHNOLOGY, assumptions=LEGACY)
    b = evaluate(hw.resnet18(16, 4), PAPER_TECHNOLOGY, assumptions=LEGACY)
    assert b.CEE > a.CEE
    assert b.counts.bits_read_act == a.counts.bits_read_act
    assert b.counts.bits_written_act == a.counts.bits_written_act


def test_quantizing_the_downsample_convolutions_is_worth_this_much():
    """How much of an EWGS row rests on quantizing the three 1x1 downsample convolutions.

    The EWGS entries quantize them like every other layer, which is what the paper describes.
    Leaving them at 8 bits instead -- the arrangement ReActNet uses -- is the obvious sensitivity
    to that reading, and it is small -- 2.5 % of the total at binary widths, less above -- because
    the downsample convolutions are three 1x1 layers out of twenty-one.  Pinned so the choice is a
    measured number rather than an assertion.
    """
    for K_A, K_W, want in [(2, 2, 1.025), (4, 4, 1.018), (16, 16, 1.009)]:
        plain = evaluate(hw.resnet18(K_A, K_W), PAPER_TECHNOLOGY, assumptions=LEGACY)
        at8 = evaluate(hw.resnet18(K_A, K_W, K_ds_W=256, K_ds_A=256), PAPER_TECHNOLOGY, assumptions=LEGACY)
        assert at8.TEE > plain.TEE
        assert round(at8.TEE / plain.TEE, 3) == want, (K_A, K_W, at8.TEE / plain.TEE)
