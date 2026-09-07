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
# Checks that the entry paths agree where they overlap, above all that tracing torchvision's
# ResNet-18 reproduces the independent hand-written geometry layer by layer and then by total
# energy. That is what makes the traced rows trustworthy.

"""The entry paths agree with each other where they overlap."""

from __future__ import annotations

import pytest

from ..spec import Assumptions
from ..model import evaluate
from . import geometry as hw

LEGACY = Assumptions(separable_fusion="groups_gt_1")


def test_handwritten_resnet18_has_the_expected_shape():
    specs = hw.resnet18(256, 256)
    assert len(specs) == 21                     # 1 stem + 16 block convs + 3 downsample + fc
    assert specs[0].name == "conv1"
    assert specs[-1].name == "fc"
    assert specs[0].pooling.kind == "max"
    assert specs[-2].pooling.kind == "adaptive_avg"
    assert all(s.groups == 1 for s in specs)


def test_handwritten_resnet50_has_no_depthwise_convolution():
    """ResNet-50 has no depthwise convolution, so no candidate for the separable fusion."""
    specs = hw.resnet50(256, 256)
    assert len(specs) == 54                     # 53 convolutions + the classifier
    assert all(s.groups == 1 for s in specs)
    assert not any(s.output_fused_into_next for s in specs)


def test_parameter_count_matches_torchvision():
    """The hand geometry is checked against a real ResNet-18/50, weights only."""
    torchvision = pytest.importorskip("torchvision")
    for builder, ref in ((hw.resnet18(256, 256), "resnet18"),
                         (hw.resnet50(256, 256), "resnet50")):
        m = getattr(torchvision.models, ref)(weights=None)
        want = sum(p.numel() for n, p in m.named_parameters()
                   if "bn" not in n and "downsample.1" not in n and p.dim() > 1)
        got = sum(s.n_weights() for s in builder)
        assert got == want, (ref, got, want)


def test_native_adapter_reproduces_the_reference_layer_list():
    """The native path uses ``fuse_model`` itself, so the lists must be the same objects' shapes.

    Pinned with ``correct_units=False`` and ``image_size=224``, i.e. the widths exactly as
    ``fuse_model`` writes them and the input size the published numbers were computed at.  That is
    the well-posed comparison against the reference; :func:`test_the_units_correction_moves_one_layer`
    and :func:`test_the_unpadded_stem_is_given_an_equivalent_input` cover the two defaults.
    """
    from ..adapters.native import specs_from_cfg
    specs = specs_from_cfg("--net 'QResNet18(gate=Tower8s)' --method ST -A 2 -W 2",
                           image_size=224, correct_units=False)
    assert len(specs) == 20
    assert specs[0].in_shape == (1, 3, 224, 224)
    assert all(s.K_in_stored == s.K_in_operand for s in specs)
    e = evaluate(specs, assumptions=LEGACY)
    assert int(e.CEE) == 25335980
    assert int(e.counts.bits_read_weight) == 20331648


def test_the_units_correction_moves_one_layer():
    """``correct_bits_as_levels`` touches the image width here and nothing else.

    ``fuse_model`` writes ``in_K = 8`` meaning 8 *bits* into a field counting *levels*, so the
    image is charged 3 bits.  For this configuration that is the only literal in play: the stem is
    a ``QConv2d`` whose weights come from a quantizer, so the second literal (``fc.K_W = 8``, for
    an unquantized layer) does not arise.
    """
    from ..adapters.native import specs_from_cfg, K_8BIT
    cfg = "--net 'QResNet18(gate=Tower8s)' --method ST -A 2 -W 2"
    raw = specs_from_cfg(cfg, correct_units=False)
    fixed = specs_from_cfg(cfg, correct_units=True)

    changed = [i for i, (a, b) in enumerate(zip(raw, fixed))
               if (a.K_in_stored, a.K_W_stored) != (b.K_in_stored, b.K_W_stored)]
    assert changed == [0]
    assert raw[0].K_in_stored == 8 and fixed[0].K_in_stored == K_8BIT
    assert evaluate(fixed, assumptions=LEGACY).CEE > evaluate(raw, assumptions=LEGACY).CEE


def test_the_unpadded_stem_is_given_an_equivalent_input():
    """``Block0_b1`` does not pad, so it must be shown the larger image the loader feeds it.

    At 224 the 7x7 stride-2 unpadded stem yields 109x109 where a padded stem yields 112x112, and
    every layer after it is then priced on a tensor the network never computes.  The default input
    size restores the 112 -> 56 -> 28 -> 14 -> 7 ladder; 229 is the smallest input that does (the
    loader's 230 gives the same ladder and differs only in how many pixels the first layer reads).
    """
    from ..adapters.native import specs_from_cfg, equivalent_input_size
    cfg = "--net 'QResNet18(gate=Tower8s)' --method ST -A 2 -W 2"
    at224 = specs_from_cfg(cfg, image_size=224)
    assert at224[0].out_shape[2:] == (109, 109)
    assert equivalent_input_size(at224[0], 224) == 229

    auto = specs_from_cfg(cfg)
    assert auto[0].in_shape == (1, 3, 229, 229)
    assert auto[0].out_shape[2:] == (112, 112)
    assert [s.out_shape[2] for s in auto[:2]] == [112, 56]
    assert evaluate(auto, assumptions=LEGACY).CEE > evaluate(at224, assumptions=LEGACY).CEE

    at230 = specs_from_cfg(cfg, image_size=230)
    assert [s.out_shape[2:] for s in at230] == [s.out_shape[2:] for s in auto]


def test_a_padded_first_layer_is_left_at_224():
    """The rule only ever grows the input, and only up to equivalence.

    A hand-written ResNet-18 stem pads, so it already produces 112x112 from 224 and must not be
    rescaled -- solving the same equation for it would otherwise shrink the input to 223.
    """
    from ..adapters.native import equivalent_input_size
    from . import geometry as hw
    assert equivalent_input_size(hw.resnet18(4, 4)[0], 224) == 224


def test_a_genuine_three_bit_configuration_is_left_alone():
    """``K = 8`` is also a legitimate setting -- 8 levels -- and must not be rewritten.

    ``-A 8 -W 8`` is a real registry row.  A blanket "every 8 becomes 256" inflates it by a third
    by rewriting nineteen genuine 3-bit activations; identifying the two literals by *where they
    come from* rather than by their value cannot do that.
    """
    from ..adapters.native import specs_from_cfg
    cfg = "--net 'QResNet18(gate=Tower8s)' --method ST -A 8 -W 8"
    raw = specs_from_cfg(cfg, correct_units=False)
    fixed = specs_from_cfg(cfg, correct_units=True)
    changed = [i for i, (a, b) in enumerate(zip(raw, fixed))
               if (a.K_in_stored, a.K_W_stored) != (b.K_in_stored, b.K_W_stored)]
    assert changed == [0]
    assert sum(1 for s in fixed if s.K_in_stored == 8) == 19


def test_traced_and_handwritten_resnet18_agree():
    """The two entry paths priced the same architecture; they must produce the same list.

    This is the cross-check that makes the tracer trustworthy: torchvision's ResNet-18 is traced
    with forward hooks, the hand geometry is written out from the paper, and neither knows about
    the other.
    """
    torchvision = pytest.importorskip("torchvision")
    from ..adapters.traced import specs_from_trace, trace, uniform_policy

    traced = specs_from_trace(
        trace(torchvision.models.resnet18(weights=None)),
        uniform_policy(256, 256, first_last_names=("conv1", "fc")),
    )
    hand = hw.resnet18(256, 256)
    assert len(traced) == len(hand) == 21
    for t, h in zip(traced, hand):
        assert t.name == h.name
        assert t.in_shape == h.in_shape, t.name
        assert t.out_shape == h.out_shape, t.name
        assert t.weight_shape == h.weight_shape, t.name
        assert (t.pooling is None) == (h.pooling is None), t.name
    assert evaluate(traced, assumptions=LEGACY).TEE == evaluate(hand, assumptions=LEGACY).TEE


def test_tracer_fuses_a_pooling_into_the_convolution_that_produced_its_input():
    torchvision = pytest.importorskip("torchvision")
    from ..adapters.traced import specs_from_trace, trace, uniform_policy

    specs = specs_from_trace(trace(torchvision.models.resnet18(weights=None)),
                             uniform_policy(256, 256))
    assert specs[0].pooling is not None and specs[0].pooling.kind == "max"
    assert specs[-2].pooling is not None and specs[-2].pooling.kind == "adaptive_avg"
    assert specs[-1].pooling is None
