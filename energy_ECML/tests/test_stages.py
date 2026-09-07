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
# Checks that the per-resolution stage breakdown is derived from the layer shapes rather than
# hard-coded, and that the stages sum to the network total.

"""Resolution stages are derived, not hand-tagged.

Derivation is what makes a per-stage table work for a foreign model nobody has annotated.
"""

from __future__ import annotations

from ..spec import Assumptions, LayerSpec
from ..model import evaluate
from . import geometry as hw

LEGACY = Assumptions(separable_fusion="groups_gt_1")


def test_resnet18_stages_are_the_five_imagenet_resolutions():
    e = evaluate(hw.resnet18(256, 256), assumptions=LEGACY)
    assert sorted(e.stages, key=lambda s: -int(s.split("x")[0])) == [
        "112x112", "56x56", "28x28", "14x14", "7x7", "1x1",
    ]
    assert e.stages["112x112"].n_layers == 1        # the stem
    assert e.stages["56x56"].n_layers == 4          # layer1, two basic blocks, no downsample
    assert e.stages["28x28"].n_layers == 5          # layer2, plus its 1x1 downsample
    assert e.stages["1x1"].n_layers == 1            # the classifier


def test_stage_totals_add_up_to_the_network_total():
    e = evaluate(hw.resnet18(4, 4), assumptions=LEGACY)
    assert sum(s.counts.bits_read_weight for s in e.stages.values()) == e.counts.bits_read_weight
    assert sum(s.counts.bits_written_act for s in e.stages.values()) == e.counts.bits_written_act
    assert sum(s.n_layers for s in e.stages.values()) == len(e.layers)


def test_an_explicit_stage_label_overrides_the_derivation():
    s = LayerSpec(name="c", in_shape=(1, 8, 5, 5), out_shape=(1, 4, 5, 5),
                  weight_shape=(4, 8, 3, 3), stage="my-stage")
    assert s.resolved_stage() == "my-stage"
    assert evaluate([s]).stages["my-stage"].n_layers == 1
