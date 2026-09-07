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
# Checks the hand-written XNOR-Net geometry against an earlier calculation of the same network
# made with the reference model, and measures the two width choices that move its number.

"""XNOR-Net's geometry, checked against the numbers the reference model computed for it.

``adapters/handwritten.xnor_alexnet`` is a port, not a new derivation: the geometry was first
established by handing a list of ``FusedConv`` to the unmodified
``evaluate_networks.net_calc_energy_stats``.  So the port has an oracle, and it is exact -- all
six printed columns of all three of that calculation's rows, including the full-precision AlexNet.

The oracle is reproduced at **the earlier calculation's network input width**, which is the
reference's ``in_K = 8``
(8 *levels*, i.e. 3 bits; ``fuse_model`` raises it to 256 only for MobileNet).  The registry entry
uses an 8-bit network input instead, like every other baseline; the difference is
one number, measured by :func:`test_the_house_convention_input_costs_this_much`, and it is the
only thing separating the entry from the slide's figure.
"""

from __future__ import annotations

import pytest

from ..adapters import handwritten as hw
from ..model import evaluate
from ..spec import PAPER_TECHNOLOGY

F32 = 2 ** 32

#: The reference-model calculation, 2026-09-06: params MB / uJ, features MB / uJ, compute uJ,
#: total uJ.
ORACLE = {
    "xnor-8bit-first-last": (dict(K_A_operand=2, K_W=2, K_first_last=256, K_in_first=8),
                             (10.68, 13440, 0.17, 216, 6, 13661)),
    "xnor-4bit-first-last": (dict(K_A_operand=2, K_W=2, K_first_last=16, K_in_first=8),
                             (8.71, 10961, 0.17, 216, 4, 11181)),
    "alexnet-f32": (dict(K_A_operand=F32, K_W=F32, K_first_last=F32, K_in_first=F32,
                         K_logits=F32),
                    (230.86, 290484, 3.17, 3987, 682, 295153)),
}


@pytest.mark.parametrize("name", sorted(ORACLE))
def test_port_reproduces_the_oracle(name):
    """Every column, at the precision the oracle table printed it."""
    kw, (p_MB, p_uJ, f_MB, f_uJ, c_uJ, t_uJ) = ORACLE[name]
    e = evaluate(hw.xnor_alexnet(**kw), PAPER_TECHNOLOGY)
    assert round(e.MB_weights, 2) == p_MB
    assert round(e.MMEE_weights / 1e6) == p_uJ
    assert round(e.MB_features, 2) == f_MB
    assert round(e.MMEE_features / 1e6) == f_uJ
    assert round(e.CEE / 1e6) == c_uJ
    assert round(e.TEE / 1e6) == t_uJ


def test_geometry():
    """Two branches of five convolutions plus three fully connected layers, none grouped.

    ``groups`` stays 1 throughout on purpose: under the legacy fusion rule a grouped layer is
    treated as fused into its successor and loses its activation writes.
    """
    specs = hw.xnor_alexnet(K_A_operand=2, K_W=2)
    assert len(specs) == 13
    assert all(s.groups == 1 for s in specs)
    assert sum(1 for s in specs if s.pooling is not None) == 6      # three per branch
    assert [s.name for s in specs[:2]] == ["branch0.conv1", "branch0.conv2"]
    assert [s.name for s in specs[-3:]] == ["fc6", "fc7", "fc8"]
    # 60.5M parameters, which is AlexNet's and what the oracle and the slide report.
    assert round(sum(s.n_weights() for s in specs) / 1e6, 1) == 60.5


def test_first_and_last_layers_are_not_binary():
    """XNOR-Net section 4.1 keeps them real; the storage assumption charges them 8 bits."""
    specs = hw.xnor_alexnet(K_A_operand=2, K_W=2)
    first = [s for s in specs if s.name.endswith("conv1")]
    assert len(first) == 2 and all(s.K_W_operand == 256 for s in first)
    assert specs[-1].K_W_operand == 256
    interior = [s for s in specs if s not in first and s is not specs[-1]]
    assert all(s.K_W_operand == 2 and s.K_in_operand == 2 for s in interior)


def test_activations_are_stored_at_one_bit_not_capped_at_eight():
    """Pricing the stored feature maps at 8 bits instead of 1, and what that is worth.

    AlexNet has no skip connection, so its binary activations really are the tensors written to
    memory -- unlike a Bi-Real style network, where the 8-bit cap exists because the real-valued
    skip path means the tensors are not binary whatever the label says.
    """
    binary = evaluate(hw.xnor_alexnet(K_A_operand=2, K_W=2), PAPER_TECHNOLOGY)
    capped = evaluate(hw.xnor_alexnet(K_A_operand=2, K_W=2, K_stored=256), PAPER_TECHNOLOGY)
    assert round(binary.MMEE_features / 1e6) == 442
    assert round(capped.MMEE_features / 1e6) == 997
    # Either way the row is dominated by weight traffic, which is identical in both.
    assert binary.MMEE_weights == capped.MMEE_weights
    assert binary.MMEE_weights / binary.TEE > 0.96


def test_the_house_convention_input_costs_this_much():
    """The registry row differs from the oracle's and the slide's 13661 uJ by the input width alone.

    The oracle used the reference's ``in_K = 8`` -- 8 levels, 3 bits -- which is what our own TNet rows
    read.  Every other baseline reads an 8-bit input instead.  The gap is +1.7 %
    and it is entirely in the two branches' first convolution, which each read the whole image.
    """
    oracle = evaluate(hw.xnor_alexnet(K_A_operand=2, K_W=2, K_in_first=8), PAPER_TECHNOLOGY)
    house = evaluate(hw.xnor_alexnet(K_A_operand=2, K_W=2), PAPER_TECHNOLOGY)
    assert round(oracle.TEE / 1e6) == 13661
    assert round(house.TEE / 1e6) == 13892
    assert house.TEE / oracle.TEE == pytest.approx(1.017, abs=5e-4)
