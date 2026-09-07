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
# Checks each charged term on its own -- dot product, pooling, affine, requantization and memory
# -- on geometries small enough that the arithmetic can be verified by hand.

"""Each charged term on its own, on geometry small enough to check by hand."""

from __future__ import annotations

import math
import pytest

from ..spec import Assumptions, LayerSpec, PoolSpec, Technology, bits_for
from ..model import count_layer, dot_product_gates, evaluate, kappa, kappa_from_definition

TECH = Technology()


def one_conv(**kw) -> LayerSpec:
    """A single 3x3 convolution, 8 input channels to 4 output, on a 5x5 map."""
    base = dict(
        name="c", in_shape=(1, 8, 5, 5), out_shape=(1, 4, 5, 5), weight_shape=(4, 8, 3, 3),
        K_in_stored=256, K_in_operand=256, K_W_stored=256, K_W_operand=256, K_out_stored=256,
    )
    base.update(kw)
    return LayerSpec(**base)


# ------------------------------------------------------------------ the dot product


def test_kappa_table_matches_its_definition_up_to_ten_bits():
    """The reference implements the appendix's table; the appendix defines a maximisation.

    Appendix: *"the stage at which the register use is necessary, is given by
    :math:`\\kappa(b_2) = \\max\\{k \\mid k b_2 + \\frac{k(k-1)}{2} \\leq \\lambda\\}`.
    For :math:`\\lambda = 10` and :math:`b_2 = (1,2,3,4,5,6...)` we get
    :math:`\\kappa(b_2) = (4,3,2,2,1,1,...)`."*

    **The two disagree for :math:`b_2 > 10`.**  The maximisation has no solution there -- even
    :math:`k = 1` costs :math:`b_2 > \\lambda` levels of circuit -- so the definition gives
    :math:`\\kappa = 0`, while the implementation's table returns 1 for every width above 4.
    This is not a rounding difference: :math:`\\kappa = 0` doubles the register term.  It
    affects only the 16-bit and 32-bit rows; ``test_kappa_above_ten_bits_is_a_divergence``
    measures how much.
    """
    for b2 in range(1, 11):
        assert kappa(b2) == kappa_from_definition(b2), b2
    assert [kappa(b) for b in (1, 2, 3, 4, 5, 6)] == [4, 3, 2, 2, 1, 1]


def test_kappa_above_ten_bits_is_a_divergence():
    for b2 in range(11, 33):
        assert kappa(b2) == 1
        assert kappa_from_definition(b2) == 0


def test_dot_product_reproduces_the_papers_table():
    """``tab:compute_cost``: dot product of 256-element vectors, in multiples of ``E1``.

    Two differences between the table and the implementation are pinned here.

    **The table was computed with :math:`E_r = 2 E_1`; the implementation uses
    :math:`E_r = E_1`.**  Every one of the fourteen published entries is reproduced exactly by
    ``first_term + 2 * second_term`` and by no other integer multiplier -- (1,1) is 768 + 2*112 =
    992, (8,8) is 20480 + 2*1408 = 23296, (32,32) is 278528 + 2*4480 = 287488.  The appendix's
    own text says :math:`E_r \\approx 2 E_1`, so the table follows the text; it is
    ``net_calc_energy_stats`` that sets ``gamma_0 = 1`` and therefore halves the register term of
    every published network row.  ``Technology.gamma_0`` makes this a settable constant rather
    than a literal in a function body, and the default is the one the network rows were computed
    with, not the one the table was.

    **The implementation adds a third term the equation does not have** -- the sum over bit
    planes (see :func:`dot_product_gates`).  Its share is measured here to be under 11 % of a
    256-element dot product, and a separate measurement reported 0.57-0.83 % of a
    whole network.
    """
    published = {
        (1, 1): 992, (1, 2): 1472, (1, 3): 2176, (1, 4): 2560,
        (1, 8): 5376, (1, 16): 9472, (1, 32): 17664,
        (2, 2): 2496, (2, 4): 4096, (2, 8): 7936,
        (4, 4): 7168, (8, 8): 23296, (16, 16): 78592, (32, 32): 287488,
    }
    n = 256
    K = 8
    GAMMA_TABLE = 2.0          # see the docstring: the table's E_r, in units of E_1
    for (b1, b2), want in published.items():
        k = kappa(b2)
        first = b1 * n * (b2 + 2)
        second = (2 ** (K - k)) * (b2 + 2 + k)
        assert first + GAMMA_TABLE * second == want, (b1, b2, first, second, want)
        # what the implementation computes for the same operands, at its own gamma_0 = 1
        adder, register = dot_product_gates(b1, b2, n)
        third = (adder - first) + (register - second)
        assert third == 2 * b1 * (b2 + b1 + K)
        assert 0 < third / want < 0.11


def test_dot_product_swaps_operands():
    """The method expands the *narrower* operand into bit planes, so the cost is symmetric."""
    assert dot_product_gates(2, 8, 256) == dot_product_gates(8, 2, 256)


# ------------------------------------------------------------------ memory


def test_activation_read_is_input_tensor_times_stored_width():
    c = count_layer(one_conv(), K_after_conv=256, K_affine_coeffs=256)
    assert c.bits_read_act == 8 * 8 * 5 * 5          # b_in * C_in * H * W


def test_weight_read_is_weights_plus_two_affine_coefficients_per_output_channel():
    c = count_layer(one_conv())
    assert c.bits_read_weight == 8 * (4 * 8 * 3 * 3) + 8 * 4 * 2


def test_write_is_output_tensor_times_stored_width_and_zero_when_not_stored():
    assert count_layer(one_conv()).bits_written_act == 8 * 4 * 5 * 5
    assert count_layer(one_conv(K_out_stored=0)).bits_written_act == 0


def test_one_bit_weights_cost_one_bit_each():
    c = count_layer(one_conv(K_W_stored=2, K_W_operand=2))
    assert c.bits_read_weight == 1 * (4 * 8 * 3 * 3) + 8 * 4 * 2


# ------------------------------------------------------------------ the skip rule


def test_the_three_skip_cases_of_the_appendix():
    """Bi-Real free, ResNet charged, Tower free -- one rule, three instances.

    The rule is a property of the tensor, not of the number of convolutions: a feature tensor is
    charged one read per fused convolution that consumes it, and a skip adds a read only when the
    tensor it carries is not already an input of the consuming convolution.
    """
    plain = count_layer(one_conv())
    bireal = count_layer(one_conv(skip_read_bits=0))            # x is conv's own input
    tower = count_layer(one_conv(skip_read_bits=0))             # concatenation is free
    resnet = count_layer(one_conv(skip_read_bits=8 * 8 * 5 * 5))  # conv1(x) is not conv2's input
    assert bireal.bits_read_act == plain.bits_read_act
    assert tower.bits_read_act == plain.bits_read_act
    assert resnet.bits_read_act == 2 * plain.bits_read_act


# ------------------------------------------------------------------ pooling


def test_max_pool_output_count_reference_versus_floor():
    """The reference divides the spatial size by the stride instead of taking the true size.

    ``issues.md`` section 7: a 55x55 map pooled by stride 2 becomes 27.5 elements per row.  The
    published numbers contain this, so ``pooling_output_count="reference"`` is the default; the
    corrected branch exists and is pinned here.
    """
    spec = one_conv(out_shape=(1, 4, 55, 55), pooling=PoolSpec("max", kernel_size=3, stride=2))
    ref = count_layer(spec, Assumptions(pooling_output_count="reference"))
    fixed = count_layer(spec, Assumptions(pooling_output_count="floor"))
    assert ref.bits_written_act == 8 * (4 * 55 / 2 * 55 / 2)       # 27.5 x 27.5
    assert fixed.bits_written_act == 8 * (4 * 27 * 27)             # floor((55-3)/2)+1 = 27
    assert ref.bits_written_act > fixed.bits_written_act


def test_adaptive_average_pool_to_one_by_one():
    spec = one_conv(out_shape=(1, 4, 7, 7), pooling=PoolSpec("adaptive_avg", output_size=(1, 1)))
    c = count_layer(spec)
    assert c.bits_written_act == 8 * 4               # one element per channel
    #  conv adds        + 49 pooling taps per pre-pool element + one affine add per output
    assert c.n_adds == 7 * 7 * (4 * 8 * 3 * 3) + 4 * 7 * 7 * 49 + 4


def test_adaptive_pool_int_output_size_is_the_references_quirk():
    """``C * (output_size + output_size)`` when the module was given a bare int, not ``C``."""
    tup = one_conv(out_shape=(1, 4, 7, 7), pooling=PoolSpec("adaptive_avg", output_size=(1, 1)))
    integer = one_conv(out_shape=(1, 4, 7, 7),
                       pooling=PoolSpec("adaptive_avg", output_size=(1, 1),
                                        output_size_was_int=True))
    assert count_layer(integer).bits_written_act == 2 * count_layer(tup).bits_written_act


# ------------------------------------------------------------------ fusion


def test_separable_fusion_explicit_versus_groups():
    """A depthwise layer is fused only when the pair is declared."""
    dw = one_conv(groups=8, weight_shape=(8, 1, 3, 3), out_shape=(1, 8, 5, 5))
    legacy = count_layer(dw, Assumptions(separable_fusion="groups_gt_1"))
    explicit_off = count_layer(dw, Assumptions(separable_fusion="explicit"))
    dw2 = one_conv(groups=8, weight_shape=(8, 1, 3, 3), out_shape=(1, 8, 5, 5),
                   output_fused_into_next=True)
    explicit_on = count_layer(dw2, Assumptions(separable_fusion="explicit"))
    assert legacy.bits_written_act == 0
    assert explicit_off.bits_written_act == 8 * 8 * 5 * 5
    assert explicit_on.bits_written_act == 0
    assert explicit_on.bits_read_act == legacy.bits_read_act


def test_a_dense_layer_is_never_fused_by_the_groups_rule():
    c = count_layer(one_conv(), Assumptions(separable_fusion="groups_gt_1"))
    assert c.bits_written_act > 0


# ------------------------------------------------------------------ policy plumbing


def test_unimplemented_policies_are_refused_rather_than_ignored():
    with pytest.raises(NotImplementedError):
        Assumptions(memory_traffic="tiling_optimal")
    with pytest.raises(ValueError):
        Assumptions(separable_fusion="magic")
    with pytest.raises(ValueError):
        Assumptions(pooling_output_count="clever")


def test_pricing_is_linear_in_the_constants():
    """Counting is separated from pricing, so doubling a constant doubles its term."""
    specs = [one_conv()]
    a = evaluate(specs)
    b = evaluate(specs, Technology(E1_pJ=2 * TECH.E1_pJ))
    c = evaluate(specs, Technology(
        E_mem_on_chip_pJ_per_bit=2 * TECH.E_mem_on_chip_pJ_per_bit))
    assert b.CEE == pytest.approx(2 * a.CEE)
    assert b.MMEE == a.MMEE
    assert c.MMEE_features == pytest.approx(2 * a.MMEE_features)
    assert c.MMEE_weights == a.MMEE_weights
    assert b.counts.to_dict() == a.counts.to_dict() == c.counts.to_dict()
