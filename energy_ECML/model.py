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
# The whole cost model. count() turns layer descriptions into integer counts of gate operations
# and memory bits, price() multiplies those counts by a Technology to get joules, and evaluate()
# does both and returns the per-layer, per-stage and total breakdown.

"""The cost model: count gate operations and memory bits, then price them.

The split is the point.  :func:`count` turns a list of :class:`~TNet.energy_ECML.
spec.LayerSpec` into :class:`~TNet.energy_ECML.spec.Counts` -- integers, no picojoules anywhere.
:func:`price` multiplies counts by a :class:`~TNet.energy_ECML.spec.Technology`.  :func:`evaluate`
does both and assembles the per-layer, per-stage and total rows.

Everything here reproduces ``energy_ECML/evaluate_networks.py:net_calc_energy_stats``, which stays in
place as the reference and the oracle (``tests/test_reference.py``).  Where that function does
something surprising, the surprise is reproduced and named -- see
:attr:`~TNet.energy_ECML.spec.Assumptions.pooling_output_count` and
:attr:`~TNet.energy_ECML.spec.Assumptions.separable_fusion`.

The formulas are those of the paper's appendix, section ``A:energy``; the paragraph each one
implements is quoted at the function that implements it.

Every cost here is a function of **bit** widths, so every local in this module that carries a
width is ``b``-named.  The ``K``-named quantities that come in -- the fields of a ``LayerSpec``,
``Technology.K_after_conv``, ``Technology.K_affine_coeffs`` -- are **numbers of states** and are
converted at the top of the function that uses them (see the naming convention in
:mod:`TNet.energy_ECML.spec`).  The one place states are used *as states* is the accumulator width,
where the reference multiplies the state counts before taking a logarithm; that line says so.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from .spec import (
    Assumptions,
    Counts,
    LayerEnergy,
    LayerSpec,
    NetworkEnergy,
    PoolSpec,
    StageEnergy,
    Technology,
    bits_for,
)

__all__ = [
    "kappa",
    "dot_product_gates",
    "count_layer",
    "count",
    "price",
    "reprice_memory",
    "evaluate",
]


# ----------------------------------------------------------------------------------------------
# The dot product (appendix, "Dot Product")
# ----------------------------------------------------------------------------------------------


def kappa(b2: int) -> int:
    """Reduction-tree stage at which pipeline registers become necessary.

    Appendix: *"the stage at which the register use is necessary, is given by
    :math:`\\kappa(b_2) = \\max\\{k \\mid k b_2 + \\frac{k(k-1)}{2} \\leq \\lambda\\}`. For
    :math:`\\lambda = 10` and :math:`b_2=(1,2,3,4,5,6...)` we get :math:`\\kappa(b_2)=(4,3,2,2,1,1,...)`."*

    The reference implements the table rather than the maximisation; both are reproduced and
    ``tests/test_terms.py`` checks they agree for the tabulated widths.
    """
    if b2 == 1:
        return 4
    if b2 == 2:
        return 3
    if b2 == 3 or b2 == 4:
        return 2
    return 1


def kappa_from_definition(b2: int, lam: int = 10) -> int:
    """:func:`kappa` computed from the appendix's definition rather than its table."""
    best = 0
    k = 1
    while k * b2 + k * (k - 1) // 2 <= lam:
        best = k
        k += 1
    return best


def dot_product_gates(b1: int, b2: int, n: int) -> tuple[int, int]:
    """Gate operations for one dot product of a ``b1``-bit and a ``b2``-bit vector of length ``n``.

    Returns ``(adder_ops, register_ops)``.  Neither depends on the technology; the energy is
    ``(adder_ops + gamma_0 * register_ops) * E1``.

    Appendix, equation ``eq:e_dot``:

    .. math::
        E_{\\rm dot} = b_1 n (b_2 + 2) E_1 + 2^{K - \\kappa(b_2)} (b_2 + 2 + \\kappa(b_2)) E_r.

    with :math:`K = \\lceil \\log_2 n \\rceil` and :math:`b_1 \\leq b_2` (the operands are
    swapped if necessary, since the method expands the *narrower* operand into bit planes).
    The appendix writes that reduction depth as :math:`K`; in this package ``K`` is reserved for
    a number of states, so the code calls it ``b_n`` -- it is a count of bits (of tree levels),
    not of quantizer states.

    **A third term is computed here that the published equation does not have.** The appendix
    says *"We assume that the cost of the sum over the bit-planes can be neglected for
    :math:`b_1 \\lll n`"*, and then drops it; ``evaluate_networks.py:423`` computes it
    unconditionally as ``b1 * (b2 + b1 + K) * (E1 + gamma_0 * E1)`` -- the final summation
    reduction over the :math:`b_1` bit planes, each accumulator being :math:`b_2 + b_1 + K` bits
    wide.  Measured, it is 0.57--0.83 % of network compute at every bit
    width, so the appendix's claim holds for a whole network, but 6--7 % on small-``n`` 1x1
    layers.  It is kept because it is what the published numbers contain.
    """
    if b1 > b2:
        b1, b2 = b2, b1
    b_n = math.ceil(math.log2(n))       # the appendix's K: the depth of the reduction tree
    k = kappa(b2)
    # first term: the conditional-accumulation reduction tree, per bit plane
    adder = b1 * n * (b2 + 2)
    # second term: pipeline registers from stage kappa(b2) down
    register = (2 ** (b_n - k)) * (b2 + 2 + k)
    # third term: the sum over bit planes (see the note above)
    bitplane = b1 * (b2 + b1 + b_n)
    adder += bitplane
    register += bitplane
    return adder, register


# ----------------------------------------------------------------------------------------------
# Counting
# ----------------------------------------------------------------------------------------------


def _pool_output_elements(spec: LayerSpec, pooling: PoolSpec, a: Assumptions) -> float:
    """Element count after the fused pooling reduction.

    See :attr:`~TNet.energy_ECML.spec.Assumptions.pooling_output_count` for why the default is the
    reference's true division rather than the correct floor.
    """
    C, H, W = spec.out_shape[1], spec.out_shape[2], spec.out_shape[3]
    if pooling.kind == "adaptive_avg":
        if pooling.output_size_was_int:
            # reproduces `r.pooling.output_size + r.pooling.output_size` in the reference
            return C * (pooling.output_size[0] + pooling.output_size[1])
        return C * (pooling.output_size[0] * pooling.output_size[1])
    if a.pooling_output_count == "reference":
        return C * H / pooling.stride * W / pooling.stride
    h = (H - pooling.kernel_size) // pooling.stride + 1
    w = (W - pooling.kernel_size) // pooling.stride + 1
    return C * h * w


def count_layer(
    spec: LayerSpec,
    a: Assumptions = Assumptions(),
    K_after_conv: int = 256,
    K_affine_coeffs: int = 256,
) -> Counts:
    """Counts for one fused convolution.

    The order of the four blocks below is the order of the appendix's description and of the
    reference implementation: convolution, optional pooling, affine, quantization, then the
    write-out.  The pooling block *changes the element count* that the affine and quantization
    blocks then use, which is why it cannot be reordered.
    """
    c = Counts()

    C_in, H_in, W_in = spec.in_shape[1], spec.in_shape[2], spec.in_shape[3]
    C_out, H_out, W_out = spec.out_shape[1], spec.out_shape[2], spec.out_shape[3]
    k_h, k_w = int(spec.weight_shape[2]), int(spec.weight_shape[3])
    n_weights = spec.n_weights()

    # states -> bits, once, at the boundary; everything below is in bits
    b_in_stored = spec.b_in_stored
    b_in_operand = spec.b_in_operand
    b_W_stored = spec.b_W_stored
    b_W_operand = spec.b_W_operand
    b_out = spec.b_out_stored
    b_after_conv = bits_for(K_after_conv)
    b_affine = bits_for(K_affine_coeffs)

    # Accumulator width in bits, "the lowest applicable" (see Assumptions.accumulator_rule).
    # This is the one expression that consumes *states* rather than bits: the reference
    # multiplies the state counts and takes one logarithm of the product, which is not the same
    # as summing the rounded-up bit widths when a K is not a power of two.  It also uses the
    # full input channel count even when the convolution is grouped.
    b_acc = math.ceil(
        math.log2(max(1, spec.K_in_operand * spec.K_W_operand * C_in * k_h * k_w))
    )

    n_out_el: float = C_out * H_out * W_out

    # -- convolution ---------------------------------------------------------------------------
    c.n_mults += H_out * W_out * n_weights
    c.n_adds += H_out * W_out * n_weights

    read_act = b_in_stored * C_in * H_in * W_in
    c.bits_read_act += read_act + spec.skip_read_bits
    c.bits_read_weight += b_W_stored * n_weights

    adder, register = dot_product_gates(b_in_operand, b_W_operand, spec.dot_length())
    c.gates_conv_adder += n_out_el * adder
    c.gates_conv_register += n_out_el * register

    # -- pooling, fused before quantization ----------------------------------------------------
    p = spec.pooling
    if p is not None:
        if p.kind == "max":
            taps = p.kernel_size * p.kernel_size
            c.n_cmps += n_out_el * taps
            c.gates_pool += n_out_el * (taps * b_after_conv)
        elif p.kind == "avg":
            taps = p.kernel_size * p.kernel_size
            c.n_adds += n_out_el * taps
            c.n_mults += n_out_el
            c.gates_pool += n_out_el * (taps * b_after_conv)
            c.gates_pool += n_out_el * (b_after_conv * math.ceil(math.log2(taps)))
        elif p.kind == "adaptive_avg":
            kw = H_out // p.output_size[0]
            kh = W_out // p.output_size[1]
            taps = kw * kh
            c.n_adds += n_out_el * taps
            c.n_mults += n_out_el
            c.gates_pool += n_out_el * (taps * b_after_conv)
            c.gates_pool += n_out_el * (b_after_conv * math.ceil(math.log2(taps)))
        n_out_el = _pool_output_elements(spec, p, a)

    # -- affine --------------------------------------------------------------------------------
    c.n_mults += n_out_el
    c.n_adds += n_out_el
    c.bits_read_weight += b_affine * C_out * 2
    c.gates_affine += n_out_el * (b_after_conv * b_affine)      # one 8x8 multiply
    c.gates_affine += n_out_el * (b_after_conv + b_affine)      # one 16-bit add

    # -- quantization --------------------------------------------------------------------------
    c.n_shifts += 2 * n_out_el
    c.gates_requantize += n_out_el * b_acc
    c.gates_requantize += n_out_el * (b_after_conv * 2)

    # -- write out -----------------------------------------------------------------------------
    if spec.K_out_stored > 0:
        written = b_out * n_out_el
        if a.separable_fusion == "groups_gt_1":
            fused = spec.groups > 1
        else:
            fused = spec.output_fused_into_next
        if fused:
            # The successor reads this tensor as its input and has already been charged for it;
            # cancel that read and charge no write.  This is the reference's bookkeeping, kept
            # verbatim: it subtracts *this* layer's output bits rather than suppressing the
            # successor's read, so the two agree exactly when the widths and shapes match.
            c.bits_read_act -= written
            written = 0
        c.bits_written_act += written

    return c


def count(
    layers: Sequence[LayerSpec],
    a: Assumptions = Assumptions(),
    K_after_conv: int = 256,
    K_affine_coeffs: int = 256,
) -> list[Counts]:
    """Per-layer counts, in layer order."""
    return [count_layer(s, a, K_after_conv, K_affine_coeffs) for s in layers]


# ----------------------------------------------------------------------------------------------
# Pricing
# ----------------------------------------------------------------------------------------------


def price(c: Counts, tech: Technology = Technology()) -> dict:
    """Turn counts into picojoules.  This is the only function that touches a constant."""
    E1 = tech.E1_pJ
    g = tech.gamma_0
    CEE_conv = (c.gates_conv_adder + g * c.gates_conv_register) * E1
    CEE_pool = c.gates_pool * E1
    CEE_affine = c.gates_affine * E1
    CEE_requantize = c.gates_requantize * E1
    return dict(
        CEE=CEE_conv + CEE_pool + CEE_affine + CEE_requantize,
        CEE_conv=CEE_conv,
        CEE_pool=CEE_pool,
        CEE_affine=CEE_affine,
        CEE_requantize=CEE_requantize,
        MMEE_weights=c.bits_read_weight * tech.E_mem_dram_pJ_per_bit,
        MMEE_features=(c.bits_read_act + c.bits_written_act) * tech.E_mem_on_chip_pJ_per_bit,
    )


def reprice_memory(totals: dict, tech: Technology) -> dict:
    """Memory energies for counts a record already holds, under a different ``Technology``.

    ``totals`` is a record's ``results["total"]`` block: it carries the bit counts alongside the
    energies they were priced at.  Only the memory terms depend on the memory constants, so
    changing them is a re-multiplication -- nothing is re-run, and a traced method does not need
    its repository cloned again.  ``CEE_pJ`` is carried through unchanged, which is what makes
    the returned ``TEE_pJ`` correct.

    Returns the four figures that change, in picojoules, ready to be merged into a copy of
    ``totals``.
    """
    w = totals["bits_read_weight"] * tech.E_mem_dram_pJ_per_bit
    f = (totals["bits_read_act"] + totals["bits_written_act"]) * tech.E_mem_on_chip_pJ_per_bit
    return {
        "MMEE_weights_pJ": w,
        "MMEE_features_pJ": f,
        "MMEE_pJ": w + f,
        "TEE_pJ": totals["CEE_pJ"] + w + f,
    }


# ----------------------------------------------------------------------------------------------
# The public entry point
# ----------------------------------------------------------------------------------------------


def evaluate(
    layers: Sequence[LayerSpec],
    tech: Technology = Technology(),
    assumptions: Assumptions = Assumptions(),
    name: str = "",
) -> NetworkEnergy:
    """Evaluate a described network.

    ``layers`` is a list of fused convolutions in execution order; see
    :mod:`TNet.energy_ECML.adapters` for the three ways to produce one.
    """
    if assumptions.pooling_K != tech.K_after_conv:
        raise ValueError(
            f"Assumptions.pooling_K={assumptions.pooling_K} disagrees with "
            f"Technology.K_after_conv={tech.K_after_conv}; the fused reduction has one width"
        )

    per_layer = count(layers, assumptions, tech.K_after_conv, tech.K_affine_coeffs)

    net = NetworkEnergy(name=name, technology=tech, assumptions=assumptions)
    stages: dict[str, StageEnergy] = {}

    for spec, c in zip(layers, per_layer):
        stage = spec.resolved_stage()
        le = LayerEnergy(name=spec.name, stage=stage, spec=spec, counts=c, **price(c, tech))
        net.layers.append(le)

        if stage not in stages:
            stages[stage] = StageEnergy(name=stage)
        st = stages[stage]
        st.n_layers += 1
        st.counts += c

        net.counts += c
        # The reference's "Max activations in/out": the raw per-layer input read (before the
        # skip surcharge and before the fusion cancellation) and the actual write.  Together
        # they bound the on-chip buffer a one-pass schedule would need.
        raw_read = spec.b_in_stored * spec.in_shape[1] * spec.in_shape[2] * spec.in_shape[3]
        net.max_bits_read_act = max(net.max_bits_read_act, raw_read)
        net.max_bits_written_act = max(net.max_bits_written_act, c.bits_written_act)

    for st in stages.values():
        for k, v in price(st.counts, tech).items():
            setattr(st, k, v)
    net.stages = stages

    for k, v in price(net.counts, tech).items():
        setattr(net, k, v)

    return net
