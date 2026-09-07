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
# The data model everything else is written against: Technology (the picojoules), Assumptions
# (the modelling policy), LayerSpec (one fused convolution's geometry and quantizer widths), and
# the result types a run produces. No torch, and no cost model -- only the descriptions.

"""Data model for the energy evaluation: what is described, and under what policy.

This module is deliberately free of ``torch``.  A :class:`LayerSpec` is a description of a
layer's *geometry and quantizer widths*, not a module; the adapters in
:mod:`TNet.energy_ECML.adapters` are the only place that knows about ``nn.Module``.  Keeping the
description pure is what lets the same list be produced by tracing a foreign repository, by
walking one of our own networks, or by typing the geometry out of a paper (see
``energy_ECML/README.md``).

**Naming convention.**  Two different quantities are both called "width"
in the literature, and confusing them has already cost a published error.
In this package they are told apart by the *first character of the name*:

``K...``
    A **number of states** -- how many distinct values a quantizer produces.  This is the
    primitive the training codebase is built on, and it need not be a power of two: ``-A 3``
    is a legitimate run with three activation states.
``b...``
    A **number of bits**, ``ceil(log2(K))``.  This is what the literature reports and what the
    paper's tables print (``1/8``, ``W=1``, "8-bit").

:func:`bits_for` converts states to bits and :func:`states_for` converts bits to states; they
are the only sanctioned way to cross between the two.  ``8`` alone is ambiguous and must never
appear as a width: as states it is three bits, as bits it is 256 states.

Three kinds of object live here:

``Technology``
    Physics.  Picojoules per bit-operation and per memory bit.  Nothing else in the package
    holds a picojoule constant.

``Assumptions``
    Modelling policy -- what may be fused, what must be written out, how a skip is charged.
    Every field carries, in its docstring, the sentence from the paper's appendix, section
    ``A:energy``, that it implements.

``LayerSpec`` / ``PoolSpec``
    The description of one fused convolution: shapes, quantizer widths in states, and an
    optional pooling reduction that happens before the affine transform and the output
    quantization.

The result types (:class:`LayerEnergy`, :class:`StageEnergy`, :class:`NetworkEnergy`) carry the
same named fields at every level, so a per-layer row, a per-stage row and the total are one kind
of thing and a table can be built by iterating.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence

__all__ = [
    "Technology",
    "Assumptions",
    "PoolSpec",
    "LayerSpec",
    "Counts",
    "LayerEnergy",
    "StageEnergy",
    "NetworkEnergy",
    "PAPER_TECHNOLOGY",
    "bits_for",
    "states_for",
]


def bits_for(K: int) -> int:
    """Number of **bits** needed to represent ``K`` distinct **states**.

    A quantizer is described by its number of states ``K``, matching the ``-A``/``-W``
    command-line arguments (``-A 8`` means eight activation states, i.e. three bits).
    ``K = 0`` means "no value is stored" and yields zero bits.

    This is one of the two functions that cross between the two units; :func:`states_for` is
    the other.  Anywhere else, a ``K``-named quantity stays states and a ``b``-named quantity
    stays bits.
    """
    if K <= 0:
        return 0
    return math.ceil(math.log2(K))


def states_for(b: int) -> int:
    """Number of **states** an unsigned ``b``-**bit** value can take, ``2 ** b``.

    The inverse of :func:`bits_for` on powers of two.  Use it at every boundary where a figure
    quoted in bits (a paper's "8-bit", the 8-bit image the network reads) has to be written into
    a ``K``-named field: ``K_in_stored = states_for(8)``, never ``K_in_stored = 8``.  Writing the
    bit count straight into a states field is the error that produced a published erratum.
    """
    if b <= 0:
        return 0
    return 2 ** b


# ----------------------------------------------------------------------------------------------
# Physics
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Technology:
    """Energy constants of the target hardware.

    **The defaults are not the published paper's.**  The compute constants are -- they are read
    from ``energy_ECML/evaluate_networks.py:net_calc_energy_stats`` (7 nm branch), not recalled --
    but the memory constant is now a *measured* one: 13.11 pJ per bit of HBM traffic on an NVIDIA
    A100.  The paper priced memory at 150 pJ/bit from a DDR4-on-a-desktop-CPU figure, roughly
    eleven times higher.  Those constants are kept, and reproduce the printed table exactly, as
    :data:`PAPER_TECHNOLOGY`.

    Because counting is separated from pricing, a record stores counts and a
    different ``Technology`` is applied by re-multiplication, without re-running anything --
    which is how the committed records were moved onto these constants without being rebuilt.
    See :func:`TNet.energy_ECML.model.reprice_memory`.
    """

    name: str = "hbm-a100"
    """Short, filename-safe identifier.  It ends up in a record's provenance and in the name of
    every figure priced under it, so it is a slug, not a sentence -- :attr:`process` and
    :attr:`memory` are where the prose goes."""

    E1_pJ: float = 0.03 / 32.0
    """Energy of a one-bit carry-on addition circuit, in picojoules.

    Appendix: *"where :math:`E_1` is the energy cost of 1-bit carry-on addition circuit (using
    2 XOR, 2 AND and 1 NOT logic gates). It costs 2-3 fJ in 45nm technology and 0.5-1 fJ in 7 nm
    technology."*  The published value ``0.03/32`` pJ = 0.9375 fJ sits in that 7 nm range.
    """

    gamma_0: float = 1.0
    """Cost of a register bit relative to :attr:`E1_pJ`, i.e. :math:`E_r / E_1`.

    Appendix: *"The energy cost :math:`E_r` of register access is about 5 fJ per bit in the 45 nm
    technology and 1-2 fJ in 7 nm, approximately double the cost of the elementary adder on the
    same technology node, :math:`E_r \\approx 2 E_1`."*  The published table nonetheless used
    ``gamma_0 = 1``; the default here is what the table used, not what the sentence suggests.
    """

    E_mem_on_chip_pJ_per_bit: float = 13.11
    """Energy per bit of feature-map (activation) traffic.

    The default is *measured*: the A100's HBM total, control (8.47) plus datapath (4.64), from
    Antepara et al., *Benchmark-driven Models for Energy Analysis and Attribution of
    GPU-Accelerated Supercomputing*, SC '25, Table 3.  The HBM3 part they measure (a Grace+H200)
    is 11.68 pJ/bit; their on-chip levels are 4.71 (L2) and 1.59 (L1).

    The published table instead used 150 pJ/bit -- appendix: *"The cost of memory accesses was
    calculated by multiplying the amount of accessed memory by 150 pJ per bit, a typical value
    modern CPUs use when accessing DDR4 memory."*  See :data:`PAPER_TECHNOLOGY`.

    The name says "on chip" because that is the variable name in the reference implementation.
    The value it is given is an *off-chip* figure, here as in the paper: this model charges every
    bit as one off-chip pass and does not model tiling or caching, so activations and weights are
    priced the same.  Measured hardware separates them by 8x (HBM against L1); that this model
    does not is an assumption, not a measurement.
    """

    E_mem_dram_pJ_per_bit: float = 13.11
    """Energy per bit of weight traffic.  See :attr:`E_mem_on_chip_pJ_per_bit`."""

    K_after_conv: int = 256
    """**States** of the accumulator presented to pooling and to the affine transform.

    Appendix: *"the reduction is again assumed to be inline, fused-in. The reduction is assumed
    to be computed in 8 bit precision."*  8 bits, so 256 states; read :attr:`b_after_conv` for
    the bit count the formulas use.
    """

    K_affine_coeffs: int = 256
    """**States** of the affine scale and bias coefficients, read once per output channel.

    8 bits, so 256 states; :attr:`b_affine_coeffs` is the bit count.  This is an assumption about
    the datapath, alongside :attr:`K_after_conv`, and **not** an instance of the 8-bit storage
    assumption that caps feature maps and weight matrices.  That one rests on the rounding error
    of a quantized operand being averaged away by a sum over many inputs; there are two affine
    coefficients per output channel and they are summed over nothing, so the argument does not
    reach them.  Being a ``Technology`` field rather than a ``LayerSpec`` one, this width is
    uniform across the network and no width policy can change it.
    """

    process: str = "7 nm CMOS"
    """The compute process node, in prose.  Descriptive: nothing computes with it."""

    memory: str = "HBM2e, measured on an NVIDIA A100 (Antepara et al., SC '25)"
    """The memory technology the per-bit constants describe, in prose.  Descriptive.

    It exists because "13.11 pJ/bit" is meaningless without it: the same model over the same
    counts gives an answer an order of magnitude different for DDR4 across a socket and for HBM on
    package, and a reader of a generated table has to be told which one they are looking at.
    """

    @property
    def b_after_conv(self) -> int:
        """:attr:`K_after_conv` in **bits**."""
        return bits_for(self.K_after_conv)

    @property
    def b_affine_coeffs(self) -> int:
        """:attr:`K_affine_coeffs` in **bits**."""
        return bits_for(self.K_affine_coeffs)


# ----------------------------------------------------------------------------------------------
# Modelling policy
# ----------------------------------------------------------------------------------------------


#: The constants the **published** ECML-PKDD 2026 table was computed with.  Passing this to
#: :func:`~TNet.energy_ECML.model.evaluate` reproduces every printed number; the default
#: :class:`Technology` no longer does, because its memory constant was replaced by a measured one.
#: ``tests/test_reference.py`` pins the paper's numbers through this object, which is what keeps
#: the change to the default from quietly moving the oracle.
PAPER_TECHNOLOGY = Technology(
    name="paper-ddr4",
    E_mem_on_chip_pJ_per_bit=150.0,
    E_mem_dram_pJ_per_bit=150.0,
    process="7 nm CMOS",
    memory="DDR4-class, from a desktop-CPU measurement (Lam, Chips and Cheese, 2025)",
)


@dataclass(frozen=True)
class Assumptions:
    """The optimistic modelling policy under which a count is produced.

    A record stores the ``Assumptions`` it was produced under, so two rows computed under
    different policies cannot be put in one table by accident -- ``report.py --check`` refuses.

    The appendix opens the relevant section with: *"When estimating the energy costs, we make
    optimistic assumptions about architecture components regarding how many read and write
    accesses are used."*  Each field below is one of those assumptions, made nameable so that a
    second policy can exist beside this one.
    """

    name: str = "paper-2026-optimistic"

    elementwise_fused: bool = True
    """Affine transforms and activations need no intermediate storage.

    Appendix: *"Element-wise operations are expected to be fused with the convolutions without
    the need of intermediate storage."*  Setting this ``False`` is not implemented; the field
    exists so the assumption is visible where it is applied.
    """

    pooling_fused: bool = True
    """A pooling reduction between a convolution and its quantization is inline.

    Appendix: *"If the convolution is followed by a max or average pooling reduction prior to
    quantization, the reduction is again assumed to be inline, fused-in."*
    """

    pooling_K: int = 256
    """**States** the fused reduction is computed at.

    Appendix: *"The reduction is assumed to be computed in 8 bit precision."*  8 bits, so 256
    states.  Kept here as well as on :class:`Technology` because it is a modelling choice, not a
    property of the silicon; :func:`TNet.energy_ECML.model.evaluate` uses ``Technology.K_after_conv``
    so that the default reproduces the published arithmetic exactly, and asserts the two agree.

    This is the one width field in the package whose name does not *begin* with ``K``.  It is
    serialized into every record's ``provenance.assumptions`` block, and renaming it would
    rewrite every file under ``energy_ECML/results/`` without changing a number; the rename is left
    for the author to take.  The ``K`` in the name still marks the unit: this is states, not bits.
    """

    accumulator_rule: str = "lowest_applicable"
    """How many bits the dot-product accumulator holds.

    Appendix: *"The bit width of dot product results is assumed to be the lowest applicable,
    :math:`b = b_1 + b_2 + \\log_2(C_{in} k^2)`, where :math:`b_1` is the number of bits per
    weight and :math:`b_2` is the number of bits per input feature."*

    ``"lowest_applicable"`` is that rule.  Note that the reference implementation computes it as
    ``ceil(log2(K_in * K_W * C_in * k_h * k_w))``, i.e. it multiplies the *state* counts and
    takes one logarithm of the product.  That is not the same as ``b_1 + b_2 + log2(...)`` when a
    ``K`` is not a power of two, since the latter rounds each width up first.  It also uses the
    *full* input channel count -- not ``C_in / groups`` -- even for a grouped convolution.  Both
    are reproduced here.
    """

    concat_free: bool = True
    """A concatenation moves no memory.

    Appendix, on the Tower architecture: *"there are no residual connections, when features are
    concatenated they become an input to the next convolution, correctly counting the memory
    accesses of all inputs."*  Author's ruling, 2026-09-06: the tensors may be pre-allocated
    contiguously, or a conv kernel can read from a list of input tensors.
    """

    skip_read_when_not_already_an_input: bool = True
    """A skip path costs one read only when its tensor is not already an input of the
    convolution that consumes it.

    Appendix: *"In ResNet, we added memory reads whenever there is a skip connection around two
    convolutional blocks... In BiReal architecture (used in ReactNet-BiReal), the skip connection
    is around one convolution block only and thus can be fused with the convolution, so we do not
    add extra memory accesses for it."*

    Stated as a property of the tensor rather than as "one conv versus two", the rule covers the
    Tower concatenation and the dense-connectivity families with no new field.  It is applied by
    the adapter, which knows the topology, by emitting an extra ``skip_read_bits`` on the
    consuming layer.
    """

    separable_fusion: str = "explicit"
    """How a depthwise+pointwise fusion is identified.

    Appendix: *"MobileNet uses a combination of a depthwise convolution followed by a pointwise
    convolution. We consider it is possible to fuse the two convolutions (with a non-linearity
    inbetween) in a single effective kernel, which needs to read the input features only once."*

    ``"explicit"``     -- the fused pairs are declared by the adapter, which sets
                          :attr:`LayerSpec.output_fused_into_next` on the first member of a pair
                          
    ``"groups_gt_1"``  -- the legacy trigger of ``energy_ECML/evaluate_networks.py``: any convolution
                          with ``groups > 1`` is treated as fused into its successor.  Kept
                          because it is what the published numbers were computed with, and it is
                          what ``tests/test_reference.py`` exercises.  It tests the right
                          property on the wrong object -- whether a *layer* is depthwise, then
                          acting as if it had asked whether a *pair* is fused.
    """

    pooling_output_count: str = "reference"
    """How the element count *after* a fused pooling reduction is obtained.

    ``"reference"`` -- what ``energy_ECML/evaluate_networks.py`` does and what the published numbers
                       are: ``C * H / stride * W / stride`` with **true** division, so an odd
                       spatial size gives a fractional element count (55/2 -> 27.5).  This is
                       arithmetically wrong -- the real output size is
                       ``floor((H - k) / stride) + 1`` -- but it is what the table was computed
                       with, so it is the default and nothing changes silently.
    ``"floor"``     -- the correct output size.  Selecting it changes published numbers.

    Recorded as an assumption rather than fixed as a bug because the difference is a number in
    the paper; ``tests/test_terms.py`` pins both branches.
    """

    memory_traffic: str = "one_pass"
    """How feature-map traffic is counted.

    ``"one_pass"``       -- one read and one write per tensor input of a convolution or linear
                            layer.  This is what the published numbers are.
    ``"tiling_optimal"`` -- the journal extension.  Not implemented; the field names the choice
                            so that the current policy is a choice rather than a silence.
    """

    def __post_init__(self):
        if self.separable_fusion not in ("explicit", "groups_gt_1"):
            raise ValueError(f"unknown separable_fusion policy: {self.separable_fusion!r}")
        if self.pooling_output_count not in ("reference", "floor"):
            raise ValueError(
                f"unknown pooling_output_count policy: {self.pooling_output_count!r}")
        if self.memory_traffic not in ("one_pass", "tiling_optimal"):
            raise ValueError(f"unknown memory_traffic policy: {self.memory_traffic!r}")
        if self.memory_traffic == "tiling_optimal":
            raise NotImplementedError(
                "memory_traffic='tiling_optimal' is named but not implemented"
            )


# ----------------------------------------------------------------------------------------------
# What is described
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolSpec:
    """A pooling reduction fused between a convolution and its output quantization."""

    kind: str
    """``"max"``, ``"avg"`` or ``"adaptive_avg"``."""

    kernel_size: int = 0
    """Square kernel side, for ``"max"`` and ``"avg"``."""

    stride: int = 0
    """Stride, for ``"max"`` and ``"avg"``."""

    output_size: tuple = ()
    """``(H_out, W_out)`` for ``"adaptive_avg"``."""

    output_size_was_int: bool = False
    """Whether the source module was given a bare ``int`` output size.

    This is not cosmetic.  ``evaluate_networks.py`` computes the post-pooling element count of an
    adaptive average pool as ``C * (output_size + output_size)`` when the module was constructed
    with an ``int``, and as ``C * (output_size[0] * output_size[1])`` when it was given a tuple.
    For ``output_size = 1`` those differ (2 versus 1).  The flag preserves the distinction so the
    reference can be reproduced; nothing in our own networks takes the ``int`` branch.
    """

    def __post_init__(self):
        if self.kind not in ("max", "avg", "adaptive_avg"):
            raise ValueError(f"unknown pooling kind: {self.kind!r}")


@dataclass
class LayerSpec:
    """One fused convolution: a convolution, an optional pooling, an affine, a quantization.

    This is the unit the cost model prices.  It corresponds to ``FusedConv`` in
    ``energy_ECML/evaluate_networks.py``, with the single ``K_in``/``K_W`` fields split into
    independent *stored* and *operand* widths.  That split is what retires the two-run
    ``compose()`` convention the published baseline rows were originally produced with: the
    reference implementation derives both the activation memory read and the convolution
    operand width from the same ``K_in``, so a row that stores 8-bit feature maps while
    charging operands at the width its paper states could only be expressed as two runs
    composed together.  Here it is one object and one evaluation.

    The five width fields are **numbers of states** (see the module docstring), which is what
    the training codebase deals in and what a quantizer's ``quant.K`` holds.  Each has a
    read-only ``b_``-named property giving the same width in **bits**, and the cost model uses
    only those; nothing multiplies or compares the two units without going through
    :func:`bits_for` / :func:`states_for`.  A figure quoted in bits is written as
    ``states_for(8)``, never as ``8`` -- that literal substitution is the published error of
    the erratum.
    """

    name: str
    kind: str = "conv"
    """``"conv"`` or ``"linear"``.  A linear layer is described with a ``(out, in, 1, 1)``
    weight shape and ``(1, in, 1, 1)`` shapes, so it is priced as a 1x1 convolution."""

    in_shape: tuple = ()
    """``(N, C_in, H_in, W_in)`` of the convolution's input, batch size 1."""

    out_shape: tuple = ()
    """``(N, C_out, H_out, W_out)`` of the convolution's output, *before* any pooling."""

    weight_shape: tuple = ()
    """``(C_out, C_in / groups, k_h, k_w)``."""

    groups: int = 1
    stride: int = 1
    dilation: int = 1

    pooling: Optional[PoolSpec] = None

    K_in_stored: int = 256
    """**States** the input activation occupies *in memory*.  Prices the read.  Bits:
    :attr:`b_in_stored`."""

    K_in_operand: int = 256
    """**States** the input activation has *as a multiplier operand*.  Prices the dot product and
    sets the accumulator width.  Bits: :attr:`b_in_operand`."""

    K_W_stored: int = 256
    """**States** a weight occupies in memory.  Prices the weight read.  Bits:
    :attr:`b_W_stored`."""

    K_W_operand: int = 256
    """**States** a weight has as a multiplier operand.  Bits: :attr:`b_W_operand`."""

    K_out_stored: int = 256
    """**States** the output activation occupies in memory.  Prices the write.  ``0`` means the
    output is not written out at all.  Bits: :attr:`b_out_stored`."""

    skip_read_bits: int = 0
    """Extra activation-read bits charged for a residual path consumed by this layer.

    Set by the adapter, which knows the topology, under
    :attr:`Assumptions.skip_read_when_not_already_an_input`.  Zero for Bi-Real style skips over a
    single convolution and for Tower concatenations; the full tensor for a ResNet skip over two.
    """

    output_fused_into_next: bool = False
    """This layer's output is consumed by a fused successor, so it is neither written to memory
    nor read back.

    Under ``Assumptions.separable_fusion == "explicit"`` the adapter declares this on the first
    member of a fused pair.  Under ``"groups_gt_1"`` the adapter sets it from ``groups > 1``, and
    the model additionally reproduces the reference's *negative read* bookkeeping -- see
    :func:`TNet.energy_ECML.model.count_layer`.
    """

    stage: Optional[str] = None
    """Resolution stage label.  Derived from the output spatial size when ``None``."""

    # -- the same five widths in bits ----------------------------------------------------------
    # Accessors, not fields: the states are the single stored copy, so the two units cannot
    # disagree and a record cannot be written with a bit count in a states slot.

    @property
    def b_in_stored(self) -> int:
        """:attr:`K_in_stored` in **bits**."""
        return bits_for(self.K_in_stored)

    @property
    def b_in_operand(self) -> int:
        """:attr:`K_in_operand` in **bits**."""
        return bits_for(self.K_in_operand)

    @property
    def b_W_stored(self) -> int:
        """:attr:`K_W_stored` in **bits**."""
        return bits_for(self.K_W_stored)

    @property
    def b_W_operand(self) -> int:
        """:attr:`K_W_operand` in **bits**."""
        return bits_for(self.K_W_operand)

    @property
    def b_out_stored(self) -> int:
        """:attr:`K_out_stored` in **bits**."""
        return bits_for(self.K_out_stored)

    def resolved_stage(self) -> str:
        """The stage label, derived from the output height if not set explicitly."""
        if self.stage is not None:
            return self.stage
        if len(self.out_shape) >= 3:
            return f"{int(self.out_shape[2])}x{int(self.out_shape[3])}"
        return "?"

    def n_weights(self) -> int:
        """Number of weight elements.  ``prod(weight_shape)``."""
        w = self.weight_shape
        return int(w[0]) * int(w[1]) * int(w[2]) * int(w[3])

    def dot_length(self) -> int:
        """Length ``n`` of the dot product the appendix prices: ``C_in k^2 / groups``.

        Appendix: *"The vector length is :math:`n = C_{in} k^2`."*  The division by ``groups`` is
        what the reference implementation does and is correct: a grouped convolution's dot
        product only runs over the channels of its own group.
        """
        return (int(self.weight_shape[2]) * int(self.weight_shape[3]) * int(self.in_shape[1])) // self.groups


# ----------------------------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------------------------


@dataclass
class Counts:
    """Technology-independent counts.  Multiply by a :class:`Technology` to get joules.

    Gate counts are in units of a one-bit adder operation.  Register accesses are counted
    separately because a register bit costs :attr:`Technology.gamma_0` times as much -- keeping
    them apart is what lets a record be re-priced for another technology without re-running the
    count.  Memory figures are in bits.

    Only the dot product uses registers; the pooling, affine and requantization terms of the
    reference model are pure adder/shifter work.
    """

    n_mults: float = 0.0
    n_adds: float = 0.0
    n_cmps: float = 0.0
    n_shifts: float = 0.0

    gates_conv_adder: float = 0.0
    gates_conv_register: float = 0.0
    gates_pool: float = 0.0
    gates_affine: float = 0.0
    gates_requantize: float = 0.0

    bits_read_act: float = 0.0
    bits_read_weight: float = 0.0
    bits_written_act: float = 0.0

    @property
    def gates_adder(self) -> float:
        """All one-bit adder/shifter operations, priced at :attr:`Technology.E1_pJ`."""
        return (self.gates_conv_adder + self.gates_pool
                + self.gates_affine + self.gates_requantize)

    @property
    def gates_register(self) -> float:
        """All register bit accesses, priced at ``gamma_0 * E1_pJ``."""
        return self.gates_conv_register

    _SUMMED = (
        "n_mults", "n_adds", "n_cmps", "n_shifts",
        "gates_conv_adder", "gates_conv_register",
        "gates_pool", "gates_affine", "gates_requantize",
        "bits_read_act", "bits_read_weight", "bits_written_act",
    )

    def __iadd__(self, other: "Counts") -> "Counts":
        for f in Counts._SUMMED:
            setattr(self, f, getattr(self, f) + getattr(other, f))
        return self

    def to_dict(self) -> dict:
        return {f: getattr(self, f) for f in Counts._SUMMED}


@dataclass
class _EnergyBase:
    """Named energy fields shared by the per-layer, per-stage and total rows.

    All energies are in picojoules.  ``uJ`` properties are provided because that is the unit the
    paper's tables print.
    """

    counts: Counts = field(default_factory=Counts)

    CEE: float = 0.0
    """Compute Energy Estimate, pJ."""

    CEE_conv: float = 0.0
    CEE_pool: float = 0.0
    CEE_affine: float = 0.0
    CEE_requantize: float = 0.0

    MMEE_weights: float = 0.0
    """Memory Movement Energy Estimate for weight reads, pJ."""

    MMEE_features: float = 0.0
    """Memory Movement Energy Estimate for activation reads *and* writes, pJ."""

    @property
    def MMEE(self) -> float:
        """Total memory movement energy, pJ."""
        return self.MMEE_weights + self.MMEE_features

    @property
    def TEE(self) -> float:
        """Total Energy Estimate, pJ."""
        return self.CEE + self.MMEE

    @property
    def MB_weights(self) -> float:
        return self.counts.bits_read_weight / 8 / 1024 / 1024

    @property
    def MB_features(self) -> float:
        return (self.counts.bits_read_act + self.counts.bits_written_act) / 8 / 1024 / 1024

    def as_row(self) -> dict:
        """The table row this object contributes, in the units the paper prints."""
        return dict(
            MB_weights=self.MB_weights,
            uJ_weights=self.MMEE_weights / 1e6,
            MB_features=self.MB_features,
            uJ_features=self.MMEE_features / 1e6,
            uJ_CEE=self.CEE / 1e6,
            uJ_MMEE=self.MMEE / 1e6,
            uJ_TEE=self.TEE / 1e6,
        )


@dataclass
class LayerEnergy(_EnergyBase):
    name: str = ""
    stage: str = ""
    spec: Optional[LayerSpec] = None


@dataclass
class StageEnergy(_EnergyBase):
    name: str = ""
    n_layers: int = 0


@dataclass
class NetworkEnergy(_EnergyBase):
    """The result of :func:`TNet.energy_ECML.model.evaluate`."""

    name: str = ""
    layers: list = field(default_factory=list)
    stages: dict = field(default_factory=dict)
    technology: Technology = field(default_factory=Technology)
    assumptions: Assumptions = field(default_factory=Assumptions)

    max_bits_read_act: float = 0.0
    """Largest single-layer activation read, in bits.  The reference prints this as
    "Max activations in"; it bounds the on-chip buffer a one-pass schedule would need."""

    max_bits_written_act: float = 0.0
