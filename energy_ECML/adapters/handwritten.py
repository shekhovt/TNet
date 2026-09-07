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
# Layer descriptions written out by hand, for an architecture with no runnable implementation to
# trace. XNOR-Net's binarized AlexNet is the only one left; conv() is the helper its geometry is
# written with.

"""Architectures described by writing their geometry out.

These geometries are where the published baseline rows come from.  They were first written for a
script that described each architecture as a list of ``FusedConv`` descriptors and priced them
with ``evaluate_networks.py`` unmodified; this is that description ported to
:class:`~TNet.energy_ECML.spec.LayerSpec`.  The one substantive difference is that a stored width
and an operand width are now separate fields, so the **two-run ``compose()`` convention that
script needed is gone**: a row is one object and one evaluation.

The storage assumption these baselines are priced under (``energy_ECML/docs/methodology.md``):

* stored feature maps are capped at 8 bits even when the paper keeps them fp32;
* convolution operands are charged at the width the paper states;
* first and last layers are not quantized, so 8 bits per weight;
* the network input is an 8-bit image.

The affine coefficients are *not* covered by that assumption; their width is a field of
``Technology``.

Those bullets are in **bits**, because that is how the papers state them.  Every argument and
every field below is a **number of states** ``K``, because that is what a
:class:`~TNet.energy_ECML.spec.LayerSpec` holds: 1 bit is ``K = 2``, 8 bits ``K = 256``, f32
``K = 2**32 =`` :data:`F32`.  The two units are told apart by the first character of the name;
see the naming convention in :mod:`TNet.energy_ECML.spec`.
"""

from __future__ import annotations

from typing import Optional

from ..spec import LayerSpec, PoolSpec, states_for

__all__ = ["F32", "conv", "xnor_alexnet"]

#: States of a value stored as f32 -- 32 bits, hence ``2**32``.  A ``K``-named constant, so it
#: is a state count: passing ``32`` instead would mean 32 states, five bits.
F32 = states_for(32)


def conv(
    name: str,
    c_in: int,
    c_out: int,
    k: int,
    stride: int,
    in_res: int,
    out_res: int,
    *,
    K_in_stored: int,
    K_in_operand: int,
    K_W_stored: int,
    K_W_operand: int,
    K_out_stored: int,
    groups: int = 1,
    pooling: Optional[PoolSpec] = None,
    skip_read_bits: int = 0,
) -> LayerSpec:
    """One fused convolution of a square feature map.

    The five ``K_``-named arguments are **numbers of states**; ``skip_read_bits`` is, as its
    name says, a count of **bits**.
    """
    return LayerSpec(
        name=name,
        kind="conv",
        in_shape=(1, c_in, in_res, in_res),
        out_shape=(1, c_out, out_res, out_res),
        weight_shape=(c_out, c_in // groups, k, k),
        groups=groups,
        stride=stride,
        pooling=pooling,
        K_in_stored=K_in_stored,
        K_in_operand=K_in_operand,
        K_W_stored=K_W_stored,
        K_W_operand=K_W_operand,
        K_out_stored=K_out_stored,
        skip_read_bits=skip_read_bits,
    )


def xnor_alexnet(
    K_A_operand: int,
    K_W: int,
    *,
    K_first_last: int = 256,
    K_in_first: int = 256,
    K_stored: Optional[int] = None,
    K_logits: int = 256,
) -> list[LayerSpec]:
    """XNOR-Net: the binarized AlexNet of Rastegari et al., ECCV 2016, at 224x224.

    The architecture is the two-branch
    Krizhevsky AlexNet that XNOR-Net's own footnote points at
    (https://gist.github.com/szagoruyko/dd032c529048492630fc), per branch::

        conv 3->48   11x11 s4 p2 | BN ReLU | maxpool 3x3 s2
        conv 48->128 5x5 p2      | BN ReLU | maxpool 3x3 s2
        conv 128->192 3x3 p1     | BN ReLU
        conv 192->192 3x3 p1     | BN ReLU
        conv 192->128 3x3 p1     | BN ReLU | maxpool 3x3 s2

    then, on the concatenation of the two branches (256 x 6 x 6 = 9216), three fully
    connected layers 9216->4096->4096->1000, described as convolutions.

    The two branches are written out as two ``groups=1`` stacks rather than folded into a
    grouped convolution.  Under ``Assumptions.separable_fusion == "groups_gt_1"`` -- the legacy
    rule, and the one ``tests/test_reference.py`` exercises -- a layer with ``groups > 1`` is
    treated as fused into its successor and loses its activation writes, which would be wrong
    here.  Concatenation is free, so two stacks and one
    grouped stack describe the same memory traffic.

    XNOR-Net does not binarize the first and last layer (paper, section 4.1), so ``conv1`` and
    the classifier take ``K_first_last`` and everything between them is binary.

    Parameters
    ----------
    All widths are **numbers of states**, not bit counts: binary is ``2``, 8-bit is ``256``,
    f32 is :data:`F32`.

    K_stored
        States of every stored feature map.  **Defaults to ``K_A_operand``, not to 256.**

        This is not an exception to the 8-bit storage assumption of
        ``energy_ECML/docs/methodology.md``: that assumption is an upper bound, and this network
        sits under it.  The 8-bit figure matters for a Bi-Real style network, whose real-valued
        skip path means its tensors are not binary in memory whatever the paper's label says.
        AlexNet has no skip connections, so its binary activations really are the tensors written
        to memory, and one bit is what the architecture stores rather than an optimistic reading
        of it.  ``tests/test_xnor.py`` measures what pricing it at 8 bits instead would cost.

    ``alpha``/``beta`` scaling factors are not modelled, so this is a lower bound -- the same
    caveat the slide carries.
    """
    if K_stored is None:
        K_stored = K_A_operand
    mp = PoolSpec(kind="max", kernel_size=3, stride=2)

    def c(name, c_in, c_out, k, stride, in_res, out_res, *, first=False, last=False,
          pooling=None):
        K_W_ = K_first_last if (first or last) else K_W
        return conv(
            name, c_in, c_out, k, stride, in_res, out_res,
            K_in_stored=K_in_first if first else K_stored,
            K_in_operand=K_in_first if first else (K_stored if last else K_A_operand),
            K_W_stored=K_W_, K_W_operand=K_W_,
            K_out_stored=K_logits if last else K_stored,
            pooling=pooling,
        )

    rr: list[LayerSpec] = []
    for b in range(2):                                  # two identical branches, concatenated
        rr += [
            c(f"branch{b}.conv1", 3, 48, 11, 4, 224, 55, first=True, pooling=mp),
            c(f"branch{b}.conv2", 48, 128, 5, 1, 27, 27, pooling=mp),
            c(f"branch{b}.conv3", 128, 192, 3, 1, 13, 13),
            c(f"branch{b}.conv4", 192, 192, 3, 1, 13, 13),
            c(f"branch{b}.conv5", 192, 128, 3, 1, 13, 13, pooling=mp),
        ]
    rr += [
        c("fc6", 256, 4096, 6, 1, 6, 1),
        c("fc7", 4096, 4096, 1, 1, 1, 1),
        c("fc8", 4096, 1000, 1, 1, 1, 1, last=True),
    ]
    return rr
