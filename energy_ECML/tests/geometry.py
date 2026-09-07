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
# Hand-written ResNet-18 and ResNet-50 geometry, kept solely as the oracle that the tracer is
# checked against. Nothing in the package imports it; it is test code.

"""ResNet-18 and ResNet-50 geometry written out by hand -- the tests' oracle.

**This is test code and nothing in the package imports it.**  The ResNet rows of the registry are
produced by tracing ``torchvision``; see :mod:`TNet.energy_ECML.methods.torchvision_resnet`.  The
geometry here exists so that the traced path has something independent to be checked against: it
was transcribed from the architecture, it knows nothing about ``torchvision``, and
``test_adapters.py`` asserts the two agree field by field.  A tracer that silently dropped a layer
or mis-read a stride would otherwise have no way of being caught.

It also carries two variations that no released row uses and that therefore have no place in the
package's public surface:

* ``bireal_downsample`` -- the Bi-Real / ReActNet downsample, which is the oracle for the traced
  ReActNet entry;
* ``skip_reads`` -- the residual read the appendix describes but the published numbers do not
  charge, kept so the size of that disagreement is a measured number.

All widths are **numbers of states**, not bit counts: binary is ``2``, 8-bit is ``256``, f32 is
:data:`F32`.  See the naming convention in :mod:`TNet.energy_ECML.spec`.
"""

from __future__ import annotations

from typing import Optional

from ..adapters.handwritten import F32, conv
from ..spec import LayerSpec, PoolSpec, bits_for

__all__ = ["F32", "conv", "resnet18", "resnet50"]


def resnet18(
    K_A_operand: int,
    K_W: int,
    *,
    K_first_last: int = 256,
    K_in_first: int = 256,
    K_stored: int = 256,
    K_ds_W: Optional[int] = None,
    K_ds_A: Optional[int] = None,
    bireal_downsample: bool = False,
    skip_reads: bool = False,
) -> list[LayerSpec]:
    """Standard ImageNet ResNet-18 at 224x224.

    Parameters
    ----------
    K_A_operand
        States of the *convolution input operand* of the quantized layers.
    K_W
        States per weight of the quantized layers.
    K_first_last
        States per weight of the unquantized stem convolution and classifier.
    K_in_first
        States of the network input.  ``256`` -- the image is 8-bit.
    K_stored
        States of every stored feature map.  ``256`` (8 bits) under the storage assumption.
    K_ds_W, K_ds_A
        States of the three 1x1 downsample convolutions' weights and input operands; default to
        the plain layers'.
    bireal_downsample
        If true the downsample is an average pool followed by a *stride-1* 1x1 convolution (the
        Bi-Real / ReActNet arrangement), so that 1x1 convolution reads a 4x smaller tensor.  If
        false it is the standard stride-2 1x1 convolution reading the full-resolution tensor.
    skip_reads
        Charge the appendix's residual read.  Appendix: *"In ResNet, we added memory reads
        whenever there is a skip connection around two convolutional blocks, i.e. we assume that
        the fused convolution reads an additional input that is added before the pooling."*

        **Default false, because that is what the published rows are.**  The code that produced
        ``tab:imagenet1k-detailed`` charges no such read, so the appendix sentence and the
        published ResNet numbers disagree.  The flag makes the size of the disagreement a measured
        number instead of an argument; ``test_reference.py`` pins it.
    """
    if K_ds_W is None:
        K_ds_W = K_W
    if K_ds_A is None:
        K_ds_A = K_A_operand

    rr: list[LayerSpec] = []
    # stem: 7x7 stride 2, then a 3x3 stride 2 max pool fused into it
    rr.append(conv(
        "conv1", 3, 64, 7, 2, 224, 112,
        K_in_stored=K_in_first, K_in_operand=K_in_first,
        K_W_stored=K_first_last, K_W_operand=K_first_last,
        K_out_stored=K_stored,
        pooling=PoolSpec(kind="max", kernel_size=3, stride=2),
    ))

    stages = [(64, 64, 56, False), (64, 128, 28, True), (128, 256, 14, True), (256, 512, 7, True)]
    for si, (cin, cout, res, ds) in enumerate(stages):
        prev_res = res * 2 if ds else res
        for bi in range(2):
            # A basic block is conv-conv with a skip around the pair: the skipped tensor is not
            # an input of the second convolution, so it costs a read (Assumptions.
            # skip_read_when_not_already_an_input).  The read is charged on the second
            # convolution, which is where the addition happens.
            first_in = cin if (ds and bi == 0) else cout
            first_res = prev_res if (ds and bi == 0) else res
            first_stride = 2 if (ds and bi == 0) else 1
            rr.append(conv(
                f"layer{si+1}.{bi}.conv1", first_in, cout, 3, first_stride, first_res, res,
                K_in_stored=K_stored, K_in_operand=K_A_operand,
                K_W_stored=K_W, K_W_operand=K_W, K_out_stored=K_stored,
            ))
            skip_bits = 0
            if skip_reads:
                # the block input, at its stored width and its own resolution
                skip_bits = bits_for(K_stored) * first_in * first_res * first_res
            rr.append(conv(
                f"layer{si+1}.{bi}.conv2", cout, cout, 3, 1, res, res,
                K_in_stored=K_stored, K_in_operand=K_A_operand,
                K_W_stored=K_W, K_W_operand=K_W, K_out_stored=K_stored,
                skip_read_bits=skip_bits,
            ))
            if ds and bi == 0:
                if bireal_downsample:      # average pool by 2, then a stride-1 1x1
                    rr.append(conv(
                        f"layer{si+1}.0.downsample.0", cin, cout, 1, 1, res, res,
                        K_in_stored=K_stored, K_in_operand=K_ds_A,
                        K_W_stored=K_ds_W, K_W_operand=K_ds_W, K_out_stored=K_stored,
                    ))
                else:                      # standard stride-2 1x1
                    rr.append(conv(
                        f"layer{si+1}.0.downsample.0", cin, cout, 1, 2, prev_res, res,
                        K_in_stored=K_stored, K_in_operand=K_ds_A,
                        K_W_stored=K_ds_W, K_W_operand=K_ds_W, K_out_stored=K_stored,
                    ))

    # global average pool fused into the last convolution, then the 1x1 classifier
    rr[-1].pooling = PoolSpec(kind="adaptive_avg", output_size=(1, 1))
    rr.append(conv(
        "fc", 512, 1000, 1, 1, 1, 1,
        K_in_stored=K_stored, K_in_operand=K_stored,
        K_W_stored=K_first_last, K_W_operand=K_first_last,
        K_out_stored=K_stored,
    ))
    return rr


_RESNET50_STAGES = [
    # (planes, blocks, out_res, in_res_of_stage)
    (64, 3, 56, 56),
    (128, 4, 28, 56),
    (256, 6, 14, 28),
    (512, 3, 7, 14),
]


def resnet50(
    K_A_operand: int,
    K_W: int,
    *,
    K_first_last: int = 256,
    K_in_first: int = 256,
    K_stored: int = 256,
    skip_reads: bool = False,
) -> list[LayerSpec]:
    """Standard ImageNet ResNet-50 at 224x224, bottleneck ``1x1 -> 3x3 -> 1x1``.

    All widths are **numbers of states**, as in :func:`resnet18`.

    All 53 convolutions have ``groups = 1``: ResNet-50 has no depthwise convolution and therefore
    no candidate for the depthwise+pointwise fusion the appendix describes.
    """

    rr: list[LayerSpec] = []
    rr.append(conv(
        "conv1", 3, 64, 7, 2, 224, 112,
        K_in_stored=K_in_first, K_in_operand=K_in_first,
        K_W_stored=K_first_last, K_W_operand=K_first_last,
        K_out_stored=K_stored,
        pooling=PoolSpec(kind="max", kernel_size=3, stride=2),
    ))
    c_prev = 64
    for si, (planes, blocks, res, in_res) in enumerate(_RESNET50_STAGES):
        for bi in range(blocks):
            first = (bi == 0)
            r_in = in_res if first else res
            stride = 2 if (first and si > 0) else 1
            rr.append(conv(
                f"layer{si+1}.{bi}.conv1", c_prev, planes, 1, 1, r_in, r_in,
                K_in_stored=K_stored, K_in_operand=K_A_operand,
                K_W_stored=K_W, K_W_operand=K_W, K_out_stored=K_stored,
            ))
            rr.append(conv(
                f"layer{si+1}.{bi}.conv2", planes, planes, 3, stride, r_in, res,
                K_in_stored=K_stored, K_in_operand=K_A_operand,
                K_W_stored=K_W, K_W_operand=K_W, K_out_stored=K_stored,
            ))
            skip_bits = bits_for(K_stored) * c_prev * r_in * r_in if skip_reads else 0
            rr.append(conv(
                f"layer{si+1}.{bi}.conv3", planes, planes * 4, 1, 1, res, res,
                K_in_stored=K_stored, K_in_operand=K_A_operand,
                K_W_stored=K_W, K_W_operand=K_W, K_out_stored=K_stored,
                skip_read_bits=skip_bits,
            ))
            if first:
                rr.append(conv(
                    f"layer{si+1}.0.downsample.0", c_prev, planes * 4, 1, stride, r_in, res,
                    K_in_stored=K_stored, K_in_operand=K_A_operand,
                    K_W_stored=K_W, K_W_operand=K_W, K_out_stored=K_stored,
                ))
            c_prev = planes * 4
    rr[-1].pooling = PoolSpec(kind="adaptive_avg", output_size=(1, 1))
    rr.append(conv(
        "fc", 2048, 1000, 1, 1, 1, 1,
        K_in_stored=K_stored, K_in_operand=K_stored,
        K_W_stored=K_first_last, K_W_operand=K_first_last,
        K_out_stored=K_stored,
    ))
    return rr
