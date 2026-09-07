# EWGS

Status: REVIEWED — 2026-09-07, built and checked with the author.

Method: EWGS, *Network Quantization with Element-wise Gradient Scaling*, Lee, Kim & Ham, CVPR 2021
(arXiv 2104.00903). Five configurations, W/A from 1/1 to 4/4, accuracies as published.

Entry path: **traced**. EWGS is a training method, not an architecture: it quantizes an
**unmodified ImageNet ResNet-18**. So the geometry is not EWGS's at all and nothing about it is
transcribed — only the widths are the method's.

## Which version is evaluated

`torchvision.models.resnet18`, traced. It is pinned in [`sources.toml`](sources.toml) as an
installed package rather than a clone, so the record stores the torchvision **version actually
imported** at build time in place of a commit. torchvision is the canonical reference
implementation and is what the paper means by "ResNet-18".

The builder is [`../methods/torchvision_resnet.py`](../methods/torchvision_resnet.py), which the
full-precision and 8-bit ResNet reference rows also use.

## How the row is built

The 8-bit storage assumption, unchanged from
[`../docs/methodology.md`](../docs/methodology.md): stored feature maps capped at 8 bits,
convolution operands at the width the paper states, first and last layers unquantized, 8-bit
network input. In code that is `uniform_policy(K_A_operand, K_W, first_last_names=("conv1", "fc"))`
— the unquantized pair is **named**, rather than inferred as "the first and last in the trace", so
the choice is visible.

One reading is not settled by the paper and is therefore recorded: **the three 1×1 downsample
convolutions are quantized like every other layer.** Leaving them at 8 bits instead — the
arrangement ReActNet uses — is the obvious alternative, and it is worth **+2.5 % of the total at
binary widths, +1.8 % at 4 bits and +0.9 % at 16**, measured by
`tests/test_bitwidths.py::test_quantizing_the_downsample_convolutions_is_worth_this_much`.

These rows have no counterpart in the published table; they were added after it.
