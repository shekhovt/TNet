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
# Turns one of our own networks into layer descriptions: builds it from a command-line string,
# runs one forward pass with shape hooks, and reuses the reference implementation's own fusion
# logic so there is only one source of truth about it. Also the boundary where the reference's
# two bits-written-as-states literals are repaired and where an unpadded first layer is given
# the input size that makes it equivalent to a padded 224.

"""Our own networks: build with ``create_net``, walk with the reference's fusion logic.

The layer *list* is produced by ``energy_ECML/evaluate_networks.py:fuse_model``, unmodified.  That is
deliberate: the fusion rules (which convolution absorbs which pooling, where a quantizer's width
comes from, the ``in_K = 8`` network input) are the thing the published numbers depend on, and
duplicating them here would create a second source of truth that could drift.  What this module
adds is the translation of each ``FusedConv`` into a :class:`~TNet.energy_ECML.spec.LayerSpec`, with

* the five state-count fields set explicitly rather than derived from one, and
* a name for every layer, recovered from ``net.named_modules()``, so a per-layer table has row
  labels instead of indices.

``fuse_model`` gives a single ``K_in`` per layer, which becomes both ``K_in_stored`` and
``K_in_operand``.  For our own networks that is correct -- a TNet activation really is stored at
the width it is multiplied at.  It is the *baselines* that need the two to differ, and those go
through :mod:`TNet.energy_ECML.adapters.handwritten` or :mod:`TNet.energy_ECML.adapters.traced`.

**This module is the units boundary.**  Everything ``fuse_model`` hands over -- ``r.K_in``,
``r.K_W``, ``r.K_out`` -- is a **number of states**, taken from a quantizer's ``quant.K`` in the
training code, and it need not be a power of two.  Two of its literals are bit counts written
into those state-valued fields; :func:`correct_bits_as_levels` is the repair, and it is the only
place in the package where a value crossing this boundary is reinterpreted.
"""

from __future__ import annotations

import io
import contextlib
from typing import Optional

import torch
import torch.nn as nn

from ..spec import LayerSpec, PoolSpec, states_for

__all__ = ["pool_spec_from_module", "spec_from_fused", "specs_from_net", "specs_from_cfg",
           "K_8BIT", "correct_bits_as_levels", "equivalent_input_size", "REFERENCE_IMAGE_SIZE"]

#: The input size the comparison is defined at.  A network whose first convolution pads sees this
#: many pixels; one that does not pad sees more -- see :func:`equivalent_input_size`.
REFERENCE_IMAGE_SIZE = 224

#: Number of **states** of an 8-bit quantity, i.e. 256.  ``K`` is a count of states everywhere in
#: this package and in the training code (a quantizer's ``K`` need not be a power of two), so
#: "8 bits" written into a ``K``-named field is ``states_for(8) = 256``, never the literal ``8``
#: -- which would mean eight states, three bits.
K_8BIT = states_for(8)


def pool_spec_from_module(p) -> Optional[PoolSpec]:
    """Translate a pooling module into a :class:`~TNet.energy_ECML.spec.PoolSpec`."""
    if p is None:
        return None
    if isinstance(p, nn.MaxPool2d):
        k, s = p.kernel_size, p.stride
        return PoolSpec(kind="max", kernel_size=_as_int(k), stride=_as_int(s))
    if isinstance(p, nn.AvgPool2d):
        k, s = p.kernel_size, p.stride
        return PoolSpec(kind="avg", kernel_size=_as_int(k), stride=_as_int(s))
    if isinstance(p, nn.AdaptiveAvgPool2d):
        o = p.output_size
        if isinstance(o, int):
            return PoolSpec(kind="adaptive_avg", output_size=(o, o), output_size_was_int=True)
        return PoolSpec(kind="adaptive_avg", output_size=tuple(o))
    raise TypeError(f"unsupported pooling module: {type(p).__name__}")


def _as_int(v) -> int:
    """The reference squares ``kernel_size`` directly, so a tuple kernel is not representable."""
    if isinstance(v, int):
        return v
    if isinstance(v, (tuple, list)):
        if len(set(v)) != 1:
            raise ValueError(f"non-square pooling {v} is not supported by the reference model")
        return int(v[0])
    raise TypeError(v)


def _is_quantized_conv(conv) -> bool:
    """Whether ``fuse_model`` took this layer's widths from a quantizer or from its own literals.

    ``fuse_model`` reads ``l.quantizer.quant.K`` -- a number of states -- for a ``QConv2d``, and
    writes a literal for a plain ``nn.Conv2d``/``nn.Linear``.  Only the literal is suspect, which
    is what makes the correction below safe: a genuine ``-W 8`` run is a quantizer's ``K``, eight
    states, and is left alone.
    """
    if conv is None:
        return False
    from ...layers import QConv2d
    return isinstance(conv, QConv2d)


def correct_bits_as_levels(specs: list[LayerSpec], convs=None) -> list[LayerSpec]:
    """Repair ``fuse_model``'s two bits-as-levels literals, in place.

    ``energy_ECML/evaluate_networks.py:fuse_model`` writes ``8`` for two quantities that are meant to
    be *8 bits* but land in fields counting *states*, where 8 bits is 256 states:

    * ``in_K = 8`` -- the network input, i.e. the image, which is charged 3 bits;
    * ``fc.K_W = 8`` -- the weights of an unquantized (plain ``nn.Conv2d``/``nn.Linear``) layer,
      likewise 3 bits, where the assumed storage width is 8.

    That both are the mistake and not a choice is established three ways: the same function writes the same quantity correctly three lines away as
    ``fc.K_out = 2**8``; its MobileNet branch already overrides *exactly these two* literals to
    256; and the image really is 8-bit (``setup_imagenet.py`` loads ``uint8`` and scales by
    1/255, with no input quantizer in the forward pass).

    **This changes published numbers** and is applied here rather than in ``evaluate_networks.py``,
    which stays untouched as the reference and the tests' oracle (``energy_ECML/SOTA/README.md`` rule
    1).  It changes published numbers, and what each row moves by is recorded with the erratum.

    Only two things are touched, and neither can catch a genuine low-state setting: the **first**
    layer's input width, identified by position, and the weight width of a layer that is **not** a
    ``QConv2d``, identified by module type.  A real ``-A 8`` or ``-W 8`` run -- 8 states, 3 bits,
    which is a legitimate configuration here -- is left exactly as it is.
    """
    if not specs:
        return specs
    if specs[0].K_in_stored == 8:
        specs[0].K_in_stored = specs[0].K_in_operand = K_8BIT
    for s, conv in zip(specs, convs or [None] * len(specs)):
        if s.K_W_stored == 8 and not _is_quantized_conv(conv):
            s.K_W_stored = s.K_W_operand = K_8BIT
    return specs


def equivalent_input_size(first: LayerSpec, image_size: int = REFERENCE_IMAGE_SIZE) -> int:
    """The input size at which an unpadded first layer sees the image a padded one would see.

    Our stem (``arch_imagenet.Block0_b1``, used by TNet and BiNeal) is a 7x7 stride-2 convolution
    with ``padding=0`` -- deliberately, so that the image is not padded with black pixels -- and
    the loader feeds it a larger crop to compensate (``setup_imagenet.py``: ``input_size = 230``).
    Evaluated at 224 it would produce a 109x109 feature map where the padded stem of a ResNet
    produces 112x112, so every layer downstream would be priced on a smaller tensor than the one
    the network actually computes.  That is not a cheaper network, it is a smaller input.

    This function restores the comparison: given the *first* layer's geometry, it returns the
    smallest input ``H`` for which that layer's output equals the output a ``padding = k // 2``
    convolution would give from ``image_size``.  With the 7x7 stride-2 unpadded stem and
    ``image_size = 224`` that is **229** -- the full 112x112 ladder, one pixel less than the 230
    the loader happens to use (230 also gives 112; 229 is the least that does).

    A layer that already pads is returned unchanged: the rule only ever grows the input, and only
    to the point of equivalence.

    The padding is inferred from the traced shapes rather than read off the module, so this works
    for any :class:`~TNet.energy_ECML.spec.LayerSpec` regardless of which adapter produced it.
    """
    if first.kind != "conv":
        return image_size
    k = first.weight_shape[2]
    s_ = first.stride
    d = first.dilation
    k_eff = d * (k - 1) + 1
    in_h = first.in_shape[2]
    out_h = first.out_shape[2]
    # out = floor((in + 2p - k_eff) / s) + 1, solved for the smallest p consistent with it
    pad = -(-(s_ * (out_h - 1) + k_eff - in_h) // 2)
    pad = max(pad, 0)
    target = (image_size + 2 * (k_eff // 2) - k_eff) // s_ + 1
    if out_h >= target and in_h >= image_size:
        return image_size                       # already padded (or larger); nothing to restore
    need = s_ * (target - 1) + k_eff - 2 * pad
    return max(need, image_size)


def spec_from_fused(r, name: str = "") -> LayerSpec:
    """Translate one ``FusedConv`` into a :class:`~TNet.energy_ECML.spec.LayerSpec`."""
    if hasattr(r, "conv"):
        w = tuple(int(x) for x in r.conv.weight.shape)
        groups = int(getattr(r.conv, "groups", 1))
        stride = getattr(r.conv, "stride", 1)
        dilation = getattr(r.conv, "dilation", 1)
        kind = "linear" if isinstance(r.conv, nn.Linear) else "conv"
    else:
        w = tuple(int(x) for x in r.weights_size)
        groups, stride, dilation, kind = 1, 1, 1, "conv"
    if len(w) == 2:                      # nn.Linear: price it as a 1x1 convolution
        w = (w[0], w[1], 1, 1)
    return LayerSpec(
        name=name,
        kind=kind,
        in_shape=tuple(int(x) for x in r.in_size),
        out_shape=tuple(int(x) for x in r.conv_out_size),
        weight_shape=w,
        groups=groups,
        stride=_as_int(stride) if not isinstance(stride, int) else stride,
        dilation=_as_int(dilation) if not isinstance(dilation, int) else dilation,
        pooling=pool_spec_from_module(r.pooling),
        # r.K_in / r.K_W / r.K_out are numbers of states, straight from fuse_model
        K_in_stored=r.K_in,
        K_in_operand=r.K_in,
        K_W_stored=r.K_W,
        K_W_operand=r.K_W,
        K_out_stored=r.K_out,
    )


def specs_from_net(net, quiet: bool = True, correct_units: bool = True) -> list[LayerSpec]:
    """Run one forward pass over ``net`` with shape hooks, fuse it, and describe the result.

    ``net`` must already have been through :func:`add_hooks` and a forward pass, or use
    :func:`specs_from_cfg`, which does the whole thing.

    ``correct_units`` applies :func:`correct_bits_as_levels`.  It is on by default; pass ``False``
    to get the widths exactly as ``fuse_model`` wrote them, which is what the published table was
    computed with.
    """
    from ..evaluate_networks import fuse_model

    names = {}
    for n, m in net.named_modules():
        names[id(m)] = n

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext():
        rr = fuse_model(net)

    specs = []
    for i, r in enumerate(rr):
        nm = names.get(id(getattr(r, "conv", None)), "")
        specs.append(spec_from_fused(r, nm or f"layer{i:03d}"))
    if correct_units:
        correct_bits_as_levels(specs, [getattr(r, "conv", None) for r in rr])
    return specs


def specs_from_cfg(
    cfg_string: str,
    num_classes: int = 1000,
    image_size: Optional[int] = None,
    quiet: bool = True,
    correct_units: bool = True,
) -> list[LayerSpec]:
    """Build one of our networks from a command line and describe it.

    This is the whole native entry path::

        specs = specs_from_cfg("--net 'QResNet18(gate=Tower8s)' --method ST -A 2 -W 2")
        energy = evaluate(specs)

    It is CPU-only shape arithmetic -- no data, no checkpoint, no GPU -- and takes a second or
    two, almost all of it importing torch.

    ``image_size=None``, the default, sizes the input so that the first layer produces the feature
    map a padded first layer would produce from :data:`REFERENCE_IMAGE_SIZE` -- 229 for our
    unpadded 7x7 stride-2 stem, 224 for anything that pads.  See :func:`equivalent_input_size` for
    why.  Passing an explicit ``image_size`` forces that size, and ``image_size=224`` is what the
    published numbers were computed at.

    ``correct_units=False`` reproduces the published numbers; see :func:`correct_bits_as_levels`.
    """
    from ..evaluate_networks import FW, add_hooks
    from ...train import o_from_str, setup_o
    from ...arch_imagenet import create_net

    buf = io.StringIO()
    ctx = contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext()
    with ctx:
        o = o_from_str(cfg_string)
        setup_o(o)
        o.num_classes = num_classes
        net = create_net(o)
        net.o = o
        net.to("cpu")
        net.eval()
        add_hooks(net)

        def forward_at(h):
            x = torch.rand(1, 3, h, h)
            with torch.no_grad():
                FW(net, x, method=net.o.m_train)
            return specs_from_net(net, quiet=True, correct_units=correct_units)

        specs = forward_at(image_size or REFERENCE_IMAGE_SIZE)
        if image_size is None and specs:
            need = equivalent_input_size(specs[0], REFERENCE_IMAGE_SIZE)
            if need != REFERENCE_IMAGE_SIZE:
                specs = forward_at(need)        # second pass: the hooks overwrite their shapes
    return specs
