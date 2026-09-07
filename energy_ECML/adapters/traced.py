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
# Turns a model someone else wrote into layer descriptions: one forward pass with hooks records
# every executed convolution, linear and pooling with its true shapes, and a per-method width
# policy assigns the quantizer widths. A hooked module that nothing can describe raises rather
# than being silently dropped.

"""Foreign repositories: trace the model for geometry, declare the quantizer widths separately.

The method's repository is cloned to ``energy_ECML/SOTA/repos/<method>/`` (gitignored; ``sources.toml``
records the URL and a pinned commit), the model is constructed, and :func:`trace` runs one forward
pass with hooks, recording every executed leaf convolution, linear and pooling module in order
with its true input and output shapes.  A per-method **width policy** (a :data:`KPolicy`) then
assigns the widths.  The policy is where the paper's claims live, and it is the only
method-specific code.

A policy returns the five ``K_``-named fields of a :class:`~TNet.energy_ECML.spec.LayerSpec`, i.e.
**numbers of states**, never bit counts; a figure a paper quotes in bits is written
``states_for(8)``.  See the naming convention in :mod:`TNet.energy_ECML.spec`.

**What a trace cannot see, and why that is survivable.**  Forward hooks fire on leaf *modules*.
Functional operations do not appear: ``F.relu``, ``x + identity``, ``torch.cat``,
``F.avg_pool2d``.  Under the modelling policy that is mostly fine -- concatenation is free by
assumption and element-wise operations are fused by assumption -- but **a residual add can cost a
read, and a trace cannot see it**.  So the tracer is not the only source of truth about topology:
an entry declares its skip structure alongside its width policy, and the tracer supplies geometry.
That is one more line per method, and it keeps the charged quantity something a reader can check
against the paper rather than something that depends on whether an author wrote ``nn.Identity()``
or a ``+``.

The autograd-graph tracer that would see those operations is deliberately not built here (author:
*"that we won't implement yet"*); every entry made through this module or through
:mod:`~TNet.energy_ECML.adapters.handwritten` becomes one of the references it would need.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn

from ..spec import LayerSpec, PoolSpec
from .native import pool_spec_from_module

__all__ = ["TracedOp", "trace", "KPolicy", "uniform_policy", "specs_from_trace",
           "ConvLike", "describe_native", "repo_dir", "read_source"]

_COMPUTE = (nn.Conv2d, nn.Linear)
_POOL = (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)


def repo_dir(name: str) -> str:
    """Path of the clone ``energy_ECML/SOTA/repos/<name>``, or a message saying how to make it.

    The clones are gitignored, so a fresh checkout has none of them; raising here with the exact
    command is better than an ``ImportError`` from halfway inside a foreign module.
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "SOTA", "repos", name)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"{path} does not exist.  Clone it at its pinned commit with:\n"
            f"    bash energy_ECML/SOTA/pull.sh {name}")
    return path


def read_source(name: str) -> dict:
    """The ``sources.toml`` entry for a clone: its URL and the commit it is evaluated at.

    A record stores these, so a traced row's provenance names an exact tree rather than
    "the ReActNet repository".
    """
    import tomllib
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "SOTA", "sources.toml"), "rb") as f:
        cfg = tomllib.load(f)
    if name not in cfg:
        raise KeyError(f"{name!r} is not listed in energy_ECML/SOTA/sources.toml")
    src = dict(cfg[name])
    if src.get("kind") == "package":
        # An installed package, not a clone: there is no commit to pin in this file, so the
        # version actually imported is read at build time and stored in its place.
        from importlib.metadata import PackageNotFoundError, version
        try:
            src["commit"] = version(src.get("package", name))
        except PackageNotFoundError:
            src["commit"] = "not installed"
    return src


@dataclass(frozen=True)
class ConvLike:
    """The geometry of one convolution-like module, however that module is written.

    A foreign repository does not have to use ``nn.Conv2d``.  ReActNet's ``HardBinaryConv``, for
    instance, holds its weights in a flat ``nn.Parameter`` and calls ``F.conv2d`` itself, so it
    has no ``.weight`` and no ``.groups``; the trace still sees it as a leaf module, and this is
    how a method says what that module computes.  ``weight_shape`` is ``(C_out, C_in / groups,
    k_h, k_w)``, as in :class:`~TNet.energy_ECML.spec.LayerSpec`.
    """

    weight_shape: tuple
    stride: int = 1
    dilation: int = 1
    groups: int = 1
    kind: str = "conv"


def describe_native(module: nn.Module) -> Optional[ConvLike]:
    """``ConvLike`` for a ``nn.Conv2d`` or ``nn.Linear``; ``None`` for anything else."""
    if not isinstance(module, _COMPUTE):
        return None
    w = tuple(int(v) for v in module.weight.shape)
    if len(w) == 2:                                # nn.Linear, priced as a 1x1 convolution
        return ConvLike(weight_shape=(w[0], w[1], 1, 1), kind="linear")
    stride = getattr(module, "stride", 1)
    dil = getattr(module, "dilation", 1)
    return ConvLike(
        weight_shape=w,
        stride=int(stride[0] if isinstance(stride, (tuple, list)) else stride),
        dilation=int(dil[0] if isinstance(dil, (tuple, list)) else dil),
        groups=int(getattr(module, "groups", 1)),
    )


@dataclass
class TracedOp:
    name: str
    module: nn.Module
    in_shape: tuple
    out_shape: tuple


def trace(model: nn.Module, input_shape=(1, 3, 224, 224), device="cpu",
          extra_compute: tuple = ()) -> list[TracedOp]:
    """One forward pass, recording every executed leaf convolution, linear and pooling.

    Order is execution order, not declaration order, which is what matters: a ResNet basic block
    declares ``downsample`` before it runs, and the energy model prices what runs.

    ``extra_compute`` names further module *types* to hook -- a repository's own convolution
    class, which is not an ``nn.Conv2d`` and would otherwise be invisible.  Whatever is hooked
    must be describable: see :class:`ConvLike` and the ``describe`` argument of
    :func:`specs_from_trace`.
    """
    ops: list[TracedOp] = []
    names = {id(m): n for n, m in model.named_modules()}
    handles = []

    def hook(module, args, output):
        x = args[0] if isinstance(args, (tuple, list)) else args
        ops.append(TracedOp(
            name=names.get(id(module), type(module).__name__),
            module=module,
            in_shape=tuple(int(v) for v in x.shape),
            out_shape=tuple(int(v) for v in output.shape),
        ))

    for m in model.modules():
        if isinstance(m, _COMPUTE + _POOL + tuple(extra_compute)):
            handles.append(m.register_forward_hook(hook))
    try:
        model.eval().to(device)
        with torch.no_grad():
            model(torch.rand(*input_shape, device=device))
    finally:
        for h in handles:
            h.remove()
    return ops


# A width policy maps (index, name, module) to the five ``K_``-named width fields of a
# LayerSpec -- numbers of states, not bit counts.  Named ``KPolicy`` for that reason.
KPolicy = Callable[[int, str, nn.Module], dict]


def uniform_policy(
    K_A_operand: int,
    K_W: int,
    *,
    K_stored: int = 256,
    K_in_first: int = 256,
    K_first_last: int = 256,
    first_last_names: Sequence[str] = (),
) -> KPolicy:
    """The 8-bit storage assumption as a policy: 8-bit stored features, stated operand widths,
    unquantized first and last layers.

    Every argument is a number of **states**: ``K_stored=256`` is the 8-bit cap, ``K_A_operand=2``
    is a binary operand.

    ``first_last_names`` names the modules left unquantized.  Naming them, rather than inferring
    "the first and last in the trace", keeps the choice visible in the entry.
    """
    def policy(index: int, name: str, module: nn.Module) -> dict:
        unquantized = name in first_last_names
        return dict(
            K_in_stored=K_in_first if index == 0 else K_stored,
            K_in_operand=K_in_first if index == 0 else (K_stored if unquantized else K_A_operand),
            K_W_stored=K_first_last if unquantized else K_W,
            K_W_operand=K_first_last if unquantized else K_W,
            K_out_stored=K_stored,
        )
    return policy


def specs_from_trace(
    ops: Sequence[TracedOp],
    policy: KPolicy,
    *,
    skip_reads: Optional[dict] = None,
    fused_pairs: Sequence[tuple] = (),
    describe: Optional[Callable[[nn.Module], Optional[ConvLike]]] = None,
) -> list[LayerSpec]:
    """Turn a trace into a layer list.

    A pooling op is *fused into the convolution that produced its input*, matching the appendix:
    *"If the convolution is followed by a max or average pooling reduction prior to quantization,
    the reduction is again assumed to be inline, fused-in."*  A pooling op whose input was not
    produced by the preceding convolution -- a Bi-Real downsample average pool, say -- is dropped
    with its geometry folded into the next convolution's input shape instead, which is what the
    trace already reports.

    ``skip_reads`` maps a layer name to the number of extra activation-read bits charged on it
    (see :attr:`~TNet.energy_ECML.spec.Assumptions.skip_read_when_not_already_an_input`).
    ``fused_pairs`` names ``(first, second)`` layer pairs whose intermediate tensor is never
    stored; the first member gets ``output_fused_into_next``.
    ``describe`` reads the geometry of a module that is not an ``nn.Conv2d``/``nn.Linear``; it
    falls back to :func:`describe_native`, and a module it cannot describe raises rather than
    being silently dropped from the layer list.
    """
    skip_reads = skip_reads or {}
    fused_first = {a for a, _b in fused_pairs}

    specs: list[LayerSpec] = []
    idx = 0
    for op in ops:
        m = op.module
        if isinstance(m, _POOL):
            if specs and tuple(specs[-1].out_shape) == tuple(op.in_shape):
                specs[-1].pooling = pool_spec_from_module(m)
            continue
        cl = describe(m) if describe is not None else None
        if cl is None:
            cl = describe_native(m)
        if cl is None:
            raise TypeError(
                f"traced module {op.name!r} of type {type(m).__name__} is neither a pooling nor "
                "a describable convolution; pass `describe` so its geometry is stated rather "
                "than dropped")
        in_shape = op.in_shape
        out_shape = op.out_shape
        if len(in_shape) == 2:                     # nn.Linear on a flattened tensor
            in_shape = (in_shape[0], in_shape[1], 1, 1)
            out_shape = (out_shape[0], out_shape[1], 1, 1)
        specs.append(LayerSpec(
            name=op.name,
            kind=cl.kind,
            in_shape=in_shape, out_shape=out_shape, weight_shape=tuple(cl.weight_shape),
            groups=cl.groups, stride=cl.stride, dilation=cl.dilation,
            skip_read_bits=int(skip_reads.get(op.name, 0)),
            output_fused_into_next=op.name in fused_first,
            **policy(idx, op.name, m),
        ))
        idx += 1
    return specs
