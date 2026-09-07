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
# Builds every unmodified-ResNet row -- the full-precision and 8-bit reference rows, and the
# five EWGS configurations -- by tracing torchvision's ResNet-18 and ResNet-50 rather than
# transcribing their geometry.

"""ResNet-18 and ResNet-50, traced from ``torchvision``.

Every unmodified-ResNet row -- the two full-precision/8-bit references and the five EWGS
configurations -- is produced here, by constructing ``torchvision.models.resnet18`` /
``resnet50`` and tracing one forward pass.  Nothing about the architecture is transcribed: the
shapes, strides, groups and weight shapes all come from the model that runs.

``torchvision`` is the canonical reference implementation of these two networks and is the
definition every paper in the comparison means by "ResNet-18", so tracing it is both less work
and more auditable than writing 21 layers out by hand.  The hand-written geometry that used to
produce these rows has not been thrown away -- it moved to ``tests/geometry.py``, where it is the
independent oracle the tracer is checked against, layer by layer and then by total energy
(``tests/test_adapters.py``).

**What the widths are.**  The 8-bit storage assumption
(``SOTA/README.md``): stored feature maps are capped at 8 bits, convolution operands are charged
at the width the paper states, and the first convolution and the classifier are not quantized.
That is exactly :func:`~TNet.energy_ECML.adapters.traced.uniform_policy`, with ``conv1`` and
``fc`` named as the unquantized pair.

**Skip connections cost nothing extra**, under
:attr:`~TNet.energy_ECML.spec.Assumptions.skip_read_when_not_already_an_input`, which is what the
published rows charge.  The appendix describes a residual read that no published number contains;
its size is measured in ``tests/test_reference.py`` rather than being charged here.
"""

from __future__ import annotations

from ..adapters.traced import specs_from_trace, trace, uniform_policy
from ..spec import LayerSpec, states_for

__all__ = ["F32", "SOURCE", "resnet18", "resnet50"]

#: States of a value stored as f32.  A ``K``-named constant, hence a state count.
F32 = states_for(32)

#: Where the traced models come from.  ``torchvision`` is a package rather than a clone, so the
#: provenance a record stores is its installed version; see ``SOTA/sources.toml``.
SOURCE = "torchvision"

#: The two layers most methods leave unquantized, by their ``torchvision`` module names.
FIRST_LAST = ("conv1", "fc")


def _build(name: str, K_A_operand: int, K_W: int, **kw) -> list[LayerSpec]:
    import torchvision

    model = getattr(torchvision.models, name)(weights=None)
    return specs_from_trace(
        trace(model),
        uniform_policy(K_A_operand, K_W, first_last_names=FIRST_LAST, **kw),
    )


def resnet18(K_A_operand: int, K_W: int, **kw) -> list[LayerSpec]:
    """Traced ``torchvision.models.resnet18``: 21 priced layers.

    All widths are **numbers of states**: binary is ``2``, 8-bit is ``256``, f32 is :data:`F32`.
    Keyword arguments are :func:`~TNet.energy_ECML.adapters.traced.uniform_policy`'s
    (``K_stored``, ``K_in_first``, ``K_first_last``).
    """
    return _build("resnet18", K_A_operand, K_W, **kw)


def resnet50(K_A_operand: int, K_W: int, **kw) -> list[LayerSpec]:
    """Traced ``torchvision.models.resnet50``: 54 priced layers, all ``groups = 1``.

    ResNet-50 has no depthwise convolution and therefore no candidate for the depthwise+pointwise
    fusion the appendix describes.  Widths as in :func:`resnet18`.
    """
    return _build("resnet50", K_A_operand, K_W, **kw)
