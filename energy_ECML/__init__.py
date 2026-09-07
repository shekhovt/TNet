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
# The package's public surface: the description types (Technology, Assumptions, LayerSpec), the
# result types, evaluate(), and the record helpers. It pulls in no torch, so the cost model can
# be used without one.

"""Energy evaluation for quantized neural networks.

A network is described as a list of :class:`LayerSpec` -- geometry and quantizer widths -- and
:func:`evaluate` prices it under a :class:`Technology` and a set of :class:`Assumptions`::

    from TNet.energy_ECML import LayerSpec, PoolSpec, Technology, evaluate

    layers = [LayerSpec(name="conv1", in_shape=(1, 3, 224, 224), out_shape=(1, 64, 112, 112),
                        weight_shape=(64, 3, 7, 7), stride=2,
                        K_in_stored=256, K_in_operand=256,
                        K_W_stored=256, K_W_operand=256, K_out_stored=256)]
    e = evaluate(layers)
    print(e.TEE / 1e6, "uJ")

Widths carry their unit in the name: a ``K``-named quantity is a **number of states** (what the
training codebase deals in; not necessarily a power of two) and a ``b``-named one is a **number
of bits** (what the paper prints).  :func:`bits_for` and :func:`states_for` are the only
crossings between them; ``LayerSpec.K_in_stored`` has the read-only companion
``LayerSpec.b_in_stored``, and so on for all five.  See :mod:`TNet.energy_ECML.spec` and
``energy_ECML/README.md``.

The three ways to obtain such a list -- walking one of our own networks, tracing a foreign
repository, or writing the geometry out by hand -- are in :mod:`TNet.energy_ECML.adapters`.  See
``energy_ECML/README.md`` for the worked example and for how to add a method.

This module is the public surface.  ``adapters`` and ``methods`` are imported lazily so that the
cost model itself needs no ``torch``.
"""

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
    states_for,
)
from .model import count, count_layer, dot_product_gates, evaluate, kappa, price
from .records import iter_records, load_record, record_from, save_record

__all__ = [
    # description
    "Technology", "Assumptions", "LayerSpec", "PoolSpec",
    # results
    "Counts", "LayerEnergy", "StageEnergy", "NetworkEnergy",
    # the model
    "evaluate", "count", "count_layer", "price", "dot_product_gates", "kappa", "bits_for", "states_for",
    # records
    "record_from", "save_record", "load_record", "iter_records",
]
