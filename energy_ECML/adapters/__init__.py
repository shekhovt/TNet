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
# The three ways to obtain the list of LayerSpec that the cost model prices: native (one of our
# own networks), traced (a model someone else wrote) and handwritten (geometry written out from
# a paper). All three return the same thing and nothing downstream can tell them apart.

"""Three ways to obtain a ``list[LayerSpec]``, in increasing order of trust needed.

``native``       -- one of our own networks, built by ``arch_imagenet.create_net`` and walked
                    with the shape hooks and the fusion logic of ``energy_ECML/evaluate_networks.py``.
``traced``       -- a foreign repository: construct the model, trace it with forward hooks for
                    geometry, and apply a per-method *width policy* (``traced.KPolicy``) for the
                    quantizer widths.
``handwritten``  -- the geometry written out from the paper.  Not a fallback of last resort: for
                    an architecture a paper states completely, it is more auditable than a traced
                    repository, and it is how the published ResNet-18 and ReActNet rows were made.

All three return the same thing, so :func:`TNet.energy_ECML.evaluate` does not know which was used;
the record's ``provenance.adapter`` field says.
"""

from . import handwritten  # noqa: F401

__all__ = ["handwritten", "native", "traced"]


def __getattr__(name):
    # `native` and `traced` need torch and a built network; import them lazily so that the pure
    # model and the hand-written geometry stay importable without them.
    if name in ("native", "traced"):
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(name)
