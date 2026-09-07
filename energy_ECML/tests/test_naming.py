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
# Checks the states-versus-bits convention structurally, since it cannot be checked by value:
# every K-named field must have a b-named accessor that agrees with bits_for, and every width
# argument of every public entry point must be named for its unit.

"""The units convention: ``K`` is a number of states, ``b`` is a number of bits.

Author, 2026-09-07: *"Let us name the bit levels as e.g. bA, bW and the number of states as KA,
KW, strictly in the code. ... at least in the code let us have a clearly distinguishable
semantics."*

The convention exists because confusing the two has already cost a published error:
``evaluate_networks.py:fuse_model`` writes ``in_K = 8`` and ``fc.K_W = 8`` meaning *8 bits* into
fields counting *states*, so the input image and some classifier weights were charged three bits
(see :func:`TNet.energy_ECML.adapters.native.correct_bits_as_levels`).

These tests pin the convention structurally rather than by inspecting values.  A width of ``8``
is *legitimate* under either reading -- eight states is a real ``-A 8`` run, eight bits is the
storage assumption -- so no test can tell a wrong ``8`` from a right one by looking at it.  What
can be checked is that the name says which unit is meant, and that the two units are only ever
crossed by :func:`~TNet.energy_ECML.spec.bits_for` / :func:`~TNet.energy_ECML.spec.states_for`.  A new
``K``-named field with no ``b``-named accessor, or an adapter argument named for the wrong unit,
fails here.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest

from ..spec import Assumptions, LayerSpec, Technology, bits_for, states_for
from ..model import count_layer, dot_product_gates, kappa, kappa_from_definition
from ..adapters import handwritten as hw
from . import geometry as geom
from ..adapters import traced
from ..methods import reactnet


# The five width fields of a LayerSpec, and the accessor each must have.
_LAYERSPEC_WIDTHS = [
    ("K_in_stored", "b_in_stored"),
    ("K_in_operand", "b_in_operand"),
    ("K_W_stored", "b_W_stored"),
    ("K_W_operand", "b_W_operand"),
    ("K_out_stored", "b_out_stored"),
]


def test_the_two_units_of_eight_are_not_the_same_number():
    """The whole point, stated once: ``8`` alone is ambiguous and the conversions disagree.

    Eight *states* is three bits; eight *bits* is 256 states.  A name that does not say which
    is meant is a name that can be read either way, which is how the published error happened.
    """
    assert bits_for(8) == 3
    assert states_for(8) == 256
    assert bits_for(states_for(8)) == 8
    assert LayerSpec(name="x", K_in_stored=8).b_in_stored == 3
    assert LayerSpec(name="x", K_in_stored=states_for(8)).b_in_stored == 8


@pytest.mark.parametrize("b", list(range(1, 33)))
def test_states_for_and_bits_for_are_inverse(b):
    assert bits_for(states_for(b)) == b


def test_zero_states_is_zero_bits():
    """``K = 0`` means "nothing is stored" -- the ``K_out_stored = 0`` case."""
    assert bits_for(0) == 0
    assert bits_for(-1) == 0
    assert states_for(0) == 0


def test_a_state_count_need_not_be_a_power_of_two():
    """The training codebase is built on states, and ``-A 3`` is a legitimate run.

    This is why states are the stored primitive and bits the derived accessor, and not the
    other way round: three states round up to two bits, and two bits would round back to four
    states.  The conversion is lossy in exactly that direction.
    """
    assert bits_for(3) == 2 and states_for(2) == 4
    assert bits_for(5) == 3 and bits_for(6) == 3 and bits_for(7) == 3


@pytest.mark.parametrize("K_field,b_field", _LAYERSPEC_WIDTHS)
def test_every_state_field_has_a_bit_accessor_that_agrees(K_field, b_field):
    for K in (0, 2, 3, 8, 16, 256, states_for(32)):
        s = LayerSpec(name="x", **{K_field: K})
        assert getattr(s, b_field) == bits_for(K)


def test_the_five_width_fields_are_exactly_these_five():
    """A new width field must arrive with its accessor and its entry above, not silently."""
    fields = [f for f in LayerSpec.__dataclass_fields__ if f.startswith("K")]
    assert fields == [k for k, _ in _LAYERSPEC_WIDTHS]


def test_no_dataclass_field_is_named_for_the_wrong_unit():
    """Every field of the three description types is ``K``-named or has no width in it.

    ``Assumptions.pooling_K`` is the single exception, and it is deliberate: it is serialized
    into every record's ``provenance.assumptions`` block, so renaming it rewrites every file
    under ``energy_ECML/results/`` without changing a number.  The ``K`` in the name still marks the
    unit.  Any *other* field carrying a width must begin with ``K``.
    """
    known_exception = {"pooling_K"}
    for cls in (Technology, Assumptions, LayerSpec):
        for f in dataclasses.fields(cls):
            if f.name in known_exception:
                continue
            # a field naming a width must declare its unit by its first character
            assert not ("_K" in f.name or f.name.endswith("K")), (
                f"{cls.__name__}.{f.name}: a state count must be named K_...")
            if f.name.startswith("b"):
                assert f.name.startswith("b_") or f.name.startswith("bits"), (
                    f"{cls.__name__}.{f.name}: a b-named field must read as a bit count")


def test_technology_state_constants_have_bit_accessors():
    t = Technology()
    assert t.b_after_conv == bits_for(t.K_after_conv)
    assert t.b_affine_coeffs == bits_for(t.K_affine_coeffs)
    assert (t.b_after_conv, t.b_affine_coeffs) == (8, 8)


# Every public function that takes or returns a width, and the unit each of its width
# parameters is in.  "K" means every width parameter must be K-named; "b" means every one
# must be b-named.
_WIDTH_APIS = [
    (geom.resnet18, "K"),
    (geom.resnet50, "K"),
    (hw.xnor_alexnet, "K"),
    (hw.conv, "K"),
    (traced.uniform_policy, "K"),
    (reactnet.K_policy, "K"),
    (count_layer, "K"),
    (dot_product_gates, "b"),
    (kappa, "b"),
    (kappa_from_definition, "b"),
]

#: Parameters of the functions above that are not widths at all.
_NOT_A_WIDTH = {
    "name", "c_in", "c_out", "k", "stride", "in_res", "out_res", "groups", "pooling",
    "skip_read_bits", "bireal_downsample", "skip_reads", "first_last_names", "spec", "a",
    "n", "lam", "self",
}


@pytest.mark.parametrize("fn,unit", _WIDTH_APIS, ids=lambda v: getattr(v, "__name__", str(v)))
def test_width_parameters_are_named_for_their_unit(fn, unit):
    """No argument of a public entry point can be read as either unit.

    This is the check that would have caught the published error at its boundary: a caller
    writing ``K_in_first=8`` for "8-bit" is writing eight states, and the only defence is that
    the parameter says ``K``, so ``states_for(8)`` is the obvious thing to write instead.
    """
    for p in inspect.signature(fn).parameters.values():
        if p.name in _NOT_A_WIDTH:
            continue
        if unit == "K":
            assert p.name.startswith("K"), f"{fn.__name__}({p.name}): a width must be K-named"
        else:
            assert p.name.startswith("b"), f"{fn.__name__}({p.name}): a width must be b-named"


def test_a_width_policy_returns_state_fields_only():
    """A policy fills ``LayerSpec``'s ``K_`` fields; nothing in it is a bit count."""
    for policy in (traced.uniform_policy(K_A_operand=2, K_W=2), reactnet.K_policy()):
        got = policy(0, "conv1", None)
        assert set(got) == {k for k, _ in _LAYERSPEC_WIDTHS}


def test_the_handwritten_f32_constant_is_states_not_bits():
    """``F32`` is 2**32 states, not the number 32 -- which would be five bits."""
    assert hw.F32 == states_for(32)
    assert bits_for(hw.F32) == 32


def test_the_native_adapters_eight_bit_constant_is_states():
    """``K_8BIT`` is the value the boundary repair writes: 256 states, i.e. 8 bits."""
    from ..adapters.native import K_8BIT
    assert K_8BIT == states_for(8) == 256
    assert bits_for(K_8BIT) == 8
