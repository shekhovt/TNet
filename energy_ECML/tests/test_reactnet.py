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
# Checks the traced ReActNet rows against the hand-written Bi-Real geometry, layer by layer and
# column by column, and pins what the originally published compute cells were and which
# substitution reproduces them.

"""The traced entry path, checked against a geometry that was written out by hand.

The ReActNet repository gives the two published ReActNet rows.  ReActNet-A
could only ever come from there -- nothing states its architecture completely enough to type out
-- so there is no independent description to check it against.  **ReActNet-BiReal18 has one**:
``tests/geometry.py:resnet18(..., bireal_downsample=True)`` is the geometry the published row was made
from, and the trace of ``resnet/2_step2/birealnet.py`` must reproduce it exactly.  That is the
test that makes the traced ReActNet-A row believable, and it is the reason the hand geometry is
kept as the oracle after the row itself moved onto the traced path.

These tests need the clone (``bash energy_ECML/SOTA/pull.sh reactnet``) and are skipped without it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from . import geometry as hw
from ..model import evaluate

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _reactnet():
    from ..methods import reactnet
    try:
        reactnet.load_module(*reactnet._BIREAL_PATH)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    return reactnet


_COLUMNS = ("MB_weights", "MMEE_weights", "MB_features", "MMEE_features", "CEE", "TEE")


def test_traced_bireal18_equals_hand_geometry():
    """All six printed columns, exactly -- not to a tolerance."""
    traced = evaluate(_reactnet().reactnet_bireal18())
    hand = evaluate(hw.resnet18(K_A_operand=2, K_W=2, K_ds_W=256, K_ds_A=256,
                                bireal_downsample=True))
    for col in _COLUMNS:
        assert getattr(traced, col) == pytest.approx(getattr(hand, col), rel=1e-12), col


def test_traced_bireal18_has_the_same_layers():
    """The same layers with the same shapes and widths, not just the same total.

    A total can agree by cancellation; this is what says the trace saw the architecture.

    The comparison is of *multisets*, because the two lists genuinely differ in order and neither
    is wrong: the trace records execution order, so a stage's 1x1 downsample convolution appears
    right after the strided convolution it runs beside, while the hand geometry writes it after
    the block's second 3x3.  Nothing in the cost model depends on order -- layers are priced
    independently and grouped into stages by output resolution -- so the totals agree exactly
    either way, which the test above checks.
    """
    def key(s):
        # repr, because the tuple has to be sortable and `pooling` is None on most layers.
        return repr((s.in_shape, s.out_shape, s.weight_shape, s.stride, s.groups, s.dilation,
                     s.K_in_stored, s.K_in_operand, s.K_W_stored, s.K_W_operand, s.K_out_stored,
                     s.skip_read_bits, s.pooling))
    traced = _reactnet().reactnet_bireal18()
    hand = hw.resnet18(K_A_operand=2, K_W=2, K_ds_W=256, K_ds_A=256, bireal_downsample=True)
    assert len(traced) == len(hand) == 21
    assert sorted(map(key, traced)) == sorted(map(key, hand))
    # The stem and the classifier are at the same position in both, and they are the two layers
    # whose widths are treated specially.
    assert key(traced[0]) == key(hand[0])
    assert key(traced[-1]) == key(hand[-1])


def test_binary_convolutions_are_seen():
    """The repository's convolutions are not ``nn.Conv2d``; a plain trace would miss all of them.

    ReActNet-A is 33 priced layers: the real-valued stem, 31 binary convolutions and the
    classifier.  Before :class:`~TNet.energy_ECML.adapters.traced.ConvLike`, forward hooks on
    ``nn.Conv2d``/``nn.Linear`` saw only the first and the last.
    """
    specs = _reactnet().reactnet_a()
    assert len(specs) == 33
    binary = [s for s in specs if s.K_W_operand == 2]
    assert len(binary) == 31
    assert all(s.K_in_operand == 2 for s in binary)
    # Feature maps are stored at 8 bits even where the operands are binary (the published rows'
    # dagger); the stem reads the 8-bit network input and the classifier is not quantized.
    assert all(s.K_out_stored == 256 for s in specs)
    assert specs[0].K_W_operand == 256 and specs[-1].K_W_operand == 256
    assert specs[-1].kind == "linear"


def test_reactnet_a_matches_the_published_memory_columns():
    """Params and features reproduce the published cells to the digit the table prints.

    The compute column does *not*, and that is a finding, not a failure: see
    :func:`test_the_originally_printed_compute_cells_swapped_one_operand_width`.
    """
    e = evaluate(_reactnet().reactnet_a())
    assert round(e.MMEE_weights / 1e6) == 5496
    assert round(e.MMEE_features / 1e6) == 13101


@pytest.mark.parametrize("build,originally_printed,both8", [
    ("reactnet_a", 89, 407), ("reactnet_bireal18", 43, 154)])
def test_the_originally_printed_compute_cells_swapped_one_operand_width(
        build, originally_printed, both8):
    """What the paper printed before the correction, and what produces it.

    The cells were 89 and 43.  They are reproduced to the printed digit by taking the binary
    convolutions' *activation* operand at the width the feature is **stored** at -- 8 bits --
    while the weight operand stays binary.  That is a single substitution of the stored width for
    the quantized one, which is exactly the distinction the original tracing script recorded as
    ``b_in`` (8) beside ``b_inq`` (1).

    It is **not** a wholesale 8-bit run: pricing *both* operands at 8 bits gives 407 and 154 uJ,
    nowhere near the printed cells.  The dot-product cost is symmetric in the two operand widths,
    so this measurement cannot say by itself which of the two was substituted.  The test below
    settles it: it is the activation operand, because the reference has only one field for it.

    With the binary operands the rows' ``1/8`` label states, the same geometries give 18.6 and
    18.1 uJ, which is what the paper now prints.  First seen on the BiReal18 row; this pins it
    on both.
    """
    specs = getattr(_reactnet(), build)()
    assert round(evaluate(specs).CEE / 1e6) != originally_printed

    swapped = getattr(_reactnet(), build)()
    for s in swapped:
        if s.K_W_operand == 2:                 # binary layers; the stem and fc are already 8-bit
            s.K_in_operand = s.K_in_stored     # the one substitution: stored width as the operand
    assert round(evaluate(swapped).CEE / 1e6) == originally_printed

    both = getattr(_reactnet(), build)()
    for s in both:
        if s.K_W_operand == 2:
            s.K_in_operand = s.K_W_operand = 256
    assert round(evaluate(both).CEE / 1e6) == both8


def test_the_reference_reproduces_the_originally_published_row_end_to_end():
    """Why the printed cells were what they were, in one run of the untouched reference.

    ``evaluate_networks.FusedConv`` has a **single** ``K_in``, and ``net_calc_energy_stats`` uses
    it twice: to derive the dot-product operand width (``evaluate_networks.py:500``) and for the
    activation memory read (line 509).  The author's original tracing notebook recorded ``b_in=8``
    and ``b_inq=1`` as separate fields, correctly; converting that into a ``FusedConv`` had to
    collapse them, and 8 bits is the value the feature-memory column needs.

    So with ``K_in = 256`` and the weights left binary, the reference reproduces the *whole*
    originally published ReactNetA row -- features 13101 uJ and compute 89 uJ -- and nothing about
    it is left unexplained.  That is the structural limitation the ``K_in_stored`` /
    ``K_in_operand`` split removes, not a mistake in an expression.
    """
    from .reference import run_reference

    specs = _reactnet().reactnet_a()
    for s in specs:
        s.K_in_stored = s.K_in_operand = 256       # the one field the reference has
    r = run_reference(specs)
    features_uJ = (r["bits_read_act"] + r["bits_written"]) * 150.0 / 1e6
    assert round(features_uJ) == 13101
    assert round(r["compute_pJ"] / 1e6) == 89
