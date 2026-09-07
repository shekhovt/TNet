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
# The registry: one Entry per method and configuration, saying how to build its layer list,
# under which Assumptions, what the row is called, and what the paper published for it.
# Everything method-specific lives here or in a sibling module; none of it is in the cost model.

"""The registry: one entry per method+configuration that we evaluate.

An entry says four things: how to build the layer list (which adapter, with which arguments),
under which :class:`~TNet.energy_ECML.spec.Assumptions`, what the row is called and what the paper
published for it.  Everything method-specific lives here; nothing method-specific is in the cost
model.

The published figures come from the paper's ``tab:imagenet1k-detailed`` and are stored so that :mod:`TNet.energy_ECML.report` can diff every row
column by column (``report.py --check``).  **An unexplained column is a finding, not something
to tune away** -- the standing rule of ``energy_ECML/SOTA/README.md``.

Accuracies are recorded with their source, because several rows carry an f32 accuracy against an
8-bit energy.  Nothing is stored at more than 8 bits here, and the accuracy was not re-measured
under that assumption; see ``energy_ECML/docs/methodology.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from ..spec import Assumptions

__all__ = ["Entry", "ENTRIES", "by_key", "FUSED", "UNFUSED"]

# The reference's rule: any convolution with groups > 1 is treated as fused into its successor.
# This is what the published PikeLPN# rows are.
FUSED = Assumptions(name="paper-2026-optimistic+separable-fused", separable_fusion="groups_gt_1")
# The declared-pairs policy with nothing declared: no convolution fusion at all.
UNFUSED = Assumptions(name="paper-2026-optimistic", separable_fusion="explicit")


@dataclass
class Entry:
    key: str
    """Filename stem of the record, unique."""

    method: str
    """Directory the record goes in, and the method's name in the table."""

    display: str
    """Row label as the table prints it."""

    bits_features: str
    """Activation width as the paper's table prints it, in **bits** -- ``"1/8"``, ``"2"``,
    ``"f32"``.  A label, not an arithmetic quantity: the widths the model is actually given are
    the ``K``-named state counts in each entry's ``build``."""

    bits_weights: str
    """Weight width as the paper's table prints it, in **bits**.  See :attr:`bits_features`."""
    accuracy: Optional[float] = None
    accuracy_source: str = ""
    cite: str = ""
    family: str = "other"
    """Plot family -- decides colour and marker.  See :mod:`TNet.energy_ECML.plotstyle`."""

    group: str = ""
    """Table group; a ``\\midrule`` boundary in LaTeX, a subheading in markdown."""

    adapter: str = ""
    repo: str = ""
    """Name in ``energy_ECML/SOTA/sources.toml``, for a traced entry.  Its URL and pinned commit go
    into the record's provenance."""

    build: Optional[Callable] = None
    assumptions: Assumptions = field(default_factory=lambda: UNFUSED)
    cfg_string: str = ""
    published: dict = field(default_factory=dict)
    """``{params_MB, params_uJ, feat_MB, feat_uJ, compute_uJ, total_uJ}``, or empty."""

    plot: bool = True
    """The ResNet-18 f32 row is table-only *by intent*, not because an axis
    range happens to exclude it."""

    notes: str = ""
    """What the row is: how it was built and what it should be read as.  Printed in the
    generated results document."""

    paper_notes: str = ""
    """How the row compares with the cell ``tab:imagenet1k-detailed`` published for it.  Kept out
    of the generated results document, which reports the current evaluation and nothing else;
    it is collected into the separate comparison document instead (``report.py --compare``)."""


def _hand(fn, **kw):
    from ..adapters import handwritten as hw
    return lambda: getattr(hw, fn)(**kw)


def _traced(module, fn, **kw):
    """Build through a method module of this package, which traces the model it names."""
    def build():
        import importlib
        m = importlib.import_module(f"{__name__}.{module}")
        return getattr(m, fn)(**kw)
    return build


def _native(cfg):
    def build():
        from ..adapters.native import specs_from_cfg
        return specs_from_cfg(cfg)
    return build


def _pub(params_MB, params_uJ, feat_MB, feat_uJ, compute_uJ, total_uJ):
    return dict(params_MB=params_MB, params_uJ=params_uJ, feat_MB=feat_MB,
                feat_uJ=feat_uJ, compute_uJ=compute_uJ, total_uJ=total_uJ)


F32 = 2 ** 32
ENTRIES: list[Entry] = []


def _add(**kw):
    ENTRIES.append(Entry(**kw))


# ---------------------------------------------------------------- full-precision references
_add(
    key="resnet18-f32", method="ResNet18", display="ResNet18", bits_features="f32",
    bits_weights="f32", accuracy=69.8, accuracy_source="torchvision metadata (69.758)",
    cite="He-ResNet", family="resnet", group="references", adapter="traced", repo="torchvision",
    build=_traced("torchvision_resnet", "resnet18", K_A_operand=F32, K_W=F32, K_first_last=F32,
                  K_in_first=F32, K_stored=F32),
    published=_pub(44.6, 56107, 15.4, 19390, 3076, 78543), plot=False,
    notes="Table-only by intent: the row is a full-precision reference, not a plotted method.",
    paper_notes="The published compute cell was not produced by this model -- it is MACs x "
                "1.6956 pJ -- and the dot-product model is outside its stated domain at 32 bits, "
                "so our 1914 uJ is not a better number, only a different one.",
)
_add(
    key="resnet18-8-8", method="ResNet18", display="ResNet18", bits_features="8",
    bits_weights="8", accuracy=69.8, accuracy_source="torchvision metadata (69.758)",
    cite="nie-22", family="resnet", group="references", adapter="traced", repo="torchvision",
    build=_traced("torchvision_resnet", "resnet18", K_A_operand=256, K_W=256),
    published=_pub(11.2, 14027, 3.9, 4848, 140, 19015),
    notes="A plotted coordinate on fig:A-CEE.",
    paper_notes="The published compute cell is MACs x 0.0772 pJ, not this model.",
)
_add(
    key="resnet50-8-8", method="ResNet50", display="ResNet50", bits_features="8",
    bits_weights="8", accuracy=76.1, accuracy_source="torchvision metadata (76.13)",
    cite="", family="resnet", group="references", adapter="traced", repo="torchvision",
    build=_traced("torchvision_resnet", "resnet50", K_A_operand=256, K_W=256),
    published=_pub(24.4, 30670, 20.1, 25291, 317, 56278),
    notes="53 convolutions, all groups=1: no depthwise convolution, so no candidate for the "
          "separable fusion.",
)

# ---------------------------------------------------------------- PikeLPN (MobileNetv1 geometry)
_PIKE = [
    ("1x", "m=1", 256, "8", 67.55, _pub(3.1, 3841, 9.9, 12407, 29, 16277),
     _pub(3.1, 3841, 6.1, 7650, 29, 11521)),
    ("2x", "m=1", 65536, "16", 69.23, _pub(3.1, 3841, 18.8, 23669, 51, 27561),
     _pub(3.1, 3841, 11.3, 14156, 51, 18047)),
    ("3x", "m=1.5", 65536, "16", 71.95, _pub(6.1, 7645, 28.1, 35413, 114, 43172),
     _pub(6.1, 7645, 16.8, 21143, 114, 28901)),
    ("6x", "m=2", 65536, "16", 73.59, _pub(10.1, 12705, 37.5, 47156, 192, 60053),
     _pub(10.1, 12705, 22.4, 28130, 192, 41027)),
]
# Only the fused rows are kept, the fusion rules being settled.  The display name stays
# `PikeLPN#`, which is what
# `tab:imagenet1k-detailed` calls the row whose published cells these are compared against.
for _tag, _m, _KA, _bits, _acc, _pub_plain, _pub_fused in _PIKE:
    _cfg = f"--net 'MobileNetv1({_m})' --method ST-det -n U -A {_KA} -W 256"
    _add(
        key=f"pikelpn-fused-{_tag}", method="PikeLPN", display=f"PikeLPN# {_tag}",
        bits_features=_bits, bits_weights="8/4", accuracy=_acc,
        accuracy_source="Neseem24PikeLPN, as published", cite="Neseem24PikeLPN",
        family="pikelpn", group="separable", adapter="native", build=_native(_cfg),
        assumptions=FUSED, cfg_string=_cfg, published=_pub_fused,
        notes="Depthwise and pointwise convolutions fused into one kernel, so the intermediate "
              "tensor is never stored. Appendix: 'we consider it is possible to fuse the two "
              "convolutions (with a non-linearity inbetween) in a single effective kernel'. The "
              "unfused variant is no longer carried as a row; its published cells are still in "
              "_PIKE should the comparison be wanted again.",
    )

# ---------------------------------------------------------------- binary baselines
_add(
    key="reactnet-bireal18", method="ReActNet", display="ReactNet-BiReal18",
    bits_features="1/8", bits_weights="1", accuracy=65.5, accuracy_source="ReActNet, as published",
    cite="ReActNet", family="reactnet", group="binary", adapter="traced", repo="reactnet",
    build=_traced("reactnet", "reactnet_bireal18"),
    published=_pub(2.0, 2494, 3.6, 4440, 18, 6952),
    notes="Traced from resnet/2_step2/birealnet.py. It reproduces the independent hand-written "
          "geometry (tests/geometry.py) in all six columns exactly, which is what "
          "tests/test_reactnet.py pins.",
    paper_notes="Params exact; features 2.1 % above the published cell. The compute cell is 18, "
                "corrected from the 43 the paper printed. 43.25 is what this geometry gives when "
                "the binary layers' activation operand is taken at the 8-bit width the feature "
                "is stored at while the weight operand stays binary -- a single substituted "
                "field, not an 8-bit run (both operands at 8 bits give 154).",
)
_add(
    key="reactnet-a", method="ReActNet", display="ReactNetA",
    bits_features="1/8", bits_weights="1", accuracy=69.4, accuracy_source="ReActNet, as published",
    cite="ReActNet", family="reactnet", group="binary", adapter="traced", repo="reactnet",
    build=_traced("reactnet", "reactnet_a"),
    published=_pub(4.4, 5496, 10.4, 13101, 19, 18616),
    notes="Traced from mobilenet/2_step2/reactnet.py. The architecture is not stated completely "
          "enough anywhere to write out by hand, so the repository is the only description of it.",
    paper_notes="The model has 29.34M parameters, which is the published row's 29.3M, so the "
                "published row came from this repository. Weights and features reproduce to the "
                "printed digit (5496 uJ and 13101 uJ). The compute cell is 19, corrected from "
                "the 89 the paper printed. 88.95 is what this geometry gives when the binary "
                "layers' activation operand is taken at the 8-bit width the feature is stored at "
                "while the weight operand stays binary -- a single substituted field, not an "
                "8-bit run (both operands at 8 bits give 407).",
)

_add(
    key="xnor-alexnet", method="XNOR-Net", display="XNOR-Net (AlexNet)", bits_features="1",
    bits_weights="1", accuracy=44.2, accuracy_source="rastegari2016xnor, as published",
    cite="rastegari2016xnor", family="xnor", group="binary", adapter="handwritten",
    build=_hand("xnor_alexnet", K_A_operand=2, K_W=2),
    notes="Hand-written geometry: XNOR-Net has no runnable implementation here to trace. It "
          "reproduces the earlier reference-model calculation of this network in all six columns "
          "when given that calculation's 3-bit network input (tests/test_xnor.py); this row uses "
          "the 8-bit network input instead, like every other baseline, which is "
          "+1.7 % on the total. Feature maps are stored at 1 bit: AlexNet has no skip "
          "connection, so its binary activations really are what is written to memory, and the "
          "8-bit storage assumption is an upper bound this network sits under. "
          "alpha/beta scaling factors are not modelled, so this is a lower bound.",
    paper_notes="No published row in tab:imagenet1k-detailed. Against the earlier calculation "
                "and the vsXNOR-Net slide, which print 13661 uJ, this row is 13892 uJ; the "
                "difference is the network input width alone.",
)

_BINEAL = [
    ("1x", "m=1", 65.0, 52.5, _pub(1.6, 1985, 1.2, 1492, 11, 3487)),
    ("1.5x", "m=1.5", 69.7, 59.5, _pub(3.4, 4270, 1.8, 2203, 20, 6494)),
]
for _tag, _m, _acc_pub, _acc_repr, _p in _BINEAL:
    _cfg = f"--net 'BiNealNet({_m})' --method ST -A 2 -W 2"
    _add(
        key=f"bineal-{_tag}", method="BiNeal", display=f"BiNeal-{_tag}",
        bits_features="1/4", bits_weights="1", accuracy=_acc_pub,
        accuracy_source="nie-22, as published (not reproduced by us)", cite="nie-22",
        family="bineal", group="binary", adapter="native", build=_native(_cfg),
        cfg_string=_cfg, published=_p,
    )
    _add(
        key=f"bineal-repr-{_tag}", method="BiNeal", display=f"BiNeal*-{_tag} (repr.)",
        bits_features="1/4", bits_weights="1", accuracy=_acc_repr,
        accuracy_source="our reimplementation", cite="nie-22",
        family="bineal-repr", group="binary", adapter="native", build=_native(_cfg),
        cfg_string=_cfg, published=_p,
        notes="Same geometry and therefore the same energy as the row above; only the accuracy "
              "differs.",
    )

_BOLD = [
    ("base 64", "m=1", 51.8, _pub(2.1, 2614, 1.2, 1492, 13, 4119)),
    ("base 192", "m=3", 65.9, _pub(17.5, 21970, 3.5, 4338, 87, 26394)),
    ("base 256", "m=4", 70.0, _pub(30.8, 38715, 4.6, 5760, 157, 44632)),
]
for _tag, _m, _acc, _p in _BOLD:
    _cfg = f"--net 'BOLDNet({_m})' --method ST -A 2 -W 2"
    _add(
        key=f"bold-{_m}", method="BOLD", display=f"BOLD({_tag})", bits_features="1",
        bits_weights="1", accuracy=_acc, accuracy_source="BOLD, as published", cite="BOLD",
        family="bold", group="binary", adapter="native", build=_native(_cfg),
        cfg_string=_cfg, published=_p,
    )

# ---------------------------------------------------------------- TNet
# The table's labels (`_bf`, `_bw`) are bit counts, as the paper prints them; the command line
# and `_KA`/`_KW` take numbers of states, K = 2**b.  The two are never the same variable.
# `("1-2", "T", 2, 4, 70.1, ...)` -- TNet with ternary weights -- is deliberately not carried
# it is clearly Pareto-suboptimal for the family.
_TNET = [
    ("1-2", "1", 2, 2, 68.9, _pub(2.4, 3050, 1.2, 1515, 26, 4591)),
    ("2", "1", 4, 2, 70.1, _pub(2.4, 3050, 1.6, 1997, 28, 5075)),
    ("3", "1", 8, 2, 71.8, _pub(2.4, 3050, 2.4, 2961, 39, 6050)),
    ("2", "2", 4, 4, 71.9, _pub(3.8, 4844, 1.6, 1997, 45, 6886)),
    ("4", "1", 16, 2, 72.1, _pub(2.4, 3050, 3.1, 3925, 45, 7020)),
    ("3", "3", 8, 8, 73.7, _pub(5.3, 6638, 2.4, 2961, 81, 9680)),
    ("4", "4", 16, 16, 74.2, _pub(6.7, 8432, 3.2, 3925, 120, 12477)),
    ("8", "8", 256, 256, 74.6, _pub(12.4, 15606, 6.2, 7781, 380, 23769)),
]
for _bf, _bw, _KA, _KW, _acc, _p in _TNET:
    _cfg = f"--net 'QResNet18(gate=Tower8s)' --method ST -A {_KA} -W {_KW}"
    _add(
        key=f"tnet-A{_KA}-W{_KW}", method="TNet", display="TNet", bits_features=_bf,
        bits_weights=_bw, accuracy=_acc, accuracy_source="ours, trained", cite="",
        family="tnet", group="tnet", adapter="native", build=_native(_cfg),
        cfg_string=_cfg, published=_p,
    )

_TNET_DIL = [
    ("1-2", "1", 2, 2, 69.7, _pub(2.4, 3050, 2.0, 2509, 66, 5625)),
    ("2", "1", 4, 2, 71.2, _pub(2.4, 3050, 2.4, 2990, 69, 6109)),
    ("3", "1", 8, 2, 72.1, _pub(2.4, 3050, 3.5, 4451, 93, 7594)),
    ("2", "2", 4, 4, 73.5, _pub(3.9, 4844, 2.4, 2990, 108, 7942)),
]
for _bf, _bw, _KA, _KW, _acc, _p in _TNET_DIL:
    _cfg = f"--net 'QResNet18(gate=Tower8s,Dilation=True)' --method ST -A {_KA} -W {_KW}"
    _add(
        key=f"tnet-dil-A{_KA}-W{_KW}", method="TNet", display="TNet with dilation",
        bits_features=_bf, bits_weights=_bw, accuracy=_acc, accuracy_source="ours, trained",
        cite="", family="tnet-dilated", group="tnet-dilated", adapter="native",
        build=_native(_cfg), cfg_string=_cfg, published=_p,
    )

_TNET_W = [
    ("0.75", 66.5, _pub(1.55, 1952, 1.18, 1488, 15, 3455)),
    ("1", 70.1, _pub(2.4, 3050, 1.6, 1961, 28, 5038)),
    ("1.5", 73.7, _pub(4.7, 5919, 2.31, 2907, 54, 8880)),
    ("2", 75.3, _pub(7.7, 9685, 3.06, 3853, 102, 13639)),
]
for _m, _acc, _p in _TNET_W:
    _cfg = f"--net 'QResNet18(gate=Tower8s,m={_m})' --method ST-det -A 4 -W 2"
    _add(
        key=f"tnet-width-m{_m}", method="TNet", display=f"TNet width $m={_m}$",
        bits_features="2", bits_weights="1", accuracy=_acc, accuracy_source="ours, trained",
        cite="", family="tnet-width", group="tnet-width", adapter="native",
        build=_native(_cfg), cfg_string=_cfg, published=_p,
    )

# ---------------------------------------------------------------- EWGS
# Not in the published table.  EWGS quantizes an unmodified ResNet-18; the accuracies are the
# paper's, the geometry is torchvision's ResNet-18, and the storage assumption applies (8-bit
# stored features, published operand widths, unquantized first and last layers).
_EWGS = {(1, 1): 55.3, (1, 2): 64.4, (2, 2): 67.0, (3, 3): 69.7, (4, 4): 70.6}
for (_bw, _ba), _acc in _EWGS.items():
    _add(
        key=f"ewgs-w{_bw}a{_ba}", method="EWGS", display=f"EWGS ResNet18",
        bits_features=f"{_ba}/8", bits_weights=str(_bw), accuracy=_acc,
        accuracy_source="EWGS, as published", cite="Lee21EWGS", family="ewgs",
        group="ewgs", adapter="traced", repo="torchvision",
        build=_traced("torchvision_resnet", "resnet18", K_A_operand=2 ** _ba, K_W=2 ** _bw),
        notes="EWGS quantizes an unmodified ResNet-18, so the geometry is torchvision's and "
              "only the widths are the method's. The three 1x1 downsample convolutions are "
              "quantized like the rest; the sensitivity to leaving them at 8 bits instead is "
              "+2.5 % of the total at binary widths and less above, measured in "
              "tests/test_bitwidths.py.",
        paper_notes="No published row in tab:imagenet1k-detailed.",
    )


def by_key(key: str) -> Entry:
    for e in ENTRIES:
        if e.key == key:
            return e
    raise KeyError(key)
