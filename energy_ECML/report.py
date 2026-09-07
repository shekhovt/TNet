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
# The command-line entry point that drives everything: it evaluates every registered entry into a
# record, then generates the results document, the three figures, the LaTeX table, and the
# separate comparison against the published table.

"""Collect the records and build the table, the plots and the LaTeX.

    python -m TNet.energy_ECML.report --build          # evaluate every entry, write energy_ECML/results/
    python -m TNet.energy_ECML.report --plots          # the three figures (SVG for the README, PDF for TeX)
    python -m TNet.energy_ECML.report --markdown       # energy_ECML/results/README.md
    python -m TNet.energy_ECML.report --latex          # energy_ECML/results/latex/*.tex
    python -m TNet.energy_ECML.report --check          # diff every row against the published table
    python -m TNet.energy_ECML.report --compare        # write that diff up as a document
    python -m TNet.energy_ECML.report --all            # all of the above, in that order

Row order, group boundaries and display names come from the registry
(:mod:`TNet.energy_ECML.methods`), not from sorting -- a generated table that sorts itself always gets
that wrong.

The three plotted quantities:

``TEE``   Total Energy Estimate: compute plus all memory movement.
``CEE``   Compute Energy Estimate: the dot products, pooling, affine and requantization.
``MMEE``  Memory Movement Energy Estimate: weight reads plus feature reads and writes
          (one number on the axis, the split kept in the record).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict

from .methods import ENTRIES, Entry
from .model import evaluate
from .records import RESULTS_DIR, load_record, record_from, save_record
from .spec import Technology

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
#: Everything this module generates is written under ``results/``, beside the records it was
#: generated from -- the figures, the results document, the LaTeX table and the comparison.
#: Nothing is written outside the package: where a generated table ends up in someone's paper
#: is that paper's business, and a document generator that reaches into a sibling directory
#: only works in the one checkout it was written in.  Both are overridable per call.
LATEX_DIR = os.path.join(RESULTS_DIR, "latex")
#: The paper-comparison document: an audit of the published table against what the model
#: computes now.  Kept out of ``results/README.md``, which reports the current evaluation only.
COMPARISON_PATH = os.path.join(RESULTS_DIR, "paper-comparison.md")

GROUP_TITLES = OrderedDict([
    ("references", "Full-precision and 8-bit references"),
    ("separable", "Separable-convolution baselines (PikeLPN)"),
    ("binary", "Binary baselines"),
    ("ewgs", "EWGS on an unmodified ResNet-18"),
    ("tnet", "TNet"),
    ("tnet-dilated", "TNet with dilation"),
    ("tnet-width", "TNet, width scaling"),
])

PLOTS = [
    ("TEE", "tee", "Total Energy Estimate [$\\mu$J]", "Accuracy vs total energy"),
    ("CEE", "cee", "Compute Energy Estimate [$\\mu$J]", "Accuracy vs compute energy"),
    ("MMEE", "mmee", "Memory Movement Energy Estimate [$\\mu$J]", "Accuracy vs memory movement"),
]


# ------------------------------------------------------------------ build


def build(entries=ENTRIES, verbose: bool = True) -> list[str]:
    """Evaluate every entry and write its record.  Returns the paths written."""
    paths = []
    for e in entries:
        try:
            specs = e.build()
        except FileNotFoundError as exc:
            # A traced entry needs its clone, and the clones are gitignored.  The record is
            # committed, so the row survives; say plainly that it was not
            # recomputed rather than either crashing or silently passing.
            print(f"  {e.key:24s} -- NOT REBUILT: {exc}")
            continue
        if not specs:
            raise RuntimeError(f"{e.key}: the adapter produced no layers")
        net = evaluate(specs, Technology(), e.assumptions, name=e.cfg_string or e.key)
        rec = record_from(
            net, method=e.method, config=e.key,
            bits_label=f"{e.bits_features}/{e.bits_weights}",
            family=e.family, cite=e.cite,
            accuracy={"value": e.accuracy, "source": e.accuracy_source,
                      "measured_at_bitwidth": e.bits_features},
            adapter=e.adapter, entry=e.key, notes=e.notes,
            **_repo_provenance(e),
        )
        rec["identity"]["display"] = e.display
        rec["identity"]["group"] = e.group
        rec["identity"]["bits_features"] = e.bits_features
        rec["identity"]["bits_weights"] = e.bits_weights
        rec["identity"]["plot"] = e.plot
        rec["published"] = e.published
        p = save_record(rec)
        paths.append(p)
        if verbose:
            print(f"  {e.key:24s} -> {os.path.relpath(p, HERE)}   "
                  f"TEE {net.TEE/1e6:8.0f} uJ  ({len(specs)} layers)")
    return paths


def _repo_provenance(e: Entry) -> dict:
    """``repo_url`` / ``repo_commit`` for a traced entry, from ``energy_ECML/SOTA/sources.toml``."""
    if not e.repo:
        return {}
    from .adapters.traced import read_source
    src = read_source(e.repo)
    return dict(repo_url=src.get("url", ""), repo_commit=src.get("commit", ""))


def _rows() -> list[dict]:
    """Records in registry order, each with its entry attached."""
    out = []
    for e in ENTRIES:
        p = os.path.join(RESULTS_DIR, e.method, f"{e.key}.json")
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} -- run --build first")
        r = load_record(p)
        r["_entry"] = e
        out.append(r)
    return out


def _uJ(rec, key):
    return rec["results"]["total"][key] / 1e6


# ------------------------------------------------------------------ check


def _decimals(x: float) -> int:
    """Decimal places the published cell was printed with, from its shortest repr."""
    t = repr(float(x))
    if "." not in t or t.endswith(".0"):
        return 0
    return len(t.split(".")[1])


def check(fh=sys.stdout) -> int:
    """Diff every row against the published table, column by column.

    **The comparison is made at the precision the table prints.**  ``tab:imagenet1k-detailed``
    gives memory in MB to one or two decimals and energies as integers, so comparing our full
    precision against the printed cell would flag 2.424 against a published "2.4" as a 1 %
    disagreement when it is the same number.  A column counts as reproduced when our value,
    rounded to the published cell's own precision, is that cell.

    Returns the number of columns that do not reproduce.  An unexplained column is a finding,
    not something to tune away -- the standing rule of ``energy_ECML/SOTA/README.md``.
    """
    cols = [
        ("params MB", "MB_weights", "params_MB", 1.0),
        ("params uJ", "MMEE_weights_pJ", "params_uJ", 1e6),
        ("feats MB", "MB_features", "feat_MB", 1.0),
        ("feats uJ", "MMEE_features_pJ", "feat_uJ", 1e6),
        ("compute uJ", "CEE_pJ", "compute_uJ", 1e6),
        ("total uJ", "TEE_pJ", "total_uJ", 1e6),
    ]
    n_bad = 0
    n_cols = 0
    n_rows = 0
    n_rows_exact = 0
    print(f"{'row':34s} {'column':11s} {'published':>10s} {'ours':>10s} {'diff':>9s}", file=fh)
    print("-" * 78, file=fh)
    for rec in _rows():
        e = rec["_entry"]
        if not e.published:
            continue
        n_rows += 1
        t = rec["results"]["total"]
        bad = []
        for label, ours_key, pub_key, scale in cols:
            pub = e.published[pub_key]
            if pub == 0:
                continue
            n_cols += 1
            ours = t[ours_key] / scale
            if round(ours, _decimals(pub)) == pub:
                continue
            bad.append((label, pub, ours, (ours - pub) / pub))
        if not bad:
            n_rows_exact += 1
            continue
        n_bad += len(bad)
        name = f"{e.display} {e.bits_features}/{e.bits_weights}"
        for label, pub, ours, d in bad:
            print(f"{name:34s} {label:11s} {pub:10.4g} {ours:10.4g} {d*100:+8.2f}%", file=fh)
            name = ""
    print("-" * 78, file=fh)
    print(f"{n_rows_exact}/{n_rows} rows reproduce every published cell exactly; "
          f"{n_bad}/{n_cols} columns differ.", file=fh)
    return n_bad


# ------------------------------------------------------------------ plots


def plots(formats=("svg", "pdf")) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plotstyle import FAMILY, apply_axes_style, family_style, place_labels

    os.makedirs(FIG_DIR, exist_ok=True)
    rows = [r for r in _rows()
            if r["_entry"].plot and r["identity"]["accuracy"]["value"] is not None]

    written = []
    for key, stem, xlabel, title in PLOTS:
        fig, ax = plt.subplots(figsize=(6.4, 3.9))
        by_family: dict[str, list] = {}
        for r in rows:
            by_family.setdefault(r["_entry"].family, []).append(r)

        pts, texts = [], []
        for fam in sorted(by_family, key=lambda f: FAMILY.get(f, FAMILY["other"])[5]):
            group = sorted(by_family[fam], key=lambda r: r["results"]["total"][f"{key}_pJ"])
            xs = [r["results"]["total"][f"{key}_pJ"] / 1e6 for r in group]
            ys = [r["identity"]["accuracy"]["value"] for r in group]
            ax.plot(xs, ys, **family_style(fam))
            for r, x, y in zip(group, xs, ys):
                pts.append((x, y))
                texts.append(_point_label(r))

        apply_axes_style(ax, xlabel)
        ax.legend(fontsize=6.5, loc="lower right", framealpha=0.8, edgecolor="none")
        fig.tight_layout(pad=0.6)
        place_labels(ax, pts, texts)
        for ext in formats:
            p = os.path.join(FIG_DIR, f"accuracy-vs-{stem}.{ext}")
            fig.savefig(p, dpi=200)
            written.append(p)
        plt.close(fig)
    return written


def _point_label(rec) -> str:
    """The short label drawn next to a point.

    Long enough to identify the row, short enough that forty of them fit: the bit widths for the
    families where the row *is* a bit setting, the method name otherwise.
    """
    e = rec["_entry"]
    if e.family == "tnet-width":
        return e.display.replace("TNet width $m=", "m=").replace("$", "")
    if e.family == "tnet":
        return f"A{e.bits_features} W{e.bits_weights}"
    if e.family == "tnet-dilated":
        return f"dil A{e.bits_features} W{e.bits_weights}"
    if e.family == "ewgs":
        return f"EWGS W{e.bits_weights}A{e.bits_features.split('/')[0]}"
    return e.display


# ------------------------------------------------------------------ markdown


def _fmt(v, nd=0):
    return f"{v:,.{nd}f}"


def markdown(path: str | None = None) -> str:
    path = path or os.path.join(RESULTS_DIR, "README.md")
    rows = _rows()
    L = []
    A = L.append

    A("# Energy evaluation — all methods, one code path")
    A("")
    A("Status: GENERATED by `python -m TNet.energy_ECML.report --all` — every figure below "
      "comes out of the model, none is transcribed. **Do not edit by hand**; edit the registry "
      "in `energy_ECML/methods/__init__.py` or the cost model in `energy_ECML/model.py` and "
      "regenerate.")
    A("")
    tech = rows[0]["provenance"]["technology"]
    A(f"Generated {rows[0]['provenance']['generated']} from TNet "
      f"`{rows[0]['provenance']['tnet_commit']}`, technology `{tech['name']}` "
      f"(E₁ = {tech['E1_pJ']*1000:.4g} fJ per one-bit add, γ₀ = {tech['gamma_0']:g}, "
      f"{tech['E_mem_on_chip_pJ_per_bit']:.0f} pJ per memory bit).")
    A("")
    A("Every row below is produced by the same cost model "
      "(`energy_ECML/model.py`), from a description of the network as layer geometry plus bit widths. "
      "The model is checked against the original energy script it replaces "
      "(`energy_ECML/evaluate_networks.py`, kept unchanged as the oracle) on 20 "
      "configurations by `energy_ECML/tests/test_reference.py`: every memory count is bit-identical "
      "and every compute total agrees at the 1 pJ resolution the reference prints.")
    A("")

    # ---- plots
    A("## Accuracy versus energy")
    A("")
    for key, stem, xlabel, title in PLOTS:
        rel = os.path.join("figures", f"accuracy-vs-{stem}.svg")
        A(f"### {title}")
        A("")
        A(f"![{title}]({rel})")
        A("")

    # ---- the table
    A("## All rows")
    A("")
    A("`Params` and `Features` are the memory moved per image (weights read once; feature maps "
      "read and written once), with the energy of moving it. `Compute` is CEE, `Total` is TEE.")
    A("")
    A("Every row is priced at the resolution its network runs at: a first layer that pads sees a "
      "224×224 image, and one that does not is given the larger input that makes its output "
      "equivalent — 229×229 for the unpadded 7×7 stride-2 stem of TNet, BiNeal and BOLD. See "
      "`adapters/native.py:equivalent_input_size`.")
    A("")
    header = ("| Method | A bits | W bits | Acc. % | Params MB | Params µJ | Feats MB | "
              "Feats µJ | Compute µJ | **Total µJ** |")
    sep = "|" + "---|" * 10
    last_group = None
    for rec in rows:
        e = rec["_entry"]
        if e.group != last_group:
            A("")
            A(f"### {GROUP_TITLES.get(e.group, e.group)}")
            A("")
            A(header)
            A(sep)
            last_group = e.group
        t = rec["results"]["total"]
        acc = rec["identity"]["accuracy"]["value"]
        A("| {} | {} | {} | {} | {} | {} | {} | {} | {} | **{}** |".format(
            e.display, e.bits_features, e.bits_weights,
            f"{acc:.1f}" if acc is not None else "—",
            _fmt(t["MB_weights"], 2), _fmt(t["MMEE_weights_pJ"] / 1e6),
            _fmt(t["MB_features"], 2), _fmt(t["MMEE_features_pJ"] / 1e6),
            _fmt(t["CEE_pJ"] / 1e6), _fmt(t["TEE_pJ"] / 1e6)))
    A("")

    # ---- provenance
    A("## How each row was produced")
    A("")
    A("| Row | Entry path | Built from | Assumptions |")
    A("|---|---|---|---|")
    for rec in rows:
        e = rec["_entry"]
        if e.cfg_string:
            src = f"`{e.cfg_string}`"
        elif e.adapter == "traced":
            prov = rec.get("provenance", {})
            src = f"[{e.repo} @ {prov.get('repo_commit', '')[:9]}]({prov.get('repo_url', '')})"
        else:
            src = "`adapters.handwritten`"
        A(f"| {e.display} {e.bits_features}/{e.bits_weights} | {e.adapter} | {src} | "
          f"`{rec['provenance']['assumptions']['name']}` |")
    A("")

    # ---- notes
    notes = [(r["_entry"], r["_entry"].notes) for r in rows if r["_entry"].notes]
    if notes:
        A("## Notes on individual rows")
        A("")
        for e, n in notes:
            A(f"- **{e.display} {e.bits_features}/{e.bits_weights}** — {n}")
        A("")

    text = "\n".join(L) + "\n"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    return path


# ------------------------------------------------------------------ comparison


def comparison(path: str | None = None) -> str:
    """Write the paper-comparison document: every row against the cell published for it.

    Kept out of ``results/README.md`` deliberately.  That document reports what the current model
    computes and nothing else; this one is the audit of the published table against it, which is
    a different question with a different audience.  Pass ``path`` to write it elsewhere.
    """
    path = path or COMPARISON_PATH
    rows = _rows()
    L = []
    A = L.append

    A("# The published table, column by column against the current model")
    A("")
    A("Status: DRAFT — generated by `python -m TNet.energy_ECML.report --compare`. "
      "**Do not edit by hand.**")
    A("")
    # Link the results document relative to wherever this one is being written, so the link
    # is correct whether the default path or an explicit one is used.
    readme_link = os.path.relpath(os.path.join(RESULTS_DIR, "README.md"), os.path.dirname(path))
    A("This is the audit of `tab:imagenet1k-detailed` against what the code now computes. It is "
      f"deliberately separate from [`{readme_link}`]({readme_link}), "
      "which reports the current evaluation and does not discuss the paper.")
    A("")
    A("**An unexplained column is a finding, not something to tune away.** The comparison is "
      "made at the precision the table prints: a column counts as reproduced when our value, "
      "rounded to the published cell's own precision, is that cell.")
    A("")

    header = ("| Method | A bits | W bits | Total µJ | published | Δ |")
    A(header)
    A("|" + "---|" * 6)
    for rec in rows:
        e = rec["_entry"]
        pub = e.published.get("total_uJ")
        if not pub:
            continue
        t = rec["results"]["total"]["TEE_pJ"] / 1e6
        A(f"| {e.display} | {e.bits_features} | {e.bits_weights} | {_fmt(t)} | {_fmt(pub)} | "
          f"{(t - pub) / pub * 100:+.1f}% |")
    A("")

    import io
    buf = io.StringIO()
    n_bad = check(buf)
    A("## Every differing column")
    A("")
    A(f"Columns differing from the published cell at its own printed precision: **{n_bad}**.")
    A("")
    A("```")
    A(buf.getvalue().rstrip())
    A("```")
    A("")

    notes = [(r["_entry"], r["_entry"].paper_notes) for r in rows if r["_entry"].paper_notes]
    if notes:
        A("## Row by row")
        A("")
        for e, n in notes:
            A(f"- **{e.display} {e.bits_features}/{e.bits_weights}** — {n}")
        A("")

    text = "\n".join(L) + "\n"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    return path


# ------------------------------------------------------------------ latex


def latex(out_dir: str | None = None) -> list[str]:
    out_dir = out_dir or LATEX_DIR
    os.makedirs(out_dir, exist_ok=True)
    rows = _rows()
    written = []

    lines = ["% Generated by `python -m TNet.energy_ECML.report --latex`.  Do not edit."]
    last_group = None
    for rec in rows:
        e = rec["_entry"]
        if last_group is not None and e.group != last_group:
            lines.append(r"\midrule")
        last_group = e.group
        t = rec["results"]["total"]
        acc = rec["identity"]["accuracy"]["value"]
        cite = f"~\\cite{{{e.cite}}}" if e.cite else ""
        n_params = "—"
        lines.append(
            f"{e.display}{cite} & {e.bits_features} & {e.bits_weights} & "
            f"{acc:.1f} & {t['MB_weights']:.1f} / {t['MMEE_weights_pJ']/1e6:.0f} & "
            f"{t['MB_features']:.1f} / {t['MMEE_features_pJ']/1e6:.0f} & "
            f"{t['CEE_pJ']/1e6:.0f} & {t['TEE_pJ']/1e6:.0f} \\\\"
        )
    p = os.path.join(out_dir, "table-imagenet1k-detailed.tex")
    with open(p, "w") as f:
        f.write("\n".join(lines) + "\n")
    written.append(p)
    return written


# ------------------------------------------------------------------ cli


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", action="store_true", help="evaluate every entry, write records")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--compare", action="store_true",
                    help="write the paper-comparison document (results/paper-comparison.md)")
    ap.add_argument("--latex-dir", metavar="DIR",
                    help="write the LaTeX table here instead of results/latex/ "
                         "(e.g. straight into a paper's generated-input directory)")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args(argv)
    if not any((a.build, a.plots, a.markdown, a.latex, a.check, a.compare, a.all)):
        ap.print_help()
        return 0

    if a.all or a.build:
        print("building records")
        build()
    if a.all or a.plots:
        print("plots:")
        for p in plots():
            print("  " + os.path.relpath(p, HERE))
    if a.all or a.markdown:
        print("markdown: " + os.path.relpath(markdown(), HERE))
    if a.all or a.latex:
        for p in latex(a.latex_dir):
            print("latex: " + p)
    if a.all or a.compare:
        print("comparison: " + comparison())
    if a.check:
        check()                             # a difference is a finding, not a build failure
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
