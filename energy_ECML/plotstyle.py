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
# Figure style for the three accuracy-versus-energy plots: one colour and marker per method
# family, the shared axes styling, and a deterministic solver that places the forty-odd point
# labels without overlapping them.

"""One place for the figure style, shared by every plot.

The three figures replace the paper's ``pgfplots`` ones rather than accompanying them, so nothing
here has to match a TikZ rendering pixel for pixel.  What carries over is
the *arrangement*: a log x axis, a grid, one colour and marker per family, an aspect ratio near
0.6, per-point labels in a small font on a translucent white background, and the legend at the
bottom right.

The colours and markers are the paper's, read from the ``pgfplots`` source of
``fig:energy-accuracy``.

**Label placement is solved, not typed.**  The paper sets ``anchor=`` by hand on forty-odd
nodes; :func:`place_labels` is a small deterministic repulsion solver that does it instead.  It
must be deterministic so that the figures do not move between builds.
"""

from __future__ import annotations

import math

__all__ = ["FAMILY", "family_style", "place_labels", "apply_axes_style"]

# name -> (colour, marker, filled, linestyle, label, z-order)
FAMILY = {
    "tnet":         ("#8000c0", "o", True,  "-",  "TNet", 10),
    "tnet-dilated": ("#8000c0", "o", False, "--", "TNet dilated", 9),
    "tnet-width":   ("#e000a0", "o", True,  "--", r"TNet A2 W1, width scaling $m$", 9),
    "bineal":       ("#268026", "D", True,  ":",  "BiNeal", 6),
    "bineal-repr":  ("#268026", "D", False, ":",  "BiNeal* -- our reimplementation", 6),
    "reactnet":     ("#e00000", "s", False, ":",  "ReActNet", 6),
    "bold":         ("#ef8a00", "X", True,  ":",  "BOLD", 6),
    "pikelpn":      ("#808080", "p", True,  ":",  "PikeLPN", 6),
    "xnor":         ("#a05000", "*", True,  ":",  "XNOR-Net", 6),
    "resnet":       ("#0050d0", "^", True,  ":",  "ResNet (8 bit)", 6),
    "ewgs":         ("#00a0a0", "v", True,  ":",  "EWGS", 7),
    "other":        ("#404040", ".", True,  ":",  "other", 5),
}


def family_style(name: str) -> dict:
    colour, marker, filled, linestyle, label, z = FAMILY.get(name, FAMILY["other"])
    return dict(
        color=colour, marker=marker, linestyle=linestyle, label=label, zorder=z,
        markerfacecolor=colour if filled else "white",
        markeredgecolor=colour, markeredgewidth=1.4, markersize=6, linewidth=1.4, alpha=0.95,
    )


def apply_axes_style(ax, xlabel: str, ylabel: str = "Accuracy [%]"):
    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
    ax.tick_params(labelsize=8)
    for s in ax.spines.values():
        s.set_linewidth(0.6)


# ------------------------------------------------------------------ label placement

_CANDIDATES = [
    (0.0, 1.0), (0.0, -1.0), (1.0, 0.0), (-1.0, 0.0),
    (0.85, 0.85), (-0.85, 0.85), (0.85, -0.85), (-0.85, -0.85),
    (0.0, 1.9), (0.0, -1.9), (1.9, 0.0), (-1.9, 0.0),
    (1.5, 1.1), (-1.5, 1.1), (1.5, -1.1), (-1.5, -1.1),
    (0.0, 2.8), (0.0, -2.8), (2.8, 0.0), (-2.8, 0.0),
    (1.1, 2.2), (-1.1, 2.2), (1.1, -2.2), (-1.1, -2.2),
]


def place_labels(ax, points, texts, *, pad_px: float = 9.0, fontsize: float = 6.0):
    """Attach ``texts`` to ``points`` avoiding overlaps, deterministically.

    ``points`` are data coordinates.  Each label is tried at a fixed list of offsets around its
    point; each candidate is scored on how much it overlaps labels already placed, the markers,
    and the axis frame; the lowest-scoring candidate is kept.  Points are processed in a fixed
    order (left to right, then bottom to top), so the result depends only on the data.

    This replaces the ``anchor=`` that the paper's figures set by hand on every node.
    """
    fig = ax.figure
    fig.canvas.draw()                       # transforms must be current to measure text
    renderer = fig.canvas.get_renderer()

    order = sorted(range(len(points)), key=lambda i: (points[i][0], points[i][1]))
    marker_boxes = [_px_box(ax, p, 5.0, 5.0) for p in points]
    placed: list[tuple] = []
    x0, y0, x1, y1 = _axes_px_box(ax)

    for i in order:
        px, py = ax.transData.transform(points[i])
        t = ax.annotate(
            texts[i], points[i], textcoords="offset points", xytext=(0, 0),
            fontsize=fontsize, ha="center", va="center", zorder=20,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
        )
        bb = t.get_window_extent(renderer=renderer)
        w, h = bb.width, bb.height

        best, best_score = None, None
        for dx, dy in _CANDIDATES:
            cx = px + dx * (w / 2 + pad_px)
            cy = py + dy * (h / 2 + pad_px)
            box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            score = 0.0
            for other in placed:
                score += 12.0 * _overlap(box, other)
            for mb in marker_boxes:
                score += 4.0 * _overlap(box, mb)
            # keep the label inside the axes
            score += 6.0 * (max(0.0, x0 - box[0]) + max(0.0, box[2] - x1)
                            + max(0.0, y0 - box[1]) + max(0.0, box[3] - y1))
            score += 0.15 * (abs(dx) + abs(dy))       # prefer the closest placement
            if best_score is None or score < best_score:
                best, best_score = (dx, dy), score
            if score == 0.15 * (abs(dx) + abs(dy)):
                break                                  # a perfect slot; take it
        dx, dy = best
        t.set_position((0, 0))
        t.xyann = (dx * (w / 2 + pad_px), dy * (h / 2 + pad_px))
        cx = px + dx * (w / 2 + pad_px)
        cy = py + dy * (h / 2 + pad_px)
        placed.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))


def _px_box(ax, point, hw, hh):
    px, py = ax.transData.transform(point)
    return (px - hw, py - hh, px + hw, py + hh)


def _axes_px_box(ax):
    bb = ax.get_window_extent()
    return bb.x0, bb.y0, bb.x1, bb.y1


def _overlap(a, b) -> float:
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    if dx <= 0 or dy <= 0:
        return 0.0
    return dx * dy
