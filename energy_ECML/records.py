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
# Reads and writes the committed JSON records under results/<method>/<config>.json. A record
# stores the counts as well as the energies, plus the provenance needed to say how that row was
# produced, so a change of Technology is a re-multiplication rather than a re-run.

"""The record format: one JSON file per method+configuration.

A record holds **counts**, not only energies, so a different :class:`~TNet.energy_ECML.spec.Technology`
is a re-multiplication rather than a re-run.  Records are committed, which makes a change to the
cost model a reviewable diff across every row.

Layout::

    energy_ECML/results/<method>/<config>.json

Three sections:

``identity``    method, config label, the bit label as a table prints it, family, citation key,
                and the accuracy figures -- each as ``{value, source, measured_at_bitwidth}``
                rather than a bare number, because several published rows carry an f32 accuracy
                against an 8-bit energy.
``provenance``  which adapter and entry produced it, the ``Technology`` and ``Assumptions``, the
                TNet git commit, the date, and for a repo-based entry the URL and pinned commit.
``results``     per-layer, per-stage and total counts and energies.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
from dataclasses import asdict, is_dataclass
from typing import Optional

from .spec import Assumptions, NetworkEnergy, Technology

__all__ = ["RESULTS_DIR", "record_from", "save_record", "load_record", "iter_records"]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _git_commit() -> str:
    """Current TNet commit, or ``"unknown"`` outside a checkout."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.run(
            ["git", "-C", here, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        dirty = subprocess.run(
            ["git", "-C", here, "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        pass
    return "unknown"


def _row(e) -> dict:
    """Counts plus energies for one level of the result tree."""
    d = e.counts.to_dict()
    d.update(
        CEE_pJ=e.CEE,
        CEE_conv_pJ=e.CEE_conv,
        CEE_pool_pJ=e.CEE_pool,
        CEE_affine_pJ=e.CEE_affine,
        CEE_requantize_pJ=e.CEE_requantize,
        MMEE_weights_pJ=e.MMEE_weights,
        MMEE_features_pJ=e.MMEE_features,
        MMEE_pJ=e.MMEE,
        TEE_pJ=e.TEE,
        MB_weights=e.MB_weights,
        MB_features=e.MB_features,
    )
    return d


def record_from(
    net: NetworkEnergy,
    *,
    method: str,
    config: str,
    bits_label: str = "",
    family: str = "",
    cite: str = "",
    accuracy: Optional[dict] = None,
    adapter: str = "",
    entry: str = "",
    repo_url: str = "",
    repo_commit: str = "",
    notes: str = "",
    per_layer: bool = True,
) -> dict:
    """Build the record dictionary for an evaluated network."""
    return {
        "identity": {
            "method": method,
            "config": config,
            "bits_label": bits_label,
            "family": family,
            "cite": cite,
            "accuracy": accuracy or {},
        },
        "provenance": {
            "adapter": adapter,
            "entry": entry,
            "repo_url": repo_url,
            "repo_commit": repo_commit,
            "tnet_commit": _git_commit(),
            "generated": _dt.date.today().isoformat(),
            "technology": asdict(net.technology),
            "assumptions": asdict(net.assumptions),
            "cfg_string": net.name,
            "notes": notes,
        },
        "results": {
            "total": _row(net),
            "max_bits_read_act": net.max_bits_read_act,
            "max_bits_written_act": net.max_bits_written_act,
            "stages": {
                k: dict(_row(v), n_layers=v.n_layers)
                for k, v in net.stages.items()
            },
            "layers": [
                dict(_row(l), name=l.name, stage=l.stage) for l in net.layers
            ] if per_layer else [],
        },
    }


def save_record(rec: dict, path: Optional[str] = None) -> str:
    """Write a record.  Returns the path written."""
    if path is None:
        method = rec["identity"]["method"]
        config = rec["identity"]["config"]
        path = os.path.join(RESULTS_DIR, method, f"{config}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(rec, f, indent=2, sort_keys=False)
        f.write("\n")
    return path


def load_record(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def iter_records(root: Optional[str] = None):
    """Yield ``(path, record)`` for every record under ``root``, in sorted order."""
    root = root or RESULTS_DIR
    for dirpath, _dirnames, filenames in sorted(os.walk(root)):
        for fn in sorted(filenames):
            if fn.endswith(".json"):
                p = os.path.join(dirpath, fn)
                yield p, load_record(p)
