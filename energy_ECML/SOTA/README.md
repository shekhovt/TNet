# `energy_ECML/SOTA/` — the methods that are not ours

Status: REVIEWED — 2026-09-07, built and checked with the author.

One file per method, saying **which version of that method is evaluated and how the row is built**
— the pinned repository or package, the architecture, and anything the tracer had to be told in
order to see the model correctly.

* `<Method>.md` — the source and the instrumentation for one method.
* `sources.toml` — every repository and package a traced entry uses, pinned to an exact commit or
  version. `pull.sh` clones the repositories into the gitignored `repos/`.

What the model assumes, and how it is checked, is in
[`../docs/methodology.md`](../docs/methodology.md); how to run and extend it is in
[`../docs/API.md`](../docs/API.md).
