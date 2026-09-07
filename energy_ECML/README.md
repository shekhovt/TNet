# `energy_ECML/` — energy evaluation for quantized networks

Status: REVIEWED — 2026-09-07, built and checked with the author.

A network is described as **layer geometry plus quantizer widths**; the model turns that into
**counts** (bits moved, gate operations by width), and a second, trivial step turns counts into
**joules**. Every row of the paper's comparison table and every point on its energy plots is
produced by this one path.

**The results are in [`results/README.md`](results/README.md)** — the full table of every method
and configuration, and the three accuracy-versus-energy plots. It reports what this model computes
under the current methodology and nothing else; it is generated, not written, by

```
python -m TNet.energy_ECML.report --all
```

## Where to read what

| for | read |
|---|---|
| the numbers | [`results/README.md`](results/README.md) — generated |
| what the numbers assume | [`docs/methodology.md`](docs/methodology.md) |
| how to run it, and how to add a method | [`docs/API.md`](docs/API.md) |
| which version of a baseline is evaluated, and how its row is built | `SOTA/<Method>.md` |

A single configuration, from the repository root:

```
python profile_networks.py --net 'BiNealNet(m=1)' --method ST -A 2 -W 2
```

It is CPU arithmetic on shapes — no data, no checkpoint, no GPU — so it runs anywhere, including a
login node, in about a second.

## In one paragraph

Each layer is a **fused convolution**: a convolution, an optional pooling reduction, an affine
transform and a quantization, with only the quantized output written to memory. `Technology` holds
the picojoules and `Assumptions` holds the modelling policy, and they are separate objects so that
a record can be re-priced without being re-run. Widths carry their unit in the name — `K...` is a
number of **states**, `b...` a number of **bits** — and a layer's *stored* width and its *operand*
width are independent fields, which is what lets a binary network with 8-bit feature maps be one
object rather than two runs. `evaluate_networks.py` — a historical name for what is really the
*original* energy script — is kept unchanged as the oracle the tests check against, and its
`fuse_model` is still what walks one of our networks into layers. All of this is stated properly
in [`docs/methodology.md`](docs/methodology.md).

## Tests

```
pytest energy_ECML/tests
```

About half a minute, no data and no GPU. `tests/geometry.py` is hand-written ResNet-18/50 geometry
used as the tracer's oracle; nothing in the package imports it.
