# Using and extending the package

Status: REVIEWED — 2026-09-07, built and checked with the author.

The assumptions behind the numbers are in [`methodology.md`](methodology.md); this page is how to
drive the code. Everything here is CPU arithmetic on shapes — no data, no checkpoint, no GPU — and
runs anywhere in about a second, almost all of it importing torch.

## The shape of it

One interface: a network is a `list[LayerSpec]`, and `evaluate` prices it.

```python
from TNet.energy_ECML import evaluate, Technology
from TNet.energy_ECML.adapters.native import specs_from_cfg

specs = specs_from_cfg("--net 'QResNet18(gate=Tower8s)' --method ST -A 2 -W 2")
e = evaluate(specs, name="TNet A1-2 W1")

print(f"weights   {e.MB_weights:6.2f} MB  {e.MMEE_weights/1e6:8.0f} uJ")
print(f"features  {e.MB_features:6.2f} MB  {e.MMEE_features/1e6:8.0f} uJ")
print(f"compute                  {e.CEE/1e6:8.0f} uJ")
print(f"total                    {e.TEE/1e6:8.0f} uJ")

for name, stage in e.stages.items():                 # per resolution stage
    print(f"  {name:9s} {stage.n_layers} layers  {stage.TEE/1e6:7.0f} uJ")

for layer in e.layers[:3]:                           # per layer
    print(f"  {layer.name:28s} {layer.TEE/1e6:7.2f} uJ  (conv {layer.CEE_conv/1e6:.3f})")
```

Re-pricing under a different technology does not re-run the counting:

```python
on_chip = Technology(name="L1", E_mem_on_chip_pJ_per_bit=6, E_mem_dram_pJ_per_bit=6)
print(evaluate(specs, on_chip).TEE / 1e6, "uJ")
```

One configuration from the command line, from the repository root:

```
python profile_networks.py --net 'BiNealNet(m=1)' --method ST -A 2 -W 2      # --layers for the breakdown
```

## The three ways to get a `list[LayerSpec]`

In increasing order of trust needed.

1. **Native** (`adapters/native.py`) — our own networks. `specs_from_cfg("--net ...")` builds the
   network, runs one forward pass with shape hooks, and reuses `evaluate_networks.py:fuse_model`
   for the fusion, so there is only one source of truth about which convolution absorbs which
   pooling. It sizes the input itself; pass `image_size=224` to force the published behaviour.

2. **Traced** (`adapters/traced.py`) — a model someone else wrote. `trace()` runs one forward pass
   and records every executed leaf convolution, linear and pooling with its true shapes, in
   execution order; a **width policy** (a `KPolicy`, returning the five state counts) assigns the
   widths. The policy is where the paper's claims live and is the only method-specific code.

   A repository need not use `nn.Conv2d` — ReActNet's convolutions are a class of its own calling
   `F.conv2d` — so an entry may pass `extra_compute` (module types to hook) and `describe` (a
   `ConvLike` giving weight shape, stride, dilation and groups). **A hooked module that nothing can
   describe raises**; it is never silently dropped from the layer list.

   Sources are pinned in `SOTA/sources.toml`: a git repository by commit, cloned by `SOTA/pull.sh`
   into the gitignored `SOTA/repos/`, or an installed package (`kind = "package"`) by the version
   actually imported. Only `--build` needs a clone — the records are committed, so the table, the
   plots and `--check` work without one, and `--build` reports `NOT REBUILT` for a row whose clone
   is missing.

3. **Hand-written** (`adapters/handwritten.py`) — the geometry written out from the paper, for an
   architecture with no runnable implementation to trace. XNOR-Net's binarized AlexNet is the only
   such row. **Everything that can be traced is traced**, because a trace cannot mis-transcribe a
   stride.

They agree where they overlap, and that is what makes the traced rows believable:
`tests/test_adapters.py` traces torchvision's ResNet-18 and checks it against an independent hand
geometry (`tests/geometry.py`, test code — nothing in the package imports it) layer by layer and
then by total energy; `tests/test_reactnet.py` does the same for ReActNet-BiReal18 traced from its
own repository. ReActNet-A has no hand geometry to check against and inherits that credit.

## Adding a method

1. Get a `list[LayerSpec]` by one of the three paths above. For a traced model, put the glue —
   how the model is imported, how its convolution class is read, which layers are quantized — in
   its own module under `methods/`, as `methods/reactnet.py` and `methods/torchvision_resnet.py`
   do.
2. Add an `Entry` to `methods/__init__.py`: display name, the bit labels the table prints
   (`bits_features` / `bits_weights`), accuracy **with its source**, family (decides the plot
   colour and marker), group (decides the table block), the builder, and the published row if
   there is one. Keep what the row *is* in `notes`; anything comparing it with a published cell
   goes in `paper_notes`, which the results document does not print.
3. `python -m TNet.energy_ECML.report --all`.
4. Write `SOTA/<Method>.md` saying which version of the method is evaluated and how the row is
   built.

## The report tool

```
python -m TNet.energy_ECML.report --build      # evaluate every entry, write results/<method>/*.json
                                  --plots      # the three figures (SVG for markdown, PDF for TeX)
                                  --markdown   # results/README.md
                                  --latex      # results/latex/*.tex
                                  --check      # diff every row against the published table
                                  --compare    # write that diff up as a document
                                  --all
```

Row order, group boundaries and display names come from the registry, not from sorting — a
generated table that sorts itself always gets that wrong.

## Records

One JSON file per method and configuration, at `results/<method>/<config>.json`, committed. Three
sections: `identity` (method, labels, accuracy with its source), `provenance` (adapter, entry,
`Technology`, `Assumptions`, the git commit, the date, and the pinned source for a traced entry),
and `results` (per-layer, per-stage and total counts and energies).

They hold **counts**, not only energies, so re-pricing is a re-multiplication. Committing them
makes a change to the cost model a reviewable diff across every row.

## Layout

```
energy_ECML/
  spec.py                 Technology, Assumptions, LayerSpec, the result types.  No torch.
  model.py                counting and pricing.  The whole cost model.
  records.py              the JSON record format
  report.py               the collector
  plotstyle.py            colours, markers, and the deterministic label-placement solver
  adapters/               native.py, traced.py, handwritten.py
  methods/                the registry, plus per-method trace glue
  docs/                   methodology.md, API.md
  results/                the records, the figures, and the generated README.md
  tests/                  pytest energy_ECML/tests -- about half a minute, no data, no GPU
                          geometry.py    hand-written ResNets: the tracer's oracle, test-only
                          test_naming.py pins the K-states / b-bits convention
  evaluate_networks.py    the original energy script, kept UNCHANGED.  The tests' oracle, and
                          the source of fuse_model, which the native path calls.  The name is
                          historical: nothing here calls it to price a row.
  mem_model.py            unchanged; not connected (the tiling-optimal traffic model)
  SOTA/                   the per-method write-ups, sources.toml, pull.sh
```
