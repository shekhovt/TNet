# ReActNet

Status: REVIEWED — 2026-09-07, built and checked with the author.

Method: ReActNet, *Towards Precise Binary Neural Network with Generalized Activation Functions*,
Liu, Shen, Savvides & Cheng, ECCV 2020 (arXiv 2003.03488). Two architectures: **ReActNet-A**
(MobileNet geometry, 69.4 %) and **ReActNet-BiReal18** (ResNet-18 geometry, 65.5 %).

Entry path: **traced**. Both rows are produced by constructing the author's models and tracing
them; the method-specific code is [`../methods/reactnet.py`](../methods/reactnet.py). Nothing in
the paper states ReActNet-A's architecture completely enough to write out by hand — the block that
doubles channels by running *two* 1×1 convolutions on one binarized tensor and concatenating is in
the code and not in the paper's figures — so the repository is the only full description of it.

## Which version is evaluated

Pinned in [`sources.toml`](sources.toml):

```
https://github.com/liuzechun/ReActNet @ a6cd08d4605d94135faea6fd73354041e4130b52
```

2021-11-11; the repository has one branch and this is its head. Clone it with

```
bash energy_ECML/SOTA/pull.sh reactnet
```

The models are the **`2_step2` variants** — weights *and* activations binarized — which is what
the published accuracies 69.4 / 65.5 are: `mobilenet/2_step2/reactnet.py` for ReActNet-A and
`resnet/2_step2/birealnet.py` for BiReal18.

The clone is only needed to *rebuild* these rows. The records are committed, so the table, the
plots and the comparison work without it.

## How the model is instrumented

**ReActNet's convolutions are not `nn.Conv2d`.** `HardBinaryConv` holds its weights as a flat
`nn.Parameter` of shape `(C_out·C_in·k·k, 1)`, reshapes them inside `forward`, and calls
`F.conv2d` itself. Forward hooks registered on `nn.Conv2d`/`nn.Linear` alone therefore see **2 of
ReActNet-A's 33 layers** — the real-valued stem and the classifier — and 5 of BiReal18's 21. A
trace taken that way produces a number, and the number is nonsense.

So `methods/reactnet.py` supplies two things to the tracer:

* `extra_compute` — the further module *type* to hook, `HardBinaryConv`;
* `describe` — a `ConvLike` giving that module's weight shape, stride, dilation and groups, since
  it has no `.weight` and no `.groups` to read.

A hooked module that no `describe` can read **raises** rather than being skipped, so this class of
silent undercount cannot recur unnoticed.

**Which layers are binary is decided by module type**, not by position: a layer is binary exactly
when it is a `HardBinaryConv`. The layers the paper calls binary are then precisely the ones the
author's code implemented with the binary convolution class, and the real-valued stem, classifier
and Bi-Real downsample 1×1 convolutions are precisely the remaining `nn.Conv2d`/`nn.Linear`.
Measured on ReActNet-A: 31 binary, 2 not.

Everything else — the 8-bit storage assumption, and the rule under which these models'
residual connections cost no extra read — is general and lives in
[`../docs/methodology.md`](../docs/methodology.md).
