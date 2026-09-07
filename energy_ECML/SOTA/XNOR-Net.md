# XNOR-Net

Status: REVIEWED — 2026-09-07, built and checked with the author.

Method: XNOR-Net, *ImageNet Classification Using Binary Convolutional Neural Networks*,
Rastegari, Ordonez, Redmon & Farhadi, ECCV 2016 (arXiv 1603.05279). Architecture evaluated: the
binarized AlexNet, 224×224, 44.2 % top-1 as published.

Entry path: **hand-written**. This is the one row with no runnable implementation here to trace,
so its geometry is written out: [`../adapters/handwritten.py`](../adapters/handwritten.py),
`xnor_alexnet`.

## Which architecture is evaluated

The paper does not print the layer table, and "AlexNet" names several different networks. The one
used is the **two-branch Krizhevsky AlexNet that XNOR-Net's own footnote points at**:

```
https://gist.github.com/szagoruyko/dd032c529048492630fc
```

Per branch:

```
conv 3->48    11x11 s4 p2 | BN ReLU | maxpool 3x3 s2
conv 48->128  5x5 p2      | BN ReLU | maxpool 3x3 s2
conv 128->192 3x3 p1      | BN ReLU
conv 192->192 3x3 p1      | BN ReLU
conv 192->128 3x3 p1      | BN ReLU | maxpool 3x3 s2
```

then, on the concatenation of the two branches (256 × 6 × 6 = 9216), three fully connected layers
9216 → 4096 → 4096 → 1000, described as convolutions. Total 60.5M parameters, which is AlexNet's.

## How the row is built

* **The two branches are written out as two `groups=1` stacks**, not folded into one grouped
  convolution. Under the legacy `separable_fusion="groups_gt_1"` rule a layer with `groups > 1`
  is treated as fused into its successor and loses its activation writes, which would be wrong
  here. Concatenation moves no memory, so two stacks and one grouped stack describe the same
  traffic.
* **First and last layers are not binarized** — the paper says so directly (section 4.1) — so
  `conv1` and the classifier take 8-bit weights and everything between them is binary.
* **Feature maps are stored at 1 bit.** This is not a departure from the 8-bit storage assumption
  of [`../docs/methodology.md`](../docs/methodology.md), which is an upper bound: AlexNet has no
  skip connections, so its binary activations really are what is written to memory, and the network
  simply stores below the bound. The 8-bit figure is what a Bi-Real style network needs, its
  real-valued residual path meaning the stored tensor is wider than the label suggests.
* **`alpha`/`beta` scaling factors are not modelled**, so this row is a lower bound.

The geometry has an oracle: `tests/test_xnor.py` reproduces, in all six printed columns, an
earlier calculation of this same network made with the untouched reference model.
