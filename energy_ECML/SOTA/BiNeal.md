# BiNeal

Status: REVIEWED — 2026-09-07, built and checked with the author.

Method: Nie, Xiao, Zhu, Chu, Shen, Li, Yang, Du, Chen, *Binary Neural Networks as a general-purpose
compute paradigm for on-device computer vision*, arXiv:2202.03716. Reported 65.0 % top-1 at `m=1`.

Entry path: **native**. There is no repository to trace — the authors' implementation is not
public — so the architecture was reconstructed from the paper as `arch_imagenet.BiNealNet` and is
walked by [`../adapters/native.py`](../adapters/native.py), the same path as our own networks.

## Which version is evaluated

Our reimplementation, built from the command line the record stores:

```
--net 'BiNealNet(m=1)'   --method ST -A 2 -W 2
--net 'BiNealNet(m=1.5)' --method ST -A 2 -W 2
```

Because the geometry is ours, it is priced exactly as our own rows are: the widths come from the
network's quantizers rather than from a policy, and the unpadded stem is given the input that makes
it equivalent to a padded 224 (see [`../docs/methodology.md`](../docs/methodology.md)).

## Two rows, one geometry

The table carries the configuration twice — `BiNeal-1x` with the **paper's** accuracy and
`BiNeal*-1x (repr.)` with **our reimplementation's**, which reaches 52.5 %. They share a geometry
and therefore share an energy row; only the accuracy differs, and each accuracy is recorded with
its source.

The reconstruction has been audited against the paper, and the audit found choices the paper does
not determine. That is a question about the reimplementation rather than about this package, and it
is recorded with the project's working notes, not here.
