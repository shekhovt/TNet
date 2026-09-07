# Scalable Binary-Quantized Neural Networks for Energy-Efficient Vision

Reference implementation for the paper

> **Scalable Binary-Quantized Neural Networks for Energy-Efficient Vision**
> Alexander Shekhovtsov and Štěpán Obdržálek, Czech Technical University in Prague.
> ECML-PKDD 2026 (research track).

A unified framework for training vision models with **integer weights and
activations** on a fixed quantization grid `{0, 1, …, K-1}` — no adaptive shift or
scale — that scales seamlessly down to **binary** (`K = 2`). Quantized weights are
trained directly rather than as a side effect of preserving real-valued signals.

## Highlights

- **One training scheme for all bit-widths.** The same code trains binary,
  ternary, and higher-integer weights/activations by changing `-W`/`-A`.
- **Elementary block** that removes the ad-hoc overparameterizations (RSign /
  RPReLU and redundant shifts) common in the binary-network literature, replacing
  them with a rationally designed initialization and per-layer learning rates.
- **Tower architecture** — a drop-in replacement for residual blocks that combines
  short and long connecting paths, giving the depth-scalability benefits of
  residual networks *without* propagating high-precision skip connections.
- **Sound gradient estimation.** Built on a stochastic-relaxation framework with a
  family of straight-through and sampling-based estimators (`ST`, `ST-det`, `ZGR`,
  `GS`, `RF`, …) that recover known heuristics as special cases.
- **Refined resource accounting** for compute, memory, and energy, correcting the
  ACE-v2 metric for low-bit operands (see `energy/`).

## Repository layout

| Path | Contents |
|------|----------|
| `train.py` | Training/evaluation entry point (CLI). |
| `methods.py` | Gradient-estimation methods (straight-through, score-function, Gumbel-Softmax, ZGR, …). |
| `layers.py` | Quantized layers, distributions, and quantization primitives. |
| `arch_imagenet.py` | Architectures: `QResNet18`, the **Tower** blocks, baselines (`resnet18`, MobileNet, Bi-Real/BOLD nets). |
| `training.py` | Training loop, schedulers, distillation, logging. |
| `setup_imagenet.py`, `setup_cifar.py`, `setup_mnist.py` | Dataset loaders (ImageNet via FFCV, CIFAR, MNIST). |
| `data_profiler.py` | Activation/gradient logging for analysis. |
| `energy/` | Energy & memory cost models and SOTA re-evaluation. |
| `utilities/` | Op counting, weight loading, visualization, plotting helpers. |
| `experiments/` | Scripts reproducing the paper's figures and tables. |

## Installation

Requires Python 3.11+ and a CUDA-capable GPU.

Clone the repository:

```bash
git clone <repo-url>
cd TNet
```

### Recommended: virtual environment

Create a virtual environment in the **parent** directory (so one venv can be shared
across related sub-projects) and install everything:

```bash
# from inside the repository root
python3 -m venv ../.venv
source ../.venv/bin/activate        # or use direnv with the provided .envrc
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Adjust the `cu128` index tag to match your CUDA version (e.g. `cu126`).
The `.envrc` in the parent directory activates `.venv` automatically when
[direnv](https://direnv.net/) is installed.

`pip install -r requirements.txt` pulls in **everything TNet needs in one step**,
including [`relimport`](https://pypi.org/project/relimport/) (installed from PyPI; no
special access required). If you co-develop `relimport`, install it editable *after*
this step (`pip install -e /path/to/relimport`) so your checkout shadows the released
copy.

### Without a virtual environment (deprecated)

Installing into user packages still works but is not recommended — system Python
paths may shadow newer packages installed via pip:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### Core dependencies

`torch` (≥ 2.7), `torchvision`, `numpy`, `scipy`, `matplotlib`, `pyparsing`,
`transformers`, `Pillow`, `nvitop`, and `relimport`. ImageNet training additionally uses
[FFCV](https://ffcv.io) for fast data loading (install separately); CIFAR/MNIST
do not need it.

## Quick start

`train.py` is the single entry point. Networks and datasets are specified as small
constructor strings, and weight/activation bit-widths are set with `-W`/`-A`
(the number of integer levels: `2` = binary, `4` = 2-bit, etc.).

Train a binary (W1/A1) Tower network on ImageNet-1k:

```bash
python train.py \
  --data 'imagenet(res=512)' --net 'QResNet18(gate=Tower8s)' \
  --method ST-det -n U --MD \
  -W 2 -A 2 \
  --batch_size 256 --epochs 200 --lr 0.005 \
  --distill --compile --cudagraph --fp16
```

Change the bit-width by varying `-W`/`-A` (e.g. `-W 4 -A 4` for 2-bit). Evaluate a
trained checkpoint:

```bash
python train.py --data 'imagenet(res=512)' --net 'QResNet18(gate=Tower8s)' \
  -W 2 -A 2 --method ST-det -n U --load best --test
```

### Smaller datasets

`imagenet-100(res=512)` and `imagenet-10(res=512)` train on the 100- and 10-class
subsets. MNIST and CIFAR-10 are available without any extra data setup; dataset
names are **case-sensitive**:

```bash
# MNIST — downloads automatically on first run
python train.py --data MNIST --net 'LeNetRQ()' \
  -W 2 -A 4 --method ST -n L --epochs 200 --batch_size 256 --MD

# CIFAR-10 — downloads automatically on first run
python train.py --data CIFAR --net 'ResNet18()' \
  -W 2 -A 4 --method ST -n L --epochs 200 --batch_size 256 --MD
```

Profile a network's compute/memory/energy cost (no data required):

```bash
python profile_networks.py
```

### Useful options

| Flag | Meaning |
|------|---------|
| `-W`, `-A` | Quantization levels per weight / activation (`0` = full precision). |
| `-m`, `--method` | Gradient estimator: `ST`, `ST-det`, `ZGR`, `GS(t=…)`, `RF(M=…)`, `ReLU`, `Clamp`, … |
| `-n`, `--noise` | Relaxation noise: `L` logistic, `U` uniform, `T` triangular, `N` Gaussian. |
| `--MD` | Mirror descent for the weights. |
| `--distill` | Train against a teacher's logits. |
| `--compile`, `--cudagraph`, `--fp16` | `torch.compile`, CUDA graphs, mixed precision. |
| `--load`, `--test` | Resume / evaluate (`best`, `last`, `final`). |

Run `python train.py --help` for the full list.

Results, checkpoints, and logs are written under `res/<data>-<net>-…/` (this
directory is git-ignored).

## Reproducing the paper

The `experiments/` directory contains the configurations behind the paper's
figures and tables — ImageNet-1k Pareto curves (`exp_imagenet1k.py`),
architecture/width scaling (`exp_arch_scaling.py`), estimator comparisons
(`exp_estimators.py`), convergence speed (`exp_convergence_speed.py`), and MNIST
ablations (`exp_MNIST.py`). Each file lists the exact `train.py` argument strings
used. Energy/memory re-evaluation of prior work lives in `energy/`.

`exp_arch_scaling.py` and `exp_imagenet1k.py` are the scripts that plot and
summarize the results for the paper's main experiments. Once the relevant
configs listed in them have been trained (their checkpoints and logs are present
under `res/`), running these scripts reads the results back and produces the
corresponding figures and summary tables.

## Citation

```bibtex
@inproceedings{shekhovtsov2026scalable,
  title     = {Scalable Binary-Quantized Neural Networks for Energy-Efficient Vision},
  author    = {Shekhovtsov, Alexander and Obdr{\v{z}}{\'a}lek, {\v{S}}t{\v{e}}p{\'a}n},
  booktitle = {Machine Learning and Knowledge Discovery in Databases (ECML-PKDD)},
  year      = {2026}
}
```

## License

This work is licensed under the
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/)
(CC BY-NC-SA 4.0). See [`LICENSE`](LICENSE) for the full text.
