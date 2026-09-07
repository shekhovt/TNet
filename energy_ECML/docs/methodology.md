# Evaluation methodology — the standing assumptions

Status: REVIEWED — 2026-09-07, built and checked with the author.

This is what the numbers in [`../results/README.md`](../results/README.md) assume. It is the
methodology as it currently stands: corrected relative to the published paper, and not final.
Everything here is a setting of a named object in the code, so a claim on this page can be checked
against a field rather than taken on trust.

## What is counted

A network is a list of **fused convolutions**. Each one is a convolution, an optional pooling
reduction, an affine transform, and a quantization, with **only the quantized output written to
memory**. This is the appendix's *Fused Convolution Assumptions* (the paper's appendix, section
`A:energy`); every assumption it makes is a named field of `Assumptions` carrying the appendix
sentence it implements.

Counting is separated from pricing. The model produces **counts** — bits moved, gate operations by
width — and a second, trivial step multiplies them by a `Technology`. Changing the technology is a
re-multiplication, not a re-run, and a record stores the counts so it can be re-priced later.

That separation was used: **the default memory constant is no longer the paper's.** The paper
charged 150 pJ per bit, described in its appendix as *"a typical value modern CPUs use when
accessing DDR4 memory"*. `Technology()` now charges **13.11 pJ/bit**, the measured HBM total
(control plus datapath) of an NVIDIA A100 in Antepara et al., *Benchmark-driven Models for Energy
Analysis and Attribution of GPU-Accelerated Supercomputing*, SC '25, Table 3. Every committed
record was moved onto it by re-multiplication, with no count and no compute figure touched
(`report --reprice`). The published constants are `PAPER_TECHNOLOGY` and still reproduce the
printed table exactly. What did not change is the *policy* below: every bit is still charged as one
off-chip pass, so the single constant applies to weights and feature maps alike — measured hardware
separates those by about 8× (HBM against L1), and this model does not.

| quantity | what it is |
|---|---|
| `CEE`  | Compute Energy Estimate: dot products, pooling, affine, requantization |
| `MMEE_weights` | weight and affine-coefficient reads |
| `MMEE_features` | feature-map reads and writes |
| `MMEE` | `MMEE_weights + MMEE_features` |
| `TEE`  | `CEE + MMEE` |

The dot product is priced by the appendix's bit-serial conditional-accumulation method, whose cost
is **sub-quadratic in the bit width at low widths**. That is the point of the metric, and it is why
it disagrees with ACE for binary networks.

## Two objects, separated on purpose

* **`Technology`** is physics — picojoules per bit-operation and per memory bit. Nothing else in
  the package holds a picojoule.
* **`Assumptions`** is modelling policy — what may be fused, what must be written out, how a skip
  connection is charged. A record stores the `Assumptions` it was made under, so rows made under
  different policies cannot land in one table by accident.

The policies that are actually in use:

| field | setting | what it means |
|---|---|---|
| `memory_traffic` | `"one_pass"` | each tensor is read once per consumer and written once. The tiling-optimal alternative is named but not implemented |
| `separable_fusion` | `"explicit"` (default) | a depthwise/pointwise pair is fused only when the pair is declared. `"groups_gt_1"` is the legacy rule — *any* `groups > 1` layer is treated as fused into its successor — and is what the PikeLPN# rows and the reference model use |
| `skip_read_when_not_already_an_input` | on | a residual add costs an extra read only when the added tensor is not already an input of the convolution that consumes it. A Bi-Real style skip around one convolution is therefore free; a ResNet skip around a pair is not |
| `pooling_output_count` | `"reference"` | the post-pooling element count divides the spatial size by the stride rather than taking the true output size. The corrected `"floor"` branch exists and is tested |

Concatenation moves no memory. Element-wise operations are fused by assumption.

## Widths: states, not bits

Two quantities are both called a "width" and both get written `8`. Confusing them has already cost
a published error, so the code marks the unit in the **first character of the name**, with no
exceptions:

| name | quantity | example |
|---|---|---|
| `K...` | **number of states** — how many distinct values a quantizer produces | `K_in_operand = 2` is binary; `K_W_stored = 256` is 8-bit |
| `b...` | **number of bits**, `ceil(log2(K))` | `b_in_operand == 1`; `b_W_stored == 8` |

States are the primitive, because that is what the training codebase deals in and a quantizer's `K`
need not be a power of two — `-A 3` is a legitimate run with three activation states, two bits.
Bits are what the literature reports and what every cost formula consumes.

`LayerSpec` *stores* the five state counts and *derives* the five bit widths as read-only
properties, so the two can never disagree. The only sanctioned crossings are `bits_for(K)` and
`states_for(b)`: a figure quoted in bits is written `states_for(8)`, never `8`, which would mean
eight states and three bits.

**Stored width and operand width are independent.** `K_in_stored` is what the feature map costs to
move; `K_in_operand` is what the convolution multiplies. A binary network that keeps 8-bit tensors
in memory is one object with two different fields, not two runs of one field.

## The 8-bit storage assumption

**No tensor and no large weight matrix is stored at more than 8 bits.** Where a paper reports its
network as keeping fp32 activations or fp32 weights, it is priced here as if those were quantized
to 8 bits.

This is an assumption, and it is the one that most affects the memory columns, so it is worth
being explicit about what it rests on. Post-training quantization to 8 bits is routinely reported
as lossless on ImageNet-scale classification, and often works well below 8; the widths that are
hard to reach without retraining are 4 and under. So an 8-bit stored representation is a
**conservative** post-training quantization scheme: it is the setting a deployed network would be
expected to reach without touching training, and assuming it costs no accuracy is a weak
assumption rather than an optimistic one.

It is applied **uniformly**, so no method is advantaged by how its own paper happened to report
storage. What it costs a row depends on where that row started:

* for a method whose paper reports fp32 storage it is a factor of four on every memory column,
  which is the ratio of the widths. The two ResNet-18 reference rows show it: 75,470 µJ of weight
  and feature traffic at 32 bits against 18,878 µJ at 8. (Their totals, 77,385 and 19,032 µJ, are
  not a clean comparison — those rows differ in operand width as well, so their compute differs
  too.)
* for a method already trained below 8 bits it binds only on the layers left unquantized, which
  is usually the first convolution and the classifier.

The direction is worth stating plainly: it **credits every baseline with a compression its authors
did not perform**, so it works against the methods being proposed here, not for them.

### What it applies to

The argument is about quantizing a quantity that is *averaged over many inputs*, where the
rounding error of individual terms does not survive the sum. So the assumption covers:

* **stored feature maps and activations** — `K_in_stored`, `K_out_stored`;
* **the weight matrices of convolution and linear layers** — `K_W_stored`, including the first and
  last layers, which most methods leave unquantized and which are therefore charged 8 bits per
  weight rather than 32;
* **the network input**, an 8-bit image.

**It does not apply to the affine (scale-and-bias) coefficients.** There are two per output
channel, they are not summed over anything, and one rounding error in a channel scale is not
averaged away — so the argument above says nothing about them. Their width is a separate
modelling choice about the datapath, described under [Precision inside a
layer](#precision-inside-a-layer) below, and it is a field of `Technology` rather than of a layer:
no width policy can reach it.

Nor does it constrain the **convolution operands**, which are charged at the width the method
states. A binary network takes binary operands out of stored tensors that are wider than binary,
which is what `K_in_stored` and `K_in_operand` being separate fields expresses, and what the `1/8†`
label and its footnote in the paper's table mean.

This is not bookkeeping. In a Bi-Real style architecture the residual path carries real values
around every binary convolution, so the tensor actually written to memory is *not* binary whatever
the method's label says — 8 bits is the assumption applied to it, exactly as to any other stored
tensor. A binary architecture with no such path, where the binary activations really are what is
written, stores at one bit and is charged for one bit.

### The one row it is not applied to

The **full-precision ResNet-18 reference row is priced uncapped**, at 32 bits stored. It is in the
table to say what an unquantized network costs, so capping it would defeat the row. Every other
row is priced under the assumption.

Note that a network storing *below* 8 bits raises no question: the assumption is an upper bound, so
a binary network whose feature maps really are written to memory one bit wide — XNOR-Net's AlexNet,
which has no skip connections — simply sits under it, and is priced at the width it stores.

## Precision inside a layer

Two widths are properties of the arithmetic datapath rather than of any layer's quantizers, and
are therefore fields of `Technology`, uniform across the network:

* **the accumulator presented to pooling and to the affine transform**, `K_after_conv`, is 8 bits.
  Appendix: *"the reduction is again assumed to be inline, fused-in. The reduction is assumed to be
  computed in 8 bit precision."*
* **the affine scale and bias coefficients**, `K_affine_coeffs`, are 8 bits, read once per output
  channel — `2 * C_out` coefficients per layer — and the affine transform is costed as one 8×8
  multiply and one 16-bit add per output element.

These are assumptions about how the fused kernel is implemented. They are not instances of the
storage assumption above, and changing one is a change of `Technology`, not of a method's widths.

## Input resolution

A row is priced at the resolution its network actually runs at. A first layer that pads sees a
224×224 image. One that does **not** pad is given the larger input that makes its output
equivalent: our stem is a 7×7 stride-2 convolution with `padding=0`, so it is evaluated at
**229×229**, which restores the 112 → 56 → 28 → 14 → 7 ladder. At 224 it would produce 109×109 and
every layer after it would be priced on a tensor the network never computes.

`equivalent_input_size` is the rule. It only ever grows the input, and only up to equivalence.

## What is not modelled

Stated so that a reader does not assume otherwise:

* XNOR-Net's `alpha`/`beta` scaling factors, so that row is a lower bound;
* the tiling-optimal memory-traffic model (`mem_model.py` is the implementation it would use, but
  it is not connected);
* anything a forward hook cannot see. Hooks fire on leaf *modules*, so `F.relu`, `x + identity`,
  `torch.cat` and functional pooling are invisible to a trace. Most are free by assumption, but a
  residual add can cost a read — so a traced entry **declares its skip structure** alongside its
  width policy rather than relying on the trace for topology.

## How this is checked

Two independent things, and it is worth being clear about which does what.

**The unit tests** check the model against itself and against arithmetic: each charged term on a
geometry small enough to verify by hand (`test_terms.py`), the states-versus-bits convention
(`test_naming.py`), the independence of stored and operand widths (`test_bitwidths.py`), and that
the entry paths agree where they overlap — tracing torchvision's ResNet-18 must reproduce an
independent hand-written geometry layer by layer (`test_adapters.py`).

**`evaluate_networks.py` is the oracle.** The name is historical and describes the file badly, so:
it is the *original* energy script, the one that produced the published numbers, and it is kept in
the package **unchanged**. Nothing in it is called to produce a row's energy — the current cost
model does that — but two of its functions matter:

* `fuse_model()` walks a constructed network into the list of fused convolutions this whole
  package is defined on. The native entry path calls it rather than reimplementing the fusion
  rules, so there is exactly one source of truth about which convolution absorbs which pooling.
* `net_calc_energy_stats()` prints energy totals. `tests/test_reference.py` runs it on 20
  configurations, parses what it prints, runs the current model on the same layer lists, and
  compares.

That comparison is what licenses the rewrite:

* every memory count is **bit-identical**;
* every compute total agrees at the **1 pJ** resolution the reference prints — about 1e-8
  relative. The reference prints `int(total)` and nothing finer, so that is the limit of the
  *comparison*, not of the agreement.

The comparison runs under the legacy fusion rule, because that is what the reference implements
and therefore what the already-published numbers contain. **Where the reference does something
surprising, the surprise is reproduced and named rather than fixed silently** — `pooling_output_count`
and `separable_fusion` are both such names, each with the corrected branch beside the legacy one,
and both tested.

The file's *computation* is what must not change; the only edits it has taken are an import fix
without which it could not be imported at all, and the comment header every file in the package
carries. `tests/test_reference.py` would fail if either had touched a number.

A new method's numbers are not reported until its entry reproduces a published row that uses the
same assumptions, column by column, with any column that does not reproduce recorded as a finding
rather than tuned away.
