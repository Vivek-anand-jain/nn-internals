# nn-internals

What actually happens inside a neural network — weights, gradients, optimizer
state and activations — traced number by number through a thirteen-parameter
model, then scaled until the reason for ZeRO, FSDP, tensor parallelism and
pipeline parallelism is arithmetic rather than jargon.

**[Read it live →](https://vivek-anand-jain.github.io/nn-internals/)**

Or clone and open `index.html` directly. No build step, no dependencies, no
server, no network — every page works from a `file://` URL.

---

## The running example

One example carries the whole project. Small enough that every number fits on
screen; big enough to contain a real matrix multiply, a real nonlinearity and a
real chain rule.

```
x  = [2.0, 3.0]          2000 sqft, 3 bedrooms
y  = 1.0                 sold for $100k

z1 = W1·x + b1  = [0.5, 1.3, -0.8]
a1 = relu(z1)   = [0.5, 1.3,  0.0]      <- hidden unit 2 is DEAD
z2 = W2·a1 + b2 = -0.13
L  = (ŷ − y)²   = 1.2769
```

Thirteen parameters: `W1` (3×2), `b1` (3), `W2` (1×3), `b2` (1).

That negative pre-activation on unit 2 is deliberate. ReLU clamps it to zero, so
its row in `dL/dW1` comes out exactly `[0, 0]` — the clearest demonstration in
the project of how blame flows backward. It recurs on almost every page.

---

## Layout

| Path | What it is |
|---|---|
| `index.html` | Hub — the example, the four tensor classes, the gradient check |
| `site/01-memory.html` | What a weight *is*: bit patterns, dtypes, strides, why `.T` is free |
| `site/02-forward.html` | Every multiply-accumulate, animated one at a time |
| `site/03-loss.html` | Turning a prediction into one scalar; the loss landscape |
| `site/04-backward.html` | The chain rule, and why activations had to be kept |
| `site/05-optimizer.html` | SGD → momentum → Adam, and the fp32 master copy |
| `site/07-loop.html` | Memory rising and falling across one iteration |
| `site/13-transformer-cost.html` | Parameter counting in `d²`, real-model reconciliation, and the seq² term |
| `site/15-scaling.html` | The 70B bill, and an interactive memory calculator |
| `slides/index.html` | The same narrative, linear and animated (arrow keys, `g`, `s`) |
| `docs/deep-dive.md` | Full symbolic derivations, floating point, memory tables |
| `code/` | Runnable Python — see `code/README.md` |
| `assets/data/trace.js` | **Generated.** Every number the site displays |

---

## The rule that keeps this honest

`code/ground_truth.py` derives all thirteen gradients **by hand** — no autograd,
every chain-rule step written out — checks them against central finite
differences, and emits `assets/data/trace.js`.

Every page, slide, animation and table reads from that file. No number is typed
twice, so the visualisations cannot drift away from the mathematics.

```bash
python3 code/ground_truth.py     # regenerate; the site follows automatically
python3 code/mlp_numpy.py        # independent numpy reimplementation, 359 checks
```

The generator also self-checks: it verifies the emitted JSON is *strict* JSON
(Python accepts bare `Infinity`; `JSON.parse` does not), that the memory
accounting is internally consistent, and that the memory timeline never frees a
tensor that was never allocated.

---

## Four tensor classes

Everything in GPU memory during training is one of four things. Every
distributed-training technique is an answer to "which of these do we split up?"

| | Lifetime | Size |
|---|---|---|
| **Weights** | entire run | 1 value per parameter |
| **Gradients** | backward → optimizer step | same shape as the weights |
| **Optimizer state** | entire run | 2 more values per parameter (Adam) |
| **Activations** | forward → backward | scales with batch × sequence |

The colours used for these throughout the site were validated with a
colour-vision-deficiency checker at all-pairs separation in both light and dark
modes. Each class also carries a glyph and a text label — colour is never the
only encoding.

---

## Two findings worth knowing before you read

**The SGD run is a fake success.** At `lr = 0.1` it overshoots on step 2 and
kills every hidden unit. From step 3 the network is a constant predictor: only
`b2` still learns, and the loss falls by exactly `(1 − 2·lr)² = 0.64` per step.
Its tidy-looking curve is a collapsed network. Adam finishes at a *higher* loss
but keeps units alive for all twelve steps. Which optimizer "wins" depends
entirely on what you measure, and the site says so rather than picking a
flattering statistic.

**ZeRO-3 alone does not save a 70B run.** Sharding optimizer state, gradients
and parameters across 64 GPUs drops model state from 1.13 TB to 17.6 GB — but at
8k sequence length the activations are still ~153 GB per GPU. You need tensor
parallelism and checkpointing too. Page 15 lets you dial this yourself.

---

## Caveats

Memory figures are first-order. They omit allocator fragmentation, the CUDA
context, communication buffers and kernel workspace; real runs measure higher.
Every page states its assumptions where it makes them.

All of `code/` has been executed and passes, including `code/mlp_torch.py`
(torch 2.13.0, 30/30 checks). PyTorch autograd agrees with the hand-derived
gradients to exactly 0.0.
