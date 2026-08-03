# nn-internals

What actually happens inside a neural network and inside a serving stack —
traced number by number, from a thirteen-parameter model up to a 70B one on
sixty-four GPUs, and from a training step to a decoded token.

**[Read it live →](https://vivek-anand-jain.github.io/nn-internals/)**

Or clone and open `index.html`. No build step, no dependencies, no server, no
network — every page works from a `file://` URL.

---

## The rule the whole thing runs on

Every number displayed anywhere — in prose, animations, tables, slides — is
read from a generated data file. The generators derive their results by hand,
with no autograd, and **verify them**. Nothing is asserted where it can be
demonstrated.

```bash
python3 code/ground_truth.py        # 13-param MLP, gradients by hand
python3 code/mlp_numpy.py           # independent numpy check: 359/359
python3 code/mlp_torch.py           # PyTorch autograd agrees at exactly 0.0
```

Fourteen generators, each exiting non-zero if its own proof fails:

| claim | check |
|---|---|
| hand-derived gradients are correct | 2.855e-10 vs finite differences |
| PyTorch agrees | **0.000e+00** — bit-identical, 30/30 |
| DDP computes the same gradient as one GPU | 5.55e-17 |
| tensor parallelism is exact | 6.94e-18 |
| gradient accumulation ≡ a larger batch | 5.55e-17 |
| FlashAttention is **exact, not approximate** | 3.47e-17 |
| sequence parallelism costs the same on the wire | identical, at every world size |
| the KV cache is exact memoisation | 0.000e+00 |
| MoE grouping changes nothing | 0.000e+00 |
| speculative decoding preserves the target distribution | TV 0.0163 vs a 0.0166 noise floor |
| attention is permutation-equivariant | 0.000e+00 — which is why RoPE exists |

---

## The five parts

Each part has one toy model, small enough that every number fits on screen.

### Part I — the mechanics · 01–07 · a 13-parameter MLP

Tensors in memory, forward, loss, backward, optimizer, training dynamics, one
full iteration. Hidden unit 2 gets a negative pre-activation on purpose, so
ReLU kills it and its gradient row is exactly `[0, 0]` — the clearest
demonstration in the project of how blame flows backward.

### Part II — a real transformer · 08–15 · a 288-parameter 2-layer block

Opens with **what it predicts**, because you cannot follow the mechanics until
you know the objective:

```
X  the  cat  sat  on   the      what it sees
Y  cat  sat  on   the  mat      what it must predict
```

The same list, shifted by one — the corpus supervises itself. Then forward,
backward (LayerNorm's two correction terms and softmax's dense Jacobian, both
derived), position, attention variants, cost, FlashAttention, and the 70B
memory wall.

### Part III — many GPUs · 16–23 · a 193-parameter MLP on 4 GPUs

Collectives, data parallel, ZeRO/FSDP, tensor parallel, sequence parallel,
pipeline parallel, partitioning a real block, 3D composition. Every strategy is
*simulated* and proven to compute what one GPU would.

### Part IV — inference · 24–31

A different machine, with its own organising idea: **arithmetic intensity**.

```
H100 ridge = 147.76 FLOP/byte
  prefill  8192.00   compute-bound
  decode      1.00   memory-bound — 0.68% of peak FLOPs
```

Decode reads every weight in the model to produce one token. Prefill vs decode,
the KV cache, batching, PagedAttention, quantisation, speculative decoding,
inference parallelism, serving.

### Part V — mixture of experts · 32–34

Routing, expert parallelism, load balancing, MoE inference. Also the only place
`all-to-all` — taught in Part III — is actually needed.

---

## Findings that argue against the usual story

- **Batching does not always fix decode.** With GQA at 2048 tokens an 8B
  model's intensity *asymptote* is 60.8 FLOP/byte, below the H100 ridge. No
  batch size makes it compute-bound.
- **More tensor parallelism can make latency worse.** TP=8 gives 12.1 ms
  inter-token latency; TP=16 gives 41.2 ms — a 3.41× regression for twice the
  GPUs, at the NVSwitch boundary, where latency is 99.8% of the collective.
- **INT4 makes prefill 18% slower** while making decode 3.79× faster.
- **The SGD run in Part I is a fake success.** At lr=0.1 it kills every hidden
  unit by step 3; the tidy loss curve is a collapsed network decaying at
  exactly `(1−2·lr)² = 0.64`.
- **ZeRO-3's quoted figure is its at-rest one.** A GPU cannot multiply by a
  quarter of a matrix — the peak is `1 + (N−1)/L` times higher.
- **Sinusoidal position survives to 4.4e-16, then collapses to 1.37** through
  `W_q`/`W_k`. That gap is the actual argument for RoPE.

---

## Layout

```
index.html              the hub
site/01..34             34 chapters, five parts
slides/index.html       animated deck (arrow keys, g for grid, s for notes)
docs/deep-dive.md       ~37k words of derivations
code/                   17 files: 14 generators + 3 verifiers
assets/data/            generated — never hand-edited
```

To change a number, edit the generator and re-run it. Never edit a page.

---

## Caveats

Memory and timing figures are first-order models, and every page that uses one
says so. They omit allocator fragmentation, CUDA context, communication buffers
and kernel workspace; real runs measure higher. Where a figure is quoted from a
paper or a vendor rather than derived here, it is labelled as such.

Model and GPU specifications are public figures, quoted not measured, and are
the only externally-sourced numbers in the data.
