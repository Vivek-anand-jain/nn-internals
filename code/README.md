# `code/` — the runnable half

Generators define reality; everything else checks itself against them or
extends their arithmetic to real hardware.

| File | Needs | Does |
|---|---|---|
| `ground_truth.py` | stdlib only | **Authoritative.** Computes the 2→3→1 MLP by hand — forward, backward, three optimizers, 12 steps — and emits `assets/data/trace.{json,js}`. Every number on the site comes from here. |
| `transformer_block.py` | stdlib only | **Authoritative for page 09.** One forward pass of a tiny pre-norm transformer block, emitting `assets/data/transformer.{json,js}`. The MLP trace has no transformer in it, so page 09 gets its own generator under the same rules. |
| `mlp_numpy.py` | numpy | The same MLP, vectorised, backward still hand-written. Reproduces the trace and asserts **359 element-wise checks at 1e-12**, then prints a PASS/FAIL table. |
| `mlp_torch.py` | **torch** | The PyTorch equivalent, plus a tour of the real memory structures: `data_ptr`, `element_size`, `nbytes`, `stride`, `is_contiguous`, `.grad` before/after `backward()`, Adam's `exp_avg`/`exp_avg_sq`, forward hooks, storage sharing under transpose, and CUDA allocator deltas. |
| `memory_accounting.py` | stdlib only | CLI calculator for per-GPU training memory of an arbitrary model, broken down by the four tensor classes, with a FITS / DOES NOT FIT verdict. |

## The anti-drift rule

```
ground_truth.py  ──►  assets/data/trace.json  ──►  everything else
```

`ground_truth.py` is the single source of truth. `mlp_numpy.py` and
`mlp_torch.py` **verify against** the trace and never the other way round.

If a check in either fails, the checking script is wrong. Do not "fix" a
failure by editing `ground_truth.py` or by hand-editing `trace.json`. If a
hyperparameter genuinely needs to change, change it in `ground_truth.py`,
re-run it to regenerate the trace, and let every consumer follow.

The same rule applies to the site: no page may type a model number by hand
(see `AGENTS.md`).

## Running them

```sh
# regenerate the trace (do this first, and after any change to the model)
python3 code/ground_truth.py

# verify — numpy only, no torch required
python3 code/mlp_numpy.py

# verify + inspect real memory — requires torch
python3 code/mlp_torch.py

# memory calculator
python3 code/memory_accounting.py --help
```

`mlp_numpy.py` and `mlp_torch.py` exit non-zero on any failed check, so they
work as-is in CI.

### Verification status

- `ground_truth.py` — runs, gradient check passes (max abs error 2.855e-10
  against finite differences).
- `mlp_numpy.py` — **runs and passes.** All 359 checks report a max absolute
  difference of exactly `0.000e+00` against the trace on numpy 2.4.1.
- `mlp_torch.py` — **not executed during authoring**; torch was not installed
  in that environment and is a multi-GB dependency. The file parses
  (`python3 -m py_compile`) and exits with a clear message if torch is
  missing, but its printed results are unverified. Treat `mlp_numpy.py` as the
  trustworthy check of the math and `mlp_torch.py` as the memory tour.
- `memory_accounting.py` — runs; every example below was executed to produce
  its output verbatim.

## `memory_accounting.py`

Formulas are documented in full in the module docstring — parameter count,
the four tensor classes, optimizer slot widths, master weights, tp/pp
sharding, the ZeRO stage table, and the activation coefficient with its
FlashAttention assumption stated. **Site page 10 (`site/10-scaling.html`) must
agree with that docstring**; it is the tiebreaker if the two ever disagree.

```sh
# 70B, Adam, bf16, 8 GPUs, plain DDP
python3 code/memory_accounting.py --params 70B --dtype bf16 --optimizer adam \
    --gpus 8 --strategy ddp --layers 80 --d-model 8192 --batch 1 --seq 4096

# the same model under ZeRO-3 on 64 GPUs, with recomputation
python3 code/memory_accounting.py --params 70B --dtype bf16 --optimizer adam \
    --gpus 64 --strategy zero3 --layers 80 --d-model 8192 --batch 1 --seq 4096 \
    --checkpointing

# every strategy side by side
python3 code/memory_accounting.py --params 70B --gpus 64 --layers 80 \
    --d-model 8192 --batch 1 --seq 4096 --table
```

The first prints **1,128 GiB per GPU → DOES NOT FIT, over by 1,050 GiB**
(that is the canonical 16 bytes per parameter for bf16 Adam, plus 85 GiB of
activations). The second prints **21.30 GiB → FITS with 55 GiB to spare**.
The gap between those two commands is the entire subject of page 10.

Sizes accept suffixes (`70B`, `6.7e9`, `175000000000`). All figures are GiB
(2³⁰ bytes), never GB.

### Presets

`--model` and `--gpu` read `T.reference_configs` out of `trace.json` — the
same published architectures and HBM capacities page 10 puts in its
dropdowns — so the CLI and the page cannot drift apart on what a "70B" is.
Explicit flags always beat a preset.

```sh
python3 code/memory_accounting.py --list-presets
python3 code/memory_accounting.py --model 70b --gpu h100 --gpus 64 \
    --strategy zero3 --checkpointing
```
