#!/usr/bin/env python3
"""
sequence_parallel.py — the half of the activation memory tensor parallelism
cannot touch, and what to do about it.

Pure Python standard library.

Page 22 states, correctly, that LayerNorm cannot be sharded across the
hidden dimension: computing a mean needs the whole vector. It then says
"that is what sequence parallelism fixes" and stops. This file finishes the
sentence.

The observation SP is built on:

  Tensor parallelism shards the ATTENTION and MLP regions of a block. It
  leaves the LayerNorm, dropout and residual regions REPLICATED -- every TP
  rank holds a full (seq x d) copy of those activations. So TP=8 divides the
  weights by 8 and the attention/MLP activations by 8, and divides the
  LayerNorm-region activations by 1.

  But those regions are all ELEMENTWISE or per-token. Nothing in them mixes
  tokens. So they can be sharded along the SEQUENCE axis instead, which TP
  has left untouched.

And the part that makes it nearly free:

  A TP region needs the full sequence; an SP region needs only its shard.
  Moving between them is an all-gather on the way in and a reduce-scatter on
  the way out. But TP already paid for an ALL-REDUCE at that boundary, and

      all_reduce == reduce_scatter + all_gather

  so the two halves of the collective TP was already running are exactly the
  two transitions SP needs. The communication VOLUME does not change.

Emits:
    assets/data/seqpar.js    (window.SEQPAR = {...})
    assets/data/seqpar.json

Run:  python3 code/sequence_parallel.py
"""

import json
import math
import os

# ============================================================================
# SHAPES
# ============================================================================
# seq 4 and TP 2 so the sequence shards evenly and every tensor prints.

SEQ = 4
D_MODEL = 4
N_HEADS = 2
D_HEAD = D_MODEL // N_HEADS
D_FF = 8
TP = 2
EPS = 1e-5

TOKENS = ["the", "cat", "sat", "down"]


def det(i, j, salt):
    return round(((((i * 7 + j * 13 + salt * 11 + 3) % 19) - 9) / 30.0), 4)


X = [[round(((t * 5 + d * 3) % 7 + 1) / 8.0, 4) for d in range(D_MODEL)]
     for t in range(SEQ)]
GAIN = [1.0 + det(0, j, 1) for j in range(D_MODEL)]
BIAS = [det(1, j, 1) for j in range(D_MODEL)]
W_IN = [[det(i, j, 2) for j in range(D_FF)] for i in range(D_MODEL)]
W_OUT = [[det(i, j, 3) for j in range(D_MODEL)] for i in range(D_FF)]


def mm(A, B):
    m, k, n = len(A), len(B), len(B[0])
    return [[sum(A[i][t] * B[t][j] for t in range(k)) for j in range(n)]
            for i in range(m)]


def add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def zeros(m, n):
    return [[0.0] * n for _ in range(m)]


def gelu(x):
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + math.tanh(c * (x + 0.044715 * x ** 3)))


def maxdiff(A, B):
    return max(abs(A[i][j] - B[i][j])
               for i in range(len(A)) for j in range(len(A[0])))


def layernorm(row, gain, bias):
    n = len(row)
    mu = sum(row) / n
    var = sum((v - mu) ** 2 for v in row) / n
    inv = 1.0 / math.sqrt(var + EPS)
    return [gain[j] * (row[j] - mu) * inv + bias[j] for j in range(n)]


# ============================================================================
# BASELINE — one GPU
# ============================================================================

def baseline():
    ln = [layernorm(X[t], GAIN, BIAS) for t in range(SEQ)]
    up = mm(ln, W_IN)
    act = [[gelu(v) for v in r] for r in up]
    down = mm(act, W_OUT)
    out = add(X, down)
    return {"ln": ln, "up": up, "act": act, "down": down, "out": out}


# ============================================================================
# TENSOR PARALLEL ONLY — what stays replicated
# ============================================================================

def tp_only(base):
    """
    W_in column-parallel, W_out row-parallel, one all-reduce at the end.
    The LayerNorm and the residual are computed REDUNDANTLY on every rank.
    """
    per = D_FF // TP
    ranks = []
    partials = []
    for r in range(TP):
        # every rank recomputes the SAME LayerNorm over the SAME full sequence
        ln = [layernorm(X[t], GAIN, BIAS) for t in range(SEQ)]
        Wi = [row[r * per:(r + 1) * per] for row in W_IN]
        up = mm(ln, Wi)
        act = [[gelu(v) for v in row] for row in up]
        Wo = W_OUT[r * per:(r + 1) * per]
        partials.append(mm(act, Wo))
        ranks.append({
            "rank": r,
            "resident": [
                {"name": "LN output", "shape": [SEQ, D_MODEL],
                 "elements": SEQ * D_MODEL, "status": "REPLICATED",
                 "why": "every rank computed the identical tensor"},
                {"name": "MLP hidden (sharded)", "shape": [SEQ, per],
                 "elements": SEQ * per, "status": "sharded",
                 "why": "column-parallel W_in gives each rank whole neurons"},
                {"name": "residual stream", "shape": [SEQ, D_MODEL],
                 "elements": SEQ * D_MODEL, "status": "REPLICATED",
                 "why": "the addition is elementwise; every rank does all of it"},
            ],
        })

    summed = zeros(SEQ, D_MODEL)
    for r in range(TP):
        summed = add(summed, partials[r])
    out = add(X, summed)

    replicated = SEQ * D_MODEL * 2          # LN output + residual stream
    sharded = SEQ * (D_FF // TP)
    return {
        "ranks": ranks, "out": out,
        "err_vs_baseline": maxdiff(out, base["out"]),
        "collectives": [
            {"op": "all_reduce", "elements": SEQ * D_MODEL,
             "where": "after the row-parallel W_out",
             "why": "each rank holds a partial sum over its slice of the "
                    "contraction"}],
        "activation_elements_per_rank": replicated + sharded,
        "replicated_elements": replicated,
        "sharded_elements": sharded,
        "note": "TP shards the MLP hidden state but leaves the LayerNorm "
                "output and the residual stream fully replicated. Raising TP "
                "does not shrink those at all.",
    }


# ============================================================================
# TENSOR + SEQUENCE PARALLEL
# ============================================================================

def tp_plus_sp(base):
    """
    The LayerNorm and residual regions are sharded along the SEQUENCE.
    Entering the TP region: all-gather along seq. Leaving it:
    reduce-scatter along seq. Together those are exactly one all-reduce's
    worth of traffic -- the same all-reduce TP was already paying for.
    """
    sper = SEQ // TP
    fper = D_FF // TP
    ranks, sched = [], []

    # ---- SP region: each rank owns seq rows [r*sper, (r+1)*sper) ---------
    ln_shards = []
    for r in range(TP):
        rows = list(range(r * sper, (r + 1) * sper))
        ln_shards.append([layernorm(X[t], GAIN, BIAS) for t in rows])

    sched.append({"step": 1, "op": "none", "elements": 0,
                  "why": "LayerNorm is per-token: rank r normalises only its "
                         "own rows and needs nothing from anyone. This is the "
                         "work TP was duplicating.",
                  "region": "SP"})

    # ---- all-gather along seq, entering the TP region --------------------
    ln_full = []
    for r in range(TP):
        ln_full.extend(ln_shards[r])
    sched.append({"step": 2, "op": "all_gather", "elements": SEQ * D_MODEL,
                  "why": "the TP region needs the full sequence, because the "
                         "matmul contracts over the hidden dimension and every "
                         "rank must see every token.",
                  "region": "SP -> TP"})

    # ---- TP region ------------------------------------------------------
    partials = []
    for r in range(TP):
        Wi = [row[r * fper:(r + 1) * fper] for row in W_IN]
        up = mm(ln_full, Wi)
        act = [[gelu(v) for v in row] for row in up]
        Wo = W_OUT[r * fper:(r + 1) * fper]
        partials.append(mm(act, Wo))

    # ---- reduce-scatter along seq, leaving the TP region -----------------
    summed = zeros(SEQ, D_MODEL)
    for r in range(TP):
        summed = add(summed, partials[r])
    sched.append({"step": 3, "op": "reduce_scatter", "elements": SEQ * D_MODEL,
                  "why": "the partial sums must be added AND the result is "
                         "only needed one shard at a time, so one collective "
                         "does both. This replaces TP's all-reduce.",
                  "region": "TP -> SP"})

    out_shards = []
    for r in range(TP):
        rows = list(range(r * sper, (r + 1) * sper))
        out_shards.append([[X[t][d] + summed[t][d] for d in range(D_MODEL)]
                           for t in rows])
        ranks.append({
            "rank": r, "seq_rows": rows,
            "resident": [
                {"name": "LN output", "shape": [sper, D_MODEL],
                 "elements": sper * D_MODEL, "status": "sharded by SEQ",
                 "why": "each rank normalises only its own tokens"},
                {"name": "MLP hidden (sharded)", "shape": [SEQ, fper],
                 "elements": SEQ * fper, "status": "sharded by HIDDEN",
                 "why": "unchanged from TP"},
                {"name": "residual stream", "shape": [sper, D_MODEL],
                 "elements": sper * D_MODEL, "status": "sharded by SEQ",
                 "why": "the addition is elementwise, so it needs only "
                        "this rank's tokens"},
            ],
        })

    out = []
    for r in range(TP):
        out.extend(out_shards[r])

    sp_elems = 2 * sper * D_MODEL + SEQ * fper
    return {
        "ranks": ranks, "out": out,
        "err_vs_baseline": maxdiff(out, base["out"]),
        "schedule": sched,
        "activation_elements_per_rank": sp_elems,
        "note": "Every activation in the block is now sharded along one axis "
                "or the other. Nothing is replicated.",
    }


# ============================================================================
# THE COMMUNICATION ARGUMENT
# ============================================================================

def comm_argument(tp, sp):
    """
    Ring costs, in elements SENT per rank. The point is that they are equal.
    """
    N = TP
    S = SEQ * D_MODEL

    def ring(op, s):
        f = {"all_reduce": 2.0 * (N - 1) / N,
             "all_gather": (N - 1) / N,
             "reduce_scatter": (N - 1) / N}.get(op, 1.0)
        return round(f * s, 6)

    tp_total = ring("all_reduce", S)
    sp_total = ring("all_gather", S) + ring("reduce_scatter", S)
    return {
        "payload_elements": S,
        "tp_only": [{"op": "all_reduce", "sent_per_rank": ring("all_reduce", S)}],
        "tp_plus_sp": [
            {"op": "all_gather", "sent_per_rank": ring("all_gather", S)},
            {"op": "reduce_scatter", "sent_per_rank": ring("reduce_scatter", S)},
        ],
        "tp_total_sent": tp_total,
        "sp_total_sent": sp_total,
        "identical": abs(tp_total - sp_total) < 1e-12,
        "why": "all_reduce IS reduce_scatter followed by all_gather. Sequence "
               "parallelism does not add a collective -- it splits the one TP "
               "was already running and does useful sharding in the gap "
               "between the halves. Same bytes on the wire, strictly less "
               "memory. That is why it is close to free and why Megatron "
               "turns it on by default alongside TP.",
        "caveat": "Latency is not free: two collectives have two launch and "
                  "sync costs where one had one. On short sequences that can "
                  "outweigh the memory win.",
    }


def at_scale():
    """
    Korthikanti et al. 2022, table 2. Per layer, per token, bytes of
    activation, with tensor parallel degree t:

        TP only     :  10 + 24/t
        TP + SP     :       34/t

    The 10 is exactly the replicated part -- LayerNorm, dropout and the
    residual. SP is what removes it. Numbers here are computed from those
    coefficients, and the coefficients are quoted from the paper.
    """
    rows = []
    for t in (1, 2, 4, 8, 16, 32):
        tp_only = 10 + 24 / t
        tp_sp = 34 / t
        rows.append({
            "tp": t,
            "bytes_per_token_per_layer_tp_only": round(tp_only, 3),
            "bytes_per_token_per_layer_tp_sp": round(tp_sp, 3),
            "saving_pct": round(100 * (1 - tp_sp / tp_only), 1),
        })
    return {
        # Explicit, so a consumer does not have to solve for them from the
        # table (page 20 did exactly that, correctly, and should not have had
        # to). A is the REPLICATED FLOOR: the part TP cannot divide.
        "coefficients": {
            "tp_only_floor_A": 10,
            "tp_only_divisible_B": 24,
            "tp_sp_C": 34,
            "reading": "TP-only = A + B/t. TP+SP = C/t, and C = A + B, which "
                       "is why the two agree exactly at t=1. A is the floor "
                       "SP exists to remove.",
            "floor_includes": "LayerNorm output, dropout mask, and the "
                              "residual stream",
        },
        "_source": "Korthikanti et al. 2022, 'Reducing Activation "
                   "Recomputation in Large Transformer Models', Table 2. "
                   "Coefficients quoted, not derived here.",
        "formula_tp_only": "s*b*h*(10 + 24/t) per layer",
        "formula_tp_sp": "s*b*h*(34/t) per layer",
        "table": rows,
        "toy_caveat": "The simulation in this file has no dropout, so its "
                      "replicated 32 elements are the LayerNorm output and "
                      "the residual stream only. Korthikanti's A=10 also "
                      "counts a dropout mask. The toy therefore understates "
                      "the floor slightly; the argument is unaffected.",
        "reading": "At t=1 the two agree, as they must. As t grows, TP-only "
                   "flattens out at 10 bytes per token per layer no matter "
                   "how many GPUs you add -- that is the replicated floor. "
                   "TP+SP keeps dividing.",
    }


def build():
    base = baseline()
    tp = tp_only(base)
    sp = tp_plus_sp(base)
    return {
        "meta": {
            "generated_by": "code/sequence_parallel.py",
            "seq": SEQ, "d_model": D_MODEL, "d_ff": D_FF, "tp": TP,
            "n_heads": N_HEADS, "tokens": TOKENS,
            "region": "LayerNorm -> MLP(column/row parallel) -> residual",
            "description": "The activation memory TP leaves replicated, and "
                           "how sequence parallelism removes it for free.",
        },
        "input": {"X": X, "gain": GAIN, "bias": BIAS,
                  "W_in": W_IN, "W_out": W_OUT},
        "baseline": base,
        "tp_only": tp,
        "tp_plus_sp": sp,
        "equivalence": {
            "tp_err": tp["err_vs_baseline"],
            "sp_err": sp["err_vs_baseline"],
            "passed": tp["err_vs_baseline"] < 1e-12 and sp["err_vs_baseline"] < 1e-12,
            "claim": "Both reproduce the single-GPU output exactly. Sequence "
                     "parallelism changes where activations live, never what "
                     "is computed.",
        },
        "memory": {
            "tp_only_per_rank": tp["activation_elements_per_rank"],
            "tp_sp_per_rank": sp["activation_elements_per_rank"],
            "saved": tp["activation_elements_per_rank"] - sp["activation_elements_per_rank"],
            "saved_pct": round(100 * (1 - sp["activation_elements_per_rank"] /
                                      tp["activation_elements_per_rank"]), 1),
        },
        "communication": comm_argument(tp, sp),
        "at_scale": at_scale(),
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(os.path.dirname(here), "assets", "data")
    os.makedirs(outdir, exist_ok=True)
    d = build()
    payload = json.dumps(d, indent=1, allow_nan=False)
    open(os.path.join(outdir, "seqpar.json"), "w").write(payload)
    with open(os.path.join(outdir, "seqpar.js"), "w") as f:
        f.write("// GENERATED by code/sequence_parallel.py -- do not hand-edit.\n")
        f.write("window.SEQPAR = " + payload + ";\n")

    m, e, mem, c = d["meta"], d["equivalence"], d["memory"], d["communication"]
    print("=" * 70)
    print("sequence_parallel.py — the activations TP cannot shard")
    print("=" * 70)
    print(f"  seq {m['seq']}  d_model {m['d_model']}  d_ff {m['d_ff']}  TP {m['tp']}")
    print()
    print(f"  TP only      vs 1 GPU : max|Δ| {e['tp_err']:.3e}")
    print(f"  TP + SP      vs 1 GPU : max|Δ| {e['sp_err']:.3e}"
          f"   -> {'PASS' if e['passed'] else 'FAIL'}")
    print()
    print(f"  activation elements per rank")
    print(f"    TP only : {mem['tp_only_per_rank']:3d}"
          f"   ({d['tp_only']['replicated_elements']} of them REPLICATED)")
    print(f"    TP + SP : {mem['tp_sp_per_rank']:3d}"
          f"   -> {mem['saved_pct']}% less")
    print()
    print(f"  sent per rank: TP {c['tp_total_sent']}  vs  TP+SP {c['sp_total_sent']}"
          f"   -> {'IDENTICAL' if c['identical'] else 'DIFFERENT'}")
    print()
    print("  at scale (Korthikanti coefficients, bytes/token/layer):")
    for r in d["at_scale"]["table"]:
        print(f"    t={r['tp']:2d}   TP only {r['bytes_per_token_per_layer_tp_only']:6.2f}"
              f"   TP+SP {r['bytes_per_token_per_layer_tp_sp']:6.2f}"
              f"   ({r['saving_pct']}% less)")
    print()
    print(f"  wrote {os.path.join(outdir, 'seqpar.js')}")
    print("=" * 70)
    if not e["passed"]:
        raise SystemExit("EQUIVALENCE FAILED")


if __name__ == "__main__":
    main()
