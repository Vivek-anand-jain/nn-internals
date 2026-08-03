#!/usr/bin/env python3
"""
inference_toy.py — why generating a token is a completely different machine
from training on one.

Pure Python standard library.

Training has one unifying idea in this project: four tensor classes. Inference
has one too, and it is ARITHMETIC INTENSITY -- FLOPs performed per byte moved
from memory. Everything else on the inference pages falls out of it:

    prefill      big GEMMs over the whole prompt      compute-bound
    decode       one token, GEMV not GEMM             BANDWIDTH-bound

A decode step reads every weight in the model to do a single token's worth of
arithmetic. The GPU's FLOPs are almost entirely idle. Once that lands,
batching, quantisation, speculative decoding and paged attention stop looking
like unrelated tricks and start looking like four attacks on one problem.

This file:

  * runs a real 2-layer transformer autoregressively, token by token
  * builds the KV cache and PROVES that decoding with it produces exactly
    the same logits as re-running the whole prefix without it
  * counts FLOPs and bytes moved for both regimes and computes the
    arithmetic intensity of each
  * places both on a roofline against real hardware
  * models batching, and shows where decode stops being bandwidth-bound

Emits:
    assets/data/infer.js     (window.INFER = {...})
    assets/data/infer.json

Run:  python3 code/inference_toy.py
"""

import json
import math
import os

# ============================================================================
# THE MODEL
# ============================================================================
# Same shape family as transformer_2layer.py so a reader moving from Part II
# recognises it, but with a vocabulary and an LM head so it can actually
# generate. Small enough to print the KV cache in full.

D_MODEL = 4
N_HEADS = 2
D_HEAD = D_MODEL // N_HEADS
N_KV_HEADS = 2            # = N_HEADS here, i.e. plain MHA; GQA is its own file
D_FF = 8
N_LAYERS = 2
VOCAB = 6
EPS = 1e-5

PROMPT = [0, 3, 1]        # token ids
N_NEW = 3                 # how many tokens to generate

TOKEN_STR = ["<s>", "the", "cat", "sat", "on", "mat"]


def det(i, j, salt):
    return round(((((i * 7 + j * 13 + salt * 11 + 3) % 19) - 9) / 30.0), 4)


def init():
    P = {"E": [[det(v, d, 0) for d in range(D_MODEL)] for v in range(VOCAB)]}
    for L in range(N_LAYERS):
        s = L * 3
        P[f"L{L}.ln.g"] = [1.0 + det(0, j, s) for j in range(D_MODEL)]
        P[f"L{L}.ln.b"] = [det(1, j, s) for j in range(D_MODEL)]
        for nm, salt in (("Wq", 1), ("Wk", 2), ("Wv", 3), ("Wo", 4)):
            P[f"L{L}.{nm}"] = [[det(i, j, s + salt) for j in range(D_MODEL)]
                               for i in range(D_MODEL)]
        P[f"L{L}.ln2.g"] = [1.0 + det(2, j, s) for j in range(D_MODEL)]
        P[f"L{L}.ln2.b"] = [det(3, j, s) for j in range(D_MODEL)]
        P[f"L{L}.W1"] = [[det(i, j, s + 5) for j in range(D_FF)]
                         for i in range(D_MODEL)]
        P[f"L{L}.W2"] = [[det(i, j, s + 6) for j in range(D_MODEL)]
                         for i in range(D_FF)]
    # LM head, untied from the embedding so the two are visibly separate
    P["head"] = [[det(i, v, 9) for v in range(VOCAB)] for i in range(D_MODEL)]
    return P


# ------------------------------------------------------------------ helpers

def mm(A, B):
    m, k, n = len(A), len(B), len(B[0])
    return [[sum(A[i][t] * B[t][j] for t in range(k)) for j in range(n)]
            for i in range(m)]


def dot(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))


def gelu(x):
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + math.tanh(c * (x + 0.044715 * x ** 3)))


def layernorm(v, g, b):
    n = len(v)
    mu = sum(v) / n
    var = sum((x - mu) ** 2 for x in v) / n
    inv = 1.0 / math.sqrt(var + EPS)
    return [g[j] * (v[j] - mu) * inv + b[j] for j in range(n)]


def softmax(v):
    m = max(v)
    e = [math.exp(x - m) for x in v]
    s = sum(e)
    return [x / s for x in e]


def maxdiff(a, b):
    fa, fb = _fl(a), _fl(b)
    return max((abs(x - y) for x, y in zip(fa, fb)), default=0.0)


def _fl(x):
    if isinstance(x, (int, float)):
        return [float(x)]
    out = []
    for v in x:
        out.extend(_fl(v))
    return out


# ============================================================================
# THE MODEL, RUN TWO WAYS
# ============================================================================

def run_block(P, L, hidden, kv_cache, use_cache):
    """
    One transformer block over a list of token hidden states.

    If use_cache, `hidden` is a SINGLE token and kv_cache holds every
    previous token's K and V for this layer. The new token's K and V are
    appended, and attention runs over the whole cache.

    If not, `hidden` is the whole prefix and everything is recomputed.
    """
    g, b = P[f"L{L}.ln.g"], P[f"L{L}.ln.b"]
    n = [layernorm(h, g, b) for h in hidden]

    Q = mm(n, P[f"L{L}.Wq"])
    K = mm(n, P[f"L{L}.Wk"])
    V = mm(n, P[f"L{L}.Wv"])

    if use_cache:
        kv_cache["K"].extend(K)
        kv_cache["V"].extend(V)
        K_all, V_all = kv_cache["K"], kv_cache["V"]
        q_offset = len(K_all) - len(Q)
    else:
        K_all, V_all = K, V
        q_offset = 0

    scale = 1.0 / math.sqrt(D_HEAD)
    O = [[0.0] * D_MODEL for _ in Q]
    attn_rows = []
    for qi in range(len(Q)):
        abs_pos = q_offset + qi
        for h in range(N_HEADS):
            lo, hi = h * D_HEAD, (h + 1) * D_HEAD
            qv = Q[qi][lo:hi]
            # causal: attend to every cached position up to and including us
            scores = [dot(qv, K_all[j][lo:hi]) * scale
                      for j in range(abs_pos + 1)]
            p = softmax(scores)
            for d in range(D_HEAD):
                O[qi][lo + d] = sum(p[j] * V_all[j][lo + d]
                                    for j in range(abs_pos + 1))
            if h == 0:
                attn_rows.append({"query_pos": abs_pos, "head": h,
                                  "n_keys": abs_pos + 1, "probs": p})

    A = mm(O, P[f"L{L}.Wo"])
    h1 = [[hidden[i][d] + A[i][d] for d in range(D_MODEL)]
          for i in range(len(hidden))]

    g2, b2 = P[f"L{L}.ln2.g"], P[f"L{L}.ln2.b"]
    n2 = [layernorm(x, g2, b2) for x in h1]
    U = mm(n2, P[f"L{L}.W1"])
    G = [[gelu(x) for x in r] for r in U]
    D = mm(G, P[f"L{L}.W2"])
    out = [[h1[i][d] + D[i][d] for d in range(D_MODEL)]
           for i in range(len(h1))]
    return out, attn_rows


def forward(P, token_ids, caches=None, only_last=False):
    """caches None -> recompute everything. caches given -> use/extend them."""
    use_cache = caches is not None
    hidden = [list(P["E"][t]) for t in token_ids]
    per_layer_attn = []
    for L in range(N_LAYERS):
        c = caches[L] if use_cache else None
        hidden, rows = run_block(P, L, hidden, c, use_cache)
        per_layer_attn.append(rows)
    logits = mm(hidden, P["head"])
    return (logits[-1] if only_last else logits), hidden, per_layer_attn


def new_caches():
    return [{"K": [], "V": []} for _ in range(N_LAYERS)]


# ============================================================================
# 1. GENERATION, AND THE PROOF THAT THE CACHE IS EXACT
# ============================================================================

def generate(P):
    """
    Prefill the prompt, then decode N_NEW tokens one at a time using the
    cache. At every step, ALSO recompute the whole prefix from scratch with
    no cache, and check the two agree.

    That check is the point. The KV cache is not an approximation: K and V
    for a token depend only on that token's hidden state, which never
    changes once computed, because attention is causal.
    """
    caches = new_caches()
    steps = []

    # ---- PREFILL: the whole prompt in one pass -------------------------
    logits, hidden, attn = forward(P, PROMPT, caches=caches, only_last=True)
    nxt = max(range(VOCAB), key=lambda v: logits[v])
    cache_len = len(caches[0]["K"])
    steps.append({
        "phase": "prefill", "step": 0,
        "tokens_in": list(PROMPT), "n_tokens_processed": len(PROMPT),
        "logits": logits, "probs": softmax(logits), "chosen": nxt,
        "cache_len_after": cache_len,
        "attn_last_row": attn[0][-1]["probs"],
        "note": "The whole prompt goes through at once. Every position is "
                "independent given causality, so this is a big matrix "
                "multiply -- the GPU's favourite shape.",
    })

    seq = list(PROMPT)
    checks = []
    for s in range(N_NEW):
        seq.append(nxt)
        # decode: ONE token, using the cache
        logits_c, _, attn_c = forward(P, [nxt], caches=caches, only_last=True)
        # the same thing the slow way: recompute the entire prefix
        logits_full, _, _ = forward(P, seq, caches=None, only_last=True)
        err = maxdiff(logits_c, logits_full)
        checks.append({"step": s + 1, "seq_len": len(seq),
                       "max_abs_err": err, "passed": err < 1e-12})

        nxt2 = max(range(VOCAB), key=lambda v: logits_c[v])
        steps.append({
            "phase": "decode", "step": s + 1,
            "tokens_in": [nxt], "n_tokens_processed": 1,
            "logits": logits_c, "probs": softmax(logits_c), "chosen": nxt2,
            "cache_len_after": len(caches[0]["K"]),
            "attn_last_row": attn_c[0][-1]["probs"],
            "recompute_check": {"max_abs_err": err, "passed": err < 1e-12},
            "note": "ONE token in. Every weight in the model is read to do "
                    "one token's worth of arithmetic.",
        })
        nxt = nxt2

    return {
        "steps": steps, "final_sequence": seq + [nxt],
        "cache_equivalence": {
            "per_step": checks,
            "max_abs_err": max(c["max_abs_err"] for c in checks),
            "passed": all(c["passed"] for c in checks),
            "claim": "Decoding with the KV cache gives bit-for-bit the same "
                     "logits as recomputing the entire prefix every time. The "
                     "cache is an exact memoisation, not an approximation -- "
                     "K and V for a token depend only on that token, and "
                     "causality means nothing later can change them.",
        },
        "cache_dump": [{"layer": L,
                        "K": caches[L]["K"], "V": caches[L]["V"],
                        "entries": len(caches[L]["K"])}
                       for L in range(N_LAYERS)],
    }


# ============================================================================
# 2. ARITHMETIC INTENSITY  -- the idea the whole part hangs on
# ============================================================================

def cost_model(n_params, n_layers, d_model, n_kv_heads, d_head,
               seq, batch, w_bytes=2, kv_bytes=2):
    """
    FLOPs and bytes moved for one prefill of `seq` tokens and one decode
    step, at batch size `batch`.

    FLOPs: a matmul of (m x k) by (k x n) is 2*m*k*n. Forward through the
    whole model is about 2 * n_params per token, plus attention's own
    quadratic term.

    BYTES: this is the half people skip. To compute anything you must first
    READ the weights out of HBM. In decode you read all of them, per step,
    and use them for one token.
    """
    # ---- prefill: `seq` tokens at once -------------------------------
    pf_flops = 2 * n_params * seq * batch
    pf_attn = 2 * 2 * n_layers * batch * n_kv_heads * d_head * seq * seq
    pf_flops += pf_attn
    pf_bytes = n_params * w_bytes                     # weights read once
    pf_bytes += 2 * n_layers * batch * seq * n_kv_heads * d_head * kv_bytes

    # ---- decode: ONE token, but the cache must be re-read -------------
    dc_flops = 2 * n_params * 1 * batch
    dc_attn = 2 * 2 * n_layers * batch * n_kv_heads * d_head * seq
    dc_flops += dc_attn
    dc_bytes = n_params * w_bytes                     # ALL of them, again
    dc_bytes += 2 * n_layers * batch * seq * n_kv_heads * d_head * kv_bytes

    return {
        "prefill": {"flops": pf_flops, "bytes": pf_bytes,
                    "intensity": pf_flops / pf_bytes},
        "decode": {"flops": dc_flops, "bytes": dc_bytes,
                   "intensity": dc_flops / dc_bytes},
    }


def roofline(models, gpus):
    """
    Where each regime sits against a device's ridge point.

    ridge = peak FLOP/s / memory bandwidth. Below it you are bandwidth-bound
    and cannot reach peak FLOPs no matter what you do.
    """
    out = []
    for g in gpus:
        ridge = g["bf16_dense_tflops"] * 1e12 / g["hbm_bw_bytes_per_s"]
        rows = []
        for m in models:
            for batch in (1, 8, 32, 128):
                c = cost_model(m["params"], m["n_layers"], m["d_model"],
                               m.get("n_kv_heads", m["n_heads"]),
                               m["d_model"] // m["n_heads"],
                               m["seq"], batch)
                rows.append({
                    "model": m["name"], "batch": batch,
                    "prefill_intensity": round(c["prefill"]["intensity"], 2),
                    "decode_intensity": round(c["decode"]["intensity"], 3),
                    "decode_bound": ("memory" if c["decode"]["intensity"] < ridge
                                     else "compute"),
                    "prefill_bound": ("memory" if c["prefill"]["intensity"] < ridge
                                      else "compute"),
                    "decode_flops_utilisation": round(
                        min(1.0, c["decode"]["intensity"] / ridge), 5),
                })
        out.append({"gpu": g["name"], "ridge_flops_per_byte": round(ridge, 2),
                    "peak_tflops": g["bf16_dense_tflops"],
                    "hbm_bw_bytes_per_s": g["hbm_bw_bytes_per_s"],
                    "rows": rows})
    return out


# ============================================================================
# 3. THE KV CACHE, AT REAL SCALE
# ============================================================================

def kv_scale(models):
    """
    KV cache bytes = 2 (K and V) * layers * seq * n_kv_heads * d_head * dtype
    per sequence. Note it does NOT depend on d_model directly -- it depends
    on the KV head count, which is exactly the lever GQA pulls.
    """
    rows = []
    for m in models:
        d_head = m["d_model"] // m["n_heads"]
        kvh = m.get("n_kv_heads", m["n_heads"])
        for seq in (1024, 4096, 8192, 32768, 131072):
            per_seq = 2 * m["n_layers"] * seq * kvh * d_head * 2   # bf16
            rows.append({
                "model": m["name"], "seq": seq,
                "n_heads": m["n_heads"], "n_kv_heads": kvh,
                "bytes_per_sequence": per_seq,
                "gb_per_sequence": round(per_seq / 1e9, 4),
                "weights_gb": round(m["params"] * 2 / 1e9, 1),
                "sequences_to_equal_weights": round(m["params"] * 2 / per_seq, 1),
                "mha_bytes_if_no_gqa": 2 * m["n_layers"] * seq * m["n_heads"] * d_head * 2,
            })
    return {
        "formula": "2 * layers * seq * n_kv_heads * d_head * dtype_bytes",
        "per": "one sequence",
        "note": "It scales LINEARLY with sequence length and with batch, and "
                "it is read in full on every single decode step. Weights are "
                "a fixed cost; the KV cache is a per-request, per-token cost, "
                "and past a few thousand tokens it is the thing that runs you "
                "out of memory.",
        "gqa_note": "The formula depends on n_kv_heads, not n_heads. That is "
                    "the entire reason grouped-query attention exists: it is "
                    "a KV-cache optimisation that happens to live in the "
                    "architecture.",
        "rows": rows,
    }


# ============================================================================
# 4. BATCHING  -- why it is nearly free in decode
# ============================================================================

def batching(model, gpu, seqs=(2048,)):
    """
    Weights are read once per STEP, not per sequence. So doubling the batch
    doubles the FLOPs and leaves the weight-reading term unchanged. Decode
    intensity therefore rises almost linearly with batch, until the KV cache
    term (which IS per-sequence) takes over.
    """
    ridge = gpu["bf16_dense_tflops"] * 1e12 / gpu["hbm_bw_bytes_per_s"]
    out = []
    d_head = model["d_model"] // model["n_heads"]
    kvh = model.get("n_kv_heads", model["n_heads"])
    for seq in seqs:
        rows = []
        for batch in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
            c = cost_model(model["params"], model["n_layers"],
                           model["d_model"], kvh, d_head, seq, batch)
            w_bytes = model["params"] * 2
            kv = 2 * model["n_layers"] * batch * seq * kvh * d_head * 2
            rows.append({
                "batch": batch,
                "intensity": round(c["decode"]["intensity"], 3),
                "weight_bytes": w_bytes, "kv_bytes": kv,
                "kv_share": round(kv / (w_bytes + kv), 4),
                "bound": "memory" if c["decode"]["intensity"] < ridge else "compute",
                "flops_utilisation": round(
                    min(1.0, c["decode"]["intensity"] / ridge), 5),
            })
        out.append({"seq": seq, "rows": rows})
    return {"ridge": round(ridge, 2), "gpu": gpu["name"],
            "model": model["name"], "by_seq": out,
            "why": "Weights are read once per step regardless of batch size. "
                   "Adding a sequence adds FLOPs but almost no weight traffic, "
                   "so intensity climbs nearly linearly -- decode batching is "
                   "close to free until the KV cache, which IS per-sequence, "
                   "starts to dominate the bytes."}


def build():
    P = init()
    gen = generate(P)

    # real configs, quoted, matching T.reference_configs in the main trace
    models = [
        {"name": "Llama 3 8B", "params": 8.03e9, "d_model": 4096,
         "n_layers": 32, "n_heads": 32, "n_kv_heads": 8, "seq": 8192},
        {"name": "Llama 3 70B", "params": 70.6e9, "d_model": 8192,
         "n_layers": 80, "n_heads": 64, "n_kv_heads": 8, "seq": 8192},
        {"name": "GPT-3 175B", "params": 175e9, "d_model": 12288,
         "n_layers": 96, "n_heads": 96, "seq": 2048},
    ]
    gpus = [
        {"name": "A100 80GB", "bf16_dense_tflops": 312,
         "hbm_bw_bytes_per_s": 2039e9},
        {"name": "H100 80GB", "bf16_dense_tflops": 495,
         "hbm_bw_bytes_per_s": 3350e9},
        {"name": "H200 141GB", "bf16_dense_tflops": 495,
         "hbm_bw_bytes_per_s": 4800e9},
        {"name": "B200 192GB", "bf16_dense_tflops": 2250,
         "hbm_bw_bytes_per_s": 8000e9},
    ]

    return {
        "meta": {
            "generated_by": "code/inference_toy.py",
            "d_model": D_MODEL, "n_heads": N_HEADS, "d_head": D_HEAD,
            "n_kv_heads": N_KV_HEADS, "d_ff": D_FF, "n_layers": N_LAYERS,
            "vocab": VOCAB, "prompt": PROMPT, "n_new": N_NEW,
            "token_strings": TOKEN_STR,
            "description": "Autoregressive generation with a KV cache, "
                           "proven exact, plus the arithmetic-intensity "
                           "argument that separates prefill from decode.",
            "_source_note": "Model and GPU specifications are quoted public "
                            "figures, external to this simulation.",
        },
        "params": {k: P[k] for k in sorted(P)},
        "generation": gen,
        "cost_model": {
            "flops_note": "a matmul (m x k)(k x n) is 2*m*k*n; a forward pass "
                          "is about 2 * params per token",
            "bytes_note": "every weight must be READ from HBM before it can "
                          "be used. In decode that read is the whole cost.",
            "toy": cost_model(sum(len(_fl(v)) for v in P.values()),
                              N_LAYERS, D_MODEL, N_KV_HEADS, D_HEAD,
                              len(PROMPT) + N_NEW, 1),
        },
        "roofline": roofline(models, gpus),
        "kv_cache": kv_scale(models),
        "batching": batching(models[1], gpus[1]),
        "models": models, "gpus": gpus,
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(os.path.dirname(here), "assets", "data")
    os.makedirs(outdir, exist_ok=True)
    d = build()
    payload = json.dumps(d, indent=1, allow_nan=False)
    open(os.path.join(outdir, "infer.json"), "w").write(payload)
    with open(os.path.join(outdir, "infer.js"), "w") as f:
        f.write("// GENERATED by code/inference_toy.py -- do not hand-edit.\n")
        f.write("window.INFER = " + payload + ";\n")

    g = d["generation"]
    ce = g["cache_equivalence"]
    print("=" * 72)
    print("inference_toy.py — prefill, decode, and the KV cache")
    print("=" * 72)
    print(f"  prompt {[TOKEN_STR[t] for t in PROMPT]}"
          f"  ->  generated {[TOKEN_STR[t] for t in g['final_sequence'][len(PROMPT):]]}")
    print()
    print(f"  KV-cache equivalence over {len(ce['per_step'])} decode steps:")
    for c in ce["per_step"]:
        print(f"    step {c['step']}  seq_len {c['seq_len']}"
              f"   max|Δ| {c['max_abs_err']:.3e}   "
              f"{'PASS' if c['passed'] else 'FAIL'}")
    print(f"  -> {'PASS' if ce['passed'] else 'FAIL'}  "
          f"(cached decode == full recompute)")
    print()
    rl = d["roofline"][1]
    print(f"  roofline on {rl['gpu']}: ridge = {rl['ridge_flops_per_byte']} FLOP/byte")
    for r in rl["rows"]:
        if r["model"] == "Llama 3 70B" and r["batch"] in (1, 32):
            print(f"    70B batch {r['batch']:3d}   prefill {r['prefill_intensity']:9.2f}"
                  f" ({r['prefill_bound']})   decode {r['decode_intensity']:7.3f}"
                  f" ({r['decode_bound']}, {r['decode_flops_utilisation']*100:.2f}% of peak FLOPs)")
    print()
    kv = d["kv_cache"]["rows"]
    print("  KV cache per sequence (bf16):")
    for r in kv:
        if r["model"] == "Llama 3 70B" and r["seq"] in (8192, 131072):
            print(f"    70B seq {r['seq']:6d}  {r['gb_per_sequence']:8.3f} GB"
                  f"   ({r['sequences_to_equal_weights']:.0f} sequences = the weights)"
                  f"   MHA would be {r['mha_bytes_if_no_gqa']/1e9:.2f} GB")
    print()
    print(f"  wrote {os.path.join(outdir, 'infer.js')}")
    print("=" * 72)
    if not ce["passed"]:
        raise SystemExit("KV CACHE EQUIVALENCE FAILED")


if __name__ == "__main__":
    main()
