#!/usr/bin/env python3
"""
attention_variants.py — MHA, MQA, GQA and MLA on the same input.

Pure Python standard library. No numpy, no torch.

All four compute attention. They differ in exactly one thing: how much
KEY/VALUE state a token leaves behind, because during decode that state is
re-read from HBM on every single generated token and it grows linearly with
context. Multi-head attention was designed before anyone was serving a
128k-token chat, and its KV cache is the reason the other three exist.

What this file establishes, numerically rather than by assertion:

  * all four are valid attention — probability rows sum to one, masked
    positions are exactly zero, and every output row lies inside the convex
    hull of the value rows it is allowed to see
  * GQA and MQA are not different mechanisms. They are ordinary multi-head
    attention with the K/V projection TIED across heads, which we prove by
    building the tied full-width matrix and showing the outputs are
    bit-identical
  * MLA is ordinary multi-head attention with the K/V projection factored
    through a rank-d_c bottleneck: W_k = W_dkv W_uk exactly
  * therefore all three savings are the same kind of thing — a rank
    constraint on the K/V projection — and we measure the rank of each
  * MLA's absorption identity, q (c W_uk)^T = (q W_uk^T) c^T, which is what
    lets you cache the latent and never materialise K
  * the KV cache arithmetic, cross-checked against the rows already
    published in assets/data/infer.json so the two pages cannot disagree

Emits:
    assets/data/attnvar.js     (window.ATTNVAR = {...})
    assets/data/attnvar.json

Run:  python3 code/attention_variants.py
"""

import json
import math
import os

# ============================================================================
# SHAPES
# ============================================================================
# 4 query heads is the smallest count that lets GQA sit strictly between MHA
# and MQA (4 -> 2 groups -> 1). d_head 2 keeps every matrix printable, and
# d_model 8 = n_heads * d_head so W_o is square, as in a real block.

D_MODEL = 8
N_HEADS = 4
D_HEAD = D_MODEL // N_HEADS       # 2
SEQ = 4
D_LATENT = 3                      # MLA's compression rank d_c
CAUSAL = True                     # decode-shaped: this page is about the cache

TOKENS = ["the", "cat", "sat", "down"]

# The four configurations, as (label, n_kv_heads). MLA is handled separately
# because its cache is not "n_kv_heads worth of K and V" at all.
VARIANTS = [("MHA", N_HEADS), ("GQA", 2), ("MQA", 1)]

DTYPE_BYTES = 2                   # bf16 / fp16, what everyone serves in
TOL = 1e-12


def det(i, j, salt):
    """Deterministic small weights. Reproducible without seeding."""
    return round(((((i * 7 + j * 13 + salt * 11 + 3) % 19) - 9) / 30.0), 4)


# ============================================================================
# Linear algebra, row-vector convention
# ============================================================================
# X is (seq, d_in), W is (d_in, d_out), Y = X @ W. Same convention as
# transformer_2layer.py and positional.py.

def mm(A, B):
    m, k, n = len(A), len(B), len(B[0])
    return [[sum(A[i][t] * B[t][j] for t in range(k)) for j in range(n)]
            for i in range(m)]


def T_(A):
    return [list(r) for r in zip(*A)]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


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


def r6(x):
    """Round for the payload. Every PASS/FAIL below is taken unrounded."""
    return round(x, 9)


def rr(A):
    return [[r6(v) for v in row] for row in A]


def softmax_row(s):
    """Numerically stable: subtract the row max before exponentiating.
    -inf entries exponentiate to exactly 0.0, which is what makes the mask
    check below an equality rather than a threshold."""
    m = max(v for v in s if v != float("-inf"))
    e = [0.0 if v == float("-inf") else math.exp(v - m) for v in s]
    z = sum(e)
    return [v / z for v in e]


def rank(A, tol=1e-9):
    """
    Rank by Gaussian elimination with partial pivoting.

    Used to make the central structural claim measurable: MHA, GQA, MQA and
    MLA all differ by the RANK of the effective d_model -> d_model key
    projection. Everything else about them is identical.
    """
    M = [list(r) for r in A]
    rows, cols = len(M), len(M[0])
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        piv = max(range(r, rows), key=lambda i: abs(M[i][c]))
        if abs(M[piv][c]) < tol:
            continue
        M[r], M[piv] = M[piv], M[r]
        pv = M[r][c]
        for i in range(r + 1, rows):
            f = M[i][c] / pv
            if f:
                for j in range(c, cols):
                    M[i][j] -= f * M[r][j]
        r += 1
    return r


# ============================================================================
# THE ONE ATTENTION IMPLEMENTATION
# ============================================================================
# Every variant below routes through this function. That is deliberate: if
# the variants used different code, "they all compute valid attention" would
# be a statement about four implementations rather than about four weight
# layouts. Only the SHAPE of Wk/Wv and the head -> kv-head map change.

def head_map(n_heads, n_kv_heads):
    """
    Which KV head each query head reads.

    n_heads // n_kv_heads query heads share one KV head. MHA is the case
    where the group size is 1 (nobody shares); MQA is the case where the
    group size is n_heads (everybody shares one). GQA is the dial in
    between, and there is nothing else to it.
    """
    if n_heads % n_kv_heads:
        raise SystemExit("n_heads must be divisible by n_kv_heads")
    g = n_heads // n_kv_heads
    return [h // g for h in range(n_heads)]


def attention(X, Wq, Wk, Wv, Wo, n_heads, n_kv_heads, d_head):
    """
    Q = X Wq                  (seq, n_heads   * d_head)
    K = X Wk, V = X Wv        (seq, n_kv_heads * d_head)
    head h reads KV head hmap[h]; O = concat_h softmax(Q_h K_g^T/sqrt(d)) V_g
    Y = O Wo

    The scores are causal: query i may read key j only if j <= i. That is
    the decode-time shape, and it is what makes K and V for token j reusable
    forever once computed — the thing the cache is exploiting.
    """
    hmap = head_map(n_heads, n_kv_heads)
    Q, K, V = mm(X, Wq), mm(X, Wk), mm(X, Wv)
    n = len(X)
    scale = 1.0 / math.sqrt(d_head)
    heads = []
    O = [[0.0] * (n_heads * d_head) for _ in range(n)]
    for h in range(n_heads):
        g = hmap[h]
        ql, qh = h * d_head, (h + 1) * d_head
        kl, kh = g * d_head, (g + 1) * d_head
        Qh = [row[ql:qh] for row in Q]
        Kg = [row[kl:kh] for row in K]
        Vg = [row[kl:kh] for row in V]
        scores = [[(dot(Qh[i], Kg[j]) * scale
                    if (not CAUSAL or j <= i) else float("-inf"))
                   for j in range(n)] for i in range(n)]
        probs = [softmax_row(r) for r in scores]
        Oh = mm(probs, Vg)
        for i in range(n):
            for j in range(d_head):
                O[i][ql + j] = Oh[i][j]
        heads.append({"head": h, "kv_head": g, "Q": Qh, "K": Kg, "V": Vg,
                      "scores": [[None if v == float("-inf") else v
                                  for v in row] for row in scores],
                      "probs": probs, "O": Oh,
                      "row_sums": [sum(r) for r in probs]})
    Y = mm(O, Wo)
    return {"Q": Q, "K": K, "V": V, "heads": heads, "hmap": hmap,
            "concat": O, "Y": Y}


# ============================================================================
# VALIDITY — what "this is still attention" actually means
# ============================================================================

def validity(res):
    """
    Three properties, all checkable, none of which any of the variants is
    allowed to give up:

      1. every probability row sums to exactly 1 (softmax's normaliser)
      2. every probability is >= 0, and every MASKED entry is exactly 0.0
         (not merely small — exp(-inf) is 0.0 in IEEE arithmetic, so this
         is an equality test)
      3. every output element lies between the min and the max of the value
         elements it was allowed to read. This is the property that makes
         attention an AVERAGE: a convex combination cannot leave the hull
         of its inputs. If a variant broke it, the thing it computed would
         not be attention regardless of what the code claimed.
    """
    worst_sum, worst_neg, worst_mask, hull_bad = 0.0, 0.0, 0.0, 0
    n = len(res["heads"][0]["probs"])
    for H in res["heads"]:
        for i in range(n):
            worst_sum = max(worst_sum, abs(sum(H["probs"][i]) - 1.0))
            for j in range(n):
                p = H["probs"][i][j]
                worst_neg = min(worst_neg, p)
                if CAUSAL and j > i:
                    worst_mask = max(worst_mask, abs(p))
            allowed = range(i + 1) if CAUSAL else range(n)
            for c in range(len(H["V"][0])):
                lo = min(H["V"][j][c] for j in allowed)
                hi = max(H["V"][j][c] for j in allowed)
                v = H["O"][i][c]
                if v < lo - 1e-12 or v > hi + 1e-12:
                    hull_bad += 1
    return {
        "row_sum_max_abs_err": worst_sum,
        "min_probability": worst_neg,
        "masked_max_abs": worst_mask,
        "convex_hull_violations": hull_bad,
        "passed": (worst_sum < TOL and worst_neg >= 0.0
                   and worst_mask == 0.0 and hull_bad == 0),
    }


# ============================================================================
# BUILDING THE FOUR LAYOUTS
# ============================================================================

def build_weights():
    """
    One Wq and one Wo shared by every variant — the query side and the
    output projection are untouched by all of this. Only Wk/Wv change.

    For GQA and MQA the K/V projection is NARROWER: (d_model, n_kv*d_head).
    To prove that this is the same thing as MHA with tied heads we also
    build the WIDE version by repeating each group's block across the heads
    that share it, and run the identical MHA code path on it.
    """
    Wq = [[det(i, j, 1) for j in range(N_HEADS * D_HEAD)] for i in range(D_MODEL)]
    Wo = [[det(i, j, 4) for j in range(D_MODEL)] for i in range(N_HEADS * D_HEAD)]
    W = {"Wq": Wq, "Wo": Wo, "kv": {}}
    for label, nkv in VARIANTS:
        W["kv"][label] = {
            "n_kv_heads": nkv,
            "Wk": [[det(i, j, 2) for j in range(nkv * D_HEAD)] for i in range(D_MODEL)],
            "Wv": [[det(i, j, 3) for j in range(nkv * D_HEAD)] for i in range(D_MODEL)],
        }
    # MLA: down-project to a rank-d_c latent, then up-project to full width.
    W["mla"] = {
        "W_dkv": [[det(i, j, 7) for j in range(D_LATENT)] for i in range(D_MODEL)],
        "W_uk": [[det(i, j, 8) for j in range(N_HEADS * D_HEAD)] for i in range(D_LATENT)],
        "W_uv": [[det(i, j, 9) for j in range(N_HEADS * D_HEAD)] for i in range(D_LATENT)],
    }
    return W


def widen(Wnarrow, n_heads, n_kv_heads, d_head):
    """
    (d_model, n_kv*d_head) -> (d_model, n_heads*d_head) by repeating each
    KV head's column block for every query head in its group.

    This is the entire content of "GQA is MHA with tied KV heads": the same
    d_head columns appear n_heads/n_kv_heads times. It is a rank-deficient
    matrix by construction, which is what the rank measurement below picks
    up, and it is why the cache only has to store one copy.
    """
    hmap = head_map(n_heads, n_kv_heads)
    out = []
    for row in Wnarrow:
        r = []
        for h in range(n_heads):
            g = hmap[h]
            r.extend(row[g * d_head:(g + 1) * d_head])
        out.append(r)
    return out


# ============================================================================
# MLA — compress K and V into one latent, cache THAT
# ============================================================================

def mla(X, Wq, W_dkv, W_uk, W_uv, Wo):
    """
    DeepSeek's multi-head latent attention.

        c_n = x_n W_dkv                (d_c, and this is ALL that is cached)
        K   = c W_uk,  V = c W_uv      (decompressed on use, full width)

    Substituting, K = X (W_dkv W_uk). So MLA is exactly multi-head
    attention whose key projection happens to be the product of two thin
    matrices — hence of rank at most d_c. Nothing about the attention
    changes; only the FACTORISATION of the projection, and therefore what
    is worth storing.

    The cache holds d_c numbers per token per layer instead of
    2 * n_heads * d_head. In exchange, every decode step pays two extra
    matmuls to decompress. That is the trade, stated plainly: fewer bytes
    moved, more FLOPs done, on hardware where decode is bandwidth-bound.
    """
    Wk_eq = mm(W_dkv, W_uk)
    Wv_eq = mm(W_dkv, W_uv)
    res = attention(X, Wq, Wk_eq, Wv_eq, Wo, N_HEADS, N_HEADS, D_HEAD)
    C = mm(X, W_dkv)
    res["C"] = C
    res["Wk_eq"] = Wk_eq
    res["Wv_eq"] = Wv_eq
    return res


def absorption_check(X, Wq, W_dkv, W_uk):
    """
    THE IDENTITY THAT MAKES IT WORTH DOING.

        q_m . k_n  =  q_m (c_n W_uk_h)^T  =  (q_m W_uk_h^T) . c_n

    The up-projection is a fixed matrix, so it can be folded into the query
    ONCE per step instead of being applied to every cached token. K is then
    never materialised at all: the score is a dot product between a
    reprojected query and the raw latent.

    Checked here for every (head, query position, key position).
    """
    Q = mm(X, Wq)
    C = mm(X, W_dkv)
    n = len(X)
    worst = 0.0
    for h in range(N_HEADS):
        lo, hi = h * D_HEAD, (h + 1) * D_HEAD
        Uh = [row[lo:hi] for row in W_uk]            # (d_c, d_head)
        for m in range(n):
            qh = Q[m][lo:hi]
            q_abs = mm([qh], T_(Uh))[0]              # (d_c,)
            for nn in range(n):
                k_h = mm([C[nn]], Uh)[0]             # decompressed key
                worst = max(worst, abs(dot(qh, k_h) - dot(q_abs, C[nn])))
    return {
        "max_abs_err": worst,
        "passed": worst < TOL,
        "claim": "q.(c W_uk) = (q W_uk^T).c for every head and every (m,n). "
                 "The up-projection folds into the query, so a decode step "
                 "reads only the latent — K is never built.",
    }


# ============================================================================
# COSTS
# ============================================================================

def variant_costs(label, n_kv_heads, cached_elements, params, rank_k, notes):
    """
    Per-token, per-layer cache in ELEMENTS first, bytes second. Elements is
    the honest unit: bytes is just elements times whatever dtype you serve
    in, and the dtype is a separate decision.
    """
    return {
        "variant": label,
        "n_query_heads": N_HEADS,
        "n_kv_heads": n_kv_heads,
        "queries_per_kv_head": (N_HEADS // n_kv_heads) if n_kv_heads else None,
        "head_map": head_map(N_HEADS, n_kv_heads) if n_kv_heads else None,
        "cache_elements_per_token_per_layer": cached_elements,
        "cache_bytes_per_token_per_layer": cached_elements * DTYPE_BYTES,
        "params": params,
        "key_projection_rank": rank_k,
        "notes": notes,
    }


def real_models(ref):
    """
    The real table. d_head is DERIVED as d_model / n_heads and n_kv_heads
    comes from the config; nothing here is typed by hand.

    KV cache for one sequence:
        2 (K and V) * n_layers * seq * n_kv_heads * d_head * dtype_bytes

    The 2 is K and V; the n_kv_heads is the only term any of the variants
    touches. That is why the whole family exists.
    """
    seqs = [1024, 4096, 8192, 32768, 131072]
    out = []
    for m in ref["models"]:
        if "n_kv_heads" not in m:
            continue
        dh = m["d_model"] // m["n_heads"]
        rows = []
        for s in seqs:
            def kv(nkv):
                return 2 * m["n_layers"] * s * nkv * dh * DTYPE_BYTES
            lat = MLA_REF["kv_lora_rank"] + MLA_REF["qk_rope_head_dim"]
            rows.append({
                "seq": s,
                "mha_bytes": kv(m["n_heads"]),
                "gqa_bytes": kv(m["n_kv_heads"]),
                "mqa_bytes": kv(1),
                "mla_bytes_hypothetical": m["n_layers"] * s * lat * DTYPE_BYTES,
            })
        out.append({
            "name": m["name"], "params": m["params"],
            "d_model": m["d_model"], "n_layers": m["n_layers"],
            "n_heads": m["n_heads"], "n_kv_heads": m["n_kv_heads"],
            "d_head": dh,
            "queries_per_kv_head": m["n_heads"] // m["n_kv_heads"],
            "kv_reduction_vs_mha": m["n_heads"] / float(m["n_kv_heads"]),
            "weights_bytes_bf16": m["params"] * DTYPE_BYTES,
            "per_token": {
                "mha_bytes": 2 * m["n_layers"] * m["n_heads"] * dh * DTYPE_BYTES,
                "gqa_bytes": 2 * m["n_layers"] * m["n_kv_heads"] * dh * DTYPE_BYTES,
                "mqa_bytes": 2 * m["n_layers"] * 1 * dh * DTYPE_BYTES,
            },
            "rows": rows,
            "native_seq": m["seq"],
        })
    return out


def crosscheck_infer(root, real):
    """
    assets/data/infer.json already publishes a KV-cache table (page 25 uses
    it). If this file's arithmetic disagreed with that one, the site would
    contradict itself two pages apart. So: recompute every published row
    from the model geometry and demand exact integer agreement, for both
    the GQA figure and the MHA counterfactual.
    """
    p = os.path.join(root, "assets", "data", "infer.json")
    if not os.path.exists(p):
        return {"available": False, "checked": 0, "max_abs_err": 0.0,
                "passed": True, "note": "infer.json not present; skipped"}
    with open(p) as f:
        inf = json.load(f)
    idx = {r["name"]: r for r in real}
    checked, worst, rows = 0, 0, []
    for row in inf["kv_cache"]["rows"]:
        mine = idx.get(row["model"])
        if not mine:
            continue
        got = [x for x in mine["rows"] if x["seq"] == row["seq"]]
        if not got:
            continue
        g = got[0]
        d1 = abs(g["gqa_bytes"] - row["bytes_per_sequence"])
        d2 = abs(g["mha_bytes"] - row["mha_bytes_if_no_gqa"])
        worst = max(worst, d1, d2)
        checked += 1
        rows.append({"model": row["model"], "seq": row["seq"],
                     "mine_gqa": g["gqa_bytes"],
                     "infer_gqa": row["bytes_per_sequence"],
                     "mine_mha": g["mha_bytes"],
                     "infer_mha": row["mha_bytes_if_no_gqa"],
                     "delta": d1 + d2})
    return {"available": True, "checked": checked, "max_abs_err": float(worst),
            "passed": worst == 0, "rows": rows,
            "formula": inf["kv_cache"]["formula"],
            "note": "every published row recomputed from d_model/n_heads/"
                    "n_kv_heads/n_layers and matched to the byte"}


# ============================================================================
# LITERATURE — quality claims we quote rather than measure
# ============================================================================
# A four-token toy cannot measure model quality. Anything about how well a
# variant WORKS is somebody else's experiment, and is labelled as such.

MLA_REF = {
    "_source": "DeepSeek-V2 / V3 released model configs and papers; quoted, "
               "not derived",
    "name": "DeepSeek-V3",
    "d_model": 7168,
    "n_layers": 61,
    "n_heads": 128,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "kv_lora_rank": 512,
    "note": "The cached state per token per layer is kv_lora_rank + "
            "qk_rope_head_dim: the compressed latent, plus one small "
            "SHARED key carrying RoPE. RoPE is position-dependent, so it "
            "cannot be absorbed into the up-projection — hence the "
            "'decoupled' rope key, kept separate and shared by all heads.",
}

LITERATURE = {
    "_source": "published papers; external to this file, quoted not measured",
    "items": [
        {"variant": "MHA",
         "claim": "The original formulation. Every head gets its own K and V, "
                  "so nothing is shared and nothing is lost.",
         "source": "Vaswani et al., Attention Is All You Need, 2017"},
        {"variant": "MQA",
         "claim": "One shared KV head. Reported as giving a large decode "
                  "speedup with 'minor quality degradation', and later work "
                  "found it can also destabilise training.",
         "source": "Shazeer, Fast Transformer Decoding, 2019"},
        {"variant": "GQA",
         "claim": "Interpolates between the two. Reported as reaching quality "
                  "close to MHA at speed close to MQA, and as being reachable "
                  "by 'uptraining' an existing MHA checkpoint with about 5% "
                  "of the original pre-training compute.",
         "source": "Ainslie et al., GQA, 2023"},
        {"variant": "MLA",
         "claim": "Low-rank joint compression of K and V. Reported as "
                  "matching or beating MHA quality, and as cutting the KV "
                  "cache by about 93% relative to DeepSeek 67B — note that "
                  "is a comparison against a DIFFERENT model's geometry. The "
                  "same-geometry counterfactual (what MLA's own 128 heads x "
                  "128 dims would have cost as plain MHA) is computed in "
                  "`deepseek` above and is a larger factor; the two numbers "
                  "are answering different questions.",
         "source": "DeepSeek-V2, 2024"},
        {"variant": "all",
         "claim": "Quality numbers above are other people's experiments. A "
                  "four-token toy can prove that each variant computes valid "
                  "attention and exactly how many bytes it caches; it cannot "
                  "prove anything about how well a trained model does.",
         "source": "this file"},
    ],
}


# ============================================================================
# BUILD
# ============================================================================

def load_reference_configs(root):
    p = os.path.join(root, "assets", "data", "trace.json")
    if not os.path.exists(p):
        raise SystemExit("missing " + p + " — run code/ground_truth.py first")
    with open(p) as f:
        return json.load(f)["reference_configs"]


def build(root):
    ref = load_reference_configs(root)
    W = build_weights()
    X = [[round(((t * 5 + d * 3) % 7 + 1) / 8.0, 4) for d in range(D_MODEL)]
         for t in range(SEQ)]

    variants = {}
    equivalences = []

    for label, nkv in VARIANTS:
        kv = W["kv"][label]
        res = attention(X, W["Wq"], kv["Wk"], kv["Wv"], W["Wo"],
                        N_HEADS, nkv, D_HEAD)
        val = validity(res)

        # the tied-MHA equivalence: widen and re-run through the n_kv = n_heads
        # path, which is literally the MHA code
        Wk_w = widen(kv["Wk"], N_HEADS, nkv, D_HEAD)
        Wv_w = widen(kv["Wv"], N_HEADS, nkv, D_HEAD)
        wide = attention(X, W["Wq"], Wk_w, Wv_w, W["Wo"],
                         N_HEADS, N_HEADS, D_HEAD)
        err = maxdiff(res["Y"], wide["Y"])
        equivalences.append({
            "variant": label, "max_abs_err": err, "passed": err < TOL,
            "claim": label + " equals MHA whose key/value projection repeats "
                     "each of its " + str(nkv) + " blocks " +
                     str(N_HEADS // nkv) + "x across the " + str(N_HEADS) +
                     " query heads. Same code path, same output.",
        })

        # cache: K and V, one copy per KV head
        cache_el = 2 * nkv * D_HEAD
        params = (D_MODEL * N_HEADS * D_HEAD          # Wq
                  + 2 * D_MODEL * nkv * D_HEAD        # Wk, Wv
                  + N_HEADS * D_HEAD * D_MODEL)       # Wo
        rk = rank(Wk_w)

        variants[label] = {
            "cost": variant_costs(label, nkv, cache_el, params, rk, {
                "shares": "each KV head serves " + str(N_HEADS // nkv) +
                          " query head" + ("s" if N_HEADS // nkv > 1 else ""),
            }),
            "Wk": rr(kv["Wk"]), "Wv": rr(kv["Wv"]),
            "Wk_widened": rr(Wk_w),
            "K": rr(res["K"]), "V": rr(res["V"]),
            "hmap": res["hmap"],
            "heads": [{
                "head": H["head"], "kv_head": H["kv_head"],
                "probs": rr(H["probs"]),
                "scores": [[None if v is None else r6(v) for v in row]
                           for row in H["scores"]],
                "O": rr(H["O"]),
                "row_sums": [r6(v) for v in H["row_sums"]],
            } for H in res["heads"]],
            "Y": rr(res["Y"]),
            "validity": val,
        }

    # ---- MLA -------------------------------------------------------------
    M = W["mla"]
    mres = mla(X, W["Wq"], M["W_dkv"], M["W_uk"], M["W_uv"], W["Wo"])
    mval = validity(mres)
    absorb = absorption_check(X, W["Wq"], M["W_dkv"], M["W_uk"])
    mla_params = (D_MODEL * N_HEADS * D_HEAD            # Wq
                  + D_MODEL * D_LATENT                  # W_dkv
                  + 2 * D_LATENT * N_HEADS * D_HEAD     # W_uk, W_uv
                  + N_HEADS * D_HEAD * D_MODEL)         # Wo
    mla_cost = variant_costs("MLA", N_HEADS, D_LATENT, mla_params,
                             rank(mres["Wk_eq"]),
                             {"shares": "no sharing at all — every query head "
                                        "keeps its own K and V; the saving "
                                        "comes from caching the rank-" +
                                        str(D_LATENT) + " latent they are "
                                        "both decompressed from"})
    mla_cost["n_kv_heads"] = N_HEADS
    mla_cost["queries_per_kv_head"] = 1
    mla_cost["head_map"] = list(range(N_HEADS))
    mla_cost["d_latent"] = D_LATENT

    variants["MLA"] = {
        "cost": mla_cost,
        "W_dkv": rr(M["W_dkv"]), "W_uk": rr(M["W_uk"]), "W_uv": rr(M["W_uv"]),
        "Wk_eq": rr(mres["Wk_eq"]), "Wv_eq": rr(mres["Wv_eq"]),
        "C": rr(mres["C"]),
        "K": rr(mres["K"]), "V": rr(mres["V"]),
        "hmap": mres["hmap"],
        "heads": [{
            "head": H["head"], "kv_head": H["kv_head"],
            "probs": rr(H["probs"]),
            "scores": [[None if v is None else r6(v) for v in row]
                       for row in H["scores"]],
            "O": rr(H["O"]),
            "row_sums": [r6(v) for v in H["row_sums"]],
        } for H in mres["heads"]],
        "Y": rr(mres["Y"]),
        "validity": mval,
        "absorption": absorb,
        "factorisation": {
            "d_latent": D_LATENT,
            "rank_Wk_eq": rank(mres["Wk_eq"]),
            "rank_bound": min(D_LATENT, D_MODEL, N_HEADS * D_HEAD),
            "passed": rank(mres["Wk_eq"]) == D_LATENT,
            "claim": "W_k = W_dkv W_uk, a product through a " +
                     str(D_LATENT) + "-dimensional bottleneck, so its rank is "
                     "exactly " + str(D_LATENT) + " and not " +
                     str(D_MODEL) + ". MLA is MHA with a low-rank K/V "
                     "projection.",
        },
    }

    # ---- the unifying statement -----------------------------------------
    # Every variant is MHA with a rank-limited effective key projection.
    # Full rank is d_model. The cache is what that rank costs to store.
    ranks = [{"variant": k,
              "rank": variants[k]["cost"]["key_projection_rank"],
              "full_rank": D_MODEL,
              "cache_elements": variants[k]["cost"][
                  "cache_elements_per_token_per_layer"]}
             for k in ["MHA", "GQA", "MQA", "MLA"]]
    ranks_ok = (ranks[0]["rank"] == D_MODEL
                and ranks[1]["rank"] == 2 * D_HEAD
                and ranks[2]["rank"] == D_HEAD
                and ranks[3]["rank"] == D_LATENT)

    # all four still produce a full-width output of the right shape
    shape_ok = all(len(variants[k]["Y"]) == SEQ and len(variants[k]["Y"][0]) == D_MODEL
                   for k in variants)

    real = real_models(ref)
    cross = crosscheck_infer(root, real)

    # ---- MLA at real scale ----------------------------------------------
    lat = MLA_REF["kv_lora_rank"] + MLA_REF["qk_rope_head_dim"]
    mha_equiv = 2 * MLA_REF["n_heads"] * MLA_REF["v_head_dim"]
    deepseek = {
        "config": MLA_REF,
        "cache_elements_per_token_per_layer": lat,
        "mha_equivalent_elements": mha_equiv,
        "compression_ratio": mha_equiv / float(lat),
        "bytes_per_token": lat * MLA_REF["n_layers"] * DTYPE_BYTES,
        "mha_bytes_per_token": mha_equiv * MLA_REF["n_layers"] * DTYPE_BYTES,
        "reduction_percent": 100.0 * (1.0 - lat / float(mha_equiv)),
        "extra_flops_note": "Decompression is two extra matmuls per step. "
                            "Absorbing W_uk into the query removes one of "
                            "them; the other becomes part of the output "
                            "projection. The trade is real but it is FLOPs "
                            "for bytes, on hardware where decode is bound by "
                            "bytes.",
    }

    checks = [
        {"id": "validity_MHA", "what": "MHA is valid attention",
         "metric": variants["MHA"]["validity"]["row_sum_max_abs_err"],
         "tol": TOL, "passed": variants["MHA"]["validity"]["passed"]},
        {"id": "validity_GQA", "what": "GQA is valid attention",
         "metric": variants["GQA"]["validity"]["row_sum_max_abs_err"],
         "tol": TOL, "passed": variants["GQA"]["validity"]["passed"]},
        {"id": "validity_MQA", "what": "MQA is valid attention",
         "metric": variants["MQA"]["validity"]["row_sum_max_abs_err"],
         "tol": TOL, "passed": variants["MQA"]["validity"]["passed"]},
        {"id": "validity_MLA", "what": "MLA is valid attention",
         "metric": variants["MLA"]["validity"]["row_sum_max_abs_err"],
         "tol": TOL, "passed": variants["MLA"]["validity"]["passed"]},
        {"id": "output_shapes", "what": "all four return (seq, d_model)",
         "metric": 0.0, "tol": TOL, "passed": shape_ok},
    ]
    for e in equivalences:
        checks.append({"id": "equiv_" + e["variant"] + "_is_tied_MHA",
                       "what": e["variant"] + " == MHA with tied KV heads",
                       "metric": e["max_abs_err"], "tol": TOL,
                       "passed": e["passed"]})
    checks += [
        {"id": "mla_low_rank",
         "what": "rank(W_dkv W_uk) == d_c",
         "metric": float(abs(variants["MLA"]["factorisation"]["rank_Wk_eq"]
                             - D_LATENT)),
         "tol": TOL, "passed": variants["MLA"]["factorisation"]["passed"]},
        {"id": "mla_absorption",
         "what": "q.(c W_uk) == (q W_uk^T).c",
         "metric": absorb["max_abs_err"], "tol": TOL,
         "passed": absorb["passed"]},
        {"id": "rank_ladder",
         "what": "ranks are d_model / 2*d_head / d_head / d_c",
         "metric": 0.0, "tol": TOL, "passed": ranks_ok},
        {"id": "infer_json_agreement",
         "what": "KV bytes match every row already published in infer.json",
         "metric": cross["max_abs_err"], "tol": TOL, "passed": cross["passed"]},
    ]

    return {
        "meta": {
            "generated_by": "code/attention_variants.py",
            "d_model": D_MODEL, "n_heads": N_HEADS, "d_head": D_HEAD,
            "seq": SEQ, "d_latent": D_LATENT, "causal": CAUSAL,
            "tokens": TOKENS, "dtype_bytes": DTYPE_BYTES,
            "dtype": "bf16",
            "variant_order": ["MHA", "GQA", "MQA", "MLA"],
            "tol": TOL,
            "convention": "row-vector: X is (seq, d_in), W is (d_in, d_out), "
                          "Y = X W. Causal mask on, because this page is "
                          "about what decode has to keep.",
            "description": "MHA, MQA, GQA and MLA on one input. All four are "
                           "valid attention; they differ only in the rank of "
                           "the effective K/V projection and therefore in how "
                           "many bytes a token leaves in the cache.",
        },
        "input": {"X": rr(X), "tokens": TOKENS},
        "shared_weights": {"Wq": rr(W["Wq"]), "Wo": rr(W["Wo"])},
        "variants": variants,
        "equivalences": equivalences,
        "ranks": {"rows": ranks, "full_rank": D_MODEL, "passed": ranks_ok,
                  "claim": "Every variant is multi-head attention with a "
                           "rank-limited effective key projection. MHA is "
                           "full rank; GQA and MQA get there by repeating "
                           "blocks; MLA gets there by factorising. The cache "
                           "is what that rank costs to store — and MLA is the "
                           "only one that stores the bottleneck ITSELF rather "
                           "than two decompressed copies of it."},
        "cache_formula": {
            "toy": "2 * n_kv_heads * d_head elements per token per layer "
                   "(MLA: d_c, once, for K and V together)",
            "real": "2 * n_layers * seq * n_kv_heads * d_head * dtype_bytes "
                    "per sequence",
            "why": "K and V for token j depend only on token j, so once "
                   "computed they never change. Decode re-reads the WHOLE "
                   "cache every step, so its size is a bandwidth cost paid "
                   "per generated token, not a one-off.",
        },
        "real": {"models": real, "_source": ref["_source"],
                 "default_model": ref["default_model"],
                 "seqs": [r["seq"] for r in real[0]["rows"]] if real else []},
        "crosscheck_infer": cross,
        "deepseek": deepseek,
        "literature": LITERATURE,
        "checks": checks,
        "passed": all(c["passed"] for c in checks),
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    outdir = os.path.join(root, "assets", "data")
    os.makedirs(outdir, exist_ok=True)

    d = build(root)
    payload = json.dumps(d, indent=1, allow_nan=False)
    with open(os.path.join(outdir, "attnvar.json"), "w") as f:
        f.write(payload)
    with open(os.path.join(outdir, "attnvar.js"), "w") as f:
        f.write("// GENERATED by code/attention_variants.py -- do not hand-edit.\n")
        f.write("window.ATTNVAR = " + payload + ";\n")

    m = d["meta"]
    print("=" * 78)
    print("attention_variants.py — MHA / MQA / GQA / MLA on one input")
    print("=" * 78)
    print(f"  d_model {m['d_model']}  query heads {m['n_heads']}  "
          f"d_head {m['d_head']}  seq {m['seq']}  d_c {m['d_latent']}  "
          f"causal {m['causal']}")
    print()
    print(f"  {'variant':8s} {'kv heads':>8s} {'q/kv':>5s} {'rank(Wk)':>9s}"
          f" {'cache el':>9s} {'cache B':>8s} {'params':>7s}  head -> kv")
    for k in m["variant_order"]:
        c = d["variants"][k]["cost"]
        print(f"  {k:8s} {c['n_kv_heads']:8d} {c['queries_per_kv_head']:5d}"
              f" {c['key_projection_rank']:9d}"
              f" {c['cache_elements_per_token_per_layer']:9d}"
              f" {c['cache_bytes_per_token_per_layer']:8d}"
              f" {c['params']:7d}  {c['head_map']}")
    print()

    print("  VALIDITY  (row sums, mask exactness, convex hull)")
    for k in m["variant_order"]:
        v = d["variants"][k]["validity"]
        print(f"    {k:5s} |sum p - 1| {v['row_sum_max_abs_err']:.3e}"
              f"   min p {v['min_probability']:.3e}"
              f"   masked {v['masked_max_abs']:.3e}"
              f"   hull violations {v['convex_hull_violations']}"
              f"   {'ok' if v['passed'] else 'FAIL'}")
    print()

    print("  EQUIVALENCE TO MHA")
    for e in d["equivalences"]:
        print(f"    {e['variant']:5s} vs MHA with tied KV  max|Δ| "
              f"{e['max_abs_err']:.3e}   {'ok' if e['passed'] else 'FAIL'}")
    f = d["variants"]["MLA"]["factorisation"]
    print(f"    MLA   rank(W_dkv W_uk) = {f['rank_Wk_eq']} "
          f"(d_c = {f['d_latent']}, full rank would be {m['d_model']})")
    print(f"    MLA   absorption q.(cW) = (qW^T).c  max|Δ| "
          f"{d['variants']['MLA']['absorption']['max_abs_err']:.3e}")
    print()

    print("  REAL MODELS  (bf16, one sequence)")
    for r in d["real"]["models"]:
        print(f"    {r['name']:16s} {r['n_heads']:4d} q heads / "
              f"{r['n_kv_heads']:2d} kv heads  d_head {r['d_head']:4d}"
              f"  layers {r['n_layers']:3d}"
              f"  -> {r['kv_reduction_vs_mha']:.0f}x KV reduction")
        for row in r["rows"]:
            print(f"        seq {row['seq']:7d}   "
                  f"MHA {row['mha_bytes'] / 1e9:8.2f} GB   "
                  f"GQA {row['gqa_bytes'] / 1e9:8.2f} GB   "
                  f"MQA {row['mqa_bytes'] / 1e9:8.3f} GB")
    print()

    x = d["crosscheck_infer"]
    print(f"  CROSS-CHECK vs assets/data/infer.json: {x['checked']} rows, "
          f"max|Δ| {x['max_abs_err']:.0f} bytes  "
          f"{'ok' if x['passed'] else 'FAIL'}")
    ds = d["deepseek"]
    print(f"  MLA at scale ({ds['config']['name']}): "
          f"{ds['cache_elements_per_token_per_layer']} el/token/layer vs "
          f"{ds['mha_equivalent_elements']} for MHA  = "
          f"{ds['compression_ratio']:.1f}x ({ds['reduction_percent']:.1f}% less)")
    print()

    print("  CHECKS")
    for c in d["checks"]:
        print(f"    {'PASS' if c['passed'] else 'FAIL'}  {c['id']:30s}"
              f" {c['metric']:.3e} < {c['tol']:.0e}   {c['what']}")
    print()
    print(f"  {'PASS' if d['passed'] else 'FAIL'}: "
          f"{sum(1 for c in d['checks'] if c['passed'])}/{len(d['checks'])} "
          f"checks — all four are valid attention; GQA/MQA/MLA are MHA with a "
          f"rank-limited K/V projection")
    print()
    print(f"  wrote {os.path.join(outdir, 'attnvar.js')}")
    print("=" * 78)
    if not d["passed"]:
        raise SystemExit("FAILED: " + ", ".join(c["id"] for c in d["checks"]
                                                if not c["passed"]))


if __name__ == "__main__":
    main()
