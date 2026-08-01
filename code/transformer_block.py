#!/usr/bin/env python3
"""
transformer_block.py — ground truth for page 09, "The Transformer Block".

Same discipline as code/ground_truth.py: pure Python standard library, no
numpy, no torch, no autograd. Every matmul is written out longhand so the
page can render the individual numbers instead of asserting them.

The main trace (assets/data/trace.js) describes a 2 -> 3 -> 1 MLP. Page 09
needs a transformer, which the MLP trace does not contain, so this file
generates it. Nothing on page 09 may hand-type a number; everything it
renders comes out of the payload emitted here.

It computes ONE forward pass of a real (if tiny) pre-norm transformer block:

    d_model = 4, n_heads = 2, d_head = 2, seq_len = 3, d_ff = 8

    h  = x  + Attn(LN1(x))
    y  = h  + MLP(LN2(h))

and emits:

    assets/data/transformer.js    (window.TFORMER = {...}  <- works over file://)
    assets/data/transformer.json  (same payload, for tooling)

Run:   python3 code/transformer_block.py
"""

import json
import math
import os

# ============================================================================
# SHAPES
# ============================================================================
# Deliberately the smallest configuration that is still a REAL transformer
# block: more than one head, more than one token, a causal mask with a
# non-trivial upper triangle, and an MLP that actually expands.

D_MODEL = 4
N_HEADS = 2
D_HEAD = D_MODEL // N_HEADS      # 2
SEQ = 3
D_FF = 8                          # 2x expansion (a real block uses 4x; see below)
EPS = 1e-5                        # LayerNorm epsilon, matching torch's default


# ============================================================================
# THE INPUT
# ============================================================================
# Three token embeddings, 4 dimensions each. Small integers and halves so the
# projections land on readable numbers. Row t is the residual stream at
# position t on the way IN to the block.

X = [
    [1.0,  0.0,  1.0,  0.0],   # token 0
    [0.0,  2.0,  0.0,  0.0],   # token 1
    [1.0,  1.0,  0.0,  2.0],   # token 2
]

TOKENS = ["the", "cat", "sat"]


# ============================================================================
# THE WEIGHTS
# ============================================================================
# Hand-picked, not random. Halves and ones keep every product exact in binary
# floating point, so the only irrational numbers on the page come from the
# two square roots (LayerNorm's std and attention's 1/sqrt(d_head)) and from
# exp() inside softmax. That is on purpose: it makes it obvious which steps
# are "just arithmetic" and which ones genuinely change the number system.
#
# Convention: weight matrices are [d_in][d_out] and applied as
#     out[j] = sum_i  row[i] * W[i][j] + b[j]
# i.e. row-vector-times-matrix, which is what nn.Linear stores transposed.

# ---- LayerNorm 1 (before attention) ----------------------------------------
LN1_GAIN = [1.2, 1.0, 0.8, 1.0]
LN1_BIAS = [0.0, 0.1, -0.1, 0.0]

# ---- Q / K / V projections (d_model -> d_model, split across 2 heads) -------
# Columns 0..1 belong to head 0, columns 2..3 belong to head 1. There is no
# separate "per-head matrix" in a real implementation either -- one matmul
# produces all heads and a reshape splits them.
W_Q = [[0.5, 0.0, 0.0, 0.5],
       [0.0, 0.5, 0.5, 0.0],
       [0.5, 0.0, 0.5, 0.0],
       [0.0, 0.5, 0.0, 0.5]]
B_Q = [0.0, 0.0, 0.0, 0.0]

W_K = [[0.5, 0.0, 0.5, 0.0],
       [0.0, 0.5, 0.0, 0.5],
       [0.0, 0.5, 0.5, 0.0],
       [0.5, 0.0, 0.0, 0.5]]
B_K = [0.0, 0.0, 0.0, 0.0]

W_V = [[1.0, 0.0, 0.5, 0.0],
       [0.0, 1.0, 0.0, 0.5],
       [0.5, 0.0, 1.0, 0.0],
       [0.0, 0.5, 0.0, 1.0]]
B_V = [0.0, 0.0, 0.0, 0.0]

# ---- Output projection (concat of heads -> d_model) ------------------------
W_O = [[0.5, 0.0, 0.0, 0.5],
       [0.0, 0.5, 0.5, 0.0],
       [0.5, 0.5, 0.0, 0.0],
       [0.0, 0.0, 0.5, 0.5]]
B_O = [0.0, 0.0, 0.0, 0.0]

# ---- LayerNorm 2 (before the MLP) ------------------------------------------
LN2_GAIN = [1.0, 1.1, 1.0, 0.9]
LN2_BIAS = [0.1, 0.0, -0.1, 0.0]

# ---- MLP: d_model -> d_ff -> d_model ---------------------------------------
W_UP = [[0.5, -0.5,  0.5,  0.0,  1.0,  0.0, -0.5,  0.5],
        [0.0,  0.5,  0.5, -0.5,  0.0,  1.0,  0.5, -0.5],
        [-0.5, 0.0,  0.5,  0.5,  0.5, -0.5,  1.0,  0.0],
        [0.5,  0.5,  0.0,  0.5, -0.5,  0.5,  0.0,  1.0]]
B_UP = [0.1, -0.1, 0.0, 0.2, -0.2, 0.0, 0.1, -0.1]

W_DOWN = [[0.5,  0.0, -0.5,  0.0],
          [0.0,  0.5,  0.0, -0.5],
          [0.5,  0.5,  0.0,  0.0],
          [0.0,  0.0,  0.5,  0.5],
          [-0.5, 0.5,  0.0,  0.0],
          [0.0,  0.0,  0.5, -0.5],
          [0.5,  0.0,  0.0,  0.5],
          [0.0, -0.5,  0.5,  0.0]]
B_DOWN = [0.0, 0.1, 0.0, -0.1]


# ============================================================================
# LONGHAND LINEAR ALGEBRA
# ============================================================================

def linear(rows, W, b):
    """rows (n x d_in) @ W (d_in x d_out) + b  ->  n x d_out.

    Also returns `work`: every individual product, so the page can animate a
    projection the same way page 02 animates the MLP's matvec.
    """
    d_in, d_out = len(W), len(W[0])
    out, work = [], []
    for t, row in enumerate(rows):
        o, wrow = [], []
        for j in range(d_out):
            products = [row[i] * W[i][j] for i in range(d_in)]
            total = sum(products) + b[j]
            o.append(total)
            wrow.append({
                "t": t, "j": j,
                "products": [{"x": row[i], "w": W[i][j], "prod": products[i]}
                             for i in range(d_in)],
                "sum_of_products": sum(products),
                "bias": b[j],
                "out": total,
            })
        out.append(o)
        work.append(wrow)
    return out, work


def layernorm(rows, gain, bias, eps=EPS):
    """LayerNorm over the LAST axis, per token, spelled out.

    Returns per-row mean, biased variance, std, the centred vector, the
    normalised vector, and the affine output. mean and var are exactly the
    two scalars per row that the backward pass has to keep -- that is the
    memory point the page makes.
    """
    d = len(rows[0])
    out, detail = [], []
    for t, row in enumerate(rows):
        mean = sum(row) / d
        centred = [v - mean for v in row]
        var = sum(c * c for c in centred) / d          # biased (1/N), as in torch
        std = math.sqrt(var + eps)
        normed = [c / std for c in centred]
        y = [normed[i] * gain[i] + bias[i] for i in range(d)]
        out.append(y)
        detail.append({
            "t": t, "x": list(row),
            "mean": mean, "var": var, "eps": eps, "std": std,
            "centred": centred, "normed": normed, "out": y,
            "saved_for_backward": {"mean": mean, "rstd": 1.0 / std},
        })
    return out, detail


def gelu(x):
    """Exact GELU: x * Phi(x), Phi the standard normal CDF.

    Written with math.erf rather than the tanh approximation so the number
    on the page is the real one.
    """
    return x * 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def softmax(scores, mask):
    """Numerically-stable softmax over one row, honouring a boolean mask.

    mask[j] == 0 means position j is not attendable: it is dropped from the
    max, gets exp() == 0, and therefore probability exactly 0. That is
    identical to adding -inf before the exponential, without putting an
    Infinity into the JSON payload.
    """
    live = [scores[j] for j in range(len(scores)) if mask[j]]
    mx = max(live)
    exps = [math.exp(scores[j] - mx) if mask[j] else 0.0 for j in range(len(scores))]
    denom = sum(exps)
    probs = [e / denom for e in exps]
    return probs, {"max": mx, "exps": exps, "denom": denom}


def add(a, b):
    """Elementwise add of two (n x d) matrices -- the residual connection."""
    return [[a[t][i] + b[t][i] for i in range(len(a[0]))] for t in range(len(a))]


# ============================================================================
# THE BLOCK
# ============================================================================

def run_block():
    rec = {}

    # ---- 1. LayerNorm 1 -----------------------------------------------------
    ln1_out, ln1_detail = layernorm(X, LN1_GAIN, LN1_BIAS)
    rec["ln1"] = {
        "name": "LayerNorm 1 (pre-attention)",
        "gain": LN1_GAIN, "bias": LN1_BIAS, "eps": EPS,
        "input": [r[:] for r in X],
        "rows": ln1_detail,
        "out": ln1_out,
    }

    # ---- 2. Q / K / V projections ------------------------------------------
    q, q_work = linear(ln1_out, W_Q, B_Q)
    k, k_work = linear(ln1_out, W_K, B_K)
    v, v_work = linear(ln1_out, W_V, B_V)

    # ---- 3. Split into heads ------------------------------------------------
    def head_slice(m, h):
        lo, hi = h * D_HEAD, (h + 1) * D_HEAD
        return [row[lo:hi] for row in m]

    scale = 1.0 / math.sqrt(D_HEAD)

    # The causal mask. mask[i][j] == 1 iff query i may attend to key j,
    # i.e. j <= i. Identical for every head and every layer, which is why
    # frameworks build it once and cache it.
    mask = [[1 if j <= i else 0 for j in range(SEQ)] for i in range(SEQ)]

    heads = []
    softmax_devs = []
    for h in range(N_HEADS):
        qh, kh, vh = head_slice(q, h), head_slice(k, h), head_slice(v, h)

        raw, scaled, masked, probs, sm_work = [], [], [], [], []
        row_sums = []
        for i in range(SEQ):
            raw_i, scaled_i, masked_i = [], [], []
            for j in range(SEQ):
                dot = sum(qh[i][c] * kh[j][c] for c in range(D_HEAD))
                raw_i.append(dot)
                scaled_i.append(dot * scale)
                # null == "-inf" for the page to render; keeps the JSON valid
                masked_i.append(dot * scale if mask[i][j] else None)
            p, work = softmax(scaled_i, mask[i])
            s = sum(p)
            row_sums.append(s)
            softmax_devs.append(abs(s - 1.0))
            raw.append(raw_i)
            scaled.append(scaled_i)
            masked.append(masked_i)
            probs.append(p)
            sm_work.append(work)

        # weighted sum of value vectors
        out_h = []
        out_h_work = []
        for i in range(SEQ):
            o, w = [], []
            for c in range(D_HEAD):
                terms = [{"p": probs[i][j], "v": vh[j][c], "prod": probs[i][j] * vh[j][c]}
                         for j in range(SEQ)]
                o.append(sum(t["prod"] for t in terms))
                w.append({"i": i, "c": c, "terms": terms,
                          "out": sum(t["prod"] for t in terms)})
            out_h.append(o)
            out_h_work.append(w)

        heads.append({
            "h": h,
            "dims": [h * D_HEAD, h * D_HEAD + D_HEAD - 1],
            "q": qh, "k": kh, "v": vh,
            "scores_raw": raw,
            "scale": scale,
            "scores_scaled": scaled,
            "mask": mask,
            "scores_masked": masked,
            "softmax_work": sm_work,
            "probs": probs,
            "row_sums": row_sums,
            "out": out_h,
            "out_work": out_h_work,
        })

    max_dev = max(softmax_devs)

    # ---- 4. Concatenate heads, output-project -------------------------------
    concat = [sum([heads[h]["out"][i] for h in range(N_HEADS)], []) for i in range(SEQ)]
    attn_out, o_work = linear(concat, W_O, B_O)

    rec["attn"] = {
        "name": "Multi-head self-attention",
        "n_heads": N_HEADS, "d_head": D_HEAD,
        "scale": scale,
        "scale_formula": "1 / sqrt(d_head) = 1 / sqrt(%d)" % D_HEAD,
        "input": ln1_out,
        "W_q": W_Q, "b_q": B_Q, "W_k": W_K, "b_k": B_K, "W_v": W_V, "b_v": B_V,
        "W_o": W_O, "b_o": B_O,
        "q": q, "k": k, "v": v,
        "q_work": q_work, "k_work": k_work, "v_work": v_work,
        "mask": mask,
        "heads": heads,
        "concat": concat,
        "out": attn_out,
        "o_work": o_work,
        "softmax_check": {
            "row_sums": [h["row_sums"] for h in heads],
            "max_abs_deviation": max_dev,
            "tolerance": 1e-12,
            "passed": max_dev < 1e-12,
            "n_rows_checked": N_HEADS * SEQ,
        },
    }

    # ---- 5. Residual add 1 ---------------------------------------------------
    resid1 = add(X, attn_out)
    rec["resid1"] = {
        "name": "Residual add 1",
        "stream_in": [r[:] for r in X],
        "delta": attn_out,
        "out": resid1,
        "note": "h = x + Attn(LN1(x)). The stream is never overwritten, only "
                "added to -- which is why the gradient has an unobstructed "
                "path from the loss back to the embedding.",
    }

    # ---- 6. LayerNorm 2 -------------------------------------------------------
    ln2_out, ln2_detail = layernorm(resid1, LN2_GAIN, LN2_BIAS)
    rec["ln2"] = {
        "name": "LayerNorm 2 (pre-MLP)",
        "gain": LN2_GAIN, "bias": LN2_BIAS, "eps": EPS,
        "input": resid1,
        "rows": ln2_detail,
        "out": ln2_out,
    }

    # ---- 7. MLP sub-layer ------------------------------------------------------
    h_pre, up_work = linear(ln2_out, W_UP, B_UP)
    h_act = [[gelu(v) for v in row] for row in h_pre]
    gelu_detail = [[{"pre": h_pre[t][j],
                     "gate": 0.5 * (1.0 + math.erf(h_pre[t][j] / math.sqrt(2.0))),
                     "out": h_act[t][j]}
                    for j in range(D_FF)] for t in range(SEQ)]
    mlp_out, down_work = linear(h_act, W_DOWN, B_DOWN)

    rec["mlp"] = {
        "name": "MLP sub-layer",
        "d_ff": D_FF,
        "expansion": D_FF / D_MODEL,
        "nonlinearity": "GELU (exact, x*0.5*(1+erf(x/sqrt(2))))",
        "input": ln2_out,
        "W_up": W_UP, "b_up": B_UP, "W_down": W_DOWN, "b_down": B_DOWN,
        "h_pre": h_pre, "up_work": up_work,
        "h_act": h_act, "gelu_detail": gelu_detail,
        "out": mlp_out, "down_work": down_work,
    }

    # ---- 8. Residual add 2 -----------------------------------------------------
    block_out = add(resid1, mlp_out)
    rec["resid2"] = {
        "name": "Residual add 2",
        "stream_in": resid1,
        "delta": mlp_out,
        "out": block_out,
        "note": "y = h + MLP(LN2(h)). Same shape in, same shape out -- which "
                "is the whole reason you can stack 80 of these.",
    }
    rec["block_out"] = block_out

    return rec


# ============================================================================
# PARAMETER ACCOUNTING
# ============================================================================

def count_params():
    d, f, H = D_MODEL, D_FF, N_HEADS
    subs = [
        {"group": "layernorm", "name": "LN1 gain",  "shape": [d],    "count": d,
         "formula": "d_model"},
        {"group": "layernorm", "name": "LN1 bias",  "shape": [d],    "count": d,
         "formula": "d_model"},
        {"group": "attention", "name": "W_q",       "shape": [d, d], "count": d * d,
         "formula": "d_model^2"},
        {"group": "attention", "name": "b_q",       "shape": [d],    "count": d,
         "formula": "d_model"},
        {"group": "attention", "name": "W_k",       "shape": [d, d], "count": d * d,
         "formula": "d_model^2"},
        {"group": "attention", "name": "b_k",       "shape": [d],    "count": d,
         "formula": "d_model"},
        {"group": "attention", "name": "W_v",       "shape": [d, d], "count": d * d,
         "formula": "d_model^2"},
        {"group": "attention", "name": "b_v",       "shape": [d],    "count": d,
         "formula": "d_model"},
        {"group": "attention", "name": "W_o",       "shape": [d, d], "count": d * d,
         "formula": "d_model^2"},
        {"group": "attention", "name": "b_o",       "shape": [d],    "count": d,
         "formula": "d_model"},
        {"group": "layernorm", "name": "LN2 gain",  "shape": [d],    "count": d,
         "formula": "d_model"},
        {"group": "layernorm", "name": "LN2 bias",  "shape": [d],    "count": d,
         "formula": "d_model"},
        {"group": "mlp",       "name": "W_up",      "shape": [d, f], "count": d * f,
         "formula": "d_model * d_ff"},
        {"group": "mlp",       "name": "b_up",      "shape": [f],    "count": f,
         "formula": "d_ff"},
        {"group": "mlp",       "name": "W_down",    "shape": [f, d], "count": f * d,
         "formula": "d_ff * d_model"},
        {"group": "mlp",       "name": "b_down",    "shape": [d],    "count": d,
         "formula": "d_model"},
    ]
    total = sum(s["count"] for s in subs)
    by_group = {}
    for s in subs:
        by_group.setdefault(s["group"], 0)
        by_group[s["group"]] += s["count"]

    # What the SAME block would cost at the standard 4x MLP expansion, so the
    # page can quote the real ~2/3 share without hand-waving.
    f4 = 4 * d
    attn4 = 4 * d * d + 4 * d
    mlp4 = d * f4 + f4 + f4 * d + d
    ln4 = 4 * d
    tot4 = attn4 + mlp4 + ln4

    return {
        "per_submodule": subs,
        "by_group": by_group,
        "total": total,
        "mlp_share": by_group["mlp"] / total,
        "attn_share": by_group["attention"] / total,
        "ln_share": by_group["layernorm"] / total,
        "toy_expansion": D_FF / d,
        "if_4x_expansion": {
            "d_ff": f4,
            "attention": attn4, "mlp": mlp4, "layernorm": ln4, "total": tot4,
            "mlp_share": mlp4 / tot4,
            "note": "the toy uses %gx expansion so the d_ff grid fits on screen; "
                    "a standard block uses 4x" % (D_FF / d),
        },
        "general_formula": {
            "attention_weights": "4 * d^2",
            "attention_biases": "4 * d",
            "mlp_weights": "d*4d + 4d*d = 8 * d^2",
            "mlp_biases": "4d + d = 5 * d",
            "layernorm": "4 * d",
            "total_leading": "12 * d^2",
            "total_exact": "12*d^2 + 13*d",
        },
    }


# ============================================================================
# ACTIVATION ACCOUNTING (this block, batch = 1)
# ============================================================================

def count_activations():
    d, f, s, H = D_MODEL, D_FF, SEQ, N_HEADS
    per = [
        {"sub": "layernorm", "name": "LN1 mean, rstd", "shape": [2, s],
         "elements": 2 * s, "cls": "activation", "scales": "seq"},
        {"sub": "layernorm", "name": "LN1 out",        "shape": [s, d],
         "elements": s * d, "cls": "activation", "scales": "seq*d"},
        {"sub": "attention", "name": "Q",              "shape": [s, d],
         "elements": s * d, "cls": "activation", "scales": "seq*d"},
        {"sub": "attention", "name": "K",              "shape": [s, d],
         "elements": s * d, "cls": "activation", "scales": "seq*d"},
        {"sub": "attention", "name": "V",              "shape": [s, d],
         "elements": s * d, "cls": "activation", "scales": "seq*d"},
        {"sub": "attention", "name": "scores (scaled)", "shape": [H, s, s],
         "elements": H * s * s, "cls": "activation", "scales": "heads*seq^2"},
        {"sub": "attention", "name": "softmax probs",  "shape": [H, s, s],
         "elements": H * s * s, "cls": "activation", "scales": "heads*seq^2"},
        {"sub": "attention", "name": "concat heads",   "shape": [s, d],
         "elements": s * d, "cls": "activation", "scales": "seq*d"},
        {"sub": "attention", "name": "attn out",       "shape": [s, d],
         "elements": s * d, "cls": "activation", "scales": "seq*d"},
        {"sub": "residual",  "name": "stream after add 1", "shape": [s, d],
         "elements": s * d, "cls": "activation", "scales": "seq*d"},
        {"sub": "layernorm", "name": "LN2 mean, rstd", "shape": [2, s],
         "elements": 2 * s, "cls": "activation", "scales": "seq"},
        {"sub": "layernorm", "name": "LN2 out",        "shape": [s, d],
         "elements": s * d, "cls": "activation", "scales": "seq*d"},
        {"sub": "mlp",       "name": "h_pre (up-proj)", "shape": [s, f],
         "elements": s * f, "cls": "activation", "scales": "seq*d_ff"},
        {"sub": "mlp",       "name": "h_act (GELU)",   "shape": [s, f],
         "elements": s * f, "cls": "activation", "scales": "seq*d_ff"},
        {"sub": "mlp",       "name": "mlp out",        "shape": [s, d],
         "elements": s * d, "cls": "activation", "scales": "seq*d"},
        {"sub": "residual",  "name": "block out",      "shape": [s, d],
         "elements": s * d, "cls": "activation", "scales": "seq*d"},
    ]
    total = sum(p["elements"] for p in per)
    quad = sum(p["elements"] for p in per if p["scales"] == "heads*seq^2")
    by_sub = {}
    for p in per:
        by_sub.setdefault(p["sub"], 0)
        by_sub[p["sub"]] += p["elements"]
    return {
        "per_tensor": per,
        "by_submodule": by_sub,
        "total_elements": total,
        "score_matrix_elements": H * s * s,
        "score_matrix_formula": "n_heads * seq^2 = %d * %d^2 = %d" % (H, s, H * s * s),
        "quadratic_elements": quad,
        "quadratic_share": quad / total,
        "linear_elements": total - quad,
        "note": "batch = 1. Everything except the score matrices grows LINEARLY "
                "in seq; the score matrices grow with seq squared.",
    }


# ============================================================================
# REAL-SCALE PARAMETER ARITHMETIC
# ============================================================================
# The model DIMENSIONS are not invented here. They come from
# T.reference_configs in assets/data/trace.json, which is the one place the
# whole site keeps published architecture figures, so pages 02, 07 and 08
# cannot quote different numbers for the same model. This file reads that
# file and does the arithmetic on top of it.
#
# The only thing added here is the ARCHITECTURE FAMILY: reference_configs
# carries dimensions, not conventions, and you cannot count parameters
# without knowing whether the MLP has two matrices or three. Two families
# cover every model in the list, and which one applies is decided by whether
# the config declares n_kv_heads.

FAMILY_INFERENCE = {
    "is_inferred": True,
    "rule": "n_kv_heads present -> llama family; absent -> gpt family",
    "why": "reference_configs records DIMENSIONS, not conventions. Parameter "
           "counts are not derivable from dimensions alone: you cannot count "
           "an MLP without knowing whether it has two matrices or three, or "
           "count a norm without knowing whether it has a bias. So the family "
           "is INFERRED here, not declared in the data.",
    "caveat": "This is a heuristic, and it is load-bearing -- pick the wrong "
              "family and the count is wrong by tens of percent. It happens to "
              "be right for all five configs in the list, but a model that "
              "used GQA with a GELU MLP, or MHA with SwiGLU, would be "
              "misclassified. The evidence that it is right here is the "
              "reconciliation column: every config lands within a fraction of "
              "a percent of its published parameter count, which would not "
              "happen if the conventions were guessed wrong.",
}

FAMILIES = {
    "gpt": {
        "detect": "no n_kv_heads field in reference_configs",
        "attention": "multi-head attention, all heads have their own K and V",
        "mlp": "2 matrices (up, down), GELU, d_ff = 4 * d_model",
        "mlp_matrices": 2,
        "ff_mult": 4,
        "norm": "LayerNorm: gain AND bias, so 2 * d_model per norm",
        "norm_mult": 4,               # 2 norms x (gain + bias)
        "proj_bias": True,            # GPT-2/3 keep biases on every projection
        "pos_embedding": "learned absolute, seq * d_model parameters",
        "tied_embeddings": True,      # GPT-2 and GPT-3 tie input/output embeddings
        "applies_to": "GPT-2, GPT-3",
    },
    "llama": {
        "detect": "n_kv_heads present in reference_configs",
        "attention": "grouped-query attention: n_kv_heads < n_heads, so K and "
                     "V projections are narrower than Q and O",
        "mlp": "3 matrices (gate, up, down) because SwiGLU gates, d_ff given "
               "explicitly and NOT 4 * d_model",
        "mlp_matrices": 3,
        "ff_mult": None,              # taken from the config's d_ff
        "norm": "RMSNorm: gain only, no bias, so 1 * d_model per norm",
        "norm_mult": 2,               # 2 norms x gain
        "proj_bias": False,           # Llama drops every projection bias
        "pos_embedding": "RoPE — rotary, computed not learned, zero parameters",
        "tied_embeddings": False,     # Llama 3 8B/70B/405B keep a separate lm_head
        "applies_to": "Llama 3 8B / 70B / 405B",
    },
}


def load_reference_configs(root):
    """Read T.reference_configs out of the MLP trace.

    Hard failure if it is missing: silently falling back to a private copy of
    the dimensions is exactly the drift this file exists to prevent.
    """
    path = os.path.join(root, "assets", "data", "trace.json")
    with open(path) as fh:
        trace = json.load(fh)
    rc = trace.get("reference_configs")
    if not rc:
        raise SystemExit(
            "trace.json has no reference_configs. Run "
            "`python3 code/ground_truth.py` first -- page 09 takes its real "
            "model dimensions from there, not from a copy in this file.")
    return rc


def real_config_arithmetic(cfg):
    """Exact parameter count for one published config, term by term."""
    name = cfg["name"]
    d = cfg["d_model"]
    L = cfg["n_layers"]
    nh = cfg["n_heads"]
    V = cfg["vocab"]
    kv = cfg.get("n_kv_heads", nh)
    fam_key = "llama" if "n_kv_heads" in cfg else "gpt"
    fam = FAMILIES[fam_key]
    dh = d // nh
    f = cfg.get("d_ff", fam["ff_mult"] * d if fam["ff_mult"] else 4 * d)

    # ---- the back-of-envelope estimate the page teaches ---------------------
    naive_per_layer = 12 * d * d
    naive_total = naive_per_layer * L

    # ---- attention ----------------------------------------------------------
    q_proj = d * (nh * dh)
    k_proj = d * (kv * dh)
    v_proj = d * (kv * dh)
    o_proj = (nh * dh) * d
    attn_bias = (nh * dh + 2 * kv * dh + d) if fam["proj_bias"] else 0
    attn = q_proj + k_proj + v_proj + o_proj + attn_bias

    attn_terms = [
        {"name": "q_proj", "shape": [d, nh * dh], "count": q_proj,
         "formula": "d * (n_heads * d_head)"},
        {"name": "k_proj", "shape": [d, kv * dh], "count": k_proj,
         "formula": "d * (n_kv_heads * d_head)" + ("  <- GQA narrows this" if kv < nh else "")},
        {"name": "v_proj", "shape": [d, kv * dh], "count": v_proj,
         "formula": "d * (n_kv_heads * d_head)" + ("  <- GQA narrows this" if kv < nh else "")},
        {"name": "o_proj", "shape": [nh * dh, d], "count": o_proj,
         "formula": "(n_heads * d_head) * d"},
    ]
    if attn_bias:
        attn_terms.append({"name": "projection biases", "shape": [attn_bias],
                           "count": attn_bias, "formula": "one per output column"})

    # ---- MLP ----------------------------------------------------------------
    mlp_terms = []
    if fam["mlp_matrices"] == 3:
        for nm, note in (("gate_proj", "  <- SwiGLU's third matrix"),
                         ("up_proj", ""), ("down_proj", "")):
            shape = [f, d] if nm == "down_proj" else [d, f]
            mlp_terms.append({"name": nm, "shape": shape, "count": d * f,
                              "formula": "d * d_ff" + note})
    else:
        mlp_terms.append({"name": "up_proj", "shape": [d, f], "count": d * f,
                          "formula": "d * d_ff  (d_ff = 4d)"})
        mlp_terms.append({"name": "down_proj", "shape": [f, d], "count": f * d,
                          "formula": "d_ff * d"})
    mlp = sum(t["count"] for t in mlp_terms)
    mlp_bias = (f + d) if fam["proj_bias"] else 0
    if mlp_bias:
        mlp_terms.append({"name": "MLP biases", "shape": [mlp_bias],
                          "count": mlp_bias, "formula": "d_ff + d"})
        mlp += mlp_bias

    # ---- norms, embeddings ---------------------------------------------------
    norms = fam["norm_mult"] * d
    per_layer = attn + mlp + norms
    layers_total = per_layer * L

    embed = V * d
    pos_embed = cfg["seq"] * d if fam_key == "gpt" else 0
    lm_head = 0 if fam["tied_embeddings"] else V * d
    final_norm = (fam["norm_mult"] // 2) * d
    exact = layers_total + embed + pos_embed + lm_head + final_norm

    published = cfg["params"]

    omissions = [
        "Embeddings sit outside 12*d^2*L entirely: %s here (%.1f%% of the "
        "model)." % (_h(embed + pos_embed + lm_head),
                     (embed + pos_embed + lm_head) / exact * 100),
    ]
    if kv < nh:
        omissions.append(
            "Grouped-query attention: only %d of the %d heads carry their own "
            "K and V, so attention costs %.2f*d^2, not 4*d^2."
            % (kv, nh, attn / (d * d)))
    else:
        omissions.append(
            "Attention here really is close to 4*d^2 (%.3f*d^2) -- this model "
            "predates GQA." % (attn / (d * d)))
    if fam["mlp_matrices"] == 3:
        omissions.append(
            "SwiGLU uses THREE d x d_ff matrices, and d_ff = %.2f*d rather "
            "than 4*d, so the MLP costs %.2f*d^2, not 8*d^2."
            % (f / d, mlp / (d * d)))
    else:
        omissions.append(
            "The MLP really is close to 8*d^2 (%.3f*d^2): two matrices, "
            "d_ff = %.0f*d." % (mlp / (d * d), f / d))
    omissions.append("%s -- %d params per layer, not the 4*d the formula "
                     "assumes." % (fam["norm"], norms))
    omissions.append(
        "Input and output embeddings are %s."
        % ("TIED, so the unembedding is free" if fam["tied_embeddings"]
           else "NOT tied, so the unembedding costs another %s" % _h(V * d)))
    if pos_embed:
        omissions.append("Learned positional embeddings add %s; RoPE models "
                         "pay nothing here." % _h(pos_embed))

    return {
        "name": name,
        "family": fam_key,
        "family_spec": fam,
        "config": dict(cfg, d_head=dh, d_ff=f, n_kv_heads=kv),
        "naive": {
            "per_layer": naive_per_layer,
            "per_layer_formula": "12 * d^2 = 12 * %d^2" % d,
            "total": naive_total,
            "total_formula": "12 * d^2 * n_layers",
            "fraction_of_exact": naive_total / exact,
        },
        "exact": {
            "attn_terms": attn_terms,
            "mlp_terms": mlp_terms,
            "attn": attn, "attn_in_d2": attn / (d * d),
            "mlp": mlp, "mlp_in_d2": mlp / (d * d),
            "norms": norms,
            "per_layer": per_layer, "per_layer_in_d2": per_layer / (d * d),
            "layers_total": layers_total,
            "embed": embed, "pos_embed": pos_embed, "lm_head": lm_head,
            "final_norm": final_norm,
            "total": exact,
            "mlp_share_of_layer": mlp / per_layer,
            "attn_share_of_layer": attn / per_layer,
        },
        "published": published,
        "exact_vs_published": exact / published,
        "omissions": omissions,
    }


def _h(n):
    """Human-readable big integer, matching NN.count on the JS side."""
    if n < 1e3:
        return str(n)
    if n < 1e6:
        return "%.1fK" % (n / 1e3)
    if n < 1e9:
        return "%.1fM" % (n / 1e6)
    return "%.2fB" % (n / 1e9)


# ============================================================================
# ACTIVATION MEMORY MODEL (for the interactive calculator)
# ============================================================================
# Per-layer activation bytes for a standard (non-Flash, dropout-on) block in
# mixed precision, following Korthikanti et al. 2022, "Reducing Activation
# Recomputation in Large Transformer Models":
#
#     bytes/layer = s*b*h * 34  +  a * s^2 * b * 5
#
# The coefficients are per-tensor byte counts: 2 for an fp16 activation,
# 1 for a boolean dropout mask. The point of splitting them this way is that
# the FIRST term does not contain seq^2 and the SECOND one does -- and the
# second term does not contain d_model at all.

ACTMEM = {
    "source": "Korthikanti et al. 2022, 'Reducing Activation Recomputation in "
              "Large Transformer Models', Table 2 accounting.",
    "formula": "bytes per layer = s*b*h*34 + a*s^2*b*5",
    "formula_note": "The 34 and the 5 are the fp16 case. Each term below is "
                    "recorded as a TENSOR COUNT plus a dtype, so the page can "
                    "re-derive both coefficients for any activation dtype in "
                    "T.memory.dtype_bytes. At 2 bytes they come back to 34 "
                    "and 5, which is the published pair.",
    # `elems` counts tensors of the given shape; `dtype` says how wide each
    # element is. "act" means the activation dtype the reader selects;
    # "mask" is a 1-byte boolean; "fp32" is always 4 bytes.
    "linear_terms": [
        {"name": "LayerNorm inputs (2 per block)", "elems": 2, "dtype": "act",
         "sub": "layernorm", "why": "one per norm"},
        {"name": "Q/K/V matmul input",             "elems": 1, "dtype": "act",
         "sub": "attention", "why": "stored once for all three projections"},
        {"name": "Q and K, held for the QKᵀ backward", "elems": 2, "dtype": "act",
         "sub": "attention", "why": "two tensors"},
        {"name": "V, held for the AV backward",    "elems": 1, "dtype": "act",
         "sub": "attention", "why": ""},
        {"name": "AV output = output-projection input", "elems": 1, "dtype": "act",
         "sub": "attention", "why": "one tensor, not two"},
        {"name": "output-projection dropout mask", "elems": 1, "dtype": "mask",
         "sub": "attention", "why": "1 byte per element regardless of dtype"},
        {"name": "MLP first-linear input",         "elems": 1, "dtype": "act",
         "sub": "mlp", "why": ""},
        {"name": "GELU input (4h wide)",           "elems": 4, "dtype": "act",
         "sub": "mlp", "why": "4x expansion"},
        {"name": "MLP second-linear input (4h wide)", "elems": 4, "dtype": "act",
         "sub": "mlp", "why": "4x expansion"},
        {"name": "MLP dropout mask",               "elems": 1, "dtype": "mask",
         "sub": "mlp", "why": "1 byte per element regardless of dtype"},
    ],
    "quad_terms": [
        {"name": "softmax output",               "elems": 1, "dtype": "act",
         "sub": "attention", "why": "one a x seq x seq matrix per batch item"},
        {"name": "softmax dropout mask",         "elems": 1, "dtype": "mask",
         "sub": "attention", "why": "same shape, 1 byte"},
        {"name": "dropout output (the input to AV)", "elems": 1, "dtype": "act",
         "sub": "attention", "why": "same shape again"},
    ],
    "flash": {
        "quad_terms": [],
        "extra_terms": [
            {"name": "per-row softmax statistics (max, log-sum-exp)",
             "elems": 1, "dtype": "fp32", "scales": "a*s*b",
             "why": "kept in fp32 for stability; linear in seq, not quadratic"},
        ],
        "what": "FlashAttention never materialises the a*s^2 score matrix in "
                "HBM. It tiles the computation in SRAM and keeps only the "
                "per-row softmax statistics (max and log-sum-exp), which is "
                "4 bytes per (batch, head, query) -- linear in seq. The "
                "backward pass recomputes the tiles it needs.",
        "caveat": "This does not make attention free: the seq^2 FLOPs are "
                  "still there, and the KV cache at inference is still linear "
                  "in seq per layer. It removes the seq^2 MEMORY term only.",
    },
    "precision_note": "The score matrix is also the tensor most sensitive to "
                      "dtype. exp() of a score overflows fp16 well before it "
                      "overflows bf16 (see T.formats.fp16.max_finite vs "
                      "T.formats.bf16.max_finite), which is why softmax is "
                      "accumulated in fp32 even when everything around it is "
                      "16-bit, and why the row max is subtracted first.",
    "assumes": "1 B dropout masks, 4x MLP expansion, no tensor or sequence "
               "parallelism, no activation recomputation. Real stacks differ; "
               "the shape of the two terms does not.",
    # Presets are NOT listed here -- the page builds them from
    # T.reference_configs so the calculator and the parameter table quote the
    # same dimensions. These ranges are chosen so every reference config is
    # exactly reachable (note heads steps by 4, because GPT-2 has 12).
    "slider_ranges": {
        "batch":   {"min": 1, "max": 32, "step": 1, "default": 1},
        "seq":     {"choices": [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
                    "default_index": 4},
        "heads":   {"min": 4, "max": 128, "step": 4, "default": 64},
        "layers":  {"min": 1, "max": 128, "step": 1, "default": 80},
        "d_model": {"choices": [768, 1024, 2048, 4096, 5120, 8192, 12288, 16384],
                    "default_index": 5},
        "dtype_default": "bf16",
    },
}
def _coeff(terms, act_bytes):
    """Bytes per (s*b*h) or per (a*s^2*b), given an activation dtype width."""
    w = {"act": act_bytes, "mask": 1, "fp32": 4}
    return sum(t["elems"] * w[t["dtype"]] for t in terms)


ACTMEM["coeff_by_dtype"] = {}
for _dt, _bytes in (("fp32", 4), ("bf16", 2), ("fp16", 2), ("fp8", 1), ("int8", 1)):
    ACTMEM["coeff_by_dtype"][_dt] = {
        "act_bytes": _bytes,
        "linear": _coeff(ACTMEM["linear_terms"], _bytes),
        "quad": _coeff(ACTMEM["quad_terms"], _bytes),
        "flash_extra": _coeff(ACTMEM["flash"]["extra_terms"], _bytes),
    }
# The fp16 case is the published pair; keep the flat names for readability.
ACTMEM["linear_coeff"] = ACTMEM["coeff_by_dtype"]["fp16"]["linear"]
ACTMEM["quad_coeff"] = ACTMEM["coeff_by_dtype"]["fp16"]["quad"]

# Anti-drift: the term table above IS the formula. If someone edits a term,
# the fp16 coefficients must still add up to the published 34 / 5, or the
# page would render a breakdown that does not match its own total.
assert ACTMEM["linear_coeff"] == 34, ACTMEM["linear_coeff"]
assert ACTMEM["quad_coeff"] == 5, ACTMEM["quad_coeff"]


# ============================================================================
# THE BRIDGE BACK TO THE MLP  (page 01-06's model)
# ============================================================================
# Page 09 opens by claiming the transformer introduces no new tensor CLASS.
# These are the numbers that claim rests on; the MLP side is re-derived from
# the same layout ground_truth.py uses so the two pages cannot drift.

MLP_MODEL = {
    "architecture": "2 -> 3 (ReLU) -> 1",
    "params": 3 * 2 + 3 + 1 * 3 + 1,
    "tensor_classes": ["weight", "gradient", "activation", "optimizer"],
    "n_classes": 4,
}


# ============================================================================
# BUILD + EMIT
# ============================================================================

def build(refcfg):
    rec = run_block()
    params = count_params()
    acts = count_activations()

    models = [real_config_arithmetic(c) for c in refcfg["models"]]
    default = refcfg.get("default_model")
    real = {
        "_source": refcfg.get("_source"),
        "family_inference": FAMILY_INFERENCE,
        "families": FAMILIES,
        "models": models,
        "default_model": default,
        "default_index": next((i for i, m in enumerate(models)
                               if m["name"] == default), 0),
    }

    return {
        "meta": {
            "generated_by": "code/transformer_block.py",
            "description": "One forward pass of a tiny but real pre-norm "
                           "transformer block. Do not hand-edit.",
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "d_head": D_HEAD,
            "seq_len": SEQ,
            "d_ff": D_FF,
            "eps": EPS,
            "tokens": TOKENS,
            "norm_placement": "pre-norm (LN before each sub-layer, residual "
                              "added after) -- what every modern LLM uses",
            "architecture": "y = h + MLP(LN2(h)),  h = x + Attn(LN1(x))",
            "nonlinearity": "GELU",
            "mask": "causal (a query at position i sees keys j <= i)",
        },
        "bridge": MLP_MODEL,
        "input": {"x": [r[:] for r in X], "tokens": TOKENS,
                  "shape": [SEQ, D_MODEL]},
        "weights": {
            "ln1": {"gain": LN1_GAIN, "bias": LN1_BIAS},
            "W_q": W_Q, "b_q": B_Q, "W_k": W_K, "b_k": B_K,
            "W_v": W_V, "b_v": B_V, "W_o": W_O, "b_o": B_O,
            "ln2": {"gain": LN2_GAIN, "bias": LN2_BIAS},
            "W_up": W_UP, "b_up": B_UP, "W_down": W_DOWN, "b_down": B_DOWN,
        },
        "ln1": rec["ln1"],
        "attn": rec["attn"],
        "resid1": rec["resid1"],
        "ln2": rec["ln2"],
        "mlp": rec["mlp"],
        "resid2": rec["resid2"],
        "block_out": rec["block_out"],
        "params": params,
        "activations": acts,
        "real": real,
        "actmem": ACTMEM,
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    outdir = os.path.join(root, "assets", "data")
    os.makedirs(outdir, exist_ok=True)

    refcfg = load_reference_configs(root)
    T = build(refcfg)

    # allow_nan=False so an accidental inf/nan is a hard error rather than
    # something that silently breaks the page.
    payload = json.dumps(T, indent=1, allow_nan=False)

    with open(os.path.join(outdir, "transformer.json"), "w") as f:
        f.write(payload)

    with open(os.path.join(outdir, "transformer.js"), "w") as f:
        f.write("// GENERATED by code/transformer_block.py -- do not hand-edit.\n")
        f.write("window.TFORMER = ")
        f.write(payload)
        f.write(";\n")

    # ---- Report --------------------------------------------------------------
    def m(rows, dg=4):
        return "\n".join("      [" + "  ".join(f"{v:>8.{dg}f}" for v in r) + "]"
                         for r in rows)

    p, a, real, chk = T["params"], T["activations"], T["real"], T["attn"]["softmax_check"]

    print("=" * 72)
    print("nn-internals -- transformer block ground truth")
    print("=" * 72)
    print(f"  d_model={D_MODEL}  n_heads={N_HEADS}  d_head={D_HEAD}  "
          f"seq={SEQ}  d_ff={D_FF}")
    print()
    print("  x (residual stream in)")
    print(m(X))
    print()
    print("  LN1 per token:")
    for r in T["ln1"]["rows"]:
        print(f"      t={r['t']}  mean={r['mean']:+.6f}  var={r['var']:.6f}  "
              f"std={r['std']:.6f}")
    print("  LN1 out")
    print(m(T["ln1"]["out"]))
    print()
    print(f"  scale = 1/sqrt(d_head) = {T['attn']['scale']:.6f}")
    for h in T["attn"]["heads"]:
        print(f"\n  --- head {h['h']}  (dims {h['dims'][0]}..{h['dims'][1]}) ---")
        print("  Q"); print(m(h["q"]))
        print("  K"); print(m(h["k"]))
        print("  V"); print(m(h["v"]))
        print("  scaled scores (upper triangle masked to -inf)")
        for i in range(SEQ):
            cells = []
            for j in range(SEQ):
                cells.append("    -inf" if h["mask"][i][j] == 0
                             else f"{h['scores_scaled'][i][j]:>8.4f}")
            print("      [" + "  ".join(cells) + "]")
        print("  softmax probabilities   (row sum)")
        for i in range(SEQ):
            row = "  ".join(f"{v:>8.4f}" for v in h["probs"][i])
            print(f"      [{row}]   sum = {h['row_sums'][i]:.12f}")
        print("  head out"); print(m(h["out"]))

    print()
    print(f"  softmax check: {chk['n_rows_checked']} rows, "
          f"max |sum - 1| = {chk['max_abs_deviation']:.3e}  -> "
          f"{'PASS' if chk['passed'] else 'FAIL'}")
    print()
    print("  concat heads"); print(m(T["attn"]["concat"]))
    print("  attn out (after W_o)"); print(m(T["attn"]["out"]))
    print("  residual stream after add 1"); print(m(T["resid1"]["out"]))
    print()
    print("  LN2 per token:")
    for r in T["ln2"]["rows"]:
        print(f"      t={r['t']}  mean={r['mean']:+.6f}  var={r['var']:.6f}  "
              f"std={r['std']:.6f}")
    print("  MLP h_pre (up-proj, d_ff=%d)" % D_FF); print(m(T["mlp"]["h_pre"], 3))
    print("  MLP h_act (GELU)");                     print(m(T["mlp"]["h_act"], 3))
    print("  MLP out (down-proj)");                  print(m(T["mlp"]["out"]))
    print("  BLOCK OUT");                            print(m(T["block_out"]))
    print()
    print("  parameters")
    for s in p["per_submodule"]:
        print(f"      {s['name']:<10s} {str(s['shape']):<10s} "
              f"{s['count']:>5d}   {s['formula']}")
    print(f"      {'-'*44}")
    for g in ("attention", "mlp", "layernorm"):
        print(f"      {g:<10s} {p['by_group'][g]:>16d}  "
              f"({p['by_group'][g]/p['total']*100:.1f}%)")
    print(f"      {'TOTAL':<10s} {p['total']:>16d}")
    print(f"      at the standard 4x expansion the same block would be "
          f"{p['if_4x_expansion']['total']} params, "
          f"MLP share {p['if_4x_expansion']['mlp_share']*100:.1f}%")
    print()
    print(f"  activation elements (batch=1): {a['total_elements']}")
    print(f"      score matrices: {a['score_matrix_formula']}  "
          f"= {a['quadratic_share']*100:.1f}% of all activation elements")
    print()
    print("  real configs (dimensions read from trace.json reference_configs)")
    print(f"      {'model':<14s} {'fam':<6s} {'naive 12d^2L':>14s} "
          f"{'exact':>16s} {'published':>14s}  {'exact/pub':>9s} {'naive/exact':>11s}")
    for mm in real["models"]:
        print(f"      {mm['name']:<14s} {mm['family']:<6s} "
              f"{mm['naive']['total']:>14,} {mm['exact']['total']:>16,} "
              f"{int(mm['published']):>14,}  "
              f"{mm['exact_vs_published']*100:>8.2f}% "
              f"{mm['naive']['fraction_of_exact']*100:>10.1f}%")
    for mm in real["models"]:
        print(f"      {mm['name']:<14s} attn {mm['exact']['attn_in_d2']:>5.2f}*d^2  "
              f"mlp {mm['exact']['mlp_in_d2']:>5.2f}*d^2  "
              f"layer {mm['exact']['per_layer_in_d2']:>5.2f}*d^2  "
              f"(textbook says 4 / 8 / 12)")
    print()
    print(f"  activation memory: {ACTMEM['formula']}")
    for dt, c in ACTMEM["coeff_by_dtype"].items():
        print(f"      {dt:<5s} act={c['act_bytes']} B  ->  linear {c['linear']:>3d}"
              f"  quad {c['quad']:>3d}  flash extra {c['flash_extra']}")
    print()
    print(f"  wrote {os.path.join(outdir, 'transformer.js')}")
    print(f"  wrote {os.path.join(outdir, 'transformer.json')}")
    print("=" * 72)

    if not chk["passed"]:
        raise SystemExit("SOFTMAX CHECK FAILED -- a probability row does not sum to 1.")


if __name__ == "__main__":
    main()
