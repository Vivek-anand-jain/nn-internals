#!/usr/bin/env python3
"""
transformer_2layer.py — a 2-layer transformer, forward AND backward, by hand.

Pure Python standard library. No numpy, no torch, no autograd.

The existing transformer_block.py covers ONE block, forward only. This file
does the thing that actually teaches you how a transformer trains:

  * two full pre-LN blocks stacked, so the residual stream is visible
    carrying gradient between them
  * every backward step derived by hand, including the two that people
    reliably get wrong:
        - LayerNorm backward, which is NOT just g/sigma. The mean and the
          variance both depend on every element of x, so the gradient picks
          up two correction terms.
        - the softmax Jacobian, which is NOT diagonal. dL/ds_i depends on
          every p_j.
  * every gradient checked against central finite differences
  * how every weight matrix is PARTITIONED under TP / PP / DP / ZeRO, down
    to which GPU owns which slice of which matrix

Emits:
    assets/data/tf2.js     (window.TF2 = {...})
    assets/data/tf2.json

Run:  python3 code/transformer_2layer.py
"""

import json
import math
import os

# ============================================================================
# SHAPES
# ============================================================================
# Deliberately tiny so every matrix fits on screen, but structurally real:
# multi-head attention with a causal mask, pre-LN placement, a 2x MLP, and
# two stacked blocks so the residual stream has somewhere to go.
#
# d_model 4 with 2 heads means d_head 2 -- and 4 divides by a world size of
# 2, so tensor parallelism splits it evenly and visibly.

D_MODEL = 4
N_HEADS = 2
D_HEAD = D_MODEL // N_HEADS      # 2
SEQ = 3
D_FF = 8                          # 2x expansion (real models use 4x or 3.5x)
N_LAYERS = 2
EPS = 1e-5
WORLD = 2                         # for the partitioning tables

TOKENS = ["the", "cat", "sat"]


def det(i, j, salt):
    """Deterministic small weights. Reproducible without seeding."""
    return round(((((i * 7 + j * 13 + salt * 11 + 3) % 19) - 9) / 30.0), 4)


# ============================================================================
# Linear algebra, row-vector convention
# ============================================================================
# X is (seq, d_in), W is (d_in, d_out), Y = X @ W.  This is the Megatron
# convention: W's COLUMNS are output features, so "column-parallel" really
# does mean splitting W's columns. (PyTorch's nn.Linear stores the transpose;
# page 12 covers that naming trap.)

def mm(A, B):
    m, k, n = len(A), len(B), len(B[0])
    return [[sum(A[i][t] * B[t][j] for t in range(k)) for j in range(n)]
            for i in range(m)]


def T_(A):
    return [list(r) for r in zip(*A)]


def add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def sub(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def had(A, B):
    return [[A[i][j] * B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def smul(A, s):
    return [[v * s for v in r] for r in A]


def addvec(A, v):
    """Broadcast a bias row across every row."""
    return [[A[i][j] + v[j] for j in range(len(v))] for i in range(len(A))]


def zeros(m, n):
    return [[0.0] * n for _ in range(m)]


def colsum(A):
    """Sum over rows -> the bias gradient."""
    return [sum(A[i][j] for i in range(len(A))) for j in range(len(A[0]))]


def flat(A):
    return [v for r in A for v in r]


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
# GELU (tanh approximation -- what real transformers actually use)
# ============================================================================

C_GELU = math.sqrt(2.0 / math.pi)


def gelu(x):
    inner = C_GELU * (x + 0.044715 * x ** 3)
    return 0.5 * x * (1.0 + math.tanh(inner))


def dgelu(x):
    """d/dx of the tanh-approximation GELU, derived by the product and
    chain rules rather than quoted."""
    inner = C_GELU * (x + 0.044715 * x ** 3)
    t = math.tanh(inner)
    dinner = C_GELU * (1.0 + 3 * 0.044715 * x ** 2)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dinner


# ============================================================================
# LAYERNORM  -- forward and the backward everyone gets wrong
# ============================================================================

def layernorm_fwd(X, gain, bias, eps=EPS):
    """
    Per ROW (per token), independently:
        mu    = mean(x)
        var   = mean((x - mu)^2)          <- biased variance, as in PyTorch
        xhat  = (x - mu) / sqrt(var + eps)
        y     = gain * xhat + bias
    """
    rows = []
    out = []
    for x in X:
        n = len(x)
        mu = sum(x) / n
        var = sum((v - mu) ** 2 for v in x) / n
        inv = 1.0 / math.sqrt(var + eps)
        xhat = [(v - mu) * inv for v in x]
        y = [gain[j] * xhat[j] + bias[j] for j in range(n)]
        rows.append({"mu": mu, "var": var, "inv_std": inv, "xhat": xhat,
                     "x": list(x), "y": y})
        out.append(y)
    return out, rows


def layernorm_bwd(dY, rows, gain):
    """
    THE DERIVATION.

    y_j = g_j * xhat_j + b_j,   xhat_j = (x_j - mu) / sqrt(var + eps)

    The naive answer is dL/dx_j = dL/dy_j * g_j / sigma. That is WRONG,
    because mu and var are themselves functions of EVERY x, so changing one
    x_j moves the normalisation for every other element in the row too.

    Writing dxhat_j = dL/dy_j * g_j and carrying both dependencies through:

        dL/dx_j = (1/sigma) * [ dxhat_j
                                - mean_k(dxhat_k)
                                - xhat_j * mean_k(dxhat_k * xhat_k) ]

    The second term is the mean's contribution; the third is the variance's.
    Both vanish only if the incoming gradient happens to be zero-mean and
    uncorrelated with xhat, which it generally is not.

    Note what this means for MEMORY: backward needs xhat and inv_std for
    every token, so LayerNorm saves them from forward.
    """
    dgain = [0.0] * len(gain)
    dbias = [0.0] * len(gain)
    dX = []
    detail = []
    for i, r in enumerate(rows):
        n = len(r["x"])
        xhat = r["xhat"]
        dy = dY[i]
        # parameter gradients accumulate over tokens
        for j in range(n):
            dgain[j] += dy[j] * xhat[j]
            dbias[j] += dy[j]
        dxhat = [dy[j] * gain[j] for j in range(n)]
        mean_dxhat = sum(dxhat) / n
        mean_dxhat_xhat = sum(dxhat[j] * xhat[j] for j in range(n)) / n
        dx = [r["inv_std"] * (dxhat[j] - mean_dxhat - xhat[j] * mean_dxhat_xhat)
              for j in range(n)]
        dX.append(dx)
        detail.append({
            "token": i, "dxhat": dxhat,
            "mean_dxhat": mean_dxhat,
            "mean_dxhat_xhat": mean_dxhat_xhat,
            "inv_std": r["inv_std"], "xhat": xhat,
            "naive_dx": [r["inv_std"] * dxhat[j] for j in range(n)],
            "dx": dx,
        })
    return dX, dgain, dbias, detail


# ============================================================================
# SOFTMAX  -- and its non-diagonal Jacobian
# ============================================================================

def softmax_row(s):
    """Numerically stable: subtract the row max before exponentiating."""
    m = max(s)
    e = [math.exp(v - m) for v in s]
    z = sum(e)
    return [v / z for v in e]


def softmax_bwd_row(dp, p):
    """
    p = softmax(s).  The Jacobian is

        dp_i/ds_j = p_i (delta_ij - p_j)

    which is DENSE -- every output depends on every input, because the
    normaliser couples them. Contracting dL/dp with it:

        dL/ds_i = p_i * ( dL/dp_i - sum_j dL/dp_j p_j )

    The subtracted term is the same scalar for every i. That scalar is what
    makes the gradient sum to zero across the row, which is the algebraic
    shadow of the fact that softmax outputs always sum to one.
    """
    dot = sum(dp[j] * p[j] for j in range(len(p)))
    return [p[i] * (dp[i] - dot) for i in range(len(p))], dot


def causal_mask(seq):
    """Position i may attend to j only if j <= i."""
    return [[(0.0 if j <= i else float("-inf")) for j in range(seq)]
            for i in range(seq)]


# ============================================================================
# PARAMETERS
# ============================================================================

def init_params():
    P = {}
    for L in range(N_LAYERS):
        s = L * 3
        P[f"L{L}.ln1.g"] = [1.0 + det(0, j, s) for j in range(D_MODEL)]
        P[f"L{L}.ln1.b"] = [det(1, j, s) for j in range(D_MODEL)]
        for nm, salt in (("Wq", 1), ("Wk", 2), ("Wv", 3), ("Wo", 4)):
            P[f"L{L}.{nm}"] = [[det(i, j, s + salt) for j in range(D_MODEL)]
                               for i in range(D_MODEL)]
        P[f"L{L}.ln2.g"] = [1.0 + det(2, j, s) for j in range(D_MODEL)]
        P[f"L{L}.ln2.b"] = [det(3, j, s) for j in range(D_MODEL)]
        P[f"L{L}.W1"] = [[det(i, j, s + 5) for j in range(D_FF)]
                         for i in range(D_MODEL)]
        P[f"L{L}.W2"] = [[det(i, j, s + 6) for j in range(D_MODEL)]
                         for i in range(D_FF)]
    return P


X_IN = [[round(((t * 5 + d * 3) % 7 + 1) / 8.0, 4) for d in range(D_MODEL)]
        for t in range(SEQ)]
TARGET = [[round(((t * 3 + d * 5) % 5 + 1) / 10.0, 4) for d in range(D_MODEL)]
          for t in range(SEQ)]


# ============================================================================
# FORWARD
# ============================================================================

def forward(P, record=True):
    """Pre-LN transformer:  h = h + Attn(LN1(h));  h = h + MLP(LN2(h))"""
    cache = {"layers": []}
    h = [list(r) for r in X_IN]
    cache["x_in"] = [list(r) for r in h]

    for L in range(N_LAYERS):
        lc = {"layer": L, "stream_in": [list(r) for r in h]}

        # ---- LayerNorm 1 -------------------------------------------------
        n1, r1 = layernorm_fwd(h, P[f"L{L}.ln1.g"], P[f"L{L}.ln1.b"])
        lc["ln1"] = {"out": n1, "rows": r1}

        # ---- Q, K, V -----------------------------------------------------
        Q = mm(n1, P[f"L{L}.Wq"])
        K = mm(n1, P[f"L{L}.Wk"])
        V = mm(n1, P[f"L{L}.Wv"])
        lc["qkv"] = {"Q": Q, "K": K, "V": V, "input": n1}

        # ---- per-head attention -----------------------------------------
        scale = 1.0 / math.sqrt(D_HEAD)
        mask = causal_mask(SEQ)
        heads = []
        O = zeros(SEQ, D_MODEL)
        for hd in range(N_HEADS):
            lo, hi = hd * D_HEAD, (hd + 1) * D_HEAD
            Qh = [row[lo:hi] for row in Q]
            Kh = [row[lo:hi] for row in K]
            Vh = [row[lo:hi] for row in V]
            raw = mm(Qh, T_(Kh))
            scaled = smul(raw, scale)
            masked = [[scaled[i][j] + mask[i][j] for j in range(SEQ)]
                      for i in range(SEQ)]
            Pr = [softmax_row(masked[i]) for i in range(SEQ)]
            Oh = mm(Pr, Vh)
            for i in range(SEQ):
                for j in range(D_HEAD):
                    O[i][lo + j] = Oh[i][j]
            heads.append({"head": hd, "cols": [lo, hi], "Q": Qh, "K": Kh,
                          "V": Vh, "raw": raw, "scaled": scaled,
                          "masked": [[None if masked[i][j] == float("-inf")
                                      else masked[i][j] for j in range(SEQ)]
                                     for i in range(SEQ)],
                          "P": Pr, "O": Oh,
                          "row_sums": [round(sum(Pr[i]), 12) for i in range(SEQ)]})
        lc["heads"] = heads
        lc["concat"] = [list(r) for r in O]
        lc["scale"] = scale

        # ---- output projection + residual --------------------------------
        A = mm(O, P[f"L{L}.Wo"])
        h1 = add(h, A)
        lc["attn_out"] = A
        lc["resid1"] = [list(r) for r in h1]

        # ---- LayerNorm 2 + MLP + residual --------------------------------
        n2, r2 = layernorm_fwd(h1, P[f"L{L}.ln2.g"], P[f"L{L}.ln2.b"])
        U = mm(n2, P[f"L{L}.W1"])
        G = [[gelu(v) for v in row] for row in U]
        D = mm(G, P[f"L{L}.W2"])
        h2 = add(h1, D)
        lc["ln2"] = {"out": n2, "rows": r2}
        lc["mlp"] = {"input": n2, "U": U, "G": G, "D": D}
        lc["resid2"] = [list(r) for r in h2]

        h = h2
        cache["layers"].append(lc)

    cache["out"] = [list(r) for r in h]
    return h, cache


def loss_fn(out):
    """Mean squared error against a fixed target, over every element."""
    n = SEQ * D_MODEL
    diff = sub(out, TARGET)
    L = sum(v * v for v in flat(diff)) / n
    dOut = smul(diff, 2.0 / n)
    return L, dOut


# ============================================================================
# BACKWARD  -- every step by hand
# ============================================================================

def backward(P, cache, dOut):
    g = {}
    steps = []
    dh = [list(r) for r in dOut]

    for L in range(N_LAYERS - 1, -1, -1):
        lc = cache["layers"][L]

        # ---- residual 2 --------------------------------------------------
        # h2 = h1 + D  ->  the SAME gradient goes to both branches. That is
        # the entire reason residual connections fix vanishing gradients:
        # the identity path carries dL/dh through untouched.
        dD = [list(r) for r in dh]
        d_skip2 = [list(r) for r in dh]
        steps.append({"layer": L, "id": "resid2", "label": "residual 2 split",
                      "note": "h2 = h1 + D. Addition sends the gradient down "
                              "BOTH paths unchanged -- the identity branch is "
                              "a gradient superhighway, which is why very deep "
                              "stacks train at all.",
                      "value": dD})

        # ---- MLP ---------------------------------------------------------
        Gm = lc["mlp"]["G"]
        n2 = lc["mlp"]["input"]
        U = lc["mlp"]["U"]
        g[f"L{L}.W2"] = mm(T_(Gm), dD)
        dG = mm(dD, T_(P[f"L{L}.W2"]))
        dU = [[dG[i][j] * dgelu(U[i][j]) for j in range(D_FF)]
              for i in range(SEQ)]
        g[f"L{L}.W1"] = mm(T_(n2), dU)
        dn2 = mm(dU, T_(P[f"L{L}.W1"]))
        steps.append({"layer": L, "id": "mlp", "label": "MLP backward",
                      "note": "dW2 = G^T dD needs the saved GELU output G; "
                              "dW1 = n2^T dU needs the saved LayerNorm output. "
                              "Same outer-product structure as any linear "
                              "layer, so the same activations must be kept.",
                      "value": dn2})

        # ---- LayerNorm 2 -------------------------------------------------
        dh1_from_ln2, dg2, db2, det2 = layernorm_bwd(dn2, lc["ln2"]["rows"],
                                                     P[f"L{L}.ln2.g"])
        g[f"L{L}.ln2.g"] = dg2
        g[f"L{L}.ln2.b"] = db2
        dh1 = add(d_skip2, dh1_from_ln2)
        steps.append({"layer": L, "id": "ln2", "label": "LayerNorm 2 backward",
                      "note": "Two correction terms, not just g/sigma. The "
                              "mean and the variance each depend on every "
                              "element of the row.",
                      "value": dh1_from_ln2, "ln_detail": det2})

        # ---- residual 1 --------------------------------------------------
        dA = [list(r) for r in dh1]
        d_skip1 = [list(r) for r in dh1]

        # ---- output projection -------------------------------------------
        O = lc["concat"]
        g[f"L{L}.Wo"] = mm(T_(O), dA)
        dO = mm(dA, T_(P[f"L{L}.Wo"]))
        steps.append({"layer": L, "id": "wo", "label": "output projection backward",
                      "note": "dWo = O^T dA, where O is the concatenated "
                              "head outputs saved in forward.",
                      "value": dO})

        # ---- per-head attention backward ---------------------------------
        scale = lc["scale"]
        dQ = zeros(SEQ, D_MODEL)
        dK = zeros(SEQ, D_MODEL)
        dV = zeros(SEQ, D_MODEL)
        head_details = []
        for hd in range(N_HEADS):
            H = lc["heads"][hd]
            lo, hi = H["cols"]
            dOh = [row[lo:hi] for row in dO]
            Pr, Vh, Qh, Kh = H["P"], H["V"], H["Q"], H["K"]

            # O = P @ V
            dPr = mm(dOh, T_(Vh))
            dVh = mm(T_(Pr), dOh)

            # P = softmax(masked scores) -- row by row, dense Jacobian
            dS = []
            dots = []
            for i in range(SEQ):
                row, dot = softmax_bwd_row(dPr[i], Pr[i])
                dS.append(row)
                dots.append(dot)
            # the mask contributed -inf, whose gradient is zero: masked
            # positions already have p = 0 so their dS is 0 automatically.

            # S = (Qh Kh^T) * scale
            dRaw = smul(dS, scale)
            dQh = mm(dRaw, Kh)
            dKh = mm(T_(dRaw), Qh)

            for i in range(SEQ):
                for j in range(D_HEAD):
                    dQ[i][lo + j] += dQh[i][j]
                    dK[i][lo + j] += dKh[i][j]
                    dV[i][lo + j] += dVh[i][j]
            head_details.append({
                "head": hd, "dP": dPr, "dS": dS, "softmax_dots": dots,
                "dQ": dQh, "dK": dKh, "dV": dVh,
                "row_sums_dS": [round(sum(r), 12) for r in dS],
            })

        # `value` carries dQ for the stepper's headline, but dK and dV are
        # equally part of this step's output. An earlier version exposed
        # only dQ, so a consumer trusting `value` saw a third of the
        # attention input-gradient; page 16 caught that. All three are now
        # here at full d_model width, already reassembled from the heads.
        steps.append({"layer": L, "id": "attn", "label": "attention backward",
                      "note": "Softmax's Jacobian is dense: dL/ds_i = "
                              "p_i (dL/dp_i - sum_j dL/dp_j p_j). The "
                              "subtracted scalar is why every row of dS sums "
                              "to zero.",
                      "value": dQ,
                      "dQ": dQ, "dK": dK, "dV": dV,
                      "value_is": "dQ; see dQ/dK/dV for all three at full "
                                  "d_model width",
                      "heads": head_details})

        # ---- Q/K/V projections -------------------------------------------
        n1 = lc["qkv"]["input"]
        g[f"L{L}.Wq"] = mm(T_(n1), dQ)
        g[f"L{L}.Wk"] = mm(T_(n1), dK)
        g[f"L{L}.Wv"] = mm(T_(n1), dV)
        dn1 = add(add(mm(dQ, T_(P[f"L{L}.Wq"])), mm(dK, T_(P[f"L{L}.Wk"]))),
                  mm(dV, T_(P[f"L{L}.Wv"])))
        steps.append({"layer": L, "id": "qkv", "label": "Q/K/V backward",
                      "note": "All three read the SAME LayerNorm output, so "
                              "their input-gradients ADD. One saved tensor, "
                              "three consumers.",
                      "value": dn1})

        # ---- LayerNorm 1 -------------------------------------------------
        dh_from_ln1, dg1, db1, det1 = layernorm_bwd(dn1, lc["ln1"]["rows"],
                                                    P[f"L{L}.ln1.g"])
        g[f"L{L}.ln1.g"] = dg1
        g[f"L{L}.ln1.b"] = db1
        dh = add(d_skip1, dh_from_ln1)
        steps.append({"layer": L, "id": "ln1", "label": "LayerNorm 1 backward",
                      "note": "And the residual adds its untouched copy back "
                              "in, so the gradient leaving this block is "
                              "never smaller than the one that entered by the "
                              "skip path alone.",
                      "value": dh, "ln_detail": det1})

    return g, steps, dh


# ============================================================================
# GRADIENT CHECK
# ============================================================================

def numerical_grad(P, key, h=1e-5):
    """Central differences on one parameter tensor."""
    val = P[key]
    is_mat = isinstance(val[0], list)
    out = []
    if is_mat:
        for i in range(len(val)):
            row = []
            for j in range(len(val[0])):
                orig = val[i][j]
                val[i][j] = orig + h
                lp = loss_fn(forward(P)[0])[0]
                val[i][j] = orig - h
                lm = loss_fn(forward(P)[0])[0]
                val[i][j] = orig
                row.append((lp - lm) / (2 * h))
            out.append(row)
    else:
        for i in range(len(val)):
            orig = val[i]
            val[i] = orig + h
            lp = loss_fn(forward(P)[0])[0]
            val[i] = orig - h
            lm = loss_fn(forward(P)[0])[0]
            val[i] = orig
            out.append((lp - lm) / (2 * h))
    return out


# ============================================================================
# PARTITIONING  -- how each matrix is split under each strategy
# ============================================================================

def partitioning(P):
    """
    For every parameter tensor, how each strategy divides it.

    TP follows Megatron: Wq/Wk/Wv and W1 are COLUMN-parallel (split the
    output dimension, so each GPU owns whole attention HEADS and whole FFN
    neurons), Wo and W2 are ROW-parallel (split the input dimension, which
    is exactly the layout the previous column-parallel layer produced).
    LayerNorm is NOT split -- it needs the whole hidden vector to compute a
    mean, which is what sequence parallelism later addresses.
    """
    N = WORLD
    entries = []
    for L in range(N_LAYERS):
        def shape_of(k):
            v = P[k]
            return [len(v), len(v[0])] if isinstance(v[0], list) else [len(v)]

        specs = [
            (f"L{L}.ln1.g", "LayerNorm 1 gain", "layernorm", "replicated",
             "needs the whole row to take a mean"),
            (f"L{L}.ln1.b", "LayerNorm 1 bias", "layernorm", "replicated",
             "needs the whole row to take a mean"),
            (f"L{L}.Wq", "W_q", "attention", "column",
             "split by output feature = whole heads per GPU"),
            (f"L{L}.Wk", "W_k", "attention", "column",
             "split by output feature = whole heads per GPU"),
            (f"L{L}.Wv", "W_v", "attention", "column",
             "split by output feature = whole heads per GPU"),
            (f"L{L}.Wo", "W_o", "attention", "row",
             "split by INPUT feature, matching the head split above; "
             "produces partial sums that must be all-reduced"),
            (f"L{L}.ln2.g", "LayerNorm 2 gain", "layernorm", "replicated",
             "needs the whole row to take a mean"),
            (f"L{L}.ln2.b", "LayerNorm 2 bias", "layernorm", "replicated",
             "needs the whole row to take a mean"),
            (f"L{L}.W1", "W_up (MLP in)", "mlp", "column",
             "split by output feature = whole FFN neurons per GPU"),
            (f"L{L}.W2", "W_down (MLP out)", "mlp", "row",
             "split by INPUT feature, matching the neuron split; all-reduce"),
        ]
        for key, disp, group, tp_mode, why in specs:
            sh = shape_of(key)
            n_el = sh[0] * (sh[1] if len(sh) > 1 else 1)
            if tp_mode == "column" and len(sh) > 1:
                per = [sh[0], sh[1] // N]
                slices = [{"gpu": r, "cols": [r * (sh[1] // N),
                                              (r + 1) * (sh[1] // N)]}
                          for r in range(N)]
            elif tp_mode == "row" and len(sh) > 1:
                per = [sh[0] // N, sh[1]]
                slices = [{"gpu": r, "rows": [r * (sh[0] // N),
                                              (r + 1) * (sh[0] // N)]}
                          for r in range(N)]
            else:
                per = list(sh)
                slices = [{"gpu": r, "full": True} for r in range(N)]

            entries.append({
                "key": key, "display": disp, "layer": L, "group": group,
                "shape": sh, "elements": n_el,
                "tp": {"mode": tp_mode, "per_gpu_shape": per,
                       "per_gpu_elements": per[0] * (per[1] if len(per) > 1 else 1),
                       "slices": slices, "why": why},
                "pp": {"stage": L,
                       "why": f"layer {L} lives entirely on pipeline stage {L}"},
                "dp": {"mode": "replicated",
                       "why": "every data-parallel rank holds a full copy"},
                "zero3": {"mode": "flat-sharded",
                          # NOTE: this is a PER-TENSOR ceiling and is NOT how
                          # ZeRO-3 actually shards. Kept for comparison only;
                          # `flat_sharding` below is the real model. Page 17
                          # caught that these coincide here only because every
                          # tensor happens to have an even element count.
                          "per_tensor_ceil": -(-n_el // N),
                          "why": "see flat_sharding -- ZeRO-3 flattens ALL "
                                 "parameters into one buffer and cuts THAT, "
                                 "so shard boundaries ignore matrix edges"},
            })
    return entries


def flat_sharding(entries, worlds=(2, 3, 4, 6, 8)):
    """
    The real ZeRO-3 model: concatenate every parameter into one flat buffer
    in declaration order, then cut the BUFFER into N equal shards.

    Boundaries land wherever the arithmetic puts them, which is the whole
    point -- a shard routinely spans the end of one weight matrix and the
    start of an unrelated one. That is the concrete difference from tensor
    parallelism, which respects what each matrix MEANS.

    (Page 17 pointed out that at world 2 the single boundary happens to land
    exactly on the layer-0/layer-1 seam, so the straddling is invisible at
    that size. Several world sizes are emitted so the effect is visible from
    the data rather than needing a caveat.)
    """
    layout, off = [], 0
    for e in entries:
        layout.append({"key": e["key"], "display": e["display"],
                       "layer": e["layer"], "group": e["group"],
                       "shape": e["shape"], "elements": e["elements"],
                       "start": off, "end": off + e["elements"]})
        off += e["elements"]
    total = off

    by_world = {}
    for N in worlds:
        size = -(-total // N)                     # ceil, then pad
        padded = size * N
        shards = []
        for r in range(N):
            lo, hi = r * size, min((r + 1) * size, total)
            spans = [t for t in layout if t["start"] < hi and t["end"] > lo]
            # does this shard START partway through a tensor?
            straddles = [t["display"] + " (L" + str(t["layer"]) + ")"
                         for t in spans
                         if t["start"] < lo or t["end"] > hi]
            shards.append({
                "gpu": r, "range": [lo, hi], "elements": max(0, hi - lo),
                "spans_tensors": [t["key"] for t in spans],
                "n_tensors_spanned": len(spans),
                "cut_through": straddles,
            })
        cut_boundaries = sum(
            1 for r in range(1, N)
            if any(t["start"] < r * size < t["end"] for t in layout))
        by_world[str(N)] = {
            "world": N, "shard_size": size, "padded_total": padded,
            "padding": padded - total, "shards": shards,
            "boundaries": N - 1,
            "boundaries_cutting_a_matrix": cut_boundaries,
            "clean_seam": cut_boundaries == 0,
        }

    return {"total_elements": total, "layout": layout, "by_world": by_world,
            "note": "ZeRO-3 shards the flat buffer, not the matrices. Shard "
                    "boundaries land on arithmetic, not meaning -- which is "
                    "why ZeRO-3 must all-gather a WHOLE tensor before it can "
                    "be used, and therefore never reduces the peak size of "
                    "any single layer. Tensor parallelism does the opposite: "
                    "it cuts along axes the maths understands, so a shard is "
                    "independently usable. They are complementary."}


def build():
    P = init_params()
    out, cache = forward(P)
    L, dOut = loss_fn(out)
    grads, steps, dx_in = backward(P, cache, dOut)

    # ---- gradient check on every parameter -----------------------------
    checks = []
    worst = 0.0
    for key in sorted(P.keys()):
        num = numerical_grad(P, key)
        err = maxdiff(grads[key], num)
        worst = max(worst, err)
        checks.append({"param": key, "max_abs_err": err,
                       "elements": len(flat(P[key])) if isinstance(P[key][0], list)
                                   else len(P[key])})

    n_params = sum(len(flat(v)) if isinstance(v[0], list) else len(v)
                   for v in P.values())
    _part = partitioning(P)

    # activation census
    act = []
    for L_ in range(N_LAYERS):
        lc = cache["layers"][L_]
        act += [
            {"layer": L_, "name": "LN1 xhat", "elements": SEQ * D_MODEL,
             "why": "LayerNorm backward needs it"},
            {"layer": L_, "name": "Q,K,V", "elements": 3 * SEQ * D_MODEL,
             "why": "attention backward needs K for dQ and Q for dK"},
            {"layer": L_, "name": "attention probs P",
             "elements": N_HEADS * SEQ * SEQ,
             "why": "THE QUADRATIC TERM: heads x seq^2"},
            {"layer": L_, "name": "concat O", "elements": SEQ * D_MODEL,
             "why": "dWo = O^T dA"},
            {"layer": L_, "name": "LN2 xhat", "elements": SEQ * D_MODEL,
             "why": "LayerNorm backward needs it"},
            {"layer": L_, "name": "MLP pre-act U", "elements": SEQ * D_FF,
             "why": "GELU backward needs the pre-activation"},
            {"layer": L_, "name": "MLP act G", "elements": SEQ * D_FF,
             "why": "dW2 = G^T dD"},
        ]

    return {
        "meta": {
            "generated_by": "code/transformer_2layer.py",
            "d_model": D_MODEL, "n_heads": N_HEADS, "d_head": D_HEAD,
            "seq": SEQ, "d_ff": D_FF, "n_layers": N_LAYERS, "eps": EPS,
            "world": WORLD, "tokens": TOKENS,
            "architecture": "pre-LN, causal, GELU MLP, 2 stacked blocks",
            "convention": "row-vector: X is (seq, d_in), W is (d_in, d_out), "
                          "Y = X W. Megatron's convention, so column-parallel "
                          "really does split W's columns.",
            "n_params": n_params,
        },
        "input": {"x": X_IN, "target": TARGET, "tokens": TOKENS},
        "params": {k: P[k] for k in sorted(P)},
        "forward": cache,
        "loss": L,
        "dOut": dOut,
        "grads": {k: grads[k] for k in sorted(grads)},
        "backward_steps": steps,
        "dx_in": dx_in,
        "gradcheck": {"per_param": checks, "max_abs_err": worst,
                      "passed": worst < 1e-6, "h": 1e-5,
                      "note": "central differences; the residual is the "
                              "finite-difference side, not the analytic one"},
        "partitioning": _part,
        "flat_sharding": flat_sharding(_part),
        "activations": act,
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    outdir = os.path.join(root, "assets", "data")
    os.makedirs(outdir, exist_ok=True)

    d = build()
    payload = json.dumps(d, indent=1, allow_nan=False)
    with open(os.path.join(outdir, "tf2.json"), "w") as f:
        f.write(payload)
    with open(os.path.join(outdir, "tf2.js"), "w") as f:
        f.write("// GENERATED by code/transformer_2layer.py -- do not hand-edit.\n")
        f.write("window.TF2 = " + payload + ";\n")

    m = d["meta"]
    print("=" * 72)
    print("transformer_2layer.py — 2-layer transformer, forward + backward")
    print("=" * 72)
    print(f"  d_model {m['d_model']}  heads {m['n_heads']}  d_head {m['d_head']}"
          f"  seq {m['seq']}  d_ff {m['d_ff']}  layers {m['n_layers']}")
    print(f"  {m['n_params']} parameters   loss = {d['loss']:.9f}")
    print()
    gc = d["gradcheck"]
    bad = [c for c in gc["per_param"] if c["max_abs_err"] > 1e-6]
    for c in gc["per_param"]:
        print(f"    {c['param']:14s} {c['elements']:3d} el   "
              f"max|Δ| {c['max_abs_err']:.3e}"
              f"   {'ok' if c['max_abs_err'] < 1e-6 else 'FAIL'}")
    print()
    print(f"  worst over all {len(gc['per_param'])} tensors: "
          f"{gc['max_abs_err']:.3e}  -> {'PASS' if gc['passed'] else 'FAIL'}")
    print()
    # softmax sanity
    rs = d["forward"]["layers"][0]["heads"][0]["row_sums"]
    print(f"  softmax rows sum to: {rs}")
    print(f"  backward steps recorded: {len(d['backward_steps'])}")
    print(f"  partitioning entries:    {len(d['partitioning'])}")
    print()
    print(f"  wrote {os.path.join(outdir, 'tf2.js')}")
    print("=" * 72)
    if not gc["passed"]:
        raise SystemExit("GRADIENT CHECK FAILED for: "
                         + ", ".join(c["param"] for c in bad))


if __name__ == "__main__":
    main()
