#!/usr/bin/env python3
"""
positional.py — why attention needs position, and how RoPE puts it back.

Pure Python standard library. No numpy, no torch.

Attention has a structural hole in it. The mechanism
    O = softmax(Q K^T / sqrt(d)) V
is built entirely out of dot products between token vectors, and a dot
product does not know where its operands came from. Permute the input rows
and every score matrix entry moves with them, so the output rows permute
too and nothing else changes. Attention is PERMUTATION-EQUIVARIANT: to it,
"the cat sat" and "sat the cat" are the same bag.

This file proves that numerically first, and only then introduces the three
answers people have actually shipped:

  * learned absolute embeddings — one trained vector per position. Simple,
    and it hits a wall at the trained length because position L_max has no
    row in the table. It also encodes no relative structure at all, which
    we measure rather than assert.
  * sinusoidal — fixed sin/cos bands. The raw encodings genuinely do have a
    relative-offset property (their dot product is a function of m - n
    exactly), but that property does NOT survive being added to the token
    embedding and pushed through W_q and W_k. We measure that too, because
    it is the exact gap RoPE closes.
  * RoPE — rotate each 2D pair of q and k by an angle proportional to the
    position. Then
            <R_m q, R_n k>  =  <q, R_{n-m} k>
    which depends only on the offset n - m. Two lines of numerics prove it,
    and that property is the whole reason RoPE won.

Emits:
    assets/data/rope.js     (window.ROPE = {...})
    assets/data/rope.json

Run:  python3 code/positional.py
"""

import json
import math
import os

# ============================================================================
# SHAPES
# ============================================================================
# Small enough that every matrix fits on screen, structurally real otherwise.
# d_head 8 gives FOUR rotation pairs, which is the minimum needed to see the
# frequency spectrum fan out (pair 0 spins fast, pair 3 barely moves).

D_MODEL = 8
N_HEADS = 1                 # position is the subject here; one head is clearer
D_HEAD = D_MODEL // N_HEADS  # 8  ->  4 rotation pairs
SEQ = 4
N_PAIRS = D_HEAD // 2

TOKENS = ["the", "cat", "sat", "down"]

# The permutation used for the equivariance proof. Deliberately not the
# identity and not a simple reversal.
PERM = [2, 0, 3, 1]

# For the learned-absolute-embedding demonstration: a model "trained" on
# sequences up to this length, then asked about longer ones.
L_TRAINED = 6
L_ASKED = 9

# RoPE bases. 10000 is the original paper's choice and what GPT-NeoX/Llama 2
# ship; 500000 is Llama 3's. Both are quoted, not derived — see LITERATURE.
BASE_DEFAULT = 10000.0
BASE_LLAMA3 = 500000.0

TOL = 1e-12


def det(i, j, salt):
    """Deterministic small values. Reproducible without seeding."""
    return round(((((i * 7 + j * 13 + salt * 11 + 3) % 19) - 9) / 30.0), 4)


# ============================================================================
# Linear algebra, row-vector convention
# ============================================================================
# X is (seq, d_in), W is (d_in, d_out), Y = X @ W — same convention as
# transformer_2layer.py so the two files compose.

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
    """Round for the payload. Keeps the JSON readable without lying: every
    PASS/FAIL decision below is taken on the unrounded value."""
    return round(x, 9)


def rr(A):
    return [[r6(v) for v in row] for row in A]


def softmax_row(s):
    """Numerically stable: subtract the row max before exponentiating."""
    m = max(s)
    e = [math.exp(v - m) for v in s]
    z = sum(e)
    return [v / z for v in e]


# ============================================================================
# PLAIN ATTENTION  —  no position information anywhere in it
# ============================================================================
# Bidirectional on purpose. A causal mask would itself break permutation
# symmetry (row i can only see j <= i, and that "i" is an index, i.e. a
# position), so masking would hide the very hole we are trying to expose.

def attention(X, Wq, Wk, Wv, rope_base=None, positions=None, rope_v=False):
    """
    Q = X Wq, K = X Wk, V = X Wv;  O = softmax(Q K^T / sqrt(d)) V.

    If `rope_base` is given, each row of Q and K is rotated by its position
    BEFORE the dot product (and V too, but only if rope_v is set — that
    switch exists so we can measure what goes wrong when you rotate V).
    """
    Q, K, V = mm(X, Wq), mm(X, Wk), mm(X, Wv)
    n = len(X)
    if rope_base is not None:
        pos = positions if positions is not None else list(range(n))
        th = thetas(len(Q[0]), rope_base)
        Q = [rope_apply(Q[i], pos[i], th) for i in range(n)]
        K = [rope_apply(K[i], pos[i], th) for i in range(n)]
        if rope_v:
            V = [rope_apply(V[i], pos[i], th) for i in range(n)]
    scale = 1.0 / math.sqrt(len(Q[0]))
    scores = [[dot(Q[i], K[j]) * scale for j in range(n)] for i in range(n)]
    probs = [softmax_row(r) for r in scores]
    O = mm(probs, V)
    return {"Q": Q, "K": K, "V": V, "scores": scores, "probs": probs, "O": O}


# ============================================================================
# ROPE
# ============================================================================

def thetas(d, base):
    """
    theta_i = base ** (-2i/d)  for i = 0 .. d/2 - 1.

    Equivalently theta_i = 1 / base^(2i/d): a geometric sweep from 1 rad per
    token down to 1/base rad per token. Pair 0 turns a full circle every
    ~6.28 tokens; the last pair takes ~2*pi*base tokens. That spread is the
    design: the fast pairs resolve "which of my immediate neighbours", the
    slow pairs resolve "roughly how far away", and together they are a
    positional fingerprint that never repeats inside a realistic context.
    """
    return [base ** (-2.0 * i / d) for i in range(d // 2)]


def rope_apply(v, pos, th):
    """
    Rotate consecutive 2D pairs (v[2i], v[2i+1]) by angle pos * theta_i.

        [x']   [cos a  -sin a] [x]
        [y'] = [sin a   cos a] [y]

    This is a block-diagonal orthogonal matrix R_pos: d/2 independent planar
    rotations. Two consequences fall straight out of orthogonality and are
    used later:
        * |R q| = |q|. RoPE changes no norms, so it cannot rescale logits.
        * R_m^T R_n = R_{n-m}. That single identity is the relative-position
          property, and everything below is a numerical restatement of it.

    (Real kernels rotate (v[i], v[i + d/2]) instead — the "split halves"
    layout. That is the same operation under a fixed permutation of the
    channel axis; pairing adjacent dims keeps the arithmetic legible here.)
    """
    out = list(v)
    for i, t in enumerate(th):
        a = pos * t
        c, s = math.cos(a), math.sin(a)
        x, y = v[2 * i], v[2 * i + 1]
        out[2 * i] = x * c - y * s
        out[2 * i + 1] = x * s + y * c
    return out


def rope_dot(q, k, m, n, th):
    """<R_m q, R_n k>. Written out rather than composed so the proof below
    is measuring the actual arithmetic a kernel performs."""
    return dot(rope_apply(q, m, th), rope_apply(k, n, th))


def rope_dot_closed_form(q, k, delta, th):
    """
    The closed form of <R_m q, R_n k>, derived rather than quoted.

    Write the i-th pair of q as (a, b) and of k as (c, e), and put
    A = m t_i, B = n t_i. Expanding the two rotations and multiplying out:

        (a cosA - b sinA)(c cosB - e sinB)
      + (a sinA + b cosA)(c sinB + e cosB)

      = ac (cosA cosB + sinA sinB)
      + be (cosA cosB + sinA sinB)
      + ae (sinA cosB - cosA sinB)
      + bc (cosA sinB - sinA cosB)

      = (ac + be) cos(A - B) + (ae - bc) sin(A - B)

    and A - B = (m - n) t_i. So each pair contributes a fixed amplitude
    times a sinusoid in the RELATIVE distance delta = m - n. No m, no n,
    only their difference — and the amplitudes (ac + be) and (ae - bc) are
    the symmetric and antisymmetric parts of the 2x2 outer product, i.e.
    pure content.

    Checking this against the brute-force rotation is a much stronger
    statement than checking that two particular rotations happen to agree.
    """
    s = 0.0
    for i, t in enumerate(th):
        a, b = q[2 * i], q[2 * i + 1]
        c, e = k[2 * i], k[2 * i + 1]
        s += ((a * c + b * e) * math.cos(delta * t)
              + (a * e - b * c) * math.sin(delta * t))
    return s


# ============================================================================
# SINUSOIDAL  (Vaswani et al. 2017)
# ============================================================================

def sinusoid(pos, d, base):
    """
    PE[pos][2i]   = sin(pos * theta_i)
    PE[pos][2i+1] = cos(pos * theta_i)

    Same theta ladder as RoPE — the frequencies are identical. The
    difference is entirely in WHERE it is applied: sinusoidal ADDS this
    vector to the token embedding at the bottom of the stack; RoPE ROTATES
    q and k inside every attention layer. That placement is the whole story.
    """
    th = thetas(d, base)
    v = [0.0] * d
    for i, t in enumerate(th):
        v[2 * i] = math.sin(pos * t)
        v[2 * i + 1] = math.cos(pos * t)
    return v


# ============================================================================
# PART 1 — THE HOLE:  attention is permutation-equivariant
# ============================================================================

def prove_permutation_equivariance(X, Wq, Wk, Wv):
    """
    Run attention on X. Run it again on X with its ROWS permuted. Claim:

        O(P X) = P O(X)

    i.e. the outputs are the same rows in the same permuted order, to
    floating-point exactness. Nothing in the mechanism resists reordering.

    Then add a learned absolute position embedding and run the identical
    experiment. The equality must now FAIL — that failure is the whole
    purpose of positional encoding, so we measure it rather than assert it.
    """
    base = attention(X, Wq, Wk, Wv)
    Xp = [list(X[PERM[i]]) for i in range(len(X))]
    perm = attention(Xp, Wq, Wk, Wv)

    # P applied to the ORIGINAL output, for comparison
    expect = [list(base["O"][PERM[i]]) for i in range(len(X))]
    err = maxdiff(perm["O"], expect)

    # The score matrix moves the same way: S'[i][j] = S[perm[i]][perm[j]].
    s_expect = [[base["scores"][PERM[i]][PERM[j]] for j in range(len(X))]
                for i in range(len(X))]
    s_err = maxdiff(perm["scores"], s_expect)

    # --- and now the same thing WITH positions -----------------------------
    Epos = [[det(p, j, 5) for j in range(D_MODEL)] for p in range(len(X))]
    Xpe = [[X[i][j] + Epos[i][j] for j in range(D_MODEL)] for i in range(len(X))]
    # the permuted sequence gets position 0,1,2,... in its OWN order, which
    # is the point: position is attached to the slot, not to the token.
    Xp_pe = [[Xp[i][j] + Epos[i][j] for j in range(D_MODEL)] for i in range(len(X))]
    b2 = attention(Xpe, Wq, Wk, Wv)
    p2 = attention(Xp_pe, Wq, Wk, Wv)
    expect2 = [list(b2["O"][PERM[i]]) for i in range(len(X))]
    err2 = maxdiff(p2["O"], expect2)

    return {
        "tokens": TOKENS,
        "perm": PERM,
        "perm_tokens": [TOKENS[PERM[i]] for i in range(len(X))],
        "X": rr(X),
        "X_permuted": rr(Xp),
        "scores": rr(base["scores"]),
        "scores_permuted": rr(perm["scores"]),
        "scores_expected": rr(s_expect),
        "scores_max_abs_err": s_err,
        "probs": rr(base["probs"]),
        "probs_permuted": rr(perm["probs"]),
        "O": rr(base["O"]),
        "O_permuted": rr(perm["O"]),
        "O_expected": rr(expect),
        "max_abs_err": err,
        "passed": err < TOL and s_err < TOL,
        "with_positions": {
            "pos_embed": rr(Epos),
            "O": rr(b2["O"]),
            "O_permuted": rr(p2["O"]),
            "O_expected": rr(expect2),
            "max_abs_err": err2,
            "broken": err2 > 1e-3,
        },
        "claim": "O(PX) = P O(X) exactly. Attention cannot tell 'the cat sat' "
                 "from 'sat the cat'; it sees a bag of vectors. Adding a "
                 "position signal is what breaks the symmetry.",
    }


# ============================================================================
# PART 2 — LEARNED ABSOLUTE EMBEDDINGS
# ============================================================================

def learned_absolute():
    """
    One trained vector per slot: a (L_max, d_model) table, looked up by
    index. GPT-1, GPT-2 and BERT all do this.

    Two measured failures:

    1. EXTRAPOLATION. Row L_max does not exist. This is not a quality
       degradation, it is an IndexError — there is no parameter to read. We
       actually perform the lookup and catch it, so the claim is executed
       rather than described.

    2. NO RELATIVE STRUCTURE. Nothing in the objective ties e_{m+1} - e_m to
       e_{n+1} - e_n. Measure it: for every offset d, collect <e_m, e_{m+d}>
       across all valid m and report the spread. If the table carried
       relative information the spread would be ~0. It is not.
    """
    E = [[det(p, j, 5) for j in range(D_MODEL)] for p in range(L_TRAINED)]

    # 1. the wall
    lookup = []
    for p in range(L_ASKED):
        try:
            row = E[p]
            if p >= L_TRAINED:                 # negative indexing would lie
                raise IndexError("out of range")
            lookup.append({"pos": p, "ok": True, "vec": [r6(v) for v in row]})
        except IndexError:
            lookup.append({"pos": p, "ok": False, "vec": None,
                           "error": "IndexError: position " + str(p) +
                                    " has no row in a table of " +
                                    str(L_TRAINED)})

    # 2. no relative structure
    offsets = []
    worst = 0.0
    for d in range(1, L_TRAINED):
        vals = [dot(E[m], E[m + d]) for m in range(L_TRAINED - d)]
        spread = max(vals) - min(vals)
        worst = max(worst, spread)
        offsets.append({"offset": d, "n_pairs": len(vals),
                        "dots": [r6(v) for v in vals],
                        "spread": r6(spread)})

    return {
        "n_pos": L_TRAINED, "d_model": D_MODEL,
        "params": L_TRAINED * D_MODEL,
        "table": rr(E),
        "lookup": lookup,
        "first_missing": L_TRAINED,
        "asked_up_to": L_ASKED - 1,
        "offset_dots": offsets,
        "max_spread": r6(worst),
        "claim": "A learned table has exactly L_max rows. Position L_max is "
                 "not approximated, it is absent. And <e_m, e_{m+d}> varies "
                 "wildly for fixed d, so the table encodes no notion of "
                 "'d apart' that could generalise.",
    }


# ============================================================================
# PART 3 — SINUSOIDAL, AND THE PROPERTY THAT DOES NOT SURVIVE
# ============================================================================

def sinusoidal_analysis(Wq, Wk):
    """
    The encodings themselves DO have the relative property:

        <PE_m, PE_n> = sum_i [ sin(m t_i) sin(n t_i) + cos(m t_i) cos(n t_i) ]
                     = sum_i cos((m - n) t_i)

    — an exact function of m - n, derived by the cosine difference identity.
    Verified below against the brute-force dot product.

    But that is not what attention computes. Attention computes

        q_m . k_n  where  q_m = (x_m + PE_m) W_q,  k_n = (x_n + PE_n) W_k

    which expands into four terms:

        x_m W_q (x_n W_k)^T    <- content x content
        x_m W_q (PE_n W_k)^T   <- content x position
        PE_m W_q (x_n W_k)^T   <- position x content
        PE_m W_q (PE_n W_k)^T  <- position x position

    Only the last one is purely positional, and even IT is not a function of
    m - n, because W_q W_k^T is an arbitrary learned matrix, not the identity
    that the clean identity above silently assumes. We measure the spread of
    that term at fixed offset. It is large. That gap is exactly what RoPE
    closes by rotating AFTER the projection instead of adding before it.
    """
    d = D_MODEL
    th = thetas(d, BASE_DEFAULT)
    n_show = 12
    table = [[r6(v) for v in sinusoid(p, d, BASE_DEFAULT)] for p in range(n_show)]

    bands = []
    for i, t in enumerate(th):
        bands.append({
            "pair": i, "theta": r6(t),
            "wavelength_tokens": r6(2 * math.pi / t),
            "deg_per_token": r6(math.degrees(t)),
        })

    # (a) the raw-encoding relative property, verified
    rel = []
    worst_rel = 0.0
    worst_closed = 0.0
    for dd in range(0, n_show):
        vals, closed = [], sum(math.cos(dd * t) for t in th)
        for m in range(n_show - dd):
            v = dot(sinusoid(m + dd, d, BASE_DEFAULT), sinusoid(m, d, BASE_DEFAULT))
            vals.append(v)
            worst_closed = max(worst_closed, abs(v - closed))
        spread = max(vals) - min(vals)
        worst_rel = max(worst_rel, spread)
        rel.append({"offset": dd, "dots": [r6(v) for v in vals],
                    "spread": r6(spread), "closed_form": r6(closed)})

    # (b) the same offset, but through W_q and W_k — the term attention
    #     actually sees
    proj = []
    worst_proj = 0.0
    PEq = [mm([sinusoid(p, d, BASE_DEFAULT)], Wq)[0] for p in range(n_show)]
    PEk = [mm([sinusoid(p, d, BASE_DEFAULT)], Wk)[0] for p in range(n_show)]
    for dd in range(0, n_show):
        vals = [dot(PEq[m + dd], PEk[m]) for m in range(n_show - dd)]
        spread = max(vals) - min(vals)
        worst_proj = max(worst_proj, spread)
        proj.append({"offset": dd, "dots": [r6(v) for v in vals],
                     "spread": r6(spread)})

    return {
        "d_model": d, "base": BASE_DEFAULT, "params": 0,
        "n_shown": n_show,
        "bands": bands,
        "table": table,
        "relative_raw": rel,
        "raw_max_spread": r6(worst_rel),
        "closed_form_max_abs_err": worst_closed,
        "raw_relative_holds": worst_rel < 1e-9 and worst_closed < TOL,
        "relative_projected": proj,
        "projected_max_spread": r6(worst_proj),
        "projected_relative_holds": worst_proj < 1e-9,
        "claim": "<PE_m, PE_n> = sum_i cos((m-n) t_i) exactly — a function of "
                 "the offset alone. Push the same encodings through W_q and "
                 "W_k, which is what attention actually does, and the offset "
                 "structure is destroyed. Sinusoidal has the property in the "
                 "wrong place.",
    }


# ============================================================================
# PART 4 — RoPE
# ============================================================================

def rope_analysis(X, Wq, Wk, Wv):
    d = D_HEAD
    th = thetas(d, BASE_DEFAULT)

    # ---- the frequency ladder ------------------------------------------
    pairs = []
    for i, t in enumerate(th):
        pairs.append({
            "pair": i, "dims": [2 * i, 2 * i + 1],
            "theta": r6(t),
            "deg_per_token": r6(math.degrees(t)),
            "wavelength_tokens": r6(2 * math.pi / t),
            "turns_in_1000": r6(1000 * t / (2 * math.pi)),
        })

    # ---- a concrete q and k, and their rotations ------------------------
    q = [det(0, j, 2) for j in range(d)]
    k = [det(1, j, 3) for j in range(d)]

    n_anim = 17
    frames = []
    for p in range(n_anim):
        rq = rope_apply(q, p, th)
        frames.append({
            "pos": p,
            "angles_rad": [r6(p * t) for t in th],
            "angles_deg": [r6(math.degrees(p * t) % 360.0) for t in th],
            "q_rot": [r6(v) for v in rq],
            "norm": r6(math.sqrt(dot(rq, rq))),
        })
    norm_q = math.sqrt(dot(q, q))
    norm_err = max(abs(math.sqrt(dot(rope_apply(q, p, th), rope_apply(q, p, th)))
                       - norm_q) for p in range(n_anim))

    # ---- THE PROOF ------------------------------------------------------
    # For every (m, n) in a 16x16 grid, <R_m q, R_n k>. Group by the
    # relative distance delta = m - n (query position minus key position,
    # so a positive delta means the key is that far in the past) and demand
    # every group be constant. Also check each value against the closed
    # form derived in rope_dot_closed_form.
    N = 16
    grid = [[r6(rope_dot(q, k, m, n, th)) for n in range(N)] for m in range(N)]
    by_offset = {}
    worst_closed = 0.0
    for m in range(N):
        for n in range(N):
            v = rope_dot(q, k, m, n, th)
            by_offset.setdefault(m - n, []).append({"m": m, "n": n, "dot": v})
            worst_closed = max(worst_closed,
                               abs(v - rope_dot_closed_form(q, k, m - n, th)))

    offsets = []
    worst_spread = 0.0
    for off in sorted(by_offset):
        vals = [e["dot"] for e in by_offset[off]]
        spread = max(vals) - min(vals)
        worst_spread = max(worst_spread, spread)
        offsets.append({
            "offset": off, "n_pairs": len(vals),
            "value": r6(vals[0]), "spread": spread,
            "examples": [{"m": e["m"], "n": e["n"], "dot": r6(e["dot"])}
                         for e in by_offset[off][:4]],
        })

    # ---- the shift test: Q,K rotated vs Q,K,V rotated -------------------
    # Slide the WHOLE window: positions 0..3 vs 4..7. Every offset inside
    # the window is unchanged, so if the logits really depend only on
    # offsets, the attention pattern must be bit-identical.
    a0 = attention(X, Wq, Wk, Wv, rope_base=BASE_DEFAULT,
                   positions=list(range(SEQ)))
    a4 = attention(X, Wq, Wk, Wv, rope_base=BASE_DEFAULT,
                   positions=list(range(4, 4 + SEQ)))
    shift_probs_err = maxdiff(a0["probs"], a4["probs"])
    shift_out_err = maxdiff(a0["O"], a4["O"])

    v0 = attention(X, Wq, Wk, Wv, rope_base=BASE_DEFAULT,
                   positions=list(range(SEQ)), rope_v=True)
    v4 = attention(X, Wq, Wk, Wv, rope_base=BASE_DEFAULT,
                   positions=list(range(4, 4 + SEQ)), rope_v=True)
    vprobs_err = maxdiff(v0["probs"], v4["probs"])
    vout_err = maxdiff(v0["O"], v4["O"])

    return {
        "d_head": d, "n_pairs": len(th), "base": BASE_DEFAULT,
        "params": 0,
        "pairs": pairs,
        "q": [r6(v) for v in q], "k": [r6(v) for v in k],
        "q_norm": r6(norm_q),
        "frames": frames,
        "n_frames": n_anim,
        "norm_max_abs_err": norm_err,
        "norm_preserved": norm_err < TOL,
        "grid": {"n": N, "values": grid},
        "offsets": offsets,
        "max_spread_over_offsets": worst_spread,
        "closed_form_max_abs_err": worst_closed,
        "relative_holds": worst_spread < TOL and worst_closed < TOL,
        "shift": {
            "window_a": list(range(SEQ)),
            "window_b": list(range(4, 4 + SEQ)),
            "qk_only": {
                "probs_a": rr(a0["probs"]), "probs_b": rr(a4["probs"]),
                "probs_max_abs_err": shift_probs_err,
                "out_a": rr(a0["O"]), "out_b": rr(a4["O"]),
                "out_max_abs_err": shift_out_err,
                "invariant": shift_probs_err < TOL and shift_out_err < TOL,
            },
            "v_rotated_too": {
                "probs_max_abs_err": vprobs_err,
                "out_a": rr(v0["O"]), "out_b": rr(v4["O"]),
                "out_max_abs_err": vout_err,
                "invariant": vout_err < TOL,
            },
            "claim": "Slide the window four tokens to the right. With RoPE on "
                     "Q and K only, every attention probability AND every "
                     "output row is unchanged to machine precision, because "
                     "the logits see only offsets. Rotate V as well and the "
                     "output moves: V is the payload being averaged, and "
                     "spinning it makes the residual stream's contents depend "
                     "on absolute position with no way to un-spin them.",
        },
        "claim": "<R_m q, R_n k> depends only on the relative distance "
                 "m - n. Every (m, n) pair in a " + str(N) + "x" + str(N) +
                 " grid, sorted into " + str(len(offsets)) + " distance "
                 "groups, and every group is constant to machine precision.",
    }


# ============================================================================
# PART 5 — EXTRAPOLATION, AND THE BASE-SCALING FIXES
# ============================================================================

def extrapolation(d_head, base, l_train, l_target):
    """
    Nothing in RoPE's arithmetic breaks past the training length: R_m is
    defined for any m. What breaks is the model, and the reason is visible
    in the frequency ladder.

    During training on length L, pair i only ever sees offsets in [0, L-1],
    so its rotation angle only ever spans theta_i * (L-1) radians. For the
    FAST pairs that is many full turns — they have seen every angle, many
    times, and they extrapolate for free. For the SLOW pairs it is a small
    arc: pair i has literally never been evaluated outside it. Ask for
    position 4L and the slow pairs are in angular territory the weights
    have no evidence about.

    Two published fixes, both quoted as literature and then verified
    arithmetically here:

      * Position Interpolation (Chen et al. 2023): use m/s instead of m.
        Every angle shrinks by s, so nothing leaves the trained arc — but
        the FAST pairs get squashed too, and they were the ones resolving
        adjacent tokens. Local resolution is the price.

      * NTK-aware base scaling (bloc97 / "NTK-aware scaled RoPE", the idea
        YaRN builds on): raise the base instead, b' = b * s^(d/(d-2)).
        Then theta'_0 = b'^0 = 1 = theta_0 — the fastest pair is untouched —
        while the slowest pair, i = d/2 - 1, gets
            theta'_last = b'^-(d-2)/d = b^-(d-2)/d * s^-1 = theta_last / s
        exactly the factor interpolation would have applied, but applied
        only where it is needed. Both identities are asserted numerically
        below rather than trusted.
    """
    s = float(l_target) / float(l_train)
    th = thetas(d_head, base)
    b_ntk = base * (s ** (d_head / (d_head - 2.0)))
    th_ntk = thetas(d_head, b_ntk)

    rows = []
    for i, t in enumerate(th):
        span_train = t * (l_train - 1)
        span_target = t * (l_target - 1)
        rows.append({
            "pair": i,
            "theta": t,
            "wavelength_tokens": 2 * math.pi / t,
            "max_angle_trained_rad": span_train,
            "turns_trained": span_train / (2 * math.pi),
            "max_angle_target_rad": span_target,
            "turns_target": span_target / (2 * math.pi),
            "seen_full_turn": span_train >= 2 * math.pi,
            "theta_pi": t / s,
            "theta_ntk": th_ntk[i],
            "ntk_shrink": th_ntk[i] / t,
        })

    n_fragile = sum(1 for r in rows if not r["seen_full_turn"])

    # the two identities the NTK trick is built on
    err_fast = abs(th_ntk[0] - th[0])
    err_slow = abs(th_ntk[-1] - th[-1] / s)

    return {
        "d_head": d_head, "base": base,
        "l_train": l_train, "l_target": l_target, "scale": s,
        "base_ntk": b_ntk,
        "rows": rows,
        "n_pairs": len(rows),
        "n_never_completed_a_turn": n_fragile,
        "fraction_fragile": n_fragile / float(len(rows)),
        "ntk_fastest_unchanged_err": err_fast,
        "ntk_slowest_scaled_err": err_slow,
        "ntk_identities_hold": err_fast < 1e-12 and err_slow < 1e-9,
        "pi_note": "Position interpolation divides every angle by s, "
                   "including pair 0, which was resolving adjacent tokens.",
        "ntk_note": "NTK-aware scaling leaves pair 0 exactly alone and "
                    "divides the slowest pair by exactly s.",
    }


def kernel_curve(d_head, base, n):
    """
    f(d) = sum_i cos(d * theta_i), normalised by d/2.

    This is <R_m q, R_n k> for the special case q = k with every pair equal
    to (1,0) — i.e. the purely positional part of the logit, stripped of
    content. It is the shape the model is trained to interpret. Plotting it
    past the training length is the clearest picture of what extrapolation
    asks for: the curve keeps oscillating, and the model has never been
    shown that stretch of it.
    """
    th = thetas(d_head, base)
    return [{"d": d, "f": r6(sum(math.cos(d * t) for t in th) / len(th))}
            for d in range(n)]


# ============================================================================
# LITERATURE  — claims we quote rather than derive
# ============================================================================

LITERATURE = {
    "_source": "published papers and released model configs; external to "
               "this file, quoted not derived",
    "items": [
        {"topic": "sinusoidal",
         "claim": "The original transformer used fixed sin/cos encodings and "
                  "reported them as performing on par with learned absolute "
                  "embeddings, choosing them for the hope of extrapolation.",
         "source": "Vaswani et al., Attention Is All You Need, 2017"},
        {"topic": "rope",
         "claim": "RoPE applies the rotation to queries and keys inside every "
                  "attention layer, with base 10000, and is the encoding used "
                  "by GPT-NeoX, PaLM, Llama and most models since.",
         "source": "Su et al., RoFormer, 2021"},
        {"topic": "base",
         "claim": "Llama 3 raised the RoPE base from 10000 to 500000 to "
                  "support long context.",
         "source": "Llama 3 released model config (rope_theta)"},
        {"topic": "interpolation",
         "claim": "Linear position interpolation (m -> m/s) extends context "
                  "with a short fine-tune, at some cost to short-range "
                  "resolution.",
         "source": "Chen et al., Extending Context Window of LLMs via "
                   "Position Interpolation, 2023"},
        {"topic": "ntk",
         "claim": "NTK-aware base scaling extends context with little or no "
                  "fine-tuning by scaling the base rather than the positions, "
                  "leaving high-frequency pairs untouched.",
         "source": "bloc97, 'NTK-Aware Scaled RoPE', 2023"},
        {"topic": "yarn",
         "claim": "YaRN combines NTK-by-parts interpolation (interpolate only "
                  "the low-frequency pairs, extrapolate the high-frequency "
                  "ones) with an attention temperature correction, and reports "
                  "matching or beating full fine-tuning at a fraction of the "
                  "tokens.",
         "source": "Peng et al., YaRN, 2023"},
        {"topic": "alibi",
         "claim": "ALiBi instead adds a linear, head-specific penalty -m*(i-j) "
                  "straight to the scores, with no embedding at all, and "
                  "reports extrapolation beyond the trained length.",
         "source": "Press et al., ALiBi, 2021"},
        {"topic": "nope",
         "claim": "Decoder-only models with NO positional encoding can still "
                  "learn position from the causal mask alone; the mask is "
                  "itself a positional signal.",
         "source": "Haviv et al. 2022; Kazemnejad et al. 2023"},
    ],
}


# ============================================================================
# BUILD
# ============================================================================

def load_reference_configs(root):
    """
    Real-model geometry comes from the site's single source of truth
    (assets/data/trace.json, emitted by code/ground_truth.py) so that no
    model number is typed twice anywhere in this repo.
    """
    p = os.path.join(root, "assets", "data", "trace.json")
    if not os.path.exists(p):
        raise SystemExit("missing " + p + " — run code/ground_truth.py first")
    with open(p) as f:
        return json.load(f)["reference_configs"]


def build(root):
    ref = load_reference_configs(root)

    X = [[round(((t * 5 + d * 3) % 7 + 1) / 8.0, 4) for d in range(D_MODEL)]
         for t in range(SEQ)]
    Wq = [[det(i, j, 1) for j in range(D_MODEL)] for i in range(D_MODEL)]
    Wk = [[det(i, j, 2) for j in range(D_MODEL)] for i in range(D_MODEL)]
    Wv = [[det(i, j, 3) for j in range(D_MODEL)] for i in range(D_MODEL)]

    equi = prove_permutation_equivariance(X, Wq, Wk, Wv)
    learned = learned_absolute()
    sinus = sinusoidal_analysis(Wq, Wk)
    rope = rope_analysis(X, Wq, Wk, Wv)

    # ---- real models ----------------------------------------------------
    # d_head is DERIVED (d_model / n_heads), never quoted.
    real = []
    for m in ref["models"]:
        dh = m["d_model"] // m["n_heads"]
        if dh % 2:
            continue
        row = {
            "name": m["name"], "d_model": m["d_model"],
            "n_heads": m["n_heads"], "d_head": dh,
            "n_pairs": dh // 2, "seq": m["seq"],
            "learned_abs_params": m["seq"] * m["d_model"],
            "learned_abs_share_of_model": m["seq"] * m["d_model"] / m["params"],
            "rope_params": 0,
        }
        for label, b in (("base_10k", BASE_DEFAULT), ("base_500k", BASE_LLAMA3)):
            th = thetas(dh, b)
            row[label] = {
                "base": b,
                "slowest_wavelength_tokens": 2 * math.pi / th[-1],
                "fastest_wavelength_tokens": 2 * math.pi / th[0],
                "turns_of_slowest_over_ctx": (m["seq"] - 1) * th[-1] / (2 * math.pi),
            }
        real.append(row)

    # spectrum for the site's default model, at both bases
    dm = [m for m in ref["models"] if m["name"] == ref["default_model"]][0]
    dh = dm["d_model"] // dm["n_heads"]
    spectrum = {}
    for label, b in (("base_10k", BASE_DEFAULT), ("base_500k", BASE_LLAMA3)):
        th = thetas(dh, b)
        spectrum[label] = {
            "base": b,
            "rows": [{"pair": i, "theta": t,
                      "wavelength_tokens": 2 * math.pi / t,
                      "turns_over_ctx": (dm["seq"] - 1) * t / (2 * math.pi)}
                     for i, t in enumerate(th)],
        }

    extrap = extrapolation(dh, BASE_DEFAULT, dm["seq"], dm["seq"] * 8)
    extrap_toy = extrapolation(D_HEAD, BASE_DEFAULT, L_TRAINED, L_TRAINED * 4)

    checks = [
        {"id": "equivariance",
         "what": "attention output permutes exactly with its input",
         "metric": equi["max_abs_err"], "tol": TOL, "passed": equi["passed"]},
        {"id": "equivariance_broken_by_position",
         "what": "adding position embeddings breaks that symmetry",
         "metric": equi["with_positions"]["max_abs_err"], "tol": 1e-3,
         "direction": "greater",
         "passed": equi["with_positions"]["broken"]},
        {"id": "learned_no_relative",
         "what": "learned absolute encodes no fixed-offset structure",
         "metric": learned["max_spread"], "tol": 1e-9, "direction": "greater",
         "passed": learned["max_spread"] > 1e-9},
        {"id": "learned_wall",
         "what": "positions past the table have no row at all",
         "metric": float(L_ASKED - L_TRAINED), "tol": 0.0,
         "direction": "greater",
         "passed": not learned["lookup"][L_TRAINED]["ok"]},
        {"id": "sinusoidal_raw_relative",
         "what": "<PE_m, PE_n> = sum_i cos((m-n) t_i), exactly",
         "metric": sinus["closed_form_max_abs_err"], "tol": TOL,
         "passed": sinus["raw_relative_holds"]},
        {"id": "sinusoidal_projected_relative_fails",
         "what": "that property does not survive W_q / W_k",
         "metric": sinus["projected_max_spread"], "tol": 1e-9,
         "direction": "greater",
         "passed": not sinus["projected_relative_holds"]},
        {"id": "rope_relative",
         "what": "<R_m q, R_n k> constant within every offset group",
         "metric": rope["max_spread_over_offsets"], "tol": TOL,
         "passed": rope["relative_holds"]},
        {"id": "rope_closed_form",
         "what": "brute-force rotation matches the derived closed form",
         "metric": rope["closed_form_max_abs_err"], "tol": TOL,
         "passed": rope["closed_form_max_abs_err"] < TOL},
        {"id": "rope_norm",
         "what": "rotation preserves the norm of q",
         "metric": rope["norm_max_abs_err"], "tol": TOL,
         "passed": rope["norm_preserved"]},
        {"id": "rope_shift_invariance",
         "what": "sliding the window leaves Q/K-RoPE attention unchanged",
         "metric": rope["shift"]["qk_only"]["out_max_abs_err"], "tol": TOL,
         "passed": rope["shift"]["qk_only"]["invariant"]},
        {"id": "rope_v_breaks_it",
         "what": "rotating V as well destroys that invariance",
         "metric": rope["shift"]["v_rotated_too"]["out_max_abs_err"],
         "tol": 1e-6, "direction": "greater",
         "passed": not rope["shift"]["v_rotated_too"]["invariant"]},
        {"id": "ntk_identities",
         "what": "b' = b*s^(d/(d-2)) fixes pair 0 and divides the last by s",
         "metric": max(extrap["ntk_fastest_unchanged_err"],
                       extrap["ntk_slowest_scaled_err"]),
         "tol": 1e-9, "passed": extrap["ntk_identities_hold"]},
    ]

    return {
        "meta": {
            "generated_by": "code/positional.py",
            "d_model": D_MODEL, "n_heads": N_HEADS, "d_head": D_HEAD,
            "n_pairs": N_PAIRS, "seq": SEQ, "tokens": TOKENS,
            "perm": PERM,
            "base_default": BASE_DEFAULT, "base_llama3": BASE_LLAMA3,
            "l_trained": L_TRAINED, "l_asked": L_ASKED,
            "tol": TOL,
            "convention": "row-vector: X is (seq, d_in), W is (d_in, d_out), "
                          "Y = X W. Attention here is BIDIRECTIONAL: a causal "
                          "mask is itself a positional signal and would hide "
                          "the permutation symmetry we are exposing.",
            "description": "Attention is permutation-equivariant. Positional "
                           "encoding is the fix, and RoPE is the fix that "
                           "made the offset structure survive the projection.",
        },
        "weights": {"Wq": rr(Wq), "Wk": rr(Wk), "Wv": rr(Wv)},
        "equivariance": equi,
        "learned": learned,
        "sinusoidal": sinus,
        "rope": rope,
        "extrapolation": extrap,
        "extrapolation_toy": extrap_toy,
        "kernel": {
            "toy": kernel_curve(D_HEAD, BASE_DEFAULT, 64),
            "real_10k": kernel_curve(dh, BASE_DEFAULT, 512),
            "real_500k": kernel_curve(dh, BASE_LLAMA3, 512),
            "note": "sum_i cos(d t_i) / (d_head/2): the purely positional part "
                    "of the logit, content stripped out.",
        },
        "real": {"models": real, "spectrum": spectrum,
                 "default_model": ref["default_model"],
                 "_source": ref["_source"]},
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
    with open(os.path.join(outdir, "rope.json"), "w") as f:
        f.write(payload)
    with open(os.path.join(outdir, "rope.js"), "w") as f:
        f.write("// GENERATED by code/positional.py -- do not hand-edit.\n")
        f.write("window.ROPE = " + payload + ";\n")

    m = d["meta"]
    print("=" * 74)
    print("positional.py — permutation equivariance, and RoPE's relative property")
    print("=" * 74)
    print(f"  d_model {m['d_model']}  heads {m['n_heads']}  d_head {m['d_head']}"
          f"  pairs {m['n_pairs']}  seq {m['seq']}  base {m['base_default']:.0f}")
    print()

    e = d["equivariance"]
    print("  1. THE HOLE")
    print(f"     tokens      {e['tokens']}")
    print(f"     permuted    {e['perm_tokens']}   (perm {e['perm']})")
    print(f"     max |O(PX) - P O(X)|            {e['max_abs_err']:.3e}")
    print(f"     max |S(PX) - P S(X) P^T|        {e['scores_max_abs_err']:.3e}")
    print(f"     same, WITH position embeddings  "
          f"{e['with_positions']['max_abs_err']:.3e}  (must be large)")
    print()

    ln = d["learned"]
    print("  2. LEARNED ABSOLUTE")
    print(f"     table {ln['n_pos']} x {ln['d_model']} = {ln['params']} params; "
          f"asked for position {ln['asked_up_to']}")
    print(f"     {ln['lookup'][L_TRAINED]['error']}")
    print(f"     max spread of <e_m, e_m+d> at fixed d: {ln['max_spread']:.4f}"
          f"   (relative structure: none)")
    print()

    s = d["sinusoidal"]
    print("  3. SINUSOIDAL")
    print(f"     raw   <PE_m,PE_n> vs closed form  max|Δ| "
          f"{s['closed_form_max_abs_err']:.3e}   spread {s['raw_max_spread']:.3e}")
    print(f"     after W_q / W_k                   spread "
          f"{s['projected_max_spread']:.4f}   (property destroyed)")
    print()

    r = d["rope"]
    print("  4. RoPE")
    for p in r["pairs"]:
        print(f"     pair {p['pair']}  theta {p['theta']:.6f}"
              f"  {p['deg_per_token']:8.3f} deg/token"
              f"  wavelength {p['wavelength_tokens']:12.2f} tokens")
    print(f"     {r['grid']['n']}x{r['grid']['n']} = "
          f"{r['grid']['n'] ** 2} (m,n) pairs in {len(r['offsets'])} offset groups")
    print(f"     max spread WITHIN an offset group   "
          f"{r['max_spread_over_offsets']:.3e}")
    print(f"     brute force vs closed form  max|Δ|  "
          f"{r['closed_form_max_abs_err']:.3e}")
    print(f"     |R_m q| - |q|               max|Δ|  {r['norm_max_abs_err']:.3e}")
    sh = r["shift"]
    print(f"     window {sh['window_a']} -> {sh['window_b']}:")
    print(f"       Q,K rotated   out max|Δ| "
          f"{sh['qk_only']['out_max_abs_err']:.3e}   (invariant)")
    print(f"       Q,K,V rotated out max|Δ| "
          f"{sh['v_rotated_too']['out_max_abs_err']:.3e}   (broken — why V is "
          f"left alone)")
    print()

    x = d["extrapolation"]
    print("  5. EXTRAPOLATION")
    print(f"     d_head {x['d_head']}  base {x['base']:.0f}  "
          f"L_train {x['l_train']} -> {x['l_target']}  (s = {x['scale']:.0f})")
    print(f"     {x['n_never_completed_a_turn']} of {x['n_pairs']} pairs never "
          f"complete one turn inside the training length "
          f"({100 * x['fraction_fragile']:.1f}%)")
    print(f"     NTK base {x['base_ntk']:.1f}: pair 0 unchanged to "
          f"{x['ntk_fastest_unchanged_err']:.3e}, last pair /s to "
          f"{x['ntk_slowest_scaled_err']:.3e}")
    print()

    print("  CHECKS")
    for c in d["checks"]:
        arrow = ">" if c.get("direction") == "greater" else "<"
        print(f"    {'PASS' if c['passed'] else 'FAIL'}  {c['id']:38s}"
              f" {c['metric']:.3e} {arrow} {c['tol']:.0e}   {c['what']}")
    print()
    print(f"  {'PASS' if d['passed'] else 'FAIL'}: "
          f"{sum(1 for c in d['checks'] if c['passed'])}/{len(d['checks'])} "
          f"checks — attention is permutation-equivariant, and "
          f"<R_m q, R_n k> depends only on m-n")
    print()
    print(f"  wrote {os.path.join(outdir, 'rope.js')}")
    print("=" * 74)
    if not d["passed"]:
        raise SystemExit("FAILED: " + ", ".join(c["id"] for c in d["checks"]
                                                if not c["passed"]))


if __name__ == "__main__":
    main()
