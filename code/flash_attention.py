#!/usr/bin/env python3
"""
flash_attention.py — why the seq^2 matrix never has to exist.

Pure Python standard library. No numpy, no torch.

The rest of the site names FlashAttention repeatedly as the reason the
quadratic activation term does not bite in practice, and never shows it.
This file shows it, and proves the claim rather than asserting it:

  * standard attention, materialising the full seq x seq probability
    matrix, exactly as pages 07 and 09 describe it
  * tiled attention with ONLINE SOFTMAX, which never holds more than one
    (Br x Bc) tile at a time
  * a numerical proof that the two produce the SAME output
  * the backward pass, which recomputes the tile instead of reading a
    stored one -- the trade that makes the whole thing work
  * the memory accounting, in elements, for both

The one idea that makes it possible: softmax can be computed incrementally
if you carry a running maximum and a running sum, and RESCALE what you have
already accumulated whenever the maximum moves. That rescale is the trick.

Emits:
    assets/data/flash.js    (window.FLASH = {...})
    assets/data/flash.json

Run:  python3 code/flash_attention.py
"""

import json
import math
import os

# ============================================================================
# SHAPES
# ============================================================================
# Bigger than the 2-layer toy, because tiling is invisible unless there is
# more than one tile. seq 8 with Bc 2 gives four key tiles; Br 4 gives two
# query tiles. Small enough to print every intermediate.

SEQ = 8
D_HEAD = 4
BR = 4          # query-block rows  (one "outer loop" step)
BC = 2          # key/value-block columns (one "inner loop" step)
CAUSAL = True

SCALE = 1.0 / math.sqrt(D_HEAD)


def det(i, j, salt):
    """Deterministic small values, reproducible without seeding."""
    return round(((((i * 5 + j * 11 + salt * 7 + 4) % 13) - 6) / 12.0), 4)


Q = [[det(i, j, 1) for j in range(D_HEAD)] for i in range(SEQ)]
K = [[det(i, j, 2) for j in range(D_HEAD)] for i in range(SEQ)]
V = [[det(i, j, 3) for j in range(D_HEAD)] for i in range(SEQ)]


# ---------------------------------------------------------------- helpers

def dot(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))


def zeros(m, n):
    return [[0.0] * n for _ in range(m)]


def maxdiff(A, B):
    return max(abs(A[i][j] - B[i][j])
               for i in range(len(A)) for j in range(len(A[0])))


# ============================================================================
# 1. STANDARD ATTENTION  -- the version that materialises seq x seq
# ============================================================================

def standard_attention():
    """
    S = QK^T * scale, masked, then P = softmax(S) row-wise, then O = PV.

    P is (SEQ x SEQ). THAT is the tensor FlashAttention refuses to build,
    and the tensor the rest of the site calls the quadratic term.
    """
    S = zeros(SEQ, SEQ)
    for i in range(SEQ):
        for j in range(SEQ):
            S[i][j] = dot(Q[i], K[j]) * SCALE
            if CAUSAL and j > i:
                S[i][j] = float("-inf")

    P = zeros(SEQ, SEQ)
    row_stats = []
    for i in range(SEQ):
        m = max(v for v in S[i] if v != float("-inf"))
        e = [(math.exp(v - m) if v != float("-inf") else 0.0) for v in S[i]]
        l = sum(e)
        P[i] = [v / l for v in e]
        row_stats.append({"row": i, "max": m, "sumexp": l,
                          "logsumexp": m + math.log(l)})

    O = zeros(SEQ, D_HEAD)
    for i in range(SEQ):
        for d in range(D_HEAD):
            O[i][d] = sum(P[i][j] * V[j][d] for j in range(SEQ))

    return {
        "S": [[None if v == float("-inf") else v for v in row] for row in S],
        "P": P, "O": O, "row_stats": row_stats,
        "stored_elements": SEQ * SEQ,
        "note": "P is seq x seq. For one head at seq 8 that is 64 numbers; "
                "at seq 8192 it is 67 million, per head, per layer.",
    }


# ============================================================================
# 2. ONLINE SOFTMAX  -- the idea, in one dimension first
# ============================================================================

def online_softmax_demo(xs, block):
    """
    softmax over a list, computed in blocks, never seeing the whole list.

    Carry two running numbers:
        m = the largest value seen so far
        l = sum of exp(x - m) over everything seen so far

    When a new block pushes the maximum up from m_old to m_new, everything
    already accumulated was scaled by exp(-m_old) and must be corrected by
    exp(m_old - m_new). That single correction factor is the whole trick.
    """
    m, l = float("-inf"), 0.0
    steps = []
    for b in range(0, len(xs), block):
        chunk = xs[b:b + block]
        m_new = max(m, max(chunk))
        correction = math.exp(m - m_new) if m != float("-inf") else 0.0
        l_rescaled = l * correction
        added = sum(math.exp(v - m_new) for v in chunk)
        l_new = l_rescaled + added
        steps.append({
            "block": b // block, "values": chunk,
            "m_old": None if m == float("-inf") else m,
            "m_new": m_new,
            "correction": correction,
            "l_old": l, "l_rescaled": l_rescaled,
            "added": added, "l_new": l_new,
            "max_moved": m_new != m,
        })
        m, l = m_new, l_new

    online = [math.exp(v - m) / l for v in xs]
    mm = max(xs)
    ee = [math.exp(v - mm) for v in xs]
    direct = [v / sum(ee) for v in ee]
    err = max(abs(online[i] - direct[i]) for i in range(len(xs)))
    return {"xs": xs, "block": block, "steps": steps,
            "online": online, "direct": direct, "max_abs_err": err,
            "passed": err < 1e-12}


# ============================================================================
# 3. FLASH ATTENTION  -- tiled, with the running rescale
# ============================================================================

def flash_attention():
    """
    Outer loop over query blocks, inner loop over key/value blocks.

    Per query block we carry:
        O_acc  (Br x d)  UNNORMALISED accumulator
        m      (Br)      running row max
        l      (Br)      running row sum of exp(s - m)

    On each inner step, if the row max moves we rescale BOTH the accumulator
    and l by exp(m_old - m_new) before adding the new tile's contribution.
    At the end, O = O_acc / l.

    Peak live memory is one (Br x Bc) tile plus m and l. The seq x seq
    matrix is never allocated.
    """
    O = zeros(SEQ, D_HEAD)
    lse = [0.0] * SEQ
    trace = []
    peak_tile = BR * BC

    for qi, q0 in enumerate(range(0, SEQ, BR)):
        rows = list(range(q0, min(q0 + BR, SEQ)))
        m = [float("-inf")] * len(rows)
        l = [0.0] * len(rows)
        acc = zeros(len(rows), D_HEAD)
        qblock = []

        for ki, k0 in enumerate(range(0, SEQ, BC)):
            cols = list(range(k0, min(k0 + BC, SEQ)))

            # A causal block can be skipped entirely if every key in it is
            # in the future of every query in it. Real kernels do exactly
            # this, and it is where causal masking actually saves work.
            if CAUSAL and k0 > rows[-1]:
                qblock.append({"kblock": ki, "cols": cols, "skipped": True,
                               "why": "every key in this tile is in the "
                                      "future of every query in it"})
                continue

            tile = [[0.0] * len(cols) for _ in rows]
            for a, i in enumerate(rows):
                for b, j in enumerate(cols):
                    tile[a][b] = (float("-inf") if (CAUSAL and j > i)
                                  else dot(Q[i], K[j]) * SCALE)

            step = {"kblock": ki, "cols": cols, "skipped": False,
                    "tile": [[None if v == float("-inf") else v for v in r]
                             for r in tile], "rows": []}

            for a, i in enumerate(rows):
                finite = [v for v in tile[a] if v != float("-inf")]
                if not finite:
                    step["rows"].append({"row": i, "all_masked": True})
                    continue
                tile_max = max(finite)
                m_old, l_old = m[a], l[a]
                m_new = max(m_old, tile_max)
                corr = math.exp(m_old - m_new) if m_old != float("-inf") else 0.0

                # rescale everything accumulated so far
                l_res = l_old * corr
                for d in range(D_HEAD):
                    acc[a][d] *= corr

                p = [(math.exp(v - m_new) if v != float("-inf") else 0.0)
                     for v in tile[a]]
                l_new = l_res + sum(p)
                for d in range(D_HEAD):
                    acc[a][d] += sum(p[b] * V[cols[b]][d]
                                     for b in range(len(cols)))

                step["rows"].append({
                    "row": i, "all_masked": False,
                    "tile_max": tile_max,
                    "m_old": None if m_old == float("-inf") else m_old,
                    "m_new": m_new, "correction": corr,
                    "max_moved": m_new != m_old,
                    "l_old": l_old, "l_rescaled": l_res,
                    "p_tile": p, "l_new": l_new,
                    "acc_after": list(acc[a]),
                })
                m[a], l[a] = m_new, l_new

            qblock.append(step)

        for a, i in enumerate(rows):
            for d in range(D_HEAD):
                O[i][d] = acc[a][d] / l[a]
            lse[i] = m[a] + math.log(l[a])

        trace.append({"qblock": qi, "rows": rows, "steps": qblock,
                      "final_m": list(m), "final_l": list(l)})

    return {"O": O, "lse": lse, "trace": trace,
            "peak_tile_elements": peak_tile,
            "saved_elements": 2 * SEQ,
            "saved_what": "one logsumexp per row (and the row max, folded "
                          "into it). Two numbers per query, not seq."}


# ============================================================================
# 4. BACKWARD  -- recompute the tile instead of reading it
# ============================================================================

def flash_backward(std, fl, dO):
    """
    Standard backward READS the stored P. Flash has not got one, so it
    RECOMPUTES each tile from Q and K, using the saved logsumexp to
    normalise -- which is why only two numbers per row had to be kept.

        p_ij = exp(s_ij - lse_i)

    That is the whole recovery. No approximation: the recomputed p is the
    same p, to floating point.

    The cost is one extra QK^T per tile in backward, i.e. roughly 30% more
    attention FLOPs, in exchange for dropping the seq^2 term from memory.
    """
    P_recomputed = zeros(SEQ, SEQ)
    detail = []
    for i in range(SEQ):
        row = []
        for j in range(SEQ):
            if CAUSAL and j > i:
                P_recomputed[i][j] = 0.0
                continue
            s = dot(Q[i], K[j]) * SCALE
            P_recomputed[i][j] = math.exp(s - fl["lse"][i])
            row.append({"j": j, "s": s, "p": P_recomputed[i][j]})
        detail.append({"row": i, "lse": fl["lse"][i], "entries": row,
                       "sum": sum(P_recomputed[i])})

    err = maxdiff(P_recomputed, std["P"])

    # dV = P^T dO ; dP = dO V^T ; then the softmax Jacobian per row
    dV = zeros(SEQ, D_HEAD)
    for j in range(SEQ):
        for d in range(D_HEAD):
            dV[j][d] = sum(P_recomputed[i][j] * dO[i][d] for i in range(SEQ))

    dP = zeros(SEQ, SEQ)
    for i in range(SEQ):
        for j in range(SEQ):
            dP[i][j] = dot(dO[i], V[j])

    dS = zeros(SEQ, SEQ)
    row_dots = []
    for i in range(SEQ):
        # the same dense-Jacobian contraction page 08 derives
        D_i = sum(dP[i][j] * P_recomputed[i][j] for j in range(SEQ))
        row_dots.append(D_i)
        for j in range(SEQ):
            if CAUSAL and j > i:
                continue
            dS[i][j] = P_recomputed[i][j] * (dP[i][j] - D_i)

    dQ = zeros(SEQ, D_HEAD)
    dK = zeros(SEQ, D_HEAD)
    for i in range(SEQ):
        for d in range(D_HEAD):
            dQ[i][d] = sum(dS[i][j] * K[j][d] for j in range(SEQ)) * SCALE
    for j in range(SEQ):
        for d in range(D_HEAD):
            dK[j][d] = sum(dS[i][j] * Q[i][d] for i in range(SEQ)) * SCALE

    return {"P_recomputed": P_recomputed, "recompute_max_err": err,
            "recompute_passed": err < 1e-12, "detail": detail,
            "row_dots": row_dots,
            "dQ": dQ, "dK": dK, "dV": dV, "dS": dS,
            "note": "p_ij = exp(s_ij - lse_i). Two saved numbers per row "
                    "reconstruct the entire probability matrix exactly."}


# ============================================================================
# 5. MEMORY, AT REAL SCALE
# ============================================================================

def memory_model():
    """Per layer, per head, activation elements for the attention probs."""
    rows = []
    for s in (128, 512, 1024, 2048, 4096, 8192, 32768, 131072):
        rows.append({
            "seq": s,
            "standard_elements_per_head": s * s,
            "flash_elements_per_head": 2 * s,
            "ratio": s // 2,
        })
    return {
        "table": rows,
        "formula_standard": "batch * heads * seq^2 per layer",
        "formula_flash": "batch * heads * 2 * seq per layer (the logsumexp "
                         "and the row max, folded into one number each)",
        "compute_cost": "roughly +30% attention FLOPs in backward, because "
                        "each tile's scores are recomputed rather than read",
        "why_it_is_also_faster": "it is not only a memory trick. The seq^2 "
                                 "matrix never leaves SRAM, so the kernel "
                                 "stops being bound by HBM bandwidth. On "
                                 "long sequences FlashAttention is faster "
                                 "AND smaller, which is why it is not a "
                                 "trade-off anyone agonises over.",
    }


def build():
    std = standard_attention()
    fl = flash_attention()
    err = maxdiff(std["O"], fl["O"])

    dO = [[det(i, d, 9) for d in range(D_HEAD)] for i in range(SEQ)]
    bwd = flash_backward(std, fl, dO)

    demo = online_softmax_demo([1.0, 3.0, 2.0, 8.0, 7.0, 4.0], 2)

    return {
        "meta": {
            "generated_by": "code/flash_attention.py",
            "seq": SEQ, "d_head": D_HEAD, "Br": BR, "Bc": BC,
            "causal": CAUSAL, "scale": SCALE,
            "n_q_blocks": (SEQ + BR - 1) // BR,
            "n_k_blocks": (SEQ + BC - 1) // BC,
            "description": "Standard vs tiled attention with online softmax, "
                           "proven identical.",
        },
        "input": {"Q": Q, "K": K, "V": V, "dO": dO},
        "standard": std,
        "flash": fl,
        "equivalence": {
            "output_max_abs_err": err,
            "passed": err < 1e-12,
            "claim": "Tiled attention with online softmax produces the SAME "
                     "output as the version that materialises seq x seq. "
                     "FlashAttention is exact, not an approximation. That is "
                     "the single most important thing to know about it.",
        },
        "backward": bwd,
        "online_softmax_demo": demo,
        "memory": memory_model(),
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    outdir = os.path.join(root, "assets", "data")
    os.makedirs(outdir, exist_ok=True)

    d = build()
    payload = json.dumps(d, indent=1, allow_nan=False)
    open(os.path.join(outdir, "flash.json"), "w").write(payload)
    with open(os.path.join(outdir, "flash.js"), "w") as f:
        f.write("// GENERATED by code/flash_attention.py -- do not hand-edit.\n")
        f.write("window.FLASH = " + payload + ";\n")

    m = d["meta"]
    print("=" * 70)
    print("flash_attention.py — the seq^2 matrix never has to exist")
    print("=" * 70)
    print(f"  seq {m['seq']}  d_head {m['d_head']}  tiles {m['Br']}x{m['Bc']}"
          f"  ({m['n_q_blocks']} query blocks x {m['n_k_blocks']} key blocks)")
    print()
    print(f"  online-softmax demo:   max|Δ| {d['online_softmax_demo']['max_abs_err']:.3e}"
          f"  {'PASS' if d['online_softmax_demo']['passed'] else 'FAIL'}")
    print(f"  flash vs standard O:   max|Δ| {d['equivalence']['output_max_abs_err']:.3e}"
          f"  {'PASS' if d['equivalence']['passed'] else 'FAIL'}")
    print(f"  recomputed P vs P:     max|Δ| {d['backward']['recompute_max_err']:.3e}"
          f"  {'PASS' if d['backward']['recompute_passed'] else 'FAIL'}")
    print()
    print(f"  standard stores  {d['standard']['stored_elements']} elements "
          f"(seq x seq)")
    print(f"  flash stores     {d['flash']['saved_elements']} elements "
          f"({d['flash']['saved_what']})")
    print(f"  peak live tile   {d['flash']['peak_tile_elements']} elements")
    print()
    for r in d["memory"]["table"][:6]:
        print(f"    seq {r['seq']:6d}  standard {r['standard_elements_per_head']:>12,}"
              f"   flash {r['flash_elements_per_head']:>8,}   {r['ratio']}x")
    print()
    print(f"  wrote {os.path.join(outdir, 'flash.js')}")
    print("=" * 70)

    if not (d["equivalence"]["passed"] and d["backward"]["recompute_passed"]
            and d["online_softmax_demo"]["passed"]):
        raise SystemExit("EQUIVALENCE PROOF FAILED")


if __name__ == "__main__":
    main()
