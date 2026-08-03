#!/usr/bin/env python3
"""
moe_toy.py — mixture of experts: where all-to-all finally gets used.

Pure Python standard library.

The site teaches eight collective operations and then applies seven of them.
all_to_all is the one that never appears, because nothing in a dense model
needs it. Mixture-of-experts is what needs it, and this file is that.

The central trick: decouple PARAMETERS from FLOPs. A dense layer uses every
weight for every token. An MoE layer holds N experts, routes each token to
only k of them, and therefore holds N/k times more parameters for the same
compute per token.

The central difficulty: which expert a token goes to is DATA-DEPENDENT and
decided at run time. So the tokens on your GPU need to reach experts on other
GPUs, and their results need to come back. That is an all-to-all, twice per
layer, in forward -- and twice again in backward.

The central failure mode: nothing makes the router balanced. Left alone it
collapses onto a few experts, which wastes the rest and stalls every GPU
waiting for the busiest one.

This file builds all three, and proves the routed computation matches a
per-token reference.

Emits:
    assets/data/moe.js     (window.MOE = {...})
    assets/data/moe.json

Run:  python3 code/moe_toy.py
"""

import json
import math
import os

# ============================================================================
# SHAPES
# ============================================================================
# 6 tokens, 4 experts, top-2 routing, 2 GPUs holding 2 experts each.
# Small enough to print the entire routing table and every all-to-all buffer.

N_TOKENS = 6
D_MODEL = 4
N_EXPERTS = 4
TOP_K = 2
D_FF = 8
EP = 2                     # expert-parallel degree: 2 experts per GPU
CAPACITY_FACTOR = 1.25

TOKEN_STR = ["the", "cat", "sat", "on", "the", "mat"]


def det(i, j, salt):
    return round(((((i * 7 + j * 13 + salt * 11 + 3) % 19) - 9) / 30.0), 4)


X = [[round(((t * 5 + d * 3) % 7 + 1) / 8.0, 4) for d in range(D_MODEL)]
     for t in range(N_TOKENS)]

W_GATE = [[det(i, e, 1) for e in range(N_EXPERTS)] for i in range(D_MODEL)]
EXPERTS = [{
    "W_up": [[det(i, j, 2 + e) for j in range(D_FF)] for i in range(D_MODEL)],
    "W_down": [[det(i, j, 7 + e) for j in range(D_MODEL)] for i in range(D_FF)],
} for e in range(N_EXPERTS)]


def mm(A, B):
    m, k, n = len(A), len(B), len(B[0])
    return [[sum(A[i][t] * B[t][j] for t in range(k)) for j in range(n)]
            for i in range(m)]


def matvec(v, W):
    return [sum(v[i] * W[i][j] for i in range(len(v))) for j in range(len(W[0]))]


def gelu(x):
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + math.tanh(c * (x + 0.044715 * x ** 3)))


def softmax(v):
    m = max(v)
    e = [math.exp(x - m) for x in v]
    s = sum(e)
    return [x / s for x in e]


def maxdiff(A, B):
    return max(abs(A[i][j] - B[i][j])
               for i in range(len(A)) for j in range(len(A[0])))


def expert_forward(v, e):
    """One expert is just an MLP. Nothing exotic."""
    u = matvec(v, EXPERTS[e]["W_up"])
    g = [gelu(x) for x in u]
    return matvec(g, EXPERTS[e]["W_down"])


# ============================================================================
# 1. THE ROUTER
# ============================================================================

def route():
    """
    logits = x @ W_gate, then softmax, then keep the top k.

    The kept gate values are RENORMALISED over the chosen k, so each token's
    weights sum to 1 -- otherwise routing to two experts would systematically
    scale the output differently from routing to one.
    """
    table = []
    for t in range(N_TOKENS):
        logits = matvec(X[t], W_GATE)
        probs = softmax(logits)
        order = sorted(range(N_EXPERTS), key=lambda e: -probs[e])
        chosen = order[:TOP_K]
        raw = [probs[e] for e in chosen]
        s = sum(raw)
        weights = [r / s for r in raw]
        table.append({
            "token": t, "text": TOKEN_STR[t],
            "logits": logits, "probs": probs,
            "chosen": chosen, "raw_gates": raw, "weights": weights,
            "top1": chosen[0],
        })
    return table


# ============================================================================
# 2. THE MoE LAYER, TWO WAYS
# ============================================================================

def reference(table):
    """Per token, loop its chosen experts and blend. The slow, obvious way."""
    out = []
    for t in range(N_TOKENS):
        acc = [0.0] * D_MODEL
        for k, e in enumerate(table[t]["chosen"]):
            y = expert_forward(X[t], e)
            w = table[t]["weights"][k]
            for d in range(D_MODEL):
                acc[d] += w * y[d]
        out.append(acc)
    return out


def grouped(table):
    """
    The way it is actually done: GROUP tokens by expert, run each expert once
    over its whole group as a single matmul, then scatter the results back.

    This is what makes MoE efficient at all -- one GEMM per expert rather
    than one tiny GEMV per token.
    """
    groups = {e: [] for e in range(N_EXPERTS)}
    for t in range(N_TOKENS):
        for k, e in enumerate(table[t]["chosen"]):
            groups[e].append({"token": t, "slot": k,
                              "weight": table[t]["weights"][k]})

    out = [[0.0] * D_MODEL for _ in range(N_TOKENS)]
    per_expert = []
    for e in range(N_EXPERTS):
        g = groups[e]
        if not g:
            per_expert.append({"expert": e, "n_tokens": 0, "tokens": [],
                               "input": [], "output": []})
            continue
        Xe = [X[item["token"]] for item in g]
        U = mm(Xe, EXPERTS[e]["W_up"])
        G = [[gelu(v) for v in r] for r in U]
        Y = mm(G, EXPERTS[e]["W_down"])
        for i, item in enumerate(g):
            for d in range(D_MODEL):
                out[item["token"]][d] += item["weight"] * Y[i][d]
        per_expert.append({"expert": e, "n_tokens": len(g),
                           "tokens": [i["token"] for i in g],
                           "weights": [i["weight"] for i in g],
                           "input": Xe, "output": Y})
    return out, per_expert, groups


# ============================================================================
# 3. EXPERT PARALLELISM  -- the all-to-all
# ============================================================================

def expert_parallel(table, groups):
    """
    Experts are split across EP GPUs. Tokens arrive sharded by SEQUENCE
    (GPU g holds tokens [g*n/EP, ...]). Each token must reach the GPU that
    owns its expert, and its result must come back.

    That is exactly an all-to-all: rank i sends its j-th chunk to rank j.
    Twice per layer -- dispatch and combine.
    """
    per_gpu = N_EXPERTS // EP
    owner = {e: e // per_gpu for e in range(N_EXPERTS)}
    tok_per_gpu = N_TOKENS // EP
    home = {t: t // tok_per_gpu for t in range(N_TOKENS)}

    # dispatch matrix: how many (token, expert) pairs go from src to dst
    dispatch = [[0] * EP for _ in range(EP)]
    detail = []
    for t in range(N_TOKENS):
        for k, e in enumerate(table[t]["chosen"]):
            src, dst = home[t], owner[e]
            dispatch[src][dst] += 1
            detail.append({"token": t, "expert": e, "slot": k,
                           "from_gpu": src, "to_gpu": dst,
                           "local": src == dst})

    gpus = []
    for g in range(EP):
        mine = [e for e in range(N_EXPERTS) if owner[e] == g]
        held = sum(len(groups[e]) for e in mine)
        gpus.append({
            "gpu": g,
            "experts": mine,
            "home_tokens": [t for t in range(N_TOKENS) if home[t] == g],
            "tokens_after_dispatch": held,
            "expert_params": sum(D_MODEL * D_FF + D_FF * D_MODEL for _ in mine),
        })

    sent = sum(dispatch[i][j] for i in range(EP) for j in range(EP) if i != j)
    local = sum(dispatch[i][i] for i in range(EP))
    return {
        "ep_degree": EP, "experts_per_gpu": per_gpu,
        "owner": owner, "home": home,
        "dispatch_matrix": dispatch,
        "detail": detail,
        "gpus": gpus,
        "pairs_total": N_TOKENS * TOP_K,
        "pairs_crossing_gpus": sent, "pairs_local": local,
        "collectives_per_layer": [
            {"op": "all_to_all", "phase": "dispatch",
             "elements": sent * D_MODEL,
             "why": "every token must reach the GPU holding its expert, and "
                    "which GPU that is was decided at run time by the router"},
            {"op": "all_to_all", "phase": "combine",
             "elements": sent * D_MODEL,
             "why": "each expert's output must return to the GPU that owns "
                    "the token, to be blended with its other expert's output"},
        ],
        "note": "Two all-to-alls per MoE layer in forward, and two more in "
                "backward. Unlike tensor parallelism's all-reduce, the "
                "payload SIZE here is data-dependent -- it depends on how "
                "the router happened to distribute this batch.",
    }


# ============================================================================
# 4. LOAD BALANCING  -- the hard part
# ============================================================================

def load_balance(table, groups):
    """
    Nothing in the router's definition makes it balanced. If it collapses
    onto two experts, the GPUs holding the others idle, and the step takes as
    long as the busiest GPU.

    The standard fix (Switch Transformer) is an auxiliary loss:

        L_aux = N * sum_e  f_e * P_e

    where f_e is the FRACTION OF TOKENS routed to expert e (a hard count) and
    P_e is the MEAN ROUTER PROBABILITY for expert e (a soft, differentiable
    quantity). The product is minimised when both are uniform, and only P_e
    carries gradient -- f_e is a constant as far as autograd is concerned.

    Its minimum is 1.0 when perfectly balanced, and it rises to N when
    everything collapses onto one expert.
    """
    counts = [len(groups[e]) for e in range(N_EXPERTS)]
    total = sum(counts)
    f = [c / total for c in counts]
    P = [sum(table[t]["probs"][e] for t in range(N_TOKENS)) / N_TOKENS
         for e in range(N_EXPERTS)]
    aux = N_EXPERTS * sum(f[e] * P[e] for e in range(N_EXPERTS))

    ideal = total / N_EXPERTS
    imbalance = max(counts) / ideal if ideal else 0

    # capacity: each expert accepts at most this many tokens; the rest are
    # DROPPED, which is a real (and deliberate) loss of computation
    cap = int(math.ceil(CAPACITY_FACTOR * total / N_EXPERTS))
    dropped = []
    kept = {e: [] for e in range(N_EXPERTS)}
    for e in range(N_EXPERTS):
        for i, item in enumerate(groups[e]):
            (kept[e] if i < cap else dropped).append(
                {"expert": e, "token": item["token"], "slot": item["slot"]})

    # what a collapsed router would score, for contrast
    collapsed_f = [1.0] + [0.0] * (N_EXPERTS - 1)
    collapsed_P = [1.0] + [0.0] * (N_EXPERTS - 1)
    aux_collapsed = N_EXPERTS * sum(collapsed_f[e] * collapsed_P[e]
                                    for e in range(N_EXPERTS))
    uniform = N_EXPERTS * sum((1 / N_EXPERTS) * (1 / N_EXPERTS)
                              for _ in range(N_EXPERTS))

    return {
        "counts": counts, "fraction": f, "mean_prob": P,
        "aux_loss": aux,
        "aux_loss_formula": "N * sum_e f_e * P_e",
        "aux_loss_if_uniform": uniform,
        "aux_loss_if_collapsed": aux_collapsed,
        "aux_gradient_note": "Only P_e is differentiable. f_e is a hard "
                             "token count with no gradient, so the loss "
                             "pushes the router's PROBABILITIES toward "
                             "balance rather than trying to differentiate "
                             "through the argmax.",
        "ideal_per_expert": ideal,
        "max_load": max(counts), "min_load": min(counts),
        "imbalance_ratio": imbalance,
        "capacity_factor": CAPACITY_FACTOR,
        "capacity_per_expert": cap,
        "kept": {str(e): kept[e] for e in kept},
        "dropped": dropped,
        "n_dropped": len(dropped),
        "straggler_note": "A step costs what the BUSIEST expert costs. An "
                          "imbalance ratio of r means the GPUs holding the "
                          "quiet experts idle for (1 - 1/r) of the layer.",
    }


# ============================================================================
# 5. PARAMS vs FLOPs  -- the whole point
# ============================================================================

def accounting():
    per_expert = D_MODEL * D_FF + D_FF * D_MODEL
    gate = D_MODEL * N_EXPERTS
    total = per_expert * N_EXPERTS + gate
    active = per_expert * TOP_K + gate
    dense_equiv = per_expert

    def flops(n_params_active, tokens):
        return 2 * n_params_active * tokens

    real = []
    for m in [
        {"name": "Mixtral 8x7B", "d_model": 4096, "d_ff": 14336,
         "n_layers": 32, "n_experts": 8, "top_k": 2},
        {"name": "DeepSeek-V3 (671B)", "d_model": 7168, "d_ff": 2048,
         "n_layers": 61, "n_experts": 256, "top_k": 8, "shared": 1},
        {"name": "a dense 70B, for contrast", "d_model": 8192, "d_ff": 28672,
         "n_layers": 80, "n_experts": 1, "top_k": 1},
    ]:
        pe = 3 * m["d_model"] * m["d_ff"]      # SwiGLU: three matrices
        tot = pe * m["n_experts"] * m["n_layers"]
        act = pe * (m["top_k"] + m.get("shared", 0)) * m["n_layers"]
        real.append({
            "name": m["name"], "n_experts": m["n_experts"],
            "top_k": m["top_k"],
            "mlp_params_total": tot, "mlp_params_active": act,
            "sparsity": round(tot / act, 2) if act else 0,
            "note": "MLP parameters only; attention and embeddings are "
                    "dense and identical either way",
        })

    return {
        "toy": {
            "params_per_expert": per_expert, "gate_params": gate,
            "params_total": total, "params_active_per_token": active,
            "dense_equivalent": dense_equiv,
            "sparsity_ratio": round(total / active, 3),
            "flops_per_token_moe": flops(active, 1),
            "flops_per_token_dense": flops(dense_equiv, 1),
        },
        "real": real,
        "why": "Parameters and FLOPs stop being the same question. An MoE "
               "holds N experts' worth of weights and uses k of them per "
               "token, so capacity grows with N while compute grows with k. "
               "The bill arrives as MEMORY: every expert must be resident "
               "somewhere, even the ones this batch never touches.",
    }


# ============================================================================
# 6. INFERENCE  -- a different problem from training
# ============================================================================

def inference_view():
    """
    In training a big batch hits every expert, so grouping gives real GEMMs.
    In decode the batch is small and scatters, so each expert may see one or
    two tokens -- and you still pay to READ its whole weight matrix.

    MoE makes the arithmetic-intensity problem of decode strictly worse.
    """
    rows = []
    pe = 3 * 7168 * 2048          # DeepSeek-V3 shaped expert
    for batch in (1, 8, 32, 128, 512, 2048):
        pairs = batch * 8         # top-8
        experts_touched = min(256, pairs)
        tokens_per_expert = pairs / experts_touched if experts_touched else 0
        bytes_read = experts_touched * pe * 2
        flops = 2 * pe * pairs
        rows.append({
            "batch": batch, "pairs": pairs,
            "experts_touched": experts_touched,
            "avg_tokens_per_expert": round(tokens_per_expert, 2),
            "bytes_read": bytes_read, "flops": flops,
            "intensity": round(flops / bytes_read, 3),
        })
    return {
        "rows": rows,
        "model": "one DeepSeek-V3-shaped MoE layer, 256 experts, top-8",
        "why": "At batch 1 you read eight whole experts to do eight tokens' "
               "worth of arithmetic. Intensity is about 2 FLOP/byte against "
               "an H100 ridge near 148, so you are at roughly 1% of peak. "
               "Batching helps more here than in a dense model, because it "
               "raises the tokens-per-expert count -- but only until you are "
               "touching all the experts anyway.",
        "caveat": "This ignores the expert-parallel all-to-all, which in "
                  "decode is a latency cost paid per token and is often the "
                  "real bottleneck rather than the weight reads.",
    }


def build():
    table = route()
    ref = reference(table)
    grp, per_expert, groups = grouped(table)
    err = maxdiff(ref, grp)
    ep = expert_parallel(table, groups)
    lb = load_balance(table, groups)

    return {
        "meta": {
            "generated_by": "code/moe_toy.py",
            "n_tokens": N_TOKENS, "d_model": D_MODEL,
            "n_experts": N_EXPERTS, "top_k": TOP_K, "d_ff": D_FF,
            "ep_degree": EP, "capacity_factor": CAPACITY_FACTOR,
            "tokens": TOKEN_STR,
            "description": "Routing, grouped expert compute, the "
                           "expert-parallel all-to-all, and load balancing.",
        },
        "input": {"X": X, "W_gate": W_GATE,
                  "experts": [{"expert": e, **EXPERTS[e]}
                              for e in range(N_EXPERTS)]},
        "routing": table,
        "reference_output": ref,
        "grouped_output": grp,
        "per_expert": per_expert,
        "equivalence": {
            "max_abs_err": err, "passed": err < 1e-12,
            "claim": "Grouping tokens by expert and running one matmul per "
                     "expert gives exactly the same answer as looping over "
                     "tokens. Grouping is what makes MoE fast; it changes "
                     "nothing about what is computed.",
        },
        "expert_parallel": ep,
        "load_balance": lb,
        "accounting": accounting(),
        "inference": inference_view(),
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(os.path.dirname(here), "assets", "data")
    os.makedirs(outdir, exist_ok=True)
    d = build()
    payload = json.dumps(d, indent=1, allow_nan=False)
    open(os.path.join(outdir, "moe.json"), "w").write(payload)
    with open(os.path.join(outdir, "moe.js"), "w") as f:
        f.write("// GENERATED by code/moe_toy.py -- do not hand-edit.\n")
        f.write("window.MOE = " + payload + ";\n")

    m, e, lb, ep, ac = (d["meta"], d["equivalence"], d["load_balance"],
                        d["expert_parallel"], d["accounting"])
    print("=" * 72)
    print("moe_toy.py — routing, all-to-all, and load balance")
    print("=" * 72)
    print(f"  {m['n_tokens']} tokens, {m['n_experts']} experts, top-{m['top_k']}, "
          f"EP {m['ep_degree']}")
    print()
    print("  routing:")
    for r in d["routing"]:
        w = ", ".join(f"E{c}:{r['weights'][i]:.3f}"
                      for i, c in enumerate(r["chosen"]))
        print(f"    '{r['text']:4s}' -> {w}")
    print()
    print(f"  grouped vs per-token reference: max|Δ| {e['max_abs_err']:.3e}"
          f"  {'PASS' if e['passed'] else 'FAIL'}")
    print()
    print(f"  expert load: {lb['counts']}  (ideal {lb['ideal_per_expert']:.1f} each)"
          f"   imbalance {lb['imbalance_ratio']:.2f}x")
    print(f"  aux loss {lb['aux_loss']:.4f}   "
          f"(uniform {lb['aux_loss_if_uniform']:.1f}, "
          f"collapsed {lb['aux_loss_if_collapsed']:.1f})")
    print(f"  capacity {lb['capacity_per_expert']}/expert -> "
          f"{lb['n_dropped']} token-slots dropped")
    print()
    print(f"  all-to-all: {ep['pairs_crossing_gpus']} of {ep['pairs_total']} "
          f"(token,expert) pairs cross a GPU boundary")
    print(f"  dispatch matrix {ep['dispatch_matrix']}")
    print()
    t = ac["toy"]
    print(f"  toy params: {t['params_total']} total, "
          f"{t['params_active_per_token']} active/token "
          f"-> {t['sparsity_ratio']}x sparsity")
    for r in ac["real"]:
        print(f"    {r['name']:26s} {r['sparsity']:5.2f}x sparse "
              f"({r['mlp_params_total']/1e9:7.1f}B total MLP, "
              f"{r['mlp_params_active']/1e9:5.1f}B active)")
    print()
    print(f"  wrote {os.path.join(outdir, 'moe.js')}")
    print("=" * 72)
    if not e["passed"]:
        raise SystemExit("MoE EQUIVALENCE FAILED")


if __name__ == "__main__":
    main()
