#!/usr/bin/env python3
"""
language_model.py — what X, Y and y-hat actually ARE, and how training and
inference are the same model run two different ways.

Pure Python standard library.

This file exists because the rest of the site had a hole in the middle of it.
transformer_2layer.py trains against mean-squared-error on an invented target
vector, which is not how a language model is trained at all. A reader could
follow every gradient in this project and still not know what the thing is
predicting.

The answer, and it is simpler than people expect:

    X = the tokens                 the  cat  sat  on   the
    Y = the SAME tokens, shifted   cat  sat  on   the  mat

Position i is asked to predict position i+1. The text is its own supervision:
there are no labels, no annotation, nothing but the corpus shifted by one.
That is the whole objective.

Two consequences the site never states:

  * EVERY POSITION TRAINS AT ONCE. Position 0 predicting "cat" and position 3
    predicting "the" are separate training examples, computed in the same
    forward pass. That is what the causal mask buys, and it is why
    transformers train so much faster than the recurrent models they
    replaced -- an RNN would need five sequential steps for what this does in
    one.

  * TRAINING AND INFERENCE DIFFER IN ONE PLACE ONLY. Training feeds the TRUE
    previous token at every position (teacher forcing). Inference has no true
    previous token, so it feeds back its OWN last prediction. Same weights,
    same forward pass, different input source.

Emits:
    assets/data/lm.js     (window.LM = {...})
    assets/data/lm.json

Run:  python3 code/language_model.py
"""

import json
import math
import os

# ============================================================================
# A TINY LANGUAGE
# ============================================================================
# Nine words, so the vocabulary fits on screen and every probability
# distribution can be printed in full.

VOCAB = ["<pad>", "the", "cat", "sat", "on", "mat", "dog", "ran", "."]
V = len(VOCAB)
STOI = {w: i for i, w in enumerate(VOCAB)}

CORPUS = ["the", "cat", "sat", "on", "the", "mat"]
IDS = [STOI[w] for w in CORPUS]

D_MODEL = 6
N_HEADS = 2
D_HEAD = D_MODEL // N_HEADS
D_FF = 12
N_LAYERS = 2
EPS = 1e-5
LR = 0.5
N_STEPS = 40


def det(i, j, salt):
    return round(((((i * 7 + j * 13 + salt * 11 + 3) % 19) - 9) / 30.0), 4)


def init():
    P = {"E": [[det(v, d, 0) for d in range(D_MODEL)] for v in range(V)],
         # learned absolute positions, the simplest thing that works.
         # RoPE is a separate file; this one is about the objective.
         "pos": [[det(p, d, 1) * 0.5 for d in range(D_MODEL)]
                 for p in range(len(IDS))]}
    for L in range(N_LAYERS):
        s = L * 4
        P[f"L{L}.ln1.g"] = [1.0 + det(0, j, s) for j in range(D_MODEL)]
        P[f"L{L}.ln1.b"] = [det(1, j, s) for j in range(D_MODEL)]
        for nm, salt in (("Wq", 1), ("Wk", 2), ("Wv", 3), ("Wo", 4)):
            P[f"L{L}.{nm}"] = [[det(i, j, s + salt) for j in range(D_MODEL)]
                               for i in range(D_MODEL)]
        P[f"L{L}.ln2.g"] = [1.0 + det(2, j, s) for j in range(D_MODEL)]
        P[f"L{L}.ln2.b"] = [det(3, j, s) for j in range(D_MODEL)]
        P[f"L{L}.W1"] = [[det(i, j, s + 6) for j in range(D_FF)]
                         for i in range(D_MODEL)]
        P[f"L{L}.W2"] = [[det(i, j, s + 7) for j in range(D_MODEL)]
                         for i in range(D_FF)]
    P["lnf.g"] = [1.0 + det(0, j, 20) for j in range(D_MODEL)]
    P["lnf.b"] = [det(1, j, 20) for j in range(D_MODEL)]
    P["head"] = [[det(i, v, 21) for v in range(V)] for i in range(D_MODEL)]
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


def logsumexp(v):
    m = max(v)
    return m + math.log(sum(math.exp(x - m) for x in v))


# ============================================================================
# THE MODEL
# ============================================================================

def forward(P, ids, cache=None):
    """
    ids -> logits, one row of `V` numbers per position.

    Every position's row is a prediction of the NEXT token. Position 0's row
    predicts what follows token 0. Causal masking makes that honest: position
    i is not allowed to look at i+1, which would be its own answer.
    """
    n = len(ids)
    h = [[P["E"][ids[t]][d] + P["pos"][t][d] for d in range(D_MODEL)]
         for t in range(n)]

    for L in range(N_LAYERS):
        nrm = [layernorm(x, P[f"L{L}.ln1.g"], P[f"L{L}.ln1.b"]) for x in h]
        Q, K, Vv = (mm(nrm, P[f"L{L}.Wq"]), mm(nrm, P[f"L{L}.Wk"]),
                    mm(nrm, P[f"L{L}.Wv"]))
        scale = 1.0 / math.sqrt(D_HEAD)
        O = [[0.0] * D_MODEL for _ in range(n)]
        for i in range(n):
            for hd in range(N_HEADS):
                lo, hi = hd * D_HEAD, (hd + 1) * D_HEAD
                sc = [dot(Q[i][lo:hi], K[j][lo:hi]) * scale
                      for j in range(i + 1)]          # CAUSAL: j <= i
                p = softmax(sc)
                for d in range(D_HEAD):
                    O[i][lo + d] = sum(p[j] * Vv[j][lo + d]
                                       for j in range(i + 1))
        A = mm(O, P[f"L{L}.Wo"])
        h = [[h[i][d] + A[i][d] for d in range(D_MODEL)] for i in range(n)]

        n2 = [layernorm(x, P[f"L{L}.ln2.g"], P[f"L{L}.ln2.b"]) for x in h]
        U = mm(n2, P[f"L{L}.W1"])
        G = [[gelu(x) for x in r] for r in U]
        Dn = mm(G, P[f"L{L}.W2"])
        h = [[h[i][d] + Dn[i][d] for d in range(D_MODEL)] for i in range(n)]

    hf = [layernorm(x, P["lnf.g"], P["lnf.b"]) for x in h]
    return mm(hf, P["head"]), hf


# ============================================================================
# THE OBJECTIVE  -- X, Y, y-hat, and cross-entropy
# ============================================================================

def make_xy(ids):
    """
    The entire supervision signal, in two lines.

    X = ids[:-1]     what the model sees
    Y = ids[1:]      what it must predict

    Same list, offset by one. No labels were harmed in the making of this
    training set.
    """
    return ids[:-1], ids[1:]


def loss_and_grad(P, X, Y):
    """
    Cross-entropy, averaged over positions.

        p    = softmax(logits)
        L_i  = -log p[i, Y[i]]
        L    = mean_i L_i

    And the gradient that falls out of it, which is the single most elegant
    result in the whole of this project:

        dL/dlogits[i][v] = (p[i][v] - 1[v == Y[i]]) / N

    "Push down whatever you predicted, push up the truth, divide by the
    number of positions." Every subtlety of softmax and of the log cancels.
    The derivation is in the docstring of cross_entropy_derivation() below.
    """
    logits, _ = forward(P, X)
    N = len(X)
    per_pos, dlogits = [], []
    for i in range(N):
        lse = logsumexp(logits[i])
        p = softmax(logits[i])
        li = lse - logits[i][Y[i]]                # = -log p[Y[i]]
        per_pos.append({
            "pos": i, "input": X[i], "input_str": VOCAB[X[i]],
            "target": Y[i], "target_str": VOCAB[Y[i]],
            "logits": logits[i], "probs": p,
            "prob_of_target": p[Y[i]], "loss": li,
            "predicted": max(range(V), key=lambda v: p[v]),
            "correct": max(range(V), key=lambda v: p[v]) == Y[i],
        })
        dlogits.append([(p[v] - (1.0 if v == Y[i] else 0.0)) / N
                        for v in range(V)])
    L = sum(x["loss"] for x in per_pos) / N
    return L, per_pos, dlogits, logits


def cross_entropy_derivation(P, X, Y):
    """
    Show the cancellation, on real numbers, rather than asserting it.

        L_i = -log( exp(z_y) / sum_v exp(z_v) ) = logsumexp(z) - z_y

        dL_i/dz_v = d(logsumexp)/dz_v  -  d(z_y)/dz_v
                  = p_v                -  1[v == y]

    The first term is the softmax probability -- because the derivative of
    logsumexp IS softmax, which is worth pausing on. The second is 1 only at
    the true token. Averaging over N positions divides by N.
    """
    logits, _ = forward(P, X)
    i = 0
    z = logits[i]
    p = softmax(z)
    y = Y[i]
    h = 1e-6
    numeric = []
    for v in range(V):
        zp, zm = list(z), list(z)
        zp[v] += h
        zm[v] -= h
        lp = logsumexp(zp) - zp[y]
        lm = logsumexp(zm) - zm[y]
        numeric.append((lp - lm) / (2 * h))
    analytic = [p[v] - (1.0 if v == y else 0.0) for v in range(V)]
    err = max(abs(analytic[v] - numeric[v]) for v in range(V))
    return {
        "position": i, "target": y, "target_str": VOCAB[y],
        "logits": z, "probs": p,
        "d_logsumexp_is_softmax": p,
        "onehot": [1.0 if v == y else 0.0 for v in range(V)],
        "analytic": analytic, "numeric": numeric,
        "max_abs_err": err, "passed": err < 1e-7,
        "note": "dL/dz = p - onehot, before the 1/N averaging. Verified "
                "against central differences on the loss itself.",
    }


# ============================================================================
# TRAINING  -- every position at once
# ============================================================================

def train(P, X, Y, steps=N_STEPS):
    """
    Numerical-gradient descent. Slow and stupid, but it needs no autograd and
    it makes the point: the loss falls, the model learns the corpus, and
    perplexity drops from `V` (uniform guessing) toward 1.
    """
    import copy
    P = copy.deepcopy(P)
    hist, snapshots = [], []
    keys = [k for k in P]
    for s in range(steps):
        L, per_pos, _, _ = loss_and_grad(P, X, Y)
        # Full distributions per step, not just p[target]. Page 08 wanted to
        # animate the whole 9-bar distribution SHARPENING over training,
        # which is the strongest visual the data can support, and could not
        # because only the target's probability was recorded.
        hist.append({
            "step": s, "loss": L, "perplexity": math.exp(L),
            "n_correct": sum(1 for x in per_pos if x["correct"]),
            "predictions": [x["predicted"] for x in per_pos],
            "target_probs": [x["prob_of_target"] for x in per_pos],
            "probs": [x["probs"] for x in per_pos],
            "entropy": [-sum(q * math.log(max(q, 1e-30)) for q in x["probs"])
                        for x in per_pos],
        })
        # Snapshot what the model GENERATES part-way through, so the page can
        # show output quality improving rather than only endpoints.
        if s in (0, 5, 10, 20, 39):
            snapshots.append({
                "step": s, "loss": L,
                "generated": generate(P, [STOI["the"]], 4)["final_str"],
            })
        if s == steps - 1:
            break
        h = 1e-4
        for k in keys:
            v = P[k]
            if isinstance(v[0], list):
                for a in range(len(v)):
                    for b in range(len(v[0])):
                        o = v[a][b]
                        v[a][b] = o + h
                        lp = loss_and_grad(P, X, Y)[0]
                        v[a][b] = o - h
                        lm = loss_and_grad(P, X, Y)[0]
                        v[a][b] = o - LR * (lp - lm) / (2 * h)
            else:
                for a in range(len(v)):
                    o = v[a]
                    v[a] = o + h
                    lp = loss_and_grad(P, X, Y)[0]
                    v[a] = o - h
                    lm = loss_and_grad(P, X, Y)[0]
                    v[a] = o - LR * (lp - lm) / (2 * h)
    return P, hist, snapshots


# ============================================================================
# INFERENCE  -- the same model, fed its own output
# ============================================================================

def sample_strategies(probs, temp=1.0, top_k=0, top_p=0.0):
    """
    Four ways to turn a distribution into a token, each shown as the
    distribution it actually samples from.
    """
    logits = [math.log(max(p, 1e-30)) for p in probs]
    if temp != 1.0:
        logits = [l / temp for l in logits]
    q = softmax(logits)
    kept = list(range(len(q)))
    if top_k:
        kept = sorted(range(len(q)), key=lambda v: -q[v])[:top_k]
    if top_p:
        order = sorted(range(len(q)), key=lambda v: -q[v])
        run, kept2 = 0.0, []
        for v in order:
            kept2.append(v)
            run += q[v]
            if run >= top_p:
                break
        kept = [v for v in kept if v in set(kept2)] if top_k else kept2
    z = sum(q[v] for v in kept)
    return {"temperature": temp, "top_k": top_k, "top_p": top_p,
            "kept": kept,
            "dist": [(q[v] / z if v in set(kept) else 0.0)
                     for v in range(len(q))]}


def generate(P, prompt_ids, n_new=4):
    """
    The contrast with training, made concrete.

    TRAINING fed position i the TRUE token i, because it had one. Here there
    is no true token, so position i is fed whatever the model itself produced
    at step i-1. Same weights. Same forward pass. Different input source, and
    an error at step 3 is an error the model must then condition on.
    """
    seq = list(prompt_ids)
    steps = []
    for s in range(n_new):
        logits, _ = forward(P, seq)
        last = logits[-1]                      # ONLY the final row matters
        p = softmax(last)
        nxt = max(range(V), key=lambda v: p[v])
        steps.append({
            "step": s, "context": list(seq),
            "context_str": [VOCAB[t] for t in seq],
            "all_rows_computed": len(seq),
            "row_used": len(seq) - 1,
            "logits": last, "probs": p,
            "chosen": nxt, "chosen_str": VOCAB[nxt],
            "prob_of_chosen": p[nxt],
            "strategies": {
                "greedy": sample_strategies(p, 1.0),
                "temp_0_5": sample_strategies(p, 0.5),
                "temp_1_5": sample_strategies(p, 1.5),
                "top_k_3": sample_strategies(p, 1.0, top_k=3),
                "top_p_0_9": sample_strategies(p, 1.0, top_p=0.9),
            },
            "wasted_note": "Every one of the " + str(len(seq)) + " rows was "
                           "computed; only the last was used. That waste is "
                           "exactly what the KV cache removes.",
        })
        seq.append(nxt)
    return {"prompt": prompt_ids, "steps": steps, "final": seq,
            "final_str": [VOCAB[t] for t in seq]}


def build():
    P = init()
    X, Y = make_xy(IDS)
    L0, per_pos0, dlog0, logits0 = loss_and_grad(P, X, Y)
    deriv = cross_entropy_derivation(P, X, Y)
    trained, hist, snapshots = train(P, X, Y)
    L1, per_pos1, _, _ = loss_and_grad(trained, X, Y)

    gen_untrained = generate(P, [STOI["the"]], 4)
    gen_trained = generate(trained, [STOI["the"]], 4)

    return {
        "meta": {
            "generated_by": "code/language_model.py",
            "vocab": VOCAB, "vocab_size": V,
            "corpus": CORPUS, "ids": IDS,
            "d_model": D_MODEL, "n_heads": N_HEADS, "d_ff": D_FF,
            "n_layers": N_LAYERS, "lr": LR, "steps": N_STEPS,
            "description": "What X, Y and y-hat are for a language model, "
                           "and how training and inference differ.",
        },
        "objective": {
            "X": X, "Y": Y,
            "X_str": [VOCAB[t] for t in X],
            "Y_str": [VOCAB[t] for t in Y],
            "pairs": [{"pos": i, "sees": VOCAB[X[i]], "must_predict": VOCAB[Y[i]],
                       "context": [VOCAB[t] for t in X[:i + 1]]}
                      for i in range(len(X))],
            "how_made": "Y is X shifted left by one. Nothing else. The corpus "
                        "supervises itself, which is why this scales to the "
                        "whole internet without a single human label.",
            "n_training_examples": len(X),
            "parallel_note": "These are " + str(len(X)) + " separate training "
                             "examples and they are all computed in ONE "
                             "forward pass. A recurrent model would need " +
                             str(len(X)) + " sequential steps. That is what "
                             "the causal mask buys.",
        },
        "before": {"loss": L0, "perplexity": math.exp(L0),
                   "per_position": per_pos0, "dlogits": dlog0,
                   "logits": logits0,
                   "uniform_loss": math.log(V),
                   "uniform_perplexity": V},
        "after": {"loss": L1, "perplexity": math.exp(L1),
                  "per_position": per_pos1},
        "cross_entropy_gradient": deriv,
        "training_history": hist,
        "generation_snapshots": snapshots,
        "trained_params": {k: trained[k] for k in sorted(trained)},
        "inference": {
            "untrained": gen_untrained,
            "trained": gen_trained,
            "teacher_forcing_contrast": {
                "training": "position i is fed the TRUE token i, always, "
                            "however wrong the model's own guess was",
                "inference": "position i is fed the model's OWN output from "
                             "step i-1, because there is no true token",
                "consequence": "a mistake at step 3 becomes the input to "
                               "step 4. Training never rehearses that, which "
                               "is called exposure bias.",
            },
        },
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(os.path.dirname(here), "assets", "data")
    os.makedirs(outdir, exist_ok=True)
    d = build()
    payload = json.dumps(d, indent=1, allow_nan=False)
    open(os.path.join(outdir, "lm.json"), "w").write(payload)
    with open(os.path.join(outdir, "lm.js"), "w") as f:
        f.write("// GENERATED by code/language_model.py -- do not hand-edit.\n")
        f.write("window.LM = " + payload + ";\n")

    o, b, a, g = d["objective"], d["before"], d["after"], d["cross_entropy_gradient"]
    print("=" * 72)
    print("language_model.py — X, Y, y-hat, and next-token prediction")
    print("=" * 72)
    print(f"  corpus: {' '.join(CORPUS)}")
    print()
    print(f"    X  {'  '.join(f'{w:5s}' for w in o['X_str'])}   <- what it sees")
    print(f"    Y  {'  '.join(f'{w:5s}' for w in o['Y_str'])}   <- what it must predict")
    print(f"       (the same list, shifted by one. {o['n_training_examples']} "
          f"training examples, all in ONE forward pass)")
    print()
    print(f"  cross-entropy gradient  dL/dz = p - onehot")
    print(f"    max|Δ| vs finite differences {g['max_abs_err']:.3e}"
          f"   {'PASS' if g['passed'] else 'FAIL'}")
    print()
    print(f"  before training: loss {b['loss']:.4f}  perplexity {b['perplexity']:.2f}"
          f"   (uniform would be {b['uniform_perplexity']})")
    print(f"  after  {N_STEPS} steps: loss {a['loss']:.4f}  "
          f"perplexity {a['perplexity']:.2f}"
          f"   {a['per_position'][0] and sum(1 for x in a['per_position'] if x['correct'])}"
          f"/{len(a['per_position'])} positions correct")
    print()
    print("  per position, after training:")
    for x in a["per_position"]:
        mark = "OK " if x["correct"] else "   "
        print(f"    {mark} sees '{x['input_str']:4s}' -> predicts "
              f"'{VOCAB[x['predicted']]:4s}'  (target '{x['target_str']:4s}', "
              f"p={x['prob_of_target']:.3f})")
    print()
    print(f"  inference from 'the':")
    print(f"    untrained {' '.join(d['inference']['untrained']['final_str'])}")
    print(f"    trained   {' '.join(d['inference']['trained']['final_str'])}")
    print()
    print(f"  wrote {os.path.join(outdir, 'lm.js')}")
    print("=" * 72)
    if not g["passed"]:
        raise SystemExit("CROSS-ENTROPY GRADIENT CHECK FAILED")


if __name__ == "__main__":
    main()
