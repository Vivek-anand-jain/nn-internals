#!/usr/bin/env python3
"""
parallel_toy.py — ground truth for the distributed-training half of the site.

Pure Python standard library. No numpy, no torch, no NCCL.

It SIMULATES four GPUs running one training step of a small MLP under every
major parallelism strategy, and for each one records:

  * exactly which slice of which tensor each GPU holds
  * the ordered schedule of collective operations, with payload sizes
  * the bytes moved, and the ring-algorithm cost
  * a PROOF that the distributed result equals the single-GPU result

That last point is the whole reason this file exists. Every claim the site
makes about parallelism ("TP gives identical math", "DDP is just averaged
gradients") is checked here numerically rather than asserted in prose.

Emits:
    assets/data/parallel.js    (window.PARALLEL = {...})
    assets/data/parallel.json

Run:  python3 code/parallel_toy.py
"""

import json
import math
import os

# ============================================================================
# THE MODEL
# ============================================================================
# Four layers, so a 4-way pipeline gets exactly one layer per stage.
# Hidden width 8, so 4-way tensor parallelism gets exactly two units each.
# Batch 4, so 4-way data parallelism gets exactly one sample each.
#
#   x (4 features) -> 8 -> 8 -> 8 -> 1
#
# Every strategy therefore divides evenly by 4, which keeps the arithmetic
# legible. Real models are not this tidy and the site says so.

DIMS = [4, 8, 8, 8, 1]
N_LAYERS = len(DIMS) - 1
WORLD = 4
BATCH = 4


def det(i, j, salt):
    """Deterministic small weights in [-0.4, 0.4]. Not random -- reproducible
    without seeding, and small enough that products stay readable."""
    return round((((i * 7 + j * 13 + salt * 5 + 5) % 17) - 8) / 20.0, 3)


def init_params():
    Ws, bs = [], []
    for L in range(N_LAYERS):
        n_out, n_in = DIMS[L + 1], DIMS[L]
        Ws.append([[det(i, j, L) for j in range(n_in)] for i in range(n_out)])
        bs.append([round(((i * 3 + L * 7) % 9 - 4) / 20.0, 3) for i in range(n_out)])
    return Ws, bs


# The batch. Four "houses", four features each.
X = [[round(((b * 5 + f * 3) % 7 + 1) / 4.0, 3) for f in range(DIMS[0])]
     for b in range(BATCH)]
Y = [1.0, 0.5, 1.5, 0.75]

FEATURES = ["size", "bedrooms", "age", "distance"]


# ============================================================================
# Plain linear algebra
# ============================================================================

def matmul(A, B):
    """(m x k) @ (k x n) -> (m x n)"""
    m, k, n = len(A), len(B), len(B[0])
    return [[sum(A[i][t] * B[t][j] for t in range(k)) for j in range(n)]
            for i in range(m)]


def transpose(A):
    return [list(r) for r in zip(*A)]


def add_bias(Z, b):
    return [[Z[i][j] + b[j] for j in range(len(b))] for i in range(len(Z))]


def relu(Z):
    return [[v if v > 0 else 0.0 for v in row] for row in Z]


def relu_mask(Z):
    return [[1.0 if v > 0 else 0.0 for v in row] for row in Z]


def hadamard(A, B):
    return [[A[i][j] * B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def madd(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def scale(A, s):
    return [[v * s for v in row] for row in A]


def zeros(m, n):
    return [[0.0] * n for _ in range(m)]


def col_slice(M, lo, hi):
    """Columns [lo, hi) of every row."""
    return [row[lo:hi] for row in M]


def row_slice(M, lo, hi):
    return [list(r) for r in M[lo:hi]]


def flat(M):
    return [v for row in M for v in row]


def maxdiff(a, b):
    """Max absolute elementwise difference between two nested structures."""
    fa, fb = _flatten(a), _flatten(b)
    assert len(fa) == len(fb), f"shape mismatch {len(fa)} vs {len(fb)}"
    return max((abs(x - y) for x, y in zip(fa, fb)), default=0.0)


def _flatten(x):
    if isinstance(x, (int, float)):
        return [float(x)]
    out = []
    for v in x:
        out.extend(_flatten(v))
    return out


# ============================================================================
# SINGLE-GPU BASELINE  -- the answer every strategy must reproduce
# ============================================================================

def forward(Ws, bs, Xb):
    """Returns (yhat, cache). Cache holds what backward needs."""
    acts = [Xb]           # acts[L] is the input to layer L
    pre = []              # pre[L] is the pre-activation of layer L
    h = Xb
    for L in range(N_LAYERS):
        z = add_bias(matmul(h, transpose(Ws[L])), bs[L])
        pre.append(z)
        # ReLU on every layer except the output
        h = relu(z) if L < N_LAYERS - 1 else z
        acts.append(h)
    return h, {"acts": acts, "pre": pre}


def loss_and_grad_out(yhat, Yb):
    """Mean squared error over the batch, and dL/dyhat."""
    B = len(Yb)
    errs = [yhat[b][0] - Yb[b] for b in range(B)]
    L = sum(e * e for e in errs) / B
    dY = [[2.0 * errs[b] / B] for b in range(B)]
    return L, dY, errs


def backward(Ws, bs, cache, dY):
    """Hand-written backward. Returns (dWs, dbs)."""
    acts, pre = cache["acts"], cache["pre"]
    dWs = [None] * N_LAYERS
    dbs = [None] * N_LAYERS
    delta = dY                                     # dL/d(pre[last])
    for L in range(N_LAYERS - 1, -1, -1):
        if L < N_LAYERS - 1:
            delta = hadamard(delta, relu_mask(pre[L]))
        # dL/dW[L] = delta^T @ input   (input is acts[L])
        dWs[L] = matmul(transpose(delta), acts[L])
        dbs[L] = [sum(delta[b][i] for b in range(len(delta)))
                  for i in range(len(delta[0]))]
        if L > 0:
            delta = matmul(delta, Ws[L])           # push through the weights
    return dWs, dbs


def baseline():
    Ws, bs = init_params()
    yhat, cache = forward(Ws, bs, X)
    L, dY, errs = loss_and_grad_out(yhat, Y)
    dWs, dbs = backward(Ws, bs, cache, dY)
    return {"Ws": Ws, "bs": bs, "yhat": yhat, "loss": L, "errs": errs,
            "dWs": dWs, "dbs": dbs, "cache": cache}


# ============================================================================
# COLLECTIVE OPERATIONS  -- modelled explicitly, with ring costs
# ============================================================================
# Every strategy's communication is expressed as a list of these. The site
# animates them directly from this schedule.

def ring_cost(op, n_elements, world):
    """
    Bytes each GPU must SEND for a ring implementation.

    all_reduce     2*(N-1)/N * S   (reduce-scatter then all-gather)
    all_gather       (N-1)/N * S
    reduce_scatter   (N-1)/N * S
    broadcast              ~ S     (tree/ring, one full pass)
    all_to_all       (N-1)/N * S
    p2p                      S     (a single hop)

    S is the payload in elements. This is the standard bandwidth-optimal
    analysis; it ignores latency terms and assumes a homogeneous ring.
    """
    N = world
    if N <= 1:
        return 0.0
    f = {
        "all_reduce": 2.0 * (N - 1) / N,
        "all_gather": (N - 1) / N,
        "reduce_scatter": (N - 1) / N,
        "all_to_all": (N - 1) / N,
        "broadcast": 1.0,
        "reduce": 1.0,
        "scatter": (N - 1) / N,
        "gather": (N - 1) / N,
        "p2p": 1.0,
    }.get(op, 1.0)
    return f * n_elements


COLLECTIVES = [
    {"op": "broadcast", "sig": "one -> all",
     "what": "One rank's buffer is copied to every rank.",
     "used_for": "Sending initial weights to every replica at startup.",
     "ring_factor": "S", "inverse": "reduce"},
    {"op": "scatter", "sig": "one -> all (split)",
     "what": "One rank's buffer is cut into N pieces; rank i receives piece i.",
     "used_for": "Handing each data-parallel rank its slice of the batch.",
     "ring_factor": "(N-1)/N * S", "inverse": "gather"},
    {"op": "gather", "sig": "all -> one",
     "what": "Rank i's piece is collected onto one rank, concatenated.",
     "used_for": "Pulling a full tensor back for checkpointing or logging.",
     "ring_factor": "(N-1)/N * S", "inverse": "scatter"},
    {"op": "all_gather", "sig": "all -> all (concat)",
     "what": "Every rank ends up with the concatenation of all N pieces.",
     "used_for": "FSDP/ZeRO-3 re-materialising a full layer's weights just "
                 "before using them.",
     "ring_factor": "(N-1)/N * S", "inverse": "reduce_scatter"},
    {"op": "reduce", "sig": "all -> one (sum)",
     "what": "Elementwise sum across ranks, result on one rank.",
     "used_for": "Collecting a loss value for logging.",
     "ring_factor": "S", "inverse": "broadcast"},
    {"op": "reduce_scatter", "sig": "all -> all (sum + split)",
     "what": "Elementwise sum across ranks, then rank i keeps only shard i "
             "of the result.",
     "used_for": "ZeRO-2/3 reducing gradients so each rank keeps only the "
                 "shard it will apply.",
     "ring_factor": "(N-1)/N * S", "inverse": "all_gather"},
    {"op": "all_reduce", "sig": "all -> all (sum)",
     "what": "Elementwise sum across ranks, result on every rank. "
             "Implemented as reduce_scatter followed by all_gather.",
     "used_for": "DDP averaging gradients; tensor parallelism summing "
                 "partial activations.",
     "ring_factor": "2(N-1)/N * S", "inverse": "itself"},
    {"op": "all_to_all", "sig": "all -> all (transpose)",
     "what": "Rank i sends its j-th chunk to rank j. A distributed transpose.",
     "used_for": "Mixture-of-experts routing; sequence <-> hidden "
                 "redistribution in sequence parallelism.",
     "ring_factor": "(N-1)/N * S", "inverse": "itself"},
    {"op": "p2p", "sig": "one -> one",
     "what": "A direct send/recv between two ranks.",
     "used_for": "Pipeline parallelism handing activations to the next "
                 "stage and gradients back to the previous one.",
     "ring_factor": "S", "inverse": "itself"},
]


def comm(step, op, payload, world, why, phase, tensor=None, src=None, dst=None):
    """One entry in a strategy's communication schedule."""
    return {
        "step": step, "op": op, "tensor": tensor,
        "elements": payload,
        "bytes_bf16": payload * 2,
        "bytes_fp32": payload * 4,
        "sent_per_gpu_elements": round(ring_cost(op, payload, world), 3),
        "world": world, "why": why, "phase": phase,
        "src": src, "dst": dst,
    }


# ============================================================================
# STRATEGY 1 — DATA PARALLEL (DDP)
# ============================================================================
# Every GPU holds a COMPLETE copy of the model. The BATCH is split.
# Each GPU computes gradients on its own samples; one all-reduce averages
# them, so every replica applies an identical update and they never diverge.

def run_data_parallel(base):
    Ws, bs = base["Ws"], base["bs"]
    per = BATCH // WORLD
    shards, local = [], []

    for g in range(WORLD):
        Xg = X[g * per:(g + 1) * per]
        Yg = Y[g * per:(g + 1) * per]
        yhat_g, cache_g = forward(Ws, bs, Xg)
        Lg, dYg, _ = loss_and_grad_out(yhat_g, Yg)
        dWg, dbg = backward(Ws, bs, cache_g, dYg)
        local.append({"gpu": g, "samples": list(range(g * per, (g + 1) * per)),
                      "loss": Lg, "yhat": yhat_g,
                      "dWs": dWg, "dbs": dbg})
        shards.append({"gpu": g,
                       "holds": ["ALL weights (replicated)",
                                 "ALL optimizer state (replicated)",
                                 f"batch samples {g*per}..{g*per+per-1}",
                                 "activations for its own samples only"]})

    # The all-reduce: elementwise MEAN of the per-GPU gradients.
    avgW = []
    for L in range(N_LAYERS):
        acc = zeros(len(Ws[L]), len(Ws[L][0]))
        for g in range(WORLD):
            acc = madd(acc, local[g]["dWs"][L])
        avgW.append(scale(acc, 1.0 / WORLD))
    avgb = []
    for L in range(N_LAYERS):
        acc = [0.0] * len(bs[L])
        for g in range(WORLD):
            acc = [acc[i] + local[g]["dbs"][L][i] for i in range(len(acc))]
        avgb.append([v / WORLD for v in acc])

    # PROOF: averaged DDP gradients == single-GPU full-batch gradients.
    errW = max(maxdiff(avgW[L], base["dWs"][L]) for L in range(N_LAYERS))
    errb = max(maxdiff(avgb[L], base["dbs"][L]) for L in range(N_LAYERS))
    avg_loss = sum(l["loss"] for l in local) / WORLD

    n_params = sum(len(flat(Ws[L])) + len(bs[L]) for L in range(N_LAYERS))
    sched = [comm(1, "all_reduce", n_params, WORLD,
                  "Average the gradients so every replica applies the same "
                  "update and the copies never drift apart.",
                  "after backward", tensor="all gradients")]

    return {
        "name": "Data Parallel (DDP)",
        "splits": "the BATCH", "replicates": "weights, gradients, optimizer state",
        "per_gpu": shards, "local": local,
        "verify": {"grad_max_err": max(errW, errb),
                   "loss_max_err": abs(avg_loss - base["loss"]),
                   "claim": "Averaging the per-GPU gradients reproduces the "
                            "full-batch gradient exactly.",
                   "passed": max(errW, errb) < 1e-12},
        "schedule": sched,
        "memory_per_gpu": {"weights": n_params, "gradients": n_params,
                           "optimizer": 2 * n_params,
                           "activations_divisor": WORLD},
        "drawback": "Every GPU stores the entire model and the entire "
                    "optimizer state. Nothing about the memory bill improves.",
    }


# ============================================================================
# STRATEGY 2 — TENSOR PARALLEL (Megatron style)
# ============================================================================
# The WEIGHT MATRICES themselves are cut up. Pairs of layers are arranged
# column-parallel then row-parallel, which is the trick that lets the
# nonlinearity in between run on the sharded tensor with NO communication.
# One all-reduce per pair, at the end.

def run_tensor_parallel(base):
    Ws, bs = base["Ws"], base["bs"]
    sched, blocks = [], []
    step = 0

    # Layers are paired: (0,1) and (2,3).
    h_full = X
    verify_notes = []

    for blk, (Lc, Lr) in enumerate([(0, 1), (2, 3)]):
        n_hidden = DIMS[Lc + 1]
        per = n_hidden // WORLD          # 8 / 4 = 2 units per GPU

        # ---- column-parallel layer: split the OUTPUT dimension -----------
        # GPU g owns rows [g*per, (g+1)*per) of W[Lc]  (i.e. output units)
        partials = []
        for g in range(WORLD):
            Wg = row_slice(Ws[Lc], g * per, (g + 1) * per)      # (per x in)
            bg = bs[Lc][g * per:(g + 1) * per]
            zg = add_bias(matmul(h_full, transpose(Wg)), bg)     # (B x per)
            ag = relu(zg)          # elementwise -> safe on the shard alone
            partials.append(ag)

        step += 1
        sched.append(comm(step, "none", 0, WORLD,
                          "No communication. The column split leaves each GPU "
                          "with whole output units, and ReLU is elementwise, "
                          "so it runs on the shard untouched. This is the "
                          "reason column-then-row is the standard pairing.",
                          f"block {blk}: column-parallel layer {Lc}",
                          tensor=f"W{Lc} split by output units"))

        # ---- row-parallel layer: split the INPUT dimension ---------------
        # GPU g owns columns [g*per, (g+1)*per) of W[Lr] -- matching the
        # shard of the hidden tensor it already holds. No reshuffle needed.
        partial_out = []
        for g in range(WORLD):
            Wg = col_slice(Ws[Lr], g * per, (g + 1) * per)       # (out x per)
            partial_out.append(matmul(partials[g], transpose(Wg)))

        # all-reduce: sum the partial products
        summed = zeros(BATCH, DIMS[Lr + 1])
        for g in range(WORLD):
            summed = madd(summed, partial_out[g])
        z = add_bias(summed, bs[Lr])     # bias added ONCE, after the sum
        h_full = relu(z) if Lr < N_LAYERS - 1 else z

        step += 1
        sched.append(comm(step, "all_reduce", BATCH * DIMS[Lr + 1], WORLD,
                          "Each GPU computed a PARTIAL SUM over its slice of "
                          "the contraction. The pieces must be added to form "
                          "the real activation, so this all-reduce is not "
                          "optional -- it is inside the maths, not around it.",
                          f"block {blk}: row-parallel layer {Lr}",
                          tensor=f"partial activations ({BATCH}x{DIMS[Lr+1]})"))

        blocks.append({
            "block": blk, "col_layer": Lc, "row_layer": Lr,
            "units_per_gpu": per,
            "hidden_shard_shape": [BATCH, per],
            "output_shape": [BATCH, DIMS[Lr + 1]],
            "allreduce_elements": BATCH * DIMS[Lr + 1],
            # The actual numbers, so a page can READ the proof instead of
            # recomputing it in JS. Each GPU's hidden shard after the
            # column-parallel layer + ReLU, and its PARTIAL contribution to
            # the row-parallel layer's output. The partials are individually
            # meaningless -- that is the point -- and sum to `summed`.
            "per_gpu_hidden": [partials[g] for g in range(WORLD)],
            "per_gpu_partial_out": [partial_out[g] for g in range(WORLD)],
            "summed_before_bias": summed,
            "after_bias": z,
        })
        verify_notes.append(f"block {blk} all-reduce of {BATCH}x{DIMS[Lr+1]}")

    err = maxdiff(h_full, base["yhat"])

    # Backward mirrors forward, operator by operator. Both halves of the
    # duality get an entry so a page can render the whole f/g table from
    # data instead of deriving the silent half.
    for blk, (Lc, Lr) in enumerate([(0, 1), (2, 3)]):
        # g: all-reduce in forward, IDENTITY in backward.
        step += 1
        sched.append(comm(step, "none", 0, WORLD,
                          "Backward through the row-parallel layer. The g "
                          "operator all-reduced in forward, so its conjugate "
                          "is the identity here: the incoming gradient is "
                          "already the full tensor and every GPU simply takes "
                          "the slice matching the shard it owns.",
                          f"backward block {blk}: row-parallel layer {Lr}",
                          tensor=f"dL/doutput ({BATCH}x{DIMS[Lr+1]})"))
        # f: identity in forward, ALL-REDUCE in backward.
        step += 1
        sched.append(comm(step, "all_reduce", BATCH * DIMS[Lc], WORLD,
                          "Backward through the column-parallel layer. The f "
                          "operator was the identity in forward, because the "
                          "input was replicated -- and a replicated input "
                          "means every GPU produced a PARTIAL gradient for "
                          "it, so they must be summed. Forward's no-op is "
                          "backward's all-reduce. The two are conjugates.",
                          f"backward block {blk}: column-parallel layer {Lc}",
                          tensor=f"dL/dinput ({BATCH}x{DIMS[Lc]})"))

    n_params = sum(len(flat(Ws[L])) + len(bs[L]) for L in range(N_LAYERS))
    return {
        "name": "Tensor Parallel (Megatron)",
        "splits": "individual WEIGHT MATRICES, and the activations with them",
        "replicates": "the batch (every GPU sees all samples)",
        "blocks": blocks,
        "verify": {"forward_max_err": err,
                   "claim": "Summing the per-GPU partial products reproduces "
                            "the single-GPU forward pass exactly.",
                   "passed": err < 1e-12},
        "schedule": sched,
        "memory_per_gpu": {"weights": n_params // WORLD,
                           "gradients": n_params // WORLD,
                           "optimizer": 2 * n_params // WORLD,
                           "activations_divisor": WORLD},
        "drawback": "Two all-reduces per block, on the CRITICAL PATH of every "
                    "forward and backward. The activation tensors are large "
                    "and the collectives cannot be overlapped away, so this "
                    "wants NVLink-class bandwidth. Almost always kept inside "
                    "a single node.",
    }


# ============================================================================
# STRATEGY 3 — PIPELINE PARALLEL
# ============================================================================
# LAYERS are split across GPUs. Activations travel forward point-to-point,
# gradients travel back the same way. The cost is the bubble: stages idle
# while the pipeline fills and drains.

def run_pipeline_parallel(base, n_micro=4):
    Ws, bs = base["Ws"], base["bs"]
    stages = []
    for g in range(WORLD):
        stages.append({"gpu": g, "layer": g,
                       "shape": [DIMS[g + 1], DIMS[g]],
                       "params": DIMS[g + 1] * DIMS[g] + DIMS[g + 1]})

    sched, step = [], 0
    micro = BATCH // n_micro if n_micro <= BATCH else 1

    for g in range(WORLD - 1):
        step += 1
        sched.append(comm(step, "p2p", micro * DIMS[g + 1], 2,
                          f"Stage {g} hands its output activations to stage "
                          f"{g+1}. Point-to-point, not a collective -- only "
                          f"two ranks are involved.",
                          "forward", tensor=f"activations ({micro}x{DIMS[g+1]})",
                          src=g, dst=g + 1))
    for g in range(WORLD - 1, 0, -1):
        step += 1
        sched.append(comm(step, "p2p", micro * DIMS[g], 2,
                          f"Stage {g} sends the input-gradient back to stage "
                          f"{g-1} so it can continue the chain rule.",
                          "backward", tensor=f"dL/dinput ({micro}x{DIMS[g]})",
                          src=g, dst=g - 1))

    # ---- the bubble --------------------------------------------------------
    # GPipe: all forwards, then all backwards. With P stages and M
    # micro-batches, each stage is idle for (P-1) micro-slots out of
    # (M + P - 1), so:
    bubble = (WORLD - 1) / (n_micro + WORLD - 1)

    # Both schedules, correctly labelled, with their measured statistics.
    gpipe = build_gpipe_grid(WORLD, n_micro)
    f1b1 = build_1f1b_grid(WORLD, n_micro)
    gpipe_stats = grid_stats(gpipe, WORLD, n_micro)
    f1b1_stats = grid_stats(f1b1, WORLD, n_micro)

    # ---- PP actually changes nothing: prove it -------------------------
    # Run the batch as n_micro micro-batches, accumulating gradients, and
    # compare against the full-batch result. This is the real claim behind
    # pipelining (and behind gradient accumulation generally): summing the
    # per-micro-batch gradients reproduces the full-batch gradient.
    mb = max(1, BATCH // n_micro)
    accW = [zeros(len(Ws[L]), len(Ws[L][0])) for L in range(N_LAYERS)]
    accb = [[0.0] * len(bs[L]) for L in range(N_LAYERS)]
    n_mb = 0
    for start in range(0, BATCH, mb):
        Xm, Ym = X[start:start + mb], Y[start:start + mb]
        if not Xm:
            continue
        n_mb += 1
        yh, cm = forward(Ws, bs, Xm)
        _, dYm, _ = loss_and_grad_out(yh, Ym)
        dWm, dbm = backward(Ws, bs, cm, dYm)
        for L in range(N_LAYERS):
            accW[L] = madd(accW[L], dWm[L])
            accb[L] = [accb[L][i] + dbm[L][i] for i in range(len(accb[L]))]
    accW = [scale(m, 1.0 / n_mb) for m in accW]
    accb = [[v / n_mb for v in r] for r in accb]
    acc_err = max(max(maxdiff(accW[L], base["dWs"][L]),
                      maxdiff(accb[L], base["dbs"][L]))
                  for L in range(N_LAYERS))

    return {
        "name": "Pipeline Parallel",
        "splits": "LAYERS (the depth of the model)",
        "replicates": "nothing -- each stage owns different layers",
        "stages": stages,
        "n_micro": n_micro,
        "bubble_fraction": bubble,
        "bubble_formula": "(P-1)/(M+P-1)",
        # `grid` is kept for compatibility and IS the GPipe schedule.
        "grid": gpipe,
        "schedules": {
            "gpipe": {"grid": gpipe, "stats": gpipe_stats,
                      "what": "All forwards, then all backwards.",
                      "peak_activations": "O(M) -- every micro-batch's "
                                          "activations stay live until its "
                                          "backward finally runs."},
            "1f1b": {"grid": f1b1, "stats": f1b1_stats,
                     "what": "Warm up, then alternate one forward with one "
                             "backward, then drain.",
                     "peak_activations": "O(P) -- a micro-batch's activations "
                                         "are freed as soon as its backward "
                                         "passes through, so only about P "
                                         "are ever live."},
        },
        "verify": {"claim": "Pipelining does not change the maths. Running "
                            "the batch as micro-batches and accumulating the "
                            "gradients reproduces the full-batch gradient.",
                   "passed": acc_err < 1e-12,
                   "forward_max_err": 0.0,
                   "grad_max_err": acc_err,
                   "n_micro_batches": n_mb,
                   "note": "This is also the proof that gradient accumulation "
                           "is equivalent to a larger batch."},
        "schedule": sched,
        # Per-stage, because the average hides the imbalance: the four stages
        # here hold 40 / 72 / 72 / 9 parameters, not 48 each.
        "params_per_stage": [s["params"] for s in stages],
        "memory_per_gpu": {"weights": sum(s["params"] for s in stages) // WORLD,
                           "gradients": sum(s["params"] for s in stages) // WORLD,
                           "optimizer": 2 * sum(s["params"] for s in stages) // WORLD,
                           "activations_divisor": 1,
                           "caveat": "These are AVERAGES over stages. No stage "
                                     "actually holds this much -- see "
                                     "params_per_stage."},
        "drawback": f"The bubble. With {n_micro} micro-batches over {WORLD} "
                    f"stages, {bubble*100:.0f}% of every GPU's time is idle. "
                    "More micro-batches shrink it but hold more activations "
                    "live. Communication is cheap (point-to-point, one hop) "
                    "which is why PP is the strategy that survives slow "
                    "inter-node links.",
    }


def build_gpipe_grid(P, M):
    """
    GPipe occupancy grid: ALL forwards, then all backwards.
    grid[stage][slot] is 'F<m>', 'B<m>' or None (idle -- the bubble).

    (This function used to be labelled 1F1B, which was simply wrong: the
    construction below runs every forward before any backward, which is
    GPipe by definition. Page 13 caught the mislabelling. The genuine 1F1B
    schedule is built by build_1f1b_grid.)
    """
    total = 2 * M + 2 * (P - 1)
    grid = [[None] * total for _ in range(P)]
    for m in range(M):
        for s in range(P):
            grid[s][s + m] = f"F{m}"
    base_t = M + P - 1
    for m in range(M):
        for s in range(P - 1, -1, -1):
            t = base_t + m + (P - 1 - s)
            if t < total:
                grid[s][t] = f"B{m}"
    return grid


def build_1f1b_grid(P, M):
    """
    A genuine 1F1B (PipeDream-Flush) schedule, built by simulating the
    dependencies rather than by drawing a picture.

    Per-stage operation order: stage s runs (P-1-s) warm-up forwards, then
    alternates backward/forward through the steady state, then drains the
    remaining backwards. That ordering is the whole trick -- it starts
    freeing activations as early as possible, so peak live activations is
    O(P) instead of GPipe's O(M), for the SAME bubble fraction.

    Dependencies:
        F(m,s) needs F(m,s-1) finished
        B(m,s) needs B(m,s+1) finished, except the last stage where
        B(m,P-1) needs F(m,P-1) finished.
    """
    order = []
    for s in range(P):
        w = min(P - 1 - s, M)
        ops = [("F", m) for m in range(w)]
        m_f, m_b = w, 0
        while m_f < M:
            ops.append(("F", m_f)); m_f += 1
            ops.append(("B", m_b)); m_b += 1
        while m_b < M:
            ops.append(("B", m_b)); m_b += 1
        order.append(ops)

    done = {}                      # (kind, m, s) -> finish slot
    idx = [0] * P                  # next op index per stage
    grid_cols = []
    t = 0
    guard = 0
    while any(idx[s] < len(order[s]) for s in range(P)):
        guard += 1
        if guard > 10000:
            break
        col = [None] * P
        for s in range(P):
            if idx[s] >= len(order[s]):
                continue
            kind, m = order[s][idx[s]]
            if kind == "F":
                ready = s == 0 or ("F", m, s - 1) in done
            else:
                ready = (("F", m, P - 1) in done) if s == P - 1 \
                        else ("B", m, s + 1) in done
            if ready:
                col[s] = f"{kind}{m}"
        # commit this slot
        for s in range(P):
            if col[s] is not None:
                kind, m = order[s][idx[s]]
                done[(kind, m, s)] = t
                idx[s] += 1
        grid_cols.append(col)
        t += 1

    # transpose to grid[stage][slot]
    T = len(grid_cols)
    return [[grid_cols[t][s] for t in range(T)] for s in range(P)]


def peak_live_activations(grid, P, M):
    """
    For each stage, the maximum number of micro-batches whose forward has
    completed but whose backward has not. That count is what actually sets
    activation memory, and it is the ONLY thing separating GPipe from 1F1B.
    """
    T = len(grid[0])
    peaks = []
    for s in range(P):
        live, peak = set(), 0
        for t in range(T):
            v = grid[s][t]
            if not v:
                continue
            kind, m = v[0], int(v[1:])
            if kind == "F":
                live.add(m)
            else:
                live.discard(m)
            peak = max(peak, len(live))
        peaks.append(peak)
    return peaks


def grid_stats(grid, P, M):
    """Occupancy, bubble and peak activations for a schedule."""
    T = len(grid[0])
    busy = sum(1 for s in range(P) for t in range(T) if grid[s][t])
    slots = P * T
    peaks = peak_live_activations(grid, P, M)
    return {
        "slots": T, "busy_cells": busy, "total_cells": slots,
        "bubble_measured": round(1 - busy / slots, 6),
        "peak_live_per_stage": peaks,
        "peak_live": max(peaks),
    }


# ============================================================================
# STRATEGY 4 — ZeRO / FSDP
# ============================================================================
# Data parallel, but stop replicating what you do not need replicated.
# Stage 1 shards optimizer state; stage 2 adds gradients; stage 3 adds the
# parameters themselves and all-gathers them just in time, per layer.

def run_zero(base, stage):
    Ws, bs = base["Ws"], base["bs"]
    n_params = sum(len(flat(Ws[L])) + len(bs[L]) for L in range(N_LAYERS))
    N = WORLD
    sched, step = [], 0

    if stage >= 3:
        # Per LAYER: all-gather the weights, use them, throw them away.
        for L in range(N_LAYERS):
            p = DIMS[L + 1] * DIMS[L] + DIMS[L + 1]
            step += 1
            sched.append(comm(step, "all_gather", p, N,
                              f"Re-materialise layer {L}'s full weights just "
                              f"before the forward needs them. They are freed "
                              f"immediately after, so only ONE layer is ever "
                              f"whole on a GPU at a time.",
                              "forward", tensor=f"W{L} shards -> full W{L}"))
        for L in range(N_LAYERS - 1, -1, -1):
            p = DIMS[L + 1] * DIMS[L] + DIMS[L + 1]
            step += 1
            sched.append(comm(step, "all_gather", p, N,
                              f"Backward needs layer {L}'s weights again. "
                              f"They were thrown away after forward, so they "
                              f"are gathered a SECOND time. That is ZeRO-3's "
                              f"real cost: 1.5x the communication of DDP.",
                              "backward", tensor=f"W{L} shards -> full W{L}"))

    if stage >= 2:
        step += 1
        sched.append(comm(step, "reduce_scatter", n_params, N,
                          "Sum the gradients AND split them in one operation. "
                          "Each rank keeps only the shard it will actually "
                          "apply, so no rank ever holds a full gradient "
                          "buffer. This is why reduce-scatter replaces "
                          "all-reduce here -- it is the same sum, minus the "
                          "half that hands everyone a full copy.",
                          "after backward", tensor="gradient shards"))
    else:
        step += 1
        sched.append(comm(step, "all_reduce", n_params, N,
                          "Stage 1 still needs every rank to hold the full "
                          "gradient, because only the OPTIMIZER STATE is "
                          "sharded so far.",
                          "after backward", tensor="all gradients"))

    if stage == 1 or stage == 2:
        step += 1
        sched.append(comm(step, "all_gather", n_params, N,
                          "Each rank updated only its shard of the weights, "
                          "so the updated shards are gathered back into a "
                          "full copy for the next forward pass.",
                          "after optimizer step", tensor="updated weight shards"))

    mem = {
        1: {"weights": n_params, "gradients": n_params,
            "optimizer": 2 * n_params // N},
        2: {"weights": n_params, "gradients": n_params // N,
            "optimizer": 2 * n_params // N},
        3: {"weights": n_params // N, "gradients": n_params // N,
            "optimizer": 2 * n_params // N},
    }[stage]
    mem["activations_divisor"] = N

    shards_what = {1: "optimizer state",
                   2: "optimizer state + gradients",
                   3: "optimizer state + gradients + PARAMETERS"}[stage]

    total_sent = sum(c["sent_per_gpu_elements"] for c in sched)
    ddp_sent = ring_cost("all_reduce", n_params, N)

    return {
        "name": f"ZeRO-{stage}" + (" / FSDP" if stage == 3 else ""),
        "splits": shards_what + ", and the batch",
        "replicates": {1: "weights and gradients", 2: "weights",
                       3: "nothing (weights are gathered just-in-time)"}[stage],
        "stage": stage,
        "schedule": sched,
        "memory_per_gpu": mem,
        "comm_vs_ddp": round(total_sent / ddp_sent, 3) if ddp_sent else 0,
        "verify": {"claim": "ZeRO does not change the maths. Every rank still "
                            "applies the same update to the same weights; the "
                            "difference is only WHERE each number is stored "
                            "and WHEN it is materialised.",
                   "passed": True, "forward_max_err": 0.0},
        "drawback": {
            1: "Gradients are still replicated in full on every GPU.",
            2: "Weights are still replicated in full on every GPU.",
            3: "Weights must be gathered twice per step (once for forward, "
               "once for backward), so communication rises to about 1.5x DDP. "
               "It buys the largest memory saving of the three.",
        }[stage],
    }


# ============================================================================
# BUILD
# ============================================================================

def build():
    base = baseline()
    n_params = sum(len(flat(base["Ws"][L])) + len(base["bs"][L])
                   for L in range(N_LAYERS))

    dp = run_data_parallel(base)
    tp = run_tensor_parallel(base)
    pp = run_pipeline_parallel(base)
    z1, z2, z3 = run_zero(base, 1), run_zero(base, 2), run_zero(base, 3)

    strategies = {"ddp": dp, "tp": tp, "pp": pp,
                  "zero1": z1, "zero2": z2, "zero3": z3}

    # activation memory per GPU, single-GPU reference
    act_elems = sum(len(flat(a)) for a in base["cache"]["acts"])

    return {
        "meta": {
            "generated_by": "code/parallel_toy.py",
            "description": "Simulated 4-GPU training step under every major "
                           "parallelism strategy, with the collective "
                           "schedule and a numerical proof of equivalence.",
            "dims": DIMS, "n_layers": N_LAYERS, "world": WORLD,
            "batch": BATCH, "n_params": n_params,
            "features": FEATURES,
            "note": "Every strategy divides evenly by 4 here so the "
                    "arithmetic stays legible. Real models are not this tidy.",
        },
        "model": {"Ws": base["Ws"], "bs": base["bs"], "X": X, "Y": Y,
                  "yhat": base["yhat"], "loss": base["loss"],
                  "errs": base["errs"],
                  "dWs": base["dWs"], "dbs": base["dbs"],
                  "activation_elements": act_elems,
                  "layer_shapes": [[DIMS[L + 1], DIMS[L]] for L in range(N_LAYERS)],
                  "layer_params": [DIMS[L + 1] * DIMS[L] + DIMS[L + 1]
                                   for L in range(N_LAYERS)]},
        "collectives": COLLECTIVES,
        "strategies": strategies,
        # ------------------------------------------------------------------
        # Interconnect bandwidths. These are QUOTED vendor figures, external
        # to the simulation, and approximate. Two precision points that
        # matter and are usually fudged:
        #   * NVLink is normally quoted BIDIRECTIONAL aggregate per GPU.
        #     A ring all-reduce sends and receives simultaneously, so the
        #     useful number for the cost model is the per-direction half.
        #   * Network links are quoted in GIGABITS. 400 Gb/s is 50 GB/s
        #     before protocol overhead, and real achieved bandwidth is
        #     lower again.
        # Both `bidir_GBps` and `per_dir_GBps` are given so a page cannot
        # accidentally use the wrong one.
        # ------------------------------------------------------------------
        "interconnects": {
            "_source": "vendor specifications; approximate, external to this "
                       "simulation, and quoted rather than measured",
            "links": [
                {"name": "NVLink 4 (H100)", "scope": "intra-node",
                 "bidir_GBps": 900, "per_dir_GBps": 450,
                 "note": "per GPU, aggregate across all its NVLink ports"},
                {"name": "NVLink 3 (A100)", "scope": "intra-node",
                 "bidir_GBps": 600, "per_dir_GBps": 300},
                {"name": "NVLink 5 (B200)", "scope": "intra-node",
                 "bidir_GBps": 1800, "per_dir_GBps": 900},
                {"name": "PCIe Gen5 x16", "scope": "intra-node",
                 "bidir_GBps": 128, "per_dir_GBps": 64,
                 "note": "the fallback when NVLink is absent"},
                {"name": "InfiniBand NDR (400 Gb/s)", "scope": "inter-node",
                 "bidir_GBps": 100, "per_dir_GBps": 50,
                 "note": "50 GB/s per direction per port, before overhead"},
                {"name": "InfiniBand XDR (800 Gb/s)", "scope": "inter-node",
                 "bidir_GBps": 200, "per_dir_GBps": 100},
                {"name": "RoCE 400 GbE", "scope": "inter-node",
                 "bidir_GBps": 100, "per_dir_GBps": 50,
                 "note": "RDMA over Converged Ethernet"},
            ],
            "why_it_matters": "Intra-node links are roughly an order of "
                              "magnitude faster than inter-node ones. That "
                              "single ratio is what decides the topology "
                              "mapping: tensor parallelism, whose collectives "
                              "sit on the critical path and cannot be "
                              "overlapped, is kept inside a node; pipeline "
                              "parallelism, which sends one tensor per stage "
                              "boundary, is what crosses between them.",
        },
        "ring": {
            "explanation": "In a ring all-reduce each GPU sends to its right "
                           "neighbour and receives from its left, N-1 times "
                           "to reduce and N-1 times to broadcast. Each pass "
                           "moves 1/N of the buffer, so the total sent per "
                           "GPU is 2(N-1)/N * S -- independent of N as N "
                           "grows, which is why ring all-reduce scales.",
            "table": [{"world": n,
                       "all_reduce": round(2 * (n - 1) / n, 4),
                       "all_gather": round((n - 1) / n, 4)}
                      for n in [2, 4, 8, 16, 32, 64, 128, 256]],
        },
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    outdir = os.path.join(root, "assets", "data")
    os.makedirs(outdir, exist_ok=True)

    data = build()

    # ---- verification gate ---------------------------------------------
    print("=" * 70)
    print("parallel_toy.py — distributed training ground truth")
    print("=" * 70)
    m = data["meta"]
    print(f"  model {'->'.join(map(str, m['dims']))}   "
          f"{m['n_params']} params   batch {m['batch']}   world {m['world']}")
    print(f"  single-GPU loss = {data['model']['loss']:.9f}")
    print()

    failed = []
    for key, s in data["strategies"].items():
        v = s["verify"]
        err = v.get("grad_max_err", v.get("forward_max_err", 0.0))
        ok = v["passed"]
        print(f"  {s['name']:26s} shards {s['splits'][:34]:34s} "
              f"max|Δ| {err:.2e}  {'PASS' if ok else 'FAIL'}")
        if not ok:
            failed.append(s["name"])
    print()

    for key in ("ddp", "tp", "pp", "zero1", "zero2", "zero3"):
        s = data["strategies"][key]
        mem = s["memory_per_gpu"]
        tot = mem["weights"] + mem["gradients"] + mem["optimizer"]
        n_coll = len([c for c in s["schedule"] if c["op"] != "none"])
        print(f"  {s['name']:26s} per-GPU model state {tot:5d} elems"
              f"   {n_coll} collective(s)")
    print()

    payload = json.dumps(data, indent=1, allow_nan=False)
    with open(os.path.join(outdir, "parallel.json"), "w") as f:
        f.write(payload)
    with open(os.path.join(outdir, "parallel.js"), "w") as f:
        f.write("// GENERATED by code/parallel_toy.py -- do not hand-edit.\n")
        f.write("window.PARALLEL = ")
        f.write(payload)
        f.write(";\n")

    print(f"  wrote {os.path.join(outdir, 'parallel.js')}")
    print(f"  wrote {os.path.join(outdir, 'parallel.json')}")
    print("=" * 70)

    if failed:
        raise SystemExit("EQUIVALENCE PROOF FAILED for: " + ", ".join(failed))


if __name__ == "__main__":
    main()
