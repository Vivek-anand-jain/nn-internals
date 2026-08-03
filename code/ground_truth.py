#!/usr/bin/env python3
"""
ground_truth.py — the single source of truth for every number in nn-internals.

Pure Python standard library. No numpy, no torch, no autograd.
Every derivative is written out by hand so you can read the chain rule
instead of trusting a framework.

It computes a full training run of a 2 -> 3 -> 1 MLP and emits:

    assets/data/trace.js    (window.TRACE = {...}   <- works over file://)
    assets/data/trace.json  (same payload, for tooling)

Every HTML page, every slide, and every animation in this project reads
from that emitted file. No number is ever typed twice. If you change a
hyperparameter here, the entire site changes with it.

Run:   python3 code/ground_truth.py
"""

import json
import math
import os
import struct

# ----------------------------------------------------------------------------
# THE PROBLEM
# ----------------------------------------------------------------------------
# Predict a house price from two features.
#
#   x[0] = size, in thousands of square feet
#   x[1] = number of bedrooms
#   y    = price, in hundreds of thousands of dollars
#
# Deliberately tiny numbers so the arithmetic stays legible on screen.

X = [2.0, 3.0]        # a 2000 sqft, 3-bedroom house
Y = 1.0               # it actually sold for $100k (normalized to 1.0)

# ----------------------------------------------------------------------------
# THE MODEL
# ----------------------------------------------------------------------------
#   z1 = W1 @ x  + b1        (3x2 @ 2  ->  3)
#   a1 = relu(z1)            (3        ->  3)
#   z2 = W2 @ a1 + b2        (1x3 @ 3  ->  1)
#   yhat = z2                (no activation on the output: this is regression)
#   L  = (yhat - y)^2        (squared error, matching torch.nn.MSELoss)
#
# 13 learnable parameters total: 6 + 3 + 3 + 1.
#
# These initial values are hand-picked, not random, for two reasons:
#   1. the forward pass lands on clean decimals
#   2. hidden unit 2 receives a NEGATIVE pre-activation, so ReLU zeroes it,
#      so its gradient is exactly zero. That dead unit is the clearest
#      possible demonstration of why backward needs the saved activation.

W1_INIT = [[0.5, -0.2],
           [-0.3, 0.8],
           [0.1, -0.4]]
B1_INIT = [0.1, -0.5, 0.2]

W2_INIT = [[0.7, -0.6, 0.9]]
B2_INIT = [0.3]

# ----------------------------------------------------------------------------
# HYPERPARAMETERS
# ----------------------------------------------------------------------------
LR = 0.1
BETA1 = 0.9           # Adam: decay for the 1st moment
BETA2 = 0.999         # Adam: decay for the 2nd moment
EPS = 1e-8
MOMENTUM_BETA = 0.9   # classical momentum: decay for the velocity
N_STEPS = 12          # how many training iterations to trace


# ============================================================================
# Small linear-algebra helpers, written out longhand.
#
# Each returns not just the answer but a `terms` list recording every
# individual multiply that went into it. The site animates those terms
# one at a time -- that is the whole point of building this by hand.
# ============================================================================

def matvec(M, v, bias):
    """out[i] = sum_j M[i][j] * v[j] + bias[i], keeping every product."""
    out, work = [], []
    for i, row in enumerate(M):
        products = [row[j] * v[j] for j in range(len(v))]
        total = sum(products) + bias[i]
        out.append(total)
        work.append({
            "index": i,
            "products": [
                {"w": row[j], "x": v[j], "prod": products[j]}
                for j in range(len(v))
            ],
            "sum_of_products": sum(products),
            "bias": bias[i],
            "out": total,
        })
    return out, work


def relu(v):
    return [x if x > 0 else 0.0 for x in v]


def relu_mask(v):
    """d relu / d z. This is the ONLY thing backward needs from the forward
    pass here -- 1 bit per element -- which is exactly why frameworks can
    store a boolean mask instead of the full float tensor."""
    return [1.0 if x > 0 else 0.0 for x in v]


def outer(col, row):
    """col (m) x row (n) -> m x n. This is how a weight gradient is formed."""
    return [[c * r for r in row] for c in col]


def matTvec(M, v):
    """M^T @ v -- pushes a gradient backwards through a linear layer."""
    n_out, n_in = len(M), len(M[0])
    return [sum(M[i][j] * v[i] for i in range(n_out)) for j in range(n_in)]


# ============================================================================
# FORWARD PASS
# ============================================================================

def forward(W1, b1, W2, b2, x, y):
    z1, z1_work = matvec(W1, x, b1)
    a1 = relu(z1)
    mask = relu_mask(z1)

    z2, z2_work = matvec(W2, a1, b2)
    yhat = z2[0]

    error = yhat - y
    loss = error ** 2

    return {
        "x": list(x),
        "y": y,
        "z1": z1, "z1_work": z1_work,
        "a1": a1, "relu_mask": mask,
        "z2": z2, "z2_work": z2_work,
        "yhat": yhat,
        "error": error,
        "loss": loss,
        # Tensors the backward pass will need to read back. Naming them
        # explicitly here is what page 04 uses to draw the "backward reads
        # this saved activation" arrows.
        "saved_for_backward": {
            "x": list(x),            # needed for dL/dW1
            "a1": a1,                # needed for dL/dW2
            "relu_mask": mask,       # needed for dL/dz1
        },
    }


# ============================================================================
# BACKWARD PASS  -- every step derived by hand
# ============================================================================

def backward(W1, b1, W2, b2, fwd):
    x = fwd["x"]
    a1 = fwd["a1"]
    mask = fwd["relu_mask"]
    steps = []

    # ---- Step 1: loss -> yhat -------------------------------------------
    # L = (yhat - y)^2   =>   dL/dyhat = 2*(yhat - y)
    dL_dyhat = 2.0 * fwd["error"]
    steps.append({
        "id": "dL_dyhat",
        "label": "dL/dŷ",
        "formula": "d/dŷ (ŷ − y)² = 2(ŷ − y)",
        "substitution": f"2 × ({fwd['yhat']:.6g} − {fwd['y']:.6g})",
        "value": dL_dyhat,
        "reads": [],
        "note": "The very first gradient. Everything downstream is this number, "
                "reshaped and rescaled by the chain rule.",
    })

    # ---- Step 2: yhat -> z2 ---------------------------------------------
    # yhat = z2 exactly (identity output), so the gradient passes straight
    # through untouched.
    dL_dz2 = dL_dyhat
    steps.append({
        "id": "dL_dz2",
        "label": "dL/dz2",
        "formula": "ŷ = z2  ⟹  dL/dz2 = dL/dŷ × 1",
        "substitution": f"{dL_dyhat:.6g} × 1",
        "value": dL_dz2,
        "reads": [],
        "note": "Identity output activation. The gradient is unchanged.",
    })

    # ---- Step 3: z2 -> W2, b2 -------------------------------------------
    # z2 = W2 @ a1 + b2
    #   dz2/dW2[0][j] = a1[j]     <-- REQUIRES THE SAVED ACTIVATION a1
    #   dz2/db2       = 1
    dL_dW2 = outer([dL_dz2], a1)
    dL_db2 = [dL_dz2]
    steps.append({
        "id": "dL_dW2",
        "label": "dL/dW2",
        "formula": "z2 = W2·a1 + b2  ⟹  dL/dW2 = dL/dz2 ⊗ a1ᵀ",
        "substitution": f"{dL_dz2:.6g} × {[round(v, 6) for v in a1]}",
        "value": dL_dW2,
        "reads": ["a1"],
        "note": "THIS is why activations are stored. To know how much W2 is to "
                "blame, you must know what W2 was multiplied by -- and that was "
                "a1, computed during the forward pass and kept in memory ever since.",
    })
    steps.append({
        "id": "dL_db2",
        "label": "dL/db2",
        "formula": "dz2/db2 = 1  ⟹  dL/db2 = dL/dz2",
        "substitution": f"{dL_dz2:.6g} × 1",
        "value": dL_db2,
        "reads": [],
        "note": "A bias adds itself in unchanged, so it receives the gradient unchanged.",
    })

    # ---- Step 4: z2 -> a1 ------------------------------------------------
    # dz2/da1[j] = W2[0][j], so dL/da1 = W2^T @ dL/dz2
    dL_da1 = matTvec(W2, [dL_dz2])
    steps.append({
        "id": "dL_da1",
        "label": "dL/da1",
        "formula": "dL/da1 = W2ᵀ · dL/dz2",
        "substitution": f"{[round(W2[0][j], 6) for j in range(3)]} × {dL_dz2:.6g}",
        "value": dL_da1,
        "reads": ["W2"],
        "note": "The gradient crosses the layer boundary. Note it reads the WEIGHTS "
                "here, not the activations -- a linear layer's input-gradient depends "
                "on its weights, its weight-gradient depends on its input.",
    })

    # ---- Step 5: a1 -> z1 (through ReLU) ---------------------------------
    # a1 = relu(z1), d relu/dz = 1 if z>0 else 0
    dL_dz1 = [dL_da1[j] * mask[j] for j in range(len(mask))]
    steps.append({
        "id": "dL_dz1",
        "label": "dL/dz1",
        "formula": "a1 = relu(z1)  ⟹  dL/dz1 = dL/da1 ⊙ 1[z1 > 0]",
        "substitution": f"{[round(v, 6) for v in dL_da1]} ⊙ {[int(m) for m in mask]}",
        "value": dL_dz1,
        "reads": ["relu_mask"],
        "note": "Hidden unit 2 had z1 = -0.8, so ReLU clamped it to 0 and its "
                "gradient is now exactly 0. It contributed nothing to the "
                "prediction, so it takes none of the blame. A dead unit.",
    })

    # ---- Step 6: z1 -> W1, b1 -------------------------------------------
    # z1 = W1 @ x + b1
    #   dz1/dW1[i][j] = x[j]      <-- REQUIRES THE SAVED INPUT x
    dL_dW1 = outer(dL_dz1, x)
    dL_db1 = list(dL_dz1)
    steps.append({
        "id": "dL_dW1",
        "label": "dL/dW1",
        "formula": "z1 = W1·x + b1  ⟹  dL/dW1 = dL/dz1 ⊗ xᵀ",
        "substitution": f"{[round(v, 6) for v in dL_dz1]} ⊗ {[round(v, 6) for v in x]}",
        "value": dL_dW1,
        "reads": ["x"],
        "note": "Same story as W2, one layer down: the weight gradient is the "
                "output gradient times the saved input. Row 2 is all zeros -- "
                "the dead ReLU unit propagated its zero backwards.",
    })
    steps.append({
        "id": "dL_db1",
        "label": "dL/db1",
        "formula": "dL/db1 = dL/dz1",
        "substitution": f"{[round(v, 6) for v in dL_dz1]}",
        "value": dL_db1,
        "reads": [],
        "note": "Biases again receive the gradient untouched.",
    })

    return {
        "dL_dyhat": dL_dyhat,
        "dL_dz2": dL_dz2,
        "dL_dW2": dL_dW2, "dL_db2": dL_db2,
        "dL_da1": dL_da1,
        "dL_dz1": dL_dz1,
        "dL_dW1": dL_dW1, "dL_db1": dL_db1,
        "steps": steps,
    }


# ============================================================================
# OPTIMIZERS
# ============================================================================
# Flatten the 13 parameters into one vector so the optimizer logic is
# written once, exactly as a real framework does it (a "flat parameter").

PARAM_LAYOUT = [
    ("W1", (3, 2)),
    ("b1", (3,)),
    ("W2", (1, 3)),
    ("b2", (1,)),
]


def flatten(W1, b1, W2, b2):
    out = []
    for row in W1:
        out.extend(row)
    out.extend(b1)
    for row in W2:
        out.extend(row)
    out.extend(b2)
    return out


def unflatten(flat):
    i = 0
    W1 = []
    for _ in range(3):
        W1.append(flat[i:i + 2]); i += 2
    b1 = flat[i:i + 3]; i += 3
    W2 = [flat[i:i + 3]]; i += 3
    b2 = flat[i:i + 1]; i += 1
    return W1, b1, W2, b2


def param_names():
    """Human-readable name for each of the 13 slots, e.g. 'W1[0][1]'."""
    names = []
    for i in range(3):
        for j in range(2):
            names.append(f"W1[{i}][{j}]")
    for i in range(3):
        names.append(f"b1[{i}]")
    for j in range(3):
        names.append(f"W2[0][{j}]")
    names.append("b2[0]")
    return names


def sgd_step(theta, grad, lr):
    return [theta[i] - lr * grad[i] for i in range(len(theta))]


def momentum_step(theta, grad, vel, lr, beta=MOMENTUM_BETA):
    vel = [beta * vel[i] + grad[i] for i in range(len(theta))]
    theta = [theta[i] - lr * vel[i] for i in range(len(theta))]
    return theta, vel


def adam_step(theta, grad, m, v, t, lr, b1, b2, eps):
    """Adam, spelled out. Returns the new params plus every intermediate
    quantity per-parameter, because page 05 animates them."""
    detail = []
    new_theta, new_m, new_v = [], [], []
    for i in range(len(theta)):
        g = grad[i]
        mi = b1 * m[i] + (1 - b1) * g            # 1st moment: mean of gradient
        vi = b2 * v[i] + (1 - b2) * g * g        # 2nd moment: mean of gradient^2
        m_hat = mi / (1 - b1 ** t)               # bias correction (t starts at 1)
        v_hat = vi / (1 - b2 ** t)
        update = lr * m_hat / (math.sqrt(v_hat) + eps)
        new_theta.append(theta[i] - update)
        new_m.append(mi)
        new_v.append(vi)
        detail.append({
            "g": g, "m": mi, "v": vi,
            "m_hat": m_hat, "v_hat": v_hat,
            "update": update,
            "before": theta[i], "after": theta[i] - update,
        })
    return new_theta, new_m, new_v, detail


# ============================================================================
# NUMERICAL GRADIENT CHECK
# ============================================================================
# Independent verification that the hand-derived chain rule above is correct.
# Perturb each parameter by +/- h and measure how the loss actually moves.
# If the analytic gradient is right, these agree to ~1e-9.

def numerical_gradient(theta, h=1e-6):
    grads = []
    for i in range(len(theta)):
        tp = list(theta); tp[i] += h
        tm = list(theta); tm[i] -= h
        Wp = unflatten(tp); Wm = unflatten(tm)
        lp = forward(Wp[0], Wp[1], Wp[2], Wp[3], X, Y)["loss"]
        lm = forward(Wm[0], Wm[1], Wm[2], Wm[3], X, Y)["loss"]
        grads.append((lp - lm) / (2 * h))
    return grads


# ============================================================================
# BYTE-LEVEL REPRESENTATION
# ============================================================================
# Page 01 shows what a float ACTUALLY is in HBM. These helpers expose the
# real IEEE-754 bit patterns rather than describing them in prose.

def fp32_bits(x):
    (u,) = struct.unpack(">I", struct.pack(">f", x))
    return format(u, "032b")


def bf16_bits(x):
    """bfloat16 = the top 16 bits of fp32, round-to-nearest-even.
    Same exponent range as fp32, 8 fewer mantissa bits."""
    (u,) = struct.unpack(">I", struct.pack(">f", x))
    lower = u & 0xFFFF
    upper = (u >> 16) & 0xFFFF
    # round to nearest even on the truncated half
    if lower > 0x8000 or (lower == 0x8000 and (upper & 1)):
        upper = (upper + 1) & 0xFFFF
    return format(upper, "016b")


def bf16_value(x):
    bits = bf16_bits(x)
    u = int(bits, 2) << 16
    (f,) = struct.unpack(">f", struct.pack(">I", u))
    return f


def fp16_bits(x):
    """fp16 has only 5 exponent bits, so values outside roughly
    [6.1e-5, 65504] cannot be represented. struct raises on overflow;
    we surface that as a real, traced fact rather than describing it."""
    try:
        (u,) = struct.unpack(">H", struct.pack(">e", x))
    except (OverflowError, ValueError):
        return None
    return format(u, "016b")


def decompose(x, kind):
    """Split a float into sign / exponent / mantissa for the bit-strip UI."""
    if kind == "fp32":
        b = fp32_bits(x)
        return {"dtype": "fp32", "bytes": 4, "bits": b,
                "sign": b[0], "exponent": b[1:9], "mantissa": b[9:],
                "exact": struct.unpack(">f", struct.pack(">f", x))[0]}
    if kind == "bf16":
        b = bf16_bits(x)
        return {"dtype": "bf16", "bytes": 2, "bits": b,
                "sign": b[0], "exponent": b[1:9], "mantissa": b[9:],
                "exact": bf16_value(x)}
    if kind == "fp16":
        b = fp16_bits(x)
        if b is None:
            # Genuinely unrepresentable: too large for fp16's 5-bit exponent.
            # `exact` is null rather than Infinity so that the emitted
            # trace.json stays STRICT JSON -- bare Infinity is accepted by
            # Python's json module but rejected by JSON.parse and by
            # Node's require(). Consumers should branch on `status`.
            return {"dtype": "fp16", "bytes": 2, "bits": None,
                    "sign": None, "exponent": None, "mantissa": None,
                    "exact": None,
                    "overflow_sign": 1 if x > 0 else -1,
                    "display": ("+inf" if x > 0 else "-inf"),
                    "status": "overflow"}
        exact = struct.unpack(">e", struct.pack(">e", x))[0]
        status = "ok"
        if x != 0 and exact == 0.0:
            status = "underflow"            # flushed to zero
        elif b[1:6] == "00000" and x != 0:
            status = "subnormal"            # representable but losing precision
        return {"dtype": "fp16", "bytes": 2, "bits": b,
                "sign": b[0], "exponent": b[1:6], "mantissa": b[6:],
                "exact": exact, "status": status}
    raise ValueError(kind)


# Exact format facts, derived from the bit-field widths rather than quoted.
def format_limits():
    """Range and precision of each format, computed not asserted."""
    out = {}
    for name, ebits, mbits, nbytes in (("fp32", 8, 23, 4),
                                       ("bf16", 8, 7, 2),
                                       ("fp16", 5, 10, 2)):
        bias = 2 ** (ebits - 1) - 1
        max_finite = (2 - 2 ** -mbits) * 2 ** bias
        min_normal = 2.0 ** (1 - bias)
        min_subnormal = min_normal * 2 ** -mbits
        out[name] = {
            "bytes": nbytes,
            "exponent_bits": ebits,
            "mantissa_bits": mbits,
            "bias": bias,
            "max_finite": max_finite,
            "min_normal": min_normal,
            "min_subnormal": min_subnormal,
            # machine epsilon: the gap between 1.0 and the next value up
            "epsilon": 2.0 ** -mbits,
            # roughly how many decimal digits survive
            "decimal_digits": round(mbits * math.log10(2), 2),
            "decades_of_range": round(math.log10(max_finite / min_normal), 1),
        }
    return out


# ============================================================================
# BUILD THE TRACE
# ============================================================================

def build():
    names = param_names()

    # ---- Step 0 in full, exhaustive detail -------------------------------
    W1, b1, W2, b2 = [r[:] for r in W1_INIT], B1_INIT[:], [r[:] for r in W2_INIT], B2_INIT[:]

    fwd0 = forward(W1, b1, W2, b2, X, Y)
    bwd0 = backward(W1, b1, W2, b2, fwd0)

    theta0 = flatten(W1, b1, W2, b2)
    grad0 = flatten(bwd0["dL_dW1"], bwd0["dL_db1"], bwd0["dL_dW2"], bwd0["dL_db2"])

    # verify the hand-derived gradients against finite differences
    num0 = numerical_gradient(theta0)
    max_err = max(abs(grad0[i] - num0[i]) for i in range(len(grad0)))

    # ---- A full training run with three optimizers -----------------------
    runs = {}
    for opt in ("sgd", "momentum", "adam"):
        theta = theta0[:]
        m = [0.0] * len(theta)
        v = [0.0] * len(theta)
        history = []
        for t in range(1, N_STEPS + 1):
            Wa, ba, Wb, bb = unflatten(theta)
            f = forward(Wa, ba, Wb, bb, X, Y)
            bk = backward(Wa, ba, Wb, bb, f)
            g = flatten(bk["dL_dW1"], bk["dL_db1"], bk["dL_dW2"], bk["dL_db2"])

            entry = {
                "t": t,
                "loss": f["loss"],
                "yhat": f["yhat"],
                "theta": theta[:],
                "grad": g[:],
                "z1": f["z1"], "a1": f["a1"], "relu_mask": f["relu_mask"],
            }

            if opt == "sgd":
                theta = sgd_step(theta, g, LR)
            elif opt == "momentum":
                theta, v = momentum_step(theta, g, v, LR)
                entry["velocity"] = v[:]
            else:
                theta, m, v, detail = adam_step(theta, g, m, v, t, LR, BETA1, BETA2, EPS)
                entry["m"] = m[:]
                entry["v"] = v[:]
                entry["adam_detail"] = detail

            entry["theta_after"] = theta[:]

            # The loss AFTER this step is applied. history[t]["loss"] is the
            # loss the step started from, which is a genuine footgun when
            # plotting -- without this you have to append final_loss by hand
            # to draw the last point.
            Wc, bc, Wd, bd = unflatten(theta)
            after = forward(Wc, bc, Wd, bd, X, Y)
            entry["loss_after"] = after["loss"]
            entry["yhat_after"] = after["yhat"]

            # How many hidden units are still alive. This is the single most
            # interesting derived quantity in the whole run: with lr=0.1 SGD
            # overshoots and kills ALL THREE hidden units by step 3, after
            # which the network is a constant predictor and only b2 learns.
            # Its tidy-looking loss curve is a collapsed network, not
            # convergence. Adam holds at 2 live units for all 12 steps.
            entry["n_live_units"] = int(sum(f["relu_mask"]))
            entry["relu_mask_after"] = after["relu_mask"]

            history.append(entry)

        Wa, ba, Wb, bb = unflatten(theta)
        final = forward(Wa, ba, Wb, bb, X, Y)
        runs[opt] = {"history": history, "final_loss": final["loss"],
                     "final_yhat": final["yhat"]}

    # ---- Memory accounting for THIS model --------------------------------
    n_params = len(theta0)
    n_activations = len(fwd0["z1"]) + len(fwd0["a1"]) + len(fwd0["z2"]) + len(X)

    memory = {
        "n_params": n_params,
        # NOTE: derived from per_tensor below, never hand-counted -- an
        # earlier hand-written value went stale the moment relu_mask was
        # added to the accounting. See the consistency assertions in main().
        "n_activation_elements": None,   # filled in after per_tensor is built
        # Bytes per element, so no page has to guess an itemsize.
        "dtype_bytes": {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1, "int8": 1},
        # Per-parameter byte cost of each realistic training configuration.
        # Stated as components so a page can show the arithmetic rather
        # than quoting a folk number like "16 bytes per parameter".
        "recipes": [
            {"name": "fp32 SGD",
             "components": {"weight": 4, "gradient": 4},
             "note": "the simplest possible setup"},
            {"name": "fp32 SGD + momentum",
             "components": {"weight": 4, "gradient": 4, "optimizer": 4}},
            {"name": "fp32 Adam",
             "components": {"weight": 4, "gradient": 4, "optimizer": 8}},
            {"name": "bf16 Adam, fp32 master, bf16 grads",
             "components": {"weight": 2, "gradient": 2, "optimizer": 12},
             "note": "optimizer = fp32 master 4 + m 4 + v 4; the common "
                     "'16 bytes per parameter' figure"},
            {"name": "bf16 Adam, fp32 master, fp32 grads",
             "components": {"weight": 2, "gradient": 4, "optimizer": 12},
             "note": "some frameworks accumulate gradients in fp32"},
            {"name": "bf16 8-bit Adam, fp32 master",
             "components": {"weight": 2, "gradient": 2, "optimizer": 6},
             "note": "optimizer = fp32 master 4 + m 1 + v 1 (quantised states)"},
        ],
        "per_tensor": [
            {"name": "W1", "shape": [3, 2], "elements": 6, "cls": "weight"},
            {"name": "b1", "shape": [3], "elements": 3, "cls": "weight"},
            {"name": "W2", "shape": [1, 3], "elements": 3, "cls": "weight"},
            {"name": "b2", "shape": [1], "elements": 1, "cls": "weight"},
            {"name": "x", "shape": [2], "elements": 2, "cls": "activation"},
            {"name": "z1", "shape": [3], "elements": 3, "cls": "activation"},
            {"name": "a1", "shape": [3], "elements": 3, "cls": "activation"},
            {"name": "z2", "shape": [1], "elements": 1, "cls": "activation"},
            # relu_mask is genuinely saved for backward (it is in
            # forward.saved_for_backward), so it belongs in the accounting.
            # Counted at fp32 here for consistency with everything else, but
            # a real framework stores it as one byte -- or one bit -- per
            # element, or re-derives it from the sign of z1.
            {"name": "relu_mask", "shape": [3], "elements": 3, "cls": "activation",
             "note": "1 byte/elem in practice, not 4"},
            # role distinguishes the gradients that PERSIST until the
            # optimizer step (one per parameter -- these are what "gradients
            # double your parameter memory" refers to) from the node
            # gradients that are consumed immediately and freed. Without
            # this, a page has to parse tensor names to tell them apart,
            # which is fragile.
            {"name": "dL/dW1", "shape": [3, 2], "elements": 6, "cls": "gradient",
             "role": "param_grad", "of": "W1"},
            {"name": "dL/db1", "shape": [3], "elements": 3, "cls": "gradient",
             "role": "param_grad", "of": "b1"},
            {"name": "dL/dW2", "shape": [1, 3], "elements": 3, "cls": "gradient",
             "role": "param_grad", "of": "W2"},
            {"name": "dL/db2", "shape": [1], "elements": 1, "cls": "gradient",
             "role": "param_grad", "of": "b2"},
            {"name": "m (Adam)", "shape": [13], "elements": 13, "cls": "optimizer"},
            {"name": "v (Adam)", "shape": [13], "elements": 13, "cls": "optimizer"},
        ],
    }

    # ---- the transient tensors backward creates on its way through -------
    # These are real allocations and they are exactly what makes the middle
    # of backward the co-resident window, so leaving them out understates
    # the peak. Added after page 07 pointed out the omission.
    memory["per_tensor"].extend([
        {"name": "loss", "shape": [1], "elements": 1, "cls": "activation",
         "note": "the scalar the whole backward pass hangs off"},
        {"name": "dL/dyhat", "shape": [1], "elements": 1, "cls": "gradient",
         "transient": True, "role": "node_grad"},
        {"name": "dL/dz2", "shape": [1], "elements": 1, "cls": "gradient",
         "transient": True, "role": "node_grad"},
        {"name": "dL/da1", "shape": [3], "elements": 3, "cls": "gradient",
         "transient": True, "role": "node_grad"},
        {"name": "dL/dz1", "shape": [3], "elements": 3, "cls": "gradient",
         "transient": True, "role": "node_grad"},
    ])

    # Tag the parameters themselves, so weight <-> gradient can be paired
    # without string surgery.
    for t in memory["per_tensor"]:
        if t["cls"] == "weight":
            t["role"] = "param"

    # ---- lifetime timeline -----------------------------------------------
    # One training iteration as an ordered list of phases, each naming the
    # tensors allocated and freed. Page 07 previously had to author this by
    # hand from read-dependencies; emitting it here makes that page
    # data-driven and keeps the phase story in one place.
    memory["timeline"] = [
        {"phase": 0,  "label": "idle",
         "desc": "Between iterations. Only the model and the optimizer's "
                 "memory of it are resident.",
         "alloc": [], "free": []},
        {"phase": 1,  "label": "load batch",
         "desc": "The input arrives on the device.",
         "alloc": ["x"], "free": []},
        {"phase": 2,  "label": "forward: linear 1",
         "desc": "z1 = W1·x + b1.",
         "alloc": ["z1"], "free": []},
        {"phase": 3,  "label": "forward: ReLU",
         "desc": "a1 = relu(z1). The mask is kept so backward knows which "
                 "units were active.",
         "alloc": ["a1", "relu_mask"], "free": []},
        {"phase": 4,  "label": "forward: linear 2",
         "desc": "z2 = W2·a1 + b2.",
         "alloc": ["z2"], "free": []},
        {"phase": 5,  "label": "loss",
         "desc": "Peak ACTIVATION memory. Every saved forward tensor is "
                 "still live and nothing has been freed yet.",
         "alloc": ["loss"], "free": [], "peak_activations": True},
        {"phase": 6,  "label": "backward: dL/dyhat, dL/dz2",
         "desc": "The first gradients appear.",
         "alloc": ["dL/dyhat", "dL/dz2"], "free": []},
        # RELEASE POLICY (one policy, applied consistently -- an earlier
        # version of this schedule mixed two and over-counted the peak by
        # 16 bytes):
        #   a saved tensor is freed at its LAST READ, because eager autograd
        #   calls release_variables() on each node as that node finishes --
        #   which is precisely why calling backward() twice without
        #   retain_graph=True raises. The exceptions are `out`/`loss`, which
        #   are live Python names for the whole iteration and cannot be
        #   freed mid-backward since `loss` is the root of the call still on
        #   the stack.
        {"phase": 7,  "label": "backward: dL/dW2, dL/db2",
         "desc": "Reads the saved a1 to form the weight gradient -- and that "
                 "is a1's last read, so it is released here, not at teardown.",
         "alloc": ["dL/dW2", "dL/db2"], "free": ["a1"], "reads": ["a1"]},
        {"phase": 8,  "label": "backward: dL/da1, dL/dz1",
         "desc": "Crosses the ReLU. Reads the mask, then releases it. "
                 "dL/da1 is consumed by dL/dz1 in this same phase.",
         "alloc": ["dL/da1", "dL/dz1"], "free": ["relu_mask", "dL/da1"],
         "reads": ["relu_mask", "W2"]},
        {"phase": 9,  "label": "backward: dL/dW1, dL/db1",
         "desc": "Reads the saved input x. All parameter gradients now "
                 "exist -- and this is the peak.",
         "alloc": ["dL/dW1", "dL/db1"], "free": ["dL/dyhat", "dL/dz2"],
         "reads": ["x"]},
        {"phase": 10, "label": "graph teardown",
         "desc": "backward() returns. The remaining forward tensors and the "
                 "loss scalar that rooted the call go out of scope.",
         "alloc": [], "free": ["x", "z1", "dL/dz1", "z2", "loss"]},
        {"phase": 11, "label": "optimizer step",
         "desc": "Reads the parameter gradients and the Adam moments, "
                 "updates the weights in place.",
         "alloc": [], "free": [], "reads": ["dL/dW1", "dL/db1", "dL/dW2",
                                            "dL/db2", "m (Adam)", "v (Adam)"]},
        {"phase": 12, "label": "zero_grad",
         "desc": "Gradients released. set_to_none=True actually frees them; "
                 "set_to_none=False only writes zeros and keeps the buffers.",
         "alloc": [], "free": ["dL/dW1", "dL/db1", "dL/dW2", "dL/db2"]},
        {"phase": 13, "label": "idle",
         "desc": "Back to the floor. Weights and optimizer state persist "
                 "into the next iteration; everything else was temporary.",
         "alloc": [], "free": []},
    ]

    # Per-class totals, so no page has to re-derive them in JS.
    by_class = {}
    for t in memory["per_tensor"]:
        c = by_class.setdefault(t["cls"], {"elements": 0, "tensors": 0, "names": []})
        c["elements"] += t["elements"]
        c["tensors"] += 1
        c["names"].append(t["name"])
    for c in by_class.values():
        c["bytes_fp32"] = c["elements"] * 4
    memory["by_class"] = by_class
    memory["total_elements"] = sum(c["elements"] for c in by_class.values())
    memory["total_bytes_fp32"] = memory["total_elements"] * 4
    # Derived, so it can never drift from per_tensor again.
    memory["n_activation_elements"] = by_class["activation"]["elements"]

    # The gradients that persist to the optimizer step. THIS is the number
    # that is 1:1 with parameters -- the full gradient class also contains
    # short-lived node gradients, so by_class.gradient.elements is larger
    # and does NOT equal n_params.
    memory["param_gradient_elements"] = sum(
        t["elements"] for t in memory["per_tensor"]
        if t.get("role") == "param_grad")
    memory["node_gradient_elements"] = sum(
        t["elements"] for t in memory["per_tensor"]
        if t.get("role") == "node_grad")

    # ---- Bit-level views of a few representative values ------------------
    bitviews = []
    bv_spec = [
        (0.5, "W1[0][0]", "a typical weight"),
        (-0.2, "W1[0][1]", "a negative weight"),
        (fwd0["loss"], "loss", "the loss"),
        (bwd0["dL_dz1"][0], "dL/dz1[0]", "a typical gradient"),
        (0.1, "learning rate", "a hyperparameter"),
        # The three cases below are why mixed precision is not just "use
        # smaller floats". They are traced, not described.
        (1.0e-8, "a tiny gradient",
         "fp16 FLUSHES THIS TO ZERO — bf16 keeps it. This is the underflow "
         "that loss scaling exists to prevent."),
        (3.0e-5, "a small gradient",
         "below fp16's smallest normal value, so fp16 stores it as a "
         "subnormal and loses most of its precision."),
        (70000.0, "a large activation",
         "beyond fp16's maximum finite value of 65504, so fp16 overflows to "
         "infinity. bf16 has fp32's exponent range and is unbothered."),
    ]
    for val, label, why in bv_spec:
        bitviews.append({
            "label": label,
            "value": val,
            "why": why,
            "fp32": decompose(val, "fp32"),
            "bf16": decompose(val, "bf16"),
            "fp16": decompose(val, "fp16"),
        })

    # ---- Row-major memory layout of W1 ------------------------------------
    # Explains strides and why transpose is free.
    w1_flat = [W1_INIT[i][j] for i in range(3) for j in range(2)]
    layout = {
        "tensor": "W1",
        "shape": [3, 2],
        "strides_elements": [2, 1],
        "strides_bytes_fp32": [8, 4],
        "order": "row-major (C contiguous)",
        "cells": [
            {"flat_index": i * 2 + j, "row": i, "col": j,
             "byte_offset": (i * 2 + j) * 4, "value": W1_INIT[i][j]}
            for i in range(3) for j in range(2)
        ],
        "flat": w1_flat,
    }

    trace = {
        "meta": {
            "generated_by": "code/ground_truth.py",
            "description": "Every number in nn-internals. Do not hand-edit.",
            "hyperparams": {"lr": LR, "beta1": BETA1, "beta2": BETA2,
                            "eps": EPS, "momentum_beta": MOMENTUM_BETA,
                            "n_steps": N_STEPS},
            "input": {"x": X, "y": Y,
                      "x_meaning": ["size (1000s sqft)", "bedrooms"],
                      "y_meaning": "price ($100k)"},
            "architecture": "2 -> 3 (ReLU) -> 1, MSE loss",
        },
        "init": {"W1": W1_INIT, "b1": B1_INIT, "W2": W2_INIT, "b2": B2_INIT},
        "param_names": names,
        "param_layout": [{"name": n, "shape": list(s)} for n, s in PARAM_LAYOUT],
        "forward": fwd0,
        "backward": bwd0,
        "gradcheck": {
            "analytic": grad0,
            "numerical": num0,
            "max_abs_error": max_err,
            "passed": max_err < 1e-6,
        },
        "runs": runs,
        "memory": memory,
        "bitviews": bitviews,
        "formats": format_limits(),
        "layout": layout,
        # Canonical real-world configurations, so every page cites the SAME
        # numbers instead of each inventing its own "4096". These are public
        # architecture figures, not values derived from our toy model -- they
        # are the only externally-sourced numbers in the trace, and they are
        # marked as such.
        "reference_configs": {
            "_source": "published model architecture parameters; external to "
                       "this model, quoted not derived",
            "models": [
                {"name": "GPT-2 small", "params": 124e6, "d_model": 768,
                 "n_layers": 12, "n_heads": 12, "seq": 1024, "vocab": 50257},
                {"name": "GPT-3 175B", "params": 175e9, "d_model": 12288,
                 "n_layers": 96, "n_heads": 96, "seq": 2048, "vocab": 50257},
                {"name": "Llama 3 8B", "params": 8.03e9, "d_model": 4096,
                 "n_layers": 32, "n_heads": 32, "n_kv_heads": 8,
                 "seq": 8192, "vocab": 128256, "d_ff": 14336},
                {"name": "Llama 3 70B", "params": 70.6e9, "d_model": 8192,
                 "n_layers": 80, "n_heads": 64, "n_kv_heads": 8,
                 "seq": 8192, "vocab": 128256, "d_ff": 28672},
                {"name": "Llama 3 405B", "params": 405e9, "d_model": 16384,
                 "n_layers": 126, "n_heads": 128, "n_kv_heads": 8,
                 "seq": 8192, "vocab": 128256, "d_ff": 53248},
            ],
            # bf16_dense_tflops is the DENSE rate. Vendors headline the
            # 2:4-structured-sparsity number, which is double and does not
            # apply to ordinary training -- quoting it silently overstates
            # throughput by 2x. hbm_bytes is exact (GiB), not the marketing
            # round number: an "80GB" H100 holds 80 GiB = 85.9 GB.
            "gpus": [
                {"name": "A100 40GB", "hbm_bytes": 40 * 1024**3,
                 "bf16_dense_tflops": 312, "hbm_bw_bytes_per_s": 1555e9,
                 "nvlink_gen": 3},
                {"name": "A100 80GB", "hbm_bytes": 80 * 1024**3,
                 "bf16_dense_tflops": 312, "hbm_bw_bytes_per_s": 2039e9,
                 "nvlink_gen": 3},
                {"name": "H100 80GB", "hbm_bytes": 80 * 1024**3,
                 "bf16_dense_tflops": 495, "hbm_bw_bytes_per_s": 3350e9,
                 "nvlink_gen": 4},
                {"name": "H200 141GB", "hbm_bytes": 141 * 1024**3,
                 "bf16_dense_tflops": 495, "hbm_bw_bytes_per_s": 4800e9,
                 "nvlink_gen": 4},
                {"name": "B200 192GB", "hbm_bytes": 192 * 1024**3,
                 "bf16_dense_tflops": 2250, "hbm_bw_bytes_per_s": 8000e9,
                 "nvlink_gen": 5},
            ],
            "mfu_note": "Real training reaches roughly 35-50% of the dense "
                        "peak (model FLOPs utilisation). Anything above ~55% "
                        "for a large transformer should be treated as "
                        "suspicious. Use a stated MFU rather than peak.",
            "default_model": "Llama 3 70B",
            "default_gpu": "H100 80GB",
        },
    }
    return trace


def _reject_constant(name):
    """Passed to json.loads so that Infinity / -Infinity / NaN in the emitted
    payload raise instead of silently parsing. Python's json accepts these by
    default; JSON.parse and Node's require() do not, so a Python-only check
    would not have caught the bug this guards against."""
    raise ValueError(f"non-standard JSON constant in payload: {name}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    outdir = os.path.join(root, "assets", "data")
    os.makedirs(outdir, exist_ok=True)

    trace = build()

    # ---- internal consistency checks -------------------------------------
    # These exist because two separate bugs shipped: a hand-written
    # n_activation_elements that went stale when a tensor was added, and a
    # bare Infinity that made the emitted .json unparseable by JSON.parse.
    # Both are now caught here rather than downstream.
    problems = []

    mem = trace["memory"]
    derived = sum(t["elements"] for t in mem["per_tensor"]
                  if t["cls"] == "activation")
    if mem["n_activation_elements"] != derived:
        problems.append(f"n_activation_elements {mem['n_activation_elements']}"
                        f" != per_tensor sum {derived}")
    if mem["by_class"]["weight"]["elements"] != mem["n_params"]:
        problems.append("weight elements != n_params")
    # The load-bearing claim of the whole project: one persisting gradient
    # per parameter. Assert it rather than trusting it.
    if mem["param_gradient_elements"] != mem["n_params"]:
        problems.append(f"param gradients {mem['param_gradient_elements']}"
                        f" != n_params {mem['n_params']}")
    # And every parameter gradient must name a parameter that exists.
    pnames = {t["name"] for t in mem["per_tensor"] if t.get("role") == "param"}
    for t in mem["per_tensor"]:
        if t.get("role") == "param_grad" and t.get("of") not in pnames:
            problems.append(f"{t['name']} claims to be the gradient of "
                            f"{t.get('of')!r}, which is not a parameter")

    # Every recipe's components must sum to its advertised bytes/param.
    for r in mem["recipes"]:
        r["bytes_per_param"] = sum(r["components"].values())

    # ---- timeline coherence ---------------------------------------------
    # Simulate the phases and make sure the schedule is actually consistent:
    # nothing is freed before it exists, nothing is allocated twice, every
    # transient is eventually freed, and the persistent classes survive.
    known = {t["name"]: t for t in mem["per_tensor"]}
    live = set()
    curve = []
    for ph in mem["timeline"]:
        for n in ph["alloc"]:
            if n not in known:
                problems.append(f"phase {ph['phase']} allocates unknown tensor {n}")
            elif n in live:
                problems.append(f"phase {ph['phase']} allocates {n} twice")
            live.add(n)
        for n in ph.get("reads", []):
            if n not in live and known.get(n, {}).get("cls") not in ("weight", "optimizer"):
                problems.append(f"phase {ph['phase']} reads {n} which is not live")
        for n in ph["free"]:
            if n not in live:
                problems.append(f"phase {ph['phase']} frees {n} which is not live")
            live.discard(n)
        # resident bytes = persistent classes + whatever is live right now
        persistent = sum(t["elements"] for t in mem["per_tensor"]
                         if t["cls"] in ("weight", "optimizer"))
        transient = sum(known[n]["elements"] for n in live if n in known)
        curve.append((persistent + transient) * 4)

    if live:
        problems.append(f"timeline leaks: {sorted(live)} still live at the end")

    mem["timeline_bytes_fp32"] = curve
    mem["peak_bytes_fp32"] = max(curve)
    mem["floor_bytes_fp32"] = curve[0]
    mem["peak_phase"] = curve.index(max(curve))

    # STRICT JSON: allow_nan=False raises on inf/nan, which is exactly the
    # failure we want at generation time rather than in a browser.
    try:
        payload = json.dumps(trace, indent=1, allow_nan=False)
    except ValueError as e:
        raise SystemExit(f"trace is not strict JSON: {e}")

    # And prove it round-trips through a strict parser.
    try:
        json.loads(payload, parse_constant=_reject_constant)
    except Exception as e:
        raise SystemExit(f"emitted JSON does not round-trip: {e}")

    if problems:
        for p in problems:
            print("  INCONSISTENT:", p)
        raise SystemExit("trace failed its internal consistency checks")

    with open(os.path.join(outdir, "trace.json"), "w") as f:
        f.write(payload)

    # A .js twin, because fetch() of a local .json is blocked by CORS when
    # the site is opened with file://. This keeps the whole project
    # double-click-able with no server.
    with open(os.path.join(outdir, "trace.js"), "w") as f:
        f.write("// GENERATED by code/ground_truth.py -- do not hand-edit.\n")
        f.write("window.TRACE = ")
        f.write(payload)
        f.write(";\n")

    # ---- Report ----------------------------------------------------------
    fwd = trace["forward"]
    gc = trace["gradcheck"]
    print("=" * 68)
    print("nn-internals ground truth")
    print("=" * 68)
    print(f"  x = {fwd['x']}   y = {fwd['y']}")
    print(f"  z1   = {[round(v, 6) for v in fwd['z1']]}")
    print(f"  a1   = {[round(v, 6) for v in fwd['a1']]}   (unit 2 killed by ReLU)")
    print(f"  yhat = {fwd['yhat']:.6f}")
    print(f"  loss = {fwd['loss']:.6f}")
    print()
    print(f"  dL/dW1 = {[[round(v, 6) for v in r] for r in trace['backward']['dL_dW1']]}")
    print(f"  dL/dW2 = {[[round(v, 6) for v in r] for r in trace['backward']['dL_dW2']]}")
    print()
    print(f"  gradcheck vs finite differences: max abs error = {gc['max_abs_error']:.3e}"
          f"  -> {'PASS' if gc['passed'] else 'FAIL'}")
    print()
    for opt in ("sgd", "momentum", "adam"):
        r = trace["runs"][opt]
        print(f"  {opt:9s} loss {r['history'][0]['loss']:.6f} -> {r['final_loss']:.6f}"
              f"   (yhat -> {r['final_yhat']:.4f}, target {Y})")
    print()
    print(f"  wrote {os.path.join(outdir, 'trace.js')}")
    print(f"  wrote {os.path.join(outdir, 'trace.json')}")
    print("=" * 68)

    if not gc["passed"]:
        raise SystemExit("GRADIENT CHECK FAILED -- the hand-derived math is wrong.")


if __name__ == "__main__":
    main()
