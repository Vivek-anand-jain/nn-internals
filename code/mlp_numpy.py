#!/usr/bin/env python3
"""
mlp_numpy.py — the same 2 -> 3 -> 1 MLP as ground_truth.py, vectorised.

Requires: numpy (tested against numpy 2.4.1). No torch, no autograd.
Status:   EXECUTED and PASSING in the authoring environment.

WHY THIS FILE EXISTS
--------------------
`ground_truth.py` writes every number the site displays into
`assets/data/trace.json`. It does so in pure Python: nested lists, explicit
`for` loops, one multiply at a time. That is readable but it is not how a
framework actually computes anything.

This file computes the identical quantities the way a framework does -- as
dense arrays hit with BLAS -- and then asserts, element by element, that the
answers agree with the trace to 1e-12. Two independent implementations of the
same chain rule landing on the same 13 gradients is real evidence that the
math on the site is right.

The direction of trust is one-way and non-negotiable:

    ground_truth.py  ->  trace.json  ->  (this file verifies against it)

If this script fails, this script is wrong. Never "fix" a failure by editing
ground_truth.py or the trace.

WHAT IS CHECKED
---------------
  forward     z1, a1, relu_mask, z2, yhat, error, loss, and every individual
              product inside z1_work / z2_work
  backward    dL_dyhat, dL_dz2, dL_dW2, dL_db2, dL_da1, dL_dz1, dL_dW1, dL_db1
  gradcheck   the flattened analytic gradient vs T.gradcheck.analytic
  training    all 12 steps x 3 optimizers: loss, yhat, theta, grad, theta_after,
              plus momentum velocity and Adam's m / v / m_hat / v_hat / update

Run:   python3 code/mlp_numpy.py
"""

import json
import os
import sys

import numpy as np

# ----------------------------------------------------------------------------
# TOLERANCES
# ----------------------------------------------------------------------------
# TOL is the contract stated in the project brief: every reproduced value must
# match the trace to 1e-12, absolute, element by element.
#
# Both implementations use IEEE-754 binary64. They differ only in the ORDER of
# the additions inside a dot product: ground_truth.py does Python's
# `sum([p0, p1])` left-to-right, numpy hands the same 2- or 3-element reduction
# to BLAS. Reassociating a sum of O(1) magnitude floats perturbs the result by
# a few ulp, i.e. ~1e-16. Twelve training steps of Adam amplify that a little.
# 1e-12 leaves four orders of magnitude of headroom, which is the point: it is
# tight enough that a real algebra error cannot sneak through, loose enough
# that summation order cannot cause a false alarm.
TOL = 1e-12

# The finite-difference gradient gets its own, much looser tolerance, and the
# reason is worth understanding rather than papering over. numerical_gradient()
# computes (L(theta+h) - L(theta-h)) / 2h with h = 1e-6. The two losses are
# both ~1.28 and differ in maybe the 7th decimal, so the subtraction cancels
# ~6 significant digits away; whatever is left is then divided by 2e-6, which
# multiplies the surviving rounding error by 500,000. A 1-ulp disagreement
# (2.2e-16) in either loss becomes ~1e-10 in the reported derivative.
#
# That is not a bug in either implementation. It is exactly why finite
# differences are a sanity check on backprop and never a replacement for it,
# and it is why frameworks ship analytic derivatives.
TOL_FINITE_DIFF = 1e-6


# ============================================================================
# LOAD THE TRACE
# ============================================================================

def load_trace():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.dirname(here), "assets", "data", "trace.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"missing {path}\n"
            "Generate it first:  python3 code/ground_truth.py"
        )
    with open(path) as f:
        return json.load(f)


# ============================================================================
# THE CHECKER
# ============================================================================
# Every comparison goes through here so the final table is uniform and so a
# single failure anywhere sets the exit code.

class Checker:
    """Accumulates named element-wise comparisons and prints one table."""

    def __init__(self):
        self.rows = []
        self.failures = 0

    def check(self, name, got, want, tol=TOL, note=""):
        got = np.asarray(got, dtype=np.float64)
        want = np.asarray(want, dtype=np.float64)

        if got.shape != want.shape:
            self.rows.append((name, f"{got.shape} vs {want.shape}", float("nan"),
                              tol, "FAIL", "shape mismatch"))
            self.failures += 1
            return got

        # Element by element, as asked. np.max of an empty array would throw,
        # so guard the (never-hit here, but cheap) zero-element case.
        diff = np.abs(got - want)
        max_err = float(diff.max()) if diff.size else 0.0
        ok = max_err <= tol
        self.rows.append((name, str(want.shape), max_err, tol,
                          "PASS" if ok else "FAIL", note))
        if not ok:
            self.failures += 1
        return got

    def section(self, title):
        self.rows.append((None, title, None, None, None, None))

    def report(self):
        name_w = max(len(r[0]) for r in self.rows if r[0]) + 2
        name_w = max(name_w, 22)
        head = (f"{'tensor':<{name_w}}{'shape':>10}{'max |Δ|':>13}"
                f"{'tol':>10}{'':>8}")
        line = "-" * (len(head) + 8)
        print(line)
        print(head + "status")
        print(line)
        for name, shape, err, tol, status, note in self.rows:
            if name is None:
                print()
                print(f"  {shape}")
                print("  " + "-" * (len(shape)))
                continue
            errs = "     n/a" if err is None else f"{err:.3e}"
            print(f"{name:<{name_w}}{shape:>10}{errs:>13}{tol:>10.0e}"
                  f"{'':>8}{status}" + (f"   {note}" if note else ""))
        print(line)
        n = sum(1 for r in self.rows if r[0])
        if self.failures:
            print(f"  {self.failures} of {n} checks FAILED")
        else:
            print(f"  all {n} checks PASS")
        print(line)
        return self.failures == 0


# ============================================================================
# THE MODEL, VECTORISED
# ============================================================================
# Shapes, fixed for the whole file:
#
#   x   (2,)      W1  (3, 2)   b1  (3,)
#   z1  (3,)      W2  (1, 3)   b2  (1,)
#   a1  (3,)      z2  (1,)     yhat scalar
#
# ground_truth.py writes `sum(row[j] * v[j] for j) + bias[i]` in a Python loop.
# Here that whole loop is one `W @ x + b`. The arithmetic is identical; only
# the loop moved from the interpreter into BLAS. On a GPU it would move again,
# into a cuBLAS GEMM kernel, and the identity would still hold.

def forward(W1, b1, W2, b2, x, y):
    """Returns every intermediate, because backward needs several of them."""
    z1 = W1 @ x + b1                    # (3,2) @ (2,) -> (3,)
    a1 = np.maximum(z1, 0.0)            # ReLU
    mask = (z1 > 0.0).astype(np.float64)  # d relu/dz, 1 bit per element

    z2 = W2 @ a1 + b2                   # (1,3) @ (3,) -> (1,)
    yhat = z2[0]                        # identity output: regression

    error = yhat - y
    loss = error * error                # squared error, matching MSELoss on 1 elem

    return {
        "z1": z1, "a1": a1, "relu_mask": mask, "z2": z2,
        "yhat": yhat, "error": error, "loss": loss,
        # The forward pass must hand these three to backward. Everything else
        # it computed can be freed. This is the entire "activation memory"
        # story in miniature -- see site page 07.
        "saved": {"x": x, "a1": a1, "relu_mask": mask},
    }


def backward(W2, fwd, x):
    """The chain rule, by hand. Same seven steps as ground_truth.backward(),
    with the loops replaced by array ops."""
    a1 = fwd["a1"]
    mask = fwd["relu_mask"]

    # Step 1: L = (yhat - y)^2  =>  dL/dyhat = 2(yhat - y)
    dL_dyhat = 2.0 * fwd["error"]

    # Step 2: yhat = z2 exactly, so the gradient passes through untouched.
    dL_dz2 = np.array([dL_dyhat])                    # (1,)

    # Step 3: z2 = W2 @ a1 + b2
    #   dL/dW2 = dL/dz2 (x) a1^T          <- READS THE SAVED ACTIVATION a1
    #   dL/db2 = dL/dz2
    # np.outer of a (1,) and a (3,) gives the (1,3) that matches W2's shape.
    dL_dW2 = np.outer(dL_dz2, a1)                    # (1,3)
    dL_db2 = dL_dz2.copy()                           # (1,)

    # Step 4: gradient crosses the layer boundary. dL/da1 = W2^T @ dL/dz2.
    # Note this reads the WEIGHTS, not the activations. A linear layer's
    # input-gradient depends on its weights; its weight-gradient depends on
    # its input. Both are needed, which is why both must be resident.
    dL_da1 = W2.T @ dL_dz2                           # (3,1)@(1,) -> (3,)

    # Step 5: through the ReLU. Elementwise multiply by the saved mask.
    # Hidden unit 2 had z1 = -0.8, so mask[2] = 0 and its gradient is exactly
    # zero -- it contributed nothing to the prediction and takes none of the
    # blame. The dead unit.
    dL_dz1 = dL_da1 * mask                           # (3,)

    # Step 6: z1 = W1 @ x + b1
    #   dL/dW1 = dL/dz1 (x) x^T           <- READS THE SAVED INPUT x
    dL_dW1 = np.outer(dL_dz1, x)                     # (3,2)
    dL_db1 = dL_dz1.copy()                           # (3,)

    return {
        "dL_dyhat": dL_dyhat, "dL_dz2": dL_dz2,
        "dL_dW2": dL_dW2, "dL_db2": dL_db2,
        "dL_da1": dL_da1, "dL_dz1": dL_dz1,
        "dL_dW1": dL_dW1, "dL_db1": dL_db1,
    }


# ============================================================================
# FLAT PARAMETER VECTOR
# ============================================================================
# Optimizers do not care about layer structure. Frameworks flatten every
# parameter into one contiguous buffer and step on that; ground_truth.py does
# the same with flatten()/unflatten(). Here the 13 slots live in one (13,)
# array and the four views are *views*, not copies -- W1 shares storage with
# theta[0:6], so writing through theta updates the model with no copy. That is
# exactly what a fused optimizer relies on.

SLICES = {
    "W1": (slice(0, 6), (3, 2)),
    "b1": (slice(6, 9), (3,)),
    "W2": (slice(9, 12), (1, 3)),
    "b2": (slice(12, 13), (1,)),
}


def unflatten(theta):
    """Four views into one flat buffer. No data is copied."""
    return tuple(theta[sl].reshape(shape) for sl, shape in
                 (SLICES["W1"], SLICES["b1"], SLICES["W2"], SLICES["b2"]))


def flatten(W1, b1, W2, b2):
    return np.concatenate([np.ravel(W1), np.ravel(b1),
                           np.ravel(W2), np.ravel(b2)])


# ============================================================================
# OPTIMIZERS, VECTORISED
# ============================================================================
# One line each, over all 13 parameters at once. Compare with the per-element
# Python loops in ground_truth.py: the update rule is identical, the memory
# footprint is what changes. SGD keeps nothing, momentum keeps one extra copy
# of the parameters, Adam keeps two. That 0x / 1x / 2x is the whole of site
# page 05 and the `--optimizer` flag in memory_accounting.py.

def sgd_step(theta, grad, lr):
    return theta - lr * grad


def momentum_step(theta, grad, vel, lr, beta=0.9):
    vel = beta * vel + grad
    return theta - lr * vel, vel


def adam_step(theta, grad, m, v, t, lr, b1, b2, eps):
    m = b1 * m + (1.0 - b1) * grad               # 1st moment
    v = b2 * v + (1.0 - b2) * grad * grad        # 2nd moment
    # Bias correction. t starts at 1. Computed with Python scalars so the
    # `b1 ** t` matches ground_truth.py's float.__pow__ exactly.
    m_hat = m / (1.0 - b1 ** t)
    v_hat = v / (1.0 - b2 ** t)
    update = lr * m_hat / (np.sqrt(v_hat) + eps)
    return theta - update, m, v, m_hat, v_hat, update


# ============================================================================
# FINITE-DIFFERENCE GRADIENT
# ============================================================================
# An independent check that does not share a single line of code with the
# chain rule above: nudge each parameter and watch the loss move.

def numerical_gradient(theta, x, y, h=1e-6):
    g = np.empty_like(theta)
    for i in range(theta.size):
        tp = theta.copy(); tp[i] += h
        tm = theta.copy(); tm[i] -= h
        lp = forward(*unflatten(tp), x, y)["loss"]
        lm = forward(*unflatten(tm), x, y)["loss"]
        g[i] = (lp - lm) / (2.0 * h)
    return g


# ============================================================================
# VERIFICATION
# ============================================================================

def main():
    T = load_trace()
    ck = Checker()

    x = np.array(T["forward"]["x"], dtype=np.float64)
    y = float(T["forward"]["y"])

    lr = T["meta"]["hyperparams"]["lr"]
    beta1 = T["meta"]["hyperparams"]["beta1"]
    beta2 = T["meta"]["hyperparams"]["beta2"]
    eps = T["meta"]["hyperparams"]["eps"]
    n_steps = T["meta"]["hyperparams"]["n_steps"]

    theta0 = flatten(T["init"]["W1"], T["init"]["b1"],
                     T["init"]["W2"], T["init"]["b2"])

    print("=" * 76)
    print("mlp_numpy.py — vectorised reproduction, verified against trace.json")
    print("=" * 76)
    print(f"  architecture : {T['meta']['architecture']}")
    print(f"  x = {x.tolist()}   y = {y}")
    print(f"  params       : {theta0.size}  (flat buffer, "
          f"{theta0.nbytes} bytes as float64)")
    print(f"  numpy        : {np.__version__}")
    print()

    # ---- FORWARD ---------------------------------------------------------
    W1, b1, W2, b2 = unflatten(theta0)
    fwd = forward(W1, b1, W2, b2, x, y)
    Tf = T["forward"]

    ck.section("FORWARD")
    ck.check("z1", fwd["z1"], Tf["z1"])
    ck.check("a1", fwd["a1"], Tf["a1"], note="unit 2 clamped to 0")
    ck.check("relu_mask", fwd["relu_mask"], Tf["relu_mask"])
    ck.check("z2", fwd["z2"], Tf["z2"])
    ck.check("yhat", fwd["yhat"], Tf["yhat"])
    ck.check("error", fwd["error"], Tf["error"])
    ck.check("loss", fwd["loss"], Tf["loss"])

    # The trace also records every individual multiply so the site can animate
    # them one at a time. Reproduce those too -- a dot product that lands on
    # the right total via the wrong products is still wrong.
    prods_want = [p["prod"] for row in Tf["z1_work"] for p in row["products"]]
    prods_got = [W1[i, j] * x[j] for i in range(3) for j in range(2)]
    ck.check("z1_work.products", prods_got, prods_want, note="6 scalar multiplies")
    ck.check("z1_work.sum_of_products",
             [float((W1[i] * x).sum()) for i in range(3)],
             [row["sum_of_products"] for row in Tf["z1_work"]])

    prods2_want = [p["prod"] for row in Tf["z2_work"] for p in row["products"]]
    prods2_got = [W2[0, j] * fwd["a1"][j] for j in range(3)]
    ck.check("z2_work.products", prods2_got, prods2_want, note="3 scalar multiplies")

    # ---- SAVED FOR BACKWARD ----------------------------------------------
    ck.section("SAVED FOR BACKWARD (the activation memory)")
    sfb = Tf["saved_for_backward"]
    ck.check("saved.x", fwd["saved"]["x"], sfb["x"], note="needed for dL/dW1")
    ck.check("saved.a1", fwd["saved"]["a1"], sfb["a1"], note="needed for dL/dW2")
    ck.check("saved.relu_mask", fwd["saved"]["relu_mask"], sfb["relu_mask"],
             note="needed for dL/dz1")

    # ---- BACKWARD --------------------------------------------------------
    bwd = backward(W2, fwd, x)
    Tb = T["backward"]

    ck.section("BACKWARD")
    ck.check("dL_dyhat", bwd["dL_dyhat"], Tb["dL_dyhat"])
    # ground_truth.py carries dL_dz2 as a bare Python float, because z2 is a
    # length-1 vector and it never needed the extra dimension. Here it is kept
    # as a (1,) array so it lines up with z2's shape, so unwrap for the
    # comparison. Same number, different container -- worth being explicit
    # about rather than silently broadcasting.
    ck.check("dL_dz2", bwd["dL_dz2"][0], Tb["dL_dz2"],
             note="trace holds this as a scalar")
    ck.check("dL_dW2", bwd["dL_dW2"], Tb["dL_dW2"])
    ck.check("dL_db2", bwd["dL_db2"], Tb["dL_db2"])
    ck.check("dL_da1", bwd["dL_da1"], Tb["dL_da1"])
    ck.check("dL_dz1", bwd["dL_dz1"], Tb["dL_dz1"], note="element 2 is exactly 0")
    ck.check("dL_dW1", bwd["dL_dW1"], Tb["dL_dW1"], note="row 2 is exactly [0, 0]")
    ck.check("dL_db1", bwd["dL_db1"], Tb["dL_db1"])

    # The dead unit is the pedagogical centrepiece, so assert it exactly
    # rather than to a tolerance. Zero times anything is zero in IEEE-754;
    # there is no rounding to hide behind here.
    assert fwd["relu_mask"][2] == 0.0, "unit 2 should be dead"
    assert bwd["dL_dz1"][2] == 0.0, "dead unit must have exactly zero gradient"
    assert np.array_equal(bwd["dL_dW1"][2], np.zeros(2)), \
        "dead unit's weight-gradient row must be exactly [0, 0]"

    # ---- GRADIENT CHECK --------------------------------------------------
    ck.section("GRADIENT CHECK")
    grad0 = flatten(bwd["dL_dW1"], bwd["dL_db1"], bwd["dL_dW2"], bwd["dL_db2"])
    ck.check("flat analytic grad", grad0, T["gradcheck"]["analytic"],
             note="13 slots, ordered as T.param_names")

    # Looser tolerance, deliberately. See TOL_FINITE_DIFF above.
    num = numerical_gradient(theta0, x, y)
    ck.check("finite-diff grad", num, T["gradcheck"]["numerical"],
             tol=TOL_FINITE_DIFF, note="cancellation-limited, see docstring")
    fd_err = float(np.abs(grad0 - num).max())

    # ---- TRAINING RUNS ---------------------------------------------------
    # Twelve steps, three optimizers, every recorded quantity checked -- not
    # just the loss history.
    for opt in ("sgd", "momentum", "adam"):
        ck.section(f"TRAINING RUN — {opt} ({n_steps} steps)")
        hist = T["runs"][opt]["history"]

        theta = theta0.copy()
        m = np.zeros_like(theta)
        v = np.zeros_like(theta)

        losses, yhats = [], []
        for t in range(1, n_steps + 1):
            h = hist[t - 1]
            W1, b1, W2, b2 = unflatten(theta)
            f = forward(W1, b1, W2, b2, x, y)
            bk = backward(W2, f, x)
            g = flatten(bk["dL_dW1"], bk["dL_db1"], bk["dL_dW2"], bk["dL_db2"])

            # State *before* the update, which is what the trace records.
            ck.check(f"  t={t:<2} theta", theta, h["theta"])
            ck.check(f"  t={t:<2} grad", g, h["grad"])
            ck.check(f"  t={t:<2} z1", f["z1"], h["z1"])
            ck.check(f"  t={t:<2} a1", f["a1"], h["a1"])
            ck.check(f"  t={t:<2} loss", f["loss"], h["loss"])
            ck.check(f"  t={t:<2} yhat", f["yhat"], h["yhat"])
            losses.append(f["loss"])
            yhats.append(f["yhat"])

            if opt == "sgd":
                theta = sgd_step(theta, g, lr)
            elif opt == "momentum":
                theta, v = momentum_step(theta, g, v, lr, beta=0.9)
                ck.check(f"  t={t:<2} velocity", v, h["velocity"])
            else:
                theta, m, v, m_hat, v_hat, upd = adam_step(
                    theta, g, m, v, t, lr, beta1, beta2, eps)
                ck.check(f"  t={t:<2} m", m, h["m"])
                ck.check(f"  t={t:<2} v", v, h["v"])
                # The trace stores Adam's per-parameter intermediates so the
                # site can animate them. Check those too.
                det = h["adam_detail"]
                ck.check(f"  t={t:<2} m_hat", m_hat, [d["m_hat"] for d in det])
                ck.check(f"  t={t:<2} v_hat", v_hat, [d["v_hat"] for d in det])
                ck.check(f"  t={t:<2} update", upd, [d["update"] for d in det])

            ck.check(f"  t={t:<2} theta_after", theta, h["theta_after"])

        # And the explicit contract from the brief: the loss history matches.
        ck.check(f"{opt}: loss history", losses, [e["loss"] for e in hist],
                 note=f"{n_steps} steps")
        ck.check(f"{opt}: yhat history", yhats, [e["yhat"] for e in hist])

        W1, b1, W2, b2 = unflatten(theta)
        fin = forward(W1, b1, W2, b2, x, y)
        ck.check(f"{opt}: final_loss", fin["loss"], T["runs"][opt]["final_loss"])
        ck.check(f"{opt}: final_yhat", fin["yhat"], T["runs"][opt]["final_yhat"])

    # ---- REPORT ----------------------------------------------------------
    print()
    ok = ck.report()

    print(f"  analytic vs finite-difference, recomputed here: "
          f"max |Δ| = {fd_err:.3e}")
    print(f"  trace's own gradcheck.max_abs_error:            "
          f"{T['gradcheck']['max_abs_error']:.3e}   "
          f"-> {'PASS' if T['gradcheck']['passed'] else 'FAIL'}")

    if ok:
        print()
        print("  Two independent implementations — pure-Python scalar loops and")
        print("  vectorised numpy — agree on all 13 gradients and all 36 training")
        print("  steps to within 1e-12. The site's numbers are sound.")
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
