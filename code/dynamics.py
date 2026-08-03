#!/usr/bin/env python3
"""
dynamics.py — the training RUN, not the training step.

Pure Python standard library. No numpy, no torch.

Pages 01-05 take one iteration apart to the last multiply. Page 07 watches
memory rise and fall across that iteration. Nothing on the site has ever
shown what happens when you do it ten thousand times in a row, which is
where every real failure lives: the run that blows up at step 400, the
initialisation that made layer 30 arrive at zero, the clip that was
computed on the wrong shard.

This file trains the SAME 13-parameter MLP that ground_truth.py traces --
imported, not re-typed -- and measures five things that only exist over a
run:

  1. LEARNING-RATE SCHEDULES. Warmup, cosine, WSD, constant, emitted as
     full curves. Then the reason warmup exists, measured rather than
     asserted: the same peak learning rate, reached immediately, destroys
     the network; reached after 10 warmup steps, it trains. Same init,
     same data, same optimizer, same total steps.

  2. INITIALISATION. Activation variance propagated through a deep stack
     under Kaiming, Xavier, too-small and too-large, measured against the
     analytic prediction. Then residual scaling: L blocks each adding a
     branch make the stream grow as sqrt(L) unless the branches are scaled
     by 1/sqrt(2L).

  3. GRADIENT CLIPPING AS A COLLECTIVE. Global-norm clipping needs the norm
     over EVERY parameter, so under any sharding it is an all-reduce of a
     scalar before any rank may touch its own shard. We compute the true
     global norm and the four local norms four shards would compute, and
     show the coefficients disagree.

  4. LOSS SPIKES. A run that eats one mislabelled sample. The gradient norm
     peaks BEFORE the loss does, which is why practitioners watch the
     gradient norm and not the loss.

  5. WEIGHT DECAY vs L2. L2 added to the gradient is divided by sqrt(v),
     so it is not weight decay. Both are run on the same parameters and the
     trajectories separate.

Emits:
    assets/data/dynamics.js    (window.DYN = {...})
    assets/data/dynamics.json

Reads, for cross-checking:
    assets/data/trace.json     (the 13-parameter model this page trains)
    assets/data/tf2.json       (a real residual stream, measured)
    assets/data/parallel.json  (collective cost model for the clip all-reduce)

Run:  python3 code/dynamics.py
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "assets", "data")
sys.path.insert(0, HERE)

# The model itself. Imported so it can never drift from the traced one --
# forward, backward, flatten/unflatten, the 13 slot names and the exact
# hand-picked initial weights all come from the file that generated
# trace.js. main() then re-checks this import against the emitted
# trace.json, because "I imported it" is a claim, not a proof.
import ground_truth as GT                                     # noqa: E402


# ============================================================================
# THE DATA
# ============================================================================
# ground_truth.py trains on ONE house. That is right for showing the chain
# rule and wrong for showing a run, for a reason worth stating: with a
# single sample a network that has lost every hidden unit can still fit it
# perfectly, because the output bias alone is enough. Capacity costs
# nothing when there is nothing to fit.
#
# So here the batch is four houses. Sample 0 is EXACTLY ground_truth.py's
# (x, y), so the per-sample forward on it must reproduce trace.json's loss
# to the last bit -- asserted in main(). The other three are new, and they
# are what makes a dead network measurably worse than a live one.

BATCH = [
    {"x": [2.0, 3.0], "y": 1.0,  "label": "2000 sqft, 3 bed",
     "note": "the house ground_truth.py traces"},
    {"x": [1.2, 2.0], "y": 0.6,  "label": "1200 sqft, 2 bed"},
    {"x": [3.4, 4.0], "y": 1.9,  "label": "3400 sqft, 4 bed"},
    {"x": [0.8, 1.0], "y": 0.35, "label": "800 sqft, 1 bed"},
]

N_PARAMS = 13

# ---- hyperparameters, all in one place -------------------------------------
HORIZON = 60          # steps in a run; every schedule is defined over this
WARMUP = 10           # linear warmup steps
PEAK_LR = 0.5         # deliberately large: this is the whole demonstration
MIN_LR = 0.0
WSD_DECAY_FRAC = 0.2  # WSD spends the last 20% of the run decaying

BETA1 = GT.BETA1      # 0.9    -- same Adam constants page 05 animates
BETA2 = GT.BETA2      # 0.999
EPS = GT.EPS          # 1e-8

CLIP_MAX_NORM = 1.0   # a chosen hyperparameter, not a derived quantity
WORLD = 4             # shards, matching parallel.json's 4-GPU world
WEIGHT_DECAY = 0.1


# ============================================================================
# The model, wrapped for a batch
# ============================================================================

def theta_init():
    """The 13 initial values, flattened. Straight out of ground_truth.py."""
    return GT.flatten([r[:] for r in GT.W1_INIT], GT.B1_INIT[:],
                      [r[:] for r in GT.W2_INIT], GT.B2_INIT[:])


def forward_one(theta, x, y):
    W1, b1, W2, b2 = GT.unflatten(theta)
    return GT.forward(W1, b1, W2, b2, x, y)


def batch_loss_grad(theta, batch=None):
    """Mean squared error over the batch, and its gradient.

    The gradient of a mean is the mean of the gradients -- the same identity
    page 17 proves for data parallelism. Nothing here is new; it is
    ground_truth.py's backward(), summed and divided.
    """
    b = batch if batch is not None else BATCH
    W1, b1, W2, b2 = GT.unflatten(theta)
    total = 0.0
    grad = [0.0] * N_PARAMS
    live = 0
    for s in b:
        f = GT.forward(W1, b1, W2, b2, s["x"], s["y"])
        bk = GT.backward(W1, b1, W2, b2, f)
        g = GT.flatten(bk["dL_dW1"], bk["dL_db1"], bk["dL_dW2"], bk["dL_db2"])
        total += f["loss"]
        live += int(sum(f["relu_mask"]))
        for i in range(N_PARAMS):
            grad[i] += g[i]
    n = len(b)
    return total / n, [v / n for v in grad], live


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def finite(x):
    return isinstance(x, float) and math.isfinite(x)


# ============================================================================
# 1 · LEARNING-RATE SCHEDULES
# ============================================================================
# Four shapes. All of them are the same two ideas -- go up slowly, come down
# eventually -- and the differences between them are entirely about what you
# are allowed to do afterwards.

def lr_at(t, kind, n_steps=HORIZON, warmup=WARMUP, peak=PEAK_LR,
          min_lr=MIN_LR, decay_frac=WSD_DECAY_FRAC):
    """Learning rate at step t (0-based), for one of four schedules.

    The warmup segment is shared by all of them and is exactly linear:
    lr(t) = peak * (t+1)/warmup. The +1 matters -- a schedule that starts at
    exactly 0 wastes a step doing nothing.
    """
    if warmup > 0 and t < warmup:
        return peak * (t + 1) / warmup

    if kind == "constant":
        return peak

    # progress through the post-warmup portion, in [0, 1]
    span = max(1, n_steps - warmup - 1)
    p = (t - warmup) / span

    if kind == "cosine":
        return min_lr + (peak - min_lr) * 0.5 * (1.0 + math.cos(math.pi * p))

    if kind == "wsd":
        # Warmup-Stable-Decay: hold the peak, then decay only at the end.
        stable_until = 1.0 - decay_frac
        if p <= stable_until:
            return peak
        q = (p - stable_until) / decay_frac       # 0 -> 1 across the decay
        return peak + (min_lr - peak) * q          # linear decay

    if kind == "linear":
        return peak + (min_lr - peak) * p

    raise ValueError(kind)


SCHEDULES = [
    ("constant", "constant (with warmup)",
     "Warm up, then never come down. Simple, and it leaves the run sitting "
     "at a learning rate that is too large to settle into a minimum."),
    ("linear", "linear decay",
     "Warm up, then straight down to zero. The oldest schedule and still "
     "competitive; the whole run is spent decaying."),
    ("cosine", "cosine decay",
     "Warm up, then a half cosine to the floor. Spends longer near the peak "
     "than linear does and lands softly. The default for most of the last "
     "five years."),
    ("wsd", "WSD (warmup-stable-decay)",
     "Warm up, hold the peak for most of the run, decay only at the end. The "
     "point is not the shape: it is that the stable phase does not depend on "
     "the total step count, so the run can be extended or checkpointed and "
     "annealed at any length."),
]


def build_schedules():
    curves = []
    for key, name, desc in SCHEDULES:
        pts = [lr_at(t, key) for t in range(HORIZON)]
        curves.append({
            "key": key, "name": name, "desc": desc,
            "points": pts,
            "mean_lr": sum(pts) / len(pts),
            "final_lr": pts[-1],
            "peak_at_step": pts.index(max(pts)),
        })

    # ---- the claims, checked -------------------------------------------
    checks = []

    # (a) the warmup segment is EXACTLY linear: second differences vanish.
    w = curves[0]["points"][:WARMUP]
    second = [w[i + 2] - 2 * w[i + 1] + w[i] for i in range(len(w) - 2)]
    checks.append({
        "claim": "warmup is exactly linear (all second differences zero)",
        "measured": max(abs(v) for v in second),
        "tol": 1e-15, "passed": max(abs(v) for v in second) < 1e-15,
    })

    # (b) every schedule shares the warmup, so they agree until step W.
    diff = 0.0
    for c in curves:
        for t in range(WARMUP):
            diff = max(diff, abs(c["points"][t] - curves[0]["points"][t]))
    checks.append({
        "claim": "all four schedules are identical during warmup",
        "measured": diff, "tol": 1e-15, "passed": diff < 1e-15,
    })

    # (c) WSD's stable phase really is flat.
    wsd = dict((c["key"], c) for c in curves)["wsd"]["points"]
    stable = wsd[WARMUP:int(WARMUP + (HORIZON - WARMUP - 1) * (1 - WSD_DECAY_FRAC))]
    spread = max(stable) - min(stable)
    checks.append({
        "claim": "WSD's stable phase is flat",
        "measured": spread, "tol": 1e-15, "passed": spread < 1e-15,
    })

    # (d) cosine and linear both land on the floor; constant does not.
    cos = dict((c["key"], c) for c in curves)["cosine"]["points"]
    checks.append({
        "claim": "cosine ends at the floor learning rate",
        "measured": abs(cos[-1] - MIN_LR), "tol": 1e-12,
        "passed": abs(cos[-1] - MIN_LR) < 1e-12,
    })

    # (e) mean learning rate ordering. This is the practical difference:
    #     how much learning actually happens over the run.
    means = dict((c["key"], c["mean_lr"]) for c in curves)
    ordered = means["constant"] > means["wsd"] > means["cosine"]
    checks.append({
        "claim": "mean LR: constant > WSD > cosine",
        "measured": [round(means[k], 6) for k in
                     ("constant", "wsd", "cosine", "linear")],
        "passed": ordered,
    })

    # (f) and the one that surprises people: a half cosine and a straight
    #     line have the SAME average. Cosine is not "more learning rate"; it
    #     is the same total, redistributed -- more of it early, less at the
    #     end. Anyone claiming cosine works because it trains at a higher
    #     rate is describing WSD.
    gap = abs(means["cosine"] - means["linear"])
    checks.append({
        "claim": "cosine and linear decay have the same mean learning rate",
        "measured": {"cosine": means["cosine"], "linear": means["linear"],
                     "gap": gap},
        "passed": gap < 1e-12,
    })

    # ---- cosine has to know the finish line, WSD does not ---------------
    # Re-derive cosine for a LONGER run and compare the first HORIZON steps.
    # They differ everywhere after warmup, which is exactly why you cannot
    # extend a cosine run: the schedule you already ran was the wrong one.
    longer = HORIZON + 30
    cos_long = [lr_at(t, "cosine", n_steps=longer) for t in range(HORIZON)]
    wsd_long = [lr_at(t, "wsd", n_steps=longer) for t in range(HORIZON)]
    cos_gap = max(abs(cos_long[t] - cos[t]) for t in range(HORIZON))
    wsd_gap = max(abs(wsd_long[t] - wsd[t]) for t in range(HORIZON))
    checks.append({
        "claim": "cosine's shape depends on the total step count; WSD's "
                 "stable phase does not",
        "measured": {"cosine_max_gap": cos_gap, "wsd_max_gap_in_stable":
                     max(abs(wsd_long[t] - wsd[t]) for t in
                         range(WARMUP, int(HORIZON * 0.6)))},
        "passed": cos_gap > 0.05 * PEAK_LR,
    })

    return {
        "n_steps": HORIZON, "warmup_steps": WARMUP, "peak_lr": PEAK_LR,
        "min_lr": MIN_LR, "decay_frac": WSD_DECAY_FRAC,
        "curves": curves,
        "checks": checks,
        "horizon_dependence": {
            "short_steps": HORIZON, "long_steps": longer,
            "cosine_short": cos, "cosine_long_truncated": cos_long,
            "wsd_short": wsd, "wsd_long_truncated": wsd_long,
            "cosine_max_gap": cos_gap, "wsd_max_gap": wsd_gap,
            "why": "A cosine schedule is a function of t/N. Change N and "
                   "every step's learning rate changes, including steps you "
                   "have already run. That is why you cannot decide to train "
                   "longer halfway through a cosine run, and why WSD -- whose "
                   "stable phase is a constant -- can be branched and annealed "
                   "at any length from one shared trunk.",
        },
    }


# ============================================================================
# 1b · WHY WARMUP EXISTS — measured, not asserted
# ============================================================================
# The mechanism, stated precisely so it can be checked:
#
#   Adam's update is  lr * m_hat / (sqrt(v_hat) + eps).
#   At t = 1,  m_hat = g  and  v_hat = g^2,  so the update is
#
#       lr * g / (|g| + eps)  =  +/- lr
#
#   EXACTLY. The first Adam step moves every parameter with a non-zero
#   gradient by precisely the learning rate, in the sign of its gradient,
#   with the magnitude of the gradient divided out. It is a jump of size lr
#   in a direction estimated from a single sample.
#
#   At lr = 0.5, on parameters whose initial magnitudes are 0.1 to 0.9, that
#   is not a step. It is a re-initialisation.
#
# Warmup makes the first steps small enough that v has accumulated a real
# estimate before the schedule hands over full-size steps.

def adam_run(schedule_kind, warmup, peak, n_steps=HORIZON, theta=None,
             batch=None, weight_decay=0.0, decay_style="none",
             clip=None, batches=None, record_updates=False, state=None):
    """One Adam run. Returns a per-step history.

    `batches[t]` overrides the batch at step t (used by the loss-spike
    simulation). `decay_style` is 'none', 'l2' (added to the gradient) or
    'adamw' (applied to the weight, outside the adaptive scaling).
    `state` continues a previous run's Adam moments and step counter, so a
    spike can be shown landing on a run that is already under way rather
    than on a fresh initialisation.
    """
    th = list(theta if theta is not None else theta_init())
    m = list(state["m"]) if state else [0.0] * N_PARAMS
    v = list(state["v"]) if state else [0.0] * N_PARAMS
    t0 = state["t"] if state else 0
    hist = []
    dead = False
    for t in range(n_steps):
        b = batches[t] if batches is not None else batch
        L, g, live = batch_loss_grad(th, b)
        clean_L = batch_loss_grad(th, BATCH)[0] if batches is not None else L

        if not finite(L) or L > 1e12:
            dead = True
            hist.append({"t": t, "lr": 0.0, "loss": None, "clean_loss": None,
                         "grad_norm": None, "live_units": 0, "diverged": True})
            break

        gn = norm(g)
        lr = lr_at(t, schedule_kind, n_steps=n_steps, warmup=warmup, peak=peak)

        # global-norm clipping, if asked for
        coef = 1.0
        if clip is not None and gn > clip:
            coef = clip / gn
            g = [x * coef for x in g]

        if decay_style == "l2":
            g = [g[i] + weight_decay * th[i] for i in range(N_PARAMS)]

        upd = [0.0] * N_PARAMS
        step_t = t0 + t + 1
        for i in range(N_PARAMS):
            m[i] = BETA1 * m[i] + (1 - BETA1) * g[i]
            v[i] = BETA2 * v[i] + (1 - BETA2) * g[i] * g[i]
            mh = m[i] / (1 - BETA1 ** step_t)
            vh = v[i] / (1 - BETA2 ** step_t)
            step = lr * mh / (math.sqrt(vh) + EPS)
            if decay_style == "adamw":
                step += lr * weight_decay * th[i]
            upd[i] = step

        row = {
            "t": t, "lr": lr, "loss": L, "clean_loss": clean_L,
            "grad_norm": gn, "clip_coef": coef, "live_units": live,
            "theta": th[:], "mean_abs_update": sum(abs(u) for u in upd) / N_PARAMS,
            "rms_v": math.sqrt(sum(x for x in v) / N_PARAMS),
        }
        if record_updates:
            row["update"] = upd[:]
            row["grad"] = g[:]
        hist.append(row)
        th = [th[i] - upd[i] for i in range(N_PARAMS)]

    final_L, _, final_live = batch_loss_grad(th, BATCH)
    return {
        "history": hist, "theta_final": th,
        "final_loss": final_L if finite(final_L) else None,
        "final_live_units": final_live,
        "diverged": dead,
        "state": {"m": m, "v": v, "t": t0 + n_steps},
    }


def constant_predictor_loss():
    """The loss of the best possible CONSTANT prediction.

    A network with every hidden unit dead computes yhat = b2 for every
    input: it is a constant. So the floor such a network can reach is the
    variance of y. If a collapsed run lands exactly here, the collapse is
    proven rather than inferred.
    """
    ys = [s["y"] for s in BATCH]
    mu = sum(ys) / len(ys)
    return mu, sum((y - mu) ** 2 for y in ys) / len(ys)


def build_warmup_demo():
    # Run A: peak learning rate from step 0. Run B: same peak, reached over
    # WARMUP steps. Both then decay on the same cosine, over the same number
    # of steps, from the same 13 initial values, on the same four houses.
    run_none = adam_run("cosine", 0, PEAK_LR, record_updates=True)
    run_warm = adam_run("cosine", WARMUP, PEAK_LR, record_updates=True)

    mu_y, floor = constant_predictor_loss()

    def first_dead(run):
        for r in run["history"]:
            if r.get("live_units") == 0:
                return r["t"]
        return None

    # ---- the mechanism: Adam's first step is exactly lr ------------------
    r0 = run_none["history"][0]
    nonzero = [i for i in range(N_PARAMS) if abs(r0["grad"][i]) > 1e-14]
    zero = [i for i in range(N_PARAMS) if abs(r0["grad"][i]) <= 1e-14]
    max_dev = max(abs(abs(r0["update"][i]) - r0["lr"]) for i in nonzero)

    # ---- proof that the collapsed run is a CONSTANT ----------------------
    # Not "its loss looks like the floor" -- literally the same prediction
    # for every input, which is what a network with no live hidden unit is.
    yhats = [forward_one(run_none["theta_final"], s["x"], s["y"])["yhat"]
             for s in BATCH]
    yhats_warm = [forward_one(run_warm["theta_final"], s["x"], s["y"])["yhat"]
                  for s in BATCH]
    spread_none = max(yhats) - min(yhats)
    spread_warm = max(yhats_warm) - min(yhats_warm)

    def series(run, key):
        return [r[key] for r in run["history"] if r.get(key) is not None]

    peak_none = max(series(run_none, "loss"))
    peak_warm = max(series(run_warm, "loss"))

    return {
        "peak_lr": PEAK_LR, "warmup_steps": WARMUP, "n_steps": HORIZON,
        "schedule": "cosine", "optimizer": "adam",
        "betas": [BETA1, BETA2], "eps": EPS,
        "runs": {
            "none": {
                "label": "no warmup — peak LR on step 0",
                "lr": [r["lr"] for r in run_none["history"]],
                "loss": [r["loss"] for r in run_none["history"]],
                "grad_norm": [r["grad_norm"] for r in run_none["history"]],
                "live_units": [r["live_units"] for r in run_none["history"]],
                "theta": [r["theta"] for r in run_none["history"]],
                "final_loss": run_none["final_loss"],
                "final_live_units": run_none["final_live_units"],
                "peak_loss": peak_none,
                "steps_to_all_dead": first_dead(run_none),
            },
            "warm": {
                "label": f"{WARMUP}-step linear warmup — same peak LR",
                "lr": [r["lr"] for r in run_warm["history"]],
                "loss": [r["loss"] for r in run_warm["history"]],
                "grad_norm": [r["grad_norm"] for r in run_warm["history"]],
                "live_units": [r["live_units"] for r in run_warm["history"]],
                "theta": [r["theta"] for r in run_warm["history"]],
                "final_loss": run_warm["final_loss"],
                "final_live_units": run_warm["final_live_units"],
                "peak_loss": peak_warm,
                "steps_to_all_dead": first_dead(run_warm),
            },
        },
        "max_live_units": len(BATCH) * 3,
        "first_step": {
            "lr": r0["lr"],
            "grad": r0["grad"],
            "update": r0["update"],
            "moved_by_exactly_lr": [GT.param_names()[i] for i in nonzero],
            "not_moved_at_all": [GT.param_names()[i] for i in zero],
            "max_deviation_from_lr": max_dev,
            "why": "At t=1 Adam's bias correction makes m_hat = g and "
                   "v_hat = g^2, so the update is lr*g/(|g|+eps) = +/- lr for "
                   "every parameter with a non-zero gradient. The gradient's "
                   "MAGNITUDE cancels out entirely; only its sign survives. "
                   "The first Adam step is a jump of size lr in a direction "
                   "estimated from one batch.",
            "dead_unit_note": "The exceptions are the three slots belonging to "
                              "hidden unit 2 — the dead ReLU. Their gradient is "
                              "exactly zero, so m and v are exactly zero, and "
                              "0/(0+eps) = 0. The one part of the network Adam "
                              "cannot damage on step 1 is the part that was "
                              "already broken.",
        },
        "collapse": {
            "mean_y": mu_y,
            "constant_predictor_loss": floor,
            "why": "Every hidden unit dead means relu(z1) = 0 for every "
                   "sample, so yhat = b2 for every sample: the network is a "
                   "constant. The best constant is the mean of y and its loss "
                   "is the variance of y, so that number is the floor a "
                   "collapsed network converges to. It has not converged — it "
                   "has run out of model.",
            "yhat_none": yhats, "yhat_warm": yhats_warm,
            "y": [s["y"] for s in BATCH],
            "prediction_spread_none": spread_none,
            "prediction_spread_warm": spread_warm,
            "none_final_vs_floor": abs((run_none["final_loss"] or 0) - floor),
            "none_gap_pct": 100 * abs((run_none["final_loss"] or 0) - floor) / floor,
            "warm_final_vs_floor": abs((run_warm["final_loss"] or 0) - floor),
        },
        "verdict": {
            "final_none": run_none["final_loss"],
            "final_warm": run_warm["final_loss"],
            "ratio": (run_none["final_loss"] / run_warm["final_loss"]
                      if run_warm["final_loss"] else None),
            "peak_none": peak_none, "peak_warm": peak_warm,
            "loss_blowup_none": peak_none / run_none["history"][0]["loss"],
        },
    }


def build_designer_samples():
    """A handful of (warmup, peak, schedule) triples run here, so the page's
    own JavaScript re-implementation of this model can be checked against
    Python rather than trusted."""
    out = []
    for kind in ("cosine", "wsd", "constant"):
        for warmup, peak in ((0, PEAK_LR), (WARMUP, PEAK_LR),
                             (WARMUP, PEAK_LR / 5), (WARMUP * 3, PEAK_LR * 1.6)):
            r = adam_run(kind, warmup, peak)
            losses = [x["loss"] for x in r["history"] if x["loss"] is not None]
            out.append({
                "schedule": kind, "warmup": warmup, "peak_lr": peak,
                "n_steps": HORIZON,
                "final_loss": r["final_loss"],
                "peak_loss": max(losses),
                "final_live_units": r["final_live_units"],
            })
    return out


# ============================================================================
# 2 · INITIALISATION
# ============================================================================
# A deterministic generator, written out rather than imported, so this file
# has no hidden state and the numbers reproduce on any machine.

class Rng:
    """Numerical-Recipes LCG plus Box-Muller. Deterministic everywhere."""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF
        self._spare = None

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s + 0.5) / 4294967296.0

    def gauss(self):
        if self._spare is not None:
            g, self._spare = self._spare, None
            return g
        u1 = self.u()
        u2 = self.u()
        r = math.sqrt(-2.0 * math.log(u1))
        self._spare = r * math.sin(2 * math.pi * u2)
        return r * math.cos(2 * math.pi * u2)


VAR_WIDTH = 32
VAR_DEPTH = 10
VAR_SAMPLES = 64
VAR_SEEDS = 12       # independent stacks; see propagate_ensemble


def second_moment(rows):
    n = 0
    tot = 0.0
    for r in rows:
        for v in r:
            tot += v * v
            n += 1
    return tot / n


def matmul(Z, W):
    """(n x k) @ (k x m). Written out; k is small enough that this is fine."""
    m = len(W[0])
    out = []
    for row in Z:
        acc = [0.0] * m
        for i, zi in enumerate(row):
            if zi == 0.0:
                continue
            wi = W[i]
            for j in range(m):
                acc[j] += zi * wi[j]
        out.append(acc)
    return out


INIT_SCHEMES = [
    ("kaiming", "Kaiming / He", "sqrt(2 / fan_in)",
     lambda fi, fo: math.sqrt(2.0 / fi),
     "Derived FOR ReLU. The 2 is there because ReLU zeroes half the "
     "distribution, halving the variance; the initialisation puts it back."),
    ("xavier", "Xavier / Glorot", "sqrt(2 / (fan_in + fan_out))",
     lambda fi, fo: math.sqrt(2.0 / (fi + fo)),
     "Derived for a symmetric activation with unit slope at zero — tanh, or "
     "no activation at all. It balances the forward and backward passes, and "
     "it is missing ReLU's factor of two."),
    ("too_small", "half of Kaiming's variance", "sqrt(1 / fan_in)",
     lambda fi, fo: math.sqrt(1.0 / fi),
     "A plausible-looking guess. It loses half the signal per layer."),
    ("too_large", "twice Kaiming's variance", "sqrt(4 / fan_in)",
     lambda fi, fo: math.sqrt(4.0 / fi),
     "The same guess in the other direction. It doubles the signal per "
     "layer, which is an exponential."),
]


def propagate(scheme_std, activation, depth=VAR_DEPTH, width=VAR_WIDTH,
              n=VAR_SAMPLES, seed=12345):
    """Push n unit-variance samples through `depth` linear layers and measure
    the second moment of the PRE-activations at each one.

    The pre-activations are the right thing to track: they are symmetric and
    zero-mean, so their second moment IS their variance, and the analytic
    argument (Var_out = fan_in * Var_W * Var_in, halved by ReLU) is a
    statement about exactly this quantity.
    """
    rng = Rng(seed)
    Z = [[rng.gauss() for _ in range(width)] for _ in range(n)]
    moments = [second_moment(Z)]
    for _ in range(depth):
        std = scheme_std(width, width)
        W = [[rng.gauss() * std for _ in range(width)] for _ in range(width)]
        A = ([[v if v > 0 else 0.0 for v in row] for row in Z]
             if activation == "relu" else Z)
        Z = matmul(A, W)
        moments.append(second_moment(Z))
    return moments


def propagate_ensemble(scheme_std, activation, seeds=VAR_SEEDS, **kw):
    """The same measurement over `seeds` INDEPENDENT stacks, averaged.

    One stack of width 32 is not enough to see the law. ReLU makes every
    activation non-negative, which collapses the effective rank of the
    activation matrix, and a low-rank matrix pushed through a random square
    matrix has a gain that fluctuates by tens of percent -- so a single
    10-layer chain can land a factor of three away from its own expectation
    and prove nothing. Width is the real fix (a 4096-wide layer concentrates
    beautifully) and width is what we cannot afford in pure Python, so we
    average over independent draws instead and report the spread as well as
    the mean. The quantity being averaged, E[z^2], is exactly the one the
    analytic argument predicts.
    """
    runs = [propagate(scheme_std, activation, seed=1000 + 7919 * s, **kw)
            for s in range(seeds)]
    depth = len(runs[0])
    mean = [sum(r[i] for r in runs) / len(runs) for i in range(depth)]
    lo = [min(r[i] for r in runs) for i in range(depth)]
    hi = [max(r[i] for r in runs) for i in range(depth)]
    return {"mean": mean, "lo": lo, "hi": hi, "n_stacks": len(runs)}


def build_init():
    schemes = []
    for key, name, formula, stdf, why in INIT_SCHEMES:
        std = stdf(VAR_WIDTH, VAR_WIDTH)
        # Analytic prediction for the per-layer ratio of pre-activation
        # variance: fan_in * Var(W), halved by ReLU's zeroing of half the
        # inputs.
        pred_relu = VAR_WIDTH * std * std / 2.0
        pred_id = VAR_WIDTH * std * std
        ens_relu = propagate_ensemble(stdf, "relu")
        ens_id = propagate_ensemble(stdf, "identity")
        mom_relu = ens_relu["mean"]
        mom_id = ens_id["mean"]
        ratios = [mom_relu[i + 1] / mom_relu[i] for i in range(VAR_DEPTH)]
        schemes.append({
            "key": key, "name": name, "std_formula": formula,
            "std": std, "why": why,
            "predicted_ratio_relu": pred_relu,
            "predicted_ratio_identity": pred_id,
            "measured_relu": mom_relu,
            "measured_relu_lo": ens_relu["lo"], "measured_relu_hi": ens_relu["hi"],
            "measured_identity": mom_id,
            "measured_ratios_relu": ratios,
            "mean_measured_ratio_relu": sum(ratios) / len(ratios),
            "final_over_initial_relu": mom_relu[-1] / mom_relu[0],
            "predicted_final_relu": pred_relu ** VAR_DEPTH,
            "predicted_curve_relu": [mom_relu[0] * pred_relu ** i
                                     for i in range(VAR_DEPTH + 1)],
        })

    # ---- the 2x2 that IS the argument -----------------------------------
    # Kaiming is right for ReLU and wrong for a linear stack. Xavier is the
    # other way round. Nothing here is a matter of taste; it is one factor
    # of two, and it is measured.
    table = []
    by_key = dict((s["key"], s) for s in schemes)
    for key in ("kaiming", "xavier"):
        s = by_key[key]
        for act, pred, meas in (("relu", s["predicted_ratio_relu"], s["measured_relu"]),
                                ("identity", s["predicted_ratio_identity"], s["measured_identity"])):
            ratios = [meas[i + 1] / meas[i] for i in range(VAR_DEPTH)]
            mean_ratio = sum(ratios) / len(ratios)
            table.append({
                "scheme": key, "scheme_name": s["name"], "activation": act,
                "predicted_ratio": pred, "measured_ratio": mean_ratio,
                "rel_error": abs(mean_ratio - pred) / pred,
                "stable": abs(pred - 1.0) < 1e-9,
                "over_10_layers": meas[-1] / meas[0],
            })

    return {
        "width": VAR_WIDTH, "depth": VAR_DEPTH, "n_samples": VAR_SAMPLES,
        "n_stacks": VAR_SEEDS,
        "schemes": schemes,
        "matrix": table,
        "argument": "Var(z_out) = fan_in * Var(W) * Var(z_in) for a linear "
                    "layer. ReLU deletes the negative half of a symmetric "
                    "distribution, which halves the variance, so a ReLU stack "
                    "needs Var(W) = 2/fan_in to hold still. That factor of two "
                    "is the entire difference between Xavier and Kaiming, and "
                    "over ten layers it is a factor of 2^10.",
    }


# ============================================================================
# 2b · RESIDUAL SCALING
# ============================================================================
# x_{l+1} = x_l + f_l(x_l).
#
# The branch of a pre-LN block begins with a LayerNorm, so its output does
# NOT scale with the stream: however large x has grown, LN(x) has unit
# variance and the branch delivers roughly the same sigma^2 every time.
# (Without that normalisation the branch would be proportional to the
# stream, the variance would multiply rather than add, and the growth would
# be exponential rather than sqrt -- which is a large part of why pre-LN
# exists.) With it, the variances add:
#
#     Var(x_L) = Var(x_0) + L * sigma^2      ->  std grows as sqrt(L)
#
# Scale every branch by 1/sqrt(2L) and the sum telescopes to a constant:
#
#     Var(x_L) = Var(x_0) + L * sigma^2/(2L) = Var(x_0) + sigma^2/2
#
# independent of depth. That is the whole trick, and both halves are
# measured below rather than quoted.

RES_WIDTH = 16
RES_SAMPLES = 48
RES_SEEDS = 4
RES_DEPTHS = [1, 2, 4, 8, 16, 32]
LN_EPS = 1e-5


def layernorm_rows(X):
    out = []
    for row in X:
        n = len(row)
        mu = sum(row) / n
        var = sum((v - mu) ** 2 for v in row) / n
        inv = 1.0 / math.sqrt(var + LN_EPS)
        out.append([(v - mu) * inv for v in row])
    return out


def residual_stack(L, scale, seed=777, width=RES_WIDTH, n=RES_SAMPLES):
    """L residual blocks, each branch = W2 · ReLU(W1 · LayerNorm(x)), both
    matrices Kaiming-initialised, so an unscaled branch contributes about
    one unit of variance no matter how large the stream already is.
    `scale` multiplies the branch output before it is added back."""
    rng = Rng(seed)
    X = [[rng.gauss() for _ in range(width)] for _ in range(n)]
    trace = [second_moment(X)]
    for _ in range(L):
        s1 = math.sqrt(2.0 / width)
        W1 = [[rng.gauss() * s1 for _ in range(width)] for _ in range(width)]
        W2 = [[rng.gauss() * s1 for _ in range(width)] for _ in range(width)]
        H = matmul(layernorm_rows(X), W1)
        H = [[v if v > 0 else 0.0 for v in row] for row in H]
        B = matmul(H, W2)
        X = [[X[i][j] + scale * B[i][j] for j in range(width)]
             for i in range(n)]
        trace.append(second_moment(X))
    return trace


def residual_ensemble(L, scale, seeds=RES_SEEDS):
    runs = [residual_stack(L, scale, seed=331 + 6151 * s) for s in range(seeds)]
    return [sum(r[i] for r in runs) / len(runs) for i in range(len(runs[0]))]


def build_residual():
    # The per-block variance increment, measured once from an unscaled stack.
    unscaled = residual_ensemble(max(RES_DEPTHS), 1.0)
    incs = [unscaled[i + 1] - unscaled[i] for i in range(len(unscaled) - 1)]
    sigma2 = sum(incs) / len(incs)
    var0 = unscaled[0]

    measured = []
    for L in RES_DEPTHS:
        u = residual_ensemble(L, 1.0)[-1]
        s = residual_ensemble(L, 1.0 / math.sqrt(2.0 * L))[-1]
        measured.append({
            "L": L, "unscaled": u, "scaled": s,
            "unscaled_predicted": var0 + L * sigma2,
            "scaled_predicted": var0 + sigma2 / 2.0,
            "std_growth": math.sqrt(u / var0),
        })

    # Analytic curves for the depth slider, built from the MEASURED sigma^2
    # and the MEASURED starting variance. No fitted constants.
    curve = []
    for L in range(1, 65):
        curve.append({
            "L": L,
            "unscaled": var0 + L * sigma2,
            "scaled": var0 + sigma2 / 2.0,
            "scale_factor": 1.0 / math.sqrt(2.0 * L),
            "std_ratio": math.sqrt((var0 + L * sigma2) / var0),
        })

    worst = max(abs(m["unscaled"] - m["unscaled_predicted"]) /
                m["unscaled_predicted"] for m in measured)

    return {
        "width": RES_WIDTH, "n_samples": RES_SAMPLES, "n_stacks": RES_SEEDS,
        "depths": RES_DEPTHS,
        "var_in": var0, "sigma2_per_block": sigma2,
        "unscaled_trace": unscaled,
        "measured": measured, "curve": curve,
        "scale_formula": "1 / sqrt(2L)",
        "why_2L": "A pre-LN transformer block has TWO residual additions — "
                  "one after attention, one after the MLP — so a stack of L "
                  "blocks performs 2L additions. The scaling is 1/sqrt(2L) "
                  "because 2L is the number of branches, not the number of "
                  "blocks. GPT-2 applies it to the output projection of each "
                  "residual branch at initialisation.",
        "worst_rel_error": worst,
    }


def build_tf2_residual():
    """The residual stream of the site's own 2-layer transformer, measured.

    tf2.json is what pages 09 and 10 animate. Its stream is visible at four
    points: entering block 0, after block 0's attention addition, after
    block 0's MLP addition (= entering block 1), and so on. Four additions,
    two blocks -- exactly the 2L the scaling rule counts.
    """
    with open(os.path.join(DATA, "tf2.json")) as f:
        tf2 = json.load(f)

    pts = []
    layers = tf2["forward"]["layers"]
    pts.append({"label": "block 0 in", "matrix": layers[0]["stream_in"]})
    for li, lay in enumerate(layers):
        pts.append({"label": f"+ attn {li}", "matrix": lay["resid1"]})
        pts.append({"label": f"+ MLP {li}", "matrix": lay["resid2"]})

    out = []
    for i, p in enumerate(pts):
        out.append({"index": i, "label": p["label"],
                    "second_moment": second_moment(p["matrix"]),
                    "elements": len(p["matrix"]) * len(p["matrix"][0])})

    # Each addition, decomposed exactly:
    #
    #     E[(x + d)^2] = E[x^2] + 2 E[x d] + E[d^2]
    #
    # "The variances add" is the middle term being zero. It is an ASSUMPTION
    # -- that the branch output is uncorrelated with the stream it is added
    # to -- and in a trained network it is false, deliberately: the branch is
    # supposed to have learned something about the stream. Here it can be
    # measured instead of assumed.
    adds = []
    for i in range(len(pts) - 1):
        before = pts[i]["matrix"]
        after = pts[i + 1]["matrix"]
        rows, cols = len(before), len(before[0])
        delta = [[after[r][c] - before[r][c] for c in range(cols)]
                 for r in range(rows)]
        n = rows * cols
        e_x2 = second_moment(before)
        e_d2 = second_moment(delta)
        e_xd = sum(before[r][c] * delta[r][c]
                   for r in range(rows) for c in range(cols)) / n
        adds.append({
            "label": pts[i + 1]["label"],
            "before": e_x2, "after": second_moment(after),
            "branch": e_d2, "cross": e_xd,
            "reconstructed": e_x2 + 2 * e_xd + e_d2,
            "residual_error": abs(second_moment(after) - (e_x2 + 2 * e_xd + e_d2)),
            "branch_over_stream": e_d2 / e_x2,
        })

    growth = out[-1]["second_moment"] / out[0]["second_moment"]
    return {
        "n_layers": tf2["meta"]["n_layers"],
        "d_model": tf2["meta"]["d_model"], "seq": tf2["meta"]["seq"],
        "n_additions": len(out) - 1,
        "points": out, "additions": adds,
        "max_decomposition_error": max(a["residual_error"] for a in adds),
        "growth": growth,
        "std_growth": math.sqrt(growth),
        "mean_branch_over_stream": sum(a["branch_over_stream"] for a in adds) / len(adds),
        "cross_term_share": sum(abs(2 * a["cross"]) for a in adds) /
                            sum(abs(a["branch"]) for a in adds),
        "source": "assets/data/tf2.json, generated by "
                  "code/transformer_2layer.py — the same two blocks pages 09 "
                  "and 10 animate.",
        "reading": "Two blocks is four additions, and each branch contributes "
                   "only a few percent of the stream's second moment, so the "
                   "stream barely moves. Growth as sqrt(2L) is a statement "
                   "about L = 126, not L = 2. What this measurement is for is "
                   "the CROSS TERM: the whole sqrt(L) argument assumes the "
                   "branch is uncorrelated with the stream, and here that term "
                   "is visible rather than assumed away.",
        "caveat": "d_model is 4 and seq is 3, so each measurement point is 12 "
                  "numbers. Treat the constants as illustrative; the "
                  "decomposition identity is exact regardless.",
    }


# ============================================================================
# 3 · GRADIENT CLIPPING IS A COLLECTIVE
# ============================================================================
# Global-norm clipping:
#
#     g <- g * min(1, max_norm / ||g||_2)      with ||g|| over ALL parameters
#
# The norm is over the whole parameter vector. Under any sharding -- ZeRO,
# FSDP, tensor parallel, all of them -- no rank holds the whole vector, so
# no rank can evaluate that norm. What every rank CAN do is its own sum of
# squares, and
#
#     ||g||^2 = sum over shards of ||g_shard||^2
#
# so one all-reduce of ONE scalar is enough. It is also mandatory: it sits
# between the backward pass and the optimizer step, and nobody may proceed
# without it.

def shard_ranges(n, world):
    """FSDP-style flat-parameter sharding: pad up to a multiple of world,
    then cut into equal contiguous pieces. 13 parameters over 4 ranks gives
    4/4/4/1 with 3 slots of padding -- which is exactly what a real flat
    parameter does, and why FSDP reports a padded numel."""
    per = -(-n // world)                      # ceil
    out = []
    for r in range(world):
        lo = min(r * per, n)
        hi = min(lo + per, n)
        out.append((lo, hi, per - (hi - lo)))
    return out


def build_clipping(grad, where):
    names = GT.param_names()
    gn = norm(grad)
    global_coef = min(1.0, CLIP_MAX_NORM / gn)

    shards = []
    sumsq = 0.0
    for r, (lo, hi, pad) in enumerate(shard_ranges(N_PARAMS, WORLD)):
        piece = grad[lo:hi]
        ln = norm(piece)
        sumsq += ln * ln
        shards.append({
            "rank": r, "lo": lo, "hi": hi, "padding_slots": pad,
            "names": names[lo:hi], "grad": piece,
            "local_norm": ln, "local_norm_sq": ln * ln,
            "local_coef": min(1.0, CLIP_MAX_NORM / ln) if ln > 0 else 1.0,
        })

    # The identity that makes one scalar enough.
    ident_err = abs(math.sqrt(sumsq) - gn)

    # What each rank would do if it clipped on its own norm, and what that
    # does to the resulting update direction.
    wrong = []
    for s in shards:
        local_scaled = [g * s["local_coef"] for g in s["grad"]]
        right_scaled = [g * global_coef for g in s["grad"]]
        err = max((abs(local_scaled[i] - right_scaled[i])
                   for i in range(len(local_scaled))), default=0.0)
        rel = abs(s["local_coef"] - global_coef) / global_coef
        wrong.append({
            "rank": s["rank"], "local_coef": s["local_coef"],
            "global_coef": global_coef, "coef_rel_error": rel,
            "max_abs_grad_error": err,
            "local_scaled": local_scaled, "correct_scaled": right_scaled,
        })

    # A locally-clipped "gradient" is not a scaled version of the true one:
    # each shard gets a different factor, so the DIRECTION changes. That is
    # worse than clipping too much or too little.
    locally = []
    for s in shards:
        locally.extend([g * s["local_coef"] for g in s["grad"]])
    globally = [g * global_coef for g in grad]
    dot = sum(locally[i] * globally[i] for i in range(N_PARAMS))
    cos = dot / (norm(locally) * norm(globally))
    angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos))))

    # ---- what the all-reduce costs, from parallel.json -------------------
    with open(os.path.join(DATA, "parallel.json")) as f:
        par = json.load(f)
    ar = [c for c in par["collectives"] if c["op"] == "all_reduce"][0]
    ring_factor = ar["ring_factor_at_world"][str(WORLD)]
    link = [l for l in par["interconnects"]["links"]
            if l["name"].startswith("NVLink 4")][0]
    cm = par["cost_model"]
    bytes_fp32 = 4
    scalar_bytes = ring_factor * 1 * bytes_fp32
    full_bytes = ring_factor * N_PARAMS * bytes_fp32
    bw = link["per_dir_GBps"] * 1e9 * cm["achieved_fraction"]
    transfer_us = scalar_bytes / bw * 1e6
    lat_us = cm["latency_us"]["intra_node"]

    return {
        "max_norm": CLIP_MAX_NORM, "world": WORLD, "where": where,
        "grad": grad, "param_names": names,
        "global_norm": gn, "global_coef": global_coef,
        "clipped": global_coef < 1.0,
        "shards": shards,
        "identity": {
            "sum_of_shard_norm_squares": sumsq,
            "global_norm_from_sum": math.sqrt(sumsq),
            "global_norm_direct": gn,
            "abs_error": ident_err,
            "why": "The square of the global norm is the SUM of the squares of "
                   "the shard norms. Sums are what a reduction does. So the "
                   "entire distributed cost of global-norm clipping is one "
                   "all-reduce of one fp32 scalar, followed by a local sqrt.",
        },
        "local_clipping": {
            "per_rank": wrong,
            "worst_coef_rel_error": max(w["coef_rel_error"] for w in wrong),
            "ranks_that_would_not_clip_at_all":
                [w["rank"] for w in wrong if w["local_coef"] >= 1.0],
            "cosine_similarity": cos,
            "angle_degrees": angle_deg,
            "why": "Every rank computes a different coefficient, so the shards "
                   "are scaled by different factors. The result is not a "
                   "clipped gradient at all — it is a DIFFERENT DIRECTION, "
                   f"{angle_deg:.2f} degrees away from the one clipping was "
                   "supposed to produce. Clipping too hard would at least "
                   "still descend.",
        },
        "comm": {
            "_status": "COST MODEL, NOT MEASUREMENT — inherited from "
                       "parallel.json's cost_model, which says the same.",
            "op": "all_reduce", "payload_elements": 1,
            "ring_factor": ring_factor,
            "ring_formula": ar["ring_factor"],
            "bytes_sent_per_gpu": scalar_bytes,
            "bytes_if_full_gradient": full_bytes,
            "link": link["name"], "link_GBps": link["per_dir_GBps"],
            "achieved_fraction": cm["achieved_fraction"],
            "transfer_us": transfer_us,
            "latency_us": lat_us,
            "latency_over_transfer": lat_us / transfer_us,
            "why": "Four bytes. The transfer time is not the cost; the "
                   "SYNCHRONISATION is. Latency exceeds transfer time here by "
                   f"a factor of {lat_us / transfer_us:.3g}, and the collective "
                   "is a hard barrier between backward and the optimizer step: "
                   "no rank may scale its own shard until every rank has "
                   "reported. It is also why frameworks fuse it with anything "
                   "else that needs a global reduction — the loss scale check, "
                   "the NaN check — into one collective.",
        },
    }


# ============================================================================
# 4 · LOSS SPIKES
# ============================================================================
# A spike is not a mystery: something enters the batch that the model is
# very wrong about, the gradient it produces is enormous, the step it
# produces is enormous, and the loss on ordinary data is bad AFTERWARDS.
#
# The order matters and it is the whole reason people log the gradient norm.

SPIKE_STEP = 10
SPIKE_STEPS = 40
SPIKE_LR = 0.08
SPIKE_WARMSTART = 40      # steps of ordinary training before the recording


def bad_batch():
    """The same four houses, with one record mistyped: a $190k house
    recorded at $900k. A data error, not an act of God."""
    b = [dict(s) for s in BATCH]
    b[2] = dict(b[2])
    b[2]["y"] = 9.0
    b[2]["note"] = "MISLABELLED: 1.9 recorded as 9.0"
    return b


def warm_start():
    """Train normally first. A spike is something that happens to a run
    already under way; starting the recording at initialisation would make
    the largest loss in the series the one at step 0, which is not a spike,
    it is a beginning."""
    r = adam_run("cosine", 6, SPIKE_LR, n_steps=SPIKE_WARMSTART)
    return r["theta_final"], r["state"], r["final_loss"]


def spike_run(clip, theta, state):
    batches = [BATCH] * SPIKE_STEPS
    batches[SPIKE_STEP] = bad_batch()
    return adam_run("constant", 0, SPIKE_LR, n_steps=SPIKE_STEPS,
                    theta=theta, state=state, batches=batches, clip=clip,
                    record_updates=True)


def build_spike():
    theta, state, warm_loss = warm_start()
    off = spike_run(None, theta, state)
    on = spike_run(CLIP_MAX_NORM, theta, state)

    def col(run, key):
        return [r[key] for r in run["history"]]

    gnorm = col(off, "grad_norm")
    clean = col(off, "clean_loss")
    arg_g = gnorm.index(max(gnorm))
    arg_l = clean.index(max(clean))

    gnorm_on = col(on, "grad_norm")
    clean_on = col(on, "clean_loss")

    # ---- what clipping actually protects, with Adam ---------------------
    # Adam's step is bounded by roughly lr per parameter whether the
    # gradient is clipped or not, so clipping does not save the step. It
    # saves the SECOND MOMENT: v absorbs g^2, and with beta2 = 0.999 a
    # gradient a hundred times too large sits in v for hundreds of steps,
    # shrinking every subsequent update by 1/sqrt(v). The recovery is what
    # clipping shortens.
    base = sum(clean[:SPIKE_STEP]) / SPIKE_STEP

    def recovery(series):
        excess = sum(max(0.0, x - base) for x in series[SPIKE_STEP:])
        back = None
        for i in range(SPIKE_STEP + 1, len(series)):
            if series[i] <= 2 * base:
                back = i - SPIKE_STEP
                break
        return excess, back

    exc_off, back_off = recovery(clean)
    exc_on, back_on = recovery(clean_on)
    v_off = col(off, "rms_v")
    v_on = col(on, "rms_v")

    return {
        "n_steps": SPIKE_STEPS, "bad_step": SPIKE_STEP, "lr": SPIKE_LR,
        "warm_start_steps": SPIKE_WARMSTART, "warm_start_loss": warm_loss,
        "max_norm": CLIP_MAX_NORM,
        "bad_batch": bad_batch(),
        "baseline_loss": base,
        "grad_at_bad_step": off["history"][SPIKE_STEP]["grad"],
        "no_clip": {
            "loss": col(off, "loss"), "clean_loss": clean,
            "grad_norm": gnorm, "clip_ratio": col(off, "clip_coef"),
            "rms_v": v_off, "mean_abs_update": col(off, "mean_abs_update"),
            "final_loss": off["final_loss"],
        },
        "clipped": {
            "loss": col(on, "loss"), "clean_loss": clean_on,
            "grad_norm": gnorm_on, "clip_ratio": col(on, "clip_coef"),
            "rms_v": v_on, "mean_abs_update": col(on, "mean_abs_update"),
            "final_loss": on["final_loss"],
        },
        "recovery": {
            "baseline_loss": base,
            "excess_loss_no_clip": exc_off, "excess_loss_clipped": exc_on,
            "steps_back_to_2x_no_clip": back_off,
            "steps_back_to_2x_clipped": back_on,
            "v_at_spike_no_clip": v_off[SPIKE_STEP],
            "v_after_spike_no_clip": v_off[SPIKE_STEP + 1],
            "v_after_spike_clipped": v_on[SPIKE_STEP + 1],
            "v_inflation": v_off[SPIKE_STEP + 1] / v_on[SPIKE_STEP + 1],
            "why": "With Adam the immediate step is bounded by roughly the "
                   "learning rate whatever the gradient was, so the spike step "
                   "itself is survivable. What is not survivable is v: the bad "
                   "gradient is squared into the second moment, and with "
                   "beta2 = 0.999 it decays with a time constant of a thousand "
                   "steps. Every parameter it touched now takes smaller steps "
                   "for a long time. Clipping is what keeps that out of v.",
        },
        "ordering": {
            "grad_norm_peaks_at": arg_g,
            "loss_peaks_at": arg_l,
            "lead_steps": arg_l - arg_g,
            "grad_norm_peak": max(gnorm),
            "grad_norm_baseline": sorted(gnorm)[len(gnorm) // 2],
            "peak_over_median": max(gnorm) / sorted(gnorm)[len(gnorm) // 2],
            "loss_peak": max(clean),
            "why": "The bad sample raises the gradient norm on the step it "
                   "arrives. The loss on ordinary data cannot move until that "
                   "gradient has been applied, so it peaks one step later. The "
                   "gradient norm is a leading indicator and the loss is a "
                   "lagging one — which is why the gradient norm is the thing "
                   "worth alerting on.",
        },
        "clip_effect": {
            "peak_clean_loss_no_clip": max(clean),
            "peak_clean_loss_clipped": max(clean_on),
            "reduction": max(clean) / max(clean_on),
            "clip_ratio_at_spike": on["history"][SPIKE_STEP]["clip_coef"],
            "grad_norm_at_spike": gnorm[SPIKE_STEP],
            "why": "Clipping does not remove the bad sample. It scales the "
                   "gradient it produced by the ratio shown, which bounds both "
                   "the step and — the part that lasts — what goes into Adam's "
                   "second moment.",
        },
        "practice": {
            "_status": "PRACTICE, NOT DERIVED. Nothing below is computed by "
                       "this file. These are the standard responses; they are "
                       "heuristics that work, not theorems.",
            "items": [
                {"name": "skip the batch",
                 "what": "If the gradient norm exceeds some multiple of its "
                         "running median, throw the step away without applying "
                         "it. Cheap, local, and it costs one batch.",
                 "cost": "one batch of data, no restart"},
                {"name": "roll back and skip the data",
                 "what": "Restore the last checkpoint and resume with the "
                         "offending data range removed from the sampler. What "
                         "large runs actually do when a spike does not recover "
                         "on its own.",
                 "cost": "every step since the last checkpoint"},
                {"name": "lower the peak learning rate",
                 "what": "Restart the run — or the annealing phase — with a "
                         "smaller peak. Treats the cause when spikes are "
                         "chronic rather than a single bad shard.",
                 "cost": "a restart, and a slower run"},
                {"name": "lengthen warmup",
                 "what": "Spikes concentrated early are usually a warmup that "
                         "is too short for the batch size.",
                 "cost": "a restart"},
                {"name": "watch the gradient norm, not the loss",
                 "what": "Alert on the gradient norm's ratio to its running "
                         "median. This page measures why: the norm moves "
                         "first.",
                 "cost": "nothing"},
            ],
        },
    }


# ============================================================================
# 5 · WEIGHT DECAY vs L2, IN ADAM
# ============================================================================
# L2 regularisation adds  wd * theta  to the GRADIENT. Adam then divides the
# whole gradient by sqrt(v_hat), so the decay term is divided too:
#
#     step = lr * (g + wd*theta) / (sqrt(v_hat) + eps)
#
# The amount a parameter shrinks therefore depends on how big its gradients
# have been. That is not weight decay. Decoupled weight decay (AdamW) applies
# it to the weight directly, outside the adaptive scaling:
#
#     step = lr * g / (sqrt(v_hat) + eps)  +  lr * wd * theta
#
# The second term is the same proportional shrink for every parameter.

DECAY_STEPS = 20
DECAY_LR = 0.1


def build_decay():
    names = GT.param_names()
    common = dict(n_steps=DECAY_STEPS, record_updates=True)
    plain = adam_run("constant", 0, DECAY_LR, **common)
    l2 = adam_run("constant", 0, DECAY_LR, weight_decay=WEIGHT_DECAY,
                  decay_style="l2", **common)
    awd = adam_run("constant", 0, DECAY_LR, weight_decay=WEIGHT_DECAY,
                   decay_style="adamw", **common)

    # Per-parameter: how much of the FIRST step is attributable to the decay
    # term, under each rule. The AdamW column is lr*wd*theta by construction.
    # The L2 column is whatever 1/sqrt(v_hat) makes of it.
    per = []
    for i in range(N_PARAMS):
        th0 = plain["history"][0]["theta"][i]
        d_l2 = l2["history"][0]["update"][i] - plain["history"][0]["update"][i]
        d_wd = awd["history"][0]["update"][i] - plain["history"][0]["update"][i]
        g = plain["history"][0]["grad"][i]
        per.append({
            "name": names[i], "theta": th0, "grad": g,
            "l2_decay_displacement": d_l2,
            "adamw_decay_displacement": d_wd,
            "ratio": (abs(d_l2) / abs(d_wd)) if abs(d_wd) > 0 else None,
            "dead": abs(g) < 1e-14,
        })

    live = [p for p in per if not p["dead"] and p["ratio"] is not None]
    dead = [p for p in per if p["dead"] and p["ratio"] is not None]
    spread = (max(p["ratio"] for p in live) / min(p["ratio"] for p in live)
              if live else None)

    traj_gap = max(abs(l2["theta_final"][i] - awd["theta_final"][i])
                   for i in range(N_PARAMS))

    def wnorm(th):
        return norm(th)

    return {
        "weight_decay": WEIGHT_DECAY, "lr": DECAY_LR, "n_steps": DECAY_STEPS,
        "param_names": names,
        "runs": {
            "none": {"label": "Adam, no decay",
                     "theta": [r["theta"] for r in plain["history"]],
                     "final_loss": plain["final_loss"],
                     "final_weight_norm": wnorm(plain["theta_final"])},
            "l2": {"label": "Adam + L2 (decay added to the gradient)",
                   "theta": [r["theta"] for r in l2["history"]],
                   "final_loss": l2["final_loss"],
                   "final_weight_norm": wnorm(l2["theta_final"])},
            "adamw": {"label": "AdamW (decay applied to the weight)",
                      "theta": [r["theta"] for r in awd["history"]],
                      "final_loss": awd["final_loss"],
                      "final_weight_norm": wnorm(awd["theta_final"])},
        },
        "per_param": per,
        "spread_across_live_params": spread,
        "trajectory_gap": traj_gap,
        "dead_unit": {
            "params": [p["name"] for p in dead],
            "ratio": dead[0]["ratio"] if dead else None,
            "l2_displacement": dead[0]["l2_decay_displacement"] if dead else None,
            "adamw_displacement": dead[0]["adamw_decay_displacement"] if dead else None,
            "why": "Hidden unit 2's gradient is exactly zero — the dead ReLU "
                   "from page 04. Under L2 its gradient becomes wd*theta and "
                   "nothing else, so m_hat/sqrt(v_hat) is exactly sign(theta) "
                   "and the step is exactly lr. A parameter with no gradient "
                   "signal at all is moved by a FULL learning rate every step, "
                   "because Adam normalises away the very smallness that was "
                   "supposed to make the decay gentle. AdamW moves it by "
                   "lr*wd*theta, which is what was asked for.",
        },
        "claim": "L2 and decoupled weight decay are the same rule only when "
                 "sqrt(v_hat) is the same for every parameter, which it never "
                 "is. Under Adam, L2 decays the parameters with small "
                 "gradients hardest — the opposite of the intent, and the "
                 "reason AdamW exists.",
    }


# ============================================================================
# BUILD
# ============================================================================

def build():
    sched = build_schedules()
    warm = build_warmup_demo()
    init = build_init()
    resid = build_residual()
    tf2r = build_tf2_residual()
    spike = build_spike()
    clip = build_clipping(spike_run(None)["history"][SPIKE_STEP]["grad"],
                          f"the gradient of the mislabelled batch at step "
                          f"{SPIKE_STEP}")
    decay = build_decay()

    return {
        "meta": {
            "generated_by": "code/dynamics.py",
            "description": "Learning-rate schedules, initialisation, gradient "
                           "clipping as a collective, loss spikes, and AdamW — "
                           "measured on ground_truth.py's 13-parameter MLP.",
            "model": GT.__doc__.strip().splitlines()[0],
            "architecture": "2 -> 3 (ReLU) -> 1, MSE loss, 13 parameters",
            "batch": BATCH,
            "n_params": N_PARAMS,
            "horizon": HORIZON, "warmup": WARMUP, "peak_lr": PEAK_LR,
            "betas": [BETA1, BETA2], "eps": EPS,
            "clip_max_norm": CLIP_MAX_NORM, "weight_decay": WEIGHT_DECAY,
            "world": WORLD,
            "why_a_batch": "ground_truth.py trains on one house. A network "
                           "that has lost every hidden unit can still fit one "
                           "house perfectly, using the output bias alone, so a "
                           "single sample cannot show the difference between "
                           "converging and collapsing. Four houses can. Sample "
                           "0 is ground_truth.py's, unchanged, and its forward "
                           "pass is checked against trace.json.",
        },
        "schedules": sched,
        "warmup": warm,
        "designer_samples": build_designer_samples(),
        "init": init,
        "residual": resid,
        "tf2_residual": tf2r,
        "clipping": clip,
        "spike": spike,
        "decay": decay,
    }


# ============================================================================
# CHECKS AND EMIT
# ============================================================================

def check_model_against_trace():
    """The one claim everything else rests on: the model trained here is the
    model trace.json describes. Not a similar one."""
    with open(os.path.join(DATA, "trace.json")) as f:
        tr = json.load(f)

    th0 = theta_init()
    s0 = BATCH[0]
    f = forward_one(th0, s0["x"], s0["y"])
    loss_err = abs(f["loss"] - tr["forward"]["loss"])
    yhat_err = abs(f["yhat"] - tr["forward"]["yhat"])

    W1, b1, W2, b2 = GT.unflatten(th0)
    bk = GT.backward(W1, b1, W2, b2, f)
    g = GT.flatten(bk["dL_dW1"], bk["dL_db1"], bk["dL_dW2"], bk["dL_db2"])
    ana = tr["gradcheck"]["analytic"]
    grad_err = max(abs(g[i] - ana[i]) for i in range(N_PARAMS))

    theta_err = max(abs(th0[i] - tr["runs"]["adam"]["history"][0]["theta"][i])
                    for i in range(N_PARAMS))
    names_ok = GT.param_names() == tr["param_names"]

    return {
        "trace_loss": tr["forward"]["loss"], "our_loss": f["loss"],
        "loss_abs_error": loss_err,
        "trace_yhat": tr["forward"]["yhat"], "our_yhat": f["yhat"],
        "yhat_abs_error": yhat_err,
        "grad_max_abs_error": grad_err,
        "theta_max_abs_error": theta_err,
        "param_names_match": names_ok,
        "n_params": N_PARAMS,
        "passed": (loss_err == 0.0 and yhat_err == 0.0 and grad_err == 0.0
                   and theta_err == 0.0 and names_ok),
        "why": "Sample 0 of this file's batch is ground_truth.py's (x, y), and "
               "the initial parameters are ground_truth.py's. So the forward "
               "pass, the prediction and all 13 gradients must agree with "
               "trace.json exactly — not to a tolerance, bit for bit.",
    }


def _reject_constant(name):
    raise ValueError(f"non-standard JSON constant in payload: {name}")


def main():
    os.makedirs(DATA, exist_ok=True)

    model_check = check_model_against_trace()
    d = build()
    d["model_check"] = model_check

    results = []

    def check(name, passed, detail):
        results.append({"name": name, "passed": bool(passed), "detail": detail})

    # ---- 0 · the model is the traced model -------------------------------
    mc = model_check
    check("model reproduces trace.json exactly",
          mc["passed"],
          f"loss {mc['our_loss']:.6f} vs {mc['trace_loss']:.6f} "
          f"(|Δ| {mc['loss_abs_error']:.1e}), "
          f"13 gradients |Δ|max {mc['grad_max_abs_error']:.1e}")

    # ---- 1 · schedules ---------------------------------------------------
    for c in d["schedules"]["checks"]:
        check("schedule: " + c["claim"], c["passed"], str(c["measured"]))

    # ---- 1b · warmup -----------------------------------------------------
    w = d["warmup"]
    v = w["verdict"]
    check("Adam's first step moves every live parameter by exactly lr",
          w["first_step"]["max_deviation_from_lr"] / w["first_step"]["lr"] < 1e-7,
          f"lr {w['first_step']['lr']:.4f}, max |{'|Δθ|'} − lr| = "
          f"{w['first_step']['max_deviation_from_lr']:.2e} (Adam's eps) over "
          f"{len(w['first_step']['moved_by_exactly_lr'])} parameters; "
          f"{len(w['first_step']['not_moved_at_all'])} zero-gradient slots "
          f"move 0")
    check("no-warmup run collapses: every hidden unit dies",
          w["runs"]["none"]["steps_to_all_dead"] is not None,
          f"all {w['max_live_units']} unit-activations dead by step "
          f"{w['runs']['none']['steps_to_all_dead']}")
    check("warmup run keeps hidden units alive",
          w["runs"]["warm"]["final_live_units"] > 0,
          f"{w['runs']['warm']['final_live_units']} of {w['max_live_units']} "
          f"unit-activations alive at the end")
    check("the collapsed network predicts the SAME number for all four houses",
          w["collapse"]["prediction_spread_none"] == 0.0
          and w["collapse"]["prediction_spread_warm"] > 0.1,
          f"no-warmup ŷ spread {w['collapse']['prediction_spread_none']:.1e} "
          f"(it is a constant); warmup ŷ spread "
          f"{w['collapse']['prediction_spread_warm']:.4f}")
    check("the collapsed run converges to the constant-predictor floor",
          w["collapse"]["none_gap_pct"] < 5,
          f"final {v['final_none']:.6f} vs var(y) "
          f"{w['collapse']['constant_predictor_loss']:.6f} "
          f"({w['collapse']['none_gap_pct']:.2f}% above it, and still "
          f"falling towards it)")
    check("same peak LR: warmup converges, no warmup does not",
          v["ratio"] is not None and v["ratio"] > 20,
          f"final loss {v['final_none']:.6f} without warmup vs "
          f"{v['final_warm']:.6f} with — {v['ratio']:.1f}x")

    # ---- 2 · initialisation ---------------------------------------------
    for row in d["init"]["matrix"]:
        check(f"variance: {row['scheme']} + {row['activation']} measured "
              f"ratio matches prediction",
              row["rel_error"] < 0.12,
              f"predicted {row['predicted_ratio']:.3f}, measured "
              f"{row['measured_ratio']:.3f} ({row['rel_error'] * 100:.1f}%)")
    byk = dict((s["key"], s) for s in d["init"]["schemes"])
    check("Kaiming holds ReLU activation variance over "
          f"{d['init']['depth']} layers",
          0.5 < byk["kaiming"]["final_over_initial_relu"] < 2.0,
          f"final/initial = {byk['kaiming']['final_over_initial_relu']:.3f} "
          f"(mean of {d['init']['n_stacks']} independent stacks)")
    check("Xavier under ReLU halves the variance every layer",
          byk["xavier"]["final_over_initial_relu"] < 0.02,
          f"final/initial = {byk['xavier']['final_over_initial_relu']:.2e} "
          f"(predicted {byk['xavier']['predicted_final_relu']:.2e})")
    check("too-large initialisation explodes",
          byk["too_large"]["final_over_initial_relu"] > 100,
          f"final/initial = {byk['too_large']['final_over_initial_relu']:.1f}")

    # ---- 2b · residual ---------------------------------------------------
    r = d["residual"]
    deepest = r["measured"][-1]
    check("unscaled residual stream variance grows linearly in L",
          r["worst_rel_error"] < 0.35,
          f"at L={deepest['L']}: measured {deepest['unscaled']:.3f}, "
          f"predicted var0 + L·σ² = {deepest['unscaled_predicted']:.3f}")
    check("1/sqrt(2L) scaling makes the stream variance depth-independent",
          max(abs(m["scaled"] - r["measured"][0]["scaled"])
              for m in r["measured"]) < 0.35 * r["measured"][0]["scaled"],
          "scaled variance at L = " +
          ", ".join(f"{m['L']}:{m['scaled']:.3f}" for m in r["measured"]))
    check("std growth of the unscaled stream is sqrt(L)-like",
          deepest["std_growth"] > 3.0,
          f"std grows {deepest['std_growth']:.2f}x over {deepest['L']} blocks "
          f"(sqrt(L) = {math.sqrt(deepest['L']):.2f})")
    t2 = d["tf2_residual"]
    check("tf2.json's residual additions decompose exactly into "
          "stream + 2·cross + branch",
          t2["max_decomposition_error"] < 1e-12,
          f"{t2['n_additions']} additions, max reconstruction error "
          f"{t2['max_decomposition_error']:.1e}; second moment "
          f"{t2['points'][0]['second_moment']:.4f} -> "
          f"{t2['points'][-1]['second_moment']:.4f} ({t2['growth']:.2f}x over "
          f"{t2['n_layers']} blocks)")

    # ---- 3 · clipping ----------------------------------------------------
    c = d["clipping"]
    check("global norm² equals the sum of the shard norms²",
          c["identity"]["abs_error"] < 1e-12,
          f"{c['identity']['global_norm_direct']:.6f} vs "
          f"{c['identity']['global_norm_from_sum']:.6f} "
          f"(|Δ| {c['identity']['abs_error']:.1e})")
    check("no shard can clip correctly on its own norm",
          c["local_clipping"]["worst_coef_rel_error"] > 0.05,
          f"local coefficients " +
          ", ".join(f"r{w['rank']}:{w['local_coef']:.4f}"
                    for w in c["local_clipping"]["per_rank"]) +
          f" vs global {c['global_coef']:.4f}")
    check("locally-clipped gradient points in a different direction",
          c["local_clipping"]["angle_degrees"] > 0.5,
          f"{c['local_clipping']['angle_degrees']:.2f}° off, cosine "
          f"similarity {c['local_clipping']['cosine_similarity']:.6f}")
    check("the clip all-reduce is latency, not bandwidth",
          c["comm"]["latency_over_transfer"] > 1000,
          f"{c['comm']['bytes_sent_per_gpu']:.0f} B per GPU, transfer "
          f"{c['comm']['transfer_us']:.2e} µs vs latency "
          f"{c['comm']['latency_us']} µs "
          f"({c['comm']['latency_over_transfer']:.3g}x)")

    # ---- 4 · spikes ------------------------------------------------------
    s = d["spike"]
    o = s["ordering"]
    check("the gradient norm peaks BEFORE the loss",
          o["grad_norm_peaks_at"] < o["loss_peaks_at"],
          f"|g| peaks at step {o['grad_norm_peaks_at']}, loss at step "
          f"{o['loss_peaks_at']} — {o['lead_steps']} step(s) of warning")
    check("the spike is visible in the gradient norm",
          o["peak_over_median"] > 5,
          f"peak |g| {o['grad_norm_peak']:.3f} is "
          f"{o['peak_over_median']:.1f}x the median")
    rec = s["recovery"]
    check("clipping bounds the damage",
          s["clip_effect"]["reduction"] > 1.2,
          f"peak clean loss {s['clip_effect']['peak_clean_loss_no_clip']:.4f} "
          f"-> {s['clip_effect']['peak_clean_loss_clipped']:.4f} "
          f"({s['clip_effect']['reduction']:.2f}x smaller); clip ratio at the "
          f"spike {s['clip_effect']['clip_ratio_at_spike']:.4f}")
    check("the lasting damage is in Adam's second moment, and clipping "
          "keeps it out",
          rec["v_inflation"] > 5,
          f"rms(v) after the spike {rec['v_after_spike_no_clip']:.4f} "
          f"unclipped vs {rec['v_after_spike_clipped']:.4f} clipped "
          f"({rec['v_inflation']:.1f}x); excess loss over the rest of the run "
          f"{rec['excess_loss_no_clip']:.4f} vs "
          f"{rec['excess_loss_clipped']:.4f}")

    # ---- 5 · weight decay ------------------------------------------------
    dc = d["decay"]
    check("L2 and AdamW produce different trajectories",
          dc["trajectory_gap"] > 1e-3,
          f"max |θ_L2 − θ_AdamW| after {dc['n_steps']} steps = "
          f"{dc['trajectory_gap']:.4f}")
    check("L2's effective decay varies across parameters; AdamW's does not",
          dc["spread_across_live_params"] > 1.5,
          f"{dc['spread_across_live_params']:.2f}x spread in the L2/AdamW "
          f"displacement ratio across the live parameters")
    check("L2 moves the DEAD unit by a full learning rate",
          dc["dead_unit"]["ratio"] is not None
          and abs(abs(dc["dead_unit"]["l2_displacement"]) - dc["lr"]) < 1e-6,
          f"{dc['dead_unit']['params']}: L2 moves it "
          f"{abs(dc['dead_unit']['l2_displacement']):.4f} = lr, AdamW moves it "
          f"{abs(dc['dead_unit']['adamw_displacement']):.6f} "
          f"({dc['dead_unit']['ratio']:.0f}x)")

    d["checks"] = results

    # ---- emit ------------------------------------------------------------
    try:
        payload = json.dumps(d, indent=1, allow_nan=False)
    except ValueError as e:
        raise SystemExit(f"payload is not strict JSON: {e}")
    try:
        json.loads(payload, parse_constant=_reject_constant)
    except Exception as e:
        raise SystemExit(f"emitted JSON does not round-trip: {e}")

    with open(os.path.join(DATA, "dynamics.json"), "w") as f:
        f.write(payload)
    with open(os.path.join(DATA, "dynamics.js"), "w") as f:
        f.write("// GENERATED by code/dynamics.py -- do not hand-edit.\n")
        f.write("window.DYN = " + payload + ";\n")

    # ---- report ----------------------------------------------------------
    print("=" * 78)
    print("dynamics.py — the training run")
    print("=" * 78)
    print(f"  model     : {d['meta']['architecture']}")
    print(f"  batch     : {len(BATCH)} samples, sample 0 = ground_truth.py's")
    print(f"  horizon   : {HORIZON} steps, warmup {WARMUP}, peak lr {PEAK_LR}")
    print()
    print(f"  no warmup : final loss {v['final_none']:.6f}  "
          f"({w['runs']['none']['final_live_units']}/{w['max_live_units']} "
          f"unit-activations alive)")
    print(f"  warmup    : final loss {v['final_warm']:.6f}  "
          f"({w['runs']['warm']['final_live_units']}/{w['max_live_units']} "
          f"unit-activations alive)")
    print()
    npass = sum(1 for r in results if r["passed"])
    for r in results:
        print(f"  [{'PASS' if r['passed'] else 'FAIL'}] {r['name']}")
        print(f"         {r['detail']}")
    print()
    print(f"  {npass}/{len(results)} checks passed")
    print(f"  wrote {os.path.join(DATA, 'dynamics.js')}")
    print(f"  wrote {os.path.join(DATA, 'dynamics.json')}")
    print("=" * 78)

    if npass != len(results):
        raise SystemExit("dynamics.py failed its own checks")


if __name__ == "__main__":
    main()
