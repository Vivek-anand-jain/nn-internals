#!/usr/bin/env python3
"""
quantisation.py — what a quantised weight actually is, and why the win lands
on DECODE rather than prefill; plus speculative decoding, simulated and proved.

Pure Python standard library. No numpy, no torch.

Two ideas run through this file, and they are the same idea seen twice.

  1. QUANTISATION. A quantised tensor is not "a smaller float". It is an
     integer grid plus a scale. Everything that matters -- per-tensor vs
     per-channel, group size, the outlier problem, SmoothQuant -- is a
     question about WHICH grid, chosen from WHICH set of values. This file
     builds the grids, snaps real weights onto them, and measures the error
     that results. Nothing here is asserted; it is all computed.

  2. SPECULATIVE DECODING. A draft model proposes k tokens and the target
     verifies all k in ONE forward pass. That works because verification is
     prefill-shaped -- k tokens through the same weights -- and decode is
     bandwidth-bound, so the weight read dominates and k barely enters the
     cost. The acceptance rule is modified rejection sampling, and it
     preserves the target distribution EXACTLY. Both claims are proved here:
     the cost claim from code/inference_toy.py's own intensity model, the
     exactness claim analytically and then by simulation against the exact
     chain distribution.

Emits:
    assets/data/quant.js     (window.QUANT = {...})
    assets/data/quant.json

Run:  python3 code/quantisation.py
"""

import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# The arithmetic-intensity model is not re-derived here. It is the one from
# Part IV, imported, so the numbers on page 28 and page 29 cannot drift from
# the numbers on pages 24-27.
import inference_toy as IT                                   # noqa: E402
from inference_toy import cost_model                         # noqa: E402


# ============================================================================
# 0.  SMALL NUMERIC HELPERS
# ============================================================================

def flat(x):
    if isinstance(x, (int, float)):
        return [float(x)]
    out = []
    for v in x:
        out.extend(flat(v))
    return out


def errs(orig, deq):
    """Every error number on page 28 comes out of this one function."""
    n = len(orig)
    mx = max(abs(a - b) for a, b in zip(orig, deq)) if n else 0.0
    mse = sum((a - b) ** 2 for a, b in zip(orig, deq)) / n if n else 0.0
    sig = sum(a * a for a in orig) / n if n else 0.0
    rmse = math.sqrt(mse)
    return {
        "max_abs": mx,
        "rmse": rmse,
        # relative RMS error: RMSE divided by the RMS of the original values.
        "rel_rmse": (rmse / math.sqrt(sig)) if sig > 0 else 0.0,
        # signal-to-quantisation-noise ratio. Every extra bit of a uniform
        # grid halves the step, so it buys about 6.02 dB. That is a fact
        # about the grid, and the measured numbers below track it.
        "sqnr_db": (10.0 * math.log10(sig / mse)) if (mse > 0 and sig > 0) else None,
    }


def sanitise(o, digits=10):
    """JSON must be strict: no Infinity, no NaN. Floats are rounded so the
    emitted file is readable and diffable."""
    if isinstance(o, float):
        if math.isnan(o) or math.isinf(o):
            return None
        if o == 0:
            return 0.0
        mag = math.floor(math.log10(abs(o)))
        return round(o, max(0, digits - mag))
    if isinstance(o, dict):
        return {k: sanitise(v, digits) for k, v in o.items()}
    if isinstance(o, list):
        return [sanitise(v, digits) for v in o]
    return o


# ============================================================================
# 1.  THE INTEGER GRID:  scale, zero-point, quantise, dequantise
# ============================================================================
# An INT-n tensor is stored as integers q in [qmin, qmax] plus two real
# numbers per group: a SCALE (the size of one step) and a ZERO-POINT (which
# integer means real zero). Dequantisation is (q - zp) * scale. That is the
# whole definition -- everything else is a choice of how many groups.

def quant_levels(nbits):
    return -(2 ** (nbits - 1)), 2 ** (nbits - 1) - 1


def quantise(vals, nbits, symmetric=True):
    """Quantise a flat list. Returns scale, zero-point, integer codes and the
    dequantised values -- the things a kernel would actually hold."""
    qmin, qmax = quant_levels(nbits)
    if symmetric:
        # Symmetric: zero maps to integer 0 exactly, and the grid is
        # centred. Costs one code (the -qmin end is never reachable) but
        # makes the zero-point term vanish from the matmul, which is why
        # weights are almost always quantised this way.
        a = max((abs(v) for v in vals), default=0.0)
        scale = a / qmax if a > 0 else 1.0
        zp = 0
    else:
        lo = min(min(vals), 0.0)
        hi = max(max(vals), 0.0)
        scale = (hi - lo) / (qmax - qmin) if hi > lo else 1.0
        zp = int(round(qmin - lo / scale))
        zp = max(qmin, min(qmax, zp))
    q = []
    for v in vals:
        # Python's round() is round-half-to-even, the same rule the hardware
        # uses. Clamping is not cosmetic: it is where outliers get destroyed.
        qi = int(round(v / scale)) + zp
        q.append(max(qmin, min(qmax, qi)))
    deq = [(qi - zp) * scale for qi in q]
    return {
        "bits": nbits, "symmetric": symmetric,
        "scale": scale, "zero_point": zp,
        "qmin": qmin, "qmax": qmax,
        "q": q, "deq": deq,
        "clipped": sum(1 for i, v in enumerate(vals)
                       if q[i] in (qmin, qmax) and abs(v / scale) > qmax + 0.5),
        "distinct_codes": len(set(q)),
    }


def quantise_grouped(vals, nbits, group, symmetric=True):
    """One scale per contiguous group of `group` values. group = len(vals)
    is per-tensor; group = 1 would be lossless and pointless."""
    out_q, out_deq, scales, zps = [], [], [], []
    for s in range(0, len(vals), group):
        chunk = vals[s:s + group]
        r = quantise(chunk, nbits, symmetric)
        out_q.extend(r["q"])
        out_deq.extend(r["deq"])
        scales.append(r["scale"])
        zps.append(r["zero_point"])
    return {"bits": nbits, "group": group, "scales": scales, "zero_points": zps,
            "q": out_q, "deq": out_deq, "n_groups": len(scales)}


def effective_bits(nbits, group, scale_bits=16, symmetric=True):
    """Storage cost per weight INCLUDING the scales. A group of 32 with an
    fp16 scale is not 4 bits per weight, it is 4.5."""
    per_group = scale_bits + (0 if symmetric else nbits)
    return nbits + per_group / group


# ============================================================================
# 2.  PER-TENSOR vs PER-CHANNEL
# ============================================================================
# A weight matrix W is (in_features x out_features). Each COLUMN feeds one
# output channel, and each column's dequantisation scale can be folded into
# that output's accumulator afterwards for free. So per-channel scales cost
# essentially nothing at inference time -- and if one channel's range is far
# wider than the others, they are the difference between a usable model and
# a broken one.

def col(M, j):
    return [r[j] for r in M]


def quantise_matrix(M, nbits, mode, symmetric=True):
    """mode = 'tensor' (one scale for everything) or 'channel' (one per column)."""
    nr, nc = len(M), len(M[0])
    if mode == "tensor":
        r = quantise(flat(M), nbits, symmetric)
        deq = [[r["deq"][i * nc + j] for j in range(nc)] for i in range(nr)]
        q = [[r["q"][i * nc + j] for j in range(nc)] for i in range(nr)]
        # `levels` is deliberately NOT emitted: every representable value is
        # (i - zero_point) * scale for i in [qmin, qmax], so the page derives
        # the ladder from the scale rather than carrying 256 numbers per grid.
        return {"mode": "tensor", "bits": nbits, "scales": [r["scale"]],
                "zero_points": [r["zero_point"]], "q": q, "deq": deq,
                "qmin": r["qmin"], "qmax": r["qmax"],
                "distinct_codes": [r["distinct_codes"]]}
    scales, zps, cols_deq, cols_q, distinct = [], [], [], [], []
    for j in range(nc):
        r = quantise(col(M, j), nbits, symmetric)
        scales.append(r["scale"])
        zps.append(r["zero_point"])
        cols_deq.append(r["deq"])
        cols_q.append(r["q"])
        distinct.append(r["distinct_codes"])
    deq = [[cols_deq[j][i] for j in range(nc)] for i in range(nr)]
    q = [[cols_q[j][i] for j in range(nc)] for i in range(nr)]
    qmin, qmax = quant_levels(nbits)
    return {"mode": "channel", "bits": nbits, "scales": scales,
            "zero_points": zps, "q": q, "deq": deq,
            "qmin": qmin, "qmax": qmax, "distinct_codes": distinct}


def per_channel_report(M, res):
    """Per-column error, so the collapse is visible channel by channel and
    not hidden inside one averaged number."""
    nr, nc = len(M), len(M[0])
    rows = []
    for j in range(nc):
        o = col(M, j)
        d = [res["deq"][i][j] for i in range(nr)]
        qs = [res["q"][i][j] for i in range(nr)]
        e = errs(o, d)
        rows.append({
            "channel": j,
            "absmax": max(abs(v) for v in o),
            "err": e,
            "distinct_codes": len(set(qs)),
            # How much of the integer range this channel actually reaches.
            # This is the number that shows the collapse: a quiet channel
            # sharing a tensor-wide scale with a loud one lives inside a
            # handful of codes out of the 256 it was allocated.
            "code_span": max(qs) - min(qs) + 1,
        })
    return rows


# ============================================================================
# 3.  FP8 — E4M3 AND E5M2, FROM THE BIT LAYOUT UP
# ============================================================================
# ground_truth.py's decompose()/format_limits() derive fp32/bf16/fp16 facts
# from the field widths rather than quoting them. The same approach extends
# to the two 8-bit float formats, with one twist: E4M3 (as specified by OCP)
# spends its all-ones exponent on ordinary numbers instead of infinities,
# reserving only the single all-ones-mantissa code for NaN. That buys it one
# extra binade of range, which is why its maximum is 448 and not 240.

FP8_SPEC = {
    # name:  (exponent bits, mantissa bits, encoding family)
    "e4m3": (4, 3, "ocp_e4m3"),
    "e5m2": (5, 2, "ieee"),
}


def fp8_codebook(name):
    ebits, mbits, family = FP8_SPEC[name]
    bias = 2 ** (ebits - 1) - 1
    ef_all = (1 << ebits) - 1
    mf_all = (1 << mbits) - 1
    book = []
    for code in range(256):
        s = code >> 7
        ef = (code >> mbits) & ef_all
        mf = code & mf_all
        kind, v = "normal", None
        if family == "ieee" and ef == ef_all:
            kind, v = ("inf" if mf == 0 else "nan"), None
        elif family == "ocp_e4m3" and ef == ef_all and mf == mf_all:
            kind, v = "nan", None
        elif ef == 0:
            v = (mf / 2.0 ** mbits) * 2.0 ** (1 - bias)
            kind = "zero" if mf == 0 else "subnormal"
        else:
            v = (1.0 + mf / 2.0 ** mbits) * 2.0 ** (ef - bias)
        if v is not None and s:
            v = -v
        book.append({"code": code, "bits": format(code, "08b"),
                     "sign": format(code, "08b")[0],
                     "exponent": format(code, "08b")[1:1 + ebits],
                     "mantissa": format(code, "08b")[1 + ebits:],
                     "value": v, "kind": kind})
    return book


FP8_BOOKS = {n: fp8_codebook(n) for n in FP8_SPEC}
# finite codes, sorted by value, for nearest-value encoding
FP8_FINITE = {n: sorted([(e["value"], e["code"]) for e in b
                         if e["value"] is not None], key=lambda t: t[0])
              for n, b in FP8_BOOKS.items()}
FP8_MAXFIN = {n: max(v for v, _ in FP8_FINITE[n]) for n in FP8_SPEC}


def fp8_encode(x, name):
    """Round x to the nearest representable value, ties to even code.
    Overflow is reported, not hidden: E5M2 has infinities and goes to one;
    E4M3 has none and saturates."""
    book = FP8_BOOKS[name]
    fin = FP8_FINITE[name]
    maxf = FP8_MAXFIN[name]
    family = FP8_SPEC[name][2]
    if x != x:
        return {"format": name, "status": "nan", "value": None, "code": None}
    if abs(x) > maxf:
        # only genuinely overflowing if it is past the halfway point to the
        # next (nonexistent) value; below that RNE lands on maxf.
        prev = max(v for v, _ in fin if v < maxf)
        if abs(x) >= maxf + (maxf - prev) / 2.0:
            if family == "ieee":
                e = next(e for e in book if e["kind"] == "inf"
                         and e["sign"] == ("1" if x < 0 else "0"))
                return dict(e, format=name, status="overflow_inf", exact=None)
            code = max(e["code"] for e in book
                       if e["value"] is not None and e["value"] == maxf)
            if x < 0:
                code |= 0x80
            e = book[code]
            return dict(e, format=name, status="overflow_saturated",
                        exact=e["value"])
    best, bestd, bestcode = None, float("inf"), None
    for v, c in fin:
        d = abs(v - x)
        if d < bestd - 1e-300 or (abs(d - bestd) <= 1e-300 and (c & 1) == 0):
            best, bestd, bestcode = v, d, c
    e = book[bestcode]
    status = "ok"
    if x != 0 and best == 0.0:
        status = "underflow"
    elif e["kind"] == "subnormal":
        status = "subnormal"
    return dict(e, format=name, status=status, exact=best)


def fp8_value(x, name):
    r = fp8_encode(x, name)
    if r["status"] == "overflow_inf":
        return math.inf if x > 0 else -math.inf
    return r.get("exact", r.get("value"))


def fp8_limits():
    """Same derivation as ground_truth.format_limits(), extended to 8 bits.
    Note E4M3's max_finite is NOT (2 - 2^-m) * 2^bias: its top exponent code
    is a normal code, and only the all-ones mantissa is stolen for NaN."""
    out = {}
    for name, (ebits, mbits, family) in FP8_SPEC.items():
        bias = 2 ** (ebits - 1) - 1
        if family == "ieee":
            top_ef = (1 << ebits) - 2
            top_mf = (1 << mbits) - 1
        else:
            top_ef = (1 << ebits) - 1
            top_mf = (1 << mbits) - 2
        max_finite = (1 + top_mf / 2.0 ** mbits) * 2.0 ** (top_ef - bias)
        min_normal = 2.0 ** (1 - bias)
        min_sub = min_normal * 2.0 ** -mbits
        out[name] = {
            "bytes": 1, "exponent_bits": ebits, "mantissa_bits": mbits,
            "bias": bias, "max_finite": max_finite,
            "min_normal": min_normal, "min_subnormal": min_sub,
            "epsilon": 2.0 ** -mbits,
            "decimal_digits": round(mbits * math.log10(2), 2),
            "decades_of_range": round(math.log10(max_finite / min_normal), 1),
            "has_infinity": family == "ieee",
            "n_finite_codes": sum(1 for e in FP8_BOOKS[name]
                                  if e["value"] is not None),
            "n_nan_codes": sum(1 for e in FP8_BOOKS[name] if e["kind"] == "nan"),
            "encoding": ("IEEE-754-shaped: all-ones exponent is inf/NaN"
                         if family == "ieee"
                         else "OCP E4M3: no infinities; only S.1111.111 is NaN, "
                              "which buys one extra binade of range"),
        }
    return out


# ============================================================================
# 4.  BUILD:  quantisation
# ============================================================================

def load(name):
    with open(os.path.join(ROOT, "assets", "data", name)) as f:
        return json.load(f)


def outlier_matrix(base, factor, channel):
    """Take a real weight matrix and give ONE output channel a large
    magnitude. This is constructed, not measured -- but the shape of it is
    what the activation-outlier literature reports, and it is the reason
    per-channel scaling exists at all."""
    nc = len(base[0])
    out = []
    for i, row in enumerate(base):
        out.append([row[j] * (factor if j == channel else 1.0)
                    for j in range(nc)])
    return out


def build_grid_demo(M, name, bits_options):
    per_bits = {}
    for nb in bits_options:
        pt = quantise_matrix(M, nb, "tensor", symmetric=True)
        pc = quantise_matrix(M, nb, "channel", symmetric=True)
        f = flat(M)
        per_bits[str(nb)] = {
            "per_tensor": dict(pt, err=errs(f, flat(pt["deq"])),
                               channels=per_channel_report(M, pt)),
            "per_channel": dict(pc, err=errs(f, flat(pc["deq"])),
                                channels=per_channel_report(M, pc)),
        }
    return {"name": name, "shape": [len(M), len(M[0])], "values": M,
            "absmax": max(abs(v) for v in flat(M)),
            "bits_options": bits_options, "per_bits": per_bits}


def build_int4_groups(vec, label, bits_options, groups):
    rows = []
    for nb in bits_options:
        for g in groups:
            if g > len(vec):
                continue
            r = quantise_grouped(vec, nb, g, symmetric=True)
            e = errs(vec, r["deq"])
            rows.append({
                "label": label, "bits": nb, "group": g,
                "n_groups": r["n_groups"],
                "effective_bits": effective_bits(nb, g),
                "bytes_per_weight": effective_bits(nb, g) / 8.0,
                "compression_vs_bf16": 16.0 / effective_bits(nb, g),
                "err": e,
                "scale_spread": (max(r["scales"]) / min(r["scales"])
                                 if min(r["scales"]) > 0 else None),
            })
    return rows


def build_quantisation():
    tf2 = load("tf2.json")
    infer = load("infer.json")

    # ---- a real weight matrix, small enough to draw every cell ----------
    W = tf2["params"]["L0.W1"]                   # (d_model 4) x (d_ff 8)
    BITS = [2, 3, 4, 5, 6, 8]

    real = build_grid_demo(W, "L0.W1 (real, from tf2.json)", BITS)

    # ---- a wider matrix, with one wide channel --------------------------
    # 16 rows x 8 output channels. Every value is a REAL weight taken in
    # order from tf2.json; the only construction is that one column has been
    # multiplied up. The factor is chosen so the loud channel is 40x the
    # others -- the activation-outlier literature reports 20-100x, and that
    # is the regime this is standing in for.
    OUT_CH, OUT_FACTOR, ROWS, COLS = 3, 40.0, 16, 8
    src = []
    for L in ("L0", "L1"):
        for k in ("W1", "W2", "Wq", "Wk", "Wv", "Wo"):
            src.extend(flat(tf2["params"][L + "." + k]))
    base = [[src[i * COLS + j] for j in range(COLS)] for i in range(ROWS)]
    Wo = outlier_matrix(base, OUT_FACTOR, OUT_CH)
    out = build_grid_demo(
        Wo, "16x8 of real tf2.json weights, channel %d scaled %dx" %
            (OUT_CH, int(OUT_FACTOR)), BITS)
    out["base_values"] = base
    out["outlier_channel"] = OUT_CH
    out["outlier_factor"] = OUT_FACTOR

    # headline contrast at INT8, the width people actually deploy
    o8 = out["per_bits"]["8"]

    def quiet_rel(res):
        """Error on the QUIET channels only. The tensor-wide average hides
        the collapse, because the loud channel carries nearly all of the
        signal energy and is quantised perfectly well either way."""
        num = den = 0.0
        for j in range(COLS):
            if j == OUT_CH:
                continue
            for i in range(ROWS):
                num += (Wo[i][j] - res["deq"][i][j]) ** 2
                den += Wo[i][j] ** 2
        return math.sqrt(num / den) if den > 0 else 0.0

    span_pt = max(c["code_span"] for c in o8["per_tensor"]["channels"]
                  if c["channel"] != OUT_CH)
    span_pc = min(c["code_span"] for c in o8["per_channel"]["channels"]
                  if c["channel"] != OUT_CH)
    qpt, qpc = quiet_rel(o8["per_tensor"]), quiet_rel(o8["per_channel"])
    out["contrast_int8"] = {
        "per_tensor_rel_rmse": o8["per_tensor"]["err"]["rel_rmse"],
        "per_channel_rel_rmse": o8["per_channel"]["err"]["rel_rmse"],
        "quiet_rel_rmse_per_tensor": qpt,
        "quiet_rel_rmse_per_channel": qpc,
        "ratio": qpt / qpc if qpc > 0 else None,
        "quiet_channel_code_span_per_tensor": span_pt,
        "quiet_channel_code_span_per_channel": span_pc,
        "n_codes_available": 256,
        "why": ("One scale for the whole tensor is set by the widest value in "
                "it. Every quiet channel then lives inside a handful of the "
                "256 available codes and most of its information is gone. A "
                "per-channel scale costs one fp16 number per output column "
                "and can be folded into that column's accumulator afterwards, "
                "so it is free at inference time. Note the tensor-wide "
                "relative error barely moves -- the loud channel dominates "
                "the signal energy and is fine either way. The damage is "
                "entirely in the quiet channels, which is why the aggregate "
                "number is the wrong one to look at."),
    }

    # ---- INT4 with group-wise scales ------------------------------------
    real_vec = []
    for L in ("L0", "L1"):
        for k in ("Wq", "Wk", "Wv", "Wo", "W1", "W2"):
            real_vec.extend(flat(tf2["params"][L + "." + k]))
    for k in sorted(infer["params"]):
        real_vec.extend(flat(infer["params"][k]))
    real_vec = real_vec[:512]

    # A constructed vector whose LOCAL scale drifts, which is the property
    # group-wise scales actually exploit. Deterministic, no RNG.
    made = []
    for i in range(1024):
        env = 0.15 * (1.0 + 6.0 * (0.5 + 0.5 * math.sin(2 * math.pi * i / 256.0)) ** 3)
        made.append(env * math.sin(i * 2.399963229728653) * (1.0 + 0.3 * math.cos(i * 0.7)))

    g4 = (build_int4_groups(real_vec, "real weights (512)", [4, 8],
                            [512, 256, 128, 64, 32]) +
          build_int4_groups(made, "drifting-scale weights (1024, constructed)",
                            [4, 8], [1024, 256, 128, 64, 32]))

    # ---- FP8 -------------------------------------------------------------
    fmt = fp8_limits()
    fp8_examples = []
    ex_spec = [
        (0.5, "a typical weight", "comfortably inside both formats"),
        (W[0][0], "L0.W1[0][0]", "a real weight from tf2.json"),
        (1.0 / 3.0, "1/3", "3 mantissa bits vs 2: the precision gap, visible"),
        (2.5e-3, "a small gradient",
         "below E4M3's smallest normal (2^-6), so E4M3 holds it as a "
         "subnormal and loses most of its precision; E5M2 still has it as an "
         "ordinary normal number"),
        (3.0e-4, "a smaller gradient",
         "E4M3 flushes this to zero -- it is below half of E4M3's smallest "
         "subnormal. E5M2 carries it as a normal, with its usual relative "
         "error. This is the underflow that makes E5M2 the gradient format."),
        (1.0e-6, "a tiny gradient",
         "past the bottom of BOTH formats. E5M2 reaches further down than "
         "E4M3 but it is still an 8-bit float, and this is why loss scaling "
         "does not go away just because you moved to FP8."),
        (5000.0, "a large activation",
         "past E4M3's maximum finite value of 448, so E4M3 saturates; E5M2 "
         "carries it as an ordinary normal number."),
        (1.0e5, "an exploding value",
         "past both formats' maxima: E4M3 saturates, E5M2 goes to infinity "
         "because it has infinities and E4M3 does not."),
    ]
    for v, label, why in ex_spec:
        fp8_examples.append({
            "label": label, "value": v, "why": why,
            "e4m3": fp8_encode(v, "e4m3"),
            "e5m2": fp8_encode(v, "e5m2"),
            "rel_err_e4m3": (abs(fp8_value(v, "e4m3") - v) / abs(v)
                             if v and math.isfinite(fp8_value(v, "e4m3")) else None),
            "rel_err_e5m2": (abs(fp8_value(v, "e5m2") - v) / abs(v)
                             if v and math.isfinite(fp8_value(v, "e5m2")) else None),
        })

    # relative error across magnitude: the range/precision tradeoff, measured
    sweep = []
    e = -9
    while e <= 6:
        for m in (1.0, 3.0):
            x = m * 10.0 ** e
            v4 = fp8_value(x, "e4m3")
            v5 = fp8_value(x, "e5m2")
            sweep.append({
                "x": x, "log10x": math.log10(x),
                "e4m3": None if not math.isfinite(v4) else v4,
                "e5m2": None if not math.isfinite(v5) else v5,
                "rel_e4m3": (abs(v4 - x) / x if math.isfinite(v4) else None),
                "rel_e5m2": (abs(v5 - x) / x if math.isfinite(v5) else None),
                "e4m3_status": fp8_encode(x, "e4m3")["status"],
                "e5m2_status": fp8_encode(x, "e5m2")["status"],
            })
        e += 1

    # which format for which tensor, MEASURED on real tensors from the trace
    tensors = {
        "weights": flat([tf2["params"]["L0." + k] for k in
                         ("Wq", "Wk", "Wv", "Wo", "W1", "W2")]),
        "gradients": flat([tf2["grads"]["L0." + k] for k in
                           ("Wq", "Wk", "Wv", "Wo", "W1", "W2")]),
    }
    # A gradient tensor as it looks BEFORE loss scaling in a real run: the
    # same shape of distribution, pushed down several orders of magnitude.
    # Constructed, and labelled as such.
    tensors["gradients (unscaled, constructed)"] = [
        g * 1e-5 * (1.0 + 4.0 * (i % 7) / 6.0) ** 3
        for i, g in enumerate(tensors["gradients"])]

    fp8_tensors = []
    for tname, vals in tensors.items():
        nz = [abs(v) for v in vals if v != 0]
        row = {"tensor": tname, "n": len(vals),
               "absmax": max(nz) if nz else 0.0,
               "absmin": min(nz) if nz else 0.0,
               "dynamic_range": (max(nz) / min(nz)) if nz else None}
        for fname in ("e4m3", "e5m2"):
            direct = [fp8_value(v, fname) for v in vals]
            direct = [0.0 if not math.isfinite(d) else d for d in direct]
            flushed = sum(1 for v, d in zip(vals, direct) if v != 0 and d == 0.0)
            sat = sum(1 for v in vals
                      if fp8_encode(v, fname)["status"].startswith("overflow"))
            # with a per-tensor scale, the way a real FP8 kernel does it
            a = max(nz) if nz else 1.0
            s = a / fmt[fname]["max_finite"]
            scaled = [fp8_value(v / s, fname) * s for v in vals]
            scaled = [0.0 if not math.isfinite(d) else d for d in scaled]
            row[fname] = {
                "unscaled": errs(vals, direct),
                "unscaled_flushed_to_zero": flushed,
                "unscaled_saturated": sat,
                "scaled": errs(vals, scaled),
                "scale": s,
            }
        fp8_tensors.append(row)

    # ---- the outlier problem, and SmoothQuant ---------------------------
    smooth = build_smoothquant(tf2)

    # ---- error vs storage, for the interactive selector ------------------
    # Error is MEASURED on the real 4x8 matrix. Storage is ARITHMETIC, and
    # deliberately quoted at a realistic shape rather than the toy one: a
    # per-channel fp16 scale costs 16 bits per COLUMN, which is 4 bits per
    # weight on a 4-row matrix and 0.002 bits per weight on an 8192-row one.
    # Quoting the toy figure would make per-channel look expensive, which is
    # exactly backwards.
    SCALE_ROWS = 8192
    sel = []
    for nb in BITS:
        for mode, label in (("tensor", "per-tensor"), ("channel", "per-channel")):
            r = quantise_matrix(W, nb, mode, symmetric=True)
            sb = 0.0 if mode == "tensor" else 16.0 / SCALE_ROWS
            sel.append({
                "kind": "int", "bits": nb, "mode": mode,
                "label": "INT" + str(nb) + " " + label,
                "err": errs(flat(W), flat(r["deq"])),
                "effective_bits": nb + sb,
                "scale_bits_per_weight": sb,
                "n_scales_here": len(r["scales"]),
            })
    for fname in ("e4m3", "e5m2"):
        a = max(abs(v) for v in flat(W))
        s = a / fmt[fname]["max_finite"]
        d = [fp8_value(v / s, fname) * s for v in flat(W)]
        sel.append({"kind": "fp8", "bits": 8, "mode": "tensor",
                    "label": "FP8 " + fname.upper() + " per-tensor",
                    "err": errs(flat(W), d),
                    "effective_bits": 8.0, "scale_bits_per_weight": 0.0,
                    "n_scales_here": 1})
    sel.append({"kind": "float", "bits": 16, "mode": "none", "label": "bf16",
                "err": errs(flat(W), bf16_round(flat(W))),
                "effective_bits": 16.0, "scale_bits_per_weight": 0.0,
                "n_scales_here": 0})

    return {
        "real_matrix": real,
        "outlier_matrix": out,
        "int4_groups": g4,
        "fp8": {
            "formats": fmt,
            "codebooks": {n: FP8_BOOKS[n] for n in FP8_SPEC},
            "examples": fp8_examples,
            "range_sweep": sweep,
            "tensors": fp8_tensors,
            "roles": {
                "_label": "literature",
                "text": ("The convention -- E4M3 for weights and activations, "
                         "E5M2 for gradients -- comes from the FP8 training "
                         "papers and the OCP FP8 specification, not from this "
                         "file. What IS computed here is the reason it makes "
                         "sense: E4M3 has one more mantissa bit (half the "
                         "relative error) but a maximum of 448 and a smallest "
                         "normal of 2^-6, while E5M2 reaches 57344 and 2^-14. "
                         "Weights and activations sit near unity after "
                         "normalisation, so precision is the binding "
                         "constraint. Gradients span orders of magnitude and "
                         "their scale is not known ahead of time, so range "
                         "is."),
            },
        },
        "smoothquant": smooth,
        "selector": sel,
        "selector_note": ("Error is measured on the real 4x8 matrix "
                          "L0.W1. Storage per weight is arithmetic, quoted "
                          "for a weight matrix with %d rows -- one fp16 "
                          "per-channel scale amortised over a whole column."
                          % SCALE_ROWS),
        "selector_scale_rows": SCALE_ROWS,
        "definitions": {
            "dequantise": "x_hat = (q - zero_point) * scale",
            "quantise": "q = clamp(round(x / scale) + zero_point, qmin, qmax)",
            "symmetric": "zero_point = 0, scale = max|x| / qmax",
            "asymmetric": ("scale = (max - min) / (qmax - qmin), "
                           "zero_point = round(qmin - min / scale)"),
            "sqnr": "10 * log10(mean(x^2) / mean((x - x_hat)^2)); "
                    "a uniform grid buys about 6.02 dB per bit",
        },
    }


def bf16_round(vals):
    """bf16 = top 16 bits of fp32, round-to-nearest-even. Same routine as
    ground_truth.bf16_bits, reimplemented here without importing struct
    machinery so this file stands alone."""
    import struct
    out = []
    for x in vals:
        (u,) = struct.unpack(">I", struct.pack(">f", x))
        lower = u & 0xFFFF
        upper = (u >> 16) & 0xFFFF
        if lower > 0x8000 or (lower == 0x8000 and (upper & 1)):
            upper = (upper + 1) & 0xFFFF
        (f,) = struct.unpack(">f", struct.pack(">I", upper << 16))
        out.append(f)
    return out


# ============================================================================
# 5.  THE OUTLIER PROBLEM AND SMOOTHQUANT
# ============================================================================
# Transformer ACTIVATIONS, unlike weights, have a few channels whose
# magnitude is one to two orders of magnitude above the rest, and they are
# the same channels for every token. Quantising activations per-tensor lets
# those channels set the scale and flattens everything else.
#
# SmoothQuant's move: activations and weights meet in a matmul, so a diagonal
# rescale can be MIGRATED from one to the other without changing the product.
#
#     X @ W  ==  (X / diag(s)) @ (diag(s) @ W)
#
# Divide the loud activation channels down; multiply the corresponding weight
# ROWS up. Weights are easy to quantise (narrow, static), activations are
# hard (wide, dynamic), so moving difficulty from the second to the first is
# a straight win. The identity is exact in full precision -- verified below.

def matmul(A, B):
    m, k, n = len(A), len(B), len(B[0])
    return [[sum(A[i][t] * B[t][j] for t in range(k)) for j in range(n)]
            for i in range(m)]


def fro_rel(ref, got):
    num = sum((a - b) ** 2 for ra, rb in zip(ref, got) for a, b in zip(ra, rb))
    den = sum(a * a for ra in ref for a in ra)
    return math.sqrt(num / den) if den > 0 else 0.0


def quantise_rows(M, nbits):
    """Per-token (per-row) activation quantisation -- the cheap thing that a
    real kernel can actually do online, since a row is one token."""
    out, scales = [], []
    for r in M:
        q = quantise(r, nbits, symmetric=True)
        out.append(q["deq"])
        scales.append(q["scale"])
    return out, scales


def build_smoothquant(tf2):
    W = tf2["params"]["L0.W1"]              # (C=4) x (O=8)
    C, O = len(W), len(W[0])
    T = 8                                   # tokens
    OUTLIER_CH = [1]
    OUTLIER_GAIN = 60.0

    # Activations: deterministic, O(1) in most channels, with one loud one.
    X = []
    for t in range(T):
        row = []
        for c in range(C):
            v = math.sin(1.7 * t + 2.3 * c) * (0.4 + 0.2 * math.cos(0.9 * c))
            if c in OUTLIER_CH:
                v *= OUTLIER_GAIN
            row.append(v)
        X.append(row)

    Y = matmul(X, W)
    act_absmax = [max(abs(X[t][c]) for t in range(T)) for c in range(C)]
    w_absmax = [max(abs(W[c][j]) for j in range(O)) for c in range(C)]

    def measure(Xm, Wm, nbits=8):
        """Quantise both operands and multiply, then compare to the exact
        product. This is the number that matters -- error in the OUTPUT."""
        res = {}
        xt = quantise_matrix(Xm, nbits, "tensor")["deq"]
        xr, _ = quantise_rows(Xm, nbits)
        wc = quantise_matrix(Wm, nbits, "channel")["deq"]
        res["act_per_tensor"] = fro_rel(Y, matmul(xt, wc))
        res["act_per_token"] = fro_rel(Y, matmul(xr, wc))
        res["act_err_per_tensor"] = errs(flat(Xm), flat(xt))
        res["act_err_per_token"] = errs(flat(Xm), flat(xr))
        res["w_err_per_channel"] = errs(flat(Wm), flat(wc))
        return res

    baseline = measure(X, W)

    def smooth(alpha):
        s = []
        for c in range(C):
            a = act_absmax[c] ** alpha
            b = w_absmax[c] ** (1.0 - alpha)
            s.append(a / b if b > 0 else 1.0)
        Xs = [[X[t][c] / s[c] for c in range(C)] for t in range(T)]
        Ws = [[W[c][j] * s[c] for j in range(O)] for c in range(C)]
        return s, Xs, Ws

    sweep = []
    for i in range(21):
        alpha = i / 20.0
        s, Xs, Ws = smooth(alpha)
        m = measure(Xs, Ws)
        sweep.append({
            "alpha": alpha,
            "scales": s,
            "act_absmax_after": [max(abs(Xs[t][c]) for t in range(T))
                                 for c in range(C)],
            "w_absmax_after": [max(abs(Ws[c][j]) for j in range(O))
                               for c in range(C)],
            "exactness": fro_rel(Y, matmul(Xs, Ws)),
            "rel_err_act_per_tensor": m["act_per_tensor"],
            "rel_err_act_per_token": m["act_per_token"],
        })
    best = min(sweep, key=lambda r: r["rel_err_act_per_tensor"])
    exact_max = max(r["exactness"] for r in sweep)

    s_best, Xb, Wb = smooth(best["alpha"])
    return {
        "X": X, "W": W, "Y": Y,
        "tokens": T, "in_channels": C, "out_channels": O,
        "outlier_channels": OUTLIER_CH, "outlier_gain": OUTLIER_GAIN,
        "act_absmax": act_absmax, "w_absmax": w_absmax,
        "baseline": baseline,
        "alpha_sweep": sweep,
        "best": {"alpha": best["alpha"],
                 "rel_err_act_per_tensor": best["rel_err_act_per_tensor"],
                 "rel_err_act_per_token": best["rel_err_act_per_token"],
                 "scales": s_best},
        "improvement_per_tensor": (baseline["act_per_tensor"] /
                                   best["rel_err_act_per_tensor"]),
        "improvement_per_token": (baseline["act_per_token"] /
                                  best["rel_err_act_per_token"]),
        "X_smoothed": Xb, "W_smoothed": Wb,
        "identity": "X @ W == (X / diag(s)) @ (diag(s) @ W), for any positive s",
        "migration_rule": "s_c = max|X[:,c]|^alpha / max|W[c,:]|^(1-alpha)",
        "exactness_max_rel_err": exact_max,
        "exactness_passed": exact_max < 1e-12,
        "_constructed": ("The activation matrix here is constructed with one "
                         "loud channel. That a handful of activation channels "
                         "in a trained transformer carry 20-100x the typical "
                         "magnitude, in the same channels for every token, is "
                         "a MEASURED result in the literature, not something "
                         "this file demonstrates. What this file computes is "
                         "everything downstream of that fact."),
    }


# ============================================================================
# 6.  WEIGHT-ONLY QUANTISATION:  the decode / prefill asymmetry
# ============================================================================
# This is the whole reason INT4 weight-only quantisation is a serving
# technique and not a training one.
#
#   decode  is BANDWIDTH-bound. Its cost is bytes/BW, and the bytes are
#           almost all weights. Quarter the weight bytes, quarter the time.
#   prefill is COMPUTE-bound. Its cost is flops/peak, and quantising weights
#           does not remove any flops -- it ADDS them, because every weight
#           has to be dequantised back into the compute dtype before the
#           tensor cores can touch it.
#
# Same change, opposite sign, and the reason is arithmetic intensity.

# ---- ASSUMPTIONS. These two numbers are not derived from anything in this
# repository. They stand in for the efficiency of a real mixed-precision
# kernel and they are labelled as assumptions everywhere they surface.
KERNEL_EFF = {2.0: 1.00, 1.0: 0.92, 0.5: 0.85}
DEQUANT_FLOPS_PER_WEIGHT = {2.0: 0.0, 1.0: 2.0, 0.5: 4.0}


def time_model(model, gpu, seq, batch, w_bytes, kv_bytes=2):
    kvh = model.get("n_kv_heads", model["n_heads"])
    d_head = model["d_model"] // model["n_heads"]
    c = cost_model(model["params"], model["n_layers"], model["d_model"],
                   kvh, d_head, seq, batch,
                   w_bytes=w_bytes, kv_bytes=kv_bytes)
    eff = KERNEL_EFF[w_bytes]
    dq = DEQUANT_FLOPS_PER_WEIGHT[w_bytes] * model["params"]
    peak = gpu["bf16_dense_tflops"] * 1e12 * eff
    bw = gpu["hbm_bw_bytes_per_s"]
    out = {}
    for phase in ("prefill", "decode"):
        flops = c[phase]["flops"] + dq
        byts = c[phase]["bytes"]
        t_c, t_m = flops / peak, byts / bw
        out[phase] = {
            "flops": flops, "dequant_flops": dq, "bytes": byts,
            "intensity": flops / byts,
            "t_compute_s": t_c, "t_memory_s": t_m,
            "time_s": max(t_c, t_m),
            "bound": "compute" if t_c > t_m else "memory",
        }
    return out


def build_weight_only():
    model = {"name": "Llama 3 70B", "params": 70.6e9, "d_model": 8192,
             "n_layers": 80, "n_heads": 64, "n_kv_heads": 8, "seq": 8192}
    gpu = {"name": "H100 80GB", "bf16_dense_tflops": 495,
           "hbm_bw_bytes_per_s": 3350e9}
    SEQ, BATCH = 8192, 1
    ridge = gpu["bf16_dense_tflops"] * 1e12 / gpu["hbm_bw_bytes_per_s"]

    base = time_model(model, gpu, SEQ, BATCH, 2.0)
    rows = []
    for wb, label in ((2.0, "bf16"), (1.0, "INT8 weight-only"),
                      (0.5, "INT4 weight-only")):
        t = time_model(model, gpu, SEQ, BATCH, wb)
        rows.append({
            "w_bytes": wb, "bits": int(wb * 8), "label": label,
            "kernel_efficiency": KERNEL_EFF[wb],
            "dequant_flops_per_weight": DEQUANT_FLOPS_PER_WEIGHT[wb],
            "weight_bytes": model["params"] * wb,
            "weight_bytes_ratio": 2.0 / wb,
            "prefill": t["prefill"], "decode": t["decode"],
            "prefill_speedup": base["prefill"]["time_s"] / t["prefill"]["time_s"],
            "decode_speedup": base["decode"]["time_s"] / t["decode"]["time_s"],
            "decode_tok_per_s": 1.0 / t["decode"]["time_s"],
        })

    # the same asymmetry as a function of batch: as batch grows, decode
    # accumulates FLOPs and KV traffic and the weight-only win erodes.
    by_batch = []
    for b in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        bb = time_model(model, gpu, SEQ, b, 2.0)
        q = time_model(model, gpu, SEQ, b, 0.5)
        by_batch.append({
            "batch": b,
            "decode_speedup": bb["decode"]["time_s"] / q["decode"]["time_s"],
            "prefill_speedup": bb["prefill"]["time_s"] / q["prefill"]["time_s"],
            "kv_share_bf16": (bb["decode"]["bytes"] - model["params"] * 2) /
                             bb["decode"]["bytes"],
            "decode_bound_int4": q["decode"]["bound"],
        })

    return {
        "model": model, "gpu": gpu, "seq": SEQ, "batch": BATCH,
        "ridge_flops_per_byte": ridge,
        "rows": rows, "by_batch": by_batch,
        "_assumptions": [
            "Mixed-precision kernel efficiency: " +
            ", ".join("%d-bit weights -> %.2f of bf16 peak" % (int(k * 8), v)
                      for k, v in sorted(KERNEL_EFF.items(), reverse=True)) +
            ". ASSUMED. Real numbers depend on the kernel and the shape.",
            "Dequantisation cost: " +
            ", ".join("%d-bit -> %.0f FLOP per weight per pass" % (int(k * 8), v)
                      for k, v in sorted(DEQUANT_FLOPS_PER_WEIGHT.items(),
                                         reverse=True)) +
            ". ASSUMED.",
            "The KV cache stays bf16 throughout. Weight-only quantisation "
            "does not touch it, which is exactly why the decode speedup "
            "falls short of the 4x the weight bytes suggest.",
        ],
        "_literature": [
            "Accuracy claims -- that INT8 per-channel weight quantisation is "
            "near-lossless, that INT4 group-wise costs a small amount of "
            "perplexity, that FP8 training matches bf16 -- are results from "
            "the quantisation literature. This file measures ERROR ON REAL "
            "TENSORS, which is a different and much narrower claim: it says "
            "nothing about downstream task quality.",
        ],
    }


# ============================================================================
# 7.  SPECULATIVE DECODING
# ============================================================================
# The trick, stated precisely:
#
#   A draft model proposes k tokens autoregressively (k cheap sequential
#   steps). The target model then evaluates all k+1 positions in ONE forward
#   pass. Because decode is bandwidth-bound, the target's cost is dominated
#   by reading its weights, which happens once regardless of k. So verifying
#   8 tokens costs almost exactly what verifying 1 costs.
#
# Then an acceptance rule decides how many of the k proposals survive, and
# that rule is constructed so the emitted tokens are distributed EXACTLY as
# if they had been sampled from the target one at a time.

def verify_cost(model, gpu, seq, batch, ks):
    """Cost of a target forward over k tokens with a warm cache. This is a
    prefill-shaped operation with m = k, which is why it is nearly free."""
    kvh = model.get("n_kv_heads", model["n_heads"])
    d_head = model["d_model"] // model["n_heads"]
    peak = gpu["bf16_dense_tflops"] * 1e12
    bw = gpu["hbm_bw_bytes_per_s"]
    rows = []
    for k in ks:
        flops = 2 * model["params"] * k * batch
        flops += 2 * 2 * model["n_layers"] * batch * kvh * d_head * seq * k
        byts = model["params"] * 2
        byts += 2 * model["n_layers"] * batch * seq * kvh * d_head * 2
        t = max(flops / peak, byts / bw)
        rows.append({"k": k, "flops": flops, "bytes": byts,
                     "intensity": flops / byts,
                     "t_compute_s": flops / peak, "t_memory_s": byts / bw,
                     "time_s": t,
                     "bound": "compute" if flops / peak > byts / bw else "memory"})
    t1 = rows[0]["time_s"]
    for r in rows:
        r["time_ratio_vs_k1"] = r["time_s"] / t1
        r["cost_per_token"] = r["time_s"] / r["k"]
    # where verification stops being free: the k at which the compute term
    # overtakes the weight-read term. Derived, not chosen.
    byts = rows[0]["bytes"]
    per_k = 2 * model["params"] * batch + \
        2 * 2 * model["n_layers"] * batch * kvh * d_head * seq
    k_cross = (byts / bw) * peak / per_k
    return {"rows": rows, "k_crossover": k_cross,
            "why": ("Verifying k tokens reads the weights exactly once, the "
                    "same as verifying one. The extra FLOPs are real but "
                    "decode has FLOPs to spare -- the whole regime is idling "
                    "on compute. Verification stays free until k reaches "
                    "about %.0f, where the arithmetic finally catches the "
                    "memory traffic." % k_cross)}


# ---------------------------------------------------------------- the rule

def softmax(v, temp=1.0):
    m = max(v)
    e = [math.exp((x - m) / temp) for x in v]
    s = sum(e)
    return [x / s for x in e]


def residual(p, q):
    """(p - q) clipped at zero and renormalised. This is where a rejected
    token's replacement comes from, and it is the entire reason the scheme
    is exact rather than merely approximate."""
    r = [max(0.0, p[i] - q[i]) for i in range(len(p))]
    s = sum(r)
    if s <= 0:
        return list(p), 0.0
    return [x / s for x in r], s


def emitted_distribution(p, q):
    """The distribution of the token this scheme emits for ONE proposal.

        accept x ~ q with probability min(1, p(x)/q(x))
        otherwise emit a draw from norm((p - q)+)

    P(emit x) = q(x)*min(1, p(x)/q(x)) + P(reject) * resid(x)
              = min(p(x), q(x))       + (p(x) - q(x))+
              = p(x)

    That last line is the whole proof, and it is an identity, not a limit."""
    V = len(p)
    acc = sum(min(p[i], q[i]) for i in range(V))
    r, mass = residual(p, q)
    return [min(p[i], q[i]) + (1.0 - acc) * r[i] for i in range(V)], acc, mass


def sample(dist, rng):
    u = rng.random()
    c = 0.0
    for i, w in enumerate(dist):
        c += w
        if u < c:
            return i
    return len(dist) - 1


class ToyPair:
    """Target = the full 2-layer toy transformer from inference_toy.py.
    Draft  = the SAME weights with layers removed. draft_layers=1 keeps the
    first block; draft_layers=0 wires the embedding table straight into the
    LM head, which is about as weak as a draft model gets and therefore the
    hardest case for the acceptance rule to survive.

    Distributions are memoised: the proof needs hundreds of thousands of
    draws and the underlying model is pure Python."""

    def __init__(self, temp=1.0, draft_layers=1):
        self.P = IT.init()
        self.temp = temp
        self.draft_layers = draft_layers
        self._c = {}

    def _logits(self, ids, n_layers):
        hidden = [list(self.P["E"][t]) for t in ids]
        for L in range(n_layers):
            hidden, _ = IT.run_block(self.P, L, hidden, None, False)
        return IT.mm(hidden, self.P["head"])[-1]

    def dist(self, ctx, which):
        key = (which, ctx)
        d = self._c.get(key)
        if d is None:
            n = IT.N_LAYERS if which == "target" else self.draft_layers
            d = softmax(self._logits(list(ctx), n), self.temp)
            self._c[key] = d
        return d

    def p(self, ctx):
        return self.dist(tuple(ctx), "target")

    def q(self, ctx):
        return self.dist(tuple(ctx), "draft")


def spec_rollout(pair, ctx, L, k, rng, rule="exact"):
    """One full speculative run: propose k, verify, accept a prefix, roll
    back the rest, emit a correction token. Returns the sequence plus a
    per-round trace so page 29 can animate an actual run."""
    seq, rounds = [], []
    while len(seq) < L:
        base = list(ctx) + seq
        drafted, qs = [], []
        c = list(base)
        for _ in range(k):
            q = pair.q(c)
            x = sample(q, rng)
            drafted.append(x)
            qs.append(q)
            c.append(x)
        ps = [pair.p(base + drafted[:i]) for i in range(k + 1)]

        accepted, verdicts = 0, []
        rejected_at, replacement = None, None
        for i in range(k):
            p, q, x = ps[i], qs[i], drafted[i]
            a = min(1.0, p[x] / q[x]) if q[x] > 0 else 0.0
            u = rng.random()
            if u < a:
                seq.append(x)
                accepted += 1
                verdicts.append({"i": i, "token": x, "p": p[x], "q": q[x],
                                 "accept_prob": a, "u": u, "accepted": True})
            else:
                if rule == "exact":
                    r, _ = residual(p, q)
                else:
                    r = p              # the plausible-looking WRONG rule
                y = sample(r, rng)
                seq.append(y)
                rejected_at, replacement = i, y
                verdicts.append({"i": i, "token": x, "p": p[x], "q": q[x],
                                 "accept_prob": a, "u": u, "accepted": False,
                                 "replacement": y})
                break
        else:
            y = sample(ps[k], rng)
            seq.append(y)
            replacement = y
        rounds.append({"drafted": drafted, "verdicts": verdicts,
                       "n_accepted": accepted, "rejected_at": rejected_at,
                       "bonus": rejected_at is None,
                       "emitted": replacement,
                       "n_emitted": accepted + 1})
    return seq[:L], rounds


def exact_chain(pair, ctx, L):
    """The exact distribution over all V^L continuations under the TARGET,
    computed by enumeration. This is what speculative decoding has to match
    -- not approximately, exactly."""
    V = IT.VOCAB
    out = {}

    def rec(prefix, prob):
        if len(prefix) == L:
            out[tuple(prefix)] = prob
            return
        p = pair.p(list(ctx) + prefix)
        for v in range(V):
            if p[v] > 0:
                rec(prefix + [v], prob * p[v])

    rec([], 1.0)
    return out


def build_spec():
    model = {"name": "Llama 3 70B", "params": 70.6e9, "d_model": 8192,
             "n_layers": 80, "n_heads": 64, "n_kv_heads": 8, "seq": 8192}
    draft = {"name": "Llama 3 8B", "params": 8.03e9, "d_model": 4096,
             "n_layers": 32, "n_heads": 32, "n_kv_heads": 8, "seq": 8192}
    gpu = {"name": "H100 80GB", "bf16_dense_tflops": 495,
           "hbm_bw_bytes_per_s": 3350e9}
    SEQ, BATCH = 8192, 1
    KS = list(range(1, 17))

    vc = verify_cost(model, gpu, SEQ, BATCH, KS)

    # Draft and target step times come from the SAME bandwidth model, so the
    # cost ratio is derived rather than assumed.
    t_tgt = time_model(model, gpu, SEQ, BATCH, 2.0)["decode"]["time_s"]
    t_drf = time_model(draft, gpu, SEQ, BATCH, 2.0)["decode"]["time_s"]

    # ---------------- the exactness proof --------------------------------
    # The draft is the target with EVERY transformer block removed -- the
    # embedding table wired straight into the LM head. A deliberately weak
    # draft, sampled at temperature 0.3 so both distributions are peaked and
    # genuinely disagree. A weak draft is the hard case: it rejects often,
    # so the residual branch is exercised, which is exactly the branch the
    # exactness argument rests on.
    TEMP, DRAFT_LAYERS = 0.3, 0

    # (a) analytic: the identity, on real distributions and on random ones
    pair = ToyPair(temp=TEMP, draft_layers=DRAFT_LAYERS)
    ident = []
    ctxs = [tuple(IT.PROMPT[:i + 1]) for i in range(len(IT.PROMPT))]
    for c in ctxs:
        p, q = pair.p(list(c)), pair.q(list(c))
        e, acc, _ = emitted_distribution(p, q)
        ident.append({"source": "toy model, context " + str(list(c)),
                      "max_abs_err": max(abs(e[i] - p[i]) for i in range(len(p))),
                      "alpha": acc})
    rr = random.Random(7)
    worst_rand = 0.0
    for _ in range(4000):
        n = rr.randint(2, 12)
        p = softmax([rr.gauss(0, 2.5) for _ in range(n)])
        q = softmax([rr.gauss(0, 2.5) for _ in range(n)])
        e, _, _ = emitted_distribution(p, q)
        worst_rand = max(worst_rand, max(abs(e[i] - p[i]) for i in range(n)))
    ident.append({"source": "4000 random (p, q) pairs, vocab 2-12",
                  "max_abs_err": worst_rand, "alpha": None})
    analytic_max = max(r["max_abs_err"] for r in ident)

    # (b) empirical: simulate the FULL loop and compare against the exact
    #     chain distribution. Also run a plausible-but-wrong rule as a
    #     control, so a passing test is not just a slack test.
    L, K, M = 3, 3, 120000
    exact = exact_chain(pair, IT.PROMPT, L)

    def simulate(rule, n):
        r = random.Random(424242 if rule == "exact" else 424243)
        counts = {}
        acc_tokens, steps = 0, 0
        for _ in range(n):
            s, rounds = spec_rollout(pair, IT.PROMPT, L, K, r, rule)
            counts[tuple(s)] = counts.get(tuple(s), 0) + 1
            for rd in rounds:
                acc_tokens += rd["n_accepted"]
                steps += 1
        return counts, acc_tokens, steps

    def compare(counts, n):
        tv = 0.0
        zmax, cells = 0.0, 0
        for seqk, pr in exact.items():
            obs = counts.get(seqk, 0)
            tv += abs(obs / n - pr)
            exp = n * pr
            if exp >= 25:
                cells += 1
                z = (obs - exp) / math.sqrt(exp * (1 - pr))
                zmax = max(zmax, abs(z))
        # anything the simulation produced that the target cannot
        extra = sum(c for s, c in counts.items() if s not in exact)
        tv = 0.5 * (tv + extra / n)
        return {"tv_distance": tv, "max_abs_z": zmax, "cells_tested": cells,
                "n_outcomes": len(exact), "draws": n,
                "impossible_outcomes": extra}

    def simulate_direct(n):
        """The control in the other direction: sample the target chain
        directly, the slow one-token-at-a-time way. Its distance from the
        exact distribution is pure sampling noise, and it is the yardstick
        the speculative run has to be measured against. A total-variation
        distance is never zero at finite n, so 'is it small' is meaningless
        without this number."""
        r = random.Random(31337)
        counts = {}
        for _ in range(n):
            s = []
            for _ in range(L):
                s.append(sample(pair.p(list(IT.PROMPT) + s), r))
            counts[tuple(s)] = counts.get(tuple(s), 0) + 1
        return counts

    c_ok, acc_tok, n_steps = simulate("exact", M)
    cmp_ok = compare(c_ok, M)
    c_bad, _, _ = simulate("naive", M)
    cmp_bad = compare(c_bad, M)
    c_dir = simulate_direct(M)
    cmp_dir = compare(c_dir, M)
    cmp_ok["tv_vs_noise_floor"] = cmp_ok["tv_distance"] / cmp_dir["tv_distance"]
    cmp_bad["tv_vs_noise_floor"] = cmp_bad["tv_distance"] / cmp_dir["tv_distance"]

    proof_passed = (analytic_max < 1e-12 and
                    cmp_ok["max_abs_z"] < 5.0 and
                    cmp_ok["impossible_outcomes"] == 0 and
                    cmp_ok["tv_distance"] < 3.0 * cmp_dir["tv_distance"] and
                    cmp_bad["max_abs_z"] > 10.0 and
                    cmp_bad["tv_distance"] > 5.0 * cmp_dir["tv_distance"])

    # measured acceptance rate for this toy draft/target pair, per context.
    # alpha = sum_x min(p(x), q(x)) is not an estimate: it IS the expected
    # acceptance probability for one proposal, exactly.
    alphas_measured = []
    for c in ctxs:
        p, q = pair.p(list(c)), pair.q(list(c))
        alphas_measured.append({"context": list(c), "alpha":
                                sum(min(p[i], q[i]) for i in range(len(p))),
                                "p": p, "q": q})
    alpha_toy = acc_tok / (n_steps * K)

    # the same measurement for a STRONGER draft (the target's first block),
    # so the effect of draft quality on alpha is visible rather than asserted
    strong = ToyPair(temp=TEMP, draft_layers=1)
    alpha_strong = []
    for c in ctxs:
        p, q = strong.p(list(c)), strong.q(list(c))
        alpha_strong.append(sum(min(p[i], q[i]) for i in range(len(p))))

    # a demonstration run, for the animation
    demo_rng = random.Random(99)
    demo_seq, demo_rounds = spec_rollout(pair, IT.PROMPT, 12, 4, demo_rng, "exact")

    # ---------------- expected tokens per step and wall clock ------------
    # E[accepted] with i.i.d. acceptance alpha = sum_{i=1..k} alpha^i.
    # Plus one token always emitted (either the correction or the bonus), so
    #   E[tokens per step] = (1 - alpha^(k+1)) / (1 - alpha).
    ALPHAS = [0.5, 0.6, 0.7, 0.8, 0.9]

    def tokens_per_step(alpha, k):
        if alpha >= 1.0:
            return k + 1.0
        return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)

    grid, optimal = [], []
    for a in ALPHAS:
        pts = []
        for k in KS:
            tps = tokens_per_step(a, k)
            t_step = k * t_drf + vc["rows"][k - 1]["time_s"]
            pts.append({
                "k": k, "tokens_per_step": tps,
                "step_time_s": t_step,
                "time_per_token_s": t_step / tps,
                "speedup": tps * t_tgt / t_step,
                "draft_share": k * t_drf / t_step,
            })
        best = max(pts, key=lambda r: r["speedup"])
        grid.append({"alpha": a, "points": pts})
        optimal.append({"alpha": a, "k_star": best["k"],
                        "speedup": best["speedup"],
                        "tokens_per_step": best["tokens_per_step"],
                        "time_per_token_s": best["time_per_token_s"]})

    # the ideal-hardware curve: free verification, free drafting
    ideal = [{"alpha": a, "points": [{"k": k, "speedup": tokens_per_step(a, k)}
                                     for k in KS]} for a in ALPHAS]

    return {
        "target": model, "draft": draft, "gpu": gpu,
        "seq": SEQ, "batch": BATCH, "ks": KS, "alphas": ALPHAS,
        "verify": vc,
        "timing": {
            "t_target_decode_s": t_tgt,
            "t_draft_decode_s": t_drf,
            "cost_ratio_draft_over_target": t_drf / t_tgt,
            "derivation": ("Both step times come from the same bandwidth "
                           "model used on pages 24-27: time = bytes / HBM "
                           "bandwidth. The draft/target cost ratio is "
                           "therefore derived from the parameter counts, not "
                           "assumed."),
        },
        "rule": {
            "steps": [
                "draft samples x ~ q(. | context)",
                "target computes p(. | context) for all k+1 positions in one pass",
                "accept x with probability min(1, p(x)/q(x))",
                "on rejection, emit a draw from norm((p - q)+) and discard the rest",
                "if all k are accepted, emit a bonus draw from p at position k+1",
            ],
            "identity": "min(p, q) + (p - q)+ = p, elementwise",
            "why_rollback": ("The draft's tokens after the first rejection "
                             "were conditioned on a token that did not "
                             "survive, so they are meaningless and are "
                             "thrown away. The KV cache entries for them are "
                             "truncated -- that is the rollback."),
        },
        "proof": {
            "analytic": {"rows": ident, "max_abs_err": analytic_max,
                         "passed": analytic_max < 1e-12,
                         "claim": ("For every (p, q), the emitted-token "
                                   "distribution equals p exactly. Checked on "
                                   "the toy model's own distributions and on "
                                   "4000 random pairs.")},
            "empirical": {
                "L": L, "k": K, "draws": M,
                "exact_rule": cmp_ok,
                "naive_rule": cmp_bad,
                "direct_sampling": cmp_dir,
                "method": ("The exact distribution over all %d continuations "
                           "of length %d is enumerated from the target model. "
                           "Then three things are run %d times each: "
                           "speculative decoding with the correct rule, "
                           "speculative decoding with a wrong rule, and plain "
                           "one-token-at-a-time sampling from the target. The "
                           "third is the noise floor -- no finite sample sits "
                           "exactly on its own distribution, so 'close' only "
                           "means anything relative to it."
                           % (len(exact), L, M)),
                "naive_description": ("A plausible-looking wrong rule: on "
                                      "rejection, resample from p instead of "
                                      "from the residual (p - q)+. It "
                                      "double-counts the mass the draft "
                                      "already got right, and the test catches "
                                      "it."),
                "top_outcomes": sorted(
                    [{"seq": list(s), "exact": pr,
                      "empirical": c_ok.get(s, 0) / M,
                      "naive": c_bad.get(s, 0) / M,
                      "direct": c_dir.get(s, 0) / M}
                     for s, pr in exact.items()],
                    key=lambda r: -r["exact"])[:16],
            },
            "passed": proof_passed,
        },
        "toy": {
            "vocab": IT.VOCAB, "tokens": IT.TOKEN_STR, "prompt": IT.PROMPT,
            "target_layers": IT.N_LAYERS, "draft_layers": DRAFT_LAYERS,
            "temperature": TEMP,
            "alpha_measured_per_context": alphas_measured,
            "alpha_measured_overall": alpha_toy,
            "alpha_stronger_draft": alpha_strong,
            "note": ("The draft used for the proof is the target with every "
                     "transformer block removed -- the embedding table wired "
                     "straight into the LM head. A weak draft rejects often, "
                     "which is the case that stresses the residual branch of "
                     "the rule. Keeping the first block instead raises alpha "
                     "to the values in alpha_stronger_draft. Both are "
                     "properties of THIS pair on a six-token vocabulary and "
                     "neither should be read as a real-world acceptance "
                     "rate."),
            "demo": {"seq": demo_seq, "rounds": demo_rounds, "k": 4,
                     "context": IT.PROMPT},
        },
        "expected": {
            "formula": "E[tokens per step] = (1 - alpha^(k+1)) / (1 - alpha)",
            "wallclock": ("step time = k * t_draft + t_verify(k); "
                          "speedup = E[tokens] * t_target / step time"),
            "grid": grid, "ideal": ideal, "optimal": optimal,
        },
        "_assumptions": [
            "The acceptance rates alpha on the speedup curves are ASSUMED "
            "values (0.5-0.9), chosen to span the range reported for real "
            "draft/target pairs. They are inputs to the model, not outputs.",
            "alpha is treated as i.i.d. across the k draft positions. In "
            "practice later positions are accepted less often, because they "
            "are conditioned on tokens the target has not endorsed, so the "
            "curves here are mildly optimistic in k.",
            "Draft and target are assumed to run sequentially on the same "
            "device with no overlap.",
        ],
        "_literature": [
            "That a small draft model reaches a useful acceptance rate "
            "against a large target is an empirical result from the "
            "speculative-decoding papers. Nothing here demonstrates it. What "
            "is demonstrated here is (a) that verification of k tokens is "
            "nearly free given the intensity model, and (b) that the "
            "acceptance rule preserves the target distribution exactly.",
        ],
    }


# ============================================================================
# 8.  ASSEMBLE, CHECK, EMIT
# ============================================================================

def build():
    q = build_quantisation()
    wo = build_weight_only()
    sp = build_spec()

    checks = []

    def chk(name, ok, detail):
        checks.append({"name": name, "passed": bool(ok), "detail": detail})
        return ok

    # --- quantisation invariants -----------------------------------------
    r8 = q["real_matrix"]["per_bits"]["8"]
    chk("dequant round-trip is (q - zp) * scale",
        all(abs((r8["per_tensor"]["q"][i][j] - r8["per_tensor"]["zero_points"][0])
                * r8["per_tensor"]["scales"][0] - r8["per_tensor"]["deq"][i][j]) < 1e-15
            for i in range(len(r8["per_tensor"]["q"]))
            for j in range(len(r8["per_tensor"]["q"][0]))),
        "every dequantised value reproduces from its integer code and scale")

    mono = all(q["real_matrix"]["per_bits"][str(b)]["per_channel"]["err"]["rmse"] <=
               q["real_matrix"]["per_bits"][str(a)]["per_channel"]["err"]["rmse"] + 1e-18
               for a, b in zip([2, 3, 4, 5, 6], [3, 4, 5, 6, 8]))
    chk("error falls monotonically with bit width", mono,
        "per-channel RMSE on the real matrix, 2 -> 8 bits")

    sq = [q["real_matrix"]["per_bits"][str(b)]["per_channel"]["err"]["sqnr_db"]
          for b in (4, 5, 6, 8)]
    gains = [sq[i + 1] - sq[i] for i in range(len(sq) - 1)]
    per_bit = (sq[-1] - sq[0]) / (8 - 4)
    chk("one extra bit buys about 6 dB of SQNR", 4.5 < per_bit < 7.5,
        "measured %.2f dB/bit from 4 to 8 bits (theory: 6.02)" % per_bit)

    c = q["outlier_matrix"]["contrast_int8"]
    chk("per-channel beats per-tensor on the outlier matrix", c["ratio"] > 10.0,
        "quiet-channel relative RMSE %.2e per-tensor vs %.2e per-channel "
        "(%.0fx); quiet channels reach %d of 256 codes per-tensor, %d "
        "per-channel"
        % (c["quiet_rel_rmse_per_tensor"], c["quiet_rel_rmse_per_channel"],
           c["ratio"], c["quiet_channel_code_span_per_tensor"],
           c["quiet_channel_code_span_per_channel"]))

    g_pt = [r for r in q["int4_groups"]
            if r["bits"] == 4 and r["label"].startswith("drifting")]
    g_pt.sort(key=lambda r: -r["group"])
    chk("smaller INT4 groups give lower error",
        all(g_pt[i + 1]["err"]["rmse"] < g_pt[i]["err"]["rmse"]
            for i in range(len(g_pt) - 1)),
        "group %s -> RMSE %s"
        % ([r["group"] for r in g_pt],
           ["%.2e" % r["err"]["rmse"] for r in g_pt]))

    f = q["fp8"]["formats"]
    chk("FP8 limits derived from the bit fields",
        abs(f["e4m3"]["max_finite"] - 448.0) < 1e-9 and
        abs(f["e5m2"]["max_finite"] - 57344.0) < 1e-9 and
        f["e4m3"]["epsilon"] == 0.125 and f["e5m2"]["epsilon"] == 0.25,
        "E4M3 max %.0f eps %.3f; E5M2 max %.0f eps %.3f"
        % (f["e4m3"]["max_finite"], f["e4m3"]["epsilon"],
           f["e5m2"]["max_finite"], f["e5m2"]["epsilon"]))

    chk("E4M3 is more precise, E5M2 has more range",
        f["e4m3"]["epsilon"] < f["e5m2"]["epsilon"] and
        f["e5m2"]["max_finite"] > f["e4m3"]["max_finite"] and
        f["e5m2"]["min_subnormal"] < f["e4m3"]["min_subnormal"],
        "eps %.3f vs %.3f; max %.0f vs %.0f; min subnormal %.3e vs %.3e"
        % (f["e4m3"]["epsilon"], f["e5m2"]["epsilon"],
           f["e4m3"]["max_finite"], f["e5m2"]["max_finite"],
           f["e4m3"]["min_subnormal"], f["e5m2"]["min_subnormal"]))

    ug = [t for t in q["fp8"]["tensors"] if t["tensor"].startswith("gradients (uns")][0]
    chk("unscaled small gradients: E4M3 flushes, E5M2 keeps",
        ug["e4m3"]["unscaled_flushed_to_zero"] > ug["e5m2"]["unscaled_flushed_to_zero"],
        "%d of %d values flushed to zero by E4M3, %d by E5M2"
        % (ug["e4m3"]["unscaled_flushed_to_zero"], ug["n"],
           ug["e5m2"]["unscaled_flushed_to_zero"]))

    s = q["smoothquant"]
    chk("SmoothQuant migration is exact in full precision",
        s["exactness_passed"],
        "max relative Frobenius error over 21 alphas: %.3e" % s["exactness_max_rel_err"])
    chk("SmoothQuant reduces output error",
        s["improvement_per_tensor"] > 2.0,
        "per-tensor activation quantisation: %.3e -> %.3e at alpha %.2f (%.1fx)"
        % (s["baseline"]["act_per_tensor"], s["best"]["rel_err_act_per_tensor"],
           s["best"]["alpha"], s["improvement_per_tensor"]))

    # --- the decode / prefill asymmetry -----------------------------------
    i4 = [r for r in wo["rows"] if r["w_bytes"] == 0.5][0]
    chk("INT4 weight-only: decode speeds up, prefill does not",
        i4["decode_speedup"] > 3.0 and i4["prefill_speedup"] < 1.0,
        "decode %.2fx faster, prefill %.2fx (i.e. %.0f%% slower); decode is "
        "%s-bound, prefill is %s-bound"
        % (i4["decode_speedup"], i4["prefill_speedup"],
           (1 / i4["prefill_speedup"] - 1) * 100,
           i4["decode"]["bound"], i4["prefill"]["bound"]))

    # --- speculative decoding ---------------------------------------------
    v8 = [r for r in sp["verify"]["rows"] if r["k"] == 8][0]
    chk("verifying 8 tokens costs what verifying 1 costs",
        v8["time_ratio_vs_k1"] < 1.02,
        "t(k=8)/t(k=1) = %.4f; free until k ~ %.0f"
        % (v8["time_ratio_vs_k1"], sp["verify"]["k_crossover"]))

    pr = sp["proof"]
    chk("acceptance rule is analytically exact", pr["analytic"]["passed"],
        "max |emitted(x) - p(x)| = %.3e over the toy model and 4000 random "
        "(p, q) pairs" % pr["analytic"]["max_abs_err"])
    e = pr["empirical"]
    chk("simulated speculative output matches the target chain exactly",
        e["exact_rule"]["max_abs_z"] < 5.0 and
        e["exact_rule"]["impossible_outcomes"] == 0 and
        e["exact_rule"]["tv_distance"] < 3.0 * e["direct_sampling"]["tv_distance"],
        "%d draws over %d outcomes: TV %.5f vs noise floor %.5f (%.2fx), "
        "max |z| %.2f across %d cells"
        % (e["draws"], e["exact_rule"]["n_outcomes"],
           e["exact_rule"]["tv_distance"], e["direct_sampling"]["tv_distance"],
           e["exact_rule"]["tv_vs_noise_floor"], e["exact_rule"]["max_abs_z"],
           e["exact_rule"]["cells_tested"]))
    chk("the control (resample from p on rejection) is caught",
        e["naive_rule"]["max_abs_z"] > 10.0 and
        e["naive_rule"]["tv_distance"] > 5.0 * e["direct_sampling"]["tv_distance"],
        "naive rule: TV %.5f (%.1fx the noise floor), max |z| %.1f -- the "
        "test has teeth"
        % (e["naive_rule"]["tv_distance"], e["naive_rule"]["tv_vs_noise_floor"],
           e["naive_rule"]["max_abs_z"]))

    opt = sp["expected"]["optimal"]
    chk("the speedup curve has a real interior maximum",
        all(1 <= o["k_star"] < max(sp["ks"]) for o in opt) and
        all(o["speedup"] > 1.0 for o in opt),
        "; ".join("alpha %.1f -> k* %d (%.2fx)"
                  % (o["alpha"], o["k_star"], o["speedup"]) for o in opt))
    chk("optimal k grows with acceptance rate",
        all(opt[i + 1]["k_star"] >= opt[i]["k_star"] for i in range(len(opt) - 1)),
        "k* = " + str([o["k_star"] for o in opt]) +
        " for alpha = " + str([o["alpha"] for o in opt]))

    return {
        "meta": {
            "generated_by": "code/quantisation.py",
            "description": ("Quantisation, measured on real tensors, and "
                            "speculative decoding, simulated and proved. "
                            "The arithmetic-intensity model is imported from "
                            "code/inference_toy.py so nothing drifts."),
            "sources": {
                "weight_matrix": "assets/data/tf2.json params.L0.W1",
                "gradients": "assets/data/tf2.json grads.L0",
                "toy_lm": "code/inference_toy.py (target = both blocks; "
                          "draft = the same weights with the blocks "
                          "removed)",
                "cost_model": "code/inference_toy.py cost_model()",
            },
            "_labels": {
                "assumption": ("A number this file chose rather than derived. "
                               "Every one is listed in _assumptions."),
                "literature": ("A claim from published work that this file "
                               "does NOT demonstrate. Every one is listed in "
                               "_literature."),
                "constructed": ("Synthetic data built to exhibit a structure "
                                "that real tensors are reported to have."),
            },
        },
        "quant": q,
        "weight_only": wo,
        "spec": sp,
        "checks": checks,
        "all_passed": all(c["passed"] for c in checks),
    }


def main():
    outdir = os.path.join(ROOT, "assets", "data")
    os.makedirs(outdir, exist_ok=True)
    d = build()
    payload = json.dumps(sanitise(d), indent=1, allow_nan=False)
    with open(os.path.join(outdir, "quant.json"), "w") as f:
        f.write(payload)
    with open(os.path.join(outdir, "quant.js"), "w") as f:
        f.write("// GENERATED by code/quantisation.py -- do not hand-edit.\n")
        f.write("window.QUANT = " + payload + ";\n")

    line = "=" * 76
    print(line)
    print("quantisation.py — integer grids, FP8 bit layouts, and speculative")
    print("                  decoding proved exact")
    print(line)

    c = d["quant"]["outlier_matrix"]["contrast_int8"]
    print()
    print("  PER-TENSOR vs PER-CHANNEL, INT8, on a matrix with one wide channel")
    print("    per-tensor   quiet-channel rel RMSE %.4e   quiet channels reach"
          " %3d of 256 codes"
          % (c["quiet_rel_rmse_per_tensor"], c["quiet_channel_code_span_per_tensor"]))
    print("    per-channel  quiet-channel rel RMSE %.4e   quiet channels reach"
          " %3d of 256 codes"
          % (c["quiet_rel_rmse_per_channel"], c["quiet_channel_code_span_per_channel"]))
    print("    -> per-channel is %.0fx better on the quiet channels"
          % c["ratio"])
    print("    (whole-tensor rel RMSE moves only %.4e -> %.4e: the loud "
          "channel hides it)"
          % (c["per_tensor_rel_rmse"], c["per_channel_rel_rmse"]))

    print()
    print("  INT4 GROUP SIZE (drifting-scale weights)")
    for r in d["quant"]["int4_groups"]:
        if r["bits"] == 4 and r["label"].startswith("drifting"):
            print("    group %5d  eff. bits %5.2f  RMSE %.4e  SQNR %5.2f dB"
                  % (r["group"], r["effective_bits"], r["err"]["rmse"],
                     r["err"]["sqnr_db"]))

    f = d["quant"]["fp8"]["formats"]
    print()
    print("  FP8, derived from the bit fields")
    for n in ("e4m3", "e5m2"):
        print("    %s  E%d M%d  max %9.0f  min normal %.3e  min subnormal %.3e"
              "  eps %.3f  %d finite codes"
              % (n.upper(), f[n]["exponent_bits"], f[n]["mantissa_bits"],
                 f[n]["max_finite"], f[n]["min_normal"], f[n]["min_subnormal"],
                 f[n]["epsilon"], f[n]["n_finite_codes"]))

    s = d["quant"]["smoothquant"]
    print()
    print("  SMOOTHQUANT")
    print("    migration exact to %.3e over 21 alphas" % s["exactness_max_rel_err"])
    print("    output rel error %.4e -> %.4e at alpha %.2f  (%.1fx better)"
          % (s["baseline"]["act_per_tensor"], s["best"]["rel_err_act_per_tensor"],
             s["best"]["alpha"], s["improvement_per_tensor"]))

    wo = d["weight_only"]
    print()
    print("  WEIGHT-ONLY QUANTISATION on %s / %s, seq %d, batch %d"
          % (wo["model"]["name"], wo["gpu"]["name"], wo["seq"], wo["batch"]))
    print("    %-18s %10s %10s %10s %10s" %
          ("", "decode ms", "speedup", "prefill s", "speedup"))
    for r in wo["rows"]:
        print("    %-18s %10.2f %9.2fx %10.3f %9.2fx"
              % (r["label"], r["decode"]["time_s"] * 1e3, r["decode_speedup"],
                 r["prefill"]["time_s"], r["prefill_speedup"]))

    sp = d["spec"]
    print()
    print("  SPECULATIVE DECODING")
    print("    verify cost, %s on %s:" % (sp["target"]["name"], sp["gpu"]["name"]))
    for r in sp["verify"]["rows"]:
        if r["k"] in (1, 2, 4, 8, 16):
            print("      k=%2d  %.2f ms  (%.4fx of k=1)  %s-bound"
                  % (r["k"], r["time_s"] * 1e3, r["time_ratio_vs_k1"], r["bound"]))
    print("      verification stays free until k ~ %.0f" % sp["verify"]["k_crossover"])
    print("    optimal draft length:")
    for o in sp["expected"]["optimal"]:
        print("      alpha %.1f -> k* = %2d, %.2f tokens/step, %.2fx speedup"
              % (o["alpha"], o["k_star"], o["tokens_per_step"], o["speedup"]))

    print()
    print("  CHECKS")
    for c in d["checks"]:
        print("    [%s] %s" % ("PASS" if c["passed"] else "FAIL", c["name"]))
        print("           %s" % c["detail"])

    print()
    print("  wrote %s" % os.path.join(outdir, "quant.js"))
    print("  wrote %s" % os.path.join(outdir, "quant.json"))
    print(line)
    n_pass = sum(1 for c in d["checks"] if c["passed"])
    print("  %d / %d checks PASS" % (n_pass, len(d["checks"])))
    print(line)
    if not d["all_passed"]:
        raise SystemExit("QUANTISATION / SPECULATIVE DECODING CHECKS FAILED")


if __name__ == "__main__":
    main()
