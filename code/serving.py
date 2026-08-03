#!/usr/bin/env python3
"""
serving.py — the scheduler and the allocator: two pieces of operating system
that had to be written before a GPU could serve more than one user.

Pure Python standard library.

inference_toy.py established the one idea Part IV hangs on: a decode step
reads every weight in the model to do a single token's worth of arithmetic,
so decode is bandwidth-bound and the GPU's FLOPs sit idle. Two consequences
follow, and each one is a whole subsystem:

  1. SCHEDULING. Because the weights are read once per STEP and not once per
     sequence, a second sequence in the same step is nearly free. So you want
     as many sequences in flight as possible. That makes *when a slot becomes
     free* and *what blocks the step* the questions that decide throughput:

        static batching     the batch runs until the LONGEST member finishes;
                            everybody else sits finished-but-blocked
        continuous batching a finished slot is refilled on the next iteration
        chunked prefill     a 2048-token prefill no longer stalls every
                            in-flight decode behind it

  2. ALLOCATION. The KV cache is per-sequence, grows one token at a time, and
     you cannot know its final length in advance. Reserving max_seq_len per
     sequence is the only thing you can do with a contiguous allocator, and
     it wastes most of what it reserves. PagedAttention is the fix, and it is
     the fix an operating system already knows: fixed-size blocks, a page
     table per process, allocation on demand, and sharing by reference.

This file:

  * simulates a queue of requests through a static and a continuous batching
    scheduler, emitting per-iteration slot occupancy and the throughput,
    latency and utilisation that fall out of it
  * simulates unchunked vs chunked prefill and emits both latency
    distributions -- honestly, including the time-to-first-token that
    chunking costs you
  * recomputes the arithmetic-intensity-vs-batch curve and locates the
    crossover where the KV cache, which IS per-sequence, overtakes the
    weights, which are not
  * allocates KV cache two ways, contiguous and paged, block by block, and
    computes internal fragmentation and the concurrency each one buys
  * models copy-on-write prefix sharing for one system prompt fanned out to
    N users

Everything here is a MODEL. The request mix, the arrival rate, the length
distribution and the MFU are assumptions, listed in meta.assumptions and
surfaced on the pages. The hardware and model shapes are quoted public
figures, the same ones inference_toy.py uses.

Emits:
    assets/data/serving.js     (window.SERVING = {...})
    assets/data/serving.json

Run:  python3 code/serving.py
"""

import json
import math
import os

# ============================================================================
# THE MACHINE  -- same specs inference_toy.py quotes, so the two agree
# ============================================================================

MODEL = {"name": "Llama 3 8B", "params": 8.03e9, "d_model": 4096,
         "n_layers": 32, "n_heads": 32, "n_kv_heads": 8, "seq": 8192}
GPU = {"name": "H100 80GB", "bf16_dense_tflops": 495,
       "hbm_bw_bytes_per_s": 3350e9, "hbm_bytes": 80 * (1 << 30)}

W_BYTES = 2                      # bf16 weights
KV_BYTES = 2                     # bf16 KV cache
D_HEAD = MODEL["d_model"] // MODEL["n_heads"]
N_KV_HEADS = MODEL["n_kv_heads"]

# bytes of KV cache one token costs, across every layer, K and V both.
# 2 (K,V) * layers * n_kv_heads * d_head * dtype
BYTES_PER_TOKEN = (2 * MODEL["n_layers"] * N_KV_HEADS * D_HEAD * KV_BYTES)

WEIGHT_BYTES = MODEL["params"] * W_BYTES
PEAK_FLOPS = GPU["bf16_dense_tflops"] * 1e12
BW = GPU["hbm_bw_bytes_per_s"]
RIDGE = PEAK_FLOPS / BW          # FLOP per byte; below it you are BW-bound

# ---------------------------------------------------------------- ASSUMPTIONS
MFU = 0.40               # fraction of peak FLOP/s a real prefill kernel hits
TOKEN_BUDGET = 256       # tokens an engine iteration will schedule when chunking
BLOCK = 16               # tokens per KV block, vLLM's default
MAX_SEQ = 2048           # what a contiguous allocator must reserve per request
SLOTS = 6                # concurrent sequence slots the scheduler runs
INTERARRIVAL_MS = 30.0   # deterministic arrivals, not a Poisson draw
WORKSPACE_BYTES = 2e9    # activations + kernel workspace held out of the budget

# A fixed request mix. NOT sampled from a fitted trace: it is a hand-built
# mix with the shape real chat traffic has -- a long tail of long prompts
# with short outputs (summarise this document) sitting alongside short
# prompts with long outputs (chat). The anti-correlation is the point: it is
# what makes a static batch wait.
PROMPTS = [64, 512, 96, 2048, 48, 128, 1024, 80, 256, 64, 1536, 112,
           192, 768, 56, 320]
OUTPUTS = [24, 8, 60, 6, 32, 20, 5, 16, 12, 48, 4, 28,
           10, 7, 56, 14]

ASSUMPTIONS = [
    {"k": "request mix", "for": "scheduler",
     "v": str(len(PROMPTS)) + " requests, prompts " + str(min(PROMPTS)) +
          "\u2013" + str(max(PROMPTS)) + " tokens, outputs " + str(min(OUTPUTS)) +
          "\u2013" + str(max(OUTPUTS)) + " tokens",
     "why": "A hand-built mix, not a fitted trace. Long prompts are paired "
            "with short outputs and short prompts with long outputs, which "
            "is the shape that makes a static batch wait."},
    {"k": "arrival process", "for": "scheduler",
     "v": "one request every " + str(INTERARRIVAL_MS) + " ms, deterministic",
     "why": "Not Poisson. A fixed spacing keeps the two schedulers "
            "comparable \u2014 they see byte-identical input."},
    {"k": "concurrency", "for": "scheduler", "v": str(SLOTS) + " sequence slots",
     "why": "Small enough to draw slot by slot. A real server runs "
            "hundreds; the shape of the argument does not change."},
    {"k": "prefill efficiency", "for": "scheduler", "v": "MFU " + str(MFU),
     "why": "Fraction of peak dense bf16 FLOP/s a real prefill kernel "
            "reaches. Decode never gets near it and is modelled from "
            "bandwidth instead."},
    {"k": "iteration time", "for": "scheduler",
     "v": "max(bytes \u00f7 HBM bandwidth, FLOPs \u00f7 (peak \u00d7 MFU))",
     "why": "A roofline, not a measurement. Decode iterations land on the "
            "bandwidth side, prefill iterations on the compute side."},
    {"k": "length distribution", "for": "allocator",
     "v": "lognormal, median 256 tokens, \u03c3 = 0.9, clipped to [32, " +
          str(MAX_SEQ) + "]",
     "why": "Chosen to look like served traffic: most requests far shorter "
            "than the advertised context. Fragmentation is a function of "
            "THIS choice, so the shape is stated rather than hidden."},
    {"k": "block size", "for": "allocator",
     "v": str(BLOCK) + " tokens per KV block",
     "why": "vLLM's default. Larger blocks mean fewer block-table entries "
            "and more waste in the final partial block."},
    {"k": "advertised context", "for": "allocator", "v": str(MAX_SEQ) + " tokens",
     "why": "What a contiguous allocator has to reserve per request, "
            "because it cannot know the true length in advance."},
    {"k": "held-out workspace", "for": "allocator",
     "v": str(int(WORKSPACE_BYTES / 1e9)) + " GB of HBM",
     "why": "Activations, CUDA graphs and kernel workspace, kept out of the "
            "KV budget. A real server tunes this; the concurrency numbers "
            "move with it."},
]


# ============================================================================
# COST MODEL  -- one engine iteration
# ============================================================================

def iteration_cost(prefill_chunk, prefill_ctx, decode_lens):
    """
    Time for one engine iteration, as a roofline.

    An iteration reads every weight ONCE, whatever it is doing, then reads
    the KV cache of every sequence that is decoding, then writes the KV of
    whatever it just computed. The FLOPs are 2*params per token plus
    attention's own term.

    Returning max(bytes/BW, flops/(peak*MFU)) is the honest two-sided bound:
    a decode-only iteration is pinned by the first term and a prefill by the
    second, and you can see which one by comparing them.
    """
    n_dec = len(decode_lens)
    tokens = prefill_chunk + n_dec

    # FLOPs. The 2*params*token term dominates; attention adds its own.
    flops = 2 * MODEL["params"] * tokens
    attn_const = 4 * MODEL["n_layers"] * N_KV_HEADS * D_HEAD
    for L in decode_lens:                       # one query vs L keys
        flops += attn_const * L
    if prefill_chunk:                           # chunk vs everything before it
        flops += attn_const * prefill_chunk * (prefill_ctx + prefill_chunk / 2.0)

    # BYTES. Weights once. Then every decoding sequence's whole cache.
    byts = WEIGHT_BYTES
    for L in decode_lens:
        byts += L * BYTES_PER_TOKEN
    if prefill_chunk:
        byts += (prefill_ctx + prefill_chunk) * BYTES_PER_TOKEN

    t_bw = byts / BW
    t_fl = flops / (PEAK_FLOPS * MFU)
    return {
        "ms": max(t_bw, t_fl) * 1e3,
        "bw_ms": t_bw * 1e3, "flops_ms": t_fl * 1e3,
        "bound": "memory" if t_bw >= t_fl else "compute",
        "tokens": tokens, "flops": flops, "bytes": byts,
        "intensity": flops / byts,
    }


# ============================================================================
# 1. THE SCHEDULER  -- static vs continuous, same requests, same machine
# ============================================================================

def make_requests():
    return [{"id": i, "arrive": i * INTERARRIVAL_MS,
             "prompt": PROMPTS[i], "out": OUTPUTS[i]}
            for i in range(len(PROMPTS))]


def simulate(policy, chunked=False, slots=SLOTS, budget=TOKEN_BUDGET):
    """
    Run the request mix through one scheduler.

      policy "static"     : a batch is admitted, runs to completion, and only
                            then are ALL its slots released together. A
                            sequence that finished on iteration 3 keeps its
                            slot -- and its padding -- until iteration 40.
      policy "continuous" : slots are refilled at the top of every iteration.

      chunked False       : a prefill runs alone and takes the whole prompt in
                            one iteration. Every in-flight decode waits.
      chunked True        : each iteration schedules decodes first, then
                            spends whatever is left of `budget` on a slice of
                            one prefill.

    Emits a per-iteration record of what every slot was doing, which is what
    the page animates, plus per-request timings.
    """
    reqs = make_requests()
    for r in reqs:
        r.update(state="waiting", prefilled=0, generated=0,
                 admit_ms=None, ttft_ms=None, finish_ms=None, tok_ms=[])

    slot_of = [None] * slots        # slot index -> request id
    queue = list(range(len(reqs)))  # not yet admitted, FCFS
    timeline = []
    clock = 0.0
    idle_ms = 0.0
    busy_ms = 0.0
    guard = 0

    def running():
        return [reqs[i] for i in slot_of if i is not None]

    while any(r["state"] != "done" for r in reqs):
        guard += 1
        if guard > 20000:
            raise SystemExit("scheduler did not converge")

        live = [r for r in running() if r["state"] != "done"]
        # ---- admission -------------------------------------------------
        # static: only when the whole previous batch has drained.
        # continuous: any free slot, every iteration.
        may_admit = (policy == "continuous") or all(i is None for i in slot_of)
        if may_admit:
            for s in range(slots):
                if slot_of[s] is not None:
                    continue
                nxt = None
                for qi in queue:
                    if reqs[qi]["arrive"] <= clock + 1e-9:
                        nxt = qi
                        break
                if nxt is None:
                    break
                queue.remove(nxt)
                slot_of[s] = nxt
                reqs[nxt]["state"] = "prefill"
                reqs[nxt]["admit_ms"] = clock

        live = [r for r in running() if r["state"] != "done"]
        if not live:
            # nothing runnable: advance the clock to the next arrival
            pend = [reqs[i]["arrive"] for i in queue]
            if not pend:
                break
            nxt_t = min(pend)
            idle_ms += max(0.0, nxt_t - clock)
            clock = max(clock, nxt_t)
            continue

        # ---- what runs this iteration ----------------------------------
        decoders = [r for r in live if r["state"] == "decode"]
        pf = next((r for r in live if r["state"] == "prefill"), None)

        if chunked:
            dec = decoders[:budget]
            room = budget - len(dec)
            chunk = min(room, pf["prompt"] - pf["prefilled"]) if pf else 0
        elif pf is not None:
            # unchunked, prefill-priority: the prefill runs ALONE.
            dec = []
            chunk = pf["prompt"] - pf["prefilled"]
        else:
            dec = decoders
            chunk = 0

        ctx = pf["prefilled"] if (pf and chunk) else 0
        cost = iteration_cost(chunk, ctx, [r["prompt"] + r["generated"]
                                           for r in dec])
        t0, t1 = clock, clock + cost["ms"]

        # ---- slot occupancy record, before state changes ---------------
        # P prefill  D decode  W admitted but stalled  B finished, still
        # holding the slot (static only)  -  empty
        row = []
        for s in range(slots):
            i = slot_of[s]
            if i is None:
                row.append(["-", None])
                continue
            r = reqs[i]
            if pf is not None and r is pf and chunk:
                row.append(["P", i])
            elif r in dec:
                row.append(["D", i])
            elif r["state"] == "done":
                row.append(["B", i])
            else:
                row.append(["W", i])

        # ---- advance ---------------------------------------------------
        for r in dec:
            r["generated"] += 1
            r["tok_ms"].append(t1)
            if r["generated"] >= r["out"]:
                r["state"] = "done"
                r["finish_ms"] = t1
        if pf is not None and chunk:
            pf["prefilled"] += chunk
            if pf["prefilled"] >= pf["prompt"]:
                # the last prefill chunk emits the first output token
                pf["generated"] = 1
                pf["ttft_ms"] = t1
                pf["tok_ms"].append(t1)
                pf["state"] = "decode" if pf["out"] > 1 else "done"
                if pf["state"] == "done":
                    pf["finish_ms"] = t1

        busy_ms += cost["ms"]
        clock = t1

        # ---- slot release ----------------------------------------------
        if policy == "continuous":
            for s in range(slots):
                i = slot_of[s]
                if i is not None and reqs[i]["state"] == "done":
                    slot_of[s] = None
        else:
            if all(i is None or reqs[i]["state"] == "done" for i in slot_of):
                slot_of = [None] * slots

        timeline.append({
            "t0": round(t0, 3), "t1": round(t1, 3), "ms": round(cost["ms"], 3),
            "slots": row, "tokens": cost["tokens"],
            "prefill_tokens": chunk, "decode_tokens": len(dec),
            "bound": cost["bound"],
            "intensity": round(cost["intensity"], 3),
        })

    # ---- metrics -------------------------------------------------------
    for r in reqs:
        r["latency_ms"] = r["finish_ms"] - r["arrive"]
        r["queue_ms"] = r["admit_ms"] - r["arrive"]
        r["ttft_from_arrival_ms"] = r["ttft_ms"] - r["arrive"]

    total_ms = clock
    out_tokens = sum(r["generated"] for r in reqs)
    cell_work = sum(1 for it in timeline for c in it["slots"]
                    if c[0] in ("P", "D"))
    # time-weighted, because iterations are wildly different lengths
    w_work = sum(it["ms"] * sum(1 for c in it["slots"] if c[0] in ("P", "D"))
                 for it in timeline)
    w_block = sum(it["ms"] * sum(1 for c in it["slots"] if c[0] == "B")
                  for it in timeline)
    w_stall = sum(it["ms"] * sum(1 for c in it["slots"] if c[0] == "W")
                  for it in timeline)
    w_empty = sum(it["ms"] * sum(1 for c in it["slots"] if c[0] == "-")
                  for it in timeline)
    denom = total_ms * SLOTS

    return {
        "policy": policy, "chunked": chunked, "slots": slots,
        "budget": budget if chunked else None,
        "timeline": timeline,
        "requests": [{"id": r["id"], "prompt": r["prompt"], "out": r["out"],
                      "arrive_ms": round(r["arrive"], 2),
                      "admit_ms": round(r["admit_ms"], 2),
                      "ttft_ms": round(r["ttft_from_arrival_ms"], 2),
                      "finish_ms": round(r["finish_ms"], 2),
                      "latency_ms": round(r["latency_ms"], 2),
                      "itl_ms": [round(b - a, 3) for a, b
                                 in zip(r["tok_ms"], r["tok_ms"][1:])]}
                     for r in reqs],
        "iterations": len(timeline),
        "makespan_ms": round(total_ms, 2),
        "busy_ms": round(busy_ms, 2), "idle_ms": round(idle_ms, 2),
        "output_tokens": out_tokens,
        "throughput_tok_s": round(out_tokens / (total_ms / 1e3), 2),
        "mean_latency_ms": round(sum(r["latency_ms"] for r in reqs) / len(reqs), 2),
        "p99_latency_ms": round(pct([r["latency_ms"] for r in reqs], 99), 2),
        "mean_ttft_ms": round(sum(r["ttft_from_arrival_ms"] for r in reqs) / len(reqs), 2),
        "slot_utilisation": round(w_work / denom, 4),
        "blocked_share": round(w_block / denom, 4),
        "stalled_share": round(w_stall / denom, 4),
        "empty_share": round(w_empty / denom, 4),
        "work_cells": cell_work,
        "cells_total": len(timeline) * slots,
    }


def pct(vals, p):
    """Linear-interpolated percentile. Small samples, so no numpy games."""
    if not vals:
        return 0.0
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def latency_stats(sim):
    """TTFT and inter-token latency distributions for one run."""
    ttft = [r["ttft_ms"] for r in sim["requests"]]
    itl = [g for r in sim["requests"] for g in r["itl_ms"]]
    def block(v, name):
        return {"name": name, "n": len(v),
                "mean": round(sum(v) / len(v), 2) if v else 0,
                "p50": round(pct(v, 50), 2), "p90": round(pct(v, 90), 2),
                "p95": round(pct(v, 95), 2), "p99": round(pct(v, 99), 2),
                "max": round(max(v), 2) if v else 0,
                "min": round(min(v), 2) if v else 0,
                "values": [round(x, 2) for x in v]}
    return {"ttft": block(ttft, "time to first token"),
            "itl": block(itl, "inter-token latency")}


def histogram(vals, edges):
    """Counts per bin, for drawing the two ITL distributions side by side."""
    bins = [0] * (len(edges) + 1)
    for v in vals:
        placed = False
        for i, e in enumerate(edges):
            if v < e:
                bins[i] += 1
                placed = True
                break
        if not placed:
            bins[-1] += 1
    labels = []
    prev = 0
    for e in edges:
        labels.append(f"{prev:g}–{e:g}")
        prev = e
    labels.append(f"≥{edges[-1]:g}")
    return {"edges": edges, "labels": labels, "counts": bins}


# ============================================================================
# 2. WHY DECODE BATCHING IS NEARLY FREE
# ============================================================================

def decode_batching(seqs=(512, 2048, 8192)):
    """
    Weights are read once per STEP. A second sequence in the same step adds
    FLOPs and its own KV cache, but not a second read of the weights. So
    intensity climbs almost linearly with batch -- until the per-sequence
    term overtakes the fixed one.

    Two crossovers, and they are different questions:

      kv_crossover     batch at which KV bytes == weight bytes. Past it, the
                       thing you are reading is mostly cache, and adding a
                       sequence stops being free.
      ridge_crossover  batch at which intensity reaches the device ridge, so
                       decode finally becomes compute-bound. With GQA and a
                       long context it may not exist at all: the curve has a
                       horizontal asymptote at 2*params / kv_bytes_per_seq,
                       and if that sits below the ridge, no batch size on
                       earth makes this decode compute-bound.
    """
    attn_const = 4 * MODEL["n_layers"] * N_KV_HEADS * D_HEAD

    def intensity(batch, seq):
        flops = 2 * MODEL["params"] * batch + attn_const * batch * seq
        byts = WEIGHT_BYTES + batch * seq * BYTES_PER_TOKEN
        return flops / byts

    out = []
    for seq in seqs:
        kv_per_seq = seq * BYTES_PER_TOKEN
        rows = []
        b = 1
        while b <= 1024:
            kv = b * kv_per_seq
            i = intensity(b, seq)
            rows.append({"batch": b, "intensity": round(i, 3),
                         "weight_bytes": WEIGHT_BYTES, "kv_bytes": kv,
                         "kv_share": round(kv / (WEIGHT_BYTES + kv), 4),
                         "bound": "memory" if i < RIDGE else "compute",
                         "flops_utilisation": round(min(1.0, i / RIDGE), 5),
                         "per_seq_cost_share": round(
                             kv_per_seq / (WEIGHT_BYTES / b + kv_per_seq), 4)})
            b *= 2

        kv_cross = WEIGHT_BYTES / kv_per_seq            # kv bytes == weights
        asym = (2 * MODEL["params"] + attn_const * seq) / kv_per_seq
        ridge_cross = None
        if asym > RIDGE:
            # solve intensity(b) = RIDGE exactly: linear in b
            num = RIDGE * WEIGHT_BYTES
            den = (2 * MODEL["params"] + attn_const * seq) - RIDGE * kv_per_seq
            ridge_cross = num / den
        out.append({
            "seq": seq,
            "kv_bytes_per_seq": kv_per_seq,
            "rows": rows,
            "asymptote": round(asym, 3),
            "kv_crossover_batch": round(kv_cross, 2),
            "ridge_crossover_batch": (round(ridge_cross, 2)
                                      if ridge_cross else None),
            "ridge_reachable": ridge_cross is not None,
        })
    return {"model": MODEL["name"], "gpu": GPU["name"],
            "ridge": round(RIDGE, 2),
            "weight_bytes": WEIGHT_BYTES,
            "bytes_per_token": BYTES_PER_TOKEN,
            "by_seq": out,
            "why": "The weights term is FIXED per step and the KV term is "
                   "PER SEQUENCE. Batch is free exactly as long as the fixed "
                   "term dominates the bytes, and stops being free the "
                   "moment the per-sequence term catches it."}


# ============================================================================
# 3. THE ALLOCATOR  -- contiguous vs paged
# ============================================================================

def lengths(n, seed=20240917):
    """
    A deterministic lognormal length sample. Reproducible without numpy:
    a linear congruential generator feeding Box-Muller.

    Median 256 tokens, sigma 0.9 -- most requests far shorter than the
    advertised 2048-token context, which is precisely why reserving the
    context is so expensive.
    """
    s = seed
    def u():
        nonlocal s
        s = (s * 1103515245 + 12345) % (1 << 31)
        return (s + 1) / float((1 << 31) + 1)
    out = []
    while len(out) < n:
        u1, u2 = u(), u()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        v = int(round(math.exp(math.log(256.0) + 0.9 * z)))
        out.append(max(32, min(MAX_SEQ, v)))
    return out


def allocator(n_pop=256, n_demo=8):
    """
    Allocate the same sequences two ways.

    CONTIGUOUS: the allocator must hand back one range it will never have to
    move, and it does not know the final length, so it reserves MAX_SEQ. Every
    token between the true length and MAX_SEQ is internal fragmentation: paid
    for, never touched.

    PAGED: fixed-size blocks handed out on demand, plus a block table per
    sequence mapping logical block -> physical block. Attention reads through
    the table, so the blocks need not be adjacent. Waste is bounded by ONE
    partial block per sequence, whatever the length.

    This is virtual memory. The block is a page, the block table is a page
    table, and the waste that disappears is external fragmentation.
    """
    pop = lengths(n_pop)
    blk_bytes = BLOCK * BYTES_PER_TOKEN
    res_blocks = MAX_SEQ // BLOCK              # blocks in one reservation

    # ---- population statistics ----------------------------------------
    cont_reserved = len(pop) * MAX_SEQ * BYTES_PER_TOKEN
    used = sum(pop) * BYTES_PER_TOKEN
    paged_blocks = sum(math.ceil(L / BLOCK) for L in pop)
    paged_alloc = paged_blocks * blk_bytes

    stats = {
        "n": len(pop),
        "mean_len": round(sum(pop) / len(pop), 1),
        "median_len": round(pct(pop, 50), 1),
        "p90_len": round(pct(pop, 90), 1),
        "max_len": max(pop), "min_len": min(pop),
        "max_seq": MAX_SEQ, "block": BLOCK,
        "bytes_per_token": BYTES_PER_TOKEN,
        "block_bytes": blk_bytes,
        "blocks_per_reservation": res_blocks,
        "contiguous_bytes": cont_reserved,
        "paged_bytes": paged_alloc,
        "used_bytes": used,
        "contiguous_waste_bytes": cont_reserved - used,
        "paged_waste_bytes": paged_alloc - used,
        "contiguous_waste_frac": round((cont_reserved - used) / cont_reserved, 4),
        "paged_waste_frac": round((paged_alloc - used) / paged_alloc, 4),
        "paged_vs_contiguous": round(cont_reserved / paged_alloc, 2),
        "length_hist": histogram(pop, [64, 128, 256, 512, 1024, 2048]),
    }

    # ---- the budget, and what fits in it -------------------------------
    kv_budget = GPU["hbm_bytes"] - WEIGHT_BYTES - WORKSPACE_BYTES
    mean_blocks = paged_blocks / len(pop)
    fit_cont = int(kv_budget // (MAX_SEQ * BYTES_PER_TOKEN))
    fit_paged = int(kv_budget // (mean_blocks * blk_bytes))
    budget = {
        "hbm_bytes": GPU["hbm_bytes"],
        "weight_bytes": WEIGHT_BYTES,
        "workspace_bytes": WORKSPACE_BYTES,
        "kv_budget_bytes": kv_budget,
        "reservation_bytes": MAX_SEQ * BYTES_PER_TOKEN,
        "mean_blocks_per_seq": round(mean_blocks, 2),
        "mean_paged_bytes_per_seq": round(mean_blocks * blk_bytes, 1),
        "concurrent_contiguous": fit_cont,
        "concurrent_paged": fit_paged,
        "ratio": round(fit_paged / fit_cont, 2),
    }

    # ---- the drawable demo, at real block granularity -------------------
    # A pool exactly the size of n_demo contiguous reservations. Under
    # contiguous allocation it holds n_demo sequences by construction. Under
    # paging it holds as many as actually fit.
    pool_blocks = n_demo * res_blocks
    demo = pop[:n_demo]

    contiguous = []
    for i, L in enumerate(demo):
        u_blocks = math.ceil(L / BLOCK)
        contiguous.append({
            "seq": i, "len": L,
            "base_block": i * res_blocks,
            "reserved_blocks": res_blocks,
            "used_blocks": u_blocks,
            "tokens_in_last_block": L - (u_blocks - 1) * BLOCK,
            "waste_tokens": MAX_SEQ - L,
            "waste_frac": round((MAX_SEQ - L) / MAX_SEQ, 4),
        })

    # PAGED: walk the SAME population into the SAME pool, block by block, in
    # arrival order, until the next sequence does not fit.
    events, tables, paged_seqs = [], [], []
    free = 0
    for i, L in enumerate(pop):
        need = math.ceil(L / BLOCK)
        if free + need > pool_blocks:
            break
        tbl = []
        for b in range(need):
            phys = free + b
            lo = b * BLOCK
            hi = min(L, lo + BLOCK)
            events.append([i, phys, hi - lo])     # seq, physical block, tokens
            tbl.append(phys)
        free += need
        tables.append(tbl)
        paged_seqs.append({"seq": i, "len": L, "blocks": need,
                           "tokens_in_last_block": L - (need - 1) * BLOCK})

    demo_block = {
        "pool_blocks": pool_blocks,
        "block": BLOCK, "block_bytes": blk_bytes,
        "blocks_per_reservation": res_blocks,
        "contiguous": contiguous,
        "contiguous_seqs": len(contiguous),
        "contiguous_used_blocks": sum(c["used_blocks"] for c in contiguous),
        "paged": paged_seqs,
        "paged_seqs": len(paged_seqs),
        "paged_used_blocks": free,
        "paged_free_blocks": pool_blocks - free,
        "events": events,
        "tables": tables,
        "fit_ratio": round(len(paged_seqs) / len(contiguous), 2),
    }

    return {"stats": stats, "budget": budget, "demo": demo_block,
            "population": pop,
            "os_analogy": "Block = page. Block table = page table. Allocation "
                          "on demand = demand paging. Sharing a prefix by "
                          "reference with a copy on first write = "
                          "copy-on-write fork. None of this is new; it is "
                          "1960s operating systems applied to a tensor."}


# ============================================================================
# 4. COPY-ON-WRITE PREFIX SHARING
# ============================================================================

SYS_PROMPT = 1000        # tokens of shared system prompt
USER_IN = 200            # tokens each user adds
USER_OUT = 200           # tokens each user generates


def prefix_sharing(fanouts=(1, 2, 4, 8, 16, 32, 64, 128)):
    """
    One system prompt, N users. Every sequence begins with the same
    SYS_PROMPT tokens, so every sequence computes the same K and V for them.

    Under paging those blocks can be SHARED: N block tables point at the same
    physical blocks, refcounted. The prompt is not a multiple of the block
    size, so the block straddling the boundary holds shared tokens and
    private tokens together -- that one block is copied on first write, per
    sequence. Everything fully inside the prefix is shared outright.

    That partial block is the whole reason this is called copy-on-write and
    not just "sharing".
    """
    blk = BLOCK
    shared_full = SYS_PROMPT // blk                 # blocks entirely prefix
    boundary_shared_tokens = SYS_PROMPT - shared_full * blk
    has_boundary = boundary_shared_tokens > 0

    per_seq_total = SYS_PROMPT + USER_IN + USER_OUT
    unshared_blocks_each = math.ceil(per_seq_total / blk)

    # with sharing: the shared full blocks once, then per sequence a copy of
    # the boundary block (if any) plus blocks for the rest of its own tokens
    private_tokens = per_seq_total - SYS_PROMPT
    if has_boundary:
        after_boundary = private_tokens - (blk - boundary_shared_tokens)
        private_blocks_each = 1 + max(0, math.ceil(after_boundary / blk))
    else:
        private_blocks_each = math.ceil(private_tokens / blk)

    blk_bytes = blk * BYTES_PER_TOKEN
    rows = []
    for n in fanouts:
        without = n * unshared_blocks_each
        with_ = shared_full + n * private_blocks_each
        rows.append({
            "n": n,
            "blocks_without": without, "blocks_with": with_,
            "blocks_saved": without - with_,
            "bytes_without": without * blk_bytes,
            "bytes_with": with_ * blk_bytes,
            "bytes_saved": (without - with_) * blk_bytes,
            "saving_frac": round((without - with_) / without, 4),
            "ratio": round(without / with_, 2),
        })

    curve = []
    for n in range(1, 129):
        without = n * unshared_blocks_each
        with_ = shared_full + n * private_blocks_each
        curve.append([n, without, with_])

    return {
        "system_prompt_tokens": SYS_PROMPT,
        "user_in_tokens": USER_IN, "user_out_tokens": USER_OUT,
        "block": blk, "block_bytes": blk_bytes,
        "shared_full_blocks": shared_full,
        "boundary_block_shared_tokens": boundary_shared_tokens,
        "has_copy_on_write_block": has_boundary,
        "blocks_per_seq_unshared": unshared_blocks_each,
        "private_blocks_per_seq": private_blocks_each,
        "rows": rows, "curve": curve,
        "note": "The prefix is " + str(SYS_PROMPT) + " tokens and the block "
                "is " + str(blk) + ", so " + str(shared_full) + " blocks are "
                "wholly inside the prefix and shared by reference. The "
                + str(blk) + "-token block at the boundary holds "
                + str(boundary_shared_tokens) + " shared tokens and the "
                "first " + str(blk - boundary_shared_tokens) + " private "
                "ones, so it is copied per sequence on first write.",
    }


# ============================================================================
# BUILD
# ============================================================================

def build():
    static = simulate("static", chunked=False)
    cont = simulate("continuous", chunked=False)
    chunk = simulate("continuous", chunked=True)

    edges = [6, 10, 20, 40, 80, 160]
    unch_itl = [g for r in cont["requests"] for g in r["itl_ms"]]
    ch_itl = [g for r in chunk["requests"] for g in r["itl_ms"]]

    long_i = PROMPTS.index(max(PROMPTS))
    ttft_long = {
        "request": long_i, "prompt": max(PROMPTS),
        "unchunked_ms": cont["requests"][long_i]["ttft_ms"],
        "chunked_ms": chunk["requests"][long_i]["ttft_ms"],
    }
    ttft_long["delta_ms"] = round(
        ttft_long["chunked_ms"] - ttft_long["unchunked_ms"], 2)

    alloc = allocator()

    return {
        "meta": {
            "generated_by": "code/serving.py",
            "description": "Static vs continuous batching, chunked prefill, "
                           "the decode-batching intensity curve, and paged "
                           "KV allocation with copy-on-write prefix sharing.",
            "model": MODEL, "gpu": GPU,
            "bytes_per_token": BYTES_PER_TOKEN,
            "weight_bytes": WEIGHT_BYTES,
            "ridge_flops_per_byte": round(RIDGE, 2),
            "slots": SLOTS, "token_budget": TOKEN_BUDGET,
            "block": BLOCK, "max_seq": MAX_SEQ, "mfu": MFU,
            "interarrival_ms": INTERARRIVAL_MS,
            "assumptions": ASSUMPTIONS,
            "_source_note": "Model and GPU specifications are quoted public "
                            "figures, external to this simulation, and match "
                            "the ones in code/inference_toy.py. Everything "
                            "else on this page is simulated here.",
        },
        "workload": {"prompts": PROMPTS, "outputs": OUTPUTS,
                     "n": len(PROMPTS),
                     "prompt_tokens": sum(PROMPTS),
                     "output_tokens": sum(OUTPUTS)},
        "batching": {
            "static": static, "continuous": cont,
            "gain": {
                "throughput_x": round(cont["throughput_tok_s"] /
                                      static["throughput_tok_s"], 2),
                "latency_x": round(static["mean_latency_ms"] /
                                   cont["mean_latency_ms"], 2),
                "utilisation_x": round(cont["slot_utilisation"] /
                                       static["slot_utilisation"], 2),
                "blocked_slot_share_static": static["blocked_share"],
            },
            "why": "Both schedulers see the same requests on the same "
                   "machine. The only difference is when a finished "
                   "sequence gives its slot back.",
        },
        "chunked_prefill": {
            "unchunked": {"sim": chunk_summary(cont),
                          "stats": latency_stats(cont)},
            "chunked": {"sim": chunk_summary(chunk),
                        "stats": latency_stats(chunk)},
            "itl_hist": {
                "unchunked": histogram(unch_itl, edges),
                "chunked": histogram(ch_itl, edges),
            },
            "ttft_longest_prompt": ttft_long,
            "budget": TOKEN_BUDGET,
            # the unchunked run IS batching.continuous -- same scheduler, same
            # requests, chunking off -- so its timeline is not duplicated here
            "timeline_unchunked_is": "batching.continuous.timeline",
            "timeline_chunked": chunk["timeline"],
            "why": "An unchunked prefill runs alone: every in-flight decode "
                   "waits the whole length of it. That is head-of-line "
                   "blocking, and it is why p99 inter-token latency is so "
                   "much worse than the mean.",
        },
        "decode_batching": decode_batching(),
        "allocator": alloc,
        "prefix_sharing": prefix_sharing(),
    }


def chunk_summary(sim):
    """The scalar half of a run, without the timeline (which is emitted once)."""
    return {k: sim[k] for k in (
        "policy", "chunked", "budget", "iterations", "makespan_ms",
        "output_tokens", "throughput_tok_s", "mean_latency_ms",
        "p99_latency_ms", "mean_ttft_ms", "slot_utilisation",
        "blocked_share", "stalled_share", "empty_share", "requests")}


# ============================================================================
# VERIFY  -- nothing ships unless the accounting balances
# ============================================================================

def verify(d):
    fails = []

    def chk(ok, msg):
        if not ok:
            fails.append(msg)

    W = d["workload"]
    for name in ("static", "continuous"):
        s = d["batching"][name]
        chk(s["output_tokens"] == W["output_tokens"],
            f"{name}: emitted {s['output_tokens']} tokens, expected "
            f"{W['output_tokens']}")
        chk(all(r["finish_ms"] is not None for r in s["requests"]),
            f"{name}: a request never finished")
        # every request appears in at most one slot per iteration
        for it in s["timeline"]:
            ids = [c[1] for c in it["slots"] if c[1] is not None]
            chk(len(ids) == len(set(ids)),
                f"{name}: a request occupied two slots in one iteration")
        # the four occupancy states must partition the grid
        tot = sum(len(it["slots"]) for it in s["timeline"])
        chk(tot == s["cells_total"], f"{name}: slot grid does not close")
        chk(abs(s["slot_utilisation"] + s["blocked_share"] +
                s["stalled_share"] + s["empty_share"] - 1.0) < 1e-6,
            f"{name}: occupancy shares sum to "
            f"{s['slot_utilisation'] + s['blocked_share'] + s['stalled_share'] + s['empty_share']}")

    st, co = d["batching"]["static"], d["batching"]["continuous"]
    chk(co["throughput_tok_s"] > st["throughput_tok_s"],
        "continuous batching did not beat static on throughput")
    chk(co["mean_latency_ms"] < st["mean_latency_ms"],
        "continuous batching did not beat static on mean latency")
    chk(st["blocked_share"] > 0,
        "static batching showed no finished-but-blocked slot time -- "
        "the workload is not exercising the thing the page claims")
    chk(co["blocked_share"] == 0,
        "continuous batching blocked a finished slot; it must not")

    cp = d["chunked_prefill"]
    u, c = cp["unchunked"]["stats"], cp["chunked"]["stats"]
    chk(c["itl"]["p99"] < u["itl"]["p99"],
        "chunking did not improve p99 inter-token latency")
    chk(c["itl"]["max"] < u["itl"]["max"],
        "chunking did not cut the worst inter-token stall")
    chk(cp["ttft_longest_prompt"]["delta_ms"] > 0,
        "chunking did not cost the longest prompt any time to first token — "
        "the honest tradeoff is missing")
    chk(cp["unchunked"]["sim"]["output_tokens"] ==
        cp["chunked"]["sim"]["output_tokens"],
        "chunked and unchunked runs produced different token counts")

    db = d["decode_batching"]
    for s in db["by_seq"]:
        prev = 0
        for r in s["rows"]:
            chk(r["intensity"] > prev,
                f"seq {s['seq']}: intensity not monotone in batch")
            prev = r["intensity"]
        chk(s["rows"][-1]["intensity"] < s["asymptote"],
            f"seq {s['seq']}: curve crossed its own asymptote")
        # kv crossover really is where the two byte terms are equal
        b = s["kv_crossover_batch"]          # rounded for display
        chk(abs(b * s["kv_bytes_per_seq"] - db["weight_bytes"]) /
            db["weight_bytes"] < 1e-3,
            f"seq {s['seq']}: kv crossover does not balance the bytes")

    a = d["allocator"]
    st_, dm, bd = a["stats"], a["demo"], a["budget"]
    chk(st_["contiguous_bytes"] == st_["used_bytes"] + st_["contiguous_waste_bytes"],
        "contiguous accounting does not balance")
    chk(st_["paged_bytes"] == st_["used_bytes"] + st_["paged_waste_bytes"],
        "paged accounting does not balance")
    chk(st_["paged_waste_bytes"] <= st_["n"] * (BLOCK - 1) * BYTES_PER_TOKEN,
        "paged waste exceeded one partial block per sequence")
    chk(st_["paged_bytes"] < st_["contiguous_bytes"],
        "paging did not save anything")
    # blocks must close: used + free == pool, and every event is one block
    chk(dm["paged_used_blocks"] + dm["paged_free_blocks"] == dm["pool_blocks"],
        "block pool does not close")
    chk(len(dm["events"]) == dm["paged_used_blocks"],
        "one allocation event per block was expected")
    chk(sum(len(t) for t in dm["tables"]) == dm["paged_used_blocks"],
        "block tables do not account for every allocated block")
    phys = [e[1] for e in dm["events"]]
    chk(len(phys) == len(set(phys)), "a physical block was handed out twice")
    chk(sum(e[2] for e in dm["events"]) == sum(p["len"] for p in dm["paged"]),
        "block token counts do not sum to the sequence lengths")
    chk(dm["paged_seqs"] > dm["contiguous_seqs"],
        "paging fit no more sequences into the same pool")
    chk(bd["concurrent_paged"] > bd["concurrent_contiguous"],
        "paging did not raise concurrency at a fixed budget")

    ps = d["prefix_sharing"]
    for r in ps["rows"]:
        expect = ps["shared_full_blocks"] + r["n"] * ps["private_blocks_per_seq"]
        chk(r["blocks_with"] == expect,
            f"fan-out {r['n']}: shared block count does not reconstruct")
        chk(r["blocks_saved"] == r["blocks_without"] - r["blocks_with"],
            f"fan-out {r['n']}: saving does not balance")
    chk(ps["rows"][0]["blocks_saved"] >= 0,
        "sharing cost blocks at fan-out 1")
    chk(ps["rows"][-1]["saving_frac"] > 0.5,
        "prefix sharing saved less than half at the largest fan-out")

    return fails


# ============================================================================
# MAIN
# ============================================================================

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(os.path.dirname(here), "assets", "data")
    os.makedirs(outdir, exist_ok=True)
    d = build()
    fails = verify(d)

    payload = json.dumps(d, indent=1, allow_nan=False)
    open(os.path.join(outdir, "serving.json"), "w").write(payload)
    with open(os.path.join(outdir, "serving.js"), "w") as f:
        f.write("// GENERATED by code/serving.py -- do not hand-edit.\n")
        f.write("window.SERVING = " + payload + ";\n")

    st, co = d["batching"]["static"], d["batching"]["continuous"]
    g = d["batching"]["gain"]
    cp = d["chunked_prefill"]
    db = d["decode_batching"]
    a = d["allocator"]
    ps = d["prefix_sharing"]

    print("=" * 72)
    print("serving.py — batching, chunked prefill, and paged KV")
    print("=" * 72)
    print(f"  {MODEL['name']} on {GPU['name']}   "
          f"{BYTES_PER_TOKEN/1024:.0f} KiB of KV per token   "
          f"ridge {RIDGE:.1f} FLOP/byte")
    print(f"  workload: {d['workload']['n']} requests, "
          f"{d['workload']['prompt_tokens']} prompt tokens, "
          f"{d['workload']['output_tokens']} output tokens, "
          f"{SLOTS} slots")
    print()
    print("  1. static vs continuous batching")
    hdr = f"    {'':12s}{'iters':>7s}{'makespan':>11s}{'tok/s':>9s}{'mean lat':>11s}{'util':>8s}{'blocked':>9s}"
    print(hdr)
    for nm, s in (("static", st), ("continuous", co)):
        print(f"    {nm:12s}{s['iterations']:7d}{s['makespan_ms']:10.1f}ms"
              f"{s['throughput_tok_s']:9.1f}{s['mean_latency_ms']:10.1f}ms"
              f"{s['slot_utilisation']*100:7.1f}%{s['blocked_share']*100:8.1f}%")
    print(f"    -> continuous is {g['throughput_x']}x the throughput, "
          f"{g['latency_x']}x lower mean latency, "
          f"{g['utilisation_x']}x the slot utilisation")
    print(f"    -> static spent {g['blocked_slot_share_static']*100:.1f}% of all "
          f"slot-time on sequences that had already finished")
    print()
    print(f"  2. chunked prefill (token budget {TOKEN_BUDGET})")
    u, c = cp["unchunked"]["stats"], cp["chunked"]["stats"]
    print(f"    {'':12s}{'ITL p50':>10s}{'ITL p99':>10s}{'ITL max':>10s}"
          f"{'TTFT mean':>12s}{'tok/s':>9s}")
    for nm, s, sim in (("unchunked", u, cp["unchunked"]["sim"]),
                       ("chunked", c, cp["chunked"]["sim"])):
        print(f"    {nm:12s}{s['itl']['p50']:9.1f}ms{s['itl']['p99']:9.1f}ms"
              f"{s['itl']['max']:9.1f}ms{s['ttft']['mean']:11.1f}ms"
              f"{sim['throughput_tok_s']:9.1f}")
    tl = cp["ttft_longest_prompt"]
    print(f"    -> p99 inter-token latency {u['itl']['p99']/c['itl']['p99']:.1f}x better; "
          f"the {tl['prompt']}-token prompt pays "
          f"+{tl['delta_ms']:.1f}ms of TTFT for it")
    print()
    print("  3. decode batching intensity")
    for s in db["by_seq"]:
        rr = ("ridge at batch %.0f" % s["ridge_crossover_batch"]
              if s["ridge_reachable"]
              else "NEVER reaches the ridge (asymptote %.1f < %.1f)"
                   % (s["asymptote"], db["ridge"]))
        print(f"    seq {s['seq']:6d}: KV bytes equal weight bytes at batch "
              f"{s['kv_crossover_batch']:.1f}; {rr}")
    print()
    print("  4. KV allocation")
    st_, bd, dm = a["stats"], a["budget"], a["demo"]
    print(f"    {st_['n']} sequences, mean length {st_['mean_len']:.1f} "
          f"(median {st_['median_len']:.1f}), advertised context {MAX_SEQ}")
    print(f"    contiguous: {st_['contiguous_bytes']/1e9:7.2f} GB reserved, "
          f"{st_['contiguous_waste_frac']*100:.1f}% never touched")
    print(f"    paged     : {st_['paged_bytes']/1e9:7.2f} GB allocated, "
          f"{st_['paged_waste_frac']*100:.1f}% waste "
          f"(<= one {BLOCK}-token block per sequence)")
    print(f"    -> {st_['paged_vs_contiguous']}x less KV memory for the same work")
    print(f"    at a {bd['kv_budget_bytes']/1e9:.1f} GB KV budget: "
          f"{bd['concurrent_contiguous']} concurrent sequences contiguous, "
          f"{bd['concurrent_paged']} paged  ({bd['ratio']}x)")
    print(f"    demo pool of {dm['pool_blocks']} blocks: "
          f"{dm['contiguous_seqs']} sequences contiguous vs "
          f"{dm['paged_seqs']} paged ({dm['fit_ratio']}x), "
          f"{dm['paged_free_blocks']} blocks still free")
    print()
    print("  5. copy-on-write prefix sharing")
    print(f"    {ps['system_prompt_tokens']}-token system prompt -> "
          f"{ps['shared_full_blocks']} shared blocks + "
          f"1 copy-on-write boundary block "
          f"({ps['boundary_block_shared_tokens']} shared tokens in it)")
    for r in ps["rows"]:
        if r["n"] in (1, 8, 64, 128):
            print(f"    fan-out {r['n']:4d}: {r['blocks_without']:6d} blocks "
                  f"({r['bytes_without']/1e9:6.2f} GB) -> "
                  f"{r['blocks_with']:6d} ({r['bytes_with']/1e9:5.2f} GB)   "
                  f"{r['saving_frac']*100:5.1f}% saved, {r['ratio']}x")
    print()
    print(f"  wrote {os.path.join(outdir, 'serving.js')}")
    print("=" * 72)

    if fails:
        for f_ in fails:
            print("  FAIL: " + f_)
        raise SystemExit("serving.py: %d consistency check(s) failed" % len(fails))
    print("  all consistency checks PASS")


if __name__ == "__main__":
    main()
