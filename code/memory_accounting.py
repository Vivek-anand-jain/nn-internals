#!/usr/bin/env python3
"""
memory_accounting.py — what does it cost, per GPU, to train this model?

Pure Python standard library. No numpy, no torch, no network, no config file.
Status: EXECUTED. Every example in code/README.md was run to produce its
output verbatim.

WHY THIS FILE EXISTS
--------------------
The toy 2 -> 3 -> 1 network in ground_truth.py has 13 parameters and its
entire training state fits in 728 bytes. Every structural fact about it
survives the jump to 70 billion parameters unchanged -- the weights are still
one buffer, the gradients are still a second buffer exactly as large, Adam
still keeps two more, and the activations still have to stay resident from
the forward pass until backward consumes them.

The only thing that changes is that the arithmetic starts to matter. This is
that arithmetic, made executable, so that "will a 70B model fit on 8 H100s"
becomes a command rather than an opinion.

Site page 11 (scaling) renders the same numbers. This file is the reference
implementation of the formulas below; if the page and this script ever
disagree, one of them has drifted and the formulas documented here are the
tiebreaker.

================================================================================
THE FOUR TENSOR CLASSES
================================================================================
Everything resident on a training GPU is one of four things. This is the whole
taxonomy the project uses, and the same four hues the site colours with.

  WEIGHTS      the parameters themselves
  GRADIENTS    one number per parameter, produced by backward()
  OPTIMIZER    momentum / variance / master-weight buffers, one or more full
               copies of the parameters
  ACTIVATIONS  forward-pass intermediates kept alive because backward needs
               them; the only class that scales with batch and sequence length

================================================================================
FORMULAS
================================================================================
Notation
    P    parameters in the whole model
    L    transformer layers            h    d_model (hidden size)
    b    micro-batch per GPU           s    sequence length
    N    total GPUs
    tp   tensor-parallel degree        pp   pipeline-parallel degree
    dp   data-parallel degree = N / (tp * pp)
    Bp   bytes per element of the parameter dtype   (--dtype)
    Bg   bytes per element of the gradient dtype    (--grad-dtype, default Bp)
    Ba   bytes per element of activations           (= Bp)

All results are in GiB = 2^30 bytes, never GB = 10^9. Vendors quote GB;
allocators report GiB. Mixing them is an 7% error, which is enough to turn a
"fits" into an OOM.

--------------------------------------------------------------------------------
1. Parameter count, if you did not supply --params
--------------------------------------------------------------------------------
    P = 12 * L * h^2

    A decoder-only transformer layer holds 4h^2 in attention (Q, K, V, O) and
    8h^2 in the MLP (up-projection h x 4h, down-projection 4h x h). Embeddings
    and the LM head are excluded -- they are a few percent at these scales and
    they are frequently sharded differently. A SwiGLU MLP uses three matrices
    of width 8h/3 and lands on the same 8h^2, so the formula holds for both.

--------------------------------------------------------------------------------
2. Model state, before any sharding
--------------------------------------------------------------------------------
    weights   = P * Bp
    gradients = P * Bg
    optimizer = P * (slots * slot_bytes + 4 * master_weights)

    slots and slot_bytes by optimizer:

        sgd         0 slots            keeps nothing at all
        momentum    1 slot,  4 B       one velocity buffer, fp32
        adam        2 slots, 4 B       exp_avg (m) and exp_avg_sq (v), fp32
        adam8bit    2 slots, 1 B       block-wise quantised m and v

    Optimizer slots are fp32 even when the weights are bf16. That is not an
    oversight: m and v accumulate tiny quantities across thousands of steps
    and bf16's 8 mantissa bits lose them to rounding. The same reasoning gives
    the fp32 MASTER WEIGHTS copy -- a bf16 weight has ~3 decimal digits, so a
    small update applied directly to it is silently discarded. You keep an
    fp32 copy, apply updates there, and cast down for the forward pass.

    --master-weights auto  turns the fp32 copy on exactly when it is needed:
    parameter dtype narrower than 4 bytes AND optimizer != sgd.

    The famous "16 bytes per parameter" for Adam is this sum with bf16 weights:
        2 (weights) + 2 (grads) + 4 (master) + 4 (m) + 4 (v) = 16
    For 70B that is 1120 GB = 1043 GiB of model state before a single
    activation exists.

--------------------------------------------------------------------------------
3. Model-parallel sharding: tp and pp
--------------------------------------------------------------------------------
    P_local = P / (tp * pp)

    Tensor parallelism splits each weight matrix across tp ranks; pipeline
    parallelism gives each rank L/pp whole layers. Both shard weights,
    gradients and optimizer state alike, and both do it before any
    data-parallel strategy is applied.

--------------------------------------------------------------------------------
4. Data-parallel strategy: what gets divided by dp
--------------------------------------------------------------------------------
                    weights   gradients   optimizer
        none          1           1           1        single GPU
        ddp           1           1           1        full replica per rank
        zero1         1           1          dp        shard optimizer state
        zero2         1          dp          dp        + shard gradients
        zero3         dp         dp          dp        + shard weights
        fsdp          dp         dp          dp        FULL_SHARD, same as zero3

    ZeRO is a sequence of decisions about what a rank is allowed to forget
    between the moments it needs it. Stage 1 forgets optimizer state it is not
    responsible for. Stage 2 also forgets gradients outside its shard, keeping
    them only long enough to reduce-scatter. Stage 3 also forgets weights,
    all-gathering each layer just before using it and dropping it after.

    Stage 3 and FSDP therefore trade memory for communication, and they need
    a transient all-gather buffer for the largest unit being gathered. That
    buffer is real and this calculator reports it as a separate line rather
    than folding it into a class -- see TRANSIENT below.

--------------------------------------------------------------------------------
5. Activations
--------------------------------------------------------------------------------
    Without checkpointing, per GPU:

        act = L_eff * s * b * h * 34 * (Ba / 2) / tp

    The coefficient 34 is the per-token, per-layer tally of stored
    intermediates in units of h elements at 2 bytes each, from Korthikanti et
    al. 2022 ("Reducing Activation Recomputation in Large Transformer
    Models"). That paper's full expression is

        s*b*h*L * (34 + 5*a*s/h)

    where the second term is the materialised s x s attention score matrix per
    head. This calculator DROPS that term because FlashAttention-style fused
    attention never materialises it, which has been the default since 2022.
    If you are running unfused attention, the true number is higher and grows
    with s^2 -- that quadratic is the reason fused attention exists.

    The (Ba / 2) factor rescales the tally from its assumed 2-byte dtype to
    whatever --dtype you asked for.

    With --checkpointing (full recomputation), only each layer's INPUT is
    kept and everything else is recomputed during backward:

        act = L_eff * s * b * h * Ba / tp

    That is a 34x reduction in activation memory for roughly a 30% increase in
    step time, because the forward pass runs twice. It is the single largest
    lever in this whole calculator.

    Activations are NOT sharded by any data-parallel strategy. Each rank owns
    its own micro-batch. This is the fact that makes ZeRO-3 stop helping: once
    model state is fully sharded, activations are all that is left, and no
    amount of dp reduces them.

    L_eff and pipeline parallelism:

        L_eff = ceil(L / pp) * min(pp, 1 if pp == 1 else pp)  =  ceil(L/pp)*pp

    which is >= L, i.e. pipeline parallelism does not reduce peak activation
    memory. Under a 1F1B schedule the first stage must keep pp micro-batches
    in flight to keep the pipe full, and it holds L/pp layers of each. The
    L/pp saving and the pp in-flight cost cancel exactly. This surprises
    people, so the report prints it explicitly.

--------------------------------------------------------------------------------
6. TRANSIENT (reported, not counted in the four classes)
--------------------------------------------------------------------------------
    zero3 / fsdp all-gather buffer, approximated as one layer's worth of
    unsharded parameters:

        transient = (P / L) * Bp / tp        (zero3, fsdp)
        transient = 0                        (everything else)

    Also reported and not counted: the CUDA context and cuBLAS/cuDNN
    workspaces, typically 1-2 GiB per process, and allocator fragmentation.
    The verdict below therefore uses a headroom band rather than a hard line.

================================================================================
USAGE
================================================================================
    python3 code/memory_accounting.py --params 70B --dtype bf16 \\
        --optimizer adam --gpus 8 --strategy ddp \\
        --layers 80 --d-model 8192 --batch 1 --seq 4096

    python3 code/memory_accounting.py --params 70B --dtype bf16 \\
        --optimizer adam --gpus 64 --strategy zero3 --table

Sizes accept suffixes: 70B, 7b, 1.5B, 175000000000, 6.7e9.
"""

import argparse
import json
import math
import os
import sys

GIB = 1024 ** 3

# ----------------------------------------------------------------------------
# DTYPE WIDTHS
# ----------------------------------------------------------------------------
# The only thing a dtype contributes to this calculator is its width. Site
# page 01 covers what the bits inside actually mean; here all that matters is
# how many of them there are. tf32 is 4 bytes in memory despite carrying 19
# bits of information -- it is a compute format, not a storage format, and
# getting that wrong understates a model by 2x.
DTYPE_BYTES = {
    "fp32": 4, "float32": 4, "tf32": 4,
    "fp16": 2, "float16": 2, "bf16": 2, "bfloat16": 2,
    "fp8": 1, "int8": 1,
}

# (slots per parameter, bytes per slot). See formula section 2.
OPTIMIZERS = {
    "sgd":      (0, 4),
    "momentum": (1, 4),
    "adam":     (2, 4),
    "adam8bit": (2, 1),
}

# Which classes a data-parallel strategy divides by dp.
STRATEGIES = {
    #            weights  grads  optimizer
    "none":     (False,  False,  False),
    "ddp":      (False,  False,  False),
    "zero1":    (False,  False,  True),
    "zero2":    (False,  True,   True),
    "zero3":    (True,   True,   True),
    "fsdp":     (True,   True,   True),
}

# Per-layer activation tally, in units of (h elements at 2 bytes), from
# Korthikanti et al. 2022 with the attention-score term dropped. See section 5.
ACT_COEFF = 34
ACT_COEFF_BASE_BYTES = 2


# ============================================================================
# PRESETS — read from the trace, never typed here
# ============================================================================
# T.reference_configs holds the published model architectures and GPU HBM
# capacities that site page 11 offers in its dropdowns. Reading them from the
# same file the page reads is the only way to guarantee the CLI and the page
# describe the same machine. AGENTS.md rule 1 applied to Python: if a number
# is in the trace, do not type it again here.
#
# Everything below degrades to "no presets available" if the trace is missing
# or predates the reference_configs key, because the calculator's own
# arithmetic does not depend on it.

def load_reference_configs():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.dirname(here), "assets", "data", "trace.json")
    try:
        with open(path) as f:
            return json.load(f).get("reference_configs")
    except (OSError, ValueError):
        return None


def find_preset(entries, query, kind):
    """Case-insensitive substring match, so '70b' finds 'Llama 3 70B'."""
    q = query.strip().lower()
    hits = [e for e in entries if q in e["name"].lower()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise SystemExit(
            f"no {kind} preset matching {query!r}. Known: "
            + ", ".join(e["name"] for e in entries))
    raise SystemExit(
        f"{query!r} is ambiguous among {kind} presets: "
        + ", ".join(h["name"] for h in hits))


def print_presets(rc):
    if not rc:
        print("  no presets: assets/data/trace.json is missing or has no "
              "reference_configs key (run python3 code/ground_truth.py)")
        return
    print(f"  source: T.reference_configs — "
          f"{rc.get('_source', 'quoted, not derived')}")
    print()
    print(f"  {'--model':<16}{'params':>10}{'layers':>8}{'d_model':>9}"
          f"{'heads':>7}{'seq':>7}")
    print("  " + "-" * 57)
    for m in rc.get("models", []):
        print(f"  {m['name']:<16}{fmt_count(m['params']):>10}"
              f"{m.get('n_layers', '-'):>8}{m.get('d_model', '-'):>9}"
              f"{m.get('n_heads', '-'):>7}{m.get('seq', '-'):>7}")
    print()
    print(f"  {'--gpu':<16}{'HBM':>12}")
    print("  " + "-" * 28)
    for g in rc.get("gpus", []):
        print(f"  {g['name']:<16}{gib(g['hbm_bytes']):>9.0f} GiB")
    print()
    print(f"  defaults on page 11: model {rc.get('default_model')!r}, "
          f"gpu {rc.get('default_gpu')!r}")


def apply_presets(args, rc):
    """Fill in whatever the user did not specify. Explicit flags always win —
    a preset is a starting point, not an override."""
    if args.model:
        if not rc:
            raise SystemExit("--model needs assets/data/trace.json; run "
                             "python3 code/ground_truth.py first")
        m = find_preset(rc.get("models", []), args.model, "model")
        if args.params is None:
            args.params = int(m["params"])
            args.params_source = f"T.reference_configs[{m['name']!r}].params"
        if not args.layers:
            args.layers = m.get("n_layers", 0)
        if not args.d_model:
            args.d_model = m.get("d_model", 0)
        if not args.seq:
            args.seq = m.get("seq", 0)
        if not args.batch:
            args.batch = 1
        args.preset_model = m["name"]
    if args.gpu:
        if not rc:
            raise SystemExit("--gpu needs assets/data/trace.json")
        g = find_preset(rc.get("gpus", []), args.gpu, "gpu")
        if args.gpu_mem is None:
            args.gpu_mem = gib(g["hbm_bytes"])
        args.preset_gpu = g["name"]
    if args.gpu_mem is None:
        args.gpu_mem = 80.0          # H100 80GB, the page-08 default
    return args


# ============================================================================
# PARSING AND FORMATTING
# ============================================================================

def parse_count(text):
    """'70B' -> 70e9. Also accepts K/M/B/T, plain integers and 6.7e9."""
    t = str(text).strip().replace(",", "").replace("_", "")
    mult = 1
    if t and t[-1].upper() in "KMBGT":
        mult = {"K": 1e3, "M": 1e6, "B": 1e9, "G": 1e9, "T": 1e12}[t[-1].upper()]
        t = t[:-1]
    try:
        return int(round(float(t) * mult))
    except ValueError:
        raise argparse.ArgumentTypeError(f"cannot parse a count from {text!r}")


def gib(n_bytes):
    return n_bytes / GIB


def fmt_gib(n_bytes):
    g = gib(n_bytes)
    if n_bytes == 0:
        return "0"
    if g < 0.01:
        return f"{n_bytes / 1024**2:.2f} MiB"
    return f"{g:,.2f} GiB"


def fmt_count(n):
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.4g}{unit}"
    return str(n)


def bar(fraction, width=40):
    """A proportional bar, ASCII only so it survives any terminal."""
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return "#" * filled + "." * (width - filled)


# ============================================================================
# THE CALCULATION
# ============================================================================

def compute(args):
    """Returns a dict with the per-GPU byte breakdown and everything needed to
    explain it. All arithmetic lives here; the printers below only format."""

    Bp = DTYPE_BYTES[args.dtype]
    Bg = DTYPE_BYTES[args.grad_dtype] if args.grad_dtype else Bp
    Ba = Bp

    # ---- parameter count -------------------------------------------------
    # main() has already filled args.params in, either from the flag or from
    # 12*L*h^2 (formula section 1), and recorded which.
    P = args.params
    P_source = getattr(args, "params_source", "--params")
    if P is None:
        raise SystemExit("need --params, or both --layers and --d-model")

    # ---- master weights: auto / on / off ---------------------------------
    if args.master_weights == "auto":
        master = (Bp < 4) and args.optimizer != "sgd"
        master_why = ("auto: on, because parameters are narrower than fp32 and "
                      "the optimizer applies updates"
                      if master else
                      "auto: off, parameters are already fp32 or SGD keeps no state")
    else:
        master = args.master_weights == "on"
        master_why = f"forced {args.master_weights}"

    slots, slot_bytes = OPTIMIZERS[args.optimizer]
    opt_bytes_per_param = slots * slot_bytes + (4 if master else 0)

    # ---- parallel degrees ------------------------------------------------
    tp, pp = max(1, args.tp), max(1, args.pp)
    gpus = max(1, args.gpus)
    mp = tp * pp
    if args.strategy == "none":
        # A single-GPU budget. Ignore --gpus rather than silently dividing.
        gpus, dp = mp, 1
    else:
        if gpus % mp != 0:
            raise SystemExit(
                f"--gpus {gpus} is not divisible by tp*pp = {tp}*{pp} = {mp}")
        dp = gpus // mp
    if dp < 1:
        raise SystemExit("tp * pp exceeds --gpus")

    # ---- model state, section 2 -> 3 -> 4 --------------------------------
    P_local = P / mp                       # tp and pp shard first
    shard_w, shard_g, shard_o = STRATEGIES[args.strategy]

    weights = P_local * Bp / (dp if shard_w else 1)
    grads = P_local * Bg / (dp if shard_g else 1)
    optimizer = P_local * opt_bytes_per_param / (dp if shard_o else 1)

    # ---- activations, section 5 ------------------------------------------
    act = 0.0
    act_note = None
    L_eff = None
    if args.layers and args.d_model and args.batch and args.seq:
        L, h, b, s = args.layers, args.d_model, args.batch, args.seq
        layers_per_stage = math.ceil(L / pp)
        # 1F1B: pp micro-batches in flight at the first stage. The L/pp saving
        # and the pp in-flight cost cancel. See section 5.
        L_eff = layers_per_stage * pp
        if args.checkpointing:
            # only each layer's input survives the forward pass
            act = L_eff * s * b * h * Ba / tp
        else:
            act = (L_eff * s * b * h * ACT_COEFF
                   * (Ba / ACT_COEFF_BASE_BYTES) / tp)
        if pp > 1:
            act_note = (f"pp={pp}: {layers_per_stage} layers/stage x {pp} "
                        f"micro-batches in flight = {L_eff} layer-equivalents "
                        f"(>= L={L}; pipelining does not cut peak activations)")
    else:
        act_note = ("activations not modelled: supply --layers --d-model "
                    "--batch --seq")

    # ---- transient, section 6 --------------------------------------------
    transient = 0.0
    if args.strategy in ("zero3", "fsdp") and args.layers:
        transient = (P / args.layers) * Bp / tp

    total = weights + grads + optimizer + act

    return {
        "P": P, "P_source": P_source, "P_local": P_local,
        "Bp": Bp, "Bg": Bg, "Ba": Ba,
        "master": master, "master_why": master_why,
        "opt_bytes_per_param": opt_bytes_per_param,
        "slots": slots, "slot_bytes": slot_bytes,
        "tp": tp, "pp": pp, "dp": dp, "gpus": gpus, "mp": mp,
        "weights": weights, "grads": grads,
        "optimizer": optimizer, "act": act,
        "act_note": act_note, "L_eff": L_eff,
        "transient": transient, "total": total,
        "shard": (shard_w, shard_g, shard_o),
    }


# ============================================================================
# REPORT
# ============================================================================

def print_breakdown(args, r):
    cap = args.gpu_mem * GIB

    print("=" * 78)
    print("  PER-GPU TRAINING MEMORY")
    print("=" * 78)

    print("  MODEL")
    if getattr(args, "preset_model", None):
        print(f"    preset              {args.preset_model}"
              f"   (from T.reference_configs)")
    print(f"    parameters          {fmt_count(r['P'])}   "
          f"({r['P']:,})   [{r['P_source']}]")
    if args.layers and args.d_model:
        print(f"    layers / d_model    {args.layers} / {args.d_model}")
    if args.batch and args.seq:
        print(f"    micro-batch / seq   {args.batch} / {args.seq}"
              f"   ({args.batch * args.seq:,} tokens per GPU per micro-step)")
    print(f"    param dtype         {args.dtype} ({r['Bp']} B/elem)"
          f"      grad dtype: {args.grad_dtype or args.dtype} "
          f"({r['Bg']} B/elem)")
    print(f"    optimizer           {args.optimizer}"
          f"   ({r['slots']} slot(s) x {r['slot_bytes']} B)")
    print(f"    master weights      {'yes' if r['master'] else 'no'}"
          f"   ({r['master_why']})")
    print(f"    recomputation       "
          f"{'full checkpointing' if args.checkpointing else 'none'}")
    print()

    print("  TOPOLOGY")
    sw, sg, so = r["shard"]
    print(f"    gpus                {r['gpus']}   "
          f"= tp {r['tp']} x pp {r['pp']} x dp {r['dp']}")
    print(f"    strategy            {args.strategy}")
    print(f"    shards by dp        weights={sw}  gradients={sg}  optimizer={so}")
    print(f"    params per gpu      {fmt_count(r['P_local'])}"
          f"   (sharded by tp*pp = {r['mp']}, before any ZeRO sharding)")
    print()

    print("  BREAKDOWN BY TENSOR CLASS")
    rows = [
        ("weights", r["weights"],
         f"{fmt_count(r['P_local'])} x {r['Bp']} B"
         + (f" / dp {r['dp']}" if sw else "")),
        ("gradients", r["grads"],
         f"{fmt_count(r['P_local'])} x {r['Bg']} B"
         + (f" / dp {r['dp']}" if sg else "")),
        ("optimizer", r["optimizer"],
         f"{fmt_count(r['P_local'])} x {r['opt_bytes_per_param']} B"
         + (f" / dp {r['dp']}" if so else "")),
        ("activations", r["act"],
         (f"L_eff {r['L_eff']} x s {args.seq} x b {args.batch} "
          f"x h {args.d_model} x "
          f"{1 if args.checkpointing else ACT_COEFF} x {r['Ba']} B / tp {r['tp']}")
         if r["act"] else "not modelled"),
    ]
    total = r["total"]
    print(f"    {'class':<13}{'bytes':>14}   {'share':>7}  "
          f"{'':<42}")
    for name, val, how in rows:
        frac = (val / total) if total else 0.0
        print(f"    {name:<13}{fmt_gib(val):>14}   {frac * 100:>6.1f}%  "
              f"{bar(frac, 28)}")
        print(f"    {'':<13}{'':>14}            {how}")
    print(f"    {'-' * 66}")
    print(f"    {'TOTAL':<13}{fmt_gib(total):>14}   "
          f"{total / r['P'] if r['P'] else 0:>6.2f} B/param overall")
    print()

    if r["transient"]:
        print(f"  TRANSIENT (not counted above)")
        print(f"    all-gather buffer   {fmt_gib(r['transient'])}"
              f"   one layer's unsharded weights, needed by "
              f"{args.strategy}")
        print()
    if r["act_note"]:
        print(f"  NOTE  {r['act_note']}")
        print()

    # ---- verdict ---------------------------------------------------------
    # A hard <= would be dishonest. A real process also pays for the CUDA
    # context, cuBLAS/cuDNN workspaces, NCCL buffers and allocator
    # fragmentation. RESERVE is a deliberately conservative flat allowance.
    RESERVE = 2.0 * GIB
    needed = total + r["transient"] + RESERVE
    print("  VERDICT")
    print(f"    gpu capacity        {args.gpu_mem:g} GiB"
          + (f"   ({args.preset_gpu})" if getattr(args, "preset_gpu", None)
             else ""))
    print(f"    tensors             {fmt_gib(total)}")
    if r["transient"]:
        print(f"    + transient         {fmt_gib(r['transient'])}")
    print(f"    + runtime reserve   {fmt_gib(RESERVE)}"
          f"   (CUDA context, workspaces, fragmentation)")
    print(f"    = required          {fmt_gib(needed)}")
    print(f"    utilisation         {needed / cap * 100:.1f}%  "
          f"[{bar(needed / cap, 40)}]")
    if needed <= cap:
        print(f"    ==> FITS with {fmt_gib(cap - needed)} to spare")
    else:
        over = needed - cap
        print(f"    ==> DOES NOT FIT — over by {fmt_gib(over)}")
        suggest_fixes(args, r, over)
    print("=" * 78)


def suggest_fixes(args, r, over):
    """Name the specific lever, with the arithmetic, rather than saying
    'reduce memory usage'."""
    tips = []
    if not args.checkpointing and r["act"]:
        saved = r["act"] - r["act"] / ACT_COEFF
        tips.append(f"--checkpointing frees {fmt_gib(saved)} of activations "
                    f"(34x), costing ~30% step time")
    if args.strategy in ("none", "ddp") and r["dp"] > 1:
        tips.append("--strategy zero1 shards optimizer state across the "
                    f"{r['dp']} dp ranks")
    if args.strategy in ("ddp", "zero1", "zero2") and r["dp"] > 1:
        tips.append("--strategy zero3 shards weights and gradients too")
    if r["Bp"] == 4:
        tips.append("--dtype bf16 halves weights, gradients and activations")
    if args.optimizer == "adam":
        tips.append("--optimizer adam8bit cuts Adam's 8 B/param of moments "
                    "to 2 B/param")
    if args.tp == 1 and args.layers:
        tips.append("--tp 8 shards weights AND activations within a node")
    if tips:
        print(f"    levers, in order of size:")
        for t in tips:
            print(f"      - {t}")


def print_table(args):
    """--table: the same model under every strategy, side by side."""
    print("=" * 96)
    print(f"  STRATEGY COMPARISON — {fmt_count(args.params)} params, "
          f"{args.dtype}, {args.optimizer}, {args.gpus} GPUs "
          f"(tp {args.tp} x pp {args.pp}), {args.gpu_mem:g} GiB each"
          + (", checkpointing" if args.checkpointing else ""))
    print("=" * 96)
    print(f"  {'strategy':<10}{'weights':>14}{'grads':>14}{'optim':>14}"
          f"{'acts':>14}{'TOTAL':>16}{'':>4}verdict")
    print("  " + "-" * 94)

    cap = args.gpu_mem * GIB
    RESERVE = 2.0 * GIB
    for strat in ("none", "ddp", "zero1", "zero2", "zero3", "fsdp"):
        sub = argparse.Namespace(**vars(args))
        sub.strategy = strat
        try:
            r = compute(sub)
        except SystemExit as e:
            print(f"  {strat:<10}  n/a  ({e})")
            continue
        needed = r["total"] + r["transient"] + RESERVE
        verdict = "FITS" if needed <= cap else "NO FIT"
        print(f"  {strat:<10}{fmt_gib(r['weights']):>14}"
              f"{fmt_gib(r['grads']):>14}{fmt_gib(r['optimizer']):>14}"
              f"{fmt_gib(r['act']):>14}{fmt_gib(r['total']):>16}"
              f"{'':>4}{verdict}")
    print("  " + "-" * 94)
    print("  'none' is the single-GPU budget: tp x pp GPUs, no data parallel.")
    print("  Every other row uses dp = gpus / (tp x pp) and shards per the")
    print("  table in this file's docstring, section 4.")
    print("  TOTAL excludes the zero3/fsdp all-gather buffer and the 2 GiB")
    print("  runtime reserve; the verdict column includes both.")
    print()
    print("  Activations are identical in every row. No data-parallel strategy")
    print("  shards them — each rank owns its own micro-batch. Once ZeRO-3 has")
    print("  driven model state to near zero, activations are the entire")
    print("  budget, and the only remaining levers are --checkpointing, --tp,")
    print("  and a smaller --batch.")
    print("=" * 96)


# ============================================================================
# CLI
# ============================================================================

def build_parser():
    p = argparse.ArgumentParser(
        prog="memory_accounting.py",
        description="Per-GPU training memory for an arbitrary model, broken "
                    "down by the four tensor classes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Formulas are documented in this file's module docstring. "
               "Site page 11 must agree with them.")

    g = p.add_argument_group("model")
    g.add_argument("--params", type=parse_count, default=None,
                   help="parameter count; accepts 70B, 6.7e9, 175000000000. "
                        "If omitted, derived as 12*L*h^2 from --layers/--d-model.")
    g.add_argument("--layers", type=int, default=0, help="transformer layers L")
    g.add_argument("--d-model", type=int, default=0, help="hidden size h")
    g.add_argument("--batch", type=int, default=0,
                   help="MICRO-batch per GPU (not global batch)")
    g.add_argument("--seq", type=int, default=0, help="sequence length s")

    g = p.add_argument_group("precision and optimizer")
    g.add_argument("--dtype", default="bf16", choices=sorted(DTYPE_BYTES),
                   help="parameter and activation dtype (default bf16)")
    g.add_argument("--grad-dtype", default=None, choices=sorted(DTYPE_BYTES),
                   help="gradient dtype (default: same as --dtype)")
    g.add_argument("--optimizer", default="adam", choices=sorted(OPTIMIZERS),
                   help="default adam")
    g.add_argument("--master-weights", nargs="?", const="on", default="auto",
                   choices=("auto", "on", "off"),
                   help="fp32 master copy of the weights. auto (default) "
                        "enables it when dtype < 4 B and optimizer != sgd.")
    g.add_argument("--checkpointing", action="store_true",
                   help="full activation recomputation: keep only each "
                        "layer's input")

    g = p.add_argument_group("parallelism")
    g.add_argument("--gpus", type=int, default=1, help="total GPUs")
    g.add_argument("--strategy", default="ddp", choices=sorted(STRATEGIES),
                   help="data-parallel strategy (default ddp)")
    g.add_argument("--tp", type=int, default=1, help="tensor-parallel degree")
    g.add_argument("--pp", type=int, default=1, help="pipeline-parallel degree")

    g = p.add_argument_group("hardware and output")
    g.add_argument("--gpu-mem", type=float, default=None,
                   help="per-GPU capacity in GiB (default 80 = H100 80GB)")
    g.add_argument("--table", action="store_true",
                   help="compare all strategies for this model")

    g = p.add_argument_group("presets (read from assets/data/trace.json)")
    g.add_argument("--model", default=None,
                   help="preset model name or substring, e.g. '70b'. Fills "
                        "--params/--layers/--d-model/--seq if you did not.")
    g.add_argument("--gpu", default=None,
                   help="preset GPU name or substring, e.g. 'h100'. Fills "
                        "--gpu-mem if you did not.")
    g.add_argument("--list-presets", action="store_true",
                   help="show the model and GPU presets and exit")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    rc = load_reference_configs()
    if args.list_presets:
        print_presets(rc)
        return 0
    args = apply_presets(args, rc)

    if args.params is None and not (args.layers and args.d_model):
        build_parser().error("need --params, or both --layers and --d-model")
    if args.params is None:
        # Formula section 1. Reported as such, so nobody mistakes an estimate
        # for a measurement.
        args.params = 12 * args.layers * args.d_model ** 2
        args.params_source = (f"12 x {args.layers} x {args.d_model}^2, "
                              f"embeddings excluded")

    if args.table:
        print_table(args)
    else:
        print_breakdown(args, compute(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
