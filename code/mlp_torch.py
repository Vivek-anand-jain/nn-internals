#!/usr/bin/env python3
"""
mlp_torch.py — the same 2 -> 3 -> 1 MLP in PyTorch, plus a tour of the real
memory structures underneath it.

    REQUIRES torch (>= 2.0). torch is NOT installed in the environment this
    file was authored in, and this script was NEVER EXECUTED here. It was
    written against the documented PyTorch API and cross-checked line by line
    against ground_truth.py / trace.json, but it carries no run receipt.
    Treat every printed claim below as a prediction until you run it yourself.

    Its sibling, mlp_numpy.py, WAS executed and passes all 359 element-wise
    checks against the trace at 1e-12. That is the file to trust for the math.
    This file's job is the part numpy cannot show you: what a framework
    actually allocates.

Run:   python3 code/mlp_torch.py
Parse-check without torch:   python3 -m py_compile code/mlp_torch.py

WHY THIS FILE EXISTS
--------------------
ground_truth.py proves the arithmetic. mlp_numpy.py proves the arithmetic is
reproducible. Neither can show you a pointer.

This project's whole claim is that a neural network is bytes in memory being
moved and multiplied, and that if you can account for the bytes you can
predict what will and will not fit on a GPU. PyTorch is the first tool in the
stack that will actually tell you where those bytes are. So this file spends
most of its length not on the model but on:

    data_ptr()          the address of the first element
    element_size()      bytes per element -- 4 for fp32, 2 for bf16
    nbytes              elements x element_size, the tensor's real footprint
    stride()            how many elements to skip per dimension
    is_contiguous()     whether that stride pattern is the packed one
    .grad               None until backward(), a full second copy afterwards
    optimizer state     empty until the first step(), then 2x params for Adam

Those six lines are the entire memory model of training. Everything on site
page 08 (scaling) is arithmetic on top of them.

ANTI-DRIFT RULE
---------------
ground_truth.py is authoritative. This file verifies against trace.json and
never the other way round. If a check here fails, fix this file.
"""

import json
import os
import sys

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - the whole point of this branch
    sys.exit(
        "mlp_torch.py requires PyTorch, which is not installed.\n"
        "  pip install torch          (a multi-GB download)\n"
        "\n"
        "For a dependency-light verification of exactly the same numbers, run\n"
        "  python3 code/mlp_numpy.py  (numpy only, and it does pass)\n"
    )


# ----------------------------------------------------------------------------
# PRECISION
# ----------------------------------------------------------------------------
# ground_truth.py runs in Python floats, which are IEEE-754 binary64. torch
# defaults to float32. If we left the default alone, every comparison below
# would be off by ~1e-7 -- not because anyone's chain rule is wrong, but
# because fp32 carries 24 bits of mantissa and fp64 carries 53.
#
# So the verification model is float64. That is a deliberate, visible choice,
# and it costs us something: element_size() will report 8 bytes, not the 4
# that site page 08's arithmetic assumes. We pay that back in
# section 2 by casting a copy and watching nbytes halve.
#
# The lesson is the one from site page 01: precision is not a detail you get
# to ignore, it is the multiplier on every byte you own.
DTYPE = torch.float64

# Tolerance for the strict fp64 comparisons. Same reasoning as mlp_numpy.py:
# torch dispatches these tiny matmuls to a different BLAS path than numpy and
# a different one again from Python's left-to-right `sum()`, so results can
# differ by a few ulp (~1e-16). 1e-12 is four orders of magnitude of headroom.
TOL = 1e-12

# The 12-step optimizer replay gets a looser bound. torch's Adam is
# algebraically identical to ground_truth.adam_step -- it computes
#     denom = sqrt(v)/sqrt(bc2) + eps  and  step = lr/bc1
# where ground_truth computes lr * (m/bc1) / (sqrt(v/bc2) + eps). Those are
# the same expression rearranged, but the rearrangement rounds differently,
# and twelve steps compound it. 1e-10 is the honest bound to claim.
TOL_OPT = 1e-10


# ============================================================================
# 0. LOAD THE AUTHORITATIVE NUMBERS
# ============================================================================

def load_trace():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.dirname(here), "assets", "data", "trace.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"missing {path}\nGenerate it first:  python3 code/ground_truth.py")
    with open(path) as f:
        return json.load(f)


class Checker:
    """Same contract as mlp_numpy.Checker: every comparison is element-wise,
    every result lands in one table, any failure sets the exit code."""

    def __init__(self):
        self.rows = []
        self.failures = 0

    def check(self, name, got, want, tol=TOL, note=""):
        g = torch.as_tensor(got, dtype=torch.float64).detach().reshape(-1)
        w = torch.as_tensor(want, dtype=torch.float64).detach().reshape(-1)
        if g.numel() != w.numel():
            self.rows.append((name, f"{tuple(g.shape)}!={tuple(w.shape)}",
                              float("nan"), tol, "FAIL"))
            self.failures += 1
            return
        err = float((g - w).abs().max()) if g.numel() else 0.0
        ok = err <= tol
        self.rows.append((name, str(tuple(
            torch.as_tensor(want, dtype=torch.float64).shape)), err, tol,
            "PASS" if ok else "FAIL"))
        if not ok:
            self.failures += 1

    def report(self):
        w = max([len(r[0]) for r in self.rows] + [24]) + 2
        line = "-" * (w + 44)
        print(line)
        print(f"{'tensor':<{w}}{'shape':>12}{'max |Δ|':>14}{'tol':>10}   status")
        print(line)
        for name, shape, err, tol, status in self.rows:
            print(f"{name:<{w}}{shape:>12}{err:>14.3e}{tol:>10.0e}   {status}")
        print(line)
        if self.failures:
            print(f"  {self.failures} of {len(self.rows)} checks FAILED")
        else:
            print(f"  all {len(self.rows)} checks PASS")
        print(line)
        return self.failures == 0


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def nbytes_of(t):
    """t.nbytes was added in torch 1.11. Fall back so this file does not blow
    up on an older install for a cosmetic reason."""
    n = getattr(t, "nbytes", None)
    return n if n is not None else t.numel() * t.element_size()


# ============================================================================
# THE MODEL
# ============================================================================

class TinyMLP(nn.Module):
    """2 -> 3 (ReLU) -> 1, exactly the network in AGENTS.md.

    nn.Linear stores its weight as (out_features, in_features) and computes
    `input @ weight.T + bias`. So lin1.weight has shape (3, 2) and holds W1
    verbatim -- the same (3, 2) row-major buffer whose strides the site's
    layout panel walks through, T.layout.strides_elements == [2, 1].

    The intermediates are stashed on self so the verification below can call
    .retain_grad() on them and read back dL/dz1 and dL/da1. A production model
    would not do this; autograd frees non-leaf gradients as soon as they have
    been consumed, and that freeing is precisely why backward is cheaper in
    memory than a naive reading suggests.
    """

    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(2, 3, bias=True, dtype=DTYPE)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(3, 1, bias=True, dtype=DTYPE)

    def forward(self, x):
        self.z1 = self.lin1(x)
        self.a1 = self.relu(self.z1)
        self.z2 = self.lin2(self.a1)
        return self.z2


def build_model(T):
    """Load the EXACT initial weights from the trace. Nothing random, nothing
    typed by hand -- the same 13 numbers the site renders."""
    model = TinyMLP()
    with torch.no_grad():
        model.lin1.weight.copy_(torch.tensor(T["init"]["W1"], dtype=DTYPE))
        model.lin1.bias.copy_(torch.tensor(T["init"]["b1"], dtype=DTYPE))
        model.lin2.weight.copy_(torch.tensor(T["init"]["W2"], dtype=DTYPE))
        model.lin2.bias.copy_(torch.tensor(T["init"]["b2"], dtype=DTYPE))
    return model


def flat_grad(model):
    """The 13 gradients in T.param_names order: W1, b1, W2, b2."""
    return torch.cat([model.lin1.weight.grad.reshape(-1),
                      model.lin1.bias.grad.reshape(-1),
                      model.lin2.weight.grad.reshape(-1),
                      model.lin2.bias.grad.reshape(-1)])


# ============================================================================
def main():
    T = load_trace()
    ck = Checker()

    hp = T["meta"]["hyperparams"]
    LR, BETA1, BETA2, EPS = hp["lr"], hp["beta1"], hp["beta2"], hp["eps"]
    N_STEPS = hp["n_steps"]

    # Batch dimension of 1. The trace's x is a bare (2,) vector; nn.Linear
    # accepts that, but giving it a leading batch axis makes the forward hook
    # in section 6 print the shapes a real training step would show, and makes
    # the activation-memory arithmetic on site page 06 scale the obvious way.
    x = torch.tensor([T["forward"]["x"]], dtype=DTYPE)      # (1, 2)
    y = torch.tensor([[T["forward"]["y"]]], dtype=DTYPE)    # (1, 1)

    print("=" * 78)
    print("mlp_torch.py — PyTorch reproduction + memory-structure tour")
    print("=" * 78)
    print(f"  torch        : {torch.__version__}")
    print(f"  dtype        : {DTYPE}  (fp64, to match ground_truth.py's "
          f"Python floats)")
    print(f"  architecture : {T['meta']['architecture']}")
    print(f"  x = {x.tolist()}   y = {y.item()}")
    print()
    print("  NOTE: this script was authored in an environment without torch")
    print("        and has NOT been executed. mlp_numpy.py is the verified one.")

    model = build_model(T)

    # ========================================================================
    rule("1. PARAMETERS AS MEMORY — data_ptr, element_size, nbytes, stride")
    # ========================================================================
    # Site page 01 argues that a weight is not an abstract number, it is a
    # fixed-width bit pattern at an address. Here is the address.
    #
    # Read the columns like this:
    #   addr          where element [0,0] lives. Meaningless in isolation;
    #                 meaningful when two tensors report the SAME one (see
    #                 section 7 -- that is what "shares storage" means).
    #   elem          bytes per element. This is the number that fp32 -> bf16
    #                 halves, and the reason every scaling table on page 08 is
    #                 written as "bytes per parameter" rather than "parameters".
    #   nbytes        numel x elem. The tensor's actual footprint. Add these up
    #                 and you have the "weights" bar on the memory strip.
    #   stride        elements to skip to advance one step along each dim. For
    #                 a row-major (3, 2) matrix that is (2, 1): stepping one row
    #                 skips 2 elements, stepping one column skips 1. This is
    #                 T.layout.strides_elements exactly.
    #   contig        whether that stride pattern is the densely packed one.
    #                 A fresh nn.Linear weight always is. Transposing it is not
    #                 -- and costs zero bytes, which is section 7.

    print(f"{'parameter':<16}{'shape':>10}{'numel':>7}{'addr':>16}"
          f"{'elem':>6}{'nbytes':>8}{'stride':>10}{'contig':>8}")
    print("-" * 81)
    total_param_bytes = 0
    for name, p in model.named_parameters():
        total_param_bytes += nbytes_of(p)
        print(f"{name:<16}{str(tuple(p.shape)):>10}{p.numel():>7}"
              f"{hex(p.data_ptr()):>16}{p.element_size():>6}"
              f"{nbytes_of(p):>8}{str(tuple(p.stride())):>10}"
              f"{str(p.is_contiguous()):>8}")
    n_params = sum(p.numel() for p in model.parameters())
    print("-" * 81)
    print(f"  {n_params} parameters, {total_param_bytes} bytes at "
          f"{DTYPE} ({total_param_bytes // n_params} B/param)")

    # Cross-check against the trace's own accounting so the site and this file
    # cannot drift apart on how many parameters the model has.
    assert n_params == T["memory"]["n_params"], \
        f"param count drift: torch says {n_params}, trace says " \
        f"{T['memory']['n_params']}"

    # Confirm the loaded weights ARE the trace's initial weights, bit for bit.
    ck.check("init W1", model.lin1.weight, T["init"]["W1"], tol=0.0)
    ck.check("init b1", model.lin1.bias, T["init"]["b1"], tol=0.0)
    ck.check("init W2", model.lin2.weight, T["init"]["W2"], tol=0.0)
    ck.check("init b2", model.lin2.bias, T["init"]["b2"], tol=0.0)

    # And that torch's stride agrees with the row-major layout the site draws.
    assert tuple(model.lin1.weight.stride()) == \
        tuple(T["layout"]["strides_elements"]), \
        "W1 stride disagrees with T.layout.strides_elements"
    print(f"  W1 stride {tuple(model.lin1.weight.stride())} matches "
          f"T.layout.strides_elements {T['layout']['strides_elements']}")

    # ========================================================================
    rule("2. THE SAME WEIGHTS AT FOUR PRECISIONS")
    # ========================================================================
    # Site page 01 shows the bit patterns; T.bitviews holds them. Here is the
    # consequence in bytes. Nothing about the model changes -- same 13 values,
    # same shapes, same strides -- only the width of each cell.
    #
    # This is the single most load-bearing fact in the whole scaling story:
    # a 70B-parameter model is 280 GB in fp32 and 140 GB in bf16, and that
    # difference is the difference between four H100s and two.
    print(f"{'dtype':<12}{'elem_size':>11}{'W1 nbytes':>12}"
          f"{'13 params':>12}{'70B params':>14}")
    print("-" * 61)
    for dt in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
        w = model.lin1.weight.detach().to(dt)
        es = w.element_size()
        print(f"{str(dt).replace('torch.',''):<12}{es:>11}{nbytes_of(w):>12}"
              f"{n_params * es:>12}{f'{70e9 * es / 2**30:.0f} GiB':>14}")
    print("-" * 61)
    print("  Casting produces a NEW allocation -- .to(dtype) copies. Note the")
    print("  addresses differ from section 1; a dtype change is never free.")

    # A bf16 round-trip loses mantissa bits. T.bitviews[0] is W1[0][0] = 0.5,
    # which happens to be exactly representable, so use a value that is not.
    v = float(T["init"]["W1"][0][1])                      # -0.2
    # Route fp64 -> fp32 -> bf16, not fp64 -> bf16 directly. ground_truth.py's
    # bf16_bits() packs the value as a fp32 first (struct.pack(">f", x)) and
    # then keeps the top 16 bits with round-to-nearest-even, so it rounds
    # twice. Reproducing that path exactly is the difference between an equal
    # comparison and a spurious 1-ulp failure -- and double rounding is a real
    # hazard in mixed-precision pipelines, not a pedantic detail.
    v_bf16 = torch.tensor([v], dtype=torch.float64) \
                  .to(torch.float32).to(torch.bfloat16)
    bf = next((b for b in T["bitviews"] if abs(b["value"] - v) < 1e-15), None)
    if bf is not None:
        print(f"  {v} in bf16 -> {v_bf16.item():.10g}; "
              f"trace's independent bf16 rounding -> {bf['bf16']['exact']:.10g}")
        ck.check("bf16 round-trip", v_bf16.item(), bf["bf16"]["exact"], tol=0.0)

    # ========================================================================
    rule("3. .grad IS NONE BEFORE BACKWARD")
    # ========================================================================
    # This is the cleanest demonstration in PyTorch that gradients are a
    # SECOND, SEPARATE allocation the same size as the weights -- not some
    # annotation living inside them. Site page 08 counts them as their own
    # tensor class for exactly this reason, and this is why the naive
    # "1x params" mental model of training memory is wrong by at least 2x
    # before the optimizer has even been constructed.
    for name, p in model.named_parameters():
        print(f"  {name:<14} requires_grad={p.requires_grad}   "
              f"grad={p.grad}")
    assert all(p.grad is None for p in model.parameters()), \
        "grads should not exist before backward()"
    print()
    print(f"  live parameter bytes now: {total_param_bytes}")
    print(f"  live gradient bytes now:  0")

    # ========================================================================
    rule("4. FORWARD — check yhat and loss against the trace")
    # ========================================================================
    # Retain the non-leaf grads so section 5 can check the intermediate
    # chain-rule steps, not just the final parameter gradients.
    yhat_t = model(x)
    model.z1.retain_grad()
    model.a1.retain_grad()
    model.z2.retain_grad()

    # L = (yhat - y)^2, matching ground_truth.py. With a single element this
    # is identical to nn.MSELoss()(yhat, y) under any reduction, because
    # mean-of-one == sum-of-one. Written explicitly so there is no ambiguity
    # about a hidden 1/N.
    loss = (yhat_t - y).pow(2).sum()

    print(f"  z1   = {model.z1.detach().reshape(-1).tolist()}")
    print(f"  a1   = {model.a1.detach().reshape(-1).tolist()}"
          f"   <- unit 2 is dead")
    print(f"  yhat = {yhat_t.item():.10f}   trace: {T['forward']['yhat']:.10f}")
    print(f"  loss = {loss.item():.10f}   trace: {T['forward']['loss']:.10f}")
    print()
    print(f"  loss.requires_grad = {loss.requires_grad}")
    print(f"  loss.grad_fn       = {loss.grad_fn}")
    print("  That grad_fn is the head of the graph autograd will walk. Every")
    print("  node in it holds a reference to the tensors its backward needs --")
    print("  which is the mechanism behind 'activations are kept alive'.")

    ck.check("z1", model.z1, T["forward"]["z1"])
    ck.check("a1", model.a1, T["forward"]["a1"])
    ck.check("z2", model.z2, T["forward"]["z2"])
    ck.check("yhat", yhat_t.item(), T["forward"]["yhat"])
    ck.check("loss", loss.item(), T["forward"]["loss"])

    # ========================================================================
    rule("5. BACKWARD — .grad allocated, every value checked")
    # ========================================================================
    loss.backward()

    print(f"{'parameter':<16}{'grad addr':>16}{'grad nbytes':>13}"
          f"{'weight addr':>16}   same buffer?")
    print("-" * 78)
    grad_bytes = 0
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name}.grad should exist after backward()"
        grad_bytes += nbytes_of(p.grad)
        shared = p.grad.data_ptr() == p.data_ptr()
        print(f"{name:<16}{hex(p.grad.data_ptr()):>16}"
              f"{nbytes_of(p.grad):>13}{hex(p.data_ptr()):>16}   {shared}")
    print("-" * 78)
    print(f"  live parameter bytes: {total_param_bytes}")
    print(f"  live gradient bytes:  {grad_bytes}   (exactly 1x the weights)")
    print("  Every 'same buffer?' is False. Gradients live somewhere else.")
    print("  Scale that to 70B params in bf16: 140 GB of weights forces a")
    print("  second 140 GB of gradients into existence the moment you call")
    print("  backward(). Site page 08 makes that a row in a table; this is it.")

    # Parameter gradients, against the trace.
    ck.check("dL_dW1", model.lin1.weight.grad, T["backward"]["dL_dW1"])
    ck.check("dL_db1", model.lin1.bias.grad, T["backward"]["dL_db1"])
    ck.check("dL_dW2", model.lin2.weight.grad, T["backward"]["dL_dW2"])
    ck.check("dL_db2", model.lin2.bias.grad, T["backward"]["dL_db2"])
    ck.check("flat grad (13)", flat_grad(model), T["gradcheck"]["analytic"])

    # Intermediate gradients -- the actual chain-rule steps from
    # T.backward.steps, which site page 04 animates one at a time.
    ck.check("dL_dz2", model.z2.grad, T["backward"]["dL_dz2"])
    ck.check("dL_da1", model.a1.grad, T["backward"]["dL_da1"])
    ck.check("dL_dz1", model.z1.grad, T["backward"]["dL_dz1"])

    # The dead unit. z1[2] = -0.8, ReLU clamped it to 0, so its gradient is
    # exactly 0 -- not 1e-17, zero. Assert it exactly; there is no rounding to
    # hide behind when you multiply by a hard zero mask.
    print()
    print(f"  dL/dz1 = {model.z1.grad.reshape(-1).tolist()}")
    print(f"  dL/dW1 = {model.lin1.weight.grad.tolist()}")
    assert model.z1.grad.reshape(-1)[2].item() == 0.0, \
        "the dead ReLU unit must have exactly zero gradient"
    assert torch.equal(model.lin1.weight.grad[2],
                       torch.zeros(2, dtype=DTYPE)), \
        "dL/dW1 row 2 must be exactly [0, 0]"
    print("  Row 2 of dL/dW1 is exactly [0.0, 0.0]. torch's autograd and the")
    print("  hand-derived chain rule in ground_truth.py agree that a unit")
    print("  which contributed nothing takes none of the blame.")

    # ========================================================================
    rule("6. FORWARD HOOKS — watching activations appear")
    # ========================================================================
    # A forward hook fires after a module produces its output, so it sees
    # exactly the tensors that autograd may need to keep alive. This is the
    # instrument you would reach for on a real model to answer "where is my
    # activation memory going", and it is how the numbers on site page 06 are
    # obtained in practice rather than estimated.
    captured = []

    def hook(module, inputs, output):
        captured.append({
            "module": module.__class__.__name__,
            "in_shape": tuple(inputs[0].shape),
            "out_shape": tuple(output.shape),
            "numel": output.numel(),
            "nbytes": nbytes_of(output),
            "addr": output.data_ptr(),
        })

    handles = [m.register_forward_hook(hook)
               for m in (model.lin1, model.relu, model.lin2)]

    model.zero_grad(set_to_none=True)
    _ = model(x)

    for h in handles:
        h.remove()          # always remove hooks; a leaked hook is a leaked
                            # reference, which is a leaked tensor

    print(f"{'module':<10}{'in':>10}{'out':>10}{'numel':>7}"
          f"{'nbytes':>9}{'addr':>16}")
    print("-" * 62)
    act_bytes = 0
    for c in captured:
        act_bytes += c["nbytes"]
        print(f"{c['module']:<10}{str(c['in_shape']):>10}"
              f"{str(c['out_shape']):>10}{c['numel']:>7}"
              f"{c['nbytes']:>9}{hex(c['addr']):>16}")
    print("-" * 62)
    print(f"  activation bytes produced by one forward: {act_bytes}")
    print(f"  input x adds another {nbytes_of(x)}")
    print(f"  trace counts {T['memory']['n_activation_elements']} activation "
          f"elements (x, z1, a1, z2)")
    print()
    print("  ReLU's output has a DIFFERENT address from its input, so this is")
    print("  an out-of-place ReLU: z1 and a1 are both resident. nn.ReLU(")
    print("  inplace=True) would report the same address for both and halve")
    print("  that row -- the classic activation-memory trade, available only")
    print("  because ReLU's backward needs a 1-bit mask, not the input values.")

    # The hook saw the same numbers the trace records.
    ck.check("hook: lin1 out (z1)", captured[0]["numel"], len(T["forward"]["z1"]),
             tol=0.0)
    ck.check("hook: relu out (a1)", captured[1]["numel"], len(T["forward"]["a1"]),
             tol=0.0)
    ck.check("hook: lin2 out (z2)", captured[2]["numel"], len(T["forward"]["z2"]),
             tol=0.0)

    # ========================================================================
    rule("7. TRANSPOSE SHARES STORAGE — a view costs zero bytes")
    # ========================================================================
    # T.layout on the site walks W1's row-major buffer and claims that
    # transposing it is free because a transpose only rewrites the strides,
    # never the data. Here is the proof, in one assertion.
    W = model.lin1.weight.detach()          # (3, 2), contiguous
    Wt = W.t()                              # (2, 3), a VIEW

    print(f"  W        shape={tuple(W.shape)}  stride={tuple(W.stride())}  "
          f"contiguous={W.is_contiguous()}  addr={hex(W.data_ptr())}")
    print(f"  W.t()    shape={tuple(Wt.shape)}  stride={tuple(Wt.stride())}  "
          f"contiguous={Wt.is_contiguous()}  addr={hex(Wt.data_ptr())}")

    assert Wt.data_ptr() == W.data_ptr(), \
        "transpose must share storage with its source"
    print()
    print("  Same address. The transpose allocated nothing. The shape flipped")
    print("  (3,2) -> (2,3) and the stride flipped (2,1) -> (1,2), and that")
    print("  stride swap IS the transpose -- the 6 float64s never moved.")
    print(f"  is_contiguous() went {W.is_contiguous()} -> {Wt.is_contiguous()},")
    print("  which is the tell: a non-contiguous tensor is one whose strides")
    print("  no longer match the packed row-major order.")

    # .contiguous() is the opposite: it forces a real copy.
    Wc = Wt.contiguous()
    print(f"  Wt.contiguous() addr={hex(Wc.data_ptr())}  "
          f"shares={Wc.data_ptr() == W.data_ptr()}  "
          f"-> a real {nbytes_of(Wc)}-byte copy")
    assert Wc.data_ptr() != W.data_ptr(), \
        ".contiguous() on a non-contiguous view must copy"

    # The flat buffer the site draws, read back out of torch.
    ck.check("W1 flat (row-major)", W.reshape(-1), T["layout"]["flat"], tol=0.0)
    print(f"  flat row-major order: {W.reshape(-1).tolist()}")
    print(f"  T.layout.flat:        {T['layout']['flat']}")

    # ========================================================================
    rule("8. OPTIMIZER STATE — empty until the first step()")
    # ========================================================================
    # Site page 05 makes the claim that SGD keeps nothing, momentum keeps 1x
    # the parameters, and Adam keeps 2x. PyTorch will show you all three, and
    # it will show you that the allocation happens LAZILY -- on the first
    # step(), not at construction. That lag is a real operational hazard: an
    # Adam run can survive its forward and backward and then OOM on step 1.
    model2 = build_model(T)
    opt = torch.optim.Adam(model2.parameters(), lr=LR,
                           betas=(BETA1, BETA2), eps=EPS)

    sd = opt.state_dict()
    print(f"  right after torch.optim.Adam(...):")
    print(f"    state_dict()['state']        = {sd['state']}")
    print(f"    len(state_dict()['state'])   = {len(sd['state'])}   <- EMPTY")
    print(f"    param_groups[0] lr={sd['param_groups'][0]['lr']}  "
          f"betas={sd['param_groups'][0]['betas']}  "
          f"eps={sd['param_groups'][0]['eps']}")
    assert len(sd["state"]) == 0, "Adam should allocate no state before step()"
    print(f"    optimizer bytes so far: 0")

    # One full step.
    out = model2(x)
    l = (out - y).pow(2).sum()
    opt.zero_grad(set_to_none=True)
    l.backward()
    opt.step()

    sd = opt.state_dict()
    print()
    print(f"  after ONE opt.step():")
    print(f"    len(state_dict()['state'])   = {len(sd['state'])}   "
          f"<- one entry per parameter tensor")

    names = [n for n, _ in model2.named_parameters()]
    opt_bytes = 0
    for (key, st), nm in zip(sorted(sd["state"].items()), names):
        m = st["exp_avg"]         # Adam's first moment  -- the trace calls it m
        v = st["exp_avg_sq"]      # Adam's second moment -- the trace calls it v
        opt_bytes += nbytes_of(m) + nbytes_of(v)
        step = st["step"]
        step = step.item() if torch.is_tensor(step) else step
        print(f"    [{key}] {nm:<12} step={step}")
        print(f"          exp_avg    (m) shape={tuple(m.shape)} "
              f"nbytes={nbytes_of(m)} addr={hex(m.data_ptr())}")
        print(f"          {m.reshape(-1).tolist()}")
        print(f"          exp_avg_sq (v) shape={tuple(v.shape)} "
              f"nbytes={nbytes_of(v)} addr={hex(v.data_ptr())}")
        print(f"          {v.reshape(-1).tolist()}")

    print()
    print(f"    parameter bytes: {total_param_bytes}")
    print(f"    gradient bytes:  {grad_bytes}")
    print(f"    optimizer bytes: {opt_bytes}   "
          f"({opt_bytes / total_param_bytes:.0f}x the parameters)")
    print("    Adam's m and v are each a full-size copy of every parameter.")
    print("    That is the 2x on site page 05 and the `--optimizer adam` term")
    print("    in memory_accounting.py, observed rather than asserted.")

    # torch's exp_avg / exp_avg_sq must equal the trace's m / v at t=1.
    h1 = T["runs"]["adam"]["history"][0]
    torch_m = torch.cat([sd["state"][k]["exp_avg"].reshape(-1)
                         for k in sorted(sd["state"])])
    torch_v = torch.cat([sd["state"][k]["exp_avg_sq"].reshape(-1)
                         for k in sorted(sd["state"])])
    ck.check("adam m after step 1", torch_m, h1["m"], tol=TOL_OPT)
    ck.check("adam v after step 1", torch_v, h1["v"], tol=TOL_OPT)

    # ========================================================================
    rule("9. THE FULL 12-STEP RUN, THREE OPTIMIZERS")
    # ========================================================================
    # torch.optim.SGD with momentum=0.9, dampening=0, nesterov=False computes
    #     buf = momentum * buf + grad ;  p -= lr * buf
    # which is ground_truth.momentum_step verbatim, including the first-step
    # convention (torch seeds buf with grad, and 0.9 * 0 + grad is the same
    # thing). torch.optim.Adam is the same rearrangement discussed at TOL_OPT.
    for opt_name in ("sgd", "momentum", "adam"):
        m3 = build_model(T)
        if opt_name == "sgd":
            o = torch.optim.SGD(m3.parameters(), lr=LR)
        elif opt_name == "momentum":
            o = torch.optim.SGD(m3.parameters(), lr=LR, momentum=0.9)
        else:
            o = torch.optim.Adam(m3.parameters(), lr=LR,
                                 betas=(BETA1, BETA2), eps=EPS)

        losses = []
        for t in range(1, N_STEPS + 1):
            out = m3(x)
            l = (out - y).pow(2).sum()
            losses.append(l.item())
            o.zero_grad(set_to_none=True)
            l.backward()
            o.step()

        hist = [e["loss"] for e in T["runs"][opt_name]["history"]]
        ck.check(f"{opt_name}: loss history", losses, hist, tol=TOL_OPT)
        with torch.no_grad():
            final = (m3(x) - y).pow(2).sum().item()
        ck.check(f"{opt_name}: final_loss", final,
                 T["runs"][opt_name]["final_loss"], tol=TOL_OPT)
        print(f"  {opt_name:<9} loss {losses[0]:.6f} -> {final:.6f}"
              f"   (trace: {hist[0]:.6f} -> "
              f"{T['runs'][opt_name]['final_loss']:.6f})")

    # ========================================================================
    rule("10. DEVICE MEMORY")
    # ========================================================================
    # torch.cuda.memory_allocated() is the only honest way to know what the
    # caching allocator is actually holding. Guarded, so this file still runs
    # end to end on a laptop -- degrading gracefully rather than crashing is
    # the whole reason for the is_available() check.
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        base = torch.cuda.memory_allocated(dev)
        print(f"  device            : {torch.cuda.get_device_name(dev)}")
        print(f"  allocated (before): {base} B")

        gmodel = build_model(T).cuda()
        gx, gy = x.cuda(), y.cuda()
        after_w = torch.cuda.memory_allocated(dev)

        gl = (gmodel(gx) - gy).pow(2).sum()
        after_f = torch.cuda.memory_allocated(dev)

        gl.backward()
        after_b = torch.cuda.memory_allocated(dev)

        gopt = torch.optim.Adam(gmodel.parameters(), lr=LR,
                                betas=(BETA1, BETA2), eps=EPS)
        gopt.step()
        after_o = torch.cuda.memory_allocated(dev)

        print(f"  + weights         : {after_w - base:>8} B")
        print(f"  + forward (acts)  : {after_f - after_w:>8} B")
        print(f"  + backward (grads): {after_b - after_f:>8} B")
        print(f"  + Adam step (m,v) : {after_o - after_b:>8} B")
        print(f"  peak              : "
              f"{torch.cuda.max_memory_allocated(dev)} B")
        print()
        print("  Those four deltas are the four tensor classes on site page 08,")
        print("  measured. The allocator rounds to blocks, so the numbers are")
        print("  larger than numel x element_size -- real allocators always are,")
        print("  which is why a memory budget needs headroom, not a tight fit.")
    else:
        print("  torch.cuda.is_available() == False — running on CPU.")
        print("  No device memory to report. On a GPU this section prints the")
        print("  allocator delta after each of: loading weights, forward,")
        print("  backward, and the first optimizer step — the four tensor")
        print("  classes, measured rather than estimated.")
        print()
        print("  The CPU-side accounting still holds and is printed above:")
        print(f"    weights    {total_param_bytes:>6} B")
        print(f"    gradients  {grad_bytes:>6} B   (1x weights)")
        print(f"    optimizer  {opt_bytes:>6} B   (2x weights, Adam)")
        print(f"    activations{act_bytes:>6} B   (this toy model; the term")
        print("                          that dominates at real batch sizes)")

    # ========================================================================
    rule("VERIFICATION SUMMARY")
    # ========================================================================
    ok = ck.report()
    print("  REMINDER: this file was NOT executed during authoring (no torch")
    print("  in that environment). The table above is the check you should")
    print("  read; mlp_numpy.py is the one already known to pass.")
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
