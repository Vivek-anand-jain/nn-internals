# Deep dive: a 13-parameter network, derived by hand

This is the written companion to the interactive site. The site animates; this
document derives. Everything here is checked against two generated data files:
`assets/data/trace.json`, written by `code/ground_truth.py`, which covers
sections 1-8; and `assets/data/parallel.json`, written by
`code/parallel_toy.py`, which covers sections 9-14. Both are pure
standard-library Python — no numpy, no autograd, no NCCL, every derivative
written out longhand and every distributed strategy simulated rank by rank. If
a number appears below, it is either a literal field in one of those files or
is stated to be arithmetic performed on their fields.

Where the site shows something interactively, the page is named:
`site/01-memory.html`, `site/02-forward.html`, `site/03-loss.html`,
`site/04-backward.html`, `site/05-optimizer.html`, `site/06-loop.html`,
`site/07-transformer-forward.html`, `site/08-transformer-backward.html`,
`site/09-transformer-cost.html`, `site/10-scaling.html`,
`site/11-collectives.html`, `site/12-data-parallel.html`,
`site/13-zero-fsdp.html`, `site/14-tensor-parallel.html`,
`site/15-pipeline-parallel.html`, `site/16-transformer-partitioned.html`,
`site/17-3d-parallelism.html`. The two-layer transformer of pages 07, 08 and
16 has no section here yet; this document derives the 13-parameter MLP and
the four-rank distributed toy only.

A note on precision. The trace stores IEEE-754 doubles as Python computed
them, so `z1[0]` appears as `0.4999999999999999` and `dL/dz1[2]` as `-0.0`.
Not typos, not errors in the math — that is what binary floating point does to
decimal literals, and section 7 is about exactly this. Below I write the exact
decimal (`0.5`) and give the stored value where it matters.

---

## 1. The model

### 1.1 Problem

Predict a house price from two features.

```
x[0] = size, in thousands of square feet
x[1] = number of bedrooms
y    = price, in hundreds of thousands of dollars
```

From `T.meta.input`:

```
x = [2.0, 3.0]     a 2000 sqft, 3-bedroom house
y = 1.0            it sold for $100k
```

`T.meta.input.x_meaning` is `["size (1000s sqft)", "bedrooms"]` and
`y_meaning` is `"price ($100k)"`.

One example, one target. No batch dimension. Everything that is true of the
gradient here is true of a batched gradient with a sum over the batch axis
bolted on; leaving the batch axis out removes a `sum(...)` from every equation
without removing any structure.

### 1.2 Architecture

`T.meta.architecture` is `"2 -> 3 (ReLU) -> 1, MSE loss"`. Written as shapes:

```
x   : (2,)          input
W1  : (3, 2)        first layer weights
b1  : (3,)          first layer bias
z1  : (3,)          pre-activation
a1  : (3,)          post-activation
W2  : (1, 3)        second layer weights
b2  : (1,)          second layer bias
z2  : (1,)          output pre-activation
yhat: scalar        prediction
L   : scalar        loss
```

### 1.3 Forward equations

```
z1   = W1 @ x  + b1          (3,2) @ (2,) + (3,)  ->  (3,)
a1   = relu(z1)              elementwise, (3,) -> (3,)
z2   = W2 @ a1 + b2          (1,3) @ (3,) + (1,)  ->  (1,)
yhat = z2[0]                 identity output — this is regression
L    = (yhat - y)^2          squared error
```

Index form, which is what the derivation actually operates on:

```
z1[i] = sum_j W1[i][j] * x[j] + b1[i]         i in 0..2,  j in 0..1
a1[i] = max(z1[i], 0)
z2    = sum_j W2[0][j] * a1[j] + b2[0]        j in 0..2
L     = (z2 - y)^2
```

Two deliberate simplifications, stated so you know what is missing. There is
no activation on the output (`yhat = z2` exactly), which makes step 2 of the
backward pass an identity. And the loss is `(yhat - y)^2`, not
`(1/n) * sum (yhat - y)^2`; with `n = 1` these coincide, which is why the
factor of 2 in `dL/dyhat` survives unhalved. `torch.nn.MSELoss` with
`reduction='mean'` on a one-element tensor gives the same thing.

### 1.4 The 13 parameters

`T.param_names` enumerates them in the order the optimizer sees them after
flattening:

| # | Name | Init (`T.init`) |
|---|---|---|
| 0 | `W1[0][0]` | 0.5 |
| 1 | `W1[0][1]` | -0.2 |
| 2 | `W1[1][0]` | -0.3 |
| 3 | `W1[1][1]` | 0.8 |
| 4 | `W1[2][0]` | 0.1 |
| 5 | `W1[2][1]` | -0.4 |
| 6 | `b1[0]` | 0.1 |
| 7 | `b1[1]` | -0.5 |
| 8 | `b1[2]` | 0.2 |
| 9 | `W2[0][0]` | 0.7 |
| 10 | `W2[0][1]` | -0.6 |
| 11 | `W2[0][2]` | 0.9 |
| 12 | `b2[0]` | 0.3 |

6 + 3 + 3 + 1 = 13. `T.memory.n_params` is 13.

The flattening order is row-major within each tensor, tensors concatenated in
declaration order (`ground_truth.py:flatten`) — exactly what a framework does
when it builds a *flat parameter*, so the optimizer step is one big
elementwise kernel instead of thirteen tiny ones. `site/01-memory.html` shows
`W1`'s row-major layout cell by cell from `T.layout`: strides `[2, 1]` in
elements, `[8, 4]` in bytes at fp32, flat contents
`[0.5, -0.2, -0.3, 0.8, 0.1, -0.4]`.

The initial values are hand-picked, not random, for two reasons stated in the
source: the forward pass lands on clean decimals, and hidden unit 2 receives a
negative pre-activation, so ReLU zeroes it and its gradient is exactly zero.
That dead unit is the centrepiece of the whole project.

Hyperparameters, from `T.meta.hyperparams`: `lr = 0.1`, `beta1 = 0.9`,
`beta2 = 0.999`, `eps = 1e-8`, `n_steps = 12`.

---

## 2. Forward pass, fully worked

Animated multiply-by-multiply on `site/02-forward.html`; every product below
is a field in `T.forward.z1_work` / `T.forward.z2_work`.

### 2.1 z1 = W1 @ x + b1

Row 0:

```
W1[0][0] * x[0] = 0.5  * 2.0 =  1.0
W1[0][1] * x[1] = -0.2 * 3.0 = -0.6
sum_of_products = 0.4
+ b1[0]         = 0.1
z1[0]           = 0.5
```

Row 1:

```
W1[1][0] * x[0] = -0.3 * 2.0 = -0.6
W1[1][1] * x[1] =  0.8 * 3.0 =  2.4
sum_of_products = 1.8
+ b1[1]         = -0.5
z1[1]           = 1.3
```

Row 2:

```
W1[2][0] * x[0] =  0.1 * 2.0 =  0.2
W1[2][1] * x[1] = -0.4 * 3.0 = -1.2
sum_of_products = -1.0
+ b1[2]         = 0.2
z1[2]           = -0.8
```

So `z1 = [0.5, 1.3, -0.8]`. The stored values in `T.forward.z1` are
`0.4999999999999999`, `1.3000000000000003`, `-0.8000000000000003`. Six
multiplies, six adds. That is the entire first layer.

### 2.2 a1 = relu(z1) — the dead unit

```
a1[0] = max(0.5,  0) = 0.5
a1[1] = max(1.3,  0) = 1.3
a1[2] = max(-0.8, 0) = 0.0     <- DEAD
```

`T.forward.a1 = [0.5, 1.3, 0.0]`, `T.forward.relu_mask = [1, 1, 0]`.

Hidden unit 2 is dead for this input. It computed a negative number, ReLU
clamped it to zero, and it therefore contributes exactly nothing to the
prediction. Keep this in view: the entire structure of the backward pass is
"assign blame in proportion to contribution", and a unit that contributed
nothing will take none of the blame. We will see its gradient row come out
exactly `[0, 0]`.

### 2.3 z2 = W2 @ a1 + b2

```
W2[0][0] * a1[0] =  0.7 * 0.5 =  0.35
W2[0][1] * a1[1] = -0.6 * 1.3 = -0.78
W2[0][2] * a1[2] =  0.9 * 0.0 =  0.00     <- the dead unit, again
sum_of_products  = -0.43
+ b2[0]          = 0.3
z2               = -0.13
```

`T.forward.z2 = [-0.13000000000000023]`.

Note that `W2[0][2] = 0.9` — a perfectly ordinary weight — has no effect on
the output at all, because the thing it multiplies is zero. Its gradient will
be zero too, and it will never be updated. Confirmed in the trace: across all
twelve steps of every optimizer run, `W2[0][2]` stays at `0.9` and `W1[2][0]`,
`W1[2][1]`, `b1[2]` stay at `0.1`, `-0.4`, `0.2`.

### 2.4 Loss

```
yhat  = z2 = -0.13
error = yhat - y = -0.13 - 1.0 = -1.13
L     = error^2 = (-1.13)^2 = 1.2769
```

`T.forward.yhat = -0.13000000000000023`, `T.forward.error =
-1.1300000000000003`, `T.forward.loss = 1.2769000000000008`.

The model predicted `-$13k` for a house that sold for `$100k`. It is a
randomly-initialised network; this is expected. `site/03-loss.html` shows the
loss surface around this point.

### 2.5 What forward left in memory

`T.forward.saved_for_backward` names it explicitly:

```
x         : needed for dL/dW1
a1        : needed for dL/dW2
relu_mask : needed for dL/dz1
```

Nothing else from the forward pass is required — `z1` itself can be freed once
the mask has been derived from it. Section 4 is about why this list is the
whole story.

---

## 3. Backward pass, from first principles

This is the heart of the document. Eight steps, in the order
`T.backward.steps` lists them. Animated on `site/04-backward.html`.

### 3.0 The shape rule, stated once

**A gradient has the same shape as the thing it is a gradient of.**

This follows from the loss being a scalar. For a scalar function `L` of a
tensor `T` with shape `S`, the derivative `dL/dT` has one entry per entry of
`T` — the partial derivative of `L` with respect to that entry — so it has
shape `S`. Concretely:

```
W1 is (3,2)   =>  dL/dW1 is (3,2)
b1 is (3,)    =>  dL/db1 is (3,)
a1 is (3,)    =>  dL/da1 is (3,)
W2 is (1,3)   =>  dL/dW2 is (1,3)
b2 is (1,)    =>  dL/db2 is (1,)
yhat scalar   =>  dL/dyhat is scalar
```

This is why `dL/dW1` is a 3x2 matrix and `dL/da1` is a length-3 vector, and it
is also why the gradient buffer for a model is always exactly the same size as
the model. The `+4 bytes per parameter` in the memory accounting of section 8
is a direct consequence of this one line.

The general Jacobian is *not* this shape. The true Jacobian of `z1` (shape
`(3,)`) with respect to `W1` (shape `(3,2)`) is a rank-3 tensor of shape
`(3, 3, 2)` — 18 numbers. Backprop never builds it. Two reasons:

1. It is enormously sparse. `dz1[k]/dW1[i][j] = x[j]` if `k == i`, else `0`,
   because `W1[i][j]` only ever appears in row `i` of the matrix-vector
   product. 12 of the 18 entries are structurally zero.
2. We never need the Jacobian, only its product with an incoming gradient
   vector — a vector-Jacobian product. The chain rule for a scalar loss is

   ```
   dL/dW1[i][j] = sum_k (dL/dz1[k]) * (dz1[k]/dW1[i][j])
   ```

   and the sparsity collapses the sum to a single term:

   ```
   dL/dW1[i][j] = dL/dz1[i] * x[j]
   ```

That last line is the outer product. Scale it up: for a linear layer with 8192
inputs and 8192 outputs, the explicit Jacobian would be `8192^3` = 5.5e11
entries = 2.2 TB at fp32, for a layer whose weights are 268 MB. Reverse-mode
autodiff exists precisely so that this object is never constructed. Every
`backward` in every framework is a hand-written VJP, not a Jacobian.

---

### Step 1 — `dL/dyhat`

**Local function.** `L(yhat) = (yhat - y)^2`, with `y` a constant.

**Symbolic derivative.** Let `u = yhat - y`, so `L = u^2`.

```
dL/du   = 2u
du/dyhat = 1
dL/dyhat = dL/du * du/dyhat = 2u * 1 = 2(yhat - y)
```

Or directly by expansion, `L = yhat^2 - 2*y*yhat + y^2`, so
`dL/dyhat = 2*yhat - 2*y = 2(yhat - y)`. Same answer.

**Chain rule.** There is nothing upstream of the loss; this is the seed of the
whole backward pass, the point where `dL/dL = 1` enters and gets multiplied by
the first local derivative.

**Substitute.** `T.backward.steps[0].substitution` is `2 × (-0.13 − 1)`.

```
dL/dyhat = 2 * (-0.13 - 1.0) = 2 * (-1.13) = -2.26
```

**Result.** `T.backward.dL_dyhat = -2.2600000000000007`. Shape: scalar.

The sign is informative. It is negative, meaning: increasing `yhat` decreases
`L`. Correct — we predicted `-0.13` and the target is `1.0`, so we are too
low.

---

### Step 2 — `dL/dz2`

**Local function.** `yhat = z2`. Identity.

**Symbolic derivative.** `d(yhat)/d(z2) = 1`.

**Chain rule.** `dL/dz2 = dL/dyhat * dyhat/dz2 = dL/dyhat * 1`.

**Substitute.** `-2.26 × 1`.

**Result.** `T.backward.dL_dz2 = -2.2600000000000007`. Shape `(1,)`.

This step is free and in a real implementation it does not exist — the output
layer's pre-activation *is* the prediction, and the framework fuses them. It
is written out here because when there *is* an output activation (a sigmoid,
a softmax, a `log_softmax` folded into cross-entropy), this is the slot it
occupies, and you want to know where to look.

---

### Step 3a — `dL/dW2`

**Local function.** `z2 = sum_j W2[0][j] * a1[j] + b2[0]`.

**Symbolic derivative.** Differentiate with respect to one weight
`W2[0][k]`. Every term in the sum is a constant with respect to it except the
`j = k` term:

```
dz2/dW2[0][k] = d/dW2[0][k] ( W2[0][k]*a1[k] + (terms without W2[0][k]) )
              = a1[k]
```

**Chain rule.**

```
dL/dW2[0][k] = dL/dz2 * dz2/dW2[0][k] = dL/dz2 * a1[k]
```

Across all `k` at once this is an outer product of a length-1 vector with a
length-3 vector, giving a `(1,3)` result — same shape as `W2`, as required:

```
dL/dW2 = dL/dz2  ⊗  a1^T
```

**Substitute.** `T.backward.steps[2].substitution` is
`-2.26 × [0.5, 1.3, 0.0]`.

```
dL/dW2[0][0] = -2.26 * 0.5 = -1.13
dL/dW2[0][1] = -2.26 * 1.3 = -2.938
dL/dW2[0][2] = -2.26 * 0.0 =  0.0
```

**Result.** `T.backward.dL_dW2 = [[-1.1300000000000001,
-2.9380000000000015, -0.0]]`. Shape `(1,3)`.

**This is why activations are stored.** To know how much `W2[0][k]` is to
blame, you must know what it was multiplied by — `a1[k]`, computed during the
forward pass — and there is no way to recover `a1` at backward time except by
keeping it or recomputing it. The third entry is the dead unit collecting its
zero share of the blame: `W2[0][2]` multiplied a zero, so changing it changes
nothing, so gradient descent will never move it.

---

### Step 3b — `dL/db2`

**Local function.** Same equation, differentiate with respect to `b2[0]`.

**Symbolic derivative.** `b2[0]` appears once, added, coefficient 1:

```
dz2/db2[0] = 1
```

**Chain rule.** `dL/db2[0] = dL/dz2 * 1 = dL/dz2`.

**Substitute.** `-2.26 × 1`.

**Result.** `T.backward.dL_db2 = [-2.2600000000000007]`. Shape `(1,)`.

A bias adds itself in unchanged, so it receives the gradient unchanged. Bias
gradients are the cheapest thing in backprop: no multiply, no saved tensor.
(In the batched case it becomes a sum over the batch axis, still no saved
tensor.)

---

### Step 4 — `dL/da1`

**Local function.** Same `z2` equation, now differentiate with respect to the
*input* `a1[k]` rather than the weight.

**Symbolic derivative.**

```
dz2/da1[k] = d/da1[k] ( W2[0][k]*a1[k] + ... ) = W2[0][k]
```

**Chain rule.** `a1[k]` influences `L` only through `z2`, so:

```
dL/da1[k] = dL/dz2 * W2[0][k]
```

Written for all `k` at once, this is a transposed matrix-vector product:

```
dL/da1 = W2^T @ dL/dz2      (3,1) @ (1,) -> (3,)
```

Shape `(3,)` — same shape as `a1`, as required.

**Substitute.** `T.backward.steps[4].substitution` is
`[0.7, -0.6, 0.9] × -2.26`.

```
dL/da1[0] =  0.7 * -2.26 = -1.582
dL/da1[1] = -0.6 * -2.26 =  1.356
dL/da1[2] =  0.9 * -2.26 = -2.034
```

**Result.** `T.backward.dL_da1 = [-1.5820000000000003, 1.3560000000000003,
-2.0340000000000007]`. Shape `(3,)`.

Two things to notice.

First, the symmetry, which is the single most useful structural fact about a
linear layer:

```
weight gradient  dL/dW = delta ⊗ input^T     <- reads the INPUT
input gradient   dL/dinput = W^T @ delta     <- reads the WEIGHTS
```

Forward uses `W`; backward-to-weights uses the input; backward-to-input uses
`W` again. This is why a linear layer must keep its input alive from forward
until backward, and why it need not keep its output.

Second, `dL/da1[2] = -2.034` is *not* zero, even though unit 2 is dead. This
is not a contradiction. It is answering a counterfactual: "if `a1[2]` were
somehow larger, would the loss go down?" Yes, by `2.034` per unit. But `a1[2]`
cannot be made larger by any change to `W1` or `b1` that is small enough for
the derivative to be valid, because ReLU has zero slope there. Step 5 is where
that counterfactual gets killed.

---

### Step 5 — `dL/dz1` (through ReLU)

**Local function.** `a1[i] = relu(z1[i]) = max(z1[i], 0)`, elementwise.

**Symbolic derivative.** ReLU is piecewise linear:

```
             { z   if z > 0
relu(z)   =  {
             { 0   if z <= 0

d relu/dz =  { 1   if z > 0
             { 0   if z < 0
             { undefined at z = 0
```

At `z = 0` the left derivative is 0 and the right is 1, so the function is not
differentiable there. Every framework picks a subgradient by convention;
PyTorch and this implementation both pick `0` (the test is `z > 0`, strictly).
The set of inputs landing exactly on 0 has measure zero, so it does not matter
numerically — but the convention must be fixed, or gradient checks stop being
reproducible.

Because ReLU is elementwise, its Jacobian is diagonal, so the vector-Jacobian
product is a Hadamard (elementwise) product rather than a matmul:

```
dL/dz1 = dL/da1  ⊙  1[z1 > 0]
```

**Chain rule + substitute.** `T.backward.steps[5].substitution` is
`[-1.582, 1.356, -2.034] ⊙ [1, 1, 0]`.

```
dL/dz1[0] = -1.582 * 1 = -1.582
dL/dz1[1] =  1.356 * 1 =  1.356
dL/dz1[2] = -2.034 * 0 =  0.0
```

**Result.** `T.backward.dL_dz1 = [-1.5820000000000003, 1.3560000000000003,
-0.0]`. Shape `(3,)`.

The `-0.0` is IEEE-754 signed zero: `-2.034 * 0.0` produces negative zero,
which compares equal to `+0.0` and behaves identically in every arithmetic
operation that matters here.

This is the step that kills the dead unit. Unit 2 had `z1[2] = -0.8`, ReLU
clamped it to `0`, its gradient is now exactly `0`, so no update reaches it.
And because nothing updates `W1[2][*]` or `b1[2]`, `z1[2]` is still `-0.8`
next step: the unit stays dead. Confirmed — across all twelve steps of the
Adam run, `T.runs.adam.history[t].relu_mask` is `[1, 1, 0]` every time.

The systems consequence: the *only* thing backward needs from forward here is
one bit per element. Not `z1`, not `a1` — a boolean mask, 1/32 the memory of
`z1` at fp32. In practice PyTorch's ReLU saves its own output and tests
`out > 0`, which is free when the next op already retains `a1` — which, per
step 3a, it does.

---

### Step 6a — `dL/dW1`

**Local function.** `z1[i] = sum_j W1[i][j] * x[j] + b1[i]`.

**Symbolic derivative.** As in step 3a, but now indexed over both rows and
columns:

```
dz1[k]/dW1[i][j] = { x[j]  if k == i
                   { 0     otherwise
```

The `k != i` case is zero because `W1[i][j]` appears only in row `i`.

**Chain rule.**

```
dL/dW1[i][j] = sum_k (dL/dz1[k]) * (dz1[k]/dW1[i][j])
             = dL/dz1[i] * x[j]              (all other terms vanish)
```

which is the outer product:

```
dL/dW1 = dL/dz1  ⊗  x^T          (3,) ⊗ (2,) -> (3,2)
```

Shape `(3,2)` — same shape as `W1`. Not a coincidence, and not a convention:
it falls out of the sparsity pattern above. Every weight `W1[i][j]` gets
exactly one term, the product of the gradient arriving at its output row and
the input feeding its input column.

**Substitute.** `T.backward.steps[6].substitution` is
`[-1.582, 1.356, -0.0] ⊗ [2.0, 3.0]`.

```
row 0:  -1.582 * 2.0 = -3.164     -1.582 * 3.0 = -4.746
row 1:   1.356 * 2.0 =  2.712      1.356 * 3.0 =  4.068
row 2:   0.0   * 2.0 =  0.0        0.0   * 3.0 =  0.0
```

**Result.** `T.backward.dL_dW1`:

```
[[-3.1640000000000006, -4.746             ],
 [ 2.7120000000000006,  4.068000000000001 ],
 [-0.0,                -0.0               ]]
```

Row 2 is exactly `[0, 0]`. The dead ReLU unit propagated its zero backwards
into both of its incoming weights.

**Why the input must be retained.** Look at what `dL/dW1 = delta ⊗ x^T`
actually requires at the moment backward runs:

- `delta = dL/dz1` comes from downstream. It was computed a microsecond ago
  and is still in a register or in L2. It is transient.
- `x` came from *upstream*, at the very start of the forward pass. Between the
  moment `x` was consumed and the moment it is needed again, the entire rest
  of the forward pass and most of the backward pass have run.

There is no algebraic path from `delta` back to `x`. The outer-product
structure means the weight gradient is *literally a function of the layer's
input*, so that input must be alive across the whole forward-backward span or
it must be recomputed. That is the entire economics of activation memory: not
"frameworks are wasteful", but "the derivative of a product is the other
factor".

### The full expanded chain for `W1[0][0]`

Every step above, composed into one product. `W1[0][0]` influences the loss
along exactly one path: it scales `x[0]` into `z1[0]`, which passes through
ReLU into `a1[0]`, which is weighted by `W2[0][0]` into `z2`, which is `yhat`,
which enters the squared error.

```
dL          dL      dyhat     dz2      da1[0]     dz1[0]
------  =  ----  *  -----  *  ------ * ------  *  ---------
dW1[0][0]  dyhat     dz2      da1[0]   dz1[0]     dW1[0][0]
```

Term by term, with the local derivative named and then evaluated:

| Factor | Local derivative | Value | Source |
|---|---|---|---|
| `dL/dyhat` | `2(yhat - y)` | `-2.26` | `T.backward.dL_dyhat` |
| `dyhat/dz2` | `1` (identity output) | `1` | step 2 |
| `dz2/da1[0]` | `W2[0][0]` | `0.7` | `T.init.W2[0][0]` |
| `da1[0]/dz1[0]` | `1[z1[0] > 0]` | `1` | `T.forward.relu_mask[0]` |
| `dz1[0]/dW1[0][0]` | `x[0]` | `2.0` | `T.forward.x[0]` |

Multiplied out:

```
dL/dW1[0][0] = (-2.26) * 1 * 0.7 * 1 * 2.0
             = (-2.26) * 0.7 * 2.0
             = (-1.582) * 2.0
             = -3.164
```

`T.backward.dL_dW1[0][0] = -3.1640000000000006`. The intermediate `-1.582` is
`T.backward.dL_dz1[0]`, which is what makes this a *chain* rather than five
independent calculations: backprop computes `-1.582` once and reuses it for
`W1[0][1]`, for `b1[0]`, and for anything else downstream of `z1[0]`.

Contrast with `W1[2][0]`, the dead unit's weight. The same chain, with the
mask factor flipped:

```
dL/dW1[2][0] = (-2.26) * 1 * 0.9 * 0 * 2.0 = 0
```

One factor of zero in the middle annihilates the entire path. That is what a
dead unit *is*: a zero in the middle of every chain that passes through it.

---

### Step 6b — `dL/db1`

**Local function.** Same `z1` equation, differentiated with respect to
`b1[i]`.

**Symbolic derivative.** `dz1[k]/db1[i] = 1` if `k == i`, else `0`.

**Chain rule.** `dL/db1[i] = dL/dz1[i]`.

**Substitute / result.** `T.backward.dL_db1 = [-1.5820000000000003,
1.3560000000000003, -0.0]`, shape `(3,)`. Identical to `dL/dz1`, as expected.

Biases again receive the gradient untouched, and again the dead unit's entry
is zero.

---

### 3.7 The complete gradient

Flattened in `T.param_names` order, this is `T.gradcheck.analytic`:

```
W1[0][0]  -3.164
W1[0][1]  -4.746
W1[1][0]   2.712
W1[1][1]   4.068
W1[2][0]  -0.0      <- dead
W1[2][1]  -0.0      <- dead
b1[0]     -1.582
b1[1]      1.356
b1[2]     -0.0      <- dead
W2[0][0]  -1.13
W2[0][1]  -2.938
W2[0][2]  -0.0      <- dead
b2[0]     -2.26
```

Four of thirteen parameters have exactly zero gradient. In a 2-3-1 network
with one example that is a curiosity; in a 70B model with ReLU-family
activations it is a structural fact that roughly half of every activation
tensor is zero, which is why sparsity-aware kernels and activation
quantization schemes exist.

---

## 4. Why activations are saved

The short version, from the derivations above:

```
dL/dW2 = dL/dz2 ⊗ a1^T      needs a1
dL/dW1 = dL/dz1 ⊗ x^T       needs x
dL/dz1 = dL/da1 ⊙ mask      needs sign(z1)
```

Three of the eight backward steps read a tensor that was produced during the
forward pass. `dL/dyhat`, `dL/dz2`, `dL/db2`, `dL/db1` read nothing — their
`reads` field in `T.backward.steps` is empty. `dL/da1` reads `W2`, which is a
parameter and would be resident anyway.

So the rule is narrow and precise: **a weight gradient needs the layer's
input; an elementwise activation's backward needs enough of its own input to
evaluate its derivative.** Everything else is free.

### 4.1 The lifetime problem

Consider the timeline of `a1` in a single training step:

```
t0  forward: a1 computed from z1
t1  forward: a1 consumed by layer 2
t2  forward: loss computed
t3  backward: dL/dz2 computed
t4  backward: dL/dW2 computed  <- a1 read here
t5  a1 can finally be freed
```

Between `t1` and `t4`, `a1` is used by nothing and can be freed by nothing.
Here that is 3 floats. In an 80-layer transformer, layer 1's activations stay
resident through 79 forward layers, the loss, and 79 backward layers. Peak
activation memory is therefore proportional to *depth x width x batch x
sequence*, and it peaks exactly at the loss — the one moment when every
layer's saved tensor is alive and none has been consumed.
`site/06-loop.html` animates this.

### 4.2 The memory consequence

For this model, `T.memory.by_class.activation` counts 13 activation elements
(`x`: 2, `z1`: 3, `a1`: 3, `relu_mask`: 3, `z2`: 1, `loss`: 1) against 13
parameters. Activations are exactly the *same size* as the parameters here,
because batch size is 1 and the network is three layers deep.

That ratio inverts hard at scale, because activation memory scales with
`batch x sequence x hidden x layers` while parameter memory scales with
`hidden^2 x layers`. Using the per-layer formula from Korthikanti et al.
(2022) — *stated as literature, not derived here* — a transformer layer with
no parallelism and no recomputation stores about

```
s * b * h * (34 + 5 * a * s / h)   bytes at 16-bit
```

where `s` = sequence length, `b` = microbatch, `h` = hidden size, `a` = number
of attention heads. For a 70B-class configuration (`h = 8192`, `a = 64`,
`layers = 80`) with `s = 4096`, `b = 1`:

```
s*b*h                = 4096 * 1 * 8192  = 33.55e6
34 * s*b*h           = 1.14e9  bytes    = 1.14 GB   (linear term)
5*a*s/h              = 5 * 64 * 0.5     = 160
160 * s*b*h          = 5.37e9 bytes     = 5.37 GB   (attention-matrix term)
per layer            = 6.51 GB
x 80 layers          = 521 GB
```

521 GB of activations for a *single microbatch* against 140 GB of bf16
weights. This is the arithmetic that made FlashAttention and gradient
checkpointing mandatory rather than optional. FlashAttention removes the
`5*a*s/h` term entirely by never materialising the `s x s` attention matrix
(it recomputes tiles of it in the backward pass from the saved
softmax statistics), taking the estimate from 521 GB to about 91 GB.

### 4.3 Gradient checkpointing and the sqrt(n) result

The trade is: **do not save, recompute.** Checkpointing marks a subset of
layers as *checkpoints*, saves only those, and discards the rest. When
backward reaches a discarded segment, it re-runs that segment's forward pass
from the nearest upstream checkpoint to regenerate the tensors it needs, uses
them, and frees them again.

Chen, Xu, Zhang & Guestrin (2016) give the optimal-in-order placement. The
reasoning, for a network of `n` uniform layers each storing one unit of
activation, checkpointing every `k`-th layer:

- **Checkpoint storage.** There are `n/k` checkpoints, each costing 1 unit.
  Cost: `n/k`.
- **Peak recompute storage.** While recomputing one segment, you hold that
  segment's `k` intermediate activations. Cost: `k`.
- **Total peak:** `M(k) = n/k + k`.

Minimise over `k`:

```
dM/dk = -n/k^2 + 1 = 0   =>   k^2 = n   =>   k = sqrt(n)
M(sqrt(n)) = n/sqrt(n) + sqrt(n) = 2*sqrt(n)
```

The second derivative `2n/k^3` is positive for `k > 0`, so this is a minimum.

So memory goes from `O(n)` to `O(sqrt(n))`, and the compute cost is exactly
**one extra forward pass** — each discarded layer is recomputed exactly once,
during the backward pass over its segment. Since a forward pass is roughly a
third of the cost of a forward-plus-backward step (backward is about 2x
forward: one matmul for the input gradient, one for the weight gradient), the
overhead is roughly **+33% step time for an `O(sqrt(n))` memory bound**.

For `n = 80` layers: `sqrt(80) ≈ 8.9`, so checkpoint every 9th layer, peak
activation storage `2*sqrt(80) ≈ 17.9` layer-equivalents instead of 80 — a
4.5x reduction.

In practice most transformer implementations checkpoint at *every* block
boundary (`k = 1`), which the uniform-cost model above says is the worst
choice. It is not, because the model's assumption — that each layer stores one
unit — is false for a transformer block. The block's *boundary* tensor is
`s*b*h`; its *interior* tensors sum to roughly `34*s*b*h` (section 4.2). So
checkpointing at block boundaries stores 1 unit and discards 34, capturing
almost the entire memory win at the same fixed 33% recompute cost. The
`sqrt(n)` result is the correct answer to the uniform-cost problem; production
systems are solving a strongly non-uniform version of it, and the answer moves
to `k = 1`.

---

## 5. Numerical verification

Hand-derived calculus is exactly as reliable as the hand that derived it. The
check is independent: perturb each parameter and measure how the loss actually
moves.

### 5.1 The central difference

From the Taylor expansions about `theta`:

```
L(theta + h) = L + h*L' + (h^2/2)*L'' + (h^3/6)*L''' + O(h^4)
L(theta - h) = L - h*L' + (h^2/2)*L'' - (h^3/6)*L''' + O(h^4)
```

Subtract. The `L` and `L''` terms cancel:

```
L(theta+h) - L(theta-h) = 2h*L' + (h^3/3)*L''' + O(h^5)
```

Divide by `2h`:

```
L(theta+h) - L(theta-h)
----------------------- = L' + (h^2/6)*L''' + O(h^4)
         2h
```

So the **truncation error is O(h^2)**. The one-sided difference
`(L(theta+h) - L(theta))/h` retains the `L''` term and is only `O(h)` — one
extra function evaluation buys you a whole order. `ground_truth.py`'s
`numerical_gradient` uses the central form.

### 5.2 Why h = 1e-6

Two error sources pull in opposite directions.

**Truncation error**, from above, is `~ (h^2/6) * |L'''|`. It *decreases* as
`h` shrinks.

**Round-off error.** `L(theta+h)` and `L(theta-h)` are nearly equal numbers;
subtracting them is catastrophic cancellation. Each is computed to a relative
accuracy of about `eps_machine`, so the absolute error in each is
`~ eps * |L|`, and the error in the difference is `~ 2*eps*|L|`. Dividing by
`2h` amplifies it:

```
round-off in the quotient  ~  eps * |L| / h
```

This *increases* as `h` shrinks — an `O(eps/h)` term.

Total error `E(h) ≈ (h^2/6)*|L'''| + eps*|L|/h`. Minimise:

```
dE/dh = (h/3)*|L'''| - eps*|L|/h^2 = 0
h^3 = 3 * eps * |L| / |L'''|
h_opt ~ eps^(1/3)   (up to the ratio of derivative magnitudes)
```

For IEEE double, `eps = 2^-52 ≈ 2.2e-16`, so `eps^(1/3) ≈ 6.1e-6`. `h = 1e-6`
sits within a factor of ~6 of that optimum, and because the total error curve
is flat near its minimum (it varies as `h^2` on one side and `1/h` on the
other), being within an order of magnitude costs almost nothing. That is the
whole justification: `1e-6` is the round number nearest `eps^(1/3)` for
doubles.

The same argument for fp32 (`eps ≈ 1.2e-7`) gives `h_opt ≈ 5e-3` — which is
why gradient checking in fp32 is nearly useless, and why every framework's
`gradcheck` upcasts to float64 first.

### 5.3 Result

`T.gradcheck` reports:

```
max_abs_error = 2.854618763592498e-10
passed        = true      (threshold: max_abs_error < 1e-6)
```

Per-parameter, analytic vs numerical (both from `T.gradcheck`):

| Parameter | Analytic | Numerical | \|diff\| |
|---|---|---|---|
| `W1[0][0]` | -3.1640000000000006 | -3.164000000110967 | 1.11e-10 |
| `W1[0][1]` | -4.746 | -4.745999999999917 | 8.35e-14 |
| `W1[1][0]` | 2.7120000000000006 | 2.7119999997937683 | 2.06e-10 |
| `W1[1][1]` | 4.068000000000001 | 4.068000000190253 | 1.90e-10 |
| `W1[2][0]` | -0.0 | 0.0 | 0 |
| `W1[2][1]` | -0.0 | 0.0 | 0 |
| `b1[0]` | -1.5820000000000003 | -1.582000000222017 | 2.22e-10 |
| `b1[1]` | 1.3560000000000003 | 1.3560000002854622 | **2.85e-10** |
| `b1[2]` | -0.0 | 0.0 | 0 |
| `W2[0][0]` | -1.1300000000000001 | -1.1299999999048183 | 9.52e-11 |
| `W2[0][1]` | -2.9380000000000015 | -2.9379999999523676 | 4.76e-11 |
| `W2[0][2]` | -0.0 | 0.0 | 0 |
| `b2[0]` | -2.2600000000000007 | -2.260000000031681 | 3.17e-11 |

The maximum, `2.85e-10`, occurs at `b1[1]`. Note the four dead parameters
agree to *exactly* zero: perturbing `W1[2][0]` by `±1e-6` moves `z1[2]` from
`-0.8` to `-0.799998` / `-0.800002`, both still negative, so ReLU still
outputs `0`, so the loss is bit-identical and the difference is exactly `0`.
The finite-difference check independently confirms the dead unit — it is not
an artifact of the derivation.

`ground_truth.py:main` raises `SystemExit` if this check fails, so the trace
cannot be generated with wrong math.

The residual `~1e-10` is round-off, as predicted: `eps*|L|/h ≈ 2.2e-16 *
1.28 / 1e-6 ≈ 2.8e-10`. The prediction and the observation agree to one
significant figure, which is about as good as this kind of estimate gets.

---

## 6. Optimizers

We now have `g = dL/dtheta` for all 13 parameters. The optimizer's job is to
turn that into an update. `site/05-optimizer.html` runs all three side by
side over `T.meta.hyperparams.n_steps = 12` steps.

### 6.1 SGD

```
theta_{t+1} = theta_t - lr * g_t
```

The derivation is one line of calculus: the gradient points in the direction
of steepest *increase* of `L`, so step against it. For small enough `lr`,
`L(theta - lr*g) ≈ L(theta) - lr*|g|^2 < L(theta)`.

With `lr = 0.1` and the gradients above, `T.runs.sgd.history[0]`:

```
W1[0][0]: 0.5 - 0.1*(-3.164) = 0.5 + 0.3164 = 0.8164
W1[2][0]: 0.1 - 0.1*(0)      = 0.1            (unchanged — dead)
b2[0]:    0.3 - 0.1*(-2.26)  = 0.3 + 0.226  = 0.526
```

`T.runs.sgd.history[0].theta_after` confirms `[0.8164, 0.2746, -0.5712,
0.3932, 0.1, -0.4, 0.2582, -0.6356, 0.2, 0.813, -0.3062, 0.9, 0.526]`.

**Memory cost: 0 tensors per parameter.** SGD keeps no state. This is why it
is still the memory-optimal choice and why it survives in vision training.

**What actually happens in this run.** From `T.runs.sgd.history[*].loss`:
`1.2769, 3.0037, 0.6734, 0.4310, ... , 0.0121`.

`lr = 0.1` is too large. Step 1 overshoots to `yhat = 2.733` against a target
of `1.0`, and the overshoot pushes `z1` negative for *every* hidden unit:
`T.runs.sgd.history[t].relu_mask` is `[1,1,0]` at `t=1`, `[1,0,0]` at `t=2`,
and `[0,0,0]` for `t=3` through `t=12`. The whole hidden layer dies.

After that `a1 = [0,0,0]`, so `yhat = b2[0]` and `b2` is the only parameter
with a nonzero gradient. The dynamics reduce to a scalar recurrence:

```
b2_{t+1} = b2_t - 0.1 * 2 * (b2_t - 1)
(b2_{t+1} - 1) = 0.8 * (b2_t - 1)
L_{t+1} / L_t = 0.8^2 = 0.64
```

Check against `T.runs.sgd.history`: `0.176535/0.275836 = 0.64` exactly, and so
is every subsequent ratio. The final loss `0.007764` with
`final_yhat = 0.9119` looks like successful training. It is not — it is a dead
network whose output bias is converging to the target. *A decreasing loss
curve is not evidence that a model is learning.*

### 6.2 Momentum

```
v_{t} = beta * v_{t-1} + g_t
theta_{t+1} = theta_t - lr * v_t
```

(This is the `torch.optim.SGD(momentum=beta)` formulation, which does *not*
scale `g` by `(1-beta)`. `ground_truth.py:momentum_step` matches it.)

Unrolling the recurrence with `v_0 = 0`:

```
v_t = g_t + beta*g_{t-1} + beta^2*g_{t-2} + ... = sum_{k=0}^{t-1} beta^k * g_{t-k}
```

An exponentially weighted sum of all past gradients, with a geometric
weighting of ratio `beta`. Its effective length is `sum_k beta^k = 1/(1-beta)`,
which for `beta = 0.9` is 10. Two consequences:

- Directions where the gradient is *consistent* accumulate: the steady-state
  step is `lr/(1-beta) = 0.1/0.1 = 1.0` — a **10x amplification** over plain
  SGD.
- Directions where the gradient *oscillates* cancel, damping the zig-zag
  across a narrow valley.

At `t = 1`, `v_1 = beta*0 + g_1 = g_1`, so the first momentum step is
*identical* to the first SGD step. Confirmed:
`T.runs.momentum.history[0].velocity` equals `T.gradcheck.analytic` entry for
entry (up to signed zero), and `T.runs.momentum.history[1].loss = 3.003748`
matches `T.runs.sgd.history[1].loss` exactly.

Thereafter they diverge. `T.runs.momentum.history[1].velocity[9]` (the
`W2[0][0]` slot) is `8.393`, against a step-1 velocity of `-1.13` for the same
slot — sign flipped and magnitude up 7.4x in one step. The loss trace
oscillates: `1.2769, 3.0037, 0.4062,
0.3828, 0.2462, 0.0823, 0.0017, 0.0355, 0.1277, 0.1917, 0.1786, 0.1053`. It
reaches a far better minimum than SGD (`0.001676` at `t=7`) and then sails
past it. Momentum is a different failure mode, not a fix for the learning rate.

**Memory cost: 1 tensor per parameter** (`v`).

### 6.3 Adam

```
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t          first moment
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2        second moment
m_hat = m_t / (1 - beta1^t)                        bias correction
v_hat = v_t / (1 - beta2^t)
theta_{t+1} = theta_t - lr * m_hat / (sqrt(v_hat) + eps)
```

**What `m` and `v` estimate.** Both are exponential moving averages, and both
are estimators of a population quantity over the distribution of minibatch
gradients:

- `m_t` estimates `E[g]`, the mean gradient. It is momentum with a `(1-beta1)`
  normalisation so that it is an average rather than a sum.
- `v_t` estimates `E[g^2]`, the uncentered second moment. `sqrt(v_hat)` is
  therefore an estimate of the *root-mean-square magnitude* of the gradient
  for that parameter.

The ratio `m_hat / sqrt(v_hat)` is thus roughly `E[g] / RMS[g]` — a
signal-to-noise ratio, dimensionless, and crucially **scale-invariant**:
multiply the loss by 1000 and every `g` scales by 1000, `m_hat` scales by
1000, `sqrt(v_hat)` scales by 1000, and the update is unchanged. This is why
Adam needs so much less learning-rate tuning than SGD, and why it survives
loss-scaled fp16 training (section 7) without the scale factor leaking into
the step size.

**Why bias correction is needed — derived.** Unroll `m_t` from `m_0 = 0`:

```
m_1 = (1-b1) * g_1
m_2 = b1*(1-b1)*g_1 + (1-b1)*g_2
m_t = (1-b1) * sum_{k=1}^{t} b1^(t-k) * g_k
```

Take expectations, assuming for the moment that `E[g_k] = E[g]` is stationary:

```
E[m_t] = (1-b1) * E[g] * sum_{k=1}^{t} b1^(t-k)
       = (1-b1) * E[g] * (1 + b1 + b1^2 + ... + b1^(t-1))
       = (1-b1) * E[g] * (1 - b1^t)/(1 - b1)
       = E[g] * (1 - b1^t)
```

So `E[m_t] = (1 - beta1^t) * E[g]`. The estimator is biased *toward zero* by
the factor `(1 - beta1^t)`, purely because it was initialised at zero and has
not yet had time to fill up. Dividing by that factor gives

```
E[m_hat_t] = E[m_t] / (1 - b1^t) = E[g]
```

which is unbiased. Identical algebra applies to `v_t` with `b2` and `g^2`.

The correction matters most early and vanishes fast. At `t = 1`,
`m_1 = 0.1*g` (an order of magnitude too small) and `v_1 = 0.001*g^2` (three
orders too small). Uncorrected, `m/sqrt(v) = 0.1g/(0.0316|g|) = 3.16*sign(g)`,
so the first step would be `3.16*lr` rather than `lr`. Uncorrected Adam does
not merely start slow; it starts wrong by a factor set by the ratio of the two
betas.

**The `t = 1` property: the update is exactly `lr * sign(g)`.** Proof. At
`t = 1`, with `m_0 = v_0 = 0`:

```
m_1     = (1-b1) * g
v_1     = (1-b2) * g^2
m_hat_1 = (1-b1)*g / (1-b1)     = g
v_hat_1 = (1-b2)*g^2 / (1-b2)   = g^2
update  = lr * g / (sqrt(g^2) + eps)
        = lr * g / (|g| + eps)
```

With `eps` negligible relative to `|g|`, `g/|g| = sign(g)`, so
`update = lr * sign(g)`. The bias correction *exactly* cancels the
initialisation shrinkage at `t = 1`, leaving a pure sign step whose magnitude
is the learning rate, independent of gradient magnitude.

Verified against `T.runs.adam.history[0].adam_detail`. For `W1[0][0]`:

```
g       = -3.164
m       = 0.9*0 + 0.1*(-3.164)         = -0.3164
v       = 0.999*0 + 0.001*(-3.164)^2   = 0.010010896
m_hat   = -0.3164 / (1 - 0.9^1)        = -0.3164/0.1        = -3.164
v_hat   = 0.010010896 / (1 - 0.999^1)  = 0.010010896/0.001  = 10.010896
sqrt(v_hat)                            = 3.164
update  = 0.1 * (-3.164)/(3.164 + 1e-8) = -0.0999999996839444
theta   = 0.5 - (-0.0999999996839444)   = 0.5999999996839444
```

Every one of these appears verbatim in `T.runs.adam.history[0].adam_detail[0]`.
The update is `-0.1` to nine significant figures; the deviation
`3.16e-10` is exactly the `eps = 1e-8` in the denominator, which is the whole
of its effect.

The full `t=1` update vector across all 13 parameters, rounded, is
`[-0.1, -0.1, +0.1, +0.1, 0, 0, -0.1, +0.1, 0, -0.1, -0.1, 0, -0.1]` — every
live parameter moves by exactly one learning rate, in the direction of its
gradient's sign, regardless of whether that gradient was `-4.746` or `-1.13`.
The four dead parameters have `g = 0`, so `m = v = 0`, so
`update = lr*0/(0 + eps) = 0`. The `eps` earns its keep here: without it this
would be `0/0`.

Consequence: `T.runs.adam.history[1]` (the state entering step 2, i.e. after
one update) shows Adam at `yhat = 0.93`, `loss = 0.0049`, against SGD's
`yhat = 2.733`, `loss = 3.0037` at the same point. Adam's fixed-magnitude
first step happened to land well. It then overshoots too — the loss trace is
`1.2769, 0.0049, 0.4845, 0.5786, 0.2674, 0.0372, ...` — because `lr = 0.1` is
aggressive for a 13-parameter problem.

But note the crucial structural difference from SGD: the ReLU mask in
`T.runs.adam.history[t].relu_mask` stays `[1, 1, 0]` for all twelve steps.
Adam's step size is bounded. Kingma & Ba give the bound on the effective step
`Delta = lr * m_hat / sqrt(v_hat)`:

```
|Delta| <= lr * (1 - beta1) / sqrt(1 - beta2)   if (1-beta1) > sqrt(1-beta2)
|Delta| <= lr                                   otherwise
```

Here `1 - beta1 = 0.1` and `sqrt(1 - beta2) = sqrt(0.001) = 0.0316`, so the
first case applies and the bound is `0.1 * 0.1 / 0.0316 = 0.316` — about
`3.16 * lr`. That bound is loose: scanning every update in
`T.runs.adam.history[*].adam_detail`, the largest step actually taken over the
whole 12-step run is `0.09999999978929626` (`W1[0][1]` at `t=1`), i.e. one
learning rate.

The point is that a bound exists at all. SGD's step is `lr * g` with no bound,
and a gradient of `-4.746` moved `W1[0][1]` by `0.4746` in one step, which is
what killed the hidden layer. Adam's trust region is what keeps the ReLU mask
at `[1, 1, 0]` for all twelve steps.

**Memory cost: 2 tensors per parameter** (`m` and `v`). In `T.memory.per_tensor`
these appear as `m (Adam)` and `v (Adam)`, 13 elements each — as many optimizer
elements (26) as there are parameters and gradients combined (13 + 13).

### 6.4 Optimizer memory summary

| Optimizer | State tensors per parameter | Bytes/param at fp32 state |
|---|---|---|
| SGD | 0 | 0 |
| SGD + momentum | 1 | 4 |
| Adam / AdamW | 2 | 8 |
| Adafactor | ~0 (factored row+col statistics, `O(n+m)` not `O(n*m)`) | ≈0 |
| 8-bit Adam (Dettmers et al.) | 2, quantized | 2 |

For a 70B model that difference is 0 GB / 280 GB / 560 GB of pure optimizer
state, before any master-weight copy. Choosing Adam is a 560 GB decision.

---

## 7. Floating point and mixed precision

`site/01-memory.html` renders the actual bit patterns from `T.bitviews`.

### 7.1 The three layouts

| Format | Bits | Sign | Exponent | Mantissa (stored) | Bytes |
|---|---|---|---|---|---|
| fp32 (IEEE binary32) | 32 | 1 | 8 | 23 | 4 |
| bf16 (bfloat16) | 16 | 1 | 8 | 7 | 2 |
| fp16 (IEEE binary16) | 16 | 1 | 5 | 10 | 2 |

Verifiable directly in the trace. `T.bitviews[0]` is `W1[0][0] = 0.5`:

```
fp32: 0 01111110 00000000000000000000000     (1+8+23)
bf16: 0 01111110 0000000                     (1+8+7)
fp16: 0 01110    0000000000                  (1+5+10)
```

The fp32 and bf16 exponent fields are *bit-identical* (`01111110`), and not by
coincidence: **bf16 is literally the top 16 bits of fp32.** The conversion
truncates the low half with round-to-nearest-even and does nothing else
(`ground_truth.py:bf16_bits`). fp16's field differs (`01110`) because it is
5 bits with a bias of 15 rather than 8 bits with a bias of 127 — converting
fp32 to fp16 re-encodes the exponent rather than dropping bits.

### 7.2 Range: why bf16 never overflows and fp16 does

The exponent field width sets the dynamic range.

| Format | Exponent bits | Bias | Max finite | Min normal | Min subnormal |
|---|---|---|---|---|---|
| fp32 | 8 | 127 | ~3.40e38 | ~1.18e-38 | ~1.4e-45 |
| bf16 | 8 | 127 | ~3.39e38 | ~1.18e-38 | ~9.2e-41 |
| fp16 | 5 | 15 | 65504 | 6.104e-5 (2^-14) | 5.96e-8 (2^-24) |

**bf16.** Same 8 exponent bits, same bias 127, therefore the same exponent
range as fp32. Every finite fp32 value has an exponent bf16 can also
represent, so casting fp32 → bf16 can never overflow to infinity and never
flushes a normal fp32 input to zero. Only precision is lost: 23 mantissa bits
truncated to 7. That is the whole design argument — trade precision, which
training tolerates, for range, which it does not.

**fp16.** 5 exponent bits, max finite 65504, min normal `2^-14 = 6.1e-5`. Both
ends bite. Gradients in deep networks routinely live at `1e-5` to `1e-8`,
inside fp16's subnormal range or below it — under `5.96e-8` a value flushes to
zero and the parameter silently stops receiving updates. At the other end,
attention logits and unnormalised residuals can exceed 65504 and become `inf`,
which poisons everything downstream via `inf - inf = nan`.

**Loss scaling** is the fix, and it works because differentiation is linear:

```
d(S*L)/dtheta = S * dL/dtheta
```

Every gradient is multiplied by `S`, shifting the distribution up by `log2(S)`
binades into fp16's range; divide by `S` in fp32 before the optimizer step.
Dynamic loss scaling tunes `S` at runtime — raise it every `N` clean steps,
halve it and skip the step on any `inf`/`nan`. bf16 needs none of this, which
is why it became the default the moment hardware supported it.

### 7.3 Precision: machine epsilon

Machine epsilon is the gap between `1.0` and the next representable number,
equal to `2^-p` where `p` is the number of *stored* mantissa bits.

| Format | Stored mantissa bits | eps = 2^-p | Decimal digits (~) |
|---|---|---|---|
| fp32 | 23 | 2^-23 = 1.1920929e-7 | 6.92 |
| bf16 | 7 | 2^-7 = 7.8125e-3 | 2.11 |
| fp16 | 10 | 2^-10 = 9.765625e-4 | 3.01 |

(Decimal digits are `stored_mantissa_bits * log10(2)`, as
`T.formats.*.decimal_digits` computes them — stored bits only, not counting
the implicit leading 1.)

Verified: `1.0 + 2^-7` is representable in bf16 and `1.0 + 2^-8` rounds back
to `1.0`; `1.0 + 2^-10` is representable in fp16 and `1.0 + 2^-11` is not;
`1.0 + 2^-23` is representable in fp32 and `1.0 + 2^-24` is not.

**bf16 carries about two decimal digits.** That is a startling
number the first time you see it. The trace makes it concrete
(`T.bitviews[*].bf16.exact` versus `.value`):

| Value | Exact (fp64) | bf16 | Absolute error |
|---|---|---|---|
| `W1[0][1]` | -0.2 | -0.2001953125 | 1.95e-4 |
| `loss` | 1.2769 | 1.2734375 | 3.46e-3 |
| `dL/dz1[0]` | -1.582 | -1.578125 | 3.88e-3 |
| `learning rate` | 0.1 | 0.10009765625 | 9.77e-5 |

The loss `1.2769` becomes `1.2734375` in bf16 — an error in the *third*
decimal place. You cannot print a bf16 loss and read a fourth significant
figure from it; the figure is not there.

The *unit in the last place* (ULP) at a value `x` is `2^floor(log2 x) * 2^-p`.
For bf16 at `x = 0.5`: `2^-1 * 2^-7 = 2^-8 = 0.00390625`. Half a ULP is
`0.001953125`, and any update smaller than that rounds straight back to `0.5`.

### 7.4 Why an fp32 master copy is necessary — worked

Take `W1[0][0] = 0.5` from `T.init`. Suppose the model is training in bf16 and
late in training its updates have shrunk to `1e-4` per step — a completely
ordinary magnitude for a converging model with a decayed learning rate.

**Naive bf16 in-place update:**

```
step 1:  bf16(0.5 + 1e-4) = bf16(0.5001) = 0.5
```

`0.5001` is `0.0001` above `0.5`. Half a ULP is `0.001953125`. `0.0001` is
19x smaller than half a ULP, so round-to-nearest returns `0.5` exactly. The
update is discarded in its entirety.

And it stays discarded, because the next step starts from `0.5` again:

```
step 2:   bf16(0.5 + 1e-4) = 0.5
step 3:   bf16(0.5 + 1e-4) = 0.5
...
step 100: bf16(0.5 + 1e-4) = 0.5
```

I ran exactly this loop with the same round-to-nearest-even bf16 conversion
that `ground_truth.py:bf16_bits` implements: after 100 iterations the weight
is still **0.5**. Not approximately 0.5 — bit-identical to its initial value.
The parameter has received 100 updates and moved zero distance. This is
sometimes called *stagnation* or *swamping*, and it is silent: no NaN, no
warning, no anomaly in the loss curve, just a parameter that has quietly
stopped learning.

**With an fp32 master copy:**

```
master (fp32) starts at 0.5
step 1:   0.5 + 1e-4 = 0.5001        (representable: fp32 ULP at 0.5 is 2^-24 = 6e-8)
step 2:   0.5002
...
step 100: 0.5100016593933105
cast to bf16 for the next forward pass: 0.51171875
```

The fp32 accumulator has ULP `2^-1 * 2^-23 = 6e-8` at this magnitude —
1600x smaller than the update — so each `1e-4` lands cleanly. After 100 steps
the master weight is `0.51`, and the bf16 copy handed to the forward pass is
`0.51171875`, one bf16 ULP above the true value. The bf16 copy is *imprecise*,
which is fine — it only has to be good enough for one matmul. The master copy
is *accurate*, which is what matters, because it is the thing being
accumulated into.

The general principle: **rounding error in a value used once is harmless;
rounding error in a value accumulated into is fatal.** Update-swamping is
systematic — it always rounds toward the current value — so it compounds
perfectly rather than averaging out.

The same argument makes the Adam moments fp32. They are accumulators
(`m = 0.9*m + 0.1*g`), and `v` additionally holds `g^2`, so a gradient of
`1e-4` puts `v` entries near `1e-8` where bf16 has no useful precision at all.

A mixed-precision Adam step therefore touches five tensors per parameter: bf16
weight, bf16 gradient, fp32 master weight, fp32 `m`, fp32 `v`. That is the
`2 + 2 + 4 + 4 + 4 = 16` bytes of the next section.

---

## 8. The memory taxonomy

Four classes. The site colours them consistently: `weight` blue, `gradient`
orange, `activation` green, `optimizer` magenta.

`T.memory.per_tensor` lists all 20 live tensors for this model, tagged by
class:

| Class | Tensors | Elements |
|---|---|---|
| weight | `W1`(6), `b1`(3), `W2`(3), `b2`(1) | 13 |
| activation | `x`(2), `z1`(3), `a1`(3), `relu_mask`(3), `z2`(1), `loss`(1) | 13 |
| gradient | `dL/dW1`(6), `dL/db1`(3), `dL/dW2`(3), `dL/db2`(1), plus the four transients `dL/dyhat`(1), `dL/dz2`(1), `dL/da1`(3), `dL/dz1`(3) | 21 |
| optimizer | `m`(13), `v`(13) | 26 |

73 elements total for a 13-parameter model (`T.memory.total_elements`). At
fp32 that is 292 bytes (`T.memory.total_bytes_fp32`). The ratio — 5.6 bytes
of *stuff* for every 1 byte of model — is the number that matters, and it
does not improve at scale. That 292 B is a census of everything the step
allocates, not a snapshot: the phase-by-phase high-water mark is
`T.memory.peak_bytes_fp32` = 264 B, over a floor of 156 B.

### 8.1 Lifetimes

| Class | Allocated | Freed | Lives for |
|---|---|---|---|
| Weights | Model construction | Process exit | The entire run |
| Optimizer state | First optimizer step | Process exit | The entire run |
| Gradients | Backward pass (or persistent buffer) | After optimizer step, or zeroed | One step |
| Activations | Forward pass, layer by layer | Backward pass, layer by layer | Part of one step |

Only activations have a lifetime shorter than a step, and only activations
depend on batch size. That is why activations are the class you attack first
when you are out of memory (reduce microbatch, checkpoint, use FlashAttention)
and why the other three set a hard floor that no batch-size change can move.
`site/06-loop.html` shows the four classes rising and falling across one
iteration.

### 8.2 Bytes per parameter

**Stated assumptions**, because this is where "folk numbers" come from and
they disagree with each other:

1. **Master weights are counted under `optimizer`.** They exist only because
   the optimizer needs a higher-precision accumulator. Some accountings put
   them under `weights`, which is why you see both "12 bytes/param of optimizer
   state" and "8 bytes/param" quoted for the same configuration. Neither is
   wrong; they are counting different boxes.
2. **Gradient dtype is stated per row.** PyTorch AMP with bf16 autocast gives
   bf16 gradients; DeepSpeed and many FSDP configurations reduce in fp32.
   A 2-bytes-per-parameter difference worth checking rather than assuming.
3. **Activations are excluded.** They scale with
   `batch x seq x hidden x layers`, not with parameter count, so a
   per-parameter figure for them is meaningless. Budgeted in section 4.2.
4. **1 GB = 1e9 bytes** (decimal), matching GPU spec sheets. An "80 GB" H100
   actually carries 80 GiB — `T.reference_configs.gpus` records it as
   85,899,345,920 bytes, i.e. 85.9 GB decimal — and less than that is usable
   after CUDA context.

| Configuration | Weights | Grads | Optimizer (incl. master) | Total B/param |
|---|---|---|---|---|
| fp32, plain SGD | 4 | 4 | 0 | **8** |
| fp32, SGD + momentum | 4 | 4 | 4 | **12** |
| fp32, Adam | 4 | 4 | 4+4 = 8 | **16** |
| bf16 mixed, Adam, bf16 grads | 2 | 2 | 4 (master) + 4 + 4 = 12 | **16** |
| bf16 mixed, Adam, fp32 grads | 2 | 4 | 12 | **18** |
| bf16 mixed, Adam, bf16 grads + separate fp32 grad accumulator | 2 | 2 + 4 | 12 | **20** |
| bf16 mixed, SGD + momentum, bf16 grads | 2 | 2 | 4 + 4 = 8 | **12** |
| bf16 mixed, 8-bit Adam, bf16 grads | 2 | 2 | 4 + 1 + 1 = 6 | **10** |
| bf16 weights + bf16 optimizer, no master (not recommended — see 7.4) | 2 | 2 | 4 | **8** |

The row that matters in practice is the fourth: **16 bytes per parameter for
mixed-precision Adam.** Note that it is numerically identical to full-fp32
Adam. Mixed precision buys you *speed* (tensor cores) and *activation* memory
(half-width activations, which is the dominant term at scale), not
optimizer-state memory. That is a common and expensive misunderstanding.

---

## 9. The communication primitives

Sections 9 through 14 are the distributed half of this document. Their ground
truth is `assets/data/parallel.json`, generated by `code/parallel_toy.py` under
the same rules as `trace.json`: pure standard-library Python, no numpy, no
torch, no NCCL. It *simulates* four ranks running one training step of a small
MLP under every major parallelism strategy, and for each one records the exact
slice each rank holds, the ordered schedule of collectives with payload sizes,
the ring cost of each, and a numerical proof that the distributed result equals
the single-rank result.

The toy model there is deliberately different from the 2-3-1 network of
sections 1-8, because a 13-parameter model has nothing to shard.
`PARALLEL.meta`:

```
dims       = [4, 8, 8, 8, 1]        4 layers, so a 4-way pipeline is 1 layer/stage
n_layers   = 4
n_params   = 193
batch      = 4                      so 4-way data parallelism is 1 sample/rank
world      = 4
```

Hidden width 8 over world 4 gives exactly two hidden units per rank under
tensor parallelism. Everything divides by four. `PARALLEL.meta.note` says the
quiet part out loud: *real models are not this tidy*.

Why this matters at all: section 8 established 16 bytes per parameter for
mixed-precision Adam. A 70B model is therefore 1.13 TB of static state before a
single activation (the term-by-term arithmetic is in 14.3), against 85.9 GB on
an H100. The model does not fit. Everything below is a different answer to the
question *which of those tensors do I refuse to replicate, and what does the
network cost me for refusing*.

`site/11-collectives.html` animates each primitive on four ranks.

### 9.0 Notation

`N` ranks, numbered `0 .. N-1`. A buffer of `S` elements. Rank `i`'s copy is
`x_i`. When a buffer is sharded, it is cut into `N` contiguous pieces of `S/N`
elements each, and shard `k` is the half-open element range

```
shard(k) = [ k*S/N , (k+1)*S/N )
```

`x[shard(k)]` is that slice. Reductions are elementwise and, unless stated,
sums — every other reduction operator (max, min, product) has identical
communication structure and is irrelevant to training, where the only reduction
that ever appears is a sum of gradients.

`S` is measured in elements throughout; multiply by 2 for bf16 bytes or 4 for
fp32. `PARALLEL` stores both (`elements`, `bytes_bf16`, `bytes_fp32` on every
schedule entry).

### 9.1 The eight collectives

`PARALLEL.collectives` is the authoritative list — nine entries, the eight
collectives below plus `p2p`, which is not a collective at all and is included
because pipeline parallelism runs on it.

| Op | Signature | Ring bytes sent per rank | Adjoint |
|---|---|---|---|
| `broadcast` | one → all | `S` | `reduce` |
| `scatter` | one → all (split) | `(N-1)/N · S` | `gather` |
| `gather` | all → one | `(N-1)/N · S` | `scatter` |
| `all_gather` | all → all (concat) | `(N-1)/N · S` | `reduce_scatter` |
| `reduce` | all → one (sum) | `S` | `broadcast` |
| `reduce_scatter` | all → all (sum + split) | `(N-1)/N · S` | `all_gather` |
| `all_reduce` | all → all (sum) | `2(N-1)/N · S` | itself |
| `all_to_all` | all → all (transpose) | `(N-1)/N · S` | itself |
| `p2p` | one → one | `S` | itself |

Every column of that table is a field in `PARALLEL.collectives[*]`: `op`,
`sig`, `ring_factor`, `inverse`. The cost column is what
`parallel_toy.py:ring_cost` implements, and it is the standard
bandwidth-optimal analysis with latency terms dropped — 9.6 puts them back.

Now each one precisely. "Before" and "after" describe the contents of every
rank's buffer; `r` denotes the root rank where one exists.

**`broadcast`.** One rank's buffer becomes everyone's.

```
before:   rank r  holds  x[0..S)
          rank i  holds  (undefined)          for i != r
after:    rank i  holds  x[0..S)              for all i
```

Used for: sending the initial weights to every replica at startup, so that all
`N` copies of a data-parallel model begin bit-identical. Also every
`torch.distributed` barrier-with-payload idiom.

**`scatter`.** One rank's buffer is cut into `N` pieces and dealt out.

```
before:   rank r  holds  x[0..S)
after:    rank i  holds  x[shard(i)]          S/N elements
```

Used for: handing each data-parallel rank its slice of the batch. In practice
the dataloader shards by index rather than scattering tensors, which is the
same operation performed on the storage side instead of the wire.

**`gather`.** The inverse of `scatter`: pieces are collected onto one rank in
rank order.

```
before:   rank i  holds  y_i                  S/N elements
after:    rank r  holds  concat(y_0, y_1, ..., y_{N-1})
          rank i  unchanged                   for i != r
```

Used for: pulling a full tensor back to rank 0 for checkpointing, logging, or
evaluation.

**`all_gather`.** `gather` followed by `broadcast`, but implemented as one
operation.

```
before:   rank i  holds  y_i                  S/N elements
after:    rank i  holds  concat(y_0, ..., y_{N-1})   for all i    S elements
```

Used for: FSDP and ZeRO-3 re-materialising a full layer's weights immediately
before the layer runs. This is the single most important collective in modern
training — it is the one that lets a rank hold `1/N` of a parameter and still
execute a dense matmul against all of it.

**`reduce`.** Elementwise sum across ranks, answer on one rank.

```
before:   rank i  holds  x_i[0..S)
after:    rank r  holds  ( sum_j x_j )[0..S)
```

Used for: collecting a scalar loss for logging. Rare in the hot path — if
every rank needs the result you want `all_reduce`, and if only one rank needs
it you usually only need it once per hundred steps.

**`reduce_scatter`.** Sum across ranks, then keep only your own shard of the
sum. The two halves of an `all_reduce`, minus the second half.

```
before:   rank i  holds  x_i[0..S)                       S elements
after:    rank i  holds  ( sum_j x_j )[shard(i)]         S/N elements
```

Used for: ZeRO-2, ZeRO-3 and FSDP reducing gradients. Each rank ends up with
the fully-summed gradient for exactly the parameters it will update, and never
allocates a full-size gradient buffer at all. Section 11.3 is about why this
is free.

**`all_reduce`.** Sum across ranks, answer on every rank.

```
before:   rank i  holds  x_i[0..S)
after:    rank i  holds  ( sum_j x_j )[0..S)     for all i
```

Used for: DDP averaging gradients; tensor parallelism summing the partial
products of a row-parallel matmul. It is the workhorse, it is its own adjoint,
and it costs exactly twice what the others do.

**`all_to_all`.** A distributed transpose. Each rank's buffer is viewed as `N`
chunks; rank `i` sends its chunk `j` to rank `j`.

```
before:   rank i  holds  in_i[0..N)            N chunks of S/N
after:    rank i  holds  out_i[0..N)  where  out_i[j] = in_j[i]
```

Read the last line as a matrix transpose of the `(rank, chunk)` index pair —
that is literally all it is. Used for: mixture-of-experts routing (tokens are
sent to whichever rank hosts their chosen expert, and the results sent back),
and for switching between sequence-sharded and hidden-sharded layouts in
sequence parallelism.

**`p2p`** (`send`/`recv`). Not a collective: two ranks, one hop, `S` elements,
no group, no synchronisation with anyone else. Used for: pipeline parallelism
handing a boundary activation to the next stage and the corresponding gradient
back to the previous one. Its irrelevance to the collectives table is the
reason it survives on slow links (section 13.5).

### 9.2 The decomposition identities

Four identities matter, and all four are one line of index algebra.

```
all_reduce   =  reduce_scatter  ;  all_gather
all_reduce   =  reduce          ;  broadcast
all_gather   =  gather          ;  broadcast
reduce_scatter = reduce         ;  scatter
```

Proof of the first, which is the one everything in section 11 hangs on. After
`reduce_scatter`, rank `i` holds `(sum_j x_j)[shard(i)]`. An `all_gather` over
those buffers concatenates shards `0 .. N-1` in rank order and leaves the
result on every rank:

```
concat_k ( (sum_j x_j)[shard(k)] )  =  (sum_j x_j)[0..S)
```

because the shards are a contiguous partition of `[0..S)`. Every rank now
holds the full sum, which is the definition of `all_reduce`. ∎

The cost decomposition follows immediately and agrees with the table:

```
(N-1)/N · S     reduce_scatter
+ (N-1)/N · S   all_gather
= 2(N-1)/N · S  all_reduce
```

This is not merely an accounting curiosity. It is *how NCCL actually
implements ring all-reduce*, and it is the reason ZeRO stage 2 costs nothing
(11.3): if you were going to throw away the `all_gather` half anyway, stop
paying for it.

### 9.3 Inverses and adjoints

The `inverse` field in `PARALLEL.collectives` pairs the operations up:

```
broadcast      <->  reduce
scatter        <->  gather
all_gather     <->  reduce_scatter
all_reduce     <->  all_reduce      (self-inverse)
all_to_all     <->  all_to_all      (self-inverse)
```

The word "inverse" is loose and worth tightening, because the precise version
is what gives you the backward pass for free. These operations are *linear
maps* on the concatenated distributed buffer. `broadcast` is not invertible —
it maps `S` numbers to `N·S` numbers and cannot be undone. What `reduce` is,
is its **adjoint**: the operator whose matrix is the transpose.

Write `broadcast` as a matrix `B` stacking `N` copies of the identity:

```
B = [ I ; I ; ... ; I ]        (N·S x S)
B^T = [ I , I , ... , I ]      (S x N·S)
```

`B^T` applied to a stack of `N` buffers sums them — that is `reduce`. Same
argument for `scatter`/`gather`: `scatter` is a selection matrix `P` whose rows
are a permutation of the identity rows, and `P^T` scatters them back, which is
`gather`. `all_gather` is `[I;I;...;I]` composed with the shard selection; its
transpose sums and selects, which is `reduce_scatter`. `all_reduce` is the
symmetric matrix `1_N ⊗ I` (an `N x N` block matrix of identities), and a
symmetric matrix is its own transpose — hence self-adjoint.

### 9.4 The conjugate pairs: forward and backward

Reverse-mode autodiff of a linear map is multiplication by its adjoint
(section 3 — every `backward` is a vector-Jacobian product, and for a linear
op the Jacobian is the op itself). So the table above *is* the table of
backward passes:

| Forward op | Backward op | Why |
|---|---|---|
| `broadcast` | `reduce` | `x` reached the loss through `N` copies; the partials sum |
| `scatter` | `gather` | each element went to exactly one rank; collect them |
| `gather` | `scatter` | each element came from exactly one rank; send it home |
| `all_gather` | `reduce_scatter` | every rank used every shard; sum, then keep your own |
| `reduce` | `broadcast` | the sum's derivative w.r.t. each term is 1 |
| `reduce_scatter` | `all_gather` | inverse of the row above |
| `all_reduce` | `all_reduce` | self-adjoint |
| `all_to_all` | `all_to_all` | a transpose is its own transpose |
| `p2p(i→j)` | `p2p(j→i)` | one hop, reversed |

Take the `broadcast` row explicitly, because it is the one that produces
Megatron's `f` operator in 12.4. Forward, rank `i` receives a copy `y_i = x`.
The loss depends on `x` through all `N` copies, so by the multivariable chain
rule

```
dL/dx = sum_{i=0}^{N-1} dL/dy_i
```

which is exactly `reduce`. If every rank needs `dL/dx` (it does — each rank
continues its own backward pass), the `reduce` becomes an `all_reduce`.

The `all_reduce` row is the one people find surprising. Forward,
`y_i = sum_j x_j` on every rank. Then `dy_i/dx_j = I` for every pair, so

```
dL/dx_j = sum_i dL/dy_i · I
```

but in the usual arrangement each rank contributes exactly one partial and
consumes exactly one output, and the sum collapses to `dL/dx_j = dL/dy_j` —
an *identity*, no communication. That asymmetry is real and is the entire
content of Megatron's `f`/`g` pair: one of them communicates in forward and is
free in backward, the other is free in forward and communicates in backward.
Details in 12.4.

### 9.5 The ring all-reduce, derived

Arrange the `N` ranks in a logical ring: rank `i` sends only to rank `i+1 mod
N` and receives only from rank `i-1 mod N`. Cut every rank's buffer into `N`
chunks of `S/N` elements.

**Phase 1 — reduce-scatter, `N-1` steps.** At step `t` (from `0`), rank `i`
sends chunk `(i - t) mod N` of its accumulator to rank `i+1`, and adds the
chunk it receives into its own accumulator. Chunk `k` therefore travels
`k+1, k+2, ...` around the ring, accumulating one rank's contribution at each
hop. After `N-1` steps, every chunk has visited all `N` ranks exactly once, so
each chunk is fully summed — and it is sitting on exactly one rank. Rank `i`
ends holding the fully-reduced chunk `(i+1) mod N`. That is `reduce_scatter`.

- Steps: `N-1`.
- Bytes sent per rank per step: `S/N`.
- Total per rank: `(N-1)/N · S`. ✓ (matches the table)

**Phase 2 — all-gather, `N-1` steps.** Same ring, same chunk size, no
arithmetic. At each step every rank forwards the chunk it most recently
obtained. After `N-1` steps every chunk has visited every rank, so every rank
holds the whole reduced buffer.

- Steps: `N-1`.
- Bytes sent per rank per step: `S/N`.
- Total per rank: `(N-1)/N · S`.

**Total.**

```
sent per rank  =  (N-1)/N · S  +  (N-1)/N · S  =  2(N-1)/N · S
```

The limit is the point:

```
lim_{N -> inf}  2(N-1)/N · S  =  2S
```

**The bytes each GPU sends do not grow with the world size.** They approach
`2S` and stop. Doubling the cluster does not double any rank's traffic; it
only adds a hop, which costs latency, not bandwidth. This single fact is why
data parallelism scales to tens of thousands of ranks, and why the naive
alternative — every rank sends its whole buffer to every other rank — does
not, since that costs `(N-1)·S` per rank and grows without bound.

`PARALLEL.ring.table` tabulates the factor:

| World `N` | `all_reduce` factor `2(N-1)/N` | `all_gather` factor `(N-1)/N` |
|---|---|---|
| 2 | 1.0 | 0.5 |
| 4 | 1.5 | 0.75 |
| 8 | 1.75 | 0.875 |
| 16 | 1.875 | 0.9375 |
| 32 | 1.9375 | 0.9688 |
| 64 | 1.9688 | 0.9844 |
| 128 | 1.9844 | 0.9922 |
| 256 | 1.9922 | 0.9961 |

By `N = 64` the factor is within 1.6% of its asymptote. Every value in that
table is a literal field in `PARALLEL.ring.table`.

Two footnotes on the model. First, this counts *bytes sent*, not time: the
ring is only bandwidth-optimal if every link is equally fast, which is false
the moment the ring crosses a node boundary, and NCCL's answer is to build
separate intra-node and inter-node rings and compose them hierarchically.
Second, `parallel_toy.py:ring_cost` charges `broadcast`, `reduce` and `p2p`
the full `S` and does not model the reduction arithmetic itself; the docstring
says so.

### 9.6 Ring versus tree: bandwidth-optimal versus latency-optimal

Drop the assumption that only bytes matter. The standard α-β cost model —
*stated as literature* (Hockney; Thakur, Rabenseifner & Gropp) — charges a
fixed `α` seconds per message hop and `β` seconds per byte:

```
T = (number of sequential hops) · α  +  (bytes sent per rank) · β
```

**Ring.** `2(N-1)` sequential steps, `2(N-1)/N · S` bytes:

```
T_ring = 2(N-1)·α  +  2(N-1)/N · S · β
```

The bandwidth term is optimal — no algorithm can move fewer than
`2(N-1)/N · S` bytes per rank for an all-reduce, a known lower bound. The
latency term is terrible: it is *linear* in `N`. At `N = 64` that is 126
sequential hops.

**Tree.** Reduce up a binary tree to the root, then broadcast back down.
Depth `log2(N)` each way:

```
T_tree = 2·log2(N)·α  +  2·log2(N)·S·β
```

The latency term is optimal — `2·log2(64) = 12` hops against the ring's 126,
a 10.5x reduction. The bandwidth term is now the bad one: each of the
`2·log2(N)` phases pushes the *whole* buffer across a link, so bytes grow
logarithmically instead of saturating at `2S`.

**When each wins.** Set them equal:

```
2(N-1)α + 2(N-1)/N · S · β  =  2 log2(N) · (α + S·β)

S*  =  2α ( N - 1 - log2 N )  /  ( β ( 2 log2 N - 2(N-1)/N ) )
```

The crossover `S*` scales like `α/β` — the *latency-bandwidth product* of the
link, in bytes. With **assumed** constants `α = 5 µs` and
`β = 1/(50 GB/s)` (representative of a 400 Gb/s InfiniBand fabric; neither
number is in the data, both are stated to make the shape concrete):

| `N` | Crossover `S*` | 1 KB | 1 MB | 100 MB | 1 GB |
|---|---|---|---|---|---|
| 8 | 0.47 MB | tree | ring | ring | ring |
| 64 | 2.84 MB | tree | tree | ring | ring |
| 256 | 8.82 MB | tree | tree | ring | ring |

So: **tree for small messages and large worlds, ring for large messages.**
Which is exactly what NCCL does — it selects between `Ring`, `Tree` and
`CollNet` algorithms per call based on message size and topology, and
`NCCL_ALGO` exists to override the choice when the heuristic is wrong. NCCL's
actual tree is a *double binary tree* (two complementary trees, each rank
interior in one and a leaf in the other) which recovers roughly half the
bandwidth penalty; that refinement is literature, not derived here.

The practical consequence, which reappears in 11.4 and 13.3: **splitting one
big collective into many small ones is free in bandwidth and expensive in
latency.** The `(N-1)/N` factor is linear in payload, so four `all_gather`s of
`S/4` send exactly as many bytes as one `all_gather` of `S` — but they cost
four times the `α`. That is the entire tension in FSDP's wrapping policy.

---

## 10. Data parallelism

`site/12-data-parallel.html` runs the four-rank version step by step.

### 10.1 The statement

Data parallelism replicates the model and shards the batch. It is correct
because of a single fact about the loss:

> **The full-batch gradient is the mean of the per-shard gradients, provided
> the shards are the same size.**

Nothing about the network, the architecture, or the optimizer enters. It is a
property of the loss being a *mean over examples* and of differentiation being
*linear*.

### 10.2 The proof

Let the loss over a batch of `B` examples be

```
L(theta) = (1/B) * sum_{b=1}^{B} l( f(x_b; theta), y_b )
```

Write `l_b` for the `b`-th term. Partition `{1..B}` into `N` disjoint shards
`S_0 .. S_{N-1}` with `|S_i| = B/N` each, and give shard `i` to rank `i`. Rank
`i` computes its *own* mean loss over the examples it holds:

```
L_i(theta) = (N/B) * sum_{b in S_i} l_b
```

(the normaliser is `1/|S_i| = N/B`, which is what any framework's
`reduction='mean'` does locally — it has no idea the other ranks exist). Then

```
(1/N) * sum_i L_i  =  (1/N) * sum_i (N/B) * sum_{b in S_i} l_b
                   =  (1/B) * sum_i sum_{b in S_i} l_b
                   =  (1/B) * sum_{b=1}^{B} l_b            (the shards partition the batch)
                   =  L
```

Differentiate with respect to `theta`. The derivative of a finite sum is the
sum of the derivatives and constants pull out — that is the whole of it:

```
grad L  =  grad [ (1/N) sum_i L_i ]  =  (1/N) * sum_i grad L_i
```

∎ One line, because linearity of differentiation is the only ingredient.

Two conditions are doing real work here and both are violated in practice
often enough to matter:

1. **Equal shard sizes.** If `|S_i| = B_i` differ, then
   `L = sum_i (B_i/B) * L_i` and the correct reduction is the *weighted* mean
   `sum_i (B_i/B) grad L_i`, not the uniform one. A uniform average silently
   up-weights the examples on ranks that got fewer of them. This is why
   `DistributedSampler` pads the last batch rather than shortening it, and why
   `drop_last=True` is the safe default. With ragged sequence lengths and
   token-level losses the same bug appears one level down: averaging per-rank
   *token* means over-weights the rank with the short sequences, and the fix
   is to reduce `(sum_of_token_losses, token_count)` and divide after.
2. **Identical parameters on every rank.** The proof evaluates every `grad L_i`
   at the *same* `theta`. This is why replicas are broadcast from rank 0 at
   startup and why any source of divergence — a rank-dependent random seed
   feeding dropout without a synchronised generator, a non-deterministic
   reduction order in a fused kernel, a rank that skipped an update because of
   a local `inf` check — is a correctness bug and not a rounding issue. It
   compounds: once two replicas differ, nothing in DDP ever pulls them back
   together.

### 10.3 Numerical verification

`parallel_toy.py:run_data_parallel` runs the four ranks with one sample each,
averages their gradients elementwise, and compares against the single-rank
full-batch backward pass. `PARALLEL.strategies.ddp.verify`:

```
grad_max_err = 5.551115123125783e-17
loss_max_err = 0.0
passed       = true
claim        = "Averaging the per-GPU gradients reproduces the full-batch
                gradient exactly."
```

The per-rank losses, from `PARALLEL.strategies.ddp.local[*].loss`:

| Rank | Sample | Local loss |
|---|---|---|
| 0 | 0 | 0.7319845177758788 |
| 1 | 1 | 0.26949076562499996 |
| 2 | 2 | 2.072808075625 |
| 3 | 3 | 0.46306323765625 |

Their mean is `0.8843366491705321`, which is `PARALLEL.model.loss` to the last
bit — hence `loss_max_err = 0.0`. Note the spread: rank 2's local loss is 7.7x
rank 1's. Per-rank loss is a nearly useless diagnostic at small microbatch;
only the reduced value means anything.

The gradient error, `5.551115123125783e-17`, is exactly `2^-54`. That is not
an approximation that happens to be small — it is the residue of *one*
floating-point rounding at magnitude ~1 (a half-ULP at `0.5`, a quarter-ULP at
`1.0`). The two computations perform the same multiplications and additions in
a different order, and one addition rounded differently. There is no
accumulating drift, no algorithmic approximation, and no tolerance being
squinted at: the distributed gradient and the single-rank gradient are the
same number.

The same is true of the two other exact-equivalence claims in the file:
`PARALLEL.strategies.tp.verify.forward_max_err = 6.938893903907228e-18`, which
is `2^-57` (section 12.5), and the ZeRO and pipeline strategies, which report
`forward_max_err = 0.0` because they do not change the arithmetic at all —
only where numbers are stored and when.

### 10.4 Why the reduction is a mean and not a sum

`all_reduce` sums. The theorem needs a mean. Something must divide by `N`.

Frameworks do this by pre-scaling: each rank multiplies its local gradient by
`1/N` *before* the collective, so the summed result is already the mean.
(PyTorch DDP does exactly this; the alternative — `all_reduce` then divide — is
one extra full pass over the gradient buffer for identical results, and in
fp16 the pre-scale also keeps the intermediate sum from overflowing.)

Getting this wrong is a learning-rate bug wearing a correctness costume. If
you `all_reduce` with `SUM` and forget the division, every gradient is `N`
times too large, so with `N = 64` you are training at `64 * lr`. The loss curve
does not report "your reduction is wrong"; it reports divergence, and the
usual response is to lower the learning rate, which papers over it.

**The learning-rate connection, stated carefully.** Sharding the batch across
`N` ranks and averaging gives you a gradient of the *same expected value* as
the single-rank gradient but with `N` times less variance:

```
E[ g_batch ]   = E[ g ]              (unbiased either way)
Var[ g_batch ] = Var[ g_single ] / N      (independent examples)
```

The gradient did not get bigger; it got *quieter*. So the total batch size
grew by `N` and the step size did not. Two heuristics exist to spend the
reduced variance — both are **literature, not derived here**:

- **Linear scaling** (Goyal et al., 2017): multiply `lr` by `N` when you
  multiply the batch by `N`, with a warm-up over the first few epochs to
  survive the early large steps. Justified by the observation that `k`
  successive small-batch steps and one `k`-times-larger step land in nearly
  the same place when the gradient does not change much over those `k` steps.
- **Square-root scaling** (Krizhevsky, and Adam-family practice): multiply
  `lr` by `sqrt(N)`, which keeps the *ratio* of step size to gradient noise
  constant rather than the step-to-gradient ratio.

Which applies depends on the optimizer. Adam's update is scale-invariant
(section 6.3: `m_hat/sqrt(v_hat)` is dimensionless), so it does not see a
gradient that got bigger — it sees one that got less noisy, and the sqrt rule
is the better match. SGD's update is proportional to the gradient, so linear
scaling is the natural one. Neither survives arbitrarily large batches;
past some critical batch size (McCandlish et al., *An Empirical Model of
Large-Batch Training*, 2018 — literature) more examples per step stop buying
proportionally faster convergence, which is the real ceiling on how far pure
data parallelism can take you.

### 10.5 Bucketing and overlap, derived

The all-reduce in `PARALLEL.strategies.ddp.schedule` is one entry:

```
step 1   all_reduce   "all gradients"   193 elements   sent/GPU 289.5
```

`289.5 = 1.5 × 193 = 2(N-1)/N · S` at `N = 4`. One collective, after backward,
covering every parameter. Done naively, that means the GPU computes for
`T_b` seconds, then communicates for `T_c` seconds, and the step takes
`T_b + T_c`. On a 70B model at `N = 64` the gradient buffer is 141.2 GB at
bf16 (`70.6e9 × 2`, using `T.reference_configs`), so `T_c` is *seconds*.
Serialising that is not an option.

But backward produces gradients incrementally, in reverse layer order, and
layer `L`'s weight gradient is final the moment its outer product is done
(section 3, step 6a) — nothing later in the backward pass touches it. So
layer `L`'s all-reduce can be launched immediately, on a side stream, while
the backward pass continues into layer `L-1`.

**The pipeline arithmetic.** Cut the gradient buffer into `K` buckets. Compute
chunk `k` takes `T_b/K`; the matching communication chunk takes `T_c/K` and
cannot start before its compute chunk finishes. Communication is serialised on
one stream, so bucket `k`'s transfer finishes no earlier than

```
finish(k) = k·(T_b/K)  +  (bucket k's own transfer, plus any queued behind it)
```

and the step ends when bucket `K` lands. Taking the maximum over the possible
binding constraints:

```
T_step = max_{k=1..K} [ k·T_b/K  +  (K - k + 1)·T_c/K ]
```

The bracket is linear in `k` with slope `(T_b - T_c)/K`, so the maximum is at
one endpoint:

```
if T_c <= T_b   (compute-bound):     T_step = T_b + T_c/K
if T_c >  T_b   (comm-bound):        T_step = T_b/K + T_c
```

Read the first line. **The exposed communication cost is `T_c/K`, not `T_c`.**
Only the final bucket — the one covering the first layers of the network,
produced last by backward — has nothing left to hide behind. Everything else
overlaps completely. With `K = 25` buckets, 96% of the communication is free.

**Why not `K = ∞`.** Because `T_c` itself depends on `K`. Under the α-β model
of 9.6, `K` buckets of total size `S` cost

```
T_c(K) = K·(2(N-1)·α)  +  2(N-1)/N · S · β
```

— the bandwidth term is unchanged (linearity again), the latency term is paid
`K` times. Substituting into the compute-bound branch:

```
T_step = T_b + 2(N-1)·α + (2(N-1)/N · S · β)/K
```

which decreases in `K`, but only until `T_c(K)` exceeds `T_b` and the
comm-bound branch takes over. The optimum is therefore the largest `K` with
`T_c(K) <= T_b`:

```
K* = floor( ( T_b - 2(N-1)/N · S · β ) / ( 2(N-1)·α ) )
```

and if the numerator is negative there is no `K` that hides the traffic and
you have a bandwidth problem, not a scheduling problem. PyTorch does not solve
this equation; it exposes `bucket_cap_mb`, default 25 MB (framework default,
stated not derived), and lets you tune it.

Three implementation details fall out of the same analysis:

- **Buckets are built in reverse order of the forward pass**, because that is
  the order backward produces gradients. PyTorch rebuilds the bucket order
  after the first iteration once it has observed the true ready order.
- **Buckets are flat contiguous buffers.** One `all_reduce` of 25 MB beats a
  thousand of 25 KB by the `α` argument, and NCCL needs contiguity anyway.
  The flat buffer is *additional* memory on top of the gradients (section
  15.1).
- **Unused parameters break it.** A parameter whose gradient is never produced
  leaves its bucket permanently incomplete, and the step hangs. Hence
  `find_unused_parameters`, which walks the autograd graph to mark them ready
  — and costs a graph traversal per step, which is why it is off by default.

### 10.6 The memory bill is unchanged

This is DDP's defining limitation and it is visible directly in the data.
`PARALLEL.strategies.ddp.memory_per_gpu`:

```
weights                193 elements
gradients              193
optimizer              386        (m and v, one each per parameter)
                      -----
per-GPU model state    772 elements
activations_divisor      4
```

193 is the whole model. Every rank holds all of it, plus a full gradient
buffer, plus full optimizer state: `4 × 193 = 772` elements per rank, exactly
what a single rank would hold running the whole batch alone. The
`per_gpu[*].holds` field spells it out — *"ALL weights (replicated)", "ALL
optimizer state (replicated)"*.

The only thing that shrinks is activations, by the `activations_divisor` of
4, because each rank forwards one sample instead of four. And activations are
the one class that was already the easiest to control (section 4.3).

So: **DDP scales throughput, not capacity.** If the model does not fit on one
device, `N` copies of it fit on `N` devices exactly as badly. The strategies in
sections 11 to 13 all start from this observation.

---

## 11. ZeRO and FSDP

`site/13-zero-fsdp.html` steps through the three stages on four ranks.

DDP replicates three things it does not need to. ZeRO (Rajbhandari et al.,
2020) removes them one at a time, in increasing order of how much
communication the removal costs. The remarkable result — derived in 11.3 — is
that the first two removals cost *nothing*.

### 11.1 The memory ladder, derived

Notation follows the ZeRO paper, so that the formulas below are directly
comparable to its Table 1:

```
Psi  = number of parameters
N    = data-parallel world size
K    = bytes of optimizer state per parameter
```

`K` is a property of the recipe, and `T.memory.recipes` supplies it without
hand-typing. The row *"bf16 Adam, fp32 master, bf16 grads"* has
`components = {weight: 2, gradient: 2, optimizer: 12}` and
`bytes_per_param = 16`, so:

```
weights   2 bytes/param   (bf16)
gradients 2 bytes/param   (bf16)
K        12 bytes/param   (fp32 master 4 + Adam m 4 + Adam v 4)
```

`K = 12` is the canonical value and the one the ZeRO paper uses. Substituting
other rows of `T.memory.recipes` changes only `K`: 8-bit Adam with an fp32
master is `K = 6`, plain fp32 SGD is `K = 0`.

**Baseline (DDP).** Every rank holds everything:

```
M_ddp = 2·Psi + 2·Psi + K·Psi = (4 + K)·Psi
```

At `K = 12`: `16·Psi`, which is section 8.2's figure, arrived at from the
other direction.

**Stage 1, `P_os` — shard the optimizer state.** Rank `i` owns `1/N` of the
master weights and moments and is the only rank that updates that slice.
Gradients and bf16 weights stay replicated:

```
M_1 = 2·Psi + 2·Psi + K·Psi/N
```

**Stage 2, `P_os+g` — also shard the gradients.** Rank `i` only ever updates
parameter slice `i`, so it only ever *reads* gradient slice `i`. The other
`(N-1)/N` of the gradient buffer is, on that rank, write-only garbage:

```
M_2 = 2·Psi + (2·Psi + K·Psi)/N
```

**Stage 3, `P_os+g+p` — also shard the parameters.** Each rank holds `1/N` of
the bf16 weights and `all_gather`s a layer's worth just before using it,
freeing it immediately after:

```
M_3 = (2·Psi + 2·Psi + K·Psi)/N = (4 + K)·Psi/N
```

The progression, written out, is the shape worth memorising:

```
2Psi + 2Psi + K·Psi           ->  2Psi + 2Psi + K·Psi/N
                              ->  2Psi + (2Psi + K·Psi)/N
                              ->  (2Psi + 2Psi + K·Psi)/N
```

**The limits are what decide the strategy.** Let `N -> infinity`:

```
M_1 -> 4·Psi     a hard floor: replicated weights AND replicated gradients
M_2 -> 2·Psi     a hard floor: replicated weights
M_3 -> 0         no floor at all
```

Only stage 3 is asymptotically free. Stages 1 and 2 buy you a constant-factor
reduction and then stop, and for a large enough model the floor alone exceeds
the device. For 70.6B parameters (`T.reference_configs`, Llama 3 70B) the
stage-2 floor is `2 × 70.6e9 = 141.2 GB` — still 1.6 H100s of bf16 weights
before a single gradient or activation. That is the whole argument for
ZeRO-3/FSDP existing.

Numerically, at `Psi = 70.6e9`, `K = 12` (arithmetic performed on
`T.reference_configs.models` and `T.memory.recipes`; GB = 1e9 bytes):

| `N` | DDP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---|---|---|---|
| bytes/param | 16 | `4 + 12/N` | `2 + 14/N` | `16/N` |
| 8 | 1129.6 GB | 388.3 GB | 264.75 GB | 141.2 GB |
| 16 | 1129.6 GB | 335.35 GB | 202.98 GB | 70.6 GB |
| 32 | 1129.6 GB | 308.88 GB | 172.09 GB | 35.3 GB |
| 64 | 1129.6 GB | 295.64 GB | 156.64 GB | **17.65 GB** |

The DDP column is constant, which is section 10.6 in one line. The ZeRO-1 and
ZeRO-2 columns flatten out toward their floors of 282.4 GB and 141.2 GB. Only
the last column keeps falling, and it is the only one that ever fits in an
H100's 85.9 GB.

### 11.2 The toy's ladder

`parallel_toy.py` counts *elements*, not bytes, with one element of weight,
one of gradient, and two of optimizer state (`m` and `v`) per parameter — so
its `K` is 2 elements and its total is `4·Psi` with `Psi = 193`. The
`memory_per_gpu` fields across the six strategies:

| Strategy | weights | gradients | optimizer | per-GPU model state |
|---|---|---|---|---|
| DDP | 193 | 193 | 386 | **772** |
| ZeRO-1 | 193 | 193 | 96 | **482** |
| ZeRO-2 | 193 | 48 | 96 | **337** |
| ZeRO-3 / FSDP | 48 | 48 | 96 | **192** |
| Tensor parallel | 48 | 48 | 96 | **192** |
| Pipeline parallel | 48 | 48 | 96 | **192** |

Every number there is a literal field in
`PARALLEL.strategies.*.memory_per_gpu`; the totals are their sum. The ladder
`772 → 482 → 337 → 192` is a 4.02x reduction on four ranks, against the ideal
4x — and the small discrepancy is honest arithmetic, not an approximation:
`parallel_toy.py` uses Python floor division, and `193 / 4 = 48.25` becomes
`48`, `386 / 4 = 96.5` becomes `96`. The exact ladder is
`772 → 482.5 → 337.75 → 193`, and `193` is precisely `772/4`. ZeRO-3 on four
ranks stores exactly one rank's fair share, which is the definition of the
stage.

Note also that ZeRO-3, tensor parallelism and pipeline parallelism land on the
identical 192 elements. They shard the same total state; they differ entirely
in *which axis* they cut it along and *what the network bill is* — which is
the subject of the next three sections.

### 11.3 Why stage 2 is communication-free relative to stage 1

This is the result that makes ZeRO look like a free lunch, and it is a direct
corollary of the decomposition identity in 9.2.

Start from stage 1 as literally scheduled in
`PARALLEL.strategies.zero1.schedule`:

```
step 1   all_reduce    "all gradients"            193 elem   sent/GPU 289.50
step 2   all_gather    "updated weight shards"    193 elem   sent/GPU 144.75
                                                  total      sent/GPU 434.25
```

The comment in the source is the reasoning: *"Stage 1 still needs every rank to
hold the full gradient, because only the OPTIMIZER STATE is sharded so far."*
Every rank ends the all-reduce holding the complete averaged gradient; each
then updates its own `1/N` slice of the master weights; then the updated bf16
weight shards are all-gathered so that every rank has a full weight copy for
the next forward pass.

Now ask what rank `i` actually *does* with that full gradient buffer. It owns
optimizer state for slice `i` only. It updates slice `i` only. It reads
gradient slice `i` only. The other `(N-1)/N` of the buffer is delivered,
stored, and never read.

Apply 9.2:

```
all_reduce  =  reduce_scatter  ;  all_gather
```

The `reduce_scatter` half is the part that computes the sum and gives rank `i`
its own slice — the part it needs. The `all_gather` half is the part that
hands every rank a full copy — the part it does not. **Stage 2 is stage 1 with
the second half deleted.** That is the entire algorithmic difference.
`PARALLEL.strategies.zero2.schedule`:

```
step 1   reduce_scatter  "gradient shards"          193 elem   sent/GPU 144.75
step 2   all_gather      "updated weight shards"    193 elem   sent/GPU 144.75
                                                    total      sent/GPU 289.50
```

The saving is exactly the deleted half:

```
434.25 - 289.50 = 144.75 = (N-1)/N · Psi = 0.75 × 193
```

and `PARALLEL.strategies.*.comm_vs_ddp` records the ratios: **1.5 for ZeRO-1,
1.0 for ZeRO-2.** Stage 2 sends *less* than stage 1 while storing less than
stage 1. There is no trade. The extra memory saving — `(N-1)/N` of the
gradient buffer, `48` of `193` elements per rank in the toy, 138.99 GB per
rank at 70.6B and `N = 64` — is a strict improvement.

The source comment on the `reduce_scatter` says it precisely: *"it is the same
sum, minus the half that hands everyone a full copy."*

**An honest note on the literature.** The ZeRO paper reports stages 1 and 2 as
having the *same* communication volume as DDP (`2·Psi`), not 1.5x and 1.0x. It
is right, and so is the toy — they are describing two different stage-1
implementations. A production ZeRO-1 also uses `reduce_scatter` for the
gradients (it has the same reason to: it only reads its own slice), which
brings its schedule to `reduce_scatter + all_gather = 2 × (N-1)/N · Psi`, the
same as DDP's single all-reduce and the same as the toy's stage 2.
`parallel_toy.py` schedules stage 1 in the naive full-`all_reduce` form
because that is the form that makes the stage-1-to-stage-2 argument visible: it
shows you the redundant `all_gather` before deleting it. The conclusion is
identical under either accounting — **sharding the gradients costs nothing** —
but the baseline it is measured against differs.

The one real cost of stage 2 is a scheduling constraint, not a bandwidth one:
gradients must be reduce-scattered *as they are produced*, bucket by bucket
during the backward pass, so that a bucket can be reduced and its non-local
portion freed before the next bucket is allocated. If you wait until backward
finishes, you have materialised the full gradient buffer and saved nothing.
ZeRO-2 therefore requires the bucketing machinery of 10.5 for correctness of
the memory claim, where DDP only wanted it for speed.

### 11.4 ZeRO-3's 1.5x, derived

Stage 3 shards the parameters themselves, so a rank no longer *has* the
weights it needs to run a layer. It must fetch them. Per step, per parameter:

1. **`all_gather` for forward.** Before layer `L` runs, its full weights are
   reconstructed from the `N` shards. Volume `Psi` over the step (each
   parameter is gathered exactly once).
2. **Free them.** The gathered copy is released as soon as the layer's forward
   is done — otherwise stage 3 would degenerate into DDP with extra steps.
3. **`all_gather` for backward.** Backward needs the same weights again
   (section 3, step 4: `dL/dinput = W^T @ delta` reads `W`). They were thrown
   away, so they are gathered a *second* time. Volume `Psi`.
4. **`reduce_scatter` the gradients.** Volume `Psi`.

In ring terms, with the `(N-1)/N` factor from 9.5:

```
ZeRO-3   = 3 · (N-1)/N · Psi
DDP      = 2 · (N-1)/N · Psi        (one all_reduce)
ratio    = 3/2 = 1.5
```

`PARALLEL.strategies.zero3.comm_vs_ddp` is `1.5`, computed by the toy as
`total_sent / ring_cost("all_reduce", n_params, N)`, and the schedule shows
where every element goes:

| Phase | Op | Payload | Sent/GPU |
|---|---|---|---|
| forward | `all_gather` W0 | 40 | 30.00 |
| forward | `all_gather` W1 | 72 | 54.00 |
| forward | `all_gather` W2 | 72 | 54.00 |
| forward | `all_gather` W3 | 9 | 6.75 |
| backward | `all_gather` W3 | 9 | 6.75 |
| backward | `all_gather` W2 | 72 | 54.00 |
| backward | `all_gather` W1 | 72 | 54.00 |
| backward | `all_gather` W0 | 40 | 30.00 |
| after backward | `reduce_scatter` grads | 193 | 144.75 |
| | | | **434.25** |

`434.25 / 289.50 = 1.5` exactly. And note that the four forward all-gathers
sum to `30 + 54 + 54 + 6.75 = 144.75`, which is exactly `(N-1)/N × 193` — the
same as one all-gather of the whole model. Splitting a collective by layer
costs nothing in bytes, because `(N-1)/N` is linear in payload. It costs four
times the latency (9.6). That is the whole of FSDP's wrapping-granularity
problem, and it is why the answer is neither "wrap every linear" nor "wrap the
whole model".

**Could you avoid the second all-gather?** Yes: keep the forward-gathered
weights alive until backward. But then every rank holds every parameter for
the duration of the step, which is DDP. The 0.5x extra traffic *is* the
payment for the `(N-1)/N` parameter memory saving; they are the same decision
seen from two sides.

The second cost is structural and worse than the bandwidth: ZeRO-3's
collectives sit on the **critical path of every layer**, not at the end of the
step. DDP's all-reduce can hide behind backward (10.5) because nothing in the
step is waiting for it. ZeRO-3's forward all-gather for layer `L` blocks layer
`L`'s matmul. The only way to hide it is to prefetch — issue layer `L+1`'s
all-gather while layer `L` computes — which requires knowing the execution
order in advance and holding *two* gathered layers at once, partially undoing
the memory saving. This is why ZeRO-3 throughput is so much more sensitive to
network quality than DDP throughput, and why its latency exposure grows with
the number of shard units rather than with the parameter count.

### 11.5 FSDP

PyTorch's FSDP is the ZeRO-3 algorithm with a different organising principle
and a different vocabulary. Worth stating the mapping, because the papers and
the API do not use the same words.

- **The unit of sharding is a *unit* (an `FSDP` module), not a tensor.** All
  parameters inside a unit are flattened and concatenated into a single 1-D
  `FlatParameter`, which is then split into `N` contiguous pieces. Rank `i`
  stores piece `i`. This matters because a transformer block has dozens of
  parameter tensors of wildly different shapes; sharding each one
  independently would produce dozens of tiny collectives per block, and
  section 9.6 says that is the expensive failure mode. Flattening turns them
  into one.
- **The wrapping policy is the tuning knob**, and it is exactly the
  granularity trade-off of 11.4. `transformer_auto_wrap_policy` wrapping one
  block per unit is the standard answer for a reason: a block is large enough
  that the collective is bandwidth-bound rather than latency-bound, and small
  enough that one gathered unit fits comfortably. Wrap too finely and you pay
  `α` per tiny unit; wrap too coarsely and the transiently gathered unit blows
  the memory budget you were trying to reduce — in the limit, wrapping the
  whole model in one unit gathers the entire model and is strictly worse than
  DDP.
- **Prefetching** (`forward_prefetch`, `backward_prefetch`) is the
  critical-path mitigation from 11.4, and it costs one extra gathered unit of
  memory.
- **`HYBRID_SHARD`** shards within a node and replicates across nodes: a
  `world = node_size × n_nodes` factorisation where the expensive per-layer
  all-gathers stay on NVLink and the inter-node traffic reverts to one
  DDP-style all-reduce per step. It exists because pure ZeRO-3 puts
  `3·(N-1)/N·Psi` of latency-sensitive, critical-path traffic on the slow
  fabric, which is the wrong place for it — the same topology argument as
  section 14.2.
- **`CPUOffload`** moves the sharded optimizer state and master weights to host
  memory, trading PCIe bandwidth for HBM. It changes the memory formulas of
  11.1 by removing `K·Psi/N` from the device entirely, at the cost of a
  host-device round trip per optimizer step.

---

## 12. Tensor parallelism

`site/14-tensor-parallel.html` splits the toy's hidden layer four ways and
shows the partial sums arriving.

ZeRO shards *storage* and reassembles the full tensor before every use.
Tensor parallelism never reassembles anything: each rank computes on its own
slice and the slices are combined only where the mathematics forces it. The
question is where that is, and the answer is a short piece of block-matrix
algebra.

### 12.1 Column-parallel: splitting the output dimension

Let `X` be `(R, k)` and `A` be `(k, n)`, so `Y = XA` is `(R, n)`. `R` is the
number of rows — tokens, i.e. batch × sequence — and is written `R` rather
than `B` because `B` is about to be the name of a matrix. Cut `A` into `N`
column blocks:

```
A = [ A_1 | A_2 | ... | A_N ]        each A_i is (k, n/N)
```

Then

```
Y = XA = [ X A_1 | X A_2 | ... | X A_N ]
```

**Proof.** `Y[r][j] = sum_{q=0}^{k-1} X[r][q] · A[q][j]`. The index `j` ranges
over output columns, and every output column `j` belongs to exactly one block
`i = floor(jN/n)`. The sum over `q` never crosses a column boundary — it runs
over the *input* dimension, which is not split. So `Y`'s column block `i`
depends on `A_i` and on nothing else in `A`. ∎

What this means operationally:

- Rank `i` stores only `A_i` — `1/N` of the weight matrix.
- Rank `i` needs all of `X` (the full input is required by every column
  block), so `X` is replicated.
- Rank `i` produces `Y_i = X A_i`, which is `1/N` of the output columns.
- **No communication.** The output is left in a sharded state, which is
  exactly what we want.
- A bias on this layer splits with the columns: `b_i` is the slice of `b`
  matching `A_i`, and each rank adds its own slice locally.

### 12.2 Row-parallel: splitting the contraction dimension

Now let `B` be `(n, m)` and cut it into `N` *row* blocks, with the input
already column-split in the matching way:

```
B = [ B_1 ; B_2 ; ... ; B_N ]        each B_i is (n/N, m)
X = [ X_1 | X_2 | ... | X_N ]        each X_i is (R, n/N)
```

Then

```
Y = XB = sum_{i=1}^{N} X_i B_i
```

**Proof.** `Y[r][j] = sum_{q=0}^{n-1} X[r][q] · B[q][j]`. Split the sum over
`q` at the block boundaries:

```
Y[r][j] = sum_{i} ( sum_{q in block i} X[r][q] · B[q][j] )
        = sum_{i} (X_i B_i)[r][j]
```

∎

The structural difference from 12.1 is everything:

- Rank `i` stores only `B_i` and needs only `X_i` — which it already has, if
  `X` came out of a column-parallel layer. **The shard boundaries line up,
  and no reshuffle is needed.**
- Rank `i` produces `X_i B_i`, which is **full-shape** `(R, m)` but is only a
  *partial sum*. It is not a piece of the answer; it is a summand.
- The partials must be added: **one `all_reduce`**. Not a design choice —
  it is inside the mathematics. `parallel_toy.py` says so in the schedule
  entry: *"Each GPU computed a PARTIAL SUM over its slice of the contraction.
  The pieces must be added to form the real activation, so this all-reduce is
  not optional — it is inside the maths, not around it."*
- A bias on this layer must be added **once, after** the sum. If every rank
  added the full bias to its partial, the reduction would count it `N` times.
  The toy does exactly this — `z = add_bias(summed, bs[Lr])`, with the
  comment *"bias added ONCE, after the sum"* — and it is a classic
  off-by-a-factor-of-`N` bug in hand-written tensor parallelism.

### 12.3 The central result: column-then-row lets a nonlinearity through

Consider the two-matmul sandwich that every transformer MLP is:

```
Z = f(X A) B          f elementwise (GeLU, SwiGLU, ReLU)
```

Split `A` column-wise and `B` row-wise. Then:

```
XA = [ X A_1 | ... | X A_N ]                          (12.1, no communication)
f(XA) = [ f(X A_1) | ... | f(X A_N) ]                 (f is elementwise)
f(XA) B = sum_i f(X A_i) B_i                          (12.2, one all_reduce)
```

The middle line is the whole trick. **`f` is elementwise, so it commutes with
the column blocking**: applying `f` to the concatenation is the same as
concatenating the `f`s, because `f` maps entry `(r, j)` to a function of entry
`(r, j)` alone and never mixes columns. Rank `i` can therefore compute
`f(X A_i)` on its own shard, with no knowledge of any other rank's columns and
no communication.

So the entire block — matmul, nonlinearity, matmul — runs as:

```
rank i:   local matmul  ->  local nonlinearity  ->  local matmul  ->  all_reduce
```

**One collective for two matmuls and an activation.**

Now the counterfactual, which is what makes it a *result* rather than a
coincidence. Suppose you had chosen row-parallel first:

```
XA = sum_i X_i A_i      -- a partial sum, full-shape, on every rank
f(sum_i X_i A_i)  !=  sum_i f(X_i A_i)     for any nonlinear f
```

A nonlinearity does not distribute over a sum. You would have to `all_reduce`
*before* applying `f`, and then `all_reduce` again after the second matmul —
two collectives instead of one, and the first one on a tensor of the wide
inner dimension (`n`, typically `4h` or `3.5h`) instead of the narrow output
dimension.

**The ordering is forced by the nonlinearity.** Column-parallel puts the shard
boundary on an axis the nonlinearity does not touch; row-parallel consumes
that same boundary. Every tensor-parallel transformer implementation is built
out of this one pairing, repeated.

### 12.4 The conjugate operators `f` and `g`

Megatron-LM names the two synchronisation points. They are the
`broadcast`/`reduce` adjoint pair of section 9.4, specialised:

```
f  :  identity in forward,   all_reduce in backward     (block input)
g  :  all_reduce in forward, identity in backward       (block output)
```

**Deriving `f`.** At the block input, `X` is replicated: every rank holds the
same `X` and uses it for its own column block. Forward, this is free — the
tensor is already where it needs to be. Backward, `X` influenced the loss
through all `N` ranks' computations, so by the chain rule

```
dL/dX = sum_{i=1}^{N} (dL/dX)|_from rank i
```

which is a `reduce`; and since every rank must continue its own backward pass
into the previous block, it is an `all_reduce`. Free forward, collective
backward.

**Deriving `g`.** At the block output, `Z = sum_i Z_i`. Forward, that is the
`all_reduce` of 12.2. Backward, `dZ/dZ_i = I` for every `i`, so

```
dL/dZ_i = dL/dZ
```

and every rank already holds `dL/dZ` — it came out of the `all_reduce`'s
adjoint on the downstream side. Collective forward, free backward.

The two are conjugates in the precise sense of 9.3: `f = g*`. The toy records
this directly. `PARALLEL.strategies.tp.schedule` step 1 is an op of type
`"none"` with the note *"No communication. The column split leaves each GPU
with whole output units, and ReLU is elementwise, so it runs on the shard
untouched"*, and step 5 is the matching backward `all_reduce` of `dL/dinput`
with the note *"Forward's no-op becomes backward's all-reduce, and vice versa
— the two are conjugates."*

### 12.5 Numerical verification

`parallel_toy.py:run_tensor_parallel` pairs the toy's four layers into two
column-then-row blocks — layers `(0,1)` and `(2,3)` — shards each 8-wide
hidden layer four ways (`units_per_gpu = 2`), runs each rank's local matmuls,
sums the partials, and compares the result against the single-rank forward
pass. `PARALLEL.strategies.tp.verify`:

```
forward_max_err = 6.938893903907228e-18
passed          = true
claim           = "Summing the per-GPU partial products reproduces the
                   single-GPU forward pass exactly."
```

`6.938893903907228e-18` is exactly `2^-57`. As with DDP in 10.3, this is a
single rounding, not a drift: the four partial products are added in a
different order than the single-rank contraction, and one addition landed on
the other side of a tie. **Tensor parallelism computes the same numbers.**

The block structure, from `PARALLEL.strategies.tp.blocks`:

| Block | Column layer | Row layer | Units/GPU | Hidden shard | `all_reduce` elements |
|---|---|---|---|---|---|
| 0 | 0 | 1 | 2 | `[4, 2]` | 32 (= 4×8) |
| 1 | 2 | 3 | 2 | `[4, 2]` | 4 (= 4×1) |

and the full schedule with ring costs at `N = 4`:

| Step | Op | Phase | Elements | Sent/GPU |
|---|---|---|---|---|
| 1 | `none` | block 0, column layer 0 | 0 | 0 |
| 2 | `all_reduce` | block 0, row layer 1 | 32 | 48.0 |
| 3 | `none` | block 1, column layer 2 | 0 | 0 |
| 4 | `all_reduce` | block 1, row layer 3 | 4 | 6.0 |
| 5 | `all_reduce` | backward block 0 (`dL/dinput`, 4×4) | 16 | 24.0 |
| 6 | `all_reduce` | backward block 1 (`dL/dinput`, 4×8) | 32 | 48.0 |
| | | | | **126.0** |

Two forward all-reduces (one per block, at the row-parallel layer) and two
backward all-reduces (one per block, at the column-parallel layer input) —
`f` and `g` firing once each per block. Total 126 elements sent per rank
against DDP's 289.5, and per-rank model state of 192 elements against DDP's
772 (11.2).

The toy's two blocks map onto one real transformer layer: block 0 plays the
attention sub-layer and block 1 plays the MLP sub-layer. Hence the standard
count — **two all-reduces forward and two backward per transformer layer.**

### 12.6 Applied to a transformer block

A transformer layer is two column-then-row sandwiches, and Megatron shards
both.

**Attention.**

- `W_Q`, `W_K`, `W_V` are **column-parallel, split by head.** This is the
  natural boundary: attention heads are independent — `softmax(QK^T/√d)V` for
  head `h` reads only head `h`'s projections — so a head-aligned column split
  keeps every head whole on one rank and the entire attention computation,
  softmax included, is local. Rank `i` owns `n_heads/N` complete heads.
- `W_O`, the output projection, is **row-parallel**, with its row blocks
  matching the head blocks. Its input is the concatenated head outputs, which
  are already sharded exactly right.
- One `all_reduce` at the end of `W_O` — the `g` operator.

**MLP.**

- The up-projection (and the gate projection, for SwiGLU) is
  **column-parallel**, splitting the `d_ff` axis.
- The nonlinearity is elementwise and runs on the shard untouched — 12.3.
- The down-projection is **row-parallel** over the same `d_ff` axis.
- One `all_reduce` at the end — `g` again.

**Total per layer: 2 all-reduces forward, 2 backward.** For Llama 3 70B's 80
layers (`T.reference_configs`), that is 320 blocking collectives per
microbatch per step.

**A GQA constraint worth naming.** Llama 3 70B has `n_heads = 64` but
`n_kv_heads = 8` (both literal fields in `T.reference_configs`). The
head-aligned column split applies to the KV projections too, so tensor
parallelism divides 8 KV heads, not 64. `TP = 8` lands exactly one KV head per
rank. `TP = 16` does not divide 8, and you must either replicate KV heads
across rank pairs or abandon the head-aligned split. The hardware answer —
8 GPUs per NVLink domain — and the model answer happen to coincide at 8, which
is not a coincidence: both are designed around the other.

**LayerNorm (and RMSNorm) is not shardable this way.** The statistic is a
reduction over the *hidden* dimension:

```
mu    = (1/h) * sum_{j=0}^{h-1} x[j]
sigma = sqrt( (1/h) * sum_j (x[j] - mu)^2 + eps )
```

If `x` is column-split across ranks, that sum crosses every shard boundary,
so a sharded LayerNorm needs a collective (two, for mean and variance, or one
fused pair) *per norm*, of which there are two per layer. Megatron's answer is
not to shard it: the block-boundary tensors — the layer input, the layer
output, the residual stream, the LayerNorms, and dropout — are **replicated**,
each rank holding the full `s × b × h`.

Which means tensor parallelism does not divide all of the activation memory.
Using Korthikanti et al.'s per-layer accounting (*literature*, the same source
as section 4.2), of the `34·s·b·h` bytes per layer, roughly `24·s·b·h` sits
inside the sharded region and divides by `t`, while `10·s·b·h` — the norms,
dropout masks and residual — stays replicated:

```
without sequence parallelism:   s·b·h·( 10 + 24/t + 5·a·s/(h·t) )
with sequence parallelism:      s·b·h·( 34/t + 5·a·s/(h·t) )
```

**Sequence parallelism** (Korthikanti et al., 2022) closes that gap. The
observation is that LayerNorm and dropout are independent *per token*: they
reduce over `h` but never over `s`. So in the regions where the hidden axis
must stay whole, shard the *sequence* axis instead — rank `i` holds tokens
`shard(i)` at full width. The `g` operator's `all_reduce` at the block output
then becomes a `reduce_scatter` (sum, and land in sequence-sharded layout),
and the `f` operator's entry point becomes an `all_gather` (back to
hidden-sharded layout). By 9.2 those two together cost exactly one
`all_reduce`:

```
reduce_scatter + all_gather  =  2·(N-1)/N·S  =  all_reduce
```

**So sequence parallelism is free in bandwidth and removes the replicated
`10·s·b·h`.** It is one of the few genuinely free wins in this document, and
like ZeRO stage 2 it is free for the same reason: an identity from 9.2 that
lets you split a collective and use both halves for something.

### 12.7 The communication volume, and why TP stays inside a node

Per transformer layer, per microbatch, tensor parallelism moves 4 all-reduces
(2 forward, 2 backward) of `s·b·h` elements each. In ring terms, bytes sent
per rank:

```
V_layer = 4 · 2(t-1)/t · s · b · h · (bytes per element)
```

For Llama 3 70B at `t = 8`, using `T.reference_configs` (`h = 8192`,
`s = 8192`, `n_layers = 80`), microbatch `b = 1`, bf16:

```
s·b·h            = 8192 · 1 · 8192      = 67.1e6 elements = 134.2 MB
2(t-1)/t         = 2 · 7/8              = 1.75
V_layer          = 4 · 1.75 · 134.2 MB  = 939.5 MB per layer per microbatch
x 80 layers      =                        75.2 GB per microbatch (whole model)
```

Now the comparison that decides the topology. Take a pipeline stage of 20
layers (the `PP = 4` recipe of 14.3), so each rank moves
`939.5 MB × 20 = 18.79 GB` per microbatch, and compare against its compute for
the same microbatch:

```
FLOPs per microbatch, whole model  = 6 · Psi · (s·b)
                                   = 6 · 70.6e9 · 8192   = 3.47e15
per stage (1/4 of layers)          = 8.68e14
per rank (÷ t = 8)                 = 1.085e14 FLOPs
at 400 TFLOP/s sustained           = 0.271 s          [assumed, see below]
```

Against that 0.271 s of compute:

| Link | Assumed bandwidth | Time for 18.79 GB | Overhead |
|---|---|---|---|
| NVLink 4 (intra-node) | 450 GB/s | 0.042 s | **15%** |
| 400 Gb/s InfiniBand (inter-node) | 50 GB/s | 0.376 s | **139%** |

**Both bandwidth figures and the 400 TFLOP/s sustained rate are assumed
spec-sheet values, not derived and not present in the data.** They are stated
to make the ratio concrete; the ratio itself — roughly 9x between the two
fabrics — is the load-bearing part, and it is robust to any plausible choice.

Fifteen percent is a tax. A hundred and thirty-nine percent is a
reclassification: the GPU spends more than half its time waiting on the
network. And unlike DDP's all-reduce (10.5), **none of this is overlappable**,
because the next matmul consumes the reduced result directly. There is no
independent work to hide it behind.

That is the whole argument for **`TP ≤ node width`.** The collectives are
frequent (4 per layer per microbatch), large (`s·b·h`), synchronous, and on
the critical path, so they must sit on the fastest link in the machine. On an
8-GPU NVLink node, that caps `t` at 8 — which, per the GQA note above, is also
where the model's own structure stops cooperating.

---

## 13. Pipeline parallelism

`site/15-pipeline-parallel.html` animates the grid below filling and draining.

Pipeline parallelism splits the model by *depth*. Stage `p` owns a contiguous
run of layers, holds their weights, gradients and optimizer state, and holds
nothing else. Activations travel forward to stage `p+1` and gradients travel
back to stage `p-1`, both point-to-point.

It is the only strategy in this document that changes *when* work happens
rather than *what* is computed. `PARALLEL.strategies.pp.verify` states it
plainly: *"Pipeline parallelism does not change the maths at all — it reorders
WHEN each layer runs, not what it computes. The single-GPU result is
reproduced by construction."* `forward_max_err = 0.0`, not `2^-54` — there is
not even a reordered summation to round differently.

The cost is not arithmetic. It is idleness.

### 13.1 The bubble, derived

`P` stages, `M` microbatches. Microbatch `m` cannot start on stage `s` until
it has finished on stage `s-1`, and stage `s` cannot start microbatch `m`
until it has finished microbatch `m-1`. Take one microbatch-on-one-stage as
the unit of time (assume the stages are balanced — 13.4 revisits that). Then
the earliest slot at which stage `s` can begin microbatch `m` is

```
start(s, m) = s + m
```

by induction: `start(0, 0) = 0`; each `+1` in `s` waits one slot for the
upstream stage; each `+1` in `m` waits one slot for the same stage's previous
microbatch. The last unit of forward work is `start(P-1, M-1) = M + P - 2`,
finishing at the end of slot `M + P - 2`, so the forward phase occupies

```
M + P - 1 slots
```

Every stage performs exactly `M` units of forward work in that window. So:

```
utilisation = M / (M + P - 1)
bubble      = 1 - M/(M + P - 1) = (P - 1) / (M + P - 1)
```

The backward phase has the identical structure with the dependency direction
reversed, so it contributes the same fraction, and the total is unchanged:
`2M` units of work in `2M + 2(P-1)` slots is again `M/(M+P-1)` busy.

**`M + P - 1` slots for `M` units of work.** That is the whole result. The
`P - 1` extra slots are the pipeline fill and drain, and they are pure loss.

`PARALLEL.strategies.pp` records `bubble_formula = "(P-1)/(M+P-1)"` and, for
`P = 4`, `M = 4`, `bubble_fraction = 0.42857142857142855` — that is `3/7`.

**Checking it against the schedule grid.** `PARALLEL.strategies.pp.grid` is a
`P × (2M + 2(P-1))` occupancy table, `4 × 14` here. Rendered, with `.` for
idle:

```
slot     0  1  2  3  4  5  6  7  8  9 10 11 12 13
stage 0 F0 F1 F2 F3  .  .  .  .  .  . B0 B1 B2 B3
stage 1  . F0 F1 F2 F3  .  .  .  . B0 B1 B2 B3  .
stage 2  .  .  F0 F1 F2 F3  .  . B0 B1 B2 B3  .  .
stage 3  .  .  .  F0 F1 F2 F3 B0 B1 B2 B3  .  .  .
```

Every row has 8 occupied slots (4 forwards + 4 backwards) out of 14:

```
busy   = 8/14  = 0.5714...
idle   = 6/14  = 0.42857142857142855   = the bubble_fraction field
```

The derived formula and the simulated grid agree to the last digit, which they
must, since `(P-1)/(M+P-1) = 2(P-1)/(2M+2(P-1))`.

How bad it is, as a function of `M/P` (arithmetic on the formula):

| `P` | `M = P` | `M = 4P` | `M = 8P` | `M = 16P` |
|---|---|---|---|---|
| 4 | 42.9% | 15.8% | 8.6% | 4.5% |
| 8 | 46.7% | 17.9% | 9.9% | 5.2% |
| 16 | 48.4% | 19.0% | 10.5% | 5.5% |

Two readings. **The bubble depends on `M/P`, not on `M` or `P` alone** — for
large `M`, `(P-1)/(M+P-1) ≈ (P-1)/M`. And **`M = P` is always about 50%
wasted**, which is why the naive "one microbatch per stage" configuration is
never used.

The catch is that `M` is not free. The global batch is
`DP × M × microbatch_size`, so driving the bubble down by raising `M` inflates
the global batch, and section 10.4's critical-batch-size ceiling is waiting at
the other end. Pipeline depth is bounded above by what your convergence
budget will tolerate in batch size.

### 13.2 GPipe versus 1F1B: same bubble, different memory

The grid above is the **GPipe** order: all `M` forwards, then all `M`
backwards. (`parallel_toy.py:build_pipeline_grid` is docstring'd as 1F1B, but
the grid it actually emits is the GPipe order — stage 0 runs `F0..F3` in slots
0-3 and does not touch `B0` until slot 10. The bubble is identical either way,
which is exactly the point of this subsection; only the memory differs.)

**GPipe peak activation memory.** Microbatch `m`'s activations are created
during its forward pass and consumed during its backward pass. Under GPipe,
*no* backward happens until *every* forward has. So at the moment the last
forward completes, stage `s` is holding the saved activations for all `M`
microbatches of its own layers:

```
peak_GPipe(s) = M · A_s          A_s = activations for one microbatch through stage s's layers
              = O(M)
```

Memory grows linearly in the very quantity you must increase to shrink the
bubble. That is a direct conflict: `M = 8P` gives a 10% bubble and 8P
microbatches of live activations.

**1F1B peak activation memory.** PipeDream's schedule keeps the same
dependency structure but reorders: after a warm-up of `P - 1 - s` forwards,
stage `s` alternates one forward and one backward for the rest of the run,
then drains. The number of microbatches simultaneously *in flight* at stage
`s` — forward done, backward not yet — is

```
in_flight(s) = P - s
```

Stage 0 is the worst case at `P`, stage `P-1` at 1 (it backwards a microbatch
immediately after forwarding it). So:

```
peak_1F1B(s) = (P - s) · A_s  <=  P · A_s  =  O(P)
```

**Independent of `M`.** You can raise `M` to 8P to kill the bubble and the
activation memory does not move. The ratio is

```
peak_GPipe / peak_1F1B = M / P
```

which at `M = 4P` is a 4x memory reduction for zero change in throughput,
zero change in the bubble, and zero change in the arithmetic. This is why
every production pipeline implementation uses 1F1B or a descendant of it, and
why GPipe survives mainly as the clean model in which the bubble is derived.

The asymmetry `in_flight(s) = P - s` also means stage 0 is the memory
bottleneck: it holds `P` microbatches while the last stage holds one. Some
implementations exploit this by giving stage 0 fewer layers.

### 13.3 Interleaving: virtual stages

The bubble `(P-1)/(M+P-1)` has `P-1` in the numerator because fill and drain
cost one *stage-time* each. Interleaved 1F1B (Narayanan et al., 2021) attacks
that constant.

Give each device `v` non-contiguous *virtual stages* — chunks of layers — so
there are `v·P` chunks in total and device `d` owns chunks
`d, d+P, d+2P, ...`. Each chunk is `1/v` of a stage's work, so a chunk-time is
`1/v` of a stage-time. The fill still costs `v·P - 1` chunk-slots, but each is
`v` times cheaper, and there are `v·M` chunk-units of work:

```
bubble_interleaved = (P - 1) / ( v·M + P - 1 )
```

The bubble falls by roughly the factor `v`. For the `P = 4`, `M = 16`
configuration of 14.3 (arithmetic on the formula):

```
v = 1:   3 / (16 + 3)  = 15.8%
v = 2:   3 / (32 + 3)  =  8.6%
v = 4:   3 / (64 + 3)  =  4.5%
```

What it costs: every chunk boundary is a device boundary, so the number of
point-to-point messages per microbatch multiplies by `v`, each one carrying
the same `s·b·h` payload. Section 13.5 shows there is a lot of headroom for
that, which is why the trade is usually worth taking. It also complicates the
schedule considerably and interacts badly with anything that assumes
contiguous layer ownership.

### 13.4 Load imbalance

The derivation in 13.1 assumed every stage takes the same time. Nothing
enforces that, and the slowest stage sets the clock for all of them — a
pipeline runs at the rate of its slowest stage, so a 20% overloaded stage
costs 20% *everywhere*, on top of the bubble.

The toy shows the problem in miniature. `PARALLEL.strategies.pp.stages`:

| Stage | Layer | Shape | Params |
|---|---|---|---|
| 0 | 0 | `[8, 4]` | 40 |
| 1 | 1 | `[8, 8]` | 72 |
| 2 | 2 | `[8, 8]` | 72 |
| 3 | 3 | `[1, 8]` | 9 |

Total 193, so an even split would be 48.25 per stage. What the stages actually
hold ranges from 9 to 72 — an **8x spread**. (`memory_per_gpu.weights` for the
`pp` strategy reports `48`, the even division; that is the *ideal*, not what
any stage holds. The `stages` array is the truth.)

In a real transformer the middle stages are genuinely uniform — a stack of
identical blocks — and the imbalance lives at the two ends:

- **Stage 0** carries the token embedding. For Llama 3 70B,
  `vocab × d_model = 128256 × 8192 = 1.051e9` parameters (arithmetic on
  `T.reference_configs`), 1.5% of the model's 70.6e9.
- **Stage `P-1`** carries the LM head, another `1.051e9` parameters if
  untied, *and* produces the logits: `s × vocab = 8192 × 128256 = 1.051e9`
  elements per microbatch, which is **2.1 GB at bf16** for a single
  microbatch of one sequence — larger than any activation anywhere else in
  the model, and it must be held for the softmax and the cross-entropy
  backward.

So the last stage is disproportionately heavy in activation memory and the
first in nothing much, while 1F1B's `in_flight(s) = P - s` makes the *first*
stage disproportionately heavy in microbatch count. The standard fixes:
give stage 0 and stage `P-1` fewer transformer layers than the middle stages;
shard the vocabulary across tensor-parallel ranks and compute the cross-entropy
in the sharded space (Megatron's vocab-parallel cross-entropy, which reduces a
scalar per token instead of gathering the logits); and chunk the loss
computation over the sequence.

### 13.5 Why pipeline parallelism tolerates slow links

Every other strategy here communicates with collectives. Pipeline parallelism
communicates with `send`/`recv` between adjacent stages, and that difference
is worth more than it sounds.

`PARALLEL.strategies.pp.schedule` — six entries, all `p2p`, all with
`world: 2`:

| Step | Op | Phase | Tensor | Elements | Sent/GPU |
|---|---|---|---|---|---|
| 1-3 | `p2p` | forward | activations `(1×8)` | 8 each | 8 each |
| 4-6 | `p2p` | backward | `dL/dinput` `(1×8)` | 8 each | 8 each |
| | | | | | **48 total** |

48 elements per rank against tensor parallelism's 126 and DDP's 289.5, for the
same model and the same step.

Three properties, each of which matters independently:

1. **The payload is `s·b·h`, independent of the parameter count.** A boundary
   activation does not grow when you make the model wider in `d_ff`, deeper
   inside a stage, or larger in vocabulary. Every other strategy's volume
   scales with `Psi` or with `s·b·h` *per layer*; pipeline's scales with
   `s·b·h` *per stage boundary*, of which there are `P-1`.
2. **No collective.** Two ranks, one hop, no group synchronisation, no
   `(N-1)/N` factor, no `α·log N` or `α·N` latency term. A `p2p` on a bad link
   is slow; a collective on a bad link is slow *for everyone in the group*.
3. **It overlaps with other stages' compute by construction.** While stage `s`
   is sending to stage `s+1`, stages `s+2 .. P-1` are computing on earlier
   microbatches. The pipeline is itself the overlap mechanism.

The 70B arithmetic, using `T.reference_configs` (`h = 8192`, `s = 8192`) with
`b = 1` and bf16:

```
one boundary tensor       = s·b·h·2 bytes = 134.2 MB
per rank per microbatch   = one forward send + one backward send = 268.4 MB
per rank per step (M=16)  = 4.29 GB
at an assumed 50 GB/s     = 0.086 s
against ~4.3 s of step compute (14.3)   ->  2%
```

Two percent, on the slowest link in the machine. Compare tensor parallelism's
139% on the same link (12.7). **This is why the 3D layout puts pipeline
parallelism across nodes**: it is the only axis whose communication is small
enough, rare enough, and asynchronous enough to survive there.

(With sequence parallelism the boundary tensor is itself sequence-sharded, so
the payload drops by a further factor of `t` — 0.537 GB per rank per step at
`t = 8`. Pipeline traffic is not the problem.)

---

## 14. Composing them

`site/17-3d-parallelism.html` lets you factor a world size and see the
resulting per-GPU budget.

### 14.1 The factorisation

The three axes are orthogonal, in the precise sense that each cuts a different
index of the problem:

```
tensor parallel     cuts the HIDDEN dimension of each weight matrix
pipeline parallel   cuts the LAYER index
data parallel       cuts the BATCH index
```

Nothing prevents cutting all three at once, and the world size factors:

```
world = TP × PP × DP
```

Each rank gets a coordinate `(t, p, d)` with `0 <= t < TP`, `0 <= p < PP`,
`0 <= d < DP`, and — with `TP` innermost, which is the layout every framework
uses and 14.2 justifies — the flat rank id is

```
rank = d·(PP·TP) + p·TP + t

t = rank mod TP
p = (rank div TP) mod PP
d = rank div (TP·PP)
```

Three communicator groups are built from those coordinates. Each rank belongs
to exactly one of each:

| Group | Members | Size | Carries |
|---|---|---|---|
| TP group | all ranks sharing `(p, d)` | `TP` | 4 `all_reduce` per layer per microbatch |
| PP group | all ranks sharing `(t, d)` | `PP` | `p2p` per microbatch per boundary |
| DP group | all ranks sharing `(t, p)` | `DP` | 1 gradient `all_reduce` (or RS+AG) per step |

The consistency check that catches most bugs: the three group sizes multiply
to the world size, every rank appears in exactly one group of each type, and
the TP group is contiguous in rank id (so it maps onto one node).

`parallel_toy.py` simulates each axis in isolation; the composition is
arithmetic on top and **is not simulated in `parallel.json`.** The numbers in
14.3 are derived from `T.reference_configs` and `T.memory.recipes` plus the
formulas of sections 11-13, and I say where each comes from.

### 14.2 The topology argument

Why `TP` innermost, `DP` outermost, `PP` across nodes? Three properties
decide it, and they point the same way.

| Axis | Volume per GPU | Frequency | On critical path? | Overlappable? | Pattern |
|---|---|---|---|---|---|
| TP | `4·2(t-1)/t·s·b·h` **per layer per microbatch** | very high | yes | **no** | collective |
| PP | `2·s·b·h` per microbatch per boundary | medium | partly | yes, by construction | `p2p` |
| DP | `2(N-1)/N·Psi/(TP·PP)` **once per step** | lowest | no | yes, by bucketing | collective |

**TP goes innermost, on NVLink.** Its traffic is the largest by an order of
magnitude (75.2 GB per microbatch for a 70B model at `t = 8`, versus 4.3 GB
per *step* for pipeline), it fires four times per layer, and it is
fundamentally non-overlappable: the `all_reduce` at the end of the MLP block
produces the input to the next LayerNorm, and there is no independent work to
interleave. The only lever left is to make the link fast. 12.7's arithmetic:
15% overhead on NVLink, 139% on InfiniBand.

**DP goes outermost, on whatever is left.** Its traffic is large in absolute
terms but it happens *once per step*, it is not on the critical path of any
layer, and 10.5 showed the exposed fraction is `1/K` with bucketing. A
gradient all-reduce that takes 88 ms on a slow fabric costs nearly nothing if
it hides behind 4 seconds of backward pass. DP is the axis that tolerates the
worst link, so it is the one that gets stretched across the most switch hops.

**PP goes across nodes.** It has the smallest payload of the three, it is
point-to-point rather than collective, and the pipeline structure already
provides the overlap. 13.5's arithmetic: 2% overhead on the slow fabric. It is
also the axis with the *fewest* messages: `P-1` boundaries, not `n_layers`
collectives.

**A corollary about ZeRO-3/FSDP.** Its per-layer all-gathers have TP's
critical-path profile with DP's group membership — the worst of both. Running
pure ZeRO-3 across a slow inter-node fabric puts `3·(N-1)/N·Psi` of
latency-sensitive, non-overlappable traffic in exactly the wrong place. That
is what `HYBRID_SHARD` (11.5) exists to fix: shard within the node, replicate
across it, so the frequent collectives stay fast and the inter-node traffic
reverts to one DDP-style all-reduce per step.

The general rule, worth extracting: **sort your parallelism axes by
communication intensity and your network links by bandwidth, then match them
in order.**

### 14.3 A worked recipe: 70B on 64 H100s

Model and device from `T.reference_configs` (`_source`: *"published model
architecture parameters; external to this model, quoted not derived"*):

```
Llama 3 70B    params 70.6e9   d_model 8192   n_layers 80
               n_heads 64      n_kv_heads 8   seq 8192   vocab 128256
H100 80GB      hbm_bytes 85,899,345,920  =  85.9 GB decimal  =  80 GiB
```

Memory recipe from `T.memory.recipes`, row *"bf16 Adam, fp32 master, bf16
grads"*: `weight 2 + gradient 2 + optimizer 12 = 16` bytes per parameter.

**Step 0 — the problem.** (`site/10-scaling.html` runs this budget
interactively against any row of `T.memory.recipes` and any device in
`T.reference_configs.gpus`.)

```
bf16 weights          70.6e9 × 2  =  141.2 GB
bf16 gradients        70.6e9 × 2  =  141.2 GB
fp32 master weights   70.6e9 × 4  =  282.4 GB
fp32 Adam m           70.6e9 × 4  =  282.4 GB
fp32 Adam v           70.6e9 × 4  =  282.4 GB
                                     ---------
total static state    70.6e9 × 16 = 1129.6 GB = 1.13 TB

64 H100s             64 × 85.9 GB = 5497.6 GB total HBM
1129.6 / 85.9 = 13.1  ->  14 GPUs just to hold the static state, with zero
                          bytes for activations
```

(The older, rounder `70e9` gives 1120 GB; `T.reference_configs` says 70.6e9
and that is what is used from here on.)

**Step 1 — factor the world.**

```
64 = TP 8 × PP 4 × DP 2
```

`TP = 8` because that is the NVLink domain width and also `n_kv_heads`
(12.6). `PP = 4` gives `80/4 = 20` layers per stage, an exact division.
`DP = 2` is what is left; ZeRO stage 1 is applied along it, sharding the
optimizer state across the 2 data-parallel replicas.

**Step 2 — model state per GPU.**

Weights, gradients and optimizer state are sharded by `TP × PP = 32`:

```
params per GPU    = 70.6e9 / 32          = 2.206e9
bf16 weights      = 2.206e9 × 2          =  4.41 GB
bf16 gradients    = 2.206e9 × 2          =  4.41 GB
optimizer (K=12)  = 2.206e9 × 12         = 26.48 GB
  with ZeRO-1 over DP=2:  ÷ 2            = 13.24 GB
                                            -------
model state per GPU                      = 22.06 GB
```

Sanity check against 11.1: the ZeRO-1 formula `2Psi + 2Psi + K·Psi/N` applied
to the per-GPU parameter count `2.206e9` with `N = DP = 2` gives
`(2 + 2 + 6) × 2.206e9 = 22.06 GB`. ✓

Without ZeRO-1 it would be 35.3 GB — sharding the optimizer state along the
one remaining axis is worth 13.2 GB per GPU for zero extra communication
(11.3).

**Step 3 — activation memory per GPU.**

Using Korthikanti et al.'s per-layer formula with tensor and sequence
parallelism (*literature*, section 12.6), and assuming FlashAttention so the
`5·a·s/(h·t)` attention-matrix term vanishes (section 4.2):

```
s·b·h                         = 8192 · 1 · 8192      = 67.1e6
per layer, no parallelism     = 34 · 67.1e6          =  2.28 GB
per layer, ÷ TP = 8           =                         0.285 GB
× 20 layers per stage         =                         5.70 GB  per microbatch
× P = 4 microbatches in flight (1F1B, stage 0, 13.2)  = 22.82 GB
```

**Step 4 — the budget.**

```
model state           22.06 GB
activations           22.82 GB
                      --------
                      44.88 GB   of 85.9 GB  ->  52% utilised
```

41 GB of headroom for allocator fragmentation, CUDA context, NCCL buffers, the
gathered-parameter prefetch buffer, and kernel workspace — all of which are
real and none of which are in the arithmetic above (section 15.1). Fifty
percent headroom is roughly what a configuration needs to actually run.

**Step 5 — the batch and the bubble.**

```
global batch  = DP × M × microbatch = 2 × 16 × 1 = 32 sequences
              = 32 × 8192 = 262,144 tokens per step
bubble (13.1) = (P-1)/(M+P-1) = 3/19 = 15.8%
  interleaved v=2 (13.3)  = 3/35 = 8.6%
```

**Step 6 — the communication, all three axes.**

Compute first, so the comparisons mean something. Using the standard
`6·Psi·tokens` estimate for forward+backward FLOPs (*literature*) and an
**assumed** 400 TFLOP/s sustained per H100:

```
FLOPs per step  = 6 × 70.6e9 × 262,144    = 1.11e17
64 GPUs × 400e12                          = 2.56e16 FLOP/s
step time                                 ≈ 4.34 s
```

| Axis | Per GPU per step | Link | Assumed BW | Time | % of step |
|---|---|---|---|---|---|
| TP | `18.79 GB × 16 microbatches` = 300.6 GB | NVLink | 450 GB/s | 0.67 s | 15% |
| PP | `2 × 134.2 MB × 16` = 4.29 GB | InfiniBand | 50 GB/s | 0.086 s | 2% |
| DP | `2(N-1)/N × 4.41 GB` at `N=2` = 4.41 GB | InfiniBand | 50 GB/s | 0.088 s | 2% |

TP dominates by two orders of magnitude in volume, which is the entire
justification for putting it on the fastest link and nowhere else. PP and DP
traffic are comparable to each other and both nearly free; both are
overlappable, TP is not.

**Every bandwidth and FLOP-rate figure in Step 6 is an assumed spec-sheet
value, not present in either data file.** The volumes are derived: from
`T.reference_configs` dimensions, the ring factors of 9.5, and the per-layer
counts of 12.7 and 13.5.

**Step 7 — what to change if it does not fit.** In rough order of what to
reach for:

1. Raise `PP` — cheapest communication, but the bubble grows and stage
   imbalance worsens.
2. Raise `DP` and move to ZeRO-2 or ZeRO-3 — memory falls as `1/N` with no
   floor at stage 3, at 1.5x traffic on the axis that tolerates it best.
3. Turn on activation recomputation at block boundaries — `+33%` step time for
   most of the activation memory (section 4.3).
4. Raise `TP` past the node width — last resort, because 12.7 says the
   overhead goes from 15% to over 100% the moment a TP group crosses a node.

### 14.4 Axes not simulated here

Three further factorisations exist. All three are **outside
`parallel.json`** — nothing below is simulated or numerically verified by
`parallel_toy.py`, and I flag it rather than smuggling it in beside the
checked numbers.

- **Sequence parallelism (SP).** Not really a fourth axis: it is a
  modification *of* tensor parallelism that shards the block-boundary tensors
  along the sequence axis in the regions where the hidden axis must stay
  whole (12.6). Same group, same degree `t`, same bandwidth (`reduce_scatter +
  all_gather = all_reduce`, by 9.2), and it removes the replicated
  `10·s·b·h` per layer. Essentially always on when TP is on.
- **Context parallelism (CP), also called sequence parallelism in a different
  sense.** A genuine fourth axis that shards the sequence dimension across
  ranks *through the attention computation*, which requires exchanging keys
  and values between ranks — ring attention and its relatives. It is what
  makes 128K-token contexts trainable, and its communication is
  attention-shaped rather than matmul-shaped: `O(s·b·h)` per ring step, `N`
  steps, overlappable with the attention compute.
- **Expert parallelism (EP).** For mixture-of-experts models, the experts of
  one MoE layer are distributed across ranks and each token is routed to its
  chosen expert(s). The communication primitive is `all_to_all` (9.1) —
  tokens out to the ranks holding their experts, results back — twice per MoE
  layer. It is the only place in mainstream training where `all_to_all` is on
  the critical path, and its cost depends on the *routing distribution*, so it
  is load-imbalanced by nature in a way none of the other axes are. Capacity
  factors, token dropping and auxiliary load-balancing losses all exist to
  manage that imbalance.

A frontier MoE training job may therefore factor its world as
`TP × CP × PP × DP × EP`, with the expert dimension overlaid on the data
dimension. The ordering principle of 14.2 is unchanged: sort the axes by
communication intensity, sort the links by bandwidth, and match them up.

---

## 15. Caveats and further reading

### 15.1 What these estimates omit

Every byte figure in sections 8, 11 and 14 is a **first-order lower bound**.
Real allocations exceed them, sometimes substantially. In rough order of size:

- **Allocator fragmentation.** PyTorch's caching allocator pools
  freed-but-not-returned blocks. Variable sequence lengths and microbatch
  shapes fragment it, and `memory_allocated()` can sit well below
  `memory_reserved()` while an allocation still fails. Typically 5-20%;
  occasionally much worse. FSDP and ZeRO-3 make this worse, not better,
  because gathered parameter buffers are allocated and freed at layer
  granularity on every step.
- **CUDA context.** Roughly 0.3-1 GB per process before a single tensor is
  allocated — driver structures, module images, the default stream. Multiply
  by processes per node.
- **Communication buffers.** NCCL allocates internal buffers per
  communicator, and a 3D-parallel job builds *three* communicators per rank
  (14.1). DDP/FSDP gradient bucketing allocates flat contiguous buffers
  *additional* to the gradients themselves; ZeRO-3 prefetch needs a second
  gathered parameter buffer in flight while the first is in use (11.4).
- **Kernel workspace.** cuBLAS split-k reductions, cuDNN algorithm
  workspaces, fused-attention scratch, and temporaries from any non-fused
  elementwise chain. Transient, but they count against the peak.
- **The peak is not the sum.** Peak memory is the max over time of live
  bytes, not the total ever allocated. Getting it right needs a timeline, not
  a table — which is what `site/06-loop.html` is for.

The **communication** estimates carry their own omissions, all of them in the
optimistic direction:

- **The α-β model is a fiction.** It assumes one message at a time on a
  contention-free homogeneous link. A real all-reduce crossing a leaf-spine
  fabric contends with every other rank's traffic, and the effective
  bandwidth depends on the oversubscription ratio, the routing, and whether
  the job's ranks landed on the same leaf switch.
- **The ring assumes a homogeneous ring.** As soon as it crosses a node
  boundary, one link in the ring is 9x slower than the rest and sets the pace.
  NCCL's real answer is hierarchical — separate intra-node and inter-node
  rings, composed — which changes the constants in 9.5 without changing its
  conclusion.
- **Sustained bandwidth is not peak bandwidth.** Every bandwidth number in
  section 12.7 and 14.3 is a spec-sheet figure, explicitly flagged where used.
  Achieved bus bandwidth for a well-tuned NCCL all-reduce is typically 60-80%
  of it, and much less for small messages.
- **Nothing here models the tail.** A single slow rank — thermal throttling, a
  bad link, an unlucky page fault — stalls every collective it participates in,
  and with three communicator groups per rank the blast radius is large.
  Straggler effects are absent from every formula above and are frequently the
  dominant term in practice.

### 15.2 What these models omit

The 2-3-1 network of sections 1-8 honestly demonstrates the chain rule, the
shape rule, the activation-lifetime problem, and the optimizer-state question.
It does not exercise batching (a sum over an axis), normalisation layers
(cross-element dependencies in backward, and more saved than just the input),
attention (quadratic in sequence, and the reason FlashAttention exists),
dropout (its mask must be saved), or weight tying (accumulation into a shared
gradient buffer). Each adds terms; none changes the structure derived in
section 3.

The four-rank simulation of sections 9-14 proves the algebra and nothing else.
Specifically, `parallel_toy.py`:

- **Has no network.** It computes payload sizes and applies closed-form ring
  costs; it never measures a transfer, never models latency, congestion,
  overlap or straggling. Every timing claim in sections 12.7, 13.5 and 14.3 is
  arithmetic on assumed bandwidths, flagged where it appears.
- **Uses floor division for shard sizes**, so `193/4` appears as `48` and the
  memory ladder reads `772 / 482 / 337 / 192` rather than the exact
  `772 / 482.5 / 337.75 / 193` (11.2).
- **Schedules ZeRO-1 in the naive full-`all_reduce` form**, which makes it
  1.5x DDP rather than the literature's 1.0x. The reason and the reconciliation
  are in 11.3.
- **Emits a GPipe schedule from a function named for 1F1B** (13.2). The bubble
  is identical; the memory is not, and the difference is derived rather than
  simulated.
- **Simulates each axis in isolation.** There is no 3D-parallel run; section
  14 is arithmetic composed on top of the verified single-axis results.
- **Has no sequence, context or expert parallelism** (14.4), no activation
  recomputation, no mixed precision (it computes in Python floats throughout,
  which is fp64), and no optimizer step — it verifies the *gradients*, not the
  updates.

What it does prove, numerically and per-strategy, is the only thing that
actually needs proving: that every one of these strategies computes the same
numbers as one GPU would.

### 15.3 Papers

Roughly this order.

**Foundations.**

- **Adam** — Kingma & Ba, *Adam: A Method for Stochastic Optimization*
  (ICLR 2015). Section 3 has the bias-correction derivation reproduced above.
  Then Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (AdamW).
- **Mixed precision** — Micikevicius et al., *Mixed Precision Training*
  (ICLR 2018). The fp32-master-copy and loss-scaling arguments.
- **Gradient checkpointing** — Chen, Xu, Zhang & Guestrin, *Training Deep Nets
  with Sublinear Memory Cost* (2016). The `sqrt(n)` result.
- **FlashAttention** — Dao, Fu, Ermon, Rudra & Ré (NeurIPS 2022) and Dao,
  *FlashAttention-2* (2023). Why the `5*a*s/h` term can be deleted.
- **Optimizer state compression** — Shazeer & Stern, *Adafactor* (2018);
  Dettmers et al., *8-bit Optimizers via Block-wise Quantization* (2022).

**Collectives.**

- **Ring all-reduce** — the bandwidth-optimal ring is folk-standard in HPC
  (it is the "recursive halving and doubling" family analysed by Thakur,
  Rabenseifner & Gropp, *Optimization of Collective Communication Operations in
  MPICH*, 2005). Its arrival in deep learning is Gibiansky's Baidu Research
  write-up, *Bringing HPC Techniques to Deep Learning* (2017), and its
  productisation is Sergeev & Del Balso, *Horovod: fast and easy distributed
  deep learning in TensorFlow* (2018). Source of the `2(N-1)/N · S` result
  derived in 9.5.

**Sharding the state.**

- **ZeRO** — Rajbhandari, Rajbhandari, Ruwase & He, *ZeRO: Memory
  Optimizations Toward Training Trillion Parameter Models* (SC 2020). Table 1
  is the per-stage memory accounting derived in 11.1; section 7 is the
  communication analysis behind 11.3 and 11.4. Then *ZeRO-Infinity* (2021) for
  the offload hierarchy.
- **PyTorch FSDP** — Zhao et al., *PyTorch FSDP: Experiences on Scaling Fully
  Sharded Data Parallel* (VLDB 2023). The engineering as distinct from the
  algorithm: `FlatParameter`, wrapping policies, prefetch, `HYBRID_SHARD`
  (11.5).

**Sharding the model.**

- **Megatron-LM tensor parallelism** — Shoeybi, Patwary, Puri, LeGresley,
  Casper & Catanzaro, *Megatron-LM: Training Multi-Billion Parameter Language
  Models Using Model Parallelism* (2019). The column-then-row split of 12.3
  and the `f`/`g` conjugate operators of 12.4.
- **Reducing activation recomputation** — Korthikanti, Casper, Lym, McAfee,
  Andersch, Shoeybi & Catanzaro, *Reducing Activation Recomputation in Large
  Transformer Models* (2022). Source of the `s*b*h*(34 + 5*a*s/h)` formula in
  section 4.2, of its tensor- and sequence-parallel refinements in 12.6, and of
  sequence parallelism itself.

**Sharding the depth.**

- **GPipe** — Huang, Cheng, Bapna, Firat, Chen, Chen, Lee, Ngiam, Le, Wu &
  Chen, *GPipe: Efficient Training of Giant Neural Networks using Pipeline
  Parallelism* (NeurIPS 2019). The bubble analysis of 13.1 and the `O(M)`
  activation peak of 13.2.
- **PipeDream / 1F1B** — Narayanan, Harlap, Phanishayee, Seshadri, Devanur,
  Ganger, Gibbons & Zaharia, *PipeDream: Generalized Pipeline Parallelism for
  DNN Training* (SOSP 2019), and Narayanan, Phanishayee, Shi, Chen & Zaharia,
  *Memory-Efficient Pipeline-Parallel DNN Training* (ICML 2021) for the
  flush-based 1F1B used in practice. The `O(P)` activation peak of 13.2.

**Composing them.**

- **3D parallelism** — Narayanan, Shoeybi, Casper, LeGresley, Patwary,
  Korthikanti et al., *Efficient Large-Scale Language Model Training on GPU
  Clusters Using Megatron-LM* (SC 2021). The interleaved pipeline schedule of
  13.3, the `TP × PP × DP` topology argument of 14.2, and the empirical
  scaling study that justifies the ordering.
- **Large-batch training** — Goyal et al., *Accurate, Large Minibatch SGD*
  (2017) for linear learning-rate scaling and warm-up; McCandlish, Kaplan,
  Amodei et al., *An Empirical Model of Large-Batch Training* (2018) for the
  critical batch size that bounds how far data parallelism goes (10.4).

### 15.4 Reproducing everything here

```
python3 code/ground_truth.py      # sections 1-8
python3 code/parallel_toy.py      # sections 9-14
```

`ground_truth.py` prints the forward pass, both weight gradients, the
gradient-check result, and the twelve-step loss trajectory for all three
optimizers, then writes `assets/data/trace.json` and `assets/data/trace.js`.
It exits non-zero if the gradient check fails, so a trace that exists is one
whose calculus has been verified against finite differences. Change `LR` at
the top and the entire site — and sections 2, 3, 5 and 6 of this document —
changes with it.

`parallel_toy.py` prints the per-strategy equivalence table, the per-GPU
memory and collective counts, then writes `assets/data/parallel.json` and
`assets/data/parallel.js`. It raises `SystemExit("EQUIVALENCE PROOF FAILED")`
if any strategy's result diverges from the single-GPU baseline, so a
`parallel.json` that exists is one in which every claim of sections 10 to 13
has been checked numerically. Change `DIMS`, `WORLD` or `BATCH` at the top and
the ladders, schedules and bubble fractions all move together.
