# Build contract for nn-internals

Read this before writing any slide or page.

This file used to contain only the sections below "Absolute rules" — the
palette, the data rule, the HTML skeleton. Those specify how to be
**consistent**. They said nothing about what to build, and the result was
consistent tables: 157 `el("table")` calls in the deck, 198 on the site, 37% of
site sections carrying no visual at all, and eight animation primitives called
zero times. That was not the authors' fault. It was this file's.

Everything above "Absolute rules" is the part that was missing.

---

# What you are building

An **uber visualization guide to neural network and LLM internals.** The
subject is *mechanism* — what actually happens, shown happening. Not a
reference manual with diagrams attached.

## The one rule

> **If a reader can only learn it by reading numbers off a grid, it is not
> built yet.**

Every table is a failure to visualise something. Before you write one, say out
loud why the thing has no shape, no motion, no before/after, and no spatial
arrangement. Almost nothing survives that question.

Things that look like tables and are not:

| looks like | actually is | draw it as |
|---|---|---|
| per-GPU values | one buffer, cut | the buffer, partition lines dropping in (`viz.cut`) |
| before / after columns | a transition | one object animating between two states |
| a value per step | a trajectory | a line the reader scrubs |
| who sends what | routing | chunks travelling between boxes (`viz.flow`) |
| a quantised value | a landing | a dot leaving the real value, arriving on a level (`viz.snap`) |
| rows summing to zero | an invariant | bars visibly cancelling |
| "4 of 20 tiles rescale" | occupancy | a grid with 4 cells lit (`viz.timeline`) |
| accept / reject list | a rollback | the chain, with everything downstream greying (`viz.chain`) |
| a cache growing | accumulation | one row landing per step (`viz.grow`) |
| compute vs bandwidth | a position | a dot on the roofline (`viz.roofline`) |

## Use the toy models. All of them, element by element.

The point of a 4×4 matrix is that the reader can watch **every element** move.
Never draw a schematic of an abstract matrix when the real one with its real 16
numbers is in the data and can be animated.

## The canonical shape for anything distributed

Every parallelism strategy owes the reader one animation of this shape, on the
real toy tensor:

```
the real tensor  →  the cut  →  what each GPU now holds
                 →  the collective firing, chunks travelling
                 →  the result reassembling  →  it equals the 1-GPU answer
```

Collectives are taught on one page and then, everywhere else, merely *named*.
Do not name a collective. Fire it, on a tensor the reader can see, and let them
watch the bytes move.

**Worked example — tensor parallelism, done right.** Not: a table of which
matrix gets which cut, plus a paragraph about partial sums. Instead: the real
4×4 `W` on screen; a partition line drops between columns 1 and 2; the halves
tint and label themselves GPU 0 / GPU 1; the input broadcasts and the reader
watches it arrive twice; each GPU produces half the output. Then the *second*
matrix is cut by rows, and each GPU now holds a complete-shaped but numerically
**wrong** partial — show both, show that neither is the answer. Then the
all-reduce fires, chunks travel, the partials add, and the result matches the
single-GPU reference. `max |difference| = 0` appears once, small.

The reader should then be able to answer, unprompted: why is the pairing always
column-then-row, and why is there exactly one collective and not two?

## Flow

A chapter is an argument, not a list of true statements about a topic.

Test: read only the section headings. Do they argue, or enumerate?

Shape that works — the problem → the naive thing failing → the fix → the fix on
the toy → it equals the reference → what it costs → what it does *not* fix,
which is the next chapter.

If a section's content would be unchanged had you deleted the section before
it, one of them is misplaced.

## Numbers, in proportion

Delete any number no sentence refers to and no picture needs. A slide averaging
48 numeric values has roughly 40 too many.

Verification is why this project is trustworthy and it is currently eating the
reader's attention: 31 slides and sections are majority PASS-table. **One tick
per claim.** A surprising worked case earns its space; twelve rows reading `ok`
do not.

## The animation kit

`assets/js/viz.js` — all eight expose `.step(i)` and `.reset()` so a `Stepper`
or a slide fragment drives them.

```js
NN.viz.grow      // a buffer gaining a row per step
NN.viz.roofline  // log-log, bandwidth-bound region shaded and named
NN.viz.timeline  // slot / stage / phase occupancy; .mark .play .step
NN.viz.flow      // labelled chunks travelling between boxes; hops take a weight
NN.viz.snap      // values leaving the real line, landing on a grid
NN.viz.chain     // accept / reject, with rollback discarding downstream
NN.viz.cut       // a matrix with an animated partition; .show .isolate
NN.viz.heat      // heatmap, hatched AND labelled masking; reveals row by row
```

Reach for these before writing a private helper. If none fits, say so
explicitly rather than falling back to a table.

---

## Absolute rules

1. **Never type a model number by hand.** Every value comes from `NN.T`
   (the object in `assets/data/trace.js`, generated by `code/ground_truth.py`).
   If you need a number that isn't in the trace, derive it in JS from what is.
   Prose may quote a value only if the same value is also rendered from `NN.T`
   nearby.
2. **Zero build, zero network.** No npm, no CDN, no `fetch()`, no imports.
   Plain `<script>` tags only. The page must work opened as `file://`.
3. **Colour never carries meaning alone.** Every tensor gets a text label and
   the class glyph (via `NN.chip` / `NN.tensor`). The four-hue palette sits in
   the 6–8 CVD band in light mode; the labels are the required mitigation.
4. **Four classes, four hues, no exceptions:** `weight` blue, `gradient`
   orange, `activation` green, `optimizer` magenta. Do not introduce a fifth
   tensor hue. `--accent` is the neutral for everything non-tensor.
5. **No dual-axis charts.** Two scales → two charts.
6. Use CSS custom properties (`var(--weight)`), never raw hex.

## Page skeleton — copy this verbatim

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NN Internals · 0X Title</title>
<link rel="stylesheet" href="../assets/css/design.css">
</head>
<body>

<main class="wrap">
  <div class="pagehead">
    <div class="eyebrow">Part 0X</div>
    <h1>Title</h1>
    <p class="lede">One or two sentences on what this page proves.</p>
  </div>

  <section class="block" id="...">
    ...
  </section>

  <div id="pagenav"></div>
</main>

<footer class="site"><div class="wrap"><div id="sitefoot"></div></div></footer>

<script src="../assets/data/trace.js"></script>
<script src="../assets/js/ui.js"></script>
<script>
(function () {
  const T = NN.T;
  // ... build the page ...
  NN.page('your-page-id');   // id from NN.PAGES
})();
</script>
</body>
</html>
```

`NN.page(id)` injects the sticky topbar, wires the theme toggle, and fills
`#pagenav` and `#sitefoot`. Call it **last**.

## The model, for reference

`2 → 3 (ReLU) → 1`, squared-error loss, 13 parameters.

```
x  = [2.0, 3.0]        size in 1000s sqft, bedrooms
y  = 1.0               price in $100k
z1 = W1·x + b1  = [0.5, 1.3, -0.8]
a1 = relu(z1)   = [0.5, 1.3, 0.0]      <- unit 2 is DEAD
z2 = W2·a1 + b2 = -0.13
L  = (ŷ − y)²   = 1.2769
```

The dead ReLU unit is the pedagogical centrepiece — its gradient row in
`dL/dW1` is exactly `[0, 0]`. Refer back to it wherever it's relevant.

## `NN.T` shape

| Path | What it is |
|---|---|
| `T.meta` | hyperparams, input meanings, architecture string |
| `T.init` | `W1 b1 W2 b2` starting values |
| `T.param_names` | 13 strings, `"W1[0][1]"` … |
| `T.forward` | `x y z1 z1_work a1 relu_mask z2 z2_work yhat error loss saved_for_backward` |
| `T.forward.z1_work[i]` | `{index, products:[{w,x,prod}], sum_of_products, bias, out}` — every multiply, for animating |
| `T.backward` | `dL_dyhat dL_dz2 dL_dW2 dL_db2 dL_da1 dL_dz1 dL_dW1 dL_db1 steps` |
| `T.backward.steps[]` | `{id,label,formula,substitution,value,reads,note}` — the **8** chain-rule steps in order. Never hardcode the count; use `.length` |
| `T.gradcheck` | `{analytic, numerical, max_abs_error, passed}` |
| `T.runs.{sgd,momentum,adam}.history[]` | per-step `{t, loss, yhat, theta, grad, z1, a1, relu_mask, theta_after}`; adam adds `m`, `v`, `adam_detail[]` |
| `T.runs.*.history[t].adam_detail[i]` | `{g,m,v,m_hat,v_hat,update,before,after}` per parameter |
| `T.memory.per_tensor[]` | `{name, shape, elements, cls, role?, of?, transient?, note?}` — the **20** tensors. `role` is `param` / `param_grad` / `node_grad`; `of` names the parameter a `param_grad` belongs to |
| `T.memory.by_class` | per-class `{elements, tensors, names, bytes_fp32}` |
| `T.memory.param_gradient_elements` | **13** — the count that is 1:1 with parameters. `by_class.gradient.elements` is 21 because it includes transient node gradients |
| `T.memory.timeline[]` | 14 phases, `{phase,label,desc,alloc[],free[],reads?[]}` — one training iteration |
| `T.memory.timeline_bytes_fp32` | resident bytes per phase; plus `peak_bytes_fp32` (248), `floor_bytes_fp32` (156), `peak_phase` (9) |
| `T.memory.recipes[]` | six real configs, `components` in bytes/param by class |
| `T.memory.dtype_bytes` | `{fp32:4, bf16:2, fp16:2, fp8:1, int8:1}` |
| `T.formats` | per-format bit widths, ranges, epsilon, decimal digits |
| `T.reference_configs` | real model + GPU specs. The ONLY externally-quoted numbers; surface `._source` |
| `T.bitviews[]` | `{label, value, fp32, bf16, fp16}`, each `{dtype,bytes,bits,sign,exponent,mantissa,exact}` |
| `T.layout` | W1 row-major layout: `{shape, strides_elements, strides_bytes_fp32, cells[], flat}` |

Inspect it live: open any page and type `NN.T` in the console.

## UI kit cheatsheet

```js
NN.el(tag, attrs, kids)                  // DOM builder; attrs.html/.text/.style/.onclick
NN.fmt(x, digits)  NN.fmtSigned(x)  NN.count(n)
NN.bytes(n)     // DECIMAL: 1 GB = 1e9. Use for model/tensor memory.
NN.gibytes(n)   // BINARY:  1 GiB = 2^30. Use for DEVICE CAPACITY only.
                // An "80 GB" H100 is 80 GiB = 85.9 GB. Never mix the two
                // in one comparison without saying which is which.
NN.chip('gradient')                      // labelled+glyphed colour chip
NN.legend(['weight','gradient'])         // legend row
NN.tensor({name, shape, values, cls, digits, signed, note})
      // returns el; el.update(vals), el.highlight([[r,c],...], dimRest)
NN.formula({rule, sub, result, cls})     // symbolic → substitution → result
NN.callout('key'|'why'|'mem'|'warn', TITLE, html)
NN.tiles([{k,v,s,cls}])                  // stat tiles
NN.memstrip([{cls,bytes,label}], total)  // proportional memory bar
NN.bitribbon(T.bitviews[0].fp32, 16)     // bit strip; 2nd arg greys bits ≥ index
new NN.Stepper({mount, count, labels, onStep(i,dir), autoplayMs})
NN.chart.line({series:[{name,color,points:[[x,y]]}], width,height,xlabel,ylabel,logY})
NN.chart.bars({items:[{label,value,color,sub}], unit, max, limit:{value,label}})
NN.tip.show(html,x,y) / NN.tip.hide()
```

Colours in JS: `'var(--weight)'`, `'var(--gradient)'`, `'var(--activation)'`,
`'var(--optimizer)'`, `'var(--accent)'`, `'var(--critical)'`.

## Tone

Explain like a good systems engineer teaching a sharp colleague. Concrete
over abstract. Say *why* a thing is that way, not just that it is. No
marketing voice, no "simply", no exclamation marks. Every claim about memory
should be backed by an arithmetic you show.

## Required per page

- At least one **interactive** element (stepper, slider, hover, or toggle).
- At least one **callout** connecting the toy model to real LLM training.
- A `<noscript>` line telling the reader the page needs JS.
- Works at 1280px and degrades acceptably at 800px.

## Verify before you finish

```
node -e "require('fs').readFileSync('site/YOURPAGE.html','utf8')"   # exists
python3 -c "import re,sys; ..."                                     # or just eyeball
```
Open the file and confirm: no console errors, topbar renders, every number
visible matches `NN.T`.
