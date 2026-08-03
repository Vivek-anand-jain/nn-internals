/* ==========================================================================
   nn-internals shared UI kit  ->  window.NN
   --------------------------------------------------------------------------
   Every page loads, in this order:

       <link rel="stylesheet" href="../assets/css/design.css">
       <script src="../assets/data/trace.js"></script>   <- window.TRACE
       <script src="../assets/js/ui.js"></script>        <- window.NN

   trace.js is a plain script (not fetch) on purpose: it means the whole
   site works from a file:// URL with no server and no CORS problems.

   NOTHING here hard-codes a number from the model. Every value comes out
   of NN.T, which is the payload emitted by code/ground_truth.py.
   ========================================================================== */
(function () {
  "use strict";

  const NN = {};
  window.NN = NN;

  /* -------------------------------------------------------------- data */
  NN.T = window.TRACE || null;
  if (!NN.T) {
    console.error("nn-internals: trace.js did not load. " +
                  "Run `python3 code/ground_truth.py` to regenerate it.");
  }

  /* ------------------------------------------------------------- pages */
  NN.PAGES = [
    /* ---- Part I — the mechanics, on the 13-parameter MLP ---- */
    { n: "01", id: "memory",      file: "01-memory.html",      title: "Tensors in Memory",
      blurb: "What a parameter actually is: bytes, dtypes, strides." },
    { n: "02", id: "forward",     file: "02-forward.html",     title: "Forward Pass",
      blurb: "Every multiply-accumulate, one at a time." },
    { n: "03", id: "loss",        file: "03-loss.html",        title: "Loss",
      blurb: "Turning a prediction into one number to minimise." },
    { n: "04", id: "backward",    file: "04-backward.html",    title: "Backward Pass",
      blurb: "The chain rule, derived — and why activations are kept." },
    { n: "05", id: "optimizer",   file: "05-optimizer.html",   title: "Optimizer & Adam",
      blurb: "SGD, momentum, Adam, and the fp32 master copy." },
    { n: "06", id: "dynamics",    file: "06-training-dynamics.html", title: "Training Dynamics",
      blurb: "Warmup, decay, initialisation, clipping, and loss spikes." },
    { n: "07", id: "loop",        file: "07-loop.html",        title: "One Full Iteration",
      blurb: "Memory rising and falling across a single step." },

    /* ---- Part II — a real transformer, on the 288-parameter block.
       The objective comes FIRST: you cannot follow the mechanics until you
       know what the thing is predicting. ---- */
    { n: "08", id: "lm",          file: "08-what-it-predicts.html", title: "What It Predicts",
      blurb: "X, Y and ŷ. Next-token prediction and cross-entropy." },
    { n: "09", id: "tfforward",   file: "09-transformer-forward.html", title: "Transformer Forward",
      blurb: "Two real blocks, every matrix, token by token." },
    { n: "10", id: "tfbackward",  file: "10-transformer-backward.html", title: "Transformer Backward",
      blurb: "LayerNorm's two correction terms. Softmax's dense Jacobian." },
    { n: "11", id: "position",    file: "11-positional.html",  title: "Position",
      blurb: "Attention is orderless. RoPE is how order gets back in." },
    { n: "12", id: "attnvariants", file: "12-attention-variants.html", title: "Attention Variants",
      blurb: "MHA → MQA → GQA → MLA, and the KV cache that drove them." },
    { n: "13", id: "transformer", file: "13-transformer-cost.html", title: "What It Costs",
      blurb: "Parameter counts, activation memory, and the seq² term." },
    { n: "14", id: "flash",       file: "14-flash-attention.html", title: "FlashAttention",
      blurb: "The seq² matrix never has to exist. Proven exact." },
    { n: "15", id: "scaling",     file: "15-scaling.html",     title: "Scaling to 70B",
      blurb: "Why ZeRO, FSDP, TP and PP have to exist." },

    /* ---- Part III — many GPUs, training ---- */
    { n: "16", id: "collectives", file: "16-collectives.html", title: "Collective Operations",
      blurb: "Broadcast, all-gather, reduce-scatter, all-reduce — and the ring." },
    { n: "17", id: "dataparallel", file: "17-data-parallel.html", title: "Data Parallel",
      blurb: "Split the batch, all-reduce the gradients. Proven identical." },
    { n: "18", id: "zero",        file: "18-zero-fsdp.html",   title: "ZeRO & FSDP",
      blurb: "Stop replicating what you don't need. Three stages." },
    { n: "19", id: "tensorpar",   file: "19-tensor-parallel.html", title: "Tensor Parallel",
      blurb: "Cut the weight matrices themselves. Column then row." },
    { n: "20", id: "seqpar",      file: "20-sequence-parallel.html", title: "Sequence Parallelism",
      blurb: "The activations TP leaves replicated — removed for free." },
    { n: "21", id: "pipelinepar", file: "21-pipeline-parallel.html", title: "Pipeline Parallel",
      blurb: "Split the layers. Live with the bubble." },
    { n: "22", id: "tfpartition", file: "22-transformer-partitioned.html", title: "Partitioning a Block",
      blurb: "Every weight matrix of a real transformer, cut four ways." },
    { n: "23", id: "combining",   file: "23-3d-parallelism.html", title: "3D Parallelism",
      blurb: "Composing TP × PP × DP onto real hardware topology." },

    /* ---- Part IV — inference. A different machine: arithmetic intensity
       replaces the four tensor classes as the organising idea. ---- */
    { n: "24", id: "regimes",     file: "24-prefill-decode.html", title: "Prefill vs Decode",
      blurb: "Two regimes, 8000× apart in arithmetic intensity." },
    { n: "25", id: "kvcache",     file: "25-kv-cache.html",    title: "The KV Cache",
      blurb: "Exact memoisation — and what it costs at 128k context." },
    { n: "26", id: "batching",    file: "26-batching.html",    title: "Batching",
      blurb: "Continuous batching, chunked prefill, and why decode batches free." },
    { n: "27", id: "paged",       file: "27-paged-attention.html", title: "PagedAttention",
      blurb: "The KV cache needed an operating system." },
    { n: "28", id: "quant",       file: "28-quantisation.html", title: "Quantisation",
      blurb: "INT8, INT4, FP8 — and why weight-only helps decode." },
    { n: "29", id: "specdec",     file: "29-speculative.html", title: "Speculative Decoding",
      blurb: "Verify k tokens for the price of one. A bandwidth trick." },
    { n: "30", id: "infpar",      file: "30-inference-parallelism.html", title: "Inference Parallelism",
      blurb: "TP for latency, PP for throughput — the opposite of training." },
    { n: "31", id: "serving",     file: "31-serving.html",     title: "Serving at Scale",
      blurb: "Disaggregated prefill/decode, scheduling, and the SLOs." },

    /* ---- Part V — mixture of experts ---- */
    { n: "32", id: "moe",         file: "32-moe.html",         title: "Mixture of Experts",
      blurb: "Routing, top-k, and decoupling parameters from FLOPs." },
    { n: "33", id: "moetrain",    file: "33-moe-training.html", title: "MoE Training",
      blurb: "Expert parallelism, all-to-all, and the load-balance problem." },
    { n: "34", id: "moeinfer",    file: "34-moe-inference.html", title: "MoE Inference",
      blurb: "A batch that scatters, and weights read for almost no math." }
  ];

  /* ------------------------------------------------------------- utils */

  /** Create an element. NN.el('div', {class:'x'}, ['text', otherEl]) */
  NN.el = function (tag, attrs, kids) {
    const e = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      const v = attrs[k];
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") e.className = v;
      else if (k === "html") e.innerHTML = v;
      else if (k === "text") e.textContent = v;
      else if (k.startsWith("on") && typeof v === "function") e.addEventListener(k.slice(2), v);
      else if (k === "style" && typeof v === "object") Object.assign(e.style, v);
      else e.setAttribute(k, v);
    }
    if (kids !== null && kids !== undefined) {
      (Array.isArray(kids) ? kids : [kids]).forEach(function (c) {
        if (c === null || c === undefined || c === false) return;
        e.appendChild(typeof c === "object" ? c : document.createTextNode(String(c)));
      });
    }
    return e;
  };

  NN.$  = function (s, r) { return (r || document).querySelector(s); };
  NN.$$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };

  /**
   * Format a number for display.
   * Keeps trailing zeros off, shows a real 0 as "0", and falls back to
   * exponential only when the magnitude genuinely demands it.
   */
  NN.fmt = function (x, digits) {
    if (x === null || x === undefined || Number.isNaN(x)) return "–";
    const d = digits === undefined ? 4 : digits;
    if (x === 0) return "0";
    const a = Math.abs(x);
    if (a >= 1e6 || a < 1e-4) return x.toExponential(2);
    let s = x.toFixed(d);
    if (s.indexOf(".") >= 0) s = s.replace(/0+$/, "").replace(/\.$/, "");
    return s === "-0" ? "0" : s;
  };

  /** Signed format, always shows + or -, for gradients and deltas. */
  NN.fmtSigned = function (x, digits) {
    const s = NN.fmt(x, digits);
    return (x > 0 ? "+" : "") + s;
  };

  /**
   * Bytes -> human string, DECIMAL (1 GB = 1e9 bytes).
   *
   * This used to divide by 1024 while labelling the result "GB", which is
   * simply wrong -- 1024-based units are KiB/MiB/GiB -- and it made the
   * same quantity render as 1.03 TB on some pages and 1.13 TB on others.
   * Decimal is also how model memory is universally quoted: 70B params in
   * bf16 is "140 GB", not "130 GiB".
   *
   * For DEVICE CAPACITY use NN.gibytes: an "80 GB" H100 really has
   * 80 GiB = 85.9 GB. Do not mix the two in one comparison without saying so.
   */
  NN.bytes = function (b) {
    const u = ["B", "KB", "MB", "GB", "TB", "PB"];
    let i = 0, v = b;
    while (Math.abs(v) >= 1000 && i < u.length - 1) { v /= 1000; i++; }
    // Raw byte counts stay integral (the toy model's "292 B" census).
    // Scaled units keep a decimal even in the 100-999 band, so "141.2 GB"
    // does not appear as "141 GB" two lines from a chart label that says
    // 141.2 -- same quantity, same base, two precisions reads like a bug.
    if (i === 0) return Math.round(v) + " " + u[i];
    const a = Math.abs(v);
    return (a < 10 ? v.toFixed(2) : v.toFixed(1)) + " " + u[i];
  };

  /** Bytes -> human string, BINARY, correctly labelled (1 GiB = 2^30 bytes). */
  NN.gibytes = function (b) {
    const u = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    let i = 0, v = b;
    while (Math.abs(v) >= 1024 && i < u.length - 1) { v /= 1024; i++; }
    return (Math.abs(v) < 10 ? v.toFixed(2) : Math.abs(v) < 100 ? v.toFixed(1)
            : Math.round(v)) + " " + u[i];
  };

  /**
   * Big counts -> 1.2K / 3.4M / 70.6B.
   * Keeps one decimal whenever it carries information, so 70.6e9 renders as
   * "70.6B" and not "71B" -- the latter reads like a typo next to a model
   * literally named "Llama 3 70B". A trailing ".0" is dropped.
   */
  NN.count = function (n) {
    const one = function (v, suffix) {
      let s = v.toFixed(1);
      if (s.endsWith(".0")) s = s.slice(0, -2);
      return s + suffix;
    };
    const a = Math.abs(n);
    if (a < 1e3) return String(n);
    if (a < 1e6) return one(n / 1e3, "K");
    if (a < 1e9) return one(n / 1e6, "M");
    if (a < 1e12) return one(n / 1e9, "B");
    return one(n / 1e12, "T");
  };

  NN.clamp = function (v, lo, hi) { return Math.min(hi, Math.max(lo, v)); };
  NN.lerp  = function (a, b, t) { return a + (b - a) * t; };

  /* ------------------------------------------------------ tensor class */
  /* The single place that knows the class -> label/glyph mapping. */
  NN.CLASSES = {
    weight:     { label: "Weight",          glyph: "■", css: "weight",
                  life: "entire run" },
    gradient:   { label: "Gradient",        glyph: "▼", css: "gradient",
                  life: "backward → step" },
    activation: { label: "Activation",      glyph: "●", css: "activation",
                  life: "forward → backward" },
    optimizer:  { label: "Optimizer state", glyph: "◆", css: "optimizer",
                  life: "entire run" }
  };

  /** A colour-plus-glyph-plus-label chip. Never use colour alone. */
  NN.chip = function (cls, textOverride) {
    const c = NN.CLASSES[cls];
    return NN.el("span", { class: "tclass tclass-" + cls }, textOverride || c.label);
  };

  NN.legend = function (classes) {
    const list = classes || Object.keys(NN.CLASSES);
    return NN.el("div", { class: "legend" }, list.map(function (c) {
      return NN.chip(c);
    }));
  };

  /* -------------------------------------------------------- tensor grid */
  /**
   * Render a tensor as a grid of cells.
   *
   *   NN.tensor({
   *     name: "W1", shape: [3,2], values: [[..],[..],[..]],
   *     cls: "weight",
   *     hot: [[0,1]],            // cells to highlight
   *     dim: [[2,0],[2,1]],      // cells to fade
   *     digits: 3,
   *     signed: false,
   *     note: "3x2 = 6 params"
   *   })
   *
   * `values` may be a flat array (treated as a column) or nested rows.
   * Returns a DOM element; also exposes .update(newValues) for animation.
   */
  NN.tensor = function (o) {
    const rows = Array.isArray(o.values[0]) ? o.values : o.values.map(function (v) { return [v]; });
    const nr = rows.length, nc = rows[0].length;
    const cls = o.cls || "weight";

    const body = NN.el("div", {
      class: "tensor-body",
      style: { gridTemplateColumns: "repeat(" + nc + ", auto)" }
    });

    const cells = [];
    for (let i = 0; i < nr; i++) {
      cells.push([]);
      for (let j = 0; j < nc; j++) {
        const v = rows[i][j];
        const c = NN.el("div", {
          class: "cell is-" + cls + (v === 0 ? " is-zero" : ""),
          "data-r": i, "data-c": j,
          title: (o.name || "") + "[" + i + "]" + (nc > 1 ? "[" + j + "]" : "") + " = " + v
        }, o.signed ? NN.fmtSigned(v, o.digits) : NN.fmt(v, o.digits));
        body.appendChild(c);
        cells[i].push(c);
      }
    }

    const shapeTxt = o.shape ? "(" + o.shape.join("×") + ")" : "(" + nr + "×" + nc + ")";
    const head = NN.el("div", { class: "tensor-label" }, [
      NN.el("span", { class: "tensor-name v-" + cls }, o.name || ""),
      NN.el("span", { class: "tensor-shape" }, shapeTxt + (o.note ? "  " + o.note : ""))
    ]);

    const root = NN.el("div", { class: "tensor" }, [head, body]);
    root.cells = cells;

    /** Repaint values without rebuilding the DOM (so CSS transitions run). */
    root.update = function (newVals, opts) {
      const nrows = Array.isArray(newVals[0]) ? newVals : newVals.map(function (v) { return [v]; });
      for (let i = 0; i < nr; i++) for (let j = 0; j < nc; j++) {
        const v = nrows[i][j];
        const c = cells[i][j];
        c.textContent = (opts && opts.signed) || o.signed ? NN.fmtSigned(v, o.digits) : NN.fmt(v, o.digits);
        c.classList.toggle("is-zero", v === 0);
        c.title = (o.name || "") + "[" + i + "]" + (nc > 1 ? "[" + j + "]" : "") + " = " + v;
      }
    };

    /** Highlight a set of [r,c] pairs; pass null to clear. */
    root.highlight = function (pairs, dimRest) {
      cells.forEach(function (row) {
        row.forEach(function (c) { c.classList.remove("hot", "dim"); });
      });
      if (!pairs) return;
      const set = new Set(pairs.map(function (p) { return p[0] + "," + p[1]; }));
      cells.forEach(function (row, i) {
        row.forEach(function (c, j) {
          if (set.has(i + "," + j)) c.classList.add("hot");
          else if (dimRest) c.classList.add("dim");
        });
      });
    };

    return root;
  };

  /* ----------------------------------------------------------- formula */
  /** A derivation block: the symbolic rule, the substitution, the result. */
  NN.formula = function (o) {
    return NN.el("div", { class: "formula" + (o.cls ? " is-" + o.cls : "") }, [
      NN.el("div", { html: o.rule }),
      o.sub ? NN.el("span", { class: "sub", html: "= " + o.sub }) : null,
      o.result !== undefined
        ? NN.el("span", { class: "sub" }, [
            "= ", NN.el("span", { class: "res v-" + (o.cls || "weight") }, o.result)
          ])
        : null
    ]);
  };

  /* ---------------------------------------------------------- callout */
  NN.callout = function (kind, title, bodyHtml) {
    return NN.el("div", { class: "callout " + kind }, [
      NN.el("div", { class: "t" }, title),
      NN.el("div", { html: bodyHtml })
    ]);
  };

  /* ------------------------------------------------------- stat tiles */
  NN.tiles = function (items) {
    return NN.el("div", { class: "tiles" }, items.map(function (it) {
      return NN.el("div", { class: "tile" }, [
        NN.el("div", { class: "k" }, it.k),
        NN.el("div", { class: "v" + (it.cls ? " v-" + it.cls : "") }, it.v),
        it.s ? NN.el("div", { class: "s" }, it.s) : null
      ]);
    }));
  };

  /* ------------------------------------------------------ memory strip */
  /**
   * NN.memstrip([{cls:'weight', bytes:52, label:'weights'}, ...], totalBytes)
   * Segments flex-grow in proportion to bytes; leftover renders as free.
   */
  NN.memstrip = function (segs, total) {
    const used = segs.reduce(function (a, s) { return a + s.bytes; }, 0);
    const cap = total || used;
    const map = { weight: "w", gradient: "g", activation: "a", optimizer: "o" };
    const strip = NN.el("div", { class: "memstrip" });
    segs.forEach(function (s) {
      if (s.bytes <= 0) return;
      strip.appendChild(NN.el("div", {
        class: "memseg " + (map[s.cls] || "free"),
        style: { flexGrow: String(s.bytes) },
        title: s.label + " — " + NN.bytes(s.bytes)
      }, s.bytes / cap > 0.07 ? s.label : ""));
    });
    if (cap > used) {
      strip.appendChild(NN.el("div", {
        class: "memseg free", style: { flexGrow: String(cap - used) },
        title: "free — " + NN.bytes(cap - used)
      }, (cap - used) / cap > 0.07 ? "free" : ""));
    }
    return strip;
  };

  /* -------------------------------------------------------- bit ribbon */
  /** Render an IEEE-754 bit pattern with sign/exponent/mantissa colouring. */
  NN.bitribbon = function (dec, droppedFrom) {
    const wrap = NN.el("div", { class: "bits" });
    const push = function (chars, kind, offset) {
      chars.split("").forEach(function (b, i) {
        const dropped = droppedFrom !== undefined && (offset + i) >= droppedFrom;
        wrap.appendChild(NN.el("div", {
          class: "bit " + kind + (dropped ? " dropped" : "")
        }, b));
      });
    };
    push(dec.sign, "sign", 0);
    push(dec.exponent, "exp", 1);
    push(dec.mantissa, "man", 1 + dec.exponent.length);
    return wrap;
  };

  /* ------------------------------------------------------------ chrome */
  /**
   * Build the sticky top bar. `base` is "" from the repo root and "../"
   * from inside site/.
   */
  NN.topbar = function (currentId, base) {
    const b = base === undefined ? "../" : base;
    const links = NN.PAGES.map(function (p) {
      return NN.el("a", {
        href: b + "site/" + p.file,
        "aria-current": p.id === currentId ? "page" : null,
        title: p.title
      }, p.n + " " + p.title);
    });

    const bar = NN.el("header", { class: "topbar" }, [
      NN.el("div", { class: "inner" }, [
        NN.el("a", { class: "brand", href: b + "index.html" },
          [document.createTextNode("nn"), NN.el("span", {}, "-internals")]),
        NN.el("nav", { class: "navlinks" }, links),
        NN.el("a", { class: "iconbtn", href: b + "slides/index.html", title: "Slide deck" }, "Slides"),
        NN.el("button", {
          class: "iconbtn", id: "themeToggle", title: "Toggle light / dark"
        }, "Theme")
      ])
    ]);
    return bar;
  };

  /** Previous / next footer links, derived from NN.PAGES order. */
  NN.pagenav = function (currentId) {
    const i = NN.PAGES.findIndex(function (p) { return p.id === currentId; });
    const prev = i > 0 ? NN.PAGES[i - 1] : null;
    const next = i >= 0 && i < NN.PAGES.length - 1 ? NN.PAGES[i + 1] : null;
    return NN.el("nav", { class: "pagenav" }, [
      prev ? NN.el("a", { class: "prev", href: prev.file }, [
        NN.el("div", { class: "d" }, "← Previous"),
        NN.el("div", { class: "t" }, prev.n + " " + prev.title)
      ]) : NN.el("a", { class: "prev", href: "../index.html" }, [
        NN.el("div", { class: "d" }, "← Back"),
        NN.el("div", { class: "t" }, "Contents")
      ]),
      next ? NN.el("a", { class: "next", href: next.file }, [
        NN.el("div", { class: "d" }, "Next →"),
        NN.el("div", { class: "t" }, next.n + " " + next.title)
      ]) : NN.el("a", { class: "next", href: "../index.html" }, [
        NN.el("div", { class: "d" }, "Done →"),
        NN.el("div", { class: "t" }, "Back to contents")
      ])
    ]);
  };

  /* ------------------------------------------------------------- theme */
  /**
   * localStorage throws (not returns null) on an opaque file:// origin in
   * some browser configurations. Since this whole site is meant to be
   * opened straight off disk, an unguarded access would take down the
   * entire page script. Theme persistence is a nicety; rendering is not.
   */
  function storeGet(k) {
    try { return localStorage.getItem(k); } catch (e) { return null; }
  }
  function storeSet(k, v) {
    try { localStorage.setItem(k, v); } catch (e) { /* non-fatal */ }
  }

  NN.initTheme = function () {
    const saved = storeGet("nn-theme");
    if (saved) document.documentElement.setAttribute("data-theme", saved);
    document.addEventListener("click", function (e) {
      const t = e.target.closest && e.target.closest("#themeToggle");
      if (!t) return;
      const cur = document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light";
      document.documentElement.setAttribute("data-theme", cur);
      storeSet("nn-theme", cur);
      window.dispatchEvent(new CustomEvent("nn:theme", { detail: cur }));
    });
  };

  /* ----------------------------------------------------------- stepper */
  /**
   * Playback controller for a list of steps.
   *
   *   const s = new NN.Stepper({
   *     mount: someEl,          // where the control bar goes
   *     count: 7,
   *     labels: [...],          // optional per-step caption
   *     onStep: (i, dir) => {...},
   *     autoplayMs: 1100
   *   });
   *
   * Keyboard: left/right arrows, space to play/pause.
   */
  /* --------------------------------------------------- stepper routing */
  NN._steppers = [];
  NN._stepperKeysInstalled = false;

  /** The stepper whose control bar is closest to the viewport centre. */
  NN._activeStepper = function () {
    const mid = window.innerHeight / 2;
    let best = null, bestD = Infinity;
    NN._steppers.forEach(function (s) {
      if (!s.bar || !s.bar.isConnected) return;
      const r = s.bar.getBoundingClientRect();
      // ignore bars scrolled far out of view entirely
      if (r.bottom < -400 || r.top > window.innerHeight + 400) return;
      const d = Math.abs((r.top + r.bottom) / 2 - mid);
      if (d < bestD) { bestD = d; best = s; }
    });
    return best;
  };

  NN._installStepperKeys = function () {
    if (NN._stepperKeysInstalled) return;
    NN._stepperKeysInstalled = true;
    document.addEventListener("keydown", function (e) {
      if (/input|textarea|select/i.test(e.target.tagName)) return;
      if (e.target.isContentEditable) return;
      const s = NN._activeStepper();
      if (!s) return;
      if (e.key === "ArrowRight") { s.pause(); s.next(); }
      else if (e.key === "ArrowLeft") { s.pause(); s.prev(); }
      else if (e.key === " " && s._spaceToPlay !== false) {
        e.preventDefault(); s.toggle();
      }
    }, true);
  };

  NN.Stepper = function (o) {
    const self = this;
    this.i = 0;
    this.count = o.count;
    this.playing = false;
    this._timer = null;

    const prev = NN.el("button", { class: "btn" }, "← Back");
    const next = NN.el("button", { class: "btn primary" }, "Next →");
    const play = NN.el("button", { class: "btn" }, "▶ Play");
    const reset = NN.el("button", { class: "btn" }, "Reset");
    const dots = NN.el("div", { class: "progressdots" });
    const count = NN.el("div", { class: "stepcount" });

    for (let k = 0; k < this.count; k++) {
      const d = NN.el("i", { title: "Step " + (k + 1) });
      d.addEventListener("click", function () { self.go(k); });
      dots.appendChild(d);
    }

    const bar = NN.el("div", { class: "controls" }, [prev, next, play, reset, dots, count]);
    if (o.mount) o.mount.appendChild(bar);
    this.bar = bar;
    this._spaceToPlay = o.spaceToPlay;

    const caption = o.captionEl || null;

    this.render = function (dir) {
      prev.disabled = self.i === 0;
      next.disabled = self.i === self.count - 1;
      count.textContent = "Step " + (self.i + 1) + " / " + self.count +
        (o.labels && o.labels[self.i] ? "  —  " + o.labels[self.i] : "");
      NN.$$("i", dots).forEach(function (d, k) {
        d.classList.toggle("done", k < self.i);
        d.classList.toggle("now", k === self.i);
      });
      if (caption && o.labels) caption.textContent = o.labels[self.i] || "";
      if (o.onStep) o.onStep(self.i, dir || 0);
    };

    this.go = function (k) {
      const t = NN.clamp(k, 0, self.count - 1);
      const dir = Math.sign(t - self.i);
      self.i = t;
      self.render(dir);
      if (self.playing && self.i === self.count - 1) self.pause();
    };
    this.next  = function () { self.go(self.i + 1); };
    this.prev  = function () { self.go(self.i - 1); };
    this.reset = function () { self.pause(); self.go(0); };

    this.play = function () {
      if (self.i === self.count - 1) self.i = 0;
      self.playing = true;
      play.textContent = "⏸ Pause";
      self._timer = setInterval(function () { self.next(); }, o.autoplayMs || 1200);
      self.render(1);
    };
    this.pause = function () {
      self.playing = false;
      play.textContent = "▶ Play";
      clearInterval(self._timer);
    };
    this.toggle = function () { self.playing ? self.pause() : self.play(); };

    prev.addEventListener("click", function () { self.pause(); self.prev(); });
    next.addEventListener("click", function () { self.pause(); self.next(); });
    play.addEventListener("click", function () { self.toggle(); });
    reset.addEventListener("click", function () { self.reset(); });

    /* Keyboard routing.
       Every Stepper used to attach its own document-level listener, so a
       page with two steppers drove BOTH from one arrow press. Pages 10 and
       16 each have several. Instead there is one shared listener, and the
       key goes to whichever stepper's control bar is nearest the centre of
       the viewport -- i.e. the one the reader is looking at. */
    NN._steppers.push(self);
    NN._installStepperKeys();

    this.render(0);
  };

  /* ------------------------------------------------------------ charts */
  NN.chart = {};

  const SVGNS = "http://www.w3.org/2000/svg";
  function svg(tag, attrs) {
    const e = document.createElementNS(SVGNS, tag);
    for (const k in attrs) if (attrs[k] !== null && attrs[k] !== undefined) {
      e.setAttribute(k, attrs[k]);
    }
    return e;
  }
  NN.svg = svg;

  /* one shared tooltip element */
  let _tip = null;
  NN.tip = {
    show: function (html, x, y) {
      if (!_tip) { _tip = NN.el("div", { class: "tooltip" }); document.body.appendChild(_tip); }
      _tip.innerHTML = html;
      _tip.classList.add("on");
      const r = _tip.getBoundingClientRect();
      _tip.style.left = NN.clamp(x + 14, 6, window.innerWidth - r.width - 6) + "px";
      _tip.style.top  = NN.clamp(y - r.height - 10, 6, window.innerHeight - r.height - 6) + "px";
    },
    hide: function () { if (_tip) _tip.classList.remove("on"); }
  };

  /**
   * Line chart with a crosshair + tooltip (the default interaction layer).
   *
   *   NN.chart.line({
   *     series: [{name:'loss', color:'var(--gradient)', points:[[x,y],...]}],
   *     width: 640, height: 240,
   *     xlabel: 'step', ylabel: 'loss', logY: false,
   *     yMin, yMax
   *   })
   */
  NN.chart.line = function (o) {
    const W = o.width || 640, H = o.height || 260;
    const M = Object.assign({ t: 14, r: 16, b: 34, l: 52 }, o.margin || {});
    const iw = W - M.l - M.r, ih = H - M.t - M.b;

    const all = o.series.flatMap(function (s) { return s.points; });
    const xs = all.map(function (p) { return p[0]; });
    const ys = all.map(function (p) { return p[1]; });
    const x0 = o.xMin !== undefined ? o.xMin : Math.min.apply(null, xs);
    const x1 = o.xMax !== undefined ? o.xMax : Math.max.apply(null, xs);
    const lg = !!o.logY;
    const tf = function (v) { return lg ? Math.log10(Math.max(v, 1e-12)) : v; };
    let y0 = o.yMin !== undefined ? tf(o.yMin) : Math.min.apply(null, ys.map(tf));
    let y1 = o.yMax !== undefined ? tf(o.yMax) : Math.max.apply(null, ys.map(tf));
    if (y0 === y1) { y0 -= 1; y1 += 1; }
    if (!lg && o.yMin === undefined) y0 = Math.min(0, y0);
    const pad = (y1 - y0) * 0.08; y0 -= pad; y1 += pad;

    const X = function (v) { return M.l + (x1 === x0 ? iw / 2 : (v - x0) / (x1 - x0) * iw); };
    const Y = function (v) { return M.t + ih - (tf(v) - y0) / (y1 - y0) * ih; };

    const root = svg("svg", {
      class: "chart", viewBox: "0 0 " + W + " " + H,
      width: "100%", height: H, role: "img"
    });

    /* gridlines + y ticks
       On a log axis the tick VALUES are already exponents, so evenly
       spacing them and rounding for the label produced duplicates like
       "1e-2, 1e-2". Place log ticks on whole decades instead, and thin
       them out if the range spans more decades than we want lines. */
    let ticks = [];
    if (lg) {
      const lo = Math.floor(y0), hi = Math.ceil(y1);
      const stride = Math.max(1, Math.ceil((hi - lo) / 6));
      for (let d = lo; d <= hi; d += stride) {
        if (d >= y0 - 1e-9 && d <= y1 + 1e-9) ticks.push({ v: d, label: "1e" + d });
      }
      if (ticks.length < 2) ticks = [
        { v: y0, label: "1e" + NN.fmt(y0, 1) },
        { v: y1, label: "1e" + NN.fmt(y1, 1) }
      ];
    } else {
      const NT = 5;
      for (let k = 0; k <= NT; k++) {
        const tv = y0 + (y1 - y0) * k / NT;
        ticks.push({ v: tv, label: NN.fmt(tv, 3) });
      }
    }
    ticks.forEach(function (t) {
      const yy = M.t + ih - (t.v - y0) / (y1 - y0) * ih;
      root.appendChild(svg("line", { class: "gridline", x1: M.l, x2: M.l + iw, y1: yy, y2: yy }));
      root.appendChild(svg("text", {
        x: M.l - 8, y: yy + 3.5, "text-anchor": "end"
      })).textContent = t.label;
    });

    /* x ticks */
    const xtCount = Math.min(8, Math.max(2, Math.round(x1 - x0)));
    for (let k = 0; k <= xtCount; k++) {
      const tv = x0 + (x1 - x0) * k / xtCount;
      root.appendChild(svg("text", {
        x: X(tv), y: M.t + ih + 18, "text-anchor": "middle"
      })).textContent = NN.fmt(tv, 1);
    }

    root.appendChild(svg("line", { class: "axis", x1: M.l, x2: M.l + iw, y1: M.t + ih, y2: M.t + ih }));
    root.appendChild(svg("line", { class: "axis", x1: M.l, x2: M.l, y1: M.t, y2: M.t + ih }));

    if (o.ylabel) {
      const t = svg("text", {
        x: -(M.t + ih / 2), y: 13, transform: "rotate(-90)", "text-anchor": "middle"
      });
      t.textContent = o.ylabel; root.appendChild(t);
    }
    if (o.xlabel) {
      const t = svg("text", { x: M.l + iw / 2, y: H - 3, "text-anchor": "middle" });
      t.textContent = o.xlabel; root.appendChild(t);
    }

    /* series */
    o.series.forEach(function (s) {
      const d = s.points.map(function (p, i) {
        return (i ? "L" : "M") + X(p[0]).toFixed(2) + " " + Y(p[1]).toFixed(2);
      }).join(" ");
      root.appendChild(svg("path", {
        class: "series", d: d, stroke: s.color,
        "stroke-dasharray": s.dash || null,
        "stroke-linejoin": "round", "stroke-linecap": "round"
      }));
      if (s.dots !== false && s.points.length <= 40) {
        s.points.forEach(function (p) {
          root.appendChild(svg("circle", {
            cx: X(p[0]), cy: Y(p[1]), r: 3, fill: s.color,
            stroke: "var(--surface-1)", "stroke-width": 2
          }));
        });
      }
    });

    /* crosshair + tooltip */
    const cross = svg("line", {
      class: "axis", x1: 0, x2: 0, y1: M.t, y2: M.t + ih,
      stroke: "var(--ink-muted)", "stroke-dasharray": "3 3", opacity: 0
    });
    root.appendChild(cross);

    root.addEventListener("mousemove", function (e) {
      const r = root.getBoundingClientRect();
      const px = (e.clientX - r.left) / r.width * W;
      if (px < M.l || px > M.l + iw) { cross.setAttribute("opacity", 0); NN.tip.hide(); return; }
      const xv = x0 + (px - M.l) / iw * (x1 - x0);
      let rows = "";
      o.series.forEach(function (s) {
        let best = null, bd = Infinity;
        s.points.forEach(function (p) {
          const d = Math.abs(p[0] - xv);
          if (d < bd) { bd = d; best = p; }
        });
        if (best) rows += '<div><span style="color:' + s.color + '">●</span> ' +
          s.name + " &nbsp;<b>" + NN.fmt(best[1], 5) + "</b></div>";
      });
      cross.setAttribute("x1", X(Math.round(xv)));
      cross.setAttribute("x2", X(Math.round(xv)));
      cross.setAttribute("opacity", 1);
      NN.tip.show("<div class='muted'>" + (o.xlabel || "x") + " " +
        Math.round(xv) + "</div>" + rows, e.clientX, e.clientY);
    });
    root.addEventListener("mouseleave", function () {
      cross.setAttribute("opacity", 0); NN.tip.hide();
    });

    return root;
  };

  /**
   * Horizontal bar chart. Used for memory breakdowns everywhere.
   *   NN.chart.bars({ items:[{label,value,color,sub}], unit:'GB', max, limit:{value,label} })
   */
  NN.chart.bars = function (o) {
    const max = o.max || Math.max.apply(null, o.items.map(function (i) { return i.value; })) * 1.05;
    const wrap = NN.el("div", { class: "barchart" });
    o.items.forEach(function (it) {
      const pct = NN.clamp(it.value / max * 100, 0, 100);
      wrap.appendChild(NN.el("div", { style: { margin: "9px 0" } }, [
        NN.el("div", {
          style: {
            display: "flex", justifyContent: "space-between",
            fontSize: ".8rem", marginBottom: "3px", fontFamily: "var(--mono)"
          }
        }, [
          NN.el("span", {}, it.label),
          NN.el("span", { class: "muted" },
            NN.fmt(it.value, 1) + (o.unit ? " " + o.unit : "") + (it.sub ? "  " + it.sub : ""))
        ]),
        NN.el("div", {
          style: {
            height: "16px", background: "var(--surface-3)",
            borderRadius: "4px", overflow: "hidden", position: "relative"
          }
        }, [
          NN.el("div", {
            style: {
              width: pct + "%", height: "100%", background: it.color,
              borderRadius: "0 4px 4px 0", transition: "width .5s cubic-bezier(.4,0,.2,1)"
            }
          })
        ])
      ]));
    });
    if (o.limit) {
      wrap.appendChild(NN.el("div", {
        class: "small",
        style: { marginTop: "8px", color: "var(--critical)", fontFamily: "var(--mono)" }
      }, "── " + o.limit.label + " = " + NN.fmt(o.limit.value, 0) +
         (o.unit ? " " + o.unit : "")));
    }
    return wrap;
  };

  /* ==================================================================== */
  /*  GPU / COLLECTIVE PRIMITIVES                                         */
  /*  Shared by pages 11-17 so every distributed diagram on the site      */
  /*  looks and behaves the same. Data comes from window.PARALLEL         */
  /*  (generated by code/parallel_toy.py).                                */
  /* ==================================================================== */
  NN.P = window.PARALLEL || null;

  NN.gpu = {};

  /** Colour a collective by what it moves, reusing the tensor grammar. */
  NN.gpu.opColor = function (op) {
    return {
      all_reduce: "var(--gradient)",
      reduce_scatter: "var(--gradient)",
      reduce: "var(--gradient)",
      all_gather: "var(--weight)",
      broadcast: "var(--weight)",
      scatter: "var(--activation)",
      gather: "var(--activation)",
      all_to_all: "var(--activation)",
      p2p: "var(--activation)",
      none: "var(--accent)"
    }[op] || "var(--accent)";
  };

  /**
   * A row of GPU cards.
   *   NN.gpu.grid({ count: 4, render: (g) => [el, ...], label: g => "GPU "+g })
   * Returns an element; .cards is the array of card elements so a caller
   * can highlight or animate them.
   */
  NN.gpu.grid = function (o) {
    const n = o.count || 4;
    const wrap = NN.el("div", { class: "gpugrid" });
    const cards = [];
    for (let g = 0; g < n; g++) {
      const body = NN.el("div", { class: "gpu-body" });
      const kids = o.render ? o.render(g) : [];
      (Array.isArray(kids) ? kids : [kids]).forEach(function (k) {
        if (k) body.appendChild(typeof k === "object" ? k
          : NN.el("div", { class: "gpu-line" }, String(k)));
      });
      const card = NN.el("div", { class: "gpu-card", "data-gpu": g }, [
        NN.el("div", { class: "gpu-head" }, o.label ? o.label(g) : "GPU " + g),
        body
      ]);
      wrap.appendChild(card);
      cards.push(card);
    }
    wrap.cards = cards;
    /** Ring the given GPU indices; pass null to clear. */
    wrap.highlight = function (idx) {
      cards.forEach(function (c, i) {
        c.classList.toggle("hot", !!idx && idx.indexOf(i) >= 0);
        c.classList.toggle("dim", !!idx && idx.indexOf(i) < 0);
      });
    };
    return wrap;
  };

  /** A labelled shard bar: which slice of a tensor a GPU owns. */
  NN.gpu.shard = function (o) {
    const n = o.parts || 4, mine = o.mine;
    const bar = NN.el("div", { class: "shardbar" });
    for (let i = 0; i < n; i++) {
      bar.appendChild(NN.el("div", {
        class: "shardseg" + (i === mine ? " mine" : ""),
        style: { background: i === mine ? (o.color || "var(--weight)") : "var(--surface-3)" },
        title: o.name + " shard " + i + (i === mine ? " (held here)" : " (elsewhere)")
      }));
    }
    return NN.el("div", { class: "shardrow" }, [
      NN.el("span", { class: "shardname" }, o.name), bar
    ]);
  };

  /**
   * Format one entry of a strategy's `schedule` array as a readable card.
   * Every page in Part III renders comm steps this way.
   */
  NN.gpu.commCard = function (c, opts) {
    const col = NN.gpu.opColor(c.op);
    const isNoop = c.op === "none";
    return NN.el("div", {
      class: "commcard" + (isNoop ? " noop" : ""),
      style: { borderLeftColor: col }
    }, [
      NN.el("div", { class: "commhead" }, [
        NN.el("span", { class: "commop", style: { color: col } },
          isNoop ? "no communication" : c.op),
        c.tensor ? NN.el("span", { class: "commtensor" }, c.tensor) : null,
        !isNoop ? NN.el("span", { class: "commbytes" },
          NN.count(c.elements) + " elems · " +
          NN.fmt(c.sent_per_gpu_elements, 1) + " sent/GPU") : null
      ]),
      NN.el("div", { class: "commwhy" }, c.why),
      c.phase ? NN.el("div", { class: "commphase" }, c.phase) : null
    ]);
  };

  /**
   * Ring all-reduce visualiser. Draws `n` nodes on a circle with arrows to
   * the right-hand neighbour, and exposes .step(k) to advance the animation
   * through the 2(N-1) phases.
   */
  NN.gpu.ring = function (o) {
    const n = o.count || 4, R = o.r || 92, S = o.size || 260;
    const cx = S / 2, cy = S / 2;
    const root = svg("svg", {
      viewBox: "0 0 " + S + " " + S, width: S, height: S, class: "ringviz"
    });
    const pos = [];
    for (let i = 0; i < n; i++) {
      const a = -Math.PI / 2 + i * 2 * Math.PI / n;
      pos.push([cx + R * Math.cos(a), cy + R * Math.sin(a)]);
    }
    const arrows = [];
    for (let i = 0; i < n; i++) {
      const [x1, y1] = pos[i], [x2, y2] = pos[(i + 1) % n];
      const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
      const bulge = 1.18;
      const q = [cx + (mx - cx) * bulge, cy + (my - cy) * bulge];
      const p = svg("path", {
        d: `M${x1} ${y1} Q${q[0]} ${q[1]} ${x2} ${y2}`,
        fill: "none", stroke: "var(--accent)", "stroke-width": 2,
        opacity: 0.28, "stroke-linecap": "round"
      });
      root.appendChild(p);
      arrows.push(p);
    }
    const nodes = [];
    for (let i = 0; i < n; i++) {
      const c = svg("circle", {
        cx: pos[i][0], cy: pos[i][1], r: 21,
        fill: "var(--surface-2)", stroke: "var(--hairline)", "stroke-width": 1.5
      });
      const t = svg("text", {
        x: pos[i][0], y: pos[i][1] + 4, "text-anchor": "middle",
        fill: "var(--ink-2)", "font-size": 11
      });
      t.textContent = "G" + i;
      root.appendChild(c); root.appendChild(t);
      nodes.push({ c: c, t: t });
    }
    /** Light the arrow leaving each GPU for phase k. */
    root.step = function (k, color) {
      arrows.forEach(function (a, i) {
        const on = k !== null && k !== undefined;
        a.setAttribute("stroke", on ? (color || "var(--gradient)") : "var(--accent)");
        a.setAttribute("opacity", on ? 0.95 : 0.28);
        a.setAttribute("stroke-width", on ? 3 : 2);
      });
    };
    root.setLabels = function (labels) {
      nodes.forEach(function (nd, i) {
        if (labels && labels[i] !== undefined) nd.t.textContent = labels[i];
      });
    };
    return root;
  };

  /**
   * Pipeline occupancy grid (the bubble picture). Takes PARALLEL's
   * strategies.pp.grid and renders stage x timeslot cells.
   */
  NN.gpu.pipelineGrid = function (grid, opts) {
    const P = grid.length, T = grid[0].length;
    const cw = (opts && opts.cell) || 30, ch = 26;
    const wrap = NN.el("div", { class: "scrollx" });
    const tbl = NN.el("div", {
      class: "pipegrid",
      style: { gridTemplateColumns: `56px repeat(${T}, ${cw}px)` }
    });
    tbl.appendChild(NN.el("div", { class: "pipehdr" }, ""));
    for (let t = 0; t < T; t++) {
      tbl.appendChild(NN.el("div", { class: "pipehdr" }, String(t)));
    }
    const cells = [];
    for (let s = 0; s < P; s++) {
      tbl.appendChild(NN.el("div", { class: "pipestage" }, "stage " + s));
      cells.push([]);
      for (let t = 0; t < T; t++) {
        const v = grid[s][t];
        const kind = v ? (v[0] === "F" ? "f" : "b") : "idle";
        const c = NN.el("div", {
          class: "pipecell " + kind,
          style: { height: ch + "px" },
          title: v ? `stage ${s}, slot ${t}: ${v[0] === "F" ? "forward" : "backward"} micro-batch ${v.slice(1)}`
                   : `stage ${s}, slot ${t}: IDLE (bubble)`
        }, v || "");
        tbl.appendChild(c);
        cells[s].push(c);
      }
    }
    wrap.appendChild(tbl);
    wrap.cells = cells;
    /** Highlight one time slot. */
    wrap.mark = function (t) {
      cells.forEach(function (row) {
        row.forEach(function (c, i) { c.classList.toggle("now", i === t); });
      });
    };
    return wrap;
  };

  /* --------------------------------------------------------- bootstrap */
  /**
   * Standard page setup. Call once at the bottom of every page:
   *     NN.page('forward');
   * Injects the topbar, wires the theme toggle, and appends prev/next.
   */
  NN.page = function (id, base) {
    NN.initTheme();
    document.body.insertBefore(NN.topbar(id, base), document.body.firstChild);
    const host = NN.$("#pagenav");
    if (host) host.appendChild(NN.pagenav(id));
    const foot = NN.$("#sitefoot");
    if (foot) {
      foot.innerHTML =
        "Every number on this page is generated by " +
        "<code>code/ground_truth.py</code> and read from <code>assets/data/trace.js</code>. " +
        "Gradients are verified against finite differences to " +
        NN.fmt(NN.T.gradcheck.max_abs_error, 1) + " max absolute error.";
    }
  };

})();
