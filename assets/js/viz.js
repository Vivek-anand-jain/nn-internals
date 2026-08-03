/* ==========================================================================
   viz.js — the animation primitives, shared by the site and the deck.
   --------------------------------------------------------------------------
   This file exists because of a measured failure, not a hunch.

   An audit of the deck found 78-89% of slides carried no diagram at all and
   averaged 11-25 numeric values each. Tables outnumbered pictures in every
   part. Part II -- the transformer, the most inherently visual subject in
   the project -- was 89% diagram-free at 25 numbers per slide.

   The cause was the toolkit, not the authors. The shared kit offered
   NN.tensor (a static grid) and two chart types. Anything that needed to
   show CHANGE OVER TIME -- a cache filling, a value snapping to a level, a
   token crossing to another GPU -- had no primitive, so it became a table.
   The deck grew its own helpers privately; the site never did.

   Every helper here was needed at least three times across the project.
   That is the bar for living in the shared kit rather than in one page.

   All of them:
     * use the four-class colour grammar and never invent a fifth hue
     * carry a label or a pattern wherever colour carries meaning
     * expose .step(i) / .reset() so a Stepper or a slide fragment drives them
     * settle to a readable final frame under prefers-reduced-motion

   Load AFTER ui.js. Attaches to window.NN.viz.
   ========================================================================== */
(function () {
  "use strict";
  const NN = window.NN;
  if (!NN) { console.error("viz.js: load ui.js first"); return; }

  const SVGNS = "http://www.w3.org/2000/svg";
  const REDUCED = !!(window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches);

  function sv(tag, attrs, kids) {
    const e = document.createElementNS(SVGNS, tag);
    for (const k in attrs || {}) {
      if (attrs[k] === null || attrs[k] === undefined) continue;
      e.setAttribute(k, attrs[k]);
    }
    (kids || []).forEach(function (c) { if (c) e.appendChild(c); });
    return e;
  }
  function tx(attrs, s) { const e = sv("text", attrs); e.textContent = s; return e; }

  const viz = {};
  NN.viz = viz;
  viz.svg = sv;
  viz.text = tx;
  viz.reduced = REDUCED;

  /* The four classes, plus a neutral. A glyph travels with every colour so
     that nothing depends on hue alone. */
  const CLS = {
    weight:     { c: "var(--weight)",     w: "var(--weight-w)",     g: "■" },
    gradient:   { c: "var(--gradient)",   w: "var(--gradient-w)",   g: "▼" },
    activation: { c: "var(--activation)", w: "var(--activation-w)", g: "●" },
    optimizer:  { c: "var(--optimizer)",  w: "var(--optimizer-w)",  g: "◆" },
    neutral:    { c: "var(--accent)",     w: "transparent",         g: "" }
  };
  viz.cls = function (k) { return CLS[k] || CLS.neutral; };

  /* ====================================================================== */
  /*  grow — a buffer that gains a row per step                             */
  /*                                                                        */
  /*  The KV cache filling, activations accumulating through a forward       */
  /*  pass, a gradient buffer appearing during backward. Replaces the        */
  /*  96-number dump that prompted this file.                               */
  /*                                                                        */
  /*    const g = NN.viz.grow({ rows, colLabels, rowLabels, cls, caption });  */
  /*    g.step(k)   // first k rows resident, newest pulsing                 */
  /* ====================================================================== */
  viz.grow = function (o) {
    const rows = o.rows || [];
    const nc = rows.length ? rows[0].length : 0;
    const cls = viz.cls(o.cls || "activation");
    const wrap = NN.el("div", { class: "vgrow" });
    if (o.title) wrap.appendChild(NN.el("div", { class: "vgrow-t" }, o.title));

    const body = NN.el("div", {
      class: "vgrow-body",
      style: {
        gridTemplateColumns: (o.rowLabels ? "auto " : "") +
          "repeat(" + nc + ", minmax(0, 1fr))"
      }
    });
    if (o.colLabels) {
      if (o.rowLabels) body.appendChild(NN.el("div", { class: "vgrow-h" }, ""));
      o.colLabels.forEach(function (c) {
        body.appendChild(NN.el("div", { class: "vgrow-h" }, c));
      });
    }
    const lines = [];
    rows.forEach(function (r, i) {
      const line = { cells: [], lab: null };
      if (o.rowLabels) {
        line.lab = NN.el("div", { class: "vgrow-rl" }, o.rowLabels[i]);
        body.appendChild(line.lab);
      }
      r.forEach(function (v) {
        const c = NN.el("div", {
          class: "vgrow-c",
          style: { color: cls.c, background: cls.w }
        }, NN.fmt(v, o.digits === undefined ? 3 : o.digits));
        body.appendChild(c); line.cells.push(c);
      });
      lines.push(line);
    });
    wrap.appendChild(body);
    const cap = NN.el("div", { class: "vgrow-cap" }, "");
    wrap.appendChild(cap);

    wrap.step = function (k) {
      lines.forEach(function (ln, i) {
        const on = i < k, fresh = (i === k - 1) && !REDUCED;
        ln.cells.forEach(function (c) {
          c.classList.toggle("off", !on);
          c.classList.toggle("fresh", fresh);
        });
        if (ln.lab) {
          ln.lab.classList.toggle("off", !on);
          ln.lab.classList.toggle("fresh", fresh);
        }
      });
      cap.textContent = o.caption ? o.caption(k)
        : (k + " of " + rows.length + " rows resident");
    };
    wrap.reset = function () { wrap.step(0); };
    wrap.step(REDUCED ? rows.length : 0);
    return wrap;
  };

  /* ====================================================================== */
  /*  roofline — the plot that explains inference                           */
  /*                                                                        */
  /*  roof = min(peak, bandwidth x intensity). Everything left of the ridge  */
  /*  is bandwidth-bound and CANNOT reach peak however hard it tries.        */
  /* ====================================================================== */
  viz.roofline = function (o) {
    const W = o.width || 640, H = o.height || 330;
    const M = { t: 18, r: 20, b: 42, l: 66 };
    const iw = W - M.l - M.r, ih = H - M.t - M.b;
    const peak = o.peakFlops, bw = o.bwBytes, ridge = peak / bw;

    const x0 = Math.log10(o.xMin || 0.1), x1 = Math.log10(o.xMax || 1e5);
    const y1 = Math.log10(peak * 1.8), y0 = Math.log10(peak / 4000);
    const X = function (v) {
      return M.l + (Math.log10(Math.max(v, 1e-9)) - x0) / (x1 - x0) * iw;
    };
    const Y = function (v) {
      return M.t + ih - (Math.log10(Math.max(v, 1e-9)) - y0) / (y1 - y0) * ih;
    };

    const root = sv("svg", {
      class: "vroof", viewBox: "0 0 " + W + " " + H, width: "100%", height: H,
      role: "img"
    });

    /* the bandwidth-bound half, shaded AND named */
    root.appendChild(sv("rect", {
      x: M.l, y: M.t, width: Math.max(0, X(ridge) - M.l), height: ih,
      fill: "var(--gradient)", opacity: 0.07
    }));
    root.appendChild(tx({
      x: (M.l + X(ridge)) / 2, y: M.t + 15, "text-anchor": "middle",
      fill: "var(--ink-muted)", "font-size": 11
    }, "memory-bound — peak is unreachable here"));

    for (let d = Math.ceil(x0); d <= Math.floor(x1); d++) {
      root.appendChild(sv("line", {
        x1: X(Math.pow(10, d)), x2: X(Math.pow(10, d)), y1: M.t, y2: M.t + ih,
        stroke: "var(--grid)", "stroke-width": 1
      }));
      root.appendChild(tx({
        x: X(Math.pow(10, d)), y: M.t + ih + 16, "text-anchor": "middle",
        fill: "var(--ink-muted)", "font-size": 10
      }, "1e" + d));
    }

    const pts = [];
    for (let k = 0; k <= 140; k++) {
      const I = Math.pow(10, x0 + (x1 - x0) * k / 140);
      pts.push((k ? "L" : "M") + X(I).toFixed(1) + " " +
               Y(Math.min(peak, bw * I)).toFixed(1));
    }
    root.appendChild(sv("path", {
      d: pts.join(" "), fill: "none", stroke: "var(--ink-2)",
      "stroke-width": 2.5, "stroke-linejoin": "round"
    }));

    root.appendChild(sv("line", {
      x1: X(ridge), x2: X(ridge), y1: M.t, y2: M.t + ih,
      stroke: "var(--ink-muted)", "stroke-dasharray": "4 4"
    }));
    root.appendChild(tx({
      x: X(ridge) + 6, y: M.t + ih - 8, fill: "var(--ink-muted)", "font-size": 11
    }, "ridge " + NN.fmt(ridge, 1) + " FLOP/byte"));

    const marks = (o.points || []).map(function (p) {
      const c = viz.cls(p.cls || "neutral");
      const g = sv("g");
      const yv = Math.min(peak, bw * p.x);
      const dot = sv("circle", {
        cx: X(p.x), cy: Y(yv), r: 7, fill: c.c,
        stroke: "var(--surface-1)", "stroke-width": 2
      });
      const lab = tx({
        x: X(p.x), y: Y(yv) - 14, "text-anchor": "middle",
        fill: c.c, "font-size": 11, "font-weight": 700
      }, p.label);
      const sub = tx({
        x: X(p.x), y: Y(yv) + 22, "text-anchor": "middle",
        fill: "var(--ink-muted)", "font-size": 10
      }, p.sub || "");
      g.appendChild(dot); g.appendChild(lab); g.appendChild(sub);
      root.appendChild(g);
      return { g: g, dot: dot, lab: lab, sub: sub };
    });

    root.appendChild(tx({
      x: M.l + iw / 2, y: H - 6, "text-anchor": "middle",
      fill: "var(--ink-muted)", "font-size": 11
    }, "arithmetic intensity — FLOP per byte moved"));

    root.setPoints = function (list) {
      marks.forEach(function (m, i) {
        const p = list[i];
        if (!p) { m.g.setAttribute("opacity", 0); return; }
        m.g.setAttribute("opacity", 1);
        const yv = Math.min(peak, bw * p.x);
        m.dot.setAttribute("cx", X(p.x)); m.dot.setAttribute("cy", Y(yv));
        m.lab.setAttribute("x", X(p.x)); m.lab.setAttribute("y", Y(yv) - 14);
        m.lab.textContent = p.label;
        m.sub.setAttribute("x", X(p.x)); m.sub.setAttribute("y", Y(yv) + 22);
        m.sub.textContent = p.sub || "";
      });
    };
    /* Reveal points one at a time, so a section separator can carry the same
       plot forward and add the dot this section is about. step(null) = all. */
    root.step = function (k) {
      marks.forEach(function (m, i) {
        m.g.setAttribute("opacity",
          (k === null || k === undefined || i < k) ? 1 : 0);
      });
    };
    root.reset = function () { root.step(0); };
    root.ridge = ridge;
    return root;
  };

  /* ====================================================================== */
  /*  timeline — slot or stage occupancy over discrete time                 */
  /*                                                                        */
  /*  Static vs continuous batching, the pipeline bubble, memory phases.     */
  /*    NN.viz.timeline({ slots, lanes:[{label, cells:[{t,span,kind,text}]}] })*/
  /* ====================================================================== */
  viz.timeline = function (o) {
    const slots = o.slots, lanes = o.lanes || [];
    const wrap = NN.el("div", { class: "scrollx" });
    const grid = NN.el("div", {
      class: "vtl",
      style: {
        gridTemplateColumns: "auto repeat(" + slots + ", " + (o.cell || 15) + "px)"
      }
    });
    grid.appendChild(NN.el("div", { class: "vtl-h" }, ""));
    for (let t = 0; t < slots; t++) {
      grid.appendChild(NN.el("div", { class: "vtl-h" },
        (o.tickEvery && t % o.tickEvery === 0) ? String(t) : ""));
    }
    const rowsEls = [];
    lanes.forEach(function (ln) {
      grid.appendChild(NN.el("div", { class: "vtl-l" }, ln.label));
      const occupied = new Array(slots).fill(null);
      (ln.cells || []).forEach(function (c) {
        for (let k = 0; k < (c.span || 1); k++) {
          if (c.t + k < slots) occupied[c.t + k] = c;
        }
      });
      const els = [];
      for (let t = 0; t < slots; t++) {
        const c = occupied[t];
        const kind = c ? (c.kind || "busy") : "idle";
        const first = c && occupied[t - 1] !== c;
        const e = NN.el("div", {
          class: "vtl-c k-" + kind,
          title: (ln.label + " @ " + t + ": " + (c ? (c.text || kind) : "idle"))
        }, first && c.text ? c.text : "");
        grid.appendChild(e); els.push(e);
      }
      rowsEls.push(els);
    });
    wrap.appendChild(grid);
    if (o.legend !== false) {
      wrap.appendChild(NN.el("div", { class: "vtl-key" }, (o.legend || [
        { kind: "prefill", label: "prefill" },
        { kind: "decode", label: "decode" },
        { kind: "idle", label: "idle / blocked" }
      ]).map(function (k) {
        return NN.el("span", { class: "vtl-kk k-" + k.kind }, k.label);
      })));
    }
    let timer = null;
    wrap.mark = function (t) {
      rowsEls.forEach(function (els) {
        els.forEach(function (e, i) { e.classList.toggle("now", i === t); });
      });
    };
    wrap.play = function (ms) {
      let t = 0; clearInterval(timer);
      if (REDUCED) { wrap.mark(slots - 1); return; }
      timer = setInterval(function () {
        wrap.mark(t++);
        if (t >= slots) clearInterval(timer);
      }, ms || 55);
    };
    wrap.stop = function () { clearInterval(timer); };
    wrap.step = function (t) { clearInterval(timer); wrap.mark(t); };
    wrap.reset = function () { clearInterval(timer); wrap.mark(-1); };
    return wrap;
  };

  /* ====================================================================== */
  /*  flow — labelled chunks physically moving between boxes                */
  /*                                                                        */
  /*  Every collective, the MoE dispatch, a pipeline hand-off. This is the   */
  /*  primitive the site never had, which is why all-to-all was a matrix.    */
  /* ====================================================================== */
  viz.flow = function (o) {
    const nodes = o.nodes || [], hops = o.hops || [];
    const W = o.width || 640, H = o.height || 210;
    const bw = o.boxW || 92, bh = o.boxH || 52;
    const gap = (W - nodes.length * bw) / (nodes.length + 1);
    const top = H / 2 - bh / 2;
    const pos = nodes.map(function (_, i) {
      const x = gap + i * (bw + gap);
      return { x: x, cx: x + bw / 2 };
    });

    const root = sv("svg", {
      class: "vflow", viewBox: "0 0 " + W + " " + H, width: "100%", height: H
    });

    /* A hop may carry a weight — a token count, a byte count. MoE's all-to-all
       is RAGGED, and that is its whole difficulty: you cannot size the buffers
       until the router has run. Equal-width arcs cannot say that, so width is
       proportional to weight, normalised against the fattest hop. */
    let maxW = 0;
    hops.forEach(function (h) {
      if (h.weight !== undefined && h.weight > maxW) maxW = h.weight;
    });

    const arcs = hops.map(function (h) {
      const a = pos[h.from], b = pos[h.to];
      const sw = (maxW && h.weight !== undefined)
        ? 1.2 + 3.4 * (h.weight / maxW) : 2;
      const above = h.from <= h.to;
      const lane = (h.lane || 0) * 14;
      const y = above ? top - 6 - lane : top + bh + 6 + lane;
      const peakY = above ? top - 44 - lane : top + bh + 44 + lane;
      const d = (h.from === h.to)
        ? "M" + a.cx + " " + (top - 6) + " a 16 16 0 1 1 0.1 0"
        : "M" + a.cx + " " + y + " Q " + ((a.cx + b.cx) / 2) + " " + peakY +
          " " + b.cx + " " + y;
      const col = h.kind === "local" ? "var(--accent)"
                                     : viz.cls(h.cls || "activation").c;
      const path = sv("path", {
        d: d, fill: "none", stroke: col, "stroke-width": sw, opacity: 0.22,
        "stroke-dasharray": h.kind === "local" ? "4 4" : null
      });
      if (h.weight !== undefined) {
        path.appendChild(sv("title", {}));
        path.lastChild.textContent =
          (h.label ? h.label + " — " : "") + h.weight + " " + (o.unit || "items");
      }
      const dot = sv("circle", { r: 5, fill: col, opacity: 0 });
      const lab = tx({
        "text-anchor": "middle", "font-size": 10, fill: col, opacity: 0
      }, h.label || "");
      root.appendChild(path); root.appendChild(dot); root.appendChild(lab);
      return { path: path, dot: dot, lab: lab, h: h, sw: sw };
    });

    const boxes = nodes.map(function (n, i) {
      const g = sv("g");
      const r = sv("rect", {
        x: pos[i].x, y: top, width: bw, height: bh, rx: 8,
        fill: "var(--surface-2)", stroke: "var(--hairline)", "stroke-width": 1.5
      });
      g.appendChild(r);
      g.appendChild(tx({
        x: pos[i].cx, y: top + bh / 2 + 4, "text-anchor": "middle",
        fill: "var(--ink-2)", "font-size": 12
      }, n.label));
      if (n.sub) g.appendChild(tx({
        x: pos[i].cx, y: top + bh + 16, "text-anchor": "middle",
        fill: "var(--ink-muted)", "font-size": 10
      }, n.sub));
      root.appendChild(g);
      return { g: g, rect: r };
    });

    root.step = function (k) {
      arcs.forEach(function (a, i) {
        const on = (k !== null && k !== undefined) && i === k;
        a.path.setAttribute("opacity", on ? 0.95 : 0.22);
        a.path.setAttribute("stroke-width", on ? a.sw + 1 : a.sw);
        a.lab.setAttribute("opacity", on ? 1 : 0);
        a.dot.setAttribute("opacity", 0);
        if (!on) return;
        let len = 0;
        try { len = a.path.getTotalLength(); } catch (e) { return; }
        let mid;
        try { mid = a.path.getPointAtLength(len / 2); } catch (e) { return; }
        a.lab.setAttribute("x", mid.x);
        a.lab.setAttribute("y", mid.y - 6);
        if (REDUCED) return;
        const dur = o.speed || 650;
        let t0 = null;
        (function frame(ts) {
          if (t0 === null) t0 = ts;
          const f = Math.min(1, (ts - t0) / dur);
          const pt = a.path.getPointAtLength(len * f);
          a.dot.setAttribute("cx", pt.x);
          a.dot.setAttribute("cy", pt.y);
          a.dot.setAttribute("opacity", 1);
          if (f < 1) requestAnimationFrame(frame);
          else a.dot.setAttribute("opacity", 0);
        })(performance.now());
      });
    };
    root.showAll = function (on) {
      arcs.forEach(function (a) {
        a.path.setAttribute("opacity", on ? 0.8 : 0.22);
        a.lab.setAttribute("opacity", on ? 1 : 0);
      });
    };
    root.highlightNode = function (idx) {
      boxes.forEach(function (b, i) {
        b.rect.setAttribute("stroke",
          i === idx ? "var(--activation)" : "var(--hairline)");
        b.rect.setAttribute("stroke-width", i === idx ? 3 : 1.5);
      });
    };
    root.reset = function () { root.step(null); root.highlightNode(null); };
    root.step(null);
    if (REDUCED) root.showAll(true);
    return root;
  };

  /* ====================================================================== */
  /*  snap — real values landing on a quantisation grid                     */
  /*                                                                        */
  /*  Shows the outlier eating the range, which is the entire argument for   */
  /*  per-channel scales and is currently a table.                          */
  /* ====================================================================== */
  viz.snap = function (o) {
    const W = o.width || 640, H = o.height || 136;
    const M = { l: 34, r: 34, t: 34, b: 36 };
    const lo = o.min, hi = o.max, iw = W - M.l - M.r;
    const X = function (v) { return M.l + (v - lo) / (hi - lo) * iw; };
    const c = viz.cls(o.cls || "weight").c;
    const levels = o.levels || [];

    const root = sv("svg", {
      class: "vsnap", viewBox: "0 0 " + W + " " + H, width: "100%", height: H
    });
    root.appendChild(sv("line", {
      x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b,
      stroke: "var(--hairline)", "stroke-width": 1.5
    }));
    levels.forEach(function (L) {
      root.appendChild(sv("line", {
        x1: X(L), x2: X(L), y1: H - M.b - 5, y2: H - M.b + 5,
        stroke: "var(--ink-muted)", "stroke-width": 1
      }));
    });
    const vals = o.values || [];
    const marks = vals.map(function (v) {
      const q = levels.length
        ? levels.reduce(function (a, b) {
            return Math.abs(b - v) < Math.abs(a - v) ? b : a;
          }, levels[0])
        : v;
      const drop = sv("line", {
        x1: X(v), x2: X(q), y1: M.t, y2: H - M.b,
        stroke: c, "stroke-width": 1, "stroke-dasharray": "3 3", opacity: 0
      });
      const dot = sv("circle", {
        cx: X(v), cy: M.t, r: 4, fill: c,
        title: NN.fmt(v, 4) + " → " + NN.fmt(q, 4)
      });
      dot.appendChild(sv("title", {}));
      dot.lastChild.textContent =
        NN.fmt(v, 4) + " snaps to " + NN.fmt(q, 4) +
        "  (error " + NN.fmt(Math.abs(v - q), 4) + ")";
      const land = sv("rect", {
        x: X(q) - 3.2, y: H - M.b - 3.2, width: 6.4, height: 6.4, fill: c,
        opacity: 0, transform: "rotate(45 " + X(q) + " " + (H - M.b) + ")"
      });
      root.appendChild(drop); root.appendChild(land); root.appendChild(dot);
      return { v: v, q: q, drop: drop, dot: dot, land: land };
    });

    root.appendChild(tx({
      x: M.l, y: M.t - 12, fill: "var(--ink-muted)", "font-size": 10
    }, o.topLabel || "real values"));
    root.appendChild(tx({
      x: M.l, y: H - 8, fill: "var(--ink-muted)", "font-size": 10
    }, o.bottomLabel || (levels.length + " representable levels")));

    /* The landing is the whole point. A value that is merely DRAWN at its
       quantised position has already lost the argument — the reader has to
       see it leave the real line and arrive somewhere it did not start. */
    root.step = function (k) {
      const n = (k === null || k === undefined) ? marks.length : k;
      marks.forEach(function (m, i) {
        const landed = i < n;
        m.drop.setAttribute("opacity", landed ? 0.55 : 0);
        m.land.setAttribute("opacity", landed ? 1 : 0);
        m.dot.setAttribute("cx", X(m.v));
        m.dot.setAttribute("cy", M.t);
        m.dot.setAttribute("opacity", landed ? 0.45 : 1);
      });
      if (REDUCED || n === 0 || n > marks.length) return;
      const m = marks[n - 1];               /* animate only the newest */
      const x0v = X(m.v), y0v = M.t, x1v = X(m.q), y1v = H - M.b;
      m.dot.setAttribute("opacity", 1);
      let t0 = null;
      const dur = o.speed || 420;
      (function frame(ts) {
        if (t0 === null) t0 = ts;
        const f = Math.min(1, (ts - t0) / dur);
        const e = 1 - Math.pow(1 - f, 3);
        m.dot.setAttribute("cx", x0v + (x1v - x0v) * e);
        m.dot.setAttribute("cy", y0v + (y1v - y0v) * e);
        if (f < 1) requestAnimationFrame(frame);
        else {
          m.dot.setAttribute("cx", x0v); m.dot.setAttribute("cy", y0v);
          m.dot.setAttribute("opacity", 0.45);
        }
      })(performance.now());
    };
    root.reset = function () { root.step(0); };
    root.step(REDUCED ? marks.length : (o.landed === false ? 0 : marks.length));
    return root;
  };

  /* ====================================================================== */
  /*  chain — an accept/reject sequence with rollback                       */
  /*                                                                        */
  /*  Speculative decoding, and any propose-then-verify loop. Rejection      */
  /*  discards everything after it, which a table cannot show.              */
  /* ====================================================================== */
  viz.chain = function (o) {
    const toks = o.tokens || [];
    const wrap = NN.el("div", { class: "vchain" });
    const row = NN.el("div", { class: "vchain-row" });
    const els = toks.map(function (t, i) {
      const e = NN.el("div", { class: "vchain-t pending" }, [
        NN.el("span", { class: "vchain-i" }, String(i + 1)),
        NN.el("span", { class: "vchain-x" }, t.text)
      ]);
      row.appendChild(e); return e;
    });
    wrap.appendChild(row);
    const cap = NN.el("div", { class: "vchain-cap" }, "");
    wrap.appendChild(cap);
    wrap.step = function (k) {
      let stopped = false, acc = 0;
      els.forEach(function (e, i) {
        e.className = "vchain-t";
        if (i >= k) { e.classList.add("pending"); return; }
        if (stopped) { e.classList.add("discarded"); e.title = "discarded — a rejection upstream invalidates everything after it"; return; }
        if (toks[i].accepted) { e.classList.add("accepted"); e.title = "accepted"; acc++; }
        else { e.classList.add("rejected"); e.title = "rejected — the chain stops here"; stopped = true; }
      });
      cap.textContent = o.caption ? o.caption(acc, k)
        : (acc + " of " + k + " proposed tokens kept");
    };
    wrap.reset = function () { wrap.step(0); };
    wrap.step(REDUCED ? els.length : 0);
    return wrap;
  };

  /* ====================================================================== */
  /*  cut — a matrix with an animated partition                             */
  /*                                                                        */
  /*  Column- and row-parallel splits, and ZeRO's shape-blind flat shards    */
  /*  whose boundaries deliberately ignore matrix edges.                    */
  /* ====================================================================== */
  viz.cut = function (o) {
    const m = o.matrix, nr = m.length, nc = m[0].length;
    const parts = o.parts || 2;
    const wrap = NN.el("div", { class: "vcut" });
    if (o.title) wrap.appendChild(NN.el("div", { class: "vcut-t" }, o.title));
    const grid = NN.el("div", {
      class: "vcut-g",
      style: { gridTemplateColumns: "repeat(" + nc + ", auto)" }
    });
    const owners = [];
    for (let i = 0; i < nr; i++) {
      owners.push([]);
      for (let j = 0; j < nc; j++) {
        const own = o.mode === "column" ? Math.floor(j / (nc / parts))
                  : o.mode === "row" ? Math.floor(i / (nr / parts))
                  : Math.floor((i * nc + j) / ((nr * nc) / parts));
        const c = NN.el("div", {
          class: "vcut-c", "data-own": String(own),
          title: (o.ownerLabel ? o.ownerLabel(own) : "GPU " + own)
        }, NN.fmt(m[i][j], o.digits === undefined ? 3 : o.digits));
        grid.appendChild(c);
        owners[i].push(c);
      }
    }
    wrap.appendChild(grid);
    const key = NN.el("div", { class: "vcut-key" });
    for (let p = 0; p < parts; p++) {
      key.appendChild(NN.el("span", { class: "vcut-kk", "data-own": String(p) },
        o.ownerLabel ? o.ownerLabel(p) : ("GPU " + p)));
    }
    wrap.appendChild(key);
    wrap.show = function (on) {
      owners.forEach(function (r) {
        r.forEach(function (c) { c.classList.toggle("split", !!on); });
      });
    };
    wrap.isolate = function (p) {
      owners.forEach(function (r) {
        r.forEach(function (c) {
          c.classList.toggle("dim",
            p !== null && p !== undefined && +c.getAttribute("data-own") !== p);
        });
      });
    };
    /* 0 = the whole matrix, undivided (this matters: the reader must see the
       thing before it is cut). 1 = the partition drops in. 2+n = isolate the
       nth owner, which is how "a GPU holds only this" gets said. */
    wrap.step = function (k) {
      const n = (k === null || k === undefined) ? 1 : k;
      if (n <= 0) { wrap.show(false); wrap.isolate(null); return; }
      wrap.show(true);
      wrap.isolate(n >= 2 ? Math.min(n - 2, parts - 1) : null);
    };
    wrap.reset = function () { wrap.step(0); };
    wrap.show(REDUCED);
    return wrap;
  };

  /* ====================================================================== */
  /*  heat — a heatmap with hover, a stated ramp, and hatched masking       */
  /*                                                                        */
  /*  Attention scores, causal masks, routing weights. Single-hue            */
  /*  sequential ramp; never a rainbow. Masked cells are hatched AND         */
  /*  labelled, so the mask never rests on colour.                          */
  /* ====================================================================== */
  viz.heat = function (o) {
    const m = o.matrix, nr = m.length, nc = m[0].length;
    const isMasked = o.masked || function () { return false; };
    let mx = o.max || 0;
    if (!o.max) {
      m.forEach(function (r) {
        r.forEach(function (v) { if (typeof v === "number" && v > mx) mx = v; });
      });
    }
    mx = mx || 1;
    const base = viz.cls(o.cls || "activation").c;
    const wrap = NN.el("div", { class: "vheat" });
    const grid = NN.el("div", {
      class: "vheat-g",
      style: {
        gridTemplateColumns: (o.rowLabels ? "auto " : "") +
          "repeat(" + nc + ", " + (o.cell || 44) + "px)"
      }
    });
    const cells = [];
    for (let i = 0; i < nr; i++) {
      if (o.rowLabels) {
        grid.appendChild(NN.el("div", { class: "vheat-rl" }, o.rowLabels[i]));
      }
      cells.push([]);
      for (let j = 0; j < nc; j++) {
        const v = m[i][j];
        const mk = isMasked(i, j) || v === null || v === undefined;
        const f = mk ? 0 : Math.max(0, Math.min(1, v / mx));
        const rl = o.rowLabels ? o.rowLabels[i] : i;
        const cl = o.colLabels ? o.colLabels[j] : j;
        const c = NN.el("div", {
          class: "vheat-c" + (mk ? " masked" : ""),
          style: mk ? {} : { background: base, opacity: 0.10 + 0.85 * f },
          title: mk ? (rl + " → " + cl + ": masked, cannot attend")
                    : (rl + " → " + cl + " = " + NN.fmt(v, 4))
        }, mk ? "−∞" : (o.showValues === false ? "" : NN.fmt(v, 2)));
        grid.appendChild(c); cells[i].push(c);
      }
    }
    wrap.appendChild(grid);
    if (o.key !== false) {
      const ramp = NN.el("div", { class: "vheat-ramp" });
      for (let k = 0; k <= 8; k++) {
        ramp.appendChild(NN.el("i", {
          style: { background: base, opacity: 0.10 + 0.85 * k / 8 }
        }));
      }
      wrap.appendChild(NN.el("div", { class: "vheat-key" }, [
        NN.el("span", {}, "0"), ramp, NN.el("span", {}, NN.fmt(mx, 2)),
        NN.el("span", { class: "vheat-mk" }, "hatched = masked")
      ]));
    }
    wrap.highlight = function (pairs) {
      cells.forEach(function (r, i) {
        r.forEach(function (c, j) {
          c.classList.toggle("hot", !!pairs &&
            pairs.some(function (p) { return p[0] === i && p[1] === j; }));
        });
      });
    };
    /* Reveal row by row by default — a score matrix built one query at a time
       is a causal mask being obeyed, not merely depicted. Pass by:'cell' when
       the fill order itself is the lesson (a tile being computed). */
    const order = o.by === "cell" ? nr * nc : nr;
    wrap.step = function (k) {
      const n = (k === null || k === undefined) ? order : k;
      cells.forEach(function (r, i) {
        r.forEach(function (c, j) {
          const idx = o.by === "cell" ? i * nc + j : i;
          c.classList.toggle("unseen", idx >= n);
        });
      });
    };
    wrap.reset = function () { wrap.step(0); };
    wrap.step(order);
    return wrap;
  };

})();
