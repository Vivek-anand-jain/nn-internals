# nn-internals — working direction

This file exists because the project drifted. Read it before touching content.

## What this is

An **uber visualization guide to neural network and LLM internals.** Slides and
site. The subject is *mechanism*: what actually happens, shown happening.

It is not a reference manual with diagrams. It is not a set of correct facts.
The verification discipline (every number generated and checked) is the
*foundation*, not the product. A reader should leave able to picture a forward
pass, a gradient arriving, a matrix cut across GPUs, a token being decoded.

## The one rule

> **If it can only be learned by reading numbers off a grid, it is not built.**

Every table is a failure to visualise something. Before writing one, name the
reason the thing has no shape, no motion, no before/after, no arrangement.
Almost nothing survives that question.

## The failure this file is a response to

Between 01:55 and 04:38 on Aug 3, fourteen commits added twelve chapters and
82 slides. Measured afterwards:

- 157 `el("table")` calls in the deck, 198 on the site
- 37% of site sections carry no visual of any kind
- the five pages with **zero** diagrams were the five topics specifically asked
  for: what-it-predicts, prefill/decode, batching, paging, MoE inference
- `NN.viz` — eight animation primitives — called **zero** times

**Root cause: briefs listed numbers to land instead of ideas to teach.** Given
"land these figures," an agent writes a table. It is the correct response to
that brief. The brief was wrong.

**Second cause: no editorial standard existed.** `AGENTS.md` specified the
palette, the data rule and the HTML skeleton — how to be *consistent*. It said
nothing about what to build. Consistency without direction produced consistent
tables.

## The two content holes to fix first

**1. Collectives are taught and then never used.**
Eight collective animations exist on one page. Then DDP, ZeRO, TP, SP, PP and
EP each *describe in prose* which collective they use. The reader never sees a
real toy matrix sitting on four GPUs with data moving between them. That is the
entire subject of Part III and it is drawn nowhere.

Every parallelism strategy owes the reader one animation of this shape:
the real toy tensor → the cut → what each GPU holds → the collective firing,
chunks travelling → the result reassembling → it matches the single-GPU answer.

**2. Inference parallelism was answered as a cost model.**
It became tables of microseconds and bandwidth. The question was how
parallelism *works* during inference: heads split across GPUs, the KV cache
sharded, the all-reduce firing on every single token, the payload now tiny.
Show the decode step on 2 GPUs the way Part III shows the training step.

## Flow

A chapter is an argument, not a list of true statements. Each section must be
caused by the previous one.

Test: read only the section headings. Do they argue, or enumerate?

Shape that works: the problem → the naive thing failing → the fix → the fix on
the toy → it equals the reference → what it costs → what it does not fix, which
is the next chapter.

There is no flow after backward propagation. Parts III–V are catalogues.

## Process rules

- **Do not expand scope.** The complaint is content quality. New architecture,
  new primitives, new pages are diversions unless asked for.
- **Detail belongs in files, not in chat.** Long analyses go in a document; the
  message says what changed and what's next.
- **Show work incrementally.** One topic, finished, shown — not five topics
  reported at the end.
- **Slides first.** Pages get a written plan until the deck is right.
- **Change generators freely** when a visual needs a field. Edit, re-run, grep
  every consumer in the same commit — field renames have broken consumers twice.

## Verification, kept in proportion

Verification is why this project is trustworthy, and it is currently eating the
reader's attention: 31 slides and sections are majority PASS-table. One tick per
claim. The worked case that is *surprising* is worth showing; twelve rows all
reading `ok` are not.
