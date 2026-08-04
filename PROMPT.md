# Prompt — for Claude, Cursor, or any coding agent

Paste everything below the line. It is self-contained; it assumes no knowledge
of this conversation.

---

## The job

Build one chapter of a visual reference on neural-network and LLM internals.

The reference already exists and is wrong in a specific way. It is accurate,
verified, and **unreadable**: it explains mechanisms with tables of numbers
instead of showing them happening. Your job is not to add pictures to text. It
is to make the mechanism visible, and let the text be a caption.

## The one rule that matters

> **If a reader can only learn it by reading numbers off a grid, it is not
> built yet.**

A table is what you write when you have a fact and no picture. Every table in
this project is a failure to visualise something. Before you write a `<table>`,
you must be able to say why the thing genuinely has no shape, no motion, no
before-and-after, and no spatial arrangement.

Almost nothing passes that test. Some things that look like tables are not:

| what it looks like | what it actually is | how to draw it |
|---|---|---|
| a list of per-GPU values | one buffer, cut | the buffer, with partition lines dropping in |
| before / after columns | a transition | the same object, animating between two states |
| a value at each step | a trajectory | a line, with the reader scrubbing it |
| which GPU sends what | routing | chunks physically travelling between boxes |
| a quantised value | a landing | a dot leaving a real value and arriving on a grid level |
| rows summing to zero | an invariant | a bar chart whose bars visibly cancel |
| 4 of 20 tiles are rescales | occupancy | a grid with 4 cells lit |

## The standard: teach the mechanism on a toy you can actually see

The reference has small verified models — a 13-parameter MLP, a 288-parameter
2-layer transformer, a 193-parameter MLP on 4 simulated GPUs. Every number
displayed anywhere is read from a generated data file, never typed by hand.

Use them. The point of a 4×4 matrix is that the reader can watch **every single
element** move. Do not draw a schematic of an abstract matrix when you could
draw the real one with its real 16 numbers and animate the actual operation.

**Worked example of the standard — "tensor parallelism" done right:**

Wrong (what the project currently does): a table listing which weight matrix
gets a column cut and which gets a row cut, and a paragraph saying partial sums
must be all-reduced.

Right: the real 4×4 `W` on screen. A partition line drops between columns 1 and
2. The two halves tint differently and label themselves GPU 0 / GPU 1. The
input vector broadcasts to both — you watch it arrive twice. Each GPU produces
half the output; the reader sees GPU 0 holds columns 0–1 of the answer and
nothing else. Then the second matrix is cut by *rows*, and now each GPU's output
is a **complete-shaped but numerically wrong** partial — show both partials, and
show that neither equals the right answer. Then the all-reduce fires: chunks
travel between the two boxes, the partials add, and the result now matches the
single-GPU reference exactly. `max |difference| = 0` appears once, small.

The reader should be able to answer, without being told: why is the pairing
always column-then-row? Why is there exactly one collective and not two?

## What "flow" means

A chapter is not a list of true statements about a topic. Each section must be
*caused* by the one before it. The test: read only the section headings. Do
they form an argument, or a table of contents?

Good shape: state the problem → show the naive thing failing → introduce the
fix → show it working on the toy → prove it equals the reference → price it →
name what it does not fix, which is the next chapter.

If a section's content would not change if you deleted the section before it,
one of them is in the wrong place.

## Hard constraints

1. **Never type a model number by hand.** Every value renders from the
   generated data object. If you need a number that isn't there, derive it in
   JS from what is, or change the generator and re-run it.
2. **No build step, no network.** Plain `<script>` tags. Must work from a
   `file://` URL.
3. **Colour never carries meaning alone.** Every coloured thing also carries a
   label or a pattern. Four tensor classes, four hues, no fifth.
4. **No dual-axis charts.** Two scales means two charts.
5. **Respect `prefers-reduced-motion`** — animations settle to a readable final
   frame instead of playing.

## Deliverable

The chapter, working. Plus three sentences: what the central picture is, what
the reader can now answer that they could not before, and what you could not
draw and why.

## How you will be judged

- Could a reader who skipped the prose still get the mechanism from the
  visuals alone?
- Is every number on screen load-bearing? Delete any number that no sentence
  refers to and no picture needs.
- Does the chapter argue, or does it list?
- Is there anything on screen that is decorative?
