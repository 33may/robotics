# Layer-1 Readout Bench — Design

**Date:** 2026-08-24 · **Ticket:** MAY-186 (feeds MAY-187) · **Status:** draft for Anton's review

## Goal

A harness that measures the inspection subagent (`inspect_view`) against ground
truth, so that every prompt/tool/model change is accepted or rejected on a
number instead of an impression. The bench is the fitness function for the
self-improvement loop; without it, refining the subagent is guesswork.

The cup and the logo are an *instance*, not the design. The bench is generic
over (object, question): cases are `(run, question)` pairs, labels are
per-view visibility verdicts, and nothing in the harness code knows what a
cup is. Questions live in label files, not in code.

## Unit under test

`inspect_view(tools, verbs, writer, model, cell, task)` — one captured frame,
one question, one finding. A **bench case** = one (capture, question) pair.

The run-level answer is derived for free by applying the answer policy over
all views of a run (one positive settles yes; coverage argues absence). This
"exhaustive baseline" is also the floor the future MAY-187 orchestrator must
beat with fewer views.

## Label schema (settled with Anton, 2026-08-24)

Per **(run, question)**:

```json
{
  "question": "is there a logo on the mug?",
  "run_answer": "yes",
  "expected_string": null,
  "per_view": {
    "007": {"verdict": "full",    "source": "model"},
    "012": {"verdict": "partial", "source": "human"},
    "015": {"verdict": "none",    "source": "rule"}
  }
}
```

- **Verdict vocabulary — exactly three:** `full` (feature fully visible),
  `partial` (partly visible / oblique / clipped), `none` (not visible in this
  frame). Anton's call: no fourth value for now.
- **Frame-scoped, always.** The label answers "what does THIS image show",
  never "what is true of the object". Object-scoped labels degenerate (an
  honest one-frame verdict could never say no); frame-scoped `none` is the
  atom the orchestrator later composes into object-level absence.
- **Question-relative.** For "is there a logo?", `partial` = part of the print
  visible. For "what does the text say?", `full` = text legible enough to
  read. Same vocabulary, meaning set by the question — this is what keeps the
  schema generic.
- **`source` is first-class:** `rule` | `model` | `human`. We always know how
  much ground truth is human-verified.
- **No direction hint.** Ruled out by Anton: no lever (we prompt-iterate, we
  don't train) and derivable in code from the fragment box + pose if MAY-187
  ever wants it.
- `expected_string` only for reading-type questions (property of the object,
  never of a frame). `run_answer` is the true object-level answer.

### Subagent output vs label

The subagent keeps its **4-way** frame-scoped verdict `yes / partial / no /
unseen` — the no/unseen split has a real consumer (the orchestrator's
coverage logic). The bench collapses it for scoring:

| subagent says | scored as |
|---------------|-----------|
| yes           | full      |
| partial       | partial   |
| no, unseen    | none      |

Consequence: the *absence-overclaim* metric (saying `no` when the honest
verdict is `unseen`) is **deferred** until labels split `none`. Accepted trade.

### Forced subagent changes (bench-driven, not taste)

1. `answer` becomes structured: `{"verdict": "<yes|partial|no|unseen>",
   "answer": "<free text>"}` — the enum is what the bench grades.
2. The prompt states explicitly that the verdict is about THIS frame even
   though the task is object-scoped — otherwise the model hedges `unseen` on
   clean back-side views.

## Datasets and seed questions

| run | object | caps | role |
|-----|--------|------|------|
| 2408-cup3 | black mug, logo | 30 | dev |
| 2408-cup4 | same mug, moved | 33 | **holdout** (untouched during iteration) |
| 2408-cup5 | white mug, no logo | 30 | dev — all-`none` by rule, free FP trap |

Seed questions (extendable by editing label files only):

- Q1 `"is there a logo on the mug?"` — all three runs. `run_answer`:
  yes / yes / no.
- Q2 `"what does the text on the mug say?"` — cup3, cup4.
  `expected_string: "VBTI"`; scored on `full`-labelled views only.

cup5 × Q1 is labelled by one rule (`none` everywhere): every hallucinated
positive there is an error caught with zero labelling effort. An all-negative
run is worth more than a labelled positive one — one FP flips the entire run
answer under our policy.

Each future captured run enters as holdout first; older holdouts roll into
dev. Guards against prompt-overfitting to ~60 frames.

## Auto-labelling pipeline

1. **Rules** — cup5×Q1 all `none`; run answers by hand (one line each).
2. **Model annotators** — two VLMs from **different families** label every
   remaining (view, question), frame-scoped, full frame + the same 3-way
   vocabulary. Agreement ⇒ label stands with `source: model`. Disagreement ⇒
   adjudication queue. Default pairing: `gemini-3.1-pro` + a Claude vision
   model (final pick at plan time — needs a key check).
3. **Human adjudication** — Anton resolves only disagreements + a random
   audit sample (~10/run) to bound the both-models-agree-and-are-wrong case.
   Estimated effort: minutes per run, not hours.

Hard line: **the bench subject's own outputs never become labels.** Annotator
≠ subject, always. Labelling frame-scoped visibility is a pure perception
question — VLMs used where they are strongest, no reasoning-level blind
spots inherited.

## Adjudication UI

The existing contact sheet (`bench/labels.py`) grows into a static HTML page:
disagreement frames only, both annotator opinions shown, keyboard `f/p/n`
per frame, "export JSON" button writes human labels. No server.

## Bench runner and metrics

`bench.py` runs the subagent over all cases for one configuration (model,
prompt version, tool policy) and writes
`inspection/data/bench/results/<name>.json`: config, git rev, per-case
verdicts + transcript refs, metric summary. A compare mode prints the delta
between two result files.

**Metrics, in priority order (each attached to a lever):**

1. **Hallucination rate** — claims `yes`/`partial` where label is `none`.
   Corrupts run-level "yes" loudly. cup5 measures it for free.
2. **Miss rate** — says `none` where label is `full`.
3. **3-way verdict accuracy** (full/partial/none).
4. **String accuracy** — reading questions, `full` views only: does the read
   match `expected_string`? (Presence and identity scored separately — the
   VBTI/YETI misread is invisible to a single combined score.)
5. **Run-level derived answer** correct per (run, question).
6. **Cost** — turns, seconds, $ per view. Reported, never optimized.

First action after building: run the baseline config **twice**; the observed
run-to-run variance is the noise floor. Improvements below it don't count.

## Improvement cycle

```
run bench → read failing transcripts → propose ONE change
  (prompt rule | tool policy | model choice)
→ re-run bench → accept iff headline metrics improve beyond noise,
  holdout does not regress
→ commit the change together with its result file
```

Human-gated: Anton approves each change. The "propose" step is automatable
later (an agent mines failing transcripts) — same loop, same gate. This is
offline distillation of failures into permanent rules — the self-improvement
pattern that works — not in-context self-correction, which degrades.

## File layout

```
inspection/eyes/bench/
  labels.py      # label store: (run, question) → verdicts + provenance (rework)
  autolabel.py   # rules + two-annotator labelling, disagreement queue
  adjudicate.py  # static HTML adjudication page generator
  bench.py       # runner + metrics + compare mode
inspection/data/bench/
  labels/<run>.json
  results/<name>.json
```

Model access goes through the existing `respond(parts, schema)` seam
(`eyes/models.py`); annotators are just more implementations of it.

## Testing

House style: plain asserts + `main()`, run as `p inspection/tests/test_X.py`,
no pytest, no network. Stub VLMs script annotator agreement/disagreement;
metric math is tested on hand-built label/result fixtures; the adjudication
page generator is smoke-tested for content, not looks.

## Out of scope

- Absence-overclaim metric (needs the none-split; deferred by the 3-way call).
- Direction hints (ruled out — no lever, derivable in code).
- Orchestrator, view selection, views-to-answer metric (MAY-187; this bench
  supplies its floor via the exhaustive baseline).
- Training/IL of any kind; levers are prompt, tool policy, model choice.
