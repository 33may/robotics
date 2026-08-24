# Image machinery — the eyes of the inspection system (design)

2026-08-20. Problem formulation for MAY-186. Companion to
`2026-08-20-ui-driven-loop-design.md`, which built the operator loop this
plugs into. **Not an implementation plan** — this fixes what the module is
for, what it owns, and where its boundaries are. Task breakdown follows
separately.

## Why this module exists

The motion system is built: the robot knows how to move, what collides, what
is reachable. What it cannot do is *see*. This module is the eyes.

It serves exactly two consumers:

1. **Answer the question** — given the views captured so far, what is the
   answer.
2. **Inform the next move** — given the views captured so far, what is worth
   looking at next. The worked example: the system inspects from several
   sides and notices that on view 12 part of a logo is visible at the right
   edge. That observation is what tells the decider to explore that area
   rather than continue a blind sweep.

The module does **not** decide where to move. It produces the understanding
that makes the decision possible.

## What it is: machinery, not a question-answerer

**The module owns no question.** It is a tool API — MCP / function-calling
shaped — providing the machinery for an AI agent to navigate our geometric
space, inspect any image, and know where it is and what is in the scene.
The agent owns the question and decides how to use the tools to answer it.

Consequence: the current demo question ("is there a logo on this cup?") is
binary, but binariness is a property of *this run's question*, not of the
machinery. Anything question-specific lives in the agent, never in the API.

Scope for v1: **one object on the table.**

## Architecture: two agent tiers

| Tier | Owns | Never does |
|---|---|---|
| **Orchestrator agent** | geometry, viewsphere, where to look, the run structure, the final answer | reads raw pixels in bulk |
| **Inspection subagent** | 2D reading of a single captured view, with its own tools | moves the robot; reasons about world geometry |

`inspect(cell, task)` is not a function call that returns a label — it is a
**subagent run**. It receives the current context, the hypothesis, and its
task; it works over one captured view with its own tools; it returns an
answer plus the content of what it saw, comments, and notes.

**Why two tiers rather than one agent with vision tools.** The subagent burns
its context on pixels and returns text, so the orchestrator never accumulates
raw images across a long run. The image-budget problem is solved
structurally rather than by imposing a cap.

### The inspection subagent

- **Never moves the robot.** It operates only on already-captured images.
  When it wants a view that does not exist, it says so as a note; the decider
  may act on it. This keeps motion budget, approval, and safety entirely
  inside the orchestrator, which already owns them.
- **May step to neighbouring captured views** (one level up / down / left /
  right). This is retrieval, not motion. A request for an uncaptured
  neighbour returns nothing — and that "nothing" is logged, because it is a
  first-class next-move hint that cost no motion to generate.
- **Knows its pose** as injected context, but reasons **camera-centric**.
  Geometry is the orchestrator's job. The subagent reports "logo fragment at
  the right edge of this frame"; the orchestrator, which knows the pose,
  translates that into a direction.
- **Its tools:** detect, segment, crop, describe. Deliberately broad —
  build everything, then trim with the evaluation framework once it exists.
  Thin separate verbs rather than one fat `inspect`, so per-tool usage data
  exists to justify a later deletion.

### The orchestrator agent

**Always-injected context** (not tools): where it is now, and the object.

**Verbs:** `view_at`, `views_near`, `get_view`, `crop`, `coverage`, `note`.
Kept deliberately small; verbs get added when a real run demonstrably cannot
express something.

**Return type:** tools ship both pixels and text for now. Which one wins is
an empirical question to settle against the dataset, not by argument.

## The run structure — the main deliverable

Every run maintains a data structure that aggregates everything the system
knows. This is the core of the task; the tools are how it gets filled and
read.

It holds: captured views and their poses, subagent run outputs, comments and
notes, relations between parts of the object, what has been visited, what has
been explored, the chain of thought so far, the current hypothesis, the
current plan, and tool results.

It must be sufficient to reconstruct enough context for an agent to continue
reasoning from any point in the run.

### Three trust levels — who is allowed to write what

| Writer | Owns | Rationale |
|---|---|---|
| **Deterministic code** | visited cells, poses, coverage, timings | Facts about what the arm actually did. No model may write these — a hallucinated "I have seen the far side" would corrupt the one signal independent of the VLM. |
| **Orchestrator agent** | plan, hypothesis, decisions | Its reasoning state. |
| **Inspection subagent** | per-view findings, comments, notes | Its observations, scoped to one view. |

### Verbatim on disk, distilled in context

The subagent's **full transcript** — its reasoning, the crops it examined,
its tool calls — is written into the run structure. Only its **summary**
returns to the orchestrator.

Rationale: a subagent that returns only a summary throws away the reasoning,
which makes "why did it say no logo on view 12" unanswerable after the fact.
Verbatim on disk costs nothing and is the raw material the evaluation
framework needs. Distillation is a read-time concern, not a write-time one —
you can always drop later, never recover.

## Answering the question

**The orchestrator decides, using everything it has in the run.** This is not
a vote and not a formula.

Views carry *different information*, not ballots. If a subagent reports a
logo in its view, that is sufficient evidence to answer yes — it does not
need corroboration from views that were looking elsewhere.

The negative case is where care is needed, and where our geometry helps: a
"no logo" from a view that was looking at the handle side is not evidence
about the far wall. Coverage is therefore available to the orchestrator as a
**supporting argument** when reasoning about absence — not as a weighting
scheme it is obliged to apply.

Recorded so the failure mode is not rediscovered: naive tallying across views
degrades presence detection as views accumulate, because most views of an
object correctly do not show a one-sided feature. PInVerify measured
positive-class accuracy falling 0.652 → 0.592 from single-view to multi-view
under vote-shaped combination.

## Context policy

The orchestrator's context holds the whole flow — subagent run results, its
own outputs, hypotheses, plans, tool results.

**No pruning in v1.** Runs are 5–30 steps. If degradation shows up in
practice, the run structure already supports reconstructing a pruned agent
with sufficient information to continue; that valve is built but not opened.

## Development approach

Software develops against **captured datasets, replayed as real runs**.

One dataset is enough to build the data structure and its operations: a full
sweep of captured positions over one object, with images and cell indices.
Broader collection — more objects, more transforms of the object — comes
later to verify and iterate.

Images are used as captured (848×480, the D405's depth-optimal resolution).
No separate high-resolution inspection path in v1.

**Known gap:** without per-cell ground truth for feature visibility, per-view
readout reliability cannot be measured and MAY-187 has no scoring function.
Deferred deliberately; it becomes mandatory before the decider can be
evaluated, and before any tool can be trimmed on evidence.

## Open questions

1. **Orchestrator agent lifecycle — undecided.** Two candidates:
   - *Image-only, per-iteration*: the orchestrator agent exists to run image
     inspection and lives for one loop iteration. The decider is a separate
     agent with its own context.
   - *Merged with the decider*: one agent shares context, plans, and
     hypothesis across both inspecting and deciding.

   This is the MAY-186 / MAY-187 boundary. Everything above holds either way,
   but the run structure's context-reconstruction contract differs between
   them.

2. **Surface stitching / unwrap.** Projecting all captured images onto a
   single canvas as a complete unwrap, for full observability at a glance.
   Where it fits is not settled — likely an orchestrator-side artifact, since
   it is a coverage representation. Note the open risk: grid-mosaicking
   separate frames is severely damaging to VLM accuracy (10 separate images
   97% → 4×4 grid 26.9%, MMNeedle), but a texture unwrap is a continuous
   geometric reparameterisation rather than a contact sheet, so that result
   may not transfer. Cheap to test on the dataset; test rather than assume.

3. **Pixels or text as the default tool return.** Ship both, decide by
   measurement.

4. **Which subagent tools survive.** Build broad now, trim once the
   evaluation framework exists.

## What this rules out

- No question-specific logic inside the module. The agent owns the question.
- No robot motion from the image tier, in any form.
- No world-frame spatial reasoning inside the inspection subagent.
- No model writing geometric ground truth.
- No pruning, summarisation, or context management in v1.
- No second high-resolution capture path in v1.
