# e02: the vision agent builds the tree

**Opened:** 2026-10-08. **Rung:** 2 (tree from one image) with rung 7 (meaning on the
nodes) pulled forward onto it. 2D only.

## What we want to achieve

From one view, the vision agent we already run in the loop — the same `VlmAgent` loop
and Gemini ER-2 model as `inspect` — given the IWM tools, produces the most logical
levelled tree of that view. The shape, in Anton's words: the side; on the side the
things (labels, the array of ports; stickers); under a port its label and the port
itself; under a sticker its codes and text. Noise is what the agent cannot name.

## Why

e01 showed the levels carry the information (UnSAMv2 granularity stack + containment)
but also surface fragments and duplicates that code cannot tell from real nodes. The
agent that sees can: it names what it sees, asks for what it expects (the vocabulary
comes from the frame, the question and, later, the existing IWM), and drops the rest.
This also settles "who names what the segmentation model looks for": the agent does.

## Deliverable

A concise report, figures first: per frame the agent's final tree level by level and
as indented text with names; the tree before and after the agent (what it dropped,
merged, grouped, added, named); the tool sequence it used; turns and seconds per view.
Scripts stay disposable.

## Questions

1. Which tools does the agent need — the verb set and what each returns (text tree,
   numbered renders)?
2. Which rules prompt makes it build levels in the expected shape, not a flat list?
3. Is "name it or drop it" (plus grouping) enough as the noise rule — what noise
   survives, which real nodes get lost?
4. What does it cost — turns, model calls, seconds per view?

## Input

The eight frames of e01 (`../e01_all_masks/evidence/frames_selected.json`) and their
e01 mask sets (UnSAMv2 ladder g 0.1–0.9, SAM 3 text) as the raw material. The agent's
view is byte-identical to `rgb.png` on disk (checked), so e01's masks line up.

## Method

**The agent.** `inspection/eyes/agents/vlm_agent.py` unchanged: a new spec (rules,
tool subset, `tree` appendix) in this folder. The IWM verbs are added for the
experiment by patching the loop's `TOOLS` and `_dispatch` from the experiment script —
no product file changes; the loop's invariants (order, one image per turn, repeat
refusal, categorical failures) stay. Model: `gemini-robotics-er-2-preview`, the one the
loop uses. Writer/store under `inspection/data/iwm/e02_agent_tree/<frame>/`; the
recorded run is only read.

**The raw tree (code, before the agent sees anything).** e01's containment tree over
UnSAMv2 g 0.1–0.9 + SAM 3 text, then two pre-filters that need no judgement: masks
that are near-duplicates across sources become one node (IoU ≥ 0.8), and a node whose
only child covers ≥ 90 % of it is collapsed. Everything else is left for the agent.

**The IWM verbs (turn 1 set, to be revised):**

| verb | args | returns |
|---|---|---|
| `tree` | — | the current tree as text: id, depth, parent, name, box, share of parent |
| `show` | node | an image: that node's region with its children outlined and numbered — "this is image N" |
| `find` | phrase | SAM 3 text masks for the phrase, placed into the tree by containment; the new ids |
| `name` | id, name | writes the name (and optional `text` read) on the node |
| `group` | ids, name | a new node = the union of the ids, under their common parent ("array of ports") |
| `drop` | ids | removes nodes (children move up to the parent) |
| `crop`, `read_text`, `note` | as today | unchanged |

**The rules prompt** states the target shape, the procedure (look at the face level,
name what you see, descend one level at a time with `show`, `find` what you expected
and do not see, `group` repeated things, `drop` what you cannot name), and the reply
`{"evidence", "reasoning", "answer", "tree"}` where `tree` is the id → name map; the
tree itself lives in the tool state and is written by code.

**Runs.** Each frame once with the full verb set; c5.6 also with `find` removed and
with `show` removed, to see which verbs carry the result. Budget 24 turns.

## What we look at

- Per frame: the final tree level by level (render) and as indented text with names.
- Before/after: nodes dropped, merged, grouped, added by `find`, named.
- The transcript: tool order, turns, seconds, model calls.
- Against the expected shape: face → things on the face → their parts, by eye.

## What would count as a no

- The agent returns a flat list of names, not levels.
- It drops real nodes (ports, labels) or keeps surface fragments with invented names.
- It needs more than 24 turns or several minutes per view.

## Done when

On the eight frames the agent's tree matches the expected shape by eye, and we know
the verb set and prompt that produced it.

## Needs

Gemini API key (set), the GPU for `find` (SAM 3 text, `robo` env). No robot.
