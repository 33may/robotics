# e01: all masks from one image, and a tree from them

**Opened:** 2026-10-06, reframed 2026-10-08. **Rung:** 1 and 2 of the ladder, as one
experiment in two parts.

## What we want to achieve

From one image of an object, a structure a robot could keep: every visible detail as a
mask, and the masks ordered as object → face → detail → sub-detail, without us naming
what to look for.

## Why

The IWM is built from these trees. A detail that gets no mask here, or lands under the
wrong parent, cannot exist in the IWM later. The planner's own questions (52 of the 100
logged were "read what is on this sticker / port / label") are what the tree has to carry.

## Deliverable

A concise report with good-looking figures that show it works: per frame, the masks
drawn, the nested outlines, the tree level by level. The scripts are disposable; what we
keep is the experience (which model, which settings, what breaks) and the figures.

## Questions

Four choices to settle, by side-by-side renders on the same frames:

1. Which model for this kind of segmentation? **SAM 3** (baseline, probably the one we
   will use) against **UnSAMv2** (SAM-2 with a granularity dial).
2. Which prompt and input/output strategy to the model?
3. Which method organises the masks into an interpretable tree?
4. Which implementation details (grid density, thresholds, crops) are needed to reach
   the small details?

## Input

Eight recorded frames, chosen by Anton from `evidence/frames_all.jpg` and listed in
`evidence/frames_selected.json`:

| id | frame | why |
|---|---|---|
| c5.6 | compute, x+ panel, 21.9 cm | main example: wordmark, 2 RJ45, 2 DB9, jacks, LEDs, labels |
| c5.2 | compute, oblique, 29.7 cm | far view paired with c5.6 |
| c5.7 | compute, x- panel, 22.0 cm | second example: USB ports, terminal block (no recorded object mask) |
| c9.6 | compute, x- panel, other run, 24.9 cm | paired with c5.7 for multi-view |
| bc5.9 | REALSENSE box, barcode label, 20.7 cm | label-rich example |
| b1.1 | cardboard box, AI KIT sticker, 20.7 cm | simple example |
| b1.4 | cardboard box, barcode sticker, 24.2 cm | (no recorded object mask) |
| b4.5 | cardboard box, same sticker, other run, 24.8 cm | paired with b1.4 for multi-view |

## Method

Same frames for everything. Part 1 (questions 1, 2, 4): each model × strategy runs on
each frame; one "all masks" render per run, the mask count, and the planner's details
marked found or missed. Part 2 (question 3): the mask sets from part 1 → each tree
method → nested outlines on the frame, the tree level by level, the tree as indented
text. Judged by eye: is every node something a person would name?

### Turn 1 (2026-10-08, run autonomously; choices made by the agent, to be revised)

Strategies: SAM 3 full-frame point grid 32×32 and 64×64 (pipeline defaults, plus one
loosened-threshold run); SAM 3 grid on the object crop; SAM 3 text prompts with the
planner's nouns; UnSAMv2 whole-image at granularities 0.1 … 0.9 on the full frame and
on the object crop. Tree: containment (dedupe near-identical masks, parent = smallest
mask holding ≥ 90 % of the child, root = object mask) on each mask set; for UnSAMv2 also
the granularity stack (coarse masks parent finer ones).

## What we look at

- Per frame and strategy: the frame with all masks drawn, the number of masks.
- Level by level: the masks at each tree depth drawn separately, then the nested outlines.
- The tree as indented text, each node with its area and the strategy that produced it.
- Per run: the planner's details, found / missed.

## What would count as a no

- Neither model gives masks for the stickers, labels and ports the planner asked about.
- Masks overlap without containment so often that no tree comes out.
- The tree nodes are surface fragments, not things a person would name.

## Done when

We can show, on the chosen frames, a tree a person would accept, and know which model,
strategy and settings produced it.

## Needs

The GPU (SAM 3 in `robo`, UnSAMv2 in `unsam`). No robot.
