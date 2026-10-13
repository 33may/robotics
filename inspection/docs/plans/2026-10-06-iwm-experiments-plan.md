# IWM: experiment ladder

**Date:** 2026-10-06. Approved by Anton the same day.
**Idea it serves:** [`../2026-10-06-iwm-research-foundation.md`](../2026-10-06-iwm-research-foundation.md), section 6.

Bottom up. Each rung answers one question, ends in a picture judged by eye, and feeds the
next. Only the next rung is planned in detail; the others are detailed when we reach
them.

Each rung is one experiment folder in `inspection/investigation/iwm/`, opened with a
note and closed with a report. The conventions are in that folder's `README.md`.

## What it runs on

Recordings only, no robot. 20 runs from 22 September in `inspection/data/runs/2209-*`,
read-only. Per view: image (1280x720), aligned depth, camera pose, object mask. Each run
has far views (about 40 cm) and close views (about 16 cm) of the same object.

## The ladder

| # | Experiment | Question it answers | What we look at | What could break it |
|---|---|---|---|---|
| 1 | All masks from one image | Do we get every detail without naming it? | The frame with all masks drawn, far view and close view | It misses the stickers and labels the planner asked about in the logs |
| 2 | A tree from one image | Does containment turn flat masks into object, face, sticker, text? | Nested outlines on the frame, plus the tree as indented text | Masks that overlap without one containing the other |
| 3 | Nodes in 3D | Where does each node sit on the object, and how big is it? | Nodes as boxes on the point cloud, sizes in mm | Depth holes on flat stickers; nodes smaller than the 10 mm shift between views |
| 4 | Two views, one tree | Does a detail seen twice become one node, while neighbours stay apart? | The merged tree, each node listing the views that saw it | Two similar stickers 3 cm apart |
| 5 | A closer view adds a level | Does a close view add children under the right parent? | The tree before and after the close view | It duplicates the parent, with no children added |
| 6 | Voxels under the tree | What does a cell hold at each level, and how well was it seen? | The object at depth 1, 2 and 3, coloured by how well seen | Cells finer than the registration allows |
| 7 | Meaning on the nodes | Can the IWM agent name, describe and read each node? | The tree with names and read text | Two readings of one label that disagree |
| 8 | Asking the tree | Are the 100 logged planner questions answered better from the tree than from one image? | Per question: the tree's answer next to what the vision agent said then | The tree lacks what the question needs |

Wiring the tree into the live loop comes after rung 8 and is not part of this plan.

## Notes on the order

- The voxels come at rung 6, after the tree works on masks. The masks and the merge carry
  the risk; voxels are known technology.
- Rung 4 has the least support in the literature and is where the shift between views
  bites.
- Rung 5 is the core of the idea: going deeper where the robot looked closer.

## Rung 1 in detail

**Views.** One far view and one close view of the same face, from a `compute` run and a
`box` run.

**Two ways of getting masks, side by side.**
- Without names: a grid of points over the image, each point asking for the mask under it.
- With names: the detail types the planner asked about in that run's log.

**What comes out.** Per view and per way: the frame with the masks drawn and the number
of masks. Per run: a checklist of the details the planner asked about, each marked found
or missed.

**Done when** we know which way gives the masks the tree needs, at far and at close
distance.

**Needs.** The GPU. No robot.
