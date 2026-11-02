# e03: the tree in 3D — projection and an interactive viewer

**Opened:** 2026-10-08. **Rung:** 3 (nodes in 3D), with the viewer that rung 4 (two
views, one tree) will need.

## What we want to achieve

Put the e02 trees into the run's 3D: every node's mask projected through its step's
aligned depth and camera pose onto the run's fused cloud, and a viewer in which we can
see the detected parts on the cloud, level by level, click a part to see the frame and
step it was reconstructed from and its parents, and go back. With that in hand we can
exercise the align and merge step: something detected from one view, then another view
observed, and the findings merged.

## Why

The 2D trees are judged good enough to carry forward. Rung 3 and 4 need a place to
SEE where a node sits in 3D, which view it came from and what it hangs under, before
any merge logic is designed. The known 10 mm per-view offset between views
(`cross-view-registration-error-2209-runs`) will show up here as the gap between the
same part from two views.

## Deliverable

- A projection utility: tree node (frame, step, mask) → points in the base frame, with
  colours, centroid and extent in mm, saved per node.
- An interactive viewer (Viser, web, Python callbacks) over one run: the fused cloud,
  the nodes of the chosen frames as coloured point sets, a level toggle, click a node →
  the right panel shows its source frame crop with the mask outline, its name and its
  parent chain; the parents are highlighted in 3D; a back button restores the previous
  selection. A frame selector to add the second view of a pair.
- A short report with static renders (the projected levels, the two-view pairs) and
  the numbers: points per node, extent in mm, the offset between the same part from two
  views.

## Questions

1. Does every 2D node survive projection — which ones lose their depth (holes on
   stickers, thin labels, specular metal)?
2. How big are the nodes in mm, and how far apart does the same node land from the
   two views of a pair (c5.7 / c9.6, b1.4 / b4.5)?
3. Is the viewer loop (level → click → frame → parents → back) enough to reason about
   the merge, or what is missing?

## Input

The e02 final trees and node masks (`inspection/data/iwm/e02_agent_tree/<frame>/full/
tree_final.json`, `masks_final.npz`) of the eight frames; per step `depth_aligned.npy`,
`T_base_cam`, `rgb_rotation_deg`; per run `session.json` (colour intrinsics, depth
scale) and `fused/{cloud,colors}.npy`.

## Method

- Projection: `cell/geometry.py:object_in_base` path (depth_aligned + colour
  intrinsics + T_base_cam), one call per node mask; the same pixels the inspect agent
  saw (orientation: upright on disk, checked in e02). Node colour from rgb.
- Viewer: Viser. Reasons: Python server with `on_click` on scene nodes, GUI panels
  with images, buttons and checkboxes — the click → frame → parents → back loop is a
  few callbacks. Rerun (what `record show` uses) has no click callback into Python in
  the native viewer and would need the notebook web viewer; Open3D's GUI is harder to
  shape. Rerun stays for record review; this viewer is research tooling.
- Static renders for the report with matplotlib (per level, per pair).
- Runs: c5.6 first (one view), then the two pairs c5.7 / c9.6 and b1.4 / b4.5 side by
  side in the same cloud.

## What we look at

- The fused cloud with the nodes coloured, per level; the same with two views.
- Per node: points, centroid, extent (mm); per pair: centroid offset of the same part.
- The viewer loop, by hand.

## What would count as a no

- Most small nodes (labels, jacks) project to nothing (depth holes).
- The same part from two views lands farther apart than its own size.
- The viewer cannot show lineage (which frame, which parent) for a clicked part.

## Done when

The viewer runs on c5.6 and on both pairs, the report shows the projected levels and
the pair offsets, and Anton has used the loop.

## Needs

No GPU, no robot. Browser for the viewer.

## Turn 2 (2026-10-08): how well do the masks land on the geometry?

**Achieve.** Before multi-round merging is designed, Anton judges by eye how well a
2D node becomes 3D, and how well it lands when carried into another view, because
every merge rule will stand on that.

**Deliverable.** Figures per frame plus a small numbers file:
1. *Mask on depth.* Each node's outline drawn over the RGB and over the depth map, and
   the pixels inside each mask that have no depth.
2. *Mask in 3D.* The node points painted on the run's fused cloud, seen from the
   capture camera, from the side (grazing) and from above. Edge bleed onto the
   background and depth error both show up here.
3. *Mask into other views.* The frame's nodes, through 3D, projected into the other
   steps of the same run that see them, with an occlusion test against that step's
   depth. Where the other step has its own e02 tree (c5.6 → c5.2), its own masks are
   drawn too, and the gap between the two is the cross-view offset.

**Numbers** (`evidence/projection_quality.json`). Per node:
- pixels with depth;
- share of points farther than 8 mm from the fused surface (bleed);
- per target view: the visible share, and for c5.6 → c5.2 the IoU with the best
  matching own node and the centroid offset in px and mm.

**A no.** Painted 3D regions spill onto the table or a neighbouring part. Or, carried
into a second view, the projected regions land on another part than the one the mask
named.
