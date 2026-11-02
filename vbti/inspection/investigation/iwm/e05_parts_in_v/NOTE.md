# e05: parts in V — the S/T/V structure, the agent keeps T, and views from it

**Opened:** 2026-10-12. **Rung:** 4–6 of the ladder in one (two views one tree, voxels
under the tree), on the S / T / V structure. Builds on e02 (agent-built trees), e03 (the
lift), e04 (V and its normals). Designed with Anton away; the **Assumed answers** at the
end are mine, for his alignment round v2.

## Aim

Show that S / T / V holds the detected parts of a run in V's geometry, stays consistent
while views arrive, and gives a view for a part from the geometry alone.

## Research questions and hypotheses

Each hypothesis states its prediction and the result that refutes it.

**RQ1 — Colouring fidelity.** Does a 2D detection, lifted into V's surface voxels and
rendered back from the same pose, reproduce the detection, and how much does it smear in 3D?
- H1: IoU(mask, label render) > 0.9 for nodes wider than 20 mm and > 0.7 for ports and
  labels; fewer than 10 % of a node's voxels lie more than 8 mm off its face plane.
- Refuted if small nodes lose more than 30 % of their mask, or smear exceeds 10 %.

**RQ2 — Cross-view consistency.** When the same physical part is lifted from two views of
one run, how far apart do the voxel sets land, and does the offset exceed the small parts?
- H2: the two lifts differ by one rigid vector per view (e03: ~6 mm for c5.6 / c5.2), so
  the voxel sets overlap > 50 % for parts wider than 20 mm but **< 50 % for ports
  (~15 mm)**: geometry alone cannot decide "same part" for small parts.
- Refuted if small parts overlap > 70 % (then geometry suffices and the agent is
  only needed for naming).

**RQ3 — Structure.** Can strict nesting (child ⊆ parent) and one deepest label per voxel
be kept while views are added, and at what conflict rate?
- H3: both hold by construction; conflicts (a voxel claimed by two unrelated parts) stay
  under 5 % of any part's voxels and sit at part boundaries or come from the offset.
- Refuted if the conflict count grows with every view added.

**RQ4 — Agent maintenance.** Does the in-loop VLM agent, shown the new frame with the
existing parts projected into it, make the right `same` / `under` calls?
- H4: it recovers ≥ 90 % of the ground-truth pairs and beats the geometry-only baseline
  exactly on the small parts of RQ2.
- Refuted if it misses clear pairs the baseline gets, or invents pairs for parts the
  view does not hold.

**RQ5 — The view for a part.** Does the pose computed from the part's geometry (centre,
face normal, extent) inspect it as well as the recorded step that found it?
- H5: on V's render metrics the computed pose scores ≥ the detecting step for > 80 % of
  parts; tilting the view loses visibility on recessed ports, not on flat labels.
- Refuted if the computed pose is > 15° off face-on, or the part sits more than 10 % of
  the frame off centre.

**RQ6 — The view for a set.** For which sets of parts does one pose exist with every
part visible, and does the union-normal rule find it?
- H6: sets on one face always; the sticker + LAN ports (two panels of one front) at the
  recorded distance; the union-normal rule finds these.
- Refuted if a set that one recorded frame plainly shows gets no pose.

Search by name is a capability check, not a hypothesis.

## Why

The two main uses of the IWM are geometry-aware view selection and semantic search that
ends in a subpart view. Both need the parts to live in V's geometry and T kept up while
views arrive. This chain is built end to end on recorded runs so the weak link shows
before anything goes live.

## Input

- V from e04, scene scope, **1 mm** (main) and 2 mm (check):
  `inspection/data/iwm/e04_voxel_scene/<run>/scene/{1,2}mm/vbg.npz`.
- The e02 final trees and masks (`inspection/data/iwm/e02_agent_tree/<frame>/full/
  tree_final.json`, `masks_final.npz`). **No new tree-building runs.**
  - `2209-compute-ray-demo5`, in capture order: **c5.2** (step 2, far: "Industrial PC"
    > heatsink top, front face, bevel; ethernet / serial ports, status LED, Chort label),
    **c5.6** (step 6, front panel close: LAN, COM, audio, antenna ports, the sticker),
    **c5.7** (step 7, the other face: USB array, HDMI, power button, DC block, labels).
  - `2209-box-ray-demo1`: **b1.1** (step 1: "box" > front face, top face, AI KIT
    sticker), **b1.4** (step 4: "front face" > product-info label, serial label, marks).
- Per step: `depth_aligned.npy`, `T_base_cam`, `rgb_rotation_deg`; `session.json`
  (colour 1280×720, fx 647 → 89° × 58° field of view).
- All views of one run share the base frame. The per-view offset (2–14 mm, one rigid
  vector per view) is measured, never corrected.
- **Ground truth for RQ4:** `evidence/truth_pairs.json`, the correspondences between the
  trees, written by me from the names and crops (c5.2 ↔ c5.6: LAN ↔ ethernet ports,
  COM ↔ serial ports, sticker ↔ Chort label, audio jacks ↔ audio ports; c5.7: object
  only; b1.1 ↔ b1.4: front face ↔ root, and the labels if the sticker is one of them).
  Marked as mine; Anton checks it.

## What we do

Ordered by what each question needs.

### 1. One view into V (RQ1)

- Two lifts of a node's mask to V's surface voxels:
  - **L1, through the view's own depth** (e03's way), each point snapped to the nearest
    surface voxel within 3 mm; unmatched points counted.
  - **L2, through V:** ray-cast V from the view's pose; the voxel each mask pixel hits.
- The node's voxel set = its own mask's voxels ∪ its children's (containment by
  construction).
- **Measure:** label render from the same pose vs the mask: IoU, precision, recall per
  node (for L2 this is ~1 by construction, so its number is the smear); share of voxels
  > 8 mm off the node's face plane; share of mask pixels with a hit (TSDF holes).
- The better lift by the numbers goes downstream; both stay in the report.
- Frames: c5.6 and b1.4 (the detail-rich ones), at 1 and 2 mm.

### 2. The structure: S / T / V in code (RQ3)

`stv.py`, used by everything after.
- **Surface table of V:** the surface voxels (weight > 0.5, |tsdf| < one voxel), one row
  each: grid index, centre, Open3D's normal, weight; hash (i, j, k) → row. V's geometry
  is indexed, not copied.
- **Part** (an item of S, a node of T): `id`, `name`, `text`, `parent`, `children`,
  `rows` (sorted int array), `detections` [(view, e02 node id)]. On demand: centre,
  extent (PCA box), normal by the e04 rule (RANSAC + PCA over the part's face's rows
  within 16 mm; face = the nearest ancestor with a planar fit), level.
- **T:** the parts under one root. **Label array:** one int per surface row = the deepest
  part holding it, −1 if none.
- **Operations:** `assign(part, rows)` adds rows to the part and every ancestor; a row's
  label becomes the part if it was −1 or an ancestor, otherwise the row keeps its label
  and (row, old, new) goes to the conflict list. `extend` = `assign`. `add(parent, name,
  rows)`. `voxels_of(part)`, `part_of(row)` → part + ancestor chain. `search(text)`.
  `view_for(part)`, `view_for_set(parts)`. `check()`: rows ⊆ parent's rows for every
  part; every row's label is a descendant-or-self of every part holding it; run after
  every operation.
- Saved as `parts.json` + `rows.npz` under `inspection/data/iwm/e05_parts_in_v/<run>/
  <mm>mm/`. **Measure:** rows, bytes, time per operation, conflicts after each view.
- Tests: the invariants on a toy grid and on the real c5.6 colouring.

### 3. Cross-view offset (RQ2)

- Colour from c5.2 only; lift c5.6's nodes separately (not assigned); for each
  ground-truth pair: voxel overlap (IoU of row sets), centroid offset (mm), the same
  after removing the view's rigid vector (diagnostic only; V stays uncorrected).
- Also the render test: the label render from c5.6's pose of the c5.2 colouring vs
  c5.6's own masks: IoU and pixel offset per pair.
- Same for b1.1 → b1.4. Figures: the pair overlaps in 3D and in c5.6's frame.

### 4. The next views, and the agent keeps T (RQ4)

Order: c5.2 → c5.6 → c5.7; b1.1 → b1.4. The first view's root is the object. For each
later view v:
1. Lift v's own e02 tree (the lift chosen in 1).
2. Render the current T from v's pose: the existing parts as named, outlined regions in
   v's frame; per new node, the share of its pixels on each existing part.
3. **The agent decides per new node, the root included:** e02's machinery (VlmAgent,
   Gemini ER-2, the e02 verbs) plus `same{new, part}` (extend part) and `under{new,
   part, name}` (new part under it). Unnamed nodes are dropped, as in e02. Decisions
   and reasons are recorded.
4. Apply `extend` / `add`; `check()`; count conflicts.
- **Baseline:** geometry only — `same` to the existing part that holds ≥ 50 % of the
  node's voxels at the same depth, else `under` its deepest holder.
- **Measure:** agent and baseline against `truth_pairs.json`: pairs recovered, wrong
  pairs, invented pairs; split by part size (< 20 mm, ≥ 20 mm). Conflicts after each
  view. The agent's run: turns, tool calls, seconds.

### 5. Search (capability)

- `search(query)` over name, text and the ancestor path ("front face / LAN ports / LAN
  1G port"); returns parts with level, centre, extent, views.
- **Measure:** 15 queries with expected hits written from the trees (LAN, serial, USB,
  HDMI, sticker, label, barcode, QR, power, audio, antenna, heatsink, logo, CE, front
  face): precision and recall.

### 6. The view for a part (RQ5)

- From the part: centre c, normal n (the rule in 2), extent s.
- Standoff d so the part fills half the frame, d = s / (2 · 0.5 · tan(FOV/2)), clamped
  to the D405 range 7–50 cm; for parts under ~60 mm the clamp decides. Camera at c + d·n
  looking along −n; up from the face's in-plane axis. The detecting step's real distance
  is reported beside d.
- **Candidates:** the normal pose; tilts of 15° and 30° in four directions; a pivot
  ring of 8 views around n at 30°; the recorded steps.
- **Scores from the ray cast of V, reported raw:** visible share of the part's rows;
  centring (the part's pixel centroid vs the image centre, as a fraction of the frame);
  median incidence angle; pixel extent; occluded share. Ranking is lexicographic:
  visible ≥ 0.9, then lowest incidence, then best centring. No weighted sum.
- **Measure:** the table over all parts; the computed pose vs the detecting step; the
  tilt loss for recessed vs flat parts. Figures: eight parts (a LAN port, a serial port,
  the sticker, a USB label, the barcode, the QR code, a face, the heatsink), a row of
  renders with scores, the step beside them; the pivot ring for one port.

### 7. The view for a set (RQ6)

- `view_for_set(parts)`: the union's centre and extent; normal from RANSAC over the
  union's rows, else the mean of the parts' normals; standoff from the union's extent;
  the same candidates; score = the minimum visible share over the parts, then the
  union's centring.
- **Measure:** sets: the two LAN ports; all ports of the lower panel; a port with its
  label; the sticker with the LAN ports; the four USB ports; the barcode with the QR
  code. Whether a pose exists with every part > 80 % visible, and its render.

### 8. The viewer

e04's viewer, extended: V coloured by the deepest label; a level slider (colour by the
ancestor at depth d); a view filter; click a voxel → part, ancestor chain, detections,
the source frame crop; pick a part → its voxels light up and its proposed view appears
as a frustum, "look from" moves this client's camera there; a search box; a multi-select
for the set view. Static label renders for the report from each capture pose, the side
and the top.

## Acceptance criteria

- AC-1 (RQ1): per-node IoU, smear and hit-share for c5.6 and b1.4, both lifts, both
  resolutions, in `evidence/lift.json`; the figure of mask vs render per node.
- AC-2 (RQ3): `stv.py` tests pass; `check()` passes after every operation of stages 1–4;
  conflicts per view in `evidence/conflicts.json`.
- AC-3 (RQ2): per ground-truth pair the overlap and offset, raw and offset-removed, in
  `evidence/pairs.json`; the pair figures.
- AC-4 (RQ4): agent and baseline decisions vs truth in `evidence/agent_vs_truth.json`,
  split by part size; the agent transcripts stored under the run's data folder.
- AC-5 (search): the query table with precision and recall.
- AC-6 (RQ5): the per-part table in `evidence/views.json`; the eight-part figure; the
  pivot-ring figure.
- AC-7 (RQ6): the six sets in `evidence/sets.json` with their renders.
- AC-8 (viewer): runs on both runs; the static label renders in `evidence/`.
- AC-9: `run.json` beside every output; the report's numbers read from `evidence/`.

## Confounds, and how they are separated

- **The per-view offset** confounds RQ2 and RQ4: measured first (stage 3), overlap
  reported with and without the rigid vector removed; V is never corrected.
- **Mask quality** (e02's masks are the detection's own errors): RQ1 compares against the
  mask itself, not against a true outline.
- **TSDF holes** (dark print, fins): the hit share is reported separately from IoU.
- **RANSAC nondeterminism:** fixed seed in every fit.
- **The agent's variance:** one run per view, transcripts kept; a second run if a
  decision looks like noise.

## What would count as a no

- RQ1: small nodes lose > 30 % of their mask, or smear > 10 %.
- RQ3: the conflict list grows with every view.
- RQ4: the agent misses pairs the geometry gets, or pairs parts the view does not hold.
- RQ5: the computed pose is not face-on and centred on V's own render, or the detecting
  step beats every candidate for most parts.
- RQ6: no pose for a set one recorded frame plainly shows.

## Done when

The report answers RQ1–RQ6 with numbers from `evidence/`, and Anton has walked the
coloured V of both runs in the viewer, searched parts, seen their proposed views and
two set views. Then alignment round v2 on the assumed answers.

## Needs

Recorded runs, the e04 V, Gemini ER-2 for stage 4 (key in the env, as for e02). No
robot. Every stage runs unattended.

## Assumed answers (grilling round 1, answered by me in Anton's stead)

| # | Question | Assumed answer | Rests on |
|---|---|---|---|
| Q1 | Who maintains T | The in-loop VlmAgent (Gemini ER-2) with the new frame + existing parts projected, verbs `same` / `under`; geometry-only baseline beside it | "the agentic maintaining of the tree"; "reuse the in-loop vision agent"; "let's try both and compare" |
| Q2 | View order, the root | Capture order (c5.2 → c5.6 → c5.7; b1.1 → b1.4); first root = the object; later roots placed by the agent like any node | the inspection runs survey → close-up; b1.4's root is a face, not the box |
| Q3 | Truth for same/under | I label `truth_pairs.json`, marked as mine | "answer your questions" |
| Q4 | Conflict policy | First label wins, conflict recorded, count reported | "don't make the decision on merge yet" |
| Q5 | What a proposed view is judged against | V-render metrics + the detecting step as the bar + Anton's look; no model grader on TSDF renders | "render views, check how optimal they are" |
| Q6 | Standoff | Half-frame rule clamped to 7–50 cm; the step's real distance reported beside it | "get the view and then execute the robot to capture that frame" |
| Q7 | Score | Five raw numbers, lexicographic ranking, no weighted sum | "storing distances in the tree is bullshit"; no invented numbers |
| Q8 | Resolution, lift | 1 mm main, 2 mm check; both lifts measured, the better one used | e04: 1–2 mm keep the details |

## Amended during the run (2026-10-12)

- Lift: a third variant L3 (ray cast gated by the view's own depth) was added after L2 smeared
  through TSDF holes; the better lift by the numbers is L3.
- The truth gained a "within" class (a node that is a band of an existing face) after the
  geometry refuted two of my pairs.
- Views: a candidate pose must have 2 cm of free space around the camera (nearest surface
  row), else it is infeasible; the face-on pose of most parts near the table is. The standoff
  clamp is the loop's D405 minimum range at 720p, 10.5 cm (not the 7 cm written above).
  Candidates gained Anton's face-plane shifts (±2/4 cm, same orientation).
- Parts keep the names later views give them as aliases (search needs them).

## Not decided here

Merging across views (conflicts are recorded, not resolved); the per-view offset; the
VLM-level search.
