# e05: parts in V — the S/T/V structure, the agent keeps T, and views from it

**Run:** 2026-10-12, autonomously (Anton away; the design questions were answered by me, see
the NOTE's *Assumed answers*). **Closes** after Anton's walk through the viewer and the
alignment round v2. Every number below is printed by `numbers.py` from `evidence/*.json`.

**Answer, per research question.**
- **RQ1 colouring.** A mask lifted into V by ray-casting V and gating each hit with the view's
  own depth (L3) lands as a clean part: no smear (0 % of rows > 8 mm off the face, except the
  real 10 % recess of the RJ45 jacks), IoU 0.75 (small parts) / 0.76 (large) against the mask
  rendered back from the same pose, recall 0.90–0.99. The plain ray cast (L2) scores a higher IoU
  but 48–83 % of a face's rows fly through TSDF holes onto interior surfaces; the own-depth lift
  (L1) has no smear but IoU 0.36–0.51. **H1 holds for L3** (small-part IoU 0.75 > 0.7, smear
  < 10 %); the IoU ceiling is voxel-vs-pixel quantisation, not error.
- **RQ2 cross-view.** The same port from c5.2 and c5.6 lands 8 mm apart along one vector
  ((0.4, 2.5, −8.2) mm); the row-set IoU of a port is 0.10, 0.26 after removing the vector.
  **H2 holds:** geometry alone cannot call "same" on a 16 mm part.
- **RQ3 structure.** Nesting and one deepest label per row hold by construction after every
  operation on both runs (`check()` never failed). Conflicts are not geometry: on the compute
  box after three views, 4956 conflicting rows = 2663 "re-parent" + 1262 "intermediate" (two
  views disagree on T) + 1031 boundary; on the box 33, all boundary. **H3 half holds:** the
  invariants hold, but conflicts grow with every view whose tree disagrees with T.
- **RQ4 agent.** Shown the new frame with the existing parts projected into it, Gemini ER-2
  pairs **10/10** ports, groups and the sticker across c5.2 → c5.6 (0 wrong, 0 missed, 6 turns,
  31 s) where the geometric baseline pairs 1/10; on c5.7 it places all 18 new parts; its one
  mistake is a name trap on the box (b1.4's "front face" paired with b1.1's "front face" strip,
  another side). **H4 holds.**
- **RQ5 view for a part.** The face-on pose from centre + face normal + extent, at the
  loop's 10.5 cm minimum range, is centred (33/38 and 15/15 within 10 % of the frame) with a
  median incidence of 23° / 14°, and the best computed candidate matches or beats the recorded
  step for **32/38 and 13/15** parts. But **29/38 and 12/15 face-on poses are infeasible**:
  parts 2–4 cm above the table put a camera at part height 0.4–1.8 cm from the table (median
  clearance 0.7 / 1.1 cm; the recorded close-ups had ≥ 2.8). Two ways out: the 15–30° tilts
  looking down (cost 16–18 % visibility) and **Anton's shift** — the face-on pose moved 2–4 cm
  up or sideways in the face plane, the loop's own rule for box faces: a shift rescues 22 of
  the 29 and 10 of the 12, and is the top candidate for 20 of 38 and 10 of 15 parts (tilts
  15 / 0, pivots 2 / 2, the plain face-on pose 1 / 3). **H5 holds on geometry and fails on
  feasibility; the shift is the answer to the feasibility half.**
- **RQ6 view for a set.** All six sets get a feasible pose with every part ≥ 0.89 visible
  (sticker + LAN ports: 0.89 vs 0.76 from the best recorded step). **H6 holds.**
- Search by name works once merged parts keep their later names as aliases (compute: 11/12
  queries at precision 1 / recall 1).

## What was run

`inspection/investigation/iwm/e05_parts_in_v/`, on e04's scene-scope V at 1 mm (2 mm as a
check), runs `2209-compute-ray-demo5` (c5.2 → c5.6 → c5.7) and `2209-box-ray-demo1`
(b1.1 → b1.4), the e02 final trees as inputs, no new tree-building.

- `stv.py`: the structure. Surface rows = voxels with weight > 0.5 and |sdf| < 1 voxel,
  **sorted by grid index** (Open3D's buffer order differs per process; a saved part in that
  order pointed at other voxels after a reload). Part = name, aliases, parent, children, sorted
  rows, detections. `assign` adds rows to the part and every ancestor; a row's label becomes the
  deepest part; a row claimed by an unrelated part keeps its first label and is recorded as a
  conflict. `check()` after every operation. 1 mm compute: 1.83 M rows, 36 MB, 0.4 s.
- `lift.py` (RQ1): L1 own depth + 3 mm snap; L2 ray cast of V; L3 = L2 gated by the view's own
  depth (10 mm). Validation as Anton asked: the labels rendered from the same pose vs the mask.
- `pairs.py` (RQ2): V coloured from c5.2, c5.6 lifted separately; per truth pair the row IoU,
  the centroid offset, the IoU after removing the view's rigid vector, the render test.
- `keep.py` (RQ3, RQ4): views in capture order; the first view's root is the object; each later
  view's nodes are placed by the baseline (same to the deepest child of the resolved parent
  holding ≥ 50 % of the node's rows, else under) or by the agent (e02's VlmAgent with
  `tree / show / existing / show_existing / same{node, P} / under{node, P or #node, name} /
  drop`), then applied with `extend` / `add`. Scored against `evidence/truth_pairs.json`
  (written by me from the e02 figures, two pairs corrected by the geometry).
- `conflict_kinds.py`: conflicts grouped by (old, new) and classified.
- `views.py` (RQ5, RQ6): per part the PCA box, the e04 normal rule (RANSAC + PCA over the
  face's rows within 16 mm), standoff for half a frame clamped to 10.5–50 cm (the loop's D405
  minimum range at 720p), candidates (face-on, ±15/30° tilts, face-plane shifts of ±2/4 cm
  along the face's up and side axes, an 8-view pivot ring at 30°, the recorded steps), scores from a ray cast of V
  (visible share of rows, occluded, off-frame, centring, median incidence, clearance), ranked
  lexicographically: feasible, visible ≥ 0.9, incidence, centring. Sets: the union's plane
  normal, score = the minimum visible share.
- `viewer.py` (port 8083): grey V, labelled rows as coloured points, level slider, view filter,
  search, part → voxels + its proposed view as a frustum, "look from proposed", set query.
- Tests: `tests/` (15), the invariants on a toy grid and the lift on c5.6.

## What we saw

- **The lift.** Mask (green) vs the labels rendered back (magenta), c5.6 at 1 mm, L3: the
  ports and jacks match to the voxel; the sticker's dark print has TSDF holes (74 % hit).
  ![](evidence/lift_c5.6_1mm_L3.jpg)
- **The per-view offset.** c5.2's lift (green) sits ~8 mm above c5.6's (magenta) on every port.
  ![](evidence/pairs_2209-compute-ray-demo5_1mm.jpg)
- **V coloured after all views** (agent), from each capture pose, the top and the side; the
  legend is T. ![](evidence/labels_2209-compute-ray-demo5_1mm.jpg)
  ![](evidence/labels_2209-box-ray-demo1_1mm.jpg) — on the box, b1.4's whole face is painted
  as P1 "front face", the strip b1.1 named: the agent's name trap.
- **The view for a part.** Ethernet Port 1: the face-on pose (centred, 14°) beside the
  detecting step's render and its real frame. ![](evidence/view_2209-compute-ray-demo5_P11.jpg)
  The pivot ring around its normal: ![](evidence/pivot_2209-compute-ray-demo5_P11.jpg)
- **The view for a set.** ![](evidence/set_2209-compute-ray-demo5_four_USB_ports.jpg)
  ![](evidence/set_2209-compute-ray-demo5_sticker_and_LAN_ports.jpg)

## Numbers

**RQ1, 1 mm** (median over nodes; small = widest extent < 20 mm; `evidence/lift.json`).

| frame | lift | small: IoU med / min | recall | smear med / max | large: IoU med / min | smear med / max |
|---|---|---|---|---|---|---|
| c5.6 | L1 | 0.36 / 0.05 | 0.38 | 0.00 / 0.11 | 0.48 / 0.40 | 0.00 / 0.11 |
| c5.6 | L2 | 0.79 / 0.74 | 0.99 | 0.00 / 0.10 | 0.80 / 0.68 | 0.18 / **0.62** |
| c5.6 | **L3** | **0.75** / 0.42 | 0.90 | 0.00 / 0.12 | 0.76 / 0.60 | 0.01 / 0.11 |
| b1.4 | L1 | 0.25 / 0.17 | 0.31 | 0.00 / 0.00 | 0.56 / 0.51 | 0.00 / 0.00 |
| b1.4 | L2 | 0.54 / 0.26 | 1.00 | 0.00 / 0.00 | 0.82 / 0.77 | 0.27 / **0.70** |
| b1.4 | **L3** | 0.44 / 0.26 | 0.70 | 0.00 / 0.00 | 0.74 / 0.70 | 0.00 / 0.00 |

The 0.10–0.12 smear on c5.6 is the LAN jacks' real 12 mm recess; b1.4's small nodes are 2–9 mm
marks (CE, recycle, "made in china") quantised by 1 mm rows. 2 mm: the same picture, IoU 0.1
lower.

**RQ2** (`evidence/pairs.json`, c5.2 → c5.6 at 1 mm): rigid vector (0.4, 2.45, −8.21) mm over
7 wide pairs; row IoU median 0.15 (wide) / 0.10 (small), after removing the vector 0.21 / 0.26.

| pair | width mm | offset mm | row IoU | shifted |
|---|---|---|---|---|
| Chort label = sticker | 30 | 5.4 | 0.19 | 0.21 |
| ethernet ports = LAN ports | 34 | 8.3 | 0.11 | 0.14 |
| Ethernet Port 1 = LAN 1G | 16 | 9.0 | 0.12 | 0.11 |
| Ethernet Port 2 = LAN 10G | 16 | 7.9 | 0.11 | 0.19 |
| Serial Ports Group = COM ports | 58 | 8.7 | 0.15 | 0.28 |
| Serial Port 1 = COM / DIO | 23 | 8.0 | 0.13 | 0.26 |
| Serial Port 2 = COM / CAN | 29 | 8.0 | 0.15 | 0.28 |
| Audio Port 1 = MIC IN | 11 | 6.7 | 0.10 | 0.34 |
| Audio Port 2 = LINE OUT | 9 | 7.1 | 0.04 | 0.43 |

Geometry also corrected two of my pairs: c5.6's "upper front panel" is the upper band of
c5.2's Front Face, not the bevel (IoU 0.02); b1.4's "front face" is another side of the box
than b1.1's strip (IoU 0.0).

**RQ3 / RQ4** (`evidence/agent_vs_truth.json`, `conflict_kinds.json`).

| run, mm, mode | view | recovered | missed | wrong | invented | within ok | new ok | conflicts + | agent turns / s |
|---|---|---|---|---|---|---|---|---|---|
| compute, 1, **agent** | c5.6 | **10** | 0 | 0 | 0 | 2 | 4 | 3942 | 6 / 30.5 |
| compute, 1, agent | c5.7 | 1 (object) | 0 | 0 | 0 | 0 | 18 | 215 | 5 / 17.0 |
| compute, 1, baseline | c5.6 | 1 (object) | **9** | 1 | 0 | 1 | 4 | 5346 | – |
| compute, 1, baseline | c5.7 | 1 | 0 | 0 | 0 | 0 | 18 | 215 | – |
| box, 1, agent | b1.4 | 0 | 0 | 0 | 1 (root) | 0 | 11 | 16 | 4 / 15.6 |
| box, 1, baseline | b1.4 | 0 | 0 | 0 | 0 (coarse) | 0 | 11 | 3 | – |
| compute, 2, baseline | c5.6 | 0 | 9 | 1 | 0 | 2 | 4 | 942 | – |

Conflict kinds after all views (1 mm, agent): compute 4956 rows = re-parent 2663 (e.g. the
Chort label hangs under the object in c5.2's tree but on the panel in c5.6's; the ports sit
under Front Face but inside the new "lower front panel"), intermediate 1262 (the new
"lower front panel" slid in above the existing ports), boundary 1031; box 33, all boundary.

**RQ5** (`evidence/views.json`, 1 mm, the agent STV).

| run | parts | computed ≥ step | face-on pose: visible ≥ 0.9 | centred ≤ 10 % | median incidence | **infeasible face-on poses** | shift rescues | 30° tilt costs |
|---|---|---|---|---|---|---|---|---|
| compute | 38 | 32 | 17 | 33 | 23° | 29 (clearance median 0.7 cm) | 22 | 18 % |
| box | 15 | 13 | 11 | 15 | 14° | 12 (median 1.1 cm) | 10 | 16 % |

Recorded steps had ≥ 2.8 cm clearance (median 6.7 cm). The best feasible candidate by family:
compute shifts 20, tilts 15, pivots 2, the face-on pose itself 1; box 10 / 0 / 2 / 3. Eight
parts (standoff 0.10 m unless the extent asks for more):

| part | extent mm | face-on pose: vis / inc / centring | best feasible | detecting step: vis / inc / centring |
|---|---|---|---|---|
| Ethernet Port 1 | 21×17 | 0.91 / 13° / 0.00 (infeasible, 0.9 cm) | tilt+15y 0.97 | step 6: 0.94 / 20° / 0.29 |
| Serial Port 2 | 27×20 | 0.96 / 15° / 0.00 | shift−2side 0.98 | step 6: 0.99 / 25° / 0.26 |
| Chort label | 32×16 | 0.95 / 22° / 0.01 | normal 0.95 | step 2: 0.90 / 51° / 0.36 |
| USB 3.2 Label 1 | 13×5 | 0.97 / 21° / 0.01 | shift−2side 0.96 | step 7: 0.96 / 30° / 0.07 |
| barcode | 56×19 | 0.99 / 13° / 0.01 | normal 0.99 | step 4: 0.99 / 17° / 0.10 |
| QR code | 10×9 | 0.44 / 7° / 0.02 | pivot5 1.00 | step 4: 0.92 / 24° / 0.36 |
| Front Face (c5.2) | 184×79 | 0.74 / 26° / 0.01 (standoff 0.19) | shift+4side 0.85 | step 2: 0.62 / 58° / 0.31 |
| Top Heatsink Face | 186×100 | 0.49 / 30° / 0.05 (standoff 0.19) | tilt+15x 0.94 | step 2: 0.88 / 45° / 0.05 |

**RQ6** (`evidence/sets.json`, min visible share over the set, best feasible candidate vs the
best recorded step): two LAN ports 0.96 tilt (step 0.94); the six lower-panel ports 0.89 pivot
(0.94); a port with its label 0.96 pivot (0.95); the sticker with the LAN ports 0.92 shift
(0.76); the four USB ports 0.96 shift (0.95); barcode + QR 0.99 shift (0.92). All six exist
above 0.8.

**Search** (`evidence/queries.json`): compute 11/12 queries at 1.0 / 1.0 ("power" recall 0.33
— my expected list named the DC block), "front face" precision 0.09 because a path match
returns everything under the face; box "ce" precision 0.06 (substring of "face").

## Limits

- One run per object, two or three views, 13 scored pairs; the agent ran once per view.
- The truth pairs are mine; two were wrong until the geometry corrected them. Anton checks them.
- The IoU of a part rendered back is bounded by 1 mm rows against 0.2 mm pixels; the real
  comparison is recall (0.9–1.0) and smear.
- V is never corrected for the per-view offset; a `same` extension therefore grows a part by
  ~8 mm (two offset copies).
- "Clearance" is the distance from the camera position to the nearest surface row, 2 cm; it
  is a free-space test, not a reach or collision check (the planner's).
- The score has no "is the detail legible" term: a 10.5 cm standoff at 89° FOV puts a 13 mm
  label on ~85 px; whether that reads is not measured here. Nor does it reward what the shift
  keeps and a tilt loses — an image plane parallel to the face (no keystone) — which is why
  the loop puts shifts before the cone.
- The standoff is a rule (half a frame, clamped at the sensor's minimum), not the loop's
  emergent "closest provably-safe sample"; no IK, no collision world — only 2 cm of free
  space around the camera position.
- The viewer's GUI was not exercised by hand (no browser automation in this session).

## What changes

- **S/T/V works as the structure.** Rows keyed by grid index, parts as sorted row arrays,
  one deepest label, nesting by construction. The merge question is now concrete: conflicts are
  T disagreements of two kinds — *re-parent* (a part's rows lie inside a non-ancestor) and
  *intermediate* (a new node between existing parent and children) — and a rule for each would
  remove 80 % of them. Anton's call.
- **The agent, not geometry, pairs parts across views** — but only when it sees the model
  projected into the frame and can reference new nodes as parents. Names are a trap both ways:
  it pairs "front face" with "front face" across sides, and the structure needs **aliases** to
  keep "LAN 1G port" on a part first called "Ethernet Port 1".
- **The view for a part is a geometry problem plus a feasibility problem.** Face-on at the
  sensor's minimum range is right for the parts of the top face and wrong for most parts near
  the table; the face-plane shift (the loop's own first rung for box faces) is the feasible
  answer for most of them, the tilts and the pivot ring for the rest; the recorded close-up
  steps were in fact 15–16 cm, 20–25° off-normal, 2.8+ cm clear. What e05 adds over the loop's
  box faces is the anchor (a part, not a face) and the scores only V can give (visibility,
  occlusion, incidence); what the loop has and e05 lacks is the emergent distance with IK and
  collision. Anton: "maybe at some point we move to planning against the voxel grid" — to be
  discussed against `view/rays.py` on his return.
- Three lifts, one answer: gate the ray cast with the view's own depth. The ray cast alone
  is not safe on a TSDF with holes.
- For alignment round v2: the eight assumed answers in the NOTE, plus: (1) re-parent / insert
  rules for the two conflict kinds; (2) whether a `same` part should be re-centred on the
  newest view (the offset); (3) the clearance threshold and who checks it; (4) the name trap:
  should `same` require a geometric overlap above some floor, or a second look.
