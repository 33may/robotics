# GraspView: Active Perception Scoring and Best-View Optimization for Robotic Grasping in Cluttered Environments

- arXiv: 2511.04199 (v1, 6 Nov 2025), 9 pages, cs.RO
- Authors: Wang, Dai, Su, Liu, Chen, Wu, Lin — Peng Cheng Laboratory / SIAT-CAS /
  Peking Univ. Shenzhen / Sun Yat-sen Univ.
- Read via: full PDF (9 pages, all equations/tables/figures transcribed directly),
  cross-checked against arXiv HTML render for section numbering.

## Core method

GraspView is an **RGB-only** (no depth sensor) tabletop grasping pipeline with three
modules (Sec. III, Fig. 1): (a) global perception scene construction, (b) active
perception scoring (NBV selection), (c) best-view global grasping. The unifying trick:
**VGGT** (Wang et al., CVPR 2025 — a feed-forward transformer that jointly predicts
camera extrinsics + dense point/depth maps from one or a few RGB frames, up-to-scale)
replaces the depth sensor entirely. A VLM is used twice, in two structurally different
roles:

1. **Grounding** (Sec. III-B, Eq. 1-3): `π_VLM = g_VLM(I_t, q)` turns a language query
   ("grasp the red bottle") into 2D prompts (boxes/points), fed to SAM2 for a mask,
   back-projected through VGGT's point map into an object-centric point cloud.
2. **Render-and-score view scoring** (Sec. III-B "Render-and-Score NBV selection",
   Eq. 13-14; Sec. III-D, Eq. 20) — **this is the part matching our branch**. Candidate
   camera poses are generated **procedurally** (not by the VLM — see below), each is
   rendered off-screen from the *current* colored point cloud, and the VLM scores each
   rendered image on a 1-10 scale (Fig. 1(b)(i) shows per-view overlaid scores like
   "7.8", "8.9", "7.2", "5", "4.7", "8.4") for visibility / grasp-suitability /
   occlusion. The robot executes only the arg-max view, takes a *real* photo there, and
   fuses it via VGGT.

### Candidate generation is 100% deterministic geometry, not VLM choice (Sec. III-B, Eqs. 4-12)

This is the paper's answer to the "who picks the render's viewpoint" chicken-and-egg
problem, and it is clean and citable:

- **Conical spherical-cap sampler egocentric to the *current* camera**, not to any
  world/object frame. Viewing axis `a = (p0-c)/‖p0-c‖` where `p0` is the *current*
  camera center and `c` the object centroid (Eq. 4). An orthonormal basis
  `{u0,u1,u2}` is built with `u0 ∥ a`. Candidates are discretized by half-angle
  `α_i = (i/n_α)·α_max` and azimuth `β_j = 2πj/n_β` around that axis (Eqs. 5-6), giving
  candidate directions `d_{i,j} = cos(α_i)u0 + sin(α_i)(cos(β_j)u1 + sin(β_j)u2)`
  (Eq. 7) and positions `p_{i,j} = c + r·d_{i,j}` (Eq. 8) at the *same radius* `r` as
  the current view. Total candidates: **1 + n_α·n_β** — exact numeric values of
  `n_α`, `n_β`, `α_max` used in the real-robot experiments are **not given anywhere in
  the paper** (a genuine gap — flag, don't assume any specific count).
- **Orientation is fully deterministic too** — a look-at rotation toward the centroid
  plus a fixed roll-stabilization rule using the *initial* camera's up-vector as
  reference (Eqs. 9-12: `u_ref = R_{BC0} e_y`; forward/right/up computed by cross
  products). The VLM never picks or reasons about orientation — it only scores
  renders of positions that are already fully posed by geometry.
- **Infeasible candidates are silently dropped before the VLM ever sees them**: "We
  retain only kinematically feasible candidates (IK + joint/velocity limits) and
  select the NBV as p* = argmax_p S(p)" (Sec. III-B, directly under Eq. 13). This is
  the **silent-removal** answer to the branch's infeasible-view-handling question —
  GraspView does not list infeasible views as "unavailable" and does not score them
  down à la SayCan's affordance factor; they are filtered out of the candidate pool
  entirely, upstream of scoring.

### The scoring function (Eq. 13, Sec. III-B / III-C)

```
S(p) = w_v·s_vis(Î_p) + w_g·s_grasp(Î_p) - w_o·s_occl(Î_p)
```

`s_vis`, `s_grasp`, `s_occl` are VLM-derived, normalized to [0,1]; `w_v, w_g, w_o ≥ 0`
are combination weights. **Numeric weight values are never given** — another
reproducibility gap. `v*_NBV = argmax_vi S(vi)` (Eq. 14). At the best-view-grasping
stage (Sec. III-D, Eq. 20), previously-traversed candidates are **re-scored** by the
VLM with a differently-emphasized score `S'` ("emphasizes complete visibility of the
target object") to pick the final grasp viewpoint `v* = argmax_vi S'(vi)`. Two
different scoring criteria at two pipeline stages, both VLM-scalar-per-image, no
menu-style single joint query across images is described in the text — Fig. 1(b)(i)'s
per-view score badges are consistent with **independent, per-candidate VLM calls**
rather than one VLM call comparing a bundled menu of images (the text never says
"compare the following N images" — it says "the VLM scores each rendering," singular).
This is a real, if implicit, design choice worth flagging for the branch's "how many
options / how similar" question: **GraspView appears to sidestep the many-similar-
options-in-one-prompt problem entirely by never putting more than one candidate image
in front of the VLM at a time**, converting "pick from a menu" into "score each item,
then argmax in code." Caveat: this is inferred from the figure and prose structure,
not an explicit implementation detail stated in the text — flag as **PLAUSIBLE, not
CONFIRMED**.

### Rendering / point-cloud quality (Sec. III-B "Render-and-Score", one paragraph)

"We synthesize an off-screen rendering Î_p of the *current* colored object cloud using
intrinsics K and a pinhole model (with near/far planes set by workspace limits). We
color points by back-projecting their appearance from the latest RGB using the known
extrinsics, and splat them with Z-buffering to obtain a clean 2D view." That is the
**entire** treatment of render quality. No discussion of holes, sparsity, or
colorless/textureless geometry anywhere in the paper. Two structural reasons this
gap is less damaging for GraspView than it would be for us: (1) the "point cloud" is
not a real depth-sensor scan but a **VGGT-predicted dense point map** — a learned
completion, not a raw scan with sensor dropout, so it tends to be visually dense/plausible
within its predicted extent even where it's geometrically wrong (up-to-scale, sometimes
inaccurate) rather than literally sparse/holed like a fused real-sensor cloud; (2) at
most 1-3 real photos are ever fused (Table II caps useful NBVs at ~2), so the "current
colored point cloud" being rendered is built from very few views and covers a small,
recently-observed region — the render is never asked to represent a large, mostly-unseen
survey cloud the way ours would be.

### Online metric alignment (Sec. III-C, Eqs. 15-19) — tangential to view selection

VGGT's predictions are up-to-scale; a scale factor `λ*` is recovered online by
least-squares alignment (Eq. 17-18, closed form Eq. 18) between robot-kinematics
translation deltas and VGGT-predicted translation deltas across the 1-3 executed NBV
images. Table IV: alignment drops rotation error 3.46°→0.72° and translation error
0.13m→0.11m vs. no-alignment (fixed λ=1.00). Out of this branch's scope (metric-pose
territory), but it establishes that GraspView's action space itself is metric/Euclidean
throughout (poses in the robot base frame) — the VLM is never asked to reason about or
emit any of this; it only scores images.

## Concrete results and numbers (Sec. IV-C)

**Table I** — 8 tabletop objects (kettle, bottle, carambola, banana, mango, tape,
ruler, ketchup), 3 methods × 2 occlusion conditions, success rate (SR, %), single-run
numbers, **no N per cell, no variance/error bars anywhere in the paper**:

| Method | w/o occ. avg SR | w/ occ. avg SR |
|---|---|---|
| RGB-D (SV, RealSense D455) | 60.6 | 23.1 |
| GraspView (SV, no active perception) | 76.9 | 33.1 |
| GraspView (MV, with NBV active perception) | 80.0 | 75.0 |

Under occlusion, active-perception NBV selection is the single largest lever: **+41.9
points SR over the RGB-D baseline (23.1→75.0) and +41.9 points over the
no-active-perception ablation (33.1→75.0)**. This is a system-level ablation
(active-perception on/off), **not** an ablation of the VLM-scoring mechanism itself
against any alternative (random view, hand-crafted coverage/information-gain
heuristic) — see "evidence discipline" below.

**Table II** — NBV budget × occlusion difficulty (easy/medium/hard, subjectively
categorized by "how much of the target is visible," no formal metric given):

| NBVs | Easy | Medium | Hard |
|---|---|---|---|
| 0 | 80 | 65 | 20 |
| 1 | 90 | 80 | 55 |
| 2 | 95 | 85 | 70 |
| 3 | 95 | 90 | 70 |

First NBV: **+35 points** on hard cases (20→55). Second NBV: **+15 more** on hard
(55→70). Third NBV: **+0** on hard (70→70), +5 on easy/medium ceiling effects. This is
the paper's entire "stopping criterion" content — a saturation curve from a manual
budget sweep (0/1/2/3), used to justify a fixed practical recommendation of 1-2 NBVs.
**No adaptive/confidence-based stopping rule, no conformal calibration, nothing
resembling Explore-until-Confident** — this paper cannot serve as positive evidence for
that RQ, only as one more data point that empirically 1-2 targeted views captures most
of the achievable gain in a small-clutter tabletop setting.

**Table III** — working distance sensitivity, RGB-D (RealSense D455) vs. GraspView
(VGGT), SR%:

| Distance | RGB-D | GraspView |
|---|---|---|
| 30 cm | 10 | 80 |
| 50 cm | 40 | 85 |
| 70 cm | 85 | 85 |

RGB-D collapses at close range (this is the D455's near-field depth failure, a known
active-stereo/ToF limitation — directly relevant background for why our D405 fused
cloud might have its own near-field or glossy-surface holes, though GraspView doesn't
measure or discuss D405 specifically).

**Table IV** — scale/pose alignment, discussed above.

## Load-bearing assumptions

1. **The rendered "point cloud" is a learned monocular reconstruction (VGGT), not a
   real sensor scan.** No stereo/ToF noise model, no glossy-surface dropout, no holes
   from occlusion in the geometric sense our D405 survey cloud has — VGGT fills in a
   plausible (if metrically uncertain) dense surface. This is the single biggest
   assumption that does **not** transfer to our defective, real-scanned, hole-ridden
   PLY.
2. **Candidates are always local, re-sampled fresh around the *current* camera pose**,
   never drawn from a fixed, named, persistent action space (no "face 1...face 6"
   labels, no adjacency graph, no viewsphere). There is no menu the VLM is shown in
   text form at all — the interface is purely "render, score one image, argmax in
   code." This sidesteps essentially every text-encoding question this branch asks
   (numeric vs. named pose, ordering, option count) by never putting geometry into the
   text channel in the first place.
3. **Orientation is never a VLM decision.** The look-at + roll-stabilization rule
   (Eqs. 9-12) is pure geometry. The VLM's entire contribution is a scalar quality
   judgment on an already-fully-specified rendered image. This matches (and supports)
   our own project's allowed pattern: "primitive-level deterministic rules are
   acceptable" — GraspView is a working existence proof of exactly that split
   (deterministic pose generation + VLM picks-among-renders).
4. **Small clutter, few objects, near-field tabletop, ≤3 fused views.** Never stress-
   tested with a large action space, many near-identical candidates, or a persistent
   multi-session survey cloud.
5. **No comparison of the VLM's contribution against any non-VLM baseline.** No
   random-view baseline, no hand-crafted information-gain heuristic baseline for the
   *view selection step specifically* (only "no active perception at all" is compared,
   which conflates "no NBV" with "no VLM scoring"). This is a serious evidence gap for
   any claim that VLM scoring specifically (rather than merely moving to a second
   viewpoint) is what drives the gains.

## What transfers to our setting

- **Deterministic geometric pose generation + VLM-as-scorer split.** Directly
  supports our allowed architecture: generate candidate poses by primitive-level
  deterministic rule (their conical cap sampler around current axis is analogous to
  "box face normal" / "cylinder orbit angle" generation), let the VLM only judge
  rendered images, never verbalize the geometry. This is the cleanest, most directly
  reusable pattern in the paper for RQ4 (presenting the action space).
- **Egocentric-by-construction candidate framing.** The conical sampler is defined
  relative to the *current* camera's line of sight (Eq. 4's `a`), never in a world/
  object allocentric frame the VLM has to reason about verbally. This is a second,
  independent data point (after MapGPT) that the way to dodge the ego/allo gap is to
  make the *candidate geometry itself* egocentric, rather than asking the VLM to
  translate an allocentric instruction ("orthogonal to face 3") into egocentric terms
  itself. Directly answers part of the branch's RQ2: restate at the geometry layer,
  before the prompt is built, not in-context by the model.
- **Silent removal of infeasible views is a validated, if unexamined, choice**: IK/
  joint-limit-infeasible candidates never reach the VLM. Usable as one concrete
  existing-system data point for the branch's infeasible-view-handling question,
  though the paper offers no comparison against listing-as-unavailable or scoring-down
  (SayCan-style) — it simply picked silent removal and didn't ablate the choice.
- **NBV-budget saturation curve (Table II)** is a real, if coarse, data point for
  "how many views are enough" in a small-clutter regime: most of the achievable gain
  (especially in hard occlusion, +35 of the total +50 points) comes from the *first*
  additional view; a second view gives a much smaller increment; a third gives
  essentially nothing. Useful prior for setting an evaluation budget, not a stopping
  *rule*.
- **Working-distance failure of RGB-D active-stereo at close range (Table III)** is
  independent corroborating background for near-field depth-sensor unreliability,
  relevant context (not proof) for why our D405 fused cloud might carry near-field
  holes even though GraspView doesn't test a D405 or report hole statistics.

## What does NOT transfer, and why (negative findings, first-class)

- **No evidence on sparse/holed/colorless real-scanned geometry.** GraspView's
  "colored point cloud" is VGGT-predicted, learned-dense, and never described as
  having holes, colorless patches, or glossy-surface dropout. This paper gives **zero**
  evidence for the branch's core "what did GraspView do about sparsity, holes, and
  colorless geometry" question beyond "nothing — they didn't have that problem in the
  form we have it." Explicitly flag this as a non-answer, not a weak answer, when
  compiling the cross-paper comparison for Branch 3's handoff.
- **No random or hand-crafted-heuristic baseline for view selection.** The paper's
  own related-work section (Sec. II-B) explicitly claims render-and-score "removes the
  need for bespoke heuristics" and can "incorporate semantic cues" versus classical
  hand-crafted NBV utilities — but this claim is **argued, not measured**: no
  experiment compares VLM scoring against a coverage/information-gain heuristic, and
  no experiment compares VLM scoring against a random candidate pick. The only ablation
  is "active perception (any kind) vs. none" (Table I, SV vs MV) and "NBV budget count"
  (Table II) — neither isolates the VLM's scoring quality from the mere fact of taking
  a second look. This is exactly the "random baseline" gap the branch questions flag as
  necessary before believing any VLM view-selection claim — GraspView does not clear
  that bar.
- **No text/language encoding of the action space at all**, so it cannot adjudicate the
  numeric-vs-named-vs-indexed pose encoding contradiction (RQ4 in the branch list,
  Thinking-with-Novel-Views vs. MapGPT) — there is no text encoding to compare, only
  images. Do not cite GraspView as evidence on either side of that contradiction.
- **No option-count/order/distinctiveness stress test.** Total candidates
  `1 + n_α·n_β` with unspecified `n_α, n_β` (likely small, given the paper never
  mentions latency or context-length issues), and — per the per-image-scoring
  inference above — candidates are (plausibly) never jointly compared in one VLM call,
  so the branch's "how many options, how similar" and "ordering matters how much"
  questions get **no data** from this paper, positive or negative. This is a genuine
  blank, not weak evidence either way — the paper's design may have avoided the
  problem rather than solved it.
- **No stopping-rule / calibration mechanism** — Table II is a manual sweep used to
  justify a fixed practical budget (1-2 views), not an adaptive or calibrated
  criterion. Do not cite as evidence for a defensible "enough views" rule; cite only
  as a saturation-curve data point.
- **No hierarchical/on-demand structure, no persistent named topology or adjacency
  graph** — candidates are regenerated fresh, locally, every step from the current
  camera pose and object centroid; there is no query interface, no collapse/expand,
  nothing resembling SayPlan or a Cypher query tool. This is consistent with (but not
  positive evidence for) the idea that a small local action space doesn't need
  hierarchical exposure — GraspView never had a large space to begin with, so it
  cannot answer whether dumping vs. querying matters at *our* scale.
- **No reported spatial-reasoning failure modes** (no wrong-face selection, no ego/
  allo confusion, no order sensitivity, no OCR/mark dependence, no viewpoint-ID
  hallucination reported or discussed). The paper's only stated limitations concern
  scale ambiguity (addressed by their alignment module) and NBV saturation — nothing
  usable for Branch 2's named-failure-mode handoff list.
- **Different robot/camera from ours**: Kinova Gen3 + wrist RealSense D455 operated
  RGB-only (depth deliberately disabled) vs. our UR5e + eye-in-hand D405 with a
  fused, real, depth-sensed survey cloud. GraspView's whole point is *avoiding* the
  depth-sensor defect problem we are trying to survive; it is evidence about a
  different failure regime (monocular reconstruction uncertainty), not about scan
  defects.

## Evidence-table entry (for the cross-paper table)

| Paper | Encoding compared against | Delta | Variance reported? |
|---|---|---|---|
| GraspView (2511.04199) | Active perception (NBV, MV) vs. no active perception (SV), w/ occlusion | +41.9 pts avg SR (33.1→75.0) | No |
| GraspView (2511.04199) | GraspView (MV) vs. RGB-D (SV) baseline, w/ occlusion | +51.9 pts avg SR (23.1→75.0) | No |
| GraspView (2511.04199) | 1 NBV vs. 0 NBV, hard occlusion | +35 pts SR (20→55) | No |
| GraspView (2511.04199) | 2 NBV vs. 1 NBV, hard occlusion | +15 pts SR (55→70) | No |
| GraspView (2511.04199) | 3 NBV vs. 2 NBV, hard occlusion | +0 pts SR (70→70) — saturation | No |
| GraspView (2511.04199) | Online metric alignment vs. no alignment (fixed λ=1) | rot 3.46°→0.72°, trans 0.13m→0.11m | No |
| GraspView (2511.04199) | **VLM scoring vs. any non-VLM baseline (random/heuristic)** | **not measured — no such ablation exists** | N/A |
| GraspView (2511.04199) | **Text/coordinate pose encoding vs. image render** | **not measured — no text encoding used at all** | N/A |

All entries above are system-level, single-run, unreported-variance, 8-object,
real-robot trials — flag every row as demonstrated, not ablated for the VLM-scoring
mechanism specifically, per the branch's evidence-discipline requirement.

## Handoffs

- **Branch 5 (topology/adjacency)**: GraspView needed no persistent adjacency graph —
  a fresh, small, egocentric local neighborhood regenerated each step was sufficient
  for its (small-clutter, ≤3-view) task. Does not *forbid* Branch 5's adjacency-graph
  ambition, but is a working counter-example showing the graph may be unnecessary
  machinery for a small view budget — worth weighing against RQ3's tractability
  pressure.
- **Branch 3 (geometry-conditioned ranking on defective clouds)**: GraspView's
  "colored point cloud" is VGGT-predicted dense reconstruction, not a real fused scan.
  It offers **no evidence** that render-and-score survives a genuinely sparse/holed/
  textureless survey cloud — flag as an open question, not a transferred capability.
- **Branch 2 (BENCH-tagged metrics / permutation test)**: No usable metric
  (View Coverage IoU / projected-area score) and no permutation-order test exist in
  this paper — do not tag any entry here as BENCH; it is a systems paper, not a
  benchmark paper.
- **RQ4 (numeric/named/indexed pose encoding contradiction)**: GraspView is silent —
  no text pose encoding is used anywhere. Cannot help resolve the Thinking-with-
  Novel-Views vs. MapGPT contradiction; exclude from that table.

## Architecture verdict contribution

GraspView's VLM plays a **narrow, read-heavy, parallelizable role**: score one
rendered image at a time against a fixed rubric, no dependency between candidate
scores, no negotiation between candidates required (argmax happens outside the model).
This is closer to the "orchestrator-worker, parallel, read-heavy" side of the
branch's final architecture question — but note the orchestrator here is *not blind*:
it is the same VLM doing the seeing (scoring the render directly), not a separate
blind planner receiving a caption or a report from a vision subagent. GraspView
therefore is not a clean data point for the blind-orchestrator-plus-vision-subagent
question — it collapses "orchestrator" and "seeing" into one VLM call per candidate,
sidestepping the hand-off question rather than resolving it in either direction.

## Structured summary

**Gist**: GraspView replaces RGB-D grasping with VGGT-predicted monocular
reconstruction, deterministic egocentric conical-sampler pose generation, and a VLM
that scores off-screen renders (visibility/grasp/occlusion, 1-10) one image at a time
to pick the next-best-view, purely by argmax — no text encoding of geometry anywhere.
On 8 tabletop objects it beats RGB-D substantially under occlusion (75% vs 23.1% SR)
and at close range (80% vs 10% SR at 30cm), with gains saturating after ~2 NBVs. All
results are single-run system-level ablations (active-perception on/off, NBV count,
alignment on/off) — none isolate VLM-scoring quality against a random or hand-crafted
baseline, and none touch sparse/holed real-scanned geometry, text pose encoding, or
option-count/order effects.

**Relevance**: High for the deterministic-geometry-plus-VLM-scorer architecture
pattern and the egocentric-candidate-framing trick (both directly reusable); low-to-
none for the branch's harder evidence questions (random baseline, defective-cloud
rendering, text pose encoding, option-count/order, stopping calibration, named
failure modes) — this paper mostly demonstrates that avoiding those questions
altogether (never verbalize geometry, never batch candidates, silently drop
infeasible views) is a viable design, not that any of the branch's open questions
have measured answers.
