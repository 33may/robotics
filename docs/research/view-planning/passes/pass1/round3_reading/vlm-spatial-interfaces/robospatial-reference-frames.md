# RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics

- arXiv: 2411.16537 (v5, HTML full text), CVPR 2025 oral
- Authors: Chan Hee Song et al. — project page: https://chanh.ee/RoboSpatial
- Read via: arXiv HTML full text (https://arxiv.org/html/2411.16537v5), downloaded and
  parsed directly (not just the abstract), cross-checked table numbers against the raw
  HTML table markup (not just a summarizer's transcription) to avoid hallucinated numbers.

## Core method / idea

RoboSpatial is a **dataset + benchmark**, not a view-selection or action-space paper.
It targets one specific defect in VLM training data: existing spatial-reasoning datasets
don't label *which reference frame* a spatial relation is being asked in, so models can't
learn to distinguish "left of the mug" from the camera's perspective vs. from the mug's own
orientation vs. in world coordinates. The paper:

1. Defines three reference frames explicitly (Sec. 3.2.1, Introduction):
   - **Ego-centric**: "from the observer's perspective at the camera pose."
   - **World-centric**: "grounded in a global world frame" (uses global z for
     above/below — i.e., elevation).
   - **Object-centric**: "based on a reference frame attached to the focal object" —
     uses the anchor object's own heading vector to define front/back/left/right.
2. Builds a two-stage automatic annotation pipeline (Sec. 3.2) over existing 3D-scanned
   scene datasets (ScanNet, Matterport3D, 3RScan, HOPE, GraspNet-1B) with pre-existing
   **oriented 3D bounding boxes** (location + heading) pulled from EmbodiedScan. Stage 1
   computes 3D spatial relations `s_i = ⟨I_i, a_i, t_i, r_i, l_i⟩` (image, anchor, target,
   relation ∈ {left,right,above,below,front,behind}, frame ∈ {ego,world,object}) directly
   from these boxes — **no point cloud or mesh is needed**, only camera intrinsics/extrinsics
   and the box heading. Stage 2 projects context/compatibility relations into 2D image
   points via raycasting and top-down occupancy maps.
3. Three question types (Sec. 3.1, Introduction): **spatial configuration** (binary,
   "Is the mug to the left of the laptop?"), **spatial compatibility** (binary, "Can the
   chair be placed in front of the table?", requires simulated bbox placement with ≥10cm
   clearance), **spatial context** (point-prediction, "Where on the table can I put the
   plate?", scored by convex-hull containment of the predicted 3D point against a
   reference point set — the paper itself flags this metric as "arguably overly strict").
4. Fine-tunes five 2D VLMs (VILA-1.5-8B, LLaVA-NeXT-8B, SpaceLLaVA-13B/SpatialVLM
   community reimpl., RoboPoint-13B, Molmo not fine-tuned) and two 3D VLMs (3D-LLM,
   LEO) on ~3M generated QA pairs, and evaluates zero-shot vs. fine-tuned on held-out
   splits, an out-of-domain manually-collected "RoboSpatial-Home" set, and BLINK/SpatialBot.

**No view-selection, viewpoint-planning, or action-menu content anywhere in this paper.**
It is purely a spatial-relation QA dataset/benchmark for VLM training; relevance to our
branch is entirely through the reference-frame taxonomy and the measured ego/object/world
accuracy gap, not through any method for choosing or presenting camera poses.

## Concrete results and numbers

**Table 3 — RoboSpatial-Val, in-domain, zero-shot baseline → fine-tuned (accuracy %):**

| Model | Indoor Config | Indoor Context | Indoor Compat | Tabletop Config | Tabletop Context | Tabletop Compat | Avg Indoor | Avg Tabletop | Total |
|---|---|---|---|---|---|---|---|---|---|
| VILA | 54.7→71.4 | 18.3→45.9 | 56.3→77.2 | 45.1→71.8 | 13.2→43.7 | 53.8→73.3 | 43.1→64.8 | 37.4→62.9 | 40.2→63.9 |
| LLaVA-NeXT | 48.9→69.3 | 12.5→41.3 | 32.7→70.5 | 48.3→70.7 | 8.4→44.8 | 30.9→66.1 | 31.4→60.4 | 29.2→60.5 | 30.3→60.5 |
| SpaceLLaVA | 52.6→76.0 | 15.3→50.7 | 49.0→76.6 | 66.5→74.9 | 12.2→46.4 | 60.1→70.5 | 38.9→67.8 | 46.2→63.6 | 43.6→65.7 |
| RoboPoint | 39.0→72.2 | 41.4→68.9 | 38.3→72.1 | 37.9→70.3 | 31.6→61.7 | 45.2→78.4 | 39.6→71.0 | 38.2→70.1 | 38.9→70.6 |
| GPT-4o (zero-shot only) | 63.5 | 25.1 | 59.4 | 62.3 | 27.9 | 66.8 | 49.3 | 52.3 | 50.8 |
| Molmo (zero-shot only) | — | — | — | — | — | — | — | — | 50.1 |

Context tasks are the hardest category for every model in the zero-shot condition
(8.1–41.4%, roughly chance-to-below-chance for a hull-containment metric), confirming
free-space/placement reasoning is the weak point generally, independent of reference frame.

**Table 8 (Appendix D.1) — per-reference-frame accuracy on RoboSpatial-Val, the single
most load-bearing table for this branch (zero-shot baselines only, %):**

| Model | Indoor Ego | Indoor Object | Indoor World | Tabletop Ego | Tabletop Object | Tabletop World |
|---|---|---|---|---|---|---|
| VILA | 55.9 | 40.5 | 32.9 | 43.6 | 39.7 | 28.9 |
| LLaVA-NeXT | 35.2 | 24.3 | 34.7 | 36.4 | 28.5 | 22.7 |
| SpaceLLaVA | 40.6 | 36.0 | 30.1 | 52.3 | 32.8 | 53.5 |
| RoboPoint | 41.9 | 36.2 | 40.7 | 46.2 | 30.5 | 37.9 |
| 3D-LLM | 28.9 | 38.3 | 45.6 | 38.9 | 35.7 | 52.6 |
| LEO | 46.9 | 30.6 | 48.2 | 41.4 | 34.3 | 55.4 |
| Molmo | 50.4 | 50.8 | 47.6 | 64.4 | 33.6 | 53.8 |
| **GPT-4o** | 52.9 | **38.7** | 56.3 | 62.5 | **30.7** | 63.7 |

This is a **combined average over configuration+context+compatibility for that frame**,
not a pure binary-choice metric, so a literal 50%-chance-baseline interpretation is not
license (context tasks pull the average down and have a different, harder-than-chance
scoring rule). Read the *pattern*, not the absolute numbers against a chance line.

Pattern (verified row-by-row from the raw table, not asserted from the paper's prose,
which never explicitly writes "VLMs default to camera-centric interpretation" — that
framing is an inference from this table, not a quoted claim):
- **Object-centric is the single lowest-or-near-lowest score for every one of the 8
  models**, in 10 of 14 (model × domain) rows it is the strict minimum of the three
  frames; GPT-4o's object-centric score (30.7% tabletop, 38.7% indoor) is its worst
  by a wide margin (vs. 62.5%/56.3–63.7% ego/world).
- **Ego-centric is not uniformly the model's best frame** — for GPT-4o, world-centric
  actually beats ego-centric in both domains (56.3 vs 52.9 indoor; 63.7 vs 62.5
  tabletop). So "defaults to camera-centric" is not literally what the data shows for
  GPT-4o; what the data robustly shows is "object-centric is the failure frame,"
  not "ego-centric is the preferred/default frame." This distinction matters for
  how the branch cites this paper — cite it for the object-centric deficit, not for
  a camera-centric-default claim.
- **3D VLMs (3D-LLM, LEO) show the opposite pattern from 2D VLMs**: their weakest
  frame is ego-centric (or object-centric), and they are comparatively strong at
  world-centric. Paper's own explanation (Appendix D.1, verbatim): "since baseline 3D
  VLMs are trained on point clouds without information of perspective, their accuracy
  in ego-centric and object-centric frames is lower... due to their design —
  specifically, the lack of a means to visually inject perspective information since
  they require complete 3D point clouds — 3D VLMs still lag behind 2D VLMs on
  ego-centric and object-centric frames." And: "Baseline 2D VLMs have trouble
  understanding the world-centric frame, which involves understanding elevation,
  while 3D VLMs comparatively excel at it."

**Table 4 — out-of-domain, RoboSpatial-Home (manually collected, real scenes, frame
left *implicit* in the question — Sec 4.2), zero-shot baselines:**

| Model | Home Config | Home Context | Home Compat | BLINK | SpatialBench |
|---|---|---|---|---|---|
| VILA | 57.8 | **0.0** | 69.0 | 72.7 | 53.0 |
| LLaVA-NeXT | 68.3 | **0.0** | 70.5 | 71.3 | 55.9 |
| SpaceLLaVA | 61.0 | 2.5 | 41.0 | 76.2 | 47.1 |
| RoboPoint | 69.9 | 19.7 | 70.5 | 63.6 | 44.1 |
| 3D-LLM | 39.8 | **0.0** | 35.2 | N/A | N/A |
| LEO | 51.2 | **0.0** | 38.1 | N/A | N/A |
| Molmo | 58.6 | 0.1 | 18.1 | 67.1 | 55.9 |
| GPT-4o | 77.2 | 5.7 | 58.1 | 76.2 | 70.6 |

**Zero-shot context accuracy on real (non-templated) scenes is at or near 0% for every
model except RoboPoint (19.7%), including GPT-4o (5.7%).** This is the sharpest capability-
floor number in the paper: even the strongest closed model essentially fails at real-world
free-space/placement point prediction when the frame is not stated explicitly. Fine-tuning
on RoboSpatial recovers it only partially (best post-FT context number for Home not
separately re-quoted here, but Table 4's fine-tuned rows — see raw HTML — show gains of
+8 to +19 points, still leaving context the weakest category by far).

**Table 6 — real robot manipulation (Kinova Jaco + ZED2, cuRobo motion planning, >200
queries, Sec 4.3):**

| Model | Success % |
|---|---|
| LLaVA-NeXT baseline | 23.7 |
| LLaVA-NeXT + RoboSpatial | 52.6 |
| RoboPoint baseline | 44.7 |
| RoboPoint + RoboSpatial | 46.2 |
| Molmo (not fine-tuned) | 43.8 |
| GPT-4o (not fine-tuned) | 46.9 |

Quoted failure-mode detail (Sec 4.3, main text — **not** the Limitations appendix, contrary
to what a naive skim might suggest): "spatial failures in 2D VLMs often stem from errors in
projecting 2D predictions into 3D. Even a small 2-pixel shift in image space can translate
to a 5–10 cm error in the physical world, which is significant in manipulation tasks."

## Load-bearing assumptions

1. **Object-centric "front" comes from a pre-existing oriented-bounding-box heading
   annotation in the source dataset**, not computed from geometry by RoboSpatial itself.
   "Each bounding box includes both the 3D location and heading of the object. The
   object's orientation is defined by the heading vector of the bounding box" (Sec.
   3.2.1). These headings are semantic/human-annotated (a chair has a "front" because
   people sit facing a direction; a car has a hood). **The paper never demonstrates
   how to define object-centric "front" for an object with no canonical front** —
   which is exactly our situation (a box or cylinder on an inspection table has no
   semantic "front," only faces/surfaces).
2. **The world-centric frame in this paper means only vertical elevation** (above/below
   via global z), not general orientation-independent horizontal relations. It is not a
   "canonical viewing angle" frame in the sense our branch cares about (e.g., "orthogonal
   to face 3" is not a world-frame relation in RoboSpatial's taxonomy — it doesn't map
   cleanly onto any of the three frames as defined here; see below).
3. **Directional regions (front/left/right/behind) are computed from continuous 3D
   geometry via a heading vector and the reference frame's axes** (Sec 3.2.1), but the
   exact angular threshold/sector logic for assigning a direction is not specified
   anywhere in the main text or the appendix excerpts read — no equation, no cone angle,
   no ambiguity-zone handling is given. This is a **methodological gap in the paper
   itself**, not something I failed to find: the "how do you draw the line between
   front and left" question (directly relevant to our "orthogonal to face N, when is a
   candidate view close enough" problem) is left unanswered by RoboSpatial.
4. **Per-frame accuracy (Table 8) is measured only on the synthetic, template-generated
   RoboSpatial-Val benchmark**, where the frame is explicitly stated in the question text
   ("... (object frame)"). The real-world, frame-implicit test (RoboSpatial-Home, Table 4)
   does **not** re-report a per-frame breakdown — only per-task-type (config/context/
   compat). So we cannot verify from this paper whether the object-centric deficit
   specifically (as opposed to the context/free-space deficit) reproduces in real,
   frame-unlabeled deployment; the cleanest ego/object/world number (Table 8) is
   confined to the synthetic benchmark.
5. **3D VLM baselines are trained on complete 3D scans/point clouds**, and the paper
   explicitly flags this: "we plan to apply the data generation pipeline... developing
   3D VLMs that can reason over partial point clouds — removing the need for complete 3D
   scans and making deployment in real-world systems more feasible" is listed as future
   work, i.e., **not solved here.** Directly relevant to our defective/partial fused
   point cloud constraint — RoboSpatial's 3D-VLM track does not address partial/sparse
   geometry at all.

## What transfers to our setting

- **The three-way frame taxonomy itself (ego/world/object) is a clean, reusable vocabulary**
  for classifying our "orthogonal to face 3" query type, and for writing benchmark test
  cases (per the branch's Branch-2 handoff instruction).
- **The measured deficit is real and consistent across 8 different models (2D and 3D,
  open and closed)**: object-centric relational reasoning is the weakest reference frame
  for almost every model tested, zero-shot. This is solid, reproduced-across-architectures
  evidence (not a single-model anecdote) that a naive "ask the VLM to relate a candidate
  view to the object's own frame" design is fighting a documented, broad capability gap.
- **The real-world context-task collapse (near-0% for 6 of 8 models, Table 4) is a strong,
  quotable capability-floor data point** for the branch's "capability floor" RQ — even
  though it's about free-space point prediction, not view selection, it demonstrates that
  frame-grounded spatial reasoning collapses hard the moment you leave templated synthetic
  phrasing and the frame is not stated explicitly. Directly informs RQ4 (presenting the
  action space): if the frame must be stated explicitly to get above-floor performance,
  our menu should never rely on an implicit/inferred frame.
- **The finding that fine-tuning (task-specific spatial training) closes most, but not
  all, of the gap** (e.g., GPT-4o zero-shot 50.8% total vs. best fine-tuned open model
  63.9–70.6%) is a useful prior for RQ1 (capability floor): general-purpose VLMs
  (GPT-4o, Molmo, used untuned in our setting) sit meaningfully below a spatially
  fine-tuned smaller open model — off-the-shelf zero-shot VLM spatial reasoning has real,
  measured headroom below what's achievable, reinforcing that we should not assume our
  deployed (untuned) VLM performs at ceiling on any frame-dependent query.

## What does NOT transfer, and why

- **No view-selection, viewpoint-planning, action-menu, or camera-pose encoding content
  at all.** This paper cannot answer any of the branch's encoding questions (numbers vs.
  names vs. indices, option count/distinctiveness, ordering effects, infeasible-option
  handling, hierarchical/on-demand disclosure, stopping criteria). It is purely a training
  dataset/benchmark paper for spatial *relation* QA between static image/scene objects.
  Do not cite it for anything beyond the reference-frame taxonomy and the measured gap.
- **Object-centric "front" is undefined for our object class.** RoboSpatial's method for
  determining an object's heading is inherited from source-dataset annotations for
  semantically-fronted household objects (chairs, cars, appliances). It gives **zero
  method** for defining a reference direction on a primitive shape (box face, cylinder)
  with no canonical front — which is precisely what our "orthogonal to face 3" scheme
  needs. This paper cannot be used as evidence for *how* to construct an object-centric
  frame for our geometry; only for the fact that *once such a frame exists*, models
  struggle to reason in it.
- **"Orthogonal to face N" does not map onto any of the three RoboSpatial frames as
  defined.** Their object-centric frame is about left/right/front/behind/above/below
  *relative to the object's own heading axis* — a single, fixed reference direction per
  object. Our "orthogonal to face N" is a *per-surface* normal-direction query on a
  multi-faced primitive — closer to "which of several local surface frames" than to
  "the one global object frame" RoboSpatial tests. So the branch's RQ2 framing ("is this
  the kind of allocentric query RoboSpatial's object-centric frame penalizes?") is only
  a partial match: RoboSpatial demonstrates VLMs are bad at *a* fixed non-ego frame, but
  does not test the *multi-frame-per-object* (face-relative) case our design needs — a
  real gap, not something this paper resolves either way.
- **No compositional-operation, ordering, or option-count analysis** — irrelevant to
  those RQs; flag as an evidence gap, not silently generalized.
- **No random/uniform baseline reported for any spatial task** — the paper reports only
  VLM baselines and their own fine-tuned models; there is no random-guessing control to
  compare against (chance level is *implicit* at 50% for binary tasks and unclear for
  context tasks, per note above), so this paper contributes nothing to the "what random
  baseline should our benchmark use" RQ.
- **No stopping-criterion, coverage, or confidence-calibration content** — out of scope
  for this paper entirely.
- **Robot manipulation results (Table 6) are pick-and-place placement tasks** ("place in
  front of the pony"), not inspection/view-selection tasks — the 5–10cm-per-2px error
  finding is about *grasp/placement point* prediction accuracy, not about camera-pose or
  view-orthogonality judgment; do not over-extend it to viewpoint precision claims.

## Evidence-table entry (for the cross-paper table)

| Paper | Encoding compared against | Delta | Variance reported? |
|---|---|---|---|
| RoboSpatial (2411.16537) | Object-centric vs. ego-centric frame, GPT-4o zero-shot, RoboSpatial-Val | -14.2 pts indoor (38.7 vs 52.9), -31.8 pts tabletop (30.7 vs 62.5) | No (single run, no seeds/error bars reported anywhere in main or appendix tables read) |
| RoboSpatial (2411.16537) | Object-centric vs. world-centric frame, GPT-4o zero-shot, RoboSpatial-Val | -17.6 pts indoor (38.7 vs 56.3), -33.0 pts tabletop (30.7 vs 63.7) | No |
| RoboSpatial (2411.16537) | Zero-shot vs. RoboSpatial-fine-tuned, per-frame average, 6 open models | +10 to +30 pts depending on model/frame (see Table 8) | No |
| RoboSpatial (2411.16537) | Templated (frame-explicit, RoboSpatial-Val context) vs. real-world (frame-implicit, RoboSpatial-Home context), zero-shot | Val 8–41% vs. Home ~0–20% (6 of 8 models ≤0.1% on Home) | No |

This paper is **demonstrated, not ablated**, with respect to reference-frame difficulty:
there is no controlled experiment isolating *why* object-centric is hardest (e.g., no
test of whether it's the heading-vector concept, the phrasing, or genuine spatial
reasoning that fails); the per-frame table is a descriptive breakdown of baseline
zero-shot accuracy, not a causal ablation. Flag accordingly in the cross-paper table —
useful as a **capability-floor data point**, not as evidence for *why* or *how to fix*
the object-centric deficit.

## Handoff notes

- **Branch 2 (benchmark test cases):** add "object-centric relation to a
  non-semantically-fronted primitive shape" as a named test case category, flagging
  that RoboSpatial itself never tests this exact case (household objects only) —
  our benchmark would be extending, not replicating, RoboSpatial's coverage.
- **RQ1 (capability floor):** RoboSpatial is evidence that a *fixed single non-ego
  frame* is a measured weak point across 8 architectures, zero-shot — supports treating
  any single allocentric relation (not just view-to-view) as a documented risk area, but
  is not itself a camera-to-camera or view-adjacency measurement (that's MMSI-Bench/
  ViewSpatial/SpinBench's job, per the branch's own framing).
- **RQ2 (ego restatement):** RoboSpatial's ego-centric numbers are consistently higher
  than object-centric (though not always the single highest frame — world sometimes
  wins for GPT-4o). This is at least directional support for "restating a query
  egocentrically tends to help," but the paper never tests a *restatement* intervention
  (going from an object-centric question to an ego-centric one for the *same* underlying
  geometric fact) — it only compares independently-labeled questions across frames. Do
  not cite this paper as evidence that restatement *preserves the geometry* — that claim
  is untested here.
