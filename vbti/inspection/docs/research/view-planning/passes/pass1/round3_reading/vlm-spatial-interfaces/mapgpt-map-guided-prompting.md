# MapGPT: Map-Guided Prompting with Adaptive Path Planning for Vision-and-Language Navigation

- arXiv: 2401.07314 (v3, HTML), accepted ACL 2024
- Authors: Chen, Judge et al. — project page: https://chen-judge.github.io/MapGPT/
- Read via: arXiv HTML (experimental) full text, multiple targeted fetches (abstract, method, tables, ablations, failure modes)

## Core method

MapGPT is a zero-shot LLM/VLM agent for Vision-and-Language Navigation (VLN) in the
**Matterport3D simulator** — "a discrete navigation environment with predefined
navigable viewpoints" (Sec. 3.1). This is the load-bearing environmental assumption:
the world is *already* a graph. Panoramic images exist at a fixed set of real-building
viewpoints, and ground-truth connectivity between them is provided by the simulator.
MapGPT does not build this graph from geometry — it only tracks which portion of a
**pre-existing** graph the agent has visited.

Three components:

1. **Online map construction (Sec. 3.1.2).** "At each step t, we record all observed
   nodes along the navigation trajectory and their connectivity into the graph
   Gt = {Vt, Et}." Nodes are typed: explored `{en_0 ... en_t}`, currently-accessible
   `{an_0^t, an_1^t, ...}`, and unexplored-but-known `{un_0, un_1, ...}`. The map is
   **not** built from scratch each turn — it accumulates as the agent explores,
   and it can shrink/reshape (adaptive planning can backtrack into it).

2. **Map-guided text prompting (Sec. 3.2.1).** The map is verbalized as short
   connectivity sentences, not coordinates: `"Place {en_0} is connected with Places
   {an_0^0}, ... Place {en_t} is connected with Places {an_t^i}"`. Node IDs are
   plain integers assigned by discovery order — arbitrary labels, not directions or
   semantics. Candidate actions are then listed with a letter label, a directional verb,
   the place ID, and a short observation: `"{label} {direction} Place {id}:
   {observation}"`, e.g. `"turn around to Place 1: <a room with blue walls>, which
   also includes <bed, curtain, picture>"`. Directions used throughout are strictly
   **egocentric movement primitives**: "go forward to", "turn left to", "turn right
   to", "turn around to", "go up to", "go down to" — never allocentric/object-relative
   labels like "opposite" or "orthogonal to". `at,0` is hard-coded as `"A. stop"`
   (Sec. 3.1.2) — stop is always an explicit, always-available menu item, not inferred.

3. **Adaptive path planning (Sec. 3.2.2).** Instead of storing a full step-by-step
   thought history, the agent maintains and rewrites a **multi-step plan** each turn:
   the previous plan is fed back in as context, and the agent decides at each step
   whether to continue it or **backtrack to a previously-seen-but-unexplored node**.
   This plan (not the raw history) is what carries state forward, explicitly to
   bound token growth and reduce dependence on possibly-wrong past decisions.

Two model variants are compared head-to-head on the identical map/prompt scaffold:
- **MapGPT + GPT-4** (text-only, "blind"): candidate nodes are described by
  **off-the-shelf caption/detector models, not the LLM's own vision** — "we follow
  NavGPT and utilize off-the-shelf vision models BLIP-2 and Faster R-CNN to acquire
  scene description and object detection respectively" (method text). This is a
  literal **caption hand-off** architecture.
  MapGPT+GPT-4 is a two-stage pipeline (vision model → caption → blind LLM).
- **MapGPT + GPT-4V** ("seeing"): "we directly require the agent to refer to images
  corresponding to various places. At the beginning of the user messages, we input
  the image IDs and observed images in an interleaved format" — actual candidate
  images, interleaved with the same textual map, go straight into the VLM context.

## Concrete results and numbers

**Table 1 (R2R, 72-scene subset, sampled for API-cost reasons; "Exp" = number of
GPT experts in the pipeline):**

| Method | LLM | Exp | NE↓ | OSR↑ | SR↑ | SPL↑ |
|---|---|---|---|---|---|---|
| NavGPT | GPT-3.5 | 3 | 8.02 | 26.4 | 16.7 | 13.0 |
| MapGPT | GPT-3.5 | 1 | 8.48 | 29.6 | 19.4 | 11.6 |
| DiscussNav | GPT-4 | 5 | 6.30 | 51.0 | 37.5 | 33.3 |
| MapGPT | GPT-4 | 1 | 5.80 | 61.6 | 41.2 | 25.4 |
| MapGPT | GPT-4V | 1 | 5.62 | 57.9 | **47.7** | **38.1** |

**Table 3 (REVERIE, 500-sample subset of val-unseen, re-tested against
retrained HAMT/DUET on the same subset):**

| Setting | Method | OSR↑ | SR↑ | SPL↑ |
|---|---|---|---|---|
| Train (from scratch) | Seq2Seq | 8.07 | 4.20 | 2.84 |
| Train (from scratch) | FAST-MATTN | 28.2 | 14.4 | 7.19 |
| Pretrain+finetune | HAMT | 35.4 | 31.6 | 29.6 |
| Pretrain+finetune | DUET | 50.0 | 45.8 | 35.3 |
| Pretrain+finetune | LAD | 64.0 | **57.0** | **37.9** |
| Zero-shot | NavGPT (GPT-3.5) | 28.3 | 19.2 | 14.6 |
| Zero-shot | MapGPT (GPT-4) | **42.6** | 28.4 | 14.5 |
| Zero-shot | MapGPT (GPT-4V) | 36.8 | **31.6** | 20.3 |

Note the **capability floor** signal: even the best zero-shot LLM/VLM navigator
(GPT-4V) lands at SR 31.6 on REVERIE vs. fine-tuned DUET's 45.8 and LAD's 57.0 —
zero-shot map-guided prompting narrows but does not close the gap to supervised
methods; it is well above the NavGPT zero-shot baseline (19.2) but well below
saturation.

**GPT-4 (blind, caption-fed) beats GPT-4V (seeing) on REVERIE OSR (42.6 vs 36.8)
while losing on SR (28.4 vs 31.6) and SPL (14.5 vs 20.3).** The paper's own
explanation: "due to information loss in the two-stage pipeline, MapGPT (GPT-4
based) tends to continue exploring when encountering insufficient description" —
i.e., caption hand-off produces *broader, less committed* exploration (the correct
node gets visited more often, inflating oracle success) but *worse convergence* to
a confident final answer and a less efficient path. This is a direct, quotable
data point for "what does a caption hand-off destroy": not coverage, but decisive
identification and path efficiency.

**Table 4 (ablation, ties encoding claims to numbers):**

| LLM | Map | Planning | NE↓ | OSR↑ | SR↑ | SPL↑ |
|---|---|---|---|---|---|---|
| GPT-4 | none | none | 6.49 | 49.5 | 32.9 | 19.4 |
| GPT-4 | Topological | none | 6.40 | 59.7 | 37.5 | 24.8 |
| GPT-4 | Topological | Adaptive | 5.80 | 61.6 | 41.2 | 25.4 |
| GPT-4V | none | none | 5.96 | 58.8 | 42.6 | 34.7 |
| GPT-4V | **Coordinate** | none | 6.12 | 55.1 | 41.2 | 32.8 |
| GPT-4V | **Topological** | none | 5.89 | 56.5 | 44.9 | 36.5 |
| GPT-4V | Topological | Action | 5.82 | 58.3 | 45.4 | 35.6 |
| GPT-4V | Topological | Adaptive | 5.62 | 57.9 | 47.7 | 38.1 |

This IS an ablation, not just a demonstration: topological map beats coordinate map
by **+3.7 SR / +3.7 SPL / +1.4 OSR** at matched planning setting (GPT-4V, no
adaptive planning). Topological map alone beats no-map by **+4.6 SR / +5.4 SPL**
(GPT-4, no adaptive planning). Adaptive planning adds a further **+3.7 SR** on top
of topological map + GPT-4 (37.5→41.2), and **+2.3 SR** on top of GPT-4V (45.4→47.7,
comparing "Action" vs "Adaptive" planning rows — "Action" appears to mean per-step
action prediction only, without the multi-step revisable plan; this distinction is
inferred from row structure, not independently re-verified against the method text).
**No variance/error bars or multi-seed numbers are reported anywhere in the tables
fetched** — single-run point estimates throughout. Flag as unablated-for-variance.

The "coordinates confuse GPT" claim is stated twice in the method text, not just
implied by the ablation: "we observe that GPT-4V struggles slightly in its attempts
to understand navigation environments based on multiple precise coordinates" and
"we discover that it seems challenging for GPT to understand precise coordinate
data" — "GPS coordinates" are explicitly named as the discarded quantity: "discard
the precise GPS coordinates that are difficult for GPT to understand, while
preserving the topological relationships."

**Stated failure modes (Appendix B, thin):** "The agent may fail to follow the
details in the instructions accurately"; "The scenes are highly challenging, and the
instructions may not provide sufficient clues." These are language-grounding
failures (instruction-following), not spatial-reasoning failures (no wrong-face,
no ego/allo confusion, no order-sensitivity, no mark/OCR dependence reported) — thin
value for the Branch-2 failure-mode handoff list.

**Stated limitation (verbatim):** "MapGPT is only experimented within a simulator
that incorporates certain ideal assumptions. Developing LLM-based agents directly
in the real world and addressing various real-world challenges would be a
meaningful future direction."

## Load-bearing assumptions

1. **The graph is given, not perceived or constructed.** Simulator ground-truth
   connectivity between panoramic viewpoints is handed to the agent; MapGPT never
   solves "which places exist and how are they connected" from raw sensor data —
   it only tracks discovery order over a pre-existing graph. This is the single
   biggest assumption that does not hold in our setting.
2. **Nodes are large-scale, visually distinct places** (different rooms/hallways in
   a building) reached by walking — not near-identical viewpoints of one small
   object. Distinctiveness of options is free in this domain; MapGPT never has to
   discriminate between options that look nearly the same.
3. **"Coordinates confuse the model" was tested for GPS-like world coordinates**
   in a large multi-room building (many digits, arbitrary global reference frame),
   at matched planning setting, on GPT-4V. It was not tested for small local
   offsets, angles, or object-relative geometric labels ("face 3", "45° from
   current pose") — a materially different regime from ours. Extrapolating the
   claim to our small local geometry is not directly licensed by this paper.
4. **Directional labels are strictly egocentric** (forward/left/right/up/down/
   turn-around relative to current heading) — the paper never asks the model to
   reason about an allocentric relation like "opposite face" or "orthogonal to
   surface N"; it sidesteps that problem entirely by keeping every action a
   first-person move.
5. **Node IDs are meaningless integers** used purely as pointers into the
   connectivity sentences, not as semantic/geometric labels (unlike a hypothetical
   "face 1, face 2, ..." scheme, where the ID itself carries geometric meaning and
   ordering).
6. Photorealistic real-building panoramas throughout — no point clouds, no sparse
   or defective geometry, no renders of holes/dropout. Irrelevant to the
   "seeing a defective point cloud" branch question.

## What transfers to our setting

- **Encoding pattern**: verbalize adjacency as short declarative sentences with
  small integer/letter IDs; keep continuous/metric quantities out of the text
  channel that the model reasons over. Directly supports keeping our face/orbit
  action menu ID-based and metric-free in the text portion of the prompt.
- **Action-menu format**: `{label} {direction/relation} {ID}: {short description}` —
  a reusable template for a discrete geometric action menu (maps cleanly onto
  "{label} orthogonal-to Face {id}: {short description}" for our case), with a
  hard-coded, always-listed terminal action (their "stop") — directly analogous to
  a "done/enough views" action in our menu.
- **Adaptive/revisable plan token** carried across turns instead of full history —
  relevant to our RQ on token growth and multi-step coverage planning; also
  supplies an explicit **backtrack** primitive (return to a previously seen but
  unresolved location), which is a candidate pattern for "revisit a face we
  under-inspected."
- **Egocentric-only action labels function well and avoid the allocentric-query
  problem entirely** — indirect support for the branch's proposed workaround
  (RQ2: restate "orthogonal to face 3" egocentrically) rather than counter-evidence;
  MapGPT never needed allocentric labels because its domain structure let it avoid
  them, which is itself informative: when you can phrase actions egocentrically,
  strong zero-shot performance is achievable (SR 47.7% well above NavGPT's 16.7%
  baseline) without ever testing allocentric relation words.
- **Caption hand-off vs seeing, head-to-head, same scaffold**: the one clean,
  controlled data point in the whole paper. Directly usable as evidence for the
  "what does caption hand-off destroy" RQ — it destroys **decisiveness and path
  efficiency** (SR/SPL drop) while **not necessarily destroying coverage** (OSR
  can even improve) because a blind, under-informed agent keeps exploring instead
  of confidently committing.

## What does NOT transfer, and why

- **No geometric construction of the action space** — MapGPT assumes the graph;
  our problem is precisely to build the graph/menu from a fused, defective point
  cloud. This paper offers zero evidence on that step (explicitly out of this
  branch's scope, but worth flagging so it isn't mistakenly cited as evidence for
  RQ2/RQ3 geometry work in Branch 5).
- **No evidence on near-identical option discrimination.** All "Handoff to Branch 2"
  option-count/order questions (how many options, how similar may they be) are
  untestable from this paper — MapGPT's candidate nodes are always visually and
  semantically distinct rooms. Cannot be used as evidence either for or against the
  "count vs. distinctiveness" hypothesis; it's simply outside the paper's design
  space. Should be flagged as an evidence gap, not silently generalized.
- **No stopping-criterion calibration** — "stop" is just another menu item the
  model can pick at any turn; there is no confidence threshold, no calibrated rule,
  nothing resembling Explore-until-Confident's conformal approach. Not usable for
  that RQ beyond "an explicit stop action in the menu is a workable pattern."
- **The "precise coordinates confuse GPT" finding is domain-specific** (GPS-scale
  coordinates, large building, matched only against a topological map, not against
  named/indexed geometric labels at small scale). Citing it as a general "numbers
  vs. names" verdict for small-scale local pose description (our face/orbit
  geometry) over-reaches what was actually measured — flag this precisely for the
  RQ4 contradiction-resolution question (MapGPT's claim and Thinking-with-Novel-
  Views' claim are not necessarily in the same measurement regime).
- **Failure-mode reporting is thin and non-spatial** — do not cite this paper's
  Appendix B as a source of named spatial-reasoning failure modes for Branch 2;
  it only reports instruction-following/grounding failures.
- **Zero-shot ceiling context is VLN-specific** (photorealistic building
  navigation, GPT-4/GPT-4V generation, 2024) — useful only as one directional data
  point for the "capability floor" framing, not a substitute for MMSI-Bench/
  ViewSpatial/SpinBench numbers, which test camera-camera/object-relative spatial
  reasoning directly rather than building-scale path navigation.

## Evidence-table entry (for the cross-paper table)

| Paper | Encoding compared against | Delta | Variance reported? |
|---|---|---|---|
| MapGPT (2401.07314) | Topological text map vs. coordinate (GPS) text map, GPT-4V, no adaptive planning | +3.7 SR, +3.7 SPL, +1.4 OSR | No |
| MapGPT (2401.07314) | Topological text map vs. no map, GPT-4, no adaptive planning | +4.6 SR, +5.4 SPL, +10.2 OSR | No |
| MapGPT (2401.07314) | Adaptive multi-step plan vs. single-step action planning, GPT-4V, topological map | +2.3 SR, +2.5 SPL | No |
| MapGPT (2401.07314) | Seeing (GPT-4V, images) vs. caption hand-off (GPT-4, BLIP-2 + Faster R-CNN captions), identical map/prompt scaffold, REVERIE | +3.2 SR, +5.8 SPL, **-5.8 OSR** (seeing worse on oracle-reachability, better on decisiveness) | No |

## Relevance note

Closest published analogue to a node-ID-augmented, connectivity-as-text action
menu with an explicit no-coordinates stance — useful template for our menu syntax
and for the egocentric-vs-allocentric and caption-vs-seeing RQs — but its central
assumption (graph given by the simulator, options large-scale and mutually
distinct) does not hold for our small set of near-identical orthogonal-view
candidates on one fused, defective point cloud, so it cannot answer the
option-count/distinctiveness or geometric-construction questions at all.
