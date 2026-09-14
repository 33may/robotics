# ViewSpatial-Bench: Evaluating Multi-perspective Spatial Localization in Vision-Language Models

- arXiv: 2505.21500 (v1: 2025-05-27, v2: 2025-09-30, HTML), preprint — no venue/conference acceptance noted
  in the arXiv metadata as of this read
- Authors: Dingming Li, Hongxing Li, Zixuan Wang, Yuchen Yan, Hang Zhang, Siqi Chen, Guiyang Hou, Shengpei
  Jiang, Wenqi Zhang, Yongliang Shen, Weiming Lu, Yueting Zhuang (ZJU-REAL lab) — project page:
  https://zju-real.github.io/ViewSpatial-Page/
- Read via: arXiv HTML (experimental, v2) full text, multiple targeted fetches (abstract, full Table 2,
  task definitions, annotation pipeline, MVSM training details, input-format confirmation, limitations)

## Core method / idea

ViewSpatial-Bench is a benchmark, not a method paper about view planning — it measures whether VLMs can
answer spatial-localization questions from **two different reference frames on the same photo**: the
camera's own frame ("egocentric"/Camera-Perspective) and a depicted person's or object's frame
("allocentric"/Person-Perspective). Five task types, split 48.4%/51.6% camera/person:

- **Cam-Rel.Dir.** (egocentric, relative direction): "Can you describe the position of {object1} relative
  to {object2}?" — answered directly from the camera's own viewpoint.
- **Cam-Obj.Ori.** (egocentric, orientation): "With the camera's viewpoint as the front, which direction is
  {object} facing?" — asks for a depicted object/person's own facing direction, judged against the camera
  frame.
- **Per-Obj.Ori.** (allocentric, orientation): "Suppose you are in {object}'s position, what direction are
  you facing?" — the model must adopt the depicted object's body frame.
- **Per-Rel.Dir.** (allocentric, relative direction): "From person A's perspective, where is person B
  located?" — relative-direction judgement re-expressed in a third party's frame.
- **Per-Sce.Sim.** (allocentric, scene/viewpoint simulation): "Imagine standing at {object1} looking
  towards {object2}, where is {object3}?" — the hardest task: mentally relocate to a novel viewpoint inside
  the scene and re-derive a third object's position from it.

**Ground truth is computed automatically from 3D metadata, not asked of humans.** ScanNet (validation set,
1,000+ indoor scenes) supplies 3D coordinates and camera parameters in the camera coordinate system;
Maximum Coverage Sampling picks a minimal set of frames per scene. MS-COCO (validation set) supplies
person images; heads/bodies are segmented and Orient-Anything-Large estimates rotation angles, with gaze
direction derived from the angular offset between head and body orientation. Raw angles are discretized
into **18 directional categories via 22.5° bins** (e.g. 22.5°–67.5° = "front-right", 67.5°–112.5° = "right",
337.5°–22.5° = "front" — full 18-way enumeration not given verbatim in the fetched text, only these
examples; no vertical/up-down categories are mentioned for any task). Multiple-choice distractors are
generated rule-based: for a single-direction ground truth, compound-direction distractors containing that
attribute are excluded; for a compound ground truth, its constituent single directions are excluded as
distractors. Exact option count (4-way vs other) is not stated in the fetched text.

**Critical methodological fact for our purposes: every question uses a single photograph.** Even
Per-Sce.Sim, which sounds like a "compare two camera views" task, is computed from **one image containing
all three objects** — the model is asked to imagine relocating within that single frame, not shown a
second photo taken from the imagined location. Quote: "we... selected object triads o1,o2,o3... and used
metadata to locate frames containing all three objects. By simulating the position and orientation at o1,
we calculated the relative position of o3 from this simulated viewpoint." This is a **mental
viewpoint-transform-within-one-image** task, not a cross-view/paired-image comparison task.

**Fine-tuned baseline (MVSM — Multi-View Spatial Model):** plain SFT of Qwen2.5-VL-3B on ~43K
(image, question, standardized-direction-answer) triples spanning ScanNet, MS-COCO, and Spatial-MM
training splits. "We freeze the vision encoder and multi-modal projector while keeping the language model
trainable" — 3 epochs, effective batch 16 (grad-accum 4), 4×A100-40GB, ~8.5 GPU-hours. No architecture
change, no multi-image input, no chain-of-thought, no auxiliary loss — the capability gain is attributed
entirely to task-matched supervised data.

## Concrete results and numbers (Table 2, all rows fetched from the HTML)

| Model | Cam-Rel.Dir | Cam-Obj.Ori | **Cam Avg** | Per-Obj.Ori | Per-Rel.Dir | Per-Sce.Sim | **Per Avg** | **Overall** |
|---|---|---|---|---|---|---|---|---|
| InternVL2.5 (2B) | 38.52 | 22.59 | 32.79 | 47.09 | 40.02 | 25.70 | 37.04 | 34.98 |
| Qwen2.5-VL (7B) | 46.64 | 29.72 | 40.56 | 37.05 | 35.04 | 28.78 | 33.37 | 36.85 |
| LLaVA-NeXT-Video (7B) | 26.34 | 19.28 | 23.80 | 44.68 | 38.60 | 29.05 | 37.07 | 30.64 |
| LLaVA-OneVision (7B) | 29.84 | 26.10 | 28.49 | 22.39 | 31.00 | 26.88 | 26.54 | 27.49 |
| InternVL2.5 (8B) | 49.41 | 41.27 | 46.48 | 46.79 | 42.04 | 32.85 | 40.20 | 43.24 |
| Llama-3.2-Vision (11B) | 25.27 | 20.98 | 23.73 | 51.20 | 32.19 | 18.82 | 33.61 | 28.82 |
| InternVL3 (14B) | 54.65 | 33.63 | 47.09 | 33.43 | 37.05 | 31.86 | 33.88 | 40.28 |
| Kimi-VL-Instruct (16B) | 26.85 | 22.09 | 25.14 | 63.05 | 43.94 | 20.27 | 41.52 | 33.58 |
| GPT-4o | 41.46 | 19.58 | 33.57 | 42.97 | 40.86 | 26.79 | 36.29 | 34.98 |
| Gemini 2.0 Flash | 45.29 | 12.95 | 33.66 | 41.16 | 32.78 | 21.90 | 31.53 | 32.56 |
| Qwen2.5-VL (3B) [MVSM backbone, pre-FT] | 43.43 | 33.33 | 39.80 | 39.16 | 28.62 | 28.51 | 32.14 | 35.85 |
| **MVSM (fine-tuned)** | **83.59** | **87.65** | **85.05** | **90.16** | **71.14** | **75.75** | **79.31** | **82.09** |
| Improvement (FT − backbone) | +40.16 | +54.32 | +45.25 | +51.00 | +42.52 | +47.24 | +47.17 | +46.24 |
| **Random baseline** | 25.16 | 26.10 | 25.50 | 24.60 | 31.12 | 26.33 | 27.12 | 26.33 |

Dataset: 5,712 total test samples across 1,338 unique scenes (48.4% camera-perspective / 51.6%
person-perspective); training set 42,875 samples. Quality control: manual filtering of ambiguous/occluded/
blurry instances, manual verification of target-object visibility.

## Load-bearing correction to the branch's stated selection rationale

The branch selected this paper on the premise "allocentric reasoning sits barely above chance while
egocentric is fine." **The paper's own headline finding is the opposite of a clean ego-good/allo-bad
split**, and the per-task breakdown shows the truth is more specific and more useful than either framing:

- Paper's verbatim headline: "Most VLMs exhibit an intriguing pattern wherein their spatial localization
  accuracy from camera perspectives (averaging 33.2%) falls below their performance from human viewpoints
  (averaging 35.7%)" — i.e. **allocentric (Per Avg) is on average slightly *better* than egocentric
  (Cam Avg)** across the reported baseline models, not worse. (My own recomputation over the 6 models in
  the abstract-page table gives Cam Avg mean ≈39.0 / Per Avg mean ≈35.4 — the direction flips depending on
  which model subset is averaged; the paper's 33.2/35.7 figures use the full 10-model set including the
  weak LLaVA/Llama video-tuned models, which are disproportionately bad at Cam tasks. Either way, the two
  averages are close and the "allocentric barely above chance, egocentric fine" framing is not what this
  table shows at the task-family level.)
- The real, load-bearing pattern is **per-task, not per-frame**:
  - **Cam-Obj.Ori (egocentric "which way is the object facing") is the single worst task overall**, and is
    at or *below* random chance (26.10) for the majority of models: InternVL2.5-2B 22.59, LLaVA-NeXT-Video
    19.28, Llama-3.2-Vision 20.98, Kimi-VL 22.09, **GPT-4o 19.58**, **Gemini-2.0-Flash 12.95** (roughly half
    of chance). Only InternVL2.5-8B (41.27), Qwen2.5-VL-7B (29.72) and InternVL3-14B (33.63) clear chance
    by a meaningful margin.
  - **Per-Obj.Ori (allocentric "imagine you are the object, which way do you face") is one of the
    *strongest* tasks**, well above chance (24.60) for nearly every model: 47.09, 37.05, 44.68, 22.39 (one
    exception), 46.79, 51.20, 33.43, 63.05, 42.97, 41.16.
  - **Per-Sce.Sim (allocentric viewpoint-transport — the task closest in spirit to "relate one viewpoint to
    another") is weak, at-or-below chance (26.33) for most models**: 25.70, 28.78, 29.05, 26.88, 18.82,
    20.27, 21.90 are all ≤ chance; only InternVL2.5-8B (32.85) and InternVL3-14B (31.86) clear it clearly.
    This is the single most relevant number in the paper for RQ1's "capability floor" question.
  - **Per-Rel.Dir (allocentric relative direction) sits close to chance (31.12) for several mid-tier
    models**: LLaVA-OneVision 31.00, Llama-3.2-Vision 32.19, Qwen2.5-VL-7B 35.04 — barely above baseline.
  - **Cam-Rel.Dir (egocentric relative direction) is the strongest camera task and generally clears
    chance comfortably** (except the two video/multimodal-chat models LLaVA-NeXT-Video 26.34 and
    Llama-3.2-Vision 25.27, which sit at chance).

**Implication for the branch's framing**: the ego/allo split is not the right lens for this paper's data.
The right lens is *what kind of spatial operation is being asked*: "where is X relative to Y" (Rel.Dir.) is
moderately tractable in **both** frames; "which way does X face" (Obj.Ori.) flips from **near/below-chance
when asked in the camera's own frame** to **one of the best-performing tasks when asked in an adopted
third-party frame**; and "transport yourself to a new vantage point and re-derive a relation" (Sce.Sim.) is
weak **regardless of frame**, because it is the only task requiring an actual novel-viewpoint simulation
rather than a frame-relabeling of the same photo. That asymmetry (Cam-Obj.Ori bad, Per-Obj.Ori good) is
explicitly named as a paper finding: "VLMs perform significantly worse on Object View Orientation tasks
from camera perspectives compared to Relative Direction tasks, yet show the opposite pattern for human
perspective tasks (42.6% for Object View Orientation vs. 36.9% for Relative Direction)" — the paper reads
this as evidence models "fail to construct a coherent 3D representation that can be flexibly navigated from
different viewpoints," i.e. performance depends on *which specific frame-pairing is asked*, not on a
monotone egocentric-easy/allocentric-hard axis.

## What this paper does and does NOT measure, precisely (important for RQ1/RQ2)

- **It does not test camera-to-camera or camera-to-object viewpoint relation from two separate renders.**
  Every question is single-image. There is no "given render A, is render B to your left/right/behind" task
  here — that is MMSI-Bench's design, not this paper's. ViewSpatial tests **frame-of-reference
  relabeling and single-image mental transform**, which is a related but distinct failure surface from
  "given two discrete candidate views, choose the correct spatial relation between them." Do not cite
  ViewSpatial's numbers as direct evidence for a "pick view B relative to view A" capability — the nearest
  proxy is Per-Sce.Sim, and even that is answered from one photo, not two.
- **"Which way is object facing" is graded against camera frame or the object's own inferred body frame
  (from Orient-Anything head/body angle offset), never against a named/labeled discrete face-ID scheme.**
  Nothing in this benchmark resembles "face 1 / face 2 / face 3" labels on a primitive shape (box, cylinder)
  — the closest analogue is Cam-Obj.Ori/Per-Obj.Ori, both of which reason about a *person's or object's
  continuous facing direction*, not a fixed discrete ID painted onto object geometry. This limits how
  directly ViewSpatial's ego/allo numbers can be read onto RQ2's "orthogonal to face 3" query. The
  candidate mapping is: "orthogonal to face 3" is structurally closer to **Per-Obj.Ori** (put yourself in
  the object's frame, state a facing direction) than to Cam-Rel.Dir or Per-Rel.Dir — and Per-Obj.Ori is
  the one task in this benchmark where allocentric framing performs *well* (24.6→~30–63% across models,
  often 40%+, well above chance), which is a mild positive signal, not a threat, for an object-face-relative
  addressing scheme — but this is an analogy, not a direct measurement, and should be flagged as such, not
  overclaimed.
- **No mark-based/visual-prompting (set-of-marks) condition is tested.** All prompting is plain
  image+question; the benchmark is silent on whether overlaid markers, numbered labels, or annotated menus
  change egocentric/allocentric accuracy. Irrelevant to the visual-prompting sub-branch questions except as
  a gap.
- **No robotics/manipulation framing, no discrete action-menu framing, no reachability/IK.** This is a
  pure perception/language spatial-QA benchmark; there is no "select next view from a candidate set"
  structure to compare against our menu design directly.

## Failure modes (for Branch 2 handoff, named and numbered)

1. **Below-chance egocentric object-facing judgement** — GPT-4o 19.58% and Gemini-2.0-Flash 12.95% on
   Cam-Obj.Ori, both under the 26.10% random baseline. A frontier closed model can be *worse than guessing*
   at "which way is this object facing, camera as front."
2. **Perspective confusion / frame bleed** — paper's qualitative claim (Fig. 4, GPT-4o vs MVSM on VSI-App
   examples): "models without perspective-aware training demonstrate inconsistent spatial judgments within
   single responses, alternating between human and camera perspectives," and specifically "GPT-4o
   frequently defaults to camera-centric judgments for orientation determinations" even when explicitly
   asked to adopt another frame. No specific QA pair with wrong-vs-right answer text was recoverable from
   the fetched HTML (only the comparative-pattern description); flag as **qualitative, not
   quantitatively itemized** beyond the Table 2 numbers.
3. **Task-type asymmetry that inverts with frame** — Obj.Ori. is the weak task in camera frame but the
   strong task in person frame (see above); Rel.Dir. is comparatively frame-stable. A single "egocentric
   good / allocentric bad" rule would mispredict this.
4. **Weakest single task industry-wide is viewpoint-transport (Per-Sce.Sim), not object-facing** — most
   models sit at or below chance (18.82–29.05%) here. This is the nearest proxy to "imagine the geometry
   from a different vantage point," and it is the floor of the whole benchmark for most non-finetuned
   models.

## Load-bearing assumptions

1. **Ground truth comes from privileged 3D metadata (ScanNet camera parameters, COCO+Orient-Anything
   pose), not from any geometry the model itself perceives or reconstructs.** The benchmark measures
   language/vision spatial reasoning over an already-known 3D structure, projected into a photo — it says
   nothing about whether a model can *build* that structure from a sparse/defective point cloud (out of
   this branch's scope; explicitly Branch 5 territory).
2. **All scenes are real, dense, well-lit, photorealistic photographs** (ScanNet frames, COCO photos) — no
   point clouds, no colourless/sparse geometry, no rendered holes or dropout. Zero evidence transfers to
   the "render of a defective fused point cloud" sub-question (GraspView/SPAZER/VLM3D territory).
   Everything measured here is on RGB photographs of real, complete scenes.
   Extrapolating these accuracy numbers to a marked render of a partial, colourless survey cloud is not
   licensed by this paper — the domain gap (missing texture, holes, single-material geometry) could make
   performance worse still, and this paper has no data on that.
3. **Direction vocabulary is a fixed, continuous-angle-derived 18-way compass label (front/back/left/right
   and diagonals), not an object-specific discrete ID scheme.** Any attempt to map these numbers onto a
   "face 1..N" or "orbit position k of M" labeling is an analogy, not a tested equivalence.
4. **Single-image-only design.** No task in this benchmark shows the model two photographs and asks it to
   relate them. This is the most consequential scope limit for our branch, since our actual interface
   (choosing among discrete candidate views / relating candidate view B to current view A) is closer to a
   cross-image comparison task than to anything measured here.
5. **MVSM's 46-point gain is from supervised fine-tuning on task-matched data, with the vision encoder
   frozen** — i.e., the fix is "teach the LM to read out spatial frames it apparently already encodes
   ambiguously," not a new visual grounding mechanism. This is evidence that the failure is partly a
   language/output-mapping problem rather than a pure perception problem, but the paper does not run the
   ablation needed to separate "encoder can't represent it" from "LM head can't read it out" (frozen
   encoder + trainable LM improving a lot is *consistent with* an LM-side readout bottleneck, but is not
   proof, since the LM could also be learning better prompt-conditioned attention over encoder features it
   already had access to).

## What transfers to our setting

- **A concrete, falsifiable warning against relying on any single VLM (including GPT-4o, Gemini) for
  "which way is this facing relative to the camera" judgements without task-specific tuning** — the
  below-chance Cam-Obj.Ori numbers for exactly the closed frontier models we would consider deploying are
  the most directly load-bearing data point for RQ1's "capability floor" question, even though the task
  format differs from ours (single-image vs. discrete-candidate-view).
- **Task-type, not frame-type, is the right unit of analysis for predicting VLM spatial failure.** When
  designing benchmark test cases (Branch 2 handoff), decompose by *operation type* ("locate relative to
  a stated frame" vs "state a facing/orientation" vs "transport to a hypothetical vantage point and
  re-derive a relation") rather than by an ego/allo binary — this paper's own data contradicts a clean
  binary.
- **Per-Sce.Sim's poor, near-chance performance is the closest available proxy (from this paper) for "can
  a model reason about a hypothetical alternate viewpoint from a single fixed observation,"** which is
  structurally close to "given the current camera view (or the fused point cloud), can the model reason
  about what a candidate discrete view would show without actually being shown it." Weak evidence, single
  proxy, single-image caveat applies — cite as a directional signal, not a load-bearing number.
- **MVSM's result is a strong existence proof that task-matched SFT closes most of this gap (26→82%
  overall)** — relevant if the branch ever considers fine-tuning a smaller VLM specifically for the
  discrete-view-menu task rather than relying on zero-shot frontier-model prompting. Not itself evidence
  that prompting/interface design alone (this branch's actual scope) can close the gap — SFT and prompting
  are different levers, and this paper only demonstrates the SFT lever.
- **The finding that object-orientation-in-adopted-frame (Per-Obj.Ori) is comparatively strong** is a mild
  positive signal (not proof) that asking a model to reason "as if standing at the object, facing outward"
  may be a more tractable phrasing than asking it to judge the object's facing purely from the camera's own
  frame — worth testing directly for our "orthogonal to face N" phrasing rather than assuming it inherits
  ViewSpatial's allocentric-bad framing.

## What does NOT transfer, and why

- **No cross-view/paired-image comparison data at all.** Cannot be used as direct evidence for any
  RQ that requires relating two actual candidate views/renders to each other (RQ1's core ask, and the
  "render B relative to render A" flavor of RQ2). The single closest proxy (Per-Sce.Sim) is explicitly a
  single-image mental-transform task, not a two-image comparison, and even that proxy performs badly —
  which if anything is a *negative* signal for two-image comparison tasks being any easier, but it is not
  a direct measurement of them.
- **No point-cloud, render, mesh, or non-photographic imagery of any kind** — nothing here transfers to
  the "render of a partial, colourless, glossy-surface-dropout fused cloud" sub-question; the benchmark's
  entire evidentiary base is real, complete, well-lit RGB photographs. Zero evidence either way about
  survey-cloud renders specifically (flagged for the SPAZER/GraspView/VLM3D/SeqVLM comparison — do not cite
  ViewSpatial numbers there).
- **No discrete action-menu / multiple-candidate-view selection structure.** This is a pure QA benchmark
  (one question, one multiple-choice answer among direction labels); it has no notion of "choose from
  N candidate views," so it offers zero evidence for the option-count/ordering/distinctiveness RQs, the
  infeasible-view-presentation RQ, or the hierarchical-disclosure RQ. Do not cite it for those.
- **No set-of-marks / visual-prompting condition.** Marks, annotated menus, and visual-prompting encodings
  are entirely untested here — irrelevant to the visual-prompting sub-branch beyond noting the gap.
- **No stopping-criterion, confidence-calibration, or "enough views" concept** — single independent
  questions, no sequential decision-making, no coverage notion. Not usable for the Explore-until-Confident
  handoff RQ.
- **No ablation of encoding choice (numbers vs. names vs. indices, text vs image) for the spatial label
  itself** — the direction vocabulary (front/back/left/right/diagonals as words) is fixed across the whole
  benchmark; there is no comparison arm testing e.g. numeric bearing angles vs. named directions vs.
  clock-position labels. This paper cannot adjudicate the RQ4 "numbers vs. names" contradiction
  (Thinking-with-Novel-Views vs. MapGPT) — it only ever uses named 18-way compass directions, with no
  alternative encoding tested against them, so it is an unablated single-encoding demonstration on that
  axis, not comparative evidence.

## Evidence-table entry (for the cross-paper table)

| Paper | Encoding compared against | Delta | Variance reported? |
|---|---|---|---|
| ViewSpatial-Bench (2505.21500) | Camera-frame vs. person/object-frame spatial query, same photo, 10 zero-shot VLMs vs. random baseline (per-task) | Cam-Obj.Ori: at/below chance for 6/10 models (e.g. GPT-4o 19.58 vs 26.10 chance, Gemini 12.95 vs 26.10 chance); Per-Obj.Ori: above chance for 9/10 models (e.g. Kimi-VL 63.05 vs 24.60 chance); Per-Sce.Sim: at/below chance for 7/10 models | No (single-run point estimates only; no seeds/CIs reported anywhere in the fetched tables) |
| ViewSpatial-Bench (2505.21500) | Zero-shot Qwen2.5-VL-3B vs. same backbone SFT'd (MVSM) on 43K task-matched samples, frozen vision encoder | +46.24 overall (35.85→82.09), task range +40.16 to +54.32 | No |

This is a **task-matched SFT ablation** (clean before/after on identical backbone/eval), not an
encoding-choice ablation — flag accordingly when the cross-paper table is assembled: it answers "does
task-specific training help" (yes, a lot), not "which prompt encoding of spatial info is best" (untested
here).

## Handoffs

- **Branch 2 (benchmark test cases)**: use failure modes 1–4 above, tagged BENCH. Suggested named test
  cases: (a) "object-facing-from-camera-frame" query on a closed frontier model, expect near-or-below-chance
  without task tuning; (b) "viewpoint-transport" query (imagine standing elsewhere, judge a third relation)
  as the hardest single-image spatial-reasoning primitive across the industry; (c) explicitly test whether
  phrasing an object-face query in an adopted-object-frame ("as if you were the face, which way do you
  point") outperforms a pure camera-frame phrasing, since this paper's Per-Obj.Ori vs Cam-Obj.Ori gap is a
  candidate signal but was never tested on discrete-face/orbit-ID geometry.
- **Branch 5 (geometry construction)**: no findings — this paper assumes 3D structure is already known via
  ScanNet/COCO metadata; it has nothing to say about constructing or scoring geometry from a survey cloud.
- **RQ1 capability-floor table**: add ViewSpatial's Per-Sce.Sim near-chance numbers as one data point
  alongside MMSI-Bench camera-camera/camera-object and SpinBench rotation numbers — but tag it explicitly
  as "single-image mental transform, not paired-view comparison" so it is not conflated with the
  camera-camera task family when the floor table is assembled.

## Relevance note

Read in full; the paper is a clean, well-instrumented capability-floor data point but does **not** support
the specific claim the branch selected it for. It does not show a uniform "allocentric near chance,
egocentric fine" gap — it shows a **task-operation-dependent** gap (object-facing-in-camera-frame is the
weak spot, not allocentric reasoning per se; person/object-frame facing is comparatively strong) and its
entire evidentiary base is single-photo, privileged-3D-metadata QA with no cross-view comparison, no
render/point-cloud imagery, and no discrete action-menu structure. High value for RQ1 (capability-floor
data, including a below-chance result for GPT-4o and Gemini on egocentric object-facing) and moderate,
analogy-only value for RQ2 (mapping "orthogonal to face N" onto its task taxonomy). Near-zero direct
evidence for RQ3 (topology/menu structure), RQ4-encoding (no alternative-encoding ablation), the
seeing-vs-caption RQ, the sparsity/render RQ, or the stopping-criterion RQ — these should be sourced
elsewhere and this paper should not be over-cited to cover those gaps.
