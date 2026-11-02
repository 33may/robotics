# Branch 6 — `vlm-spatial-interfaces` — round-4 accumulation

## Presenting spatial structure and view actions to VLM agents

**Date:** 2026-08-27 · **Pass 1, round 4 (branch synthesis)**
**Feeds:** RQ4 (primary), RQ3 (tractability pressure back onto the representation),
RQ1/RQ2 (via the judge/selector architecture), Branch 2 (benchmark methodology),
the blind-vs-seeing architecture decision.

**Inputs:** [`../round1_plan.md#branch-6`](../round1_plan.md) ·
[`../round2_map/vlm-spatial-interfaces.md`](../round2_map/vlm-spatial-interfaces.md) ·
all 20 files in [`../round3_reading/vlm-spatial-interfaces/`](../round3_reading/vlm-spatial-interfaces/)
**Contract:** [`../../../geometry-aware-view-planning.md`](../../../geometry-aware-view-planning.md)

---

## 0. What was actually read, and what was not

**Read in full (20 papers/sources):** MMSI-Bench, ViewSpatial-Bench, SiT-Bench,
RoboSpatial, Thinking-with-Novel-Views, MapGPT, SayPlan, Structured-Interfaces-3DSG
(Cypher), Set-of-Mark, MOKA, PIVOT, SCAFFOLD, AP-VLM, GraspView, SeeGround, Think3D,
MV-GEL, Explore-until-Confident, LLM-option-order-sensitivity, How-Many-Tools.

**Not read this round — several load-bearing round-2 claims therefore remain
UNVERIFIED and must not be cited as evidence:**

| Round-2 claim | Source | Status |
|---|---|---|
| SPAZER: agent-selected views beat random by only **+3.6%** | `spazer-progressive-3d-grounding` | **unverified** — this was the branch's headline caution; nothing read this round substitutes for it |
| TagaVLM: MapGPT-style text topology adds only **+0.94%** | `tagavlm-topology-critique` | **unverified** — the single most direct published threat to the blind orchestrator |
| SoM marks cost **−16.2%** on relational questions | `som-limitations-contrastive-region-guidance` | **unverified** — the highest-value Family-B critique |
| FlipSet compositional deficit (ToM + rotation collapse when composed) | `egocentric-bias-flipset` | **unverified** — but AP-VLM supplies an independent measured substitute (§3, Q3) |
| 3D-Mem frontier snapshots; Manipulate-Anything contact sheet; RAVEN caption bottleneck; SayCan affordance factor; Zheng et al. token bias | map-only | **unverified** |

Round-3 readers repeatedly caught summarizer-model fabrication (Thinking-with-Novel-Views:
an entire invented per-backbone table; MV-GEL: a false "keyword absent" result). Every
number below is one a reader verified against primary text. Where a reader flagged a
number as figure-only or inconsistent, that flag is carried forward.

**Branch-wide evidence quality.** Of 20 sources, **four** report any variance:
How-Many-Tools (±std over 3 seeds), SCAFFOLD (ablation subset averaged over 2 runs),
PIVOT (Table 4 ± values), Think3D (claims 3-run averaging, shows no std). Everything
else in this branch — including every capability-floor benchmark, every view-selection
system, and every encoding ablation — is single-run point estimates. Treat every delta
under ~5 points anywhere in this document as noise-level until replicated.

---

## 1. What is known — the eight findings that survived reading

These are ordered by how much weight they can carry. Confidence tags: **[strong]** =
ablated in ≥2 independent papers with consistent direction; **[moderate]** = ablated
once, or demonstrated repeatedly; **[weak]** = single unablated observation.

### 1.1 Interface richness is capability-gated, and below the gate it is net-negative **[strong]**

This is the branch's most robust cross-paper convergence, and it was not anticipated in
the round-2 map. Six independent papers report that adding spatial machinery to a
below-threshold model *hurts*:

| Source | Intervention | Effect on the weak model |
|---|---|---|
| Think3D §4.2 | Give Qwen3-VL-4B the same camera-control toolkit as GPT-4.1 | avg **+0.61**; MindCube-Around **−8.34**, VSI Route-Plan **−4.08** |
| Thinking-with-Novel-Views App. B | Generative view augmentation on Qwen3-VL-8B | **−4.3** overall, **−26.7** on Size |
| MMSI-Bench §4.3 | Zero-shot CoT | improves GPT-4o only; **degrades the rest** |
| MMSI-Bench §4.3 | Visual prompting (PATS correspondence lines) | slight gain 2 models, **degradation 2 models** |
| SiT-Bench §3.3 | "Spatial-specialist" fine-tunes (SpaceR, Robobrain2.0, Space-Qwen, Cosmos-Reason2) | all **underperform** general-purpose VLMs of equal or smaller scale |
| Set-of-Mark §4.4 | Alphanumeric marks on LLaVA-1.5 / MiniGPT-v2 | "can hardly interpret the marks" — categorical failure, not graded degradation |
| AP-VLM §V | Add a second rotation axis (`3Dx`→`3Dxy`, a strict superset) | SR **0.8→0.5** (S1), **0.5→0.1** (S2) |

Think3D's and TwNV's own diagnoses agree on the mechanism: below the threshold the
dominant failure is *instruction-following / task comprehension*, not missing evidence.
TwNV states it directly — "the understanding backbone should meet a minimum
instruction-following competence for the augmentation to take effect."

**Consequence for us:** the RQ4 question "which interface" is not answerable
independently of "which model." Any interface decision must be paired with a named
model tier, and a cheap instruction-following pre-check must gate the whole design.

### 1.2 Selecting a label beats emitting a value, by a very large margin **[strong]**

Set-of-Mark, Table 2, RefCOCOg REC: GPT-4V asked to emit a bounding box scores **25.7%**;
the same model asked to name an overlaid mark scores **86.4%** — **+60.7 points**, the
largest single delta in this branch, and it is a within-paper ablation (same model, same
benchmark, two prompting modes). MOKA and PIVOT independently built entire systems on
this premise, and MOKA additionally refuses to trust the VLM's chosen point directly
(snapping to the nearest of 30 classical antipodal grasp candidates). PIVOT's founding
sentence is the same claim: VLMs "struggle to produce precise spatial outputs directly,
[but] can readily select among a discrete set of coarse choices."

This validates the contract's given ("VLMs cannot emit metric poses; selection stays
discrete") with real numbers, and it validates it at the *output* layer specifically.
It says nothing about how the choices should be described.

### 1.3 Any explicit addressing scheme beats none; the choice among schemes matters much less **[moderate]**

SCAFFOLD's Table 3 is the cleanest encoding ablation in the branch (150-sample subset,
each condition run twice and averaged — one of the four variance-reporting results here):

| Coordinate format | Score |
|---|---|
| Cartesian 2D grid (theirs) | **79.3** |
| Alphabet | 77.0 |
| Absolute pixel | 76.7 |
| One-dimensional | 74.7 |
| **None (baseline)** | **69.0** |

Baseline→any-scheme gap: **5.7–10.3 points**. Best-scheme→worst-labelled-scheme gap:
**4.6 points**. The ordering (coarse discretized grid > raw continuous pixel values >
over-collapsed 1D index) says the axis that matters is *granularity relative to the
addressed space*, not numeric-vs-symbolic.

### 1.4 A new observation beats more deliberation over the same observation **[strong]**

Two independent matched-budget controls:

- **Thinking-with-Novel-Views, Table 2:** text-only self-reflection swept over R∈{1..10}
  **never beats its own R=0 baseline** (76.7%), fluctuating 72.7–75.7%. At matched
  2-call budget, looking at a new (even *generated*) view scores 78.7% vs 75.7% —
  **+3.0**. "The gap is not from extra compute but from reasoning over a new viewpoint
  instead of the same one."
- **Think3D §5.6:** a Self-Refine baseline matched on rounds gives "only marginal
  improvements" at comparable-or-higher token cost; the gain is attributed to explicit
  3D interaction.

This is the strongest ammunition against any design that hopes better prompting will
substitute for taking another view — and, symmetrically, against padding the loop with
extra reasoning turns.

### 1.5 …but a low-fidelity observation is worse than no extra observation at all **[moderate]**

TwNV Table 1, generator ablation with the reasoner held fixed: QwenImageEdit **−1.8pp
vs. not generating a view at all**; NanoBananaPro +0.5; GPTImage1 +1.3; CamCraft +2.5.
The paper's own conclusion: "a generator must reach a certain fidelity threshold before
its outputs help rather than hurt." PIVOT's Appendix G Table 7 gives the marks-side
analogue: arrow-direction classification on a blank background is 88–100% accurate, but
matching an arrow to the object it indicates collapses to **0–44% on "Very Hard"
(eccentric-geometry) objects**.

Together these two set a hard prior on the seeing-orchestrator option: a marked render
of a sparse, holed, colourless survey cloud is exactly the regime where the added visual
channel could be net-negative, and nobody has measured it (§5).

### 1.6 A blind, text-fed model is far from useless — and its specific loss is measurable **[moderate]**

MapGPT is the only clean head-to-head in this branch: identical map/prompt scaffold,
GPT-4 fed BLIP-2 + Faster R-CNN captions vs. GPT-4V fed the actual images.

| REVERIE | OSR | SR | SPL |
|---|---|---|---|
| MapGPT + GPT-4 (blind, captions) | **42.6** | 28.4 | 14.5 |
| MapGPT + GPT-4V (seeing) | 36.8 | **31.6** | **20.3** |

The blind arm **wins on oracle success** and loses on success and path efficiency. The
paper's own explanation: "due to information loss in the two-stage pipeline, MapGPT
(GPT-4 based) tends to continue exploring when encountering insufficient description."
The caption hand-off destroys **decisiveness and efficiency, not coverage** — a blind,
under-informed agent keeps looking instead of committing.

Corroborating: SayPlan reaches **86.6% executability on long-horizon plans with zero
visual input**, purely via a symbolic verify-then-correct loop (13.3%→86.6% vs. the
open-loop baseline). SiT-Bench shows VLM-pretrained backbones outperform their pure-text
siblings **on identical text-only items** (Qwen2.5-VL-72B 45.45 vs Qwen2.5-72B 42.57;
Qwen3-VL-8B 42.10 vs Qwen3-8B 37.91) — i.e. a blind orchestrator should still be a
vision-pretrained model.

### 1.7 Nobody asks a prompted VLM to construct the candidate view set — geometry does it **[strong]**

Across every system read, candidate generation is deterministic and the language model
never sees the geometry as text:

| System | Candidate generation | VLM's role |
|---|---|---|
| GraspView | conical spherical-cap sampler **egocentric to the current camera** (Eqs. 4–8); look-at + roll-stabilisation orientation (Eqs. 9–12); IK-infeasible silently dropped | scores one render at a time, 1–10; argmax **in code** |
| MV-GEL | uniform 60-view grid (elev ∈[−60°,60°], azim 30° steps) | none — a trained ranker picks; the VLM only consumes the winner |
| Think3D | integer (azimuth, elevation) delta **relative to a chosen anchor camera** | picks the delta from an 8-item menu (RL) or an open integer range (frontier models) |
| SeeGround | closed-form: look at a text-identified anchor, back off and up (Pytorch3D `look_at_view_transform`) | reads the one resulting render |
| AP-VLM | ArUco+FK-projected workspace grid on the **live** RGB frame — no render at all | picks a grid vertex (+ optional ±35° rotation) |
| MOKA | 5×5 chess-notation image grid + FPS boundary keypoints | picks a tile / keypoint label |

This is the branch's clearest structural answer to RQ2/RQ4 jointly: **geometry proposes,
the model disposes, and the disposal is always a label.** It matches the contract's
"primitive-level deterministic rules are acceptable" allowance exactly.

### 1.8 Two-stage decision framing is worth more than menu presentation **[moderate]**

MOKA's Table IX is the largest ablated effect in the visual-prompting literature we
read: removing the high-level decomposition step (annotate everything at once, ask one
VQA question) collapses reasoning success from **0.6–1.0 to 0.0–0.2**. The other two
ablations — removing keypoint/tile descriptions, removing CoT — cost only ~0.2. The
ranking is unambiguous: *hierarchy ≫ description ≈ CoT*.

Critically, MOKA's "hierarchy" is **not** collapsing the action space — the leaf menu
stays flat (~25–40 simultaneous options). It is decomposing *what is being decided*
before showing the menu. That maps onto our problem as: decide *which surface/primitive*
first, then *which view of it* — never both in one turn.

---

## 2. Consolidated evidence-discipline table (branch question 14)

Every encoding/interface claim from the 20 papers, with whether it was ablated and
whether variance was reported.

| Paper | Comparison | Delta | Ablated? | Variance? |
|---|---|---|---|---|
| Set-of-Mark | coordinate emission vs mark-ID selection (RefCOCOg REC) | **+60.7** (25.7→86.4) | Yes | No (N=100 img) |
| Set-of-Mark | Number+Mask vs Number+Mask+Box (Flickr30K) | +4.8 R@1 | Yes | No |
| Set-of-Mark | predicted vs GT masks (RefCOCOg mIoU) | +14.5 | Yes | No |
| Set-of-Mark | GPT-4V vs open LMMs reading marks | not quantified | No (qualitative) | No |
| SCAFFOLD | dot-matrix + Cartesian vs no scaffold | +10.3 (69.0→79.3) | Yes | **Yes** (2 runs) |
| SCAFFOLD | Cartesian / Alphabet / Pixel / 1D | 79.3 / 77.0 / 76.7 / 74.7 | Yes | **Yes** |
| SCAFFOLD | Binary vs Black/White/Complementary colouring | 79.3 / 78.0 / 76.0 / 76.3 | Yes | **Yes** |
| SCAFFOLD | dot perturbation σ=¼ spacing | retains gain; VSR *improves* | Yes | No |
| SCAFFOLD | 3×3…7×7 matrix size | 6×6 best on aggregate, not per-subset | Yes | No |
| PIVOT | image annotation vs 3×3-grid text, same choice (nav, L2↓) | 0.21±0.002 vs 0.26±0.15; 0.23 vs 0.35; 0.44 vs 0.46 | Yes | **Yes** (±) |
| PIVOT | PIVOT arrows vs text-only direction words (manip) | "large margin" (figure only) | Yes | No |
| PIVOT | 5 / 10 / 20 samples per iteration | more → better iter-1, worse refinement (marker occlusion) | Yes | No |
| PIVOT | prompt-element order (preamble/image/task) | 0.43 vs 0.41 — "small margin" | Yes | No |
| PIVOT | arrow direction on blank bg vs object-referential match | 88–100% vs **0–44%** (Very Hard) | Yes | No |
| PIVOT | depth via colour+size | **"none of the VLMs we tested are capable"** (§4.6) | negative result | No |
| MOKA | with vs without hierarchical decomposition | **0.6–1.0 → 0.0–0.2** | Yes | No (10 trials) |
| MOKA | with vs without keypoint/tile description | −0.2 to −0.4 | Yes | No |
| MOKA | index/tile encoding vs raw coordinates | **asserted, never ablated** | No | — |
| MapGPT | topological text map vs coordinate (GPS) text map, GPT-4V | +3.7 SR / +3.7 SPL / +1.4 OSR | Yes | No |
| MapGPT | topological map vs no map, GPT-4 | +4.6 SR / +5.4 SPL / +10.2 OSR | Yes | No |
| MapGPT | adaptive multi-step plan vs single-step | +2.3 SR / +2.5 SPL | Yes | No |
| MapGPT | seeing (GPT-4V) vs caption hand-off (GPT-4), same scaffold | +3.2 SR, +5.8 SPL, **−5.8 OSR** | Yes | No |
| SayPlan | iterative replan + sim feedback vs open-loop | exec 13.3→86.6 (long-horizon) | Yes | No |
| SayPlan | GPT-3.5 vs GPT-4 semantic search | 6.6/0.0 vs 86.7/73.3 | Yes (capability, not encoding) | No |
| SayPlan | collapse/expand token footprint | 6731→878 (Office), 6598→1817 (Home) | Yes | No |
| SayPlan | JSON graph encoding vs any alternative | **never compared** | No | — |
| SayPlan | 30→300 room-node scale sweep | failures were **token-limit only**, not reasoning degradation | Yes | No |
| Structured-Interfaces | Cypher(A) vs context-dump, **small** graph (65 obj) | Q&A **+10** (0.85 v 0.75); PDDL **tied** (0.65 v 0.65) | Yes | No |
| Structured-Interfaces | same, **large** graph (15,944 places) | Q&A +44 (0.77 v 0.33); tokens 2,395 v **582,202** | Yes | No |
| Structured-Interfaces | agentic (multi-call) vs single-call | +3 to +13, sometimes **−4** (Python/PDDL-large) | Yes | No |
| SeeGround | query-aligned render vs BEV / 3 point-to-point heuristics | +2.8 overall; **+4.4 Hard, +5.4 View-Dependent** | Yes | No |
| SeeGround | FAM cross-modal binding on/off | +3.6–3.8 | Yes | No |
| SeeGround | drop 3D-position text, add PAM (row e vs d) | **+1.7** — render substitutes for coordinate text | Yes | No |
| SeeGround | marker vs contour vs bbox vs mask overlay | 47.7 / 47.5 / 45.1 / 45.0 | Yes | No (40/130 rooms) |
| SeeGround | 7B/8B vs 72B VLM, identical encoding | **−12.8** | Yes | No |
| SeeGround | Mask3D vs GT 3D proposals | 44.1 vs 59.5 — **40.5% still fail with a perfect detector** | Yes | No |
| AP-VLM | 8-way action-space comparison incl. `3Dx`→`3Dxy` | SR 0.8→0.5, 0.5→0.1 | Yes | No (N=10) |
| AP-VLM | annotated (`2DA`) vs unannotated (`2DNA`) grid | SR tied; OSR 9 v 0, PE 0.30 v 0.44 | Yes | No |
| AP-VLM | discrete (`3DD`) vs continuous (`3DC`) vertex output | 0.8/0.3 vs 0.9/0.2 — **inconclusive** | Yes | No |
| AP-VLM | vs random/uniform view baseline | **no such baseline exists** | — | — |
| GraspView | active perception vs none, under occlusion | +41.9 pts SR (33.1→75.0) | Yes (system-level) | No |
| GraspView | 1 vs 0 / 2 vs 1 / 3 vs 2 NBVs (hard occlusion) | +35 / +15 / **+0** | Yes | No |
| GraspView | **VLM scoring vs random or classical utility** | **not measured — no such ablation** | — | — |
| Think3D | camera-anchor / choice / ego-view components | +4.35 → +14.29 → +6.19 → +2.26 (BLINK) | Yes (additive) | claimed 3-run, no std |
| Think3D | Pi3 vs VGGT reconstructor | −4.26 BLINK / −0.96 MindCube | Yes | No |
| Think3D | **learned RL view policy vs random from same 8-item menu** | **+10.43 avg**; RL-backbone-with-random-exploration only **+0.9** over random | Yes | No |
| Think3D | integer-angle encoding vs named/free-form | **never ablated** | No | — |
| Think3D | 8-item menu size vs 4 or 16 | **never swept** | No | — |
| MV-GEL | geometry-aware ranking vs Only-CLIP (LISA-CAD, Face F1) | 0.598 vs 0.377 (**−37% rel.**) | Yes | No |
| MV-GEL | geometry-aware vs **random view** | 0.598 vs 0.250 (**−58% rel.**) | Yes | No |
| MV-GEL | FiLM vs Cross-Attn vs No-Fusion | 0.598 / 0.582 / 0.584 — **fusion mechanism barely matters** | Yes | No |
| MV-GEL | vanilla LISA vs CAD-domain-tuned, view selector fixed | Face F1 0.391→0.598, Edge 0.079→0.358 | Yes | No |
| MV-GEL | ViewNMS (suppress near-duplicate views) | **−4.5 pts** — redundancy *helps* a soft aggregator | Yes | No |
| MV-GEL | "CLIP fails on textureless industrial parts" | **asserted in related work**, no textured/textureless split | No | — |
| TwNV | numerical vs natural-language vs discrete instruction (4 combos) | numerical **+0.5…+1.3** consistently; NL **+0.3 / −2.2 / −2.6**; discrete "near-zero" (**no numbers given**) | Yes | No |
| TwNV | generator swap, reasoner fixed | QwenImageEdit **−1.8**, CamCraft +2.5 | Yes | No |
| TwNV | text self-reflection R∈{1..10} vs image iteration N∈{0,1,2} | text never beats R=0; image N=1 best, N=2 regresses | Yes | No |
| TwNV | Qwen3-VL-8B capability threshold | **−4.3** overall, **−26.7** Size | Yes | No |
| Explore-until-Confident | conformal stopping vs relevance-only | 71% vs 85% of budget at matched 58% success | Yes | No |
| Explore-until-Confident | LSV/GSV components, τ sweep, view orientation | GSV-only ≈FBE; orientation **+0.9** | Yes | No |
| Explore-until-Confident | vs random baseline | **none** (baselines are FBE + Relevance, both non-random) | — | — |
| Option-order (Pezeshkpour) | oracle-best vs oracle-worst option ordering | **13.1–73.4 pts** (GPT-4 / InstructGPT) | Yes | No; **N not stated** |
| Option-order | majority vote over 10 random reorderings | **0 to +8 pts**, benchmark-dependent | Yes | No |
| Option-order | MEC calibration | mixed, **−6.9** in places | Yes | No |
| Option-order | option count vs sensitivity | **no correlation found** | Yes | No |
| How-Many-Tools | presented-set K≈2.2 vs K=5, identical descriptions **and ordering** | **−6.0 pp** choice acc (93.1→87.1); **−30 pp** on medium-difficulty with embedding scorer | Yes | **Yes** (3 seeds) |
| How-Many-Tools | similarity/near-duplicate distractors at fixed K | **not tested** — cited from other papers | No | — |
| How-Many-Tools | option ordering | **not tested** — order fixed at scorer rank | No | — |
| MMSI-Bench | CoT on/off; PATS visual prompting on/off; spatial SFT | all null-to-negative | Yes (paired) | No |
| ViewSpatial | task-matched SFT (MVSM) vs backbone | **+46.24** (35.85→82.09) | Yes | No |
| ViewSpatial | encoding of the spatial label | **never varied** (18-way compass words throughout) | No | — |
| RoboSpatial | object-centric vs ego / world frame, GPT-4o | −14.2/−17.6 indoor; **−31.8/−33.0 tabletop** | descriptive, not causal | No |
| RoboSpatial | frame-explicit (Val) vs frame-implicit (Home) context task | 8–41% → **~0–20%** (6/8 models ≤0.1%) | Yes | No |
| SiT-Bench | thinking mode on/off | +5.16 to +7.13 | Yes | No |
| SiT-Bench | VLM-pretrained vs text-only backbone, identical text items | +2.88 / +4.19 | Partially | No |
| SiT-Bench | coordinate-free prose vs raw coordinates | **design choice, never compared** | No | — |

---

## 3. Answers to the branch's questions

### Capability floor — what may an encoding assume at all?

#### Q1 — Is relating one viewpoint to another above chance for models we would deploy?

**Answer: it depends entirely on model tier, and for the open-weights tier the answer on
the closest measured task is no.**

MMSI-Bench, the closest task shape (two images, camera–camera and camera–object
relations), 4-option MC, chance = 25.0:

| Model | Cam–Cam | Cam–Obj | Cam-Motion | Avg |
|---|---|---|---|---|
| Human | 95.7 | 94.2 | 98.6 | 97.2 |
| GPT-5 (reasoning) | 43.0 | 48.8 | 32.4 | 41.9 |
| o3 | 45.2 | 44.2 | 31.1 | 41.0 |
| Gemini-2.5-Pro | 39.7 | 45.3 | 36.4 | 36.9 |
| **Qwen2.5-VL-72B (best open)** | **25.8** | **23.3** | 27.0 | 30.7 |
| Blind GPT-4o (no image) | 20.2 | 13.9 | 20.2 | 22.7 |
| Random | 25.0 | 25.0 | 25.0 | 25.0 |

The best open-source model is **at or below chance** on both camera-relation categories.
Scaling does not fix it (72B beats 32B by 3 points; InternVL3-78B beats InternVL3-1B by
1.5). Frontier closed reasoning models clear chance by 10–25 points and remain 45–70
points below human. MMSI-Bench's human–AI gap (55.3) is the largest among comparable
spatial benchmarks.

Corroborating, from three other benchmark families:

- **SiT-Bench** (blind, text-only, best-case expert-curated descriptions): *View
  Consistency* best 50.5% (Gemini-3-Flash) vs human 75.0, random 25.0; *Pure Mental
  Rotation* best 45.0 vs human 68.0. Every model clusters 30–50%.
- **ViewSpatial-Bench**: *Per-Sce.Sim* (transport yourself to a new vantage point and
  re-derive a relation — the nearest single-image proxy) is **at or below chance for 7 of
  10 models**.
- **Think3D**: BLINK multi-view is 36.8–50.9 for everything including frontier models;
  MindCube-Among 20–49; but MindCube-Rotation is 60/85 for GPT-4.1/Gemini. **No flat
  verdict is available even within one paper's own benchmark selection.**

**Consequence, stated precisely:** RQ3's topology ambition does not die, but it changes
owner. What dies is *asking the model to infer or reason over viewpoint relations*. What
survives is *precomputing relations offline and handing them to the model as flat,
already-resolved facts or labels*. The decisive number for that is SiT-Bench's Cognitive
Mapping subtask — literally "reconstruct a topological map from textual navigation logs"
— where the best of 33 models scores **8.34%** against a human 26.77%. That is close to
non-functional, on pre-digested leakage-filtered text.

#### Q2 — Is "orthogonal to face 3" an allocentric query, and can it be restated egocentrically?

**This is the branch's sharpest live contradiction, and the round-2 framing was wrong.**

- **RoboSpatial** (8 models, 2D and 3D, open and closed): object-centric is the strict
  minimum frame in 10 of 14 model×domain rows. GPT-4o: object-centric **30.7%** tabletop
  vs 62.5 ego / 63.7 world; 38.7 indoor vs 52.9 / 56.3. But note the correction its
  reader flagged: **ego is not uniformly best** — world beats ego for GPT-4o in both
  domains. Cite RoboSpatial for the *object-centric deficit*, not for a
  "camera-centric-default" claim.
- **ViewSpatial-Bench** points the *other* way, and its reader explicitly corrected the
  round-2 selection rationale: the paper's own headline is that camera-perspective
  accuracy (avg 33.2%) sits *below* person-perspective (avg 35.7%). Per-task:
  **Cam-Obj.Ori** (egocentric "which way is this facing") is the worst task in the whole
  benchmark — GPT-4o **19.58%**, Gemini-2.0-Flash **12.95%**, both *below* the 26.10%
  chance floor. **Per-Obj.Ori** (allocentric "imagine you are the object, which way do
  you face") is one of the *strongest* — up to 63.05%.

**The reconciliation the readers converged on** (offered as the best available reading,
not as proof): the predictive variable is the *operation*, not the frame.
Re-anchoring — "adopt this reference and speak first-person from it" — is comparatively
easy. Composing a relation *between two other things inside a third thing's frame*, or
transporting to a hypothetical vantage point, is hard regardless of label.

Two further scoping facts matter:

1. RoboSpatial's object-centric frame is **one global heading vector per object**,
   inherited from human-annotated oriented bounding boxes on semantically-fronted
   household objects. Our "orthogonal to face N" is a **per-surface local frame on a
   multi-faced primitive with no canonical front**. RoboSpatial gives *zero method* for
   constructing such a frame and never tests the multi-frame-per-object case. The mapping
   is partial, not clean.
2. RoboSpatial's real-world, **frame-implicit** test collapses to near-0% for 6 of 8
   models (GPT-4o 5.7%) on context tasks. Whatever frame we use, **it must be stated
   explicitly in every prompt** — never left implicit or inferred.

**The engineering answer, which three systems converge on independently: do the
restatement at the geometry layer, before the prompt exists, and never ask the model to
perform the transform.**

- **GraspView**: candidate poses are sampled in a cone around *the current camera's own
  viewing axis* (Eq. 4), never in a world or object frame.
- **Think3D**: rotations are integer deltas around *a named anchor camera's own vertical
  and right axes*; `(azimuth=0, elevation=0)` *is* the anchor view.
- **MapGPT**: every action label is an egocentric movement primitive ("go forward to",
  "turn around to") — the paper never needed an allocentric relation word and never
  tested one.

Caveat honestly: **no paper anywhere runs a restatement intervention** — the same
geometric fact phrased two ways, measured. RoboSpatial compares independently-labelled
questions across frames; ViewSpatial compares different task types. That the restatement
*preserves the geometry* is untested in the literature.

#### Q3 — What is the measured penalty for composing two spatial operations?

FlipSet was not read, but **AP-VLM supplies a direct, measured, robotics-native
substitute** — and it is arguably better evidence because the action space is ours:

Adding a second rotation axis (`3Dx` → `3Dxy`) drops success from **0.8→0.5** (Scene 1)
and **0.5→0.1** (Scene 2) despite `3Dxy` being a strict superset that can always choose
zero y-rotation. The authors' own diagnosis: "the VLM adds x or y-rotations when both or
at least one of them are not required. The VLM seems to get more confused with multiple
options." Note what did *not* change: the number of actions per step is still one. Only
the *dimensionality within one action* grew, and that alone was enough.

Corroborating: MOKA's hierarchy ablation (§1.8) is the same finding from the other side —
pre-decomposing *what is being decided* is worth 0.6–0.8 of success rate. SiT-Bench's
named "arithmetic-spatial hallucination" failure (entities identified correctly, final
composition wrong) and MMSI-Bench's Multi-Step Reasoning being its own hardest category
are consistent.

**Consequence: one spatial operation per turn. Compose the geometry offline.** Our
interface should never present position and orientation as two simultaneously-adjustable
parameters of one choice; the orientation must be baked into each enumerated candidate.

#### Q4 — Which pose encoding does the model handle best? Resolving TwNV vs MapGPT

**Verdict: the contradiction is a category error. The two papers measure different
pipeline stages, and both can be true.**

- **Thinking-with-Novel-Views §4.3**: numerical (6-DOF, metres + degrees) is consistently
  **+0.5…+1.3** across four reasoner×painter combos; natural language is strongly
  reasoner-dependent (+0.3 with Gemini-3-Flash, **−2.2/−2.6** with GPT-5). But the
  paper's own stated mechanism is about the **generative image model**: "Numerical
  instructions, by conveying only geometric transformation parameters… minimize
  [hallucinations] and ensure **the generative module** produces viewpoint changes that
  faithfully preserve the original scene layout." This is a claim about a diffusion
  model's susceptibility to over-interpreting semantic content, **not** about a VLM's
  comprehension of numbers as reasoning input. Worse for us: TwNV's **"Discrete"
  condition — categorical named views, the format closest to our actual interface — is
  reported only as "near-zero or marginal gains" with no numbers given anywhere.**
- **MapGPT**: "GPT-4V struggles slightly in its attempts to understand navigation
  environments based on multiple precise coordinates"; the discarded quantity is named as
  **GPS coordinates** in a multi-room building. Ablated at matched planning setting:
  topological text map beats coordinate text map **+3.7 SR / +3.7 SPL**. This is
  world-scale absolute coordinates with many digits and an arbitrary global frame — a
  materially different regime from a small local angle.

**What the branch can actually assert about encoding, with sources:**

1. **Selection over emission** — the largest effect by far (SoM +60.7).
2. **Some explicit address over none** — +5.7…+10.3 (SCAFFOLD).
3. **Coarse discretised addresses over raw continuous metric values** — SCAFFOLD's
   Cartesian-grid > absolute-pixel ordering, and MapGPT's topology > GPS. Both point the
   same way.
4. **Numbers are fine as *labels of an enumerated set*, bad as *quantities to reason over
   or emit*.** Think3D's deployed interface is the clean instance: signed integer degrees,
   integers only, restricted to 8 enumerated `(azimuth, elevation)` pairs for the small
   model. SiT-Bench's construction convention agrees — prose with occasional embedded
   distances/angles, raw coordinate triples explicitly kept out of the model-facing text.
5. **Names over opaque indices** — SayPlan's position (semantically legible node names,
   metric pose as a side-channel attribute for the executor only). **Demonstrated, never
   ablated** — flag it as a pattern, not evidence.
6. **Redundant cues stack** — SoM Table 3, Number+Mask+Box beats Number+Mask by +4.8.

**The genuine measurement gap:** nobody has ablated named vs indexed vs angular labels
for a discrete camera-pose menu. Think3D uses integers and never varies them; MOKA uses
chess notation and never varies it; ViewSpatial uses compass words and never varies them.
This is a gap our own benchmark could fill cheaply and probably should (§5).

---

### Family choice — blind, seeing, or hybrid?

#### Q5 — Text vs image on paired encodings: which wins, under what conditions?

Not a verdict — a set of conditions.

**Image wins when the task is "point at / indicate a location in what you are looking
at."** PIVOT Table 4 is the clean ablation: image annotation beats a 3×3-grid *text*
description of the identical choice on every navigation category, and with markedly
lower variance (In-View 0.21±0.002 vs 0.26±0.15).

**A new observation wins over more deliberation, always** (§1.4, TwNV + Think3D).

**Text wins, or ties, when the information is symbolic and already resolved.** SayPlan
reaches 86.6% executability blind; MapGPT's blind arm beats its seeing arm on oracle
success; SiT-Bench shows blind models do 30–50% on viewpoint-relation items that seeing
models do barely better on.

**The image only helps if it is explicitly bound to the text.** This is SeeGround's
mechanism finding, and it is the most useful single result for a hybrid design: adding
layout or texture channels alone to raw 3D-position text buys only ~2pp (37.7→39.5/39.7);
adding the Fusion Alignment Module that ties in-image markers to text IDs buys
**+3.6–3.8**. And row (e) — *drop* the 3D-position text, *add* query-aligned rendering —
beats row (d) which has coordinates but no aligned render (45.0 > 43.3). A well-aimed,
identity-bound render can substitute for metric coordinate text.

**The image hurts when its fidelity is below threshold** (§1.5) **or when the model is
below the mark-reading threshold** (§1.1).

**Design rule extracted from SeeGround Table 9:** point markers (47.7) ≈ contour (47.5) >
bbox (45.1) ≈ mask (45.0). Region overlays *hurt* because they occlude the appearance
cues the model needs and duplicate information already carried by the text channel.
**Don't let a visual prompt re-encode what the text already says; let it carry only what
the text cannot — identity binding.**

#### Q6 — What exactly does a caption hand-off destroy, and can better questions recover it?

**Measured answer (MapGPT, the only clean head-to-head):** it destroys **decisiveness and
path efficiency** (SR −3.2, SPL −5.8), not coverage (OSR **+5.8**). A blind agent fed
lossy descriptions keeps exploring rather than committing.

**How much is recoverable without vision:** SayPlan's verify-then-correct loop drives
long-horizon executability from 13.3% to **86.6%** on a purely textual, first-violated-
precondition diagnostic — with no camera anywhere in the pipeline. That bounds how much
visual feedback is actually needed: a large fraction of the "blind orchestrator problem"
is recoverable through structured *symbolic* feedback.

**What is not recoverable:** the TwNV result (§1.4) shows re-deliberation over the same
information never beats not re-deliberating. If the missing thing is *visual evidence*,
better questions do not manufacture it.

RePLan and RAVEN — the two papers that would sharpen this — were not read.

#### Q7 — For the seeing option, what is the render?

**Branch-wide blank. Nobody in this reading pass rendered a defective, real, sparse,
partially-holed survey point cloud for a VLM.**

| System | What was actually rendered | Why it does not transfer |
|---|---|---|
| GraspView | off-screen pinhole render of the "current colored point cloud," colours back-projected from latest RGB, splatted with Z-buffering — *that is the entire treatment* | the cloud is **VGGT-predicted** (learned dense monocular reconstruction), not a sensor scan; learned completion is visually dense/plausible where a real scan has holes; ≤3 real views ever fused |
| SeeGround | ScanNet dense RGB-D room scans; depth-aware z-buffer visibility (Eqs. 15–17) with an unspecified threshold α | the z-buffer test **cannot distinguish "occluded by something" from "never sensed"** — exactly our hole ambiguity. Risk acknowledged (§9.2) but never measured |
| Think3D | Pi3 fused "**clean** colored point cloud" — the word "clean" is asserted once and never tested | no discussion of holes, sparsity, or dropout anywhere |
| MV-GEL | 60 synthetic renders of a clean watertight CAD B-rep mesh; shading/material spec not stated | visibility oracle ray-casts against error-free ground truth and presupposes the target's exact 3D extent is known |
| AP-VLM | **no render at all** — an ArUco+FK-projected grid overlaid on the *live* RGB frame; "we did not use depth" | a real third answer, but it requires fiducial infrastructure and yields a **workspace-anchored, geometry-blind** grid |
| SoM / SCAFFOLD / MOKA / PIVOT | clean, complete, textured RGB photographs | nothing about non-photographic substrates |

**What we can carry forward as risk-shaping evidence rather than answers:**

- **A fidelity floor exists and is measurable** (TwNV: a bad visual artifact is **worse
  than none**, −1.8pp).
- **Referential mark accuracy collapses on hard geometry** (PIVOT Table 7: 0–44% on
  eccentric objects vs 88–100% on a blank background) — the single closest signal for
  "will marks survive on a holey render."
- **Mark placement precision is *not* the bottleneck** (SCAFFOLD dot perturbation at
  σ = ¼ inter-dot spacing retains the gain, and *improves* VSR) — mild positive for
  placing marks on geometry with a few-mm/few-cm offset.
- **Marks must not collide with image content** (SoM's own caution: "avoid numeric marks
  on an arithmetic image"). Inspected industrial parts carry serial numbers and printed
  text — this is a live named risk for us, mitigated in SoM only by manual per-image
  judgement.
- **Mark type matters and region overlays hurt** (SeeGround Table 9).

#### Q8 — The chicken-and-egg of choosing the render's own viewpoint: five published answers

| Answer | Who | Cost / limit |
|---|---|---|
| **Text-only call picks a semantic anchor; closed-form geometry places the camera** (back off and up from the anchor) | SeeGround (PAM) | works only because "back off and up" is a good *generic* viewpoint. **The paper contains zero mentions of orthogonality, incidence angle, or surface normals** — a fixed offset rule has no way to express "normal to face 3" |
| **Deterministic egocentric sampler around the current view; render all feasible; VLM scores one at a time; argmax in code** | GraspView | never compares candidates jointly; no persistent named space; **VLM scoring never validated against random or a classical utility** |
| **Don't render — project marks onto the live camera frame via calibration + FK** | AP-VLM | needs 9 ArUco markers; grid is workspace-anchored, i.e. it *instantiates* our stated problem |
| **Anchor to an already-captured view and specify an integer delta from it** | Think3D | the anchor must be one of the *original input* images; no notion of a candidate not yet visited |
| **Remove the prompted model from selection entirely — a trained geometry-conditioned ranker picks, the VLM only consumes the winner** | MV-GEL | needs a visibility oracle on clean geometry plus training data; the objective is segmentation F1, **not orthogonality** |

3D-Mem's frontier snapshots and Manipulate-Anything's tiled contact sheet — the two
answers the round-2 map flagged — were not read.

---

### Interface mechanics — if the space is a menu, how is the menu built?

#### Q9 — How many options, and how similar may they be?

**Count: measured, and the effect is real.** How-Many-Tools §4.3 is the only variance-
reporting result here — Claude Sonnet 4.6, forced single choice, **tool descriptions and
ordering held identical, only set size varies**, 120 queries, 3 seeds:

| Presented set | Choice accuracy |
|---|---|
| K ≈ 2.2 (BoR) | **93.1 ± 0.5** |
| K = 5 (fixed) | **87.1** |

−6.0 pp from distractor count alone. On the medium-difficulty bucket (gold ranked 2nd–5th)
with an embedding scorer, the gap widens to **80.1 vs 50.0 — 30 points**.

Corroborating operating points from elsewhere: Think3D's authors **collapsed an open
continuous integer space to 8 enumerated `(az, el)` pairs** for the small open model
while leaving it open for frontier models (never ablated, but a revealing engineering
choice); PIVOT settled on 10 samples because more crowd the image and cause marker
occlusion; Explore-until-Confident fixed 3 marks and never varied it; MOKA runs ~25–40
simultaneous options for GPT-4V (an untested operating point, not a measured tolerance).

**Similarity: assumed everywhere, measured nowhere.** This is a correction to the round-2
premise. How-Many-Tools' own reader flagged it: the paper's "semantically similar tools
hurt" claim is a **citation to LongFuncEval and Rabinovich & Anaby-Tavor, not a result it
produces**; no ablation varies distractor similarity at fixed K. SoM, MOKA, MapGPT and
PIVOT all lack any similarity manipulation. MapGPT's options are always visually distinct
rooms — its reader states explicitly that the count-vs-distinctiveness hypothesis is
"simply outside the paper's design space."

The one datum pointing the *other* way — MV-GEL's ViewNMS ablation, where **suppressing
near-duplicate views hurts by 4.5 points** because nearby views act as a fallback — is
about a *differentiable multi-view aggregator's recall*, not a discrete-choice LLM. Its
reader explicitly flags it as likely non-transferring in direction. **Do not cite it as
evidence that redundant candidates are harmless for a language menu.**

**Net for RQ3:** the branch's founding worry ("20 near-identical geometry-conditioned
views may be *harder* than 36 uniform cells") is **neither confirmed nor refuted by any
paper read.** It remains the branch's most important untested hypothesis and belongs in
our own benchmark (§5).

#### Q10 — How much does ordering matter?

**Correct the number first.** The round-2 map says 13–85%. The paper (Table 1) says
**13.1–73.4 points**, and — load-bearing — it is an **oracle best-vs-worst bound found by
searching orderings using ground-truth knowledge**, not the swing between two arbitrary
orderings. The practically-realised effect is better estimated by the paper's own
majority-voting experiment over 10 *random* reorderings: **0 to +8 points recoverable**.

**The mechanism is the transferable part, and it lands squarely on our hard case.**
Conjecture 4.1: sensitivity arises from *uncertainty among the model's top-2/3 candidates*
× *positional bias*. Sort-and-truncate confirms the wobble lives inside the top-2/3;
self-verification confirms it (>94% of order-sensitive questions are ones the model
itself flags as having multiple plausible answers). **Placing the top-2 candidates at the
first and last slots amplifies the bias; placing them adjacent dampens it.**

**And "we only have a few options" is not a defence:** the paper found **no correlation
between option count and sensitivity magnitude** — its 3-option benchmark produced the
single largest gap in the whole table (73.4, InstructGPT).

Caveats: GPT-4 and `text-davinci-003` only, 2023, text-only knowledge MCQs, no
multimodal condition, no stated N for the main experiment, no CIs. PIVOT's prompt-*element*
ordering ablation (0.43 vs 0.41, "small margin") is a different question and does not
substitute. Zheng et al.'s token-vs-position decomposition was not read.

**Consequence:** if our menu's order is geometrically meaningful (face 1, face 2, …;
orbit 0°, 30°, 60°), that order is a *variable*, not a neutral formatting choice. Our
benchmark must permute (§6).

#### Q11 — How should infeasible (IK/collision-rejected) views appear?

Four patterns are observed in the wild. **None is ablated against any other, anywhere.**

1. **Silent removal upstream** — GraspView: "We retain only kinematically feasible
   candidates (IK + joint/velocity limits)" *before* the VLM sees anything. Chosen, never
   examined.
2. **Unhandled** — AP-VLM: grid vertices map to pre-calibrated base coordinates with no
   dynamic feasibility check; the paper's own limitations state "Sometimes, the actions
   are out of reachable robot workspace." A documented negative example.
3. **Reactive rejection with a textual diagnostic** — SayPlan: a scene-graph simulator
   forward-simulates the plan and returns the *first* violated precondition as prose; the
   LLM revises. Measured worth: executability 13.3→86.6%. Measured cost: up to 5
   replanning rounds, and **2 of 15 long-horizon tasks never converged within the cap.**
4. **Affordance down-weighting (SayCan)** — in the map, not read.

**No paper preserves the model's ability to reason about coverage in the face of
infeasibility.** GraspView's silent removal means the model cannot know a surface is
unreachable; SayPlan's reactive loop means it discovers this only after committing. This
is an unfilled design space, and it matters for us because the contract states plainly
that an orthogonal view that exists geometrically may not exist kinematically.

#### Q12 — Is dumping a small space worse than exposing it through a query or hierarchy?

**Answer: at our scale, no — dump it flat.** The evidence is unusually clean.

- **Structured-Interfaces (Cypher)** is the direct test, and its *small*-graph column is
  the relevant one (65 objects, 96 places, 5 regions — roughly our order of magnitude):
  Q&A Cypher(A) 0.85 vs context-dump 0.75 (+10); **PDDL 0.65 vs 0.65 — tied**; token
  reduction only 3.4×. The dramatic win (0.77 vs 0.33; **240× tokens**) requires 15,944
  places. The paper's own words: "the context window baseline is a competitive method on
  small scene graphs."
- **SayPlan**'s collapse/expand exists to fit an 8192-token budget. Its scalability
  ablation (30→300 room nodes) found that **every failure at scale was a context-limit
  failure, not reasoning degradation from distractor nodes** — "the LLM's reasoning
  capabilities or ability to attend to the relevant parts of the 3DSG is not
  significantly impacted by the presence of 'noisy' or increasing number of nodes." Its
  smallest tested point is already an order of magnitude above a per-primitive view menu.
- **A query interface does not guarantee correct use.** Structured-Interfaces §5: "all
  methods struggle with language that requires traversing the hierarchy… While Cypher
  does provide a transitive containment operation, GPT-4.1 inconsistently chooses to
  generate queries that take advantage of it." Exposing the operation ≠ the model using it.

**But the hierarchy that *does* pay is at the decision level, not the menu level** — MOKA's
0.6–1.0 → 0.0–0.2 collapse when the high-level decomposition step is removed (§1.8).

**Net:** flat dump of the leaf menu; a cheap high-level framing turn above it.

#### Q13 — Stopping criterion

**Exactly one defensible calibrated rule exists in the literature, and it is portable as
a pattern but not as an implementation.** Explore-until-Confident, Sec. V:

1. Per-view confidence is **relevance-gated**: `ρ_y(x) = Rel(x)·(f̂_y(x) − 1)`, where
   `Rel` is a separate yes/no VLM query "is this view relevant to the question at all."
   A confident answer from an irrelevant view is discounted. *(This is the structurally
   important idea for us: a confident defect judgement from a view of the wrong face must
   not count.)*
2. Episode confidence is the **running minimum** over timesteps, not the average
   (Eq. 9) — one bad/occluded view still constrains the conclusion.
3. **Split conformal prediction** converts the heuristic score into a covered threshold:
   `q̂` = the `⌈(N+1)(1−ε)⌉/N` empirical quantile of `1 − ρ̄` over a calibration set
   (300 calibration / 200 test scenarios here).
4. Stop when the **intersection** of per-timestep prediction sets collapses to a
   singleton.

Measured payoff: at matched 58% success, **71% of budget used vs 85%** for the
relevance-only baseline. The win is *efficiency*, not accuracy.

**Hard constraints on porting it:** it requires per-token logprobs for custom label sets
(the authors deliberately used the open-weight Prismatic VLM because GPT-4V lacked
reliable logprob access); the calibration set must be representative and labelled (our
real-defect calibration set will be far smaller than 300); and the label space `Y` is a
QA answer set — for a *coverage* task we must invent the analogous label space ourselves,
and the paper offers no guidance on how.

**Everything else in the branch is a cap or a saturation curve, not a rule:**

- GraspView Table II: **1st NBV +35 pts on hard occlusion, 2nd +15, 3rd +0.**
- TwNV: iterative refinement N=1 optimal (79.6), **N=2 already regresses** (79.3) —
  attributed to error accumulation and drift.
- Think3D: hard cap K=3, model-judged STOP (which is simply "emit `<answer>` instead of a
  tool call"), no calibration. And turn count does not help without RL.
- AP-VLM: 10-iteration cap plus self-report, and the paper is internally inconsistent
  between "binary signal Z" and "a high certainty score" — **uncalibrated**.
- How-Many-Tools' BoR: a chance-corrected reward that decays mathematically with depth,
  requiring **no calibration set** — but it assumes single-hit retrieval, not coverage.
- MapGPT: an always-listed `"A. stop"` menu item. Cheap, and worth copying regardless.

**Convergent practical reading across four independent systems: 1–2 extra views captures
essentially all the achievable gain, and the third is free at best.**

---

### Evidence discipline

#### Q14 — Ablated or demonstrated? → §2 above.

Summary of what that table shows: the encoding claims that are genuinely ablated are
SoM's coordinate-vs-mark, SCAFFOLD's coordinate-format and colouring, PIVOT's
image-vs-text and sample-count, MapGPT's topology-vs-coordinates and seeing-vs-caption,
SeeGround's five view-placement heuristics and its component stack, MV-GEL's selector
ablation, TwNV's generator and instruction-format swaps, Think3D's component and
random-vs-learned-policy ablations, AP-VLM's 8-way action-space comparison, and
How-Many-Tools' K sweep. Everything else — MOKA's index encoding, SayPlan's JSON schema,
Think3D's integer-angle format and 8-item menu size, GraspView's entire VLM-scoring
premise, SiT-Bench's coordinate-free prose, MV-GEL's textureless claim — is **asserted or
demonstrated, not compared**.

#### Q15 — What is the random baseline?

**The branch's most consequential negative finding: no published result shows a
*prompted* VLM beating a random view-selection baseline, with variance.**

The two positive random comparisons both use *trained* selectors:

- **Think3D Table 6** (the branch's only genuine random-vs-policy view-selection
  comparison): Random (2 of 8 candidate angles) **36.68** avg vs learned RL policy
  **47.11** (+10.43). Critically, an RL-tuned backbone with *random* exploration scores
  **37.56** — only **+0.9 over random**. "Gains mainly come from spatial policy
  learning." Nearly all the benefit is attributable to *which* views, not to the backbone.
- **MV-GEL**: random view Face F1 **0.250** vs geometry-aware ranking **0.598** (>2×).
  The chooser is a LoRA-trained ranker, not a prompted model.

The systems where a *prompted* VLM does the choosing have no random baseline at all:

- **GraspView**: baselines are "no active perception" and an RGB-D sensor — never random
  view, never a classical coverage/information-gain utility. Its related-work claim that
  render-and-score "removes the need for bespoke heuristics" is **argued, never measured**.
- **AP-VLM**: baselines are a single fixed home view (NAP) and TGCSR's 5 *hand-chosen*
  fixed cameras. No random arm.
- **Explore-until-Confident**: baselines are FBE (already a non-random heuristic) and
  relevance-only.
- SPAZER's +3.6%-over-random — the nearest thing to a negative result — remains
  **unverified in this pass**.

**Usable machinery for building the baseline properly:** How-Many-Tools' chance-corrected
formalism, `P_rand = 1 − C(N−R_q, K)/C(N, K)` (→ `K/N` for single-hit), and
`BoR = log₂(P_obs / P_rand)`, with the ceiling `BoR_opt(K) = log₂(N/K)`. Caveat from its
own reader: the paper only ever runs the "≥1 relevant item" success rule; a coverage
version (Recall@K) is supported in principle but never validated. And MMSI-Bench's
pattern of reporting a **per-category** chance floor plus a **blind (no-image) baseline**
to prove the task isn't solvable from priors.

#### Q16 — Named failure modes, for Branch 2

Grouped by where they bite. Each is sourced; each is a candidate test case.

**Frame and relation errors**
- *Allocentric / situation-transformation error* — MMSI-Bench's own grader taxonomy names
  this verbatim as a top-level category.
- *Perspective bleed* — ViewSpatial: models "alternat[e] between human and camera
  perspectives" within a single response; "GPT-4o frequently defaults to camera-centric
  judgments" even when explicitly asked to adopt another frame.
- *Below-chance egocentric object-facing* — GPT-4o 19.58%, Gemini-2.0-Flash 12.95% vs
  26.10 chance.
- *Object-centric frame collapse* — RoboSpatial, GPT-4o 30.7% tabletop vs 62.5 ego.
- *Frame-implicit collapse* — RoboSpatial-Home context: ~0% for 6 of 8 models.
- *Live ego/allo tangling mid-CoT* — SiT-Bench §B.3.1 has a quotable Qwen3-30B trace
  visibly confusing "the right wall" with its own right, reaching the right answer by luck.

**Structure and aggregation errors**
- *Overlap-matching / scene-reconstruction error* — MMSI's largest error class.
- *Cognitive-mapping failure* — SiT-Bench, best 8.34% vs human 26.77%.
- *Hierarchy-traversal failure even with the tool available* — Structured-Interfaces §5.
- *Negation failure* — SayPlan: "find an office that does **not** have a cabinet."
- *Distance / count comparison failure* — SayPlan: "the closest fire extinguisher to X."
- *Transitivity failure and hallucinated relations* — MMSI's spatial-logic subtypes.
- *Arithmetic-spatial hallucination* — SiT-Bench: right entities, wrong final composition.

**Menu and mark errors**
- *Multi-axis menu confusion* — AP-VLM `3Dxy`: the model adds rotations that were not
  required.
- *Positional/order bias under top-2 ambiguity*, amplified by extremes placement —
  Pezeshkpour & Hruschka.
- *Distractor-count degradation at matched descriptions/order* — How-Many-Tools, −6 to
  −30 pp.
- *Mark-attention mislocalisation* — SoM: model picks the mark *near* the target rather
  than *on* it (backpack chosen instead of person).
- *Content/mark symbol collision* — SoM's OCR-confusion caution; live risk on parts
  carrying serial numbers or printed text.
- *Categorical mark-blindness in weaker models* — SoM: LLaVA-1.5, MiniGPT-v2.
- *OCR dependence of the whole pipeline* — SeeGround §7.1.
- *Referential mark collapse on hard geometry* — PIVOT Table 7: 0–44%.
- *ID hallucination* — SayPlan's only surviving error class after replanning (6.67%).

**Execution and stopping errors**
- *Unreachable-workspace commands* — AP-VLM's own limitations.
- *Discretisation granularity floor* — AP-VLM's fixed ±35° provably fails to reach the
  needed steep angle in Scene 3.
- *Confidently wrong despite seeing the right thing* — Explore-until-Confident (2 of 6
  hardware failures).
- *Miscalibrated raw relevance* — Explore-until-Confident, Fig. A5: successful episodes
  often had top relevance < 0.5.
- *Sub-cm precision failure through discrete commands* — FAEA (map only, unverified);
  MOKA's architectural workaround (snap to nearest classical candidate) is the read
  evidence that this is a real boundary.

**Meta / benchmark-hygiene**
- *Correct answer, wrong reasoning* — MMSI §5.2: reasoning-process accuracy is 8–10 points
  **below** answer accuracy (GPT-4.5 37.5% vs 40.3%; Qwen2.5-VL-72B 21.5% vs 30.7%).
  Never grade a stated justification as evidence of understanding.
- *Wrong ground truth* — SoM Fig. 8: some benchmark "failures" are annotation errors.
  Manually verify every scored-wrong answer.

---

### Cross-branch handoffs

#### Q17 — To Branch 5: does any measured limit forbid a view-adjacency graph?

**It forbids the model *deriving* one. It does not forbid Branch 5 *building* one.**

Forbidding evidence: SiT-Bench Cognitive Mapping at **8.34%** (best of 33 models) is the
most direct measurement in the literature of "can an LLM build the adjacency structure
itself from descriptions" — and the text was pre-digested and leakage-filtered.
MMSI-Bench Cam–Cam at/below chance for the deployable open tier means relational
addressing ("the view opposite this one") has no measured floor for that tier.
Structured-Interfaces §5 shows that even handing the model a transitive-containment
operator does not make it use the operator reliably.

Permitting evidence: MapGPT's ablation shows a **precomputed** topology verbalised as
flat connectivity sentences is worth **+4.6 SR / +5.4 SPL** over no map, and beats a
coordinate map by +3.7 SR. That is a topology the *simulator* supplied and the agent only
*tracked*.

**Instruction to Branch 5:** build whatever adjacency structure the geometry supports, but
budget for it to be consumed as **flat, already-resolved facts** ("Face 3's opposite is
Face 5") or as pre-labelled menu items — never as a structure the model must traverse,
compose over, or infer. Also note Think3D and GraspView both operate *below* the
adjacency abstraction (raw pose deltas from an anchor, locally regenerated candidates) and
neither needed a graph — GraspView is a working counter-example that the graph may be
unnecessary machinery for a 1–2-view budget.

#### Q18 — To Branch 3: does geometry-conditioned view ranking survive a defective cloud?

**Mechanism supported on clean geometry; transfer to our cloud completely unresolved.**

MV-GEL's supporting numbers (LISA-CAD rows): geometry-aware ranking **0.598** Face F1 vs
Only-CLIP **0.377** (−37% relative) vs Random **0.250** (−58% relative); Edge F1 0.358 vs
0.210 vs 0.101. Its conclusion — "language-driven 3D localization is fundamentally a
geometry-aware view prioritization problem, not merely an image-text alignment task" — is
directly aligned with our analytical track.

Four caveats Branch 3 must carry:

1. The **"textureless industrial parts" claim is a single related-work sentence** (§2.4),
   not a controlled textured/textureless split. The Only-CLIP row supports the mechanism
   (semantic-only ranking fails on fine geometric targets) but does not isolate
   texturelessness as the cause.
2. **The visibility oracle is the part most likely to break first.** It ray-casts against
   a ground-truth, error-free watertight mesh and presupposes the target entity's exact
   3D extent is already known (needed to compute the top-k positives for training). On a
   cloud with pose error, dropout and a ~4 cm fused-vs-single-view offset, that
   presupposition is the failure point — not the ranking architecture.
3. **The objective is segmentation F1, not orthogonality.** A top-ranked GELview could be
   oblique if obliqueness maximises visible entity area. Do not import "GELviews ≈
   orthogonality proxy" without re-deriving the objective.
4. **The downstream model dominates the view selector.** Domain-adapting the segmenter
   (vanilla LISA → LISA-CAD) at a *fixed* view selector moves Face F1 0.391→0.598 and
   Edge F1 0.079→0.358 — larger than every view-selection effect in the paper. A caution
   against over-crediting view selection when the consumer is undertrained for the domain.

Also for Branch 3: MV-GEL's edge localisation is roughly **half** as good as face
localisation for every selector including the oracle (0.358 vs 0.598; oracle 0.390 vs
0.632), attributed to thin projections, discretisation sensitivity and view-dependent
occlusion. Structural, not selector-fixable.

#### Q19 — To Branch 2: metrics, protocols, and mandatory arms

**Metric candidates surfaced or confirmed in this pass:**
- **OSR (Oracle Success Rate)** — AP-VLM: success if the closest trajectory point to the
  goal was within 0.1 m even if the episode did not stop there. Separates "got close but
  stopped/answered wrong" from "never got close." Cheap, diagnostic, directly reusable.
- **Geometry-weighted IoU** — MV-GEL: weight by triangle area (faces) / edge length
  (edges), not raw point or pixel count. Our coverage metric should weight by real
  surface area for the same reason.
- **Visibility-oracle count** — MV-GEL: `S(v, E) = Σ 1{point of E visible from v}` by
  ray-casting. Adaptable as the geometric backbone of an orthogonality + framing score.
- **Chance-corrected BoR** — How-Many-Tools: `P_rand = 1 − C(N−R,K)/C(N,K)`,
  `BoR = log₂(P_obs/P_rand)`. Needs re-derivation for a coverage (Recall@K) success rule.
- I-Perceive's View Coverage IoU and SeqVLM's projected-area score remain from the
  round-2 map, **unread**.

**Protocol templates:**
- **PIVOT Appendix G** — a two-part annotation-legibility protocol: (a) classify an
  isolated mark on a blank background, (b) match a mark to the object/geometry it
  indicates on a real scene. The gap between the two (88–100% vs 0–44%) is the actual
  measurement. This is exactly the test we need before trusting marks on a point-cloud
  render.
- **TwNV Table 3** — error attribution as *wrong instruction / bad render / reasoning
  failure*, tracked **before and after** each intervention. TwNV found that fixing the
  instruction loop shifted blame to generation (Wrong-Instruction 81→24, Bad-Generation
  85→101). Our benchmark should expect the same blame-shifting.
- **SCAFFOLD Tables 2–3** — clean A/B/C/D/E encoding comparison on a fixed subset with
  each condition run ≥2×. The template for our own marked-render ablation
  (no-mark / numbered / coordinate-grid / named-index).
- **MMSI-Bench** — per-category chance floor + a blind (no-image) baseline row.
- **AP-VLM's 8-way action-space table** — the template for sweeping our own action-space
  variants, with the correction that it should report variance (it does not).

---

## 4. What is contested — kept visible, not smoothed

1. **Ego vs allocentric.** RoboSpatial: object-centric is the weakest frame across 8
   models (GPT-4o −31.8 pts vs ego on tabletop). ViewSpatial: adopting the object's frame
   to state a facing (Per-Obj.Ori) is one of the *strongest* tasks, while doing the same
   in the camera's frame (Cam-Obj.Ori) is *below chance*. Both are single-image QA on
   photographs; neither tests a per-surface local frame on a primitive; neither runs a
   restatement intervention. The proposed reconciliation (operation, not frame, is the
   predictor) is a reading, not a result.

2. **Does the orchestrator need to see?** Four incompatible architectures, no head-to-head:
   AP-VLM (one seeing agent judges sufficiency *and* picks the next view off the same
   annotated frame) · GraspView (VLM sees renders but only scores; argmax in code) ·
   SeeGround (blind text call decides *where*, seeing call decides *what*) · MV-GEL (no
   prompted model in the selection loop at all). MapGPT's blind arm beats its seeing arm
   on one metric and loses on two.

3. **Numbers vs names.** TwNV says numerical beats language; MapGPT says precise
   coordinates confuse. Reconciled here as different pipeline stages (§Q4) — but note
   TwNV's Discrete condition, the one closest to our interface, is the least documented
   thing in that paper. SCAFFOLD complicates both by showing the choice *among* schemes
   matters far less than having one.

4. **Do near-identical options hurt or help?** How-Many-Tools measures count hurting
   (−6 to −30 pp) but never isolates similarity. MV-GEL's ViewNMS measures redundant
   near-duplicate views *helping* (+4.5) — for a soft aggregator, explicitly flagged as
   likely non-transferring. The branch's founding hypothesis is untested.

5. **Hierarchical exposure.** SayPlan: collapse/expand is essential (token reduction 86.9%
   / 72.5%). Structured-Interfaces: at small-graph scale, dumping ties. MOKA: hierarchy is
   the single biggest lever — but at the *task-decomposition* level, with a flat leaf
   menu. All three are true; they are about different objects.

6. **Do more iterations help?** PIVOT: non-monotonic (Franka reach success sometimes drops
   to 0% with more iterations). TwNV: N=1 best, N=2 regresses. Think3D: turn count does
   not help without RL. GraspView: saturates at 2. Explore-until-Confident: the gain is
   efficiency, not accuracy. Convergent practical answer: 1–2, and no free lunch beyond.

7. **Is AP-VLM prior art or a cautionary tale?** It is the closest published system to
   ours — UR5, in-hand camera, zero-shot VLM picks the next viewpoint from a marked grid,
   loops until it can answer. And its grid is anchored to the *workspace origin* via
   ArUco markers, uses **no depth at all** ("we did not use depth"), and its own
   limitations report unreachable commands and a 35° rotation granularity that provably
   fails on an inclined object. **It instantiates the exact flaw Albert flagged rather
   than solving it.** Its transferable content is three measured failure modes and a
   documented absence of every baseline we would need.

---

## 5. What is missing — the branch's gap list

**Measurement gaps in the field (each is a candidate contribution for us):**

1. **Nobody renders a defective real survey cloud for a VLM.** Every render in this branch
   is either a photograph, a learned dense reconstruction (VGGT/Pi3), a ScanNet room scan,
   or a clean CAD mesh. Sparsity, holes, colourless geometry and glossy dropout are
   untouched. *This is the single largest blank.*
2. **No prompted VLM has been shown to beat random view selection with variance.** Both
   positive random comparisons use trained selectors (Think3D RL, MV-GEL ranker).
3. **Option similarity has never been varied at fixed count** for any menu.
4. **Option ordering has never been tested for a spatial or multimodal action menu.**
5. **No benchmark measures whether VLMs can interpret textual azimuth/elevation
   descriptions** — Think3D deploys them and never ablates the format. This is precisely
   our current {h,v} interface.
6. **No restatement intervention exists** — the same geometric fact phrased in an object
   frame vs the camera's frame, measured on the same model.
7. **No comparative study of infeasible-option presentation** (silent removal vs listed-
   unavailable vs affordance-scored vs reactive diagnostic).
8. **Variance reporting is essentially absent branch-wide** (4 of 20 sources).
9. **Object-centric frames for primitives with no canonical front** — RoboSpatial's
   headings come from human annotations on chairs and cars; nothing addresses a box or
   cylinder on an inspection table.

**Reading gaps in this pass** (§0): SPAZER's random-view margin, TagaVLM's +0.94% text-
topology ablation, the SoM relational-degradation critique, FlipSet's compositional
deficit, 3D-Mem, Manipulate-Anything, RAVEN, SayCan, Zheng et al. Four of these carry
claims the round-2 map treated as load-bearing. **Round 5 should verify SPAZER and
TagaVLM first** — they are the two most direct threats to the current design and neither
has a substitute in what was read.

---

## 6. Implications

### RQ1 — Judging a view

- **Build the judge in geometry space; do not let a VLM be the primary judge.** GraspView
  is the archetype of the risk: it replaces classical utilities with VLM scoring and
  **never benchmarks the VLM's scoring against a classical utility or a random pick**. Its
  "hand-crafted utilities are brittle" claim is argued, not measured. Any VLM-as-judge in
  our system must be benchmarked against a geometric judge, or the metric is circular.
- **Two concrete, implementable geometric primitives are available now:** MV-GEL's
  ray-cast visibility oracle `S(v,E) = Σ 1{visible}`, and its geometry-weighted IoU
  (weight by triangle area / edge length, not point count). Both work directly on a PLY.
- **The judge must be relevance-gated.** Explore-until-Confident's structural insight —
  a confident answer from an irrelevant view is discounted, not trusted — maps exactly
  onto "a confident defect judgement from a view of the wrong face must not count toward
  coverage."
- **Do not import "geometry-conditioned view ranking ≈ orthogonality."** MV-GEL optimises
  segmentation F1; a top-ranked view can be oblique. Orthogonality has to be in the
  objective explicitly.
- **The downstream consumer may dominate the judge.** MV-GEL's domain-adaptation effect
  (+0.207 Face F1 at fixed selector) is larger than every selector effect. Before
  optimising the view score, check that the thing consuming the view is competent on our
  imagery.

### RQ2 — Selecting best views

- **Deterministic geometric candidate generation is the published consensus** (§1.7). Six
  of six systems generate candidates by rule; none lets a prompted model construct the
  set. This aligns with the contract's allowance for primitive-level deterministic rules
  and is not an ADAM regression — the rule is per-primitive, not per-object.
- **Frame the candidates egocentrically at generation time.** GraspView's conical cap
  around the *current* viewing axis and Think3D's integer delta from a *named anchor
  camera* are the two worked patterns. This dodges the object-centric deficit at the
  geometry layer, where it costs nothing, rather than in the prompt, where it is measured
  to be expensive.
- **Filter for reachability before the menu exists** — GraspView's silent removal is the
  only pattern with a working real-robot deployment, and AP-VLM is the documented failure
  when you skip it. But see RQ4: silent removal destroys the model's ability to know a
  surface is unreachable, so pair it with an explicit textual statement of what was
  removed and why.
- **A small candidate set suffices.** MV-GEL closes to oracle within 7–8 of 60 views;
  GraspView saturates after 2 NBVs; Think3D's deployed menu is 8. Nothing in this branch
  supports a 36-cell space.
- **Orientation must be baked into each candidate, never offered as a separate axis** —
  AP-VLM's `3Dxy` collapse and PIVOT's explicit refusal to attempt rotation via visual
  prompting ("we fix these due to the difficulty of expressing them") both say the same
  thing. PIVOT's admission matters especially: our core requirement is an *orientation*
  problem, and PIVOT — the paper the round-2 map called load-bearing — explicitly dropped
  the orientation DOF as too hard.

### RQ3 — Composing the action space

- **Precompute every relation; expose none as an inference.** SiT-Bench's 8.34% Cognitive
  Mapping and MMSI's at-chance camera–camera for open models mean the model cannot build
  or traverse topology. MapGPT's ablation means a *given* topology, verbalised flat, is
  worth +4.6 SR. Those two facts are compatible and jointly prescribe the design.
- **Two-level decision structure, flat leaf menu.** MOKA's ablation (0.6–1.0 → 0.0–0.2) is
  the largest measured effect in the visual-prompting literature and it is about
  decomposing the decision, not the menu. For us: turn 1 = which surface/primitive;
  turn 2 = which view of it. Never both.
- **Target menu size in the low single digits per decision.** Measured: K≈2 beats K=5 by
  6 pp (30 pp on hard cases) at matched descriptions and order. Deployed: Think3D 8,
  PIVOT 10, Explore 3. The two-level structure is what makes a small leaf menu possible.
- **Keep the space *interface-invariant* across shapes even when the geometry differs.**
  The two patterns that achieve this are (a) a fixed enumerated menu with stable label
  semantics regardless of shape, and (b) egocentric deltas from a named anchor view. Both
  keep the prompt shape constant while the geometry varies underneath — which is what
  "consistent across different object shapes" needs to mean operationally.
- **Always include an explicit terminal action** (MapGPT's hard-coded `"A. stop"`) and an
  **anti-repeat instruction** (Think3D's "Do not request the same angle as before").
  Both are free.
- **The similarity trap is unresolved and must be tested, not assumed either way.** The
  round-2 worry that 20 near-identical geometry-conditioned views could be *harder* than
  36 uniform cells has no evidence for or against it in this branch. If it turns out true,
  the RQ4 framing needs restating — and that is a benchmark question, not a literature
  question.

### RQ4 — Presenting to the VLM

- **The three-family framing does not survive contact with the evidence.** No family wins;
  what wins is a specific combination that cuts across them: **geometry proposes a small
  labelled set; the model selects a label; if the model is shown anything visual, that
  visual is explicitly bound by identity to what the text says** (SeeGround's FAM is where
  the gain lives: +3.6–3.8 pp, vs ~2 pp for adding raw visual channels unbound).
- **The blind orchestrator is not refuted.** MapGPT's blind arm beats its seeing arm on
  oracle success; SayPlan reaches 86.6% executability with zero vision via symbolic
  feedback; SiT-Bench shows vision-*pretrained* backbones do best on text-only items. What
  blind costs is **decisiveness and efficiency**, measured (−3.2 SR, −5.8 SPL), and it
  cannot substitute deliberation for observation (TwNV: text self-reflection never beats
  not reflecting).
- **The seeing orchestrator is not free.** It requires clearing three thresholds none of
  which we have measured for our imagery: the model must read marks at all (SoM:
  categorical failure in weaker models), the render must clear a fidelity floor (TwNV:
  −1.8 pp below it), and marks must be referentially legible on our geometry (PIVOT:
  0–44% on hard shapes).
- **The lowest-risk architecture supported by what was read:** keep the orchestrator blind
  for *navigating the action space* — which is symbolic, small, precomputed and
  IK-filtered — and give sight only to a **scoped, identity-bound judgement step** (is
  this evidence view acceptable / what does it show). This is SeeGround's blind-decides-
  where / seeing-decides-what split, which is the pattern our current architecture already
  approximates. It is supported as a *pattern*, at the scale of one anchor and one render;
  it is **not** demonstrated to scale to navigating a structured multi-option space.
- **A fourth architecture deserves explicit consideration and was not in the round-2
  framing:** MV-GEL removes the prompted model from selection entirely — a geometric or
  trained scorer picks, the VLM only consumes the winner and answers the inspection
  question. This is structurally identical to our own analytical/optimization track. Given
  §Q15 (no prompted VLM has been shown to beat random view selection), this is arguably
  the *default* against which the VLM-selects designs must justify themselves.
- **Encoding rules, with the evidence attached:** select a label, never emit a value (SoM
  +60.7) · any explicit address beats none (SCAFFOLD +5.7…+10.3) · coarse discrete beats
  raw continuous (SCAFFOLD, MapGPT) · integers are fine as labels of an enumerated set
  (Think3D) and bad as quantities to reason over · stack redundant cues (SoM +4.8) · point
  markers, not region overlays (SeeGround) · state the reference frame explicitly in every
  prompt (RoboSpatial's frame-implicit collapse to ~0%) · one spatial operation per turn
  (AP-VLM) · carry metric pose as an executor side-channel the model never sees (SayPlan,
  GraspView, MOKA all do this).
- **Model tier is a first-class design parameter, not an implementation detail** (§1.1).
  The same interface that gains +11.6 on GPT-4.1 gains +0.6 on Qwen3-VL-4B and *loses*
  8.3 on one of its subtasks. Pin the tier before choosing the interface, and gate the
  design on an instruction-following pre-check.

### Benchmark methodology (for Branch 2)

**Mandatory arms — any view-selection claim without these is not believable, and this
branch contains multiple published examples of exactly that failure:**

1. **Random / uniform baseline.** The current 12×3 {h,v} sphere is the natural one.
   Report chance-corrected (BoR: `P_rand = 1 − C(N−R,K)/C(N,K)`), per-category, not just
   aggregate. *Precedent for its absence: GraspView, AP-VLM, Explore-until-Confident all
   lack one.*
2. **Blind / no-image baseline** to prove the task actually requires the geometry (MMSI's
   Blind GPT-4o row changes the interpretation of its whole leaderboard).
3. **Geometric-judge baseline** against any VLM judge. Without it the loop is closed and
   the metric is meaningless — this is Branch 2's standing circularity mandate, and
   GraspView is the published archetype of violating it.
4. **Permuted-order condition.** At minimum: canonical geometric order, reversed, two
   random shuffles, plus the adversarial construction (top-2 candidates at first-and-last
   vs adjacent). Treat answer instability across permutations as a first-class failure
   signal, not noise to average away. *Do not report a single accuracy number under one
   canonical order.*
5. **Matched-budget control**: the same number of model calls with **no new view** (TwNV's
   text self-reflection, Think3D's Self-Refine). This is the only way to show a gain comes
   from the view rather than from the extra deliberation.
6. **Model-tier stratification**: at least one frontier-closed and one deployable-open
   model. Nearly every result in this branch changes sign across that boundary.

**Reporting discipline:**

- **Report variance.** ≥3 seeds, ±std. Four of twenty sources do this; it is why almost
  nothing in the field can be compared.
- **Never grade the stated justification** as evidence of spatial understanding — MMSI's
  reasoning accuracy runs 8–10 points below its answer accuracy.
- **Manually verify every scored-wrong answer** before counting it — SoM found benchmark
  ground truth itself was wrong in some failure cases.
- **Use TwNV's three-way error attribution** (wrong view requested / defective render /
  correct view but wrong reasoning), tracked before *and* after each intervention, and
  expect blame to shift between buckets rather than disappear.
- **Report OSR alongside success** (AP-VLM) — "got close but stopped wrong" and "never got
  close" are different bugs.
- **Weight coverage by surface area, not point count** (MV-GEL).
- **Run the mark-legibility protocol before trusting any marked render** (PIVOT App. G:
  isolated-mark classification vs object-referential matching, on *our* renders).

**Named test-case categories to seed the benchmark:** the failure-mode catalogue in
§Q16 — with the four frame/relation modes, the seven structure/aggregation modes, the
eight menu/mark modes, and the five execution/stopping modes each mapped to a concrete
query template over our primitives.

**Two benchmark contributions the field is missing and we are positioned to make cheaply:**
(a) whether VLMs can interpret textual azimuth/elevation descriptions at all — this is
literally our current interface and nobody has measured it; (b) option similarity at fixed
count — near-identical orthogonal views vs the same number of well-separated ones.
