# Lei, Yang, Yang, Xu, Wang, Wang, Duan, Chen — "SCAFFOLD: Scaffolding Coordinates to Promote Vision-Language Coordination in Large Multi-Modal Models" — arXiv:2402.12058v1 (19 Feb 2024)

Source: full HTML (`arxiv.org/html/2402.12058v1`) fetched and parsed section-by-section, cross-checked against WebFetch summarization. All numbers below transcribed directly from Table 1–5 / Fig. 5–8 text in the extracted HTML, not from the summarizer's paraphrase alone.

## Core method / idea

**Scaffold prompting** overlays a uniformly-spaced rectangular dot matrix on an input image; each dot is labeled with a multi-dimensional Cartesian coordinate, and the model is instructed (in text) to reference the nearest coordinate whenever it names an object, before reasoning about spatial relations numerically. The claim is that this closes a "semantic granularity mismatch" between vision and language: words are discrete, image regions are not, and the dot-matrix gives the model a shared discrete addressing scheme that exists simultaneously in the pixels and in the text.

**Exact construction (§3.1):**
- Matrix size: 6×6 "for general vision-language tasks" — chosen from an ablation sweep of 3×3 to 7×7 (Fig. 5), not a theoretical optimum; the paper is explicit that 6×6 is a one-size-fits-all compromise, not best on every subset (see Limitations, below).
- Density: uniform across the image ("without prior visual knowledge... providing LMMs equal assistance when reasoning across different regions").
- Color: each dot is colored black or white, whichever contrasts more against its local background pixel — the "Binary" strategy, chosen because "coordinates are designed to be recognizable by LMMs using their OCR capabilities."
- Coordinate system, single image: 2D `(x, y)` on an `h × w` matrix; **x ascends top-to-bottom within a column, y ascends left-to-right within a row** — note this maps x to the vertical (row) axis and y to the horizontal (column) axis, the reverse of typical image-array `(row, col)` naming, but internally consistent as long as the accompanying text guideline states the convention (verbatim guideline text is reproduced in Appendix A.1.1 and quoted below).
- Coordinate system, image sequences: 3D `(t, x, y)`, where `t` is constant within one frame and increases across the sequence — this is the only accommodation for multiple images/viewpoints anywhere in the paper (see "What does NOT transfer").
- Both the original and the dot-overlaid image are given to the model side by side in the single-image setting, "to preserve original information."
- Exact system/task-guideline text (single-image setting, Appendix A.1.1, verbatim): *"I will provide you with two images of the same scene. The second image is overlaid with a dot matrix of the shape of HEIGHT * WIDTH... each dot is labeled with two-dimensional coordinates (x,y). 1. When you mention any key objects in the image, first output their nearest coordinates then identify them. 2. You use the coordinates to determine the spatial relationships of the objects... 3. You can search and reason region by region with the help of the dots. 4. Finally, conclude your answer in format [[ANSWER]]."*

**Alternative encodings explicitly considered and rejected in the design (§3.1, prose, not ablated at this point in the text — ablated later in §5, see below):** absolute pixel coordinates ("consume excessive space and are complicated to perceive"), one-dimensional Cartesian coordinates, and alphabetic coordinates — all "fall short in providing adequate positional information" relative to 2D Cartesian.

**Active-perception extension (§6.1, Fig. 8):** a two-phase procedure — (1) ask the model to identify key objects and output the nearest coordinate(s) in the format `[[(x,y)]]` without yet answering the question, (2) crop the image around the returned coordinate(s) and re-query the model on the crop. This is the paper's only "zoom"/multi-step mechanism; it is not iterative refinement of a candidate set (contrast PIVOT, arXiv:2402.07872, read separately) — it is a single fixed two-step localize-then-inspect pipeline.

## Concrete results and numbers

**Table 1 — full results, GPT-4V, temperature 0, 11 benchmarks, 1852 samples total** (Naive / CoT / Scaffold, accuracy or task-specific metric):

| Category | Benchmark | n | Metric | Naive | CoT | Scaffold | Δ vs CoT |
|---|---|---|---|---|---|---|---|
| Spatial Reasoning | MME (Position) | 60 | Accuracy | 51.7 | 51.7 | 75.0 | **+23.3** |
| Spatial Reasoning | VSR | 200 | Accuracy | 67.8 | 70.4 | 74.4 | +6.6 |
| Spatial Reasoning | EgoThink (Spatial) | 50 | LLM-as-judge | 66.0 | 74.0 | 76.0 | +10.0 |
| Compositional | Winoground | 100 | Group score | 17.0 | 33.0 | 33.0 | +0.0 |
| Compositional | WHOOPS! VQA | 200 | BEM score | 58.6 | 57.6 | 62.7 | +5.1 |
| Compositional | CLEVR | 200 | LLM-as-judge | 43.5 | 43.0 | 48.0 | +5.0 |
| Fine-Grained | V* Bench | 238 | Accuracy | 27.2 | 30.8 | 44.6 | +13.8 |
| Fine-Grained | Spotting Differences | 100 | Accuracy | 13.0 | 14.0 | 19.0 | +5.0 |
| Hallucination | POPE (Adversarial) | 100 | Accuracy | 79.0 | 80.0 | 86.0 | +6.0 |
| Hallucination | HallusionBench (Hard) | 504 | Accuracy | 45.6 | 48.8 | 53.0 | +4.2 |
| Hallucination | Mementos | 100 | LLM-as-judge | 33.3 | 33.5 | 36.1 | +2.6 |
| **Overall** | All | 1852 | Average | **45.7** | **48.8** | **55.3** | **+6.5 (paper reports "+9.6" vs Naive, i.e. 55.3−45.7)** |

Category-level averages stated in prose (§4.3): Spatial Reasoning **+13.3** (avg of the three Δ-vs-CoT above: (23.3+6.6+10.0)/3 = 13.3 — this is the number our branch selected the paper for), Compositional Reasoning +8.2 (their stated number — the per-row deltas I compute above sum differently under Δ-vs-CoT if using +16.0 for Winoground on the original Naive-scale reporting the paper itself uses in-table; treat the category averages as the paper's own arithmetic, not independently re-derived here, since the in-table per-row "+X" annotations mix Δ-vs-Naive and Δ-vs-CoT bases inconsistently across the transcription — see caveat below), Fine-Grained +11.7, Hallucination +5.7.

**Caveat on arithmetic (important for evidence discipline):** the in-table "(+X.X)" annotations printed next to each Scaffold score in the original Table 1 are Δ **vs Naive** in some rows and read as Δ **vs CoT** in the prose category summaries — e.g. Winoground shows "(+16.0)" in-table (33.0 − 17.0 = 16.0, i.e. vs Naive) but the category prose average of +8.2 for Compositional only reconciles if Winoground's contribution is 0.0 (33.0 − 33.0, vs CoT). I did not find an explicit statement of which basis the headline "+13.3" spatial-reasoning average uses; recomputing from the three spatial rows using Δ-vs-CoT gives exactly 13.3 ((23.3+6.6+10.0)/3), which matches, so **the +13.3 figure is confirmed as Δ-vs-CoT**, not Δ-vs-Naive. Flagging this because our branch was selected specifically for this number — it is reproducible from the table but the paper's own table annotations are not internally consistent about which baseline each parenthetical uses, so any downstream citation should use "+13.3 over CoT" explicitly, not just "+13.3."

**Ablation — Table 2, coloring strategy** (150-sample subset: 50 each from Winoground/VSR/POPE-Adversarial, GPT-4V, each run twice and averaged):

| Strategy | WNG | VSR | POPE | Overall |
|---|---|---|---|---|
| None (baseline) | 70.0 | 64.0 | 73.0 | 69.0 |
| White | 78.0 | 72.0 | 78.0 | 76.0 |
| Black | 79.0 | 73.0 | 82.0 | 78.0 |
| Complementary | 77.0 | 71.0 | 81.0 | 76.3 |
| **Binary (ours)** | 81.0 | 73.0 | 84.0 | **79.3** |

**Ablation — Table 3, coordinate format** (same 150-sample subset):

| Format | WNG | VSR | POPE | Overall |
|---|---|---|---|---|
| None (baseline) | 70.0 | 64.0 | 73.0 | 69.0 |
| Alphabet | 80.0 | 72.0 | 79.0 | 77.0 |
| Pixel (absolute) | 78.0 | 75.0 | 77.0 | 76.7 |
| One-Dimensional | 72.0 | 71.0 | 81.0 | 74.7 |
| **Cartesian (ours)** | 81.0 | 73.0 | 84.0 | **79.3** |

Note every format beats the no-coordinate baseline; the gap between the best (Cartesian, 79.3) and worst-of-the-labeled-options (One-Dimensional, 74.7) is 4.6 points — the choice of *which* textual coordinate scheme matters less than *whether one exists at all* (baseline-to-any-scheme gap is 5.7–10.3 points, scheme-to-scheme gap among labeled options is ≤4.6 points).

**Ablation — matrix size (Fig. 5, heatmap, 3×3 to 7×7, same subset):** 6×6 reported as optimal for the ablation set; the paper notes larger-width/shorter-height matrices (upper-right of the heatmap) tend to perform better, speculated to align with the sampled images typically being wider than tall. Explicitly *not* optimal for every one of the three individual subsets (WNG/VSR/POPE) — stated as an open problem (see Limitations).

**Ablation — dot perturbation (§5.5, Eq. 1, Fig. 7):** Gaussian noise added independently to each dot's x/y pixel position, `σ = (1/4) × inter-dot-spacing` on each axis. Perturbed-coordinate performance "retains the enhancements provided by standard coordinates" and on the VSR subset *exceeds* the unperturbed version — read by the authors as evidence of robustness to imprecise dot placement, and as a hint that dot placement could potentially be further optimized rather than kept uniform.

**Active perception integration — Table 4** (V* Bench `direct_attributes` subset, Not-Found-Rate ↓ / Success-Rate ↑):

| Method | N.F.R. (%) | S.R. (%) |
|---|---|---|
| Naive | 72.2 | 21.7 |
| CoT | 71.3 | 21.7 |
| Scaffold | 26.2 | 31.3 |
| **Scaffold + Active Perception** | **14.8** | **45.2** |

The CoT→Scaffold jump is attributed to coordinates reducing not-found/give-up behavior during search; the Scaffold→Scaffold+AP jump (+14.1pp success rate) is attributed to the crop step letting the model actually resolve fine attributes once localized.

**Compatibility with CoT — Table 5:** Winoground Naive/CoT/Scaffold/Scaffold+CoT = 17.0/33.0/33.0/**41.0**; V* Bench = 27.2/30.8/44.6/**47.9**. Stacking is additive, not redundant, on both tasks tested.

## Benchmarks used — what each actually measures (relevant for evidence discipline against our branch's capability-floor questions)

- **MME (Position split, n=60):** yes/no questions about 2D object-object spatial relations in a single photo, e.g. "Is the cricket bat above the batter's body?" — pure image-plane left/right/above/below judgment between two named, already-visible objects in one photo.
- **VSR (n=200 of full set):** true/false judgment of a spatial-relation sentence against one photo, 66 relation types, e.g. "The person is facing the banana." — again single image, two objects, no camera/viewpoint concept.
- **EgoThink (Spatial split, n=50):** first-person questions like "Where is the phone, on my left or on my right?" graded by GPT-4 as judge. This is the *closest* of the eleven benchmarks to an egocentric/camera-frame framing our branch cares about — but it is still a single static photo with the camera as an implicit fixed observer, not a task requiring the model to relate **two different camera poses/viewpoints** to each other. There is no benchmark item anywhere in this paper's suite that asks the model to compare view A to view B, judge relative rotation, or reason about "opposite face" / adjacency between viewpoints.
- Compositional (Winoground, WHOOPS!, CLEVR), fine-grained (V* Bench, Spotting Differences), hallucination (POPE, HallusionBench, Mementos): all single-image (or, for the "double-image"/"sequence" coordinate variant described in §3.1/Appendix A.1.1, multiple images treated as an ordered *sequence* for temporal tasks like Mementos — not as multiple *viewpoints of the same scene* to be geometrically related). No benchmark item requires reasoning about camera-to-camera pose relationships.

## Author-stated limitations (verbatim, §"Limitations")

1. **Fixed matrix size is not adaptive:** "our ablation study in Section 5 suggests that a one-size-fits-all matrix size can yield good, but not the best results across all datasets... Addressing the dynamic and automatic adjustment of these attributes to better suit different scenarios remains an area for future exploration." — unsolved by the paper.
2. **Localization/grounding imprecision persists:** "particularly in complex or clustered scenes, GPT-4V occasionally struggles to accurately associate textual reasoning with the nearest coordinates... underscores the need for LMMs to achieve improved visual localization and grounding capabilities in intricate environments." — the coordinate scaffold reduces but does not eliminate the underlying grounding failure mode; the authors frame this as a still-open capability gap in the base model, not something their prompting scheme fixes.

## Load-bearing assumptions

- **Single flat 2D RGB (or synthetic-but-2D-rendered, e.g. CLEVR) photographic image as the sole visual substrate** — the dot matrix is drawn in image (pixel) space, and its legibility depends on the underlying image having continuous, fully-populated, reasonably textured pixel content for the dots to contrast against (§3.1's whole justification for binary black/white coloring is "contrast against the background," which presupposes a background exists everywhere). Nothing in the method or its evaluation touches point clouds, sparse geometry, holes, or colorless/textureless regions.
- **The coordinate scaffold addresses locations *within one image plane*, not camera poses or viewpoints.** Even the 3D `(t,x,y)` extension for sequences uses `t` purely as a frame index for temporal ordering (used for Mementos-style "reason over an image sequence" tasks), not as an encoding of *where the camera was* for each frame. There is no equation, ablation, or benchmark item anywhere in the paper that encodes or evaluates relative camera pose, viewpoint adjacency, or orientation.
- **Model: GPT-4V exclusively**, zero-shot, temperature 0, evaluated under "limited GPT-4V access quota" (stated repeatedly as the reason for subsampling benchmarks to 50–500 items each). No second model family, no open-weight VLM, no replication — a single-model, single-vendor, ~2.5-year-old (Nov 2023 GPT-4V vintage) result set with no stated variance/seed information beyond the coloring/coordinate ablations being run twice and averaged (§5.1). The main Table 1 numbers are apparently single runs (no repeat/variance reported there), only the 150-sample ablation subset (Tables 2–3) states "we run each experiment twice and report average accuracy."
- **OCR-dependent labeling:** the entire coordinate-legibility mechanism assumes the base model's OCR capability can read small numeric labels rendered into the image at 6×6 = 36 dot density — this is asserted, not measured as a separate capability (no OCR-accuracy-conditioned breakdown is reported).

## What transfers to our setting

- **General design principle, useful for RQ4/RQ5 (text vs image encoding, does a seeing orchestrator benefit from drawn addressing):** when a VLM must reference a specific location in an image it is looking at, drawing a discrete, textually-nameable address (a coordinate) directly into the pixels and asking the model to cite it before reasoning measurably improves grounding accuracy over both a no-scaffold baseline and over plain CoT, on tasks that are fundamentally about connecting a textual reference to a place in a **single image**. This is directly ablated (Tables 2–3: any coordinate scheme beats none, by 5.7–10.3 points on the 150-sample subset) — genuine, checkable evidence of the *type* our branch's evidence-discipline question wants, not merely a demonstration.
- **Coordinate-format ablation (Table 3) is a small, real data point for the "MapGPT says precise coordinates confuse, Thinking-with-Novel-Views says numeric pose helps" contradiction (branch Q4):** here, numeric Cartesian coordinates (79.3%) beat absolute pixel coordinates (76.7%) beat alphabetic (77.0%, close) beat one-dimensional numeric (74.7%). The ordering suggests it is not "numeric vs symbolic" that matters but **coordinate granularity/dimensionality relative to the addressed space** — 2D discretized-grid coordinates outperform both raw continuous pixel values (too fine-grained/precise, "complicated to perceive") and 1D-collapsed coordinates (too coarse/ambiguous, "fall short in providing adequate positional information"). This is weak, indirect evidence, not a resolution of the MapGPT/Thinking-with-Novel-Views contradiction, since neither compared scheme here is a camera-pose encoding — but it does support the general shape of an answer: coarse discrete labels tied to a small enumerated grid may beat both raw continuous coordinates and maximally-collapsed 1D indices for the same reason in a camera-pose menu.
- **Dot-perturbation robustness (§5.5) is mild positive evidence that exact marker placement precision is not the bottleneck** — if we mark candidate viewpoints or object faces on a render, this suggests the system may tolerate imprecise mark placement (e.g. from a defective/offset point cloud) without losing the addressing benefit, *for the single-image-grounding task type this was tested on*. Should not be over-generalized to camera-pose selection without direct evidence.
- **The active-perception two-step pattern (localize coordinate → crop → re-query, Table 4) is a reusable interaction shape** — "coarse localize, then request a closer/different look" — loosely analogous to a coarse-select-then-refine view-planning loop, though it operates entirely within one static image (cropping), not by choosing among physically different camera poses.

## What does NOT transfer, and why

- **Cannot serve as capability-floor evidence for branch Q1 (relating one viewpoint to another above chance) or Q2 (ego/allo, "orthogonal to face").** None of SCAFFOLD's 11 benchmarks contain a single item requiring the model to relate two camera poses, judge relative rotation, identify an "opposite" or "adjacent" viewpoint, or reason about surface orientation relative to a camera. EgoThink (Spatial) is the nearest analog and is still single-photo left/right-of-me judgment with an implicit fixed camera, not cross-view reasoning. Citing SCAFFOLD's spatial-reasoning gain (+13.3) as evidence that a drawn-coordinate scheme would similarly rescue camera-to-camera or viewpoint-relation queries (MMSI-Bench, ViewSpatial, SpinBench-style tasks) would be an unsupported extrapolation — flag per the evidence-discipline branch question: **encoding claim is ablated, but only for within-image 2D object-relation grounding, not for the viewpoint-relation task family our branch actually needs.**
- **No 3D, point-cloud, or defective-geometry content anywhere.** The paper's entire method and evaluation assume a complete, well-textured 2D image. It offers zero evidence — positive or negative — about whether a dot-matrix-style coordinate overlay survives being drawn onto a rendered, partial, holed, glossy-dropout point cloud (our actual render source, per the framing doc's ~4cm fused-vs-single-view offset). This should be handed to the GraspView/SPAZER/VLM3D/SeqVLM sparsity question (branch Q7) as a gap, not as supporting evidence either way.
- **No discrete action-menu / tool-selection framing.** SCAFFOLD's coordinates are a dense, continuous-ish grid (36 addressable cells uniformly tiling one image) used for grounding references *during* reasoning, not a small enumerated menu of qualitatively distinct, semantically named options (e.g., "face 1" vs "face 2" vs "orbit position k") that the model must choose *between* as an action. It says nothing about option count, option similarity, order sensitivity, or infeasible-option handling — the branch's tool-count/ordering/feasibility questions (Q9, Q10, Q11) get no evidence from this paper.
- **Chicken-and-egg / render-viewpoint-choice problem (branch Q8) is entirely out of scope** — SCAFFOLD always receives one already-given image; it never chooses or influences which camera pose produced that image.
- **Single-model (GPT-4V, Nov-2023 vintage), no cross-model replication.** Nothing here licenses a claim about how a 2024–2026-era open-weight VLM (Qwen2-VL, InternVL, etc. — the models our branch would actually deploy) would perform with or without a coordinate scaffold; treat all numeric deltas as GPT-4V-specific.
- **The headline "+9.6 overall / +13.3 spatial" numbers, read in isolation, overstate what was actually shown** for our purposes: the 11 benchmarks skew toward tasks that are near-trivial 2D left/right/above judgments between two already-salient objects in ordinary photos (MME Position, VSR) — categorically easier and structurally different from the viewpoint-relation, allocentric-rotation, and cross-view-correspondence tasks (MMSI-Bench, ViewSpatial, SpinBench, FlipSet) that define our branch's actual capability floor. Any downstream citation of "+13.3" without this scope caveat would misrepresent the evidence.

## Evidence-discipline table entry (for Branch-2 handoff)

| Paper | Encoding compared against what | Δ | Variance reported? | Task type | Usable for our RQs? |
|---|---|---|---|---|---|
| SCAFFOLD (2402.12058) | Dot-matrix + Cartesian coords vs. no-scaffold baseline (Table 3) | +10.3 pts (79.3 vs 69.0), 150-sample subset | Yes — each ablation run twice, averaged (main Table 1 results are single-run, no variance reported) | Within-image object-relation grounding (left/right/above, existence, attribute ID) — **not** camera-pose/viewpoint relation | Weak/indirect — supports "draw the address into the pixels" as a general principle for RQ4/RQ5, but the benchmark suite never tests viewpoint relation, so cannot answer Q1/Q2 capability-floor questions directly. Flag as scope-limited when cited. |
| SCAFFOLD — coordinate format (Table 3) | Cartesian vs. Alphabet vs. Pixel-absolute vs. 1D vs. None | 79.3 / 77.0 / 76.7 / 74.7 / 69.0 | Yes (2 runs averaged) | Same as above | Weak indirect data point for Q4 (numeric vs symbolic pose encoding contradiction) — same caveat, not a pose encoding. |
| SCAFFOLD — coloring strategy (Table 2) | Binary vs Black vs White vs Complementary vs None | 79.3 / 78.0 / 76.0 / 76.3 / 69.0 | Yes (2 runs averaged) | Same as above | Not directly relevant to any branch RQ; marker-color legibility detail only. |

## Checkable citations

| Claim | Section | Table/Fig/Eq |
|---|---|---|
| Method description, dot matrix + coordinates | §3.1 | — |
| Guideline prompt text (single-image) | Appendix A.1.1 | — |
| Main results, all 11 benchmarks | §4.3 | Table 1 |
| "+13.3" spatial reasoning average | §4.3, para "Spatial Reasoning:" | derived from Table 1 rows 1–3, Δ-vs-CoT |
| Coloring ablation | §5.3 | Table 2 |
| Coordinate format ablation | §5.4 | Table 3 |
| Matrix size ablation | §5.2 | Fig. 5 |
| Dot perturbation robustness | §5.5, Eq. (1) | Fig. 7 |
| Active perception integration | §6.1 | Table 4, Fig. 8 |
| Scaffold + CoT compatibility | §6.2 | Table 5 |
| Limitation 1 (fixed matrix size) | "Limitations" (post-§7) | — |
| Limitation 2 (grounding imprecision in cluttered scenes) | "Limitations" (post-§7) | — |
| EgoThink (Spatial) benchmark description/example | Appendix A.2.1 | — |
| Related-work framing vs Set-of-Marks (Yang et al. 2023a) | §2, "Multi-Modal Prompting Methods" | — |

## Handoff notes

- **To Branch 2 (benchmarking):** the coordinate-format and coloring ablations (Tables 2–3) are clean, checkable A/B/C/D/E comparisons on a fixed 150-sample subset with repeated runs — a usable template for how to structure our own marked-render ablation (compare no-mark / numbered / coordinate-grid / named-index conditions on the same view-selection task, run each condition ≥2×). Add to the evidence-discipline table above.
- **To Branch 5 (structuring the viewpoint space):** SCAFFOLD offers no evidence for or against a view-adjacency graph — it never addresses multiple viewpoints of the same scene, only locations within one fixed image. No tractability pressure flows from this paper either direction.
- **To Branch 3 (primitive structure from survey clouds):** no relevant evidence — SCAFFOLD never operates on point clouds or non-photographic renders.
- **Caution flag for anyone citing "+13.3" or "+9.6" elsewhere in the project:** both numbers are GPT-4V-specific, single-model, and drawn from tasks that are structurally easier (2D within-image object-relation judgments) than the viewpoint-relation tasks our branch's capability-floor questions (Q1, Q2) are actually about. Do not use these numbers as evidence that a coordinate/marker scaffold would close the gap on MMSI-Bench/ViewSpatial/SpinBench-style camera-relation queries — that claim is untested here.
