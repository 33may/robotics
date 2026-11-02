# MMSI-Bench: A Benchmark for Multi-Image Spatial Intelligence

- arXiv: 2505.23764 (HTML v3 read in full: https://arxiv.org/html/2505.23764v3)
- Authors: Sihan Yang, Runsen Xu, Yiman Xie, Sizhe Yang, Mo Li, Jingli Lin, Chenming Zhu, Xiaochen Chen, Haodong Duan, Xiangyu Yue, Dahua Lin, Tai Wang, Jiangmiao Pang
- Read method: fetched raw HTML, parsed the LaTeXML-rendered tables directly (Table 3's leaderboard is rendered as nested `<span class="ltx_tabular">` markup, not a real `<table>`, so a naive table-tag scrape misses it — verified by depth-aware span parsing against the raw HTML, cross-checked against a second independent extraction pass; numbers below are the ones that survived verification).

## Core idea

A VQA benchmark that isolates **multi-image spatial reasoning**: every question requires combining information across ≥2 images (except Multi-Step Reasoning, all categories use exactly **2 images per question** — Sec 3.2, "*Except for the Multi-Step Reasoning category, all other categories contain only two images per question.*"). 1,000 four-option MC questions, human-curated (six 3D-vision researchers, 300+ hours, >120,000 candidate images inspected down to 1,990 used, avg 2.55 images/question, max 10 — Table 1/Sec 3.2). No templates — "all questions are created in free form" (Sec 3.2). 37 models evaluated (open-source + proprietary).

**Taxonomy (Table 1, Sec 3.1)** — three fundamental elements (camera / object / semantic region) crossed into 11 task types:
- Positional Relationship: Camera–Camera, Camera–Object, Camera–Region, Object–Object, Object–Region, Region–Region
- Attribute: Measurement, Appearance
- Motion: Camera, Object
- Multi-Step Reasoning (combines the above; the only category with >2 images)

Explicitly excluded by design (Sec 3.1): "*our questions are designed to be human-answerable; thus, we do not include questions about camera attributes (parameters)*" — i.e. no numeric pose/intrinsics ground truth ever appears in a question or is required to answer one. Regions get no motion category (static by definition).

**Data sources** (Sec 3.2, Table 1): Matterport3D, ScanNet (indoor scans), DTU (local multi-view reconstruction), nuScenes, Waymo (driving), AgiBot-World (robot manipulation), DAVIS 2017 (video seg), Ego4D (egocentric video). All real-world imagery, no synthetic/CAD renders, no tabletop/close-range manipulation-camera imagery specifically (AgiBot-World is the closest to robotics but is mobile-manipulation, not eye-in-hand inspection).

## Concrete results (Table 3, verified against raw HTML span parse)

Selected rows, all as % accuracy, columns = [Cam-Cam, Obj-Obj, Reg-Reg, Cam-Obj, Obj-Reg, Cam-Reg, Meas, Appr, Cam-Motion, Obj-Motion, MSR, **Avg**]:

| Model | Cam-Cam | Cam-Obj | Cam-Motion | **Avg** |
|---|---|---|---|---|
| GPT-5 (reasoning) | 43.0 | 48.8 | 32.4 | **41.9** |
| o3 (reasoning) | 45.2 | 44.2 | 31.1 | **41.0** |
| GPT-4.5 | 34.4 | 51.2 | 41.9 | **40.3** |
| Gemini-2.5-Pro | 39.7 | 45.3 | 36.4 | **36.9** |
| Qwen2.5-VL-72B (best open-source) | **25.8** | **23.3** | 27.0 | **30.7** |
| NVILA-15B | 30.1 | 36.0 | 18.9 | 30.5 |
| Blind GPT-4o (no image) | 20.2 | 13.9 | 20.2 | 22.7 |
| Random Guessing | 25.0 | 25.0 | 25.0 | 25.0 |
| Human Level | 95.7 | 94.2 | 98.6 | 97.2 |

**This is the load-bearing number for RQ "capability floor above chance":** the *best open-source model* (Qwen2.5-VL-72B — the class of model we'd actually deploy on-prem, not a closed API) scores **25.8% on Camera-Camera and 23.3% on Camera-Object**, i.e. at or *below* the 25% random-guess baseline for a 4-option MC. Only frontier proprietary reasoning models (GPT-5, o3, GPT-4.5, Gemini-2.5-Pro) clear chance by a real margin (34–51% on these two categories), and even they sit 45–70 points below the human ceiling (94–96%). Blind GPT-4o (text-only, no image) scores near chance too (20.2/13.9%), confirming the questions aren't solvable from language priors alone (Sec 4.2).

Camera-Motion (ego-motion direction) is uniformly hard even for GPT-5/o3 (31–32%, barely above chance), and the paper flags it explicitly: "*most models, particularly open-source ones, perform poorly on camera motion tasks, suggesting that MLLMs, as embodied agents, struggle to understand their own movement*" (Sec 4.2, "Multi-step reasoning and camera motion comprehension are challenging").

**Scaling doesn't fix it** (Sec 4.2): Qwen2.5-VL-72B beats Qwen2.5-VL-32B by only 3 points overall; InternVL3-78B beats InternVL3-1B by only 1.5 points. NVILA-15B outperforms most 70B+ models.

**Benchmark-comparison table (Table 2/App. D):** MMSI-Bench's human-AI gap (55.3%, i.e. 97.2 − 41.9) is the **largest reported among comparable spatial benchmarks** — larger than BLINK (44.4%), VSI-Bench (33%), LEGO-Puzzle (39.5%), SpatialRGPT (<42%), MMIU (<45%). This is the citation basis for the branch-selection note ("hardest capability bound").

## Four-way failure taxonomy (Sec 5.1, Tables 6/7 — exact grader-prompt definitions)

1. **Grounding error** — "failure to accurately detect, locate, or reference objects or elements in the image."
2. **Overlap identification and scene-reconstruction error** — "mistakes in identifying overlaps between elements or errors in reconstructing spatial relationships within the scene," e.g. failing to recognize the same tree across two images as one object.
3. **Situation-transformation reasoning error** — "errors in reasoning about directions or spatial relations from the perspective of an entity other than the observer (**i.e., allocentric transformation errors**)" — includes converting relative (left/right) to absolute (N/S/E/W) directions. **The paper's own grader taxonomy names this failure mode "allocentric transformation error" verbatim** (Table 6/7 prompt text) — direct evidence for the branch's ego/allo framing.
4. **Spatial-logic error** — three subtypes: (a) hallucinating spatial relationships that don't exist, (b) failing transitivity (A east-of B, B east-of C ⇒ wrongly infers A west-of C), (c) picking the wrong reference object in motion reasoning (e.g. using a moving object instead of the ground as the motion reference frame).

Qualitative ranking (Sec 5.2, Fig 7 — figure is a chart image, no OCR'd percentages available from this pass): "*overlap-matching and scene-reconstruction errors account for the largest proportion of mistakes across all models.*" Exact per-model percentages are in Figure 7 as a stacked bar chart — not extractable as text from the HTML build; would need the PNG read visually for exact numbers if a table x model x % breakdown is needed later.

**Reasoning ≠ answer correctness** (Sec 5.2): even when the final MC answer is right, the annotated reasoning chain is often wrong. Reasoning-process accuracy: GPT-4.5 37.5%, GPT-4o 29.9%, Qwen2.5-VL-72B 21.5% — all lower than each model's own answer accuracy (Qwen2.5-VL-72B's reasoning accuracy is ~10 points below its answer accuracy). **Implication for us:** if we ever eval a VLM's stated justification for a view choice, don't trust a correct pick as evidence of correct spatial reasoning — it may be a guess/shortcut.

**Automated error-labeling reliability:** GPT-4o labels the correct error type only 53% of the time given just the correct answer; 78% once given the full human-annotated reasoning trace (Sec 5.2). I.e. the paper's own automated pipeline needs gold reasoning chains to be trustworthy — a caution against building our own auto-graded failure taxonomy without similarly rich annotations.

## Ablations (evidence-discipline table)

| Intervention | Section | Setup | Result | Ablated (Y/N) | Variance reported |
|---|---|---|---|---|---|
| Zero-shot CoT ("let's think step by step") | Sec 4.3, Fig 5, Table 5 | Prepended to prompt, compared vs direct-answer prompt | "modest improvement only for GPT-4o... performance degradation on the other models" | Y (paired same-model comparison) | No (single Fig 5 chart, no error bars in extracted text) |
| Visual prompting (PATS correspondence lines) | Sec 4.3, Fig 5, Table 5 | Concatenated image pairs + PATS-drawn correspondence lines between matched points, vs concatenated images without lines, on 4 representative models | "yields only slight performance improvements for two models, while resulting in performance degradation for the other two" | Y (paired) | No |
| Spatial-reasoning fine-tuning | Table 4 (verified numbers below) | 3 base/fine-tuned pairs: Qwen2.5-VL-3B→Spatial-MLLM-3B, InternVL2.5-8B→InternSpatial, Qwen2.5-VL-32B→RoboBrain2.0-32B | Marginal/mixed: 26.5→27.7 (+1.2), 28.7→26.9 (−1.8), 27.7→28.5 (+0.8) avg accuracy | Y (paired) | No |

Author interpretation of the CoT/visual-prompt null result (Sec 4.3): "*this limited effectiveness of linguistic prompting may be attributed to the models' inherent lack of spatial reasoning capabilities; simply encouraging 'thinking' does not address their fundamental limitations*" and for visual prompting: "*most models still lack the basic spatial intelligence required to benefit from such cues.*" **These are the paper's own hypotheses, not independently re-verified in the paper** — flag as plausible-but-asserted, not proven.

**No set-of-marks, grid-overlay, or coordinate-annotation intervention is tested anywhere in this paper.** The only "visual prompt" tried is PATS correspondence-line drawing for cross-image point matching — a different mechanism (feature-matching aid) from set-of-marks (spatial-choice aid). Do not cite this paper as evidence for or against set-of-marks/annotated-menu prompting specifically.

## Downstream-task correlation (Appendix E.5, Table 11) — most directly relevant transfer evidence

The paper computes Pearson/Spearman correlation between MMSI-Bench accuracy and three EmbodiedBench (Yang et al. 2025) tasks, across a set of frontier models:
- MMSI-Bench → EB-Navigation: Pearson 0.8147, Spearman 0.7333
- MMSI-Bench → EB-Manipulation: Pearson 0.7402, Spearman 0.7500
- MMSI-Bench → EB-Habitat (active perception/planning): Pearson 0.7299, Spearman 0.6167

Authors' claim: "models performing well on MMSI-Bench also tend to achieve strong performance on both downstream embodied control (navigation, manipulation) and active perception... tasks." This is the paper's own transfer-validity evidence, and it's a correlational claim over a small set of frontier models (not ablated/causal) — treat as suggestive, not proof, but it is the concrete link between "static multi-image VQA score" and "acts-in-the-world" performance that justifies taking MMSI-Bench's numbers seriously for our orchestrator.

## Load-bearing assumptions (do not silently inherit)

1. **Every task pair uses exactly 2 images**, both static photographs from a pre-existing dataset — the model never *chooses* a viewpoint or acts; it only compares/reasons over two already-given views. MMSI-Bench measures the "relate two views" capability our orchestrator would need *after* candidate views exist, not the exploration/selection loop itself.
2. **Free-form natural-language questions**, no structured schema, no image indices/marks/coordinate systems presented to the model as an addressing scheme (explicitly: no camera parameters, ever). This means MMSI-Bench gives us zero direct evidence on *encoding* choices (index vs. name vs. numeric angle) for a view menu — it only tells us models are bad at the underlying relational reasoning regardless of encoding, since encoding was never varied.
3. Scenes are **real-world, room/street/driving/robot-workspace scale** (indoor rooms, streets, DTU multi-view objects, tabletop-scale AgiBot-World clips) — no glossy/textureless/defective industrial part scans, no dense point-cloud renders. Sparsity/hole/colourless-geometry failure modes (relevant to our fused PLY) are outside this paper's scope entirely.
4. 4-option MC with a defined 25% chance floor is what makes the "below random" finding for Qwen2.5-VL-72B legible — this framing (chance floor as a bright line) is directly reusable for our own benchmark design.

## What transfers to our setting

- **The chance-floor framing itself**: define a random/uniform baseline for our {h,v} sphere or face-menu, and treat "does the deployable model beat it" as the first gate before any view-selection claim is trusted (this directly answers the branch's "what random baseline" question — MMSI-Bench's own Random-Guessing row is the template: fixed-probability baseline computed from option count, reported per-category, not just overall).
- **The allocentric-transformation-error category is empirical evidence, not just terminology**, that models confuse egocentric and allocentric frames when converting between viewpoints — directly supports treating "orthogonal to face 3" as an allocentric-risk query that should be restated in the camera's current (egocentric) frame wherever possible.
- **The reasoning-accuracy-vs-answer-accuracy gap** (Sec 5.2) is a methodological warning for any eval we build on top of a VLM orchestrator: don't grade the stated justification as evidence of correct spatial understanding.
- **Downstream correlation with EB-Manipulation (Pearson 0.74)** is the best available (if indirect) evidence that a model's MMSI-Bench-style capability floor predicts its usefulness for something adjacent to our task.
- Camera-Camera task structure (compare exactly two viewpoints, is there a relation) is the closest published analog to "does the model understand two candidate views are on the same object / adjacent faces / opposite faces" — and on that closest analog, deployable-tier open models (Qwen2.5-VL-72B class) are **at or below chance**.

## What does NOT transfer, and why

- No evidence on encoding (index/name/numeric) — never varied. Cannot use this paper to resolve the "which pose encoding" branch question (RQ about Thinking-with-Novel-Views vs MapGPT contradiction) — that needs other papers.
- No evidence on option count / ordering effects for a menu of near-identical items — MMSI-Bench is always exactly 4 options per question, never a variable-size menu, so it can't speak to the "how many options, how similar" branch question.
- No evidence on point-cloud/PLY-derived renders, sparse/defective geometry, or glossy-surface dropout — all images are ordinary photos from established datasets.
- No evidence on set-of-marks/visual-menu prompting for action *selection* (only tested visual prompting for cross-image *correspondence*, a different task).
- Camera-Region and other categories mix in semantic-region judgments (e.g., "kitchen") that have no analog in our tabletop/primitive-shape setting — don't over-generalize the aggregate "Avg." number, use the Cam-Cam/Cam-Obj/Cam-Motion rows specifically.
- The correlation-with-embodied-tasks result (App E.5) is correlational across a handful of frontier models, not a controlled causal ablation — flag as PLAUSIBLE evidence, not CONFIRMED.

## Answers to the two branch questions this document was selected for

**Q: "Capability floor... is relating one viewpoint to another above chance for the models we would actually deploy?"**
No, for open-source/deployable-class models. Qwen2.5-VL-72B (best open-source, the realistic on-prem deployment tier) scores 25.8% on Camera-Camera and 23.3% on Camera-Object — at/below the 25% random-guess baseline (Table 3). Only frontier closed reasoning models (GPT-5 43.0/48.8%, o3 45.2/44.2%) clear chance meaningfully, and they remain 45-70 points below human (94-96%). This is direct support for the branch's conclusion that "the action space must be flat and independently addressed" if the deployed model is open-source-class — the topology/adjacency ambition in RQ3 has no measured floor to stand on for that model tier. (If the project's orchestrator will run on a frontier closed API, the picture is less bleak but still far from human-reliable — 32-51% depending on category, not remotely saturating.)

**Q: Is "orthogonal to face 3" an allocentric query, and what's the measured ego/allo cost?**
The paper's own grader taxonomy defines category 3 explicitly as "allocentric transformation errors" (Table 6/7) — reasoning about directions from a perspective other than the observer's, or converting relative (left/right) to absolute frames. This is direct terminological and empirical confirmation that viewpoint-relative geometric queries (an object-centric "face 3" label needing conversion to the camera's current egocentric frame) are exactly the class MMSI-Bench shows models fail at. No isolated ego-vs-allo accuracy delta is reported in this paper (that split would need ViewSpatial/SpinBench specifically) — MMSI-Bench only tells us the failure mode exists and is qualitatively frequent (grouped under one of four named categories, not the single largest per Fig 7's finding that overlap/reconstruction errors dominate) — the allocentric-transformation category is present but not shown to be the plurality failure mode. Don't overstate: cite the *existence and definition* of the allocentric-error category from MMSI-Bench, get the *quantified* ego/allo delta from ViewSpatial/SpinBench.

## Handoff notes for Branch 2 / Branch 5 (per orchestrating brief)

- BENCH-tag: MMSI-Bench Camera-Camera / Camera-Object / Camera-Motion subsets are directly reusable as difficulty-calibrated test cases for a "relate two views" capability check in our own benchmark.
- Named failure modes to carry forward (Sec 5.1): grounding error, overlap-matching/scene-reconstruction error, allocentric/situation-transformation error, spatial-logic error (hallucination / transitivity failure / wrong reference-object in motion).
- Random-baseline design pattern to carry forward: report accuracy per category against a category-level (not just aggregate) chance floor, and include a blind (no-image) LLM baseline to confirm the task isn't solvable from priors — both are cheap and this paper shows they materially change interpretation (Blind GPT-4o ≈ random, confirming the benchmark requires real visual-spatial grounding).
- No adjacency/topology structure is present in MMSI-Bench's action space at all (it's pure pairwise VQA, not action selection) — it cannot inform RQ3's tractability question directly; it only bounds the *reasoning* capability that would need to underlie any such structure.
