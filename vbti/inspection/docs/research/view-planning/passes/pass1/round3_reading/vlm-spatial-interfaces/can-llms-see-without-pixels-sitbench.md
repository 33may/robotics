# Can LLMs See Without Pixels? Benchmarking Spatial Intelligence from Textual Descriptions (SiT-Bench)

**arXiv:2601.03590v1** (Guo, Yang, Li, Zhang, Gao, Wang, Li, Liu, Jian — Beijing Institute of Technology / BUCT). Submitted 07 Jan 2026.
Full text: https://arxiv.org/html/2601.03590v1 (fetched directly, all sections + both appendices read in full — this is a complete read, not an abstract skim).

## 0. One-line identity

Not a viewpoint-selection or embodied-agent paper — it's a **capability-isolation benchmark**: strips pixels out entirely and asks whether an LLM's spatial reasoning survives on VLM-generated, human-verified textual scene descriptions alone. Directly measures the blind-orchestrator hypothesis's ceiling: if the model can't relate viewpoints to each other from rich text, no caption format will save a blind orchestrator.

## 1. Core method

- **Task taxonomy** (§2.1): 5 categories, 17 subtasks, 3,892 items. Categories: Global Perception & Mapping (601, 15.4%), Navigation & Planning (900, 23.1%), Multi-View & Geometric Reasoning (836, 21.5% — "core module... rigorous 3D geometric modeling and coordinate transformations"), Embodied & Fine-grained Perception (1,105, 28.4%), Logic & Anomaly Detection (450, 11.6%). Full subtask→source→count→rationale table is Table 4 (§A.2).
- **Two-path construction** (§2.2, Fig. 2): Path A generates QA from scratch (robotic manipulation, floor plans, GTA-V game scenes, sim) via GPT-4o spatial-complexity filtering then VLM-authored captions+QA. Path B *adapts existing vision benchmarks* — CoSpace, SpinBench (not ViewSpatial-Bench despite it being named in §2.2 prose; the actual Appendix A.1 source list has no ViewSpatial-Bench entry — see §6 caveats below), Ego3D-Bench, Mindcube, ResPlan, Play4Data (GTA-V), POEM-v2, LEGO-puzzles, WM-ABench — by captioning the images and keeping only sub-tasks solvable from text alone, discarding ones needing absolute metric estimation.
- **Quality control** (§2.2): DeepSeek-R1 auto-filters for entity-visibility, answer-leakage, spatial-deductibility, ambiguity (full prompt in §A.4); then human experts review with R1's CoT attached. From 34,917 candidate samples down to ~3,900 kept (§A.1).
- **The actual textual encoding is *not* raw coordinates.** Despite "coordinate-aware" in the title, the delivered scene descriptions are natural-language prose with occasional explicit distances (m) / angles (°) embedded — e.g. "positioned 5 meters ahead on your right flank" — plus qualitative direction words (left/right/front/back, cardinal N/S/E/W, "Far Left / Center-Left / Center / Center-Right / Far Right" zones) and mandatory **boundary-anchor naming** ("explicitly name the objects at the Far Left and Far Right edges... this is the only way to align this view with others") used as a stitching mechanism for cross-view entity correspondence. Numeric coordinates only appear as an *internal* scratch step for the item-construction VLM (a "Right-Handed Coordinate System, origin at observer" — §B.3.2 Perspective Shift prompt) and in the *derivation* field of the ground-truth JSON, not in what the eval model reads. Full construction prompts for every subtask are given verbatim in Appendix B (§B.1–B.4).
- **Evaluation protocol** (§3.1): mostly 4-way MCQ scored by exact match; Cognitive Mapping scored as structured JSON match. Models: proprietary (GPT-4o, Gemini-3-Flash-preview, DeepSeek-V3.2 ± thinking), open VLM/LLM pairs (Qwen2.5/3-VL vs Qwen2.5/3 backbone, InternVL3/3.5, LLaVA-1.5, Llama-3.1), and 5 "spatial-specialist" models (Space-Qwen, SpaceThinker, Robobrain2.0, SpaceR, Cosmos-Reason2). Thinking-mode budget = 8,192 tokens, temperature 0.0 (§A.5). Human baseline collected under a documented instruction protocol (§A.7, verbatim).

## 2. Concrete numbers (Table 2, §3.2 — 33-row table, all models; full per-model per-subtask breakdown also in Table 6 §A.8)

**Overall (Avg. column, out of ~3,900 items across all 17 subtasks):**

| Model | Avg. | Rank |
|---|---|---|
| Human Level | 74.42 | 1 |
| Gemini-3-Flash-preview | 59.46 | 2 |
| Qwen3-VL-32B-thinking | 51.06 | 3 |
| Qwen3-VL-32B (no thinking) | 45.90 | 5 |
| GPT-4o | 45.70 | 6 |
| Qwen2.5-VL-72B | 45.45 | 8 |
| DeepSeek-V3.2-thinking | 43.74 | 10 |
| DeepSeek-V3.2 (no thinking) | 37.06 | 22 |
| Random Level | 27.30 | 32 |
| Space-Qwen-3B (worst spatial-specialist) | 27.26 | 33 |

**The three subtasks closest to our branch questions' named benchmarks (View Consistency ≈ "camera-camera relation", Perspective Shift ≈ ego/allo POV switch, Pure Mental Rotation ≈ SpinBench rotation) — best model vs. human vs. random, all 4-way MCQ so random=25.00 for all three:**

| Subtask | Best model | Best score | 2nd best | Human | Random |
|---|---|---|---|---|---|
| **View Consistency** (SpinBench "View-spatial" subset — recognize the *same object across different camera viewpoints*, closest proxy to camera-camera relation) | Gemini-3-Flash | **50.50%** | Qwen3-VL-32B-thinking / InternVL3.5-8B-thinking, both 39.00% | 75.00% | 25.00% |
| **Perspective Shift** (Play4Data/Ego3D — reason from a non-observer POV, or after the *same* observer's rotation) | Qwen3-VL-32B-thinking | **59.54%** (Gemini 72.65% is highest overall — beats human) | Gemini 72.65 | 71.23% | 25.00% |
| **Pure Mental Rotation** (LEGO-puzzles — abstract geometric rotation with semantic interference removed) | Gemini-3-Flash | **45.00%** | Qwen3-VL-32B (no thinking) 39.00%, GPT-4o 37.00% | 68.00% | 25.00% |

Full per-model rows for these three columns are in Table 2/Table 6; e.g. for View Consistency: DeepSeek-V3.2=38.00, DeepSeek-V3.2-thinking=37.00, InternVL3-8B=33.50, Qwen2.5-VL-72B=36.50, Robobrain2.0-7B=33.50, SpaceR-7B=36.50 — the entire model population clusters in a narrow 30–50% band, none escaping toward the 75% human ceiling.

**The single worst subtask in the whole benchmark — Cognitive Mapping** ("reconstruct unstructured textual navigation logs into structured 2D grid layouts or JSON topological maps," §2.1, inspired by VSI-Bench/MindCube): best model (Gemini-thinking) **8.34%**, human **26.77%** (Table 2) — stated in prose (§3.2 "Overall Performance") as "26.44%", a minor internal inconsistency between the table and the running text; random is undefined ("-", not a fixed-cardinality MCQ). Every model in the 33-row table scores under 10% except SpaceR-7B (7.60%) and Gemini's few points above — this is not "worse than human," it is close to non-functional.

**Random baseline row is reported per-subtask** (§3.2 "Random Baseline Comparison", Table 2 "Random Level" row) — useful as our own benchmark's floor calibration reference: most subtasks are 4-way MCQ (25.00%), Outdoor Navigation is 8-way (12.50%), Ego/Objects-motion Perception is binary (50.00%). Overall random average = 27.30%.

**CoT/"thinking" boost** (§3.3 "Emergence of Explicit Spatial Reasoning"): Qwen3-8B 37.91%→45.04% (+7.13 pts); Qwen3-VL-32B 45.90%→51.06% (+5.16 pts); DeepSeek-V3.2 37.06%→43.74%. 32B Qwen3-VL-thinking (51.06%) beats the much larger DeepSeek-V3.2 non-thinking (37.06%) — scaling alone is not the lever, explicit reasoning is (§3.2 "Scaling vs. Reasoning Backbone").

**Latency cost of that boost** (Table 3, §A.3): standard models 0.27–4.94s; thinking-enabled models 2.31–190.24s. Gemini-3-Flash-thinking = 40.11s; DeepSeek-V3.2-thinking = 190.24s (70× its own non-thinking latency). Explicit statement: this "limit[s] deployment in latency-sensitive embodied tasks" (§6 Limitations).

**Visual grounding helps even in text-only eval** (§3.3): Qwen2.5-VL-72B (45.45%) > Qwen2.5-72B backbone (42.57%); Qwen3-VL-8B (42.10%) > Qwen3-8B (37.91%). VLM-trained backbones outperform pure-text-trained LLMs of the same base *on the exact same text-only items* — multimodal training "bakes" spatial priors into language weights.

**Spatial-specialist models underperform generalists** (§3.3): Cosmos-Reason2-8B (42.13%) < Qwen3-VL-8B (45.66%); Robobrain2.0-7B (35.52%), SpaceR-7B (36.42%), Space-Qwen-3B (27.26% — tied with random) all underperform general-purpose VLMs of comparable or smaller scale. Paper's hypothesis: narrow/template-reliant spatial fine-tuning data causes loss of general reasoning flexibility or catastrophic forgetting of the linguistic common sense needed to parse a dense caption.

## 3. Load-bearing assumptions

- **The text is expert-curated and leakage-filtered by construction.** DeepSeek-R1 explicitly screens out any caption that states the answer, that omits a needed entity, or that's logically ambiguous (§2.2, prompt in §A.4). This is a *best-case* textual encoding — richer, more consistent, and more leakage-free than anything an automated view-description pipeline in our system would produce on the fly. The measured "spatial gap" is therefore a **floor on the floor**: our system's captions will likely be noisier, so real-world performance on an equivalent task should be assumed *worse*, not better, than these numbers.
- **All source imagery is ordinary RGB photography or synthetic renders of everyday scenes** (indoor rooms, street scenes, GTA-V, LEGO sets, hand-object interaction) — never sparse/defective point clouds, never industrial parts, never a single object under multiple orthogonal camera angles at close range. Zero overlap with our object-domain or camera-modality.
- **MCQ scoring with 4 options is the near-universal protocol** (only Outdoor Navigation and Ego/Objects-motion Perception deviate, to 8-way and binary respectively). No subtask varies option-count or option-order as an independent variable — this paper cannot answer the "how many options / does order matter" branch questions; it simply wasn't designed to.
- **"Thinking mode" is on/off only, budget fixed at 8,192 tokens** — no ablation of reasoning budget, no permutation testing, no repeated-seed variance reporting anywhere in the paper (confirmed by direct read — no std-dev, no CI, no seed count mentioned in §3, §A.5, or Table 2/6 legends).

## 4. Answers to branch questions this document was selected for

**Q1 — Capability floor: is relating one viewpoint to another above chance for deployable models?**
Above chance, but only barely functional, not reliable. **View Consistency** (SiT-Bench's closest analog to MMSI-Bench camera-camera relation — "recognizing objects from different perspectives within the same scene," §2.1, sourced from SpinBench's View-spatial subset) tops out at Gemini-3-Flash's **50.50%** against a 25% random floor and a 75.00% human ceiling — i.e. best deployable model is ~2× random but ~33 points short of human, and every other model (including all "spatial-specialist" ones) sits in the 30–40% band, statistically closer to random than to human. **Pure Mental Rotation** (closest analog to SpinBench rotation) is worse in absolute terms: 45.00% best vs. 68.00% human. This is a genuine capability floor for the "relate viewpoint A to viewpoint B" primitive, not a chance-level wall — but it is far too weak to trust for anything beyond coarse binary judgments ("have I already seen roughly this face?"), and RQ3's ambition of a rich, LLM-reasoned view-adjacency topology should be built to *not depend* on the model doing this relation implicitly from description alone. Confirmed further by **Cognitive Mapping** — literally "reconstruct navigation logs into a topological map from text" — scoring 8.34% best-of-33-models vs. 26.77% human; this is the single most direct measurement in the literature so far of "can an LLM build the adjacency graph itself," and the answer is close to no, even with the text handed to it pre-digested and leakage-filtered.

**Q4 — Which encoding of a camera pose does the model handle best?**
SiT-Bench doesn't run this as a controlled ablation (see gap list below), but its *own design choice* is informative: the authors explicitly discarded raw numeric coordinates from the delivered text and instead used **qualitative directional language + explicit distance(m)/angle(°) numbers embedded in prose + named boundary anchors for cross-view stitching** (construction prompts, §B.1–B.3, especially the Perspective Shift prompt's Step 2 "Descriptive Prose" instruction, which explicitly forbids raw coordinate output to the model — coordinates are only used *internally* by the captioning VLM and appear in the machine-checkable `derivation` field, never in the prompt the eval model reads). This is a real-world existence proof, from a paper built specifically to stress-test textual spatial reasoning, that **natural-language-with-embedded-numbers beats raw coordinate triples** as the delivered format — consistent with MapGPT's "precise coordinates confuse the model" side of the contradiction this branch was asked to resolve, at least for LLM-consumption of static scene description (not for pose *specification* by the model, which this paper never asks for — models here only select/reason, never emit poses).

**Q5 — Text vs. image, under what conditions?**
The paper's own headline finding is the closest thing to a controlled comparison: **VLM-trained backbones beat pure-text LLM backbones on identical text-only items** (Qwen2.5-VL-72B 45.45% vs. Qwen2.5-72B 42.57%; Qwen3-VL-8B 42.10% vs Qwen3-8B 37.91%, §3.3). This is *not* an image-vs-text-on-the-same-task ablation though — both conditions here are text-only; the variable is whether the *model's training* included vision, not whether the *eval input* did. The paper never reports the same items' accuracy when the *original* image is shown (i.e. it never reruns the models on the Path-B source benchmarks' native image format to get a true paired image-vs-caption number for the same questions). This is a genuine, flaggable gap for RQ5 — treat as "not ablated, only demonstrated" (see evidence table below).

**Q14 — Evidence discipline table (encoding claim: ablated or demonstrated?):**

| Claim | Ablated (controlled, same items, isolated variable)? | Variance reported? |
|---|---|---|
| Coordinate-free prose beats raw coordinates for delivery | No — design choice only, never compared against a raw-coordinate variant within the paper | No |
| VLM-trained backbone > pure-text LLM backbone on same text | Partially — same base model family, vision vs. no-vision pretraining is the isolated variable, both eval conditions are text-only (Table 2 rows exist for both) | No (single run, temp=0.0) |
| Thinking mode boosts spatial reasoning | Yes — same model, thinking on/off, same items | No |
| Text-only vs. image-in eval on the *same* question | **Not tested at all** — no paired condition exists in this paper | N/A |
| Spatial-specialist fine-tuning helps | Ablated by comparison across models, not within one model's pre/post fine-tuning — confounded with base-model choice, dataset, scale | No |

Flag: none of this paper's headline claims carry error bars, seeds, or significance tests — every number in Table 2/6 is a single run at temperature 0.0. Treat every delta under ~5 points as noise-level until independently replicated.

**Q15 — Random baseline:** answered concretely above (§2) — per-subtask random floors of 12.5%/25%/50% depending on option cardinality, aggregate 27.30%. Directly useful as the calibration reference our own benchmark needs: confirms that a 4-way MCQ random floor of 25% is the right default assumption unless our action space has a different option count, and that "above 25%" is a very low bar to clear — SiT-Bench's View Consistency result (50.5% best) shows that even 2× random is still a long way from usable.

**Q16 — Failure modes to hand to Branch 2 (verbatim from the paper):**
- **Arithmetic-spatial hallucination** (§3.3): model correctly identifies which entities/viewpoints overlap but miscalculates the final count or coordinate offset — reasoning trace is qualitatively right, final answer wrong. Concrete example given in Appendix A.6 (Gemini-3-Flash full reasoning traces for two panoramic-counting items) — worth using as literal test-case templates.
- **View-consistency collapse under near-identical distractors**: best model only 50.5% on "is this the same object across two camera views" — directly relevant to our "near-identical candidate views" concern (branch question on option distinctiveness).
- **Cognitive-mapping/topology near-failure**: 8.34% best vs. 26.77% human on reconstructing a structured map from text — hand off as the single strongest argument for why any adjacency/topology exposed to the LLM must be pre-computed, not LLM-derived.
- **Left/right and ego/allo mixing under CoT** (Appendix B.3.1, Qwen3-30B-A3B-thinking trace, verbatim): the model's own reasoning trace visibly tangles itself trying to decide whether "the right wall" is to its right or behind it depending on which way it's facing — a live, quotable example of ego/allo confusion mid-reasoning, even when it reaches the correct final answer by luck.
- **Latency wall on thinking mode**: 40–190s per call for the strongest reasoning configurations (Table 3) — any design that needs multiple LLM-reasoned view judgments per inspection episode should budget for this explicitly.

## 5. What transfers to our setting

- **Prefer a VLM-trained backbone for the orchestrator even if its runtime input stays text-only** — the visual-grounding-helps-even-text-only finding (§3.3) is a concrete, actionable model-selection signal for our blind-orchestrator design.
- **Avoid "spatial-specialist" fine-tuned checkpoints by reputation alone** — SpaceR/Robobrain2.0/Space-Qwen/Cosmos-Reason2 all underperformed general-purpose VLMs of similar or smaller scale in this benchmark; general-purpose + CoT beat domain fine-tuning here.
- **Enable CoT/thinking for any per-view spatial judgment call, but budget the latency** — 5–7 point gains are real and replicate across multiple models, but the cost is 10–70× the non-thinking latency.
- **Design our view-menu text to avoid raw coordinate triples**; favor short natural-language descriptors with the occasional explicit number (distance/angle), following this paper's own construction-prompt convention — it is the field's most recent, most deliberately-tested attempt at exactly this delivery problem.
- **Do not lean on the LLM to build cross-view topology/adjacency implicitly from description.** Cognitive Mapping's ~8% ceiling is the strongest single piece of evidence in the whole reading pass against RQ3 attempting an LLM-derived adjacency graph; any adjacency structure we expose must be computed offline and handed to the model as a fact, not something we ask it to infer from a set of view descriptions.

## 6. What does NOT transfer, and negative findings / caveats

- **Zero domain overlap with our setting.** No point clouds, no industrial parts, no eye-in-hand camera, no orthogonality-to-surface concept anywhere in the benchmark. Nothing here speaks to render quality, sparsity, or glossy-surface dropout (branch question on GraspView/SPAZER/VLM3D-style sparse-cloud rendering) — this document is silent on that entire cluster of questions; do not cite it there.
- **The benchmark measures relating *given* textual descriptions of views, not selecting *which* view to look at next.** There is no view-selection, no action space, no tool-count, no menu at all — every item is closed-form MCQ over pre-supplied multi-view captions. The branch questions about option count, option order, infeasible-view marking, and hierarchical/on-demand exposure are simply out of scope for this paper; it cannot answer them, and I did not force an answer.
- **No permutation/order testing, no option-count ablation.** Explicitly absent — confirmed by full read of §3, §A.5, and every table's legend. Flag as a benchmark-design gap in the literature, not just an omission in our extraction.
- **"Coordinate-aware" is a slight overclaim relative to what's delivered.** The title's framing implies numeric coordinates are the substrate; in practice the eval-facing text is prose with embedded numbers, and coordinates are confined to internal construction-time scratch work. Don't cite this paper as evidence that "coordinate-heavy" textual encodings work — the evidence, if anything, points the other way (see Q4 answer above).
- **Two internal inconsistencies worth flagging if this paper is cited for precise numbers:** (1) Human Cognitive Mapping score is given as 26.77% in Table 2 but as "26.44%" in the §3.2 prose — pick the table value (26.77%) as authoritative since it's the machine-checkable source. (2) §2.2 prose names ViewSpatial-Bench (Li et al. 2025) as a Path-B source alongside CoSpace and Ego3D-Bench, but Appendix A.1's actual enumerated data-source list has no ViewSpatial-Bench entry (only SpinBench's "View-spatial" subset, a different paper by Zhang et al. 2025b) — likely a naming collision between "ViewSpatial-Bench" and SpinBench's internal "View-spatial" subset that the authors didn't disambiguate. Treat "View Consistency comes from ViewSpatial-Bench" as unverified; the paper's own appendix says SpinBench.
- **No comparison to our specific benchmarks (MMSI-Bench, ViewSpatial, SpinBench, FlipSet) as standalone leaderboards** — SiT-Bench absorbs pieces of SpinBench and (nominally) ViewSpatial-Bench as *source material* for its own re-annotated text-only items, but never reports what the same models score on those benchmarks' original image-based protocol. So this document cannot directly confirm or deny numbers quoted elsewhere for MMSI-Bench/ViewSpatial/FlipSet themselves — it's adjacent evidence, not a replication.

## 7. Section/figure index for follow-up citation checks

- Task taxonomy & definitions: §2.1
- Construction pipeline, Path A/B, filtering: §2.2, Fig. 2
- Dataset stats: §2.3
- Main results table: **Table 2**, §3.2 (also reproduced with more rows as **Table 6**, §A.8)
- Latency table: **Table 3**, §A.3
- Subtask→source→count table: **Table 4**, §A.2
- DeepSeek-R1 filter prompt (verbatim): §A.4
- Implementation/hyperparameters: **Table 5**, §A.5
- Full Gemini-3-Flash reasoning traces (2 examples, panoramic counting): §A.6
- Human-participant instructions (verbatim): §A.7
- All per-subtask construction prompts + one worked example + one correct/incorrect model trace each: Appendix B, §B.1 (Scene Layout Reason, Panoramic Counting), §B.2 (Path Planning Logic, Ego/Objects-motion Perception), §B.3 (View Consistency, Perspective Shift, Pure Mental Rotation), §B.4 (Hand-Object Interaction)
- No numbered equations anywhere in the paper.
