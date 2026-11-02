Yanbing Zhang, Bo Wang, Jianhui Liu, Nan Jiang, Jiaxiu Jiang, Haoze Sun, Yijun Yang, Shenghe Zheng, Lin Song, Haoyang Huang, Nan Duan, Wenbo Li (Joy Future Academy) — "Thinking with Novel Views: A Systematic Analysis of Generative-Augmented Spatial Intelligence" (TwNV), arXiv:2605.10588v1 [cs.CV], 11 May 2026, CC BY 4.0

Source: full HTML (`arxiv.org/html/2605.10588v1`) downloaded with `curl` and converted to plain text with `pandoc -t plain` (1166 lines, read in full, including all appendices A–C and the reference list). Every claim/number below was verified against this raw text unless marked "unverified" below.

**Methodology caveat, load-bearing for how to trust this note:** I first used the `WebFetch` tool (which summarizes via a small helper model) to extract details, twice. Cross-checking those summaries against the raw downloaded text found that the helper model **fabricated exact-looking numbers that do not exist anywhere in the source** — specifically a full per-backbone baseline/GEN_AUG accuracy table for Figure 5(b) (invented values like "Qwen3-VL-32B: 65.2%→71.9%", "GPT-5: 70.4%→71.7%" reused out of context, "Gemini-3-Flash: 75.4%→76.9%") and a full per-category baseline/delta table for Figure 4(b) ("Orientation 66.7%→68.4%"). I grepped the raw HTML for every one of these numbers; **none of them appear in the source** (only 68.4 appears once, in an unrelated table cell of Table 2). Figures 4(b), 5(b), and the "Discrete" instruction condition in Figure 7 are bar/heatmap images whose exact values are simply not present in the paper's text — only the prose deltas the authors chose to state explicitly are recoverable. This note reports only what is directly quotable from text/tables; anywhere the underlying figure would be needed for an exact number I say so and omit the invented figure. Treat this as a general warning about trusting `WebFetch` for numeric extraction from image-only figure content.

## Core method / idea

**The pipeline (§3.1, Fig. 2), three mandatory stages plus an optional iterative extension:**

1. **Stage 1 — Camera Movement Planning:** given the source image I₀ and question Q, a VLM ("Planner") produces a 6-DOF ego-motion instruction (Δx, Δy, Δz, yaw, pitch, roll).
2. **Stage 2 — Novel View Synthesis:** a generative image model ("Painter"/"Synthesizer"), conditioned on I₀ and the instruction, **generates** (does not render/photograph) a synthesized image Iₜ purporting to show the target perspective.
3. **Stage 3 — Spatial Reasoning:** in Simple Mode (N=0) the VLM ("Reasoner") reasons jointly over {I₀, Iₜ}. In Iterative Mode (N>0) a **Quality Verifier** VLM call evaluates Iₜ on "object visibility, generation quality, and scene consistency"; a rejected view triggers diagnostic feedback back to the Planner for a revised instruction, for up to N rounds. Final reasoning is over {I₀, I*} where I* is the best accepted view (falls back to I₀ if every round is rejected).

**Critical structural fact for our branch (verified directly, not previously flagged by either WebFetch summary):** there is **no menu of candidate views anywhere in this pipeline**. Exactly one continuous 6-DOF instruction is proposed per turn/round, and exactly one image is generated and added to context. TwNV is not a "pick one of k discrete pre-enumerated options" system at any stage — it is a "propose one continuous camera delta, hallucinate that view, look again" system. This is the single biggest scope mismatch with our RQ3/RQ4 (a fixed discrete action space of geometry-derived candidate views).

**The "numerical beats language" finding's actual locus (§4.3, verbatim mechanism explanation):** "Numerical instructions, by conveying only geometric transformation parameters (translation in meters, rotation in degrees), minimize such hallucinations and ensure **the generative module** produces viewpoint changes that faithfully preserve the original scene layout." The effect being measured is **the image generator's (diffusion/edit model's) susceptibility to producing unintended scene edits** when the instruction carries semantic content ("zoom in on the red mug") vs. when it carries only geometry (Δx=1.5, yaw=−20°). It is not a claim about a VLM's own comprehension of numeric coordinates as a reasoning input — see "What does NOT transfer" below for why this matters for resolving the branch's TwNV-vs-MapGPT question.

**CamCraft (Appendix A):** the paper's own purpose-built Painter, a 16B-parameter MMDiT backbone + Qwen3-VL-8B-Instruct semantic encoder for the pose instruction, with the source image latent (Wan2.1 VAE) concatenated channel-wise to the generation-noise latent and joint self-attention across both. Trained on 7.6M image pairs from DL3DV-10K, ScanNet++, Blender, and Ego-Exo4D (Appendix A.2), with ORB-feature overlap filtering and a monocular-metric-depth-based scale normalization step to unify translation units across heterogeneous source datasets into real metres (Appendix A.3).

## Concrete results and numbers

**Headline (Abstract, Fig. 1):** +1.3 to +3.9 pp accuracy across "four spatial subtask categories and four LMM architectures (both closed- and open-source)." The +3.9pp figure is explicitly Qwen3-VL-32B's absolute gain, +1.3pp is GPT-5's (Introduction, verbatim: "e.g., +3.9 pp for Qwen3-VL-32B vs. +1.3 pp for GPT-5").

**Evaluation set (§3.2, Fig. 3):** 695 samples = 575 from 3DSRBench (stratified across all 12 subcategories) + 120 from RealWorldQA (orientation/size/position subcategories). Four task categories: Orientation n=225 (viewpoint/heading/egocentric left-right/front-behind), Location n=230 (depth ordering, vertical comparison, proximity, relative layout), Size n=45 (physical dimension comparison), Multi-Object n=195 (facing direction, relative proximity, co-orientation, alignment). **Neither MMSI-Bench, ViewSpatial-Bench, SpinBench, nor FlipSet is mentioned anywhere in the paper (main text, related work, or references) — confirmed by direct grep of the full text.** This paper is not a data point on the branch's "closest measured tasks" capability-floor question; it uses a disjoint benchmark family.

**Tool-based grounding baseline comparison (§1, Introduction, motivating experiment):** giving GPT-5 a crop/zoom tool-calling pipeline **drops** overall accuracy vs. single-image baseline: 69.6% vs. 70.4% (−0.8pp), widening to −2.0pp on multi-object questions. This is presented as the paper's founding negative result for 2D re-framing tools generally (motivates going to generative 3D synthesis instead). TwNV then beats this tool-baseline: 71.7% vs 69.6% overall on GPT-5, "a striking 5.6pp advantage on multi-object relationships."

**Table 1 (§4.4), generator ablation, GPT-5 fixed as reasoner, verbatim table:**

| Category | w/o Gen | QwenImageEdit | NanoBananaPro | GPTImage1 | CamCraft |
|---|---|---|---|---|---|
| Orientation | 63.6 | 61.8 | 60.9 | 65.3 | 65.3 |
| Location | 80.9 | 77.8 | 83.0 | 80.4 | 82.6 |
| Size | 93.3 | 86.7 | 91.1 | 91.1 | 91.1 |
| Multi-object | 60.5 | 61.5 | 63.6 | 64.1 | 66.2 |
| **Overall** | **70.4** | **68.6** | **70.9** | **71.7** | **72.9** |

Deltas over no-generation baseline: QwenImageEdit **−1.8pp**, NanoBananaPro +0.5pp, GPTImage1 +1.3pp, CamCraft +2.5pp. **QwenImageEdit makes things worse than not generating a novel view at all** — the paper's own words: "low-fidelity edits can introduce misleading visual evidence that actively harms reasoning" and "a generator must reach a certain fidelity threshold before its outputs help rather than hurt the downstream reasoner."

**§4.3 instruction-format ablation, GPT-5/Gemini-3-Flash reasoners × GPT-Image-1/Nano-Banana-Pro painters (4 combinations), exact deltas quoted from text:**
- Numerical (6-DOF, meters+degrees): **consistently positive, +0.5% to +1.3%** across all 4 combos.
- Natural language (free-form): **strongly reasoner-dependent** — +0.3% with Gemini-3-Flash, but **−2.2% / −2.6% with GPT-5**.
- Discrete (categorical, e.g. "side view", "zoom in"): "near-zero or marginal gains across most settings" — **no exact per-combination numbers are given anywhere in the text**; only this qualitative summary and a heatmap figure (values not recoverable from the extracted text).

**§4.1 scene-coverage breakdown, GPT-5 + GPT-Image-1, exact numbers as stated in prose (no full table exists in text for this figure):** overall 70.4%→71.7% (+1.3pp); Orientation +1.7pp; Multi-Object +3.6pp; Location "comparable" 80.9%→80.4% (a *decrease*, described as "comparable"); Size: "minor degradation" (no number given in text). General-purpose (non-spatial) benchmarks: "negligible" / "no change" (no number given).

**§4.2 model robustness, four backbones (Gemini-3-Flash, GPT-5, Qwen3-VL-235B, Qwen3-VL-32B):** paradigm helps all four; relative gain (aug−base)/base "peaking at +6.7% for Qwen3-VL-32B," Gemini-3-Flash improves least (no exact number given). **Absolute per-backbone baseline/augmented percentages for all four models are not present in the extracted text — they exist only inside Figure 5(b) as an image.**

**Table 2 (§4.5), iterative refinement vs. text-only self-reflection, best configuration = Gemini-3-Flash + CamCraft, verbatim table:**

| Strategy | Config | Orient. | Loc. | Size | Multi-obj | Overall |
|---|---|---|---|---|---|---|
| Text reflection | Baseline (R=0) | 68.9 | 82.6 | 93.3 | 74.9 | 76.7 |
| | Best (R=1) | 69.3 | 81.3 | 93.3 | 72.3 | 75.7 |
| | Worst (R=4) | 68.4 | 77.0 | 91.1 | 68.2 | 72.7 |
| Image iterative | N=0 (Simple) | 73.3 | 85.2 | 91.1 | 74.4 | 78.7 |
| | N=1 | 72.4 | 86.5 | 91.1 | 76.9 | 79.6 |
| | N=2 | 72.4 | 86.1 | 91.1 | 76.4 | 79.3 |

Text R was swept over R∈{1,…,10}; only "best" (R=1) and "worst" (R=4) are reported, and self-reflection **never beats its own R=0 baseline at any tested R** — it "fluctuat[es] between 72.7% and 75.7% versus the single-pass baseline of 76.7%." This is a clean negative finding: adding text-only re-examination rounds over the *same* image monotonically-in-effect (though non-monotonic in R) **hurts** spatial accuracy here.

Image-based scaling: N=1 (79.6%) beats best text self-reflection (75.7%) by **+3.9pp**; at matched 2-VLM-call budget (text R=1 vs. image N=0), image wins by **+3.0pp** (78.7% vs 75.7%) — "the gap is not from extra compute but from reasoning over a new viewpoint instead of the same one." N=2 already regresses vs N=1 (79.3% < 79.6%); paper attributes this to "error accumulation... each additional generation–verification cycle risks introducing artifacts or drifting from the original scene context."

**Table 3 (§4.5), error attribution before/after one iterative round, GPT-5+CamCraft-class setup:**

| Error type | Simple # | Simple % | Iterative # | Iterative % |
|---|---|---|---|---|
| Wrong Instruction | 81 | 43.1 | 24 | 16.9 |
| Bad Generation | 85 | 45.2 | 101 | 71.1 |
| VL Failure | 22 | 11.7 | 17 | 12.0 |
| Total | 188 | 100 | 142 | 100 |

Wrong-Instruction count drops 81→24 (a 70% reduction) — the verification loop mostly fixes *what to ask for*, not *how faithfully it gets rendered*; Bad Generation becomes the dominant remaining failure (71.1%) — "the bottleneck shift[s] back to generation fidelity."

**Appendix B, capability-threshold analysis, Qwen3-VL-8B (a 5th, weaker backbone not in the main paper), verbatim Table B.1:**

| Variant | Orientation | Location | Size | Multi-Object | Avg |
|---|---|---|---|---|---|
| Baseline | 43.1 | 64.8 | 80.0 | 56.9 | 56.5 |
| GEN_AUG | 46.7 | 59.1 | 53.3 | 50.3 | 52.2 |

Overall **−4.3pp** (56.5%→52.2%); Size collapses **−26.7pp**. Error breakdown: Qwen3-VL-8B is 40.06% Wrong-Instruction / 42.47% Bad-Generation / 17.47% VL-Failure vs. GPT-5's 24.88% / 60.83% / 14.29% — the weak model's failures are dominated by *misinterpreting/mis-following the instruction task itself*, not by missing visual evidence. Explicit conclusion: "the understanding backbone should meet a minimum instruction-following competence for the augmentation to take effect" — below that floor, generative augmentation actively hurts.

**No variance/error bars/CIs are reported anywhere in the paper** (confirmed by full-text grep for "varianc", "std", "±", "confidence interval") beyond the single methodological note that every condition (including baseline) is a majority vote over K=3 independent runs, "so that observed differences reflect paradigm choice rather than sampling variance" (§3.3). This is an assertion, not a measurement — no per-condition spread is ever shown.

**Appendix A.5–A.7, CamCraft's own geometric-fidelity benchmark** (168 pairs, DL3DV-10K-VAL in-domain + ETH3D/MVImgNet2/RealEstate10K out-of-domain), judged by Gemini-3-Pro on 3 Likert-1–5 axes: Instruction Compliance (does the generated view match the target camera pose), Consistency with Original (are real objects preserved), Rationality of New Content (**how plausible the *hallucinated* previously-occluded regions look**). Best checkpoint (16k steps/~2 epochs): IC 2.941, CO 2.696, RN 2.683, avg 2.773 — on a 1–5 scale, i.e. **roughly the midpoint**, not near-ceiling, even for the paper's own best, purpose-trained, purpose-built generator.

## Load-bearing assumptions

- **The "view" the Reasoner sees is a diffusion-generated hallucination conditioned on a single source photo, never a photograph or a render of measured/real geometry.** The entire pipeline exists to imagine what an alternate camera pose *would* show, via a 16B-parameter generative model trained to fabricate plausible unseen content (explicitly graded on "Rationality of New Content" for made-up occluded regions). This is the opposite of our constraint that evidence images must show *real* defects.
- **The action space per turn is a single continuous 6-DOF delta, produced fresh by the Planner every time — never a fixed, pre-enumerated, discrete menu.** There is no analogue anywhere in this paper to "here are 8 named candidate views, pick one."
- **The three-way instruction-format comparison (Numerical/Discrete/Natural) is a comparison of how to instruct a *downstream generative model*, not how to present options for a VLM's own decision-making or comprehension.** The stated causal mechanism is generator hallucination-suppression (numbers carry no exploitable semantic redundancy for the diffusion model to over-interpret), not VLM legibility of numeric vs. named content.
- **All four main reasoning backbones (Gemini-3-Flash, GPT-5, Qwen3-VL-235B, Qwen3-VL-32B) are frontier-or-near-frontier scale models.** The one below-floor model tested (Qwen3-VL-8B, Appendix B) shows the paradigm actively hurts, gated on "basic instruction-following competence."
- **Answer judging is deterministic string match with a Gemini-3-Flash semantic-judge fallback (§3.3)** — an LLM-as-judge dependency for the ambiguous cases, not disclosed what fraction of the 695 samples needed the fallback.
- **The generator-comparison ablation (Table 1) and the instruction-format ablation (Fig. 7/§4.3) both hold the *reasoner* fixed as a controlled variable while varying one thing (generator or instruction format) — genuinely controlled, single-factor ablations, not just before/after demonstrations.** This is worth crediting: unlike many cited papers in this space, the encoding claim is directly ablated (see "Evidence discipline" note below).
- **CamCraft's own fidelity benchmark (Appendix A.5) shows even the best purpose-built generator averages ~2.77/5 across fidelity axes** — i.e., the paper's own best-case generator is measured as mediocre in absolute terms, not near-perfect; the entire pipeline's benefit rides on "better than the alternatives," not "good in an absolute sense."

## What transfers to our setting

- **The clean, matched-budget, ablated finding that reasoning over a *new visual observation* (even a generated/imagined one) beats spending the same compute on textual re-deliberation over the *same* image is directly load-bearing for the branch's "text vs. image / does a caption hand-off destroy something view selection needs" question (Table 2):** at matched 2-call budget, image-based Simple Mode (78.7%) beats text self-reflection R=1 (75.7%), a **+3.0pp** gap, and text self-reflection *never* beats its own no-reflection baseline (76.7%) at any tested round. This is strong, controlled, first-party evidence that a blind orchestrator asking itself to "think harder" about a caption/description **cannot substitute** for actually being shown new visual information — directly useful ammunition against any architecture that tries to close the gap with better prompting alone rather than adding a view.
- **The generator/render-fidelity gating result (Table 1: QwenImageEdit −1.8pp, i.e., *worse than no augmentation at all*) is the closest available analogue to our branch's "does anything survive our ~4cm fused-vs-single-view offset and glossy-surface dropout" question.** The transferable *principle*, not the numbers: showing the reasoning VLM a defective/low-fidelity visual artifact can be actively worse than showing it nothing extra — there is a fidelity floor below which "more information" backfires. This directly supports treating our point-cloud-render legibility as an open risk requiring its own ablation (a defective/holey render could play the QwenImageEdit role relative to a clean single-view photo baseline), rather than assuming any additional view helps by default.
- **The capability-threshold finding (Appendix B, Qwen3-VL-8B: −4.3pp overall, −26.7pp on Size) is directly relevant to the branch's "capability floor" question, orthogonally to the missing MMSI/ViewSpatial/SpinBench data:** it is independent, controlled evidence that spatial-reasoning augmentation schemes (of which a discrete-view-selection action space is one) can *actively hurt* below some backbone-capability floor, and that the dominant failure mode below that floor is *instruction-following/task-comprehension*, not missing-evidence. If we deploy on a smaller/local VLM, this is a concrete reason to validate baseline instruction-following competence before trusting a richer action-space design to help.
- **The Reasoner→Planner→Verifier→Planner iterative-refinement loop with an explicit stopping-diminishing-returns result (N=1 optimal, N=2 already regresses; §4.5) is a usable, if narrow, data point for the branch's "stopping-criterion" question:** more view-refinement rounds is not free; the paper's own explanation (error accumulation, drift from original context) is a concrete mechanism to watch for if our action space supports iterative view-request-and-verify loops.
- **The error-attribution methodology (Table 3: Wrong-Instruction / Bad-Generation / VL-Failure, tracked before/after an intervention) is a directly reusable diagnostic template** for Branch 2's benchmark: classify failures the same three ways (wrong view requested vs. defective render vs. correct-view-but-wrong-reasoning) to see which bottleneck our own system hits and whether fixing one shifts blame to another, exactly as this paper found (better generator shifts blame to instructions; better instructions shift blame back to generation).
- **The Reasoner-identifies-ambiguity → requests-new-evidence → re-examines control loop (Fig. 2) is structurally the same shape as a "blind orchestrator requests a view from a vision subagent" architecture**, even though the content of what gets requested (a continuous 6-DOF delta vs. a discrete view-ID) differs completely. Worth citing as one more example of the "ask again with a specific evidence request" pattern generalizing across the imagined-view and real-view literatures.

## What does NOT transfer, and why

- **The paper's core mechanism — generating a novel view via diffusion synthesis conditioned on a single photo — is fundamentally the wrong tool for an inspection task requiring real defect evidence.** CamCraft is explicitly graded on "Rationality of New Content," i.e. how plausible its *hallucinations of unseen regions* look, and even the best checkpoint only reaches ~2.7/5 on that axis. A generative model that fabricates plausible-looking geometry for regions it cannot see is close to the worst possible tool for surfacing or verifying a real physical defect — it could just as easily paint over a defect as reveal one. This is not a tunable detail of the paper; it is the paper's entire premise, so essentially none of the *generation* half of TwNV transfers to our setting (our "Painter" must be a real camera + robot pose, not a diffusion model).
- **There is no discrete action-space/menu anywhere in this pipeline to compare against RQ3/RQ4 at all.** The paper never asks "given k pre-enumerated named views, which format helps a VLM pick among them" — it asks "given one continuous 6-DOF delta to hand to an image generator each turn, which encoding avoids generator hallucination." The branch's Q4 framing ("which encoding of a camera pose does the model handle best — integers/angles, indices, or names?") is only half-answered here: TwNV gives evidence for *integers/angles* as an instruction format to a downstream generator, but says **nothing** about *indices or names* as a way to address discrete pre-fixed options, because that condition never appears in the paper.
- **Resolving the branch's TwNV-vs-MapGPT contradiction: they are not measuring the same failure mode, so "resolve the contradiction" may be a category error rather than a fact to reconcile.** TwNV's own stated mechanism (§4.3, quoted above) is that numeric instructions reduce **a generative image model's tendency to over-interpret semantically redundant language and hallucinate scene edits**. That is a claim about image-synthesis fidelity, not about a reasoning VLM's ability to parse/use numeric coordinates as decision input (which is what MapGPT's claim would need to be, per the branch's framing, though MapGPT itself was not read for this note). If MapGPT's "precise coordinates confuse the model" finding is about a VLM's own spatial-reasoning/planning step reading raw coordinates, the two papers may both be true simultaneously: numbers are good instructions *to send to a generator* (this paper) while being bad *content for a VLM to reason over directly* (if that is what MapGPT shows) — the resolution the branch needs requires actually reading MapGPT's exact pipeline stage to confirm, but this paper alone should not be read as license to assume "numerical always wins for VLMs."
- **The Discrete-instruction condition (Figure 7) — the format closest to our actual proposed interface (named/indexed candidate views) — is reported only as "near-zero or marginal gains" with no numbers ever given.** This is the format most relevant to us and it is the one the paper documents least; treat "numerical beats discrete" as asserted, not quantified, for the Discrete condition specifically.
- **No variance or error bars are reported for any comparison** (only a K=3 majority-vote note asserting, not measuring, robustness to sampling noise). Per the branch's "evidence discipline" question, this paper's ablations are genuinely controlled (single-factor swaps: Table 1 fixes the reasoner and swaps the generator; Fig. 7 crosses 2 reasoners × 2 painters × 3 formats) but **not variance-reported** — flag accordingly in any evidence table (ablated: yes; variance: no).
- **The benchmark (3DSRBench + RealWorldQA, 695 samples across Orientation/Location/Size/Multi-Object) is a different benchmark family from MMSI-Bench/ViewSpatial-Bench/SpinBench/FlipSet** that the branch's capability-floor question is actually asking about. This paper cannot answer "is relating one viewpoint to another above chance for the models we'd deploy" on those specific benchmarks — it simply doesn't touch them. Its Orientation category ("egocentric left/right and front/behind judgments") is adjacent in spirit but not the same measured task.
- **Every backbone tested (except the Appendix-B outlier) is Qwen3-VL-32B or larger, GPT-5, or Gemini-3-Flash — all substantially larger than typical locally-deployable models.** The one small-model data point (Qwen3-VL-8B) shows the paradigm *hurting*, which is a caution against assuming these gains generalize downward to smaller/cheaper models we might actually deploy.
- **The "Small-Model Dividend" framing (smaller-but-still-large models gain more in relative terms) should not be read as "small models benefit most" without qualification** — it holds only *within* the four large/frontier backbones tested; the one genuinely small backbone (Qwen3-VL-8B) breaks the trend catastrophically (−4.3pp), so the relationship between model size and augmentation benefit is non-monotonic across the full range, not a clean inverse-scaling law, and the paper explicitly frames this as gated by a "capability threshold" rather than a continuous trend.

## Checkable citations

| Claim | Section | Table/Fig. |
|---|---|---|
| Pipeline: Planner → Synthesizer → Reasoner (+ Verifier) | §3.1 | Fig. 2 |
| Evaluation set: 695 samples, 4 categories, 2 sources | §3.2 | Fig. 3 |
| Tool-based grounding baseline hurts (69.6% vs 70.4%) | §1 | Text |
| Numerical vs Natural vs Discrete instruction comparison | §4.3 | Fig. 6 (qualitative), Fig. 7 (heatmap, values not in text) |
| Generator ablation (QwenImageEdit −1.8pp … CamCraft +2.5pp) | §4.4 | Table 1 |
| Model robustness / Small-Model Dividend (+6.7% relative, Qwen3-VL-32B) | §4.2 | Fig. 5(b) (exact per-model % not in text) |
| Scene-coverage breakdown (Orientation +1.7pp, Multi-obj +3.6pp, Location 80.9→80.4, Size degrades) | §4.1 | Fig. 4 |
| Error taxonomy (Wrong Instr./Bad Gen./VL Failure) baseline split | §4.1 | Fig. 5(a) |
| Text reflection vs image-iterative refinement | §4.5 | Table 2 |
| Error attribution before/after iterative refinement | §4.5 | Table 3 |
| Capability threshold, Qwen3-VL-8B (−4.3pp, Size −26.7pp) | Appendix B | Table B.1 |
| CamCraft architecture (16B MMDiT + Qwen3-VL-8B encoder) | Appendix A.1 | Text |
| CamCraft training data (7.6M pairs, 4 sources) | Appendix A.2 | Fig. A.1 |
| Data processing (ORB overlap filter, metric-scale normalization) | Appendix A.3 | Text |
| Camera-pose fidelity benchmark (168 pairs, 4 datasets) | Appendix A.5 | Fig. A.2 |
| Fidelity metrics: IC/CO/RN, 1–5 Likert, Gemini-3-Pro judge | Appendix A.6 | Text |
| Training convergence, best checkpoint 16k steps avg 2.773/5 | Appendix A.7 | Table A.1 |
| K=3 majority vote per condition, no variance reported | §3.3 | Text |
| Answer judging: string match + Gemini-3-Flash semantic fallback | §3.3 | Text |
| Limitations: pipeline brittleness across separate networks, static/rigid-scene scope only | §5 | Text |
