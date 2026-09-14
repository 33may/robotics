# Measuring what Matters: Construct Validity in Large Language Model Benchmarks

- arXiv: 2511.04703 (v1), HTML full text: https://arxiv.org/html/2511.04703v1
- Domain: LLM benchmark methodology (text-first; explicitly *not* vision/embodied). Read for its checklist and diagnosed failure modes, not for any domain content transferable to view planning.

## Core method

A systematic review of **445 benchmark papers** from ICML/ICLR/NeurIPS (2018–2024) and ACL/NAACL/EMNLP (2020–2024), coded by **29 expert reviewers** against a codebook grounded in Cronbach & Meehl's construct-validity framework (face, content, ecological, convergent/discriminant, predictive validity).

Corpus funnel: 46,114 articles → keyword filter → 2,189 → GPT-4o-mini screening (validated at F1=0.84 against a 50-article human-labelled sample, Table 8) → 522 → manual review → 445 included (Table 7). Each article was double-coded (primary reviewer + second reviewer mapping), inter-rater reliability reported as mean Brennan–Prediger Kappa = **0.524** across 30 codebook fields (Table 9) — i.e. even expert reviewers only moderately agree on whether a benchmark satisfies these criteria, which is itself a signal about how contestable "does this benchmark measure the construct" judgments are.

Central definitional move (the one thing worth importing structurally): **a benchmark = a "task" + a "metric" used together to represent a "phenomenon."** Construct validity failures are then classified by *where* the phenomenon→task→metric chain breaks:
- Phenomenon ill-defined or contested (definitional stage)
- Task doesn't represent the phenomenon (sampling/representativeness stage)
- Metric doesn't measure what the task intends (scoring stage)
- No evidence offered that any of the above holds (validation stage — nobody checked)

Sections: 1 = Methods (study design, codebook, expert review); 2 = Results (Phenomenon, Task, Metric, Claims subsections); 3 = Recommendations (3.1–3.8); 4 = Discussion; 5 = Conclusion. Figure 1: corpus overview (top-3 phenomenon categories; articles/year vs. whether construct validity is discussed). Figure 2: codebook results across 5 ordered items, shaded region = "best practice" compliance band.

## Concrete results / numbers (Section 2)

**Phenomenon:**
- 21.8% of benchmarks give **no definition** of the phenomenon they claim to measure.
- Of those that do define it, **47.8% are contested** — "many possible definitions or no clear definition at all" (paper's example: alignment benchmarks measuring "harmlessness").
- 61.2% treat the phenomenon as composite (multiple sub-abilities) but only a minority of those separately measure each sub-component; 36.5% collapse a composite phenomenon into a single unified score anyway.
- Only 53.4% of benchmarks provide **any** evidence of construct validity at all.

**Task (representativeness/sampling):**
- Only **10%** of benchmarks used complete real-world tasks.
- 40.7% use constructed/artificial tasks at least partially; 28.5% exclusively.
- Convenience sampling used at least partially in 39.3% (12.3% exclusively, 27.0% "incorporated"); only 17.5% used exclusively targeted/purposive sampling.
- 38.2% reuse data from prior benchmarks or human exams; 7.7% source tasks *solely* from another benchmark (i.e. no new task design at all — pure repackaging).
- 31.2% use LLM-generated data at least partially.
- 21.1% require a specific output format that is itself a confound (format compliance gets conflated with the target capability).
- Human baselines present in only 32.4% of benchmarks.
- Ecological validity: only 5.6% explicitly compare the benchmark task to a realistic deployment setting; 26.8% claim the benchmark task itself is realistic.

**Metric:**
- 81.3% use exact-match as the (at least partial) primary metric.
- LLM-as-judge used at least partially in 17.1%, exclusively in 3.1%.
- **Only 16.0% use any uncertainty estimate or statistical test** to compare model scores. This is the headline statistical-rigor gap.

**Reliability of the review itself:** Brennan–Prediger Kappa = 0.524 mean across coded fields (Table 9) — moderate, not high, agreement among expert coders on construct-validity judgments.

## Eight recommendations (Section 3, verbatim structure)

1. **3.1 Define the phenomenon** — precise operational definition; state scope and explicitly excluded aspects; identify sub-components and measure them separately if composite.
2. **3.2 Measure the phenomenon and only the phenomenon** — control for confounding tasks (e.g., format compliance riding along with the target skill); assess whether output-format constraints distort measurement; if an LLM is used to parse/extract responses, that extractor LLM must itself be validated so it doesn't inject new bias.
3. **3.3 Construct a representative dataset for the task** — explicit sampling strategy justified as representative; quality-check large/auto-generated data; deliberately include items that probe known LLM failure sensitivities.
4. **3.4 Acknowledge limitations of reusing datasets** — document what was adapted from prior work, its known limitations, and report performance against the original for comparability.
5. **3.5 Prepare for contamination** — "benchmark contamination is likely to occur even with model developers acting in good faith"; run contamination-detection tests, keep a held-out item set, investigate pre-exposure in training corpora.
6. **3.6 Use statistical methods to compare models** — report sample size and justify statistical power; report uncertainty estimates for every primary score; describe rater demographics and mitigate demographic bias; for subjective labels use metrics that capture rater variability rather than single-point aggregation/exact match.
7. **3.7 Conduct an error analysis** — qualitative + quantitative failure-mode breakdown; check whether failures correlate with constructs the benchmark did *not* intend to measure; identify and report scoring biases.
8. **3.8 Justify construct validity** — argue relevance to real-world application; give explicit task/metric rationale tied back to the phenomenon definition; compare/contrast against existing evaluations of the same or adjacent phenomenon; discuss limitations and design trade-offs openly.

## Load-bearing assumptions

- The framework assumes a **text-generation LLM benchmark**: task = prompt, metric = scoring function over a textual response, phenomenon = a named capability/property (reasoning, harmlessness, etc.). The phenomenon→task→metric decomposition is the transferable abstraction; the specific failure-rate numbers are corpus-specific to that population of papers and don't transfer as base rates.
- Inclusion criterion explicitly filtered to benchmarks "compatible with text and vision models," excluding **audio and video** — but the paper does not substantively discuss multimodal, spatial, or embodied benchmarks anywhere in its analysis. It is not a VLM-benchmark or robotics-benchmark paper; any vision-model benchmarks in the 445 are folded into aggregate text-LLM statistics, uncharacterized separately.
- No dedicated treatment of **judge-model circularity or self-preference bias** (the phenomenon where a model scores its own or a similar model's outputs more favorably). Recommendation 3.2's "validate the extractor LLM" is the closest the paper gets, and it's framed as a parsing-bias concern, not a self-preference/collusion concern. This is a **negative finding**, not an oversight to work around — the self-preference literature the branch question (Q3) references is *not* in this document; it must be sourced elsewhere.
- No minimum-detectable-effect-size or formal power-analysis guidance is given despite recommending "justify statistical power" — the GSM8K example is cited only as "large sample size justified by diverse question creation," not a worked power calculation.

## What transfers to our setting

- **The phenomenon→task→metric decomposition is directly usable as an audit template for our own benchmark design.** For the view-planning benchmark: phenomenon = "the evidence view is orthogonal/informative for surface region X"; task = (object, region) or (object, question) episode; metric = the RQ1 judge / coverage score. Recommendation 3.1 says: write down the operational definition of "orthogonal enough" now, state what's excluded (e.g., self-occluded regions), before building the judge — directly actionable for Q7.
- **Recommendation 3.2 (measure the phenomenon and only the phenomenon) is the sharpest tool for Q19's contamination check.** It reframes "does the benchmark measure view quality or inspection outcome without one contaminating the other" as: is answer-correctness confounded by the VLM's language/reasoning ability riding along with view quality? This directly motivates Q15's view-correct × answer-correct grid — the paper's generic warning about confounding metrics is exactly the failure mode Q15 already anticipates.
- **Recommendation 3.5 (prepare for contamination) analogized to Q3**: the paper's contamination concern is about training-data leakage, not judge circularity, but the structural fix — hold out an item set, use an independent detection mechanism — is the same shape as "different provenance (third-party metric implementation as held-out score)" that Q3 proposes. Reinforces treating held-out/independent scoring as the correct move rather than a same-family reimplementation.
- **Recommendation 3.6 is a direct citation for Q12** (statistical treatment given few objects/trials): report sample size + power justification, uncertainty estimates on every primary score, and don't rely on single-point aggregation. The 16% compliance rate in the reviewed corpus is a citable "field average is this bad, we should not be" baseline to differentiate against.
- **Recommendation 3.7 (error analysis correlated with non-targeted constructs)** is a direct precedent for Q16 (naming individual VLM spatial failure modes as separate reported numbers) — the paper's generic advice ("investigate whether failures correlate with non-targeted constructs") is exactly what motivates decomposing failures into named categories rather than one aggregate score.
- **The 47.8%-contested-definition and 21.8%-no-definition numbers are usable rhetorically** to justify why Branch's Q2/Q7 (nail down an operational, checkable definition of "best view" before building anything) matters — this paper is evidence that skipping this step is the modal failure in the field, not a hypothetical risk.

## What does NOT transfer, and why

- **No formula, protocol, or numeric result about view planning, coverage, reconstruction error, or view-efficiency metrics.** This paper contains zero domain content on 3D vision, NBV, or active reconstruction — confirmed by the fetch: it is scoped to text-LLM benchmarks with audio/video explicitly excluded and vision/multimodal not discussed. Nothing here answers Q4/Q5/Q9/Q10/Q11 (mesh-free scoring, baselines, sensor-defect injection, fused-vs-single-view offset, object-set stratification) — those need domain-specific NBV/active-reconstruction literature.
- **No treatment of LLM-as-judge circularity/self-preference**, despite the branch task description asserting this paper covers it "verbatim." It does not. It mentions LLM-as-judge only as a metric-type statistic (17.1%/3.1%) and flags extractor-LLM bias as a parsing concern (Rec 3.2), not a self-preference or judge-contamination concern. **Q3 needs a different source** (the self-preference / LLM-evaluator bias literature, e.g. work on LLMs preferring their own outputs) — this paper should not be cited as covering that ground.
- **No guidance transfers on sample-size math** despite recommending it — Recommendation 3.6 says "report sample size and justify statistical power" but the paper gives no formula, no effect-size convention, no worked example beyond a hand-wave at GSM8K. Q12's "fix the effect size, derive required objects × trials" cannot be answered from this source; it needs a stats-methodology or small-sample experimental-design reference instead.
- **The failure-rate percentages (e.g., 81.3% exact-match, 16% statistical testing) are base rates over a corpus of ML conference LLM papers**, not general truths about benchmarking quality — they should be cited as "field practice in LLM benchmarking is this weak" context, not as a target or a validated methodology to imitate.
- **Kappa = 0.524 double-coding reliability** is a caution about the review's own rigor (moderate agreement among 29 experts on construct-validity judgments) — worth noting if citing this paper's percentages as if they were precise measurements; they carry real inter-rater noise.

## Answers / non-answers to branch questions

- **Directly useful for:** Q1 (phenomenon/task/metric decomposition frames "what is the unit of evaluation" as a phenomenon-definition question), Q12 (Rec 3.6, citable field-average statistical-rigor gap), Q15/Q19 (Rec 3.2's confounding-construct framing), Q16 (Rec 3.7's error-analysis-by-construct framing), Q17 partially (LLM-as-judge usage stats, but not the controls Q17 actually asks about), Q18 (the phenomenon→task→metric template is a candidate skeleton for the intake format).
- **Not answered here, needs other sources:** Q2, Q4, Q5, Q7, Q9, Q10, Q11, Q13, Q14, Q20, Q21, Q22 (all domain-specific to 3D vision/view planning/metrology — outside this paper's scope). Q3's specific "self-preference literature" claim is **not substantiated by this document** — flag for correction upstream in the branch's document-selection notes.

## Bottom line

Useful as a **generic audit checklist and citable field-weakness statistic**, not as a source of domain methodology. Its real contribution to this branch is procedural: force explicit phenomenon definitions before task/metric design (Rec 3.1), keep the judge/metric from measuring anything beside the intended construct (Rec 3.2), and hit the statistical-rigor bar the rest of the field is missing (Rec 3.6). It does **not** contain the self-preference/judge-circularity content the branch brief attributed to it, and contains no NBV/view-planning content at all — treat it as a methodology-hygiene reference, cite specific recommendation numbers rather than paraphrasing generically, and source Q3's circularity concern elsewhere.
