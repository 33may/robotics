# Large Language Models Are Sensitive to the Order of Options in Multiple-Choice Questions

**arXiv:2308.11483** (submitted 2023-08-22). Pezeshkpour, Hruschka (Megagon Labs). Source used: `arxiv.org/html/2308.11483v1` (full HTML), cross-checked with two further targeted fetches for exact table values, methodology, and quotes.

Branch: Presenting spatial action spaces to LLM/VLM agents. Selected because: our view menu is functionally a multiple-choice list over candidate viewpoints, and this paper is the load-bearing citation for "reordering options alone changes accuracy" — i.e., that a geometry-derived, semantically meaningful ordering (face 1, face 2, ... or orbit angle order) is not free, and any benchmark we build must test under permuted orderings.

---

## 1. Core method

**Problem framing:** GPT-4 and InstructGPT (`text-davinci-003`) are evaluated on standard MCQ benchmarks, but instead of asking "how accurate is the model," the paper asks "how much does accuracy change if you keep the question and the correct answer fixed and only permute the order the options are listed in." This isolates *positional/order* effects from *content* effects.

**Sensitivity-gap metric (their core measurement, Table 1):** for each question, the model is queried under different orderings of its options; the paper reports "the difference between the maximum and minimum LLMs' performance when using an oracle ordering" — i.e., an **oracle search over orderings** (using ground-truth answer knowledge to find best-case and worst-case option placements), not a single random shuffle and not an average over random shuffles. This is the single most important methodological detail for us: **the headline 13–75% numbers are the range between the best possible and worst possible ordering for accuracy, not the typical swing you'd see between two arbitrary orderings.** The paper does not fully specify (and my fetches could not confirm) whether the oracle search is an exhaustive enumeration of all permutations or a constrained search; treat the number as an upper bound on order sensitivity, not an expected/average effect size.

**Conjecture 4.1 (their explanatory model, stated formally):**
> "The sensitivity of LLMs to the order of options in multiple-choice questions arises from the interaction of two colluding forces: (1) Uncertainty of LLMs regarding the correct answer among the top possible choices. And (2) positional bias, leading LLMs to favor specific options based on the order they appear."

I.e., order only matters when the model is genuinely torn between its top-2/top-3 candidate answers; for questions the model is already confident about, reordering doesn't move the answer.

**Diagnostic chain for the conjecture:**
1. **Sort-and-truncate test (Table 2):** ask the model to *rank* the options by probability (Hits@1/2/3 metrics — does the ground truth appear in the model's own top-k ranking), then re-run the MCQ task keeping *only* the top-2 or top-3 ranked options **in their original relative order**. Accuracy is roughly unchanged (sometimes marginally better/worse) versus the full option set — meaning the "wrong" options being present or absent doesn't explain the sensitivity; the wobble lives inside the top-2/3 set.
2. **Error-rate correlation (Fig. 3):** a scatter plot of sensitivity gap vs. model error rate across benchmarks/models — visually correlated, but **no correlation coefficient (r or R²) is reported**, so this is qualitative-visual evidence, not a quantified statistical claim.
3. **Self-verification (94% figure):** the model is asked, in a separate prompt, "Can more than one of the choices be a highly probable answer to the question? Please respond with 'yes' or 'no'." On the subset of questions flagged as "sensitive" (i.e., where reordering changed the answer), the model itself answers "yes" (multiple plausible answers) **>94% of the time**, across all five benchmarks and both models — used as self-report evidence that sensitivity tracks genuine top-choice ambiguity, not arbitrary noise.

**Placement pattern finding (their actionable result, Section on positional patterns + Table 3):**
> "To amplify the positional bias... the optimal strategy involves positioning the top two choices as the first and last options. Conversely, to mitigate bias, we recommend placing these choices among the adjacent options."

I.e., if the model's two leading candidate answers are placed at the two extremes of the list (first and last), positional bias is maximized (largest accuracy swing depending on which extreme carries the true answer); if the top-2 candidates are placed next to each other (adjacent), the swing shrinks. Table 3 shows model- and option-count-specific "patterns" (e.g., different optimal adjacency patterns for GPT-4 vs InstructGPT at 5 options) — the effect is not a single universal rule but is at minimum directionally consistent (extremes amplify, adjacency dampens).

**Calibration methods proposed as partial fixes:**
- **Majority voting:** run the same question under 10 random reorderings, take the majority-vote answer. Table 5: improvements from 0 to **+8 percentage points**, benchmark- and model-dependent (e.g., InstructGPT Logical Deduction +8.0%, GPT-4 Logical Deduction +2.0%, Abstract Algebra +0.0% for both models — no free lunch, sometimes no measurable gain).
- **Multiple Evidence Calibration (MEC):** prompt the model to generate an explanation before predicting (borrowed method, Wang et al. 2023b prompts, not reproduced verbatim in this paper). Results are **mixed and often negative**: decreases InstructGPT accuracy by −5.0% to −6.9% in some settings, inconsistent gains for GPT-4. The paper's own conclusion: MEC "diverges significantly from majority voting, casting doubts on its applicability."

---

## 2. Concrete results and numbers

**Table 1 — zero-shot sensitivity gap (max-min accuracy across oracle-searched orderings), the paper's headline number:**

| Benchmark | # options | GPT-4 gap | InstructGPT gap |
|---|---|---|---|
| CSQA (CommonsenseQA) | 5 | 22.9 pts | 43.1 pts |
| Logical Deduction (BIG-Bench) | 3 | **13.1 pts** | **73.4 pts** |
| Abstract Algebra (MMLU) | 4 | 53.0 pts | 70.0 pts |
| High School Chemistry (MMLU) | 4 | 41.8 pts | 66.5 pts |
| Professional Law (MMLU) | 4 | 24.8 pts | 50.6 pts |

The abstract's "approximately 13% to 75%" range is exactly the min (GPT-4/Logical Deduction, 13.1) and max (InstructGPT/Logical Deduction, 73.4, rounded) of this table. **Note for our routing brief:** the brief describing this document says "13-85%"; the paper's actual figure is 13–75% (rounding InstructGPT/Logical Deduction's 73.4 up loosely) — cite 13.1–73.4 pts (Table 1) as the precise, checkable number, not 85%.

**No clear relationship between option count and sensitivity magnitude:** the paper states directly (Section 3.1 area): "The domain and the number of options in the MCQ tasks seem to affect the model's performance. However, we do not observe a clear correlation between these factors and the sensitivity gap." Logical Deduction (3 options) has both the *smallest* GPT-4 gap (13.1) and the *largest* InstructGPT gap (73.4) in the table — option count alone does not predict sensitivity; content/domain and model identity dominate.

**Table 2 — sort-and-truncate test:** Hits@1/2/3 (does the model's self-ranking place the ground truth in its top-k) plus accuracy when only the top-2/top-3 (in original order) are kept. Used to support Conjecture 4.1's "ambiguity lives in the top-2/3" claim, not itself a headline transferable number.

**Table 3 — placement patterns:** model- and option-count-specific tables of which relative positions of the top-2 choices amplify vs. mitigate bias; confirmed pattern direction (extremes amplify, adjacency mitigates) but exact optimal slot varies by model.

**Figure 2 — few-shot mitigation:** tested at k ∈ {1, 2, 5} shots. Qualitative finding: "the sensitivity gap consistently remains substantial even with the inclusion of more demonstrations in the few-shot setting" — few-shot examples reduce but do not close the gap. Exact per-k numbers live in an appendix I could not extract in this pass (flagged, not fabricated).

**Figure 3 — error-rate correlation:** scatter plot only, **no r/R² value reported**. Treat as suggestive, not statistically quantified.

**Table 5 — majority-voting calibration:** up to +8 pts improvement (InstructGPT, Logical Deduction), 0 pts on Abstract Algebra for both models — inconsistent, benchmark-dependent, and the paper flags it as "computationally expensive" (10x inference cost per question).

**Models tested: only GPT-4 and InstructGPT (`text-davinci-003`).** No GPT-3.5-turbo, no open-source models, no VLM of any kind appears anywhere in the paper, including any appendix reachable via my fetches. No API version/date is given beyond citing OpenAI (2023).

**No human baseline anywhere in the paper.**

**No explicit sample-size (N questions) statement for the Table 1 zero-shot experiments** was locatable — the only concrete N I could confirm is "randomly choose 100 samples" for selecting few-shot demonstrations, which is a different number (demonstration pool, not eval set size). This is a real gap in the paper's reporting, not an extraction failure on my part — flag it as such.

---

## 3. Load-bearing assumptions

1. **The 13–75% figure is an oracle best-vs-worst gap, not an expected/average effect of a single reordering.** This is the single most important caveat for citing this paper. A random shuffle of a real deployed system will very likely produce a *smaller* swing than the oracle max-min range, because the oracle search specifically hunts for the worst-case ordering using ground-truth knowledge the deployed system doesn't have access to at prompt-construction time. The paper's own majority-voting experiment (10 *random* reorderings, not oracle) shows the practically-realized effect is smaller — up to +8 pts of *recoverable* accuracy, not 13–75 pts — which is consistent with random reordering producing a milder, but still real and non-zero, swing than oracle reordering.
2. **Sensitivity is conditional on model uncertainty, not universal.** Per Conjecture 4.1 and the sort-truncate test (Table 2), reordering does nothing to accuracy on questions the model is already confident about; it only moves the needle on the (self-identified, ~94%-flagged) ambiguous top-2/3 cases. If our view-selection prompts have a genuinely dominant, unambiguous best option, order sensitivity should be low; if two candidate views are geometrically/visually near-tied, expect the described bias.
3. **Domain and option count don't predictably set the magnitude — the paper found no clean correlation.** We cannot use "we only have 6–12 view options" to argue sensitivity will be small; Logical Deduction with only 3 options showed the single largest gap (73.4 pts, InstructGPT). Option count is not a safe proxy for how much ordering will matter.
4. **Evaluated only on text-only knowledge/reasoning MCQs, on 2023-era non-multimodal, non-agentic models.** GPT-4 (2023 chat/completion, not GPT-4V), InstructGPT (`text-davinci-003`, a legacy completion model, not even chat-tuned). No VLM, no spatial task, no image-grounded choice, no tool-calling / discrete-action agent setting appears. Whether the magnitude (not just the qualitative existence) of order sensitivity replicates on current (2026) multimodal agentic models answering "which viewpoint" is **untested by this paper** — the mechanism (top-2 ambiguity + positional bias) is plausible to generalize; the specific percentages are not.
5. **Calibration fixes are weak/inconsistent, not a solved problem.** Majority voting over 10 reorderings recovers up to 8 pts but sometimes 0; MEC often backfires. There is no proposed fix in this paper that reliably neutralizes order sensitivity — this is a paper that documents and partially diagnoses the problem, not one that solves it.

---

## 4. What transfers to our setting

- **The core warning transfers directly and is the reason this paper is in our reading list: a geometrically meaningful option order (face 1, face 2, face 3... or camera angle 0°, 30°, 60°...) is not a neutral choice of encoding — it is itself a variable that can bias the model's selection independent of which view is actually best.** This is exactly RQ11's concern. Any benchmark we build to test view-selection quality must include a permuted-order control condition (at minimum: reversed order, and a few random shuffles) to separate "the model picked the geometrically-best view" from "the model picked whichever view sat in a favored slot."
- **The mechanism (bias bites hardest when the model is genuinely torn between its top-2 candidates) is the actionable design principle, not the raw percentage.** For our task, the risk case is exactly when two candidate views are near-equally good — e.g., two adjacent orbit positions both roughly orthogonal to a curved surface. This is precisely the "near-identical items" condition Q9 asks about, and this paper's finding says: *that* is when order will matter most, regardless of how many total options are on the menu.
- **The amplify/mitigate placement finding gives a concrete construction rule if we want to *stress-test* a benchmark:** to find a model's worst-case order-driven failure, put the two most plausible views at the first and last slot; to *reduce* the risk of order artifacts influencing production behavior, keep visually/geometrically similar candidate views adjacent in the presented list rather than scattered. This is a testable, falsifiable recommendation, not just a warning.
- **The majority-voting result is a cheap, literature-grounded ablation we could adopt for the benchmark itself (not necessarily production):** re-run a view-selection question under several reorderings and check answer stability, exactly analogous to their 10-reorder majority vote. Gives us a concrete, adaptable protocol rather than needing to invent a permutation-testing method from scratch.
- **The "no correlation between option count and sensitivity" finding is a direct, load-bearing answer to part of RQ9:** it rules out "we have few enough options that order won't matter" as a valid inference. Count is not evidence of safety from order effects.

## 5. What does NOT transfer, and why

- **The exact 13–75% (or any Table 1 number) must not be quoted as "the swing our system will see."** It is (a) an oracle best/worst bound, not a realistic single-reordering effect, (b) measured on GPT-4/InstructGPT circa 2023, not on any current VLM, and (c) measured on unrelated text-knowledge MCQ tasks (commonsense QA, logical deduction, algebra, chemistry, law), not spatial/viewpoint selection over images. Citing "13-75%" as *our* expected benchmark swing would misuse the paper exactly the way the branch's evidence-discipline requirement warns against.
- **Nothing here is multimodal.** No image, no rendered view, no point cloud, no camera pose appears anywhere in this paper — it answers zero of the branch's viewpoint-relation, ego/allo, or visual-encoding questions (Q1, Q2, Q5, Q7). Its scope is strictly the *text list-ordering* problem, one layer of our interface (the menu), not the *content* of what's in the menu.
- **No option-similarity/distinctiveness variable is manipulated.** The paper's options are semantically distinct MCQ answers (e.g., different numeric/legal/chemical answers), not near-duplicate candidates like two adjacent viewpoints 15° apart. We can extrapolate the "ambiguity concentrates the effect" mechanism to near-duplicate options, but the paper provides no direct evidence at that end of the similarity spectrum — flagged as an extrapolation, not a measured finding.
- **No production-grade fix is validated.** Majority voting is inconsistent (0 to +8 pts) and 10x more expensive; MEC is unreliable and sometimes harmful. Nothing here can be lifted as a ready-made calibration for our orchestrator — at best, it tells us permutation-averaging is worth testing empirically, not that it will reliably work.
- **No statistical rigor on the headline correlation (Fig. 3 has no r/R²) and no reported sample size for the main Table 1 experiment.** Treat Table 1 gaps as point estimates from an unspecified N, not as numbers with confidence intervals — consistent with the branch's general "flag missing variance reporting" instruction (Q14).
- **Zero engagement with agentic/tool-use framing.** No discrete-action-space, no embodied task, no multi-turn tool selection — this is single-turn, single-question MCQ answering. Nothing here addresses the "one operation per turn" (Q3), infeasible-option handling (Q11 relative), or hierarchical menu structure (Q12) parts of the branch.

---

## 6. Evidence-discipline entry (for Branch's ablation table, Q14)

| Comparison | Ablated within-paper? | Delta | Variance reported? | N |
|---|---|---|---|---|
| Oracle-best vs oracle-worst option ordering, zero-shot (Table 1, 5 benchmarks × 2 models) | Yes (same model, same question set, ordering is the only manipulated variable) | 13.1–73.4 pts (see table above) | No CI/variance; unspecified eval-set N | not stated |
| Full option set vs. top-2/3-only (order preserved), sort-truncate test (Table 2) | Yes | "nearly unchanged or incremental" (no single delta number extracted) | No | not stated |
| Zero-shot vs. few-shot (1/2/5-shot) sensitivity gap (Fig. 2) | Yes | gap "remains substantial" at all k — qualitative, exact deltas not extracted from this pass | Not confirmed | not stated |
| Naive vs. majority-vote-over-10-random-reorders (Table 5) | Yes | 0 to +8 pts, benchmark-dependent | No | not stated |
| Naive vs. MEC calibration | Yes | mixed, −6.9 to positive but inconsistent | No | not stated |
| Sensitivity gap vs. model error rate (Fig. 3) | No formal ablation — correlational plot only | not quantified (no r/R²) | No | not stated |

**Flag:** no confidence intervals or variance anywhere in the paper; no stated per-benchmark eval sample size for the core Table 1 experiment; only two models tested, both pre-GPT-4V/non-multimodal, both 2023-era. This is a real, load-bearing warning about MCQ order effects but a methodologically thin paper by the branch's evidence-discipline standard — cite the *mechanism* and *qualitative direction*, flag any specific percentage as oracle-bound-not-typical-case.

---

## 7. Failure modes catalog (handoff to Branch 2)

- **Positional/order bias under top-choice ambiguity**: when a model is torn between two (or three) leading candidates, its final pick shifts measurably depending on where those candidates sit in the presented list — not because of their content, purely their slot.
- **Extremes-amplify-adjacency-mitigates pattern**: placing the two most plausible options at the first and last position of a list maximizes the chance that ordering (not content) decides the outcome; placing them next to each other minimizes it. A named, constructible test case: for any view-selection menu, construct one ordering with the two best-looking candidate views at opposite ends and one with them adjacent, and check if the model's choice flips.
- **Option-count is not a safe predictor of susceptibility.** A 3-option menu (Logical Deduction) showed the single worst measured swing (73.4 pts, InstructGPT) in this paper's own data — don't assume "few options = safe from order effects."
- **Self-reported ambiguity as a diagnostic, not a fix**: models can be asked directly whether multiple options seem equally plausible, and this self-report correlates strongly (>94% "yes" rate on paper's flagged-sensitive cases) with actual order-driven answer instability — a cheap probe Branch 2 could reuse to flag risky view-selection questions in the benchmark, even though it doesn't fix the underlying bias.
- **Calibration overhead / no cheap fix**: the only tested mitigations either cost 10x inference (majority voting over reorderings) or actively hurt accuracy in some conditions (MEC). Don't assume a benchmark run "with calibration" is comparable in cost to a naive single-order run.

---

## 8. Answers to / effect on branch questions

- **Q11 (how much does ordering matter; must the benchmark permute):** Directly served, this is the paper's whole contribution. Answer: order sensitivity is real, mechanistically tied to top-choice ambiguity (not to option count or domain), and the effect concentrates exactly where our hardest cases will live (near-tied candidate views). **The benchmark must evaluate under permuted orderings** — at minimum, present the same view-selection question with the candidate list in original order, reversed order, and one or two random shuffles, and treat answer instability across those as a first-class failure signal, not noise to average away. Do not report a single "accuracy" number under one canonical order (e.g., face 1→face 2→face 3) without a permutation check — that number is exactly the kind of oracle-favorable-or-unfavorable ordering this paper shows can swing by tens of points on other MCQ tasks. Caveat honestly: cite the *mechanism and qualitative magnitude-is-large* finding, not the specific 13–75% number, since that's an oracle bound on unrelated text tasks with 2023 models.
- **Q9 (option count vs. distinctiveness as the binding constraint):** Directly served, partial answer. This paper found **no correlation between option count and sensitivity magnitude** — the smallest-option benchmark (3 choices) had the single worst gap. This weakens any argument that "we only have ~6-12 views so order won't matter much" and shifts weight toward distinctiveness/ambiguity (near-duplicate candidate views) as the more likely binding constraint — consistent with Conjecture 4.1's uncertainty-among-top-choices mechanism, though this paper never directly manipulates option *similarity*, so the distinctiveness side is extrapolation, not measured.
- **Q14 (evidence discipline table):** Directly served — table in §6 above, ready to merge into Branch 2's master ablation table, all entries correctly flagged for missing variance/N.
- **Q16 (failure-mode catalog):** Directly served — §7 above, including one directly reusable adversarial test-construction recipe (extremes vs. adjacent placement of top-2 candidates).
- **Q7/Q8 (rendering, chicken-and-egg viewpoint selection):** Not addressed — zero multimodal content in this paper.
- **Handoff to Branch 5 / RQ4 tractability:** No structural capability limit found here — the finding is about *presentation order* of an already-fixed menu, not about whether relational/topological structure (adjacency, "opposite face") can be reasoned over. Silent on that question, not informative either way.
- **Architecture verdict (blind vs seeing orchestrator):** Not informative — this paper's models are text-only in every experiment; it cannot speak to whether seeing the render changes order sensitivity. Worth flagging as an open question for Branch generally: does giving the orchestrator vision (so it can perceptually compare options rather than reasoning purely over a text list) reduce reliance on position as a tiebreaker? Not tested by this paper or, as far as this reading pass found, by any paper in this directory yet.

---

## 9. Bottom line

This paper is the right citation for the *existence and mechanism* of option-order sensitivity in LLM multiple-choice answering: reordering alone swings accuracy by double digits, the effect is driven by ambiguity among the model's top-2/3 candidates (not option count, not domain), extremes-placement amplifies it while adjacent-placement dampens it, and the only tested fixes (majority voting over reorderings, self-explanation calibration) are inconsistent and partial (0–8 pts recovered, sometimes negative). **Precise numbers must be cited carefully**: the headline 13.1–73.4-point range (Table 1) is an oracle best-vs-worst bound on 2023-era GPT-4/InstructGPT over unrelated text-knowledge MCQs, not a measured typical-case swing and not evidence about multimodal or spatial tasks. For our branch, the actionable takeaway is the mechanism, not the magnitude: **our view-selection menu, if it encodes a geometrically meaningful order, must be tested under permuted orderings, with the sharpest expected risk exactly where candidate views are visually/geometrically near-tied** — which is also where our design is most likely to actually need the model to make a real judgment call.
