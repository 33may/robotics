# How Many Tools Should an LLM Agent See? A Chance-Corrected Answer

- arXiv: 2605.24660 (v2, HTML)
- Authors: Vyzantinos Repantis, Ameya Gawde, Harshvardhan Singh, Joey Blackwell II
- Read via: arXiv HTML full text (all sections fetched and parsed directly from
  the LaTeXML HTML, not summarized secondhand) — Introduction, Related Work,
  Method (3.1–3.3), Empirical Evaluation (4.1–4.3), Discussion (5.1–5.3),
  Conclusion, Appendix A. Equations and table numbers below are copied from the
  source, not reconstructed.

## Domain flag (read this first)

This is a **tool-retrieval / RAG-shortlist-depth paper**, not a spatial- or
vision-agent paper. There are no images, no cameras, no viewpoints, no VLMs —
every "tool" is a text-described function (BFCL function-calling entries,
MetaTool API descriptions, ToolBench API catalog) or a text document (SciFact,
NFCorpus, MS MARCO in the appendix), scored by BM25 or a sentence-embedding
model, and the only "agent" decision is a binary STOP/CONTINUE over a
score-ranked list. It answers **zero** of the branch's spatial-reasoning
questions (ego/allo framing, composition, encoding of camera pose, wrong-face
selection, etc.) directly. Its relevance is narrow and specific: it is the
best available **quantified, ablated account of what happens to LLM choice
accuracy as candidate-set size grows**, plus a reusable **chance-corrected
random-baseline formalism**. Treat it as answering RQ9 (count vs.
distinctiveness) and the random-baseline RQ, and as a stopping-criterion
alternative to Explore-until-Confident — not as evidence for anything
spatial.

## Core method

**Bits-over-Random (BoR)**, introduced in a prior paper (Repantis et al., cited
as [30]) and applied/extended here. For corpus size `N`, `R_q` items relevant
to a query, `K` items presented, the random baseline is the hypergeometric
probability of including at least one relevant item by chance:

```
P_rand = 1 - C(N-R_q, K) / C(N, K)                     (Eq. 1, Sec 3.1)
```

which simplifies to `P_rand = K/N` for single-relevant queries (`R_q=1`,
true for BFCL/MetaTool/ToolBench/SciFact; NFCorpus has `R_q` from 1–475).
`P_obs` is Success@K (fraction of queries where ≥1 relevant item lands in the
top K). Then:

```
BoR = log2(P_obs / P_rand)                              (Eq. 2, Sec 3.1)
```

BoR=0 is random-level; BoR=10 is 1,024× better than random; BoR<0 is worse
than random. Two derived quantities: `BoR_max(K) = -log2(P_rand(K))` (ceiling
at perfect success), and `BoR_opt(K) = log2(N/K)` (an R_q=1-assuming
upper bound usable across systems/methods on the same corpus — "self-
calibrating," Sec 3.1).

**Self-pruning property (Sec 3.3, load-bearing for the "how many is enough"
RQ):** because `P_rand` rises with every added candidate, the BoR reward
*mathematically* decays with depth without any hand-tuned penalty. Worked
example given in the text: "At K=3 from a 500-tool registry, the reward is
approximately 7 bits, but at K=100 it drops to 2 bits" — a 5-bit drop, purely
from the combinatorics, not an engineered term. Doubling rule: once `P_obs`
plateaus, doubling K costs ≈1 bit (`ΔBoR ≈ -log2(K2/K1)`); once `P_obs>0.5`,
gaining more selectivity by going deeper becomes mathematically impossible
(P_obs is capped at 1).

**MDP formulation (Sec 3.2).** One query = one episode. A scorer (BM25 or
embeddings) ranks all N candidates once, up front. The RL agent walks the
ranked list one item at a time.
- **State:** scores of items examined so far (top score, gap between 1st and
  current, score spread), current depth `k_t`, registry size `N`, BoR ceiling
  at current depth.
- **Action:** binary STOP / CONTINUE.
- **Reward:** at STOP, if presented set contains ≥1 relevant item,
  `reward = -log2(P_rand(k_stop; R_q))`, else 0. Per-step continuation cost
  (`step_cost=0.01, γ=0.95` default; `γ=1.0` for MetaTool+BM25;
  `step_cost=0.005` for the main-table BFCL+BM25 agent) discourages
  unnecessary depth without encoding a depth preference itself. Agents are
  trained with **oracle R_q** and assume `R_q=1` at inference (every query in
  these benchmarks has exactly one correct tool) — this single-hit assumption
  is structural to the whole method, see "does NOT transfer" below.

**Baselines (Sec 3.3):** (1) Fixed-K (FK=1/5/10/20/50, same K for every
query); (2) an `F1 = 2/(K+1)` ablation — a depth-aware but **not**
chance-corrected reward, used to test whether *any* depth penalty (not
specifically BoR's random-chance correction) would produce the same adaptive
behavior. Agents: deliberately small — DQNs and tabular Q-learning on CPU,
explicitly "a probe of the metric's properties, not a proposed production
architecture" (Sec 1, Scope).

## Concrete results and numbers

**Table 2 (cross-condition summary, all 9 tool-selection conditions):**

| Condition | BoR Found% | F1 Found% | Best FK Found% | BoR K | F1 K |
|---|---|---|---|---|---|
| BFCL+BM25 (N=370) | 90.3±2.4 | 88.9±1.4 | 90.8 (FK=50) | 7.4±2.5 | 6.4±1.9 |
| BFCL+embed (N=370) | 85.0±3.0 | 85.8±1.4 | 97.5 (FK=5) | 1.4±0.1 | 1.6±0.1 |
| MetaTool+BM25 (N=100) | 96.2 | 82.8 | 83.7 (FK=50) | 80.7 | 57.2 |
| MetaTool+embed (N=100) | 73.3 | 69.0 | 74.7 (FK=3) | 2.3 | 3.0 |
| MetaTool+BGE (N=100) | 71.4 | 61.4 | 57.2 (FK=1) | 2.4 | 1.2 |
| MetaTool (N=20) | 70.6 | 63.3 | 59.2 (FK=1) | 1.6 | 1.3 |
| MetaTool (N=50) | 74.7 | 63.9 | 59.2 (FK=1) | 2.6 | 1.2 |
| MetaTool (N=100) | 72.8 | 63.3 | 59.2 (FK=1) | 2.1 | 1.5 |
| ToolBench (N=50) | 61.9±0.6 | 47.6±1.3 | 77.3 (FK=20) | 4.4±0.4 | 1.5±0.3 |

**Difficulty-bucket adaptivity (ToolBench N=50, Sec 4.1, Fig. 1)** — the
clearest, variance-backed demonstration of query-adaptive depth: easy
(gold ranked 1st, n=272) K=2.5±0.2 → 100% found; medium (2nd–5th, n=116)
K=4.8±0.5 → 74.4±0.4%; hard (6th–20th, n=76) K=5.7±0.5 → **16.7±4.3%** (FK=5,
F1, and FK=1 all find **0%** on this bucket); very hard (21+, n=136) K=6.9±0.7
→ 0.2% (all alternatives 0%). FK=5 wins on aggregate Found% (64.7% vs BoR's
61.9%) purely because most test queries are easy/medium and a uniform K=5
catches them without needing to adapt — "FK=5 achieves higher aggregate
coverage... but finds nothing on hard queries" (Sec 4.1).

**Scorer-quality ablation (Sec 4.2, Fig. 2), MetaTool, same data, same BoR
reward, only the scorer changes:** BM25 (found@1=33%) → K=80.7 learned,
96.2% found but only 1.04 bits selectivity ("nearly showing all tools");
MiniLM (found@1=60%) → K=2.3; BGE (found@1=57%) → K=2.4. "The two embedding
models, which have similar retrieval quality, produce nearly identical
policies. The difference is between lexical and semantic scoring, not
between embedding architectures." This is a genuine ablation (same queries,
same reward, only scorer swapped) with a clean, quotable delta.

**Downstream tool-choice validation (Sec 4.3, Table 1) — the single most
relevant result for our branch.** BFCL, N=370, BM25 scorer, Claude Sonnet
4.6 forced to pick exactly one tool from the presented set, 120 test
queries, 3 seeds, temperature=0, **tool descriptions and ordering held
identical across methods — only set size K varies**:

| Method | Presented% | Choice Acc% | End-to-End% | Avg K |
|---|---|---|---|---|
| BoR | 76.9±0.4 | **93.1±0.5** | 71.7±0.0 | 2.2±0.4 |
| F1 | 72.8±2.7 | 94.3±1.7 | 68.6±1.7 | 1.7±0.3 |
| FK=5 | 84.2 | **87.1** | 73.3 | 5.0 |
| FK=1 | 65.0 | 100.0 | 65.0 | 1.0 |

Gold present, but Claude picks it 93.1% of the time under BoR's short list
(K≈2.2) vs. **87.1%** under FK=5 — a **6pp gap purely from distractor count**,
with descriptions/ordering held fixed. On the medium-difficulty bucket
(gold ranked 2nd–5th, n=23) the gap widens sharply: FK=5 presents gold on
100% of these queries but Claude selects it only **60.9%** of the time;
BoR presents on 62.3±2.0% of these queries but at **76.8±2.5%** choice
accuracy when it does. Replicated with an embedding scorer: 96.1±2.0%
(BoR) vs 84.6% (FK=5) overall (11.5pp), widening to **80.1% vs 50.0%
(30pp)** on medium-difficulty queries. FK=1 hits 100% choice accuracy
trivially (no distractors possible) but only 65% end-to-end because it
often fails to present the gold tool at all.

**Appendix A (document retrieval, structurally identical framework, not
tool-specific):** SciFact (N=5,183) BoR K=7.2, 78.9% found, vs FK=50's 85.6%
at 7× depth; NFCorpus (N=3,633, R_q up to 475) BoR K=22.9, 71.1% found — "the
separation is directional rather than dramatic" here because R_q is often
large; MS MARCO (N=8.8M) BoR K=24.0, 82.7% found, exceeding FK=50's 80.7% at
under half the depth.

## Load-bearing assumptions

1. **Single relevant item per query almost everywhere (`R_q=1`).** BFCL,
   MetaTool, ToolBench, SciFact all effectively R_q=1; only NFCorpus varies
   R_q (1–475). The whole BoR/MDP apparatus is built around "is the one
   right item in the top K," a **retrieval** framing, not a **coverage**
   framing (need several/all of a set). Trained with oracle R_q, inference
   assumes R_q=1 (Sec 3.2).
2. **A relevance scorer exists prior to and independent of the LLM**
   (BM25 or embeddings over text descriptions), producing a single fixed
   rank ordering that the RL agent walks monotonically. Depth selection is
   entirely a **pre-LLM shortlisting** problem; the LLM only sees the final
   presented set once, in Sec 4.3's validation.
3. **Registries are large relative to our setting in 7 of 9 tool-selection
   conditions** (100–3,251). Only MetaTool's N=20 condition (Table 2 row 6)
   approaches our scale, and there the BoR-vs-best-FK gap is much smaller
   (70.6% vs 59.2% found, BoR K=1.6) than at N=3,251 (ToolBench) — the paper
   itself never states this trend explicitly but it is directly readable off
   Table 2, and Sec 3.1 flags the mechanism: "for tool selection N is small
   (typically 50–500), so P_rand rises quickly with K" — meaning the ceiling
   on achievable selectivity shrinks as N shrinks, and our action space
   (order 10–30 discrete views) sits below even the smallest N tested here.
4. **Distractor identity varies "naturally" with the benchmark, never as a
   controlled independent variable.** BFCL's near-duplicate functions
   (`calc_area_triangle` vs `calculate_triangle_area` vs `calculate_area`,
   Sec 4.1) are offered as evidence that semantic near-duplicates exist in
   the registry, but the paper is explicit that it does **not** decompose
   this: "no controlled ablation isolates similarity-based confusion from
   count effects" is not a verbatim line but the paper never runs a
   matched-K, varied-similarity experiment — every ablation in Table 2 and
   Sec 4.1–4.2 varies K or scorer, never distractor similarity at fixed K.
5. **Ordering is held fixed, never manipulated.** "Tool descriptions and
   ordering from the scorer are kept identical across methods" (Sec 4.3) —
   order is always scorer-rank order; there is no permutation/randomization
   condition anywhere in the paper.
6. **Downstream validation is single-shot forced-choice**, not sequential/
   adaptive selection with dependent outcomes (choosing view 1 changes the
   value of view 2) — closer to "pick the one right function to call" than
   to "plan a sequence of views that jointly cover a surface."

## What transfers to our setting

- **A reusable, self-calibrating random-baseline formalism directly answers
  the branch's "what random baseline does our benchmark need" question.**
  `P_rand = 1 - C(N-R_q,K)/C(N,K)` (Eq. 1) and `BoR = log2(P_obs/P_rand)`
  (Eq. 2) give a chance-corrected way to state "our view-selection agent is
  X bits better than picking views uniformly at random from the candidate
  set," provided we can define an analogous binary success rule (e.g., "did
  the chosen view set include an adequately-orthogonal view for surface S")
  and treat N as the candidate view count. This is a stronger, more
  principled baseline than a bare success-rate comparison, and it is exactly
  the {h,v}-sphere-as-uniform-baseline instinct in the branch questions,
  formalized. Caveat below under "does NOT transfer": it needs restating for
  a coverage task, not a single-hit task.
- **A controlled, ablated, quantified distractor-count penalty on LLM choice
  accuracy** (Table 1, Sec 4.3): shorter presented sets improve choice
  accuracy by 6–30pp depending on scorer and difficulty bucket, at matched
  descriptions and matched ordering. This is real, usable evidence (not
  merely demonstrated — genuinely ablated, with variance over 3 seeds) that
  **raw candidate count degrades LLM selection even when the target is
  present**, independent of whether the extra candidates are similar to the
  target. Directly supports the "keep the space small" instinct behind RQ3
  and gives it a number: at N=370 with BM25, going from K≈2 to K=5 costs
  ~6pp choice accuracy overall and ~30pp on the hardest-to-rank-correctly
  subset (with a stronger scorer).
- **Self-pruning as an alternative "how many is enough" stopping rule**
  (Sec 3.3): a chance-corrected reward that decays mathematically with depth
  requires no hand-tuned penalty and produces genuinely adaptive stopping
  (K ranges 1.4→80.7 across conditions with zero per-condition tuning,
  Sec 5.1). Worth handing to the stopping-criterion RQ as a second family
  alongside Explore-until-Confident's conformal calibration — cheaper to
  implement (no calibration set needed) but requires the R_q=1/known-corpus
  assumptions below.
- **The F1-ablation methodology itself is a reusable pattern**: to claim
  "our metric causes adaptive behavior," compare against a depth-aware but
  *not* chance-corrected baseline (F1=2/(K+1)) at matched conditions,
  showing the naive depth penalty stays flat while the chance-corrected one
  adapts to N/scorer/difficulty (Sec 5.1, third observation). Reusable
  template for any future benchmark of "does this stopping rule actually
  respond to the problem, or just to a fixed penalty."

## What does NOT transfer, and why

- **Zero vision content.** No images, no camera poses, no VLMs — every
  branch RQ about ego/allo framing, view encoding (integers vs. names),
  render sparsity, marked renders, caption hand-off, etc. is untouched by
  this paper. Do not cite it for anything spatial; it is purely a text/
  retrieval-depth paper mistakenly adjacent-sounding to "tool count" because
  our action-space items are also discrete named options.
- **Single-hit retrieval framing does not match a coverage task.** Our
  problem is "select a *set* of views that jointly cover the object's
  surfaces orthogonally," closer to Recall-of-a-set than Success@K-for-one-
  item. BoR's math explicitly supports Recall@K in principle ("BoR applies
  to other success rules as well, e.g., Recall@K, but all experiments in
  this paper use the ≥1 rule," Sec 3.1) but **the paper never runs that
  variant** — every reported number uses the "found at least one relevant
  item" rule. Reusing BoR for our coverage task would require our own
  re-derivation and re-validation of the Recall@K/multi-hit case; nothing
  here is directly reusable as measured.
- **No evidence on similarity/distinctiveness as a factor separate from
  count.** The paper's own related-work section cites LongFuncEval and
  Rabinovich & Anaby-Tavor for the claim that "function-calling accuracy
  degrades as tool catalogs grow or as semantically similar tools are
  added" (Sec 2.2) — but this is a **citation to other papers**, not a
  result BoR itself produces. The selection rationale for this document
  ("entry point to the tool-scaling literature showing semantic overlap
  hurts as much as raw count") over-states what this specific paper shows:
  it demonstrates a count effect cleanly (Table 1) but explicitly does not
  isolate a similarity effect. For RQ9 (count vs. distinctiveness), this
  paper only answers the count half; the distinctiveness half needs its own
  primary source (candidates: LongFuncEval, RAG-MCP [N: 1→11,100],
  Rabinovich & Anaby-Tavor — cited but not read for this pass).
- **No ordering-effect data.** Order is held constant at scorer-rank
  throughout; the branch's "does the benchmark need permuted action
  orderings" question gets no evidence either way from this paper — flag as
  an evidence gap, not a null result, consistent with what MapGPT also could
  not supply.
- **N regime is a poor match at the small end.** Even the smallest tested
  condition (MetaTool N=20) is comparable to our upper bound (order 10–30
  candidate views); most conditions (100–3,251, plus 3,633–8.8M in the
  appendix) are 1–5 orders of magnitude larger than our action space. Per
  Sec 3.1's own logic ("N small → P_rand rises quickly with K"), the paper's
  central selling point — large selectivity gains from adaptive depth —
  should be expected to shrink or vanish at our scale; Table 2's N=20 row is
  the weakest adaptive-depth margin in the whole tool-selection table,
  consistent with that expectation, though the paper does not discuss this
  trend explicitly.
- **Downstream validation is single-shot, not sequential/adaptive.** The
  Sec 4.3 experiment is "present K items once, force one choice" — it
  cannot speak to multi-turn view planning where later choices depend on
  earlier ones (our actual RQ7/RQ8/RQ12 territory: chicken-and-egg render
  selection, on-demand/hierarchical exposure, coverage-tracking across
  turns).
- **No stated failure modes of any spatial/perceptual kind** — Sec 5.3's
  limitations are about experimental scale (single-seed MetaTool, DQN vs.
  production RL, tool-choice-vs-argument-correctness scope) and are not
  usable for the Branch-2 named-failure-modes handoff list.

## Evidence-table entry (for the cross-paper table)

| Paper | Encoding compared against | Delta | Variance reported? |
|---|---|---|---|
| How Many Tools (2605.24660) | Presented-set size K (BoR-selected, K≈2.2) vs Fixed K=5, identical descriptions/ordering, BFCL+BM25, Claude Sonnet 4.6 forced choice (Table 1, Sec 4.3) | +6.0pp choice accuracy (93.1 vs 87.1), +8.5pp end-to-end offset by lower presentation rate | Yes (±std, 3 seeds; FK=5 deterministic) |
| How Many Tools (2605.24660) | Same, embedding scorer, medium-difficulty bucket only (gold ranked 2nd–5th) | +30pp choice accuracy (80.1 vs 50.0) | Partial (BoR has std; FK=5 deterministic single run) |
| How Many Tools (2605.24660) | BoR reward vs F1=2/(K+1) reward (chance-corrected vs. flat depth penalty), MetaTool N=20/50/100, embedding scorer (Table 2) | Found% +6–11pp for BoR at comparable-or-lower K; F1 stays ~63–64% found flat across N while BoR rises 70.6→74.7% | No (MetaTool is single-seed) |
| How Many Tools (2605.24660) | BM25 vs MiniLM vs BGE scorer, identical BoR reward, MetaTool (Sec 4.2, Fig. 2) | Learned K: 80.7 (BM25) vs 2.3 (MiniLM) vs 2.4 (BGE) — scorer quality, not architecture, drives depth | No |
| How Many Tools (2605.24660) | Semantic-similarity-driven degradation | **Not tested by this paper** — cited from LongFuncEval / Rabinovich & Anaby-Tavor (Sec 2.2), no primary data here | N/A — flag as citation, not evidence |
| How Many Tools (2605.24660) | Option-order effects | **Not tested** — order held fixed at scorer rank throughout (Sec 4.3) | N/A — evidence gap |

## Random-baseline formalism (for the random/uniform-baseline RQ)

`P_rand = 1 - C(N-R_q,K)/C(N,K)` → `K/N` for single-relevant queries (Eq. 1).
`BoR = log2(P_obs/P_rand)` (Eq. 2). `BoR_opt(K) = log2(N/K)` is a
method-independent ceiling given only corpus size and depth (Sec 3.1). This
is directly reusable machinery for stating our benchmark's "beats-random"
number, once (a) success is redefined for a coverage task rather than a
single-hit task, and (b) N is set to the true candidate-view count of
whatever action space design wins (RQ2/RQ3), not the current 12×3=36-cell
{h,v} sphere by default.

## Relevance note

Answers the branch's tool-count-degradation RQ only partially and with an
important correction to the selection rationale: it gives a clean, ablated,
variance-backed number for **raw candidate count** hurting LLM choice
accuracy (up to 30pp on hard-to-rank subsets) and a reusable chance-corrected
random-baseline formalism — but it does **not** itself show that semantic
overlap/near-duplicate distractors hurt independently of count (that claim
is cited from other papers, not measured here), it never tests option
ordering, and its single-hit retrieval framing (find the one right tool)
does not match our set-coverage view-selection task. Use it for: the
distractor-count-hurts-choice number, the random-baseline math, and as one
data point (alongside Explore-until-Confident) in the stopping-criterion
literature — not as evidence for the count-vs-distinctiveness question
itself, which still needs a dedicated source.
