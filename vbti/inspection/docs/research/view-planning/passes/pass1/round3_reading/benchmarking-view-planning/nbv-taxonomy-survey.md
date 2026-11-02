# Alsadik, Mahdi, Abdulateef — "A structured review and taxonomy of next-best-view strategies for 3D reconstruction"

ISPRS Open Journal of Photogrammetry and Remote Sensing 17 (2025) 100098. https://doi.org/10.1016/j.ophoto.2025.100098
Full text obtained (ScienceDirect blocks scraping with Cloudflare 403; PDF pulled via Unpaywall → University of Twente repository mirror: `https://ris.utwente.nl/ws/files/504240448/1-s2.0-S2667393225000171-main.pdf`). 21 pages, read in full, all figures/tables/bibliography reviewed. CC-BY, no external funding declared.

## Core method/idea

This is a **taxonomy/categorization review**, not a benchmark or empirical paper. It systematically searched Scopus/Web of Science/IEEE Xplore/Google Scholar (>150 candidates screened, 100 selected) and sorted the NBV literature into five non-overlapping strategy families (Fig. 2, Section 2):

1. **Rule-based** (§2.1) — predefined heuristics/visibility rules, `v* = argmax_v G(v)`, `G(v) = Σ_c G_c(v)` (Eq. 2–3, Algorithm 1). Pros/cons per Table 3: "Efficient, simple, task-specific" vs. "Inflexible, limited in unstructured environments." **Example application listed: "Surface inspection, structured environments."**
2. **Uncertainty-based** (§2.2) — selects views maximizing expected entropy reduction, `ΔU(v) = max(0, H_before − H_after(v))` (Eq. 4–5, Algorithm 2). Subtypes (Fig. 5): information-theoretic, probabilistic modeling, neural-rendering uncertainty (NeRF/NeVF/BavsNeRF), active-learning (E-criterion), semantic-aware.
3. **Sampling-based** (§2.3) — random/directed candidate generation then scored (Eq. 6, Algorithm 3). Praised for scalability/flexibility/robustness against local minima; weak point is compute cost.
4. **Learning-based** (§2.4) — supervised, RL (Eq. 7, Algorithm 4), regression (CNN), zero-shot, semantic-driven, hybrid-with-uncertainty (Fig. 7).
5. **Prediction-based** (§2.5) — uses a prediction model (shape completion, coverage anticipation) to estimate unseen regions and pick views that will reveal them (Eq. 8–9, Algorithm 5): `G(v) = Coverage(U(v; P))`. SCONE and MACARONS are called out as state of the art here.

A cross-cutting axis (Table 1, p.3, independent of the five-family taxonomy) is **model-based vs. free-model-based NBV**: model-based methods "select views by initially defining models or structures of the environment... assume at least partial or complete knowledge about the environment's geometry is available," used for "inspecting targeted objects, recognizing objects, or mapping structured environments," but "often lack flexibility in unpredictable or dynamic environments... rely on previous information." Free-model-based methods impose no prior model, adapt continuously, cost more compute. **Our project's setting (survey cloud known in advance, primitive fitting, deterministic per-primitive rules) is a model-based, rule-based NBV system in this taxonomy** — which is exactly the family the survey says fits "surface inspection, structured environments" best.

Section 3 catalogs 20 open-source NBV libraries with GitHub links, one per named method, organized by the same five categories (e.g., `nbvplanner`/Bircher et al. 2016 for sampling; `PC-NBV`/Zeng et al. 2020b, `NBV-Net`/Mendoza et al. 2018, `ScanRL`/Peralta et al. 2020 for learning-based; `SCONE`/Guédon et al. 2022, `MACARONS`/Gu'edon et al. 2023 for prediction-based).

## Concrete results and numbers

**None.** This paper reports zero quantitative benchmark results, no coverage percentages, no reconstruction-error values, no view-count numbers, no comparison tables of method performance. It contains algorithmic objective functions (Eq. 1–9) and pseudocode (Algorithms 1–5) but these are generic templates, not results from running any method. The only "tables" are: Table 1 (list of references split model-based vs. free-model-based, no numbers), Table 2 (surface-based vs. volume-based method references with one-line descriptions, no numbers), Table 3 (qualitative pros/cons/example-application comparison of the 5 strategies, no numbers). Confirmed by reading every page — no figure or table contains a metric value, dataset name with size, or a reported accuracy/coverage/error number anywhere in the body text.

## Load-bearing assumption behind why this document has no numbers

The paper's own inclusion criterion (§1, p.2): papers were kept only if "(1) the study must explicitly focus on NBV or viewpoint planning; (2) the method should be applied to 3D scene understanding, reconstruction, or exploration; (3) implementation details or evaluation metrics must be provided." **This confirms the task brief's framing exactly**: this survey pre-filtered ~100 papers that *do* report evaluation metrics/protocols, and organized them into 5 families + subfamilies with one-line descriptions — but it does not itself extract or synthesize what those metrics/protocols/ground-truth strategies *are*. It is a **triage index into the bibliography**, not a source of harvestable evaluation-methodology content in its own right. The actual protocol/metric/GT-strategy harvesting this branch needs has to happen at the level of the individual cited papers, not this spine document.

## What transfers to our setting

- **Vocabulary/classification**: our proposed system (primitive fit from PLY → deterministic per-primitive view rules) is unambiguously "model-based, rule-based NBV" in this taxonomy (Table 1 + §2.1), and the survey's own characterization validates the direction: rule-based is the right family for "structured environments" / "surface inspection" (Table 3, Example application column) — directly matches our tabletop-inspection use case. This is a useful framing citation for our own writeup, not new information for the RQs themselves.
- **Named index leads worth follow-up reads** (from the reference list, pp.19–21, cross-checked against Table 1/2/Fig.4–8 categorization), ranked by relevance to this branch's questions:
  - **Karaszewski et al. 2016**, *"Assessment of next-best-view algorithms performance with various 3D scanners and manipulator,"* ISPRS J. Photogramm. — listed in Table 2 as a rule-based/geometric-rules reference (Fig. 4). Description explicitly says it "evaluated the performance of various NBV algorithms on a number of different 3D scanners and manipulators" — this reads as an actual **comparative benchmark paper** and is the single highest-priority lead this survey surfaces for Q5 (baselines) and Q12 (statistical treatment). Not yet read — flagged for a dedicated pass.
  - **Trummer et al. 2010**, extended E-criterion online NBV — accuracy-driven heuristic evaluation; relevant background for Q7 (per-primitive analytical optimum) though the E-criterion itself is Branch-1 (metric formula) territory.
  - **Vasquez-Gomez et al. 2014**, probabilistic NBV robust to "robot positioning error" (both sensor and robot pose uncertainty modeled) — directly relevant to Q9 (injected pose-error sensor defects) as an existing precedent for modeling positioning error in an NBV evaluation.
  - **Zeng et al. 2020b (PC-NBV)** and **Liu et al. 2023** (sampling-based NBV + Point Cloud Completion Network) — both take **point clouds as the native input representation** and, in Liu et al.'s case, use PCN-predicted-shape confidence as the view-selection signal. This is a concrete precedent for Q4's "abandon Chamfer/coverage for a mesh-free metric" exit: PCN confidence-on-partial-cloud is an alternative ground-truth-free proxy worth comparing against a per-region view-angle metric.
  - **Vasquez-Gomez et al. 2021**, 3D-CNN NBV regression — trains against some notion of "best view" ground truth; worth checking their GT construction method for Q2.

## What does NOT transfer, and why (negative findings — first class per instructions)

- **No VLM/LLM content at all.** Searched the full text: "vision-language," "VLM," "LLM," "language model" do not appear anywhere in this paper. The learning-based section (§2.4) covers only classical deep learning (CNN regression, RL, supervised labels). **This survey provides zero evidence for Q16 (VLM spatial failure modes), Q17 (LLM-judge controls), or how VLM-active-perception papers report success protocols** — that branch question must be answered from a different, more recent source; this ISPRS survey (accepted Aug 2025, covering literature largely pre-2025) simply predates/excludes the VLM-orchestrator paradigm our project uses.
- **No ground-truth-for-best-view discussion.** No mention of how any reviewed method obtains its GT "best view" (CAD-derived analytical normal vs. dense pre-scan vs. human label) — Q2 and Q4 are untouched here.
- **No sensor-defect injection methodology.** No discussion of synthetic defect injection (pose error, dropout, fused-vs-single-view offset) for evaluation purposes — Q9, Q10 untouched.
- **No statistical treatment.** No mention of trial counts, effect sizes, confidence intervals, or object/trial scaling — Q12, Q13 untouched.
- **No symmetry/view-equivalence discussion** — Q8 untouched.
- **No metrology/photogrammetric-network-design connection.** No mention of VDI/VDE 2634, ISO 10360-13, or network-design criterion matrices — Q21, Q22 remain genuinely unsearched; this survey does not bridge into the metrology/photogrammetry-standards literature despite being published in a photogrammetry journal.
- **No object/dataset catalog.** No named benchmark datasets or object sets are given anywhere (not even generically) — Q11 untouched.
- Because the whole document is deliberately abstracted to the level of five strategy families with generic pseudocode, essentially **every one of the 22 branch questions that asks "what does a specific paper do/report" is unanswerable from this text** — they all require going one level deeper into the ~100 individual references.

## Citations (for checkability)

- Inclusion criteria and 3-database/100-paper methodology: §1, p.2, para. beginning "After identifying over 150 potential publications..."
- Five-strategy taxonomy figure: Fig. 2, p.3.
- Model-based vs. free-model-based definitions and reference table: p.3, para. "In the domain of NBV strategies..." + Table 1.
- Rule-based objective/algorithm: Eq. 2–3, Algorithm 1, §2.1 (pp.4–6).
- Uncertainty-based objective/algorithm: Eq. 4–5, Algorithm 2, §2.2 (pp.7–9).
- Sampling-based objective/algorithm: Eq. 6, Algorithm 3, §2.3 (pp.9–11).
- Learning-based objective/algorithm + 7 subcategories: Eq. 7, Algorithm 4, §2.4 (pp.12–14).
- Prediction-based objective/algorithm: Eq. 8–9, Algorithm 5, §2.5 (pp.14–15).
- Open-source library index: §3 (pp.16–17).
- Strategy comparison table (pros/cons/example application): Table 3, p.18.
- Karaszewski et al. 2016 pointer: Table 2, p.4 (row: "Karaszewski et al. (2016) — 3D Scanner Assessment"); full ref p.20.
- Liu et al. 2023 PCN description: §2.3, p.10, para. "Liu et al. (2024) [sic, cited as 2023 in refs], a sampling-based path planner for NBVs is presented... utilizes NBV planning in conjunction with shape completion using a Point Cloud Completion Network (PCN)... confidence score can be used to choose candidate views."
- Vasquez-Gomez et al. 2014 pointer: §2.2, p.7, para. "Vasquez-Gomez et al. (2014) described the NBV planning problem in scenarios confronted with sensor positioning errors..."
- No-funding declaration and AI-writing-tool disclosure: end matter, p.19.

## Assessment for the branch

Low direct yield, but legitimate as the "spine" it was chosen to be: it costs one read to confirm the field's vocabulary (rule-based/model-based = our track) and to surface a short, prioritized list of individual papers (Karaszewski et al. 2016 above all) that must be read next to actually answer Q5/Q7/Q9/Q12. It should not be cited as a source for any metric, dataset, GT-strategy, baseline, or statistical claim — none exist in the text.
