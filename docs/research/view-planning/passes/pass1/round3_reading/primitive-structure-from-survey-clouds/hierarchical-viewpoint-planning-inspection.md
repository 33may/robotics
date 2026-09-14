# Hierarchical Viewpoint Planning for Complex Surfaces in Industrial Product Inspection (IEEE/ASME T-Mech, 2024)

**DOI:** 10.1109/TMECH.2023.3340312 | **IEEE Xplore:** https://ieeexplore.ieee.org/document/10375828/
**Venue:** IEEE/ASME Transactions on Mechatronics, vol. 29, no. 5, pp. 3289–3299. Early access 2023-12-28, issue date 2024-10.
**Authors:** Shaohu Wang, Yuchuang Tong, Xiuqin Shang, Zhengtao Zhang — Institute of Automation, Chinese Academy of Sciences (CASIA) (confirmed via OpenAlex author affiliations, not inferred).

## Access status and sourcing (read this before trusting any claim below)

A prior pass on this document failed outright (see git history of this file) —
IEEE Xplore, Semantic Scholar, Unpaywall, ResearchGate, CORE, and Crossref all
returned bibliographic metadata or nothing. This pass got further, but **the
paper's own body text, figures, and result tables were still never read
directly** — IEEE Xplore is paywalled past the abstract, and IEEE/ASME
Transactions on Mechatronics has no legal open-access copy (confirmed again
this pass). Two substitute sources were used instead, and every claim below
is tagged with which one it came from:

1. **[IEEE-abstract]** — the public abstract and first paragraph of Section I
   ("Industrial product appearance inspection refers to surface observation
   and defect inspection..."), fetched directly from the IEEE Xplore page.
   Page range (3289–3299) and date confirmed against Crossref/OpenAlex.
2. **[BLOG]** — a Chinese-language section-by-section reading-notes post
   ("论文精读", CSDN, published 2025-07-28, https://blog.csdn.net/weixin_46361119/article/details/149401071)
   that walks through the paper's structure, transcribes/describes its
   numbered equations (1)–(18) in words, names its section headers verbatim
   (matching the IEEE table-of-contents structure exactly — Problem
   Definition, Hierarchical Viewpoint Planning Approaches parts A–D,
   Experiment parts A–E), names its three evaluation datasets and seven
   metrics, and describes what each figure/table shows qualitatively.
   **This is a third-party secondary source, not the paper.** It transcribes
   equation *meaning* and section *structure* faithfully enough to be useful,
   but does **not** transcribe the numeric contents of any results table —
   those are described only qualitatively ("our method was best on combined
   efficiency and accuracy"), because the blog's tables/figures are embedded
   images the author did not OCR.
3. **[COMPANION]** — the full text (via legitimately downloaded PDF) of a
   2025 systems paper by an overlapping author subset from the same lab:
   T. Yang, S. Wang, Y. Tong, M. Zou, X. Shang, W. Ma, Z. Zhang, "Human-like
   Observation-inspired Universal Image Acquisition System for Complex
   Surfaces in Industrial Product Inspection," *Machine Intelligence
   Research*, vol. 22, no. 4, pp. 677–695, 2025, DOI 10.1007/s11633-025-1561-5
   (open access at mi-research.net). Its Section 4.3.2 explicitly names our
   target paper as reference **[17]** and states verbatim: "we have developed
   a hierarchical view planning method based on the divide and conquer[17]"
   — i.e. this companion paper **reuses the exact algorithm from the target
   paper as its viewpoint-generation module**, and reports quantitative
   comparison numbers for it (Table 2, Table 4) in a different experimental
   harness than the target paper's own. These numbers characterize the
   target paper's method as deployed by its own authors, but are **not the
   target paper's own published numbers** — no claim below conflates the two.
   The companion paper's reference [16] is also identified: S. Wang, F. Qin,
   Y. Tong, X. Shang, Z. Zhang, "Probabilistic Boundary-Guided Point Cloud
   Primitive Segmentation Network," IEEE TIM, vol. 72, 2023 (DOI
   10.1109/TIM.2023.3322509) — this is the upstream primitive-segmentation
   method the target paper's "multilevel geometric representation" stage
   depends on (confirmed via its own abstract, see below).

Net effect: this note can describe the method's structure, equations
(by number and meaning), evaluation protocol, and *some* real numbers (from
the companion's reuse) with reasonable confidence, but **cannot cite a single
number from the target paper's own Tables I–V**, and cannot verify anything
about its actual figures beyond the blog author's description.

## Core method (per [BLOG] structure + [IEEE-abstract])

**Problem formulation (Section III, "Problem Definition and Task Model"):**
A viewpoint `v` is `(p^v, e^v)` — 3D position + normal — in object frame `{O}`.
A point cloud point is `s_i = (p_i, n_i)`. Transformed into the viewpoint's
own frame `{VP}`, coverage of point `s_i` by viewpoint `v` requires (a) the
transformed point lies within the viewpoint's field-of-view/depth-of-field
frustum (Eq. 2) and (b) a visibility/normal condition — the point must not be
back-facing (Eq. 3). A "coupled point" `(s_i, v_i)` pairs each 3D point with
its best viewpoint (Eq. 4); Eq. 5 adds the constraint needed to recover a
full 6-DoF pose (position alone under-constrains orientation). The overall
viewpoint-planning objective (Eq. 6) is a three-part minimization: (i)
minimize the number of viewpoints needed for full task coverage, (ii)
maximize pose quality by penalizing deviation from the ideal orthogonal
normal alignment, (iii) subject to every point being covered by at least one
viewpoint. **This is precisely a set-cover-style formulation with an added
orthogonality term** — directly the same shape as our project's implicit
objective (minimal, orthogonal, complete view set).

**Multilevel geometric representation / primitive segmentation (Section
IV-A):** the point cloud is first segmented into **regular primitives —
plane, sphere, cylinder, cone** — plus a residual **irregular** category, via
the authors' own prior method (companion paper's ref [16], PBPS — see below).
[BLOG] figure 4 is described as showing segmentation results on real
objects; regular vs. irregular receive categorically different downstream
treatment (not a single unified fitter).

**Bottom-up, for regular primitives (Section IV-B):** each regular primitive
type gets an explicit parametric surface equation, and the primitive's
*bounded* surface is **gridded** (rasterized into cells) — grid cell size is
derived (Eqs. 8–9, per [BLOG]) from the camera's field-of-view and depth-of-
field, i.e. sensor optics directly set the sampling density of the surface
grid. Eq. 10 unifies all primitive types' point sets into one common
gridded-surface representation so the rest of the pipeline is
primitive-type-agnostic downstream of this step. **This directly answers the
branch's "bounded face" question**: the bottom-up path does not fit infinite
planes/cylinders — it produces a bounded, gridded patch per primitive with
one viewpoint generated per grid cell, each orthogonal to the local primitive
surface by construction.

**Top-down, for irregular primitives (Section IV-C), three stages:**
1. **Hybrid Viewpoint Confidence Representation (HVCR)**, Eqs. 14–16 combined
   in Eq. 17: (a) Eq. 14 — boundary-edge probability density `b_i` (from the
   segmentation stage) multiplied by local surface curvature `κ_i` (computed
   via Open3D), rewarding viewpoints that see edges/high-curvature regions
   well; (b) Eq. 15 — raw count of 3D points a candidate viewpoint covers;
   (c) Eq. 16 — normal-alignment term: dot product between the viewpoint's
   normal and each covered point's normal, rewarding near-antiparallel
   (i.e. orthogonal-incidence) alignment, zeroed out via a max() when the
   angle is acute (wrong side). These three terms combine into one scalar
   confidence per candidate viewpoint (Eq. 17) used to accept/reject it as a
   candidate.
2. **Dynamic NBVsearch** — a neighbor-search restricted by a distance-ratio
   test (Eq. 18: ratio of computed inter-viewpoint distance to actual
   distance) so that HVCR does not have to be recomputed for every remaining
   candidate at every step — an explicit computational-cost mitigation for
   what is otherwise the classic "recompute confidence for all remaining
   viewpoints" bottleneck the companion paper's related-work section
   attributes to prior NBV-style methods ([COMPANION] §2.3, citing refs
   [37],[39],[40] as suffering exactly this cost).
3. **Dynamic viewpoint pose optimization ("FOV-Shift")** — starting from an
   initial viewpoint, iteratively intersect with NBV-search neighbors,
   recompute covered-point count, and repeat until the covered set stops
   growing; the paper is said to include pseudocode for the resulting
   best-viewpoint-set algorithm ([BLOG], described as following a figure the
   blog author declined to paraphrase in detail).

**Viewpoints merging and path planning (Section IV-D):** merges the
bottom-up and top-down viewpoint sets and runs a path-planning step over
them. [BLOG]'s own annotation states this part is explicitly **not the
paper's research focus** ("这一部分不是本文研究重点") — i.e. even the
paper itself treats final trajectory sequencing as a secondary, bolted-on
step rather than a core contribution.

## Evaluation protocol (Section V, per [BLOG]; no numeric values obtained from the target paper itself)

- **Simulated evaluation**, three dataset sources: **"Synthesis"** (not
  further identified — likely an author-generated synthetic set, unverified),
  **ABCParts** (subset of the ABC CAD dataset, Koch et al., CVPR 2019 — this
  citation is independently corroborated: [COMPANION] cites the identical
  ABC dataset, ref [68], when it reuses this method), and **T-LESS**
  (Hodaň et al.'s texture-less real industrial-parts pose-estimation
  dataset — real scanned CAD-backed objects, not raw survey scans).
- **Seven metrics** (per [BLOG]): `Crate` (surface coverage rate), `Vnum`
  (number of generated viewpoints), `Pleng` (total path length), `Nvc`
  (view-direction/normal consistency across the viewpoint set — *not* a
  cross-rescan repeatability metric, see below), `Ptime` (planning runtime),
  `Srate` (success rate), `Etime` (total platform execution time).
- **Comparison experiments**: multiple baseline viewpoint-planning methods
  per dataset; [BLOG] states only that the paper's method wins on combined
  efficiency+accuracy — no numbers transcribed.
- **Ablation study**: isolates HVCR, Dynamic NBVsearch, and FOV-Shift;
  [BLOG] states all three together give best performance (again, no numbers).
- **Robustness checks**: sensitivity to field-of-view size (method reported
  to favor/tolerate small FOV) and to point-cloud density (method reported
  robust to density variation) — qualitative only, per [BLOG].
- **Real-world experiments**: two physical platforms — a 5-axis stage and a
  6-DoF robot arm — tested on a **phone** and a **turbine/compressor blade**,
  said to show reduced redundant viewpoints/motions and more complete
  corner/edge imaging versus baselines. No numbers obtained.

## Concrete numbers that DO exist — but from [COMPANION]'s reuse, not the target paper

These characterize the exact algorithm (cited as ref [17] = this paper) as
redeployed by its own authors in a different harness — useful as a lower
bound on real-world performance, but **not substitutable for the target
paper's own reported numbers** when checking specific branch claims.

**[COMPANION] Table 2 — viewpoint planning on 40 complex ABC-dataset point
clouds, 10 repeated trials, averaged, vs. sampling/clustering baselines**
(K-Greedy [42], P-Greedy [37], TCluster [46]):

| Method | Vnum | Etime (s) | ACrate (%) |
|---|---|---|---|
| K-Greedy | 39.0 | 127.75 | 92.80 |
| P-Greedy | 35.0 | 114.25 | 91.15 |
| TCluster | 29.0 | 97.75 | 83.95 |
| **Ours (=this paper's method)** | **25.0** | **82.75** | **96.45** |

Text states a "15.34% improvement" in efficiency over TCluster and the
highest coverage match rate (96.45%) vs. the prior best of 92.8%.

**[COMPANION] Table 4 — real-object viewpoint planning, 5 objects, 100%
coverage achieved on all**:

| Object | Vnum | Crate | RepeatCrate |
|---|---|---|---|
| iPhone frame | 11 | 100% | 10.43% |
| Automotive part | 44 | 100% | 41.72% |
| Irregular part 01 | 47 | 100% | 44.83% |
| Irregular part 02 | 63 | 100% | 46.17% |
| Irregular part 03 | 60 | 100% | 47.31% |

Min viewpoints = 11 (simple regular object), max = 63 (most irregular
object) — a >5x spread driven entirely by shape complexity, and
`RepeatCrate` (redundant/overlapping coverage) stays under ~50% across
all objects, framed by the authors as evidence of efficient, low-redundancy
coverage.

Note: the companion paper's own upstream **normal-vector correction**
numbers (its Table 1, "Coulomb force" method, AE=0.04 rad on regular
objects / 0.09 rad on complex objects / 0.13 rad overall) are a
**preprocessing contribution of the companion paper itself**, not part of
the target paper's method — they are the *input quality* the target paper's
viewpoint algorithm is fed in this reuse, not a result the target paper
reports. Flagged here only because it's directly relevant to Branch 1's
angular-error-vs-noise question, with the explicit caveat that it is a
different paper's number.

## Load-bearing assumptions (why this pipeline works, and what it needs to keep working)

- **Primitive segmentation is a trained, supervised network, not RANSAC.**
  [COMPANION]'s ref [16] (Wang et al., IEEE TIM 2023, "Probabilistic
  Boundary-Guided Point Cloud Primitive Segmentation Network," own abstract
  confirmed via OpenAlex) uses a Gaussian-distributed boundary-probability
  representation, a learned boundary-guided feature-fusion module, and a
  primitive instance contrastive loss — trained and validated on "two
  CAD-based datasets and two real-scene datasets." This is a **learned**
  front end requiring labeled training data, not an off-the-shelf
  Open3D/CGAL/PCL RANSAC call. This directly bears on the branch's question
  about off-the-shelf, no-retraining coverage — this lab's own answer is no,
  they built and trained a custom segmentation network for this purpose.
- **Sensor optics set primitive granularity.** Grid cell size for the
  bottom-up path is derived directly from FOV/depth-of-field (Eqs. 8–9 per
  [BLOG]) — the decomposition is not sensor-agnostic; porting this requires
  redoing that derivation for the D405's own optics.
- **High-quality point normals are assumed as input.** The companion
  system's own architecture (its Fig. 4, "Software algorithm architecture")
  places point-cloud preprocessing with explicit normal-vector correction
  (KNN, distance-weighted, Coulomb-force methods; [COMPANION] §4.2) *before*
  primitive segmentation and viewpoint generation — implying the pipeline
  this paper's method sits inside treats normal quality as a first-order
  bottleneck worth a dedicated correction stage, consistent with our own
  Branch 1 concern about noisy normals from a defective survey cloud.
- **Binary regular/irregular split, no explicit uncertainty or fallback
  tier.** Every primitive is routed to exactly one of two fixed strategies
  (bottom-up gridization or top-down HVCR) based on its segmentation label;
  there is no described mechanism for detecting a *failed* or *unreliable*
  segmentation/fit and falling back to something coarser (e.g. our
  contract's "uniform sphere" fallback). Irregular geometry gets a
  different *algorithm*, not a degraded/safer one.
- **Coverage-oriented, not orientation-covariance-oriented.** The entire
  optimization (Eq. 6) and confidence formulation (Eq. 17) are deterministic
  point-estimate quantities (counts, dot products, curvature) — nothing in
  what was recovered treats primitive-fit orientation as a random variable
  with covariance, and nothing propagates per-point sensor noise or
  registration error into it.

## What transfers to our setting

- **Architectural precedent, at production scale, for exactly our accepted
  design**: primitive-level deterministic rules (bottom-up gridization per
  regular primitive) driving orthogonal per-cell viewpoints is validated
  here across two papers by the same lab, on real industrial hardware
  (UR5e + turntable + industrial camera — notably the *same UR5e* family
  robot as our platform), not just in simulation. This is strong existence
  proof that "primitive decomposition → deterministic per-primitive view
  rule" is a workable, shippable pattern, matching our framing doc's
  explicitly accepted non-goal exception.
- **A concrete, checkable, non-learned view-quality score usable for our
  RQ1** ("judging a view"): HVCR (Eq. 17) is fully computable from an
  off-the-shelf pipeline — Open3D-computed local curvature × a boundary/edge
  probability × a viewpoint-normal-to-surface-normal dot product — without
  needing the rest of the paper's machinery. This is a legitimate candidate
  building block for our own analytical/optimization track, independent of
  whether we adopt their segmentation network.
- **Direct, positive answer to "does anyone do object→primitives→per-primitive
  viewpoints"**: yes, unambiguously, at the exact granularity our contract
  cares about (plane/sphere/cylinder/cone bottom-up rules), which is the
  single most direct precedent found for this branch's core question. The
  "irregular" top-down path is the nearest precedent for what to do when
  primitive decomposition *doesn't* close the object — but note its
  substitute is a different algorithm class (confidence-driven NBV search),
  not a coarser primitive (like our uniform-sphere fallback) or a different
  vocabulary (like superquadrics).
- **Gridization-of-bounded-primitive-surface is a usable template** for
  producing "face center + face size" rather than an infinite plane
  equation — directly responsive to the branch's bounded-face question,
  even though the exact grid-cell-to-viewpoint mapping would need adaptation
  to our sensor and reachability constraints.

## What does NOT transfer / negative findings (first-class)

- **No defect-mode analysis of any kind was found.** Nothing in what was
  recovered — abstract, blog walkthrough, or companion full text —
  discusses registration/fusion error across multiple views, depth dropout
  on glossy/dark surfaces, or systematic depth bias. The evaluated point
  clouds are CAD-derived (ABC), an unspecified "Synthesis" set, and T-LESS
  (real but curated, texture-less parts) plus turntable-captured real
  objects under a controlled, purpose-built rig (industrial lighting,
  dedicated positioning camera, hand-eye calibrated). This is a much
  cleaner acquisition setting than our single eye-in-hand D405 fused survey
  cloud with a measured ~4 cm fused-vs-single-view offset — **this paper
  cannot answer any of Branch 1's noise/dropout/registration-covariance
  questions**, and its authors give no indication they consider it.
- **Primitive segmentation requires training, contradicting the branch's
  "off-the-shelf, no retraining" preference.** See load-bearing assumptions
  above — this is a hard negative for the specific RQ about whether
  Open3D+CGAL/PCL alone (no training) covers box/cylinder recovery the way
  this paper's pipeline does; this paper's own answer is that off-the-shelf
  RANSAC-family fitting was apparently judged insufficient by this lab
  (they built a learned segmentation network instead), though no direct
  RANSAC-vs-their-method ablation was found to confirm *why*.
- **No superquadrics, no cuboid decomposition — but also no explicit
  justification that plane/sphere/cylinder/cone is sufficient for chamfered/
  rounded real parts.** The "irregular" bucket is presumably where such
  parts land, handled by switching planning *strategy*, not by adding a
  richer primitive vocabulary. This only partially answers the branch's
  vocabulary question: the answer from this paper is "don't extend the
  vocabulary, extend the algorithm," which is a real design alternative
  worth weighing against superquadric/cuboid approaches from other papers
  in this same folder (SuperDec, SuperFlex, Yang & Chen cuboid abstraction),
  not a resolution of the question on its own.
- **No re-scan repeatability / primitive-identity-stability study.** `Nvc`
  (normal-vector/view-direction consistency) is a within-single-run metric
  over the generated viewpoint set, not a metric comparing primitive
  identity, ordering, or orientation across repeated scans of the same
  object. RQ3's "does face 3 stay face 3" concern is **not addressed** by
  this paper in any form found.
- **No partial-occlusion / symmetry-completion treatment for cylinders or
  any other primitive.** No mention of mirror-plane estimation, PotSAC-style
  axis completion, or any explicit handling of a half-scanned primitive with
  additional glossy-side dropout.
- **No uncertainty quantification or covariance for fitted primitive
  parameters**, and no fit-reliability score calibrated against downstream
  task failure (their HVCR confidence score is a *view-selection* heuristic,
  not a *fit-quality* certificate — it scores candidate viewpoints given an
  already-trusted primitive segmentation, it does not score whether the
  segmentation/fit itself should be trusted).
- **No benchmark-dataset overlap with a corrupted-defect-profile protocol.**
  ABC/T-LESS/"Synthesis" are clean or synthetically-clean sources; nothing
  suggests any of the three datasets were corrupted with registration
  jitter, view-dependent dropout, or depth bias matching our specific
  defect profile.

## Answers to specific branch questions (direct hits only)

- *"Is there any published system that decomposes a scanned object into
  geometric primitives and then generates per-primitive viewpoints?"* — **Yes,
  this is exactly that system**, at production/real-hardware scale, and its
  "regular vs. irregular" split with different strategies per branch is the
  closest precedent found so far for a primitive-decomposition-driven
  action space. Its substitute for primitives on irregular geometry is a
  confidence-driven top-down search (HVCR + NBV-search + FOV-Shift), not a
  richer primitive vocabulary or a mesh/skeleton representation.
- *"Which methods recover bounded faces (extent, boundary, centroid) rather
  than infinite primitives?"* — this one does, via gridization of the
  primitive's bounded surface region (Section IV-B) — but the note above on
  sourcing applies: the exact grid-to-face-descriptor mapping was not
  independently verified beyond the blog's equation-by-equation description.
- *"Does the contract's object set close with plane+cylinder+sphere+cone
  RANSAC, or do real chamfered/rounded parts force superquadrics or cuboid
  decomposition?"* — this paper's regular-primitive vocabulary is exactly
  plane/sphere/cylinder/cone, matching our contract, but it does not extend
  to superquadrics for the residual — it falls back to a different
  *algorithm* for "irregular," which is informative but does not by itself
  settle whether our chamfered/rounded real objects will segment cleanly
  into that four-primitive vocabulary or get shunted into "irregular."
- *"Is there a closed-form covariance for a fitted plane's or cylinder's
  orientation...?"* / *"How does registration/alignment covariance
  propagate...?"* / *"What computable signals do papers use to declare a fit
  unreliable...?"* — **not addressed**; none of these questions can be
  answered from this paper. Look elsewhere in the branch's source set.

## Recommendation for future access attempts

If institutional IEEE Xplore access becomes available, re-reading the
target paper directly would upgrade essentially every claim above from
"per a third-party summary" to primary-sourced, and would specifically
unlock: the actual numeric values in its own Tables I–V, whatever
"Synthesis" dataset actually is, the exact FOV-Shift pseudocode, and any
discussion (if present) of failure modes or limitations the abstract/blog
did not surface. The companion paper (open access, full PDF obtained,
saved reasoning at DOI 10.1007/s11633-025-1561-5) is a legitimate standing
resource for this branch independent of this note — it documents a fuller
production pipeline (point cloud preprocessing, primitive segmentation,
hierarchical viewpoint generation, posture alignment, motion control) with
real quantitative tables throughout, and may be worth its own separate
reading-notes file if the branch wants a second, more instrumented
data point on the same lab's approach.
