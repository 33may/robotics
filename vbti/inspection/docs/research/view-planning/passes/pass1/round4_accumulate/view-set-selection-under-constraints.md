# Branch 4 — Accumulated Synthesis
## Deriving the view set: coverage, selection, reachability

**Date:** 2026-08-27
**Role:** branch synthesizer, pass 1 / round 4.
**Branch plan:** [`../round1_plan.md` §Branch 4](../round1_plan.md)
**Map + questions:** [`../round2_map/view-set-selection-under-constraints.md`](../round2_map/view-set-selection-under-constraints.md)
**Reading notes:** [`../round3_reading/view-set-selection-under-constraints/`](../round3_reading/view-set-selection-under-constraints/) — 20 documents, all read.
**Contract:** [`../../../../geometry-aware-view-planning.md`](../../../../geometry-aware-view-planning.md)
**Feeds:** RQ2 (primary), RQ3 (secondary), benchmark methodology (hand-off to Branch 2).

---

## 0. Provenance and reliability

20 documents were read. Access quality varies and this matters for how hard we
can lean on each:

| Tier | Documents | Note |
|---|---|---|
| **Primary, full text** | Scott 2003, Scott 2001, Scott 2009 (green-OA ERB-1125), Peuzin-Jubert 2021, Alsadik 2025, Magaña 2023, Mosbach 2021, Gospodnetić 2022, Glorieux 2020, Spensieri 2025, Bircher 2015, Zacharias 2007, Wang 2026, Pan 2025, MA-SCVP 2024, ObjView-Bench 2026 | 16 of 20 |
| **Primary substitute** | Yu & Gupta — read the author's SFU PhD thesis (168 pp.), which is the pre-condensation source of both target papers | strong |
| **Secondary reconstruction** | Tarbox & Gottschlich 1995 — no OA copy exists; reconstructed from Scott 2003 §5.1 and Gospodnetić 2022 §2.1 | **do not cite for numbers** |
| **Secondary reconstruction** | SCVP 2022 — IEEE-paywalled; reconstructed from MA-SCVP's own recap and controlled re-runs | numbers are MA-SCVP's re-measurement, not SCVP's own tables |
| **Weak / survey-only** | CMM visibility maps (Kweon & Medeiros 1998) — genuinely closed, zero OA footprint; built from Sadaoui & Phan 2021 (open access, read in full) | **citation-chain anchor only** |

Two follow-up reads would materially strengthen the branch and could not be
obtained: **Prieto 2000** (the 35° source) and **Tarbox & Gottschlich 1995**
primary text (the actual grazing-angle numbers, the actual selected-set sizes).

---

## 1. Executive summary — what this branch now knows

**Seven findings, ordered by how much they change what we build.**

1. **The field's canonical formulation is minimum set cover over a
   measurability matrix, solved as a 0/1 ILP.** This is not contested. It is
   unanimous across 30 years, four independent lineages, and every document in
   this branch that formalizes selection at all. Maximum-k-coverage — "the VLM
   gets k views, cover as much as possible" — has **essentially no precedent**;
   it appears exactly once in the corpus, as a one-line infeasibility fallback
   in Scott 2001.

2. **At our instance size the entire heuristic apparatus of the field is
   irrelevant, and this is quantified, not asserted.** Exact ILP set cover
   solves in ~1 s at 800 candidates × thousands of voxels (Pan 2025, Gurobi),
   <10 s at 32 candidates × 32³ voxels (MA-SCVP/SCVP, Gurobi, run at
   20,000-instance dataset scale), 15.6 s at Glorieux's smallest industrial
   config. Glorieux's own tractability threshold is "<1000 admissible
   viewpoints." Our instance — tens of candidates, tens of regions — is one to
   two orders of magnitude below every one of those. And greedy leaves a
   *measured* tax on the table: **12.5–40% more views than the best-known plan**
   (Scott 2009) and **5.5–23.8% more** (Glorieux). We should solve exactly and
   say so explicitly.

3. **The bottleneck is never the solver — it is the visibility computation.**
   Four independent sources agree: Mosbach (ray-tracing 41 s–52 min vs. greedy
   0.4 s–5 min), Gospodnetić (3.81 s per candidate ray-trace vs. <10 s total
   greedy for 1000 candidates), Pan 2025 (visibility 250–2000 s vs. set cover
   1 s), Scott 2009 (measurability matrix is the dominant cost, O(s²v), two ray
   traces per entry). Engineering effort belongs on the coverage predicate, not
   on the optimizer.

4. **The field is split on whether "coverage" already means "coverage at
   quality," and the split is generational.** The classical CAD-inspection
   lineage bakes a quality gate into the matrix entry (Tarbox's grazing-angle
   thresholds; Scott 2009's `m_ij` = frustum ∧ visibility ∧ spec-compliance;
   Glorieux's `f_scat∘f_ray∘f_fov`; Gospodnetić's 75° incidence cut). The
   modern reconstruction/NBV lineage — SCVP, MA-SCVP, Pan 2025, ObjView-Bench,
   all Bennewitz-lab — uses **pure binary visibility with zero quality term**,
   confirmed by full-text keyword searches in four separate reading notes. If
   we adopt the modern lineage's SCOP skeleton, we must supply the quality gate
   ourselves; the classical lineage already hands us the pattern.

5. **The fallback ladder for an unreachable ideal view has five rungs, all with
   citations, and they compose rather than compete.** This was the branch's
   headline deliverable and it is now enumerable (§5.1). The single most
   important correction from round 3: **rung (iii) "optimize in joint space"
   does not eliminate the failure mode** — Wang 2026, the flagship joint-space
   paper, delegates collision to MoveIt as a post-hoc ranked-retry and still
   fails 8/100 end-to-end.

6. **Learned selection exists in our regime for a reason we do not have.**
   SCVP/MA-SCVP need a network *because the object is unknown at inference
   time* — the network is a learned surrogate for a set-cover oracle that
   cannot be run online. We have the fused survey cloud before selection runs.
   The condition that forces the field toward learning does not hold for us.
   Additionally, ObjView-Bench Table 4 shows learned selection over a fixed
   discrete action space is *fragile to that action space shrinking underneath
   it* (NBV-Net: ΔNSC 0.516 collapse under reachability restriction) — which is
   precisely our VLM-over-discrete-actions design.

7. **Nobody plans views from a defective self-acquired survey cloud.** Confirmed
   by four independent surveys and every method paper in the branch. The single
   close entry is Pan 2025's previous-session planning, whose *conservative
   inflation* mechanism (union the transferred prior with the current coarse
   scan plus points where they diverge) is the only measured technique in the
   corpus for treating prior geometry as fallible evidence rather than truth.

---

## PART I — The formulation

### 2.1 Q1: minimum set cover, maximum k-coverage, or per-region quality under a floor?

**The field's answer is unambiguous: minimum set cover.** The citation chain,
in order:

- **Tarbox & Gottschlich 1995** — measurability matrix M(i,j), rows = surface
  points, columns = admissible viewpoints; greedy "select the viewpoint
  maximizing the void surface visible" until void is exhausted. The origin, and
  the origin of the NP-completeness claim.
- **Scott, Roth & Rivest 2001 (NRC-44892)** — the formal ILP. Verbatim:
  `min Z = Σ x_j` s.t. `Σ m_ij x_j ≥ 1 ∀i`, `x_j ∈ {0,1}`; explicitly named
  "the *unicost set covering problem*."
- **Scott 2003 survey** — restates it three times, names it NP-complete three
  times; grep-verified: "k-coverage" does not appear anywhere in the document.
- **Scott 2009** — the same ILP as Eq. 6–9, now with a working algorithm.
- **Peuzin-Jubert 2021** — the "SCP transposition" macrocategory (§3.1); Karp
  1972 cited directly for NP-completeness, an independent trail.
- **Glorieux 2020 §2.1** — explicit formal set system `(S, Q)`, "find the
  minimum number of viewpoints such that every primitive is observed."
- **Mosbach 2021 Eq. 11**, **Gospodnetić 2022 §5** ("equivalent to a minimum
  set coverage problem"), **Spensieri 2025 Eq. 3** (binary ILP via HiGHS),
  **SCVP/MA-SCVP Eq. 4** (SCOP via Gurobi), **Pan 2025 Eq. 6** (Gurobi),
  **ObjView-Bench Eq. 3** (`d_plan` as literal minimum set cover).
- **CMM metrology** — every method in Sadaoui & Phan's Table 2 (Hwang/Chvátal,
  Kamrani, Stojadinovic, Ziemian, Kweon, Corrigall, Lai) is described with the
  same goal: minimize part setups subject to covering all tolerances. Never
  "given k setups, cover as much as possible."

**Maximum k-coverage has no home in this literature.** One sentence, in
Scott 2001 §2.1: *"Occasionally, there may be no feasible solution, in which
case we wish to have the minimal number of non-compliant constraints, that is —
maximum surface covering."* An exception path, not a parallel framing.
ObjView-Bench uses budget regimes (K=5, K=30, Auto) but as an **evaluation
protocol**, not as any planner's objective.

**Where quality enters — two architectures, both viable, and the choice
matters for us:**

- **(a) Quality folded into the coverage predicate as a hard admissibility
  gate.** The dominant move. Scott 2009's `m_ij` = 1 iff frustum occupancy ∧
  visibility ∧ spec-compliance (precision + sampling density meet a numeric
  spec) — so "coverage" *already means* "coverage at acceptable quality," by
  construction. Glorieux's `f_obs = f_scat(f_ray(f_fov(·)))`, a hard AND of
  three tests. Gospodnetić's binary admissibility (focusing distance + incidence
  ≤ 75°) checked before the visibility matrix is built. Spensieri's cone-shaped
  admissible regions. Bircher's incidence-angle half-spaces as QP constraints.
- **(b) Quality guaranteed by construction at generation, coverage solved
  separately over already-good candidates.** Mosbach 2021: Eq. 6 places every
  candidate camera *on the surface normal* at distance `d`, so **every candidate
  is exactly orthogonal to its own patch by construction**; the residual
  combinatorial problem (which of these already-orthogonal candidates to keep)
  is then plain unweighted set cover. Scott 2009's Optimal Scanning Zone does
  the same at per-vertex granularity (normal dilation by `f_d·R_o`,
  `f_d = 1.02`).

**Route (b) is the one that matches Albert's requirement.** His demand is that
an *evidence image* be the most orthogonal view of the inspected surface — not
that the object be covered cheaply. Mosbach's architecture says exactly this:
orthogonality is a property of how a candidate exists at all, not a term traded
off inside an objective. Peuzin-Jubert gives it a citable category name —
**"coverage optimization"** (§3.2), the class of methods that construct one
viewpoint per patch directly from segmentation rather than sampling a superset
and culling.

**The one instance of a literal "quality subject to a coverage floor"** in the
corpus is Wang 2026: hard geometric constraints `C_obs` (axis error ≤ θ_max,
lateral offset ≤ 5 mm, standoff ∈ [0.08, 0.38] m) **AND** a bottom-visible
ratio `V ≥ V_min = 0.85`. Clean pattern, but single-view/single-region — there
is no set being selected.

> **Contradiction to keep visible.** Scott 2003/2009 argue coverage already
> subsumes quality, so no separate quality layer is needed. Peuzin-Jubert's
> negative finding argues the opposite for our purposes: where incidence angle
> *is* a named criterion in the field (their criterion `n`), it always appears
> as **one weighted term among several** in a multi-criterion objective —
> never as a hard bar an evidence image must clear. Albert's requirement is
> structurally different from the field's convention, and no surveyed method
> treats orthogonality the way he asks for it.

### 2.2 Q2: submodularity — the question dissolves rather than resolves

**Zero of the 20 documents discuss submodularity.** Grep-verified in Scott 2003,
Scott 2001, Peuzin-Jubert, Alsadik, Glorieux, Gospodnetić, Mosbach, Spensieri,
Pan 2025. Chvátal 1979 and Nemhauser–Wolsey–Fisher were never read in round 3.

What round 3 established instead is *why* nobody needs it:

- The algorithm the field actually runs — Tarbox's "maximize void surface
  visible," Gospodnetić's Algorithm 1, Glorieux's Eq. 1 — is the textbook
  submodular-greedy max-marginal-gain move over an **unweighted** coverage
  function. The ln(n)+1 / (1−1/e) guarantees hold *by construction*, and
  Tarbox implemented the algorithm without ever invoking them.
- **Everyone stays boolean deliberately.** Scott 2009's `m_ij` is a 0/1
  spec-compliance test rather than a continuous quality score. Glorieux's
  `Pr_k` is 0/1. Gospodnetić's `a_ij` is 0/1. Spensieri's shapes are
  inside/outside. This is the design lesson: **the field preserves the
  guarantee by refusing to make quality a continuous weight.**
- The one weighted formulation in the corpus — Mosbach's Eq. 11, with
  `c_i = 1/(coverage count)` — is **posed but not solved as posed**; they run
  plain greedy anyway, for comparability with prior work.

**For us the question is moot.** At our instance size we solve exactly (§2.3),
so approximation guarantees do not bind. The finding is still load-bearing as a
*fallback contract*: if we ever need to scale to an instance where greedy is
necessary, keep the per-region contribution boolean and the guarantee comes
free; introduce a continuous orthogonality weight and it evaporates with no
literature to fall back on.

### 2.3 Q3: at our instance size, can we just solve it exactly? — Yes, quantified

This is the branch's strongest evidence cluster.

| Source | Instance | Solver | Time |
|---|---|---|---|
| **MA-SCVP / SCVP** | 32 candidates × ~32³ voxel universe | Gurobi | **<10 s per instance**, run across a ~20,700-sample dataset pipeline |
| **Pan 2025** | 32–63 (plants) / 800 (greenhouse rows) candidates × thousands of voxels | Gurobi | **~1 s** — vs. 250–2000 s for the visibility computation |
| **Spensieri 2025** | 413 aerospace inspection points → 4 sensor locations; 77 weld tasks → 3 robots | HiGHS | near-instant; even 2000 abstract disks: ~210 s avg / ~600 s worst |
| **Glorieux 2020** | door panel 2283 primitives, ~60–250 admissible viewpoints | ILP | **15.6 s** (smallest config) → 1062 s (>1000 candidates); explicit tractability threshold "<1000 admissible viewpoints" |
| **Mosbach 2021** | cylinder, 36–48 candidates | greedy NBV | 0.4–1.0 s (ray-tracing the matrix: 41–62 s) |
| **Gospodnetić 2022** | 70–2002 candidates × 74k–100k mesh faces | greedy | <10 s for 1000 candidates (ray-tracing: 3.81 s/candidate) |
| **Scott 2009** | mask s=244; bunny ~700 → 72 candidates | greedy | matrix build 3.6 min (mask) / 28.1 min (bunny) on 2005 hardware |
| **Tarbox 1995** | ~15,000 candidates for 1000–2500-vertex meshes | greedy + SA | judged "prohibitive" *at that scale* |

**Our instance is smaller than every row.** And the tax for not solving exactly
is measured, not hypothetical:

- **Scott 2009**: greedy used **40% more views** than the best-known plan on the
  mask (7 vs. 5) and **12.5–25% more** on the bunny (18 or 20 vs. 16).
- **Glorieux 2020**: ILP gave **5.5% fewer** viewpoints (case study 1) and
  **23.8% fewer** (case study 2) than greedy approximation.

> **Caution when citing Glorieux's 23.8%.** The abstract and conclusion phrase
> it as a targeted-vs-random-sampling gain. Traced back to §7.3, it is the gain
> from switching the *solver* from greedy to ILP with sampling held fixed. The
> genuine targeted-vs-SOTA viewpoint-count gains are 3.7–17.2%.

> **Counter-evidence, weak but honest.** The CMM literature standardized on
> heuristics (Chvátal greedy per Hwang et al., genetic algorithms) at instance
> sizes *comparable to or smaller than* ours. This is a mild negative data point
> against "small instance ⇒ obviously solve it exactly" — though it most likely
> reflects 1990s–2000s compute norms and ILP-tooling availability rather than a
> considered rejection. Do not read it as a verdict.

> **Nuance that must survive into our writeup.** The field solves set cover
> exactly but *rarely deploys it as the online planner*. ObjView-Bench computes
> `d_plan` as an offline difficulty label; SCVP/MA-SCVP run Gurobi offline as a
> training-label oracle. The genuine online-exact deployments are Pan 2025,
> Spensieri 2025 and Glorieux 2020. If we say "the field shows ILP is
> deployable," those three are the citations — not the Bennewitz lineage.

### 2.4 Q4: does "generate candidates then select" earn its keep?

**Yes — for three independent reasons, only one of which is ours.**

1. **Numerical tractability (the field's own reason).** Scott 2003 §9.2 splits
   viewpoint generation into *synthesis* (solve directly over continuous
   viewpoint space) vs. *generate-and-test*, and states plainly that synthesis
   is a minority approach because *"in practice, computational difficulties of
   nonlinearity and convergence usually arise"* and that *"the great majority of
   view planning methods follow the generate-and-test paradigm."* This predates
   and is unrelated to any VLM argument.
2. **Constraint modularity.** Peuzin-Jubert §3.2.5 is the concrete trade-off:
   coverage-optimization (derive-directly) methods are *"not optimized to easily
   add or remove constraints... the implementation of the constraints was split
   between the segmentation and optimization of the viewpoint position,"* unlike
   SCP-transposition methods where *"constraints are mainly included in the cost
   function."* Their criterion (d) — generic treatment of constraints — is
   satisfied more often by the enumerate-then-select camp.
3. **Revocability.** Scott 2009's argument for global set-cover selection over
   sequential NBV: early selections in an iterative loop *"cannot be revoked."*
4. **VLM legibility (ours).** The contract's given: VLMs cannot emit metric
   poses; selection stays discrete. SCVP makes the same point mechanically —
   binding every network output bit to a fixed candidate index is a *hard
   requirement* of formulating selection as fixed-length multi-label
   classification at all.

**Empirical support:**

- **Gospodnetić 2022 ran the ablation.** Their single-stage "generate-and-use"
  (pick an uncovered primitive, place a viewpoint on its normal, test and keep)
  reaches ≥90% coverage on all three objects but *"shows no other benefits over
  the generate-and-test approach"* — it is not minimal, and is recommended only
  for rapid prototyping.
- **Glorieux 2020**: Jing et al.'s *joint* (non-decoupled) selection+sequencing
  formulation cost **1 hour vs. 5 seconds** for equivalent quality — ~720× more
  compute for no gain.

**But generation should be constraint-driven, not grid-driven.** Spensieri 2025
is the sharpest result here: their divide-and-conquer intersection-refinement
candidate generation reaches the same optimum as uniform-grid sampling but far
cheaper, and **uniform sampling ran out of memory in 3 of 4 industrial
scenarios** at fine resolution. Gospodnetić reaches the same conclusion from the
heuristic side: Space Sampling (sphere-based) discards 33–77% of candidates as
inadmissible and still converges to poor coverage (as low as 53% on the Gear,
0% on some Hirth rows), while Patch Sampling discards 0.08–32%.

**Counter-evidence worth keeping:**

- **Wang 2026's ablation is the sharpest attack on our current architecture.**
  "View Sampling + IK" — sample Cartesian views, test IK, keep survivors — is
  the closest published analog to our viewsphere+IK-filter, and it scores
  **65/100 success, 80.2% bottom-visible** against joint-space direct search's
  **92/100, 91.65%**. Their ranked candidate list is never exposed to any
  chooser, though, so this says nothing about VLM presentability.
- **Magaña 2023 shows the derive-directly track does not let us skip
  enumeration.** Its output is a continuous SE(3) manifold; validation only
  spot-checks **2 extreme vertices** of it per task; the paper never proposes a
  principled way to turn a 𝒞-space into a well-distributed, budget-controlled
  candidate set. Adopting constraint-intersection generation leaves RQ3/RQ4
  fully open.
- **The CMM chain has the same hole.** Spyridi & Requicha, Kweon & Medeiros and
  Vafaeesefat & ElMaraghy all compute continuous admissible-direction regions
  and cluster analytically — no discrete candidate pool — and hand the result to
  a CMM controller that executes every orientation deterministically. As the
  reading note puts it: *the enumeration-for-presentability problem has no
  precedent here* — their "user" was a machine executing a fixed plan, not an
  agent choosing among options.

**Verdict:** keep both stages. Make generation constraint-driven (per-primitive
admissible regions), keep an explicit exact selection pass on top.

---

## PART II — Per-primitive view recipes

### 3.1 Q5: the precedent chain for "one orthogonal view per face, a ring per cylinder"

**No paper in the corpus names this recipe.** Confirmed across all 20
documents. Scott 2003 §3.1 has a single unattributed aside — *"Some view
planning techniques are applicable to restricted shape classes, such as
manufactured objects constructed of a limited set of shape primitives"* — which
confirms the *category* was recognized in the field's own 2003 taxonomy but
attaches no citation.

**The strongest chain assemblable from round 3, in the order to cite it:**

1. **Metrology accessibility cones + feature clustering (1990–2000, three
   independent groups).** Spyridi & Requicha 1990 (Local/Global Accessibility
   Cones, computed via Gaussian images and Minkowski sweeps, explicitly to
   *"generate a minimal set of directions for inspecting a part"*) →
   Vafaeesefat & ElMaraghy 2000 (per-point accessibility domains over STL
   facet vertices, Boolean-combined, then grouped) → Kweon & Medeiros 1998
   (visibility maps + "VMap dimensionality" clustering, C++, minimal part
   orientations). Three groups converging on the same two-stage recipe —
   *per-feature admissible-direction set, then cluster/intersect to minimize
   orientation count* — and industrially adopted for CMM planning. **This is
   the "primitive-level deterministic rules stand on precedent" argument
   Anton needs.**
   - **Scope limit, confirmed**: Sadaoui & Phan's Table 2 lists Kweon et al.'s
     disadvantage as *"only applied for prismatic measurement parts."* This
     chain covers the **box-face half only**. The cylinder half must come from
     elsewhere.
   - **Caveat**: none of the three *names* "one view per face." A box-face
     answer falls out as a special case for prismatic parts. Cite the chain
     for the **principle**, not the specific rule.
2. **Normal-deviation-bounded adaptive subdivision — two independent
   lineages.** (a) Mosbach 2021: recursive subdivision of a B-spline parameter
   domain until a feature functional drops below threshold, one candidate per
   leaf placed **on the surface normal** (Eq. 6); functionals include max
   normal deviation `E_ND` and max incidence angle `E_IA`. (b) Sadaoui 2018 /
   Lartigue 2014 / Koutecky 2016 adaptive voxelization, via Peuzin-Jubert
   §3.2.1: *"a grid of voxels is created, with each voxel having the size of
   the sensor field. It is ensured that the angle between all the surface
   normals contained in each voxel is not greater than a certain threshold.
   Subsequently, the voxel is divided in two, and so on, iteratively."* Two
   communities (German MVA, French CAD-inspection) reaching the same mechanism.
3. **Scott 2009's Optimal Scanning Zone** — the fine-grained ancestor. Each
   rough-model surface point is dilated along its local normal by `f_d·R_o`
   (`f_d = 1.02`), boresight = negated dilation vector. **The candidate is
   orthogonal to the local surface by construction.** Generalizes directly to
   primitive level: face centroid + normal for a planar face; radial dilation
   at fixed height intervals for a cylinder.
4. **Peuzin-Jubert's "coverage optimization" macrocategory (§3.2)** — the
   citable *name* for the whole class of methods that generate one viewpoint
   per patch directly from segmentation. Includes Lee, Germani, Phan, Souzani,
   Bircher, Wu, and the visibility-map family.
5. **Gospodnetić 2022's empirical lever, Table 4** — the strongest
   non-theoretical support for the orthogonality-first design: orienting
   candidates toward the **surface normal** rather than the **object centroid**
   cuts the inadmissible-candidate rate from 29–76% down to 14–28%. Independent
   of any quality argument, normal-oriented generation is simply better
   engineering.
6. **The closest thing to a published "ring per cylinder" data point** is
   Mosbach's Fig. 6d: a cylinder subdivided under the incidence-angle
   functional at `t = 60°, d = 20` yields **48 candidates arranged as rings
   around the curved mantle plus a few on the flat caps**, reduced to 22
   selected. Emergent from an incidence tolerance, not hand-specified — which
   is exactly the defence we want.

**Explicit anti-citations — do not use these for the recipe:**

- **Wang 2026's cylinder is an axial cavity viewed borescope-style through a
  circular aperture**, not an exterior cylindrical surface orbited by a ring of
  cameras. No azimuthal sweep, no orbit concept anywhere. Citing it for "ring
  per cylinder" would be a category error.
- **Bircher 2015's unit of work is a mesh triangle**, arbitrarily fine or
  coarse depending on upstream ACVD remeshing. No normal-similarity clustering.
- **Tarbox, Scott 2001, Gospodnetić, Glorieux, Spensieri, SCVP, MA-SCVP,
  Pan 2025, ObjView-Bench** — all whole-object or mesh-facet, no primitive
  decomposition anywhere.

### 3.2 Q6: how many views does a primitive deserve?

**No closed form exists anywhere in the corpus.** The field's uniform answer is
that **view count is an output of the selection solve, not a computed
quantity** — MA-SCVP literally uses the SCOP-solved covering-view count as its
object-complexity metric.

**Closest approaches to a formula:**
- **Mosbach's area functional `E_A` (Eq. 10)** — motivated explicitly by *"a
  narrow camera frustum might impose limiting factors on the amount of surface
  that can be processed from a single viewpoint... a subdivision by area will
  always ensure a certain sampling resolution."* This licenses
  `view count ≈ surface area / FOV-limited footprint at working distance d` as
  a defensible starting point — but the footprint formula is never written down.
- **Scott 2009's sampling-density model (Eq. 2–5)** — closed-form chord-based
  density from FOV, image resolution, scan length, slant range and both
  inclination angles. This is a per-point density predicate, not a count.
- **Scott 2009's rough-model rules of thumb**: sampling 32× lower than the
  target fine model suffices for measurability prediction "for deep cavities";
  lower bound `A_o/25` where `A_o` is the frustum cross-sectional area. These
  bound *model resolution*, not view count.

**Spensieri's reframe is the most useful and we should adopt it:** count is
emergent from geometry × tolerance and can be **computed exactly per primitive,
cheaply**, so we may not need a closed form at all. Their evidence is stark —
**4 sensor locations for 413 inspection points** (wide admissible cone) vs.
**~80 locations for 427 points** (narrow cone), a 20× difference in count from
comparable feature counts, driven purely by admissible-cone width.

**Empirical anchors for sanity-checking our own counts:**

| Source | Object | Candidates | Selected |
|---|---|---|---|
| Mosbach 2021 | cylinder | 36–48 | **10–24** |
| Mosbach 2021 | spring | 194–482 | 9–13 |
| Mosbach 2021 | tangle cube | 1325–1571 | 32–38 |
| Gospodnetić 2022 | Gear / Hirth / Crankshaft | 70–2002 | **34 / 12–17 / 23–24** |
| Gospodnetić 2022 | same, *human experts* | — | **44/44, 21/13, 34/32** |
| SCVP / MA-SCVP | generic tabletop objects | 32 (fixed hemisphere) | **9.6–13.1** (range 4–16 by complexity) |
| Pan 2025 | maize / tomato | 32 hemisphere / 63 sphere | 7.94 / 14.17 |
| Glorieux 2020 | door panel / window frame | 60–250 admissible | 53.8–64.8 / 14.0–19.0 |
| ObjView-Bench | Objaverse++ pool | 128 Tammes | `d_plan` mean **18.5** (main) / **45.9** (slow-sat) |
| Scott 2009 | mask / bunny | 244 / 72 | 5–7 / 16–20 |
| Spensieri 2025 | aerospace / weld cell | continuous | **4 / 3** |

For tabletop-scale objects the minimal set clusters in the **single digits to
mid-teens**. That is the order of magnitude our VLM action space should be
sized against.

### 3.3 Q7: the maximum admissible deviation from orthogonality — **the numbers, and why there is no single one**

This is the branch's richest numeric harvest and its most contested finding.

| Value | Source | Sensor / context | Nature |
|---|---|---|---|
| **8.6°** (0.15 rad) | Wang 2026, `θ_max` | narrow cavity, ID 5 cm, L/D ≈ 1.1 | task-occlusion-driven, not sensor-driven |
| **30°** | Bircher 2015, all four scenarios | UAV camera, incidence floor as QP constraint | uniform, re-declared per scenario, never justified |
| **35°** | Prieto 2000 PhD, via Peuzin-Jubert §3.1.4 | *"beyond this angle, the resulting point cloud is extremely noisy and cannot obtain accurate measurements"* | **noise-derived** |
| **45° / 60° / 90°** | Gospodnetić 2022, patch-sampling `E_ND` sweep | subdivision controls, not admissibility | swept parameters |
| **50°** | Raffaeli et al., via Peuzin-Jubert §3.2.1 | glancing-angle cutoff for visibility-map ray projection | method-internal |
| **60°** | Mosbach 2021, `E_IA` threshold used for the cylinder result | subdivision control | design choice |
| **60° / 70°** | Scott 2009 Eq. 1, `t_yz` / `t_xz` | triangulation sensor — **asymmetric**: 60° in the baseline plane, 70° in the orthogonal scanning plane | sensor-geometry-derived |
| **75°** | Gospodnetić 2022 §8 | their camera: 2448×2050 px, f = 12.93 mm, focusing distance 140 mm, DOF 135–155 mm | system-specific constant |
| *undisclosed* | Glorieux 2020, `f_scat` | *"within the limitations of the gauge"* — CogniTens WLS400a | **proprietary, withheld** |
| *undisclosed* | Spensieri 2025 | cone half-angle not in main text | not reported |
| *unrecoverable* | Tarbox 1995 | separate grazing-angle thresholds for camera and light source | primary text inaccessible |
| *exogenous* | Magaña 2023 §III-E-5 | max incidence angle is *"a sensor manufacturer-provided input parameter"* | explicitly not derived |

> **The contested finding, stated plainly.** The published numbers span
> **8.6° to 75°** — nearly an order of magnitude. Every source that gives a
> number ties it to its own sensor geometry (Scott, Gospodnetić, Prieto) or its
> own task occlusion structure (Wang, Bircher). Magaña states outright that the
> number is exogenous, supplied by the manufacturer. Peuzin-Jubert's two
> numbers (35°, 50°) come from independent lineages with different purposes.
> **There is no universal threshold and we must measure ours for the D405.**

Two things *are* consistent and usable:
- Every number is well under 90°, and the metrology cluster (**35–75°**) sets
  the plausible band for a depth sensor.
- **The tolerance may not be isotropic.** Scott 2009's 60°/70° asymmetry —
  tighter in the triangulation baseline plane than in the orthogonal scanning
  plane — is a direct warning for a stereo sensor with a baseline. Our D405 has
  one. This should be tested, not assumed away.

**Also negative:** no RGB-D / active-stereo depth camera appears anywhere in the
CAD-inspection literature. Peuzin-Jubert's sensor taxonomy (§2.4.1) is laser
scanner / structured light / "other"; the closest analog (structured light) is
a calibrated high-precision industrial device, not a consumer stereo camera with
per-pixel confidence and dropout on glossy or dark surfaces. Any sensor-quality
formula transplanted from this literature needs **re-derivation, not
re-parameterization**.

---

## PART III — The constraint stack (the branch's distinguishing question)

### 4.1 Q8: the fallback ladder — **five rungs, with citations, and they compose**

This was the branch's headline deliverable. Round 3 delivered it, plus one
important correction and two non-rungs.

**Rung (i) — hard filter and discard, at generation time.**
- **Scott 2001 §2.2, verbatim**: *"A separate issue relates to physical
  limitations on the range of motion and degrees of freedom of the positioning
  system which effect viewpoint feasibility. There is no need to include such
  constraints in the above IP formulation. It is more efficient to avoid
  generating infeasible viewpoints at the viewpoint generation stage."*
  Scott 2009 §3 restates it.
- Tarbox 1995: octree occupancy grid culls viewpoints occluded by mounting
  fixture and supporting plane, *before* the measurability matrix is built.
- Glorieux 2020 Eqs. 4–5: `g_kin(Q_i) ≥ 1` and `g_col(Q_i) ≥ 1` as **hard
  constraints inside the search** — infeasible poses are excluded outright, not
  penalized.
- Jing et al. 2018, via Peuzin-Jubert §3.1.2: *"The viewpoints for which no
  configuration existed, or which involved a collision with the environment,
  were directly removed from the set."*
- ObjView-Bench Table 1: "remove infeasible views" for set-prediction planners.

  **The justification everywhere is tractability, not quality-neutrality.**
  Scott 2001 says so explicitly. Scott 2009 adds the crucial mitigating
  condition: hard-delete is safe *"with the expectation that associated surface
  regions will be covered by adjacent viewpoints on `S_o`"* — i.e. discard only
  works when generation is dense enough that neighbours recover the region.
  **That condition does not hold for a small, sparse, primitive-derived
  candidate set.**

**Rung (ii) — slide inside the admissible region and re-optimize.**
- **Bircher 2015** is the cleanest mechanism: a per-facet QP whose *feasible
  set is the admissible region itself* (incidence half-spaces off each triangle
  edge, min/max range planes, FOV cone convexified by angular slicing). The
  viewpoint can never be pushed outside admissibility; it only slides inside it,
  chasing a cheaper tour, iteration after iteration.
- Tarbox 1995: morphological erosion of peripheral viewpoints (for pose-error
  robustness, not reachability — the mechanism transfers, the motivation
  differs).
- Tarabanis et al. 1995, via Scott 2003 §9.3: synthesis that *"seeks to
  centralize viewpoints in the admissible domain."*
- **Scott 2009 §5, verbatim**: *"Pose constraints due to mobility limitations
  are applied on a case-by-case basis, mapping the optimum viewpoint to the
  nearest feasible pose. Excessive mobility constraints will negate the
  advantages of the proposed viewpoint generation scheme."*
- MA-SCVP §V-F, the crude real-world version: *"we adjust the candidate view
  positions by moving them up by 0.008 m to ensure reachability"* — a single
  blanket offset for the whole candidate set, tuned once for their rig.

**Rung (iii) — never leave the feasible set: sample/optimize in configuration
space.**
- **Yu & Gupta 2000/2001** is the founding argument, and it is quotable:
  *"'Where to move' is best posed and answered in configuration space... the
  view planning problem is also appropriately posed in the configuration space
  of the robot."* Their view-node candidates are drawn **only from the
  roadmap's currently-reachable connected component** — functionally a live
  capability map. Their contrast with recognition-oriented CV view planning is
  the citable framing sentence: those works *"do not include kinematics
  constraint of the robot... They usually assume that the sensor can move to
  arbitrary positions and directions and there are no obstacles."*
- **Wang 2026**: joint-space multistart coordinate search, no IK inversion
  anywhere in the optimizer. Result: **92/100 success** vs. View-Sampling+IK's
  65/100 and Fixed-Pose+IK's 34/100; bottom-visible 91.65% vs. 80.2% vs. 44.6%.
- Vasquez-Gomez 2014/2017, via Peuzin-Jubert §4.2: viewpoints sampled directly
  in the robot's configuration space, then collision-tested.
- Spensieri 2025 Algorithms 2/3: `findAllVisible` prunes candidate cells
  *during* the divide-and-conquer search via IK + collision-distance queries.

  > **Correction from round 3, load-bearing.** Wang 2026 does **not** satisfy
  > all constraints by construction. §III-B, verbatim: *"The optimizer handles
  > observation geometry and joint limits; collision geometry and trajectory
  > connectivity are delegated to MoveIt."* Collision is a post-hoc ranked-retry
  > over ≤16 candidates, and **8/100 cases still fail end-to-end** despite it.
  > The honest statement for our architecture: **searching in joint space
  > reduces but does not eliminate the ideal-view-unreachable failure mode.**

**Rung (iv) — defer rather than delete: fall back to the next-best target.**
- Yu & Gupta's `getWhereToScan()` returns the next-best pre-ranked point
  `x'_max` when no reachable view node exists for `x_max`, with a bounded 2-step
  lookahead to decide whether to scan a blocking unknown region first.
- Wang 2026's ranked list of ≤16 configurations, tried in rank order by MoveIt,
  advancing on planning failure.
- ObjView-Bench Table 1: "select best feasible view" for score-based iterative
  planners.
- ObjView-Bench's own observation: **none of its 7 evaluated planners implements
  true defer-and-retest.** An observed gap, not just our omission.

**Rung (v) — predict feasibility (proposed, never built).**
- MA-SCVP §VI, verbatim: *"This does not allow the system to drop some
  unreachable candidate views, such as due to obstacles and robot workspace.
  A possible solution is using a regression network and training it with the
  state of unreachable views."* Floated as future work in a 2024 T-RO paper.
  A fifth answer, unimplemented.

**Two non-rungs that appear in the corpus:**
- **Move the base.** Zacharias 2007 §V: reposition the torso so the
  high-reachability region (D ∈ [68,76]) sits over the object. **Not available
  to us** — our base is fixed.
- **Terminate and report.** Vasquez-Gomez, via Peuzin-Jubert §4.3: *"the process
  was terminated if the robot did not identify any path between its current
  position and the positions of the candidate viewpoints"* — reachability
  exhaustion treated as a **valid stop condition, not an error state.**
  Scott 2001's objective-switch version: when no feasible cover exists, minimize
  non-compliant constraints, i.e. maximize surface covered.

**The documented worst case:** ObjView-Bench Table 1 — NBV-Net outputs a single
discrete index and **terminates the whole run if that index is infeasible.**
Under reachability restriction it collapses at ΔNSC 0.516 / ΔSC 0.545 while
classical planners lose only ΔSC ~0.13. This is the failure mode of "a learned
policy over a fixed discrete action space, when the action space shrinks
underneath it" — the exact structure of a VLM choosing from an enumerated menu.

**They compose.** Yu & Gupta is the proof: candidates drawn only from a proven-
feasible set (iii) *and* graceful fallback to the next-best target (iv), in one
implemented system. Spensieri is the second proof: coarse conservative geometric
filter (i-flavoured) *and* exact per-candidate query (post-filter), explicitly
because the coarse filter alone admits false positives. **Presenting the ladder
as four mutually exclusive alternatives would be a misreading.**

### 4.2 Q9: capability map up front, or per-candidate IK post-filter?

**Zacharias 2007 is the origin of "filter up front," and its numbers matter:**
- Reachability index `D = R/N · 100` over N=200 Saff–Kuijlaars sphere points per
  25 mm-radius subcube, with 12 twist samples at 30° per direction (twist
  discarded once *any* succeeds).
- **`D ∈ [0, 76]` across a 7-DOF arm's workspace** — *even the single best region
  reaches only 76% of sampled approach directions*, not 100%.
- Compact shape fitting to cone / cylinder-t1 / cylinder-t2; **mixed-map Shape
  Fit Error mean 7.2%, σ = 10.7** (Table I).
- Build cost: **12.3 h** for the IK sampling, **6.4 min** for the shape fitting,
  **O(1) angle test** at query time (Table II). The expense is entirely in the
  offline sampling.

> **The founding paper of the technique names our exact question and declines to
> answer it.** §VI, verbatim: *"Some of the shapes fitted to the data still
> include a few points on the sphere for which no inverse kinematics solution
> was found. It remains to be evaluated how often these positions are
> encountered by e.g. grasp planners using the map."* Nobody in this corpus
> closed that loop in the 19 years since. **We cannot cite a false-positive rate
> — we would have to measure our own.**

Two further Zacharias limitations that bite for us:
- **Collision is explicitly out of scope** (§VI: *"it remains an open question
  whether this should be done by or external to the inverse kinematics"*). Our
  setting has object inflation, gripper+camera hull, and table — all
  scene-specific, none in the map.
- **z-twist is discarded by construction.** A capability-map hit certifies that
  *some* twist works at that position+direction, not that our fully-specified
  pose does. **A hit is necessary but not sufficient.**

**Spensieri 2025's answer is "both, hierarchically," with a stated reason.**
A cheap convex over-approximation of the manipulator's reach (a solid ball per
task point) narrows the candidate set; the expensive exact query (IK + collision
distance, via Industrial Path Solutions) runs only on survivors — *precisely
because* the conservative filter admits false positives.

**Gospodnetić 2022 supplies the structurally identical finding for optical
constraints, and it partly dissolves the question:** the admissibility
pre-filter barely changes the final selected set *when the base inadmissible
rate is already low* (Patch Sampling, 0.08–32%), but destabilizes results for
high-rate generators (Space Sampling, 33–77%). **The pre-filter-vs-post-filter
choice only matters when candidate generation is bad.** Good, normal-oriented,
constraint-driven generation makes it largely moot.

**Portability caveat:** Zacharias's Monte-Carlo construction leans on a 7-DOF
arm's null space, which a 6-DOF UR5e does not have for a fully-specified 6D
pose. But the map already discards twist, so it solves a **5D** target
(position + approach direction) — under which a 6-DOF arm retains exactly one
free DOF, structurally analogous to the redundancy the sampling exploits. An
IK-branch-enumeration analogue is plausible. *This is the reading note's
inference, not Zacharias's claim, and is untested.*

**What a false positive costs *us*, specifically** — nobody in the corpus prices
this, so we must state it ourselves: offering the VLM an action it cannot
execute is an **action-space integrity failure**, not a wasted cycle. The
orchestrator commits, then execution fails, and the agent's model of its own
action space is now wrong. ObjView-Bench's NBV-Net (terminates the run on an
infeasible index) is the closest documented analogue of that cost.

### 4.3 Q10: how much of the surface becomes unreachable on a tabletop arm?

**One number exists in the entire corpus.** ObjView-Bench §6.3: *"a physical
reachability check in the real-world tabletop setup of Figure 1 shows that only
**45.4/128 candidate views are reachable on average**, with counts ranging from
**26/128 to 74/128** across five objects and five placements."* Roughly a **35%
average reachable fraction**, with wide per-object and per-placement variance.

> **Two caveats that must travel with this number.**
> 1. **No robot make/model, IK solver, collision-checking method, or workspace
>    geometry is given anywhere in the paper.** Figure 1 is the only record. It
>    is a precedent and an order-of-magnitude anchor, **not a reproducible
>    pipeline**.
> 2. **The ΔSC 0.13–0.17 coverage-loss figure is a different experiment.**
>    Table 4 compares a *simulated* whole-sphere vs. quarter-sphere proxy; the
>    45.4/128 measurement is a separate real-robot IK+collision check. Do not
>    say "ObjView-Bench measured that our failure mode costs 0.13–0.17
>    coverage." Say: it measured a real reachability rate, **and separately**
>    showed a comparably restrictive simulated constraint costs 0.13–0.17.

**The coverage-loss numbers themselves (Table 4), with a metric warning:**
classical/hybrid planners lose **ΔSC ≈ 0.128–0.131** while **ΔNSC ≈ 0**; learned
BENBV-Net/MA-SCVP lose ΔSC 0.157–0.165; NBV-Net collapses at ΔSC 0.545. The
reason ΔNSC ≈ 0 is that **NSC renormalizes by the restricted space's own
observable ceiling — it hides the loss; SC exposes it.** This is a direct
benchmark-design instruction (§8).

**The other relevant number:** Zacharias's `D ∈ [0, 76]` — the best-reachability
region of a redundant 7-DOF arm still reaches only 76% of sampled approach
directions.

**Everyone else reports full coverage achieved.** Glorieux, Spensieri,
Gospodnetić, Mosbach, Bircher: no partial-coverage case, no unreachable-fraction
statistic, anywhere. The set-cover formulation's implicit assumption that a
cover exists was never stressed.

### 4.4 Q11: IK multiplicity — is a "view" a Cartesian pose or a joint configuration?

**Near-total silence, confirmed document by document.** Scott 2003 (grep: "joint
space", "inverse kinematic" → zero hits; anachronistic — no surveyed positioning
system had more than ~2 DOF), Scott 2001, Scott 2009, Peuzin-Jubert (*"robot
configuration"* and *"Cartesian viewpoint"* treated as interchangeable),
ObjView-Bench (black-box binary), Spensieri (IK is an opaque oracle),
Pan 2025 (no IK at all — Euclidean TSP with a bounding-sphere collision check).

**Three positions exist implicitly:**

1. **Cartesian pose, IK-existence as a gate** — the mainstream. Glorieux is
   explicit: the optimization variable is `W_p ⊆ ℝ⁶ = (x,y,z,α,β,γ)`, and
   `g_kin(Q_i) ≥ 1` merely **counts** feasible IK solutions and requires ≥1.
   **He never discusses which one gets used, or whether they differ in collision
   status.** This is exactly our execution-failure-after-commitment risk, and it
   is unengaged with in the corpus.
2. **Joint configuration as the action** — Wang 2026, Yu & Gupta,
   Vasquez-Gomez. Wang sidesteps multiplicity entirely by never inverting IK.
3. **Coarser than either: (position, approach direction), twist and IK branch
   deferred to execution** — Zacharias's capability map is *deliberately* at
   this level. And two independent lineages converge on it from a different
   direction: SCVP/MA-SCVP use a **5-DOF view pose** (camera Z+ at object
   centroid, sixth rotational DOF *"has almost no effect,"* resolved
   arbitrarily), and Wang 2026 explicitly leaves camera **roll unconstrained**
   in `C_obs` because *"camera roll can vary without changing whether the cavity
   bottom is visible."*

**The design lesson this yields for us:** an orthogonality requirement
constrains **2 rotational DOF (the boresight) plus position** — roll is free.
So the natural action representation is `(position, approach direction)`, with
twist and IK branch resolved at execution time rather than committed to in the
action space. Zacharias's abstraction and Wang's roll-free `C_obs` converge on
this from opposite ends of the literature. It also *shrinks* the action space,
which serves RQ3/RQ4.

### 4.5 Q12: does the reachable set reshape the answer, or only prune it?

**Zacharias 2007 §III-E is the one direct answer, and it says: it reshapes.**
Reading Figs. 9–10: near the workspace **border**, reachable approach directions
form a narrow **cone** (one dominant direction survives); moving inward, cones
**open out**, pass through **double-cone** structures, and the innermost,
best-reachability spheres show a **ring** structure. **Position-dependent
anisotropy, not a uniform discount factor.**

**ObjView-Bench cannot answer it.** Re-queried explicitly in round 3: the paper
reports only pooled counts (45.4/128 average, 26–74 range) and **does not report
which views are systematically excluded** — no top-down-for-tall-objects
analysis, no grazing/near-table analysis. It never verifies that its
quarter-sphere abstraction's shape matches real exclusion patterns. It gives us
the *magnitude* of the loss, not its *structure*.

**Yu & Gupta names three structural exclusion modes**, one of which is
permanent rather than a prune: **self-occlusion** — *"the robot can never scan
the back of itself"* (Fig. 2.5) — a goal behind the robot's own body from every
reachable vantage is permanently unreachable even though a free path exists in
principle.

**Objective-redefinition precedent is thin to the point of being a gap:**
- Scott 2001's one-line infeasibility fallback (switch to maximum surface
  covering).
- Vasquez-Gomez's reachability-exhaustion-as-valid-stop.
- That is all. **Scott 2003, Scott 2009, Tarbox, Mosbach, Gospodnetić,
  Glorieux, Spensieri all implicitly assume a full cover exists** — that every
  surface row has at least one admissible column — and **none of them discusses
  what replaces the objective when it does not.** Scott 2003's own open-problems
  list (§10.1) names efficiency, accuracy, robustness and reflectance; structural
  uncoverability is not on it.

**This is a first-class gap that bites us specifically.** Our hard-minimum shell
radius + tabletop + gripper/camera hull make empty rows plausible. The field
gives us no guidance, so the design decision — best-available view per region,
plus an explicit unreachable-region report — is ours to make and defend.

---

## PART IV — Set size, redundancy, and stopping

### 5.1 Q13: what set sizes do methods emit, and how bad is redundancy?

**Bircher's redundancy is now documented from a primary comparative source.**
Glorieux §6, p.8, verbatim: *"The methodologies proposed by Bircher et al.
manages to provide full coverage however the number of selected viewpoints was
**300 times larger** compared to other successful methodologies... The
assumption that there is no need to minimise the number of viewpoints is thus
unrealistic for dimensional inspection of sheet metal parts."* This is a
feasibility screen rejecting a named published method purely for redundancy.

> **But round 3 corrects the framing, and the correction should survive.**
> Bircher's 1-viewpoint-per-mesh-triangle is a **deliberate, stated design
> goal**, made twice in the paper: *"with a continuously sensing sensor the
> number of viewpoints (and if this is minimal or not) is not necessarily
> important but mostly their configuration in space."* Dense overlapping views
> are *strictly better* for the downstream SfM/photogrammetry objective. Framing
> it as "documented to emit redundant views" is accurate as a **consequence for
> us**, not as a self-diagnosed flaw. And it means "iterative resampling →
> redundant views" is **not a causal law** — adopt the resampling mechanism with
> a cardinality-penalized objective and the redundancy would not follow.

**Spensieri's structural point is stronger and should lead our treatment:**
**redundancy is a property of greedy or incomplete selection, not of candidate
generation.** Their uniform-sampling *baseline* reaches the same true optimum as
their sophisticated D&C generation in every scenario, because an exact selector
sits on top. The D&C advantage is entirely in time and memory, not in answer
quality. **Exact selection immunizes against redundancy regardless of how
candidates were generated.**

**The quantified greedy tax** (already in §2.3): Scott 2009 40% / 12.5–25%;
Glorieux 5.5% / 23.8%.

**Generation-side over-production, for calibration:** Mosbach over-produces
**3–13×** (candidates vs. selected). Gospodnetić's discard rates: Space Sampling
33–77%, Vertex Sampling 26–47%, Patch Sampling 0.08–32%.

> **A genuine contradiction against minimality, and it deserves airtime.**
> Gospodnetić §9.4 ran the first manual-vs-automated comparison in this
> literature. Two human experts **consistently used 1.5–2× more views than the
> automated optimum** (Gear 44 & 44 vs. 23; Hirth 21 & 13 vs. 12; Crankshaft
> 34 & 32 vs. 23) — and they did so **deliberately**: their placements are
> *patterned*, clustered on one side and replicated by rotating the object on a
> turntable, because that simplifies the physical acquisition rig. The paper's
> own conclusion: *"the inspection can only be implemented using a manipulator
> which entails other challenges such as path planning or timing restriction.
> In that case, the benefit of having a smaller set of viewpoints becomes
> questionable."* **Minimality is not free — it shifts cost to trajectory
> planning.**
>
> For us the pressure toward a small set comes from the VLM's context and
> navigability, not from cycle time, so this argument does not directly bind.
> But it does mean "fewer views is better" is not self-evident and needs its own
> stated justification in our writeup, rather than being assumed.

**And ObjView-Bench's cross-method spread is the empirical version of the same
point:** under the Automatic budget, methods emit **9.6 (RSE) to 49.5
(MA-SCVP)** views for comparable NSC ≈ 0.99 — a **>3× spread at equal
coverage**. View-count minimality is not correlated with coverage quality.

### 5.2 Q14: is there a principled stopping/minimality criterion we can implement?

**Seven distinct answers exist. Ranked by how defensible they are for us:**

1. **Exact ILP optimality certificate.** Scott 2001 (Eq. 1–3), Scott 2009,
   Glorieux, Spensieri, Pan 2025, SCVP/MA-SCVP's labeling. **The only
   *principled minimality* answer in the corpus** — the solve terminates with a
   proof. Spensieri adds the anytime version (Algorithm 3, Eq. 4): track upper
   bound `n̄` (best confirmed solution) and lower bound `n̲` (relaxed ILP
   including unverified queue cells), stop when `n̲ = n̄` (proven optimal) or at
   a time limit, **reporting the gap**. This is a genuine, gap-bounded,
   non-learned criterion.
2. **Greedy until void is exhausted** — Tarbox; Chvátal's heuristic applied by
   name in the CMM literature (Hwang, Tsai & Chang 2004, via Sadaoui & Phan's
   Table 2 — a real, applied precedent in a structurally identical problem, not
   just a textbook reference). Guarantee holds only while the contribution
   stays boolean.
3. **Diminishing-marginal-return thresholds.** Gospodnetić's **2% of the
   first-viewpoint contribution** (deliberately caps at 98–99%, residual gap
   surfaced to a human via the coverage-visualization tool — an honest,
   implementable design). ObjView-Bench's **Map-Stabilization MS@δ** (relative
   gain <1% for three consecutive steps, hard cap 128). Plus, via
   Peuzin-Jubert §4.3: Pito/Loriot's redundancy percentage, Vasquez-Gomez's
   "surface factor," Kriegel's hole-coverage index.
4. **Coverage-redundancy `k`** — Glorieux: cover every primitive ≥ k times.
   Empirically: increasing k reduces the final count **with a documented
   saturation point** (three highest configs show no further gain) and **does
   not reduce travel cost at all** (§7.5 — an explicit null result).
5. **Learned auto-stop** — SCVP/MA-SCVP: the predicted 32-bit mask *is* the
   stop. Works **only because the training labels are themselves exact ILP
   minimum covers** — the network distills an optimizer. Cite it precisely:
   this is not a network inventing minimality.
6. **Reachability exhaustion** — Vasquez-Gomez. No path to any candidate ⇒
   terminate. A valid stop, not an error.
7. **Max-iteration cap** — the universal fallback (Peuzin-Jubert §4.3;
   Bircher's fixed 25/50 iterations; ObjView-Bench's 128-view cap).

> **The warning that binds any threshold-based criterion.** ObjView-Bench
> Table 5: MS's stopping point means something *different* across difficulty
> regimes — residual coverage gap **0.0026 on the main pool vs. 0.0081 on the
> slow-saturation pool**, a 3× difference. **A naive "gain has plateaued" rule
> is not difficulty-invariant.** Any criterion we adopt must be validated
> against an object-hardness axis the way that paper checks MS against
> `d_plan`/`d_sat`.

**Historical note worth citing:** Scott 2003 §3.2.7 named "Self-Terminating" as
a first-class view-planning requirement in 2003, and §8.1.7 criticized every
surveyed method for using criteria *"not related to explicit requirements to be
met by the reconstructed object model."* Twenty-three years later, per this
corpus, that gap is still open outside the exact-ILP camp. An ILP optimality
certificate tied to a per-region quality floor closes a **named, decades-old,
explicitly-flagged** gap rather than reinventing something.

---

## PART V — Learning vs. analysis, and the defective-cloud gap

### 6.1 Q15: does learned selection beat analytical selection in our regime?

**Chronology first.** Peuzin-Jubert 2021 §5, explicit: *"although machine
learning is widely used in the field of computer graphics, **none** of the
studied methods use this type of paradigm, likely because these methods require
a large amount of data."* ~35 methods surveyed, zero using ML as of 2021. The
field's uptake is **post-2021**, which dates the entire learned-NBV literature
as very recent relative to the CAD-inspection tradition.

**Alsadik 2025 names our regime and blesses the analytical track textually.**
Table 1 defines **"model-based NBV"** — partial or complete prior geometry,
targeted-object inspection, structured environments — as a first-class, named
subproblem, explicitly distinguished from the free-model exploration NBV our
branch excludes. Table 3's example application for the **rule-based** strategy
class is literally *"Surface inspection, structured environments,"* and the
narrative repeatedly ties rule-based methods to *"industrial settings where the
scene is predictable and well-defined."* **Deterministic per-object view rules
are a recognized, still-current strategy class for exactly our regime — not a
regression to something superseded.**
- Caveats: Alsadik contains **zero experiments and zero numbers** — it cannot
  adjudicate anything empirically. And its single use of the word "primitive" is
  dismissive (*"rule-based methods... seem somewhat 'primitive' to modern
  machine-learning-driven methods"*).

**The decisive structural argument comes from the learned lineage itself.**
SCVP and MA-SCVP need a network **because the object is unknown at inference
time** — the network is a learned surrogate for a set-cover oracle that cannot
be run online, since the true remaining surface is unknown. Their ILP runs
**offline only**, as a training-label oracle. *We do not have that constraint.*
Our fused survey cloud exists before selection runs. **The condition that forces
this literature toward learning does not hold for us; the strongest reading of
their own engineering choices is that we should skip the network and run the
optimizer.**

**The empirical evidence for our specific design.** ObjView-Bench Table 4:
under reachability restriction, **classical and hybrid planners degrade
gracefully** (ΔSC ≈ 0.13, ΔNSC ≈ 0, because they re-optimize over whatever
candidate set they are given) while **NBV-Net collapses** (ΔNSC 0.516) — it
emits a single discrete index and terminates if that index is infeasible. This
is a concrete failure-mode precedent for *"learned selection over a fixed
discrete action space is fragile to that action space shrinking underneath
it"* — structurally identical to a VLM choosing from an enumerated menu that
reachability has just pruned.

> **Tempering counter-evidence, same paper, kept visible.** §6.5: rebalancing
> MA-SCVP's *training distribution* toward medium/high-`d_plan` objects, with
> **no architecture change**, recovers its coverage on hard objects. Learned
> brittleness under distribution shift is partly a data-curation problem, not
> fundamental to the paradigm. A purely "learned methods are brittle" conclusion
> is too strong.

**One more observation that reframes the whole comparison.** SCVP, MA-SCVP,
Pan 2025 and ObjView-Bench all use **uniform Tammes / Spherical-Code
hemispheres or spheres** as their candidate space — 32, 63, 128 evenly-spaced
points, fixed regardless of whether the object is boxy, cylindrical or organic.
MA-SCVP justifies "32" only by citing prior conventions (NBV-Net's 14,
PC-NBV's 33) and a packing algorithm, with **no incidence, FOV, or overlap
calculation behind it**. The modern learned lineage is therefore **a
sophisticated instance of exactly the geometry-agnostic sphere we are
replacing** — not a counter-example to our thesis.

### 6.2 Q16: does anyone plan views from a defective self-acquired survey cloud?

**No.** Confirmed independently by four surveys and every method paper:

- **Scott 2003**: the model-based / non-model-based split has no middle. Either
  a genuine prior model, or a model built live and incrementally during the same
  session. "Survey once into a fused defective PLY, then plan a separate later
  set of views against that static artifact" is neither.
- **Peuzin-Jubert 2021**: every method needs a clean discretized proxy of a
  *trusted* model (mesh / voxel / NURBS / point cloud / B-Rep). Point clouds
  appear **only as intermediate segmentation substrates derived from a trusted
  model**, never as noisy end products of a prior scanning pass.
- **Tarbox** (trusted CAD tolerance-reference model), **Mosbach** (explicitly
  warns B-spline reconstruction needs *"highly accurate scanning with a low
  degree of noise"*), **Gospodnetić** (clean synthetic CAD meshes),
  **Glorieux** (assumed-available reference geometry, constant sheet thickness),
  **Spensieri** (process-given feature points), **Magaña** (hand-specified CAD
  parameters), **SCVP/MA-SCVP** (privileged watertight meshes),
  **ObjView-Bench** (§8, explicit: *"no depth noise or calibration error"*).

**The one genuinely close entry is Pan 2025.** A degraded, incomplete
previous-session reconstruction drives one-shot view selection. Its transferable
mechanism is **conservative inflation** (Eq. 4): `𝒫̃⁺ = 𝒫̃ ∪ 𝒬 ∪ ℐ`, where ℐ
adds points near the current coarse scan but *far from* the aligned prior — i.e.
capture where reality diverges from the transferred geometry rather than
trusting it. **Measured contribution: coverage 96.99% → 98.06% (maize)**,
98.98% → 99.15% (tomato). Small but consistent.
- **What does not transfer:** plants *deform* between sessions; our object does
  not. The non-rigid registration machinery solves a problem we do not have.
  Only the *pattern* — treat self-acquired geometry as fallible evidence,
  augment for regions it may be wrong about — transfers.
- **Noise realism caveat:** simulation-only, virtual sensor, synthetic Gaussian
  pose noise (σ_t = 0.01 m, σ_r = 2°) recoverable by a lightweight ICP pass
  (Chamfer 13.87 → 6.08 mm; F-score precision 42.1% → 99.3%). Not our
  depth-dropout / glossy-surface / ~4 cm fusion-offset profile. Treat their
  robustness numbers as a **lower bound on difficulty**.

**Adjacent and mechanistically useful: Scott 2009's pose-error analysis.**
His rough model is a real scan, *synthetically* degraded (sampling noise
0.3% R_min; pose error σ_p = 4.26–6.15 mm, σ_a = σ_t = 3°). Two findings
transfer:
- **Pose error primarily degrades *frustum occupancy*** — the geometric "is this
  point even in view" test — more than visibility or spec-compliance, and
  redundancy is the main mitigant. This is a **mechanism-level explanation for
  why our ~4 cm fused-vs-live-depth offset matters**: it can silently exclude
  points from a candidate's frustum that the plan assumed were covered.
- **Compensation's main measured effect is variance reduction, not mean
  recovery** (bunny: mean measurability 0.9978 → 0.9992, but σ 0.0029 → 0.0011).

**Also worth carrying:** Scott 2001's "rough model" terminology (a preliminary,
imperfect geometric input used to plan subsequent precision views) is the
closest vocabulary precedent for our survey-PLY pipeline — a two-stage
rough-model → precision-measurement shape, named in 2001.

---

## 7. What is contested — contradictions preserved

Seven live disagreements. None of these should be smoothed in the final writeup.

1. **Does coverage already mean coverage-at-quality?**
   *Yes* — Scott 2003/2009, Tarbox, Glorieux, Gospodnetić, Spensieri: quality is
   a hard admissibility gate baked into the matrix entry.
   *No* — SCVP, MA-SCVP, Pan 2025, ObjView-Bench: pure binary visibility, zero
   quality term, confirmed by keyword search in four separate notes.
   *Neither* — Peuzin-Jubert's negative finding: where incidence angle is a
   named criterion, it is always **one weighted term among several** in a
   multi-criterion objective, never a hard bar an evidence image must clear.
   Albert's requirement matches none of the three exactly.

2. **What is the maximum admissible deviation from orthogonality?**
   Published values span **8.6° to 75°** — 8.6° (Wang, narrow-aperture),
   30° (Bircher, UAV), 35° (Prieto, noise-derived), 50° (Raffaeli), 60°/70°
   (Scott, anisotropic), 75° (Gospodnetić). Magaña states outright that the
   number is exogenous, supplied by the sensor manufacturer. Glorieux withholds
   his as proprietary. **There is no field consensus; there may be no
   field-level answer at all.**

3. **Does small instance size imply "just solve it exactly"?**
   *Yes* — Pan 2025 (1 s at 800 candidates), MA-SCVP (<10 s), Spensieri (exact
   ILP everywhere, HiGHS), Glorieux (ILP below ~1000 candidates), plus the
   measured 5.5–40% greedy tax.
   *Weak counter* — the CMM literature standardized on **heuristics** (Chvátal
   greedy, GAs) at instance sizes comparable to or smaller than ours. Probably a
   compute-era artifact, but it is a real data point and should be named rather
   than hidden.

4. **Is minimality actually good?**
   *Yes, for the VLM* — small action spaces are the contract's given.
   *Not obviously, in practice* — Gospodnetić's experts used **1.5–2× more
   views than the optimum, on purpose**, because patterned redundant sets are
   easier to execute; and the paper's own conclusion questions whether a smaller
   set is a benefit at all once a manipulator with path-planning constraints is
   doing the moving. Our small-set pressure is legitimate but comes from a
   different source (agent context) and must be argued on those terms.

5. **Does joint-space optimization eliminate the unreachable-view problem?**
   *The claim* (Yu & Gupta's founding argument; the branch's own round-2
   framing): visibility, kinematics and collision hold by construction.
   *The correction* (Wang 2026 §III-B, verbatim): only visibility and joint
   limits hold by construction. **Collision is delegated to MoveIt as a
   post-hoc ranked-retry, and 8/100 cases still fail end-to-end.**

6. **Filter up front, or post-filter?**
   *Up front* — Scott 2001/2009 state it as an explicit efficiency
   recommendation; Zacharias's capability map is built for it.
   *Both, hierarchically* — Spensieri, explicitly because the cheap filter
   admits false positives; Zacharias's own twist-discarding and
   collision-exclusion make a map hit necessary-but-not-sufficient.
   *Largely moot* — Gospodnetić: the choice only matters when candidate
   generation is bad.
   *Nobody has the false-positive rate* — Zacharias named the question in 2007
   and it is still open.

7. **Is learned selection brittle in our regime?**
   *Yes* — ObjView-Bench Table 4, NBV-Net collapsing at ΔNSC 0.516 under
   reachability restriction while classical planners lose ~0.13 SC.
   *Partly fixable* — same paper §6.5: difficulty-aware training-set rebalancing
   recovers MA-SCVP's coverage on hard objects with no architecture change.

---

## 8. What is missing — gaps confirmed or newly opened

**Confirmed gaps** (each now supported by multiple independent documents, not
just the round-2 suspicion):

1. **No paper conditions viewpoint generation on a fitted primitive
   decomposition.** Confirmed across all 20 documents, including four surveys
   whose explicit purpose was exhaustive categorization (Scott 2003,
   Peuzin-Jubert 2021, Alsadik 2025, and ObjView-Bench's dataset taxonomy). The
   Bennewitz lineage — NBV, one-shot, prior-guided, benchmark — spans four
   papers and never once introduces shape-primitive conditioning. **This remains
   the branch's clearest novelty claim, and its clearest risk: no precedent for
   how well it works.**
2. **Almost nothing plans views from a defective self-acquired survey cloud.**
   Four independent confirmations (§6.2). Pan 2025 is the only close entry.
3. **Reachability is overwhelmingly a post-filter, and the false-positive rate
   of any pre-filter is unmeasured.** Zacharias named the question in 2007;
   nobody in the corpus answered it.
4. **No numeric orthogonality threshold that transfers.** §7.2.
5. **No closed-form per-primitive view count.** §3.2.
6. **No treatment of structural uncoverability.** The set-cover formulation the
   whole field inherits assumes every surface row has an admissible column; no
   paper discusses what replaces the objective when it does not. Scott 2003
   does not even list it as an open problem.
7. **IK multiplicity is unaddressed everywhere.** `g_kin ≥ 1` (an IK solution
   exists) does not imply the *executed* branch is collision-free, and no paper
   engages with this.
8. **No RGB-D / active-stereo sensor in the CAD-inspection literature at all**
   (Peuzin-Jubert §2.4.1). Any sensor-quality model needs re-derivation.
9. **No defect-detectability content.** Peuzin-Jubert: zero mentions of
   "defect," "flaw," "scratch," "crack." "Inspection" throughout means
   dimensional conformance to a model spec. **The CAD-inspection-planning
   literature and the visual-defect-inspection literature are different bodies
   of work, and our project straddles them — neither serves us fully.**
10. **The enumeration-for-presentability problem has no precedent.** Magaña
    produces a continuous manifold and never discretizes it principledly; the
    CMM chain hands continuous direction clusters to a machine that executes
    every one deterministically. Nobody in this branch has had a *choosing
    agent* downstream.

**Newly opened by round 3:**

11. **Our reachable workspace is non-convex and this is a concrete technical
    obstacle.** Spensieri's exact shape-intersection machinery has an explicit
    convexity requirement (§3, *"the convexity assumption is needed"*), and
    models a manipulator's reach as a **solid ball**. Our reachable shell — ball
    minus a smaller concentric ball, because the standoff radius has a hard
    minimum — is **not convex**. Encoding reachability as a "feature shape"
    would require either convex decomposition (thin slabs) or falling back to an
    opaque per-candidate query.
12. **The ray-casting/visibility computation, not selection, is where our
    engineering budget should go.** Four independent sources (§1.3).
13. **Additivity as a design rule for any per-region functional we define.**
    Mosbach's formal condition (a functional must be additive over disjoint
    partitions to guarantee monotone convergence) plus his quantified
    instability finding: **non-additive functionals — normal deviation and
    incidence angle, the two we most want — can flip between "no subdivision at
    all" and "several rounds" on small threshold changes** (§7.3, Fig. 13–14).
    A real, transferable engineering constraint, not just his internal concern.
14. **Nominal-pose distance is a bad proxy for task success.** Wang 2026: the
    baselines with the *lowest* position error (37.9 mm) and *lowest* angular
    error (15.51°) are **not** the ones with the highest end-to-end success —
    the proposed method wins on success (92/100) while being third on position
    error. Argues directly against scoring views by distance to a canonical
    orthogonal pose.
15. **Naive coverage maximization can be worse than random.** Yu & Gupta,
    Fig. 3.2: after 14 scans, "maximize new physical volume" reaches **33%**
    known C-space vs. **random views at 47%** and MER at 70%. Their explanation
    generalizes: raw "new area seen" ignores structure that matters downstream.
    A numeric caution for any pure-coverage objective.

---

## 9. Implications per research question

### RQ1 — Judging a view

- **The field's answer to "how do I score a view" is: don't score it, gate it.**
  Every mature system (Scott 2009's three-test `m_ij`, Glorieux's
  `f_scat∘f_ray∘f_fov`, Gospodnetić's admissibility pre-filter, Spensieri's
  cones, Bircher's half-spaces) reduces view quality to a **boolean
  admissibility predicate** and lets the combinatorics work on counts. There is
  a good structural reason for this (§2.2: it keeps the coverage function
  submodular for free) and a bad one (§7.1: it makes orthogonality a floor, not
  an objective — which is not what Albert asked for).
- **Our score must therefore be two-layered**: a hard admissibility predicate
  (in-FOV ∧ unoccluded ∧ incidence ≤ τ ∧ standoff ∈ [d_min, d_max] ∧ reachable)
  for the selection problem, plus a **continuous orthogonality/framing score**
  used only to rank among admissible candidates and to pick the per-region
  representative. Keeping these separate preserves the boolean regime for
  selection while still answering "is this the best view within its
  neighbourhood."
- **The threshold τ must be measured for the D405, not inherited.** §3.3's
  table spans 8.6°–75°. Plan an experiment. Scott 2009's asymmetry (60° in the
  baseline plane, 70° orthogonal) says to check whether the D405's tolerance is
  **anisotropic about its stereo baseline** — if it is, our score has an
  orientation dependence nobody would guess.
- **Score against observation constraints, not against a nominal pose.**
  Wang 2026's result — lowest pose error ≠ highest success — is a direct warning
  against "distance to the ideal orthogonal pose" as a proxy metric.
- **Any per-region functional we define should be additive** (Mosbach's formal
  condition), or we inherit his documented threshold instability — which is
  worst precisely for the incidence-angle and normal-deviation functionals we
  most want to use.
- Hand to Branch 1: Scott 2009 Eq. 1–5 (precision + sampling-density models),
  Mosbach's six feature functionals, Magaña's frustum (Eq. 4) and incidence
  (Eq. 5) constraint forms.

### RQ2 — Selecting the best views *(this branch's primary)*

**The recommended architecture, with its citation for every step:**

1. **Generate per-primitive, constraint-driven, orthogonal by construction.**
   One candidate per patch, camera placed **on the surface normal** at the
   working distance (Mosbach Eq. 6; Scott 2009's Optimal Scanning Zone,
   `f_d = 1.02`). Patch size set by recursive subdivision until an incidence-
   angle / normal-deviation / area functional falls below threshold (Mosbach;
   Sadaoui/Lartigue/Koutecky adaptive voxelization). **Not** grid or sphere
   sampling — Spensieri and Gospodnetić both show constraint-driven generation
   strictly dominates it.
2. **Filter feasibility at generation time, hierarchically.** Cheap conservative
   pre-filter first (capability-map-style, Zacharias; or a convex reach
   approximation, Spensieri), exact IK + collision on survivors (Spensieri's
   explicit reason: the cheap filter admits false positives; Zacharias's own
   §VI caveats: no collision, twist discarded). **A capability-map hit is
   necessary, not sufficient.**
3. **Select by exact ILP minimum set cover** over the admissible candidates,
   with the coverage predicate gated by orthogonality (route (b) of §2.1 — bake
   quality into what counts as covering, per Scott 2001's structural argument
   that per-view properties belong in the matrix entry, not in a relational
   constraint). At our instance size this is instantaneous (§2.3) and removes
   the 5.5–40% greedy tax.
4. **Stop on the ILP's own optimality certificate**, with Spensieri's
   gap-bounded anytime variant as the fallback if we ever scale. This closes the
   "Self-Terminating" requirement Scott named in 2003 and criticized as unmet.
5. **When a region has no admissible reachable view, do not silently drop it.**
   The field's set-cover formulation assumes this never happens and gives no
   guidance (§4.5). Our design decision: switch that region's objective to
   *best-available view* and emit an **explicit unreachable-region report** as a
   first-class output. Precedents to lean on: Scott 2001's infeasibility
   objective switch; Vasquez-Gomez's reachability-exhaustion-as-valid-stop.
6. **Do not use a learned selector.** §6.1: the condition that forces this
   literature toward learning (unknown object at inference) does not hold for
   us; and ObjView-Bench documents that learned selection over a fixed discrete
   action space collapses when that space shrinks — which is what reachability
   does to ours.

**The fallback ladder we take** (and must state explicitly, per §4.1):
rung **(i)** hard-filter at generation, composed with rung **(ii)** slide to the
nearest feasible pose *within the region's admissible band* rather than
discarding, composed with rung **(iv)** defer to the next-best candidate for
that region. We explicitly do **not** take pure rung (iii) — Wang 2026 shows it
does not eliminate the failure mode and it destroys the enumerated action space
RQ4 requires. Scott 2009's warning is our stopping condition on rung (ii):
*"excessive mobility constraints will negate the advantages of the... viewpoint
generation scheme"* — if too much of the ideal band is slid away, the
orthogonality premise stops paying for itself and we should report rather than
degrade.

**View count per primitive**: compute it, do not derive it (Spensieri). Expect
single digits to mid-teens for tabletop objects (§3.2's anchor table). Mosbach's
cylinder at `E_IA = 60°, d = 20` → 48 candidates in rings → 22 selected is the
closest published anchor for a cylinder orbit.

### RQ3 — Composing the action space

- **Represent a view as `(position, approach direction)`, deferring twist and
  IK branch to execution.** Three independent lineages converge here: Zacharias
  (the capability map is deliberately direction-level, twist discarded),
  SCVP/MA-SCVP (5-DOF pose, sixth rotational DOF *"has almost no effect"*),
  Wang 2026 (roll explicitly unconstrained in `C_obs`). Orthogonality constrains
  only 2 rotational DOF plus position — roll is free. This is both principled
  and shrinks the space.
- **The primitive decomposition is the natural addressing skeleton** — per-face
  and per-orbit-station indices are geometry-anchored and consistent across
  shapes in a way `{h,v}` bins are not. The CMM clustering chain (§3.1) is the
  precedent for "features that share accessible directions get one address."
- **Reachability must be resolved *before* the space is composed, not after.**
  Zacharias §III-E: reachability is anisotropic and **reshapes** the admissible
  direction set (cone at the workspace border → ring in the interior). If we
  compose first and prune second, the space's topology changes underneath the
  agent — ObjView-Bench's NBV-Net collapse is what that failure mode looks like.
- **Size the space against the empirical anchors** (§3.2): the field's minimal
  covers for tabletop objects sit at 4–16 views. Gospodnetić's expert-vs-optimum
  result (1.5–2× more, deliberately) is the honest counter-pressure — but their
  reason (turntable-replicable patterns simplify the rig) does not apply to us.
- Hand to Branch 5: Tarbox's icosahedral view-sphere and its critique
  (Scott 2003 §9.2.3 — the reduction of 6D pose to 1–2D, orientation about
  boresight fixed by everyone except Tarbox); Mosbach's parameter-domain
  subdivision as an addressing scheme; Zacharias's cone/cylinder SO(3)
  discretization.

### RQ4 — Presenting to the VLM

- **Enumeration is not a concession to the VLM — it is what the field already
  does, for its own reasons.** Scott 2003 §9.2 (generate-and-test dominates
  because synthesis has nonlinearity/convergence problems), Peuzin-Jubert
  §3.2.5 (direct derivation costs constraint modularity), Scott 2009
  (revocability), Gospodnetić §9.3 (their single-stage ablation is not minimal).
  **Four independent, non-VLM reasons converge on our architecture.** That is a
  strong argument and we should make it.
- **But the presentability problem itself has no precedent in this branch.**
  Every method here hands its output to a machine that executes the whole plan
  deterministically — a CMM controller, a MoveIt retry loop, a flight tour.
  **Nobody has had a choosing agent downstream.** This is genuinely ours to
  solve, and Branch 6 owns it.
- **Never offer an action that cannot be executed.** The cost of a
  capability-map false positive is not a wasted cycle but an **action-space
  integrity failure** — the agent commits and execution fails, and its model of
  its own action space becomes wrong. Nobody in the corpus prices this; we must
  state it ourselves. This argues for exact per-candidate IK + collision
  confirmation on the final small set, on top of any cheap pre-filter.
- **A small set is defensible on VLM-context grounds, not on cycle-time
  grounds.** Gospodnetić's experts-use-more result means "fewer views is better"
  needs its own argument in our writeup rather than being assumed.
- Hand to Branch 6: the `(position, approach direction)` representation and its
  three-lineage support; the ranked-candidate-list pattern (Wang's ≤16 configs,
  internal-only) as a design to *avoid* — the ranking must be visible to the
  chooser, not consumed silently.

---

## 10. Implications for benchmark methodology *(hand-off to Branch 2)*

1. **Report absolute coverage, not only renormalized coverage.**
   ObjView-Bench's own numbers make the case: under reachability restriction,
   **ΔNSC ≈ 0 while ΔSC ≈ 0.13** — NSC renormalizes by the restricted space's
   own observable ceiling, so **it hides exactly the loss we care about**. Our
   metric set must include an absolute-to-full-surface measure.
2. **Reachability must be inside the protocol, not controlled for.**
   ObjView-Bench shows method rankings **reorder** under both budget regime and
   reachability restriction — evaluating on an unconstrained viewsphere is
   meaningless for us. And MA-SCVP's practice of applying a uniform reachability
   patch to *all* baselines *"to ensure a valid and fair comparison"* is a
   documented anti-pattern: they controlled *for* reachability instead of
   measuring it.
3. **Measure our own reachable fraction; it is the honest denominator.**
   ObjView-Bench's 45.4/128 (range 26–74) is the only anchor and its robot is
   **undocumented**. Reproduce the equivalent number for UR5e + RG2 + D405 +
   minimum shell, per object and per placement, and report the **structure** of
   the loss (which approach-direction families are excluded), which
   ObjView-Bench explicitly does not.
4. **`e_v = n_Opt / n` becomes measurable for the first time.** Scott 2003
   defined view-plan efficiency in 2003, surrogated `n_Opt` by "the best plan
   anyone has found," and §8.1.6/Table III confirm it was **essentially never
   rigorously measured**. At our instance size we can compute `n_Opt` **exactly,
   by ILP** — so we can report a true efficiency ratio and screen every
   candidate method for redundancy against a proven optimum.
5. **Stratify the benchmark by an object-difficulty axis.** ObjView-Bench's
   `d_plan` (the exact minimum-set-cover count) is the actionable one, and §6.5
   shows it is a real axis, not a post-hoc statistic. Their Table 5 warning is
   the reason: MS's residual gap differs 3× between difficulty tiers. Any
   stopping criterion or headline number we report must be checked against
   difficulty, or it means different things on different objects.
6. **Three concrete ground-truth-for-best-view protocols now have precedent**
   (Branch 2 wanted at least three):
   - **Analytical** — the face-normal / Optimal-Scanning-Zone construction
     (Scott 2009's `f_d·R_o` normal dilation; Mosbach's Eq. 6). On primitives
     this is exact by construction.
   - **Optimization-oracle** — solve exact ILP min-set-cover over a dense
     admissible candidate set; the resulting set *is* the reference answer
     (Spensieri, Pan 2025, SCVP's labeling pipeline all do exactly this).
   - **Human expert** — Gospodnetić §8.3/Fig. 12: two experts placing viewpoints
     with an interactive coverage-visualization tool, compared head-to-head
     against the automated optimum. The only such comparison in the corpus, and
     directly reusable as a protocol.
7. **The uniform sphere baseline is our incumbent and it is already the field's
   control.** MA-SCVP Table XVI's random-set-at-matched-count ablation and
   BENBV's "Random Uniform Sphere" baseline both play this role. Any number
   reported for it elsewhere is a free estimate of our current system's
   performance. Adopt MA-SCVP's ablation design directly: compare our chosen set
   against a same-size random/uniform set to prove targeting rather than size.
8. **Watch the judge–benchmark circularity trap, which this branch makes
   *worse*.** The field's dominant pattern — fold quality into the admissibility
   gate — means our coverage predicate and our benchmark metric will both be
   orthogonality-based unless we deliberately separate them. Scott 2009's
   boolean-spec-compliance design is the one that most invites it. Mitigations:
   evaluate against a *different* quantity than the one the predicate gates on
   (e.g. measured depth quality or defect detectability on the captured image,
   not the planned incidence angle), and include the human-expert arm as an
   independent reference.
9. **Budget the visibility computation, not the solver.** Four independent
   sources (§1.3). If the benchmark is to support iteration over selection
   objectives, the ray-casting/coverage-matrix stage is what needs caching and
   engineering.
10. **Simulate the right noise.** No benchmark in the corpus models our defect
    profile. ObjView-Bench is explicitly idealized (§8); Pan 2025 injects
    synthetic Gaussian pose noise recoverable by ICP; Scott 2009 injects
    sampling noise (0.3% R_min) and pose error (σ_p 4.26–6.15 mm, σ_a = σ_t = 3°).
    Our profile — per-view depth dropout on glossy/dark surfaces plus a ~4 cm
    fused-vs-live offset — is more severe than all three. **Scott 2009's finding
    that pose error primarily corrupts *frustum occupancy* tells us the specific
    failure to instrument:** points the plan believed were framed, that the live
    view does not actually contain.

---

## 11. Cross-branch hand-offs

- **Branch 1 (view quality):** Scott 2009 Eq. 1–5; Mosbach's six feature
  functionals + the additivity condition and its quantified instability;
  Magaña's frustum/incidence closed forms; the full §3.3 orthogonality-threshold
  table; Wang 2026's "nominal-pose distance is a bad proxy" result.
- **Branch 2 (benchmark):** everything in §10. Highest-value seeds:
  ObjView-Bench (difficulty axes, budget regimes, reachability protocol,
  SC-vs-NSC lesson), Gospodnetić (manual-vs-automated ground-truth protocol,
  per-method view-count tables), MA-SCVP (random-baseline-at-matched-count
  ablation, released simulator/dataset-generator/ROS interface), Glorieux
  (ILP-vs-greedy tractability curve and the 23.8% citation caveat).
- **Branch 3 (primitive structure from survey clouds):** Mosbach's explicit
  warning that B-spline reconstruction needs low-noise scans — and the sharper
  concern that a thin-plate-regularized fit over a dent or scratch **smooths
  away exactly the defect we need a view of**. Pan 2025's conservative-inflation
  pattern for treating self-acquired geometry as fallible. Scott 2009's
  pose-error-degrades-frustum-occupancy mechanism.
- **Branch 5 (viewpoint-space structure):** Tarbox's icosahedral view sphere
  and Scott 2003 §9.2.3's critique of it; Zacharias's SO(3) discretization and
  cone/cylinder shape primitives; the `(position, approach direction)`
  representation argument; Mosbach's parameter-domain subdivision as addressing.
- **Branch 6 (VLM interfaces):** the enumeration-for-presentability gap
  (§8, item 10) — nobody in this branch has had a choosing agent downstream;
  the action-space-integrity framing of a reachability false positive;
  ObjView-Bench's NBV-Net collapse as the canonical failure mode of a fixed
  discrete action space that shrinks underneath its chooser.

---

## 12. Follow-up reads worth their cost

Ordered by how load-bearing the missing content is.

1. **Prieto 2000** (PhD thesis, ÉTS) — the source of the 35° threshold. The one
   number in the corpus that is explicitly noise-derived rather than
   system-specific.
2. **Tarbox & Gottschlich 1995** primary text (institutional access) — the
   actual grazing-angle values and the actual selected-set sizes. Needed before
   any claim citing them for numbers ships.
3. **Sadaoui 2018 / Lartigue 2014 / Koutecky 2016** — the adaptive-voxelization
   normal-deviation threshold values, not given in Peuzin-Jubert.
4. **Chvátal 1979 + Nemhauser–Wolsey–Fisher 1978** — never read; the only place
   the submodularity question can actually be settled if we ever need it.
5. **Vafaeesefat & ElMaraghy 2000** — operates on facet/point representations,
   closer to our point-cloud setting than Kweon & Medeiros, and the CMM
   clustering chain's most transferable member.
6. **GS-NBV (Song & Karydis 2025)** — mapped but not read in round 3; still the
   only candidate cylinder-orbit precedent, since Wang 2026 turned out to be an
   axial-cavity paper.
7. **OccamView 2026 / BENBV 2024 / Mizutani FPT 2024 / VF-Plan 2025** — mapped,
   unread. OccamView's Geo-Floor and BENBV's third-party learned-vs-analytical
   comparison table are the two most likely to change a conclusion here.
