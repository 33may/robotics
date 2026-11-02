# Mosbach, Gospodnetić, Rauhut, Hamann, Hagen — "Feature-Driven Viewpoint Placement for Model-Based Surface Inspection"

Machine Vision and Applications 32(1):8, 2021. https://doi.org/10.1007/s00138-020-01116-y
PDF fetched: https://web.cs.ucdavis.edu/~hamann/MosbachGospodneticRauhutHamannHagenMachineVisionAndAppplicationsPaperRevision2AsPublished10292020.pdf (21 pages, open access CC-BY 4.0).

Read in full (all 8 sections + references + author bios). Local copy of extracted text: `/tmp/mosbach2021.txt` (not committed).

---

## Core method

**Pipeline** (Fig. 1, Sect. 1): Virtual Model → Viewpoint Candidates → Selected Viewpoints → Path Planning → Inspection. This paper covers only the second box — **object space exploration** (candidate generation) — and validates candidate quality using a standard downstream **optimization** step (greedy next-best-view set-cover) borrowed from prior work, not proposed here.

**Steps** (Sect. 3, enumerated list before Fig. 2):
1. Approximate the input triangle mesh by a network of **C⁰-continuous B-spline surfaces** (Sect. 3.1; uses Eck & Hoppe [7] reconstruction: surface Voronoi tessellation → dual Delaunay complex → quad merging → per-quad B-spline fit by least-squares + thin-plate fairing, Eq. 2–4).
2. **Recursively subdivide** each B-spline surface in parameter space, driven by a user-chosen scalar **feature functional** `E(S,Ω)` and threshold `t` (Sect. 3.2, Algorithm 1). A region is split into 4 equal quadrants whenever `E(S,Ω) > t`; recursion stops when `E` drops below `t`. Requires `E` to be *additive* (splitting a region partitions its value) for guaranteed convergence to zero as `Ω`→ point.
3. **Place one viewpoint candidate per leaf segment** (Sect. 3.3, Fig. 5): pivot point = surface evaluated at the segment's parameter-center `(u_c,v_c)`; camera position is a **pure fixed-distance offset along the analytic surface normal**:

   `C_{S,Ω} = S(u_c,v_c) + n(u_c,v_c)·d`  (Eq. 6)

   with `n(u,v) = (∂_u S × ∂_v S)/‖∂_u S × ∂_v S‖`. **View direction is the reversed normal — orthogonality to the surface is baked into the pose formula by construction, not scored or optimized.** `d` is a manually chosen constant (system engineer sets it for the desired depth-of-field/standoff), identical for every candidate on a surface — no per-point standoff adaptation.

An **iterative variant** (Algorithm 2, Sect. 3.2.3) instead subdivides globally by always splitting the single highest-`E` segment across the whole model, until a user-specified viewpoint budget `t_n` is reached — useful when the number of candidates is externally constrained rather than driven by a threshold.

## Feature functionals (Sect. 4) — the paper's "quality terms"

All are **parametric-surface-space (B-spline) integrals or point-wise maxima**, not point-cloud/PLY terms and not image-space terms:

| Functional | Formula | Type | Eq. |
|---|---|---|---|
| Curvature | `E_κ(S,Ω) = ∬_Ω (κ1²+κ2²) ‖∂_uS×∂_vS‖ du dv` | integral, additive | Eq. 7–8 |
| Thin-plate energy | `E_TP(S,Ω) = ∬_Ω ‖∂_uuS‖²+2‖∂_uvS‖²+‖∂_vvS‖² du dv` (closed form via control-point matrix `M`, Eq. 9) | integral, additive | Eq. 3, 9 |
| Max normal deviation | `E_ND(S,Ω) = max_{(u,v)∈Ω} cos⁻¹(n_avg(S,Ω)·n(u,v))` | **max-pooled**, non-additive | Sect. 4.3 |
| Max incidence angle | `E_IA(S,Ω) = max_{(u,v)∈Ω} cos⁻¹( n(u,v)·(C_{S,Ω}−S(u,v))/‖C_{S,Ω}−S(u,v)‖ )` | **max-pooled**, non-additive | Sect. 4.4 (unlabeled eq.) |
| Area | `E_A(S,Ω) = ∬_Ω ‖∂_uS×∂_vS‖ du dv` | integral, additive | Eq. 10 |
| Region of interest (ROI) | `E_ROI(S,Ω) = ∬_Ω f_ROI(S(u,v)) ‖∂_uS×∂_vS‖ du dv`, `f_ROI∈{0,1}` binary mask | integral, additive | Sect. 4.6 |

**Important precision on the incidence-angle functional**: it is evaluated *after* the camera pose is already fixed by Eq. 6 (normal offset). It measures, for the *already-placed* candidate, the worst-case (max) angle between the surface normal and the camera ray **anywhere in the segment** — i.e. it answers "is this segment small enough that the whole patch stays under an incidence-angle tolerance from this one orthogonal-at-center camera," not "which pose direction minimizes incidence angle." It is used purely as a **subdivision stopping criterion** (hard threshold: keep subdividing until every point in the segment is within the angle budget), not as a smooth weight summed into a score. Concrete thresholds used in experiments: 60° (cylinder), 85° (spring), 35° (tangle cube) (Table 1; note these vary by object — chosen ad hoc per model, not derived from any sensor model).

## Concrete results / numbers (Table 1, Sect. 6)

Three test objects, all synthetic B-spline models (not raw scans): cylinder (21,800 mesh primitives, 6 B-spline surfaces), spring (51,800 primitives, 14 surfaces), "tangle cube" (82,672 primitives, 452 surfaces, reconstructed from volumetric/medical-like data). For each, 5 functionals compared at thresholds chosen to yield roughly comparable candidate counts:

- Cylinder: `n_vpc` 36–48 candidates depending on functional → NBV greedy selection reduces to `n_opt` 10–24 final viewpoints. Incidence-angle (t=60°) gave 48 candidates → 22 selected.
- Spring: `n_vpc` 194–482 → `n_opt` 9–13.
- Tangle cube: `n_vpc` 1325–1571 → `n_opt` 32–38.

Timing breakdown (subdivision SD, ray-trace visibility-matrix RT, greedy NBV) also reported; RT dominates cost (up to ~3100 s for tangle cube), while subdivision itself is cheap for thin-plate energy (0.001–0.008 s) but expensive for exact curvature (up to 79.9 s for the spring). Machine: i7-4790 @ 3.6 GHz, 16 GB RAM (2020-era single-core CPU, no GPU).

**Redundancy data (directly relevant to "which quality terms are redundant")** — Tables 3 & 4 quantify overlap between thin-plate energy and true curvature by surface region:
- Cylinder (Fig. 11, Table 3): TP puts 18.83% of its value-mass on the flat top/bottom, curvature puts **0%** there — the two functionals **diverge sharply on simple objects with flat regions**. Root cause identified (Sect. 7.1): TP is a physics-motivated deformation-energy proxy that reacts to *parameterization distortion* (mapping a rectangular parameter domain onto an elliptical patch), not to true geometric bending — it is provably nonzero on a flat disk (`κ1=κ2=0` but `E_TP≠0` shown analytically). This is a **named failure mode / negative finding**: thin-plate energy is not a faithful curvature proxy on flat-but-distorted patches.
- Spring (Fig. 12, Table 4): TP and curvature distributions are **nearly identical** (e.g. 8.46% vs 8.71% on "lower surfaces") — on geometrically complex free-form objects the two terms **are redundant**, and the paper explicitly concludes this ("both functionals behave similarly for this model").
- Takeaway citable as precedent: redundancy between two geometric quality terms is **object-dependent** — cheap-but-approximate and exact-but-expensive versions of "the same" concept can diverge exactly where it matters (flat regions), so redundancy can't be assumed a priori from formula similarity; it has to be checked per shape class, and simple primitives (flat panels) are precisely where cheap proxies are least trustworthy.

Qualitative comparison across all 6 functionals (Table 2, "Intuition of values / Computational cost / Meeting design objectives / Additivity") — a compact rubric worth mimicking for our own term-comparison table (Branch 1 deliverable).

## What this paper does NOT do (negative findings, several directly load-bearing for this branch)

1. **No sensor physics.** Zero mention of depth-camera incidence-angle-vs-error curves, active-stereo behavior, RealSense/D405, or any real sensor noise model. Purely a synthetic-geometry / CAD-style planning paper. Does not answer branch questions on σ(range,θ) curves or knee effects — those must come from other sources.
2. **No specular/material modeling.** Section 8 (Conclusions) explicitly lists "optimizing camera positions... to consider additional properties like material behavior or light source placement" as **future work, not done here**. The paper confirms this is an acknowledged, still-open gap in the literature it belongs to — supports the branch's "gap = contribution claim" framing for RQ on off-normal specular avoidance.
3. **No defect-detectability modeling.** The ROI functional (Sect. 4.6) lets an engineer *mark* known-defect-prone regions for denser sampling, but there is no model of how defect visibility varies with viewing angle (no raking-light / grazing-angle detectability treatment at all).
4. **No image-space judging whatsoever.** 100% geometric/analytic; no captured images, no rendering, no VLM. Nothing here informs image-space or render-and-score questions.
5. **No local/gradient optimization of pose.** Pose is a single closed-form normal offset (Eq. 6); no refinement loop, no gradient ascent, no differentiable rendering. The paper's own "future work" is precisely to add camera-position optimization — meaning that as of 2021 this line of work had not yet closed that gap either. Directly answers the branch's "what does the field report about local refinement" question: for this feature-driven strand, the answer is *nothing yet, acknowledged as future work*.
6. **No visibility/self-occlusion in the scoring itself.** Self-occlusion and collision of candidates with the object are explicitly named as **unaddressed limitations** (Sect. 7.4: "the algorithm does not deal with self-occlusions and collisions of viewpoint candidates with the object's geometry. Holes or tight cavities still pose a challenge."). Visibility only enters later, downstream, in the NBV selection step, via a binary `visibility matrix A ∈ {0,1}^{n_F×n_vpc}` built by **ray tracing against the original triangle mesh** (Sect. 5) — i.e. it requires a watertight mesh, not a raw/holey point cloud, and it is a hard 0/1 predicate with no smoothing/soft-visibility treatment discussed anywhere in the paper.
7. **No coverage guarantee, by the authors' own admission** (Sect. 7.4): "The subdivision approach does not explicitly address the total coverage of an object" — it works empirically for the 3 tested models only because subdivision happened to be fine enough everywhere.
8. **No error propagation from noisy input geometry into the score.** Sect. 3.1 only says qualitatively that B-spline fit quality "strongly decreases when exhibiting significant noise or incorrect topology" — there is no quantitative treatment of how upstream geometric uncertainty propagates into `E(S,Ω)` or into the resulting camera pose. This is a genuine gap the branch can claim as unaddressed by this source.
9. **Score is unbounded and not object-comparable.** Feature-functional values are raw geometric integrals; the only normalization offered (Sect. 3.2.1, Eq. 5) is dividing by a **per-object** average `t_avg = (Σ E(S_i))/n_s` to set a threshold as "a percentage of the average" — this makes thresholds *interpretable within one object*, but values/thresholds are **not on a common bounded scale across different objects** (cylinder uses 60°, spring 85°, tangle-cube 35° for the *same* incidence-angle functional — chosen per-model by the engineer, not derived). This is a citable negative precedent for the branch's "is there a bounded, normalized, cross-object score" question: the answer in this line of work is explicitly no.

## What transfers to our setting

- **The constructive "derive pose from local surface normal + fixed standoff" move** (Eq. 6) is exactly the primitive-level deterministic rule Albert is asking for (box face → normal-offset camera; cylinder → radial-normal offset around the axis). It is dead simple and directly reusable for primitive shapes (box faces, cylinder mantle) where we already have (or can fit) clean local normals.
- **Max-pooling (worst-point-in-region) as the aggregation rule for angle-based constraints** (`E_ND`, `E_IA` both use `max` over the segment, not mean) is a real, citable precedent that worst-case pooling is the standard/defensible choice specifically for *angle/tolerance*-type inspection criteria — supports using worst-point pooling for our "is the region within a viewing-angle tolerance" checks, as opposed to mean-pooling which is used for the integral-based (area, curvature, ROI) terms instead.
- **The "threshold-as-stopping-criterion" pattern**: treating incidence angle as a hard, per-segment feasibility gate rather than a smoothly weighted term is a real precedent for framing orthogonality as "within tolerance" rather than "most orthogonal," which matches the branch's stated decision point (soft weight vs hard feasibility constraint) — though note it's applied here to decide *candidate density*, not to rank/select among already-generated candidate poses.
- **The candidate-generation-offline / selection-online pipeline shape** (Fig. 1: candidates generated once from geometry, selection/optimization happens downstream) matches our own intuition that geometry-driven refinement belongs at action-space construction time, not online per VLM request.
- Cylinder test case (Table 1a, Fig. 6) is a literal primitive-shape data point usable for Branch 2's replicability-on-primitives question — though it is B-spline-space, not point-cloud, so the pipeline itself would need re-implementing on our data representation.
- Table 2's rubric (intuition of values / computational cost / meets design objective / additivity) is a good template shape for our own Branch 1 term-comparison table.

## What does NOT transfer, and why

- **Hard representation mismatch**: the entire method requires a *watertight, C⁰-continuous B-spline surface network*, itself requiring successful mesh reconstruction (Eck-Hoppe: Voronoi tessellation → Delaunay quads → parameterization) from a clean triangle mesh. Our input is a **fused PLY with pose error (mm), per-view depth dropout holes, and no mesh at all**. Per the branch's own partition scheme, this entire technique sits in the "**needs a mesh we do not have**" tier — none of it is directly computable on our raw defective point cloud today. The paper itself flags that its upstream fitting step degrades under exactly the kind of noise/topology problems (holes, incorrect topology) our PLY has.
- **No sensor-error or defect-physics content** — cannot answer any of the branch's sensor-curve, specular-avoidance, or defect-detectability-vs-angle questions; those need other sources.
- **Fixed, manually-set standoff distance `d`** (not derived from GSD/DOF requirements algorithmically) — less directly useful than a source that ties standoff to resolution/DOF formulas.
- **Visibility computed by ray-tracing a mesh**, not by any point-cloud-native technique (e.g., hidden-point-removal) — doesn't help answer "how is visibility computed without a surface," and actually reinforces that mesh-dependence is the norm in this literature strand, which is a data point for why HPR-on-raw-PLY is a genuinely separate (and less precedented) approach we'd need to validate ourselves.
- **No ground-truth notion of "best single view"** is proposed or validated — the paper never claims any individual candidate view is optimal; only the *downstream selected subset* is validated (via a greedy heuristic, itself NP-hard-approximate) for *coverage*, not per-view quality. So this source cannot serve as a "best view" ground-truth precedent for Branch 2 — it only ever evaluates viewpoint **sets**, and does so with the same visibility criteria used to generate the set (mildly circular, as flagged generically by the branch's own risk list).

## Section/page map for citation checking

- Abstract, Sect. 1 (Introduction) — pp. 1–4 (paper's own "Page 1–4 of 21").
- Sect. 2 (Related work; MVP/Tarabanis, Tarbox & Gottschlich, Scott, Gronle & Osten, etc.) — pp. 3–4.
- Sect. 3 (Method) — pp. 4–11: 3.1 B-spline model (pp. 5–6), 3.2 Subdivision + Algorithm 1 (pp. 6–8), 3.2.1–3.2.3 threshold/modifications/iterative (pp. 7–8, Algorithm 2), 3.3 Computing viewpoint candidates + Eq. 6, Fig. 5 (pp. 10–11).
- Sect. 4 (Feature functionals) — pp. 8–11: 4.1 Curvature (Eq. 7–8), 4.2 Thin plate energy (Eq. 9), 4.3 Max normal deviation, 4.4 Max incidence angle, 4.5 Area (Eq. 10), 4.6 ROI.
- Sect. 5 (Selecting final viewpoints; set-cover formalism Eq. 11, NBV greedy) — pp. 10–11.
- Sect. 6 (Results, Table 1, Figs. 6–10) — pp. 11–14.
- Sect. 7 (Discussion) — pp. 14–19: 7.1 curvature-vs-TP artifact + Table 3/4 (pp. 15–17), 7.2 intuition (p. 17), 7.3 stability (pp. 17–18, Figs. 13–14), **7.4 Limitations (p. 17, self-occlusion/collision/coverage gaps)**.
- Sect. 8 (Conclusions/future work — material behavior & light-source placement future work) — pp. 18–19.
- References [5,11,14,19,20-24,26-28] are the prior-art lineage (Tarabanis MVP, Tarbox & Gottschlich dense sphere, Scott model-based, Gronle & Osten octree reduction, Sheng et al. bounding-box normal placement, Prieto et al. NURBS voxelization) worth pulling directly for the incidence-angle/threshold branch question if not already covered by other passes.
