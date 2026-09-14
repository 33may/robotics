# Gospodnetić, Mosbach, Rauhut, Hagen — "Viewpoint placement for inspection planning" (Machine Vision and Applications 33(1):2, 2022)

DOI: 10.1007/s00138-021-01252-z · Open access (CC-BY 4.0). Fetched as PDF from
`https://d-nb.info/1249061261/34` (German National Library mirror; Springer page
requires IdP auth). Read in full, 21 pages.

## What this paper actually is

Not a viewpoint-scoring paper. It is a **survey + head-to-head benchmark of
viewpoint-*candidate-generation* methods** (the "object space exploration"
step that happens *before* any optimization/selection), evaluated by feeding
each candidate set through one common **next-best-view (NBV) greedy set-cover
selector** and comparing achieved **face coverage** on three CAD test parts.
It also runs, for the first time, a **manual-expert-vs-automated comparison**.
There is no continuous per-view quality score anywhere in this paper — only a
binary **admissible/inadmissible** gate (does a viewpoint satisfy focusing
distance + max incidence angle?) and a **coverage count** (how many mesh faces
does it newly observe?). This matters a lot for how much this document can
serve the branch's "scoring" RQs — see the RQ mapping at the end.

## Core definitions (Sect. 3–4, Fig. 1)

- **Viewpoint** `v = [p, d, u]`, a 9D vector, `p,d,u ∈ ℝ³`: `p` = camera
  position, `d` = optical-axis direction, `u` = up-vector. Camera modeled as a
  "typical industrial area-scan camera producing a 2D image" (not RGB-D, not
  active stereo).
- **Requirements a candidate set must satisfy** (Sect. 3, R1–R4): R1 full
  coverage; R2 adaptive coverage redundancy (more viewpoints where geometry is
  complex); R3 minimize inadmissible viewpoints; R4 scalable coverage
  redundancy (density is a tunable knob). Useful vocabulary, not a scoring
  formula.
- **Inspection constraints used to gate admissibility** (Sect. 5, camera
  settings in Sect. 8): **focusing distance / depth-of-field** (viewpoint
  admissible only if the observed surface point falls inside the DOF
  interval) and **maximum angle of incidence** (viewpoint admissible only if
  the optical axis is within a threshold angle of the local surface normal).
  In the experiments (Sect. 8, p.10): sensor 2448×2050 px, focal length
  12.93 mm, focusing distance 140 mm, DOF interval 135–155 mm, **maximum
  incidence angle = 75°**. Both constraints are implemented as **hard
  thresholds inside ray-traced visibility computation**, not as continuous
  weights or penalty terms — a viewpoint either passes or is discarded before
  it enters the visibility matrix (Sect. 5.1: "By assuming that the viewpoint
  is admissible only if it satisfies the inspection constraints in the center
  of the image, it is possible to filter out inadmissible viewpoints before
  ray tracing takes place").
- **Anchored vs. floating viewpoints** (Sect. 6, Sect. 7): anchored = focal
  point lies on the object surface; floating = focal point lies off-surface
  (e.g., on the convex hull) but the viewpoint may still be admissible if
  within DOF. Useful terminology for our "anchored to fused-PLY surface"
  design.

## Candidate-generation taxonomy (Sect. 2, 4)

1. **Space Sampling** — sample a sphere/space around the object independent
   of geometry, then optionally reposition to satisfy constraints ("passive
   geometry awareness": ray-cast along the initial direction, snap focal
   point to the surface or convex hull intersection, optionally reorient
   `d = -n`). This is literally the "geometry-agnostic viewsphere with
   post-hoc constraint repair" pattern — structurally the same idea as our
   current {h,v} viewsphere.
2. **Vertex Sampling** — viewpoint per mesh vertex, oriented either toward
   object centroid or along `-vertex normal`; optionally octree-filtered to
   thin near-duplicate orientations (Gronle & Osten's method, reproduced in
   Sect. 4.2.2).
3. **Patch Sampling** — subdivide the surface (B-spline patches) using a
   **feature functional** `E(S,Ω)` that measures local geometric complexity,
   and place viewpoints non-uniformly, denser where the functional is larger.
   Four functionals given as closed forms (Sect. 4.3, useful for the
   view-quality-term table, RQ5):
   - Curvature energy: `E_κ(S,Ω) = ∬_Ω (κ₁²+κ₂²) ‖∂_uS×∂_vS‖ du dv` — vanishes
     iff surface is flat.
   - Thin-plate energy: `E_TP(S,Ω) = ∬_Ω ‖∂_uuS‖² + 2‖∂_uvS‖² + ‖∂_vvS‖² du dv`
     — cheap curvature-bending proxy, can have small nonzero values on flat
     regions.
   - Surface area: `E_A(S,Ω) = ∬_Ω ‖∂_uS × ∂_vS‖ du dv` — near-uniform sampling.
   - Normal deviation: `E_ND(S,Ω) = max_{(u,v)∈Ω} cos⁻¹(n_avg(S,Ω)·n(u,v))`,
     with `n_avg` the area-weighted average normal — "geometrically
     intuitive" but sensitive to local deformation/noise.
   These are **region-complexity terms that drive viewpoint density**, not
   per-view image-quality scores; they are the closest thing in this paper to
   a "surface-normal entropy" analog for RQ5's table.

## Selection method used for the benchmark (Sect. 5)

**Next-best-view greedy set cover** (Algorithm 1, p.9): build a visibility
matrix (viewpoint × mesh primitive, via ray tracing); repeatedly pick, among
viewpoints that see the currently-least-covered primitive, the one that
covers the most primitives; repeat until coverage threshold or exhaustion.
Two efficiency tricks introduced (Sect. 5.1): (a) filter inadmissible
viewpoints *before* ray tracing (long pole is ray tracing at ~3.81 s/viewpoint
average); (b) a **stalling threshold** (2% of the best single-viewpoint
marginal-coverage contribution) to stop the "long-tail" phase where
additional viewpoints add redundant, low-value coverage — typically 80%
coverage is reached within the first 30% of viewpoints (Fig. 10 discussion,
p.9). This selection loop is explicitly out of this branch's scope (branch 4
territory) but is the mechanism that turns "admissible" into "selected," so
it's worth knowing the paper never scores views beyond this binary gate + a
raw newly-covered-primitive count (which is unbounded and object/mesh-size
dependent, not normalized).

## Generate-and-use (Sect. 6, new contribution)

A baseline that skips candidate-set generation: pick a random uncovered
primitive, place an anchored viewpoint along its normal at the focusing
distance, keep it if admissible (else discard the primitive), repeat.
Achieves ≥90% coverage convergence on all three test objects within ~100
random draws (Fig. 11) but produces **non-minimal, redundant** view sets —
"no other benefits over the generate-and-test approach" (Sect. 9.3). Useful
only as a fast-prototyping lower bound, not as a quality method.

## Test objects and quantitative results (Sect. 8, Tables 1–4, Figs. 8–12)

Three CAD objects, all **complex free-form industrial parts, not primitives**:
- Gear (99,998 faces / 50,000 vertices) — teeth (freeform) + flat faces,
  self-occlusion inside the central hole.
- Crankshaft (74,018 faces / 37,011 vertices) — elongated, has a purely
  cylindrical section, a parallel-offset section, and a perpendicular
  section; no holes/cavities (so directionally simpler — least self-occlusion
  of the three, Sect. 9.1.2).
- Hirth (87,360 faces / 43,680 vertices) — like the gear, central hole +
  repetitive locally-flat top pattern, smooth sides/bottom.

Selected best-performing numbers (Table 1, filtered version Table 2 — bold =
best per object):
- Gear: Octree vertex sampling, **120** candidates → **23** optimal
  viewpoints → **99.69%** coverage. Patch "exact curvature": 514 → 33 →
  99.03%.
- Hirth: Patch "no subdivision": 70 candidates → 15 → **98.79%**.
- Crankshaft: Patch "no subdivision": **120** → **24** → **99.02%**.

Cross-cutting findings (Sect. 9.1, verbatim-paraphrased):
- **Space (sphere) Sampling "shows no benefits and their use is not
  recommended"** (Sect. 9.1.1, final sentence) — high rate of inadmissible
  viewpoints (avg. discarded 61.9–77% depending on object and variant, Table
  3), and coverage plateaus far below 98% for several variants (e.g., Gear
  min-bbox: 53.33% at 1000 candidates → only 26 optimal views, Table 1).
  Distance-aware repositioning (snap-to-surface/convex-hull + reorient to
  `-normal`) helps a lot over fixed-radius sphere sampling but still
  underperforms geometry-native methods.
- **Vertex Sampling and Patch Sampling are "both good approaches"** (Sect.
  9.4) — geometry-native placement (per-vertex or feature-functional-driven)
  consistently reaches >98% coverage with far fewer candidates and far lower
  discard rates (Table 3: Vertex ~26–46% discarded, Patch ~0.3–27%, vs. Space
  ~34–77%).
- Direct **vertex-normal orientation beats centroid-facing orientation**
  (Sect. 9.1.2) for coverage and inadmissible-viewpoint rate — i.e., aiming
  each candidate at the local surface normal (closer to "orthogonal") is
  empirically better than aiming everything at the object center, on these
  three parts.

## Manual (expert) vs. automated planning (Sect. 8.3, 9.4, Fig. 12)

Two experts planned each object by hand in an interactive tool (Gospodnetić
et al. 2020, same group). Findings:
- Experts used **more** viewpoints than the best automated result (e.g., Gear:
  44 vs. automated 23; Crankshaft: 34/32 vs. 23) but their viewpoints show a
  **structured spatial pattern** — e.g., on the Gear, most side-inspection
  viewpoints were two fixed positions replicated around the object via a
  turntable rotation (Sect. 7's "turntable option," Fig. 7). Automated sets
  show **no such pattern** — "the inspection can only be implemented using a
  manipulator, which entails other challenges such as path planning" (Sect.
  9.4).
- Direct quote, load-bearing for us: *"An automated approach, as currently
  formulated, will always start the evaluation of a good viewpoint placement
  on the surface of the object. That way, an object will surely be observed
  in the center of the image, however, it does not necessarily maximize the
  surface which the viewpoint could cover if it was placed nearby without
  touching the surface. A need for such viewpoints is likely to be higher for
  topologically complex objects. The automated viewpoint generation provides
  no possibility to include such reasoning..."* (Sect. 9.4, p.18). This is a
  **negative finding about anchoring-to-surface / orthogonality-first
  heuristics**: coverage-maximizing viewpoints for complex topology can
  legitimately want to be off-surface / off-normal, and no method in this
  survey models that trade-off.

## What transfers to our setting

- **Empirical support for abandoning a geometry-agnostic viewsphere.** This
  is the single most load-bearing transfer: on a controlled benchmark, sphere
  (viewsphere) sampling is explicitly the worst-performing family and "not
  recommended," while geometry-native (vertex-normal / feature-functional)
  generation wins on coverage-per-candidate and admissibility rate (Sect.
  9.1.1, 9.4). This directly backs the framing doc's premise that a
  geometry-agnostic {h,v} sphere is the wrong substrate for an
  orthogonality-seeking action space, from an independent, quantified source.
- **A precise, industrially-used, binary formalization of "orthogonal
  enough."** Their max-incidence-angle constraint (75° in their experiments)
  and DOF constraint are exactly the "within tolerance and reachable" framing
  the branch question poses, applied in a real inspection-planning tool. It's
  evidence that **practitioners treat incidence angle as a hard admissibility
  threshold**, not a smooth weighted term, at least for image sharpness/
  self-occlusion purposes (not for depth-sensor noise, which this paper never
  discusses — see below).
- **Vocabulary/requirements list (R1–R4)** and **anchored/floating
  viewpoint** terms are directly reusable in our own spec.
- **Feature functionals** (curvature, thin-plate, area, normal-deviation) are
  citable, closed-form region-complexity measures — candidates for the RQ5
  view-quality-term table under "surface-normal entropy"/"curvature-driven
  density," tagged PLY-space (need normals + local surface fit).
- **Negative caution on orthogonality-first anchoring** (Sect. 9.4 quote
  above): even in pure-coverage terms (no sensor noise involved), forcing the
  focal point onto the surface can suppress opportunistic higher-coverage
  viewpoints on complex topology. Worth flagging against our "evidence views
  must be orthogonal" constraint if we ever want to loosen it for coverage
  reasons — though our setting is single-region evidence capture, not
  full-object coverage planning, so the tension is milder for us.

## What does NOT transfer, and why (first-class negative findings)

- **No sensor error model at all.** The camera is treated as an idealized
  area-scan device; "quality" = in-DOF + within-incidence-threshold +
  ray-traced visible. No depth noise, no active-stereo baseline geometry, no
  D405, no `sigma(range, theta)` curve of any kind. **Does not answer branch
  Q1.** This is a coverage-planning paper for 2D image acquisition
  (defect-detection by image analysis downstream), not a 3D
  reconstruction/metrology paper.
- **No specular/glossy-surface modeling, no off-normal-is-better case.**
  Not discussed; incidence-angle constraint is purely about image sharpness/
  self-occlusion avoidance and framing, never about reflectance. **Does not
  answer branch Q3.**
- **No defect-detectability model.** "Coverage" here means a mesh face is
  geometrically visible within DOF and incidence-angle limits — nothing
  about whether a scratch/dent is *more visible* at that angle. **Does not
  answer branch Q4.**
- **Operates on clean, complete, watertight CAD meshes (tens of thousands of
  faces), not fused point clouds.** No pose error, no dropout holes, no
  fused-vs-single-view offset, no discussion of point-cloud density
  non-uniformity, no hidden-point-removal for visibility (visibility is exact
  ray-tracing against a known mesh with known face normals). **This paper's
  visibility/coverage machinery does not carry over to our defective-PLY
  setting without first solving mesh reconstruction** — directly relevant
  negative for RQ10/RQ11/RQ12/RQ13 (needs-a-mesh-we-do-not-have bucket).
- **No continuous, bounded, per-view score.** Quality is binary admissibility
  (pass/fail) plus a raw "newly covered primitive count" used only inside the
  greedy selector — unbounded, mesh-resolution-dependent, and not comparable
  across objects or normalized in any way. **Does not answer branch Q8**; if
  anything it's a counter-example (a real, published, industrially-grounded
  system that never needed a continuous per-view score because its use case
  is offline set-cover planning, not per-view interactive judging).
- **No local/gradient-based viewpoint optimization or differentiable
  rendering.** Viewpoints are generated then only *repositioned* by a single
  ray-surface intersection snap (Sect. 4.1.2) — a closed-form projection, not
  an iterative refinement or gradient step. **Does not answer branch
  Q14–Q16.**
- **No image-space judging, no VLM, no learned model anywhere.** Entirely a
  classical ray-tracing / computational-geometry pipeline. **Does not answer
  branch Q17–Q21.**
- **Test objects are complex free-form/industrial parts (gear, crankshaft,
  hirth spline), not primitives.** Flags for **Branch 2 handoff (RQ23)**:
  none of these three objects is a box or cylinder in our sense; our
  primitive-level regime (single flat face, single cylindrical shell) is
  simpler than anything benchmarked here, so absolute coverage/viewpoint-count
  numbers from this paper are not directly usable as baselines for us, though
  the qualitative ranking (geometry-native > sphere) should still hold or be
  easier to achieve on primitives.
- **Ablation structure is candidate-generation-method-vs-method under one
  fixed selector, not quality-term-vs-quality-term ablation.** For **Branch 2
  handoff (RQ25)**: the useful protocol to imitate here is the *"same NBV
  selector held fixed while candidate-generation strategy varies"* design
  (Fig. 9 pipeline) — it isolates generation quality from selection-algorithm
  effects, which is a clean protocol pattern, but it is not an
  information-gain-vs-quality-term ablation of the kind the branch question
  asks for.

## Branch-question coverage summary

Directly informs: orthogonality-as-threshold precedent (Q2, partial — no
sensor-noise curve, but a real hard-threshold industrial practice, 75°);
sphere-vs-geometry-native evidence (transfers to our core premise, not a
listed RQ verbatim but foundational); feature functionals for the term table
(Q5, partial entries); anchored/floating terminology and the
anchoring-suppresses-coverage caution (tangential to Q3's "does orthogonality
ever hurt," though for a coverage reason, not a specular-reflection reason).
Explicitly does NOT inform: Q1 (sensor error curve), Q3's specular case, Q4
(defect detectability), Q6 (redundant-term correlation — no scoring terms to
correlate), Q7 (normal-estimation-error sensitivity — no such model), Q8
(bounded normalized score — answer is "no, not here"), Q9 (aggregation
unit — closest is per-face binary visibility, no worst-point/mean pooling
discussion), Q10–Q13 (defective-PLY questions — wrong input representation
entirely), Q14–Q16 (local optimization), Q17–Q21 (image-space/VLM judging).
Branch 2 handoff: Q23 (object set mismatch — free-form parts, not
primitives), Q25 (candidate-generation-vs-selector ablation protocol worth
imitating).
