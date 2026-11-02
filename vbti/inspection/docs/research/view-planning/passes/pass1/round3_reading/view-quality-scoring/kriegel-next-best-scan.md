# Kriegel, Rink, Bodenmüller, Suppa — "Efficient Next-Best-Scan Planning for Autonomous 3D Surface Reconstruction of Unknown Objects" (J. Real-Time Image Processing 10, 2015)

Fetched as full PDF from `https://www.robotic.dlr.de/fileadmin/robotic/krie_si/kriegel13a.pdf`
(DLR author mirror, open access). Read in full, 20 pages, all sections and both
tables.

## What this paper actually is

An **online, real-time autonomous 3D-modeling system** (not a static scoring
paper): an industrial robot (Kuka KR16) with an eye-in-hand **laser stripe
profiler** (Micro-Epsilon ScanControl 2700-100, triangulation sensor, not
RGB-D/active-stereo) incrementally sweeps an unknown object, building a
triangle mesh and a probabilistic voxel space (PVS) live during each scan
motion, then after each scan **plans the Next-Best-Scan (NBS)** — a
continuous scan *path* (not a single viewpoint, since the sensor is a line
profiler) — using a **utility function that explicitly sums an exploration
(information-gain) term and a mesh-quality term**, weighted by a schedule
that shifts from explore-first to quality-first as scanning progresses. This
is the "IG-only vs IG+quality ablation on a real industrial robot" the branch
selected it for, and that ablation is real, quantified, and run on 9 physical
objects (Sect. 8, Tables 2–3).

Important framing correction for how this feeds the branch: **candidates are
generated online from the currently-known mesh's boundaries/holes, not
scored against a fixed action space or a static externally-supplied point
cloud.** The "action space" here is continuously regenerated from geometry
the system itself is still building and correcting via ICP. This is
structurally different from our setting (VLM picks from a discrete,
pre-enumerated set of views against a static, already-fused survey PLY).

## Core quality-scoring machinery (the part in this branch's scope)

### Per-voxel surface features (Sect. 4.3, computed once, stored in the PVS)

- **Sampling density** (Sect. 4.3.1, Eq. 2–3): `d(v_i) = N_act/N_max ∈ [0,1]`,
  where `N_act` = actual point count in the normal-estimation ball
  neighborhood (radius `R_n`) around vertex `v_i`, and `N_max` is the
  close-packing-of-spheres upper bound `N_max = √2·π·R_n²/R_r²` (`R_r` =
  density-limitation radius). Averaged per voxel to `d̄_i`. **Bounded [0,1],
  PLY/mesh-space, needs local point density only** — directly usable entry
  for the RQ5 term table ("sampling density").
- **Average surface normal per voxel** `n̄_i` (Sect. 4.3.2): averaged over
  all *vertex normals* within the voxel, extracted directly from the mesh.
  Explicitly justified as an **approximation adopted because exact
  per-mesh-face incidence-angle computation "proved to be too time
  consuming."** No error bound is given for how much this per-voxel
  averaging degrades the angle estimate — it is an engineering shortcut, not
  an analyzed one. This is the paper's only brush with RQ7 (score
  sensitivity to normal error): the authors accept a coarsened/averaged
  normal for the sole purpose of an incidence-angle *gate*, not for a
  continuous orthogonality *score*, and never quantify the effect.
- **Border-edge fraction** `b_i` (Sect. 4.3.3, Eq. 4): fraction of mesh
  edges within a voxel that are boundary edges (only one adjacent triangle,
  see Fig. 2/Algorithm 2). `b_i ∈ [0,1]`, a completeness proxy, not an
  orthogonality or angle term.

### The incidence-angle gate — hard threshold, not a smooth weight (Sect. 2.3, Sect. 5.4 Eq. 14)

This is the paper's most load-bearing content for **branch Q2**. Sect. 2.3
("Surface Quality Measure") reviews prior work explicitly on this question:

- Massios & Fisher [26, BMVC 1998] — first to add angle of incidence as a
  quality criterion alongside visibility.
- Vasquez-Gomez et al. [41] — same quality criterion plus travel distance
  and overlap with previous range images; sample 80 candidate views over a
  sphere.
- **Johnson et al. [12, 3DIM 1997] and Mehdi-Souzani et al. [28, JCISE
  2006] both investigate measurement error vs. angle of incidence and "both
  come to the conclusion that the noise is almost constant up to a certain
  angle and increases enormously at an incidence angle greater than 60°.
  Therefore the angle of incidence should be used as quality criterion but
  only as a binary."** (verbatim paraphrase, Sect. 2.3, p.3). This is a
  **citable secondary claim of a knee near 60°**, not a curve fit by Kriegel
  et al. themselves — the primary sources (Johnson 1997, Mehdi-Souzani 2006)
  would need to be pulled directly for the actual `sigma(range,theta)`
  shape; this paper only reports the qualitative conclusion.
- Also from the same section: Suppa [37] finds **surface-to-sensor
  distance** "plays a very important role for laser scanners and should also
  be considered as quality criterion," and Scheibe's evaluation [33] finds
  "3D point quality depends a lot more on the distance than on the angle of
  incidence." **Both are about laser triangulation scanners, not active
  stereo — flag before using for D405.**

Kriegel et al. **operationalize this as a hard threshold at 70°**, one step
more conservative than the ~60° knee reported in the literature they cite —
their own per-voxel quality term (Eq. 14):

```
q_i = { λ·b_i + (1-λ)·d̄_i   if θ < 70°
      { 0                    otherwise
```

with `θ` the incidence angle between the simulated scan beam and the
voxel's average surface normal `n̄_i`, and `λ = 0.7` (completeness weighted
above point density; "proved to be good," no ablation on λ given). **Above
70°, the term is forced to exactly zero — the paper's explicit rationale
(Sect. 5.4, p.12) is that "we assume that a re-scan of this surface area
will not increase the quality of the surface model due to the large angle
of incidence."** This is a genuine feasibility gate baked into a utility
score, not a soft weight — directly answers branch Q2 with a real precedent:
**threshold, not smooth penalty**, and the threshold used in practice (70°)
is slightly *more* conservative than the ~60° knee reported by the primary
sources they cite.

### Aggregation to path/view level — mean pooling (Sect. 5.4, Eq. 15)

`q_s = (1/k) Σ_{i=1}^{k} q_i`, i.e., the surface-quality score for a full
candidate scan path is the **mean** of per-voxel `q_i` over all `k` voxels
intersected by the path's rays. **Not min/worst-point pooling.** This is a
direct, if partial, answer to **RQ9**: this system pools per-voxel quality
to per-candidate quality by simple averaging, with no discussion of whether
worst-point pooling was considered or why mean was chosen (likely because
their target quantity is "expected mesh-quality *improvement* over a
region," not "is the region uniformly good," so mean is the natural
aggregator for that objective — but this is inference, not stated).
`q_s ∈ [0,1]` because both `b_i` and `d̄_i` are individually bounded to
`[0,1]` — a genuine example of a **bounded, normalized quality score**
(partial answer to **RQ8**), though note it scores "how much would scanning
here still improve completeness/density," not "how good is this view of
this surface" in our sense — see negative-transfer notes below.

### Information gain / entropy term (Sect. 5.4, Eq. 13)

`e_v = -(1/k) Σ_{i=1}^{k} [p_i log(p_i) + (1-p_i)log(1-p_i)]` — mean Shannon
entropy of occupancy probability `p_i` over the `k` voxels a ray intersects
until it reaches an occupied voxel. `p_i ∈ [0,1]` from the Bayesian-updated
PVS (Sect. 4.2). This is the **volumetric exploration term**, computed
against the *unknown-space occupancy grid*, entirely separate machinery from
the surface-quality term above (`q_s`), and it is what makes visibility
"soft"/continuous rather than a hard visible/occluded binary — relevant to
**RQ15**: probabilistic occupancy-grid entropy is itself one of the
"smoothing tricks" the branch is cataloguing for a visibility function
that's discontinuous in camera pose, here used for a different purpose
(exploration) but architecturally the same smoothing device (soft occupancy
probability instead of hard raycast hit/miss).

### The combined utility function (Sect. 5.4, Eq. 11–12)

```
f_utility = (1-ω)·e_v + ω·(1-q_s)
w(n_s) = (n_s/n_q) / (n_s/n_q + 1),  n_q = 5
```

`ω` ramps from 0 (pure exploration) toward 0.5 as scan count `n_s` grows;
at `n_s = n_q = 5` scans, exploration and quality are weighted equally, and
`ω` keeps rising slowly thereafter (asymptotic, never fully quality-only in
this formula — check: as `n_s→∞`, `ω→1`, so it does approach pure-quality
eventually). Candidate scan paths are evaluated by casting simulated rays
through the PVS and mesh; the path with **highest** `f_utility` is selected
(Fig. 10 shows red-to-green color-coded candidate rating). This is a
**discrete argmax over a finite candidate set generated per iteration** —
not a gradient-based or differentiable-rendering local optimizer. Directly
negative for **RQ14–RQ16**: view "optimization" here means evaluate-and-pick
the best of a handful of boundary-search-generated candidates, not climb a
continuous quality landscape.

## The ablation: IG-only vs IG+Quality (Sect. 8, Tables 2–3, Fig. 15) — the experiment the branch flagged this paper for

Setup: 9 real objects across 3 categories (cultural heritage: camel, Mozart
bust, Zeus bust; household: cookies box, dog spray bottle, Santa Claus
figurine; industrial: control valve, filter, pressure valve — Fig. 14).
Compared: prior method `IG` (`ω=0` in Eq. 11, i.e., exploration-only NBS
selection, from their earlier paper [19]) vs new method `IG/Quality` (full
Eq. 11 with the scheduled `ω`). Both use the same real-time space update,
ICP pose correction, and termination criteria for fair comparison
(Sect. 8.4).

**Table 2 (verbatim numbers, format `IG`/`IG-Quality`)**:
- Number of scans `n_s` to termination: Camel 30/16, Mozart 15/14, Zeus
  14/15, Cookies 6/5, Spray 30/14, Santa 6/6, Control 9/11, Filter 17/15,
  Pressure 30/14. **Three objects (camel, spray, pressure) never reached the
  target quality under IG-only and aborted at the fixed 30-scan cap** —
  "this also shows that the IG approach does not really aim at increasing
  the quality" (Sect. 8.4.2, p.16–17).
- Total execution time `t` (min): Camel 27.5/9.3, Mozart 10.8/5.5, Zeus
  10.7/6.5, Cookies 3.0/1.5, Spray 21.7/5.8, Santa 3.1/1.9, Control 5.5/3.8,
  Filter 11.7/6.1, Pressure 19.6/5.4. **Average time reduction factor: 2.3×**
  (Sect. 8.4.1) — driven both by needing fewer scans and by a faster
  per-iteration pipeline (Table 3: 17 s/iteration average, only 4 s of
  robot-idle wait per scan vs. 20 s in the prior method — a 16 s/iteration
  speedup attributed independently to the real-time space update, not to the
  quality term itself; don't conflate the two speedups).
- Actual completeness `c_a` (mesh vs. hand-guided ground-truth scan,
  ~25× more accurate reference system): **higher for IG/Quality on all
  objects except spray and Santa Claus**; for spray, IG needed >2× the
  scans and *still* never reached the target point density (Sect. 8.4.2).
- Model RMS error `ē`: mostly sub-millimeter for both methods; pressure
  valve is an outlier (~1.5 mm both methods) attributed to specular
  reflections during scanning, not to the quality term.

**Fig. 15** plots `c_a` vs. scan number for camel and filter: IG/Quality is
**>20 percentage points more complete than IG after only 7 scans** on both
objects; both curves plateau (0–2 pp gap) between scans 12–17, i.e. the
quality term's advantage is concentrated in the early-to-mid regime, not the
long tail. This is the clean "ablation shows a quality term earns its
place" result for **Branch 2's RQ25** — extract verbatim: same protocol
(same modeling/registration/termination machinery, same 9 objects), one
binary switch (`ω=0` fixed vs. scheduled `ω`), four independent outcome
metrics (scans, time, completeness, point density), run on a real robot,
not simulation.

## What transfers to our setting

- **Threshold-not-smooth precedent for the incidence-angle term, with a
  literature trail two levels deep** (Kriegel's own 70° cutoff, backed by
  Johnson 1997 / Mehdi-Souzani 2006's reported ~60° knee) — strong, if
  indirect, evidence for **RQ2**. The consistent message across three
  independent groups (Massios/Fisher, Johnson, Mehdi-Souzani, Kriegel) is
  that incidence-angle degradation for triangulation-style range sensors is
  treated as a **hard admissibility gate in practice**, not a continuously
  weighted penalty — even though physically the underlying noise curve is
  presumably smooth-then-steep, not literally discontinuous. Worth going to
  the primary sources (Johnson 1997, Mehdi-Souzani 2006) for the actual
  curve shape before treating "knee at ~60°" as settled.
- **A working, bounded [0,1], mean-pooled, per-region quality score
  actually used inside a real robot's argmax loop** (Eq. 14–15) — concrete
  precedent for **RQ8** (bounded score) and **RQ9** (mean aggregation, not
  min/worst-point) that a practitioner system found sufficient.
- **A concrete, tunable explore/quality utility blend with a scan-count
  schedule** (Eq. 11–12) — directly the pattern the branch's framing doc
  gestures at (view-quality algorithm as an auxiliary signal, not the whole
  planner) and a template worth reusing/citing if we build an analogous
  two-term score.
- **A worked example of tolerating normal-estimation imprecision for an
  angle-gate purpose** (per-voxel averaged normal, Sect. 4.3.2) — weak but
  real anecdotal input to **RQ7**: the practical answer this paper gives is
  "average normals over a small neighborhood and accept the imprecision,"
  with no quantified tolerance — a gap, not an answer.
- **A concrete analog for the fused-vs-single-view offset problem**
  (**RQ13**): Sect. 6 / Fig. 11 show that without ICP pose correction,
  successive scans misalign by "approximately four millimeter" for one test
  object, and that *too much* ICP correction over-contracts the model —
  i.e., the correction itself introduces a different bias. Same phenomenon
  our framing doc flags at cm-scale (not mm-scale — different sensor/setup)
  but same qualitative lesson: **registration error and registration
  over-correction are both live risks, and "fix it with ICP" is not free.**
- **Probabilistic occupancy-grid entropy as a smoothing device for
  visibility**, independent of its use here for exploration — a real,
  citable instance of the smoothing-trick catalog for **RQ15**.

## What does NOT transfer, and why (first-class negative findings)

- **Sensor modality mismatch — this is a laser triangulation stripe
  profiler, not active stereo.** Depth of field 300–600 mm, operating
  distance 450 mm (Sect. 8.1) — **an order of magnitude farther than our
  10–40 cm D405 range**, and a different noise mechanism (triangulation
  baseline geometry vs. active-stereo IR-pattern correlation). The 60–70°
  threshold and the "distance matters more than angle" claim (Scheibe [33])
  are about this different sensor class. **Does not directly answer branch
  Q1** — no `sigma(range,theta)` curve for anything resembling a D405 is in
  this paper; it only supplies a secondary pointer to two other papers
  (Johnson 1997, Mehdi-Souzani 2006) that would need separate reading.
- **No specular/off-normal-is-better modeling** (**Q3**): the paper
  acknowledges reflective/dark industrial surfaces cause elevated error (the
  pressure valve's ~1.5 mm RMS error and its unreachable back-of-clock area,
  Sect. 8.4.2) but proposes **no directional strategy** to mitigate it —
  reflective surfaces are simply logged as a limitation, not compensated by
  choosing an off-normal viewing angle. No planner in this paper reasons
  about specular lobes.
- **No defect-detectability model whatsoever** (**Q4**): this is a pure
  geometric-reconstruction-quality paper (mesh completeness, point density,
  RMS error against ground truth); "quality" never means "is a scratch
  visible," it means "is the mesh complete and accurate." Silent on this
  question, not even tangentially.
- **Action space is continuously regenerated online from a live,
  self-corrected mesh — not a fixed, pre-enumerated discrete set scored
  against a static externally supplied PLY.** This is the single biggest
  architectural mismatch with our setting. Scan-path candidates come from
  *Boundary Search* (Sect. 5.1) over the mesh the system is itself building,
  and holes trigger *Hole Rescan* (Sect. 5.3) — there is no notion of "score
  this candidate view against a cloud we are handed and cannot change."
  Their PVS/mesh are simultaneously the geometry being judged and the
  geometry being corrected (via ICP, Sect. 6) — the paper never confronts
  "the judge's coordinate frame might be wrong relative to what the camera
  will actually see," because for them the judge *is* the accumulating
  ground truth, continuously self-correcting. **Does not inform Q12
  (propagating input-geometry uncertainty into the score) or Q13's
  "calibration vs. modelling-requirement" framing** — the paper's answer to
  drift is "keep correcting the shared model," which isn't available to us
  (our fused PLY is a fixed input, not a thing we keep re-scanning to
  refine).
- **No hidden-point-removal-style visibility computation** (**Q11**):
  visibility/occlusion is handled by ray-tracing against the PVS occupancy
  grid (built from real sensor sweeps with known viewpoints), not by
  reasoning about visibility from an already-static point cloud with no
  surface. Not applicable to "given only a static PLY, is a point visible
  from a candidate pose" — this paper always has calibrated multi-view
  history to build the occupancy grid from.
- **No image-space judging, no VLM, no learned model anywhere** (**Q17–Q21**):
  entirely classical mesh/voxel geometry and hand-tuned thresholds/weights
  (70°, `λ=0.7`, `n_q=5`) — several of which are stated as "proved to be
  good" with no ablation shown for the specific constant, only for the
  headline IG-vs-IG+Quality switch.
- **No gradient-based or differentiable-rendering local refinement**
  (**Q14–Q16**): "optimization" = evaluate a handful of discretely generated
  candidates (from boundary geometry) and take the argmax of Eq. 11. No
  continuous pose refinement step exists anywhere in the pipeline.
- **Objects are freeform/industrial parts, not primitives** (**Branch 2
  handoff, Q23**): busts, a camel figurine, a spray bottle, a filter/valve
  assembly (Fig. 14/16) — the industrial subset (control valve, filter,
  pressure valve) is the closest to box/cylinder primitives in this set but
  still has integrated caps, threads, and cavities; none is a clean
  isolated box face or cylindrical shell in our sense. Absolute numbers
  (scan counts, timings) are not directly usable as baselines for a
  primitive-only regime, though the qualitative ablation *pattern* (quality
  term saves scans and time, most on early/mid scans) is a reasonable prior.
- **Termination/coverage machinery (Sect. 7, Eq. 16–20) is about knowing
  when the *whole object* is sufficiently modeled**, not about judging a
  *single already-captured view* — tangential to this branch, core to
  branch 4 (coverage/selection), flagged only so it isn't mistaken for an
  RQ1-relevant per-view score.

## Branch-question coverage summary

Directly informs: **Q2** (threshold-vs-smooth — strong, with a two-source
literature trail, though numeric knee is second-hand at ~60° vs. their own
70° operating cutoff); **Q8** (bounded [0,1] score — real precedent, Eq.
14–15); **Q9** (aggregation — mean pooling at path/region level, explicit
formula); **Q13** (registration-offset analog at mm-scale, and the
overcorrection risk); **Q15** (occupancy-grid entropy as a visibility
smoothing device — one citable instance); **Q25 / Branch 2 handoff**
(IG-vs-IG+quality ablation — this *is* the protocol to imitate, full
numbers extracted above, Table 2/Fig. 15).
Partially informs: **Q1** (wrong sensor modality/range — only a pointer to
two other primary papers, no curve of our own); **Q7** (normal-error
tolerance — anecdotal, unquantified); **Q23 / Branch 2 handoff** (object set
is industrial-but-not-primitive; qualitative pattern may transfer, absolute
numbers should not).
Explicitly does NOT inform (negative, first-class): **Q3** (specular/
off-normal — acknowledged as a problem, zero mitigation strategy); **Q4**
(defect detectability — entirely absent, this is a reconstruction-accuracy
paper); **Q10–Q12** (defective static-PLY questions — wrong system model,
this system continuously rebuilds and self-corrects its own geometry rather
than scoring against a fixed given cloud); **Q11** (HPR-style visibility —
uses ray-traced occupancy grid instead, not applicable without calibrated
multi-view sensing history); **Q14, Q16** ("best view in a neighbourhood" /
gradient refinement — discrete argmax over generated candidates only, no
continuous optimization anywhere); **Q17–Q21** (image-space/VLM judging —
absent).
