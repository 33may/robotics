# Branch 1 — `view-quality-scoring` — Round 4 accumulation

## Scoring a single view: metrics and local optimality

**Date:** 2026-08-27
**Role:** branch synthesizer, pass 1 / round 4
**Inputs:** [`../round1_plan.md`](../round1_plan.md) § Branch 1 ·
[`../round2_map/view-quality-scoring.md`](../round2_map/view-quality-scoring.md) (53 candidates, ranked) ·
[`../round3_reading/view-quality-scoring/`](../round3_reading/view-quality-scoring/) (20 documents read in full)
**Contract:** [`../../../../2026-08-27-geometry-aware-view-planning.md`](../../../../2026-08-27-geometry-aware-view-planning.md)

**Scope note.** This document accumulates everything the branch established. It is
organised by the branch's own questions (Q1–Q25 from the round-2 map), then by what
is contested, then by what is missing, then by explicit implications per RQ1–RQ4 and
for the benchmark. Contradictions between sources — and between sources and this
project's own framing — are preserved verbatim, not reconciled.

---

## 0. Executive summary

Twenty documents were read in full. The branch's central technical question — *what
is the depth-error-versus-incidence-angle curve for our sensor* — **was not answered
by any of them**, and the reading surfaced a load-bearing correction to the project's
own sensor model in the process: **the Intel RealSense D405 is a passive
visible-light stereo camera with an IR-cut filter and no projector**, not the active
stereo device the branch (and the framing doc) assumed. That single fact invalidates
the analogy to the entire structured-light noise-model literature the branch was
built around, and rewrites the specular-failure question.

What *was* established is substantial:

1. **A large, checkable term inventory** — roughly 45 distinct computable view-quality
   terms across point/PLY, voxel, mesh and image space, with formulas and equation
   numbers, partitioned by what our defective fused PLY can actually support today.
2. **A firm answer on smooth-vs-threshold**: the field's practice is *both, at
   different pipeline stages* — a continuous quality model, and a hard feasibility
   gate near the failure knee. Six independent groups do exactly this. But the knee
   value is contested across a 45°–75° band, and none of those numbers was measured
   on a passive-stereo camera at 10–40 cm.
3. **A unanimous negative on input-geometry uncertainty**: across 20 documents and
   three separate survey traditions, *nobody* propagates uncertainty in the input
   geometry into a view-quality score. The field names this gap at its own origin
   (Tarabanis 1995 §III-C-1) and again in 2003 (Scott §8.1.5) and again in 2020
   (Zeng §1). This is the branch's strongest contribution claim.
4. **A unanimous positive on refinement timing**: five independent systems refine
   candidate views *once, offline, at action-space construction time* — never in
   response to a live request. This directly settles a design fork the plan expected
   the literature to leave open.
5. **Two ablation protocols worth imitating verbatim** (Kriegel's IG-vs-IG+quality
   switch on a real robot; Delmerico's isolate-the-score head-to-head), and one clean
   current example of the exact circularity trap the benchmark must avoid.

And what was *not* covered: the entire image-space tier (Tier 7 of the round-2 map)
went unread, as did the four sources specifically designated for the bounded-score
and term-correlation questions. Those are the branch's largest coverage holes.

---

## 1. Source inventory: what each document actually delivered

| Document | Delivered | Did not deliver | Access note |
|---|---|---|---|
| `intel-d405-product-and-datasheet` | **D405 is passive stereo, no projector** (Tbl 2-2/3-15/3-49); Min-Z by resolution (100 mm @ 1280×720, 70 mm @ 848×480, 40 mm @ 424×240); ±2 % Z-accuracy ≤0.5 m; invalid-depth-band formula `B·F/Z` | Any angle data whatsoever | Primary datasheet, read directly |
| `nguyen-izadi-kinect-noise-model` | Closed-form σ_L, σ_z vs range **and** angle (Eqs 1–4); 70 ° operational cutoff; σ-weighted fusion | D405-usable numbers (Kinect v1, 0.5–2.75 m); any specular data | Author PDF |
| `soudarissanane-incidence-angle-tls` | α = arccos(P·N/\|P\|\|N\|); noise ∝ 1/cos α; **~20 % of room-scan noise attributable to incidence alone**; RMSE point→patch pooling | Fitted joint σ(ρ,θ); D405-class physics | Free workshop PDF |
| `soudarissanane-scanning-geometry-isprsj` | Physical derivation `P_R ∝ cos α / ρ²` (Eq 4.16); `c_I(α)=cos α`, `c_R(ρ)=(ρ_max²−ρ²)/(ρ_max²−ρ_min²)`; point→patch→view propagation; **Cartesian-patch density confound (negative finding)**; **corner viewpoint beats centre viewpoint by ~25 %** | Anything for a bistatic/stereo sensor | **Journal PDF closed everywhere**; read via the author's TU Delft thesis Ch. 4–5 |
| `tarabanis-sensor-planning-survey` | The 8-term constraint vocabulary; MVP margin formalism; **feature-uncertainty named as an open gap in 1995**; Anderson's minimal-enclosing-cone; Cowan & Bergman's specular-region subtraction | Any depth-sensor content | Read via browser bot-challenge bypass on Columbia Academic Commons |
| `tarabanis-mvp-system` | Generalized viewpoint (8-D); Eqs 1–6 for resolution/DOF/FOV; visibility-as-signed-distance; **authors' own disavowal of weighted-sum scoring**; focus-constraint discontinuity mechanism | Any incidence-angle term at all | Open green copy |
| `tarabanis-feature-detectability-constraints` | **Nothing from the paper itself** — plus two proxies: Roberts & Marshall's "best view = smallest angular offset from inverse surface normal within the visibility region"; Magaña 2023's φ_sf constraint and the 60 °/45 ° thresholds | Its own equations | **Closed access, no OA copy exists**; the given URL pointed at a different paper |
| `scott-roth-rivest-vpp-survey` | 15-requirement taxonomy; `m_v` = area-weighted compliance ratio; **the canonical view-sphere critique**; pose-error framing (§9.3); **Tarbox threshold critique**; "no standardized benchmarks" (§8.1.6) | Any formula for measurement error | NRC accepted manuscript |
| `zeng-view-planning-survey` | `f_quality = n̂·v̂` (Massios & Fisher); Delmerico entropy formalism; **Table 4: never more than two utility terms combined across 25 systems** | FOV/DOF/GSD (zero occurrences); any sensor content | Open access |
| `bonaventura-viewpoint-selection-survey` | **22 measures with exact equations** in one table; top-5 ranking on Dutagaci GT; viewpoint entropy's resolution-divergence; visibility ratio as the one genuinely bounded measure; stability via JS divergence | **No correlation matrix — deferred to future work by the authors** | Open access (PMC) |
| `delmerico-volumetric-ig-comparison` | 7 VI formulations with equations; **Area Factor's bounded unimodal target-band score**; occlusion-aware visibility relaxation; timing (<1 % spread); **entropy-AUC decoupled from coverage-AUC** | Any incidence/orthogonality content (its "visibility" is occlusion, not facing angle) | Author PDF; Table 1 partially garbled in extraction |
| `wu-quality-driven-poisson-autoscanning` | `w_o = exp(−(1+n^T d)²/(1−cos σ)²)` soft orthogonality (σ = 25 °); `w_d` standoff Gaussian; bounded [0,1] composite; three different pooling operators in one paper; **sphere-constrained vs adaptive search-space ablation** | Any sensor-measured angle tolerance (σ = 25 ° is a tuned hyperparameter); mesh-free visibility | Open PDF |
| `border-gammell-see` | **Mesh-free everything**: point-proximity occlusion (Alg 8), HPR-style normal-sign resolution (Alg 6), maximin least-occluded direction (Alg 10), density↔standoff↔FOV closed form (Alg 2) | Any inspection-quality criterion; a bounded score (its out-degree count is a negative exemplar) | Open (arXiv/IJRR CC-BY) |
| `kriegel-next-best-scan` | Bounded [0,1] quality `q_i` with a **hard 70 ° gate**; sampling-density term `d(v_i)=N_act/N_max`; scheduled explore/quality blend; **the IG-vs-IG+quality ablation on 9 real objects** | A σ(ρ,θ) curve (only a secondhand ~60 ° knee via Johnson 1997 / Mehdi-Souzani 2006) | DLR mirror |
| `burusa-gradient-local-nbv-plant-nodes` | Differentiable ray sampling + transmittance (Eqs 4–8); **10× fewer ray-trace calls than local sampling at parity**; 0.06 s/call | Any normal/orthogonality term (the formulation has no normals at all); IK-constrained validation | Open (arXiv) |
| `defect-detectability-viewpoint-optimization-aiat25` | 4 image-space terms (Vis/Size/DisB/DisC, Eqs 1–5); sample-then-gradient-refine architecture; **a clean current circularity example** | **Any numbers at all** — no table, no ablation, weights undisclosed, N = 1 object. Also: the round-2 blurb's "handles unknown defect location by ROI sampling" is **future work, not implemented** | Open (CC BY, via ACM PDF endpoint) |
| `staderini-visual-quality-inspection-planning` | Binary visibility matrix (occlusion ∧ FOV ∧ DOF ∧ incidence-threshold ∧ projector-and-camera); `Cov(ι_j)∈[0,1]` mean-pooled; **BC/WS/BO ablation, Table 6.1**; Bayesian optimization chosen *because* the objective is non-differentiable | A σ(ρ,θ) curve; material-stratified numbers | **IROS PDF gated**; read via the author's TU Wien PhD thesis Ch. 2 + Ch. 6 |
| `staderini-spatial-resolution-metric` | Abstract-level only: sampling-density matrix concept, mesh requirement, resolving-power-not-noise scoping correction | **Everything else** — no formula, no numbers | **Closed, confirmed zero OA locations** |
| `mosbach-feature-driven-viewpoint-placement` | `C = S(u_c,v_c) + n(u_c,v_c)·d` — pose derived from the normal by construction; six feature functionals; **max-pooling for angle criteria**; **TP-vs-curvature redundancy is object-dependent, diverging exactly on flat regions** | Sensor physics; local optimization (named as future work); self-occlusion in scoring | Open (CC-BY author PDF) |
| `gospodnetic-viewpoint-placement-inspection-planning` | Head-to-head of candidate-*generation* methods; **"Space (sphere) sampling shows no benefits and their use is not recommended"**; 75 ° incidence threshold in practice; **expert-vs-automated: anchoring to the surface suppresses higher-coverage off-normal views** | Any continuous score; any sensor model | Open access (DNB mirror) |

**Provenance flag for the write-up:** three of the branch's designated formula sources
were not read as published — Soudarissanane's ISPRS-J paper and Staderini's IROS paper
were read via the authors' own theses (equation/figure numbering therefore differs from
the published versions), and Tarabanis's CVGIP 1994 paper — flagged HIGH priority as the
branch's highest formula-density source — **could not be read at all**. Any citation to
these must carry the provenance.

---

## 2. The orthogonality claim, grounded (Q1–Q4)

### 2.1 Q1 — σ(range, θ) for the D405 at 10–40 cm: **not answered, and the question needs restating**

No document in the read set supplies a depth-error-versus-incidence-angle relationship
for anything resembling our sensor at our range. What exists:

| Source | Sensor | Range tested | Model |
|---|---|---|---|
| Nguyen/Izadi | Kinect v1, structured light, 7.5 cm baseline | 0.5–2.75 m | `σ_L(θ)[px] = 0.8 + 0.035·θ/(π/2−θ)`; `σ_z(z,θ) = 0.0012 + 0.0019(z−0.4)² + (0.0001/√z)·θ²/(π/2−θ)²` |
| Soudarissanane 2009/2011 | FARO LS880 / Leica HDS6000, monostatic phase TLS | 5–50 m (plate), 0–6.5 m (room) | `P_R ∝ cos α / ρ²` (Eq 4.16); `c_I(α)=cos α`; `c_R(ρ)=(ρ_max²−ρ²)/(ρ_max²−ρ_min²)` |
| Khoshelham/Elberink (mapped, unread) | Kinect v1 | indoor | random error quadratic in range |
| D405 datasheet | **passive visible stereo, 18 mm baseline** | ≤0.5 m qualified | ±2 % Z-accuracy, ≤1 % RMS spatial noise, ≥99.5 % fill — **a band spec, not a curve; zero angle dependence anywhere in 159 pages** |

The Kinect model is the closest in *form* — its shortest tested range (0.5 m) already
exceeds our longest (0.4 m), and near-field active-stereo/passive-stereo noise is
dominated by effects it never measured (minimum-disparity floor, pattern/texture defocus
at close range). The TLS model is closest in *physical rigour* but is derived from a
monostatic backscatter link budget that does not describe a triangulating stereo pair at
all.

**The load-bearing correction.** The D405 datasheet read establishes, at table level:
IR Projector = **X / N/A** (Tbl 2-2, Tbl 3-49); left/right imager filter = **IR-cut**
(Tbl 3-15); and footnote 4 to Tbl 4-14: *"KPIs for D405, a passive camera, are measured
using a textured target and typical ambient room light (~250 Lux). All other models are
active…"* Two other reading notes in this same branch (`nguyen-izadi-…` and
`tarabanis-feature-…`) assert the opposite — that the D405 is active stereo with an IR
texture projector — and the branch's own round-2 framing text says *"an active-stereo
sensor like the D405."* **The datasheet is primary and wins.** Consequences:

- The structured-light noise lineage (Kinect, fringe projection, Bertagnolli's 45 °
  structured-light threshold) is the *wrong* analogy. The right analogy is
  **passive/ambient stereo correspondence noise** — texture-dependent, ambient-light
  dependent, and degrading at grazing angle through foreshortened correspondence-window
  overlap rather than through projected-pattern decorrelation.
- The specular mechanism the framing doc assumes ("the D405's projector reflects back
  into the imager") **cannot occur — there is no projector.** A plausible replacement
  mechanism exists (an ambient specular highlight locally saturating both imagers and
  killing correspondence) but is unconfirmed by any source read.
- **A second candidate root cause for our stated "depth dropout on glossy/dark
  surfaces" appears**: on a low-texture matte primitive face, passive stereo can lose
  fill rate *independently of incidence angle*, purely from lack of texture, because
  there is no projector to paint any on. The branch's current framing attributes all
  dropout to angle/specularity; these are separable causes and should be separated
  experimentally.

**Two sensor-intrinsic numbers that *are* citable and do bind on us:**

- **Min-Z is resolution-dependent**: 100 mm at 1280×720, 70 mm at 848×480, 40 mm at
  424×240 (Tbl 4-11). The marketed "7 cm" is only true at 848×480 or below. If the
  reachable-shell minimum must clear Min-Z *at full resolution*, the binding number is
  **10 cm, not 7 cm** — a real, quantified resolution-versus-standoff trade the shell
  design has to make explicitly.
- **Stereo-baseline invalid-depth band** (§4.5, derived with D405 values): a strip of
  width `B·F/Z` px at one image edge has no depth by construction —
  ≈10 % of image width at 10 cm, 5 % at 20 cm, 2.5 % at 40 cm. Angle-independent,
  computable from `(B, Z, HFOV, HRES)` alone with no PLY and no normal, and **worst
  exactly where we want to operate.** This argues against pushing standoff to the
  minimum even before incidence angle enters.

**Verdict: Q1 is a genuine gap that must be closed by our own measurement.** No
published passive-stereo-at-10–40 cm incidence characterisation surfaced. The
functional templates available for fitting are: Nguyen's `a + b(z−z₀)² + c·θ²/(π/2−θ)²`
form, and Soudarissanane's separable `cos α / ρ²` form.

### 2.2 Q2 — smooth or threshold: **both, at different stages** — but the knee is contested

**The field's actual practice is a two-tier design**, and it is precedented by six
independent groups:

| Group | Continuous model | Hard gate |
|---|---|---|
| Soudarissanane et al. | `c_I(α) = cos α`, `c_R(ρ)` — explicitly framed as [0,1] *reliability coefficients* | Same authors' planning chapter: `α_max` — "parts observed with incidence angles greater than α_max are considered not visible anymore" |
| Nguyen/Izadi | Hyperbolic `σ_z(z,θ)`, continuous everywhere | **70 °** ICP exclusion cutoff |
| Kriegel et al. | `q_i = λ b_i + (1−λ) d̄_i`, bounded [0,1] | **`q_i = 0` above 70 °** |
| Staderini/Gospodnetić (AIT) | — | **75 °** binary admissibility, inside the visibility matrix |
| Magaña 2023 (proxy) | — | **60 °** generic; **45 °** for structured-light sensors |
| Mosbach et al. | — | Per-object stopping thresholds: **60 ° / 85 ° / 35 °** |
| Wu et al. | `w_o = exp(−(1+n^T d)²/(1−cos σ)²)`, σ = 25 ° — smooth, no knee | none |

**The knee band spans 45 °–75 °, and every value comes from a different sensor class.**
Secondary evidence for a real physical knee near 60 °: Soudarissanane's own finding
("above 60 ° the incidence angle dominates the scan point precision"); Lichti 2007's
independent 65 ° a-priori recommendation; and, via Kriegel §2.3, **Johnson 1997 and
Mehdi-Souzani 2006 independently concluding that noise is "almost constant up to a
certain angle and increases enormously at an incidence angle greater than 60 °,
therefore the angle of incidence should be used as quality criterion but only as a
binary."** Those two primary sources were reached only secondhand and remain unread.

**A genuine physical cliff also exists on top of the smooth curve.** Soudarissanane
discarded 9 of 54 plate scans outright because at 60 ° and 50 m *"the return signal
strength is too low for the detection unit to be detected"* — no measurement at all,
not a degraded one. So a hard gate is not purely an engineering convenience.

**And the field's canonical survey says the thresholding practice is wrong.** Scott,
Roth & Rivest, criticising Tarbox & Gottschlich's grazing-angle thresholds: *"The only
grazing angle effects considered were the impact on dilution of surface sampling
density and viewability at acute angles… This masked the most important impact of
grazing angle, which is on measurement error."* Read together with the practice above:
**model smooth, gate hard, and do not mistake the gate for the physics.**

### 2.3 Q3 — does orthogonality ever hurt: **contested, weak, and mechanism-dependent**

Evidence *for* an off-normal optimum:

- **Two independent within-group TLS observations of near-normal degradation.** The 2009
  plate experiment has an off-trend noise point near 0–10 °; the 2011 room experiment
  shows *"a circular pattern of higher noise levels… corresponding to a surface oriented
  nearly perpendicular to the laser beam… A possible explanation… saturation of the
  detection unit."* Both are explicitly labelled speculative by the authors. The
  mechanism (backscatter intensity overload) does not transfer to a passive stereo pair.
- **Cowan & Bergman** (via the Tarabanis survey) model reflectance as Lambertian plus a
  specular lobe with angular tolerance and **explicitly subtract the specular region
  from the admissible illuminator domain** — a citable precedent that specular geometry
  can be planned around rather than merely suffered, though for a separate light+camera
  rig, not a coaxial one.
- **A coverage-side argument, from a different direction entirely** (Gospodnetić §9.4):
  *"An automated approach… will always start the evaluation of a good viewpoint
  placement on the surface of the object… however, it does not necessarily maximize the
  surface which the viewpoint could cover if it was placed nearby without touching the
  surface… The automated viewpoint generation provides no possibility to include such
  reasoning."* Human experts chose off-surface, non-normal views that covered more.
  This is orthogonality hurting for a *coverage* reason, not an optical one — milder for
  us (we do single-region evidence capture, not full-object coverage) but real.

Evidence *against*, or pointing the other way:

- For **monostatic** TLS the failure runs backwards: *"in the particular case of
  specular reflections no light signal is returned back to the scanner if the surface is
  not perpendicular to the emitted beam."* Near-normal is the *only* geometry that
  returns anything from a mirror-like surface for that sensor class.
- Every reconstruction-lineage planner read (SEE, Wu, Mosbach, Kriegel, Staderini)
  builds orthogonality in by construction with no exception path, and none discusses
  specular avoidance. Wu's silence is the loudest: a leading quality-driven autoscanner
  never considers the failure mode at all.
- **For the D405 specifically the branch's stated mechanism does not exist** (no
  projector). The question must be re-posed as: does an ambient specular highlight at
  near-normal incidence saturate both imagers and kill correspondence? Nothing read
  answers this.

**Verdict: unresolved.** The direct source exists and was identified during reading but
not read: **Staderini et al., "Photometric visibility matrix for the automatic selection
of optimal viewpoints," 3DV 2024** — a visibility matrix built on Blinn-Phong /
Cook-Torrance BRDFs that *excludes saturated and underexposed regions*. That is the
single highest-value unread document for this question.

### 2.4 Q4 — does defect detectability peak at normal incidence: **the literature does not answer this, and the reason is structural**

- Every reconstruction-lineage source defines "inspection" as *dimensional conformance
  to a model specification*, never flaw visibility. Scott, Roth & Rivest — a survey with
  "Inspection" in its title — contains **zero occurrences of "defect", "scratch",
  "flaw" or "raking."**
- The **only** defect-detectability paper read (AIAT'25) has **no incidence-angle term
  at all**. Orthogonality enters only implicitly, via `Size` (foreshortening shrinks the
  defect's pixel footprint) and `DisB` (grazing views push defects toward the silhouette
  boundary). And its optical design — a **coaxial ring light rigidly aligned with the
  camera axis** — structurally excludes the classic industrial practice of raking light,
  so the paper cannot even pose the question.
- The only quantified angle→detectability link anywhere in the set is **foreshortening**:
  the effective sampling density / GSD degrades as 1/cos θ, empirically visible in
  Soudarissanane's plate experiment where point count on a fixed-size target fell from
  **38 500 at ~0 ° to 4 900 at 80 °**. That is a *resolving-power* argument, not an
  *appearance* argument.

**Verdict for the write-up:** Albert's requirement is defensible on measurement-geometry
grounds — sampling density, depth precision, framing, evidence-image legibility — and it
is what the entire viewpoint-generation literature does by construction. It is **not**
supported on defect-contrast grounds by anything read, and the industrial-inspection
practice of raking illumination points the other way for shallow surface flaws. This
should be stated plainly rather than papered over; it also marks joint
viewpoint+illumination planning (flagged as a known gap in round 2, and named as its own
future work by AIAT'25) as the place where the tension would actually be resolved.

---

## 3. The score function itself (Q5–Q9)

### 3.1 Q5 — consolidated term inventory

Tagged by the space it needs. **P** = raw points + normals + intrinsics (our PLY today);
**V** = probabilistic voxel grid; **M** = mesh / parametric surface; **I** = a rendered
or captured image; **S** = pure sensor model, no geometry.

#### Geometry-of-measurement terms

| Term | Formula | Source | Space | Plain intuition |
|---|---|---|---|---|
| Incidence angle | `α = arccos(P·N / \|P\|\|N\|)` | Soudarissanane Eq 4.1 | P | The angle between the camera's line of sight and the surface's "straight out" direction. 0° = looking dead-on at the surface, 90° = skimming along it. |
| Cosine quality | `q(v,x) = \|n̂(x)·v̂\|` | Connolly / Massios & Fisher, via Zeng Eq 1 & Scott p.79 | P | The same idea packaged as a 0-to-1 score: 1.0 when looking straight at the surface, 0 when skimming. Just the cosine of the angle above. |
| Incidence constraint | `φ_sf = arccos(n_f·e_sz/…) ≤ φ_max` | Magaña 2023 Eq 5 (proxy) | P | A pass/fail check: is the viewing angle inside the allowed cone around the normal? No grading, just admissible or not. |
| Soft orthogonality | `w_o = exp(−(1+n^T d)² / (1−cos σ)²)`, σ = 25 ° | Wu Eq 4 | P | A bell curve on the angle: full marks looking straight on, falling off smoothly as you tilt. The width of the bell (25°) is a hand-picked tolerance, not a measured one. |
| Incidence reliability | `c_I(α) = cos α` ∈ [0,1] | Soudarissanane Eq 4.23 | P | Reads the cosine as "what fraction of measurement reliability survives at this tilt" — grounded in their physical noise model, not just a convenient shape. |
| Range reliability | `c_R(ρ) = (ρ_max²−ρ²)/(ρ_max²−ρ_min²)` ∈ [0,1] | Soudarissanane Eq 4.24 | P | Same reliability idea, but for distance: 1 up close, sliding to 0 as you approach the sensor's maximum range. |
| Joint geometry | `quality ∝ cos α / ρ²` | Soudarissanane Eq 4.16 | P | Angle and distance combined in one number: quality drops as you tilt AND as you back away — and backing away hurts quadratically. |
| Standoff Gaussian | `w_d = exp(−(‖v−s‖−d_o)²/(d_n/4)²)`, `d_o=(d_n+d_f)/2` | Wu Eq 4 | P | A bell curve on distance: the camera has a sweet-spot standoff, and the score penalises being closer or farther than it. |
| Sampling density | `d(v_i) = N_act/N_max`, `N_max = √2·π·R_n²/R_r²` ∈ [0,1] | Kriegel Eqs 2–3 | P | Of all the measurement points that *could* land on this surface area under ideal conditions, what fraction actually did? Low = the view samples the surface sparsely. |
| Density↔standoff↔FOV | `d = √( ω_xω_y / (12ρ·tan(θ_x/2)tan(θ_y/2)) − 2r²/3 )` | SEE Alg 2 (**fronto-parallel; no cos θ**) | P | Answers "how far should I stand so the point density comes out right, given my camera's field of view?" Caution: derived assuming you look straight on — no tilt correction. |
| Smoothness confidence | `f_s = Σ_j θ(‖s−q_j‖)·φ(n,q_j−s)` (bilateral over K-NN) | Wu Eq 2 | P | Do the neighbouring points agree this is a smooth surface? Neighbours lying nicely in the same plane → trust the data here; jumpy neighbours → don't. |
| Per-patch precision | `σ = √(ê^T ê /(n−m))`, RMSE of plane-fit residuals | Soudarissanane Eq 4.21 | P | Fit a plane to a small patch of points and measure how much they scatter around it. The scatter IS the measured noise of that patch. |
| Border-edge fraction | `b_i` = boundary edges / edges in voxel | Kriegel Eq 4 | M | How much of what this view sees lies at the ragged edge of the current model — edges mean unfinished territory worth scanning. |
| Completeness confidence | `f_g = Γ(s)·n` (Poisson field gradient) | Wu Eq 1 | M | How confident the reconstruction is that the surface here is already fully captured — low confidence marks the spots still worth a view. |
| Resolution / magnification | MVP `g1` (Eq 1) | Tarabanis MVP | P | Do enough pixels land on the feature to resolve the smallest detail we care about? A "can you even see it at this size" check. |
| Angle-subtended resolution | angle subtended at lens perspective centre — **explicitly orientation-independent, therefore carries no 1/cos θ foreshortening** | Cowan & Kovesi, via Tarabanis survey | P | Measures resolution by how big an angle the feature occupies in the camera's eye. Deliberately ignores surface tilt — so it can never punish slanted views, and cannot substitute for the angle term. |
| Depth of field | `D1 = a f d/(a(d−f) − c f)`, `D2 = a f d/(a(d−f) + c f)` | MVP Eqs 2–5 | S | The near and far distances between which the image is actually in focus. The whole target region must sit inside that slab. |
| Field of view | `g3 = (r_K−r_V)·ν − ‖r_K−r_V‖cos(α/2) ≥ 0` | MVP Eq 6 | P | Pass/fail: does the whole target region fit inside the camera frame from this pose, or does something stick out? |
| Invalid depth band | `band[px] = B·F/Z`; `DBR = B/(2Z·tan(HFOV/2))` | D405 §4.5 | S | The strip at one image edge where a stereo camera physically cannot compute depth (the two eyes don't overlap there). It grows as you move closer — ~10 % of the image at 10 cm. |
| Min-Z gate | resolution-dependent: 100/70/55/45/40 mm | D405 Tbl 4-11 | S | The closest distance the camera can measure at all, and it depends on resolution: at full resolution anything nearer than 10 cm returns no depth, period. |

#### Visibility / occlusion terms

| Term | Formula | Source | Space | Plain intuition |
|---|---|---|---|---|
| Point-proximity occlusion | walk sightline, check any point within radius υ | SEE Alg 8 | **P (mesh-free)** | Walk along the straight line from camera to target point; if any cloud point sits close to that line on the way, something is in front — the view of this point is blocked. Works on raw points, no mesh needed. |
| Normal-sign disambiguation | HPR-style spherical projection from the current view | SEE Alg 6 | **P (mesh-free)** | A normal has two possible directions (into or out of the surface). This decides which one points "out": the side you can look at it from without other points in the way. |
| Least-occluded direction | maximin antipole of the smallest containing spherical cap | SEE Alg 10 | **P (mesh-free)** | Of all directions you could look at a point from, find the one farthest from every blocker — the widest clear corridor to the surface. |
| Frontier classification | core/frontier/outlier by density, `k_min = ⌈(4/3)πρr³⌉` | SEE Alg 2/§3.2 | P | Labels each point by counting its neighbours: many = solid interior, few = the edge of what's been scanned, almost none = probably noise. |
| Visibility as signed distance | `g4 = ±d_V` to the CSG visibility-region boundary | MVP Eqs 12–14 | M (solid) | Instead of a yes/no "visible," measure how far the camera is from the edge of the visible zone — so a view that's barely unblocked scores worse than one comfortably in the clear. |
| Binary visibility matrix | `V_ij = 1` iff occlusion-free ∧ in-FOV ∧ in-DOF ∧ θ < θ_max (∧ projector ray too) | Staderini Eq 2.8, Gospodnetić §5 | M | A big yes/no table: can view i properly see surface patch j? "Properly" = unblocked, inside the frame, in focus, and within the angle limit — all at once. |
| Soft visibility likelihood | `P_v(x_n) = Π_{i<n} P̄_o(x_i)` | Delmerico Eq 5 | V | The probability the sight line actually gets through: multiply the "probably empty" odds of every map cell the ray crosses. One likely-occupied cell on the way tanks it. |
| Transmittance | `T(i) = Π_{j<i}(1 − p_o(j))` | Burusa Eq 6 | V | The same idea told as light through fog: how much of the ray survives all the partially-occupied cells between camera and target. Smooth, so you can differentiate through it. |

#### Information-gain terms *(answer a different question — exploration, not evidence quality)*

| Term | Formula | Source | Plain intuition |
|---|---|---|---|
| Occupancy entropy | `H(x) = −P_o ln P_o − P̄_o ln P̄_o` | Delmerico Eqs 2–3 | How uncertain the map is about one cell of space: highest when the cell is completely unknown, zero once we're sure it's empty or full. |
| Occlusion-aware VI | `I_v(x) = P_v(x)·H(x)` | Delmerico Eq 6 | Uncertainty × the chance we can actually see the cell — so a view gets no credit for "interesting unknown stuff" that's hidden behind a wall. |
| Rear-side / proximity-count VI | `I_p(x) = d_max − d(x)` for unobserved voxels | Delmerico Eq 11 | Prefers unknown cells that are close to the sensor — chase the surprises next to you before the ones far away. |
| Average entropy VI | `(1/n) Σ H(x)` | Kriegel Eq 13 | The average uncertainty over all the cells this view would touch — "how confused is the map, on average, where I'm about to look." |
| **Area factor** | `f(α,α_des)` piecewise-cubic, **bounded, unimodal, peaks at a target** (α_des,oc = 0.2, α_des,op = 0.8) | Vasquez-Gomez, via Delmerico Eqs 17–20 | Doesn't want "as much new stuff as possible" — it wants the *right mix* of new and already-seen (for reliable alignment), so the score peaks at a target ratio instead of always climbing. |
| Semantic IG | `I_s(ξ) = Σ_r Σ_i T(i)·I_s(i)` | Burusa Eqs 5–8 | Same uncertainty-hunting, but only counts uncertainty about the *things you care about* (labelled targets), ignoring uninteresting empty space. |

#### Mesh / graphics measures

Bonaventura's 22 (VQ1–VQ22) with exact equations: `#visible triangles`, `projected
area`, `Plemenos–Benayada`, **`visibility ratio = Σ vis_z A_z / A_t` — the one genuinely
bounded, resolution-insensitive measure in the set**, `viewpoint entropy` (→∞ as mesh
refines), `I₂/I₃/VMI/VKL`, the silhouette family (length/entropy/curvature/extrema — all
*poor performers* on the benchmark), the depth family (max depth, Gini–Simpson depth
diversity, Stoev & Straßer), **`instability` = mean Jensen–Shannon divergence to
graph-neighbour views**, `depth-based visual stability` (NCD threshold 0.87),
**`curvature entropy`** (angle-deficit `K_i = 2π − Σφ_j`, histogrammed — ranked 4th of
22), `mesh saliency` (`S(x)=|G(C,σ)−G(C,2σ)|`, DoG on mean curvature), `projected
saliency` (**ranked 1st of 22**), `saliency-based EVMI`.

Mosbach's feature functionals: `E_κ` (curvature integral), `E_TP` (thin-plate energy,
closed form via the control-point matrix), **`E_ND` and `E_IA` (max-pooled normal
deviation and max-pooled incidence angle — non-additive by design)**, `E_A` (area),
`E_ROI` (binary mask integral).

Staderini/Gospodnetić: `Cov(ι_j) = (1/M) Σ_i V_ij` ∈ [0,1]; viewability
`ν(s_i) = (1/N) Σ_j V_ij`; Scott's `m_v` = compliant surface area / total surface area.

#### Image-space terms

AIAT'25: `Vis` (hard gate: 0 unless *every* defect is detected), `Size` (labelled-pixel
count), `DisB` (mean defect-centroid distance to the object silhouette boundary), `DisC`
(mean offset from image centre), composed as
`Score = Vis·(α·Size + β·DisB + γ·(1−DisC))` — **weights never disclosed anywhere in the
paper.** Downstream detector metric `GIoU = IoU − |C\(A∪B)|/|C|`.

Bonaventura's rendered-image subfamily (silhouette VQ10–13, depth VQ14–16, VQ18) needs
only a 2-D raster and is therefore **plausibly point-splat-computable from our PLY
without meshing** — an untested but cheap transfer.

**Structural observation.** Notice what the term families do *not* overlap on. Delmerico's
"occlusion-aware visibility" is not a competitor to an orthogonality term — it answers
"can the ray reach the surface at all," while incidence answers "at what angle does it
strike once it arrives." Bonaventura's "depth" means z-buffer diversity within a frame,
not standoff to a physical sensor. Cowan & Kovesi's "resolution" is deliberately
orientation-independent and therefore carries none of the foreshortening penalty our
framing cares about. **Term-name collisions across these literatures are a real hazard
for the term table and must be disambiguated explicitly.**

### 3.2 Q6 — which terms are redundant: **largely unanswered, with four useful data points**

- **The most answerable version of this question is unanswered in the published
  literature.** Bonaventura benchmarks 22 measures against one ground truth and reports
  **no correlation matrix**; the authors defer "a convex linear combination of I₁, I₂,
  I₃" to future work. A cross-measure redundancy study of viewpoint-quality metrics does
  not exist.
- **The field's own combination behaviour is a hard constraint.** Zeng's Table 4 tags 25
  systems by utility function from a five-label vocabulary (information gain / quality /
  moving cost / ambiguity / entropy). **No row combines more than two terms.** A
  five-term composite score would be unprecedented in this literature.
- **Redundancy is object-dependent, and primitives are the failure case.** Mosbach's
  Tables 3–4: thin-plate energy and true curvature are *nearly identical* on the spring
  (8.46 % vs 8.71 % of value mass on the same region) but *diverge sharply* on the
  cylinder — TP places **18.83 %** of its mass on the flat top/bottom where curvature
  places **0 %**, because TP reacts to parameterisation distortion and is provably
  nonzero on a flat disk. **Cheap proxies for "the same" quantity break exactly on flat
  faces — i.e. exactly on our boxes.**
- **Two terms that look related can be measurably decoupled.** Delmerico's central
  negative finding: Average Entropy VI achieves the *best entropy-AUC* on most models
  while being **worst-correlated with surface coverage**, because reducing a voxel's
  entropy requires re-observing it, which competes with exploring new surface. A proxy
  can be internally well-behaved and still track the wrong outcome.
- **Counter-example — two terms that are demonstrably not redundant:** Soudarissanane
  shows that after regressing out the incidence contribution (`ê_d = ê_ρ cos α`) the
  residual noise is close to angle-independent and is instead explained by range. Angle
  and range each explain a distinct share of variance (~20–22 % attributable to incidence
  in a real room scan).
- **The authors of the founding formalism disown weighted-sum composition.** MVP §III-D,
  verbatim: *"The result of the optimization will be locally optimal only with respect to
  the chosen objective function. This result will generally be different than the optimal
  generated by a multiple objective approach."* And on the weights themselves:
  *"determination of the appropriate weights is a problem of its own."* The companion
  survey repeats it in Future Directions.

### 3.3 Q7 — normal-estimation-error sensitivity: **nobody measures it. This remains the branch's top risk.**

Across all twenty documents there is **no `d(score)/d(normal error)` anywhere.** What is
available:

- **The analytic handle, for the cosine term only** (derivable from Zeng Eq 1): for
  `score = cos θ`, `d(score) ≈ −sin θ · Δθ`. Sensitivity is **second-order near normal
  incidence and worst at grazing**. This is a genuinely favourable result: normal error
  matters *least* exactly where we want views to be, and most where the view is already
  marginal. It also argues that an orthogonality term is best used as a soft weight near
  the optimum plus a hard gate far from it — the same two-tier shape §2.2 found
  independently.
- **Everyone else punts, and says so.** Kriegel uses a *per-voxel averaged* normal
  because exact per-face incidence *"proved to be too time consuming,"* with no error
  bound. Nguyen uses a **constant θ_mean = 30 °** in the smoothing stage and a per-pixel
  normal only in the fusion stage — i.e. the authors themselves treated angle accuracy as
  optional where it was inconvenient. Wu's noise sweep (Fig 16) perturbs point *positions*
  only, never normal directions. Bonaventura's curvature measures assume clean manifold
  connectivity with no robustness analysis. Soudarissanane fits normals over thousands to
  tens of thousands of points on *known-planar* surfaces — far better conditioned than
  anything our fused PLY will produce on a curved primitive with dropout holes.
- **One concrete failure mechanism is documented.** SEE §3.3.2: the naive normal-sign
  heuristic *"fails when [the current view] observes the surface at an acute angle"*
  because *"the sightline and surface normal are nearly perpendicular and measurement
  noise can corrupt the identification of the outwards facing normal direction."* Normal
  error does not just perturb the score — near grazing it can flip the normal's sign.

**Branch obligation:** state the tolerance we need (Branch 3 owes the achieved number),
and note that the tolerance is angle-dependent — tight at grazing, loose at normal.

### 3.4 Q8 — bounded, normalised, cross-comparable per-view score

**Precedent exists, but every instance normalises against something object-specific.**

Bounded examples:

| Score | Construction | Comparable across…? |
|---|---|---|
| Kriegel `q_i`, `q_s` ∈ [0,1] | weighted sum of two individually-bounded factors, mean-pooled | within a session |
| Wu `f(s_k)`, `g(v_i)` ∈ [0,1] | **product of Gaussians, each ≤ 1** | within-object only; cross-object never tested |
| Staderini `Cov(ι_j)` ∈ [0,1] | fraction of *that object's own* face count | asserted comparable across objects by construction; never tested |
| Vasquez-Gomez area factor | piecewise-cubic, peaks at a target | across views |
| Bonaventura VQ4 visibility ratio | ratio of areas; **resolution-insensitive** | the strongest bounded example in the graphics set |
| Scott `m_v`, `e_v` | area-weighted compliance ratio; goal = 1.0 | **plan-level, not per-view** |
| AIAT'25 `Score` | terms normalised to [0,1] then weighted; **reused unchanged as a loss `L_VA = 1 − Score`** | one object only |

Unbounded counter-examples, several stated as open problems by their own authors:

- MVP's `h` — an unbounded weighted sum of *mixed-unit* signed margins (mm distances, a
  dimensionless magnification margin, an angular-cosine difference), with weights
  hand-tuned only to equalise order of magnitude at the *start* of optimisation.
- The Tarabanis survey names this as unresolved in Future Directions: *"The question of
  weighting constraints according to the importance of each in the general planning
  objective is a notable one… this may be somewhat task dependent."*
- SEE's view value is a **raw out-degree count** in a frontier visibility graph —
  unbounded, scaling with model complexity, and explicitly only used to rank within one
  ongoing observation of one object. A clean negative exemplar.
- Delmerico's IG is extensive — it grows with voxel count and object size.
- Bonaventura documents that **viewpoint entropy "goes to infinity for finer and finer
  resolutions of the mesh"**; `I₂` was invented specifically to patch this by subtracting
  a per-mesh constant, and even `I₂` is only "more stable," not interval-bounded.
- Mosbach's functionals need per-object thresholds (60 ° / 85 ° / 35 ° for the *same*
  functional on three objects), chosen by the engineer.

**Synthesis: boundedness is achievable and there are exactly two reliable constructions**
— (a) a product of individually-bounded factors (Wu), (b) a fraction of the region that
satisfies a threshold (Scott's `m_v`, Staderini's `Cov`). Construction (b) additionally
produces a directly narratable number, which matters for RQ4. Cross-object calibration
is *asserted* by every source that claims it and *tested* by none.

### 3.5 Q9 — the unit of scoring and the aggregation operator: **purpose-dependent, and the choice can flip the answer**

Every pooling operator in the field appears, with a defensible rationale:

| Operator | Used for | Source |
|---|---|---|
| **RMSE / quadratic mean** of per-point residuals → patch → average across patches → whole-view scalar | measurement precision | Soudarissanane Eq 4.21, §5.3.2.5 |
| **Area-weighted binary compliance ratio** — threshold *first*, then area-weight; do not average raw scores | plan/reconstruction quality | Scott `m_v`, footnote 3 |
| **Mean** over faces / voxels | coverage; expected improvement | Staderini Eq 6.11, Kriegel Eq 15, AIAT'25 DisB/DisC |
| **Max / worst-point** | **angle and tolerance criteria specifically** | Mosbach `E_ND`, `E_IA` |
| **Max over surface points** | choosing *where* to scan next (worst-confidence site) | Wu Eq 4 `g(v_i)` |
| **Hard AND over points** | visibility of an extended feature | MVP; AIAT'25 `Vis` |
| **Sum / count** | information gain (cumulative along a ray, not bottlenecked) | Delmerico Eq 1, SEE out-degree, Wu `ζ` |
| **Entropy of a normalised histogram; KL / JS divergence** | distributional view measures | Bonaventura VQ5/11/19, VQ7/17/21/22 |
| **Inverse-noise-weighted running average** | per-point → per-voxel fusion | Nguyen Listing 2 |

Three findings make this more than a menu:

1. **Wu uses three different operators for three sub-questions in one paper** — max to
   find the worst spot, sum to estimate aggregate coverage, max again to find a
   registration anchor. Pooling tracks *purpose*, not a universal rule.
2. **The pooling choice can flip which viewpoint wins.** Soudarissanane's room
   experiment: moving the scanner ~2 m from the room centre to a corner *worsens*
   incidence angles at the far edges of segments, yet **improves mean per-patch quality
   by ~25 % (0.0023 m → 0.0017 m)** because it trades a few high-incidence far patches
   for many more near-perpendicular near ones. *"Most orthogonal at the centroid"* and
   *"argmax of pooled quality over the region"* are different objectives that select
   different views.
3. **Fixed-area patching is a documented mistake on non-uniform density.** Same source,
   §5.3.2.4–5: *"the quality of the Cartesian patches is dominated by the number of
   points, which biases the comparison of the patches. The Cartesian patches are
   therefore not really suitable when assessing the quality of local patches… with big
   variations in point density."* They switched to fixed-angular (constant-point-count)
   patches specifically to decouple noise-quality from density. **This is the same
   confound our dropout holes and non-uniform density will produce**, in a different
   guise.

---

## 4. Computability on our input (Q10–Q13)

### 4.1 Q10 — the three-bucket partition

**Bucket A — computable today on the fused PLY (points + normals + intrinsics, no mesh):**
incidence/cosine in all variants · `c_I`, `c_R`, and the joint `cos α / ρ²` form · Wu's
`w_o` (soft orthogonality) and `w_d` (standoff) and `f_s` (smoothness confidence, needs
only raw points + normals — the Poisson step in Wu's pipeline is not logically required
for Eq 2) · Kriegel's sampling density `N_act/N_max` · SEE's entire mesh-free stack
(density classification, point-proximity occlusion, HPR-style normal-sign, maximin
least-occluded direction, density↔standoff relation) · Soudarissanane's per-patch RMSE
propagation · MVP's resolution / DOF / FOV formulas (they need only feature points and
lens parameters) · the D405 invalid-band and Min-Z terms (pure sensor model) ·
**plausibly** Bonaventura's rendered-image subfamily via point splatting — cheap to try,
untested by anyone.

**Bucket B — needs Branch-3 structure first (voxelisation, plane/primitive fits, or
reliable normals):** every VI/occupancy term (Delmerico, Kriegel's entropy, Burusa's
transmittance) · Soudarissanane's patch pipeline (presupposes a segmentation into
locally-planar patches with well-conditioned fits) · any curvature-based term.

**Bucket C — needs a mesh or CAD we do not have:** Wu's `f_g` (Poisson-field gradient)
and his mesh-ray-cast `w_v` · Bonaventura VQ1–9, 17, 19–22 (need per-face area or
per-vertex curvature) · Mosbach's entire B-spline pipeline (needs a watertight mesh, and
the paper itself flags that its fitting step *"strongly decreases when exhibiting
significant noise or incorrect topology"* — exactly our defect profile) ·
Gospodnetić/Staderini visibility matrices and `Cov` · Staderini's sampling-density matrix
· **AIAT'25's whole term family** (needs a watertight textured CAD mesh *and* known
defect ground truth).

**The headline ratio matters:** of the twenty documents read, only **SEE** operates
mesh-free end to end. Everything else in the applied-inspection lineage requires a mesh.
Our input is in Bucket A's regime, which means the branch is working with a comparatively
thin — but real and pseudocode-level — precedent base.

### 4.2 Q11 — visibility without a surface

**SEE is the only worked positive answer in the read set**, and it is worked at
pseudocode level:

- Occlusion test: walk sample points along the sightline; occluded if any measurement
  lies within a radius υ of any sample (Alg 8). Pure point-proximity, no ray/mesh
  intersection.
- Normal-sign resolution: HPR-inspired spherical projection — sample outward along ±e_n,
  project nearby measurements onto a unit sphere centred at the current view, pick the
  direction whose projected neighbourhood is empty first (Alg 6).
- Least-occluded direction: project occluders onto a unit sphere and solve for maximum
  angular separation from all of them — the antipole of the smallest containing spherical
  cap (Alg 10).

**Two cautions.** (i) The search radii υ and ψ are hand-tuned per scene with **no
reported ablation** — ψ was halved for large models purely to bound compute. (ii) SEE's
own density classification has a blind spot for our failure mode: a true dropout hole —
a region with zero neighbouring measurements, isolated from any core point — **would
never satisfy the frontier condition** and could sit unclassified rather than being
flagged. The paper does not test this.

**Katz's HPR primary paper and Open3D's `hidden_point_removal` parameter semantics were
not read this round**, so the branch's own flagged hazard (radius-parameter sensitivity on
non-uniform density, and whether holes read as "visible through") remains unvalidated.

Everyone else requires structure: MVP does point-in-polyhedron against a CSG solid;
Wu/Mosbach/Gospodnetić/Staderini ray-trace a watertight mesh; Delmerico/Kriegel/Burusa
ray-cast an occupancy grid. Scott's 2003 survey confirms this is structural — every
surveyed visibility mechanism needs *some* intermediate representation, and HPR postdates
the survey entirely.

### 4.3 Q12 — propagating input-geometry uncertainty into the score: **unanimous negative, and the field names its own gap**

Twenty documents, three survey traditions, thirty years. **Not one propagates uncertainty
in the input geometry into a view-quality score.**

The gap is named at each of the field's checkpoints:

- **1995**, Tarabanis survey §III-C-1: *"Underlying most of the work done in this area of
  sensor planning has been the assumption that the environment is known. This is often not
  the case in actual sensing situations. As a first step towards relaxing this assumption,
  sensor planning systems could investigate how to incorporate feature uncertainty."*
- **2003**, Scott §8.1.5: *"None of the methods incorporated both sensor and positioning
  system error models."* And §8.3.3: *"the view planning literature generally assumes
  idealized, error-free sensors… No author addressed the nonisotropic and nonstationary
  nature of residual geometric noise."*
- **2020**, Zeng §1 lists *"imprecision of motion, and unreliability of visual
  perception"* among the key hindrances, then never operationalises it in any of the six
  equations or 25 tabulated systems.

The nearest adjacent work — none of which closes it:

- Nguyen's σ-weighted ICP (Eq 5) and σ-normalised TSDF fusion propagate depth uncertainty
  into *pose estimation and fusion*, not into a view score.
- Tarbox & Gottschlich's morphological erosion of the admissible viewpoint set (via Scott
  §9.3) operates on the *viewpoint* set, not on object-geometry uncertainty.
- Vasquez-Gomez's expected-utility-under-pose-uncertainty (mapped, **unread**) is the
  closest published machinery and remains the highest-value follow-up for this question.

Every single reading note independently reached the same verdict. **This is the branch's
strongest defensible contribution claim.**

### 4.4 Q13 — the ~4 cm fused-vs-single-view offset: calibration or modelling requirement?

Three pieces of evidence, pointing to a hybrid answer.

**Calibration is a more plausible partial fix than expected**, because of a D405 fact:
RGB and depth are natively pixel-registered on the D405 — the left depth imager *is* the
colour sensor (Tbl 2-2 fn 4; Tbl 3-15 and 3-20 are the same OV9782 part twice). So the
offset **cannot** come from RGB↔depth sensor extrinsics; it must originate in the
survey/registration pipeline, which is a more tractable target.

**But the literature frames it as a modelling requirement, not a nuisance.** Scott §9.3,
Fig 9, describing exactly our failure mode: *"When the NBV list is sent to a positioning
system whose position and orientation accuracy is inferior to that of the sensor, the
coverage of individual viewpoints and of the NBV list as a whole is compromised… We can
recover a refined pose estimate post facto by employing suitable registration techniques…
However, we are still left with data acquisition differing from that which had been
planned."* Framed as open, with only Tarbox's erosion cited as a mitigation.

**And "just run ICP" is documented as not free.** Kriegel §6/Fig 11: without ICP,
successive scans misalign by *approximately four millimetres*; with *too much* ICP
correction, the model over-contracts — the correction introduces its own bias.

**Two design responses are precedented:**

1. **Compute the incidence angle in the live camera frame, not the fused frame.** Nguyen's
   pipeline always derives θ from the same frame's own locally-estimated normal, avoiding
   any cross-source geometry mismatch by construction. Soudarissanane's method structure —
   re-fit locally, don't trust the input cloud as ground truth beyond the neighbourhood you
   fit — is compatible with per-candidate local normal re-fitting.
2. **Build margin into the admissible set** (Tarbox-style erosion) rather than trusting a
   nominal pose.

**Verdict:** treat the offset as *both* — chase it as a registration/calibration defect
(the sensor cannot be its source), and simultaneously design the judge so it does not
depend on the fused frame being right, by computing the angle locally and by carrying a
margin. The literature offers no method that does this today, which is the same
observation as Q12.

---

## 5. Local optimality (Q14–Q17)

### 5.1 Q14 — formal definitions of "best view within a neighbourhood": **eight families, one explicitly ruled out**

| # | Definition | Source | Computable for us? |
|---|---|---|---|
| 1 | **Constraint-satisfaction with margin** — maximise `Σ α_i g_i` where each `g_i` is a *signed distance to an infeasibility boundary*; local SQP from a seed | MVP Eqs 4–5, 12–15 | yes, if we can define boundaries; unbounded and hand-weighted |
| 2 | **argmax over the visibility-constrained region of −\|angular offset from the inverse surface normal\|** | Roberts & Marshall (proxy read) | **yes — this is exactly the analytical box-face rule** |
| 3 | **Minimal enclosing cone / spherical Chebyshev centre** — minimise the maximum off-axis angle to any object point | Anderson, via Tarabanis survey §II-D-3 | yes; **weight-free and non-circular**; good primitive-level candidate |
| 4 | **Viewpoint stability** — mean Jensen–Shannon divergence of `p(Z\|v)` to graph-neighbour views; or NCD-threshold neighbour counting on depth images | Bonaventura VQ17 (Eq 21), VQ18 (Eqs 22–23) | needs mesh (VQ17) or renders (VQ18) |
| 5 | **Gradient of a utility field** — fixed-budget ascent on a differentiable score | Burusa Eq 9 | yes if voxelised; **does not verify a stationary point** |
| 6 | **Target-band / unimodal peak** — "best" is not "most", it is closest to a desired value | Vasquez-Gomez area factor, via Delmerico Eqs 17–20 | yes; structurally relevant if off-normal is ever preferred |
| 7 | **Restrict-then-ratio-maximise on a local visibility graph** — restrict to views dominating the nearest proposal, then `argmax deg⁺(m)/‖x−x_c‖` | SEE Alg 12 | yes, mesh-free |
| 8 | **Bi-objective argmax over a fixed local candidate neighbourhood** — `argmax η(s')·ζ(s')` balancing registration anchoring against coverage | Wu Eq 5 | yes |

**Explicitly ruled out:** aspect graphs. Scott §5.2: an aspect graph partitions viewpoint
space by *qualitatively equivalent visible topology*, but *"viewpoints with the same
visibility (i.e., the same aspect) are by no means equivalent for sensing purposes."*
Topological equivalence is not quality equivalence.

**Recommendation for the branch:** families 2 and 1 are the cheapest and least circular
for a primitive-level rule. Family 2 gives the answer directly for a box face. Family 1
naturally handles the case the contract actually creates — a view that is orthogonal but
*barely* reachable or *barely* unoccluded — because margin, not just alignment, is what
it maximises.

### 5.2 Q15 — is the landscape smooth enough to climb: **no, and here is the catalogue of fixes and the catalogue of breakages**

**Documented discontinuity mechanisms:**

- Visibility is a boolean set-membership predicate almost everywhere in the field
  (Tarabanis, Wu's `w_v` ray-cast indicator, Gospodnetić, Staderini, Kriegel).
- **A subtler, more general mechanism, admitted by MVP's authors** (§III-D): *"in the case
  of the focus constraint, as the viewing direction varies, different feature points become
  the farthest and nearest points along this viewing direction, thus leading to a
  discontinuity in the associated constraint."* **Any constraint defined as an argmin/argmax
  over a discrete point set has combinatorial switching discontinuities.** By direct
  analogy: **worst-point pooling for a defect-detectability or angle-tolerance term
  inherits exactly this.** That is an important, under-appreciated cost of the max-pooling
  choice §3.5 found precedented.
- **One team hit this wall and chose not to smooth it.** Staderini selected Bayesian
  optimisation (derivative-free, 60 iterations per pose, CPU only) *because* the
  ray-cast-visibility objective is non-differentiable in pose.

**Smoothing tricks catalogue, with cost:**

| Trick | Mechanism | Cost |
|---|---|---|
| Signed distance to the infeasibility boundary replaces the boolean | MVP `g4 = ±d_V` | closed-form for polyhedral regions; **needs the boundary known analytically**, which a holed PLY does not give |
| Probabilistic-occupancy relaxation | `P_v(x_n) = Π P̄_o(x_i)` | Delmerico Table 2: **<1 % of total per-view time** — the score formula is negligible next to shared ray casting |
| Volume-rendering transmittance + trilinear voxel interpolation | Burusa Eqs 4–6, autodiff through the grid | **0.06 s per ray-trace call**, 128 samples/ray |
| Occupancy-grid entropy as a soft visibility surrogate | Kriegel Eq 13 | same ray-cast cost |
| Differentiable ray tracing (Mitsuba 3) | AIAT'25 §4.5 | unreported |

**Empirical evidence the landscape is climbable *in a narrow case*:** Burusa's gradient
ascent reached parity with 10-candidate local sampling on ROI coverage (93.8 % vs 95.1 %)
and F1 (87.3 % vs 88.5 %). But the test was **four single-box occlusion configurations
around one small point-like target** — it does not stress-test the multi-optimum landscape a
multi-face box or a self-occluding cylinder plausibly creates.

### 5.3 Q16 — sampling versus gradient: **one credible speed result against a field-wide consensus for sampling**

**The gradient case (Burusa), and why it is stronger than it looks:** the sampling
baseline it beats is *already* restricted to a 0.1 m radius around the current pose —
comparable to our own working scale — so the speedup is genuinely a *small-neighbourhood*
result, not an artefact of comparing against a global search. Numbers: **1 ray-trace call
per viewpoint vs 10** (20 vs 200 calls over 20 viewpoints; 5 vs 50 over 5), trajectory
**0.31 m vs 0.43 m** at 5 views (the "28 %" headline) and **0.31 m vs 1.67 m** at 20
views, at parity on coverage and F1.

**Four caveats, all first-class:**

1. **No IK or collision constraint is enforced anywhere in the optimisation.** `V^C`
   appears as a constraint in Eq 3 and is never projected onto in the update rule Eq 9.
   Our contract's binding constraint is exactly the one that was not tested.
2. **No convergence criterion.** Termination is a fixed budget (5 or 20 steps). "Found
   the local optimum" is not demonstrated, only "walked for N steps."
3. **The paper's own efficiency multipliers are internally inconsistent** — Table I's
   caption says "four times", the body text says "three times", and the raw table numbers
   give 5.4×. Cite the raw numbers, never the prose.
4. **Delmerico shows the cost is in the ray casting, not the score.** All seven VI
   formulations cost 3.7618–3.7889 s/view — a spread of <0.03 s (<1 %). The gradient win is
   therefore about *fewer forward evaluations*, not a cheaper objective, which is precisely
   what Burusa's 10:1 call-count ratio measures.

**The consensus on the other side is overwhelming.** Zeng's Table 4: **all 25 surveyed
systems use explicit candidate sampling; none uses gradient-based local refinement as its
primary mechanism.** Wu: voxel grid + local-maxima suppression + finite-neighbourhood
argmax, no gradients, no discussion. SEE: one-shot maximin solve at proposal time. Kriegel:
argmax over boundary-generated candidates. Staderini: Bayesian optimisation, explicitly
chosen over gradients. Bonaventura: brute force over a fixed 642-point sphere. AIAT'25 is
the only other gradient user, and it *initialises from a 91-candidate hemisphere scan*.

### 5.4 Q17 — offline or online: **unanimous, offline, at construction time**

Five independent systems, no dissent:

- **AIAT'25**: 91-candidate hemisphere scan → argmax → 50 gradient steps, once, per fixed
  defect configuration. No timing reported that would support an interactive loop.
- **SEE**: the maximin correction is applied once, when a proposal is first found
  occluded, and never revisited.
- **Wu**: direction refinement (Eq 5) runs once per selected NBV location, before the
  robot moves.
- **Staderini**: 60 Bayesian iterations per pose, CPU only, with the explicit rationale
  *"increasing the number of Bayesian iterations extends computation time, [but] this is
  inconsequential for industrial applications with offline inspection planning."*
- **Mosbach / Gospodnetić**: candidates generated once from geometry; selection is a
  separate downstream stage.
- **Scott's future directions** independently favour a staged coarse-then-fine pipeline
  ("scene exploration, rough modeling, fine modeling, problem resolution").

**This settles a fork the plan expected the literature not to decide.** Refinement belongs
at action-space construction time. The action space can therefore carry a *precomputed*
per-view score, which is directly enabling for RQ3 and RQ4.

---

## 6. Image-space judging (Q18–Q21): **the branch's largest coverage hole**

The entire Tier 7 of the round-2 map (ViewActive, GraspView, Q-Align, FisherRF, COVER,
blur NR-IQA, SML focus measures) went **unread in round 3.** What the read set contains:

- **AIAT'25 is the only image-space judge read**, and it is thin: four mask-derived terms,
  no numbers of any kind, no ablation, undisclosed weights, one unnamed object, and it
  requires a watertight textured CAD mesh plus **known defect ground truth**. Its one
  transferable claim for Q19 is real but narrow — defect appearance *under a specific
  illumination* cannot be read off geometry alone — and is scoped to Phong shading,
  foreshortening and framing, **not** exposure, motion blur, focus or specular blowout,
  which is exactly the list the branch hypothesised image space would win on.
- **Wu argues against image-space cues explicitly**, in a reconstruction-quality context:
  colour is captured but deliberately excluded from NBV scoring *"because color information
  may not be accessible for all scanners and high color variation may not correlate with
  rich geometric details."*
- **Bonaventura's "image-based" measures are renders of clean meshes, not photographs** —
  a terminology trap. Its silhouette family, the most purely image-derived, was the
  *worst-performing* family on the human benchmark.
- **Zeng's only image-space NBV** (Johns et al.) scores recognition-hypothesis
  disambiguation from RGB pairs — not surface quality, framing, or defect contrast.
- The round-2 search trail records an explicit negative: **no paper was found that uses a
  VLM as a judge of viewpoint quality for NBV.** Q21's "do nothing clever" control has no
  precedent to inherit.
- One enabling D405 fact: RGB and depth are natively co-registered (shared left imager),
  so fusing an image-space score with a PLY-space one carries **no extrinsic calibration
  burden from the sensor**.

**Verdict: Q18–Q21 are effectively unanswered by this branch's round 3.** The fork the
contract flags — whether image-space judging is *required*, and therefore whether the
deferred generative-unseen-view question should reopen — cannot be closed on this evidence.

---

## 7. What is contested — contradictions preserved

1. **D405 sensor class.** Datasheet (primary, table-level, four independent citations):
   passive visible stereo, IR-cut, no projector. Two reading notes in this same branch and
   the branch's own round-2 framing text: active stereo with an IR texture projector. The
   datasheet wins; the consequences (wrong analogue literature, non-existent specular
   mechanism, texture-dependent dropout as a competing root cause) propagate through
   Q1–Q3 and must be corrected wherever the branch text asserts otherwise.
2. **Where the knee is.** 45 ° (structured-light, secondhand) / 60 ° (generic, and
   Soudarissanane's own finding, and Johnson/Mehdi-Souzani secondhand) / 65 ° (Lichti) /
   70 ° (Nguyen's ICP cutoff, Kriegel's gate) / 75 ° (AIT group in practice) — plus Wu's
   25 ° Gaussian tolerance which is a tuned hyperparameter with no measurement behind it,
   and Mosbach's per-object 35 °/60 °/85 °. **None measured on a passive stereo camera at
   10–40 cm.**
3. **Smooth or threshold.** Practice says both (§2.2). But Scott's survey says the
   thresholding tradition *"masked the most important impact of grazing angle, which is on
   measurement error"* — i.e. the field's canonical critique is that gating hides the real,
   smooth physics. And a genuine hard cliff (total signal loss) also exists. All three are
   simultaneously true and should be stated as such.
4. **Does the angle even dominate?** Soudarissanane: incidence accounts for ~20 % of
   room-scan noise, and *"above 60 ° the incidence angle dominates."* Scheibe (via Kriegel
   §2.3): *"3D point quality depends a lot more on the distance than on the angle of
   incidence."* Suppa (same section): standoff *"plays a very important role for laser
   scanners."* Both camps are laser-based; neither is our sensor. Unreconciled.
5. **Is orthogonality optimal?** SEE and Mosbach place views on the normal by
   construction and never question it. Soudarissanane's room result shows a
   less-central, less-orthogonal viewpoint winning on pooled quality. Gospodnetić's
   experts deliberately choose off-surface views for coverage. TLS shows near-normal
   saturation halos. Monostatic TLS on mirrors requires near-normal to get any return at
   all. All five coexist.
6. **Bounded score: solved or open?** Kriegel, Wu, Staderini, Vasquez-Gomez and
   Bonaventura's VQ4 all exhibit bounded scores. The Tarabanis survey, MVP, SEE, Delmerico
   and Bonaventura's entropy family all say it is unsolved or exhibit the failure. Both
   are right: bounded-within-an-object is solved; bounded-and-calibrated-across-objects is
   asserted by everyone who claims it and tested by no one.
7. **Aggregation operator.** RMSE (Soudarissanane) vs area-weighted compliance (Scott) vs
   mean (Kriegel, Staderini) vs max/worst-point (Mosbach for angle terms) vs sum
   (Delmerico, SEE) vs entropy/divergence (Bonaventura). No convergence, and the choice
   demonstrably changes which view wins.
8. **Redundancy is not a property of the formula.** Mosbach's thin-plate-vs-curvature pair
   is redundant on a spring and divergent on a cylinder. Formula similarity does not imply
   redundancy, and the divergence appears **exactly on flat faces** — our regime.
9. **Sampling vs gradient.** One credible 10× small-neighbourhood result against a field
   in which 25/25 surveyed systems sample, and in which one team rejected gradients on
   principle because the objective is non-differentiable.
10. **Internal inconsistencies flagged inside the sources themselves.** Bonaventura's
    Table 2 marks VQ4's best direction "H" while the prose says *"The best viewpoint
    corresponds to the minimum value of the measure"* (verified against raw HTML — a genuine
    source error). Burusa's 4× / 3× / 5.4×. AIAT'25's round-2 blurb credits it with ROI
    sampling for unknown defect locations, which is Section 6 future work.
11. **Access as a finding.** Tarabanis CVGIP 1994 (the branch's designated
    highest-formula-density source) — **no OA copy exists anywhere**, confirmed via
    Unpaywall; the given URL pointed at a different paper. Staderini ICVS 2023 — confirmed
    zero OA locations, abstract-only. Soudarissanane ISPRS-J 2011 and Staderini IROS 2024 —
    read via the authors' own theses, so equation and figure numbering differs from the
    published versions. Three of the branch's formula sources are therefore not verifiable
    at equation level as published.

---

## 8. What is missing

**Unread documents that directly own an open question:**

| Question | Designated source, unread |
|---|---|
| Q3 — does orthogonality hurt on glossy surfaces | **Staderini et al., "Photometric visibility matrix…", 3DV 2024** — Blinn-Phong/Cook-Torrance visibility that excludes saturated and underexposed regions. Identified during reading; the single highest-value follow-up. |
| Q2 / Q4 — angle penalty shape, relaxing orthogonality | **Haneda et al., IEEE Access 2025** (gold OA, zero access cost) — explicitly relaxes the orthogonal-viewing restriction under a data-quality-preservation condition. |
| Q1 — the ~60 ° knee, at source | **Johnson et al. 1997 (3DIM)** and **Mehdi-Souzani et al. 2006 (JCISE)** — the actual primary measurements, reached only via Kriegel's paraphrase. |
| Q6 — attribute correlation clusters | **Secord et al., "Perceptual Models of Viewpoint Preference"** (TOG 2011) — the map's designated source for the three correlation clusters. |
| Q8 — bounded scalar performance metric | **Mavrinac et al.** (T-Mech 2015) and **Mavrinac & Chen** (IJCV 2013) — the designated bounded-score and continuous-coverage-strength sources. |
| Q11 — HPR parameter hazard | **Katz, Tal & Basri** (TOG 2007) primary + **Open3D** parameter semantics. The mesh-free visibility route's known hazard is unvalidated. |
| Q12 / Q13 — uncertainty machinery | **Vasquez-Gomez** ×2 (IJARS 2014, AuRo 2015) — expected utility under pose uncertainty; the closest published machinery to the gap we intend to claim. |
| Q5 / Q9 / Q22 — the best single formal treatment | **Scott, "Model-based View Planning"** (MVA 2009) — measurability matrix extended with shadow effect, measurement precision and incident angle; plus the *optimal scan zone generated by offsetting from the sampled surface patch*, which is precedent for abandoning the uniform sphere. Referenced by four of the documents read; read by none. |
| Q22 — human-labelled GT | **Dutagaci et al.** (3DOR 2010) primary + NIST dataset. |
| Q18–Q21 — the entire image-space tier | ViewActive, GraspView, Q-Align, FisherRF, COVER, blur NR-IQA, SML focus measures. |

**Empirical gaps no reading can close:**

- **No σ(range, θ) for a passive stereo camera at 10–40 cm exists in the read set**, and
  the sensor-class correction makes it unlikely one exists at all in the RGB-D
  characterisation literature (which is dominated by active devices). **This probably has
  to be measured on our own rig.**
- **No `d(score)/d(normal error)` anywhere.** Must be established by us.
- **No test of whether HPR-style mesh-free visibility behaves on dropout holes.**
- **No joint viewpoint + illumination planning read** — flagged as a known gap in round 2,
  and it is precisely where the orthogonality-vs-raking-light tension (§2.4) would resolve.
- **No cross-object calibration test of any bounded score.**

---

## 9. Implications

### 9.1 RQ1 — Judging a view

**The score decomposes into three families with sharply different evidential status, and
they should not be blended.**

- **(a) Geometry-of-measurement terms** — incidence cosine, standoff, sampling density,
  FOV framing, mesh-free visibility. Well-precedented, computable today on our PLY
  (Bucket A), and each has at least one closed form with an equation number. **This is
  where the judge should be built.**
- **(b) Information-gain terms** — entropy, occlusion-aware VI, area factor. They answer
  a *different question* (how much unknown volume will this reveal) than RQ1 asks (how good
  is this evidence view of a known region). **Borrow their smoothing tricks, not their
  objective.**
- **(c) Appearance/defect terms** — essentially unread, and the one instance we have is
  numbers-free and needs a CAD mesh plus defect ground truth. **Do not build on this yet.**

**Concrete design consequences:**

1. **Keep the score small.** Across 25 surveyed systems the field never combined more than
   two utility terms, and MVP's own authors disowned weighted-sum composition in 1995 as
   producing a different answer than a proper multi-objective treatment, with weight
   selection *"a problem of its own."* A five-term composite would be unprecedented and
   the branch would inherit an unsolved weighting problem.
2. **Adopt the two-tier shape the field converged on independently six times:** a
   continuous quality model (`cos α`, or Wu's Gaussian `w_o`, or `cos α / ρ²`) for
   ranking, plus a hard feasibility gate near the knee for admissibility. Our gate value
   cannot be inherited — the 45–75 ° band is sensor-class-specific and none of it is
   passive stereo at our range — so **the gate threshold is a parameter to measure, and
   should be exposed as such rather than hard-coded from a citation.**
3. **Make it bounded, by one of the two reliable constructions**: a product of
   individually-bounded factors (Wu), or a *fraction of the region satisfying a threshold*
   (Scott's `m_v`, Staderini's `Cov`). The second is preferable — see RQ4.
4. **Choose and justify the pooling operator per term, and state it.** Max/worst-point for
   angle-tolerance gates has real precedent (Mosbach's `E_ND`, `E_IA` — angle criteria
   specifically); mean or area-weighted-compliance for regional quality; and **never
   fixed-area patching on non-uniform density** (Soudarissanane's explicit negative
   finding, which is our dropout confound in disguise). Note the hidden cost: worst-point
   pooling inherits MVP's combinatorial switching discontinuity, so it is a gate, not
   something to climb.
5. **Decide explicitly what the objective is.** "Most orthogonal at the region centroid"
   and "argmax of pooled quality over the region" select **different views** —
   Soudarissanane's room experiment is the proof, with a ~25 % pooled-quality swing in
   favour of the less-orthogonal, less-central viewpoint. Albert's requirement is naturally
   read as the first; a quality-maximising judge computes the second. This is a
   specification question for Albert, not a research question.
6. **The normal-error story is more favourable than feared, but only for the cosine term:**
   `d(score) ≈ −sin θ · Δθ`, so sensitivity is second-order at normal incidence and worst at
   grazing. Combined with SEE's documented normal-*sign* flip at acute angles, this argues
   for the same two-tier design from a third independent direction.
7. **Two sensor-side terms are free and should be in the score today**: the resolution-
   dependent Min-Z gate (100 mm at full resolution, not 70 mm) and the stereo invalid-depth
   band (10 % of image width at 10 cm). Both are computable with no geometry at all, and
   both argue against pushing standoff to the minimum.

**On the second half of RQ1 ("is it the best within its neighbouring region"):** eight
formal families exist and one (aspect graphs) is explicitly disqualified. The two to
implement and compare are **argmax of alignment with the inverse surface normal over the
feasible set** (family 2 — directly the analytical box-face rule) and **MVP-style margin**
(family 1 — maximise distance to the nearest infeasibility boundary), because margin is
what actually handles the case the contract creates: a view that is orthogonal but barely
reachable or barely unoccluded.

### 9.2 RQ2 — Selecting best views

Branch 4 owns the selection loop, but this branch supplies four load-bearing inputs:

1. **The analytical route is legitimate and precedented.** Deriving the pose directly from
   the local surface is a 30-year-old, mainstream construction:
   `C = S(u_c,v_c) + n(u_c,v_c)·d` (Mosbach Eq 6), `x = f + d·e_n, φ = −e_n` (SEE Alg 4),
   and Scott's *optimal scan zone generated by offsetting from the sampled surface patch*.
   Under this construction **the view is orthogonal by construction rather than by scoring**
   — which means for primitives, RQ2 may not need RQ1's judge at all.
2. **The geometry-agnostic viewsphere is empirically the worst option, from an independent
   quantified source.** Gospodnetić: space (sphere) sampling *"shows no benefits and their
   use is not recommended"*, with 34–77 % of candidates discarded as inadmissible versus
   0.3–27 % for geometry-native generation. Wu's Fig 14 corroborates from a different angle
   (sphere-constrained search misses self-occluded geometry at equal budget). And Scott's
   canonical survey pre-condemns it: *"To the degree that object shape is not spherical,
   viewpoint position is nonoptimal… It is not possible to optimize sensor performance with
   such limitations."* **Three independent citations back the project's founding premise.**
3. **Standoff must be derived, not fixed by hand.** Mosbach's manually-chosen constant `d`
   is the weakness of the constructive route; SEE's density↔standoff↔FOV closed form is the
   better skeleton — but **it assumes fronto-parallel viewing and therefore contains no
   `cos θ` term**, so it must be re-derived with the foreshortening correction before use
   off-normal. Our own binding constraints are Min-Z-at-resolution and the invalid-band
   growth at close range.
4. **Two documented cautions against surface-anchoring as an absolute rule:**
   Gospodnetić §9.4 (anchoring structurally cannot express the higher-coverage off-surface
   views experts choose), and Soudarissanane's pooled-quality result. Neither is fatal for
   us — we do single-region evidence capture on primitives, not full-object coverage of
   complex topology — but both should be acknowledged rather than discovered later.

### 9.3 RQ3 — Composing the action space

1. **Refinement belongs at construction time — this is settled, five independent sources,
   no dissent.** The consequence is architectural: **the action space can carry a
   precomputed per-view score.** Views are refined and scored once when the space is built;
   nothing needs to be re-scored during inspection.
2. **Cross-object consistency is a scoring-normalisation problem, and the only precedent
   is normalise-by-the-object's-own-denominator** (Staderini's `Cov`, Scott's `m_v`). Every
   source that claims cross-object comparability got it this way, and none tested it. If
   the space must stay consistent across shapes, this normalisation choice *is* the
   consistency mechanism, and validating it is our job because nobody has.
3. **The hard gate belongs in the space, not in the score.** The field's practice is to
   filter inadmissible candidates *before* the expensive computation (Gospodnetić filters
   pre-ray-tracing; Staderini bakes θ_max into the visibility matrix; Kriegel zeroes the
   quality term above 70 °). Applied to us: IK/collision/Min-Z/angle-gate filtering happens
   at construction, and the space the VLM sees contains only admissible views — which also
   removes an entire class of failure from RQ4.
4. **Primitive-level determinism is compatible with everything above.** The constructive
   pose rule (§9.2) plus a construction-time gate plus a precomputed bounded score is a
   deterministic per-primitive recipe, which the contract explicitly permits.

### 9.4 RQ4 — Presenting to the VLM

1. **An unbounded score is unusable as an interface, and unbounded is the field's default.**
   MVP's `h` (mixed units, hand-tuned weights), SEE's raw out-degree (explicitly not
   cross-object comparable), Delmerico's extensive IG, Bonaventura's divergent viewpoint
   entropy. Any number shown to the VLM must be bounded and object-normalised, and the
   branch must build that rather than inherit it.
2. **Prefer the threshold-then-fraction construction** (Scott's `m_v` pattern: threshold
   per point first, then area-weight) over a product-of-Gaussians. It is bounded, it is
   object-normalised, it is precedented twice — and, uniquely among the candidates, **it
   produces a directly narratable sentence**: *"this view sees 87 % of the face within
   orthogonality tolerance."* A product of Gaussians produces 0.62, which is
   uninterpretable to a blind orchestrator.
3. **Give the VLM the two-tier structure, not a scalar.** The field's own model/gate split
   maps naturally onto an interface: a pass/fail admissibility flag plus a bounded magnitude.
   That matches how the score is actually computed and gives the orchestrator something to
   reason about rather than a number to compare.
4. **Boundedness has to be validated, not asserted.** Every bounded precedent claims
   cross-object comparability by construction and none tests it. If the VLM will compare
   scores across objects, that test is a benchmark requirement (§9.5).
5. **The seeing-vs-blind fork cannot be resolved on this branch's evidence.** The entire
   image-space tier went unread, and the round-2 search found no VLM-as-viewpoint-judge
   paper at all. Both the "let the VLM see the candidate views" option and the "beat the
   off-the-shelf VLM judge" control remain unevidenced.

### 9.5 Benchmark methodology

1. **Imitate two protocols verbatim.**
   - **Kriegel's ablation switch** (the one that answers "does the quality term earn its
     place"): hold the modelling, registration and termination machinery fixed; flip one
     binary switch (`ω = 0` vs scheduled `ω`); measure four independent outcomes (scan
     count, time, completeness, density); on real hardware, 9 objects. Their result shape is
     also the expected shape of ours: **>20 percentage points of advantage concentrated in
     the first ~7 views, converging to a 0–2 pp gap by views 12–17.** *A quality term's value
     is front-loaded, not asymptotic.* And their honesty is worth copying: they separate the
     2.3× time reduction into the part caused by the quality term and the part caused by a
     faster pipeline, and say so.
   - **Delmerico's isolate-the-score design**: fix everything except the score formula;
     report **both** the task metric and the proxy metric the score nominally optimises, so
     decoupling is visible; always include a **random control**; report per-formulation
     compute cost as a first-class result; run enough random-seed trials (20) to average out
     start-state variance and **report variance, not just means** — their headline caveat
     about the random baseline is entirely about variance.
   - Secondary: Staderini's Table 6.1 (three refinement strategies, same visibility model,
     same objects, same candidate count) and Gospodnetić's design (same selector held fixed
     while the generation strategy varies) both isolate one pipeline stage cleanly.
     Bonaventura's is the template for many-scores-one-protocol, with settings recoverable
     (642-point icosphere, 6× bounding-sphere radius, 640×640 render, no back-face culling,
     one shared codebase).
2. **Carry Delmerico's central negative finding into the design as a rule.** Average
   Entropy VI wins on entropy-AUC and loses on the thing that matters, because *"the ability
   of a VI formulation to effectively reduce the entropy in the map is not well correlated
   with its effectiveness in surface reconstruction."* **Every proxy term must be validated
   against a downstream task metric, not just checked for internal good behaviour.**
3. **The circularity trap is real and current.** AIAT'25 defines the best view as
   `argmax(Score)`, optimises `Score`, and demonstrates success by showing `Score` improved —
   a 2025/26 textbook instance. Delmerico's coverage-vs-watertight-mesh is circular by
   construction (declared, as a deliberate isolation choice). SEE's real-world coverage is
   computed *against SEE's own final point cloud* because no ground truth exists — an IJRR
   paper accepting the circularity openly. **Mitigation: use analytical ground truth on
   primitives as primary** (on a box face the right answer is derivable), and use
   metric-based ground truth only for terms *not* used in constructing that ground truth.
4. **Ground-truth families and their risks:** *analytical* (face normal through the region
   centroid at optimal standoff — precedented by Mosbach Eq 6, SEE Alg 4, Roberts &
   Marshall; risk: it inherits Branch 3's primitive-fit error, so the GT is only as good as
   the fit); *metric-based* (circularity, as above); *human-labelled* (Dutagaci's protocol —
   68 models × 26 subjects, single-pick web interface, symmetry-corrected bounded 0–1 error —
   but it measures **representativeness for a human recognising a familiar object, not
   measurability for a sensor**, a distinction two separate reading notes flag independently).
5. **Baselines the benchmark must include:** the current uniform {h,v} sphere (our built-in
   baseline, and now backed by three citations saying sphere sampling is the wrong
   substrate), a **random-view control** (Delmerico: random is "reasonably well on average"
   with much larger variance — so a method that only beats random on the mean has not
   proven much), and the analytical constructive rule (normal-offset pose) as the
   deterministic reference.
6. **Object set: primitives — and this is not merely "easier".** Every published object
   set read is free-form or industrial (Stanford models, DTU MVS, statuettes, gears,
   crankshafts, Hirth couplings, busts, a church, one unnamed metal component). Only
   Staderini's parallelepiped, Mosbach's cylinder and MVP's primitive-assembled block are
   near-analogues, and Staderini's parallelepiped is the **only object in their table
   reaching 100.00 % coverage** — published magnitudes are not informative baselines for us.
   But primitives are a *different* regime, not just an easier one: Mosbach shows cheap
   geometric proxies diverge from exact ones **precisely on flat regions**, and Bonaventura's
   top-ranked measures include curvature entropy, which is near-degenerate on a box face.
   **Some published "winning" metrics will silently fail on our objects.**
7. **Noise injection has no validated model.** We have no D405-relevant σ(range, θ). Three
   honest options, and the benchmark must state which it used: (a) inject Nguyen's functional
   form with re-fitted coefficients and declare it a template not a measurement; (b) measure
   our own curve first (the branch's recommendation, since it also closes Q1 and Q7); or
   (c) inject angle-invariant noise and explicitly declare the angle term untested. Also
   inject the *other* defects the contract names — mm pose error, dropout holes, and the
   ~4 cm fused-vs-single-view offset — since **no published evaluation includes any of them**
   and the offset in particular is the failure mode Scott §9.3 frames as unresolved.
8. **Add a cross-object calibration test.** Every bounded-score precedent asserts
   cross-object comparability and none tests it. Since RQ4 depends on the VLM comparing
   numbers across objects, this is our benchmark's obligation, not an optional extra.
9. **Add a normal-error sensitivity sweep.** It is the branch's top risk, nobody in the
   literature measures it, and it is cheap to measure: perturb estimated normals by a known
   Δθ and record score error and rank inversions. Expect `−sin θ · Δθ` behaviour for the
   cosine term, and watch for SEE's normal-sign flip at grazing angles.
10. **Cite the field's own admission that it lacks benchmarks** (Scott §8.1.6: *"Given the
    lack of standardized performance benchmarks, it is difficult to make quantitative
    performance comparisons between proposed view planning methods"*) — it is the canonical
    justification for building one, and it is still true 23 years later, since nothing read
    in this branch reports a shared benchmark for *per-view* quality scoring.

---

## 10. Round-5 priorities for this branch

1. **Measure σ(range, θ) on the D405** at 10–40 cm on a matte target and a glossy target,
   with and without added texture. This single experiment closes Q1, sets the gate threshold
   for Q2, tests the texture-vs-angle dropout hypothesis for Q3, and supplies the benchmark's
   noise injector. Nothing in the literature can substitute for it.
2. **Read the four documents that own still-open questions**: Staderini's 3DV 2024
   photometric visibility matrix (Q3), Haneda et al. IEEE Access 2025 (Q2/Q4, gold OA),
   Scott MVA 2009 (Q5/Q9/Q22 — the map's own best-single-formal-treatment candidate,
   referenced by four documents read and read by none), Vasquez-Gomez ×2 (Q12/Q13).
3. **Read the image-space tier** — it is the branch's largest hole and it gates the
   contract's deferred generative question.
4. **Correct the sensor model everywhere** the branch asserts the D405 is active stereo.
5. **Establish `d(score)/d(normal error)` ourselves**, and hand Branch 3 the tolerance
   requirement — noting it is angle-dependent, tight at grazing and loose at normal.
