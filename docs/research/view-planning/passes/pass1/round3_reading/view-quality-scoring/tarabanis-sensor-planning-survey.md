# Tarabanis, Allen & Tsai — "A Survey of Sensor Planning in Computer Vision" (IEEE Trans. Robotics and Automation, 11(1):86–104, Feb 1995)

Source: full text obtained as the author-hosted PDF at Columbia Academic Commons
(`academiccommons.columbia.edu/doi/10.7916/D8348W4X`, file `00345940.pdf`, 2.26 MB, 19
pages, DOI `10.1109/70.345940`). The site fronts every request with an Anubis
proof-of-work bot-challenge (curl/WebFetch both got a "Making sure you're not a bot!"
page or a raw 403); a real browser session (chrome-devtools MCP) passed the challenge
automatically, and the resulting `techaro.lol-anubis-auth` cookie was reused with curl
to pull the actual PDF bytes. Two dead ends worth recording so they aren't retried:
(1) `academia.edu/23026194/...` — Cloudflare-blocked, no bypass found; (2)
`cs.columbia.edu/~allen/PAPERS/tro.final.pdf` — despite the "tro" filename this is a
**different** paper (Georgiev & Allen, "Localization Methods for a Mobile Robot in
Urban Environments," also IEEE T-RA) — a filename false positive, do not cite it as
the Tarabanis survey. Semantic Scholar's API rate-limited every attempt (429) and
never yielded `openAccessPdf`. Full paper read (all sections, both constraint taxonomy
tables, Figs. 1–9, Eqs. 1–5, full reference list), not abstract-only.

## What this document actually is

The founding survey of the *model-based, feature-detectability* tradition in vision
sensor planning — not a benchmark, not a new method. It (1) frames "sensor planning"
as computing camera + illumination parameters that satisfy a *task* given a *model* of
the environment (Fig. 1, Section I-A), (2) taxonomizes the field into three areas —
object feature detection, model-based recognition/localization, scene reconstruction —
and explicitly scopes itself to the first (Section I-C, Section II intro), and (3)
reviews ~15 systems (HEAVEN, VIO, ICE, the SRI Cowan–Kovesi/Cowan–Bergman line, MVP,
Anderson's viewer-orientation work, LIGHTING ADVISOR, VANTAGE, ROSI) under two
methodological families: **generate-and-test** (discretize a viewing sphere, evaluate
each cell) and **synthesis** (derive closed-form admissible regions analytically, then
optimize). This is the paper the branch was selected for specifically because it is
the *origin* of "visibility / field-of-view / focus / resolution" as separately
characterizable constraint domains (Section I title: "the branch's selection reason"),
and that is exactly what the paper delivers — a vocabulary and a taxonomy, not numbers
for any real depth sensor.

## The vocabulary and taxonomy (Q5's table, direct answer)

Section II-B enumerates the "sensor parameters" (Section II-B-1) and "feature
detectability constraints" (Section II-B-3) that structure the entire field this
branch inherits from. This is the closest thing in the whole 1995–2026 literature to a
canonical originating list — reproduced here as the requested table, PLY-space vs.
image-space tagged from our project's perspective (the paper itself predates that
distinction; all of its "image-space" constraints are actually *predicted from a CAD
model*, not measured from a captured image):

| Term | Paper's definition (§ cite) | Formula / inputs | PLY vs. image space (our framing) |
|---|---|---|---|
| Visibility | "all lines of sight from the camera to each point of the feature are not obstructed" (§II-B-3) | Ray-cast from each feature point to candidate viewpoint; occluded iff any object intersects the ray. Realized in MVP as a piecewise-analytic 3D occlusion region derived from the union of occluded regions of all polygon faces (§II-D-1, Fig. 7) | PLY-space (needs full scene geometry, not just the captured image) |
| Field of view | rays from the feature must land inside the sensor's active area, not be clipped (§II-B-3) | Closed-form via perspective projection + sensor extent; one of MVP's `g_3` constraints (Eq. 4) | PLY-space (planning-time), verifiable in image-space post hoc |
| Resolution / magnification | "the minimum angle subtended by a given incremental surface length at the perspective center of the lens" (Cowan & Kovesi's definition, §II-D-1) — explicitly *not* pixels-per-length, and explicitly *independent of viewing orientation* since subtended angle doesn't depend on which way the camera is rotated about that ray | Angle subtended at lens perspective center; MVP's `g_1` (Eq. 4) | PLY-space |
| Focus / depth of field | Near/far distance limits from Krotkov's blur-circle formula: blur circle diameter < minimum pixel dimension (§II-D-1, citing [48]) | Axial (along optical-axis) distance range only — no incidence-angle term anywhere in this formulation | PLY-space |
| Perspective distortion | "minimizes the maximum local distortion" under projection (§II-B-3); operationalized by Anderson as minimizing the maximum off-axis angle between viewing direction and any object point (§II-D-3) | Smallest enclosing circular cone on the unit sphere containing all object-point projections | PLY-space |
| Illuminability | feature must be visible to *some* point of the light source too, else in shadow (§II-B-3) | Same ray-cast machinery as visibility, but from illuminator instead of sensor | PLY-space |
| Dynamic range | image irradiance at the feature must fall inside the sensor's brightness range — not saturated, not below noise floor (§II-B-3) | Requires BRDF + illumination geometry (Torrance–Sparrow model used downstream in ICE/VANTAGE/ROSI, §II-E-1/F) | Requires photometric model, PLY+material |
| Contrast | edge-detectability requires sufficient intensity disparity across the edge (§II-B-3) | ICE's "contrast graph" fits piecewise-continuous polynomials along an edge from simulated intensities on both adjacent faces (§II-E-3, Fig. 4) | Simulated/predicted, not measured-image |

Two taxonomy axes worth preserving verbatim (§II-B-1, §II-B-3, third paragraph):
**geometric vs. optical** sensor parameters (position+orientation vs. lens settings
`d, f, a`), and **geometric vs. optical/radiometric** constraints (visibility/
illuminability depend only on geometric parameters; resolution/focus/FOV/dynamic-
range/contrast also depend on optics or illumination). Nothing in this paper's
vocabulary is viewpoint entropy, saliency, information gain, surface-normal entropy,
sampling density, or defect detectability — those all belong to *other* traditions
(graphics best-view, NBV, industrial inspection) this survey does not touch. Explicit
negative finding: **do not treat this as a source for the graphics/NBV terms in
Q5's minimum-coverage list** — it supplies exactly 8 of them (visibility, FOV,
resolution, focus/DoF, perspective distortion, illuminability, dynamic range,
contrast) and zero of the rest.

## Not addressed at all (first-class negative findings)

- **No incidence-angle-vs-depth-error curve, no active-stereo/RGB-D content of any
  kind.** This 1995 paper predates structured-light/active-stereo depth cameras
  entirely; its "range finders" are light-stripe triangulation systems mentioned only
  in passing (ROSI's illumination model, §II-F-2). Q1 (D405 σ(range,θ) curve) has zero
  answer here — do not cite this paper for any depth-noise number.
- **No specular-reflects-into-imager treatment for a coaxial projector-camera D405-like
  system**, but a *closely related* physics is modeled one level removed: Cowan &
  Bergman's SRI extension (§II-D-1, penultimate paragraphs, citing [23]) models surface
  reflectance as Lambertian diffuse **plus a specular lobe with angular tolerance
  around the perfect mirror angle**, and explicitly *computes and subtracts* the
  specular region from the admissible illuminator-placement domain — i.e., they
  formally exclude camera/light geometries that fall inside the specular return cone.
  This is for a **separate point-light-source + camera** rig, not a coaxial
  projector-in-camera IR system, so it doesn't transfer as a number, but it is a
  citable precedent that "orthogonal (specular) geometry can be actively excluded by a
  planner, not just avoided by luck" — answers the branch's "does any planner model
  this?" with a qualified yes, one level removed from our sensor type.
- **No industrial-inspection defect-detectability-vs-angle result.** Batchelor's
  LIGHTING ADVISOR (§II-E-1) is inspection-flavored (asks about "specular, diffuse,
  opaque, transparent, translucent" surfaces and "type of feature or defect to be
  highlighted") but is a rule-based expert system that outputs a *lighting sketch*, not
  a quantitative angle-vs-detectability relationship — no formula, no curve, no
  raking-light data. Q4 gets no numeric answer from this source.
- **No VLM, no image-space captured-quality judging, no generative unseen-view
  prediction.** Entirely a priori, model-based planning; there is no step anywhere in
  this paper that scores an already-captured image.
- **No feature/geometry uncertainty propagated into the view score** — flagged
  explicitly *by the authors themselves* as future work, not attempted (see below,
  Section III-C-1). This is the strongest single finding for the branch's contribution-
  gap question (Q12): even the founding survey names this gap and it appears to remain
  open on the branch's other reading so far.

## The "best view in a neighbourhood" formalism (Q14, direct hit)

MVP (§II-D-2, pp. 96–98) is the paper's most fully worked synthesis system and gives an
exact, quotable formal answer:

- **Generalized viewpoint**: `V(r_o, ν, d, f, a)` — an 8-dimensional point: 3D position
  `r_o(x,y,z)`, 2D orientation `ν` (unit viewing-direction vector via pan+tilt only —
  MVP explicitly does **not** plan rotation about the optical axis / "swing"), plus
  optical settings `d` (image distance), `f` (focal length), `a` (aperture) (p. 96,
  "In MVP... a point in this space is defined as a generalized viewpoint").
- **Constraint-satisfaction-with-margin**: each task requirement becomes an inequality
  `g_i(r_o, ν, d, f, a) ≥ 0` (Eq. 4), where `i=1` resolution, `i=2a,2b` near/far depth-
  of-field limits, `i=3` field of view, `i=4` visibility (signed distance to the
  visibility-region boundary, `g_4 = ±d_v`). A large positive `g_i` means "comfortably
  satisfied," near zero means "marginal," negative means "violated" (p. 97, right
  column). This *is* the branch's "constraint-satisfaction-with-margin" definition,
  cited verbatim.
- **Objective**: `f = max(α₁g₁ + α_{2a}g_{2a} + α_{2b}g_{2b} + α₃g₃ + α₄g₄)` (Eq. 5),
  subject to `g_i ≥ 0` and an auxiliary unit-vector equality constraint `g_5 = 0`.
  Weights `α_i` are chosen only "so that the contribution of each constraint... is of
  the same order of magnitude and prevents a subset of the constraints from dictating
  the optimization" (p. 97) — explicitly **ad hoc, not learned or derived**.
- **Result claimed**: "a generalized viewpoint that is globally admissible and locally
  optimal... For this computed viewpoint, all constraints are satisfied with the
  largest margin **in the neighborhood of the initial point**" (p. 98, left column) —
  i.e. MVP's own definition of "best view in a neighbourhood" is exactly: start from an
  initial guess, run constrained nonlinear optimization on Eq. 5, converge to a local
  optimum of the weighted-margin sum. Demonstrated on a real CAD test object (Fig. 6,
  small/large-hole feature edges) with before/after camera views shown (Fig. 8).
- **Self-flagged open problems, verbatim** (end of §II-D-2, p. 98): "further work is
  needed... to study the convergence of the method, **the dependence of the method on
  the quality of the initial guess**, and **the selection of appropriate weight
  settings**." This is a load-bearing caveat for any gradient/local-search track we
  build: the exact same failure modes (init sensitivity, weight tuning) were already
  unresolved at the birth of this formalism 30 years ago.

Anderson's viewer-orientation work (§II-D-3, p. 98) is a **second, narrower, non-
circular** family: given only that the object must be centered/fit in frame, find the
viewing direction minimizing the maximum off-axis angle to any object point — reduces
to the smallest enclosing circular cone on the unit sphere (a spherical-geometry
Chebyshev-center problem), solved with algorithms from [5],[6]. Explicitly does **not**
consider magnification, focus, or occlusion (paper's own words, p. 98: "Requirements
such as magnifying the object features, having them in focus, and avoiding occlusion
of the features are not considered"). This is a clean, non-circular, purely-geometric
"best whole-object framing" answer — useful precedent for the box-face/cylinder-orbit
primitive-level rules our framing doc wants, since it needs no task-specific weights at
all, only object-point geometry.

The generate-and-test family's analogous but weaker answer is HEAVEN's/ICE's/VIO's
composite criterion: `C = t·C₁ + (1−t)·C₂` (HEAVEN, Eq. 3, §II-C-1) — again a linear
blend of two attribute scores with an arbitrary mixing weight `t`, evaluated over
tessellated-sphere facets rather than continuous optimization. Same ad hoc-weighting
character as MVP's `α_i`, just discretized instead of optimized.

## Smoothness of the visibility landscape (Q15) — a mixed, mostly-negative finding

Visibility itself is treated as a **hard, Boolean set-membership** predicate throughout
— "a feature is visible" is inside/outside a polyhedral occlusion region, not a
continuous field. But MVP's optimization formulation applies one specific, reusable
smoothing trick: it replaces the Boolean visibility test with a **signed distance to
the visibility-region boundary**, `g_4 = ±d_v` (positive inside, negative outside,
Eq. 4/§II-D-2, p. 97) — turning a discontinuous indicator into a continuous margin that
a local optimizer can climb. No fuzzy filters, no occupancy fields, no differential ray
sampling, no alpha compositing anywhere in this paper — those are all later-era (2000s–
2020s NeRF/differentiable-rendering) techniques absent from 1995 sensor planning. The
one technique this paper *does* offer — Boolean feasibility → signed distance to the
nearest infeasibility boundary — is cheap (closed-form for polyhedral visibility
regions) but requires the polyhedral occlusion boundary to be known analytically in the
first place, which a fused, holed, noisy PLY does not give you directly.

## Aggregation unit and pooling (Q9, direct precedent for both hard-AND and soft pooling)

The paper's unit of scoring is the **feature** (a line segment / edge, occasionally a
point), not a point, patch, or whole view — and it shows both pooling conventions
coexisting in the same literature:

- **Hard AND / worst-point pooling**: MVP's visibility constraint requires *every*
  point along a feature edge to be unoccluded simultaneously (§II-B-3 definition quoted
  above) — algebraically equivalent to a min/AND over per-point binary visibility. This
  is direct precedent that worst-point pooling for occlusion/visibility is an
  established, not ad hoc, convention.
- **Soft/area-weighted pooling**: HEAVEN's/ICE's edge-visibility criterion is instead a
  **ratio**: "unoccluded portion of these edges[,] to their actual edge lengths"
  (§II-C-4, "Robustness against potential object placement uncertainty" list, item 2)
  — a continuous fraction, not a Boolean AND.
- **View-level aggregation**: the per-view score is then built by combining several
  per-feature margins via the weighted-sum-then-max objective (Eq. 5) — i.e., "per-
  region quality aggregated into per-viewpoint utility" is already the paper's own
  structure, just with linear weights rather than mean/min/area-weighting rules.

## Bounded, normalized, cross-comparable score (Q8) — explicit negative finding

There is **no bounded or normalized score** anywhere in this survey. MVP's objective
`f` (Eq. 5) is an unbounded weighted sum of heterogeneous signed margins (angle
margins, distance margins) in different physical units, combined via hand-tuned
weights explicitly chosen only to equalize *order of magnitude*, not to produce a
comparable [0,1] utility. The paper's own "Future Directions" section (III-D,
"Constraint Integration") names this as unresolved: "The question of weighting
constraints according to the importance of each in the general planning objective is a
notable one. It may be more important to have a feature visible at a poor resolution
than not being visible at all... this may be somewhat task dependent." No later part of
this paper offers a fix. Treat this as a citable "no precedent" answer to Q8, and as
independent corroboration (from the field's own origin point) that a bounded, VLM-
facing per-view score is a genuine gap, not a solved problem being reinvented.

## Redundancy / correlation between terms (Q6) — silent, not addressed

The survey performs no correlation analysis between its constraint terms and does not
discuss whether e.g. resolution and focus (both `d,f,a`-dependent, both optical) are
redundant. It does note the terms are **not always separable**: "the sensor and
illumination planning problems are not separable" in general (§II-B-3, penultimate
paragraph) because of BRDF coupling (specular term) and lens vignetting's cos⁴ off-axis
falloff [35], with two named exceptions where they decouple (diffuse-only surfaces;
narrow FOV). This is evidence of *coupling*, not measured *correlation* — useful as a
qualitative caution but not a number to cite for Q6.

## Feature/geometry uncertainty (Q12/Q13) — explicit, load-bearing gap

Section III-C-1 ("Feature Uncertainty," p. 102) states plainly: "Underlying most of the
work done in this area of sensor planning has been the assumption that the environment
is known. This is often not the case in actual sensing situations. As a first step
towards relaxing this assumption, sensor planning systems could investigate how to
incorporate feature uncertainty... combine the deterministic sensor planning techniques
with approaches that use a decision theoretic framework [15], [31]." No system reviewed
in the paper (HEAVEN, VIO, ICE, MVP, Cowan-Kovesi/Bergman, VANTAGE, ROSI) propagates
input-geometry uncertainty into the view score — all assume an exact CAD/polyhedral
model. This directly answers the branch's "gap = contribution claim" question: **yes,
this is a named, still-open gap at the field's origin**, which strengthens (does not
weaken) the case that propagating our ~mm pose error / ~4cm fused-vs-single-view
offset into a view-quality score would be a genuine contribution rather than a
reinvention.

## Object sets tested (Branch 2 handoff, Q11/Q23)

Every quantitative demonstration in this paper is on **polyhedral, CAD-defined
industrial parts** — explicitly *not* Stanford models or ShapeNet-style organic shapes
(that tradition postdates this paper by years): MVP is tested on "a piece of sheet
metal with a number of different size slots and holes" and "a tray of three-dimensional
parts" (§II-D-1, p. 96), and shown with a specific CAD object with two holes (Fig. 6,
"feature edges to be observed"). Anderson's viewer-orientation work assumes general
convex/concave polyhedral objects. This is a positive precedent for the branch's
primitive-shape regime: **the founding paper of this whole tradition already evaluated
on primitive/near-primitive polyhedral parts**, which is arguably a closer match to our
box/cylinder tabletop objects than most modern NBV benchmarks (which use organic
scanned models). Worth flagging to Branch 2 as a historical anchor when arguing that
published organic-mesh baselines may be uninformative for our regime.

## Ablation / validation protocols actually reported (Branch 2 handoff, Q24)

No formal ablation (with-term vs. without-term) is reported anywhere in this survey.
The two closest things to quantitative validation, worth extracting verbatim for
Branch 2:

1. **Cowan & Bergman's illumination-region validation** (§II-D-1, penultimate
   paragraph): "the illuminator was moved to positions along the constraint curves of
   minimum and maximum brightness and **the resulting images were consistent with the
   expected values (within roughly 5%)**." — a predicted-vs-measured brightness check
   against real captured images, the paper's single hardest number.
2. **MVP's qualitative before/after demonstration** (Fig. 6/7/8, §II-D-2): two
   different initial guesses (one starting inside the visibility region, one outside,
   one violating all constraints) both converge under the optimizer to viewpoints
   verified — by taking an actual camera view from the robot — to satisfy all
   constraints. This is a feasibility demonstration, not an ablation of which
   constraint terms matter.

## Load-bearing assumptions (apply across nearly every system reviewed)

- Exact, known, polyhedral (or otherwise clean parametric) object geometry — no PLY,
  no point cloud, no scan noise, no holes, no registration error anywhere in this
  paper. Every "visibility region" is a closed-form polyhedral construction from CAD
  face data.
- Single static scene / single target object per planning problem (dynamic sensor
  planning for moving objects/robots is flagged as a separate, less-developed area,
  §III intro and the VIO moving-target extension, §II-C-2 last paragraph).
- Illumination and sensor planning are coupled in general but most reviewed systems
  either ignore illumination entirely (MVP has no illumination planning yet, §II-D-2,
  "At this stage the MVP system does not include illumination planning") or plan it
  with strong simplifying assumptions (Lambertian + one specular lobe, point or
  Lambertian-extended sources, no interreflections — §II-D-1, §II-F-2).
- "Resolution" throughout is angle-subtended-at-the-lens, not pixels-per-unit-length —
  a definition chosen specifically to be *orientation-independent* (§II-D-1, p. 95,
  right column: "the advantage of formulating the resolution constraint independently
  of the viewing orientation since the angle subtended at the viewpoint does not
  depend on the viewing orientation"). This means the survey's canonical "resolution"
  term does **not** encode the `1/cos θ` foreshortening penalty our branch is
  interested in — a real mismatch between this paper's resolution definition and the
  GSD-vs-incidence-angle framing our RQs use; don't conflate the two without noting
  this.

## What transfers to our setting

- The eight-term geometric/optical constraint vocabulary (visibility, FOV, resolution,
  focus/DoF, perspective distortion, illuminability, dynamic range, contrast) as a
  checked, citable base layer for Q5's table — this is literally what the branch
  curator selected the paper for.
- MVP's constraint-satisfaction-with-margin formalism (Eq. 4/5) as one clean, quotable
  candidate answer to "what does best-view-in-a-neighbourhood mean," directly usable
  once we substitute our own constraint functions (orthogonality margin, standoff/GSD
  margin, framing margin) for `g_1...g_4`.
- The signed-distance-to-infeasibility trick (turning Boolean visibility into a
  continuous margin) as a cheap, reusable smoothing technique — provided we can compute
  or approximate an analogous boundary for our hidden-point-removal-based visibility
  route.
- Anderson's minimal-enclosing-cone framing metric as a weight-free, non-circular
  "whole-object best direction" primitive, potentially useful for the "orbit a
  cylinder" / "center a box face" rules our framing doc wants at the primitive level.
- The worst-point (AND) vs. ratio (soft) pooling duality, as direct historical
  precedent that both conventions are legitimate design choices in this literature, not
  something we'd be inventing from scratch.
- The paper's own list of open problems (initial-guess dependence, ad hoc weighting,
  feature/geometry uncertainty, illumination/sensor inseparability) as a pre-vetted
  risk list for anything we build in the optimization track.

## What does NOT transfer, and why

- No depth-sensor noise model, no incidence-angle-vs-error curve, no active-stereo
  content — this paper cannot answer Q1, Q2, or Q3 with any number; it is 15+ years
  earlier than commodity RGB-D and reasons entirely about CCD cameras + CAD models.
- No industrial-inspection defect-detectability-vs-angle result (Q4) — LIGHTING ADVISOR
  is qualitative/rule-based, not a measured curve.
- No point-cloud-native visibility method (Q10/Q11) — every visibility computation
  assumes exact polyhedral face data; nothing here validates or invalidates hidden-
  point-removal on a real, holed, non-uniform-density cloud. That question is untouched.
- No bounded/normalized score (Q8) and no correlation/redundancy analysis (Q6) — both
  explicit gaps, not just omissions of no interest to the paper.
- No image-space captured-image judging, no VLM, no generative view prediction — the
  entire "judge from the photo you already took" track (Q16, Q17, Q18, Q19, Q20) is out
  of scope for this 1995 model-based-planning paper by construction; it plans *before*
  any image is captured.
- No human-labelled ground truth for "best view" anywhere (Q21) — the closest is a
  5%-accuracy predicted-vs-actual brightness check (Cowan & Bergman) and MVP's
  qualitative feasibility demo, neither of which is a labeled best-view dataset.

## Checkable citations

- Sensor planning definition, Fig. 1 diagram: Section I-A, p. 86.
- Three-area taxonomy (feature detection / recognition-localization / reconstruction),
  survey's scope decision: Section I-C, p. 88 (citing Maver & Bajcsy's classification).
- Sensor parameters (geometric `x,y,z` + pan/tilt/swing; optical `d,f,a`) and
  illumination parameters (geometric + radiometric): Section II-B-1, pp. 89–90.
- Feature detectability constraints, full list with definitions (visibility, FOV,
  focus, resolution, perspective distortion, illuminability, dynamic range, contrast),
  geometric-vs-optical/radiometric split: Section II-B-3, pp. 90–91.
- HEAVEN system, geodesic-dome discretization, occlusion-free region via convex hull,
  distance-to-occluded-facet ranking, composite criterion `C = tC₁ + (1−t)C₂`: Section
  II-C-1, Eqs. (1)–(3), pp. 91–93, Figs. 2–3.
- VIO system, edge-visibility ratio, camera-illuminator pair evaluation: Section
  II-C-2, pp. 93–94.
- ICE system, edge-contrast graph, Torrance–Sparrow-based reflectance with specular
  polarization term: Section II-C-3, p. 94, Fig. 4.
- Cowan & Kovesi resolution definition (angle subtended, orientation-independent),
  Krotkov depth-of-field/blur-circle formula: Section II-D-1, pp. 94–96, Fig. 5.
- Cowan & Bergman illumination extension, Lambertian+specular-lobe reflectance model,
  specular-region subtraction, 5%-consistency validation: Section II-D-1, p. 96
  (penultimate/final paragraphs).
- MVP generalized viewpoint `V(r_o,ν,d,f,a)`, constraint inequalities `g_i≥0` (Eq. 4),
  objective `f=max(Σα_ig_i)` (Eq. 5), "globally admissible and locally optimal," open
  problems (initial guess, weights): Section II-D-2, pp. 96–98, Eqs. (4)–(5), Figs. 6–8.
- Anderson's minimal-enclosing-cone viewer-orientation algorithm, explicit scope
  limitation (no magnification/focus/occlusion): Section II-D-3, p. 98.
- LIGHTING ADVISOR expert system: Section II-E-1, p. 99.
- VANTAGE sensor composition tree, generalized sources: Section II-F-1, pp. 99–100,
  Fig. 9.
- ROSI sensor simulation, Torrance–Sparrow rendering pipeline: Section II-F-2, pp.
  100–101.
- Feature Uncertainty open problem (verbatim quote used above): Section III-C-1, p. 102.
- Constraint-weighting/integration open problem (verbatim quote used above): Section
  III-D, p. 102.
- Illumination planning as generally unsolved/simplified: Section III-E, p. 102.
