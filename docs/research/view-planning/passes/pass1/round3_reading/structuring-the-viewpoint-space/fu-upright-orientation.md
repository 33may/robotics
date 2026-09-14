# Fu, Cohen-Or, Dror, Sheffer — "Upright Orientation of Man-Made Objects"

ACM TOG 27(3) (SIGGRAPH 2008), Article 42. Fetched directly from the authors' page
(`https://www.cs.tau.ac.il/~dcor/articles/2008/upright_orientation.pdf`), a 7-page
PDF containing the full paper (no separate abstract-only version existed to
mistake for it). All 7 sections read in full: Introduction, Related Work, System
Overview, Shape Analysis and Feature Extraction (§4.1–4.2), Assessment Function
Learning (§5), Implementation and Results (§6), Conclusion. Section numbers below
match the paper.

## Core idea

The paper solves a narrower problem than the branch's selection blurb implies:
**given a free-standing man-made 3D mesh in an arbitrary/unlabeled pose, predict
which one of its convex-hull faces is the "natural base" it would rest on under
gravity** — i.e., recover the up-axis (1 rotational DOF), not a full canonical
frame. The method has two stages:

1. **Candidate-base generation (§4.1), purely geometric, no learning.** A
   candidate base is a set of coplanar points on the mesh lying on its convex
   hull; its *supporting polygon* is the 2D convex hull of those points in the
   supporting plane; the upright direction is the plane's outward normal. Convex
   hull facets are first merged via Variational Shape Approximation
   [Cohen-Steiner et al. 2004] to eliminate noise, then supporting polygons are
   extracted directly from the (simplified) convex hull. Typically **"at most a
   dozen" candidates per object** (§3). This set is further pruned by a
   **static-equilibrium stability test**: a candidate survives only if the
   projection of the object's center of mass onto the supporting plane falls
   inside the supporting polygon (§4.1) — the necessary condition for the object
   to actually balance on that face without tipping.
2. **Learned ranking over 8 geometric features per candidate (§4.2–§5).**
   Features: static stability (`min_θ d_inner(θ)/d_outer(θ)`, a normalized
   0–1 ratio of center-of-mass-to-supporting-polygon-boundary distance vs.
   center-of-mass-to-convex-hull-projection-boundary distance), 2 symmetry
   attributes (coincidence distance and collinearity distance of 4 characteristic
   points: projected center of mass, supporting-polygon barycenter, hull-
   projection barycenter, actual-base barycenter) plus a base-symmetry distance,
   2 parallelism attributes (face parallelism = area of mesh faces parallel to
   the candidate plane; base parallelism = area of other supporting polygons
   parallel to this one), and visibility (per-face visibility computed by
   rendering from 32 uniform view directions and color-ID'ing faces; a face
   seen in <5% of views is "occluded"; per-base visibility = total visible-face
   area after adding the supporting polygon as an occluder, normalized by
   baseline mesh visibility) plus base area. These 8 features feed a **combined
   Random Forest (2000 trees) + quadratic-kernel SVM** classifier, each scoring
   candidates independently (not jointly, despite most objects having exactly
   one correct base — framed and tried as a structured-estimation problem first,
   but exponential-model output gave "mediocre results," §5): `Y_combined =
   α·Y_rf + (1-α)·Y_svm`, Eq. 1, α=0.6 tuned by cross-validation (§5.3). The
   argmax-scoring candidate is the predicted upright base.

**The PCA negative finding (§2, exactly the branch's selection premise).**
Quoted directly: *"it is well known that principal axes are not robust for pose
normalization of many models [Kazhdan et al. 2003] and they certainly do not
always produce compatible alignments with the upright orientations of objects...
More important, PCA does not fully resolve the orientation problem, but simply
reduces the orientation candidate set to six candidates."* Six = 3 principal
axes × 2 directions each. This is stated as motivation for the paper's own
approach, not evaluated empirically anywhere in the paper (no PCA baseline
number in Table 1) — it is a related-work claim, not a measured comparison. A
second, milder critique of *symmetry*-based axis-narrowing is also given: even
principal *symmetry* axes [Podolak et al. 2006] are "more robust than principal
axes," but "not all man-made models are symmetric, nor does symmetry necessarily
imply their upright orientations" (§2, citing Figure 6) — symmetry only narrows,
it doesn't decide, echoing the PCA critique one level down.

## Concrete results and numbers

- **Table 1 — prediction accuracy** (10-fold CV on 345-model training set /
  independent 819-model test set, avg. 5–5.75 candidate bases/model): SVM alone
  81.9% / 78.6%; RF alone 83.4% / 85.3%; **Combined classifier 90.5% / 87.5%**.
- **Single-feature ablation:** stability attribute alone → SVM accuracy only
  **50.1%** (§6.2, footnote: RF/combined can't be run on a single feature — RF
  needs more features to train meaningfully).
- **Confidence-gated accuracy:** limiting to the 30% most-confident predictions
  raises accuracy to **~95%** (§6.2, unlabeled inline figure plotting
  accuracy(t) vs. fraction(t) of retained models — no exact curve values given
  beyond the 30%→95% headline and the axis range 0.86–1.0).
- **Failure-mode breakdown** (§6, qualitative, not exhaustive/exclusive
  percentages — paper's own words: "difficult to precisely analyze... we could
  suggest several possible reasons"): **~20%** of failures = multiple plausible
  orientations, mostly box-like shapes (refrigerator, cabinet); **~30%** = models
  (bed, sofa) where "geometric features do not seem to provide enough
  information... geometry alone might not be sufficient"; **~5%** = mesh
  representation problems (one-sided representation, holes, noise) causing
  incorrect visibility or unstable convex-hull computation; remainder attributed
  to feature conflicts (e.g., visibility vs. stability disagreeing for a mirror
  model, Fig. 7). Overall: **misorients ~10% of models tested** (Fig. 7 caption).
- **Timing (Table 2)**, Intel Core 2 2.13GHz, 2GB RAM, single-threaded: avg
  model 17,148 vertices / 22,358 faces. Candidate selection **7.09s**
  (bottleneck = convex-hull computation, "can be accelerated by simplifying the
  model in advance"); feature extraction **3.53s** (bottleneck = visibility
  computation, "can be sped up using graphics hardware"); classifier testing
  **2.2 ms**; **total 10.62 s/model**. One-time training-set-wide assessment-
  function training: **12.00 s** (for the whole 345-model set, not per-model).
- Training/test corpora: Princeton Shape Benchmark [Shilane et al. 2004] plus
  Google 3D Warehouse / TurboSquid / 3D Cafe for the test set; no duplication
  between train/test verified; candidate-base natural/not labels are
  **manually assigned** (§6.1).

## Load-bearing assumptions

- **Objects must be free-standing man-made shapes with a well-defined gravity-
  driven "natural base"** — explicitly excludes floating objects (ships,
  airplanes) (§1). The whole candidate-generation step (§4.1) presupposes an
  object that rests stably on *some* flat supporting plane; nothing in the
  method applies to objects with no valid stable base at all.
- **Candidate generation needs a convex hull, ideally a clean closed mesh.**
  Convex-hull computation (CGAL) is the stated bottleneck of candidate
  selection (7.09s of the 10.62s total, Table 2) and 5% of test failures are
  attributed to "representation problems... leading to incorrect visibility
  computation or unstable convex-hull computation" (§6). The paper does
  deliberately avoid 2nd-order-derivative attributes (curvature, feature lines)
  specifically because "input meshes might suffer from poor sampling,
  inconsistent vertex/face normals, or isolated components" (§4.2) — so the
  *feature* computation is somewhat noise-tolerant by design, but the
  *candidate-generation* geometry (convex hull + coplanar-facet merging via
  Variational Shape Approximation) is not shown to tolerate a raw noisy point
  cloud; every example and every timing number is mesh-based (vertices/faces
  reported, never points).
- **Requires labeled training data at corpus scale.** 345 hand-labeled training
  models, 819 test models, both drawn from general-purpose 3D shape
  repositories. This is a supervised-learning method, not a closed-form rule —
  there is no analytic "here is the formula for the natural base" fallback; the
  8-feature vector only becomes a decision via a trained RF+SVM ensemble.
- **Solves only 1 of 2 orientation DOF.** Explicitly stated twice: the abstract
  frames the problem as reducing "the two-dimensional (spherical) orientation
  space" (S², i.e., which direction is "up") to a discrete candidate set; §2
  states the practical payoff as reducing "the orientation alignment problem
  from two to one degrees of freedom." **Rotation about the resulting up-axis
  (front/side/which-face-faces-which-way) is never addressed anywhere in the
  paper** — no attribute, no candidate set, no discussion targets azimuthal
  disambiguation at all.
- **Generalization across function classes is empirical, not architecturally
  guaranteed.** The 8 attributes are deliberately "abstracted from concrete
  shapes" to generalize (§4.2), and the paper reports it works on humanoid
  figures despite zero humanoid training examples (§6.2) — but this is an
  empirical claim about the training corpus's diversity, not a proof the
  method transfers to shape classes qualitatively unlike anything in PSB/
  Warehouse/TurboSquid/3D Cafe (i.e., unlike our tabletop industrial parts).

## What transfers to our setting

- **The PCA negative finding is directly citable for RQ13.** "Principal axes are
  not robust... and do not always produce compatible alignments with the
  upright orientations of objects... PCA does not fully resolve the orientation
  problem, but simply reduces the orientation candidate set to six candidates"
  (§2) is an exact match to the branch's premise: PCA narrows a search space,
  it does not solve the addressing problem, and any canonical-frame machinery
  built on PCA principal axes inherits this non-robustness (axis sign/order
  flips under small perturbations, well-known separately in the PCA-alignment
  literature this paper cites [Kazhdan et al. 2003]).
- **Confirms front/side disambiguation is still an open, unsolved sub-problem**
  even after "up" is fixed. This paper — the one dedicated specifically to
  orientation canonicalization — only ever claims to resolve 1 of 2 rotational
  DOF (§2: "two to one degrees of freedom"). For RQ13's question ("is
  front/side disambiguation still needed for an inspection action space?") this
  is direct, explicit evidence: **yes, nothing in this literature closes that
  gap; even the paper built to solve upright orientation leaves it open.**
- **The stability-filtered, convex-hull-face candidate-generation step (§4.1) is
  a reusable, learning-free geometric primitive**, independent of the paper's
  learned ranker: "take convex hull faces (after coplanar merging), keep only
  those whose center-of-mass projection lands inside the face" is a cheap,
  deterministic way to generate a small set of physically-meaningful candidate
  reference planes for an object — conceptually adjacent to (though not the
  same problem as) selecting a primitive's addressable faces in our setting.
  It would need re-validation against a noisy fused point cloud rather than a
  clean PSB mesh before reuse (see below).
- **The stability metric formula itself is a reusable, closed-form geometric
  quality function** (`min_θ d_inner(θ)/d_outer(θ)`, §4.2, "Static Stability")
  — a cheap way to score how well a candidate supporting plane would hold an
  object, if our inspection setting ever needed to reason about resting-pose
  quality (it currently does not: our object's pose is already given by
  physics once placed on the table for inspection).
- **The visibility-attribute construction (32-view render, color-ID per face,
  <5% visibility threshold ⇒ occluded, §4.2 "Visibility") is a concrete,
  cheap recipe for a per-face visibility/coverage score** that could inform an
  RQ1-style "how visible/coverable is this face" judgment, independent of the
  paper's orientation-selection use case.

## What does NOT transfer, and why (negative findings)

- **The paper's core problem is largely moot for our setting.** Our object
  arrives already resting stably on the table under gravity — the "which face
  is the natural base" question this paper spends its entire method on is
  directly observable from the survey cloud (whichever surface contacts/faces
  the table *is* the base), not something we need to predict from geometry
  alone. **This paper solves a problem we don't have**, not the problem
  ("front/side disambiguation," "canonical up-axis machinery for addressing")
  the branch selected it for.
- **It explicitly does not solve the sub-problem we'd actually need (azimuthal/
  front-side disambiguation).** As above — 1 of 2 DOF only, stated plainly, no
  mechanism offered even as future work beyond a vague closing remark about
  possibly using "dominant line and plane features" for *other* object classes
  (§7, Conclusion) — not a concrete proposal.
- **Requires supervised learning over a large labeled multi-object-class
  corpus** (345+819 models, hand-labeled). This is heavyweight, general-purpose-
  shape-repository infrastructure, mismatched to our setting of a handful of
  known industrial part classes (box, cylinder, box+cylinder assembly) where a
  deterministic primitive-fitting rule (already acceptable per the framing doc:
  "primitive-level deterministic rules are acceptable") is far cheaper and more
  appropriate than training an RF+SVM ensemble on thousands of labeled meshes.
- **No evidence of robustness to a noisy, fused point cloud.** Every input in
  the paper is a mesh (vertex/face counts reported in Table 2); the visibility
  and stability computations both depend on a valid convex hull and coherent
  face structure. 5% of the paper's *own* test-set failures are attributed to
  mesh defects (holes, noise, one-sided faces) *even on curated 3D-repository
  models* — our actual input (fused PLY, ~4cm cross-view offset, per-view depth
  dropout, real defects, per the framing doc) is a harder input class than
  anything validated here, and the paper offers zero point-cloud-native
  variant of convex-hull/coplanar-merging/visibility-rendering.
- **No cross-shape addressing consistency, adjacency, ordering, or seam
  discussion whatsoever.** Like every other paper sampled in this branch so
  far (Magaña et al., Faugeras et al.), this is entirely single-object-instance
  scoped: candidates are ranked and one winner is picked per object; there is
  no notion of a stable "base_1"/"face_3" identity meaningful across different
  shapes, no adjacency between candidate bases, no ordering. Reinforces
  (further, independently) the branch's accumulating finding that cross-shape
  view-addressing consistency has no precedent in the sampled literature
  (relevant to RQ20).
- **No comparison to a uniform/tessellated-sphere-style baseline on any metric**
  — not applicable in the same sense as the view-planning papers (this is an
  orientation-classification paper, not a view-planning paper), but it's worth
  noting for RQ19's benchmark-gap tracking that this paper's own "baseline" is
  PCA, cited but never run as a quantitative comparison point in Table 1 (no
  PCA accuracy number appears anywhere) — the PCA critique is asserted from
  prior literature [Kazhdan et al. 2003], not measured by this paper itself.

## Citations (for checkability)

- Six-candidate PCA critique, non-robustness of principal axes: §2 ("Pose
  Normalization" paragraph), citing [Kazhdan et al. 2003].
- Symmetry-axis critique (Podolak et al. 2006 more robust but still only
  narrows, doesn't decide): §2, citing Figure 6.
- Two-to-one DOF reduction framing: §2 ("3D Shape Retrieval" paragraph, last
  sentence) and Abstract ("two-dimensional (spherical) orientation space").
- Candidate-base definition, supporting polygon, upright direction: §4.1,
  Figure 2.
- Coplanar-facet merging via Variational Shape Approximation: §4.1, citing
  [Cohen-Steiner et al. 2004].
- Stability necessary condition (center of mass over supporting polygon),
  candidate pruning: §4.1.
- 8-feature vector: stability formula: §4.2 "Static Stability"; symmetry
  4-point construction: §4.2 "Symmetry", Figure 2/3; parallelism: §4.2
  "Parallelism", Figure 4; visibility (32 views, 5% threshold): §4.2
  "Visibility", Figure 5.
- Combined classifier formula, α=0.6: §5.3, Eq. 1.
- Structured-estimation attempt with exponential models gave "mediocre
  results": §5.
- Table 1 accuracy numbers: §6.2.
- Stability-alone 50.1% ablation, footnote on single-feature RF limitation:
  §6.2, footnote 2.
- Confidence-gated 30%→95% accuracy: §6.2 "Confidence Dependent Performance."
- Failure-mode breakdown (20%/30%/5%), "misorient only about 10%": §6.2 text
  and Figure 7 caption.
- Table 2 timings, model statistics: §6, "Timings."
- Training/test corpus construction (PSB, Warehouse, TurboSquid, 3D Cafe;
  345/819 models): §6.1–§6.2.
- Conclusion, future work on non-standing/natural shapes and dominant-line/
  plane candidate generation: §7.

## Assessment for the branch

This paper delivers exactly the negative finding it was selected for — the PCA
critique in §2 is a clean, citable, directly-quotable statement that principal
axes are non-robust and only narrow orientation to six candidates, useful
ammunition for RQ13's canonical-frame skepticism — but it does **not** deliver
the "unbudgeted prerequisite" the selection blurb hoped for. The paper's actual
contribution (predicting which convex-hull face an object would rest on) is a
problem our setting doesn't have, because our objects arrive already resting on
the table — the up-axis is free, observable directly from the survey geometry,
not something requiring an 8-feature learned classifier over a 345+819-model
corpus. What the paper actually confirms, load-bearingly, is the *negative*
half of RQ13: **even a paper purpose-built for upright-orientation canonicalization
solves only 1 of 2 rotational degrees of freedom** (§2: "two to one degrees of
freedom") and offers no mechanism, not even a sketch, for the front/side
(azimuthal) disambiguation that any cross-shape "face_3 means the same thing on
every box" addressing convention would actually need. Combined with zero
discussion of cross-shape consistency, adjacency, or ordering (consistent with
every other paper sampled in this branch), this document should be read as
confirming an open gap rather than supplying the missing machinery: canonical
up-axis recovery has real, if heavyweight and learning-based, solutions in the
literature, but canonical *rotation-about-up* recovery — the piece our
addressing scheme would actually need to make "face 3" mean the same physical
face across every box we survey — remains unaddressed anywhere we've looked so
far.
