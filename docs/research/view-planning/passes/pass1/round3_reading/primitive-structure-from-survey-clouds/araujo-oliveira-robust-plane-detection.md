# Araújo & Oliveira (2020) — "A Robust Statistics Approach for Plane Detection in Unorganized Point Clouds"

**Source:** Pattern Recognition 100 (2020) 107115 (preprint dated March 2019). PDF fetched from
https://www.inf.ufrgs.br/~oliveira/pubs_files/RE/Araujo_Oliveira_plane_detection_low_res.pdf
(full text read via `pdftotext -layout`, all 34 pages / 5 sections + refs).

**Ships as:** Open3D's `detect_planar_patches` (per task brief; not stated in the paper itself — paper predates
that Open3D API by a few years). Local copy: `/tmp/paper_dl/araujo_oliveira_plane_detection.{pdf,txt}`.

---

## 1. Core method (Section 3, Fig. 2)

Deterministic, split → grow → merge pipeline (RSPD = "Robust Statistics Plane Detection"), O(n log n):

1. **Split (§3.1, Algorithm 1).** Recursively subdivide the cloud with an octree until each leaf has
   fewer than `ε` samples (`ε` = 0.1% of total point count, fixed empirically for *all* datasets — this
   is itself a global tuned constant, see caveats below). Test each leaf **bottom-up** (leaves → root):
   if a leaf is planar, stop recursing into it; if none of a node's 8 children are planar, re-test the
   parent itself as a candidate leaf. This bottom-up direction (vs. Vo et al.'s top-down octree) is
   explicitly claimed to preserve small parallel planar structures that top-down stops on early
   (Fig. 3, §2.3).

2. **Robust planarity test (§3.2, the paper's main contribution).** Instead of classic PCA
   eigenvalue-ratio planarity (`|λ3|/|λ1| > τ`, Eq. 1) — which the paper argues is broken because
   covariance/PCA is mean-based and a single outlier can wreck it (breakdown point 0%, §3.2) — it
   estimates the plane from **medians**, which have a 50% breakdown point:
   - Plane point `C` = per-axis **median** of sample positions (Eq. 3).
   - Plane normal `N` = normalized per-axis **median** of sample *normals* (Eq. 4) — note: this
     requires normals to already be estimated per-point before the planarity test runs (they use
     FAST-MCD, robust normal estimation, k=50 neighbors, §4.3 — this is an external dependency, not
     part of RSPD itself).
   - Robust dispersion via **MAD** (median absolute deviation, Eq. 2, k=1.4826 for Gaussian
     consistency), also 50% breakdown point.
   - **T1 — plane-sample distance test (§3.2.1):** per-point distance to plane `dᵢ = |(Pᵢ-C)·N|`
     (Eq. 5); build a robust inlier interval `ID = median(D) ± 3·MAD(D)` (Eq. 6, α=3 chosen to match
     the 99.7% / 3σ rule for normal distributions). The upper bound is called **MDP** (maximum
     distance to plane). Critically, MDP is not used directly as a threshold — it is converted into an
     **angular** threshold via a constructed vector `F = 0.5·max(w,l)·U + MDP·N` (Eq. 7, using patch
     width/length `w,l` and an arbitrary in-plane basis `U,V`) and the angle `θ` between `F` and `N`
     (Fig. 4). Large MDP relative to patch size → curved surface → small θ. Global threshold:
     **θ > 75°** ("experimentally found to work for all tested datasets", §3.2.1, line ~259). This is
     the mechanism that makes the distance threshold scale with patch size — bigger patches tolerate
     more absolute deviation for the same angular threshold.
   - **T2 — normal deviation test (§3.2.2):** angle `φᵢ = acos(|Nᵢ·N|)` (Eq. 8) between each sample's
     own normal and the candidate plane normal; robust interval `IΦ` (Eq. 9) → upper bound **MND**
     (maximum normal deviation). Global threshold: **MND < 60°** (§3.2.2).
   - **T3 — outlier percentage test (§3.2.3):** if ≥25% of a patch's points fall outside `ID` or `IΦ`,
     discard the whole patch as too noisy.
   - So the method is **not literally parameter-free** — it replaces per-dataset tuned thresholds with
     three *global* magic constants (75°, 60°, 25%) plus the leaf-size fraction ε=0.1%, all picked once
     across the 7 test datasets. "Virtually independent of parameter tuning" (abstract) means these four
     numbers are reused everywhere, not that there are zero hyperparameters.

3. **Grow (§3.3, Algorithm 2).** Build a k-NN neighborhood graph (k=50) over the whole cloud once;
   BFS-flood each surviving patch outward, admitting a neighbor point if its distance to the patch's
   plane is `< MDPᵢ` **and** its normal deviation is `< MNDᵢ` (both re-using that patch's own robust
   thresholds — this is what makes growing threshold-free per-patch). Patches are grown in ascending
   order of MND (least-noisy patch first) so clean patches don't get starved by noisier ones claiming
   shared boundary points.

4. **Merge (§3.4, Fig. 5).** Two patches merge iff: (M1) adjacent in the neighbor graph, (M2) normal
   angle between them `< max(MND_A, MND_B)`, and (M3) at least one point of A satisfies B's inlier
   condition (or vice versa) — this last condition specifically prevents merging **parallel but
   spatially separated** patches (e.g., near-parallel walls a few cm apart), which the paper calls out
   as a known failure mode of Vo et al.'s technique (§2.3, and visible in their Computer-dataset result,
   Fig. 8).

5. **Iterate grow↔merge (§3.5)** until "stable" (a patch's re-estimate is only re-triggered if it grew
   by >50% of its prior sample count, exploiting the 50% breakdown point of median/MAD so stats don't
   need updating on every small addition). Worst case `O(log₁.₅ n)` iterations → total cost **O(n log n)**
   (§3.6, full complexity derivation given per phase).

**Output:** a set of delimited planar patches, each carrying `(C, N, MDP, MND)` — i.e., a centroid, a
normal, and two robust dispersion measures — plus its actual inlier point subset (from which width/length
`w,l` were already estimated internally as a bounding-box-like size, Eq. 7 / §3.2.1). No explicit
polygon/OBB boundary is output in this paper's formulation — the patch's spatial extent is the point
subset itself, with `w,l` used only internally for the angular threshold, not exposed as a first-class
result.

---

## 2. Concrete results and numbers (§4, Tables 1–4, Fig. 6–8)

**Datasets (Table 1):** Box (964,806 pts, 6 planes, synthetic cube + **2.5% uniformly-distributed
noise**, 100% planar), Computer (68,852 pts, 9 planes, 94% planar), Room (112,586 pts, 15 planes, 81%),
Utrecht (160,256 pts, 11 planes, 70%, "highly noisy" per §4.3, building facades), Museum (179,744 pts, 23
planes — most of any dataset, 74%), Plant (358,116 pts, 9 planes, 57%, petrochemical plant — many
cylinders/pipes), Boiler Room (5,990,481 pts, 10 planes, 73%, industrial, "mostly planes and pipes").
Box is the only dataset with a *known, controlled* noise model; the rest are real-sensor scans
(different sensors, unspecified per-dataset in the excerpt beyond "different acquisition sensors").

**Ground truth (§4.1):** No CAD/reference models used. Ground truth built by **interactively hand-selecting
inlier points per plane**, then fitting `(C,N)` via the same median-position / FAST-MCD-normal recipe as
the detector. This is the field's accepted protocol in absence of CAD — directly answers the "what is the
ground-truth story" branch question: human-annotated segmentation, not object models.

**Evaluation protocol (§4.2, Eq. 10, Fig. 7):** Voxel-regularize the cloud before scoring (voxel sizes in
Table 2, e.g. 0.02% of largest dimension for Box/Computer/Museum, 0.78% for Room/Utrecht/Boiler Room, 0.19%
for Plant) to counteract **sensor proximity bias** (planes near the sensor are oversampled, would
otherwise dominate precision/recall). A detected plane counts as matching ground truth iff (i) ≥50% of
its voxels overlap the GT plane's voxels, **and** (ii) normal angular difference **≤ 30°** (§4.2, this
30° bound is itself picked "empirically" — i.e., the paper's own definition of "the correct primitive"
tolerates up to 30° of normal error before calling a detection wrong; important context for how loose the
literature's own correctness bar is). Note voxelization here is an **evaluation-only** step — "the
techniques still perform the actual plane-detection process in the original (non-regularized) point
clouds" (§4.2) — so this is not a preprocessing-improves-fitting result, it's a scoring-fairness step.

**Table 3 per-dataset results (RSPD row only, Precision/Recall/F1/Time):**
- Box: 6/6 planes, P=0.97, R=0.84, F1=0.90, 3.35s. Recall hurt because "added noise disturbed the normal
  estimation for some samples ... difference in normal orientation was bigger than MND" (§4.3) — i.e., at
  2.5% uniform noise, some legitimately-planar points get normal-estimation errors exceeding the 60° MND
  cutoff and are excluded. This is the paper's only semi-quantitative noise→exclusion anecdote; it does
  not give a clean noise-σ vs. angular-error curve.
- Computer: 9/9, P=0.96, R=0.93, F1=0.95, 0.11s — best F1 of all methods, distinguished nearby parallel
  wall planes other methods (Vo, KHT-3D) merged.
- Room: 10/15, P=0.85, R=0.81, F1=0.83, 0.24s — best F1 among all compared methods.
- Museum (most planes, 23): 17/23, P=0.95, R=0.75, F1=0.83, 0.49s.
- Utrecht ("highly noisy"): 11/11 (**all** planes detected), P=0.75, R=0.74, F1=0.74, 0.44s — best P/R/F1
  of all methods on the noisiest real dataset.
- **Plant (many cylinders/curved surfaces):** 9/9, P=0.63 (low), R=0.92, F1=0.75, 0.42s.
- **Boiler Room (many cylinders/pipes):** 9/10, P=0.86, R=0.60 (low), F1=0.71, 6.39s.

**Table 4 averages across all 7 datasets:** RSPD: detection ratio A%=0.90 (best), avg precision AP=0.85
(2nd best after RG-Farid's 0.88), avg recall AR=0.79 (best), avg F1=0.81 (best), avg normalized time
ANT=0.0021 ms/sample (2nd fastest after KHT-3D 0.0005 and RG-Vo 0.0003, but those two have much worse F1:
0.67 and 0.69 respectively). Runtime scaling for our "realistic per-object runtime" question: from Table
3, RSPD handles ~360K points in 0.42s, ~1M points in 3.35s, and **~6M points in 6.39s** — sub-linear-looking
in practice, consistent with the O(n log n) bound; single-threaded C++/Eigen on a 2017 desktop CPU (i7-7700K,
32GB RAM, §4.3).

---

## 3. The one explicit negative finding directly about curved surfaces (§4.3 + §5)

This is the single most load-bearing result for our branch:

> "The Plant and Boiler Room datasets contain a large number of cylinders and other curved surfaces.
> Several techniques, **including ours**, identify approximately planar patches in small cylindrical
> regions, thus affecting their performance. Nevertheless, our technique achieved good F-1 score on both
> datasets (0.75 and 0.71, respectively)." (§4.3)

And in the Conclusion (§5), stated as an **acknowledged, unfixed limitation**:

> "Currently, we only perform local planarity tests, disregarding the global structure of the point cloud.
> This may lend to the detection of false planes on large curved structures (e.g., the detection of
> rectangular sections along the axis of a cylindrical element with a relatively large radius). This
> problem could be mitigated by analyzing the surface curvature inside the patch's parent octree node,
> and preventing the detection of local planar patches in curved surfaces." — proposed as **future work**,
> not solved here.

So: a local, octree-cell-scale planarity test **structurally cannot distinguish** a locally-flat patch of
a large-radius cylinder from a true plane — it will emit spurious rectangular "plane" patches tiled along
the cylinder's axis unless a separate curvature check is added (which this paper does not implement). Any
use of this method (or Open3D's `detect_planar_patches` port of it) on cylindrical/rounded features in our
object set needs an explicit curvature-consistency guard downstream, or expect fragmented/spurious planar
patches on rounded parts.

---

## 4. Load-bearing assumptions

- **Per-point normals must already exist and be reasonably good** before RSPD runs — the whole plane
  estimate (Eq. 4) and both robustness tests (T1 via MDP-angle construction, T2 directly) depend on
  input normals. The paper uses FAST-MCD (robust normal estimation, k=50) for all normal-requiring
  methods in its comparison to keep this fair (§4.3) — it does **not** propose or evaluate its own normal
  estimator, and does not quantify normal-estimation error propagation into plane-normal error. If our
  pipeline's normals are noisier/biased (e.g., from registration error or depth bias near edges), RSPD's
  accuracy is bounded by that upstream estimate — untested here.
- **Global constants (75° angular threshold, 60° MND, 25% outlier fraction, ε=0.1% leaf size) are fixed
  once, not derived per-dataset from a noise model.** They happened to generalize across these 7
  datasets (spanning building facades to industrial pipe rooms), but there is no guarantee they transfer
  to a tabletop D405 survey cloud with a very different noise/defect profile (registration error,
  specular dropout, ~4cm fused-vs-single-view offset) — none of which resemble any of the 7 test scenes.
- **Datasets are architectural/industrial scale (rooms, buildings, plants), not small tabletop objects.**
  Smallest-scale dataset (Computer, a desk) is still room-scale; none approximate a single small object
  (box/cylinder ~tens of cm) scanned close-up with an eye-in-hand depth camera. Sample counts (69K–6M)
  and voxel regularization percentages (Table 2) were tuned for these scales; no evidence the same
  constants hold at our object scale and point-cloud density.
- **Ground-truth normal-angle tolerance is loose (≤30°, §4.2)** — the paper's own bar for "correctly
  detected" allows fairly large orientation error. This is useful context but is *not* an angular-error
  budget we should adopt uncritically for an "orthogonal view" requirement, which likely needs a tighter
  bound than 30°.
- **No treatment of multi-view registration/fusion error as a distinct noise source.** The only mention is
  a single throwaway line in §3.2's intro: "errors in registering partial point clouds also lends to
  noise" — not modeled, not separated from sensor noise, no propagation analysis. Does not address the
  branch's registration-covariance-into-primitive-orientation question at all.

---

## 5. What transfers to our setting

- **The robust-statistics recipe (median + MAD instead of mean + PCA-eigenvalue-ratio) is a genuinely
  useful, implementation-ready idea** for plane fitting under our defect profile (registration jitter,
  depth dropout creating outlier clusters) — median/MAD have 50% breakdown point vs. PCA's 0%, directly
  relevant to robustness against the kind of localized dropout/bias artifacts we expect.
- **The per-patch confidence signals (MDP, MND, outlier fraction T3)** are cheap, already-computed,
  interpretable quantities that could seed a fit-reliability score without retraining anything — directly
  answers "what computable signals... obtainable from an off-the-shelf implementation" for the plane case.
  MND in particular is already an angular deviation measure in degrees, plausibly reportable as
  "normal ± MND"-style resolution, though the paper treats it as a hard threshold, not a calibrated
  uncertainty (no covariance derivation, see §4 caveats above).
- **The evaluation protocol structure (voxel-regularize to remove density bias; match on ≥50% voxel
  overlap + angular tolerance; precision/recall/F1)** is a reusable benchmark template even though the
  specific numbers won't carry over — useful for the "which datasets/protocols can seed our benchmark"
  branch question as a *method*, not as ready-made ground truth.
- **Bounded-patch output (centroid C, normal N, plus internally-estimated width/length) is structurally
  close to what a "scan this face orthogonally" action needs** — centroid + normal is exactly a face pose;
  width/length exist internally (Eq. 7) even though not exposed as a first-class polygon output in this
  paper. This is the strongest fit to the "bounded faces, not infinite planes" branch question — but note
  it is not a full oriented-bounding-rectangle, just a scalar size estimate used for thresholding.
- **O(n log n), sub-7-second even at ~6M points, single-threaded, no training** — practical for an
  on-the-fly per-object pipeline step at our expected cloud sizes.

## 6. What does NOT transfer, and why

- **Planes only.** No cylinder, sphere, or cone primitive in this paper at all — it is a plane-detection
  paper, full stop. Directly relevant to the branch question "does the contract's object set close with
  plane+cylinder+sphere+cone" — this single method covers only 1 of 4 required primitive types. Would need
  pairing with a separate cylinder/sphere/cone method (e.g., Efficient RANSAC, Open3D's own RANSAC
  primitives, or a superquadric/cuboid abstraction) to cover our object set; this paper is silent on how
  to combine them.
- **Curved surfaces actively break the local planarity test** (§5, quoted above) — not a graceful
  degradation but a **structural false-positive mode**: locally-flat patches of a cylinder get reported as
  spurious rectangular planes. For our tabletop objects with rounded/chamfered edges, this is a concrete
  failure mode to test for, not a hypothetical.
- **No uncertainty/covariance model tied to a sensor range/incidence-angle error model** — MDP/MND are
  robust *dispersion* statistics of the observed residuals, not a physically-derived covariance from range
  bias or incidence angle. Does not answer the M3C2-EP-style "closed-form covariance under a realistic
  range-error model" branch question; would need to be paired with an external sensor error model.
- **No registration/multi-view fusion error propagation** — single mention, unmodeled (see §4 above).
- **No repeat-scan / identity-stability experiment.** Patches are found independently per run with no
  persistent ID scheme; nothing in the paper speaks to whether "patch 3" stays "patch 3" (in count,
  ordering, or orientation) across two scans of the same object under slightly different noise/registration
  realizations — directly relevant to, and unanswered by, the RQ3 consistency requirement in our branch
  questions.
- **No failure-predictor calibration against downstream task success** — T1/T2/T3 are used only as
  internal accept/reject gates for patch formation, never calibrated against "does the derived pose/action
  fail" in any downstream task. The mapping from fit-score to task success would have to be built by us.
- **Scene/object scale mismatch** — architectural and industrial scans (rooms, facades, plant/boiler
  rooms), not small tabletop objects scanned close-in with an eye-in-hand RGB-D camera. Noise
  characteristics (LiDAR/TLS range noise, room-scale registration) differ qualitatively from D405
  specular/dark dropout and short-standoff depth bias — none of the 7 datasets stress-test that defect
  profile.

---

## 7. Section/figure/equation index (for checkability)

- Core algorithm: §3, Fig. 2 (pipeline diagram), Algorithm 1 (split, §3.1), Algorithm 2 (grow, §3.3).
- Robust planarity test: §3.2, Eqs. 1–9; MDP construction and Fig. 4 (curved vs. planar F-vector diagram),
  §3.2.1; MND, §3.2.2; outlier test, §3.2.3.
- Merge conditions: §3.4, Fig. 5.
- Complexity: §3.6 (derivation to O(n log n)).
- Datasets: Table 1, Fig. 6 (point clouds + ground truth renders).
- Ground truth construction: §4.1 (uses FAST-MCD [ref 39] for GT normals too).
- Evaluation protocol: §4.2, Eq. 10, Table 2 (voxel sizes), Fig. 7 (proximity-bias illustration).
- Per-dataset results table: Table 3; averages: Table 4; qualitative comparison figure: Fig. 8.
- Curved-surface limitation: §4.3 (Plant/Boiler Room discussion) and §5 (Conclusion, explicit future-work
  item on curvature-aware octree analysis).
- FAST-MCD robust normal estimator: ref [39] Rousseeuw & Driessen 1999 (used as the shared normal source
  for RSPD and other normal-dependent baselines, not proposed by this paper).
