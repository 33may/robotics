# Barath & Matas, "Multi-Class Model Fitting by Energy Minimization and Mode-Seeking" (Multi-X, ECCV 2018 / arXiv:1706.00827v2)

Read from: arXiv HTML (cross-checked) + full PDF text (`pdftotext -layout` on arXiv v2 PDF, primary source for all quotes/numbers below). Single-author-team paper, MTA SZTAKI + CTU Prague.

## 1. Core method

**Problem framing.** Multi-class multi-instance fitting: interpret a point set as a mixture of noisy observations from multiple instances of *multiple model classes at once* (e.g. k planes + l cylinders together), not just multiple instances of one class. General formalism in Definition 1 (§2): a class `H_i = {(θ_i, φ_i, τ_i)}` is a parameter space, a distance function, and an estimator; the "multi-class model" is the union `H_∀ = ∪ H_i` including an explicit outlier class `H_o` with constant distance `k`.

**Energy (Eq. 1, §2):** `E(L) = E_d(L) + w_g·E_g(L) + w_c·E_c(L)`
- `E_d` (Eq. 2): data term, sum of per-point distances to its assigned instance's model, `Σ_p φ_{L(p)}(θ_{L(p)}, p)`.
- `E_g` (Eq. 3): spatial-coherence term over a neighborhood graph `N`, pairwise Potts penalty `w_pq·[L(p)≠L(q)]`; `w_pq = 1` throughout the paper. Explicitly noted as swappable for a higher-order geometric-prior term (e.g. perpendicularity) from Pham et al. [27] if needed.
- `E_c` → weighted version `Ê_c` (Eq. 4, Def. 2): per-instance label cost `ψ_l`, from Delong et al.'s label-cost formulation, extended to be class-dependent (a plane and a cylinder can have different costs).

**Optimization = PEARL (Isack & Boykov 2012) + one new move.** PEARL alternates (1) α-expansion labeling given fixed instance set, (2) re-estimating each instance's parameters from its current inliers. Its only way to kill an instance is starving it of points, which the paper states directly is "often unable to delete instances having similar parameters" and "sensitive to the choice of label cost w_c" (§3). Multi-X's contribution is a third move: **mode-seeking in parameter space** — periodically replace the whole cluster of near-duplicate instance hypotheses with the *modes* of their density (Eq. 5, §3):
`H_{i+1} = Θ(H_i)` if `E(L_Θ(H_i)) ≤ E(L_i)`, else `H_i` — i.e. accept the mode-replaced instance set only if it doesn't increase energy, so convergence is preserved.
- Mode-seeking operator Θ = **Median-Shift** (Shapira et al. 2009) using Tukey- or Weiszfeld-medians, chosen over Mean-Shift because it "does not generate new elements in the vector space" (always returns an actual candidate instance) and was empirically more robust (§4, step 2). `|Θ_med(H_i)| ≤ |H_i|` always (monotonic non-increasing instance count).
- Instances are represented as **point sets** (e.g. a line = 2 points, a homography = 4 correspondences) with **Hausdorff distance** as the instance-to-instance metric, specifically to get a well-behaved, non-anisotropic distance for mode-seeking (§4, step 2, and footnote 1: "Details on the choice of model representation are provided in the supplementary material" — the *plane/cylinder* canonical point-set representation is NOT given in the main paper).
- Bandwidth `ε_i` per instance is set automatically as the distance to its k-th nearest neighbor (Georgescu et al. 2003 procedure) — no manual bandwidth tuning (§4, "Automatic parameter setting").
- Label cost `w_c = m·log(|P|)/h_max` (Pham et al. 2014 formula), where `m` = minimal sample size for that class, `h_max` = max expected instance count. Paper states "the choice of h_max is not restrictive" because mode-seeking already suppresses duplicate/weak instances — label cost is a secondary safety net, not the primary count-control mechanism.

**Algorithm 1 (§4)** full loop: `InstanceGeneration` (NAPSAC guided sampling — spatially-local random minimal samples, same neighborhood graph as `E_g`) → repeat{`ModeSeeking` (Median-Shift) → `Labeling` (α-expansion) → `ModelFitting` (Weiszfeld/L2 re-estimation, i.e. truncated-L2 given the labeling)} until convergence.

**Algorithm 2, "Model Validation" (§4, step 5):** post-hoc pruning of *spatially coherent outlier clusters* that survived as instances. For each surviving instance, run `t` trials (default t=100) of: draw a minimal subset from its inliers, refit, compute mean distance from the refit model to all its inliers; accept the instance only if this cross-validated mean distance is below the same outlier threshold `γ` used for fitting. This is the paper's only formal accept/reject signal — **binary**, not a graded confidence score, and it is a self-consistency check (does resampling reproduce a similar model), not an absolute correctness check.

**Distance functions given explicitly only for lines and circles** (§2, worked examples): `φ_line = |cos(α)x + sin(α)y + c|`, `φ_circle = |r − √((c_x−x)²+(c_y−y)²)|`. **The plane and cylinder distance functions / parametrizations are never written out in the main paper** — deferred to supplementary material not fetched here (footnote 1). This is a real gap for reproducing exact axis/normal DOF handling.

## 2. Concrete numbers

**Plane+cylinder fitting on LIDAR data — Table 6 (§5, "Simultaneous Plane and Cylinder Fitting"), misclassification error %:**

| Test | PEARL | T-Linkage | RPA | Multi-X |
|---|---|---|---|---|
| (1) | 10.63 | 57.46 | 46.83 | **8.89** |
| (2) | 10.88 | 41.79 | 53.39 | **4.72** |
| (3) | 37.34 | 52.97 | 61.64 | **2.84** |
| (4) | 38.13 | 38.91 | 41.41 | 19.38 |
| (5) | 17.20 | 51.83 | 53.34 | 16.83 |
| (6) | 17.35 | 61.77 | 51.21 | **21.72** (worst here — PEARL wins case 6 too, 17.35 < 21.72) |
| (7) | 6.12 | 12.49 | 80.45 | **5.72** |

Text: "Table 6 reports that Multi-X is the most accurate in all test cases except one" (§5, referring to case 4 where PEARL=38.13 vs Multi-X=19.38 — Multi-X is still better there numerically; re-reading, the "except one" case with Multi-X *not* best is case (6): PEARL 17.35 < Multi-X 21.72). Either way: **misclassification error is a point-labeling accuracy metric (points correctly assigned to the right instance, matched to ground truth by optimal permutation), not a geometric/angular parameter error.** No normal-angle or axis-angle error is reported anywhere for plane or cylinder fits, in this section or any other.

**Dataset (§5):** "The annotated database consists of traffic signs, balusters and the neighboring point clouds truncated by a 3-meter-radius cylinder parallel to the vertical axis. Points were manually assigned to signs (planes) and balusters (cylinders)." Footnote: "It will be made available after publication" — no point counts, no ground-truth instance counts, no sensor spec given for this specific dataset (contrast with the AdelaideRMF/Hopkins datasets used elsewhere in the paper, which are named public benchmarks). Fig. 6 shows two example scenes, captioned "there is only one cylinder on the two scenes: the pole of the traffic sign on top" — i.e. scenes are street/urban-furniture scale (traffic sign pole + plate, balusters/railings), a 3 m capture radius. This is outdoor terrestrial/aerial LIDAR survey scale, not tabletop object scale.

**Outlier threshold used for planes/cylinders: 10 cm** (§4, "Automatic parameter setting": "planes and cylinders (10 cm)"). This is a *user-set, problem-specific* constant, not automatically derived — automatic in this paper only refers to bandwidth `ε` and label cost `w_c`. A 10 cm inlier threshold is roughly 3–4 orders of magnitude coarser than the mm-level defects our framing doc describes (few-mm registration error, ~4 cm fused-vs-single-view offset) — i.e. it's tuned to LIDAR range noise at meters-scale, not to a D405 tabletop scan.

**Spatial coherence weight `w_g = 0.3`** "performed well in the experiments" (§4) — a single fixed value used across all problem types in the paper, not tuned per plane/cylinder case specifically.

**Runtime — Table 8, column (5) "planes and cylinders" (§5.1):**

| # points | Multi-X | T-Linkage |
|---|---|---|
| 100 | 0.1 s | 0.4 s |
| 500 | 3.8 s | 15.9 s |
| 1000 | 7.5 s | 120.9 s |

(Correction vs. a naive read of the table: columns are (1) lines/circles, (2) homographies, (3) two-view motions, (4) video motions, (5) planes/cylinders — I initially mis-mapped column (1)'s numbers to column (5) on a first pass; the numbers above are column (5), verified against the PDF layout.) Claim: "Multi-X runs in time approximately linear in the number of data points" and "is orders of magnitude faster than currently available Matlab implementations of J-Linkage, T-Linkage and RPA" (§5.1). **Point counts tested top out at 1000 points** — far below a typical fused survey PLY (thousands–millions of points); no runtime data at that scale is given, so the "linear" claim is asymptotic conjecture. No CPU is even specified, and comparisons are against MATLAB reference implementations of competitors (favorable to Multi-X, likely C++/optimized).

**Noise-sensitivity experiment (Fig. 3b, §"Comparison of PEARL and Multi-X")** exists only for the **synthetic 2D line-fitting toy problem** (3 lines × 100 points + 200 outliers, Gaussian pixel noise σ swept 0–12 px). It shows Multi-X is "less sensitive to the noise than PEARL" — no numbers beyond the plot, and **no equivalent noise-vs-accuracy sweep exists for the plane/cylinder 3D case.** The 66.7% outlier ratio (200 outliers / 300 inliers) in this same toy setup is the only quantified outlier-ratio robustness data point in the whole paper, and again it's for 2D lines, not 3D primitives.

**Fitted-model comparison for two-view plane segmentation on AdelaideRMF (Table 2/3)** and motion segmentation (Tables 4/5/7) are image-correspondence problems (homographies/motion subspaces), not 3D point-cloud plane fitting — not directly relevant to our RANSAC-family primitive question but confirm the same misclassification-% metric convention throughout the paper (never a geometric-parameter error).

## 3. What this paper does NOT do (negative findings — first-class for our branch)

- **No normal or axis angular error is ever reported.** Search of the full text confirms zero occurrences of "normal" as a surface-normal concept; normals are not computed, used as fitting features, or evaluated. Fitting is done directly on raw 3D point coordinates via minimal-sample estimators `τ`. **This paper cannot supply Branch 1's requested angular-error number** — its only accuracy metric is point-to-instance labeling correctness (misclassification %), which conflates segmentation quality and geometric accuracy and gives no way to back out a normal/axis angle error.
- **No bounded-face output.** Instances are infinite planes/cylinders (a plane in `θ = (normal, offset)`-style form, a cylinder as axis+radius, per the general Definition 1 formalism); the method returns a **point-to-instance labeling** plus fitted infinite-primitive parameters. Extent, boundary, or centroid of a face is not computed or evaluated anywhere. Getting a "face center + face size" (what the branch's orthogonal-scan action needs) would require a separate post-hoc step (e.g., convex hull / OBB of each instance's inlier points) that this paper doesn't perform.
- **No partial/incomplete-cylinder or occlusion treatment.** Full-text search for "partial", "incomplete", "occlusion", "missing" returns nothing. The method implicitly assumes the observed inliers are sufficient to run α-expansion + Median-Shift + re-estimation; there is no symmetry completion, no mention of how a half-scanned cylinder axis is recovered versus a fully-scanned one, no experiment stressing partial coverage.
- **No instance-identity/order stability treatment, and no path to it.** Instance generation is random (NAPSAC minimal samples), mode-seeking prunes clusters based on density in parameter space, and α-expansion's local optimization is run once per instance-set — nothing in the algorithm guarantees that re-running on a re-scanned (even nominally identical) point cloud reproduces the same instance count, ordering, or per-instance point membership. This is a direct negative for RQ3's "face 3 stays face 3" consistency requirement: Multi-X's output is an unordered, run-dependent labeling with no persistent identity across runs, by construction.
- **No continuous/graded confidence score.** Algorithm 2 (Model Validation) is the only introspection mechanism: a binary accept/reject test per instance (mean cross-validated residual < threshold `γ`). It's a self-consistency check (does refitting from a random subset reproduce something close to the fitted model), not a calibrated probability, and it is never validated against any downstream task-failure criterion — the paper only reports that it removes "statistically insignificant instances," with no experiment quantifying false accept/reject rates.
- **No code or dataset release confirmed in the paper text** — both are promised ("will be made available after publication," §5 footnote 5 and §6 conclusion) but no repository URL appears anywhere in the v2 PDF. (Not independently re-verified via a live GitHub search in this pass — flag for a follow-up check if the "off-the-shelf implementation" branch question needs a firm answer; the paper text itself gives no link.)
- **No citation of the classical point-cloud RANSAC-family lineage** (no Schnabel et al. "Efficient RANSAC", no explicit citation of prior point-cloud plane/cylinder detection work) — Multi-X positions itself against 2D/image-correspondence multi-model fitting baselines (PEARL, T-Linkage, RPA, J-Linkage, RCMSA) that were themselves mostly developed for line/circle/homography/motion problems and only secondarily applied to 3D LIDAR by the authors themselves. This is a paper about the *labeling/energy-optimization* problem transplanted onto plane+cylinder fitting as one of five demonstrated applications, not a paper specialized for point-cloud primitive extraction.
- **Scale mismatch to our setting is structural, not incidental.** The plane/cylinder outlier threshold (10 cm), the capture geometry (3 m-radius truncation around traffic-sign/baluster street scenes), and the runtime experiments (capped at 1000 points) all indicate this method was built and tuned for outdoor terrestrial/aerial LIDAR survey scale — objects meters apart, cm-level noise floor. Our tabletop D405 setting is mm-registration-error, sub-cm object scale, likely thousands-to-millions of points per fused PLY. None of Multi-X's reported operating points bracket our regime; every number here would need re-tuning and re-validation from scratch (outlier threshold, `w_g`, `h_max`, and almost certainly the runtime/scaling behavior at real cloud sizes).

## 4. What plausibly transfers to our setting

- **The framing itself is the main transferable idea**, not the numbers: fit multiple primitive classes *jointly* under one global energy instead of sequential greedy RANSAC, specifically to avoid the "phantom shape" failure mode the branch cites Multi-X for (greedy plane-then-cylinder extraction leaving spurious low-support instances behind). The label-cost + mode-seeking mechanism is a principled, largely-automatic way to answer "how many instances, of which classes" without hand-tuned per-scene iteration counts — directly relevant to the "which primitive type and how many" framing question in our selection rationale.
- **The energy formulation is class-agnostic and extensible** (Definition 1/2): adding "sphere" and "cone" alongside plane/cylinder is architecturally trivial (just define `θ, φ, τ` for each), which is reassuring for the contract's plane+cylinder+sphere+cone vocabulary question — Multi-X's *framework* would close that, even though this specific paper only demonstrates plane+cylinder.
- **Model Validation (Algorithm 2)'s pattern** — cross-validated refit-and-compare-residual as an accept/reject gate — is a reusable, implementation-cheap idea for a failure-detection tier (Branch's "two-tier design, else uniform sphere" question), even though it's not calibrated or graded in this paper. We'd have to build the calibration ourselves.
- **NAPSAC-style spatially-guided instance generation** (vs. pure random sampling) is a reasonable prior for our single-object tabletop scans, where inliers are much more spatially coherent than in an urban street scan with heavy background clutter.

## 5. What does not transfer / must be re-derived

- All quantitative operating points (10 cm threshold, `w_g=0.3`, up-to-1000-point runtime figures) — wrong scale, need re-derivation for mm-level object scans with potentially far larger point counts.
- The misclassification-% metric alone is insufficient for our branch — we need geometric parameter error (normal/axis angle vs. ground truth), which this paper's evaluation protocol does not produce. Any benchmark we build on top of Multi-X-style output needs an added geometric-error layer.
- No mechanism here addresses bounded-face extraction, partial/occluded-cylinder axis recovery, or run-to-run identity stability — three of the branch's harder questions (RQ3 consistency, bounded faces, partial-cylinder symmetry) get no evidence from this paper and must be answered from other literature.

## 6. Citation map (for checkability)

- Energy formulation: §2, Eqs. 1–4, Definitions 1–2.
- PEARL baseline and its weakness: §3, prose paragraph before Eq. 5.
- Mode-seeking / Median-Shift move: §3 Eq. 5; §4 step 2.
- Full algorithm: §4, Algorithm 1 (steps 1–5 enumerated in prose after the box), Algorithm 2 (Model Validation).
- Automatic parameter setting (bandwidth, label cost, thresholds, `w_g`): §4, subsection "Automatic parameter setting."
- Plane+cylinder results table: §5 "Simultaneous Plane and Cylinder Fitting," Table 6, Fig. 6.
- Runtime: §5.1, Table 8, column (5).
- Noise/outlier synthetic experiment: "Comparison of PEARL and Multi-X" subsection, Fig. 3(a)/(b).
- Dataset description + "will be made available" promise: §5, footnote 5; code promise: §6 Conclusion, last paragraph.
- Missing plane/cylinder distance-function definition: footnote 1 (points to supplementary material, not fetched in this pass).

## 7. Answers to specific branch questions this document can/cannot serve

- **"Angular error on recovered face normal or cylinder axis at X noise/incompleteness"** — **not answered**. No angular-error metric anywhere in the paper; only point-labeling misclassification %.
- **"Does the object set close with plane+cylinder+sphere+cone?"** — Multi-X's *formalism* supports arbitrary classes trivially, but only plane+cylinder is empirically demonstrated; sphere/cone are not evaluated here.
- **"Bounded faces (extent/centroid) vs infinite primitives"** — Multi-X returns infinite primitives + point labels only; no bounded-face output.
- **"Primitive decomposition → per-primitive viewpoints, precedent?"** — no view-planning content in this paper at all; purely a geometric-fitting method.
- **"Repeatability of decomposition across re-scans / face-identity stability"** — no evidence for it, and by algorithm construction (random instance generation + local optimization) there is no built-in identity/order stability.
- **"Partial cylinder + symmetry completion"** — not addressed; no partial-data experiments.
- **"Noise sigma / neighbourhood k / normal angular error relationship"** — not applicable; no normals used, no such sweep.
- **"Closed-form covariance for fitted plane/cylinder orientation"** — not present; the paper has no uncertainty quantification, only a binary cross-validation accept/reject gate.
- **"Computable signals to declare a fit unreliable, off-the-shelf"** — Algorithm 2's cross-validated mean-residual-vs-threshold test is the one candidate signal here; binary, uncalibrated, and would need our own downstream-failure calibration experiment.
- **"Datasets/protocols to seed our benchmark"** — the paper's own plane+cylinder LIDAR dataset was only promised for release (footnote 5), no confirmed public link found in the text; not verified as actually available (would need a live check, e.g. author's GitHub, separate from this reading pass).
