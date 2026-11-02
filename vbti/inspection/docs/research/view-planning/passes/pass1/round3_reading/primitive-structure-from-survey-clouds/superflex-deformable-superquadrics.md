# SuperFlex: Deformable Superquadrics for Point Cloud Decomposition (2026)

**arXiv:** 2607.01015 · **HTML:** https://arxiv.org/html/2607.01015v1 · **Project page:** https://superflex3d.github.io · **Code:** https://github.com/GabrielTavernini/superflex
**Authors:** Gabriel Tavernini, Elisabetta Fedele, Tiago Novello, Leonidas Guibas, Marc Pollefeys, Francis Engelmann

Read via arXiv HTML full text (multiple targeted fetches covering abstract, §1–§6, Appendix A/B/D, Tables 1–5, 9). Section numbers below are as given in the HTML rendering; treat table/eq numbers as approximate transcriptions from a fetched summary, not a hand-verified PDF read — flagged where precision matters.

## 1. Core method

Learned, feed-forward, **not** RANSAC/optimization-only (though it has an optional test-time optimization stage). Transformer decoder (PVCNN point-voxel encoder → 3-layer transformer decoder) outputs a fixed **P=16** slot tokens per object, each regressing 19 params: 11 rigid superquadric params (translation, rotation, 3 scales `s=(sx,sy,sz)`, 2 shape exponents `ε1,ε2`) + 8 deformation params (2 tapering `τx,τy`, 6 bending `(k_x,α_x,k_y,α_y,k_z,α_z)` — one curvature/orientation pair per axis) + a scalar existence probability `e_p ∈ [0,1]`.

**Superquadric implicit surface (canonical form, "Eq. 1")**:
`((x/sx)^(2/ε2) + (y/sy)^(2/ε2))^(ε2/ε1) + (z/sz)^(2/ε1) = 1` — a closed, bounded solid parameterized entirely by pose (t, R) + scale `s` + two shape exponents. This is the property most relevant to our branch: **every primitive is inherently a bounded volume with an explicit centroid, orientation and half-extents (s)** — not an infinite plane. But note (§6 below): the paper never derives per-*face* boundaries from this; it stays at whole-primitive granularity.

**Deformations** (Appendix A): tapering linearly scales the cross-section along z as a function of z; bending maps the primitive's central axis onto a circular arc of curvature k. Purpose: let one primitive faithfully cover a "bent and tapered" part instead of requiring several rigid superquadrics — directly named as fixing SuperDec's "inherently convex and symmetric" primitives.

**Loss (§4.1)**: `L = λ_IoU·L_IoU + λ_SDF·L_SDF + L_reg`, replacing SuperDec's Chamfer-only objective:
- `L_IoU`: soft volumetric occupancy match on points sampled in the unit cube (global shape coverage).
- `L_SDF`: per-point signed-distance-to-surface loss weighted by the predicted soft segmentation assignment (surface precision), stabilized with an asymmetric truncated tanh.
- `L_reg` = overlap penalty + sparsity (parsimony) + existence-probability supervision.

**Test-time optimization (§4.2)**: given the feed-forward prediction, refine only "active" primitives (e_p>0.5) with 1000 Adam iterations on the same loss. Large quality jump, large cost (see §2).

**Robustness-to-occlusion training (§4.3)**: take a *complete* point cloud, run the fully-trained + test-time-optimized model on it to get a high-quality decomposition, use that as **pseudo-ground-truth**, then fine-tune a second network on *partial* versions of the same cloud (synthetic occlusion) to regress toward the pseudo-GT via Hungarian-matched set loss: `L_sup = L_geom + λ_IoU·L_IoU + L_rigid` (the auxiliary rigid loss front-loads coarse rigid shape before letting deformation parameters move).

## 2. Concrete numbers

**Table 1 — ShapeNet, complete clouds, 13 categories, canonical pose, N=4096 pts (FPS)**

| Method | IoU↑ | F-score↑ | L1↓(×10²) | L2↓(×10²) | #Prim↓ | Runtime |
|---|---|---|---|---|---|---|
| SQ | 0.27 | 0.13 | 3.67 | 0.29 | 10 | 0.0065s |
| CSA | 0.42 | 0.14 | 3.52 | 0.30 | 7.85 | 0.004s |
| EMS | 0.40 | 0.29 | 4.40 | 1.06 | 5.67 | 7.82s |
| Marching Primitives | 0.56 | 0.18 | 2.08 | 0.10 | 24.10 | 163.41s |
| SuperDec | 0.59 | 0.28 | 1.77 | 0.050 | 5.79 | 0.0075s |
| SuperFlex (rigid, no deform) | 0.70 | 0.35 | 1.75 | 0.049 | 6.04 | 0.0075s |
| **SuperFlex (full)** | **0.72** | **0.37** | **1.54** | **0.043** | 5.64 | 0.0082s |

**Table 2 — test-time optimization on top of feed-forward (complete clouds)**: IoU 0.72→**0.87**, F-score 0.37→**0.49**, runtime 0.0082s → **55s**. This is the *best case* the paper reports anywhere — full ShapeNet supervision, complete cloud, canonical pose, 4096 pts, ~1 minute of optimization per object.

**Table 3 — partial clouds, ASE dataset (synthetic occlusion, see §3)**

| Method | Occlusion aug. | ASE fine-tune | IoU↑ | F-score↑ | L1↓ | L2↓ | #Prim |
|---|---|---|---|---|---|---|---|
| SuperDec | ✗ | ✗ | 0.20 | 0.14 | 5.98 | 2.45 | 6.0 |
| SuperFlex (standard) | ✗ | ✗ | 0.20 | 0.16 | 5.85 | 2.09 | 6.3 |
| SuperFlex (robust) | ✓ | ✗ | 0.48 | 0.16 | 4.49 | 1.62 | 4.9 |
| **SuperFlex (robust)** | ✓ | ✓ | **0.54** | **0.18** | 3.98 | 1.39 | 4.3 |

Read this row-by-row: a model trained only on complete clouds (SuperDec, SuperFlex-standard) **collapses to IoU≈0.20 on partial input** — a 3.5× drop from the complete-cloud number. Occlusion-augmented + fine-tuned training claws most of that back but still only reaches IoU 0.54 / **F-score 0.18** — i.e. even the best-case partial-cloud result is *worse on F-score* than the worst complete-cloud baseline in Table 1 (SQ's F=0.13 aside, everything else in Table 1 beats 0.18). F-score (tight surface-distance threshold) staying under 0.2 on partial data means the fitted surfaces are geometrically coarse even after the robustness pipeline.

**Table 4 — loss ablation (complete clouds)**: Chamfer-only (SuperDec's loss) gives the *lowest* overlap (4%) but worst accuracy (IoU 0.59). IoU-loss-only pushes accuracy up (IoU 0.74) but overlap balloons to 32% (primitives redundantly cover the same volume). IoU+SDF+Overlap-penalty lands at IoU 0.72, overlap 8% — the accuracy/tidiness trade-off is explicit and quantified. (Relevant aside for our failure-detection branch question: **overlap % between fitted primitives is a directly obtainable, un-trained signal of decomposition quality** — the paper uses it as a loss term, not a runtime confidence score, but the mechanism transfers.)

**Table 5 — deformation ablation (after TTO)**: no-deform 0.81 IoU → +tapering 0.83 → +bending 0.84 → both 0.85. Bending contributes slightly more than tapering; gains are modest (2-4 points of IoU) relative to the loss-function change (which alone moved IoU 0.59→0.72, i.e., 13 points).

**Appendix Table 9 — "normal completeness"**: for each GT point, find nearest point on the reconstructed superquadric surface, take `|dot(n_pred, n_gt)|`. Values across SDF-weight ablation: ω=1→0.83, ω=0.5→0.84, ω=0.1→0.85, ω=0.01→0.85 (best-case, complete ShapeNet clouds). **This is the only normal-accuracy number in the paper**, and it is a global, aggregate, absolute-dot-product statistic, not a per-primitive angular error distribution. Naively converting the best value (0.85) via `arccos` gives ≈32° — but this is not a clean per-axis angular error (it mixes well-fit and poorly-fit surface regions, and `|dot|` folds sign ambiguity in) so treat as a rough upper-bound-on-quality proxy, not a number to budget against directly. **No normal-completeness number is reported for the partial/occluded setting at all** — the metric only appears in the complete-cloud SDF-weight ablation.

## 3. What "partial real-world" actually means here — read carefully

This is the load-bearing point for our branch. The paper's own "robustness to real-world" claim (in the abstract and §4.3/§5.3) is entirely a **visibility/occlusion story**, not a sensor-defect story:

- ASE (Aria Synthetic Environments) is a **synthetic** multi-room dataset populated with ABO CAD furniture models — not real depth-sensor captures.
- The only two corruption mechanisms applied are (1) Hidden Point Removal (removes points not visible from a synthetic camera viewpoint — a pure line-of-sight occlusion operator) and (2) random spherical masking (removes a random ball of points, standing in for foreground-object occlusion). No parameter values (HPR radius, sphere size distribution) are given in the fetched text.
- **No Gaussian/range-dependent noise is injected. No registration/pose jitter across multiple synthetic "views" is simulated. No glossy/dark dropout model is used. No depth bias model is used.** The three defect modes our framing doc names (registration error, specular/dark dropout, depth bias) are simply outside what this paper tests.
- ScanNet++ and Replica (real/real-rendered indoor scans) appear **only qualitatively** (Fig. 7, Appendix B.1) using ground-truth instance masks to isolate objects — no quantitative metric is reported against them, presumably because no GT superquadric decomposition exists for real scans to compare against. So the "SuperDec on ScanNet++, SuperFlex on partial real data" framing in the branch's selection rationale is optimistic: SuperFlex's *quantitative* partial-cloud numbers are all on synthetic ASE occlusion, not on real sensor data; the real-scan (ScanNet++) results are unquantified qualitative figures only.

This directly weakens the paper's usefulness for branch questions about depth-dropout bias, registration-covariance propagation, and range/incidence-dependent noise (Branch Qs on M3C2-EP, depth-error-vs-incidence-angle, dropout-induced systematic bias) — **this paper provides no evidence on any of those**, only on missing-point occlusion.

## 4. Load-bearing assumptions that don't hold in our setting

- **Canonical pose at training and evaluation** — §5.1 states explicitly: *"All objects are pre-aligned to a canonical orientation."* All ShapeNet numbers (Table 1, 2, 4, 5) are on canonically-posed objects. There is no stated rotation-invariance/equivariance mechanism in the architecture (PVCNN + transformer with absolute-coordinate point features is not rotation-equivariant by construction), and **no experiment evaluates arbitrary object pose** — this is a direct violation of our tabletop setting where the object sits at an arbitrary yaw/orientation on the table. The ASE partial-cloud experiments don't clarify whether ASE objects are also canonicalized before being fed to the network; the paper doesn't say, which itself is a gap.
- **Fixed downsampling to N=4096 points via FPS** for every experiment — the paper never tests or discusses behavior at the point counts a fused multi-view survey cloud would produce (tens of thousands to hundreds of thousands of points before any subsampling). Whether FPS-to-4096 preserves the fine edges/creases that later matter for face-boundary or chamfer/fillet detection is untested.
- **Fixed, small primitive budget** (P=16 slots, converging to ~4-6 active primitives per object on furniture-scale ShapeNet categories) — untested on industrial/mechanical objects with many small features.
- **Ground truth for training/eval is CAD-model occupancy**, not human-annotated real-scan segmentation — so its accuracy numbers assume a GT that we would not have for our own real scanned objects (no CAD models mentioned for our test objects in the framing doc).

## 5. What transfers to our setting

- **Bounded-volume primitives with explicit pose+extent** (superquadric params t, R, s, ε) are exactly the representation type our "scan this face orthogonally" action needs a level up from — a superquadric gives a primitive centroid and axis-aligned half-extents directly, which is a strictly richer geometric object than an infinite RANSAC plane/cylinder equation. (Still not a *face*: see §6.)
- **Loss-design lesson**: pure Chamfer distance (SuperDec) under-covers/gaps and pure volumetric IoU over-covers/overlaps; a combined surface (SDF) + volumetric (IoU) + explicit overlap penalty is what got both accuracy and primitive-set tidiness. If we ever move toward a learned or hybrid abstraction step (not RANSAC-family), this is a directly reusable objective design, and the reported overlap-% is a candidate confidence signal (obtainable without retraining anything — it's just intersection volume between fitted primitives) worth prototyping for our failure-detection branch question.
- **Existence probability** (`e_p`) is a cheap, per-primitive scalar the network already produces — a template for "does this primitive actually exist" scoring, though it is trained as a binary indicator of "was this slot used," not calibrated against any downstream task-failure signal (see §6 — branch question about calibrating a score against task failure is unanswered here too).
- **Deformation as an explicit tell for when rigid convex primitives are insufficient**: the fact that bending/tapering only buys 2-4 IoU points over rigid superquadrics on ShapeNet furniture (Table 5) is a soft signal that most everyday objects are *close to* rigid-convex-decomposable, and the marginal gain from non-convex deformation is small — useful context (not proof) for our own question about whether plane+cylinder+sphere+cone RANSAC "closes" our object set: this paper's own ablation suggests most of the reconstruction quality comes from getting the *loss function* and *primitive placement* right, not from needing non-convex shape families.
- Code + weights are released (github.com/GabrielTavernini/superflex) — usable off-the-shelf if we want to benchmark it directly on our own PLYs rather than only reading numbers.

## 6. What does NOT transfer / negative findings (first-class per instructions)

- **No per-primitive axis/normal angular-error distribution reported anywhere.** Table 9's "normal completeness" is a single aggregate absolute-dot-product number (0.83–0.85, complete clouds only), not decomposable into "expected angular error on this face's normal." **Branch Q on angular-error-vs-noise/incompleteness is not answered by this paper** — no noise sweep exists at all (confirmed by the explicit fetch: *"No explicit Gaussian noise/dropout experiments on point clouds reported in main paper"*).
- **No primitive-type confusion analysis.** The paper never reports "at X% incompleteness the fitted primitive stops being the correct type" — it only reports IoU/F-score degradation in aggregate (Table 3). Branch Q1's second half ("at what noise/incompleteness level do methods stop returning the correct primitive type") is unanswered; would require a bespoke experiment.
- **No uncertainty quantification / covariance on fitted pose.** `e_p` is an existence probability, not a pose or extent uncertainty. No covariance propagation from input noise or registration error into primitive orientation is modeled or discussed — branch questions on closed-form covariance and M3C2-EP-style propagation get **zero support** from this paper.
- **No RANSAC/region-growing comparison or even discussion** — §2 Related Work covers only optimization-based (EMS, Marching Primitives, Light-SQ) and learning-based (SQ, CSA, SuperDec) superquadric/cuboid methods. Classical plane/cylinder/sphere/cone RANSAC is not cited, benchmarked, or discussed at all, so this paper cannot answer whether our RANSAC-family vocabulary "closes" against superquadric-family alternatives — it simply operates in a different, adjacent literature.
- **No face/boundary extraction.** The representation is a whole bounded solid (box-like, cylinder-like, or ellipsoid-like depending on ε), not a decomposition into individual bounded planar/curved *faces* with their own centroid+extent+boundary. Getting a "face 3 centroid + size" out of a SuperFlex primitive would require an extra, unpublished step (e.g., slicing the superquadric's 6 canonical faces and testing which ones the input points actually cover) — branch Q on "which methods recover bounded faces" is only partially served: SuperFlex gives bounded *primitives*, not bounded *faces*.
- **No identity/order consistency guarantee across re-scans.** Slot assignment is an unordered set (P=16 tokens) matched via Hungarian algorithm *during training against pseudo-GT*; at inference on two different point clouds of the same physical object (e.g., two re-scans), nothing in the architecture or loss enforces that "primitive index 3" corresponds to the same physical face/part both times — Hungarian matching is a training-time device, not an inference-time correspondence guarantee. This is a direct negative for branch Q3 (RQ3 consistency: "does face 3 stay face 3 in identity, ordering and orientation") — the paper gives us reason to expect it does **not**, without an explicit correspondence/tracking step bolted on.
- **No registration-error, depth-bias, or glossy/dark-dropout modeling** (detailed in §3 above) — the "robust to real-world" claim is narrower than the branch's selection rationale implied; it is robust to *missing points from occlusion*, not to the *specific defect triad* (registration jitter, dropout, depth bias) our framing doc names.
- **Runtime**: feed-forward is fast (8ms) but low quality (0.72 IoU complete / 0.20-0.48 IoU partial without fine-tuning); the actually-good numbers (0.87 IoU complete, 0.54 IoU partial) require the 55s-per-object test-time optimization. For our offline per-object survey pipeline this cost is probably acceptable (one-time cost per object before inspection starts), but it means "real-time primitive extraction from a fresh scan" is not what this method offers — consistent with our own pipeline shape (survey once, then plan), so this is a mild positive, not a blocker, just worth noting as a design constraint if we ever adopt it.
- **13 ShapeNet object categories are household/furniture-scale** (chairs, tables, etc., inferred from ABO/ShapeNet context) — no evaluation on tabletop industrial/mechanical parts with tight chamfers/fillets of the kind the branch's own selection rationale hoped this paper would address ("real chamfered/rounded parts force superquadrics or cuboid decomposition"). The superquadric shape-exponent (ε1, ε2) does *already* interpolate continuously between box-like and rounded/ellipsoid shapes without needing the bending/tapering deformation — so chamfered edges are partially covered by the base superquadric family itself, but this was never tested against our object class.

## 7. Mapping to branch questions (explicit)

- **Angular error vs. noise/incompleteness (Q1)**: not answered quantitatively; only a single aggregate normal-completeness number on complete clouds (0.83-0.85 dot product), nothing on our defect profile. **No.**
- **Primitive vocabulary closure — RANSAC vs. superquadric/cuboid (Q2)**: paper doesn't engage with RANSAC at all; offers only its own deformation ablation as indirect evidence that rigid convex primitives already capture most reconstruction quality on furniture-scale objects. **Weak, indirect.**
- **Bounded faces vs. infinite primitives (Q3-face)**: superquadrics are bounded *primitives* with centroid+extent, but not decomposed into individual bounded *faces*. **Partial.**
- **Decompose-then-view-plan precedent (Q4-precedent)**: no mention of robotics, view planning, or downstream viewpoint generation anywhere in the paper or project page. **No — not a precedent, only a possible geometric building block.**
- **Order/identity stability across re-scans (Q-RQ3-consistency)**: architecture uses an unordered slot-based set prediction with training-time Hungarian matching; no inference-time correspondence mechanism — actively suggests instability, not stability. **Negative evidence.**
- **Partial cylinder / symmetry completion (Q-cylinder)**: not tested; ASE occlusion experiments report only aggregate object-level IoU/F-score, not per-primitive-type (e.g., cylinder axis) recovery under partial view. **No.**
- **Noise-sigma vs. k vs. normal-angular-error curve (Q-normal-estimator)**: absent; this paper does no normal-estimator-family comparison at all — it's not that kind of paper (no PCA/jet/VCM/Hough/learned-normal comparison). **No.**
- **Closed-form covariance for fitted primitive orientation (Q-covariance)**: absent. **No.**
- **Registration covariance propagation, M3C2-EP transfer (Q-M3C2)**: absent — paper's occlusion model has no registration-jitter component at all. **No.**
- **Depth dropout mechanism/bias on glossy surfaces (Q-dropout-bias)**: absent — occlusion model is purely visibility-based (HPR + spherical masking), not sensor-photometric. **No.**
- **Preprocessing (voxel/outlier-removal) effects on fitting (Q-preprocessing)**: not discussed; input is always FPS-subsampled to 4096 points, no ablation on denoising/outlier-removal choices. **No.**
- **Computable unreliability signals (inlier ratio, MDL, etc.) (Q-confidence-signal)**: existence probability `e_p` and primitive-overlap % (from the loss ablation, Table 4) are the two candidate signals available off-the-shelf from this codebase; neither is validated against downstream task failure. **Partial — candidate signals only, uncalibrated.**
- **Calibrating a fit-score against downstream failure (Q-calibration)**: not attempted. **No — open problem, as the branch assumed.**
- **Two-tier design / geometric-vs-learned failure predictor (Q-two-tier)**: not discussed; no fallback or failure-detection philosophy in the paper. **No.**
- **Open3D+CGAL/PCL+abstraction-method coverage & runtime (Q-toolchain)**: not applicable — this is a from-scratch trained network, not a toolchain composition; only informs "what a learned alternative would cost" (8ms feed-forward / 55s with TTO, at 4096 pts, GPU-bound — no CPU-only or PLY-scale numbers given).
- **Released weights & real-scan generalization vs. canonical-pose assumption (Q-weights-canonical)**: weights are released (GitHub), and qualitative ScanNet++/Replica figures exist, but **quantitatively validated only on canonically-posed ShapeNet + synthetic ASE occlusion** — the paper itself confirms the branch's suspicion that this class of method assumes canonical orientation, a **direct constraint violation** for our arbitrarily-posed tabletop object. **Confirmed negative.**
- **Datasets/protocols to seed our benchmark (Q-benchmark-seed)**: ShapeNet+ABO / ASE occlusion protocol (HPR + spherical masking) is a candidate *recipe* for building a partial-cloud benchmark, but it is a visibility-only corruption model — would need to be extended with our specific defect profile (registration jitter, glossy dropout, depth bias) to be useful; the paper doesn't validate against Fit4CAD, SHREC 2022, or ABC. **Partial — protocol idea, not directly reusable data.**
- **Ground-truth story for "correct decomposition" of a real object (Q-groundtruth)**: this paper sidesteps the problem exactly the way we'd have to — it never evaluates real scans quantitatively because it has no GT decomposition for them; qualitative-only figures on ScanNet++/Replica. This is itself informative: **even a well-resourced 2026 paper in this space has no real-scan quantitative GT story**, reinforcing that we will likely need human-annotated segmentation or synthetic-with-known-GT benchmarking rather than expecting a public real-object GT decomposition to exist.

## 8. Open items if we want more from this paper

- Appendix D ("3D Scene Decomposition from RGB", referencing SAM 3D) was not retrievable in full from the fetch — worth a direct PDF read if we ever care about RGB-only (no-depth) primitive extraction.
- HPR radius / spherical-mask-size parameters weren't retrievable — would need the PDF or supplementary material for exact corruption severities if we wanted to compare against our own defect magnitudes.
- Could run the released code (github.com/GabrielTavernini/superflex) directly on one of our fused PLYs as a quick empirical check rather than relying only on paper-reported numbers — closest ISO the paper trained on is furniture-scale ABO objects, so expect a moderate domain gap for our tabletop objects.
