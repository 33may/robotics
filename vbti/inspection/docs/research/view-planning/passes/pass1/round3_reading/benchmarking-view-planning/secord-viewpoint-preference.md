# Secord, Lu, Finkelstein, Singh, Nealen — "Perceptual Models of Viewpoint Preference" (ACM TOG 30(5) art. 109, 2011)

**Source read:** full PDF, `https://gfx.cs.princeton.edu/pubs/Secord_2011_PMO/secord2011.pdf` (author's preprint copy; ACM DOI `10.1145/2019627.2019628` returned 403/paywalled, so citations below use the preprint's own pagination: pp. 1–13, its internal Section/Table/Figure numbers, which should match the published version).

**Branch:** Benchmarks and evaluation methodology for view planning — read for the human-study design template (pairwise comparison → attribute regression → attribute correlation clustering) and the "surface visibility dominates" finding.

---

## 1. What the paper is

A whole-*object* viewpoint-goodness model: given a 3D mesh, predict which of two nearby camera viewpoints a human would judge to "reveal its shape better." Built from 14 hand-picked geometric/semantic attributes (§3.1, Fig. 1), fit to a large (127K-choice) crowdsourced pairwise-preference dataset (§3.2) via a Bradley-Terry paired-comparison model (§3.3), then validated on 4 held-out models with an independent AMT study (§5). Applications: N-best-view selection, smooth orbit paths, and force-guided trackball navigation (§4) — none of these are relevant to our benchmark question, but the measurement methodology is.

This is a **shape-legibility / aesthetic viewpoint-selection** paper, not a metrology/inspection paper. No robot, no camera reachability, no defects, no surface regions, no primitives-with-known-normals. Read it purely as a methodology template.

## 2. Attribute taxonomy (§3.1, Fig. 1) — 14 attributes, 5 groups

| Group | Attributes |
|---|---|
| Area (§3.1.1) | a1 projected area; a2 **surface visibility** (visible surface area / total surface area, Plemenos & Benayada 1996); a3 viewpoint entropy (Vázquez et al. 2001) |
| Silhouette (§3.1.2) | a4 silhouette length; a5 silhouette curvature (∫\|κ(ℓ)\|dℓ, Vieira et al. 2009); a6 silhouette curvature extrema (∫κ(ℓ)²dℓ) |
| Depth (§3.1.3) | a7 max depth; a8 depth distribution (1 − ∫H(z)²dz, entropy-like spread of depth histogram) |
| Surface curvature (§3.1.4) | a9 mean curvature; a10 Gaussian curvature; a11 mesh saliency (Lee et al. 2005) |
| Semantic (§3.1.5) | a12 "above preference" — a Gaussian bump in latitude φ, `G(φ; 3π/8, π/4)`, peaked 3π/8 above the equator, encoding the Blanz et al. 1999 finding that people prefer slightly-elevated views; a13 eyes (hand-marked, summed visibility with cosine falloff); a14 base (hand-marked, "is the object's supporting base visible") |

All computed from an **image-based pipeline**: render color/depth/ID buffers per view, no dependency on defect-free mesh *per se*, but every attribute assumes a clean, complete, single-object mesh — no notion of a fused/noisy point cloud (§3.1.6). Table I: precompute cost per view for the 14 attributes, 194 ms (11K-face model) to 1256 ms (526K-face Lucy), single-threaded 2.26 GHz.

Attribute a12 ("above preference") is the closest thing in the paper to what we need for Q7: it is an **analytically-defined angular Gaussian centered on an ideal viewing direction**, with a known peak/ceiling and falloff — structurally identical to what an "orthogonality score" would need to be (peak at the surface normal, falloff with angular deviation), just aimed at a global "slightly above equator" prior rather than a per-surface normal.

## 3. Human study design (§3.2) — the template to copy

- **Task framing (2AFC):** "Which of the two views of the object shown below reveals its shape better? ... Just click on the one that you think is better." — deliberately generic, non-expert-friendly language (inspired by Blanz et al. 1999), not tied to any specific attribute.
- **Models:** 16 total — 8 "familiar" (armadillo, dragon, etc.) + 8 from separate Princeton Shape Benchmark categories, chosen to be recognizable even to non-experts (upper 4 rows of Fig. 6). Rationale given: recognizable shapes → less noisy, more meaningful preference judgments than abstract "blobbies."
- **Pair selection is *not* random-uniform on the sphere.** Rejection sampling used three criteria (§3.2, p.5):
  1. angular separation fixed near **π/8 radians (~22.5°)** — far enough that the difference is obvious, close enough it doesn't feel like an "apples-to-oranges" comparison;
  2. reject pairs where the view has rotated substantially in the longitudinal direction near the poles (avoids near-identical-content pairs from up-vector wraparound);
  3. reject pairs with low attribute variance — bias sampling toward pairs where the sum of (per-attribute, std-normalized) absolute differences is large, so every pair has *some* discriminative signal.
- **Scale:** 120 pairs/model × 16 models = 1920 pairs, 3840 images total (Fig. 2 shows per-object pair distributions in θ×φ space, colored by dominant differing attribute).
- **Platform & QC:** Amazon Mechanical Turk. Each HIT = 30 pairs shown once each, **then the same 30 pairs shown again, shuffled** — workers kept only if ≥22/30 (p>0.99 under random-guessing null) repeat-answered consistently. Result: 2,119 HIT assignments retained, **127,140 total pairwise choices**, from 524 unique workers (power-law activity distribution, most active worker did 51 HITs).
- **Up-vector fixed** during rendering so nearby views share orientation (removes rotation as a confound).

**Modeling pipeline (§3.3):** Bradley-Terry model, `P(G0,G1) = 1/(1+exp(-(G0-G1)))`, log-likelihood fit per candidate goodness function G. Two reference points define a **normalized fitness scale F[G] ∈ [0,1]**:
- **Naive** (random-guess, P=1/2): F=0 by construction, `L*_naive = -33203`.
- **Oracle** (uses each pair's *empirical* human selection frequency k/n as its predicted probability — the best any model *could* do given the noise in repeated human judgment): F=1, `L*_oracle = -3688`.

This naive-floor/oracle-ceiling normalization, with the oracle defined from **empirical human agreement rate** (not "1.0 = perfect"), is the single most transferable idea in the paper for Q5/Q12 — it gives every candidate metric a dataset-specific, noise-aware ceiling instead of comparing raw accuracy numbers.

## 4. Results — the "surface visibility dominates" finding

**Single-attribute models (Table II, §3.3.4)**, cross-validated 100× via random half/half object-level train/test split (mean F, σ(F)):

| Attribute | F | σ(F) |
|---|---|---|
| Oracle | 1 | 0 |
| **Surface visibility (a2)** | **0.38** | 0.072 |
| Viewpoint entropy (a3) | 0.37 | 0.092 |
| Projected area (a1) | 0.28 | 0.110 |
| Mesh saliency (a11) | 0.26 | 0.100 |
| Silhouette length (a4) | 0.20 | 0.120 |
| Above (a12) | 0.16 | 0.064 |
| Base (a14) | 0.13 | 0.087 |
| Eyes (a13) | 0.09 | 0.033 |
| Silhouette curvature (a5) | 0.01 | 0.040 |
| Sil. curv. extrema, depth distribution, max depth, abs mean/Gaussian curvature | ≈0 or slightly negative | — (statistically indistinguishable from 0 at p=0.05) |
| Naive | 0 | 0 |

Surface visibility alone (a single ratio: visible-surface-area / total-surface-area) beats every other individual attribute, including the more elaborate/expensive mesh saliency. Silhouette curvature — heavily used in prior literature — is **not statistically distinguishable from useless** on its own.

**Linear-K combined models (§3.3.5):** all 16,383 possible linear combinations of the 14 attributes tested (100 trials each); Tukey HSD used to find the *pool* of models statistically indistinguishable from the best. Key negative finding stated explicitly: **"using more than five attributes does not improve performance of the linear models."** Recommended models (Table III weights):
- linear-3 {a2, a12, a4}: F=0.55
- linear-5 {a2, a12, a4, a1, a7}: F=0.58
- linear-5b {a2, a12, a4, a1, a13}: F=0.58 (swap in "eyes" when hand-marked features available)

**Quadratic model (§3.3.6):** adds all pairwise attribute-product cross-terms (14+105=119 weights) → F=0.63, "a moderate improvement." Authors explicitly raise and reject the idea that this closes the gap to the oracle.

**Non-parametric K-nearest-neighbors (§3.3.7):** F=0.77 at K=10 (Table IV) — best of all fitted models, but (a) expensive (must compute all 14 attributes for both query views and find nearest neighbors against the training set of 960 pairs at inference time) and (b) only gives *pairwise* preference, not a per-view absolute goodness score — ruled out for the paper's downstream applications for that reason.

**Table IV full ranking:** Oracle 1.00 > 10-NN 0.77 > Quadratic 0.63 > Linear-5b/Linear-5 0.58 > Linear-3 0.55 > Single 0.38 > Naive 0.

**Attribute correlation structure (Fig. 3, correlation matrix):** three strong clusters — (1) area attributes (projected area, surface visibility, viewpoint entropy) highly inter-correlated; (2) the two silhouette-curvature attributes; (3) the two surface-curvature attributes (mean, Gaussian). Authors' interpretation: since silhouette curvature is "surprisingly unimportant after model fitting" despite prior literature, and surface visibility is "by far the most influential metric," this **"points to significant redundancies in any combined measure of viewpoint goodness."** They also report discarding a promising-looking attribute ("number of disconnected silhouette loops") after finding it was just a proxy for silhouette length — an explicit warning that attribute redundancy inflates apparent metric variety without adding real signal, and they call for "a more robust methodology for investigating the correlations between various attributes" (§6) — i.e., they flag this as unsolved.

## 5. Validation study (§5) — different-provenance check on new models

Distinct from the cross-validated training procedure: linear-5b (fit entirely on the 16 training models) evaluated against **4 unseen models** (bunny, David, Buddha, city) using a **new, independent AMT study with a different procedure**:
- 100 random single viewpoints per model (rejection-sampled individually, not as pairs).
- Fresh 2AFC study: each HIT presents 30 *random* pairs (not fixed-separation pairs), same repeat-and-filter consistency QC as before → 1075 HITs retained, 233 subjects, 400 viewpoints shown 234–370 times each.
- "True goodness" per viewpoint = empirical fraction of times it was picked when pitted against a random other view of the same object (an approximation of the oracle, now on new data).
- Logistic regression of linear-5b's predicted goodness against this empirical true-goodness (Table V): coefficients all highly significant (t = 45.0–70.6, p < 10⁻⁶⁵ for all 4 models, df=98).
- **Weakest fit was the city model** (t=45.0, still significant but visibly lower correlation in Fig. 10 than bunny). Authors attribute this to the city model's qualitative difference from the training set — "boxiness," planarity, multiple disconnected components — i.e., **a model fit on organic/statuary shapes transfers measurably worse to a geometrically different (box-like, multi-part) object class.** This is indirect evidence (from a domain the paper wasn't built for) that a universal viewpoint-goodness model degrades on box-like objects — mildly supportive of our project's choice to use *per-primitive* analytical rules (box vs. cylinder) rather than one blended geometric score.

## 6. Author-stated limitations (§6) — first-class negative findings

- Quantitative claims are scoped to the 16 (+4 validation) models actually studied; explicitly flagged as not covering "certainly more classes of models," and the optimization in §3.3 only searches 2 camera parameters (view direction on a sphere) — no field-of-view, up-vector, velocity, or free 3D position in cluttered scenes.
- Many candidate attributes from the literature were *not* included at all — notably **object symmetry (Podolak et al. 2006)**, lighting variation, occluding contours, texture. Authors call this future work. → **The paper never engages our Q8 (view equivalence under symmetry) at all** — it's an acknowledged gap in the source itself, not something to borrow a solution from here.
- The fitted model is explicitly called "error-prone," with named failure modes (Fig. 4/5): a pair predicted <20% preference for the left image was actually chosen >80% of the time (a full anti-correlation, not just noise); a pair predicted near 50/50 was picked almost unanimously for one image because it made a child's head more visible in the group-statue model — "a quality that our attributes do not capture." This is a concrete instance of the attribute set being incomplete for face/occlusion-sensitive content, i.e., a purely geometric attribute basis has a real, demonstrated ceiling below the oracle (F=0.58–0.77 vs 1.0).
- Authors' own closing suggestion: "special-case combinations of measures will create better combined goodness measures for classes of models" — i.e., they themselves conclude that a single universal geometric model is not going to close the remaining gap, and that class-specific (their word) models are the way forward. This resonates with our project's constraint that primitive-level deterministic rules (box face, cylinder orbit) are acceptable rather than a single general blended-attribute score.

## 7. Relevance to our branch questions

**Transfers (methodology, not domain):**
- **Q3 (circularity):** the paper's two-layer defense against judge/GT contamination is a direct template — (1) cross-validated *object-level* held-out testing during model fitting (train attributes' weights on 8 objects, test on the other 8, repeated 100×, §3.3.4–5), and (2) a fully independent validation study on **4 objects never used in training, with a freshly collected human-preference dataset and a different sampling procedure** (§5). That second step is a "different provenance" separation in the terms of Q3: not just a held-out split of the same collection run, but a new study. Directly informs how we'd defend a benchmark against the RQ1 judge being the same construct as the GT.
- **Q5/Q12 (baselines & statistics):** the **naive-floor / oracle-ceiling normalized fitness scale F[G]** is a strong, transferable pattern for reporting any of our candidate view-planning metrics against a random baseline and an empirical upper bound, rather than raw scores. Their **repeated random sub-sampling (100 trials of half/half object splits) + Tukey HSD "indistinguishable pool"** is a concrete, citable statistical procedure for exactly our Q12 problem (few objects, noisy comparisons): report mean ± σ over many resamples, and identify a *pool* of statistically tied top performers instead of a single point estimate.
- **Q7 (metric family / analytical ceiling):** attribute a12 ("above preference," a Gaussian in latitude peaked at an ideal angle) is a working example of exactly the kind of metric Q7 wants — an analytically defined, angle-based score with a known per-object ceiling. Swap "peak at 3π/8 above equator" for "peak at the surface normal direction" and this becomes a template for an orthogonality-score. The paper's strongest empirical result — **surface visibility (a2) alone is the single best predictor (F=0.38) and dominates any combined model** — is a real, checkable argument for keeping our view-quality score simple (one dominant geometric term) rather than stacking many correlated attributes; their explicit finding that >5 attributes stops helping is a caution against over-engineering our own metric family.
- **Q17 (judge controls, human-labelling QC):** their AMT consistency filter (show every pair twice, keep only workers with ≥22/30 repeat agreement, p>0.99) is a reusable, concrete QC recipe if we ever collect our own human "which view is better evidence" labels for a held-out slice.

**Does NOT transfer (explicitly, with reasons):**
- **Construct mismatch with our whole RQ1/Q2/Q7 target.** This paper's "best view" is *whole-object shape legibility for a human viewer*, not *orthogonality of a view to one specific inspected surface region*. There is no per-surface-region unit of analysis anywhere in the paper (Q1) — everything is a whole-object 2AFC judgment. Its methodology (pairwise + regression + correlation clustering) transfers; its target construct does not.
- **No defect robustness whatsoever (Q9).** Every attribute is computed from a clean, complete, single-material triangle mesh rendered offline (color/depth/ID buffers, §3.1.6). No pose error, no dropout, no fused-vs-live offset is modeled or discussed anywhere. This paper gives zero evidence about metric robustness under our survey-cloud defects — treat as an open question, not something answered here.
- **No mesh-free / point-cloud-only path (Q4).** The attribute pipeline structurally requires a clean mesh with well-defined normals, curvature, and silhouette — it does not degrade gracefully to a raw defective point cloud; several attributes (mean/Gaussian curvature, mesh saliency) are second-derivative quantities that would be materially noisier on a real fused cloud with mm-level pose error. Not a source for "how to score without ground-truth mesh."
- **No primitive-shape structure, no symmetry handling (Q8).** Models are organic/statuary or generic scanned shapes; nothing about boxes/cylinders/faces or view-equivalence under symmetry. Authors explicitly list object symmetry (Podolak et al. 2006) as an attribute they *didn't* include and left to future work — this is a genuine absence in the source, not a place to mine a symmetry-handling recipe from.
- **No robot/reachability constraints, no VLM, no task-outcome coupling (Q6, Q14–Q16, Q19).** Purely a virtual-camera, human-preference study; there's no orchestrator, no downstream inspection task, no stopping criterion, and no way to map its findings onto grounded-success vs. lucky-answer questions.
- **Weak transfer signal on box-like objects (Q11):** the one piece of evidence that touches our primitive/box concern is indirect — the city model (their most "boxy," multi-part, planar validation object) fit the worst of the 4 held-out models (t=45.0 vs. 57–71 for organic models), suggesting a universal geometric-attribute model degrades on box-like structure. This is weak, incidental evidence (not a designed test) but consistent with our project's decision to use primitive-specific deterministic rules rather than one blended score.

## 8. Bottom line

Read this paper for **method, not content**: it is the best available template for (a) a naive/oracle-normalized fitness scale, (b) repeated-subsampling + HSD statistics for small-N noisy human comparisons, (c) a two-tier train/validate split that defends against judge–GT contamination, and (d) a worked example of an analytically-defined, ceiling-having angular preference score (a12) that Q7 could imitate for orthogonality. Its actual subject — whole-object aesthetic viewpoint preference from a clean mesh — does not touch surface regions, primitives, defects, robots, or VLMs, and the authors themselves flag symmetry and class-specific modeling as open gaps.
