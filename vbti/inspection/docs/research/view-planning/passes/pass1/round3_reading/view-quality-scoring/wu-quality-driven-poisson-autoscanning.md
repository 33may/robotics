# Wu et al. 2014 — Quality-driven Poisson-guided Autoscanning (ACM TOG / SIGGRAPH Asia 33(6):203)

**Full text read**: yes, full 12-page PDF fetched and read in full (all sections, all equations, all figures/tables transcribed below).
Source: https://pub-085d059a8619439494873b401c0986ce.r2.dev/Wu2014QualitydrivenPoisson/Wu2014QualitydrivenPoisson.pdf (via https://graphics.uni-konstanz.de/publikationen/Wu2014QualitydrivenPoisson/index.html). DOI 10.1145/2661229.2661242.
Authors: Shihao Wu, Wei Sun, Pinxin Long, Hui Huang (SIAT), Daniel Cohen-Or (Tel-Aviv), Minglun Gong (Memorial U. Newfoundland), Oliver Deussen (Konstanz), Baoquan Chen (Shandong).

---

## 1. Core method / idea

Autonomous next-best-view (NBV) scanning system whose stated objective is explicitly **not** "cover the surface with the fewest scans" but **"ensure high fidelity scanning of the model"** (Abstract, §1, §3). Pipeline (Fig. 2, §3):

1. Blind all-around initial scan → incomplete raw point cloud `Q` (§4, Fig. 2b).
2. **Point consolidation**: WLOP operator (Huang et al. 2009) → cleaned, oriented point set `P = {(p_i, n_i)}` (§4.1).
3. **Screened Poisson surface reconstruction** (Kazhdan & Hoppe 2013) on `P` → watertight iso-surface (§4.1, Fig. 3). Note: Poisson interprets an **inward-pointing** normal field as the gradient of the solid indicator function — this sign convention matters for reading Eq. 4 below.
4. Discretize the iso-surface via Poisson-disk sampling (Corsini et al. 2012) into oriented **iso-points** `S = {(s_k, n_k)}` (§4.2).
5. Compute a per-iso-point **confidence score** `f(s_k)` = completeness × smoothness (Eqs. 1–3, §4.2).
6. Build a 3D **viewing vector field (VVF)** over a voxelized bounding region: at each empty voxel, a scalar NBV utility `g(v_i)` and a preferred viewing direction (Eq. 4, §5.1).
7. Select a sparse, mutually non-adjacent set of NBV **locations** by local-maxima suppression on the VVF (§5.2, Fig. 6b–c), then refine each location's **direction** by a local neighborhood search that balances "cover unconfident points" against "keep enough confident points visible for registration" (Eq. 5, §5.2).
8. Scan from the NBVs, register with confidence-weighted ICP (Masuda 2002), and selectively re-insert new points with probability weighted toward low-confidence regions (Eq. 6, §5.3). Iterate until global confidence plateaus (Δ < 0.005 default) or scanning is exhausted.

This is a **coverage/reconstruction-quality** NBV planner, not a defect-inspection or image-space quality judge. The "quality" being optimized is *fidelity of the reconstructed 3D shape*, measured entirely on the (reconstructed-mesh) geometry side — never on the captured 2D image.

---

## 2. Key formulas (verbatim, citable)

All from §4.2–§5.2.

**Completeness confidence** (Eq. 1):
`f_g(s_k, n_k) = Γ(s_k) · n_k`
where `Γ(s_k)` is the gradient of the Poisson scalar field at `s_k` (already computed as a byproduct of Poisson reconstruction — no extra cost). High value ⇒ the zero-crossing is sharply/stably located ⇒ reliable iso-point.

**Smoothness confidence** (Eq. 2), a bilateral weighted sum over the `K`-nearest raw input points `Ω_k` of `s_k` (`K=100` default):
`f_s(s_k, n_k) = Σ_{j∈Ω_k} θ(‖s_k − q_j‖) φ(n_k, q_j − s_k)`
`θ(‖s_k−q_j‖) = exp(−‖s_k−q_j‖² / (h_k/2)²)`  — spatial term, decays with support radius `h_k = max_{j∈Ω_k} ‖s_k−q_j‖`
`φ(n_k, q_j−s_k) = exp(−‖n_k^T(q_j−s_k)‖² / (h_k/4)²)`  — orthogonal-distance term: penalizes raw points that lie far from the *tangent plane* at `s_k` along the normal direction.
High `f_s` ⇔ the local raw points form a thin plate hugging the surface (low noise / high sample density); low `f_s` ⇔ either noisy scans or genuine high-frequency geometric detail — both read as "needs more scans."

**Combined confidence** (Eq. 3, normalized to [0,1]):
`f(s_k) = f_g(s_k,n_k) · f_s(s_k,n_k)`

**NBV utility field** (Eq. 4, §5.1) — for candidate free-space voxel `v_i`:
`g(v_i) = max_{s_k} { w(v_i,s_k) · (1 − f(s_k)) }`
`w(v_i,s_k) = w_d(s_k,v_i) · w_o(s_k,v_i) · w_v(s_k,v_i)`  (product of three Gaussians/indicator, each ∈[0,1] ⇒ `g(v_i)∈[0,1]`)

- `w_d = exp(−(‖v_i−s_k‖ − d_o)² / (d_n/4)²)`, `d_o=(d_n+d_f)/2` — **standoff/GSD term**, Gaussian around the scanner's optimal working distance, tolerance tied to the near-limit `d_n`.
- `w_o = exp(−(1 + n_k^T·d(v_i,s_k))² / (1−cos σ)²)`, `d(v_i,s_k) = (v_i−s_k)/‖v_i−s_k‖`, `σ=25°` default — **orthogonality term**. Given the inward-normal convention, this is maximized (→1) exactly when `d(v_i,s_k) = −n_k`, i.e. the viewing ray from the surface point to the candidate location is anti-parallel to the inward normal = parallel to the outward normal = straight-on incidence. It is a **smooth Gaussian falloff in angular deviation**, not a hard cutoff; `σ=25°` sets the half-tolerance-ish width. This is the one directly reusable, closed-form, computable "orthogonality weight" in the paper.
- `w_v ∈ {0,1}` — binary **visibility** term via ray casting from `v_i` through `s_k`, tested against the reconstructed mesh (occlusion test requires a surface — see §6 "does NOT transfer" below).

**Local direction refinement** (Eq. 5, §5.2) — given a chosen VVF location `v_i` (candidate anchor `s_k`), search a local neighborhood `Υ_k` (iso-points within 10% of `d_f`, default) for the best aim point `s_k'`:
`argmax_{s_k'∈Υ_k} η(s_k')·ζ(s_k')`
`η(s_k') = max_{j∈Υ_k} { w(v_i,s_j) f(s_j) }`  — rewards having a *confident* iso-point nearby (anchors registration).
`ζ(s_k') = Σ_{j∈Υ_k} w(v_i,s_j) (1−f(s_j))`  — rewards *many* low-confidence iso-points nearby (coverage).
This is a discrete, grid/neighborhood-based local search — **not** gradient-based / differentiable-rendering refinement. No gradients of the score w.r.t. pose are ever computed; the "optimization" is exhaustive evaluation over a finite candidate set (iso-points within a fixed-radius neighborhood).

**Selective point insertion probability** (Eq. 6, §5.3): `P(p_i) = (1 − f(s_k))^ρ`, `ρ=3` default, `s_k` = nearest iso-point to new raw point `p_i`. Concentrates new samples in low-confidence regions.

---

## 3. Concrete results / numbers

- **Table 1** (avg. per scan-iteration timing, 5–15 NBVs/iteration): robot motion+capture 463 s (65%), scan consolidation/registration 178 s (25%), NBV computation 71 s (10%) — total ≈712 s (~12 min) per iteration.
- **Table 2** (per-object totals): e.g. Fig. 1 figures: 1.83M pts / 8 iters / 105 NBVs / 124 min. Church model (Fig. 18, largest): 4.14M pts / 15 iters / 213 NBVs / 235 min (~4 h).
- **Fig. 9/11 quantitative comparison** (synthetic model, ground-truth distance error): vs. Khalfaoui et al. 2013 (visibility-based) and Kriegel et al. 2013 (PVS/boundary-based): both baselines *converge* (stop finding barely-visible/boundary points) after **55 scans**; this method takes **67 scans** but the resulting cloud has **74K points** vs 85K (visibility) / 86K (PVS) — fewer points, "yet captures more details" (fewer points because of selective insertion, Eq. 6). Error histogram (Fig. 9) shows a higher fraction of points in the lowest error bins for their method than either baseline.
- **Fig. 12** (real dragon toy, same NBV budget = 25 scans): visibility-based misses the left wing entirely (incomplete); PVS is better (detects open boundaries) but still incomplete; this method is complete with the same 25-NBV budget.
- **Fig. 13 ablation — which confidence term earns its place** (real goat figurine, `#Iter=10` per Table 2): comparing NBV-selection criteria head-to-head at fixed iteration count —
  - curvature-based confidence: **incomplete after 20 iterations**
  - point-density-based confidence: more complete after 20 iterations, but **holes remain** (highlighted regions)
  - `f_g·f_s` (this paper): **complete with rich detail after only 10 iterations** — half the budget of the weaker baselines for a *better* result. This is the paper's direct "does the quality term earn its place" ablation.
- **Fig. 14** — geometry-adaptive NBV search space vs. **sphere-constrained** NBV search space (Vásquez-Gómez et al. 2009), both given 36 scans on a complex robot-figure model: adaptive search is complete; **sphere-based search misses the raised weapon and the feet** (self-occluded, off-sphere-normal geometry). This is a direct, citable negative result for uniform/sphere-based view spaces on complex/self-occluded shapes — germane to our viewsphere-agnostic-of-geometry problem, though note the failure mode here is *coverage* (unreachable/occluded regions), not *orthogonality*.
- **Fig. 15** — manual expert scanning (Artec Studio, interactive) vs. autonomous: manual took **200+ scans, >3 h**; autonomous took **~100 scans, <2 h**, comparable overall quality, but the autoscanner **missed some parts the human captured** (attributed to physical scanner-mount size / robot reach limits — an accessibility/reachability failure, not a scoring failure) while achieving *smoother* results in some regions due to more precise registration.
- **Fig. 16** — noise robustness: synthetic noise (Berger et al. 2013 model) injected at levels 0→1; reconstruction error increases with noise level (plot shown, not a closed-form curve); final model under strongest corruption "still shows satisfactory quality." This is a **point-position noise** sweep, not a normal-direction-error-specific sensitivity study (see gap analysis below).
- **Fig. 17** — texture/color capture is demonstrated but explicitly **excluded from the NBV/quality computation**: *"we choose not to use color information for computing NBVs, because color information may not be accessible for all scanners and high color variation may not correlate with rich geometric details"* (§6, "More applications"). Explicit deliberate exclusion of image-space cues.
- Termination criterion: iterate until "the global confidence difference between two consecutive iterations is below a threshold (0.005 by default)" (§6).

---

## 4. Load-bearing assumptions

1. **A watertight mesh must exist.** Every confidence and visibility computation is defined on the Poisson iso-surface / iso-points, not on the raw point cloud. `f_g` needs the Poisson scalar-field gradient `Γ`; the visibility term `w_v` needs ray–surface intersection against a closed mesh.
2. **Consolidated, well-oriented normals feeding the reconstruction.** WLOP preprocessing is assumed to produce reliable normals before Poisson reconstruction even runs — normal quality is a hidden precondition, not something the confidence score itself audits (see gap 4 below).
3. **Isotropic sensor, 5-DOF search.** "Our current approach assumes that the scan sensor used is isotropic. That is, we search for NBVs in 5D pose space, ignoring the effects of sensor rotation about its normal" (§7, explicitly flagged as future work for 6D).
4. **Scanner behavior fully captured by (near, far) working range + a single angular tolerance `σ=25°`.** No dependence of the model on incidence-angle-vs-error data from an actual sensor; `σ` is a tuned/assumed constant, not derived from measurement.
5. **Greedy, single-pass optimization**: "at each iteration the scanner seeks to improve the quality of the regions with the currently lowest confidence. There could be more efficient sequences of scanning that can reach the same quality faster, with fewer scans, or shorter paths" (§7) — acknowledged as suboptimal but left unaddressed.
6. **Accessibility/reachability is a known, unsolved ceiling**: "due to the limitation of the scanner's accessibility... we cannot guarantee a given quality requirement is satisfied everywhere on the model. This limitation in fact applies to virtual scanners as well... since deep or complex cavities are hard or impossible to cover" (§7) — this is the same "not every desired view is reachable" constraint our framing doc states, independently arrived at.

---

## 5. What transfers to our setting

- **`w_o` (Eq. 4) is a directly reusable, closed-form soft-orthogonality weight**: Gaussian in `(1 + n·d)`, tolerance `σ`. Answers "smooth or threshold?" (branch Q2) for *this* paper's design choice: **smooth**, not a hard knee — but see gap below, it is a *design choice*, not measured sensor physics. Still useful as a template: `w_o = exp(−(1+n^T d)² / (1−cosσ)²)` is a clean, bounded, differentiable-in-pose function we could parameterize with a `σ` fit to actual D405 incidence-error data once that curve exists (Branch/RQ1 empirical piece is still missing — this paper doesn't supply it).
- **Bounded, [0,1]-normalized per-point and per-voxel scores** (`f(s_k)` and `g(v_i)`, both products of terms ≤1). Direct precedent for branch Q "bounded, normalized per-view score" — yes, this exists and the construction (product of Gaussian sub-scores, each capped at 1) is a reusable pattern. Caveat: normalization is *within one object's scanning session*; the paper never tests or claims cross-object comparability of raw magnitudes — bounded ≠ calibrated.
- **Worst-point / argmax pooling used deliberately for one purpose**: NBV location scoring `g(v_i)` pools over iso-points by **max** (not mean, not area-weighted) — i.e., a candidate viewpoint's score is set by the single least-confident, best-aligned point it can see (Eq. 4). This is a citable precedent for "worst-point pooling" (branch Q on aggregation) — but note it's used to *find where to scan next*, not to *rate an already-decided view's overall quality* (that role instead uses **sum**-pooled `ζ` in Eq. 5, and **max**-pooled `η` for a different purpose — registration anchoring). Three different pooling strategies for three different sub-questions in the same paper is itself the useful data point: pooling choice tracks the *purpose* (find worst spot vs. estimate aggregate coverage vs. find a good anchor), not a universal "the" right pooling operator.
- **Direct evidence that geometry-agnostic (sphere-constrained) view spaces under-cover self-occluded/complex shapes vs. adaptive spaces** (Fig. 14, 36-view budget, missed weapon+feet) — independent, citable support for our framing doc's core motivating problem (uniform viewsphere is blind to geometry). Useful as an external citation for our RQ1/RQ2 motivation, though the failure mode shown is coverage/occlusion, not "missing the orthogonal angle."
- **Explicit acknowledgment that reachability/accessibility, not scoring quality, is often the binding constraint** (Fig. 15, §7) — matches our "not every desired view is reachable" constraint; useful as precedent that even a well-resourced autoscanner (own robot, own scanner, full 3D search) hits this ceiling and treats it as out of scope rather than solved.
- **Selective/weighted point re-insertion (Eq. 6) and confidence-weighted ICP** are a reusable pattern for propagating a per-point quality/confidence value downstream into registration weighting — relevant if we ever want per-point confidence to inform how much we trust a given region of the fused PLY during pose refinement (tangential to Branch 3 handoff, not core to this branch).

---

## 6. What does NOT transfer, and why (negative findings — first class)

1. **No incidence-angle-vs-depth-error sensor model at all.** The scanner (Artec Spider, structured-light) is abstracted to `[d_n, d_f]` working range + a single tuned angular Gaussian (`σ=25°`) with **no citation, no calibration, no measured curve**. This paper supplies **zero data** toward branch Q1 (`sigma(range,theta)` for an active-stereo sensor like the D405). The `σ=25°` value is a hyperparameter picked by the authors for aesthetic/reconstruction-quality tuning on their setup, not derived from a sensor error study — do not cite it as if it were measured incidence tolerance.
2. **No treatment of specular/glare-driven off-normal preference at all.** No discussion of glossy or dark surfaces, no case where the model would deliberately choose an off-normal view. The orientation term is monotonically increasing toward normal incidence with no exceptions built in — this paper offers **no data or mechanism** for branch Q3 ("does orthogonality ever hurt"). This silence is itself informative: a leading reconstruction-quality autoscanner paper does not even consider the specular failure mode, so if our project needs it, we cannot lean on this reference.
3. **No image-space quality term whatsoever, and a deliberate argument against using one**: color/appearance is captured (Fig. 17) but explicitly excluded from scoring because "high color variation may not correlate with rich geometric details" (§6). This is a *reconstruction-quality* framing where geometry is ground truth; it gives no traction on branch Qs about defect detectability, image-space judging, specular blowout/exposure/focus, or VLM-as-judge — those are out of scope for this paper by the authors' own design choice.
4. **No normal-error sensitivity study.** Every scoring term (`f_g`, `f_s`, `w_o`) is a function of `n_k`, yet the paper never isolates normal-direction noise; Fig. 16's noise sweep uses a general point-position noise model (Berger et al. 2013) and reports only end-to-end reconstruction error, not `d(score)/d(normal error)`. Branch's top technical risk question is **not answered** here — this is a genuine literature gap this paper does not fill, and normal accuracy requirements would need to come from elsewhere (or be established empirically ourselves).
5. **Visibility/self-occlusion is computed by ray-casting against a full watertight mesh** (§5.1), not by any point-cloud-only technique (no hidden-point-removal, no occupancy field, no splat/alpha-compositing approximation). This paper gives **no evidence at all** on branch Q "how is visibility computed without a surface" — it sidesteps the question entirely by requiring the mesh to already exist. Since our setting has "no mesh" as a listed defect condition, the visibility mechanism here is **not directly applicable**; only the *idea* (ray-cast occlusion test) transfers, the *implementation* (needs a closed surface) does not.
6. **No propagation of input geometry uncertainty into the view score.** The confidence score treats the current Poisson surface as the thing to *improve*, not as a possibly-biased reference frame — there is no notion of "the fused cloud itself might be offset from what a live camera would report," which is precisely our ~4 cm fused-vs-single-view offset problem. This paper's `f_g`/`f_s` only ever measure *internal consistency of the reconstruction with itself*, never *calibration/agreement with an independent live sensor reading*. That gap is real and would be a genuine contribution if we filled it (branch Q on "does any method propagate input geometry uncertainty" — answer for this paper: **no**).
7. **Not applicable to primitive-shape / industrial-inspection object classes as tested.** Test objects are all complex, self-occluded artistic/statuette models (action figures, animals, church model) evaluated on *overall reconstruction fidelity*, never simple primitives (box, cylinder) and never with a "defect" ground truth. Branch-2-handoff relevant: this paper's benchmark protocol (ground-truth distance error against a known digital model, Fig. 9) is not directly reusable for our primitive-shape / real-defect regime, though the *protocol structure* (closest-point distance error histogram vs. ground truth) is a reasonable template to imitate.
8. **Gradient-based/differentiable local refinement is absent.** Despite the branch's interest in "gradient magnitude of a utility field" and differentiable-rendering pose refinement, this paper's local optimization (Eq. 5) is pure discrete search over a finite candidate set of already-sampled iso-points within a fixed-radius neighborhood — no gradients of `g` or `η·ζ` w.r.t. camera pose are ever computed or claimed to exist in closed form (indeed `w_v`, the visibility indicator, is explicitly discontinuous — ray-cast hit/miss — so the field is *not* smooth in general; the paper never addresses or smooths this discontinuity, contrary to the branch's "is the landscape smooth enough to climb" question). This is evidence *against* naive gradient climbing being used even by a paper motivated to find good views — they chose discretize-and-search instead, and never discuss why (no ablation of sampling vs. gradient).

---

## 7. Cross-reference to branch questions this document actually informs

| Branch question (paraphrased) | What this paper gives |
|---|---|
| Q1 incidence-angle-vs-error curve | **Nothing** — no sensor error model, only a tuned angular tolerance `σ=25°` with no measurement behind it. |
| Q2 smooth vs. threshold angle penalty | For this paper's own construction: smooth Gaussian (Eq. 4 `w_o`), not a hard knee — but a *design choice*, not a measured sensor response. |
| Q3 orthogonality ever hurts (specular) | Not addressed; no mechanism, no data, no discussion. |
| Q4 defect detectability vs. angle | Out of scope; paper is reconstruction-fidelity-only, explicitly excludes appearance (§6). |
| Q5 table of computable view-quality terms | Contributes: completeness (`f_g`, Eq.1, PLY-space, needs Poisson gradient), smoothness (`f_s`, Eq.2, PLY-space, needs raw points+normals+consolidated K-NN), standoff/GSD (`w_d`, Eq.4), orthogonality (`w_o`, Eq.4), visibility (`w_v`, Eq.4, needs mesh), a composite NBV utility (`g`, Eq.4). No sampling-density, viewpoint-entropy, surface-normal-entropy, or defect-detectability terms defined here (those come from elsewhere in the field). |
| Q6 redundant terms | `f_g` and `f_s` are explicitly presented and ablated as complementary, not redundant (Fig. 5 shows they peak in different regions/times: `f_g` improves early with coverage, `f_s` lags and needs detail-focused scans, §6 "Quantitative evaluation" discussion of Fig. 8). Useful data point for a correlation-cluster analysis even though this paper itself doesn't report a correlation coefficient. |
| Q7 normal-error sensitivity | **Not answered** — flagged as this branch's top risk; this paper's Fig. 16 noise sweep is position-noise, not normal-error-specific. |
| Q8 bounded normalized per-view score precedent | **Yes**: `f(s_k)∈[0,1]` and `g(v_i)∈[0,1]` by construction (Eqs. 3–4), but only within-object-session comparable, not validated across objects. |
| Q9 unit of scoring / pooling | Point (iso-point) is the atomic unit; per-view aggregation uses **max** for NBV location scoring (Eq.4), **sum** for coverage estimate (Eq.5 `ζ`), **max** for registration-anchor estimate (Eq.5 `η`) — pooling choice is purpose-dependent, not a fixed rule. |
| Q10 survives a defective fused PLY? | Partition: `w_d`, `w_o`, `w_v`(concept only) are **computable today** given points+normals (no mesh needed for the formulas themselves, though `w_v` as implemented needs one); `f_g` and the mesh-based `w_v` implementation are **needs-a-mesh-we-do-not-have**; `f_s` is **computable today** directly on raw/fused points if we have per-point normals (no Poisson step logically required for Eq. 2 itself, only convention borrowed from their pipeline). |
| Q11 visibility without a surface | **Not answered** — this paper always ray-casts against a completed mesh; gives no evidence on point-cloud-only HPR robustness, dropout-hole behavior, or parameter sensitivity. |
| Q12 propagating input geometry uncertainty into the score | **No** — confidence is purely internal-consistency-of-reconstruction; never models "the reference cloud itself may be wrong relative to a live sensor." Flagged as an open gap / possible contribution. |
| Q13 4 cm fused-vs-single-view offset → judge orthogonal to wrong surface | Paper doesn't model this scenario (single robot+scanner rig, presumably self-consistent calibration); no data either way — cannot confirm calibration-fixable vs. must-enter-the-score from this source. |
| Q14 formal "best view within a neighbourhood" definitions | Contributes one concrete formal definition: Eq. 5, a bi-objective (registration-anchor confidence `η` × coverage `ζ`) argmax over a fixed local candidate neighborhood — worth comparing against MVP's constraint-satisfaction-with-margin and mutual-information viewpoint stability from other sources. |
| Q15 is the landscape smooth enough to climb | Negative evidence: visibility term `w_v` is a hard binary ray-cast indicator (discontinuous in pose), and the paper never smooths it nor discusses the discontinuity — suggests visibility discontinuity is a real, unaddressed problem even in a well-cited quality-driven planner. |
| Q16 sampling vs. gradient | This paper uses discrete sampling/search exclusively (voxel grid + local-maxima suppression + finite neighborhood argmax), never gradients — silent on why, no comparison given, but is one more data point on the "sampling wins in practice" side of the ledger. |
| Q17 offline vs. online local refinement | Refinement here (Eq. 5) happens once per selected NBV location, offline relative to execution (computed before the robot moves) — consistent with "refine each candidate once at construction time" being a viable, precedented pattern. |
| Q18–Q20 image-space judging, render-and-score, VLM-as-judge | Not addressed at all — deliberately excluded (§6, color-exclusion quote). No transfer. |
| Q21 ground truth for "best view" | Contributes the **metric-based-with-noted-circularity** family: Fig. 9's ground-truth-distance-error benchmark uses closest-point distance to a known digital model, not a "best view" label per se — useful for Branch 2 handoff as a benchmark-protocol example, not a best-view ground truth per se. |
| Q22 replicable on primitive shapes | All test objects are complex/artistic (statuettes, action figures, animals, a church model); **no primitive-shape evaluation** exists in this paper — Branch 2 handoff: flag as non-primitive benchmark set. |
| Q23 ablation showing a term earns its place | **Yes, directly usable**: Fig. 13, curvature-based vs. density-based vs. `f_g·f_s` NBV selection at matched iteration budgets, `f_g·f_s` reaches a complete detailed model in half the iterations of the runner-up. Verbatim protocol worth imitating for Branch 2 handoff. |

---

## 8. Gist / summary for reuse

Wu et al. 2014 replace "minimize scan count" with "maximize reconstruction fidelity" by scoring confidence directly on a **Poisson-reconstructed mesh** (not the raw point cloud): a completeness term from the Poisson scalar-field gradient stability (Eq. 1) times a smoothness term from local raw-point bilateral consistency (Eq. 2), combined into a bounded [0,1] per-point confidence (Eq. 3). This confidence field drives a 3D viewing-vector field (Eq. 4) whose per-voxel utility is (optimal-standoff Gaussian) × (orthogonality Gaussian, `σ=25°` tolerance) × (binary mesh-ray-cast visibility) × (1 − confidence), max-pooled over visible surface points — directly reusable as a template for an orthogonality/standoff weighting term, and as one precedent for a bounded per-view score. The paper is a strong, well-ablated **negative case** for several of our branch's harder questions: no sensor incidence-error model, no specular/off-normal handling, no image-space quality term (deliberately excluded), no normal-error sensitivity study, no point-cloud-only visibility method (mesh required throughout), and no propagation of input-geometry uncertainty into the score. Its core machinery (`f_g`, mesh-based `w_v`) needs a mesh we do not have; `f_s`, `w_d`, `w_o` need only points+normals and are computable today on our fused PLY.
