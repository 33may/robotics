# MV-GEL: Language-Driven Multi-View Geometric Entity Localization on Meshes

Bali & Aydin (Helmholtz Zentrum Hereon / TU Hamburg; DFKI Saarbrücken), **arXiv:2606.31533v2** (submitted 30 Jun 2026, rev. 23 Jul 2026), accepted ECCV 2026.
Source: full HTML (`arxiv.org/html/2606.31533v2`), read via targeted section-by-section extraction (multiple passes to cross-check a keyword-search contradiction, see §7 below). No PDF/figures inspected directly — all numbers below are from HTML text/tables as returned by extraction.

## 0. One-line identity — read this before anything else

**MV-GEL is not a paper about a VLM/LLM agent choosing among a menu of candidate views via language reasoning.** It is a small, LoRA-trained, differentiable **ranking network** (CLIP encoders + a cross-view transformer) that scores 60 uniformly-sampled synthetic renders of a clean CAD mesh, picks the top-1 (or top-k), and only *then* hands that single image to a downstream reasoning-segmentation VLM (LISA) for 2D mask prediction, which is lifted back to 3D mesh faces/edges. **No LLM ever sees a menu, reasons in text about which view is better, or is prompted with marks/labels over multiple candidates.** This is the single most important scoping fact for this branch: the paper answers "does geometry-aware ranking beat semantic-only ranking for picking a good view" (yes, clearly) but says essentially nothing about egocentric/allocentric framing, visual-prompting menus, option count/order, or any VLM-agent capability-floor question — because the choosing agent here isn't a prompted VLM at all, it's a trained scorer. Treat this paper as evidence for **RQ1 (view judging)** and the **architecture-verdict** question ("should a VLM even be doing the choosing"), not for RQ4 presentation-to-VLM questions.

## 1. Core method

Pipeline (§5.3): **ŷ = ℒ ∘ Γ ∘ Ψ(ℳ, q)** — mesh ℳ, text query q, predicted 3D entity labeling ŷ.

- **Ψ = GELviews**, the view-ranking module (§5.1):
  - CLIP head: LoRA-adapted (r=8, α=16, dropout 0.1) CLIP image/text encoders give initial semantic score `s_v^clip = ⟨f_v/‖f_v‖, t/‖t‖⟩`.
  - Learnable per-view positional embedding `p_v`: `f̃_v = f_v + p_v` (breaks view-permutation invariance so the transformer knows *which* view is which).
  - Modality fusion (ablated, §6): **FiLM** `γ(t)⊙f̃_v + β(t)`, **Cross-Attention** (token-level text↔view), or **No-Fusion** (skip straight to self-attention over position-aware features).
  - View Interaction Head: a multi-layer Transformer encoder 𝒯 attends **across all 60 candidate views simultaneously** (not sequentially): `h_v = 𝒯({f̂_v}_{v=1}^{V})_v`, then `s_v^geo = w^⊤LN(h_v)`. Exact depth/heads/hidden-dim **not stated anywhere in the text** — flagged, not found even after a targeted search.
  - Final score: `s_v = α(s_v^clip + s_v^geo)`, α a learned temperature.
  - Training objective (Eq. 1), pairwise hinge ranking loss: `ℒ_rank = (1/|𝒫|) Σ_{(i,j)∈𝒫} log(1+exp(-(s_i-s_j-m)))`, margin m=0.1, positive set 𝒱⁺ = top-k views by a **ground-truth visibility oracle** (k = max(1, ⌊V·ρ⌋), ρ=0.09 → top 5 of 60), all others negative.
  - Visibility oracle `S(v_i, ℰ) = Σ_{p_j∈𝒫_ℰ} 𝟙{p_j visible from v_i}` computed by **ray-casting against the ground-truth clean mesh** — i.e. supervision requires already knowing the exact 3D location/extent of the target entity ℰ on an error-free mesh.
- **Γ**: LISA-CAD, a domain-adapted (CAD-render-finetuned) version of LISA (reasoning-segmentation VLM: multimodal LLM + SAM-style mask decoder), run on the single top-ranked image (Table 1) or top-3 (Table 2/Fig. 5).
- **ℒ**: geometric lifting, `ŷ = ℒ({m_k}_{k∈S}, {π_k}_{k∈S}, ℳ)` — predicted 2D masks projected back onto mesh faces/edges via ray projection, multi-view when S has >1 view. **Exact aggregation rule (voting/union/intersection) across multiple views is not specified in the text** — only that additional views act as a "fallback ... especially in cases where [the] VLM might miss [the entity] on the main view" (§6).

**Candidate view generation** (§4/§6, "Training"): uniform grid, elevation ∈ [-60°, 60°], azimuth ∈ [0°, 360°) at 30° increments → **60 candidate views per mesh instance**, each rendered `I_k = ℛ(ℳ, π_k)`. Rendering shading/texture/material convention is **not specified** in the extracted text (checked explicitly — only found: "an unannotated RGB image for model inference and a secondary mask-highlighted rendering for supervised training," no color/shading spec).

## 2. Concrete numbers — full Table 1 (Section 6)

Columns: View Selector | Face IoU / Prec / Rec / F1 | Edge IoU / Prec / Rec / F1.

**Vanilla LISA (rows 1–6, domain-unadapted):**
| Selector | Face F1 | Edge F1 |
|---|---|---|
| Cross-Attention | 0.390 | 0.074 |
| FiLM | 0.391 | 0.079 |
| No-Fusion | 0.389 | 0.077 |
| Only-CLIP | 0.272 | 0.061 |
| Random | 0.174 | 0.046 |
| GTviews (oracle) | 0.389 | 0.079 |

Note the oracle row barely beats FiLM/No-Fusion here — with an un-adapted segmentation backbone, *which view you pick barely matters* because the bottleneck is the segmenter, not the view.

**LISA-CAD (rows 7–12, CAD-domain-finetuned segmenter):**
| Selector | Face IoU | Face F1 | Edge IoU | Edge F1 |
|---|---|---|---|---|
| Cross-Attention | 0.486 | 0.582 | 0.287 | 0.359 |
| FiLM | **0.501** | **0.598** | 0.284 | 0.358 |
| No-Fusion | 0.490 | 0.584 | 0.283 | 0.355 |
| Only-CLIP | 0.305 | 0.377 | 0.160 | 0.210 |
| Random | 0.196 | 0.250 | 0.071 | 0.101 |
| GTviews (oracle ceiling) | 0.529 | 0.632 | 0.311 | 0.390 |

Paper's own framing: FiLM realizes "95% of oracle's 0.63 F1" for faces, ">90% of 0.39 F1 ceiling" for edges.

**Table 2, zero-shot point-cloud-native baselines vs. MV-GEL (render-then-segment-then-lift):**
| Model | Face IoU | Face F1 | Edge IoU | Edge F1 |
|---|---|---|---|---|
| PartSLIP | 0.073 | 0.114 | 0.012 | 0.022 |
| Find3D | 0.171 | 0.260 | 0.017 | 0.032 |
| MV-GEL (FiLM@top1) | **0.501** | **0.598** | **0.284** | **0.358** |

Find3D: "0.95 Recall but only 0.17 Precision" on faces — severe over-segmentation/"bleeding" without topological boundaries.

**Dataset scale** (§4/§6): ABC CAD dataset (>5,000 geometries), 54k mined query–entity pairs; test set 1535 query–entity pairs over 218 meshes (655 faces, 880 edges). Queries constructed programmatically from ground-truth B-rep attributes (curvature, orientation, adjacency, symmetry, continuity, relative position) to be intrinsically unambiguous — i.e. the language-grounding-under-ambiguity problem is largely engineered away by construction, not solved.

**Compute** (§6, "Training"): 4×H100, LoRA + mask-decoder only trainable, lr=1e-4. Epoch count not stated. **No variance/seeds/error bars reported anywhere in Table 1/2** — single-run point estimates throughout (same evidence-discipline gap as the sibling Cypher/3DSG paper).

## 3. Ablations that matter to us

- **Domain adaptation of the segmenter dominates everything else**: Vanilla→CAD-tuned LISA jumps Face F1 0.391→0.598 (+53% rel.) and Edge F1 0.079→0.358 (+353% rel.) *holding the view selector fixed (FiLM)*. This dwarfs the view-selector ablation below — a caution against over-crediting view-selection quality when the downstream reasoning model itself is undertrained for the domain.
- **View-selector design, LISA-CAD only**: FiLM (0.598) ≈ No-Fusion (0.584) ≈ Cross-Attention (0.582) — a tight ~2.4pt band, i.e. **once you have any cross-view geometric interaction, the exact fusion mechanism barely matters**. What matters is *whether* the model reasons jointly across views at all:
  - Only-CLIP (semantic score alone, no cross-view transformer): Face F1 0.377 vs FiLM 0.598 = **−37% relative**; Edge F1 0.210 vs 0.358 = **−41% relative**.
  - Random: Face F1 0.250 vs 0.598 = **−58% relative**; more than 2× worse than geometry-aware ranking.
- **ViewNMS ablation (Table 2, row 7 vs 8)**: suppressing near-duplicate/nearby views *hurts* — greedy top-3 Face F1 0.540 vs ViewNMS-filtered top-3 0.495 (**−4.5pts**). Paper's own read: "Views nearby act as fallback for entity segmentation [when the] VLM might miss them on the main view." **This is a genuine negative finding for a naive "reduce distractor views" intuition** — but it is about a *differentiable multi-view feature aggregator's* recall, not about a discrete-choice LLM's confusability between near-identical menu items. Flag explicitly: this likely does **not** transfer to the branch's "how similar may options be" question about a language menu — different mechanism (soft multi-view pooling helped by redundancy vs. discrete single-choice hurt by redundancy is the more plausible direction for an LLM, per other branch sources on option-count degradation). Do not cite this as evidence that redundant candidates are harmless for an LLM menu.
- **Multi-view aggregation curves (Fig. 5a/5b, described not tabulated)**: FiLM/Cross-Attention/No-Fusion "rapidly close gap to oracle within first 7–8 views" (of 60 available) as top-K grows. Order-of-magnitude data point only — mechanism is a trained aggregator's recall-vs-K curve, not a "how many menu items can a VLM handle" curve.
- **Inference efficiency (Fig. 6, described)**: ranking module overhead ≈0.1s; Random-5-views pipeline 0.6s vs FiLM-top-1 0.25s (2.4× faster, because top-1 doesn't need multi-view segmentation); VRAM ~16GB (Random) vs ~17GB (FiLM) — minor, not load-bearing for us.

## 4. The textureless/industrial-CAD claim — evidence discipline check

The framing-doc selection blurb cites this paper for "reports CLIP-style semantic ranking fails on textureless industrial parts." Verified exact quote, **Section 2.4 ("Prompt-Aware View Selection"), related-work discussion**, single sentence: *"However, global semantic matching fails on homogeneous, textureless industrial CAD parts, where pinpointing fine-grained features (e.g., 'the inner chamfered edge') requires explicit geometric awareness."*

This is **asserted as related-work motivation** (framed as a critique of Cap3D/ViewRefer-style CLIP-similarity view selection), **not run as a controlled ablation with a textured-vs-textureless split of their own benchmark**. The closest empirical support is the Only-CLIP row of Table 1 (§3 above: −37%/−41% relative F1 vs the full geometry-aware model) — but since the entire ABC test set is CAD B-rep geometry with rendering appearance unspecified (no confirmed material/texture in the extracted text), Only-CLIP's collapse is consistent with the textureless-failure claim but does not isolate "textureless" as the causal variable (could equally be "any fine, small, structurally-defined target," textured or not). **Verdict for evidence-discipline table: demonstrated + partially ablated (Only-CLIP row), not a controlled textured/textureless-subset ablation. Cite the *mechanism* (semantic-only ranking underperforms on fine geometric targets) with confidence; cite "textureless" as the *specific causal factor* only as the authors' own claim, not as something they isolated.**

(Note on process: an earlier extraction pass returned "industrial" and "textureless" as literally absent from the document via keyword search — this was a summarizer-model artifact, not real. A second, more targeted verbatim-quote extraction found the single sentence above. Recording this because it's exactly the kind of silent hallucination this branch's evidence-discipline requirement exists to catch — always re-verify a load-bearing quote with a second, narrower fetch.)

## 5. Failure modes / limitations

No dedicated Limitations section exists in the paper (checked directly). Implicit findings scattered through §6:
- **Edge localization is roughly half as good as face localization everywhere** (FiLM: Edge F1 0.358 vs Face F1 0.598; oracle: 0.390 vs 0.632) — attributed to "thin projections, discretization sensitivity, and view-dependent occlusions." Consistent across every selector — a structural, not selector-fixable, difficulty.
- Point-cloud-native zero-shot methods (PartSLIP, Find3D) are far worse than render-and-lift (§2 above) — the paper's implicit argument is that going through calibrated 2D renders + 2D segmentation + geometric lifting currently beats reasoning natively in 3D/point-cloud space for this task.
- Conclusion (§7, verbatim): "language-driven 3D localization is fundamentally a geometry-aware view prioritization problem, not merely an image-text alignment task."

## 6. What transfers to our setting

- **Mechanism, not numbers**: semantic-only (CLIP) view scoring underperforms geometry-aware scoring by a large, consistent margin (−37 to −58% relative F1) on fine/structurally-defined targets on clean CAD geometry. Directly supports the **Branch 3 handoff** question in the abstract sense — geometry-conditioned ranking is a real, demonstrated win over pure semantic similarity, at least on clean meshes. It does **not** by itself establish this survives on a defective survey cloud (see §7).
- **Random-baseline discipline**: their own Random-view row (Face F1 0.250, more than 2× worse than the geometry-aware method 0.598) is a concrete data point for the branch's "what random baseline must a view-selection claim beat" question — here the bar clears easily and by a wide margin, i.e. on this task geometric awareness is not a marginal win. Useful comparator magnitude when we design our own random-viewsphere baseline, though the target metric differs (segmentation F1 vs. our orthogonality/coverage judging).
- **Architecture pattern worth stealing for RQ1/RQ4's tension**: MV-GEL is a working existence proof that "which view is best" does **not** have to be solved by a prompted VLM/LLM reasoning over a menu — it can be delegated to a small trained (or, in our offline/no-training-data case, a hand-built geometric) scoring function, with the VLM only ever consuming the single winning image. This directly informs the branch's **architecture-verdict** question: MV-GEL's answer to "should the orchestrator see the render or should selection happen elsewhere" is neither of the branch's two options (blind orchestrator-worker vs. sighted single agent) — it removes the *reasoning* VLM from the selection step entirely and gives selection to a cheap differentiable module trained against a geometric oracle. This is structurally the same move our framing doc's "Optimization track" and RQ1 propose (build an objective judge of view quality in PLY space, let optimization/ranking — not VLM prompting — pick the view).
- **Visibility-oracle-via-ray-casting as a template for RQ1**: `S(v_i, ℰ) = Σ 𝟙{visible}` is a simple, reusable pattern for scoring candidate views against a target region in geometry space — directly adaptable to our "how orthogonal / how well does it frame the region" view-quality question, modulo replacing "entity visibility count" with an orthogonality + framing + coverage objective. Feed to **Branch 2** as a candidate metric-construction pattern (alongside the geometry-weighted IoU below).
- **Geometry-weighted IoU metric** (weight by triangle area for faces / edge length for edges rather than raw point/pixel count) — a reusable metric-design idea for **Branch 2**: our own coverage metric should probably weight by real-world surface area, not raw point count, for the same reason (avoid biasing toward large flat regions vs. small important ones).

## 7. What does NOT transfer, and why — first-class negative findings

- **No language-agent view selection exists in this paper at all.** It cannot answer, and should not be cited for, any of: capability-floor benchmarks (MMSI-Bench/ViewSpatial/SpinBench/FlipSet), ego/allo framing, visual-prompting/set-of-marks menu design, option count/order effects for an LLM, composition of two spatial operations, stopping-criterion policy, or infeasible-view presentation. The "choice" in this paper is made by backprop against a ranking loss, not by a model reading a prompt and picking an option — architecturally a different problem from everything else in this branch.
- **Clean, complete, watertight CAD B-rep mesh, not a defective survey cloud.** The visibility oracle ray-casts against ground-truth, error-free geometry, and presupposes the target entity's exact 3D location/extent is already known (needed to compute k-of-60 positive views for training). Our setting has pose error (few mm), per-view depth dropout on glossy/dark surfaces, and a measured ~4cm fused-vs-single-view offset — none of which this paper's method or evaluation touches. **This is the paper's single largest non-transferring assumption.** Nothing in the text says what happens to GELviews-style ranking when the "ground truth" surface is itself uncertain, or has holes — that question is simply out of scope for MV-GEL, not answered-and-negative. Flag for **Branch 3/4 handoff**: do not assume geometry-conditioned ranking "is available" on our cloud just because it beats semantic ranking on a clean mesh — the oracle-construction step (which requires already knowing where the target is, on clean geometry) is the part most likely to break first on a sparse/holed/noisy cloud.
- **No kinematic/reachability constraints anywhere.** All 60 candidate views are free synthetic renders; no IK, no collision checking, no notion of an infeasible view, no reachable-shell minimum. The branch's SayCan-style "how should infeasible views appear" question gets zero evidence from this paper.
- **No orthogonality objective, despite surface-level resemblance.** GELviews optimizes for *segmentation-metric performance* (max IoU/F1 of the downstream mask), not camera-normal alignment to a surface. A top-ranked view could in principle be oblique if that happens to maximize visible/discriminable entity area. Do not import "GELviews-style ranking = orthogonality proxy" without an explicit re-derivation — the framing doc's requirement (evidence views orthogonal to the inspected surface) is not what this objective optimizes for.
- **No count/order/menu-composition findings for a language interface.** The "top-k" selection here is soft/differentiable ranking over a fixed 60-view grid consumed internally by the network, not a discrete enumerated menu read and chosen from by an LLM — nothing here informs token-count degradation, permutation sensitivity, or near-identical-option distinctiveness for a text/visual-prompt menu (contrast with the ViewNMS finding in §3, which points the *opposite* direction for a different mechanism — flagged there, not reused here).
- **Fully offline, no embodiment, no sequential acquisition cost.** 54k mined query–entity pairs evaluated as a static dataset; no robot, no eye-in-hand camera, no cost-of-acquiring-another-view concept, no stopping criterion. The branch's Explore-until-Confident-style "enough views" question gets nothing from this paper.

## 8. Branch-question-by-branch disposition

- **Handoff to Branch 3 ("does geometry-conditioned view ranking survive a defective survey cloud")**: MV-GEL is the strongest available evidence that CLIP-only/semantic ranking underperforms geometry-aware ranking *on clean CAD meshes* (§3, §6) — but it is silent on defective/sparse/noisy point clouds by construction (§7). **Answer: partial support for the general mechanism, zero support for our specific cloud conditions — flag as an open gap, not resolved by this paper.**
- **"What is the random baseline?"**: answered for this paper's own task (Face F1 0.250 random vs 0.598 geometry-aware, >2× gap) — useful magnitude reference, wrong task (segmentation quality, not VLM view-choice accuracy) to import directly. Hand to **Branch 2** as a comparator, not as our benchmark's actual random-baseline number.
- **"What is the sparsity/holes/colorless-geometry answer (GraspView/SPAZER/VLM3D/SeqVLM question)"**: MV-GEL is not one of the systems that question is about, and offers no sparsity-handling technique itself — its rendering pipeline assumes clean watertight meshes throughout. No transfer here; the question must be answered by the other four named systems.
- **"How many options, and how similar may they be?"**: MV-GEL's ViewNMS finding (redundancy helps a soft aggregator) is a real, reportable data point but explicitly flagged as likely non-transferring in direction to a discrete LLM menu (§3). Do not let it stand in for the LLM-specific answer Branch 1 needs elsewhere.
- **Architecture verdict question**: MV-GEL's existence as a working system is itself an argument that "let a trained/geometric scorer pick the view, keep the reasoning VLM blind to the menu entirely" is a viable third option beyond the orchestrator-worker vs. single-sighted-agent framing — worth stating explicitly when that verdict section is written.
- All remaining branch questions (capability-floor benchmarks, ego/allo restatement, compositional-operation cost, pose encoding, text-vs-image paired encoding, caption hand-off, chicken-and-egg render viewpoint, ordering/token-bias, infeasible-view display, hierarchical/on-demand pattern, stopping-criterion) — **this paper has no evidence to contribute; do not force a citation.**

## 9. Evidence-discipline summary (for the branch's cross-paper table)

| Claim | Ablated or demonstrated? | Variance reported? |
|---|---|---|
| Geometry-aware (FiLM/Cross-Attn/No-Fusion) beats Only-CLIP beats Random | **Ablated** — controlled swap of one module (Table 1, LISA-CAD rows 7–11), same data/backbone otherwise | No — single-run numbers, no seeds/error bars anywhere |
| CLIP fails specifically on "textureless industrial CAD parts" | **Asserted** (§2.4, related-work sentence) with only indirect ablation support (Only-CLIP row, no textured/textureless split) | N/A |
| Domain-adapted segmenter (LISA-CAD) beats vanilla LISA | **Ablated** (full Table 1, rows 1–6 vs 7–12) | No |
| ViewNMS hurts recall | **Ablated** (Table 2, one row pair) | No |
| Render-and-lift beats point-cloud-native zero-shot (PartSLIP/Find3D) | **Demonstrated**, not ablated in a controlled sense — different architectures entirely, not a component swap | No |
