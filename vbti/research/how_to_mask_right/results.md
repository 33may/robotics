# Accepted Results

## Part 1: Masking Feasibility — Embedding Analysis (2026-04-07)

**Question:** Are masked image embeddings in-distribution for the frozen SigLIP encoder?

**Method:** Embedded 3000 frames × 4 cameras from 01_02_black (clean) + 500 frames × 10 mask conditions (2 shapes × 5 fills). Added 3000 frames from 09_merged (diverse dataset) to test if wider clean distribution absorbs masked points. Total: 44,000 embeddings (1152-d). Analyzed via UMAP, cosine similarity, k-NN intrusion (k=10), MMD (RBF kernel).

**Key Finding:** Masked embeddings are consistently OOD. They form separate clusters in UMAP regardless of dataset diversity, masking shape, or fill strategy. Even with the larger 09_merged dataset expanding the clean manifold, masked points sit outside it.

**Evidence:**
- Cosine similarity: all conditions mean 0.94–0.97 (seemingly high but structurally distinct in UMAP)
- Destructive fills (noise, shuffle) drift furthest (mean cos ~0.94)
- Blur/mean stay closest (mean cos ~0.97) but still form separate islands
- results/umap.html, cosine_violin.html, knn_mmd.html, records.npz

**Implication:** Image masking before a frozen contrastive encoder cannot be used as training regularization. The encoder's embedding space is sensitive to image corruption. Data diversity is the primary lever for improving generalization.

---

## Part 2a: Attention Rollout — Negative Result (2026-04-07)

**Question:** Can standard attention rollout reveal what SigLIP attends to spatially?

**Method:** Extracted attention weights from all 27 layers of SigLIP-so400m with `output_attentions=True`. Computed attention rollout (matrix product across layers) and per-layer mean attention maps.

**Key Finding:** Near-uniform heatmaps with tiny corner artifacts from positional embeddings. No useful spatial signal.

**Evidence:** attention_rollout.ipynb — per-layer overlays show uniform attention across all 729 tokens.

**Implication:** Standard attention interpretation doesn't work for contrastive ViTs. SigLIP's mean pooling + contrastive training distributes attention broadly. Need methods designed for CLIP-family models (CLIP Surgery, token ablation).

---

## Part 2b: CLIP Surgery Spatial Attribution (2026-04-08)

**Question:** What does SigLIP attend to spatially? Does it encode cup position strongly during grasping phases?

**Method:** CLIP Surgery on frozen SigLIP-so400m. Extract 729 patch token embeddings, compute per-patch element-wise product with text embeddings for 6 prompts (rubber duck, red cup, robot gripper, wooden table, background wall, task instruction), subtract mean redundancy, sum to get spatial similarity maps. 5 frames × 4 cameras across trajectory.

**Key Findings:**
1. SigLIP encoding is **temporally static** — attribution doesn't shift across approach/grasp/transport phases
2. Duck and cup have **partially overlapping spatial encoding** (r ≈ 0.02–0.35) — not cleanly separated
3. Robot gripper is most salient individual object (0.013–0.014 mean), beating duck (0.009–0.011) and cup (0.007–0.009)
4. Signal-to-noise is low (peak ~0.06 on [-0.07, 0.08] range) — consistent with contrastive encoders distributing information broadly

**Evidence:** research/results/exp01_clip_surgery/plots/ — frame heatmaps, summary bar chart, temporal evolution, duck-cup overlap scatter

**Implication:** The cup-conditioning problem is baked into the encoder from the start. SigLIP encodes everything with roughly equal weight, the representation is temporally static, and there's no clean spatial signal to exploit at the patch level. Only data diversity can address this — the encoder needs to see that cup position doesn't predict the correct action.

**Caveat (discovered after):** CLIP Surgery probes SigLIP's text-image alignment, but in SmolVLA the task prompt never goes through SigLIP. It goes through SmolVLM's language side, which cross-attends with the vision tokens. So CLIP Surgery tells us what SigLIP's text encoder thinks, not what the policy actually uses. The spatial attribution findings are still informative about the vision encoder's behavior, but don't capture how the prompt modulates attention in the full pipeline.

---

## Exp03: SigLIP Baseline — Raw Vision Encoder Spatial Structure (2026-04-08)

**Question:** Does SigLIP's raw output (before any text conditioning) show spatially meaningful variance patterns?

**Method:** Extracted 729×1152 SigLIP patch tokens from 55 real captures (3 datasets: duck-fixed/cup-moves, duck-moves/cup-fixed, both-move) across 4 cameras. Computed per-patch variance (L2 norm of per-dim std across captures) as 27×27 spatial heatmaps.

**Key Finding:** SigLIP's spatial variance cleanly tracks which objects move. "Duck fixed, cup moves" shows high variance where the cup is; "duck moves, cup fixed" shows variance where the duck is. UMAP yields 3 clean dataset clusters. Within-dataset cosine similarity is 0.997+.

**Evidence:** exp03_prompt_stability/plots/siglip_variance_top.png, siglip_umap_top.png, siglip_similarity_top.png

**Implication:** The raw vision encoder has spatially meaningful representations that respond to physical scene changes. This establishes that any text-based modulation should be measurable on top of this baseline.

---

## Exp03: Prompt-Conditioned VLM Token Analysis (2026-04-08) — NEGATIVE RESULT

**Question:** Does changing the task prompt change which image regions influence the VLM observation embedding? Specifically: does "pick up the duck" make the VLM ignore cup-position patches?

**Method:** Ran SmolVLA's full VLM prefix forward pass (SigLIP → connector → SmolVLM self-attention) on 55 real captures × 3 prompts ("pick up the duck and place it in the cup", "pick up the duck", "place it in the cup") × 4 cameras = 660 forward passes. Extracted per-camera vision tokens (64×960) after VLM self-attention with text. Analyzed: per-patch variance, prompt effect, UMAP, cosine similarity, prompt sensitivity.

**Key Finding:** The task prompt has **negligible effect** on VLM vision tokens.
- Variance maps are identical across prompts (same spatial structure for all 3 prompts per dataset)
- UMAP: 3 clusters by dataset, prompts completely interleaved within each cluster
- Prompt sensitivity: cosine distance between prompts per patch = 0.0001–0.0003 (effectively zero)
- Prompt effect (duck vs cup prompt variance difference): ±0.04 on a 4–16 scale

**Evidence:** exp03_prompt_stability/plots/vlm_variance_grid_top.png, vlm_umap_top.png, vlm_prompt_sensitivity_top.png, vlm_prompt_effect_top.png, vlm_similarity_by_prompt_top.png

**Implication:** Stage-aware prompting is NOT viable. The frozen VLM backbone (SmolVLM2-500M, never fine-tuned for robot tasks) doesn't route visual information based on task-specific language content. The observation representation is determined almost entirely by what the cameras see, not what the text says. Any prompt-dependent behavior must emerge in the downstream action expert's cross-attention, not in the observation embedding itself.
