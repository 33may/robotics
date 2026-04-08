# Research TODO

## Current Phase: Wrapping up — finishing Part 2 + writing paper

### In Progress
- [ ] Write complete paper into Obsidian research doc

### Up Next
- [ ] Final synthesis — compare SigLIP baseline vs VLM-processed results

### Skipped
- [~] Token ablation on SigLIP — skipped, CLIP Surgery pivot made it unnecessary
- [~] CLIP Surgery — partially useful (shows SigLIP spatial encoding is diffuse/static), but answers wrong question (probes SigLIP text encoder, not SmolVLM cross-attention). Pivoted to Exp02.

### Completed
- [x] Frame extraction pipeline — cached_frames/ with 47 shards from 01_02_black (3000 frames)
- [x] Masking library — gaussian noise + rectangular cutout, 5 fill modes each
- [x] Embedding analysis (run_masking_analysis.py) — UMAP, cosine violin, k-NN/MMD
- [x] Added 09_merged dataset (3000 frames) — records.npz has 44K embeddings total
- [x] FiftyOne exploration + image saving — images/ folder with clean + masked JPEGs
- [x] Attention rollout analysis — confirmed near-uniform on SigLIP (negative result)
- [x] Research on attribution methods — CLIP Surgery selected as primary method
- [x] Part 1 conclusion: image masking before frozen SigLIP is NOT viable as regularization
- [x] Exp03 Stage 0: SigLIP baseline — raw patch variance, UMAP, similarity
- [x] Exp03 Stage 1: VLM token extraction — 660 arrays (55×3×4), (64, 960) each
- [x] Exp03 Stage 1: VLM analysis — variance, UMAP, similarity, prompt sensitivity
- [x] Exp03 conclusion: prompt has negligible effect on VLM vision tokens — stage-aware prompting NOT viable
