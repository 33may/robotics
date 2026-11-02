# Research Flow Journal

## 2026-04-07 — decision: research direction established
Started with two hypotheses: (1) image masking as regularization, (2) embedding dropout. Decided to test masking first since it's stronger in principle — full spatial exclusion vs distributed dropout.

## 2026-04-07 — insight: cosine similarity is misleading
Cosine similarities are all 0.94+ which sounds close, but UMAP reveals clear structural separation. High cosine sim in 1152-d space doesn't mean in-distribution. The manifold structure matters more than the scalar distance.

## 2026-04-07 — insight: camera-specific patterns
Left and right cameras produce similar cluster shapes and island structures (symmetric camera placement). Gripper camera has distinct structure — close-up view means masking affects it differently.

## 2026-04-07 — insight: gaussian noise connecting lines
The "connecting lines" in UMAP between clean clusters and masked points are paths toward noise attractor points — embeddings of fully corrupted images. Not useful signal, just the geometry of increasing corruption.

## 2026-04-07 — dead_end: attention rollout on contrastive ViTs
Attention rollout produces uniform heatmaps on SigLIP. Expected in hindsight — contrastive training distributes attention broadly, unlike classifiers that focus. Mean pooling (no CLS) diffuses attention further.

## 2026-04-07 — decision: CLIP Surgery as primary attribution method
After researching alternatives (GradCAM, Chefer, Grad-ECLIP, token ablation), selected CLIP Surgery for simplicity (1 forward pass, no gradients) and direct applicability (text-conditioned spatial maps). Token ablation as ground-truth validation.

## 2026-04-08 — decision: wrapping up today
Goal: implement CLIP Surgery, run spatial attribution analysis, write complete research paper. All Part 1 data is in. Focus on Part 2 execution + full writeup.

## 2026-04-08 — insight: CLIP Surgery results — partial accept
CLIP Surgery shows spatial structure but signal is noisy. Key reason: using last_hidden_state (pre-head) for patches but pooler_output (post-head) for text — different levels of processing. The method was designed for CLIP which has explicit projection layers; SigLIP doesn't.

Despite noise, the qualitative findings are solid: encoding is temporally static, duck/cup partially overlap spatially, gripper dominates attention. These are meaningful conclusions for the research.

## 2026-04-08 — decision: skip token ablation
Token ablation would validate CLIP Surgery findings but not change conclusions. Given we're wrapping up today, the evidence is sufficient. The main research questions are answered:
1. Image masking → OOD, not viable (Part 1)
2. SigLIP encodes everything broadly, static across trajectory → data diversity is the fix (Part 2)

## 2026-04-08 — insight: cross-experiment synthesis
Both parts converge on the same conclusion: the frozen SigLIP encoder treats the entire scene holistically. You can't trick it (masking produces OOD), and its internal representations don't offer clean handles for intervention (spatial attribution is diffuse). The only lever is what you show it during training — data diversity.

## 2026-04-08 — pivot: CLIP Surgery answers the wrong question
Realized CLIP Surgery probes SigLIP's text-image alignment, but in SmolVLA the text prompt never goes through SigLIP. The actual pipeline is: SigLIP (vision only) → patch tokens → SmolVLM (cross-attends patches with task text) → observation embedding → action head. The text influences which vision patches matter at the SmolVLM level, not SigLIP.

This opens a much more interesting question: does the task prompt control which image regions influence the observation embedding? If "pick up the duck" makes SmolVLM ignore cup patches → stage-aware prompting could fix the cup-conditioning problem without any data changes.

## 2026-04-08 — decision: Exp02 — prompt-conditioned embedding stability
New experiment: capture real images with duck in fixed position, cup in varied positions. Run full SmolVLA pipeline with different prompts ("pick up the duck and place it in the cup" vs "pick up the duck" vs "place it in the cup"). Measure how stable the observation embedding is across cup positions for each prompt. If "pick up the duck" is stable → stage-aware prompting works.

## 2026-04-08 — insight: SigLIP baseline shows spatial variance tracks object movement
Raw SigLIP tokens (729 patches, 27×27 grid) show per-patch variance that spatially tracks which objects move in each dataset. "Duck fixed, cup moves" has high variance in cup region; "duck moves, cup fixed" has variance where duck is. UMAP shows 3 clean clusters by dataset. Cosine similarity within datasets is 0.997+. The encoder's spatial structure is meaningful and measurable.

## 2026-04-08 — insight: VLM connector compresses 729→64 patches (8×8 grid)
The SmolVLM connector downsamples SigLIP's 27×27 patch grid to 8×8. This is an 11× spatial compression. Analysis must work at 8×8 resolution for VLM-processed tokens.

## 2026-04-08 — surprise: prompt has ZERO effect on VLM vision tokens
VLM token extraction completed: 55 captures × 3 prompts × 4 cameras = 660 token arrays, each (64, 960).

Results across all 5 analysis plots:
1. Variance maps: identical across prompts within each dataset
2. Prompt effect (duck vs cup prompt): ±0.04 on scale of 4-16 — noise
3. UMAP: 3 clusters by dataset, prompts completely interleaved within each
4. Cosine similarity: identical matrices across prompts (~0.95+)
5. Prompt sensitivity: cosine distance between prompts per patch = 0.0001-0.0003

The frozen VLM backbone processes images essentially identically regardless of task prompt. The text tokens pass through self-attention alongside vision tokens, but the pretrained weights (never fine-tuned for robot tasks) don't learn to route visual information based on task-specific language.

## 2026-04-08 — decision: stage-aware prompting is NOT viable
Since the prompt doesn't change which image regions matter in the VLM's vision tokens, changing the prompt mid-trajectory (e.g., "pick up the duck" during grasping) won't make the model ignore cup position. The text influence on vision representation is negligible in the frozen VLM. Only the action expert head downstream has to deal with this — and it works via cross-attention on the cached KV pairs, which are prompt-invariant.
