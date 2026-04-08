# Exp01: CLIP Surgery Spatial Attribution — Findings

## What was implemented

CLIP Surgery on frozen SigLIP-so400m-patch14-384. Extracts 729 patch token embeddings from `last_hidden_state`, computes per-patch element-wise product with normalized text features, subtracts mean across prompts (redundancy removal), sums over feature dim to get spatial similarity maps.

6 text prompts: "a rubber duck", "a red cup", "a robot gripper", "a wooden table", "background wall", "Pick up the duck and place it in the cup"

5 frames sampled across trajectory (0, 21721, 43471, 65221, 86971) × 4 cameras.

## Architecture finding

SigLIP has NO separate visual/text projection layers. Both encoders output 1152-d natively. We use `last_hidden_state` (post-layernorm, pre-head) for patches and `pooler_output` (post-head) for text. These are approximately but not perfectly aligned — contributes to noisy signal.

## Key observations

### 1. Task instruction prompt dominates (not useful spatially)
"Pick up the duck and place it in the cup" has ~3× the mean positive similarity (0.033) vs individual objects (0.008–0.014). It activates uniformly across all patches — not spatially discriminative. This is expected: full-sentence prompts match the global scene embedding, not specific regions.

### 2. Robot gripper is most salient individual object
0.013–0.014 mean positive similarity across all cameras. Beats duck (0.009–0.011) and cup (0.007–0.009). The gripper is visually prominent and moves across the scene — the encoder picks it up strongly.

### 3. Duck and cup are partially separated spatially
Correlation between duck and cup surgery maps: r ≈ 0.02–0.35 across frames. They're somewhat spatially distinct but with significant overlap. Neither has a clean, localized hotspot — the signal is distributed across many patches.

### 4. Temporal evolution is flat
Attribution values for each prompt barely change across the 5 trajectory timepoints. SigLIP's spatial encoding is static — it doesn't shift attention based on manipulation phase (approach vs grasp vs transport).

### 5. Signal-to-noise is low
Peak surgery similarity ~0.06 on a range of [-0.07, 0.08]. The heatmaps show structure but are noisy. This is consistent with working in pre-head feature space and with contrastive encoders distributing information broadly.

### 6. Raw cosine (no surgery) is less discriminative
Without the redundancy subtraction, all prompts light up similar regions. Confirms the surgery correction is necessary for any spatial discrimination.

## Limitations

- Patch features from `last_hidden_state` are pre-attention-pooling-head, while text features from `pooler_output` are post-head. Not perfectly aligned spaces.
- Only 5 frames sampled — sufficient for qualitative conclusions but not for statistical claims.
- CLIP Surgery was designed for CLIP, not SigLIP specifically. The method transfers but with reduced signal quality.

## Interpretation for the research

The frozen SigLIP encoder does partially separate objects spatially at the patch level, but the separation is weak and noisy. Cup and duck information are distributed across overlapping patch regions. The encoding is temporally static — the encoder doesn't "know" which manipulation phase we're in.

This means: the cup-conditioning problem is baked into the encoder's representation from the start. The encoder encodes everything in the scene with roughly equal weight, and the representation doesn't change based on task phase. Only data diversity (showing the policy that cup position doesn't matter) can address this — architectural tricks at the encoder level won't help because there's no clean spatial signal to exploit.
