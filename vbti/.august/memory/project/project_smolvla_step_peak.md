---
name: project_smolvla_step_peak
description: SmolVLA fine-tuning on duck_cup datasets peaks at 20–25k steps with the v013/v014 cosine schedule (50k total, lr=1e-4, BS=32) — train shorter
type: project
originSessionId: 672e223a-9b37-4a53-84d3-29a68457f36d
---
SmolVLA fine-tuning on the duck_cup_smolvla family consistently produced its best evaluation checkpoint between **20k and 25k steps** under the v013/v014 schedule (50 000 steps, batch 32, lr 1e-4, cosine decay → 2.5e-6, warmup 1000). v014's sensitivity ablation explicitly used `lerobot_output_r1/checkpoints/020000/pretrained_model`. Later checkpoints (>25k) were observed to be worse — the training is overfitting on the second half of the cosine schedule.

**Why:** dataset is small (~70k–135k frames) relative to model capacity, so the policy memorizes after ~20 epochs even with a frozen vision encoder. v017/v018 inherit the same regime.

**How to apply:**
- For new SmolVLA fine-tuning runs on this task family, default to **35k steps** (captures the peak with margin past 25k to confirm the pattern still holds, saves ~30% wall time vs 50k).
- Save cadence at 5000 steps gives 7 candidates (5/10/15/20/25/30/35k) — pick best by val loss / real-robot eval.
- If the dataset grows substantially (e.g. > 200k frames), revisit — more data will push the peak later.
- LR + warmup + decay shape stays the same; only `steps` shrinks. Cosine schedule will compress proportionally so the model reaches the LR floor faster, which is a feature not a bug.
