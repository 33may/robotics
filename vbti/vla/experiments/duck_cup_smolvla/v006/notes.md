# v006 — Notes

Single-dataset baseline with official SmolVLA LR schedule. Back to basics after v005's mixed-data plateau.

## Purpose

Establish a clean convergence baseline on `08-merged_trimmed` alone. v005 mixed 3 datasets (2 real + sim) and hit a loss floor at 0.050 — diagnosed as conflicting gradients from sim data (see [v005 analysis](../v005/notes.md#training-analysis-2026-03-23)). Before iterating on data mixing strategies, we need to know:
1. How fast does SmolVLA converge on this single dataset?
2. What's the achievable loss floor with no distribution conflict?
3. Does the official LR schedule (cosine decay, 1e-4 peak) behave differently from our previous 1e-5 runs?

## What changed from v005

| Parameter | v005 | v006 | Why |
|-----------|------|------|-----|
| datasets | 3 (2 real + sim, w=0.5) | 1 (08-merged_trimmed) | Remove sim noise source |
| lr | 1e-5 | 1e-5 | Same (config LR applied via SmolVLAConfig) |
| lr_schedule | WSD | cosine_decay_with_warmup | Official SmolVLA default |
| steps | 80,000 | 30,000 | Match official decay window |
| warmup | 1,000 | 1,000 | Same |

## Dataset

| | |
|---|---|
| repo_id | eternalmay33/08-merged_trimmed |
| episodes | 57 |
| frames | 22,331 |
| fps | 30 |
| cameras | top, left, right, gripper (480x640) |
| actions | 6 DoF (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper) |
| source | real |

- ~1,396 batches/epoch at batch_size 16
- 30k steps ≈ 21.5 epochs
- Dataset verified clean: no NaN/Inf, stats.json matches raw data, indices continuous

## LR Schedule

Config LR is applied through `SmolVLAConfig(optimizer_lr=training_cfg.lr)` at model init (line 65 of smolvla.py). The preset then uses this value, not the hardcoded default.

Actual schedule for v006 (30k steps, warmup=1000, lr=1e-5):
- Steps 0–1,000: linear warmup (0 → 1e-5)
- Steps 1,000–30,000: cosine decay (1e-5 → 2.5e-6)

Key difference from v004: the scheduler auto-scales when `total_steps < num_decay_steps`. v004 (10k steps) got warmup compressed to 333 steps and aggressive cosine decay. v006 (30k steps = preset default) gets no scaling — full warmup, gentle decay.

## v004 vs v006 — loss gap investigation

v006 train_loss ~0.04 at step 4600. v004 was at ~0.005 at the same step. Both use same dataset, same LR (1e-5), same code. Verified:
- [x] Config LR IS applied (via SmolVLAConfig constructor, not ignored)
- [x] Dataset unchanged (stats.json from Mar 18, data parquet from Mar 18)
- [x] No code changes affecting loss computation
- [x] LR logged values match simulation for v006
- [x] v004's logged LR consistent with 1e-5 peak, 500-step warmup (auto-scaled to 333)

Root cause: **scheduler auto-scaling**. v004 got compressed warmup (333 steps) + aggressive cosine decay over 10k steps. v006 gets full 1000-step warmup + gentle decay over 30k. The schedule shape is fundamentally different.

## Key finding: loss ≠ policy quality

Despite 10x higher train_loss, v006 step_004000 checkpoint produces **better real-world behavior** than v004 step_005000. Crucially, **v006 has no resting bias** — the main failure mode in v002/v003/v004 where the model gets stuck in rest position is gone. v004's low loss was likely overfitting — aggressive LR decay on small data = memorization of rest-heavy modes. v006's gentler schedule learns a smoother, more robust policy that actually moves.


![[v004_vs_v006_comparison.png]]


This reframes v005's 0.050 plateau too — the loss floor may not have been the problem. The sim data conflict (diagnosed per-dataset) was real, but the loss number alone is misleading for BC policy quality.



2,4 are nonsense

6 starts to get to the duck properly

8 turned out to be nonsense in eval too

10 didnt perform weel, a bit better then 8 but technically the same

12 also didn't perform well

