---
name: v018 vs v019 training comparison
description: SmolVLA v018 (175ep, 1 cup, depth_turbo) vs v019 (671ep, 2 cups + 3 tasks + padded depth) — train loss curves and what they imply
type: project
originSessionId: 92f50957-8ae3-4bb7-a2e5-c27a13fabe74
---
v018 (35k @ BS24) and v019 (50k @ BS48) are NOT directly comparable on the loss-vs-step axis. v019 saw only 7.83 epochs vs v018's 11.93, so it's likely **underfit** — when plotted vs epoch, the two curves overlap through epoch ~6. The 50k schedule is too short for the bigger v019 corpus.

Numbers:
- v018 final train_loss 0.0087, min 0.0055 @ 31.4k. Grad_norm median 0.19.
- v019 final train_loss 0.0144, min 0.0097 @ 22.4k. Grad_norm median 0.14 (calmer, more averaged gradients = diverse data).
- LR/sample identical at 3.125e-6 (linear scaling rule was applied correctly).

**Why:** Big-corpus SmolVLA fine-tuning needs proportional step count, not just step-count + LR scaling. With 4× the data v019 should have run ~75–90k steps to reach v018's epoch coverage.

**How to apply:** When sizing future schedules, target a fixed epoch count (~12 worked on v018), not a fixed step count. Check `train/epochs` in wandb summary at end of run — if < 10, extend.

Eval baseline as of 2026-05-07:
- v013 (single cup, no depth): 73% SR on id_scale_60 (44/60). Single-task is solved.
- v017 dual-cup no-depth: 37% (11/30). v018 dual-cup with depth: 47% (14/30) — depth gives +10pp.
- v019 not yet evaluated.

Plot: `vbti/experiments/duck_cup_smolvla/v019/v018_vs_v019_curves.png`
