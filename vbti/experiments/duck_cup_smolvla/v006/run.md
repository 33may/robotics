# v006

**Hardware**: RTX 4070 Ti Super (16GB)

---

## Attempt 1 — config LR 1e-4

**Started**: 2026-03-23 10:14 | **Killed**: 2026-03-23 ~10:35 | **Steps**: 800/30000

train_loss stuck at 0.055, rising as LR ramped up. Killed — thought LR was too aggressive.

---

## Attempt 2 — config LR 1e-5 (current run)

**Started**: 2026-03-23 ~10:40 | **Status**: running

Config change: `lr: 1e-4 → 1e-5`. Metrics cleared, fresh start.

### LR investigation

Spent time debugging why v006 converges slower than v004 (same dataset, same LR). Initially thought config LR was ignored — **wrong**. Config LR IS applied through `SmolVLAConfig(optimizer_lr=training_cfg.lr)` at model load (smolvla.py:65). Both v004 and v006 peak at 1e-5.

Real difference: **scheduler auto-scaling**. SmolVLA's `CosineDecayWithWarmupSchedulerConfig.build()` auto-scales when `total_steps < num_decay_steps(30000)`:
- v004 (10k steps): warmup compressed 1000→333, aggressive cosine over 10k
- v006 (30k steps): no scaling, full 1000 warmup, gentle cosine over 30k

This produces very different schedule shapes despite same peak LR.

### Training progress

| Step | train_loss | LR |
|------|-----------|-----|
| 1000 | 0.056 | 9.98e-6 (peak) |
| 2000 | 0.047 | 9.92e-6 |
| 3000 | 0.045 | 9.82e-6 |
| 4000 | 0.044 | 9.68e-6 |
| 4600 | 0.041 | 9.57e-6 |

Plateau forming around 0.04. v004 was at 0.005 at same step — 8x gap.

### Real-world eval — step_004000

**Despite 10x higher loss, v006 produces better real-world behavior than v004.** v004 was stuck escaping rest position. v006 shows more purposeful, smoother actions. The lower v004 loss was overfitting from aggressive LR decay, not better policy learning.

Okay the loss floor is settled at 0.04 and wiggling arouind this value, however we continue training since the performance of model improves

Will now eval it on the 12k checkpoint

