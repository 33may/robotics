---
name: training_experiments
description: SmolVLA/GR00T training findings — LR behavior, loss vs policy quality, scheduler auto-scaling, env setup
type: project
---

## Key Finding: Loss ≠ Policy Quality in BC (2026-03-23)

Train loss is a bad proxy for behavior cloning policy quality. Confirmed by:
- Robomimic study: best val loss checkpoint is 50-100% worse than best performing policy
- Our v004 (loss 0.004) vs v006 (loss 0.041): v006 performs better on real robot despite 10x higher loss
- v006 has no resting bias (main v002/v003/v004 failure mode), v004 does

**Why:** v004's aggressive LR decay (cosine over 10k steps) caused overfitting/memorization. v006's gentle decay (cosine over 30k) learned a smoother policy.

**How to apply:** Don't chase low train_loss. Eval on real robot is the only reliable metric. Eval multiple checkpoints — policy quality can converge early while loss keeps moving.

## SmolVLA Scheduler Auto-Scaling

`CosineDecayWithWarmupSchedulerConfig.build()` auto-scales when `total_steps < num_decay_steps(30000)`:
- 10k steps → scale_factor=0.333, warmup 1000→333, aggressive decay
- 30k steps → no scaling, full warmup, gentle decay

This produces very different training dynamics despite same peak LR.

## Config LR Path in SmolVLA

Config `lr` IS applied via `SmolVLAConfig(optimizer_lr=training_cfg.lr)` at model init (smolvla.py:65). It's NOT ignored — the preset picks up this value. Scheduler params (warmup, decay_lr) also flow through the same path.

## flash-attn Build Fix (Fedora 42)

flash-attn's setup.py does `os.rename` across /tmp (tmpfs) and /home (ext4) = cross-device link error.
**Fix:** Set TMPDIR to same filesystem:
```bash
TMPDIR=/home/may33/tmp/build CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 \
  CUDA_HOME=/usr/local/cuda-12.9 TORCH_CUDA_ARCH_LIST="8.9" \
  pip install flash-attn==2.7.4.post1 --no-build-isolation \
  --cache-dir=/home/may33/tmp/pipcache
```

## Conda Environments

| Env | Python | Purpose | Key packages |
|-----|--------|---------|-------------|
| isaac | 3.11 | Isaac Sim, SmolVLA training, inference | torch 2.7.0+cu128, lerobot 0.4.3 |
| rbts | 3.10 | GR00T training | torch 2.7.1+cu128, gr00t 0.1.0, flash-attn 2.7.4, lerobot 0.4.4 |

## SmolVLA: 50 Episodes Should Be Enough

Official docs: ~50 episodes for basic pick-place. Reference used 50 eps across 5 positions (10 per position). 25 eps was insufficient. Quality > quantity — slow smooth demos matter more than count.
