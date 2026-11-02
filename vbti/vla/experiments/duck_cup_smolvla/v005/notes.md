# v005 — Notes

3-dataset mix: 2 real + sim. Fresh start from smolvla_base. Long run to verify convergence.

## Hypothesis
Mixing 08-merged_trimmed (clean distribution) + so101_trimmed (ep37 removed) + sim data should:
1. Dilute the rest-bias from so101 (8% rest-zone → ~4% in mix)
2. Add diversity from sim (no rest bias, wider action range)
3. Train long enough (40k steps ≈ 9 epochs) for BC to converge

## Datasets

| dataset | eps | frames | SL<-85 | weight |
|---|---|---|---|---|
| 08-merged_trimmed | 57 | 22,331 | 1.1% | 1.0 |
| so101_trimmed (no ep37) | 50 | 18,439 | 8.2% | 1.0 |
| sim | 132 | 29,801 | 3.4% | 0.5 |
| **effective combined** | **239** | **~56k** | | |

- ~3,500 batches/epoch at batch_size 16
- 40k steps ≈ 11 epochs
- sim at weight 0.5 → contributes ~21% of effective training signal

## Checkpoint strategy
- save_freq: 2000 → 20 checkpoints (~18GB total)
- val_freq: 1000

## Training Analysis (2026-03-23)

### Plateau behavior
- Loss dropped fast during warmup (0.118 → 0.062 in 1k steps)
- Descent phase 1k–2.5k: train_loss 0.055 → 0.050 (-4.2% per 500 steps)
- **Plateau from step ~2500 onward**: train_loss stuck at 0.050 ± 0.001
- Micro-trend: 0.0526 → 0.0495 across plateau (6% over 4500 steps), then truly flat from step 5000
- val_loss: 0.046 (only 2 data points), comparable to v003's 0.043

### Cross-version comparison at ~5k steps

| Version | Dataset(s) | train_loss | val_loss |
|---------|-----------|------------|----------|
| v002 | 08-merged (1 real) | 0.004 | 0.003 |
| v003 | so101_trimmed (1 real) | 0.004 | 0.043 |
| v004 | 08-merged (1 real) | 0.005 | 0.006 |
| **v005** | **3 mixed** | **0.050** | **0.046** |

v005 train_loss 10x higher than single-dataset runs — expected for mixed data (model can't memorize 3 distributions). Val losses are comparable to v003, suggesting generalization is similar.

### LR schedule problem
Original config: cosine decay over 80k steps. At step 5.3k, LR barely moved (9.92e-6 vs peak 1e-5 — only 0.8% reduction). Compare v002 which was at 5.1e-6 (half peak) at the same step count. SmolVLA default is 30k decay steps, not 80k.

**Fix**: switched to WSD (Warmup-Stable-Decay) schedule:
- Steps 0–1k: warmup
- Steps 1k–56k: stable at 1e-5 (70%)
- Steps 56k–80k: cosine decay to 2.5e-6

Added `--reset_lr` flag to engine to support scheduler reset on resume.

### Loss spike diagnosis (from step_006000 checkpoint)

| Dataset | mean loss | p99 max | worst max |
|---------|-----------|---------|-----------|
| 08-merged (real) | 0.057 | 10.8 | 15.2 |
| so101_trimmed (real) | 0.052 | 10.9 | 33.9 |
| **sim** | **0.099** | **26.9** | **43.5** |

- Sim dataset is the main spike source: 2x mean loss, 2.5x spike severity vs real
- Two real datasets are well-matched — no conflict between them
- Worst sim episodes: 47, 114, 22, 20, 33, 56 (all max loss >30)
- 15% of all batches had max_loss >15, mostly from sim samples
- so101 ep 0 also problematic (max=33.9)

### Conclusion
The 0.050 plateau is a **loss floor** for this data mix — the sim distribution conflicts with real, creating noisy gradients. The model oscillates (std=0.001) rather than converging further.

**Recommendation for v006**: keep sim but drop 6 worst episodes (47,114,22,20,33,56), lower weight 0.5→0.25. Or try real-only to see if lower train_loss translates to better policy.