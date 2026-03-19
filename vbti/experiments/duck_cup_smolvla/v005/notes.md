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