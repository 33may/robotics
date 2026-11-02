# v011 — Notes

## Goal
475 eps merged (01+02+09+50eps_no37). Testing if more diverse real data breaks the generalization ceiling.

## Hypothesis
- v010 proved ~90% in-dist with 144 eps but <10% out-of-dist (boundary stickiness, cup-conditioning)
- 3.3x more data covering more positions should reduce spatial conditioning
- If this doesn't break through, next step is RL hardening, not more BC data

## Dataset
- `eternalmay33/01_02_09_00_merged` — 475 episodes, 235,123 frames
- Sources: 01 (43 eps), 02 (101 eps), 09-merged (281 eps), 50eps_no37 (50 eps)
- Note: patched robot_type on 01/02 from `so_follower` → `so101_follower` for merge compatibility
- 4 cameras: top, left, right, gripper

## Config (same as v010)
- LR: 1e-4, cosine decay, warmup 1000, decay_ratio 0.3
- batch_size: 64, bf16, 50k steps, save every 5k
- Sweet spot expected: 20k-30k based on v008/v010 evidence

## Known Risks
- 09-merged is 59% of data — its biases will dominate
- v008 flagged left-drift pattern in 09 at episode start
- Overfitting window may shift with more data

## Results
(pending)
