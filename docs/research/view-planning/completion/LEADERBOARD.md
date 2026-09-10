# Completion bench — leaderboard (append-only)

Step set FROZEN to `half` (every 2nd step + final) for ALL iterations —
bench timing rule, GUIDE §3.0.3. `runs=N/9` = truth-annotated runs at
bench time; `no-truth` = no GT existed, metrics pending backfill.
Headline = min-over-runs worst-face coverage at final steps.
err_mm / halluc = mean over runs at each run's FINAL benched step.
Mid step = benched step nearest floor(0.5*(n_steps-1)), tie -> lower.

**BAR FROZEN (15:14, per GUIDE §1 freeze rule):** baseline poisson20
(iter 5: headline 0.148, min-mid 0.098, halluc 0.303) beats NO bar
number, so the bar freezes exactly as written: min-run worst-face >=
0.85 final AND >= 0.60 mid; mean halluc (final) <= 0.15; mean err
(final) <= 12 mm. No further movement, ever.

| iter | time | algo | hash | runs | steps | headline | mean wf final | mean wf mid | err_mm | halluc | s/step | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 14:43 | star | 18bbe0b6 | 0/9 | half | no-truth | no-truth | no-truth | no-truth | no-truth | ~1.5 | keep (renders) |
| 2 | 14:44 | boxsnap | e86f8de7 | 0/9 | half | no-truth | no-truth | no-truth | no-truth | no-truth | ~0.5 | pending-truth |
| 3 | 14:44 | planes | e86f8de7 | 0/9 | half | no-truth | no-truth | no-truth | no-truth | no-truth | ~0.5 | pending-truth |
| 4 | 15:14 | poisson6 | e86f8de7 | 9/9 | half | 0.121 | 0.428 | 0.138 | 7.5 | 0.318 | 0.30 | kill (control; dominated by poisson20) |
| 5 | 15:14 | poisson20 | e86f8de7 | 9/9 | half | 0.148 | 0.479 | 0.275 | 7.4 | 0.303 | 0.30 | keep (BASELINE; best mean wf_mid) |
| 6 | 15:14 | hull | e86f8de7 | 9/9 | half | 0.410 | 0.666 | 0.132 | 4.7 | 0.057 | <0.01 | keep (LEADER on headline) |
| 7 | 15:14 | star | e86f8de7 | 9/9 | half | 0.000 | 0.412 | 0.063 | 5.3 | 0.184 | 1.0 | kill (wavy walls > tau on noisy runs; 2 fix cycles spent in iter 1) |
| 8 | 15:14 | boxsnap | e86f8de7 | 9/9 | half | 0.312 | 0.826 | 0.176 | 2.2 | 0.066 | <0.01 | iterate (1.000 on 6/9 runs; extent overshoot b2/b3 from ghost points) |
| 9 | 15:14 | planes | e86f8de7 | 9/9 | half | 0.000 | 0.375 | 0.072 | 5.6 | 0.234 | <0.01 | kill (slab medians polluted by orthogonal faces; fix folded into next candidate) |
| 10 | 15:31 | boxguard | 861d9542 | 9/9 | half | 0.361 | 0.929 | 0.169 | 1.9 | 0.027 | 0.01 | parked (8/9 runs at 1.000; b2 x- unfixable within gates; shave gates OVERFIT-SUSPECT — memorized from 5 contested faces; hull keeps headline lead 0.410) |
| 11 | 15:42 | poisson6 | 5646f52e | 9/9 | half | 0.121 | 0.428 | 0.138 | 7.6 | 0.320 | 0.30 | rebench:seeded-sampler |
| 12 | 15:42 | poisson20 | 5646f52e | 9/9 | half | 0.148 | 0.479 | 0.275 | 7.4 | 0.301 | 0.30 | rebench:seeded-sampler |
| 13 | 15:42 | hull | 5646f52e | 9/9 | half | 0.410 | 0.666 | 0.132 | 4.7 | 0.059 | <0.01 | rebench:seeded-sampler |
| 14 | 15:42 | star | 5646f52e | 9/9 | half | 0.000 | 0.412 | 0.063 | 5.2 | 0.182 | 1.0 | rebench:seeded-sampler |
| 15 | 15:42 | boxsnap | 5646f52e | 9/9 | half | 0.312 | 0.826 | 0.176 | 2.2 | 0.068 | <0.01 | rebench:seeded-sampler |
| 16 | 15:42 | planes | 5646f52e | 9/9 | half | 0.000 | 0.375 | 0.072 | 5.5 | 0.228 | <0.01 | rebench:seeded-sampler |
| 17 | 15:42 | boxguard | 5646f52e | 9/9 | half | 0.361 | 0.929 | 0.169 | 1.9 | 0.027 | 0.01 | rebench:seeded-sampler (byte-identical across fresh processes verified) |

Rebench rows 11-17: the GUIDE §4 reproduction check caught unseeded
`sample_points_uniformly` in `_eval_mesh` (halluc jittered ~0.001
across processes). Sampler now seeded per call; coverage columns are
sampling-free and unchanged; halluc/err shifted <= 0.005. Rankings
and all verdicts unchanged.
