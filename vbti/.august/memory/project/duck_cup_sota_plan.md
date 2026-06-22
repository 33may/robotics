---
name: duck-cup SOTA plan (post-v020, 2026-05-11)
description: Current SOTA = v020@step150k, 93% on dual_cup_60. Hypothesis (unfreeze vision + augmentation + right-side data) validated. Next bottleneck = placement precision on single-cup; need harder protocol before v021.
type: project
originSessionId: 8f0cd7a3-6f05-494a-9eb5-6468840fe79b
---
Goal: SOTA on **pick duck → place in {red, black} cup, full-table coverage**.

## Where we stand 2026-05-11

| Version | Lever | Best result | Protocol |
|---|---|---|---|
| v017 | — | 37% | dual_cup |
| v018 | + depth | 47% | dual_cup |
| v019 | + 671ep + 3 tasks | 67% [49.5–80.5] @ 20k | dual_cup_30 |
| **v020** | **+ vision unfreeze + aug + right-side data** | **93% [84–97] @ 150k** | **dual_cup_60** |

v020 also hit **30/30 (100%) on dual_cup_30** @ step 150000. Non-overlapping CI vs v019 — real win. The right-side cell that broke v019 is solved.

## What v020 confirmed

Hypothesis from the prior plan validated:
- **Unfreeze vision** (vision_lr_scale=0.1) — single biggest lever. v019 plateau of 67% was the frozen-SigLIP color-discrimination ceiling. Unfreezing lifted it by ≥26pp.
- **Lighting-robust augmentation** (brightness/contrast/sharpness/affine, saturation+hue=0) — non-destructive vs cup color signal.
- **+94 right-side episodes** addressed v019's worst cell directly.

## Remaining failure mode (v020@150k, 4/60)

- All 4 failures = single-cup trials. Dual-cup = 20/20 → language conditioning works.
- 3/4 are single_black (small-n color skew hint).
- Failure step count mean 1239 vs success 425 — model retries placement, doesn't give up. **Failure = placement/drop precision**, not perception/language.

## Why

v019 → v020 jump is large enough that dual_cup_60 is now near saturation. Further training-side changes can't be measured on this protocol — CIs will be too wide vs the headline 93%.

## How to apply

When the user asks "what's next for duck-cup":
1. **First**, propose building a harder protocol (`dual_cup_hard`?) — extreme rotations, corner positions, occlusion, lighting variation. Don't propose v021 architecture changes before there's a protocol that can measure improvement vs v020's 93% ceiling.
2. **If placement precision is the target**, suggest action-chunking / RTC tuning or gripper/depth signal upgrades, not more language/vision data.
3. **GR00T N1.6** is still future-only — only consider after v020 is confirmed beaten or saturated under the harder protocol.

## References

- Session log: `vbti/sessions/sprint4/11-05-2026.md`
- Eval sessions: `vbti/experiments/duck_cup_smolvla/v020/eval_sessions/chkpt_step_150000_ah_10_pr_dual_cup_{30,60}_*`
- v020 notes: `vbti/experiments/duck_cup_smolvla/v020/notes.md` (## Evaluation results section)
