# Eval Session — chkpt_step_070215_ah_10_pr_checkpoint_sweep_20260529_134447

**Date:** 2026-05-29T13:44:47  
**Checkpoint:** `step_070215`  
**Protocol:** `checkpoint_sweep`  

## Config
- experiment: `duck_cup_smolvla`
- version: `v023`
- action_horizon: `10`
- max_steps: `10000`
- fps: `30`
- delta_actions: `False`
- detection: `False`
- enable_rtc: `True`
- record: `True`

## Overall
- **Success rate:** 6/20 (30%)

## Per scene
- both_black: 1/5 (20%)
- both_red: 0/5 (0%)
- single_black: 2/5 (40%)
- single_red: 3/5 (60%)

## Per target color
- black: 3/10 (30%)
- red: 3/10 (30%)

## Failures
- Trial 01 [single_red] steps:637  prompt: "Pick up the duck and place it in the red cup"
- Trial 03 [single_red] steps:460  prompt: "Pick up the duck and place it in the red cup"
- Trial 05 [both_red] steps:666  prompt: "Pick up the duck and place it in the red cup"
- Trial 06 [both_red] steps:464  prompt: "Pick up the duck and place it in the red cup"
- Trial 07 [both_red] steps:781  prompt: "Pick up the duck and place it in the red cup"
- Trial 08 [both_red] steps:359  prompt: "Pick up the duck and place it in the red cup"
- Trial 09 [both_red] steps:431  prompt: "Pick up the duck and place it in the red cup"
- Trial 11 [both_black] steps:404  prompt: "Pick up the duck and place it in the black cup"
- Trial 12 [both_black] steps:1017  prompt: "Pick up the duck and place it in the black cup"
- Trial 13 [both_black] steps:454  prompt: "Pick up the duck and place it in the black cup"
- Trial 14 [both_black] steps:409  prompt: "Pick up the duck and place it in the black cup"
- Trial 16 [single_black] steps:462  prompt: "Pick up the duck and place it in the black cup"
- Trial 18 [single_black] steps:531  prompt: "Pick up the duck and place it in the black cup"
- Trial 19 [single_black] steps:647  prompt: "Pick up the duck and place it in the black cup"

## Videos
- Trial 00 (success): `trial_00_success.mp4`
- Trial 01 (failure): `trial_01_failure.mp4`
- Trial 02 (success): `trial_02_success.mp4`
- Trial 03 (failure): `trial_03_failure.mp4`
- Trial 04 (success): `trial_04_success.mp4`
- Trial 05 (failure): `trial_05_failure.mp4`
- Trial 06 (failure): `trial_06_failure.mp4`
- Trial 07 (failure): `trial_07_failure.mp4`
- Trial 08 (failure): `trial_08_failure.mp4`
- Trial 09 (failure): `trial_09_failure.mp4`
- Trial 10 (success): `trial_10_success.mp4`
- Trial 11 (failure): `trial_11_failure.mp4`
- Trial 12 (failure): `trial_12_failure.mp4`
- Trial 13 (failure): `trial_13_failure.mp4`
- Trial 14 (failure): `trial_14_failure.mp4`
- Trial 15 (success): `trial_15_success.mp4`
- Trial 16 (failure): `trial_16_failure.mp4`
- Trial 17 (success): `trial_17_success.mp4`
- Trial 18 (failure): `trial_18_failure.mp4`
- Trial 19 (failure): `trial_19_failure.mp4`

## Insights

_Fill in observations after reviewing videos / running eval-copilot._
