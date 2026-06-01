# Eval Session — chkpt_step_028128_ah_10_pr_checkpoint_sweep_20260528_162706

**Date:** 2026-05-28T16:27:06  
**Checkpoint:** `step_028128`  
**Protocol:** `checkpoint_sweep`  

## Config
- experiment: `duck_cup_smolvla`
- version: `v022`
- action_horizon: `10`
- max_steps: `10000`
- fps: `30`
- delta_actions: `False`
- detection: `False`
- enable_rtc: `True`
- record: `True`

## Overall
- **Success rate:** 4/20 (20%)

## Per scene
- both_black: 2/5 (40%)
- both_red: 1/5 (20%)
- single_black: 0/5 (0%)
- single_red: 1/5 (20%)

## Per target color
- black: 2/10 (20%)
- red: 2/10 (20%)

## Failures
- Trial 01 [single_red] steps:437  prompt: "Pick up the duck and place it in the red cup"
- Trial 02 [single_red] steps:574  prompt: "Pick up the duck and place it in the red cup"
- Trial 03 [single_red] steps:325  prompt: "Pick up the duck and place it in the red cup"
- Trial 04 [single_red] steps:699  prompt: "Pick up the duck and place it in the red cup"
- Trial 05 [both_red] steps:393  prompt: "Pick up the duck and place it in the red cup"
- Trial 06 [both_red] steps:363  prompt: "Pick up the duck and place it in the red cup"
- Trial 07 [both_red] steps:495  prompt: "Pick up the duck and place it in the red cup"
- Trial 09 [both_red] steps:266  prompt: "Pick up the duck and place it in the red cup"
- Trial 10 [both_black] steps:245  prompt: "Pick up the duck and place it in the black cup"
- Trial 11 [both_black] steps:349  prompt: "Pick up the duck and place it in the black cup"
- Trial 13 [both_black] steps:245  prompt: "Pick up the duck and place it in the black cup"
- Trial 15 [single_black] steps:330  prompt: "Pick up the duck and place it in the black cup"
- Trial 16 [single_black] steps:343  prompt: "Pick up the duck and place it in the black cup"
- Trial 17 [single_black] steps:399  prompt: "Pick up the duck and place it in the black cup"
- Trial 18 [single_black] steps:236  prompt: "Pick up the duck and place it in the black cup"
- Trial 19 [single_black] steps:771  prompt: "Pick up the duck and place it in the black cup"

## Videos
- Trial 00 (success): `trial_00_success.mp4`
- Trial 01 (failure): `trial_01_failure.mp4`
- Trial 02 (failure): `trial_02_failure.mp4`
- Trial 03 (failure): `trial_03_failure.mp4`
- Trial 04 (failure): `trial_04_failure.mp4`
- Trial 05 (failure): `trial_05_failure.mp4`
- Trial 06 (failure): `trial_06_failure.mp4`
- Trial 07 (failure): `trial_07_failure.mp4`
- Trial 08 (success): `trial_08_success.mp4`
- Trial 09 (failure): `trial_09_failure.mp4`
- Trial 10 (failure): `trial_10_failure.mp4`
- Trial 11 (failure): `trial_11_failure.mp4`
- Trial 12 (success): `trial_12_success.mp4`
- Trial 13 (failure): `trial_13_failure.mp4`
- Trial 14 (success): `trial_14_success.mp4`
- Trial 15 (failure): `trial_15_failure.mp4`
- Trial 16 (failure): `trial_16_failure.mp4`
- Trial 17 (failure): `trial_17_failure.mp4`
- Trial 18 (failure): `trial_18_failure.mp4`
- Trial 19 (failure): `trial_19_failure.mp4`

## Insights

_Fill in observations after reviewing videos / running eval-copilot._
