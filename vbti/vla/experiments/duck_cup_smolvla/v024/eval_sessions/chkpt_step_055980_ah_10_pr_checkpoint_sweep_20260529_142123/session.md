# Eval Session — chkpt_step_055980_ah_10_pr_checkpoint_sweep_20260529_142123

**Date:** 2026-05-29T14:21:23  
**Checkpoint:** `step_055980`  
**Protocol:** `checkpoint_sweep`  

## Config
- experiment: `duck_cup_smolvla`
- version: `v024`
- action_horizon: `10`
- max_steps: `10000`
- fps: `30`
- delta_actions: `False`
- detection: `False`
- enable_rtc: `True`
- record: `True`

## Overall
- **Success rate:** 11/20 (55%)

## Per scene
- both_black: 3/5 (60%)
- both_red: 1/5 (20%)
- single_black: 3/5 (60%)
- single_red: 4/5 (80%)

## Per target color
- black: 6/10 (60%)
- red: 5/10 (50%)

## Failures
- Trial 03 [single_red] steps:428  prompt: "Pick up the duck and place it in the red cup"
- Trial 05 [both_red] steps:764  prompt: "Pick up the duck and place it in the red cup"
- Trial 07 [both_red] steps:587  prompt: "Pick up the duck and place it in the red cup"
- Trial 08 [both_red] steps:685  prompt: "Pick up the duck and place it in the red cup"
- Trial 09 [both_red] steps:329  prompt: "Pick up the duck and place it in the red cup"
- Trial 10 [both_black] steps:308  prompt: "Pick up the duck and place it in the black cup"
- Trial 14 [both_black] steps:749  prompt: "Pick up the duck and place it in the black cup"
- Trial 18 [single_black] steps:781  prompt: "Pick up the duck and place it in the black cup"
- Trial 19 [single_black] steps:350  prompt: "Pick up the duck and place it in the black cup"

## Videos
- Trial 00 (success): `trial_00_success.mp4`
- Trial 01 (success): `trial_01_success.mp4`
- Trial 02 (success): `trial_02_success.mp4`
- Trial 03 (failure): `trial_03_failure.mp4`
- Trial 04 (success): `trial_04_success.mp4`
- Trial 05 (failure): `trial_05_failure.mp4`
- Trial 06 (success): `trial_06_success.mp4`
- Trial 07 (failure): `trial_07_failure.mp4`
- Trial 08 (failure): `trial_08_failure.mp4`
- Trial 09 (failure): `trial_09_failure.mp4`
- Trial 10 (failure): `trial_10_failure.mp4`
- Trial 11 (success): `trial_11_success.mp4`
- Trial 12 (success): `trial_12_success.mp4`
- Trial 13 (success): `trial_13_success.mp4`
- Trial 14 (failure): `trial_14_failure.mp4`
- Trial 15 (success): `trial_15_success.mp4`
- Trial 16 (success): `trial_16_success.mp4`
- Trial 17 (success): `trial_17_success.mp4`
- Trial 18 (failure): `trial_18_failure.mp4`
- Trial 19 (failure): `trial_19_failure.mp4`

## Insights

_Fill in observations after reviewing videos / running eval-copilot._
