# Eval Session — chkpt_step_139950_ah_10_pr_checkpoint_sweep_20260601_111446

**Date:** 2026-06-01T11:14:46  
**Checkpoint:** `step_139950`  
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
- **Success rate:** 13/20 (65%)

## Per scene
- both_black: 3/5 (60%)
- both_red: 2/5 (40%)
- single_black: 5/5 (100%)
- single_red: 3/5 (60%)

## Per target color
- black: 8/10 (80%)
- red: 5/10 (50%)

## Failures
- Trial 02 [single_red] steps:868  prompt: "Pick up the duck and place it in the red cup"
- Trial 04 [single_red] steps:502  prompt: "Pick up the duck and place it in the red cup"
- Trial 05 [both_red] steps:1013  prompt: "Pick up the duck and place it in the red cup"
- Trial 08 [both_red] steps:956  prompt: "Pick up the duck and place it in the red cup"
- Trial 09 [both_red] steps:605  prompt: "Pick up the duck and place it in the red cup"
- Trial 10 [both_black] steps:1057  prompt: "Pick up the duck and place it in the black cup"
- Trial 14 [both_black] steps:1704  prompt: "Pick up the duck and place it in the black cup"

## Videos
- Trial 00 (success): `trial_00_success.mp4`
- Trial 01 (success): `trial_01_success.mp4`
- Trial 02 (failure): `trial_02_failure.mp4`
- Trial 03 (success): `trial_03_success.mp4`
- Trial 04 (failure): `trial_04_failure.mp4`
- Trial 05 (failure): `trial_05_failure.mp4`
- Trial 06 (success): `trial_06_success.mp4`
- Trial 07 (success): `trial_07_success.mp4`
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
- Trial 18 (success): `trial_18_success.mp4`
- Trial 19 (success): `trial_19_success.mp4`

## Insights

_Fill in observations after reviewing videos / running eval-copilot._
