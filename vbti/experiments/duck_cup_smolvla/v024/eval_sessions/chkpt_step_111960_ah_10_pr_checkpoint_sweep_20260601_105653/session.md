# Eval Session — chkpt_step_111960_ah_10_pr_checkpoint_sweep_20260601_105653

**Date:** 2026-06-01T10:56:53  
**Checkpoint:** `step_111960`  
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
- both_black: 2/5 (40%)
- both_red: 3/5 (60%)
- single_black: 4/5 (80%)
- single_red: 4/5 (80%)

## Per target color
- black: 6/10 (60%)
- red: 7/10 (70%)

## Failures
- Trial 01 [single_red] steps:543  prompt: "Pick up the duck and place it in the red cup"
- Trial 05 [both_red] steps:913  prompt: "Pick up the duck and place it in the red cup"
- Trial 08 [both_red] steps:648  prompt: "Pick up the duck and place it in the red cup"
- Trial 11 [both_black] steps:788  prompt: "Pick up the duck and place it in the black cup"
- Trial 12 [both_black] steps:1402  prompt: "Pick up the duck and place it in the black cup"
- Trial 14 [both_black] steps:1078  prompt: "Pick up the duck and place it in the black cup"
- Trial 18 [single_black] steps:633  prompt: "Pick up the duck and place it in the black cup"

## Videos
- Trial 00 (success): `trial_00_success.mp4`
- Trial 01 (failure): `trial_01_failure.mp4`
- Trial 02 (success): `trial_02_success.mp4`
- Trial 03 (success): `trial_03_success.mp4`
- Trial 04 (success): `trial_04_success.mp4`
- Trial 05 (failure): `trial_05_failure.mp4`
- Trial 06 (success): `trial_06_success.mp4`
- Trial 07 (success): `trial_07_success.mp4`
- Trial 08 (failure): `trial_08_failure.mp4`
- Trial 09 (success): `trial_09_success.mp4`
- Trial 10 (success): `trial_10_success.mp4`
- Trial 11 (failure): `trial_11_failure.mp4`
- Trial 12 (failure): `trial_12_failure.mp4`
- Trial 13 (success): `trial_13_success.mp4`
- Trial 14 (failure): `trial_14_failure.mp4`
- Trial 15 (success): `trial_15_success.mp4`
- Trial 16 (success): `trial_16_success.mp4`
- Trial 17 (success): `trial_17_success.mp4`
- Trial 18 (failure): `trial_18_failure.mp4`
- Trial 19 (success): `trial_19_success.mp4`

## Insights

_Fill in observations after reviewing videos / running eval-copilot._
