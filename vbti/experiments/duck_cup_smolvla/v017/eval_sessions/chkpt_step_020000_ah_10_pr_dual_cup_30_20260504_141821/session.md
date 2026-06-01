# Eval Session — chkpt_step_020000_ah_10_pr_dual_cup_30_20260504_141821

**Date:** 2026-05-04T14:18:21  
**Checkpoint:** `step_020000`  
**Protocol:** `dual_cup_30`  

## Config
- experiment: `duck_cup_smolvla`
- version: `v017`
- action_horizon: `10`
- max_steps: `10000`
- fps: `30`
- delta_actions: `False`
- detection: `False`
- enable_rtc: `True`
- record: `True`

## Overall
- **Success rate:** 11/30 (37%)

## Per scene
- both: 2/10 (20%)
- single_black: 5/10 (50%)
- single_red: 4/10 (40%)

## Per target color
- black: 6/15 (40%)
- red: 5/15 (33%)

## Dual-cup cells (target_color × side × closer)
- ('black', 'left', False): 0/1
- ('black', 'left', True): 0/1
- ('black', 'right', False): 1/1
- ('black', 'right', True): 0/2
- ('red', 'left', False): 1/2
- ('red', 'left', True): 0/1
- ('red', 'right', False): 0/1
- ('red', 'right', True): 0/1

## Failures
- Trial 00 [single_red] steps:869  prompt: "Pick up the duck and place it in the red cup"
- Trial 02 [single_red] steps:370  prompt: "Pick up the duck and place it in the red cup"
- Trial 03 [single_red] steps:327  prompt: "Pick up the duck and place it in the red cup"
- Trial 05 [single_red] steps:744  prompt: "Pick up the duck and place it in the red cup"
- Trial 06 [single_red] steps:311  prompt: "Pick up the duck and place it in the red cup"
- Trial 09 [single_red] steps:399  prompt: "Pick up the duck and place it in the red cup"
- Trial 10 [both] steps:372  prompt: "Pick up the duck and place it in the red cup"
- Trial 12 [both] steps:1174  prompt: "Pick up the duck and place it in the red cup"
- Trial 13 [both] steps:499  prompt: "Pick up the duck and place it in the red cup"
- Trial 14 [both] steps:1048  prompt: "Pick up the duck and place it in the red cup"
- Trial 15 [both] steps:803  prompt: "Pick up the duck and place it in the black cup"
- Trial 16 [both] steps:568  prompt: "Pick up the duck and place it in the black cup"
- Trial 17 [both] steps:498  prompt: "Pick up the duck and place it in the black cup"
- Trial 18 [both] steps:829  prompt: "Pick up the duck and place it in the black cup"
- Trial 20 [single_black] steps:492  prompt: "Pick up the duck and place it in the black cup"
- Trial 23 [single_black] steps:747  prompt: "Pick up the duck and place it in the black cup"
- Trial 25 [single_black] steps:638  prompt: "Pick up the duck and place it in the black cup"
- Trial 26 [single_black] steps:712  prompt: "Pick up the duck and place it in the black cup"
- Trial 29 [single_black] steps:440  prompt: "Pick up the duck and place it in the black cup"

## Videos
- Trial 00 (failure): `trial_00_failure.mp4`
- Trial 01 (success): `trial_01_success.mp4`
- Trial 02 (failure): `trial_02_failure.mp4`
- Trial 03 (failure): `trial_03_failure.mp4`
- Trial 04 (success): `trial_04_success.mp4`
- Trial 05 (failure): `trial_05_failure.mp4`
- Trial 06 (failure): `trial_06_failure.mp4`
- Trial 07 (success): `trial_07_success.mp4`
- Trial 08 (success): `trial_08_success.mp4`
- Trial 09 (failure): `trial_09_failure.mp4`
- Trial 10 (failure): `trial_10_failure.mp4`
- Trial 11 (success): `trial_11_success.mp4`
- Trial 12 (failure): `trial_12_failure.mp4`
- Trial 13 (failure): `trial_13_failure.mp4`
- Trial 14 (failure): `trial_14_failure.mp4`
- Trial 15 (failure): `trial_15_failure.mp4`
- Trial 16 (failure): `trial_16_failure.mp4`
- Trial 17 (failure): `trial_17_failure.mp4`
- Trial 18 (failure): `trial_18_failure.mp4`
- Trial 19 (success): `trial_19_success.mp4`
- Trial 20 (failure): `trial_20_failure.mp4`
- Trial 21 (success): `trial_21_success.mp4`
- Trial 22 (success): `trial_22_success.mp4`
- Trial 23 (failure): `trial_23_failure.mp4`
- Trial 24 (success): `trial_24_success.mp4`
- Trial 25 (failure): `trial_25_failure.mp4`
- Trial 26 (failure): `trial_26_failure.mp4`
- Trial 27 (success): `trial_27_success.mp4`
- Trial 28 (success): `trial_28_success.mp4`
- Trial 29 (failure): `trial_29_failure.mp4`

## Insights

_Fill in observations after reviewing videos / running eval-copilot._
