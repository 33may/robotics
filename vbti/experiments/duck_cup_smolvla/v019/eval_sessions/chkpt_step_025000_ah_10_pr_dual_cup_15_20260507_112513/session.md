# Eval Session — chkpt_step_025000_ah_10_pr_dual_cup_15_20260507_112513

**Date:** 2026-05-07T11:25:13  
**Checkpoint:** `step_025000`  
**Protocol:** `dual_cup_15`  

## Config
- experiment: `duck_cup_smolvla`
- version: `v019`
- action_horizon: `10`
- max_steps: `10000`
- fps: `30`
- delta_actions: `False`
- detection: `False`
- enable_rtc: `True`
- record: `True`

## Overall
- **Success rate:** 10/15 (67%)

## Per scene
- both: 1/5 (20%)
- single_black: 5/5 (100%)
- single_red: 4/5 (80%)

## Per target color
- black: 6/7 (86%)
- red: 4/8 (50%)

## Dual-cup cells (target_color × side × closer)
- ('black', 'left', False): 1/1
- ('black', 'right', True): 0/1
- ('red', 'left', True): 0/1
- ('red', 'right', False): 0/1
- ('red', 'right', True): 0/1

## Failures
- Trial 03 [single_red] steps:362  prompt: "Pick up the duck and place it in the red cup"
- Trial 05 [both] steps:379  prompt: "Pick up the duck and place it in the red cup"
- Trial 06 [both] steps:451  prompt: "Pick up the duck and place it in the red cup"
- Trial 08 [both] steps:668  prompt: "Pick up the duck and place it in the black cup"
- Trial 09 [both] steps:694  prompt: "Pick up the duck and place it in the red cup"

## Videos
- Trial 00 (success): `trial_00_success.mp4`
- Trial 01 (success): `trial_01_success.mp4`
- Trial 02 (success): `trial_02_success.mp4`
- Trial 03 (failure): `trial_03_failure.mp4`
- Trial 04 (success): `trial_04_success.mp4`
- Trial 05 (failure): `trial_05_failure.mp4`
- Trial 06 (failure): `trial_06_failure.mp4`
- Trial 07 (success): `trial_07_success.mp4`
- Trial 08 (failure): `trial_08_failure.mp4`
- Trial 09 (failure): `trial_09_failure.mp4`
- Trial 10 (success): `trial_10_success.mp4`
- Trial 11 (success): `trial_11_success.mp4`
- Trial 12 (success): `trial_12_success.mp4`
- Trial 13 (success): `trial_13_success.mp4`
- Trial 14 (success): `trial_14_success.mp4`

## Insights

_Fill in observations after reviewing videos / running eval-copilot._
