# Eval Session — chkpt_step_020000_ah_10_pr_dual_cup_15_20260507_103400

**Date:** 2026-05-07T10:34:00  
**Checkpoint:** `step_020000`  
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
- both: 3/5 (60%)
- single_black: 2/5 (40%)
- single_red: 5/5 (100%)

## Per target color
- black: 3/7 (43%)
- red: 7/8 (88%)

## Dual-cup cells (target_color × side × closer)
- ('black', 'left', False): 1/1
- ('black', 'right', True): 0/1
- ('red', 'left', True): 1/1
- ('red', 'right', False): 1/1
- ('red', 'right', True): 0/1

## Failures
- Trial 08 [both] steps:689  prompt: "Pick up the duck and place it in the black cup"
- Trial 09 [both] steps:540  prompt: "Pick up the duck and place it in the red cup"
- Trial 10 [single_black] steps:652  prompt: "Pick up the duck and place it in the black cup"
- Trial 13 [single_black] steps:489  prompt: "Pick up the duck and place it in the black cup"
- Trial 14 [single_black] steps:461  prompt: "Pick up the duck and place it in the black cup"

## Videos
- Trial 00 (success): `trial_00_success.mp4`
- Trial 01 (success): `trial_01_success.mp4`
- Trial 02 (success): `trial_02_success.mp4`
- Trial 03 (success): `trial_03_success.mp4`
- Trial 04 (success): `trial_04_success.mp4`
- Trial 05 (success): `trial_05_success.mp4`
- Trial 06 (success): `trial_06_success.mp4`
- Trial 07 (success): `trial_07_success.mp4`
- Trial 08 (failure): `trial_08_failure.mp4`
- Trial 09 (failure): `trial_09_failure.mp4`
- Trial 10 (failure): `trial_10_failure.mp4`
- Trial 11 (success): `trial_11_success.mp4`
- Trial 12 (success): `trial_12_success.mp4`
- Trial 13 (failure): `trial_13_failure.mp4`
- Trial 14 (failure): `trial_14_failure.mp4`

## Insights

_Fill in observations after reviewing videos / running eval-copilot._
