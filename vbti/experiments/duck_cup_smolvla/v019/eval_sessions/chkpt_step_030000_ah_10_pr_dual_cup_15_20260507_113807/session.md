# Eval Session — chkpt_step_030000_ah_10_pr_dual_cup_15_20260507_113807

**Date:** 2026-05-07T11:38:07  
**Checkpoint:** `step_030000`  
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
- **Success rate:** 6/15 (40%)

## Per scene
- both: 1/5 (20%)
- single_black: 3/5 (60%)
- single_red: 2/5 (40%)

## Per target color
- black: 3/7 (43%)
- red: 3/8 (38%)

## Dual-cup cells (target_color × side × closer)
- ('black', 'left', False): 0/1
- ('black', 'right', True): 0/1
- ('red', 'left', True): 1/1
- ('red', 'right', False): 0/1
- ('red', 'right', True): 0/1

## Failures
- Trial 00 [single_red] steps:392  prompt: "Pick up the duck and place it in the red cup"
- Trial 01 [single_red] steps:883  prompt: "Pick up the duck and place it in the red cup"
- Trial 03 [single_red] steps:314  prompt: "Pick up the duck and place it in the red cup"
- Trial 06 [both] steps:698  prompt: "Pick up the duck and place it in the red cup"
- Trial 07 [both] steps:413  prompt: "Pick up the duck and place it in the black cup"
- Trial 08 [both] steps:333  prompt: "Pick up the duck and place it in the black cup"
- Trial 09 [both] steps:376  prompt: "Pick up the duck and place it in the red cup"
- Trial 10 [single_black] steps:522  prompt: "Pick up the duck and place it in the black cup"
- Trial 13 [single_black] steps:599  prompt: "Pick up the duck and place it in the black cup"

## Videos
- Trial 00 (failure): `trial_00_failure.mp4`
- Trial 01 (failure): `trial_01_failure.mp4`
- Trial 02 (success): `trial_02_success.mp4`
- Trial 03 (failure): `trial_03_failure.mp4`
- Trial 04 (success): `trial_04_success.mp4`
- Trial 05 (success): `trial_05_success.mp4`
- Trial 06 (failure): `trial_06_failure.mp4`
- Trial 07 (failure): `trial_07_failure.mp4`
- Trial 08 (failure): `trial_08_failure.mp4`
- Trial 09 (failure): `trial_09_failure.mp4`
- Trial 10 (failure): `trial_10_failure.mp4`
- Trial 11 (success): `trial_11_success.mp4`
- Trial 12 (success): `trial_12_success.mp4`
- Trial 13 (failure): `trial_13_failure.mp4`
- Trial 14 (success): `trial_14_success.mp4`

## Insights

_Fill in observations after reviewing videos / running eval-copilot._
