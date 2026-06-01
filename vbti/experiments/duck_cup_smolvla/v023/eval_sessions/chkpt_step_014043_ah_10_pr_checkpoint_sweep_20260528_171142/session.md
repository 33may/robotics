# Eval Session — chkpt_step_014043_ah_10_pr_checkpoint_sweep_20260528_171142

**Date:** 2026-05-28T17:11:42  
**Checkpoint:** `step_014043`  
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
- **Success rate:** 4/20 (20%)

## Per scene
- both_black: 1/5 (20%)
- both_red: 0/5 (0%)
- single_black: 1/5 (20%)
- single_red: 2/5 (40%)

## Per target color
- black: 2/10 (20%)
- red: 2/10 (20%)

## Failures
- Trial 02 [single_red] steps:508  prompt: "Pick up the duck and place it in the red cup"
- Trial 03 [single_red] steps:596  prompt: "Pick up the duck and place it in the red cup"
- Trial 04 [single_red] steps:493  prompt: "Pick up the duck and place it in the red cup"
- Trial 05 [both_red] steps:768  prompt: "Pick up the duck and place it in the red cup"
- Trial 06 [both_red] steps:583  prompt: "Pick up the duck and place it in the red cup"
- Trial 07 [both_red] steps:382  prompt: "Pick up the duck and place it in the red cup"
- Trial 08 [both_red] steps:553  prompt: "Pick up the duck and place it in the red cup"
- Trial 09 [both_red] steps:392  prompt: "Pick up the duck and place it in the red cup"
- Trial 10 [both_black] steps:649  prompt: "Pick up the duck and place it in the black cup"
- Trial 11 [both_black] steps:542  prompt: "Pick up the duck and place it in the black cup"
- Trial 12 [both_black] steps:543  prompt: "Pick up the duck and place it in the black cup"
- Trial 13 [both_black] steps:471  prompt: "Pick up the duck and place it in the black cup"
- Trial 16 [single_black] steps:542  prompt: "Pick up the duck and place it in the black cup"
- Trial 17 [single_black] steps:660  prompt: "Pick up the duck and place it in the black cup"
- Trial 18 [single_black] steps:446  prompt: "Pick up the duck and place it in the black cup"
- Trial 19 [single_black] steps:602  prompt: "Pick up the duck and place it in the black cup"

## Videos
- Trial 00 (success): `trial_00_success.mp4`
- Trial 01 (success): `trial_01_success.mp4`
- Trial 02 (failure): `trial_02_failure.mp4`
- Trial 03 (failure): `trial_03_failure.mp4`
- Trial 04 (failure): `trial_04_failure.mp4`
- Trial 05 (failure): `trial_05_failure.mp4`
- Trial 06 (failure): `trial_06_failure.mp4`
- Trial 07 (failure): `trial_07_failure.mp4`
- Trial 08 (failure): `trial_08_failure.mp4`
- Trial 09 (failure): `trial_09_failure.mp4`
- Trial 10 (failure): `trial_10_failure.mp4`
- Trial 11 (failure): `trial_11_failure.mp4`
- Trial 12 (failure): `trial_12_failure.mp4`
- Trial 13 (failure): `trial_13_failure.mp4`
- Trial 14 (success): `trial_14_success.mp4`
- Trial 15 (success): `trial_15_success.mp4`
- Trial 16 (failure): `trial_16_failure.mp4`
- Trial 17 (failure): `trial_17_failure.mp4`
- Trial 18 (failure): `trial_18_failure.mp4`
- Trial 19 (failure): `trial_19_failure.mp4`

## Insights

_Fill in observations after reviewing videos / running eval-copilot._
