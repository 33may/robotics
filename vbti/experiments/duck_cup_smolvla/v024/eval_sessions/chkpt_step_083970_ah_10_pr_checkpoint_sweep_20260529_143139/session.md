# Eval Session — chkpt_step_083970_ah_10_pr_checkpoint_sweep_20260529_143139

**Date:** 2026-05-29T14:31:39  
**Checkpoint:** `step_083970`  
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
- single_black: 4/5 (80%)
- single_red: 3/5 (60%)

## Per target color
- black: 7/10 (70%)
- red: 4/10 (40%)

## Failures
- Trial 01 [single_red] steps:707  prompt: "Pick up the duck and place it in the red cup"
- Trial 03 [single_red] steps:390  prompt: "Pick up the duck and place it in the red cup"
- Trial 05 [both_red] steps:217  prompt: "Pick up the duck and place it in the red cup"
- Trial 07 [both_red] steps:730  prompt: "Pick up the duck and place it in the red cup"
- Trial 08 [both_red] steps:415  prompt: "Pick up the duck and place it in the red cup"
- Trial 09 [both_red] steps:192  prompt: "Pick up the duck and place it in the red cup"
- Trial 10 [both_black] steps:915  prompt: "Pick up the duck and place it in the black cup"
- Trial 14 [both_black] steps:522  prompt: "Pick up the duck and place it in the black cup"
- Trial 18 [single_black] steps:675  prompt: "Pick up the duck and place it in the black cup"

## Videos
- Trial 00 (success): `trial_00_success.mp4`
- Trial 01 (failure): `trial_01_failure.mp4`
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
- Trial 19 (success): `trial_19_success.mp4`

## Insights

_Fill in observations after reviewing videos / running eval-copilot._
