# Eval Session — chkpt_step_110000_ah_10_pr_checkpoint_sweep_20260601_115100

**Date:** 2026-06-01T11:51:00  
**Checkpoint:** `step_110000`  
**Protocol:** `checkpoint_sweep`  

## Config
- experiment: `duck_cup_smolvla`
- version: `v020`
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
- both_red: 3/5 (60%)
- single_black: 2/5 (40%)
- single_red: 5/5 (100%)

## Per target color
- black: 5/10 (50%)
- red: 8/10 (80%)

## Failures
- Trial 08 [both_red] steps:657  prompt: "Pick up the duck and place it in the red cup"
- Trial 09 [both_red] steps:660  prompt: "Pick up the duck and place it in the red cup"
- Trial 11 [both_black] steps:561  prompt: "Pick up the duck and place it in the black cup"
- Trial 14 [both_black] steps:926  prompt: "Pick up the duck and place it in the black cup"
- Trial 16 [single_black] steps:1092  prompt: "Pick up the duck and place it in the black cup"
- Trial 18 [single_black] steps:1841  prompt: "Pick up the duck and place it in the black cup"
- Trial 19 [single_black] steps:1656  prompt: "Pick up the duck and place it in the black cup"

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
- Trial 10 (success): `trial_10_success.mp4`
- Trial 11 (failure): `trial_11_failure.mp4`
- Trial 12 (success): `trial_12_success.mp4`
- Trial 13 (success): `trial_13_success.mp4`
- Trial 14 (failure): `trial_14_failure.mp4`
- Trial 15 (success): `trial_15_success.mp4`
- Trial 16 (failure): `trial_16_failure.mp4`
- Trial 17 (success): `trial_17_success.mp4`
- Trial 18 (failure): `trial_18_failure.mp4`
- Trial 19 (failure): `trial_19_failure.mp4`

## Insights

_Fill in observations after reviewing videos / running eval-copilot._
