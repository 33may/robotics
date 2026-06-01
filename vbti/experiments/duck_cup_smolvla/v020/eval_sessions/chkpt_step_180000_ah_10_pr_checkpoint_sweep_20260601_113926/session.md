# Eval Session — chkpt_step_180000_ah_10_pr_checkpoint_sweep_20260601_113926

**Date:** 2026-06-01T11:39:26  
**Checkpoint:** `step_180000`  
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
- both_black: 2/5 (40%)
- both_red: 3/5 (60%)
- single_black: 3/5 (60%)
- single_red: 5/5 (100%)

## Per target color
- black: 5/10 (50%)
- red: 8/10 (80%)

## Failures
- Trial 07 [both_red] steps:645  prompt: "Pick up the duck and place it in the red cup"
- Trial 08 [both_red] steps:578  prompt: "Pick up the duck and place it in the red cup"
- Trial 10 [both_black] steps:1046  prompt: "Pick up the duck and place it in the black cup"
- Trial 11 [both_black] steps:750  prompt: "Pick up the duck and place it in the black cup"
- Trial 13 [both_black] steps:321  prompt: "Pick up the duck and place it in the black cup"
- Trial 18 [single_black] steps:1522  prompt: "Pick up the duck and place it in the black cup"
- Trial 19 [single_black] steps:631  prompt: "Pick up the duck and place it in the black cup"

## Videos
- Trial 00 (success): `trial_00_success.mp4`
- Trial 01 (success): `trial_01_success.mp4`
- Trial 02 (success): `trial_02_success.mp4`
- Trial 03 (success): `trial_03_success.mp4`
- Trial 04 (success): `trial_04_success.mp4`
- Trial 05 (success): `trial_05_success.mp4`
- Trial 06 (success): `trial_06_success.mp4`
- Trial 07 (failure): `trial_07_failure.mp4`
- Trial 08 (failure): `trial_08_failure.mp4`
- Trial 09 (success): `trial_09_success.mp4`
- Trial 10 (failure): `trial_10_failure.mp4`
- Trial 11 (failure): `trial_11_failure.mp4`
- Trial 12 (success): `trial_12_success.mp4`
- Trial 13 (failure): `trial_13_failure.mp4`
- Trial 14 (success): `trial_14_success.mp4`
- Trial 15 (success): `trial_15_success.mp4`
- Trial 16 (success): `trial_16_success.mp4`
- Trial 17 (success): `trial_17_success.mp4`
- Trial 18 (failure): `trial_18_failure.mp4`
- Trial 19 (failure): `trial_19_failure.mp4`

## Insights

_Fill in observations after reviewing videos / running eval-copilot._
