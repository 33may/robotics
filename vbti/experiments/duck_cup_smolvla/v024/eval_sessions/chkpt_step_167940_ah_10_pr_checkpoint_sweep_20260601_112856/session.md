# Eval Session — chkpt_step_167940_ah_10_pr_checkpoint_sweep_20260601_112856

**Date:** 2026-06-01T11:28:56  
**Checkpoint:** `step_167940`  
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
- **Success rate:** 14/20 (70%)

## Per scene
- both_black: 4/5 (80%)
- both_red: 3/5 (60%)
- single_black: 3/5 (60%)
- single_red: 4/5 (80%)

## Per target color
- black: 7/10 (70%)
- red: 7/10 (70%)

## Failures
- Trial 03 [single_red] steps:649  prompt: "Pick up the duck and place it in the red cup"
- Trial 07 [both_red] steps:1234  prompt: "Pick up the duck and place it in the red cup"
- Trial 09 [both_red] steps:657  prompt: "Pick up the duck and place it in the red cup"
- Trial 14 [both_black] steps:1027  prompt: "Pick up the duck and place it in the black cup"
- Trial 18 [single_black] steps:499  prompt: "Pick up the duck and place it in the black cup"
- Trial 19 [single_black] steps:5  prompt: "Pick up the duck and place it in the black cup"

## Videos
- Trial 00 (success): `trial_00_success.mp4`
- Trial 01 (success): `trial_01_success.mp4`
- Trial 02 (success): `trial_02_success.mp4`
- Trial 03 (failure): `trial_03_failure.mp4`
- Trial 04 (success): `trial_04_success.mp4`
- Trial 05 (success): `trial_05_success.mp4`
- Trial 06 (success): `trial_06_success.mp4`
- Trial 07 (failure): `trial_07_failure.mp4`
- Trial 08 (success): `trial_08_success.mp4`
- Trial 09 (failure): `trial_09_failure.mp4`
- Trial 10 (success): `trial_10_success.mp4`
- Trial 11 (success): `trial_11_success.mp4`
- Trial 12 (success): `trial_12_success.mp4`
- Trial 13 (success): `trial_13_success.mp4`
- Trial 14 (failure): `trial_14_failure.mp4`
- Trial 15 (success): `trial_15_success.mp4`
- Trial 16 (success): `trial_16_success.mp4`
- Trial 17 (success): `trial_17_success.mp4`
- Trial 18 (failure): `trial_18_failure.mp4`
- Trial 19 (failure): `trial_19_failure.mp4`

## Insights

_Fill in observations after reviewing videos / running eval-copilot._
