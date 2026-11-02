# Eval Session — chkpt_step_150000_ah_10_pr_dual_cup_60_20260511_135739

**Date:** 2026-05-11T13:57:39  
**Checkpoint:** `step_150000`  
**Protocol:** `dual_cup_60`  

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
- **Success rate:** 56/60 (93%)

## Per scene
- both: 20/20 (100%)
- single_black: 17/20 (85%)
- single_red: 19/20 (95%)

## Per target color
- black: 27/30 (90%)
- red: 29/30 (97%)

## Dual-cup cells (target_color × side × closer)
- ('black', 'left', False): 2/2
- ('black', 'left', True): 3/3
- ('black', 'right', False): 2/2
- ('black', 'right', True): 3/3
- ('red', 'left', False): 3/3
- ('red', 'left', True): 2/2
- ('red', 'right', False): 3/3
- ('red', 'right', True): 2/2

## Failures
- Trial 11 [single_red] steps:954  prompt: "Pick up the duck and place it in the red cup"
- Trial 43 [single_black] steps:1194  prompt: "Pick up the duck and place it in the black cup"
- Trial 45 [single_black] steps:782  prompt: "Pick up the duck and place it in the black cup"
- Trial 51 [single_black] steps:2026  prompt: "Pick up the duck and place it in the black cup"

## Videos
- Trial 00 (success): `trial_00_success.mp4`
- Trial 01 (success): `trial_01_success.mp4`
- Trial 02 (success): `trial_02_success.mp4`
- Trial 03 (success): `trial_03_success.mp4`
- Trial 04 (success): `trial_04_success.mp4`
- Trial 05 (success): `trial_05_success.mp4`
- Trial 06 (success): `trial_06_success.mp4`
- Trial 07 (success): `trial_07_success.mp4`
- Trial 08 (success): `trial_08_success.mp4`
- Trial 09 (success): `trial_09_success.mp4`
- Trial 10 (success): `trial_10_success.mp4`
- Trial 11 (failure): `trial_11_failure.mp4`
- Trial 12 (success): `trial_12_success.mp4`
- Trial 13 (success): `trial_13_success.mp4`
- Trial 14 (success): `trial_14_success.mp4`
- Trial 15 (success): `trial_15_success.mp4`
- Trial 16 (success): `trial_16_success.mp4`
- Trial 17 (success): `trial_17_success.mp4`
- Trial 18 (success): `trial_18_success.mp4`
- Trial 19 (success): `trial_19_success.mp4`
- Trial 20 (success): `trial_20_success.mp4`
- Trial 21 (success): `trial_21_success.mp4`
- Trial 22 (success): `trial_22_success.mp4`
- Trial 23 (success): `trial_23_success.mp4`
- Trial 24 (success): `trial_24_success.mp4`
- Trial 25 (success): `trial_25_success.mp4`
- Trial 26 (success): `trial_26_success.mp4`
- Trial 27 (success): `trial_27_success.mp4`
- Trial 28 (success): `trial_28_success.mp4`
- Trial 29 (success): `trial_29_success.mp4`
- Trial 30 (success): `trial_30_success.mp4`
- Trial 31 (success): `trial_31_success.mp4`
- Trial 32 (success): `trial_32_success.mp4`
- Trial 33 (success): `trial_33_success.mp4`
- Trial 34 (success): `trial_34_success.mp4`
- Trial 35 (success): `trial_35_success.mp4`
- Trial 36 (success): `trial_36_success.mp4`
- Trial 37 (success): `trial_37_success.mp4`
- Trial 38 (success): `trial_38_success.mp4`
- Trial 39 (success): `trial_39_success.mp4`
- Trial 40 (success): `trial_40_success.mp4`
- Trial 41 (success): `trial_41_success.mp4`
- Trial 42 (success): `trial_42_success.mp4`
- Trial 43 (failure): `trial_43_failure.mp4`
- Trial 44 (success): `trial_44_success.mp4`
- Trial 45 (failure): `trial_45_failure.mp4`
- Trial 46 (success): `trial_46_success.mp4`
- Trial 47 (success): `trial_47_success.mp4`
- Trial 48 (success): `trial_48_success.mp4`
- Trial 49 (success): `trial_49_success.mp4`
- Trial 50 (success): `trial_50_success.mp4`
- Trial 51 (failure): `trial_51_failure.mp4`
- Trial 52 (success): `trial_52_success.mp4`
- Trial 53 (success): `trial_53_success.mp4`
- Trial 54 (success): `trial_54_success.mp4`
- Trial 55 (success): `trial_55_success.mp4`
- Trial 56 (success): `trial_56_success.mp4`
- Trial 57 (success): `trial_57_success.mp4`
- Trial 58 (success): `trial_58_success.mp4`
- Trial 59 (success): `trial_59_success.mp4`

## Insights

_Fill in observations after reviewing videos / running eval-copilot._
