# v007 — Notes

Official `lerobot-train` baseline. Eliminate all custom engine variables to isolate whether the problem is our training pipeline or the data/task itself.

## Purpose

v006 showed intentional movement but inconsistent actions — the model doesn't reliably execute pick-and-place despite 57 real episodes. Deep analysis found no obvious bugs in our pipeline, but several differences from the official recipe:
- batch_size 16 vs official 64
- lr 1e-5 vs official 1e-4
- custom engine vs official lerobot-train with Accelerate

This run uses the **exact official training script** with default settings to establish a clean reference. If this works → our engine has a subtle issue. If this also fails → the problem is data or task complexity.

## What changed from v006

| Parameter | v006 | v007 | Why |
|-----------|------|------|-----|
| engine | custom vbti engine | official lerobot-train | Eliminate engine as variable |
| dataset | 08-merged_trimmed (22k frames) | 08-merged (27k frames, untrimmed) | More data, no trimming artifacts |
| batch_size | 16 | 64 | Official default |
| lr | 1e-5 | 1e-4 | Official default |
| steps | 30,000 | 20,000 | Official recommendation |
| scheduler | cosine (custom impl) | cosine_decay_with_warmup (LeRobot preset) | Official preset |

## Dataset

| | |
|---|---|
| repo_id | eternalmay33/08-merged |
| episodes | 57 |
| frames | 27,475 (untrimmed) |
| fps | 30 |
| cameras | top, left, right, gripper |
| robot_type | so_follower |

## Training command

```bash
lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=eternalmay33/08-merged \
    --batch_size=16 \
    --steps=20000 \
    --output_dir=vbti/experiments/duck_cup_smolvla/v007/lerobot_output \
    --job_name=v007_official \
    --wandb.enable=true \
    --save_freq=1000 \
    --policy.repo_id=eternalmay33/so101_suck_baseline \
    --policy.empty_cameras=1 \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.gripper": "observation.images.camera2", "observation.images.right": "observation.images.camera3", "observation.images.left": "observation.images.empty_camera_0"}'
```



1: hard start, undertrained

2: just bad

3: grassped the duck undertrained

4: overshoots the duck

5: 




8: grassped with help but drpped

9: grassped alone

10: overshhot

11: grasped with help


19: still overshooting the duck ALMOST performed task





New clenaed eval

2: is just reachoing phantome, even reachinmg doesant work

5: reaching is still poor, get to the duck but overshoots it

10: better reaching, still overshoots

18:  still overshoot, grabbed the duck but droped it

19: way better gripper possitioning, grasping and holdeing, fail to put in the cup, feels stuck and just wait

20: also fine id say



Lets continue training


![[Pasted image 20260324132310.png]]

Why the pattern is the same?

So after training for 20k steps we have decided to resume the train with 30k more steps, while refreshing the cosine scheduler

This way the idea is to use warmup to escape the local optimum we landed in r1 and then converge and refine trajectories to hopefully better optima




Here I run the trained model 5 times for every check

10:30

| ckpt | front | bot   | left  | right | uniform | note                                                                            |
| ---- | ----- | ----- | ----- | ----- | ------- | ------------------------------------------------------------------------------- |
| 1/10 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0   | Not precise motions                                                             |
| 1/19 | 0/0/0 | 1/0/0 | 0/0/0 | 0/0/0 | 0/0/0   |                                                                                 |
| 2/10 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0   | Failed to exit rest position most of times                                      |
| 2/20 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0   | Failed to exit rest position most of times                                      |
| 2/26 | 0/0/0 | 1/1/0 | 1/1/0 | 1/0/0 | 0/0/0   | Doing fine, but unreliable                                                      |
| 2/30 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0   | Stuck sometimes in back and forth rest position, under or overshooting the duck |
| 3/05 | 1/1/0 | 0/0/0 | 1/1/0 | 0/0/0 | 0/0/0   | Performance looks the same                                                      |
| 3/10 | 0/0/0 | 1/1/0 | 0/0/0 | 1/1/0 | 0/0/0   | Performance looks the same                                                      |

The rest loops happens when duck facing back

## Eval Graphs

![[eval_grabs.png]]
![[eval_pipeline.png
![[eval_heatmap.png]]




So the model moves reasonable but it is not reliable at all.

I think this is mainly the data problem so the next v008 run will use more data from 2 or maybe even 3 datasets

![[08-merged_may_compare.png]]

