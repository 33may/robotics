# v008 — Notes

Mixed real dataset: 08-merged (57 eps) + 50eps_no37 (50 eps) = 107 eps. Same hyperparams as v007. Testing if more diverse real data improves grab reliability.



## Eval

| ckpt   | front | bot   | left  | right | uniform | note                                                                                                                        |
| ------ | ----- | ----- | ----- | ----- | ------- | --------------------------------------------------------------------------------------------------------------------------- |
| 005000 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0   | Undertrained                                                                                                                |
| 010000 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0   | Undertrained                                                                                                                |
| 015000 | 1/1/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0   | Don't distinguish bot and front                                                                                             |
| 020000 | 1/1/0 | 1/1/1 | 1/1/1 | 1/1/1 | 0/0/0   | Model actually go rest after executing the task                                                                             |
| 025000 | 1/1/1 | 1/1/1 | 0/0/0 | 1/1/1 | 0/0/0   | Picked the duck after initial fail from the side position<br>Model gets quite confused when close to the cup(camera inside) |
| 030000 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0   | Overshoots                                                                                                                  |
| 035000 | 0/0/0 | 1/1/1 | 0/0/0 | 0/0/0 | 0/0/0   | Overshoots                                                                                                                  |
| 040000 | 0/0/0 | 1/1/0 | 0/0/0 | 0/0/0 | 0/0/0   | Overfitted, Overshoots                                                                                                      |
| 045000 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0   | Overfitted, Overshoots                                                                                                      |
| 050000 | 1/1/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0   | Overfitted, Overshoots                                                                                                      |

## Analysis

![[v008_eval_bars.png]]

![[v008_eval_heatmap.png]]

![[v008_loss_vs_eval.png]]

![[v008_conditional.png]]

### Summary

- **Sweet spot: 20K-25K steps** — performance peaks then collapses after 30K (overfitting)
- 20K: 80% grab, 80% place, 60% rest
- 25K: 60% grab, 60% place, 60% rest (need to be verified in the next section)
- After 30K: drops to near zero, overshooting motions
- The 20K-25K window where loss is ~0.018-0.015 seems to be the generalization sweet spot for this dataset size


So the bottleneck is grab phase, we might collect more data on that, or now I could form the only grabbing dataset from the 09-merged and fine-tune on grabbing




Okay so the best performance is steps 20k and 25k, lets evaluate them more


## Front - Red cup

| Run # | 020000 | 025000 |
| ----- | ------ | ------ |
| 1     | **1**  | **1**  |
| 2     | **1**  | **1**  |
| 3     | **1**  | 0      |
| 4     | 0      | 0      |
| 5     | 0      | **1**  |
| 6     | 0      | **1**  |
| 7     | **1**  | **1**  |
| 8     | **1**  | 0      |
| 9     | 0      | **1**  |
| 10    | **1**  | **1**  |
|       | **6**  | **7**  |

The grasping in 25 ckpt looks way more solid, the 20k ckpt mostly grabbed the nose of the duck, where the 25k more stable head grip. The failure was always at the last put stage, next we will verify if the performance increase if we change the cup from red to green.


## Front - Green cup

| Run # | 020000 | 025000 |
| ----- | ------ | ------ |
| 1     | 0      | 1/1/0  |
| 2     | 0      | 0/0/0  |
| 3     | 0      | 0/0/0  |
| 4     | 1      | 0/0/0  |
| 5     | 0      | 1/1/0  |
| 6     | 1      | 1/1/0  |
| 7     | 0      | 0/0/0  |
| 8     | 0      | 1/1/1  |
| 9     | 0      | 1/1/0  |
| 10    | 0      | 1/1/0  |
|       | 2      | 1      |

somehow the grabbing success decreased.

Looks like the color of the cup produce completely separate visual conditioning, and the model actions are strongly conditioned on that, creating 2 separate action modes.

to verify that we will run 10 more runs on the 25k ckpt to verify that grasping is more stable with red cup


## Front - Red cup

| Run # | 025000 |
| ----- | ------ |
| 1     | 1/1//1 |
| 2     | 0/0/0  |
| 3     | 1/1/0  |
| 4     | 1/1/1  |
| 5     | 0/0/0  |
| 6     | 1/1/0  |
| 7     | 1/1/1  |
| 8     | 1/1/0  |
| 9     | 1/1/0  |
| 10    | 1/1/0  |
|       | 3      |

No idea what changed between the first run, the lightning became poorer in office, but it should not matter that much. maybe the eval size of 10 episodes is just not large enough to be representative

The gripping became more stable, 8 with red and 6 in green, however the eval size of 10 might be not enough to judge



## Rotation Center- Red cup

| Run # | 025000 | Side           |
| ----- | ------ | -------------- |
| 1     | 1/1/1  | left           |
| 2     | 0/0/0  | back           |
| 3     | 1/1/1  | right          |
| 4     | 0/0/0  | left-back 45   |
| 5     | 0/0/0  | right-back 45  |
| 6     | 1/1/0  | left-front 45  |
| 7     | 0/0/0  | right-front 45 |
| 8     | 0/0/0  | back           |
| 9     | 0/0/0  | right          |
| 10    | 1/1/0  | left           |
|       |        |                |


the model doesnt work reliable, only the front red cup works but still fails to put the duck in cup


The thing we should consider is the fact the model was trained with the white gripper, but now we replced the gripper to the black one, however I don't think this is a gripper issue 


# dataset cleanup

## 08-merged
### Bad
2, 4, 6, 7 10 11 12 - 15, 17 18 19 21 25 28 31 36 37 42 45 50 52 54 56

## 09-merged
### Bad


## 10-black

| Zone | Clock positions | Distance | Episodes |
|------|----------------|----------|----------|
| **Front** | 11, 12, 1 | near + far | 6 × 3 pos × 2 dist = **18** |
| **Left** | 9, 10 | near + far | 6 × 2 pos × 2 dist = **12** |
| **Right** | 2, 3 | near + far | 6 × 2 pos × 2 dist = **12** |
| **Back-left** | 7, 8 | near + far | 6 × 2 pos × 2 dist = **12** |
| **Back-right** | 4, 5 | near + far | 6 × 2 pos × 2 dist = **12** |
| **Center** (near cup) | — | close to cup | **6** |
| **Edge cases** | extreme reach | far | **8** |
| | | **Total** | **80** |

```
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=frodeo-test \
    --robot.cameras="{ \
        top:     {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, \
        left:    {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, \
        right:   {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, \
        gripper: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30} \
    }" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM2 \
    --teleop.id=frodeo-test \
    --dataset.repo_id=eternalmay33/01_black_gripper \
    --dataset.num_episodes=80 \
    --dataset.single_task="Pick up the duck and place it in the cup" \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --display_data=true
```