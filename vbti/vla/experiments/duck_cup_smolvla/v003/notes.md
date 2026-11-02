# v003 — Notes

changed to trimmed dataset to bypass rest bias

okay we have run for 5k steps enough for now. turned out we need more training for robotics

---

Key finding: the rest pose for real data is shoulder_lift≈-98, elbow_flex≈99 — that's the extreme end of the range, not the mean. If the model was just outputting the dataset mean, it would produce [-23, 17] (trimmed) — which is NOT the resting pose.                      

So it's not a "predicting the mean" problem. Something is actively driving the output toward the rest configuration. Let me check what happens if the model outputs near-zero (default/untrained output) — what does the postprocessor denormalize that to?

```
Eval: duck_cup_smolvla/v003
Checkpoints to run: 1
  step_005000

  top: RealSense serial=123622270073 OK
  left: RealSense serial=123622270367 OK
  right: RealSense serial=126122270644 OK
  gripper: OpenCV path=/dev/video11 OK
Initialized 4/4 cameras
Connecting to robot on /dev/ttyACM0...

============================================================
Checkpoint: step_005000
============================================================
Moving to rest position...
WARNING:root:Relative goal position magnitude had to be clamped to be safe.
{   'gripper': {   'original goal_pos': 9.941176470588236,
                   'safe goal_pos': 10.120320855614974}}
WARNING:root:Relative goal position magnitude had to be clamped to be safe.
{   'gripper': {   'original goal_pos': 6.9411764705882355,
                   'safe goal_pos': 7.31283422459893}}
WARNING:root:Relative goal position magnitude had to be clamped to be safe.
{   'gripper': {   'original goal_pos': 3.9411764705882355,
                   'safe goal_pos': 4.572192513368984}}
WARNING:root:Relative goal position magnitude had to be clamped to be safe.
{   'gripper': {   'original goal_pos': 0.9411764705882355,
                   'safe goal_pos': 1.7647058823529402}}
  At rest: [  0. -95.  95.  45.   0.   0.]

Press Enter to start inference...
Loading policy from /home/may33/projects/ml_portfolio/robotics/vbti/experiments/duck_cup_smolvla/v003/checkpoints/step_005000
Reducing the number of VLM layers to 16 ...
Loading weights from local directory
Policy loaded: 450,046,176 params
qt.qpa.plugin: Could not find the Qt platform plugin "wayland" in "/home/may33/miniconda3/envs/isaac/lib/python3.11/site-packages/cv2/qt/plugins"
  step 20/10000  shoulder=    3.4  shoulder=  -99.3  elbow_fl=   93.7  wrist_fl=   45.0  wrist_ro=    1.1  gripper=    3.0
  step 40/10000  shoulder=    2.9  shoulder=  -97.9  elbow_fl=   94.8  wrist_fl=   45.2  wrist_ro=    0.6  gripper=    3.2
  step 60/10000  shoulder=    3.6  shoulder= -101.9  elbow_fl=   91.4  wrist_fl=   43.2  wrist_ro=   -0.3  gripper=    6.0
  step 80/10000  shoulder=    3.4  shoulder= -102.7  elbow_fl=   94.8  wrist_fl=   41.7  wrist_ro=   -0.8  gripper=   11.9
  step 100/10000  shoulder=    3.5  shoulder= -102.4  elbow_fl=   96.0  wrist_fl=   41.2  wrist_ro=   -0.8  gripper=   11.0
  step 120/10000  shoulder=    3.8  shoulder= -101.5  elbow_fl=   95.2  wrist_fl=   40.8  wrist_ro=   -1.8  gripper=   17.5
  step 140/10000  shoulder=    3.1  shoulder=  -98.7  elbow_fl=   98.0  wrist_fl=   41.1  wrist_ro=   -2.5  gripper=   23.9
  step 160/10000  shoulder=    3.0  shoulder= -100.5  elbow_fl=   92.0  wrist_fl=   41.0  wrist_ro=   -2.5  gripper=   28.2
  step 180/10000  shoulder=    3.4  shoulder= -100.8  elbow_fl=   94.5  wrist_fl=   41.3  wrist_ro=   -3.1  gripper=   29.3
  step 200/10000  shoulder=    4.3  shoulder=  -99.8  elbow_fl=   96.4  wrist_fl=   41.7  wrist_ro=   -3.0  gripper=   29.1
  step 220/10000  shoulder=    3.0  shoulder=  -99.6  elbow_fl=   90.6  wrist_fl=   42.9  wrist_ro=   -3.0  gripper=   34.4
  step 240/10000  shoulder=    3.5  shoulder=  -97.5  elbow_fl=   92.2  wrist_fl=   42.3  wrist_ro=   -3.5  gripper=   34.8
  step 260/10000  shoulder=    3.1  shoulder=  -99.9  elbow_fl=   88.3  wrist_fl=   43.4  wrist_ro=   -3.8  gripper=   38.5
  step 280/10000  shoulder=    3.2  shoulder=  -98.3  elbow_fl=   89.6  wrist_fl=   44.2  wrist_ro=   -4.4  gripper=   39.2
  step 300/10000  shoulder=    3.1  shoulder=  -98.6  elbow_fl=   89.8  wrist_fl=   44.2  wrist_ro=   -4.1  gripper=   39.4
  step 320/10000  shoulder=    2.9  shoulder=  -99.1  elbow_fl=   86.1  wrist_fl=   45.5  wrist_ro=   -4.4  gripper=   42.0
  step 340/10000  shoulder=    2.4  shoulder=  -99.5  elbow_fl=   86.2  wrist_fl=   45.4  wrist_ro=   -5.1  gripper=   43.5
  step 360/10000  shoulder=    1.6  shoulder=  -99.8  elbow_fl=   80.1  wrist_fl=   46.8  wrist_ro=   -5.0  gripper=   45.5
  step 380/10000  shoulder=    1.8  shoulder= -100.6  elbow_fl=   82.3  wrist_fl=   45.7  wrist_ro=   -5.3  gripper=   47.0
  step 400/10000  shoulder=    1.3  shoulder= -101.7  elbow_fl=   83.4  wrist_fl=   45.1  wrist_ro=   -5.3  gripper=   47.4
  step 420/10000  shoulder=    2.2  shoulder= -101.0  elbow_fl=   77.6  wrist_fl=   48.0  wrist_ro=   -5.4  gripper=   47.5
  step 440/10000  shoulder=    1.3  shoulder= -102.0  elbow_fl=   77.0  wrist_fl=   47.6  wrist_ro=   -5.3  gripper=   48.6
  step 460/10000  shoulder=    1.1  shoulder= -100.8  elbow_fl=   74.6  wrist_fl=   50.3  wrist_ro=   -5.4  gripper=   50.2
  step 480/10000  shoulder=    1.5  shoulder= -101.0  elbow_fl=   74.0  wrist_fl=   49.8  wrist_ro=   -5.1  gripper=   49.8
  step 500/10000  shoulder=    2.8  shoulder= -101.1  elbow_fl=   75.6  wrist_fl=   48.5  wrist_ro=   -5.0  gripper=   49.8
  step 520/10000  shoulder=    0.8  shoulder=  -99.9  elbow_fl=   75.3  wrist_fl=   50.3  wrist_ro=   -4.8  gripper=   50.5
  step 540/10000  shoulder=    2.1  shoulder= -101.5  elbow_fl=   77.3  wrist_fl=   49.9  wrist_ro=   -5.5  gripper=   51.4
  step 560/10000  shoulder=    1.8  shoulder= -100.5  elbow_fl=   72.5  wrist_fl=   51.3  wrist_ro=   -4.6  gripper=   51.5
  step 580/10000  shoulder=    1.6  shoulder= -100.7  elbow_fl=   72.6  wrist_fl=   52.0  wrist_ro=   -5.3  gripper=   51.7
  step 600/10000  shoulder=    2.0  shoulder= -100.9  elbow_fl=   74.5  wrist_fl=   51.9  wrist_ro=   -5.4  gripper=   51.4
  step 620/10000  shoulder=    1.6  shoulder=  -98.9  elbow_fl=   71.8  wrist_fl=   50.1  wrist_ro=   -6.0  gripper=   52.8
  step 640/10000  shoulder=    2.0  shoulder= -101.3  elbow_fl=   72.3  wrist_fl=   50.0  wrist_ro=   -5.9  gripper=   52.1
  step 660/10000  shoulder=    2.1  shoulder= -100.8  elbow_fl=   69.9  wrist_fl=   52.1  wrist_ro=   -5.9  gripper=   53.0
  step 680/10000  shoulder=    0.5  shoulder= -100.6  elbow_fl=   70.4  wrist_fl=   49.7  wrist_ro=   -6.1  gripper=   53.8
  step 700/10000  shoulder=   -0.1  shoulder= -100.5  elbow_fl=   74.0  wrist_fl=   48.6  wrist_ro=   -5.7  gripper=   54.3
  step 720/10000  shoulder=    0.7  shoulder= -101.5  elbow_fl=   72.8  wrist_fl=   48.7  wrist_ro=   -6.5  gripper=   54.6
  step 740/10000  shoulder=   -0.3  shoulder= -101.5  elbow_fl=   73.0  wrist_fl=   48.9  wrist_ro=   -6.4  gripper=   54.4
  step 760/10000  shoulder=    0.8  shoulder= -103.9  elbow_fl=   69.1  wrist_fl=   48.7  wrist_ro=   -6.7  gripper=   55.4
  step 780/10000  shoulder=    0.9  shoulder= -103.1  elbow_fl=   70.3  wrist_fl=   49.3  wrist_ro=   -7.3  gripper=   55.8
  step 800/10000  shoulder=    1.2  shoulder= -103.3  elbow_fl=   69.7  wrist_fl=   49.2  wrist_ro=   -7.4  gripper=   56.3
  step 820/10000  shoulder=    0.8  shoulder= -101.3  elbow_fl=   69.0  wrist_fl=   49.3  wrist_ro=   -7.4  gripper=   56.8
  step 840/10000  shoulder=    0.4  shoulder= -100.4  elbow_fl=   71.4  wrist_fl=   48.2  wrist_ro=   -6.8  gripper=   56.3
  step 860/10000  shoulder=    1.0  shoulder= -101.3  elbow_fl=   66.7  wrist_fl=   50.3  wrist_ro=   -6.6  gripper=   56.8
  step 880/10000  shoulder=    1.0  shoulder= -100.7  elbow_fl=   68.0  wrist_fl=   50.6  wrist_ro=   -7.1  gripper=   56.4
  step 900/10000  shoulder=    0.7  shoulder= -100.4  elbow_fl=   68.5  wrist_fl=   50.2  wrist_ro=   -7.0  gripper=   55.9
  step 920/10000  shoulder=    1.2  shoulder= -101.5  elbow_fl=   66.5  wrist_fl=   52.6  wrist_ro=   -7.4  gripper=   56.7
  step 940/10000  shoulder=    0.7  shoulder= -102.3  elbow_fl=   68.0  wrist_fl=   51.9  wrist_ro=   -7.5  gripper=   57.2
  step 960/10000  shoulder=    1.7  shoulder= -102.3  elbow_fl=   62.8  wrist_fl=   54.1  wrist_ro=   -7.7  gripper=   57.4
^C
Checkpoint step_005000 interrupted — saving video and continuing.
Encoding 975 frames to /home/may33/projects/ml_portfolio/robotics/vbti/experiments/duck_cup_smolvla/v003/eval/videos/eval_v003_step_005000.mp4...
Video saved: /home/may33/projects/ml_portfolio/robotics/vbti/experiments/duck_cup_smolvla/v003/eval/videos/eval_v003_step_005000.mp4
Robot disconnected.

Eval complete. Videos saved to: /home/may33/projects/ml_portfolio/robotics/vbti/experiments/duck_cup_smolvla/v003/eval/videos
(isaac) 
 Thu 19 Mar - 12:15  ~/projects/ml_portfolio/robotics   origin ☊ main 1☀ 9● 3‒ 
```


```
  ┌───────────────┬───────────┬───────────┬──────────────┬────────────────────┐
  │     Joint     │ Predicted │ Rest Pose │ Dataset Mean │    Closest to?     │                                                               
  ├───────────────┼───────────┼───────────┼──────────────┼────────────────────┤
  │ shoulder_pan  │ 2.5       │ 0         │ 2.0          │ both (all similar) │                                                               
  ├───────────────┼───────────┼───────────┼──────────────┼────────────────────┤                                                             
  │ shoulder_lift │ -100.5    │ -95       │ -23          │ REST               │                                                               
  ├───────────────┼───────────┼───────────┼──────────────┼────────────────────┤
  │ elbow_flex    │ 85.0      │ 95        │ 17.3         │ REST               │
  ├───────────────┼───────────┼───────────┼──────────────┼────────────────────┤
  │ wrist_flex    │ 46.1      │ 45        │ 67.7         │ REST               │
  ├───────────────┼───────────┼───────────┼──────────────┼────────────────────┤
  │ wrist_roll    │ -3.3      │ 0         │ 8.0          │ rest               │
  ├───────────────┼───────────┼───────────┼──────────────┼────────────────────┤
  │ gripper       │ 32.4      │ 0         │ 24.0         │ mean               │
  └───────────────┴───────────┴───────────┴──────────────┴────────────────────┘
```


  The model is NOT outputting zero (which would give dataset mean). It's actively predicting rest-adjacent values — at -1.8σ and +1.5σ in
  normalized space.

  ★ Insight ─────────────────────────────────────
  - In MEAN_STD normalization, predicting the mean = output 0. But the model outputs ±1.2-1.8σ → it has specifically learned the rest pattern.
  - Interestingly, the gripper drifts from 3→57 and elbow from 94→63 over 960 steps — the model IS slowly moving, just anchored to rest as its "base state."
  - This looks like a partially trained flow-matching model that learned the dominant mode (near-rest) but hasn't learned the visual conditioning to produce task-specific trajectories.
  ─────────────────────────────────────────────────
![[Pasted image 20260319122129.png]]

So after plotting the values again we see that trimming didn't particually help in the rest pose bias. 

Okay I know Koen was training the model and it worked for him, so here is distribution of the 08-merged dataset

![[Pasted image 20260319122456.png]]


Okay here is the trimmed version of 08-merged

```
python vbti/logic/dataset/trim_utils.py distribution eternalmay33/08-merged_trimmed
```

![[Pasted image 20260319122647.png]]

Looks way better for the rest pose, however ion both the gripper pose is strange. lets look the videos to verify

```
lerobot-dataset-viz --repo-id eternalmay33/08-merged_trimmed --episode-index 33
```

okay that is actually fine, just how Koen teleoperated



```
python vbti/logic/dataset/trim_utils.py distribution eternalmay33/so101_real_pick_place_50eps_trimmed
```

![[Pasted image 20260319123918.png]]

here is how the trim dataset looks now with 15 thresh

![[Pasted image 20260319124008.png]]



Iyt m,ght be just under train prolem and actually training for while will train the visual condition to move from the resting state, or it  
is more fundamental problem with data. I am quite concerned since the sim data showed way better and reasonable training   
in smaller time, where the real dataste stayed in this rest bias after even longer train. the large train run is time      
consuming so I want to verify and test the hypothersis before starting the train. 