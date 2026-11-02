# Dataset knowledge

## Joint order

Action and state are 6-D, in this order (IDs 1-6 on the bus):

```
shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos, gripper.pos
```

## Units

- Real recordings: body joints in degrees relative to the calibration profile (`MotorNormMode.DEGREES`), gripper `[0, 100]` (`RANGE_0_100`). See `logic/servos/profiles.py`.
- `convert_utils` output: body joints `[-100, 100]`, gripper `[0, 100]`.
- Joint values are only meaningful under the calibration they were recorded with. `convert_utils recalibrate` re-expresses a dataset under another profile.

## Sim to LeRobot conversion

`logic/dataset/convert_utils.py`, per joint `i`:

```python
sim_deg  = sim_rad * 180 / pi
t        = (sim_deg - SIM_LIMITS_DEG[i].lo) / (SIM_LIMITS_DEG[i].hi - SIM_LIMITS_DEG[i].lo)
real_deg = t * (REAL_LIMITS_DEG[i].hi - REAL_LIMITS_DEG[i].lo) + REAL_LIMITS_DEG[i].lo
frac     = (real_deg - REAL_LIMITS_DEG[i].lo) / (REAL_LIMITS_DEG[i].hi - REAL_LIMITS_DEG[i].lo)
out      = frac * 100            # gripper  -> [0, 100]
out      = frac * 200 - 100      # body     -> [-100, 100]
```

`SIM_LIMITS_DEG` come from the robot USD joint limits, `REAL_LIMITS_DEG` from one servo calibration. Read the tables in the file; they are not symmetric and the gripper sim range starts below zero. Unit conversion does not align sim and real by itself: the calibration zero pose must also match the sim zero (a sim/real zero mismatch was found and fixed on 30/04, see `docs/case-study/README.md` §6).

## LeRobot v3.0 layout

```
<root>/
├── meta/
│   ├── info.json        # codebase_version, robot_type, fps, features, totals
│   ├── stats.json       # per-feature min/max/mean/std
│   ├── episodes/        # episode metadata parquet
│   └── tasks.*          # task strings
├── data/chunk-*/file-*.parquet
└── videos/observation.images.<camera>/chunk-*/file-*.mp4
```

Parquet columns: `action`, `observation.state`, `episode_index`, `frame_index`, `timestamp`, `task_index`.

## Action/state correspondence

`check_utils report` checks which holds:

- absolute: `action[t] ≈ state[t]`
- next-state: `action[t] ≈ state[t+1]`
- delta: `action[t] ≈ state[t+1] - state[t]`

Teleop recordings here are absolute position targets. `to_delta` makes a delta copy (v012 trained on delta actions).

## Cameras

| Source | Key | Example |
|---|---|---|
| Sim HDF5 | `obs/<cam>` | `cam_top`, `wrist` |
| LeRobot | `observation.images.<name>` | `observation.images.top` |
| Real rig | `top`, `left`, `right`, `gripper` (+ `gripper_depth` from v018) | |

`convert --camera_map` bridges sim keys to LeRobot names. Training configs list cameras in `dataset.cameras.names`; order and names must match the dataset and the inference preset.

## Depth

Depth enters the model as an image stream `observation.images.gripper_depth`: clipped to 0.05-0.20 m, turbo-colorized. Training data and live inference must use the same clip and colormap. A dataset without depth gets a placeholder with `add_zero_depth_turbo` before it is aggregated with depth datasets.
