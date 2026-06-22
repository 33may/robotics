# Runbook: Real Data Collection

Use this to record real SO-101 demonstrations into a LeRobot dataset.

## 1. Preflight

Run:

```bash
documentation/runbooks/preflight.md
```

In practice, execute the commands from that runbook manually.

## 2. Use X11 If Keyboard Controls Fail

LeRobot recording controls can fail under Wayland. If `B`, `N`, or `R` do not work reliably, log into an X11 session.

## 3. Pick Dataset Name And Task Text

Example:

```text
repo_id: eternalmay33/<new_dataset_name>
task: Pick up the duck and place it in the cup
episodes: 80
```

Use precise task text because it is stored in the dataset and used by VLA training.

## 4. Recording Command Template

Modern preference is stable symlink/camera mapping. If using LeRobot's direct `lerobot-record`, adapt the camera block to the current camera backend.

Historical OpenCV-style command pattern:

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=frodeo-test \
  --robot.cameras="{ \
    top:     {type: opencv, index_or_path: /dev/cam_top_raw, width: 640, height: 480, fps: 30}, \
    left:    {type: opencv, index_or_path: /dev/cam_left_raw, width: 640, height: 480, fps: 30}, \
    right:   {type: opencv, index_or_path: /dev/cam_right_raw, width: 640, height: 480, fps: 30}, \
    gripper: {type: opencv, index_or_path: /dev/cam_gripper_raw, width: 640, height: 480, fps: 30} \
  }" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM2 \
  --teleop.id=frodeo-test \
  --dataset.repo_id=eternalmay33/<new_dataset_name> \
  --dataset.num_episodes=80 \
  --dataset.single_task="Pick up the duck and place it in the cup" \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --display_data=true
```

Adjust:

- follower port from `scan_all`;
- leader port from `scan_all`;
- camera paths after `view_cameras` verification;
- dataset repo ID;
- episode count;
- task text.

## 5. Recording Controls

Common LeRobot episode controls:

- `B`: start recording episode.
- `N`: mark success, save, reset.
- `R`: discard episode, reset.

Only save successful/clean demonstrations unless intentionally collecting failures.

## 6. During Recording

Watch for:

- camera freezes;
- bad gripper cable motion;
- robot calibration drift;
- too many idle/rest frames;
- inconsistent task setup;
- accidental human-only visual cues.

## 7. Post-Recording Inspection

```bash
python -m vbti.logic.dataset.check_utils report eternalmay33/<new_dataset_name>
python -m vbti.logic.dataset.check_utils cameras eternalmay33/<new_dataset_name>
python -m vbti.logic.dataset.check_utils plot_actions eternalmay33/<new_dataset_name> --save=/tmp/<new_dataset_name>_actions.png
```

Record in notes:

- dataset repo ID;
- episode count;
- frame count;
- camera schema;
- task text;
- calibration profile;
- known issues.

## Common Failures

| Failure | Fix |
|---|---|
| Keyboard controls do nothing | Use X11 session. |
| Camera names wrong | Verify presets/symlinks before recording. |
| Gripper camera disconnects | Cable/camera/USB isolation before more episodes. |
| Dataset includes bad episodes | Trim/remove before training. |
| Lots of idle frames | Use trimming/curation before model training. |
