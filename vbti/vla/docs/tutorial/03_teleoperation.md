# 3. Teleoperation

**What you have:** a calibrated leader/follower pair (block 2) and four named cameras (block 1). Demonstrations are recorded with LeRobot's `lerobot-record` straight into a LeRobot dataset in `~/.cache/huggingface/lerobot/<repo_id>`.

**What you need to do:** pick a dataset name and the task text — the text is stored with every frame and is what the VLA is conditioned on, so keep it exact and identical across sessions. Record from an X11 session (LeRobot's keyboard controls don't work under Wayland).

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=<follower_port> --robot.id=<name> \
  --robot.cameras="{ \
    top:     {type: opencv, index_or_path: /dev/cam_top_raw,     width: 640, height: 480, fps: 30}, \
    left:    {type: opencv, index_or_path: /dev/cam_left_raw,    width: 640, height: 480, fps: 30}, \
    right:   {type: opencv, index_or_path: /dev/cam_right_raw,   width: 640, height: 480, fps: 30}, \
    gripper: {type: opencv, index_or_path: /dev/cam_gripper_raw, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader --teleop.port=<leader_port> --teleop.id=<name> \
  --dataset.repo_id=<user>/<dataset_name> \
  --dataset.single_task="<task text>" \
  --dataset.num_episodes=5 \
  --dataset.streaming_encoding=true --dataset.encoder_threads=2 \
  --display_data=true
```

Controls: `B` start episode · `N` save + reset · `R` discard + reset.

**Done when:**
- 5 episodes saved, the leader drives the follower smoothly with no lag
- all 4 camera streams are present in the dataset for every episode
- `python -m vla.logic.dataset.check_utils report <user>/<dataset_name>` runs on it
