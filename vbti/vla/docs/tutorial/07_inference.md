# 7. Real inference

**What you have:** a checkpoint's `pretrained_model/` dir (`config.json`, `model.safetensors`, `policy_preprocessor.json`, `policy_postprocessor.json`) and the robot and cameras from blocks 1–2. `run_real_inference run` moves the follower to rest, then runs the policy asynchronously: each predicted action chunk is executed for `action_horizon` steps before the next one takes over, and RTC blends the new chunk into the committed trajectory. No protocol, no session files: this is for watching a policy, demos and recorded runs.

**What you need to do:** pass the checkpoint, a camera preset whose camera names match the training dataset's, the task text used in training, and `--depth=true` only for a policy trained with `gripper_depth`.

```bash
python -m vla.logic.cameras.view_cameras --preset <preset>
python -m vla.logic.inference.run_real_inference run \
  --checkpoint=experiments/<exp>/<vNNN>/<run_name>/checkpoints/<NNNNNN>/pretrained_model \
  --port=<follower_port> \
  --task="<task text>" \
  --cameras=<preset> \
  --depth=<true|false> \
  --action_horizon=10 \
  --max_steps=10000 \
  --move_to_start=true
# recorded run: same command + --record=<out.mp4>
# prompt switching while running: same command + --prompt_gui=True --prompt_update_mode=responsive
python -m vla.logic.servos.rest --port=<follower_port>
```

**Done when:**
- the policy loads and the camera grid shows every training camera, plus the `gripper_depth` tile for a depth policy
- the follower goes to rest, then executes the task; `q` stops it
- with `--record`, the mp4 contains the full camera grid of the run

## Inference commands

| Command | What it does |
|---|---|
| `python -m vla.logic.inference.run_real_inference run --checkpoint=<pretrained_model> [flags]` | Policy on the robot with a live camera grid; `q` quits |
| `  --cameras=<preset>` | Preset from `CAMERA_PRESETS` (`logic/cameras/cameras.py`): `realsense`, `realsense_depth`, `opencv`, `opencv_depth`, `sim_4cam_opencv`, `sim_4cam_realsense`, … |
| `  --depth=true` | Adds live D405 depth as `gripper_depth`; needs a RealSense gripper in the preset (`opencv_depth`, `realsense_depth`) |
| `  --action_horizon=<n>` · `--enable_rtc=<bool>` | Actions executed per chunk before re-planning · Real-Time Chunking on/off |
| `  --max_steps` · `--fps` · `--move_to_start` | Run length · control rate · go to rest before starting |
| `  --record=<out.mp4>` | Save the camera grid of the run |
| `  --prompt_gui=True` | Prompt window: free-text prompt, red/black quick buttons, update mode |
| `  --live_prompt_toggle=True [--red_prompt --black_prompt]` | `t` in the camera window toggles between two prompts |
| `  --voice_prompt=True [--voice_backend --voice_model --voice_command_mode=red_black\|direct]` | `v` starts/stops recording; speech becomes the prompt |
| `  --prompt_update_mode=smooth\|responsive` | New prompt at the next chunk · reset the runner immediately |
| `  --camera_name_map` · `--robot_id` · `--print_actions_every=<n>` · `--replay_pkl` | Rename live cameras to training names · calibration id (default: active profile) · print actions · replay stored actions instead of the policy |
| `python -m vla.logic.inference.run_real_inference preview` | Camera grid only, no robot, no model |

Full reference: `docs/modules/inference/`.
