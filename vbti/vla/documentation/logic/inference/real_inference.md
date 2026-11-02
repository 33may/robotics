# Real Inference

## Preview Cameras

```bash
python -m vbti.logic.inference.run_real_inference preview
```

## Run Policy

```bash
python -m vbti.logic.inference.run_real_inference run \
  --checkpoint=/path/to/checkpoint/pretrained_model \
  --port=/dev/ttyACM1 \
  --cameras=realsense \
  --task="Pick up the duck and place it in the red cup" \
  --max_relative_target=10.0 \
  --action_horizon=10 \
  --fps=30 \
  --max_steps=500
```

## Important Defaults

| Arg | Default | Meaning |
|---|---|---|
| `checkpoint` | `""` | Pretrained model/checkpoint path. |
| `port` | `/dev/ttyACM0` | Follower arm port in `run_real_inference.py`. |
| `cameras` | `realsense` | Camera preset. |
| `task` | `pick up the object` | Language instruction. |
| `max_relative_target` | `10.0` | Safety clamp per command. |
| `move_to_start` | `True` | Move to rest/start before run. |
| `action_horizon` | `10` | Number of actions from each chunk to execute. |
| `enable_rtc` | `True` | Async realtime chunking. |
| `fps` | `30` | Control/camera loop target. |
| `max_steps` | `500` | Max control steps. |
| `show_cameras` | `True` | Live grid display. |
| `record` | `""` | Optional video path. |
| `device` | `auto` | Torch device. |
| `detection` | `False` | Add live 22D detection-state path. |
| `depth` | `False` | Add live `gripper_depth` image stream. |

## Runtime Data Flow

```text
SO-101 follower state
camera frames
optional D405 depth
optional detector state
task text
-> LeRobot policy observation
-> preprocessor
-> SmolVLA select_action
-> postprocessor
-> joint target clamp
-> follower command
```

## Policy Loading

Checkpoint directory must include model config/weights plus pre/post processors. The postprocessor is required because the policy outputs normalized actions.

Expected files include:

```text
config.json
model.safetensors
preprocessor.json or policy_preprocessor.json
postprocessor.json or policy_postprocessor.json
```

## Depth Run

```bash
python -m vbti.logic.inference.run_real_inference run \
  --checkpoint=/path/to/depth_checkpoint/pretrained_model \
  --cameras=opencv_depth \
  --depth=true
```

Requires checkpoint trained with `observation.images.gripper_depth`.

## Detection-State Run

```bash
python -m vbti.logic.inference.run_real_inference run \
  --checkpoint=/path/to/detection_state_checkpoint/pretrained_model \
  --detection=true
```

Requires checkpoint trained with 22D state.

## Cleanup

Inference cleanup should:

- stop async runner;
- save video if recording;
- release cameras;
- disconnect robot;
- move to rest when configured.

If cleanup fails, manually check robot state before the next run.
