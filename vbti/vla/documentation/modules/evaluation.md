# Inference And Evaluation Modules

## Scope

Evaluation code lives in:

- `logic/inference/run_real_inference.py`
- `logic/inference/eval_engine.py`
- `logic/inference/eval_helpers.py`
- `logic/inference/eval_render.py`
- `logic/inference/async_chunk_runner.py`
- `logic/inference/protocols/`

It deploys trained policies on the real robot and turns physical trials into structured evidence.

## Real Inference

Main command:

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

Preview cameras:

```bash
python -m vbti.logic.inference.run_real_inference preview
```

Legacy eval wrapper:

```bash
python -m vbti.logic.inference.run_real_inference eval \
  --checkpoint=all \
  --experiment=duck_cup_smolvla \
  --version=v020 \
  --n_tries=1
```

Important defaults:

- `port=/dev/ttyACM0` in `run_real_inference.run()`.
- `cameras=realsense`.
- `task="pick up the object"`.
- `max_relative_target=10.0`.
- `action_horizon=10`.
- `enable_rtc=True`.
- `fps=30`.
- `max_steps=500`.
- `device=auto`.
- `detection=False`.
- `depth=False`.

Workflow:

1. Resolve camera preset.
2. Optionally modify gripper camera for depth.
3. Connect SO-101 follower.
4. Load policy, preprocessor, and postprocessor.
5. Start async chunk runner.
6. Capture state and camera frames.
7. Optionally append detection-state vector.
8. Optionally inject `gripper_depth` virtual camera.
9. Predict action in policy space.
10. Postprocess/denormalize and send joint targets with relative clamp.
11. Save/display video if enabled.
12. Stop cameras, robot, and async runner; move to rest.

Postprocessor is mandatory because SmolVLA outputs normalized actions. The real robot command path expects denormalized joint targets.

## Depth In Inference

Use `--depth=true` only when the checkpoint was trained with `observation.images.gripper_depth`.

Examples:

```bash
python -m vbti.logic.inference.run_real_inference run \
  --checkpoint=/path/to/pretrained_model \
  --cameras=opencv_depth \
  --depth=true
```

Depth path:

```text
D405 z16 depth -> aligned to gripper RGB -> clip 0.05-0.20m -> turbo RGB -> observation.images.gripper_depth
```

OpenCV cannot provide D405 depth. Hybrid presets use OpenCV for fixed RGB cameras and RealSense for gripper depth.

## Detection-State In Inference

Use only for models trained on augmented state:

```bash
python -m vbti.logic.inference.run_real_inference run \
  --checkpoint=/path/to/pretrained_model \
  --detection=true
```

State layout:

```text
6 joints + [left,right,top,gripper] x [duck,cup] x [cx,cy]
```

Live detector input is already RGB; do not BGR-swap before detection.

## Protocol Evaluation

Main command:

```bash
python -m vbti.logic.inference.eval_engine \
  --checkpoint=v020-150k \
  --protocol=dual_cup_60 \
  --experiment=duck_cup_smolvla \
  --version=v020 \
  --port=/dev/ttyACM1 \
  --action_horizon=10 \
  --max_steps=10000 \
  --fps=30 \
  --record=True
```

Important: `eval_engine.py` has `fire.Fire(run)`, so the module form exposes run arguments directly. Do not rely on a `run` subcommand unless tested.

Defaults:

- `protocol=checkpoint_sweep`
- `port=/dev/ttyACM1`
- `cameras=realsense`
- `action_horizon=10`
- `max_steps=600`
- `fps=30`
- `max_relative_target=10.0`
- `record=False`
- `resume=False`
- `depth=False`

Operator keys:

- `SPACE`: start trial.
- `S`: stop and mark success.
- `F`: stop and mark failure.
- `V`: save video.
- `Q`/Esc: quit session.

Workflow:

1. Resolve checkpoint shorthand such as `v020-150k`.
2. Create or resume an `eval_sessions/...` folder.
3. Persist session config.
4. Load protocol JSON.
5. Initialize cameras, robot, policy, optional depth/detector.
6. Show live top-camera placement overlay.
7. Run inference after operator starts trial.
8. Save per-trial result and optional video.
9. Move robot to rest after each trial.
10. Write `session.json` and `session.md`.

Output layout:

```text
experiments/duck_cup_smolvla/v020/eval_sessions/chkpt_step_150000_ah_10_pr_dual_cup_60_YYYYMMDD_HHMMSS/
  session.json
  session.md
  videos/
    trial_XX_success.mp4
    trial_XX_failure.mp4
```

## Protocol Schema

Legacy protocol shape:

```json
{
  "name": "id_scale_60",
  "version": "v1",
  "task": "pick up the duck and place it in the cup",
  "total_trials": 60,
  "trials": [
    {
      "trial_id": 0,
      "zone": "ID",
      "duck_px": [289, 245],
      "cup_px": [221, 117],
      "duck_dir_deg": 219.8,
      "cup_group": 0
    }
  ]
}
```

Entity protocol shape:

```json
{
  "name": "checkpoint_sweep",
  "schema": "entities",
  "task_template": "Pick up the duck and place it in the {color} cup",
  "trials": [
    {
      "trial_id": 0,
      "entities": [
        {"name": "duck", "kind": "duck", "color": "yellow", "px": [391, 250]},
        {"name": "A", "kind": "cup", "color": "red", "px": [248, 297]}
      ],
      "target": "A",
      "task": "Pick up the duck and place it in the red cup",
      "tags": {"scene": "single_red", "target_color": "red"}
    }
  ]
}
```

Entity protocols support multiple cups, per-trial task text, and tags for analysis.

## Protocol Tools

```bash
python -m vbti.logic.inference.protocols.protocols render id_scale_60
python -m vbti.logic.inference.protocols.protocols verify id_scale_60
python -m vbti.logic.inference.protocols.protocols edit id_scale_60
python -m vbti.logic.inference.protocols.render_protocol dual_cup_60
python -m vbti.logic.inference.protocols.generators.make_checkpoint_sweep
python -m vbti.logic.inference.protocols.generators.make_dual_cup_60
```

Outputs go under `logic/inference/protocols/renders/` and protocol JSON files.

## Session Analysis

List sessions:

```bash
python -m vbti.logic.inference.eval_helpers ls
python -m vbti.logic.inference.eval_helpers ls --detailed
```

Inspect:

```bash
python -m vbti.logic.inference.eval_helpers info latest
python -m vbti.logic.inference.eval_helpers info v020-150k --group_by=scene
```

Open video:

```bash
python -m vbti.logic.inference.eval_helpers play latest 3
```

Render heatmap:

```bash
python -m vbti.logic.inference.eval_render heatmap /path/to/session --output=/tmp/heatmap.png
```

## Result Anchors

| Experiment/checkpoint | Protocol | Result |
|---|---|---|
| `v017` step 20k | `dual_cup_30` | 11/30 = 37% |
| `v018` step 20k | `dual_cup_30` | 14/30 = 47% |
| `v019` step 20k | `dual_cup_30` | 20/30 = 67% |
| `v020` step 150k | `dual_cup_30` | 30/30 = 100% |
| `v020` step 150k | `dual_cup_60` | 56/60 = 93% |
| `v020` step 336940 | `checkpoint_sweep` | 19/20 = 95% |
| `v024` step 167940 | `checkpoint_sweep` | 14/20 = 70% |

Interpretation from v020: dual-cup language disambiguation was solved in the saturated protocol; remaining failures were more likely drop/placement precision than language/perception.

## Pitfalls

- Real-robot eval is the final metric; training loss is diagnostic only.
- Compare only sessions from the same protocol for quantitative claims.
- `eval_engine.py` CLI docs and Fire binding differ; use direct module args.
- Depth requires RealSense gripper depth.
- Detection-state models require matching 22D state at inference.
- Checkpoint shorthand resolution is convenient but should be verified before a long eval.
- v020 `dual_cup_60` is near saturation; use harder protocols for further progress.
