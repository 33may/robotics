# SmolVLA Training Internals (LeRobot)

## 1. Training Loop Structure

**File**: `lerobot/src/lerobot/scripts/lerobot_train.py`

Raw PyTorch loop wrapped with HuggingFace **Accelerate** (NOT HF Trainer).

```
train(cfg) ->
  make_dataset(cfg)
  make_policy(cfg.policy, ds_meta)
  make_pre_post_processors(cfg.policy, ..., dataset_stats=dataset.meta.stats)
  make_optimizer_and_scheduler(cfg, policy)
  accelerator.prepare(policy, optimizer, dataloader, lr_scheduler)

  for step in range(cfg.steps):
      batch = next(dl_iter)          # infinite cycling dataloader
      batch = preprocessor(batch)    # normalize, tokenize, device
      update_policy(policy, batch, optimizer, ...)
```

`update_policy()` function:
- `policy.forward(batch)` returns `(loss, output_dict)`
- `accelerator.backward(loss)`
- `accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)`
- `optimizer.step()` / `optimizer.zero_grad()`
- `lr_scheduler.step()` — stepped every batch, not every epoch

## 2. SmolVLA forward() Return Signature

**File**: `lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py`, line 355

```python
def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> tuple[Tensor, dict]:
    losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
    # losses shape: (batch, chunk_size, max_action_dim) — per-element MSE

    # Mask out-of-episode padding
    if actions_is_pad is not None:
        losses = losses * in_episode_bound.unsqueeze(-1)

    loss = losses.mean()
    loss_dict = {"loss": loss.item(), ...debug keys...}
    return loss, loss_dict
```

Inner `VLAFlowMatching.forward()` does **flow matching**:
- Samples noise and time from `Beta(1.5, 1.0) * 0.999 + 0.001`
- Interpolates `x_t = t * noise + (1-t) * actions`
- Target velocity `u_t = noise - actions`
- Loss = `MSE(u_t, v_t, reduction="none")`

## 3. Optimizer & Scheduler from Config Presets

**File**: `lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py`

When `cfg.use_policy_training_preset = True` (default):
```python
self.optimizer = self.policy.get_optimizer_preset()
# -> AdamWConfig(lr=1e-4, betas=(0.9,0.95), eps=1e-8, wd=1e-10, grad_clip=10)

self.scheduler = self.policy.get_scheduler_preset()
# -> CosineDecayWithWarmupSchedulerConfig(peak_lr=1e-4, decay_lr=2.5e-6, warmup=1000, decay=30000)
```

Factory builds them:
```python
params = policy.get_optim_params()       # returns self.parameters()
optimizer = cfg.optimizer.build(params)   # AdamW
lr_scheduler = cfg.scheduler.build(optimizer, cfg.steps)  # LambdaLR with cosine decay
```

## 4. Checkpoint Saving

**File**: `lerobot/src/lerobot/utils/train_utils.py`

```python
save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, scheduler, preprocessor, postprocessor)
```

Directory structure:
```
005000/
├── pretrained_model/
│   ├── config.json              # policy config
│   ├── model.safetensors        # weights (safetensors format)
│   ├── train_config.json        # full TrainPipelineConfig
│   ├── preprocessor.json
│   └── postprocessor.json
└── training_state/
    ├── optimizer_state.safetensors
    ├── optimizer_param_groups.json
    ├── scheduler_state.json
    ├── training_step.json
    └── rng_state.safetensors
```

## 5. Validation Approach

Eval happens if `cfg.env` is set and `cfg.eval_freq > 0`. Calls `eval_policy_all()` — runs policy in sim with `torch.no_grad()` + `accelerator.autocast()`.

For real-world data training: **no built-in validation split** — eval is done externally.

Metrics tracked: `avg_sum_reward`, `pc_success`, `eval_s`.

## 6. Multi-Camera & delta_timestamps

`prepare_images()` iterates over all keys in `config.image_features`. Images resized/padded to 512x512, normalized to [-1, 1]. Missing cameras filled with `-1` padding.

For temporal observations, images with `ndim == 5` (B, T, C, H, W) sliced to **last timestep only**: `img = batch[key][:, -1, :, :, :]`. Same for state.

`delta_timestamps` handled at **dataset level**, not policy level. Policy config: `n_obs_steps=1`, `chunk_size=50`, `n_action_steps=50`.

## 7. Preprocessor/Postprocessor Pipeline

**File**: `lerobot/src/lerobot/policies/smolvla/processor_smolvla.py`

**Preprocessor** steps (in order):
1. `RenameObservationsProcessorStep` — remap feature keys
2. `AddBatchDimensionProcessorStep` — unsqueeze for single-sample inference
3. `SmolVLANewLineProcessor` — appends `\n` to task strings
4. `TokenizerProcessorStep` — tokenizes language (max_length=48, right-padding)
5. `DeviceProcessorStep` — moves to GPU
6. `NormalizerProcessorStep` — normalizes state (MEAN_STD), images (IDENTITY), actions (MEAN_STD)

**Postprocessor** steps:
1. `UnnormalizerProcessorStep` — denormalizes actions
2. `DeviceProcessorStep` — moves to CPU

Preprocessor called on each batch before `policy.forward()`. Postprocessor only during eval/inference. Both serialized and saved with checkpoints.
