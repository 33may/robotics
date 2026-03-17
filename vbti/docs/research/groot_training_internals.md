# GR00T N1.6 Training Internals (Isaac-GR00T)

## 1. Training Loop (Gr00tTrainer)

Subclasses HuggingFace `Trainer` with 4 overrides:

- **`compute_loss()`** — calls `super().compute_loss()` (which calls `model.forward(inputs)` and extracts the `"loss"` key), then computes action-token accuracy every N steps for logging
- **`get_train_dataloader()`** — forces `ignore_data_skip=True`; on resume, reseeds the dataset (`seed + global_step`) rather than skipping batches. This is because `ShardedMixtureDataset` is an `IterableDataset` with no `__len__`
- **`train()`** — manually loads `TrainerState` from checkpoint before calling `super().train()` to ensure correct dataloader initialization
- **`log()`** — hides the epoch field since it's meaningless with iterable datasets

The model itself returns loss — `Gr00tN1d6.forward(inputs: dict) -> {"loss": tensor, "action_loss": ..., "action_mask": ...}`. The Trainer just extracts `outputs["loss"]`.

## 2. Model forward() and Loss

```
Gr00tN1d6.forward(inputs: dict):
  backbone_inputs, action_inputs = prepare_input(inputs)  # splits VLM vs action data
  backbone_outputs = self.backbone(backbone_inputs)        # Eagle VLM → vl_embeds
  action_outputs = self.action_head(backbone_outputs, action_inputs)  # DiT
  return action_outputs
```

The action head uses **flow-matching diffusion**:
- Sample time `t ~ Beta(alpha, beta)`, create `noisy_trajectory = (1-t)*noise + t*actions`, target `velocity = actions - noise`
- Encode state via embodiment-conditioned MLP, encode noisy actions, concatenate, pass through 32-layer DiT with cross-attention to VLM features
- Loss: `F.mse_loss(predicted_velocity, velocity, reduction="none") * action_mask` — mask handles variable action dims across embodiments

Inference uses 4-step Euler integration: `actions += dt * pred_velocity`.

## 3. Data Pipeline

**LeRobot v2 parquet + mp4** → `LeRobotEpisodeLoader` → `ShardedSingleStepDataset` → `ShardedMixtureDataset` (IterableDataset)

- `modality.json` (per dataset, in `meta/`): maps state/action array indices to semantic fields (e.g., `"single_arm": {"start": 0, "end": 5}`) and video keys
- `ModalityConfig` (per embodiment, Python file): defines `delta_indices`, `modality_keys`, `ActionConfig` (relative/absolute, EEF/non-EEF)
- `Gr00tN1d6Processor`: normalizes state/action, applies albumentations augmentations, pads to `max_state_dim=29`/`max_action_dim=29`, converts images to Eagle chat template format
- **VLM tokenization happens in the collator** (`Gr00tN1d6DataCollator`), not the dataset
- ShardedMixtureDataset streams shards with weighted sampling across datasets, background caching via ThreadPoolExecutor, distributed shard assignment via modulo

## 4. Checkpoints

- Uses HuggingFace `save_pretrained()` (model inherits `PreTrainedModel`)
- `CheckpointFormatCallback` copies experiment config + processor config into each `checkpoint-{step}/` directory
- `BestMetricCheckpointCallback` tracks best eval metric, calls `model.save_pretrained()` for the best checkpoint
- Loading: `AutoModel.from_pretrained()` + `AutoProcessor.from_pretrained()`

## 5. Evaluation

`open_loop_eval.py`: compares predicted vs ground truth actions using MSE/MAE. Runs policy at `action_horizon` intervals (default 16 steps), concatenates predicted action chunks. Generates multi-panel trajectory plots. No closed-loop feedback.

## 6. DeepSpeed

- **ZeRO-2** (`zero2_config.json`): optimizer state + gradient partitioning. Good default for single-node multi-GPU.
- **ZeRO-3** (`zero3_config.json`): + parameter partitioning, `stage3_gather_16bit_weights_on_model_save=true`. For multi-node or VRAM-constrained setups.
- Both use bf16, auto batch sizes. Selected via `config.training.deepspeed_stage`.

## 7. Key Differences vs SmolVLA

| Aspect | GR00T (HF Trainer) | SmolVLA (raw loop) |
|--------|--------------------|--------------------|
| Dataset | `IterableDataset`, shard-based, no `__len__` | Map-style dataset |
| Loss | Computed inside `model.forward()`, returned as dict key | Returned as tuple `(loss, loss_dict)` |
| Collator | Does VLM tokenization | Simpler stacking |
| Optimizer | Managed by Trainer | Manual |
| Checkpoints | `save_pretrained()` with config/processor | `save_pretrained()` + pre/postprocessor |
| Resume | Reseed dataset, load TrainerState | Skip to step |
| Multi-GPU | DeepSpeed via Trainer flag | Accelerate DDP |
| Action masking | Padded to max_dim=29, masked in loss | Fixed dim |
| Action head | 32-layer DiT, flow matching | Flow matching (simpler) |
| VLM backbone | Eagle (custom) | SmolVLM (HF) |
