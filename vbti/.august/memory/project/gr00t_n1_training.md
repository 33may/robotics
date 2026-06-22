# GR00T N1.6 Training Infrastructure (2026-03-17)

## Repo & Model
- **Repo**: `github.com/NVIDIA/Isaac-GR00T` (6.4k stars, active as of 2026-03-16)
- **Model**: `nvidia/GR00T-N1.6-3B` (3B params, VLA = Vision-Language-Action)
- **Architecture**: Cosmos-Reason-2B VLM backbone (Eagle) + 32-layer DiT action head (flow-matching diffusion)
- **Package**: `gr00t` v0.1.0, Python >=3.10 <3.13, managed via `uv`
- **NOT cloned locally** — repo at NVIDIA/Isaac-GR00T on GitHub

## Training Loop (HF Trainer Subclass)

### Gr00tTrainer customizations over HuggingFace Trainer:
1. **compute_loss()**: Calls `super().compute_loss()`, then computes token-level accuracy on action predictions every `logging_steps`. Applies `action_offset` filtering when configured.
2. **get_train_dataloader()**: Sets `ignore_data_skip=True`. On resume, reseeds dataset (`seed + global_step`) instead of skipping batches. Creates DataLoader with custom `multiprocessing_context`.
3. **train()**: Loads `TrainerState` from checkpoint JSON before calling `super().train()` — ensures correct dataloader init on resume.
4. **log()**: Hides epoch field (meaningless for IterableDataset).

### Pipeline Assembly (experiment.py → run()):
```
Config → MODEL_REGISTRY.get(Gr00tN1d6Config) → Gr00tN1d6Pipeline
pipeline.setup() → model, train_dataset, eval_dataset, data_collator, processor
TrainingArguments(28 params from config) → Gr00tTrainer(model, args, datasets, collator)
Callbacks: CheckpointFormatCallback, BestMetricCheckpointCallback, ProfCallback
trainer.train(resume_from_checkpoint=...)
```

## Model forward() Signature & Loss

```python
# Gr00tN1d6.forward(inputs: dict) -> BatchFeature
backbone_inputs, action_inputs = self.prepare_input(inputs)  # splits dict
backbone_outputs = self.backbone(backbone_inputs)              # Eagle VLM
action_outputs = self.action_head(backbone_outputs, action_inputs)  # DiT
return action_outputs  # contains "loss", "action_loss", "action_mask"
```

### Action Head Loss (flow-matching):
```python
# In Gr00tN1d6ActionHead.forward():
noise = torch.randn(actions.shape)
t = self.sample_time(batch_size)  # Beta distribution
noisy_trajectory = (1 - t) * noise + t * actions
velocity = actions - noise
# ... encode state, encode noisy actions, concat, pass through DiT ...
pred_actions = action_decoder(dit_output)
action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
loss = action_loss.sum() / (action_mask.sum() + 1e-6)
```

### Inference (get_action):
- 4-step denoising (configurable via `num_inference_timesteps`)
- Euler integration: `actions = actions + dt * pred_velocity`
- Returns `BatchFeature({"action_pred": actions, ...})`

## Data Pipeline

### Flow:
```
LeRobot v2 parquet + mp4 → LeRobotEpisodeLoader → ShardedSingleStepDataset
  → extract_step_data() → VLAStepData → Gr00tN1d6Processor → model-ready tensors
Multiple datasets → ShardedMixtureDataset (IterableDataset) → Gr00tTrainer
```

### ShardedMixtureDataset (IterableDataset):
- Weighted sampling across datasets, normalized by shard sizes
- `generate_shard_sampling_schedule()` creates (dataset_idx, shard_idx) pairs
- Distributed: modulo assignment `i % (world_size * num_workers) == rank * num_workers + worker_id`
- Background shard caching via ThreadPoolExecutor
- Shuffles within each shard before yielding

### Gr00tN1d6Processor transforms:
1. State/action: normalize (percentile or standard), optional sincos encoding, pad to max_dim
2. Images: albumentations augmentations (replay-consistent across views), resize → (T, C, H, W)
3. VLM: images→PIL + text → Eagle chat template → tokenize
4. Output: `{state, action, action_mask, vlm_content, embodiment_id}`

### BasicDataCollator:
- Simple `torch.stack()` per field key — nothing fancy

### Gr00tN1d6DataCollator (in model):
- Handles VLM content tokenization at collation time
- Returns `{"inputs": BatchFeature(...)}`

## modality.json (per dataset, in meta/)
```json
{
  "state": {"single_arm": {"start": 0, "end": 5}, "gripper": {"start": 5, "end": 6}},
  "action": {"single_arm": {"start": 0, "end": 5}, "gripper": {"start": 5, "end": 6}},
  "video": {"front": {"original_key": "observation.images.front"}, "wrist": {...}},
  "annotation": {"human.task_description": {"original_key": "task_index"}}
}
```

## ModalityConfig (per embodiment, Python file)
```python
config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["front", "wrist"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=["single_arm", "gripper"]),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),  # 16-step action horizon
        modality_keys=["single_arm", "gripper"],
        action_configs=[ActionConfig(rep=RELATIVE, type=NON_EEF, format=DEFAULT), ...]
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
}
register_modality_config(config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
```

## Checkpoints
- **Saving**: HF `save_pretrained()` (inherits from PreTrainedModel)
- **CheckpointFormatCallback**: copies exp config dir + processor dir + wandb config into each `checkpoint-{step}/`
- **BestMetricCheckpointCallback**: tracks best eval metric, calls `model.save_pretrained()` into best checkpoint dir, removes previous best
- **Loading**: `AutoModel.from_pretrained()` for model, `AutoProcessor.from_pretrained()` for processor

## DeepSpeed
- **ZeRO-2** (`zero2_config.json`): optimizer state + gradient partitioning. overlap_comm, reduce_scatter enabled. Good for single-node multi-GPU.
- **ZeRO-3** (`zero3_config.json`): + parameter partitioning. `stage3_gather_16bit_weights_on_model_save=true`. For >1 node or VRAM-constrained.
- Both: bf16 auto, batch sizes auto, gradient clipping auto (set from TrainingArguments)
- Config selected via `config.training.deepspeed_stage` → loads JSON from `gr00t/configs/deepspeed/`

## Tunable Components
- `tune_llm`: False — freeze VLM backbone LLM
- `tune_visual`: False — freeze vision encoder (SigLIP2)
- `tune_top_llm_layers`: 4 — last N LLM layers trainable
- `tune_projector`: True — state/action encoders/decoders
- `tune_diffusion_model`: True — DiT action head
- `tune_vlln`: True — VL layer norm

## Config System
```python
@dataclass
class Config:
    model: Gr00tN1d6Config  # architecture, tuning flags
    data: DataConfig         # datasets, modalities, sharding
    training: TrainingConfig # HF TrainingArguments + custom fields

# FinetuneConfig is a flat CLI-friendly dataclass (tyro)
# that gets expanded into Config internally
```

## Key Diffs vs Standard PyTorch Loop (for unified backend)
1. **IterableDataset** — no `__len__`, no epoch concept, shard-based streaming
2. **Loss inside model.forward()** — model returns dict with "loss" key, Trainer extracts it
3. **Collator does tokenization** — VLM text processing happens at collation, not in dataset
4. **DeepSpeed integration** — via HF Trainer, not manual
5. **No explicit optimizer creation** — Trainer handles it from TrainingArguments
6. **Checkpoint = save_pretrained** — not `torch.save(state_dict)`, includes config/processor
7. **Resume = reseed** — doesn't skip batches, reseeds dataset RNG instead
8. **Action masking** — variable-dim actions across embodiments, padded + masked in loss

## Hardware
- **Training**: Single H100 or L40 recommended. A6000 works but slower.
- **Inference**: RTX 5090 27Hz, H100 26Hz, RTX 4090 23Hz, Jetson Thor 10Hz (4 denoise steps)
- **DeepSpeed ZeRO-2/3** for multi-GPU

## Pre-registered Embodiments
ROBOCASA_PANDA_OMRON, GR1, UNITREE_G1, LIBERO_PANDA, OXE_GOOGLE, OXE_WIDOWX, OXE_DROID, BEHAVIOR_R1_PRO, NEW_EMBODIMENT

## Example Finetune Command
```bash
python gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path <lerobot_v2_dataset> \
    --modality_config_path <embodiment_config.py> \
    --embodiment_tag NEW_EMBODIMENT \
    --output_dir /tmp/finetune \
    --global_batch_size 32 --learning_rate 1e-4 --max_steps 10000 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
```
