# v020 — Notes

Vision unfrozen + image augmentation (lighting-robust, depth bypassed) + 94 ep right-side coverage data (datasets 11+12+13). Target: beat v019@20k (67% on dual_cup_30) by exceeding 80% overall, fixing the (right, closer) failure cell.

## Final config (r7 — running 2026-05-07 15:56)

| | Value |
|---|---|
| Dataset | `eternalmay33/duck_cup_v020_all` (765 ep, 336,940 frames) |
| Cameras | top, left, right, gripper (RGB) + gripper_depth (turbo, transform-bypassed) |
| Batch size | 16 |
| Steps | 252,000 (~11.97 epochs at 21,059 steps/epoch — matches v018's peak) |
| Optimizer | smolvla-adamw (vision_lr_scale=0.1) |
| Peak LR | 1e-4 (vision: 1e-5, expert: 1e-4) |
| Warmup | 1500 steps |
| Cosine decay | 1500 → 151,800 (decay_ratio=0.6), then flat at 2.5e-6 |
| Vision encoder | unfrozen, gradient checkpointing ON |
| Text encoder | frozen (train_expert_only=true with vbti-patch override) |
| Augmentation | enable=true, max_num_transforms=3 (brightness/contrast/sharpness/affine), saturation+hue weight=0 |
| save_freq | 10,000 (25 checkpoints) |
| push_to_hub | false |
| use_policy_training_preset | false |

## Hardware

- Remote: `vbti@10.11.100.151`, RTX 5090 32 GB
- VRAM at step 200: 13.1 GB / 32 GB (with gradient checkpointing)
- Step time: 0.773s
- Data load: 0.029s (no dataloader bottleneck)
- ETA: ~54 hours for 252k steps

## What changed vs v019

1. **Vision encoder unfrozen** — main lever to address v019's color-discrimination ceiling. v019 plateaued at 67% across all checkpoints.
2. **+94 episodes right-side coverage** (datasets 11/12/13) addressing v019's worst cell (right-far, both colors).
3. **Lighting-robust augmentation** — brightness, contrast, sharpness, affine ±5°. Saturation/hue disabled (would destroy cup color signal).
4. **Per-group LR** — vision encoder at lr × 0.1 to avoid catastrophic forgetting of pretrained SigLIP features.
5. **Gradient checkpointing on vision** — pure memory optimization (~17 GB savings at BS=16).
6. **No state augmentation** — reverted to 6-dim raw state (vs v014's 22-dim detection-augmented). The unfrozen vision should learn the grounding directly.

## Iteration log

- r1: Crashed (`tfs.saturation/hue` flags rejected by draccus — engine.py was emitting them, removed in subsequent edits)
- r2: User-launched concurrent with r1 → both OOM'd from GPU competition
- r3: Killed (user took over launch)
- r4: Crashed (`use_policy_training_preset=True` silently overrode all our optimizer/scheduler flags — needed `--use_policy_training_preset=false`)
- r5: Crashed (`smolvla-adamw requires named_parameters() dict` — factory.py passes plain `policy.parameters()` when preset disabled; patched factory.py)
- r6: OOM at BS=16 on clean GPU (vision unfrozen activation memory at 512×512 input far exceeded estimate)
- **r7: Running** — gradient checkpointing patch on vision_model freed 17 GB; BS=16 fits comfortably

## Patches applied to remote LeRobot 0.4.3

(See `~/.claude/projects/.../memory/remote_lerobot_patches.md` for md5s + reapply scripts)

1. `streaming_dataset.py` — skip image_transforms on depth cameras (turbo→distance mapping breaks under color jitter)
2. `smolvlm_with_expert.py` — three patches: vision-only unfreeze (override train_expert_only), keep vision in train mode, **enable gradient checkpointing on vision_model**
3. `optim/optimizers.py` — register `smolvla-adamw` per-group-LR optimizer (matches `"vision_model"` substring precisely, not greedy `"vlm"`)
4. `datasets/transforms.py` — saturation/hue weight=0 in defaults (CLI can't drill into dict-of-dataclass)
5. `optim/factory.py` — pass `dict(policy.named_parameters())` when optimizer.type=smolvla-adamw

## Known caveats

- **Step time**: ~0.773s/step → 54 hours wall clock. Could potentially halve with BS=24 (we have 19 GB headroom, gradient checkpointing makes BS scale cheap), but didn't restart to avoid more iteration risk.
- **W&B run**: https://wandb.ai/eternalmay33/vbti-training/runs/3esoatva
- **Tmux session**: `train_duck_cup_smolvla_v020_lerobot_output_r7` on remote
- **Output dir**: `/home/vbti/anton/experiments/duck_cup_smolvla/v020/lerobot_output_r7`

## Next: post-training plan

1. Sweep eval at checkpoints 20k, 30k, 50k, 100k, 150k on dual_cup_15 to find peak.
2. If peak >80% on dual_cup_15: confirm with dual_cup_60 + heatmap.
3. If peak ≤67%: vision unfreeze didn't help → consider auxiliary tasks (color classification head, position regression) for v021.

## Evaluation results (2026-05-11)

Tested at **step 150000** (just past cosine decay end at step 151,800).

| Protocol | n | Result | CI 95% |
|---|---|---|---|
| dual_cup_30 | 30 | **30/30 (100%)** | [89–100] |
| dual_cup_60 | 60 | **56/60 (93%)** | [84–97] |

vs v019@20k baseline: dual_cup_30 = 20/30 (67%) [49.5–80.5]. **Non-overlapping CIs — real improvement.**

### Per-bucket breakdown (dual_cup_60)

- By scene: both 20/20, single_black 17/20 (85%), single_red 19/20 (95%)
- By color: red 29/30 (97%), black 27/30 (90%)
- By side / target_closer: 10/10 on every bucket

### Failure mode

4/60 failures are **all single-cup trials** (dual-cup = 20/20 → no language-conditioning failure).

- T11 (single_red, 954 steps), T43 (single_black, 1194 steps), T45 (single_black, 782 steps), T51 (single_black, 2026 steps)
- Failure step counts mean **1239 vs 425 for successes** — the model retries placement for ~3× as long, doesn't give up. Likely cause: placement/drop precision, not perception or language.
- 3/4 failures are black cups — small-n hint of residual color skew.
