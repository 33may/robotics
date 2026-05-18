# SmolVLA-UVA: Auxiliary Future-Feature Prediction for SmolVLA

| Field | Value |
|---|---|
| Author | may33 |
| Date | 2026-05-13 (rev. 2026-05-18) |
| Status | Implemented; see Design Revision below |
| Target dataset (v0) | `eternalmay33/06_black_cup_red_bg_depth` |
| Teacher checkpoint (v0) | `vbti/experiments/duck_cup_smolvla/v020/lerobot_output_r12/checkpoints/150000/pretrained_model` (step 150k — eval'd at 100% SR on duck_cup) |
| Reference paper | UVA — Unified Video Action Model ([arXiv 2503.00200](https://arxiv.org/abs/2503.00200)) |
| Reference repo | [github.com/ShuangLI59/unified_video_action](https://github.com/ShuangLI59/unified_video_action) |

---

## 0. Design Revision (2026-05-18) — Fix B: bake the future window

The original design (Sections 4–5 below) had the bake store **one** feature `(S,S,D)` per frame and relied on the policy config's `observation_delta_indices = [0, 1, ..., T]` to make the dataloader gather frames `t+1..t+T` at training time.

**This is broken.** `lerobot/datasets/factory.py:resolve_delta_timestamps` applies a single `observation_delta_indices` list to **every** `observation.*` key uniformly. Setting `[0..T]` would make the dataloader also gather `observation.images.*` and `observation.state` at future offsets, and SmolVLA's `prepare_images`/`prepare_state` take `batch[key][:, -1]` (the **last** index) — silently feeding a future frame as the current observation.

**Fix B (adopted):** the bake script assembles the **full future window per row**. For frame `t` it stores a `(t_future, S, S, D)` tensor = features of frames `t+1..t+t_future`, clamped at episode boundaries. Consequences:

- `SmolVLAUVAConfig` does **not** override `observation_delta_indices` — it stays the parent's `[0]`.
- The dataloader gathers `observation.video_features.*` at a single timestep, adding a size-1 obs-step dim: batch shape `(B, 1, t_future, S, S, D)`.
- `SmolVLAUVAPolicy.forward` drops that obs-step dim and reads the window directly — no `t=0` slicing.
- The dataset column stores a 4-dim per-row value; `meta/info.json` feature shape is `[t_future, S, S, D]`.
- `add_video_features.py` gains a `--t-future` CLI arg.

Where Sections 4.2 / 5.1 below describe the `observation_delta_indices` override or per-frame `(S,S,D)` storage, **this revision supersedes them.**

---

## 1. Goal

Add a UVA-style auxiliary loss to SmolVLA to improve sample efficiency on data-scarce manipulation fine-tuning. The aux task is **per-step prediction of future visual features**, computed against a frozen teacher checkpoint (v020). The auxiliary head is **dropped at inference** — only the backbone benefits from the regularization signal.

Concretely: at each training step, alongside the standard flow-matching action loss, the model predicts the SigLIP features of frames `t+1, ..., t+T_future` from the same camera. Loss is the equal-component MSE sum:

```
total_loss = action_loss + λ · video_loss      (λ = 0.3, constant)
```

The architecture follows UVA's pattern (shared backbone + two parallel heads) but **decouples target generation from training**:

- UVA uses a pretrained VAE and trains a diffusion video head end-to-end.
- We use a **frozen task-trained teacher (v020 SigLIP)** as the target encoder and a **deterministic MSE head**. No VAE dependency, no diffusion machinery, no online encoder forward at train time — all target features are baked into the dataset once.

---

## 2. Background and rationale

### 2.1 Why aux future-feature prediction

Aux feature prediction is a regularizer that forces the backbone to develop temporally-aware representations. Gradient from `video_loss` flows through the video head, back into `suffix_out`, through the action expert and (via cross-attention) into the trainable SigLIP vision tower. The head is discarded at inference; **the capability is absorbed into SigLIP + action expert weights**.

This is the same pattern as MAE, BYOL, DINO, V-JEPA: throwaway projection/predictor head, retained backbone.

### 2.2 Why UVA-style decoupled heads, not UWM-style deep mixing

UWM (arXiv 2504.02792) mixes action and video tokens in a single transformer with independent diffusion timesteps. Inference must run both tracks. UVA (arXiv 2503.00200) shares a backbone but **decouples decoding via two heads**, so video head is skipped at inference. Decision: UVA pattern — faster inference, simpler integration with SmolVLA's flow matching, suitable for real-time SO-ARM101 on 4070 Ti SUPER 16GB.

### 2.3 Why frozen v020 teacher instead of a generic VAE

UVA uses a pretrained VAE because they need video generation for their dynamic/inverse model variants. We need only the aux loss for representation learning. Trade-off:

- **Generic VAE (UVA-faithful)**: captures all visual statistics; task-agnostic.
- **v020 SigLIP frozen (our choice)**: captures *what duck_cup imitation learning shaped SigLIP to attend to* — gripper, cup, manipulation-relevant cues.

For data-scarce fine-tuning on a subset of v020's training distribution, the task-aware target is more efficient. This is feature distillation with a fixed teacher (DistillBert / BYOL-fixed-teacher analog).

### 2.4 Why deterministic MSE, not diffusion

Diffusion heads in UVA exist because they need to sample video at inference. We never sample — the head is dropped. Deterministic MSE on features is the standard for aux representation learning (BYOL, SimSiam, MAE, V-JEPA).

### 2.5 Why constant λ, not a schedule

Literature review (Annealing-KD, Auxiliary Task Reweighting for Min-Data Learning, SLGrad, UVA paper) supports **constant λ as the default**:

- UVA itself uses constant equal weights.
- Monotonic decay (high → low) risks catastrophic forgetting of temporal features when aux signal vanishes late.
- Data-scarce fine-tuning needs regularization *more* late in training, not less.
- The transition step for decay is ill-defined without separate aux-quality metric.

Constant is the conservative choice; schedules can be added in v1 if v021 evidence demands it. λ is configurable per run.

---

## 3. System architecture

Two-piece split — **policy code in the fork, bake script in vbti/**. They communicate exclusively via the dataset parquet (a new feature column).

```
┌─────────────────────── lerobot/ (our fork, vbti/main branch) ──────────────────────┐
│                                                                                     │
│  src/lerobot/policies/smolvla_uva/        ◄── NEW DIRECTORY                        │
│  ├── __init__.py                                                                    │
│  ├── configuration_smolvla_uva.py         # SmolVLAUVAConfig(SmolVLAConfig)        │
│  ├── modeling_smolvla_uva.py              # SmolVLAUVAPolicy + VLAFlowMatchingUVA  │
│  └── video_head.py                        # VideoHead module                       │
│                                                                                     │
│  src/lerobot/policies/smolvla/modeling_smolvla.py    ◄── REFACTOR PATCH            │
│       extract `_compute_suffix_out` helper from VLAFlowMatching.forward            │
│       (pure code motion, no behavior change)                                       │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                              │
                              │  push origin vbti/main; pull on remote
                              ▼
┌─────────────────────── vbti/ (project-local, not in fork) ─────────────────────────┐
│                                                                                     │
│  logic/dataset/                                                                     │
│  ├── add_video_features.py        # CLI bake script                                │
│  └── target_extractors/           # Modular extractor registry                     │
│      ├── __init__.py              # @register decorator + lookup                   │
│      └── siglip_output.py         # L2 extractor (v0 default; L1/L3 stubs)         │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Why fork-internal policy

Remote training ships `lerobot-train` only. The policy must be importable as `lerobot.policies.smolvla_uva.*`. Custom vbti-local policies cannot be loaded by remote training (the `SmolVLABackend` in `vbti/logic/train/backends/smolvla.py` is dead code on remote).

### 3.2 Why bake script is vbti-local

The bake script runs once per dataset/teacher combination, on the laptop. It never deploys to remote. It depends on the v020 checkpoint and laptop GPU; the remote machine never needs to load the teacher.

### 3.3 Factory registration via naming convention

Per `lerobot/src/lerobot/policies/factory.py:542` (`_get_policy_cls_from_policy_name`):

- Config class `SmolVLAUVAConfig` in `configuration_smolvla_uva.py` with `@PreTrainedConfig.register_subclass("smolvla_uva")`
- → factory looks for `SmolVLAUVAPolicy` in `modeling_smolvla_uva.py` via dynamic import

**No edit to `factory.py` needed.** Same for `make_pre_post_processors`: `isinstance(policy_cfg, SmolVLAConfig)` matches our subclass.

---

## 4. Data flow

### 4.1 Phase 1 — Bake (one-time, laptop)

```
Input:
  - dataset:            eternalmay33/06_black_cup_red_bg_depth
  - teacher checkpoint: vbti/experiments/duck_cup_smolvla/v020/lerobot_output_r12/
                        checkpoints/150000/pretrained_model
                        (step 150k — only checkpoint we've evaluated; 100% SR on duck_cup)
  - target camera:      observation.images.wrist
  - layer:              siglip_output  (L2)
  - spatial_size:       4              (avg-pool 16x16 -> 4x4)

Per frame (no_grad, fp16 storage):
  img (3, 512, 512)
    → resize_with_pad (SmolVLA preprocessing) → (3, 384, 384)
    → v020.vlm_with_expert.embed_image → feats (256, 1152)
    → reshape (16, 16, 1152)
    → avg_pool2d(kernel=4) → (4, 4, 1152)
    → fp16
    → parquet column: 'observation.video_features.siglip_output_4x4'

Output:
  - dataset gains one new column
  - per-frame storage: ~9.4 KB (4*4*1152*2 bytes)
  - ~4 min bake time on 4070 Ti SUPER for 50k frames
```

### 4.2 Phase 2 — Training (remote)

```
Batch:
  observation.images.wrist          (B, 3, 512, 512)
  observation.images.{others}       (B, 3, 512, 512)
  observation.video_features.       (B, T_future+1=5, 4, 4, 1152) fp16
    siglip_output_4x4                ◄── via observation_delta_indices = [0..T_future]
  observation.state                 (B, 1, 7)
  observation.language.tokens       (B, 48)
  action                            (B, 50, 7)

Forward (SmolVLAUVAPolicy):

  noise, time      = flow-matching noise schedule
  suffix_out       = self.model._compute_suffix_out(images, ..., x_t, time)  ◄── (B, 50, 720)
  v_t              = self.model.action_out_proj(suffix_out)                  ◄── (B, 50, 32)
  z_pred           = self.model.video_head(suffix_out[:, :T_future])         ◄── (B, 4, 16, 1152)
                     reshape → (B, 4, 4, 4, 1152)

  action_losses    = MSE(u_t, v_t)                                           ◄── (B, 50, 32)
  video_losses     = MSE(z_pred, batch[feature_key][:, 1:])                  ◄── (B, 4, 4, 4, 1152)

  action_loss      = action_losses.mean()
  video_loss       = video_losses.mean()
  total_loss       = action_loss + aux_weight * video_loss

  return total_loss, {action_loss, video_loss, loss, aux_weight}

Gradient (autograd-driven):
  Both losses feed into suffix_out. Combined gradient backpropagates through:
    expert layers → cross-attention → VLM (frozen text part) → SigLIP (trainable)
```

### 4.3 Phase 3 — Eval / Inference (anywhere)

```
- Dataset does NOT need video_features column.
- video_head loaded from checkpoint but never called.
- select_action / predict_action_chunk paths identical to vanilla SmolVLA.
- VRAM / latency cost: IDENTICAL.
```

### 4.4 Dataset column naming convention

```
observation.video_features.{layer}_{spatial}x{spatial}
```

Examples:
- `observation.video_features.siglip_output_4x4` ← v0 default
- `observation.video_features.siglip_output_8x8` ← v1 candidate (higher spatial res)
- `observation.video_features.siglip_layer8_4x4` ← v1 candidate (intermediate layer L1)

Multiple bakes can coexist in the same dataset. The policy config's `teacher_features_key` selects which one is read.

---

## 5. Components

### 5.1 `SmolVLAUVAConfig`

```python
@PreTrainedConfig.register_subclass("smolvla_uva")
@dataclass
class SmolVLAUVAConfig(SmolVLAConfig):
    # === UVA-specific ===
    teacher_features_key:  str   = "observation.video_features.siglip_output_4x4"
    teacher_feature_dim:   int   = 1152      # SmolVLM-500M SigLIP last-layer dim
    teacher_spatial_size:  int   = 4         # k x k grid in the target
    t_future:              int   = 4         # number of future frames predicted
    aux_weight:            float = 0.3       # constant λ
    video_head_hidden:     int   = 720       # = expert_hidden_size (auto-derived if -1)
    enable_aux_loss:       bool  = True      # False => bypass to parent forward

    @property
    def observation_delta_indices(self) -> list[int]:
        base = super().observation_delta_indices  # [0]
        if self.enable_aux_loss:
            return base + list(range(1, self.t_future + 1))
        return base

    def __post_init__(self):
        super().__post_init__()
        if self.enable_aux_loss:
            if self.t_future > self.chunk_size:
                raise ValueError(f"t_future ({self.t_future}) exceeds chunk_size ({self.chunk_size})")
            # Validate that feature_key encodes the same spatial_size we expect
            import re
            m = re.search(r"_(\d+)x(\d+)$", self.teacher_features_key)
            if m:
                n_h, n_w = int(m.group(1)), int(m.group(2))
                if n_h != self.teacher_spatial_size or n_w != self.teacher_spatial_size:
                    raise ValueError(
                        f"teacher_features_key '{self.teacher_features_key}' encodes {n_h}x{n_w} "
                        f"but teacher_spatial_size={self.teacher_spatial_size}"
                    )
```

### 5.2 `VideoHead`

Token-aligned MLP, ~1.1M params. Reads `suffix_out[:, :T_future]`, outputs per-patch features.

```python
class VideoHead(nn.Module):
    def __init__(self, h_in: int, feat_dim: int, t_future: int, num_patches: int):
        super().__init__()
        self.temporal_emb = nn.Parameter(torch.zeros(t_future, h_in))   # zero init
        self.spatial_emb  = nn.Parameter(torch.zeros(num_patches, h_in))  # zero init
        self.norm = nn.LayerNorm(h_in)
        self.mlp  = nn.Sequential(
            nn.Linear(h_in, h_in),
            nn.GELU(),
            nn.LayerNorm(h_in),
            nn.Linear(h_in, feat_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, T, h_in)  → out: (B, T, num_patches, feat_dim)
        z = z + self.temporal_emb[None]
        z = z.unsqueeze(2) + self.spatial_emb[None, None]
        z = self.norm(z)
        return self.mlp(z)
```

**Total parameters**: ~1.1M (vs SmolVLA's 450M backbone → 0.24% of model).

### 5.3 `VLAFlowMatchingUVA`

Inherits `VLAFlowMatching`. Adds `video_head`. Overrides `forward` to return `(action_losses, video_losses)` tuple. Reuses parent's `_compute_suffix_out` helper.

### 5.4 `SmolVLAUVAPolicy`

Inherits `SmolVLAPolicy`. Owns `VLAFlowMatchingUVA` as `self.model`. Overrides `forward` to combine losses. Inference paths (`predict_action_chunk`, `select_action`) are inherited unchanged.

### 5.5 `add_video_features.py` (vbti-local)

CLI script with arguments matching the modular abstraction:

```
usage: add_video_features.py --dataset <path-or-hub-id>
                             --teacher  <checkpoint-path>
                             --layer    siglip_output
                             --spatial-size 4
                             --target-camera observation.images.wrist
                             [--dtype fp16]
```

Loads target_extractor from registry, encodes frames in batches, writes new column.

### 5.6 Refactor patch on `VLAFlowMatching`

In `lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py`, extract `_compute_suffix_out` helper from existing `forward()`. Pure code motion, no behavior change. Tested via tier-1 invariance test.

---

## 6. Configuration (full)

```python
SmolVLAUVAConfig fields:
  # === Inherited from SmolVLAConfig ===
  ...all parent fields...

  # === UVA additions ===
  teacher_features_key:  str   = "observation.video_features.siglip_output_4x4"
  teacher_feature_dim:   int   = 1152
  teacher_spatial_size:  int   = 4
  t_future:              int   = 4
  aux_weight:            float = 0.3
  video_head_hidden:     int   = 720
  enable_aux_loss:       bool  = True
```

| Field | v0 default | Configurable later? |
|---|---|---|
| Target representation (teacher choice) | v020 SigLIP frozen | Yes — different teacher_checkpoint at bake |
| Layer | L2 (SigLIP output) | Yes — `layer` bake arg, plus new `teacher_features_key` at train |
| Spatial granularity | 4×4 avg-pool | Yes — `spatial_size` bake arg |
| Target camera | wrist | Yes — `target_camera` bake arg |
| Video head architecture | Token-aligned MLP, ~1.1M params | Yes — `video_head_hidden` |
| λ (aux weight) | 0.3, constant | Yes — `aux_weight` |
| λ schedule | Constant | No (v1+ extension) |
| Vision encoder during aux | Always on, no warm-up | No (v1+ extension) |
| Missing-features handling | Hard fail at config init | No (strict requirement) |
| Checkpoint loading | Permissive both ways, log drops | No (sensible default) |
| Factory registration | Dynamic fallback (no edit) | n/a |

---

## 7. Behavior contracts

### 7.1 Three modes

| Mode | `enable_aux_loss` | Behavior | Use case |
|---|---|---|---|
| UVA on | `True` | Both losses computed; hard-fail if feature column missing | Round 1+ training |
| UVA off | `False` | Bypasses to `super().forward()`; video_head dormant; column not required | Ablation, eval, continued vanilla fine-tune from UVA checkpoint |
| Inference | n/a | No loss path; video_head never called | Robot rollouts |

Mode is per-run config (startup-only); changing mid-training invalidates optimizer state.

### 7.2 Error handling

| Case | Detected at | Behavior |
|---|---|---|
| `enable_aux_loss=True` + feature column missing | `__post_init__` (reads `ds_meta.features`) | `ValueError` with offending key + bake-script hint |
| Feature column shape ≠ config | First batch (torch shape error) | `RuntimeError` surfaces via autograd |
| `teacher_features_key` pattern ≠ `teacher_spatial_size` | `__post_init__` (regex check) | `ValueError` |
| `t_future > chunk_size` | `__post_init__` | `ValueError` |

### 7.3 Checkpoint save/load

| Source | Target | Behavior |
|---|---|---|
| vanilla SmolVLA → `SmolVLAUVAPolicy` | Cold start UVA | Load matching keys, init video_head fresh (Kaiming linears, zero pos embeds). One-line info log of fresh-init keys. |
| UVA → `SmolVLAUVAPolicy` | Continued / eval | Full bit-identical load |
| **UVA → vanilla `SmolVLAPolicy`** | **Most common inference path** | **Load everything except `video_head.*`. One-line info log of dropped keys. Permissive default.** |
| UVA → `SmolVLAUVAPolicy` with `enable_aux_loss=False` | Vanilla-mode eval from UVA ckpt | Full load; head dormant |

Implementation: both directions use `load_state_dict(strict=False)` with explicit logging of unexpected/missing keys.

### 7.4 Forward-pass invariants

1. `enable_aux_loss=False` produces bit-identical loss & gradients to vanilla `SmolVLAPolicy`.
2. `aux_weight=0.0` (with `enable_aux_loss=True`) produces identical gradients to vanilla, but loss dict still logs `video_loss` for monitoring.
3. Same checkpoint loaded with `enable_aux_loss` True vs False produces identical `select_action` outputs.

---

## 8. Lifecycle / iteration plan

This design supports iterated knowledge distillation. Each round produces a stronger teacher for the next round.

```
Round 0 (DONE)      v020/150k     ◄── existing baseline, trained on duck_cup
                                      (eval'd at step 150k → 100% SR; unknown perf at other steps)
                     │  freeze SigLIP from this checkpoint
                     │  bake features into 06_black_cup_red_bg_depth
                     ▼
Round 1 (THIS DESIGN) v021_uva    ◄── train SmolVLAUVAPolicy with teacher=v020/150k.siglip
                     │  evaluate: v021_uva > v020 on real eval?
                     │            no  → UVA didn't help on this task; stop
                     │            yes → continue
                     │  freeze v021.siglip, re-bake features
                     ▼
Round 2 (FUTURE)    v022_uva      ◄── train with teacher=v021.siglip
                     │
                     ▼
                    plateau or diminishing returns → stop iterating
```

Rules:

1. Only adopt the next teacher if it beats the previous on real eval (not just training loss).
2. Re-bake required when teacher changes (~4 min per dataset on 4070 Ti SUPER).
3. Practical cap: 2 iterations. Round 3+ rarely yields meaningful gains in literature.

---

## 9. Out of scope (v1+ candidates)

Deferred items, in priority order:

| Item | Reason deferred |
|---|---|
| L1 (intermediate SigLIP layer) target | Need L2 baseline first to know if signal saturates |
| L3 (after VLM) target | Requires running full v020 VLM per frame; heavier bake |
| Multi-camera target features | One target cam sufficient; multi-cam is 4× cost |
| Diffusion video head (UVA-faithful) | Deterministic MSE is simpler and likely sufficient |
| EMA teacher (V-JEPA style) | Fixed v020 teacher is more deterministic for first comparison |
| λ warm-up (β: low → target) | v0 evidence may show it unnecessary |
| λ decay (high → low) | Risk of forgetting; only add if late-training interference observed |
| Two-stage training (Annealing-KD style) | Add only if persistent task interference |
| GradNorm / uncertainty-based dynamic λ | Out of scope; constant only |
| Per-position video loss logging | Scalar mean sufficient for v0 |
| Validation-set aux loss tracking | Possible diagnostic; not required |

---

## 10. Testing strategy

### 10.1 Tier 1 — Refactor invariance

```python
def test_vlaflowmatching_forward_unchanged():
    """Refactor of _compute_suffix_out helper is pure code motion."""
    # Fixed seed; same batch, noise, time
    # Loss tensor matches pre-refactor reference
```

### 10.2 Tier 2 — `enable_aux_loss=False` parity

```python
def test_uva_policy_with_aux_disabled_matches_vanilla():
    """SmolVLAUVAPolicy(enable_aux_loss=False) == SmolVLAPolicy bit-exactly."""
```

### 10.3 Tier 3 — Shape & dtype sanity

```python
def test_video_head_shapes()
def test_config_validation_catches_key_size_mismatch()
def test_config_validation_catches_t_future_exceeds_chunk_size()
```

### 10.4 Tier 4 — End-to-end overfit smoke

```python
def test_end_to_end_overfit_one_episode():
    """Bake one episode, train 100 steps, action_loss and video_loss both drop."""
```

### 10.5 Tier 5 — Bake script smoke

```python
def test_bake_script_writes_expected_shape()
```

### 10.6 Manual integration checks (pre-first-real-training)

1. Bake on `06_black_cup_red_bg_depth`. Inspect 5 rows. Verify storage size matches estimate.
2. `scripts/distill_overfit_sanity.py`-style run for 200 steps, batch=8. Verify both losses drop, no NaN, wandb logs all three losses + aux_weight separately.
3. Save checkpoint. Load into vanilla `SmolVLAPolicy`. Compare `select_action` outputs to UVA-class load — must be bit-identical (video_head not called).

---

## 11. Summary of decisions

| Dimension | Decision | Rationale |
|---|---|---|
| Aux task type | Future visual-feature prediction | UVA-style, regularizes backbone for temporal awareness |
| Backbone integration | UVA-style (decoupled heads), not UWM-style (mixed tokens) | Faster inference; cleaner SmolVLA integration |
| Target representation | v020 SigLIP frozen (Option D), not VAE (Option A) or moving target (Option B) | Task-aware target without VAE dep; stable target during training |
| Target layer | L2 (SigLIP last-layer output) | Natural architectural boundary; rich spatial info without VLM dependency at bake |
| Spatial granularity | 4×4 (8× downpool from 16×16) | Retains quadrant-level layout; ~470 MB dataset bloat |
| Target camera | Wrist only | Most action-informative; single-cam bake cost |
| Video head | Token-aligned MLP, ~1.1M params | Small enough to force backbone signal; H2 variant from design |
| Loss function | Deterministic MSE on features | No need for diffusion (no sampling at inference) |
| λ (aux weight) | 0.3, constant | Standard practice; data-scarce needs sustained regularization |
| λ schedule | Constant | Decay risks catastrophic forgetting; literature mostly constant |
| Vision-encoder gating | Always on, no warm-up | Simplest; short training schedule; add warm-up only if instability observed |
| Missing-feature handling | Hard fail at config init | Permissive fallback hides misconfiguration silently |
| Checkpoint loading | Permissive both directions, log drops | UVA → vanilla path is the common inference path |
| Factory.py edit | None — use dynamic fallback | Cleaner fork patch surface |
| Refactor of parent | Extract `_compute_suffix_out` helper | Enables subclass reuse without copy-paste |
| Dataset for v0 | `eternalmay33/06_black_cup_red_bg_depth` | Subset of v020's training distribution → teacher fully aligned |
| Teacher checkpoint | v020 step 150k (`lerobot_output_r12/checkpoints/150000/pretrained_model`) | Only checkpoint we have eval evidence for (100% SR on duck_cup); other steps unknown |
| Lifecycle | Round 1 = train v021_uva with v020 teacher; iterate only if eval improves | Born-Again Networks / IKD pattern; max 2 rounds practical |

---

## 12. Open questions / risks

| Risk | Mitigation |
|---|---|
| v020 teacher features may be too task-specific, locking student to v020's narrow representation | Round 1 evaluation will show; can swap teacher per dataset column without code change |
| λ=0.3 may be wrong by 2-3×; only constant means we can't adapt mid-run | Run a small sweep (λ ∈ {0.1, 0.3, 1.0}) before committing to a Round 1 trainer |
| Aux loss may interfere with action loss late in training | Both losses logged separately; if action_loss climbs while video_loss drops, add λ warm-up or decay in v1 |
| VRAM headroom is tight (15.6 GB / 16 GB at BS=32) | Video head adds ~1.1M params; expected overhead <2%. Verify in tier-4 smoke test. |
| `_compute_suffix_out` refactor breaks existing SmolVLA training | Tier-1 invariance test gates the refactor |

---

## 13. References

- Li, Gao, Sadigh, Song. *Unified Video Action Model.* RSS 2025. [arXiv:2503.00200](https://arxiv.org/abs/2503.00200)
- Zhu et al. *Unified World Models (UWM).* [arXiv:2504.02792](https://arxiv.org/abs/2504.02792)
- Jafari et al. *Annealing Knowledge Distillation.* EACL 2021.
- Shi et al. *Auxiliary Task Reweighting for Minimum-data Learning.* NeurIPS 2020.
- He et al. *Masked Autoencoders Are Scalable Vision Learners.* CVPR 2022 (MAE pattern reference).
- Grill et al. *Bootstrap Your Own Latent (BYOL).* NeurIPS 2020 (fixed-teacher reference).
- Caron et al. *DINO / DINOv2.* (target-distillation pattern reference).
- Bardes et al. *V-JEPA.* (predictive feature learning reference).

Memory references:
- `remote_lerobot_patches.md` — fork base v0.4.4, 10 vbti patches, sync fingerprint
- `project_remote_training_path.md` — remote.py ships lerobot-train; custom code must live in fork
- `project_smolvla_vram_anchor.md` — VRAM budget on 4070 Ti SUPER
- `duck_cup_sota_plan.md` — broader research roadmap context
