# SmolVLA-UVA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `smolvla_uva` policy class to our lerobot fork that augments SmolVLA training with a UVA-style auxiliary future-feature-prediction loss (target = frozen v020 SigLIP features, baked into the dataset offline).

**Architecture:** Two-piece split. (1) Fork-internal: new `smolvla_uva/` policy directory + one refactor patch to `modeling_smolvla.py`. (2) vbti-local: bake script + target-extractor registry. They communicate only via a new parquet column on the dataset. Inference path is identical to vanilla SmolVLA (video head is dropped). See spec: `docs/superpowers/specs/2026-05-13-smolvla-uva-design.md`.

**Tech Stack:** Python 3.10/3.12, PyTorch 2.7+cu128, lerobot v0.4.4 (our fork `33may/lerobot` `vbti/main`), pytest, safetensors, huggingface_hub.

---

## File Structure

### Lerobot fork additions

| File | Responsibility | New/Modified |
|---|---|---|
| `lerobot/src/lerobot/policies/smolvla_uva/__init__.py` | Package marker; re-export config & policy | New |
| `lerobot/src/lerobot/policies/smolvla_uva/video_head.py` | `VideoHead` `nn.Module` — token-aligned MLP producing per-patch feature predictions | New |
| `lerobot/src/lerobot/policies/smolvla_uva/configuration_smolvla_uva.py` | `SmolVLAUVAConfig` dataclass extending `SmolVLAConfig` + `__post_init__` validation | New |
| `lerobot/src/lerobot/policies/smolvla_uva/modeling_smolvla_uva.py` | `VLAFlowMatchingUVA` (inner) + `SmolVLAUVAPolicy` (outer) | New |
| `lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py` | Extract `_compute_suffix_out` helper from existing `VLAFlowMatching.forward` (pure code motion) | Modified |
| `lerobot/tests/policies/smolvla/test_refactor_invariance.py` | Characterization test for `VLAFlowMatching.forward` before/after the refactor | New |
| `lerobot/tests/policies/smolvla_uva/__init__.py` | Test package marker | New |
| `lerobot/tests/policies/smolvla_uva/test_video_head.py` | Unit tests for `VideoHead` shapes/init/gradients | New |
| `lerobot/tests/policies/smolvla_uva/test_config.py` | `SmolVLAUVAConfig` validation tests | New |
| `lerobot/tests/policies/smolvla_uva/test_modeling.py` | `SmolVLAUVAPolicy` parity + loss-combiner tests | New |
| `lerobot/tests/policies/smolvla_uva/test_checkpoint.py` | UVA↔vanilla checkpoint loading tests | New |

### vbti additions

| File | Responsibility | New |
|---|---|---|
| `vbti/logic/dataset/target_extractors/__init__.py` | Registry decorator + lookup function | New |
| `vbti/logic/dataset/target_extractors/siglip_output.py` | L2 extractor: forward image through teacher's SigLIP, optional spatial pool | New |
| `vbti/logic/dataset/add_video_features.py` | CLI bake script: load teacher, walk dataset, write new parquet column | New |
| `scripts/smolvla_uva_overfit_sanity.py` | End-to-end smoke test (mirrors `distill_overfit_sanity.py`) | New |

---

## Pre-flight verification

Before Task 1, confirm working tree is clean:

```bash
cd /home/may33/projects/ml_portfolio/robotics
git -C lerobot status
# Should be on vbti/main with no uncommitted changes
```

If lerobot has uncommitted changes, stash or commit them before proceeding.

---

### Task 1: Pre-refactor characterization test for `VLAFlowMatching.forward`

**Why this task matters:** The refactor in Task 2 is pure code motion — but "pure code motion" is famously where subtle bugs hide. This characterization test pins down the *exact* forward behavior of the current `VLAFlowMatching.forward` so we can prove the refactor changes nothing.

**Files:**
- Create: `lerobot/tests/policies/smolvla/test_refactor_invariance.py`

**Note on test design:** This is a "characterization test" — captures current behavior, doesn't fail. After the refactor in Task 2, it must still pass.

- [ ] **Step 1: Write the characterization test**

```python
# lerobot/tests/policies/smolvla/test_refactor_invariance.py
"""Characterization test pinning VLAFlowMatching.forward behavior before/after refactor.

The refactor extracts a _compute_suffix_out helper from forward(). This test asserts
that for a fixed seed/batch/noise/time, the returned losses tensor is bit-identical.
"""
import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.utils.random_utils import set_seed
from tests.utils import require_cuda, require_package


@require_package("transformers")
@require_cuda
def test_vlaflowmatching_forward_deterministic():
    """For a fixed seed and inputs, VLAFlowMatching.forward must produce identical losses."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    set_seed(42)
    config = SmolVLAConfig(max_action_dim=7, chunk_size=10, num_vlm_layers=4)
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    config.device = "cuda"

    policy = SmolVLAPolicy(config).to("cuda").eval()

    # Synthetic batch
    B = 2
    batch = {
        "observation.images.wrist": torch.rand(B, 1, 3, 224, 224, device="cuda"),
        "observation.state": torch.randn(B, 1, 7, device="cuda"),
        "action": torch.randn(B, config.chunk_size, 7, device="cuda"),
        "observation.language.tokens": torch.zeros(B, 4, dtype=torch.long, device="cuda"),
        "observation.language.attention_mask": torch.ones(B, 4, dtype=torch.long, device="cuda"),
    }

    # Fixed noise & time to make the forward fully deterministic
    set_seed(0)
    noise = torch.randn(B, config.chunk_size, config.max_action_dim, device="cuda")
    time = torch.full((B,), 0.5, device="cuda")

    loss1, _ = policy.forward(batch, noise=noise, time=time)
    loss2, _ = policy.forward(batch, noise=noise, time=time)

    assert torch.allclose(loss1, loss2), "forward must be deterministic for fixed inputs"
    # Save reference value for cross-refactor comparison (printed to log)
    print(f"REFERENCE_LOSS={loss1.item():.10f}")
```

- [ ] **Step 2: Run test against current (pre-refactor) code**

Run: `cd /home/may33/projects/ml_portfolio/robotics && pytest lerobot/tests/policies/smolvla/test_refactor_invariance.py::test_vlaflowmatching_forward_deterministic -v -s`

Expected: PASS. Record the printed `REFERENCE_LOSS=...` value.

- [ ] **Step 3: Pin the reference value into the test**

Edit the test to add the recorded reference value as a stricter assertion:

```python
    # Pinned reference from pre-refactor run (regression guard)
    EXPECTED_LOSS = <recorded value>  # e.g., 1.2345678901
    assert abs(loss1.item() - EXPECTED_LOSS) < 1e-6, \
        f"forward output drifted from pre-refactor reference: {loss1.item()} != {EXPECTED_LOSS}"
```

- [ ] **Step 4: Re-run test to confirm**

Run: `pytest lerobot/tests/policies/smolvla/test_refactor_invariance.py -v`
Expected: PASS.

- [ ] **Step 5: Commit characterization test**

```bash
cd /home/may33/projects/ml_portfolio/robotics/lerobot
git add tests/policies/smolvla/test_refactor_invariance.py
git commit -m "test: pin VLAFlowMatching.forward characterization for upcoming refactor"
```

---

### Task 2: Refactor `VLAFlowMatching` — extract `_compute_suffix_out` helper

**Why this task matters:** `VLAFlowMatchingUVA` (Task 7) needs to reuse the prefix-embed → VLM-forward → expert pipeline without copy-pasting it. Extracting a helper lets the subclass call it once and tap `suffix_out` for both heads.

**Files:**
- Modify: `lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py:762-798`

- [ ] **Step 1: Read current `VLAFlowMatching.forward`**

Open `lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py` and re-read lines 762-798. Confirm the structure: `forward()` takes `(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)`, sets up `x_t`/`u_t`, calls `embed_prefix`/`embed_suffix`, runs `vlm_with_expert.forward(...)`, slices `suffix_out[:, -self.config.chunk_size:]`, casts to float32, then `action_out_proj` + MSE.

- [ ] **Step 2: Extract `_compute_suffix_out` helper**

Replace the current `forward` method (lines 762-798) with:

```python
    def _compute_suffix_out(self, images, img_masks, lang_tokens, lang_masks, state, x_t, time):
        """Run prefix-embed + suffix-embed + VLM+expert forward, return suffix_out.

        Returns:
            Tensor of shape (B, chunk_size, expert_hidden_size), upcast to float32.
        """
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return suffix_out

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        suffix_out = self._compute_suffix_out(images, img_masks, lang_tokens, lang_masks, state, x_t, time)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses
```

- [ ] **Step 3: Run the characterization test from Task 1**

Run: `pytest lerobot/tests/policies/smolvla/test_refactor_invariance.py -v`
Expected: PASS. Bit-identical loss to pre-refactor reference.

- [ ] **Step 4: Run full smolvla test suite**

Run: `pytest lerobot/tests/policies/smolvla/ -v`
Expected: All existing tests still PASS.

- [ ] **Step 5: Commit refactor**

```bash
cd /home/may33/projects/ml_portfolio/robotics/lerobot
git add src/lerobot/policies/smolvla/modeling_smolvla.py
git commit -m "vbti: refactor VLAFlowMatching to extract _compute_suffix_out helper

Pure code motion. Enables subclass reuse for UVA aux-loss policy
without copy-pasting the prefix/suffix-embed + expert-forward pipeline.

Characterization test pins identical forward output pre- and post-refactor."
```

---

### Task 3: `VideoHead` module + unit tests

**Why this task matters:** The aux head is small (~1.1M params) by design — to force the backbone to carry the prediction signal rather than the head explaining everything. This task pins down its shapes and gradient behavior in isolation.

**Files:**
- Create: `lerobot/src/lerobot/policies/smolvla_uva/__init__.py`
- Create: `lerobot/src/lerobot/policies/smolvla_uva/video_head.py`
- Create: `lerobot/tests/policies/smolvla_uva/__init__.py`
- Create: `lerobot/tests/policies/smolvla_uva/test_video_head.py`

- [ ] **Step 1: Create package skeletons**

```bash
mkdir -p /home/may33/projects/ml_portfolio/robotics/lerobot/src/lerobot/policies/smolvla_uva
mkdir -p /home/may33/projects/ml_portfolio/robotics/lerobot/tests/policies/smolvla_uva
```

Write `lerobot/src/lerobot/policies/smolvla_uva/__init__.py`:
```python
# Re-exports are added at the bottom of this file after class definitions in
# Tasks 6 and 8. For now, this file marks the package.
```

Write `lerobot/tests/policies/smolvla_uva/__init__.py` as an empty file.

- [ ] **Step 2: Write failing unit tests for `VideoHead`**

Write `lerobot/tests/policies/smolvla_uva/test_video_head.py`:

```python
"""Unit tests for VideoHead — the aux-loss prediction head."""
import pytest
import torch

from lerobot.policies.smolvla_uva.video_head import VideoHead


def test_video_head_output_shape():
    """Output shape is (B, T, num_patches, feat_dim)."""
    head = VideoHead(h_in=64, feat_dim=128, t_future=4, num_patches=16)
    z = torch.randn(2, 4, 64)
    out = head(z)
    assert out.shape == (2, 4, 16, 128)


def test_video_head_positional_embeddings_zero_init():
    """Temporal & spatial pos embeddings must start at zero so head starts neutral."""
    head = VideoHead(h_in=64, feat_dim=128, t_future=4, num_patches=16)
    assert torch.equal(head.temporal_emb, torch.zeros(4, 64))
    assert torch.equal(head.spatial_emb, torch.zeros(16, 64))


def test_video_head_gradient_flows_to_input():
    """Backward from MSE loss must produce non-zero gradient on input z."""
    head = VideoHead(h_in=64, feat_dim=128, t_future=4, num_patches=16)
    z = torch.randn(2, 4, 64, requires_grad=True)
    target = torch.randn(2, 4, 16, 128)
    loss = torch.nn.functional.mse_loss(head(z), target)
    loss.backward()
    assert z.grad is not None
    assert (z.grad.abs().sum() > 0).item(), "gradient must propagate to input"


def test_video_head_parameter_count_small():
    """Head must stay ~1M params — large enough to project, small enough to be lossy."""
    head = VideoHead(h_in=720, feat_dim=1152, t_future=4, num_patches=16)
    n_params = sum(p.numel() for p in head.parameters())
    assert 500_000 < n_params < 2_000_000, f"got {n_params} params, expected ~1.1M"
```

- [ ] **Step 3: Run test, verify failure**

Run: `cd /home/may33/projects/ml_portfolio/robotics && pytest lerobot/tests/policies/smolvla_uva/test_video_head.py -v`
Expected: FAIL with `ImportError: cannot import name 'VideoHead'`

- [ ] **Step 4: Implement `VideoHead`**

Write `lerobot/src/lerobot/policies/smolvla_uva/video_head.py`:

```python
"""VideoHead: token-aligned MLP for UVA-style auxiliary feature prediction.

Reads suffix_out tokens from the action expert and projects them to per-patch
target features in the teacher's space. Lightweight by design (~1.1M params) so
the backbone, not the head, carries the prediction signal.
"""
import torch
from torch import Tensor, nn


class VideoHead(nn.Module):
    def __init__(self, h_in: int, feat_dim: int, t_future: int, num_patches: int):
        """
        Args:
            h_in: backbone hidden dim (matches SmolVLA expert_hidden_size).
            feat_dim: target feature dim (e.g., 1152 for SmolVLM-500M SigLIP last layer).
            t_future: number of future timesteps to predict.
            num_patches: spatial grid size flattened (e.g., 4x4 -> 16).
        """
        super().__init__()
        self.temporal_emb = nn.Parameter(torch.zeros(t_future, h_in))
        self.spatial_emb = nn.Parameter(torch.zeros(num_patches, h_in))
        self.norm = nn.LayerNorm(h_in)
        self.mlp = nn.Sequential(
            nn.Linear(h_in, h_in),
            nn.GELU(),
            nn.LayerNorm(h_in),
            nn.Linear(h_in, feat_dim),
        )

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z: backbone tokens, shape (B, T, h_in). T == t_future.

        Returns:
            (B, T, num_patches, feat_dim) predicted features per patch.
        """
        z = z + self.temporal_emb[None]                           # (B, T, h_in)
        z = z.unsqueeze(2) + self.spatial_emb[None, None]          # (B, T, num_patches, h_in)
        z = self.norm(z)
        return self.mlp(z)
```

- [ ] **Step 5: Run tests, verify pass**

Run: `pytest lerobot/tests/policies/smolvla_uva/test_video_head.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/may33/projects/ml_portfolio/robotics/lerobot
git add src/lerobot/policies/smolvla_uva/__init__.py \
        src/lerobot/policies/smolvla_uva/video_head.py \
        tests/policies/smolvla_uva/__init__.py \
        tests/policies/smolvla_uva/test_video_head.py
git commit -m "vbti: add VideoHead module for UVA aux-loss prediction

Token-aligned MLP. ~1.1M params. Zero-init pos embeddings.
Reads suffix_out tokens, outputs per-patch feature predictions.
Unit tests cover shape contracts, gradient flow, parameter count."
```

---

### Task 4: `SmolVLAUVAConfig` + validation tests

**Why this task matters:** The config holds every UVA-specific knob (target key, λ, t_future, etc.) and its `__post_init__` catches the two common misconfigurations: feature-key/spatial-size mismatch and t_future exceeding chunk_size. Catching these at config-init is much faster than discovering shape errors mid-training.

**Files:**
- Create: `lerobot/src/lerobot/policies/smolvla_uva/configuration_smolvla_uva.py`
- Create: `lerobot/tests/policies/smolvla_uva/test_config.py`

- [ ] **Step 1: Write failing config tests**

Write `lerobot/tests/policies/smolvla_uva/test_config.py`:

```python
"""Tests for SmolVLAUVAConfig validation."""
import pytest

from lerobot.policies.smolvla_uva.configuration_smolvla_uva import SmolVLAUVAConfig


def test_default_config_valid():
    """Defaults must pass __post_init__ unchanged."""
    cfg = SmolVLAUVAConfig()
    assert cfg.aux_weight == 0.3
    assert cfg.t_future == 4
    assert cfg.teacher_spatial_size == 4
    assert cfg.teacher_feature_dim == 1152
    assert cfg.enable_aux_loss is True


def test_config_observation_delta_indices_with_aux():
    """When aux is on, delta indices must include the future-frame offsets."""
    cfg = SmolVLAUVAConfig(t_future=4)
    assert cfg.observation_delta_indices == [0, 1, 2, 3, 4]


def test_config_observation_delta_indices_without_aux():
    """When aux is off, delta indices match parent (single-frame observation)."""
    cfg = SmolVLAUVAConfig(enable_aux_loss=False)
    assert cfg.observation_delta_indices == [0]


def test_config_validates_key_vs_spatial_size_mismatch():
    """Feature key encoding ..._4x4 with teacher_spatial_size=8 must raise."""
    with pytest.raises(ValueError, match="teacher_features_key"):
        SmolVLAUVAConfig(
            teacher_features_key="observation.video_features.siglip_output_4x4",
            teacher_spatial_size=8,
        )


def test_config_validates_t_future_within_chunk_size():
    """t_future > chunk_size is structurally impossible (head reads first T tokens)."""
    with pytest.raises(ValueError, match="t_future"):
        SmolVLAUVAConfig(t_future=100, chunk_size=50)


def test_config_registered_as_smolvla_uva():
    """Config must register so policy factory's dynamic fallback finds it."""
    from lerobot.configs.policies import PreTrainedConfig
    assert "smolvla_uva" in PreTrainedConfig.get_known_choices()
```

- [ ] **Step 2: Run tests, verify failure**

Run: `pytest lerobot/tests/policies/smolvla_uva/test_config.py -v`
Expected: All FAIL with `ImportError`.

- [ ] **Step 3: Implement `SmolVLAUVAConfig`**

Write `lerobot/src/lerobot/policies/smolvla_uva/configuration_smolvla_uva.py`:

```python
"""SmolVLAUVAConfig — extends SmolVLAConfig with UVA aux-loss fields."""
import re
from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("smolvla_uva")
@dataclass
class SmolVLAUVAConfig(SmolVLAConfig):
    # === UVA-specific fields ===
    teacher_features_key: str = "observation.video_features.siglip_output_4x4"
    teacher_feature_dim: int = 1152          # SmolVLM-500M SigLIP last-layer dim
    teacher_spatial_size: int = 4            # k x k grid in the target
    t_future: int = 4                        # number of future frames predicted
    aux_weight: float = 0.3                  # constant lambda for video_loss
    video_head_hidden: int = 720             # = SmolVLA expert_hidden_size
    enable_aux_loss: bool = True             # False => bypass to parent forward

    @property
    def observation_delta_indices(self) -> list:
        base = super().observation_delta_indices  # [0]
        if self.enable_aux_loss:
            return base + list(range(1, self.t_future + 1))
        return base

    def __post_init__(self):
        super().__post_init__()
        if not self.enable_aux_loss:
            return
        if self.t_future > self.chunk_size:
            raise ValueError(
                f"t_future ({self.t_future}) cannot exceed chunk_size ({self.chunk_size}); "
                f"video head reads suffix_out[:, :t_future]"
            )
        # Validate that teacher_features_key encodes the same spatial size as config
        m = re.search(r"_(\d+)x(\d+)$", self.teacher_features_key)
        if m:
            n_h, n_w = int(m.group(1)), int(m.group(2))
            if n_h != self.teacher_spatial_size or n_w != self.teacher_spatial_size:
                raise ValueError(
                    f"teacher_features_key '{self.teacher_features_key}' encodes "
                    f"{n_h}x{n_w} but teacher_spatial_size={self.teacher_spatial_size}"
                )
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest lerobot/tests/policies/smolvla_uva/test_config.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/may33/projects/ml_portfolio/robotics/lerobot
git add src/lerobot/policies/smolvla_uva/configuration_smolvla_uva.py \
        tests/policies/smolvla_uva/test_config.py
git commit -m "vbti: add SmolVLAUVAConfig with __post_init__ validation

Registers 'smolvla_uva' policy type. Validates feature-key/spatial-size
match and t_future<=chunk_size at config-init time (fail fast)."
```

---

### Task 5: `VLAFlowMatchingUVA` — inner model with video head

**Why this task matters:** The inner model owns the video head and exposes the dual-loss forward. By inheriting from `VLAFlowMatching` and reusing `_compute_suffix_out`, the prefix/suffix/VLM/expert pipeline is computed *once* per batch — both heads consume the same `suffix_out`.

**Files:**
- Modify: `lerobot/src/lerobot/policies/smolvla_uva/modeling_smolvla_uva.py` (create)
- Create: `lerobot/tests/policies/smolvla_uva/test_modeling.py`

- [ ] **Step 1: Write failing test for `VLAFlowMatchingUVA.forward` return shape**

Write `lerobot/tests/policies/smolvla_uva/test_modeling.py`:

```python
"""Tests for VLAFlowMatchingUVA + SmolVLAUVAPolicy."""
import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla_uva.configuration_smolvla_uva import SmolVLAUVAConfig
from lerobot.utils.random_utils import set_seed
from tests.utils import require_cuda, require_package


def _make_uva_config(**overrides) -> SmolVLAUVAConfig:
    """Helper: minimal UVA config that can instantiate without real dataset."""
    cfg = SmolVLAUVAConfig(
        max_action_dim=7, chunk_size=10, num_vlm_layers=4,
        t_future=2, teacher_feature_dim=64, teacher_spatial_size=2,
        video_head_hidden=64,
        teacher_features_key="observation.video_features.siglip_output_2x2",
        **overrides,
    )
    cfg.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    cfg.device = "cuda"
    return cfg


@require_package("transformers")
@require_cuda
def test_inner_forward_returns_action_and_video_losses():
    """VLAFlowMatchingUVA.forward returns (action_losses, video_losses) tuple."""
    from lerobot.policies.smolvla_uva.modeling_smolvla_uva import SmolVLAUVAPolicy

    set_seed(42)
    cfg = _make_uva_config()
    policy = SmolVLAUVAPolicy(cfg).to("cuda").eval()

    B = 2
    target_shape = (B, cfg.t_future, cfg.teacher_spatial_size, cfg.teacher_spatial_size, cfg.teacher_feature_dim)
    batch = {
        "observation.images.wrist": torch.rand(B, 1, 3, 224, 224, device="cuda"),
        "observation.state": torch.randn(B, 1, 7, device="cuda"),
        "action": torch.randn(B, cfg.chunk_size, 7, device="cuda"),
        "observation.language.tokens": torch.zeros(B, 4, dtype=torch.long, device="cuda"),
        "observation.language.attention_mask": torch.ones(B, 4, dtype=torch.long, device="cuda"),
        cfg.teacher_features_key: torch.randn(B, cfg.t_future + 1, *target_shape[2:], device="cuda"),
    }

    loss, loss_dict = policy.forward(batch)
    assert "action_loss" in loss_dict
    assert "video_loss" in loss_dict
    assert "aux_weight" in loss_dict
    assert loss_dict["aux_weight"] == cfg.aux_weight
    # total = action + aux_weight * video
    expected = loss_dict["action_loss"] + cfg.aux_weight * loss_dict["video_loss"]
    assert abs(loss.item() - expected) < 1e-5
```

- [ ] **Step 2: Run test, verify failure**

Run: `pytest lerobot/tests/policies/smolvla_uva/test_modeling.py::test_inner_forward_returns_action_and_video_losses -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `VLAFlowMatchingUVA`**

Write the FIRST HALF of `lerobot/src/lerobot/policies/smolvla_uva/modeling_smolvla_uva.py`:

```python
"""SmolVLA-UVA: SmolVLA + UVA-style auxiliary future-feature prediction.

Adds a video head reading suffix_out[:, :t_future] and an MSE-vs-teacher-features
auxiliary loss. Inference path is identical to SmolVLA (video head dropped).
"""
import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.smolvla.modeling_smolvla import (
    SmolVLAPolicy,
    VLAFlowMatching,
)
from lerobot.policies.smolvla_uva.configuration_smolvla_uva import SmolVLAUVAConfig
from lerobot.policies.smolvla_uva.video_head import VideoHead


class VLAFlowMatchingUVA(VLAFlowMatching):
    """Inner model: SmolVLA + video head sharing suffix_out."""

    def __init__(self, config: SmolVLAUVAConfig, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)
        self.config: SmolVLAUVAConfig
        h_in = (
            config.video_head_hidden if config.video_head_hidden > 0
            else self.vlm_with_expert.expert_hidden_size
        )
        self.video_head = VideoHead(
            h_in=h_in,
            feat_dim=config.teacher_feature_dim,
            t_future=config.t_future,
            num_patches=config.teacher_spatial_size ** 2,
        )

    def forward(
        self,
        images, img_masks, lang_tokens, lang_masks, state,
        actions, target_features,
        noise=None, time=None,
    ) -> tuple[Tensor, Tensor]:
        """Forward returning (action_losses, video_losses) — both reduction='none'.

        Args:
            target_features: (B, t_future, S, S, feat_dim) future-frame teacher features.
        """
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        suffix_out = self._compute_suffix_out(
            images, img_masks, lang_tokens, lang_masks, state, x_t, time
        )
        v_t = self.action_out_proj(suffix_out)
        action_losses = F.mse_loss(u_t, v_t, reduction="none")

        # Video head consumes the first t_future tokens of suffix_out
        z_pred = self.video_head(suffix_out[:, : self.config.t_future])
        # z_pred: (B, T, num_patches, feat_dim)
        # target: (B, T, S, S, feat_dim) → flatten S*S to match num_patches
        B, T, S1, S2, D = target_features.shape
        target_flat = target_features.reshape(B, T, S1 * S2, D)
        video_losses = F.mse_loss(z_pred, target_flat.to(z_pred.dtype), reduction="none")
        return action_losses, video_losses
```

- [ ] **Step 4: Implement minimal `SmolVLAUVAPolicy` stub so test can import**

Append to `lerobot/src/lerobot/policies/smolvla_uva/modeling_smolvla_uva.py`:

```python
class SmolVLAUVAPolicy(SmolVLAPolicy):
    """Outer policy: combines action and video losses; inference is vanilla."""

    config_class = SmolVLAUVAConfig
    name = "smolvla_uva"

    def __init__(self, config: SmolVLAUVAConfig, **kwargs):
        # Replicate SmolVLAPolicy.__init__ but instantiate VLAFlowMatchingUVA
        from lerobot.policies.pretrained import PreTrainedPolicy

        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = VLAFlowMatchingUVA(config, rtc_processor=self.rtc_processor)
        self.reset()

    def forward(self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"):
        if not self.config.enable_aux_loss:
            return super().forward(batch, noise, time, reduction)

        if self.config.teacher_features_key not in batch:
            raise KeyError(
                f"UVA training requires '{self.config.teacher_features_key}' in batch. "
                f"Bake the dataset via vbti/logic/dataset/add_video_features.py."
            )

        from lerobot.utils.constants import (
            ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE,
        )
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_id_pad")

        # delta_indices include t=0 at index 0; drop it and keep t=1..t_future
        target_features = batch[self.config.teacher_features_key]
        if target_features.shape[1] == self.config.t_future + 1:
            target_features = target_features[:, 1:]

        action_losses, video_losses = self.model.forward(
            images, img_masks, lang_tokens, lang_masks, state,
            actions, target_features, noise, time,
        )

        if actions_is_pad is not None:
            action_losses = action_losses * (~actions_is_pad).unsqueeze(-1)
        action_losses = action_losses[:, :, : self.config.max_action_dim]

        action_loss = action_losses.mean()
        video_loss = video_losses.mean()
        total_loss = action_loss + self.config.aux_weight * video_loss

        loss_dict = {
            "loss": total_loss.item(),
            "action_loss": action_loss.item(),
            "video_loss": video_loss.item(),
            "aux_weight": self.config.aux_weight,
        }
        return total_loss, loss_dict
```

- [ ] **Step 5: Update `smolvla_uva/__init__.py` to re-export**

Replace `lerobot/src/lerobot/policies/smolvla_uva/__init__.py` content with:

```python
from lerobot.policies.smolvla_uva.configuration_smolvla_uva import SmolVLAUVAConfig
from lerobot.policies.smolvla_uva.modeling_smolvla_uva import (
    SmolVLAUVAPolicy,
    VLAFlowMatchingUVA,
)
from lerobot.policies.smolvla_uva.video_head import VideoHead

__all__ = ["SmolVLAUVAConfig", "SmolVLAUVAPolicy", "VLAFlowMatchingUVA", "VideoHead"]
```

- [ ] **Step 6: Run test, verify pass**

Run: `pytest lerobot/tests/policies/smolvla_uva/test_modeling.py -v`
Expected: `test_inner_forward_returns_action_and_video_losses` PASSES.

- [ ] **Step 7: Commit**

```bash
cd /home/may33/projects/ml_portfolio/robotics/lerobot
git add src/lerobot/policies/smolvla_uva/modeling_smolvla_uva.py \
        src/lerobot/policies/smolvla_uva/__init__.py \
        tests/policies/smolvla_uva/test_modeling.py
git commit -m "vbti: add VLAFlowMatchingUVA + SmolVLAUVAPolicy

Inner model adds VideoHead, returns (action_losses, video_losses).
Outer policy combines via total = action + lambda * video, logs all three.
Hard-fails at forward if teacher_features_key missing from batch."
```

---

### Task 6: `enable_aux_loss=False` parity test

**Why this task matters:** This is the most important behavioral test in the suite. If `SmolVLAUVAPolicy(enable_aux_loss=False)` produces bit-identical losses to `SmolVLAPolicy` on the same batch, then UVA is provably opt-in — vanilla training is never accidentally broken by adopting the UVA policy class.

**Files:**
- Modify: `lerobot/tests/policies/smolvla_uva/test_modeling.py` (append test)

- [ ] **Step 1: Write the parity test**

Append to `lerobot/tests/policies/smolvla_uva/test_modeling.py`:

```python
@require_package("transformers")
@require_cuda
def test_uva_policy_with_aux_disabled_matches_vanilla():
    """SmolVLAUVAPolicy(enable_aux_loss=False) must produce bit-identical loss to SmolVLAPolicy."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla_uva.modeling_smolvla_uva import SmolVLAUVAPolicy

    # Build SmolVLAConfig with identical hyperparameters as UVA config (aux disabled)
    set_seed(42)
    vanilla_cfg = SmolVLAConfig(max_action_dim=7, chunk_size=10, num_vlm_layers=4)
    vanilla_cfg.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    vanilla_cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    vanilla_cfg.device = "cuda"

    set_seed(42)
    uva_cfg = _make_uva_config(enable_aux_loss=False)

    vanilla = SmolVLAPolicy(vanilla_cfg).to("cuda").eval()
    set_seed(42)  # reseed before UVA so identical inits
    uva = SmolVLAUVAPolicy(uva_cfg).to("cuda").eval()

    # Copy vanilla weights into UVA so they're bit-identical on shared parameters
    vanilla_state = vanilla.state_dict()
    uva_state = uva.state_dict()
    for k in vanilla_state:
        if k in uva_state and uva_state[k].shape == vanilla_state[k].shape:
            uva_state[k] = vanilla_state[k].clone()
    uva.load_state_dict(uva_state, strict=False)

    B = 2
    set_seed(0)
    batch = {
        "observation.images.wrist": torch.rand(B, 1, 3, 224, 224, device="cuda"),
        "observation.state": torch.randn(B, 1, 7, device="cuda"),
        "action": torch.randn(B, vanilla_cfg.chunk_size, 7, device="cuda"),
        "observation.language.tokens": torch.zeros(B, 4, dtype=torch.long, device="cuda"),
        "observation.language.attention_mask": torch.ones(B, 4, dtype=torch.long, device="cuda"),
    }
    noise = torch.randn(B, vanilla_cfg.chunk_size, vanilla_cfg.max_action_dim, device="cuda")
    time = torch.full((B,), 0.5, device="cuda")

    loss_v, _ = vanilla.forward(batch, noise=noise, time=time)
    loss_u, _ = uva.forward(batch, noise=noise, time=time)

    assert torch.allclose(loss_v, loss_u), \
        f"UVA(aux=False) must match vanilla: vanilla={loss_v.item()}, uva={loss_u.item()}"
```

- [ ] **Step 2: Run test, verify pass**

Run: `pytest lerobot/tests/policies/smolvla_uva/test_modeling.py::test_uva_policy_with_aux_disabled_matches_vanilla -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
cd /home/may33/projects/ml_portfolio/robotics/lerobot
git add tests/policies/smolvla_uva/test_modeling.py
git commit -m "test: UVA policy with aux disabled is bit-identical to vanilla SmolVLA"
```

---

### Task 7: Checkpoint loading tests (UVA ↔ vanilla, permissive both directions)

**Why this task matters:** Most inference of UVA-trained models will use vanilla `SmolVLAPolicy` (since the head is dead weight at inference). The loader must drop `video_head.*` keys with a logged warning, not crash. And the reverse — initializing UVA from a vanilla checkpoint — must fresh-init the head without breaking.

**Files:**
- Create: `lerobot/tests/policies/smolvla_uva/test_checkpoint.py`

- [ ] **Step 1: Write failing tests**

Write `lerobot/tests/policies/smolvla_uva/test_checkpoint.py`:

```python
"""Checkpoint loading tests: UVA ↔ vanilla SmolVLA."""
import pytest
import tempfile
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla_uva.configuration_smolvla_uva import SmolVLAUVAConfig
from lerobot.policies.smolvla_uva.modeling_smolvla_uva import SmolVLAUVAPolicy
from lerobot.utils.random_utils import set_seed
from tests.utils import require_cuda, require_package


def _common_features(cfg):
    cfg.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    cfg.device = "cuda"


@require_package("transformers")
@require_cuda
def test_load_vanilla_checkpoint_into_uva_fresh_inits_video_head(tmp_path):
    """Loading a vanilla SmolVLA checkpoint into SmolVLAUVAPolicy fresh-inits video_head."""
    set_seed(42)
    cfg_v = SmolVLAConfig(max_action_dim=7, chunk_size=10, num_vlm_layers=4)
    _common_features(cfg_v)
    vanilla = SmolVLAPolicy(cfg_v).to("cuda").eval()
    vanilla.save_pretrained(tmp_path / "vanilla_ckpt")

    cfg_u = SmolVLAUVAConfig(
        max_action_dim=7, chunk_size=10, num_vlm_layers=4,
        t_future=2, teacher_feature_dim=64, teacher_spatial_size=2, video_head_hidden=64,
        teacher_features_key="observation.video_features.siglip_output_2x2",
    )
    _common_features(cfg_u)
    cfg_u.pretrained_path = str(tmp_path / "vanilla_ckpt")
    # SmolVLAUVAPolicy.from_pretrained must succeed with strict=False under the hood
    uva = SmolVLAUVAPolicy.from_pretrained(
        cfg_u.pretrained_path, config=cfg_u, strict=False,
    ).to("cuda").eval()

    # video_head must exist and be initialized (its params were not in checkpoint)
    assert hasattr(uva.model, "video_head")
    n_vh_params = sum(p.numel() for p in uva.model.video_head.parameters())
    assert n_vh_params > 0


@require_package("transformers")
@require_cuda
def test_load_uva_checkpoint_into_vanilla_drops_video_head(tmp_path):
    """Loading a UVA checkpoint into vanilla SmolVLAPolicy drops video_head.* keys."""
    set_seed(42)
    cfg_u = SmolVLAUVAConfig(
        max_action_dim=7, chunk_size=10, num_vlm_layers=4,
        t_future=2, teacher_feature_dim=64, teacher_spatial_size=2, video_head_hidden=64,
        teacher_features_key="observation.video_features.siglip_output_2x2",
    )
    _common_features(cfg_u)
    uva = SmolVLAUVAPolicy(cfg_u).to("cuda").eval()
    uva.save_pretrained(tmp_path / "uva_ckpt")

    set_seed(42)
    cfg_v = SmolVLAConfig(max_action_dim=7, chunk_size=10, num_vlm_layers=4)
    _common_features(cfg_v)
    cfg_v.pretrained_path = str(tmp_path / "uva_ckpt")
    vanilla = SmolVLAPolicy.from_pretrained(
        cfg_v.pretrained_path, config=cfg_v, strict=False,
    ).to("cuda").eval()

    # vanilla policy must not have a video_head attribute
    assert not hasattr(vanilla.model, "video_head")
```

- [ ] **Step 2: Run tests, verify they pass with `strict=False`**

Run: `pytest lerobot/tests/policies/smolvla_uva/test_checkpoint.py -v`
Expected: Both PASS (PyTorch's `load_state_dict(strict=False)` already handles unexpected/missing keys).

If tests FAIL because `from_pretrained` doesn't accept `strict=False`, proceed to step 3.

- [ ] **Step 3: If needed, override `_load_state_dict` in `SmolVLAUVAPolicy` and add `load_state_dict` warning logger**

In `lerobot/src/lerobot/policies/smolvla_uva/modeling_smolvla_uva.py`, append to `SmolVLAUVAPolicy`:

```python
    def load_state_dict(self, state_dict, strict: bool = True):
        """Permissive load: log dropped/missing UVA-specific keys."""
        result = super().load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            import logging
            video_head_missing = [k for k in result.missing_keys if k.startswith("model.video_head.")]
            if video_head_missing:
                logging.info(
                    f"[SmolVLAUVAPolicy] Initialized {len(video_head_missing)} video_head keys "
                    f"fresh (not in checkpoint): {video_head_missing[:3]}{'...' if len(video_head_missing) > 3 else ''}"
                )
        return result
```

Re-run step 2.

- [ ] **Step 4: Commit**

```bash
cd /home/may33/projects/ml_portfolio/robotics/lerobot
git add tests/policies/smolvla_uva/test_checkpoint.py \
        src/lerobot/policies/smolvla_uva/modeling_smolvla_uva.py
git commit -m "test+feat: permissive checkpoint loading between SmolVLA and SmolVLA-UVA

Vanilla→UVA: fresh-init video_head with INFO log.
UVA→vanilla: drop video_head.* keys via strict=False."
```

---

### Task 8: vbti — target extractor registry

**Why this task matters:** The modular extractor abstraction lets us add L1 (intermediate SigLIP layer), L3 (after VLM), or new pooling strategies without touching the bake script's CLI surface. Today we only ship `siglip_output` (L2); future variants register themselves.

**Files:**
- Create: `vbti/logic/dataset/target_extractors/__init__.py`

- [ ] **Step 1: Implement registry**

Write `vbti/logic/dataset/target_extractors/__init__.py`:

```python
"""Registry of target-feature extractors for UVA aux-loss bake.

Each extractor maps (teacher_model, image_batch) -> (B, S, S, feat_dim) feature tensor.
"""
from collections.abc import Callable
from typing import Any

import torch

_EXTRACTORS: dict[str, Callable[..., torch.Tensor]] = {}


def register(name: str):
    """Decorator: register an extractor under a unique name."""
    def decorator(fn: Callable[..., torch.Tensor]):
        if name in _EXTRACTORS:
            raise ValueError(f"target extractor '{name}' already registered")
        _EXTRACTORS[name] = fn
        return fn
    return decorator


def get(name: str) -> Callable[..., torch.Tensor]:
    """Look up an extractor by name. Raises KeyError if not found."""
    if name not in _EXTRACTORS:
        raise KeyError(
            f"target extractor '{name}' not registered. "
            f"Available: {sorted(_EXTRACTORS.keys())}"
        )
    return _EXTRACTORS[name]


def list_available() -> list[str]:
    """List all registered extractor names."""
    return sorted(_EXTRACTORS.keys())
```

- [ ] **Step 2: Smoke-test the registry interactively**

Run:
```bash
cd /home/may33/projects/ml_portfolio/robotics
python -c "
from vbti.logic.dataset.target_extractors import register, get, list_available

@register('dummy')
def dummy_extractor(teacher, imgs, **kwargs):
    return imgs.mean(dim=(2,3,4), keepdim=False)

assert 'dummy' in list_available()
assert get('dummy') is dummy_extractor
print('OK')
"
```
Expected output: `OK`.

- [ ] **Step 3: Commit**

```bash
cd /home/may33/projects/ml_portfolio/robotics
git add vbti/logic/dataset/target_extractors/__init__.py
git commit -m "vbti: add target_extractors registry for UVA aux-loss bake"
```

---

### Task 9: vbti — `siglip_output` extractor (L2)

**Why this task matters:** This is the v0 default extractor. It takes the teacher SmolVLAPolicy, runs the future-frame image through `embed_image` (raw SigLIP output), reshapes to a spatial grid, and avg-pools to `(spatial_size, spatial_size, feat_dim)`.

**Files:**
- Create: `vbti/logic/dataset/target_extractors/siglip_output.py`

- [ ] **Step 1: Implement L2 extractor**

Write `vbti/logic/dataset/target_extractors/siglip_output.py`:

```python
"""L2 extractor: SigLIP final-layer patch tokens from the teacher SmolVLA model.

For each input image:
  1. Forward through teacher.vlm_with_expert.embed_image -> (B, N_patches, feat_dim)
  2. Reshape to spatial grid (B, sqrt(N), sqrt(N), feat_dim)
  3. Avg-pool to (B, spatial_size, spatial_size, feat_dim)
"""
import math

import torch
import torch.nn.functional as F

from vbti.logic.dataset.target_extractors import register


@register("siglip_output")
def siglip_output_extractor(
    teacher,  # SmolVLAPolicy
    images: torch.Tensor,  # (B, 3, H, W) preprocessed to teacher's expected range
    spatial_size: int = 4,
    **kwargs,
) -> torch.Tensor:
    """Returns (B, spatial_size, spatial_size, feat_dim) features, no_grad."""
    with torch.no_grad():
        # embed_image returns (B, N_patches, feat_dim) — raw SigLIP output (no sqrt-dim scale)
        feats = teacher.model.vlm_with_expert.embed_image(images)  # (B, N, D)

        B, N, D = feats.shape
        side = int(math.sqrt(N))
        if side * side != N:
            raise ValueError(
                f"SigLIP output has {N} patches; expected square grid. "
                f"Teacher model may have non-square patch layout."
            )

        # Reshape (B, N, D) -> (B, D, side, side) for avg_pool2d
        feats_2d = feats.transpose(1, 2).reshape(B, D, side, side)

        if spatial_size != side:
            if side % spatial_size != 0:
                raise ValueError(
                    f"spatial_size={spatial_size} must divide native patch grid side={side}"
                )
            kernel = side // spatial_size
            feats_2d = F.avg_pool2d(feats_2d, kernel_size=kernel)

        # (B, D, spatial_size, spatial_size) -> (B, spatial_size, spatial_size, D)
        return feats_2d.permute(0, 2, 3, 1).contiguous()
```

- [ ] **Step 2: Smoke-test the extractor with a tiny synthetic teacher**

Run:
```bash
cd /home/may33/projects/ml_portfolio/robotics
python -c "
import torch
from unittest.mock import MagicMock
# Lazy import so registry registers the extractor
import vbti.logic.dataset.target_extractors.siglip_output  # noqa
from vbti.logic.dataset.target_extractors import get

# Build a mock teacher that returns 256 patch tokens of dim 64
mock_teacher = MagicMock()
mock_teacher.model.vlm_with_expert.embed_image.return_value = torch.randn(2, 256, 64)

extractor = get('siglip_output')
out = extractor(mock_teacher, torch.rand(2, 3, 224, 224), spatial_size=4)
assert out.shape == (2, 4, 4, 64), f'got shape {out.shape}'
print('OK')
"
```
Expected output: `OK`.

- [ ] **Step 3: Commit**

```bash
cd /home/may33/projects/ml_portfolio/robotics
git add vbti/logic/dataset/target_extractors/siglip_output.py
git commit -m "vbti: add siglip_output (L2) target extractor for UVA bake"
```

---

### Task 10: vbti — bake script (`add_video_features.py`)

**Why this task matters:** This is the entry point that turns "v020 checkpoint + dataset" into "dataset with the new video_features column ready for UVA training". It's the only piece that touches LeRobot dataset internals — every other piece works on the resulting parquet.

**Files:**
- Create: `vbti/logic/dataset/add_video_features.py`

- [ ] **Step 1: Implement the bake script**

Write `vbti/logic/dataset/add_video_features.py`:

```python
"""Bake teacher visual features into a LeRobot dataset for UVA aux-loss training.

Adds a new column `observation.video_features.{layer}_{S}x{S}` containing fp16
tensors of shape (S, S, feat_dim) per row, where S = --spatial-size and feat_dim
comes from the teacher's SigLIP last layer.

Usage:
  python -m vbti.logic.dataset.add_video_features \
      --dataset eternalmay33/06_black_cup_red_bg_depth \
      --teacher vbti/experiments/duck_cup_smolvla/v020/lerobot_output_r12/checkpoints/150000/pretrained_model \
      --layer siglip_output \
      --spatial-size 4 \
      --target-camera observation.images.wrist
"""
import argparse
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# Import to trigger registry registration
import vbti.logic.dataset.target_extractors.siglip_output  # noqa: F401
from vbti.logic.dataset.target_extractors import get as get_extractor


def build_feature_key(layer: str, spatial_size: int) -> str:
    return f"observation.video_features.{layer}_{spatial_size}x{spatial_size}"


def main():
    p = argparse.ArgumentParser(description="Bake video features into a LeRobot dataset")
    p.add_argument("--dataset", required=True, help="Local path or HF hub repo_id")
    p.add_argument("--teacher", required=True, help="Path to teacher SmolVLAPolicy checkpoint")
    p.add_argument("--layer", default="siglip_output", help="Extractor registry name")
    p.add_argument("--spatial-size", type=int, default=4)
    p.add_argument("--target-camera", default="observation.images.wrist")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--root", default=None, help="LeRobot dataset cache root")
    p.add_argument("--force", action="store_true", help="Overwrite existing column")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("bake")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"loading teacher from {args.teacher}")
    teacher = SmolVLAPolicy.from_pretrained(args.teacher).to(device).eval()

    log.info(f"opening dataset {args.dataset}")
    dataset = LeRobotDataset(args.dataset, root=args.root)

    feature_key = build_feature_key(args.layer, args.spatial_size)
    log.info(f"feature column: {feature_key}")

    if feature_key in dataset.features and not args.force:
        raise RuntimeError(
            f"Column '{feature_key}' already exists. Pass --force to overwrite."
        )

    extractor = get_extractor(args.layer)

    # Walk the dataset row-by-row, batched
    out_chunks = []
    n_frames = len(dataset)
    log.info(f"baking {n_frames} frames at spatial_size={args.spatial_size}")

    for start in tqdm(range(0, n_frames, args.batch_size), unit="batch"):
        idxs = list(range(start, min(start + args.batch_size, n_frames)))
        imgs = []
        for idx in idxs:
            sample = dataset[idx]
            img = sample[args.target_camera]  # (3, H, W) float in [0, 1]
            if img.ndim == 4:  # (T, 3, H, W) — take t=0
                img = img[0]
            imgs.append(img)
        imgs_t = torch.stack(imgs, dim=0).to(device)
        imgs_t = imgs_t * 2.0 - 1.0  # match SmolVLA preprocessing

        feats = extractor(teacher, imgs_t, spatial_size=args.spatial_size)
        if args.dtype == "fp16":
            feats = feats.half()
        out_chunks.append(feats.cpu())

    out = torch.cat(out_chunks, dim=0)
    log.info(f"baked tensor shape={tuple(out.shape)} dtype={out.dtype}")

    # Persist into the dataset — uses LeRobot's add_column / dataset.hf_dataset rewrite
    _persist_column(dataset, feature_key, out, args.target_camera)
    log.info("done")


def _persist_column(dataset: "LeRobotDataset", key: str, tensor: torch.Tensor, ref_camera: str):
    """Write the baked tensor as a new column in the underlying HF Arrow dataset.

    Stored as nested list-of-lists; consumer (LeRobot dataloader) wraps to tensor.
    """
    import datasets as hf_ds

    hf = dataset.hf_dataset
    # Convert (N, S, S, D) tensor to list of (S, S, D) per row
    new_col = [tensor[i].tolist() for i in range(len(tensor))]
    hf_new = hf.add_column(key, new_col)
    # Persist updated dataset to disk; LeRobot reads from cache on next open
    hf_new.save_to_disk(str(Path(dataset.root) / "data" / "hf_dataset_with_uva"))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test that the script imports and parses args**

Run:
```bash
cd /home/may33/projects/ml_portfolio/robotics
python -m vbti.logic.dataset.add_video_features --help
```
Expected: argparse usage printed without import errors.

- [ ] **Step 3: Commit**

```bash
cd /home/may33/projects/ml_portfolio/robotics
git add vbti/logic/dataset/add_video_features.py
git commit -m "vbti: add bake script for UVA video_features dataset column"
```

---

### Task 11: End-to-end overfit smoke script

**Why this task matters:** This is the integration test that the whole pipeline produces a model that actually trains. It mirrors `distill_overfit_sanity.py`. After the bake step, this trains for 100 steps on a tiny subset and asserts both losses drop.

**Files:**
- Create: `scripts/smolvla_uva_overfit_sanity.py`

- [ ] **Step 1: Implement the smoke script**

Write `scripts/smolvla_uva_overfit_sanity.py`:

```python
"""End-to-end overfit smoke for SmolVLA-UVA.

Steps:
  1. Bake video features on a small subset of the target dataset.
  2. Train SmolVLAUVAPolicy for 100 steps, batch=4.
  3. Assert action_loss and video_loss both decrease monotonically over a window.
"""
import argparse
import logging
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla_uva.configuration_smolvla_uva import SmolVLAUVAConfig
from lerobot.policies.smolvla_uva.modeling_smolvla_uva import SmolVLAUVAPolicy


def run_bake(dataset_path: str, teacher: str, spatial_size: int = 4):
    cmd = [
        sys.executable, "-m", "vbti.logic.dataset.add_video_features",
        "--dataset", dataset_path,
        "--teacher", teacher,
        "--layer", "siglip_output",
        "--spatial-size", str(spatial_size),
        "--target-camera", "observation.images.wrist",
        "--batch-size", "8",
    ]
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Path to a SMALL local dataset (1-2 episodes)")
    p.add_argument("--teacher", required=True, help="Path to v020 teacher checkpoint")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--skip-bake", action="store_true", help="Skip baking (already done)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("smoke")

    if not args.skip_bake:
        log.info("--- step 1: bake ---")
        run_bake(args.dataset, args.teacher)

    log.info("--- step 2: train ---")
    cfg = SmolVLAUVAConfig()
    dataset = LeRobotDataset(args.dataset, delta_indices_override=cfg.observation_delta_indices)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    policy = SmolVLAUVAPolicy(cfg).to("cuda").train()
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-4)

    action_history, video_history = [], []
    it = iter(loader)
    for step in range(args.steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in batch.items()}

        loss, loss_dict = policy.forward(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()

        action_history.append(loss_dict["action_loss"])
        video_history.append(loss_dict["video_loss"])
        if step % 10 == 0:
            log.info(
                f"step={step} action={loss_dict['action_loss']:.4f} "
                f"video={loss_dict['video_loss']:.4f} total={loss_dict['loss']:.4f}"
            )

    # Window-mean comparison: first 20 vs last 20
    early_action = sum(action_history[:20]) / 20
    late_action = sum(action_history[-20:]) / 20
    early_video = sum(video_history[:20]) / 20
    late_video = sum(video_history[-20:]) / 20

    log.info(f"action_loss: early={early_action:.4f} late={late_action:.4f}")
    log.info(f"video_loss: early={early_video:.4f} late={late_video:.4f}")
    assert late_action < early_action, f"action_loss did not decrease: {early_action} -> {late_action}"
    assert late_video < early_video, f"video_loss did not decrease: {early_video} -> {late_video}"
    log.info("SMOKE PASS")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Dry-run the script's `--help` to confirm it imports cleanly**

Run:
```bash
cd /home/may33/projects/ml_portfolio/robotics
python scripts/smolvla_uva_overfit_sanity.py --help
```
Expected: argparse usage printed.

- [ ] **Step 3: Commit**

```bash
cd /home/may33/projects/ml_portfolio/robotics
git add scripts/smolvla_uva_overfit_sanity.py
git commit -m "vbti: add overfit smoke script for SmolVLA-UVA end-to-end"
```

---

### Task 12: Run the smoke test on a tiny subset

**Why this task matters:** This is the moment of truth — the first time the full pipeline runs against a real dataset and a real teacher checkpoint. Catches any data-loading or shape-mismatch issues that escaped the synthetic tests.

**Files:** none (execution only)

- [ ] **Step 1: Prepare a tiny local subset (1-2 episodes from `06_black_cup_red_bg_depth`)**

```bash
cd /home/may33/projects/ml_portfolio/robotics
mkdir -p /tmp/uva_smoke_dataset
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('eternalmay33/06_black_cup_red_bg_depth')
print(f'total episodes: {ds.num_episodes}, frames: {len(ds)}')
print('Use the first 2 episodes for smoke test (manual export below)')
"
```

Use the printed info to subset the dataset to ~200 frames. If the dataset is small enough already (<500 frames), use it as-is.

- [ ] **Step 2: Run the smoke**

```bash
cd /home/may33/projects/ml_portfolio/robotics
python scripts/smolvla_uva_overfit_sanity.py \
  --dataset eternalmay33/06_black_cup_red_bg_depth \
  --teacher vbti/experiments/duck_cup_smolvla/v020/lerobot_output_r12/checkpoints/150000/pretrained_model \
  --steps 100 --batch-size 4
```

Expected:
- Bake completes in ~few minutes.
- Training shows both `action_loss` and `video_loss` decreasing.
- Final line: `SMOKE PASS`.

If FAILS, diagnose: check feature column shape in batch, check teacher embed_image output dim, check delta_indices.

- [ ] **Step 3: Record results in spec doc as a "validation log"**

Append to `docs/superpowers/specs/2026-05-13-smolvla-uva-design.md` at the bottom:

```markdown

---

## 14. v0 Validation Log

| Date | Test | Result |
|---|---|---|
| YYYY-MM-DD | Tier 1 invariance | PASS |
| YYYY-MM-DD | Tier 2 enable_aux_loss=False parity | PASS |
| YYYY-MM-DD | Tier 5 bake smoke | PASS |
| YYYY-MM-DD | E2E overfit smoke | PASS: action_loss X.X→Y.Y, video_loss X.X→Y.Y |
```

Fill in actual numbers from the run.

- [ ] **Step 4: Commit validation log**

```bash
cd /home/may33/projects/ml_portfolio/robotics
git add docs/superpowers/specs/2026-05-13-smolvla-uva-design.md
git commit -m "docs: record SmolVLA-UVA v0 smoke validation results"
```

---

### Task 13: Push fork branch and verify remote sync

**Why this task matters:** All the fork-internal changes (refactor patch + new policy directory + tests) must reach the remote training machine before any real Round 1 training run. This task closes the deployment loop.

**Files:** none (deployment only)

- [ ] **Step 1: Verify fork commits are stacked correctly**

```bash
cd /home/may33/projects/ml_portfolio/robotics/lerobot
git log --oneline v0.4.4..HEAD
```

Expected: Several `vbti:` commits including:
- `vbti: refactor VLAFlowMatching to extract _compute_suffix_out helper`
- `vbti: add VideoHead module for UVA aux-loss prediction`
- `vbti: add SmolVLAUVAConfig with __post_init__ validation`
- `vbti: add VLAFlowMatchingUVA + SmolVLAUVAPolicy`
- plus test commits

- [ ] **Step 2: Push to origin (the 33may/lerobot fork)**

```bash
cd /home/may33/projects/ml_portfolio/robotics/lerobot
git push origin vbti/main
```

Expected: push succeeds.

- [ ] **Step 3: Sync the remote machine**

```bash
ssh vbti@10.11.101.240 "cd /home/vbti/anton/lerobot && git pull origin vbti/main"
```

Expected: pull succeeds; new files appear on remote.

- [ ] **Step 4: Verify policy is importable on remote**

```bash
ssh vbti@10.11.101.240 "/home/vbti/anton/env/bin/python -c \"from lerobot.policies.smolvla_uva import SmolVLAUVAConfig, SmolVLAUVAPolicy; print('OK')\""
```

Expected output: `OK`.

- [ ] **Step 5: Update memory file**

Append to `/home/may33/.claude/projects/-home-may33-projects-ml-portfolio-robotics/memory/remote_lerobot_patches.md` patches table:

```markdown
| `<new SHA>` | vbti: refactor VLAFlowMatching to extract _compute_suffix_out helper |
| `<new SHA>` | vbti: add VideoHead + smolvla_uva package |
| `<new SHA>` | vbti: add SmolVLAUVAConfig with validation |
| `<new SHA>` | vbti: add VLAFlowMatchingUVA + SmolVLAUVAPolicy |
```

Update the patch-count and sync-fingerprint:

```bash
cd /home/may33/projects/ml_portfolio/robotics/lerobot
git log --format="%s" v0.4.4..vbti/main | sort | md5sum
```

Replace the `fcdc6b9f...` fingerprint in `remote_lerobot_patches.md` with the new value.

- [ ] **Step 6: No git commit for memory file (memory is not git-tracked)**

---

## Self-Review Checklist (post-plan)

Run through each item, fix issues inline.

### Spec coverage

Cross-check each section of `docs/superpowers/specs/2026-05-13-smolvla-uva-design.md`:

- [x] Section 3 (architecture): two-piece split — Tasks 3,4,5,6,7 (fork) + Tasks 8,9,10 (vbti)
- [x] Section 4 (data flow): bake (Task 10), train (Tasks 5,6), eval — bake script enables it
- [x] Section 5 (components): VideoHead (Task 3), Config (Task 4), Inner+Outer (Task 5), bake (Tasks 8-10)
- [x] Section 6 (config): all fields covered in Task 4
- [x] Section 7.1 (modes): test in Task 6 covers `enable_aux_loss=False`; UVA-on tested in Task 5
- [x] Section 7.2 (error handling): config validation in Task 4; missing-key hard-fail in Task 5
- [x] Section 7.3 (checkpoint loading): Task 7 covers both directions
- [x] Section 7.4 (forward-pass invariants): #1 in Task 6, #2 implicit, #3 via override
- [x] Section 8 (lifecycle): Round 1 is what this plan produces; iteration is post-plan
- [x] Section 9 (out of scope): nothing in plan implements deferred items
- [x] Section 10 (testing): all 5 tiers covered (T1=Task 2, T2=Task 6, T3=Tasks 3+4, T4=Tasks 11+12, T5=Tasks 9+12)

### Placeholder scan

Searched the plan above for "TBD", "TODO", "later", "similar to" — none found.

### Type consistency

- `VideoHead` constructor signature `(h_in, feat_dim, t_future, num_patches)` — consistent across Task 3 (definition), Task 5 (used in `VLAFlowMatchingUVA.__init__`).
- `SmolVLAUVAConfig` field names — consistent: `teacher_features_key`, `teacher_feature_dim`, `teacher_spatial_size`, `t_future`, `aux_weight`, `video_head_hidden`, `enable_aux_loss`. Same names used in Tasks 4, 5, 6, 7.
- `target_features` parameter name in `VLAFlowMatchingUVA.forward` — consistent with what `SmolVLAUVAPolicy.forward` passes.
- Feature-key naming pattern `observation.video_features.{layer}_{S}x{S}` — consistent in spec, Task 4 (regex), Task 10 (`build_feature_key`).

All consistent. No issues found.

---

## Plan summary

**Total tasks:** 13
**Total commits expected:** 12-13 (one per task, except Task 11 which is execution-only)
**Estimated effort:** 4-6 hours for implementation Tasks 1-10, plus 1-2 hours for smoke + deployment Tasks 11-13.

**Critical path:**
Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6 → Task 7 → (parallel: Tasks 8,9,10) → Task 11 → Task 12 → Task 13.

**Dependencies external to the plan:**
- v020 teacher checkpoint at `vbti/experiments/duck_cup_smolvla/v020/lerobot_output_r12/checkpoints/150000/pretrained_model` (verified to exist)
- Dataset `eternalmay33/06_black_cup_red_bg_depth` reachable on HF hub
- Remote machine SSH access at `vbti@10.11.101.240` (per memory)
- Lerobot fork pushable to `origin` (`33may/lerobot`)
