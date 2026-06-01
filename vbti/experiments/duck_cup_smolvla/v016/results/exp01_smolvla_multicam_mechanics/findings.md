# exp01 — SmolVLA multicam mechanics

Architecture investigation (Read-only). Source files referenced under `lerobot/src/lerobot/`.

## Q-A: Multi-camera mechanics

**Where image becomes tokens.** `policies/smolvla/modeling_smolvla.py:653` calls `self.vlm_with_expert.embed_image(img)` inside `embed_prefix`. `embed_image` (`policies/smolvla/smolvlm_with_expert.py:179-192`) runs `vision_model(pixel_values=...)` (SigLIP) then `connector` (modality projection). **One forward pass per image tensor.**

**Joint vs independent encoding.** Independent. `modeling_smolvla.py:634-664` loops `for img, img_mask in zip(images, img_masks)` — each camera gets its own `embed_image` call. There is **no batched ViT pass over all cameras**.

**Token combination.** Concatenated along the sequence dim, then attended. `modeling_smolvla.py:704` does `embs = torch.cat(embs, dim=1)` after appending image, language, and state tokens. The action expert reads VLM K/V via cross-attention over the concatenated prefix (`smolvlm_with_expert.py:403-455`, `self_attn_every_n_layers=2` mixing).

**Image-key set frozen at policy init?** Yes. `policies/factory.py:458-472` pulls `features` from `ds_meta.features`, then `cfg.input_features = {...}` is set once. `configs/policies.py:148-152` exposes `image_features` as the visual subset of `input_features`. `modeling_smolvla.py:409-410` iterates `self.config.image_features`, not the batch keys.

**Missing key at inference — silent partial fallback.**
- `modeling_smolvla.py:412-415` — if **all** keys missing → `ValueError`.
- `modeling_smolvla.py:436-442` — missing keys filled with `-1` images and mask=0, **but only up to `self.config.empty_cameras`**. Beyond that count, silently dropped (no token at all).
- `configuration_smolvla.py:53` — `empty_cameras: int = 0` default.
- `modeling_smolvla.py:427-430` — if `batch[f"{key}_padding_mask"]` is provided, it is used as the per-sample image mask. Mask flows via `pad_masks.append(img_mask)` (`:663-664`) into `make_att_2d_masks` (`:783, :102-132`), zeroing attention to those tokens. **This is the natural hook for "depth missing for this sample"** — but only consulted for keys that ARE present, so a zero-tensor must still be supplied.

## Q-B: Code surface for each Q2 path

### B1. Extra-camera (depth as additional `observation.images.<cam>_depth`)
- **Architecture changes: zero.** `embed_prefix` is generic over the present-key list.
- **Edits:**
  1. Capture: enable `rs.stream.depth` + `rs.align(rs.stream.color)` in `vbti/logic/cameras/cameras.py` (~10 lines).
  2. Dataset: write depth as `dtype="image"`, `shape=(3,H,W)` — `datasets/utils.py:719-720` rejects non-3D image features (depth must be colorized to 3 channels OR repeat-broadcast).
  3. Inference: include `observation.images.gripper_depth` in `vbti/logic/inference/run_real_inference.py::_build_observation`.
- **Token cost:** doubles vision tokens for any camera given depth.
- **Difficulty: trivial.** No SmolVLA source modifications.
- **Caveat:** SigLIP is RGB-pretrained. Raw uint16/grayscale is far OOD. Colorize (turbo) so the encoder sees something in-distribution.

### B2. Side-branch (separate depth encoder, fuse via cross-attention)
- Concat lives in `VLAFlowMatching.embed_prefix` (`modeling_smolvla.py:625-717`). `VLAFlowMatching` must be subclassed/forked, not just `SmolVLAPolicy`.
- **Sketch:**
  1. In `__init__` (`:555-600`) add a depth encoder (small CNN, or a second SigLIP) + projection to `expert_hidden_size`.
  2. In `prepare_images` (`:403-443`) split RGB vs depth keys (suffix-based) into two lists.
  3. In `embed_prefix` (`:625-717`) embed depth via the new encoder, append to `embs` with their own `pad_masks`.
  4. Optionally add a learned modality-type embedding.
- **Files touched:** 1 + a new `depth_encoder.py`. ~150-300 LoC.
- **Difficulty: moderate.**

### B3. Channel-concat (4th channel on RGB)
- SigLIP patch embedding is `vision_model.embeddings.patch_embedding` — Conv2d(3, hidden=768, kernel=16, stride=16) (HuggingFaceTB/SmolVLM2-500M-Video-Instruct).
- **Sketch:** replace patch_embedding with 4-channel Conv2d, copy pretrained 3-ch weights into ch 0-2, zero-init ch 3. Stack RGB+depth → `(B,4,H,W)` in `prepare_images`. `resize_with_pad` is channel-agnostic. Normalization at `:423` (`img * 2.0 - 1.0`) assumes [0,1].
- **Risk:** patch embedding is part of the frozen vision encoder by default (`configuration_smolvla.py:73 freeze_vision_encoder=True`). Either unfreeze patch_embedding only or unfreeze the ViT — both break SmolVLA's "frozen VLM, train expert only" paradigm.
- **Difficulty: heavy.** High-impact, low-payoff.

## Q-C: Mixed RGB-only / RGB-D co-training

**Single-dataset loader.** Feature-uniform per dataset. `datasets/lerobot_dataset.py:246-258` derives image/video/camera keys from `meta.features`. All episodes in one dataset share the schema; `__getitem__` (`:1082-1118`) iterates `meta.video_keys`. No per-episode override.

**Merging two datasets with different image keys — hard error.**
- `MultiLeRobotDataset` is **disabled**: `datasets/factory.py:115` raises `NotImplementedError`.
- Supported merge path: `tools.merge_datasets` → `aggregate_datasets` → `validate_all_metadata`. At `datasets/aggregate.py:75-78`:
  ```python
  if features != meta.features:
      raise ValueError(f"Same features is expected, but got features={meta.features} ...")
  ```
- **Workaround**: `tools.modify_features` (`datasets/dataset_tools.py:275+`) to add a dummy depth feature to RGB-only datasets first (zeros, or DA3-estimated), then merge.

**Variable image-keys per sample at runtime — effectively unsupported.** `modeling_smolvla.py:706,715` builds `att_masks` as a single `(1, seq_len)` then expands across the batch. Variable per-sample sequence length would break this.

**Cleanest "dual-path with optional depth" path.** Use the `<key>_padding_mask` hook:
1. Backfill a dummy depth tensor (zeros, or DA3 estimate) into RGB-only datasets via `tools.modify_features` — features become uniform, merge passes.
2. Add a per-frame boolean; thin collate translates to `<key>_padding_mask` keyed by the depth image-key.
3. `prepare_images` reads the mask, flows into `pad_masks`, zeroes attention to depth tokens for those samples.
4. Token count constant across batch; only attention is masked.

## Recommendation

**Q2-B1 (extra-camera) is the most viable for v016.** Zero edits to SmolVLA source. Combined with **Q1-A backfill** (DA3 depth into v013/v014/v015) and the `<key>_padding_mask` trick (mask=False on backfilled episodes), we get clean co-training.

**Avoid Q2-B3.** Defrosts the SigLIP patch conv, breaks `freeze_vision_encoder=True`, and the 4th channel has no pretraining signal.

**Q2-B2 is overkill for v016 pilot** — keep as v017 escalation if B1 plateaus.

## Surprises

1. **`empty_cameras=0` by default** ⇒ missing image keys are silently dropped. The padding-mask hook is only checked **inside** the `for key in present_img_keys` loop. To mask depth on RGB-only episodes we MUST emit a (zero) depth tensor, not omit the key.
2. **`MultiLeRobotDataset` is hard-disabled.** Co-training must go through pre-aggregation, which strict-checks feature equality. Backfilling via `tools.modify_features` is the only sanctioned merge path.
3. **Image dtype must be 3-D.** 1-channel depth fails policy-feature parsing. Save as 3-channel on disk.
4. **Each camera = independent ViT pass.** 4 depth cameras = 4 extra ViT forwards per step; doubles token budget. Consider gripper-only depth first.
