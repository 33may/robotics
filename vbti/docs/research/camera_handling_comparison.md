# Camera Handling: SmolVLA vs GR00T N1.6

## Camera Key Naming

**SmolVLA**: `observation.images.<name>` — direct 1:1 from dataset features. Names are arbitrary (front, wrist, gripper, top).

**GR00T**: Two-layer mapping. `modality.json` maps short names to full keys:
```json
"video": {
    "front": {"original_key": "observation.images.front"},
    "wrist": {"original_key": "observation.images.wrist"}
}
```
Then `ModalityConfig.modality_keys` references short names: `["front", "wrist"]`

## Camera Ordering — YES, IT MATTERS FOR BOTH

**SmolVLA**: `prepare_images()` iterates `config.image_features` in dict insertion order. Each image becomes a separate SigLIP embedding block:
```
[img_start] [img_0_embeds] [img_end] [img_start] [img_1_embeds] [img_end] ... [lang] [state]
```

**GR00T**: Views stacked in `modality_keys` list order → flattened into Eagle chat conversation as sequential images. NVIDIA convention: wrist = `wrist`, third-person = `front`.

## Image Resolution & Normalization

| | SmolVLA | GR00T N1.6 |
|---|---|---|
| Resize | 512x512 aspect-preserving + left/top pad | shortest_edge=256, 95% center crop |
| Pad value | 0 (becomes -1 after normalization) | N/A |
| Normalization | `[0,1]` → `[-1,1]` manual | uint8 → Eagle processor handles internally |
| VLM backbone | SmolVLM2-500M (SigLIP) | Eagle-Block2A-2B-v2 (SigLIP-2) |

## Adding/Removing Cameras

**SmolVLA**: Add `FeatureType.VISUAL` entry to `input_features`. Set `empty_cameras` config param for absent views — creates placeholder entries with `-1` fill + zero mask.

**GR00T**: Add to `modality.json` video section + `modality_keys` list. New embodiments use `"new_embodiment"` tag (projector index 10). **No built-in missing camera fallback** — processor asserts all view keys present.

## Missing Camera Handling

**SmolVLA** — graceful:
- Training: keys should be present, missing silently skipped but ≥1 required
- Inference: missing → `-1` filled tensors + zero attention mask

**GR00T** — strict:
- Training: `assert view in images` — missing = crash
- Inference: same assertion, no fallback

## Temporal Frames

**SmolVLA**: Takes **last frame only** (`batch[key][:, -1]`). `n_obs_steps=1`.

**GR00T**: Configurable via `delta_indices` (e.g., `[-2, -1, 0]` for 3 temporal frames). More flexible.

## Our Camera Setup

From `convert_utils.py`:
- HDF5 obs keys: `cam_top`, `cam_left`, etc. (auto-discovered)
- Mapped via `camera_map`: `{"cam_top": "top", "cam_left": "left"}`
- Resolution: 480x640 uint8 RGB
- Inference uses 3 cameras: `front`, `third_person`, `gripper`

## Summary Table

| Aspect | SmolVLA | GR00T N1.6 |
|---|---|---|
| Key format | `observation.images.<name>` | Short name + `modality.json` |
| Order matters? | Yes (dict insertion) | Yes (list order) |
| Image size | 512x512 padded | shortest_edge=256 cropped |
| Max cameras | No hard limit | No hard limit |
| Missing camera | Graceful (pad + mask) | Assert crash |
| Temporal | Last frame only | Multi-frame via delta_indices |
| Embodiment | Single | Multi (32 projector slots) |

## Design Implications for Unified Backend

1. **Camera registry with explicit ordering** — both models are order-sensitive. Config must define camera order as a list, not rely on dict iteration.

2. **Camera name mapping** — our existing `camera_map` from convert_utils is the right abstraction. For GR00T it also needs to generate `modality.json`.

3. **Resolution is model-specific** — SmolVLA=512x512 padded, GR00T=256-shortest-edge cropped. Transforms MUST be per-backend, never shared.

4. **Missing cameras** — SmolVLA handles it; for GR00T we'd need to wrap the processor to inject black frames or always provide all configured cameras.

5. **Temporal** — GR00T supports multi-frame natively; SmolVLA only uses last frame. If we want temporal for SmolVLA, it needs frame stacking at dataset level.
