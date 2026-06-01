# v016 — Adding Depth Information

## Status
**Research / planning phase.** Goal is to add depth from D405 cameras to the SmolVLA observation pipeline. No training run yet — design space is open.

## Hypothesis
Explicit depth cues from D405 stereo will lift grasping precision and contact-rich behavior over the v013-style RGB+state baseline. Depth gives the encoder direct 3D geometry it currently has to infer from parallax alone.

## Hard constraint
**Existing datasets (v013/v014/v015 — `01_02_03_merged_may-sim_detection*` and friends) must remain usable in training.** They are RGB-only. Whatever path we take, it has to answer "what is the depth tensor for an old episode?"

---

## Hardware findings (verified 2026-04-29)
- All 4 cameras are Intel RealSense **D405**. Verified via `rs-enumerate-devices` (top/left/right) and `udevadm info /dev/cam_gripper` → `ID_MODEL=Intel(R) RealSense(TM) Depth Camera 405`, serial `130523070141`.
- D405 spec: stereo-IR depth, ideal range ~7–50cm, sub-mm precision at close range — the right sensor for gripper-mounted manipulation.
- `vbti/logic/cameras/cameras.py` currently only enables `rs.stream.color`. The gripper entry in `CAMERA_PRESETS["realsense"]` even routes through OpenCV (`/dev/cam_gripper`) instead of the RealSense pipeline — so the gripper depth path doesn't exist at all yet.
- To enable depth: add `rs.stream.depth` (Z16) per camera, align with `rs.align(rs.stream.color)` so depth and color share extrinsics, and persist into the dataset.
- Gripper RealSense serial (currently absent from preset): `128422270260` (V4L2 serial `130523070141`, see `camera_udev_setup.md`).

---

## Design space (open)

### Q1 — How to handle existing RGB-only datasets?

| Option | Mechanism | Tradeoff |
|---|---|---|
| **A. Backfill estimated depth** | Run Depth Anything v2 / DA3 over old episodes, save as new image streams. New data uses real D405 depth. | Distribution shift between estimated (relative, fuzzy) and real (metric, sharp). |
| **B. Estimated depth everywhere** | Use Depth Anything on both old and new data, ignore D405 depth at training time. | Uniform input, but throws away the metric signal we paid for. |
| **C. Depth Helps-style dual path** | RGB encoder runs on everything; depth branch active only when present. | More architecture work; cleanest principled answer. |

### Q2 — How to feed depth into SmolVLA?

| Option | Mechanism | Tradeoff |
|---|---|---|
| **Extra camera (simplest)** | Add `observation.images.<cam>_depth` keys. SmolVLM2 encoder treats them as additional images. | No architecture change. But encoder is RGB-pretrained, depth is OOD — colorize (turbo) to bridge. |
| **Side-branch (Depth Helps)** | Separate depth encoder + cross-attention into RGB tokens. | Better feature extraction, more engineering. |
| **Early fusion (4th channel)** | Stack depth as alpha-channel onto RGB. | Breaks pretrained encoder conv weights — likely worst option. |

### Q3 — Depth representation choices
- **Colorize vs grayscale**: turbo/viridis colormap makes depth look more like a natural image to the RGB-pretrained encoder. Grayscale preserves "single scalar per pixel" semantics but is far OOD.
- **Fixed clip range vs per-frame normalization**: fixed range (e.g. 0.07–0.5m for D405) preserves metric meaning across frames. Per-frame min-max destroys absolute distance but maximizes contrast.
- **Bit depth on disk**: uint16 PNG (lossless metric) vs uint8 colorized (smaller, viewer-friendly, lossy). Probably uint16 in the dataset, colorize on the fly during training.

---

## Research references
- [Depth Helps (IROS 2024)](https://arxiv.org/html/2408.05107v1) — depth injection on top of pretrained RGB policy with minimal RGB-D data.
- [UniLACT](https://arxiv.org/html/2602.20231) — depth-aware RGB latent action learning for VLAs.
- [SmolVLA paper](https://arxiv.org/abs/2506.01844) — base model, RGB-only by default. Confirms encoder treats per-camera images independently.
- [VLM2VLA](https://vlm2vla.github.io/) — LoRA-based fine-tuning without catastrophic forgetting.
- [CORAL](https://arxiv.org/html/2603.09298) — LoRA experts per task on a frozen VLA backbone.

---

## Open questions for next sessions
1. **Real-time capture pipeline**: FPS hit when enabling depth on 4 D405s simultaneously, USB bandwidth limits, depth-color alignment quality, holes/dropouts near object edges.
2. **Backfilling old data**: compute cost of Depth Anything across v013/v014/v015 frame counts. Does estimated depth align (in scale/orientation) with what D405 reports, or do we need a calibration step?
3. **SmolVLA handling of missing modality**: does the codebase tolerate per-sample-variable image keys, or must every sample have the same set?
4. **Token budget impact**: 4 RGB + 4 depth doubles per-step vision tokens. How much does this slow training and inference (real-robot latency budget is ~50ms/step today)?
5. **Whether v013 already implicitly learned depth** from stereo parallax across the 4 cameras — if yes, explicit depth may give diminishing returns.
6. **Co-training schedule**: combine all datasets uniformly, or warm up on RGB-only then introduce depth episodes? Affects whether old data still steers the policy.

---

## Decisions to make (in order)
1. Q1 path — backfill vs estimated-everywhere vs dual-branch.
2. Q2 path — extra-camera vs side-branch.
3. Pilot a small depth-capture dataset to verify hardware pipeline + storage.
4. Train a v016_pilot on a subset to confirm the encoder picks up signal from depth.
