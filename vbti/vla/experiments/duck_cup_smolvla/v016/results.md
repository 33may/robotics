# v016 — Accepted Results Log

*Append-only. Each entry: question, method, key finding, evidence pointer, implication.*

---

## exp01 — SmolVLA multicam mechanics — 2026-04-29

**Question:** How does SmolVLA handle multi-camera input, can the codebase tolerate per-sample-variable image keys, and what is the code surface for each Q2 path (extra-camera / side-branch / channel-concat)?

**Method:** Pure code investigation of the editable `lerobot` install at `lerobot/src/lerobot/`. Read `policies/smolvla/modeling_smolvla.py`, `smolvlm_with_expert.py`, `configuration_smolvla.py`, plus dataset-side `lerobot_dataset.py`, `aggregate.py`, `dataset_tools.py`, `utils.py`, `factory.py`.

**Key findings:**

1. **Each camera = one independent ViT forward pass.** Tokens are concatenated along the sequence dim and the action expert cross-attends over the joint prefix. Adding a depth camera = adding a SigLIP forward pass, doubling vision tokens for that view.

2. **Image-key set is frozen at policy init time** (read from `ds_meta.features` once). Per-sample variable image keys are unsupported because `att_masks` is built once and expanded across the batch.

3. **The `<key>_padding_mask` hook is the legitimate way to mask out depth on RGB-only episodes** — but only works if the depth tensor is *present* (e.g. zeros). Missing keys are silently dropped because `empty_cameras=0` by default.

4. **`MultiLeRobotDataset` is hard-disabled.** Co-training requires pre-aggregation, which strict-checks feature equality. Mixed RGB/RGB-D datasets must be unified first via `tools.modify_features` (backfill a dummy depth feature on the RGB-only side).

5. **Image features must be 3-D `(C,H,W)`** with `dtype in {image,video}`. Single-channel depth fails parsing. Must save as 3-channel (turbo-colorized uint8 OR 3× repeat of grayscale uint16-mapped-to-uint8).

**Evidence:** `results/exp01_smolvla_multicam_mechanics/findings.md` and `code_excerpts.md` (verbatim quotes with file:line refs).

**Implications for v016:**
- **Q2-B1 (extra camera) is trivial code-wise** and is the recommended starting point. No SmolVLA source edits required.
- **Q2-B3 (channel concat) is dead** — defrosts the patch conv, breaks `freeze_vision_encoder=True`, no pretraining signal on the 4th channel.
- **Q2-B2 (side-branch) is moderate effort** (~150-300 LoC in one file plus a new depth-encoder module) — defer to v017 if B1 plateaus.
- **Q1 path A (backfill old data) is the only co-training-compatible option** given the merge-time strict feature equality check. Path B (estimate-everywhere) is logically equivalent for old data, just different on new data. Path C (dual-branch) on its own doesn't escape the dataset-merge requirement.

---
