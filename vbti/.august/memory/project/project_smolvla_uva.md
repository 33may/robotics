---
name: smolvla-uva-auxiliary-video-prediction-loss-v0-built-validated
description: "UVA-style future-feature-prediction aux loss on SmolVLA for sample-efficiency; v0 implemented in the fork as of 2026-05-18, not yet trained for real"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5831fc1f-f3a8-4242-8572-532eea191b91
---

UVA = Unified Video Action Model (arXiv 2503.00200). We added a UVA-style
**auxiliary future-feature-prediction loss** to SmolVLA to improve sample
efficiency on data-scarce fine-tuning. Shared backbone, two decoupled heads
(action + video); video head is dropped at inference, so a UVA-trained
checkpoint inferences as plain SmolVLA. `loss = action_loss + 0.3·video_loss`
(constant λ — literature review showed no robust gain from λ scheduling).

**Status (2026-05-18): v0 built, validated, and the v025–v029 UVA sweep is LAUNCHED.**
- Lives in fork `33may/lerobot` `vbti/main` — package `policies/smolvla_uva/`,
  policy type `smolvla_uva`. 20 commits, see [[remote_lerobot_patches]].
- v0 validation: 18 unit tests + end-to-end overfit smoke pass
  (action_loss 1.31→0.50, video_loss 25.3→17.2 over 100 steps).
- Design doc: `docs/superpowers/specs/2026-05-13-smolvla-uva-design.md`
  (Section 14 = validation log). Plan: `docs/superpowers/plans/2026-05-13-smolvla-uva-implementation.md`.

**Remote training integration (how `lerobot-train` runs UVA):**
- Fork patch `policies/__init__.py` imports `SmolVLAUVAConfig` so
  `@register_subclass("smolvla_uva")` fires — `get_policy_class`'s plugin
  fallback then resolves the policy by naming convention.
- `--policy.path` must point at a checkpoint whose `config.json` says
  `type: smolvla_uva`. We built `/home/vbti/anton/data/smolvla_uva_base`
  (remote) = `lerobot/smolvla_base` weights + patched config; `from_pretrained`
  loads it non-strict so `video_head.*` stays fresh-init (warm-start).
- vbti engine: `ModelType.SMOLVLA_UVA` added (`config_utils.py` + `engine.py`),
  reuses `SmolVLAModelConfig` + `SmolVLABackend`.
- `empty_camera_N` slots ARE real model inputs when the dataset provides them
  (`prepare_images` zero-pads only *missing* keys) — so gripper (cam #4 of 5 in
  duck_cup_v020) is a genuine input and a valid UVA target camera.

**v025–v029 sweep (launched 2026-05-18):** UVA counterpart of v021–v024 +
a full-data point. Strides 16/8/4/2/1 → 1/16, 1/8, 1/4, 1/2, full of
`duck_cup_v020_all`. Baked dataset = `duck_cup_v020_all_uva` (remote-only,
~41 GB fp16, gripper cam). Configs carry explicit `steps`
(21716/42197/84258/167941/336940 — W&B-verified vs v021–v024/v020) not `epochs`.
Chain dispatched via `chain_remote.py --wait_for_session uva_bake`: a remote
tmux waits for the bake, then runs all 5 sequentially. `chain.py` /
`chain_remote.py` patched for remote-only datasets (lazy frame-count, skip
push_data, `os.path.isdir` for un-stat-able remote paths).

**Key design choices:**
- Teacher = v020's frozen SigLIP encoder (v020 trained on a superset of the
  target data, so its features capture the real task). Target = `siglip_output`
  L2 features, 4×4 spatial grid, `t_future=4`, 960-dim (SmolVLM2-500M
  `embed_image` connector output, NOT raw 1152-d SigLIP).
- Targets are **baked into a new dataset copy** by
  `vbti/logic/dataset/add_video_features.py` — the full `(t_future,S,S,D)`
  future window per row (Fix B), because LeRobot's `observation_delta_indices`
  applies globally to all `observation.*` keys and would corrupt image/state.

**Next step:** when the sweep finishes, evaluate v025–v029 vs the vanilla
v021–v024 / v020 — heatmap SR(epoch, dataset_size), UVA row vs vanilla row, to
measure the sample-efficiency gain ([[phase1_data_efficiency_plan]]).
Pull checkpoints: `python -m vbti.logic.train.remote pull --checkpoint=all
--run_name=lerobot_output_r1 --version=v0XX`.

**Caveat:** the `lerobot-train` + UVA CLI path had no full end-to-end run before
launch (only the in-process smoke test). v025 is the de-facto pre-flight — if
the CLI integration is broken it fails fast at policy-build / dataset-load.
