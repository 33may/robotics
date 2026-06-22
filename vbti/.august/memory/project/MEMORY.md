# Project Memory Index

## General
- [skills_location.md](skills_location.md) — Custom skills go in ~/.claude/skills/, not plugins
- [vbti_codegraph_baseline.md](vbti_codegraph_baseline.md) — VBTI Graphify/August knowledge graph baseline built 2026-06-03 for design-doc grounding
- [documentation_agents_2026_06_04.md](documentation_agents_2026_06_04.md) — Current module docs and specialist agent specs live under `vbti/documentation/`

## Feedback
- [feedback_learn_before_recommending.md](feedback_learn_before_recommending.md) — Learn project deeply before making strategy recommendations
- [feedback_main_focus_not_simulation.md](feedback_main_focus_not_simulation.md) — Lead with robot learning and evaluation, not simulation as the headline
- [feedback_notebook_dataloader.md](feedback_notebook_dataloader.md) — No DataLoader workers in notebooks; cache frames via script instead
- [feedback_sequential_sweeps.md](feedback_sequential_sweeps.md) — Run multi-run research sweeps sequentially, not batched — mid-sweep bugs/insights must steer next run
- [feedback_autonomous_waiting.md](feedback_autonomous_waiting.md) — Autonomous agent waits: always `run_in_background` for unknown-duration; avoid 300–1200s sleeps (worst cache zone); read tail -1/3, not -15
- [feedback_show_plots_inline.md](feedback_show_plots_inline.md) — After generating any plot/image, show it inline (Read tool) AND cite absolute file path
- [feedback_no_destructive_inplace.md](feedback_no_destructive_inplace.md) — Default to new artifact for data transforms; never rename-over-source unless user explicitly asks for in-place
- [feedback_one_liner_status.md](feedback_one_liner_status.md) — Before every non-trivial tool call, prefix with one-line "what + why". Don't drop it mid-session.
- [feedback_ultrawide_layouts.md](feedback_ultrawide_layouts.md) — Ultrawide monitor: prefer wide grid layouts (more cols, fewer rows). Target aspect ≈3.5
- [feedback_no_delete_without_guidance.md](feedback_no_delete_without_guidance.md) — Cleanup: survey + report only; never `rm`/purge/empty without user explicitly approving each action
- [feedback_remote_password.md](feedback_remote_password.md) — SSH password is `vbti25robot` (in remote_machine.md), not `vbti` — check memory before guessing
- [codex_subscription_belongs_in_pi.md](codex_subscription_belongs_in_pi.md) — Pi, not Claude Code, is the sanctioned Codex-subscription harness
- [feedback_codex_proxy_claude_code.md](feedback_codex_proxy_claude_code.md) — Claude Code TUI stays, Codex runs behind a proxy

## Hardware
- [gpu_upgrade.md](gpu_upgrade.md) — RTX 4070 Ti SUPER 16GB confirmed 2026-04-21 (prior 5090 upgrade claim was wrong)
- [servo_leader_follower_voltage.md](servo_leader_follower_voltage.md) — leader arm ≈5V, follower ≈12V in scan_all; tells the ttyACM ports apart
- [camera_viewer_preset_fix.md](camera_viewer_preset_fix.md) — `logic/cameras/view_cameras.py` should use `cameras.py` presets/utilities, not raw RealSense discovery

## Project state
- [project_smolvla_step_peak.md](project_smolvla_step_peak.md) — SmolVLA fine-tuning on duck_cup peaks at 20–25k steps with cosine 50k schedule; default 35k steps for new runs
- [project_remote_training_path.md](project_remote_training_path.md) — remote.py ships lerobot-train; SmolVLABackend dead code on remote — custom dataloader features must bake into dataset OR patch lerobot itself
- [project_smolvla_vram_anchor.md](project_smolvla_vram_anchor.md) — VRAM anchor: SmolVLA BS=32 + 4 cams ≈ 15.6 GB on 5090; +2 GB per cam, linear-ish in BS
- [dataset_may_sim_suffix.md](dataset_may_sim_suffix.md) — `_may-sim` suffix = recalibrated to current calibration (NOT sim-rendered); originals without it are on old calibration
- [v018_v019_training_analysis.md](v018_v019_training_analysis.md) — v019 underfit (7.83 epochs vs v018's 11.93); size schedules by epoch count (~12), not step count
- [duck_cup_sota_plan.md](duck_cup_sota_plan.md) — Roadmap to SOTA on dual-cup pick-duck: eval v019 → unfreeze vision → collect data → GR00T last
- [phase1_data_efficiency_plan.md](phase1_data_efficiency_plan.md) — Phase 1 (2026-05-11→18): 4-slice sweep v021–v024 (6.25/12.5/25/50%) → heatmap SR(epoch, dataset_size) on dual_cup_60
- [project_smolvla_uva.md](project_smolvla_uva.md) — SmolVLA-UVA aux video-prediction loss: v0 built & validated 2026-05-18 in the fork, not yet trained for real
- [project_robotics_reframe_hbo_to_research.md](project_robotics_reframe_hbo_to_research.md) — Project is now framed as a robotics research arc, not just a company task
- [project_state_2026_05_26.md](project_state_2026_05_26.md) — Current graduation-table story: pipeline → VLA baseline → 100% lab task → harder evaluation/UVA next
- [project_plan_2026_05_26_update.md](project_plan_2026_05_26_update.md) — 26-05 project plan rewrite: robot-learning workflow, VLA baseline, validation, sim-real extensions
- [remote_lerobot_patches.md](remote_lerobot_patches.md) — 29 vbti patches now live as commits on `33may/lerobot` `vbti/main` (v0.4.4 base); both machines run editable installs of the fork
- [vbti_user_account.md](vbti_user_account.md) — second Linux user `vbti` (uid 1001) for employee data collection; standard user, shared conda via ACL on /home/may33
- [reference_hf_duck_cup_repos.md](reference_hf_duck_cup_repos.md) — public HF Hub repos: duck_cup_v020_all dataset + smolvla-duck-cup-v019/v020 model repos under eternalmay33
- [hf_cli_base_env.md](hf_cli_base_env.md) — do HF Hub work via the conda envs' own huggingface_hub (~0.35); don't upgrade base to 1.x — breaks transformers

## Pipeline
- [detection_pipeline.md](detection_pipeline.md) — OWLv2 detection + phase detection + augmentation pipeline (2026-04-16)
- [inference_state_aug.md](inference_state_aug.md) — v014+ uses 22-d augmented observation.state; --detection=true does both aug + overlay; RGB pitfall fixed

## Topic Files
- `remote_machine.md` — SSH access to remote robot machine (10.11.101.240 as of 2026-05-11, DHCP-assigned, vbti)
- [remote_host_fingerprint.md](remote_host_fingerprint.md) — Canonical ED25519 host-key `pPDbgPe7…` to find the box across DHCP IP changes
- `pipeline_architecture.md` — robot_utils.py pipeline, Cosmos Transfer, HDF5 structure, replay determinism
- `depth_estimation_3d_reconstruction.md` — DA3, VGGT, MASt3R, MapAnything; COLMAP alternatives for GS
- `lerobot_dataset_format.md` — LeRobot v2.1/v3.0 format spec, parquet schema, create API, HDF5 conversion, sim-to-real
- `training_experiments.md` — SmolVLA/GR00T training: LR behavior, loss vs policy quality, scheduler auto-scaling, env setup
- `camera_udev_setup.md` — RealSense D405 udev symlinks (/dev/cam_*), UYVY vs YUYV streams, LeRobot opencv fixes
- `remote_pretrained_auto_sync.md` — remote.py auto-syncs local pretrained paths to remote before training
- `lerobot_env_rebuild.md` — torchcodec ABI fix: use torchcodec 0.5 with PyTorch 2.7+cu128
- `dataset_resolve_function.md` — Shared resolve_dataset_path in dataset/__init__.py, vbti.pth makes package importable

---

## SmolVLA Inference Fix (2026-02-05)

**Root cause**: Missing postprocessor + unit mismatch (degrees vs radians).
**Fix**: Load postprocessor, denormalize outputs, convert degrees→radians.

### Data Flow
```
Collection: Isaac (radians) → convert → dataset (degrees)
Training:   dataset (degrees) → normalize → model (normalized)
Inference:  model output → postprocessor (degrees) → radians → env.step()
```

---

## NuRec + TiledCamera INCOMPATIBLE (2026-02-13)

TiledCamera hangs with GS splat (NuRec USDZ). NuRec not rendered in synthetic data passes.
**Decision**: Convert GS to mesh (MILo). Viewport works fine, only synthetic data broken.

---

## leisaac Package Fix

Missing `__init__.py` in `assets/{robots,scenes,__init__}.py`.
Added `"template"` to blacklist in `tasks/__init__.py`.
Set `LEISAAC_ASSETS_ROOT` env var.

---

## MILo Installation (2026-02-13)

**Location**: `vbti/libs/MILo/`, **Env**: `gsplat-pt25` (PyTorch 2.5.1+cu124, Python 3.11)

### Compilation Fixes (Fedora 42)
1. glibc 2.41 cospi/sinpi conflict → patch CUDA math_functions.h
2. GCC 15 unsupported → use `CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14`
3. Missing `#include <cstdint>` in 9 headers across 3 rasterizer submodules
4. Missing `#include <cfloat>` in simple_knn.cu
5. tetra_triangulation cmake needs CUDA include path flag

---

## Cosmos Transfer 2.5 (2026-02-20)

Full reference in `pipeline_architecture.md`. Key points:
- Model: 2.36B params, HF: `nvidia/Cosmos-Transfer2.5-2B`, BF16 only
- Python 3.10, CUDA 12.8, PyTorch 2.9, uv package manager
- 720p needs 65GB VRAM (A100/H100), 480p ~24GB
- Control types: depth, edge, seg, vis

---

## Conda Environments

| Env | Python | Purpose |
|-----|--------|---------|
| lerobot | 3.12 | LeRobot v0.4.4 (editable), PyTorch 2.7.0+cu128, torchcodec 0.5, SmolVLA training/inference |
| groot | 3.10 | GR00T N1.6 training (torch 2.7.1, flash-attn, gr00t) — renamed from rbts |
| isaac | 3.11 | Isaac Sim |
| gsplat-pt25 | 3.11 | MILo / Gaussian Splatting |

---

## Auth & Accounts
- **HuggingFace**: username `eternalmay33`, token at `~/.cache/huggingface/token`
- **GitHub**: `33may` (verified via `gh auth status`; local username is `may33`, not the same)

## User Environment
- Shell: zsh, GPU: RTX 4070 Ti SUPER (16GB, SM 89) — confirmed 2026-04-21 (earlier 5090 claim was wrong)
- GCC 14 at `/usr/bin/gcc-14`, CUDA 12.9 at `/usr/local/cuda-12.9` (patched)
- flash-attn cross-device fix: set `TMPDIR=/home/may33/tmp/build` (same filesystem as pip cache)
