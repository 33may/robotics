---
name: HF Hub duck_cup Repos
description: Public HuggingFace Hub repos for the duck_cup dual-cup pick-place project — dataset + v019/v020 SmolVLA policies
type: reference
originSessionId: 6764f38e-c482-44ba-ad27-1fce66892784
---
Public HF Hub repos under `eternalmay33` for the duck_cup project (pushed 2026-05-18, shared with a collaborator):

- **Dataset** — `eternalmay33/duck_cup_v020_all` — 765 ep / 336,940 frames, LeRobot v3.0, 20.8 GB.
- **Model (unfrozen vision)** — `eternalmay33/smolvla-duck-cup-v020` — v020 step-150000 SmolVLA policy; SigLIP fine-tuned; 93–100% SR on dual_cup.
- **Model (frozen vision)** — `eternalmay33/smolvla-duck-cup-v019` — v019 step-20000 SmolVLA policy; frozen SigLIP; 67% baseline.

Local sources: v020 = `vbti/experiments/duck_cup_smolvla/v020/lerobot_output_r12/checkpoints/150000`; v019 = `vbti/experiments/duck_cup_smolvla/v019/lerobot_output_r1/checkpoints/020000`.
