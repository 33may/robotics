# Documentation And Agents Update — 2026-06-04

Created current documentation source of truth under `vbti/documentation/`.

Key files:

- `documentation/README.md`
- `documentation/SYSTEM_TEXTBOOK.md`
- `documentation/logic/README.md`
- `documentation/logic/cameras/`
- `documentation/logic/servos/`
- `documentation/logic/dataset/`
- `documentation/logic/depth/`
- `documentation/logic/detection/`
- `documentation/logic/train/`
- `documentation/logic/inference/`
- `documentation/logic/reconstruct/`
- `documentation/runbooks/`
- `documentation/modules/dataset.md`
- `documentation/modules/hardware.md`
- `documentation/modules/training.md`
- `documentation/modules/evaluation.md`
- `documentation/modules/reconstruction.md`
- `documentation/evidence/terminal_and_session_evidence.md`
- `documentation/agents/dataset_agent.md`
- `documentation/agents/hardware_agent.md`
- `documentation/agents/training_agent.md`
- `documentation/agents/reconstruct_agent.md`
- `documentation/agents/evaluation_agent.md`

The docs were grounded in the portfolio baseline document, code inspection, existing docs, `.august/memory/project`, experiment folders, zsh history, and Claude session command patterns.

Update after user feedback: the readable/source-of-truth code docs now mirror actual `logic/` folders under `documentation/logic/`. The broad `documentation/modules/*.md` files are only overview summaries.

Second update: added `documentation/runbooks/` with executable scenarios for preflight, data collection, dataset prep, experiment version creation, local/remote training, training chains, checkpoint pulling, real inference, protocol evaluation, evaluation analysis, and common scenario shortcuts.

Important captured decisions:

- Remote training uses `lerobot-train`; local `SmolVLABackend` changes do not affect remote runs.
- Dataset transforms should normally create new artifacts, not mutate source datasets.
- Real protocol evaluation is the main validation signal.
- Depth is represented as `observation.images.gripper_depth` and must match train/inference preparation.
- Camera/servo state should be checked before model debugging.
