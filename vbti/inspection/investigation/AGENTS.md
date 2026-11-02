# investigation — research beyond the PoC

## Purpose
One-off research scripts and notes, one folder per investigation branch.
Offline replays over recorded runs, benches and probes that informed (or may
inform) the shipped modules. Nothing here is imported by the robot loop.
Write-ups and literature: `docs/research/`.

## Folders
- `run_replay/`: replay a recorded run offline: `cloud_replay` (depth-only
  accumulation), `step_replay` (exact per-step cloud), `mask_probe` (drives the
  shipped `run.segmenter.object_view`), `retreat_replay` (the retreat decision).
- `partial_prompt/`: partial prompt box → partial SAM mask, mitigated by a
  detector-anchored prompt; `REPORT.md` + `evidence/`.
- `ray_views/`: replay bench for ray views (`ray_replay`, `frame_render`);
  design `docs/blueprints/ray-views.md`.
- `face_replay.py`: boxguard face-evidence gate and resolution ladder over a run.
- `view_framing/`: how big the object looks per cell (`fill_probe`) and with a tighter shell (`frame_preview`).
- `view_hints/`: whether a subagent crop box / `framing.facing` phrase can
  name the next cell (`normal_probe`, `phrase_calib`); design `docs/blueprints/view-hints.md`.
- `tool_hull/`: honest tool collision hull: per-part bounds from USD and
  photos, reachable-cell trade-off, blocked-cell attribution, blueprint drawing.
  Inputs in `data/tool/`.
- `loop_stability/`: why the RTDE stream and camera stalled: reader threads
  starved of CPU. `SYNTHESIS.md` + four sub-reports.
- `iwm/`: IWM research (a structure the robot builds from images and 3D and
  decides from). Rules in `iwm/README.md` (NOTE before, REPORT after, evidence
  committed, heavy data in `data/iwm/`). Experiments:
  - `e01_all_masks`: UnSAMv2 multi-granularity masks stacked into a part tree; SAM 3 text for naming.
  - `e02_agent_tree`: the unchanged `VlmAgent` loop with tree verbs turns the draft tree into a named tree.
  - `e03_tree_3d`: tree nodes projected into the base frame; Viser viewer.
  - `e04_voxel_scene`: TSDF voxel grid V of a run; 1–2 mm resolves the details.
  - `e05_parts_in_v`: masks lifted into V (own-depth-gated ray cast), cross-view offset measured.
  - `e06_view_offset`: source of the per-view rigid offset (~10 mm); NOTE only, no REPORT.
  - `e07_update_loop`: design of the agentic IWM update loop; NOTE only.
  Docs: `docs/research/iwm-research-foundation.md`, `iwm-experiments-plan.md`,
  `iwm-e05-parts-in-v.md`, `docs/research/iwm/`.

## Rules
- One folder per branch; no loose files at this level (`face_replay.py` predates the rule).
- Recorded runs are read-only: refer by run and step, never copy or modify.
- May import any `inspection.*` module; nothing outside `investigation/` may import from here.
- Scripts here are not covered by `tests/` and may be stale against current code.

## Does NOT belong here
- Code the loop runs. A result that ships moves into its module with tests.
