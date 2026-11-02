# IWM research

This folder is research, not product code. It develops the IWM: a structure a robot
builds from images and 3D data, holds what it has seen in, and decides from. The work
may become a short paper, so every result here has to be traceable and kept.

Nothing in this folder is imported by the robot loop.

## Where things are

| What | Where |
|---|---|
| The idea and the research questions | `vbti/inspection/docs/research/iwm-research-foundation.md` |
| The experiment ladder | `vbti/inspection/docs/research/iwm-experiments-plan.md` |
| Literature map | `docs/research/iwm/passes/pass1/` |
| Papers, as PDF and markdown | `docs/research/iwm/papers/` |
| Experiments | this folder, one folder each |
| Heavy experiment data | `inspection/data/iwm/`, one folder per experiment, not in git |

## Rules

1. **One folder per experiment**, named `eNN_short_name`, numbered in the order they are
   opened. A second attempt at the same question gets a letter: `e04b_...`.
2. **A note before.** No script runs until `NOTE.md` says what we want to achieve.
3. **A report after.** `REPORT.md` is written when the experiment closes, and it is
   concise. An experiment that fails gets a report too.
4. **Data is stored.** Nothing an experiment produces goes to `/tmp`.
   - What the report shows (figures, small result files) goes in the experiment's
     `evidence/` folder and is committed.
   - Heavy outputs (masks, clouds, arrays) go in `inspection/data/iwm/eNN_short_name/`.
5. **Every run leaves a record.** Each run of a script writes a `run.json` next to its
   outputs: date, git commit, command, the recorded views used (run and step), model
   and settings.
6. **Recorded runs are read-only.** They are referred to by run and step, never copied
   and never changed.
7. **Figures come from scripts.** No figure is edited by hand. The report names the
   command that made it.
8. **Numbers come from files.** A number in a report is read from a result file in
   `evidence/`, not typed from a terminal.

## An experiment folder

```
e01_all_masks/
  NOTE.md      written before the first run
  REPORT.md    written when the experiment closes
  *.py         the scripts, run as  p inspection/investigation/iwm/e01_all_masks/<script>.py
  evidence/    figures and result files the report uses, committed
```

## NOTE.md

```markdown
# eNN: title

**Opened:** date. **Rung:** which rung of the ladder.

## Question
## Why
## Input
## What we do
## What we look at
## What would count as a no
## Done when
```

## REPORT.md

```markdown
# eNN: title

**Closed:** date. **Answer:** one or two lines.

## What was run        inputs, settings, commit, command
## What we saw         figures from evidence/
## Numbers             a table, read from evidence/
## Limits              what this does not show
## What changes        in the idea or the plan, and what comes next
```

## Experiments

| # | Experiment | Status | Answer |
|---|---|---|---|
| e01 | [All masks from one image, and a tree from them](e01_all_masks/NOTE.md) | running, turn 1 closed ([report](e01_all_masks/REPORT.md)) | UnSAMv2 at six granularities + containment gives object → faces → details → sub-details on all eight frames; SAM 3 text prompts name the details, its point mode cannot make levels |
| e02 | [The vision agent builds the tree](e02_agent_tree/NOTE.md) | running, turn 1 closed ([report](e02_agent_tree/REPORT.md)) | The in-loop agent (Gemini ER-2) with tree/show/find/name/group/drop turns the e01 draft into a named tree of 6–19 nodes in 8–12 turns, ~45 s; `show` is indispensable, "name it or drop it" removes the fragments |
| e03 | [The tree in 3D: projection and viewer](e03_tree_3d/NOTE.md) | running, turn 1 ([report](e03_tree_3d/REPORT.md)) | Every node projects (85–100 % of its pixels carry depth, 1–8 mm from the fused cloud); Viser drill-down viewer: sides → click → children, source frame + parents on the right |

| e04 | [The voxel geometry V of a run: build, navigate, resolution](e04_voxel_scene/NOTE.md) | running, turn 1 ([report](e04_voxel_scene/REPORT.md)) | Open3D TSDF fuses a run in < 1 s at 1–10 mm; at 1–2 mm ports, wordmark and label marks survive as shapes, V matches each step's own depth to ~1–2 mm median; object scope drops faces of unmasked steps, scene scope keeps them |
| e04 / normals | [A reliable surface normal at a point of V](e04_voxel_scene/normals/NOTE.md) | run ([report](e04_voxel_scene/normals/REPORT.md)) | RANSAC plane + PCA over a 16 mm region: 2.2° median / 5.3° p90 off the face (1 mm V) vs 11–17° / 38–54° for the triangle / TSDF-gradient normals; a 5–20 mm part's own points are worse than its surroundings. Turn 2, edges: no geometric trick fixes points < 8 mm from an edge (30–40° at 0–4 mm); fitting only the own FACE's points (from T) holds 3–7° |
| e05 | [Parts in V: the S/T/V structure, the agent keeps T, views from it](e05_parts_in_v/NOTE.md) | running, turn 1 ([report](e05_parts_in_v/REPORT.md)) | A mask lifted by ray-casting V gated by the view's own depth lands as a clean part (no smear, IoU 0.75 vs the mask); S/T/V keeps nesting by construction; geometry pairs 1/10 ports across views (8 mm offset), the agent with the projection pairs 10/10; conflicts are T disagreements, not geometry; the computed view for a part matches or beats the recorded step for 32/38 parts, but the face-on pose is infeasible for most parts near the table — the face-plane shift (the loop's rule) rescues them; all six part sets get a pose ≥ 0.89 visible |

Status is one of: note written, running, closed. The rungs still to come are in the
ladder.
