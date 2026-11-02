# 1. Cell and calibration

**What you have:** a UR5e welded to its table, an RG2 on the flange and a wrist RealSense D405 (serial `WRIST_SERIAL` in `logic/perception/camera.py`) on a bracket under the gripper. The controller is at `192.168.2.50`; every module that talks to it carries that as `ROBOT_IP` and takes `--ip`. The host needs an address on `192.168.2.0/24`. Everything that plans or reconstructs reads one world model, `logic/cell/cell.yaml`: the probed `base → table` frame, the taught keep-out `corners` and `planes`, the `floor` slab, and the `tool` boxes welded to the flange. The camera pose on the flange is `calib/T_flange_cam_<date>.npy` (the loader takes the newest by name; the `.yaml` beside it is its provenance). The run starts from `logic/run/survey_pose.json`.

**What you need to do:** get the arm into Remote Control and check that the model's base frame matches the controller's. Measure the cell into `cell.yaml` from freedrive taps and check it visually. Solve the hand-eye transform and commit it as a dated artifact, teach the survey pose, then Tare the D405 depth and keep the candidate that scores best on the baseline cells.

```bash
# robot: pendant in Remote Control
python -m inspection.motion.execute preflight            # read-only: robot mode, safety mode, remote control, go
python -m inspection.motion.execute bringup              # power on + brake release, then preflight
python -m inspection.calib.frame_check --n=3             # freedrive between samples; model vs controller base frame

# cell: freedrive taps with the closed RG2 tip, pendant Feature = Base
python -m inspection.calib.tap_probe                     # "s" prints the TCP point in the base frame
# table marks -> calib/table_probe_<date>.yaml, wall corners -> calib/planes_<date>.yaml
python -m inspection.calib.probe_fit                     # newest table_probe_*.yaml -> frames.table pose for cell.yaml
python -m inspection.calib.plane_fit                     # newest planes_*.yaml -> corners/planes snippet for cell.yaml
python -m inspection.cell.viewer                         # meshcat; re-renders on every save of cell.yaml
python -m inspection.cell.topview                        # logic/cell/topview.png: footprints, keep-out sides, reach circle
python -m inspection.cell.world                          # collision smoke: truth table + timings
python -m inspection.cell.world --demo                   # truth-table poses in meshcat, coloured by verdict

# hand-eye: ChArUco 8x6/20 mm (data/inspection/charuco_8x6_20mm_a4_print.pdf) flat on the table
python -m inspection.perception.handeye collect --outdir=vbti/inspection/data/inspection/handeye_ur
python -m inspection.perception.handeye solve --outdir=vbti/inspection/data/inspection/handeye_ur
cp vbti/inspection/data/inspection/handeye_ur/T_flange_cam.npy vbti/inspection/logic/calib/T_flange_cam_<YYYY-MM-DD>.npy
# write calib/T_flange_cam_<YYYY-MM-DD>.yaml (pairs, method, residual); move the lens station in cell.yaml tool.cam_stack if it changed

# survey pose: freedrive the camera over the object area, looking down
python -m inspection.motion.wrist_watch                  # every joint vs survey_pose.json, red = the planner will refuse
python -m inspection.run.app teach                       # current q -> logic/run/survey_pose.json

# depth: baseline is 1310-calib-red (red object with a yellow dot, survey + 6 cells, untared camera)
python -m inspection.calib.depth_tare measure            # board under the camera: depth error here, no motion
python -m inspection.calib.depth_tare scan               # asks before each move; one Tare candidate per height
# per candidate in calib/d405_tare/<MMDD-HHMM>/: write it, re-run the baseline cells, compare
python -m inspection.calib.depth_tare apply --cand=vbti/inspection/logic/calib/d405_tare/<MMDD-HHMM>/h240.json --yes
python -m inspection.calib.depth_eval --label=h240       # red object on the table; survey + baseline cells, auto-confirmed
# leave the winner applied; roll back with
python -m inspection.calib.depth_tare restore --backup=vbti/inspection/logic/calib/d405_tare/backup_before_apply_<stamp>.json --yes
```

**Done when:**
- `preflight` reports `go` as `True`
- `frame_check --n=3` prints `MODEL AGREES WITH THE ROBOT`
- `probe_fit` and `plane_fit` output is pasted into `cell.yaml`, and in meshcat the table, floor slab and keep-outs sit where the real ones are, with the tool boxes enclosing the gripper and camera in every pose of `world --demo`
- `topview.png` hatches each keep-out on the wall side, not the robot side
- `handeye solve` puts PARK, DANIILIDIS and HORAUD within the "good" or "nominal-DH floor" band of the recipe it prints, and `calib/T_flange_cam_<date>.npy` is the newest artifact there
- `wrist_watch` is green on every joint at the taught pose
- `data/depth_val/<run>/compare.json` exists for every candidate, and the applied one beats `1310-calib-red` on dot error and table height

## Cell and calibration commands

| Command | What it does |
|---|---|
| `python -m inspection.motion.execute preflight \| bringup \| demo [--ip]` | Go/no-go report (read-only); power on + brake release; validated ±5° wrist_3 wiggle |
| `python -m inspection.calib.frame_check [--n=N] [--ip]` | Flange → TCP offset at N freedrive poses; constant offset = model and controller agree |
| `python -m inspection.calib.tap_probe [--ip]` | Freedrive tap logger: `s` prints the TCP point (m, base frame), `q` quits |
| `python -m inspection.calib.probe_fit [--data <yaml>]` | Kabsch fit of table touch points, holdout check, `frames.table` pose line |
| `python -m inspection.calib.plane_fit [<planes.yaml>]` | Corner taps → oriented keep-out planes, `corners:`/`planes:` snippet |
| `python -m inspection.cell.viewer [--no-robot] [--once] [--yaml]` | meshcat live preview of `cell.yaml`, reloads on save |
| `python -m inspection.cell.topview [--yaml] [-o <png>]` | Top-view render of what the yaml says |
| `python -m inspection.cell.world [--demo \| --replay]` | Collision world smoke; meshcat truth-table poses or a path replay frozen at impact |
| `python -m inspection.cell.usd_preview [--print] [--slide-mm <mm>]` | Isaac USD of UR5e + RG2 in meshcat, gripper measured in the flange frame |
| `python -m inspection.perception.camera live` | Live D405 colour view |
| `python -m inspection.perception.handeye collect --outdir=<dir>` | Freedrive hand-eye pairs, auto-captured when the arm is still and the board is seen, `q` ends; resumes an existing dir |
| `python -m inspection.perception.handeye solve --outdir=<dir>` | `calibrateHandEye` PARK + cross-methods, board-in-base residual, `<dir>/T_flange_cam.npy` |
| `python -m inspection.motion.wrist_watch [<ip>]` | Live per-joint delta to `survey_pose.json` during freedrive |
| `python -m inspection.run.app teach [--ip]` | Save the current joints as the survey pose |
| `python -m inspection.calib.depth_tare measure \| scan [--heights] [--auto]` | Board depth error at the current pose; straight-down Tare candidates per height, nothing persisted |
| `python -m inspection.calib.depth_tare apply --cand=<h.json> --yes` | Write a candidate to the camera EEPROM, flash table backed up first |
| `python -m inspection.calib.depth_tare restore --backup=<json> --yes \| --factory --yes` | Put a backup or the factory table back |
| `python -m inspection.calib.depth_eval <label> [--baseline] [--skip_capture=<run>]` | Survey + baseline cells as a collect run, `depth_val` on it, before/after table |
| `python -m inspection.calib.depth_val --run=<run>` | Yellow-dot triangulation and table plane per view of a recorded run (read-only) |

Design and history of the cell model: `logic/cell/AGENTS.md`, `docs/demo/context/04-motion-and-safety.md`.
