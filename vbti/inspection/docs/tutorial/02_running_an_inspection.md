# 2. Running an inspection

**What you have:** the calibrated cell from block 1, the `inspection` env, and two model providers: Gemini for the vision agents (`eyes/`, `GEMINI_API_KEY`) and Claude for the planner (`brain/`, the Claude Agent SDK, authenticated by `ANTHROPIC_API_KEY` or a Claude Code login). `run/app.py run` is the composition root: it checks the key and the robot preflight, starts the porthole bus, the `Supervisor` (`run/machine.py`), the real rig and the operator window. The question is not a flag; it is typed into the window. Nothing moves until the operator confirms it. `ui/app.py mock` is the same `Supervisor`, bus and window over a `FakeRig` (`run/rigs.py`): a real planner, stub vision and stub segmentation, and no robot or camera.

**What you need to do:** export the keys, build the frontend once, rehearse the operator flow on the mock, then start the live app, ask, approve every move, and read the answer.

```bash
export GEMINI_API_KEY=<key>                  # run refuses to start without it
export ANTHROPIC_API_KEY=<key>               # or a logged-in Claude Code

# frontend, once per checkout
cd vbti/inspection/vendor/porthole && npm ci && npm run build -w @porthole/framework
cd ../../ui && npm ci && npm run build && cd ../../..

# rehearsal: no robot, no camera, run dir in a tempdir
python -m inspection.ui.app mock
python -m inspection.ui.app mock --collect --auto        # sweep flow: no question, every cell auto-confirmed after its preview

# live: pendant in Remote Control, object on the table, arm at or near the survey pose
python -m inspection.motion.execute preflight
python -m inspection.run.app run --name=<name> --object=box        # -> data/runs/<DDMM>-<name>
python -m inspection.ui.tap --show --rtde                           # second terminal: topic rates, pose motion vs the arm's own joints

# data sweep instead of a question: all reachable cells, no AI
python -m inspection.run.app collect --name=<name> --object=box --auto   # -> data/datasets/<DDMM>-<name>
```

In the window: wait for `sam3 warm` in the log, then type the question into the trace panel. The planner's first move is the survey. Every move is the same two presses. The request plans the move and loops a preview in the 3D panel, and the confirm executes it. When the planner drives, it makes the request and the actions panel shows a single confirm button. `approve`/`grid` toggles the full grid of cells and faces, where both presses are yours. A refused plan turns the cell amber and the arm stays put. STOP is never disabled and halts a move in progress. `exit` ends the run cleanly; on a sweep, `finish` closes it as completed. Ctrl-C in the terminal is a safe shutdown; a second Ctrl-C is a hard exit.

**Done when:**
- on the mock, a question runs survey → moves → answer card, every move waiting at its preview until confirmed
- live, `preflight` reports `go` as `True` and the app prints the `ui` URL (port 8767) and opens the window
- during a move, `tap` shows `scene/poses` and `camera/wrist` streaming and the pose moving with the arm
- STOP during a move halts the arm
- the trace panel shows the verdict with cited views, and `python -m inspection.record ls` lists the run as `completed`

## Run commands

| Command | What it does |
|---|---|
| `python -m inspection.run.app run --name=<n> --object=<o> [--outdir] [--fit] [--ip] [--port] [--bus_port] [--no_window] [--gui]` | Live inspection run into `data/runs/<DDMM>-<n>` (`-2`, `-3` if taken); `--object` ∈ `box`, `cup`, `cylinder`, `duck`; `--fit none` disables the box fitter; `--no_window` serves the UI in the browser |
| `python -m inspection.run.app collect --name=<n> --object=<o> [--auto] [--random] [--cells]` | Sweep without cognition into `data/datasets/<DDMM>-<n>`; `--auto` confirms each view after its preview, `--random` shuffles the order, `--cells` visits exactly these cells |
| `python -m inspection.run.app teach` | Save the current joints as `run/survey_pose.json` |
| `python -m inspection.ui.app mock [--collect] [--auto] [--no_window] [--bus_port] [--run_root]` | Real `Supervisor` + `FakeRig` over the real bus; stub vision, real planner |
| `python -m inspection.ui.app serve --run_dir=<dir>` | Window only, no bus; for a loop that already owns the bus |
| `python -m inspection.ui.tap [--topic=<t>] [--show] [--rtde]` | Read-only bus monitor: per-topic rates, pose motion, arm motion from its own RTDE receive |
| `python -m inspection.motion.wrist_watch` | Per-joint distance to the survey pose, for freedriving back after an abort |
| `pytest vbti/inspection/tests/e2e -m e2e` | Headless-browser runs of the mock app, both flows |

Loop design: `docs/blueprints/ui-driven-loop.md`, `docs/blueprints/loop-v1.md`; UI contract: `ui/AGENTS.md`.
