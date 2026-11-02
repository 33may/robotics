# inspection — AI-driven inspection on a UR5e

The purpose is to use AI for decision making in the inspection process, so the robot can answer any question about any object.

We do this with an agent loop. The robot commands, the representation of the current state of the world and the state of the object are exposed as tools, and a VLM uses them to guide the inspection.

## What's inside

| Part | Module | Role |
|---|---|---|
| Planner | `logic/brain/` | The orchestrator agent. Owns the question, the plan and the answer; never sees images |
| Vision | `logic/eyes/` | VLM agents that read captured views and report back as text |
| World | `logic/cell/`, `logic/perception/`, `logic/calib/`, `logic/fit/` | Workcell model, camera capture and hand-eye calibration, object shape from captured clouds |
| Views | `logic/view/` | Where the camera can stand around the object, as safe, addressable views |
| Robot | `logic/motion/` | IK, planning and execution on the UR5e |
| Loop | `logic/run/`, `logic/ui/` | The inspection loop, plus the operator UI that watches and controls it |
| Record | `logic/record/` | Every run saved to one schema, for review and replay |
| Research | `investigation/`, `logic/viewplan/` | Experiments beyond the PoC |

## Install

One conda env (Python 3.11) and Node for the operator UI.

```bash
conda env create -f vbti/inspection/envs/inspection.yml
conda activate inspection
pip install -e vbti/inspection
pip install -e vbti/inspection/vendor/porthole/python

cd vbti/inspection/vendor/porthole && npm ci && npm run build -w @porthole/framework
cd ../../ui && npm ci && npm run build
```

Keys: `GEMINI_API_KEY` for the vision agents (`logic/eyes/`), Anthropic auth for the Claude Agent SDK (`logic/brain/`).

## Documentation

```
docs/
  tutorial/     hands-on, in order: 01 cell & calibration · 02 running an inspection · 03 reviewing a run
  blueprints/   how each part is designed and why: loop, image machinery, views, face evidence, data engine
  report/       the PoC: what we built, what it achieved, limits and next directions
  research/     post-PoC research foundation (IWM) and its experiments in investigation/
  demo/         demo script, recordings and context
```

**How to read:** start with `report/` for what the system does and where it stopped. The system itself is described in `AGENTS.md`: this folder's for the high level, each module's for its main details. Use `tutorial/` on the real cell. Go to `blueprints/` before changing how a part works.
