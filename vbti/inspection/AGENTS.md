# inspection — system

**Two layers.** The reasoning layer decides *what* to look at, and the execution layer does *how*.

- **Planner** (`brain/`): a Claude Agent SDK loop that owns the question, the plan and the answer. It never sees images. Its tools are plan, move, inspect and answer.
- **Eyes** (`eyes/`): VLM subagents that read one captured view and return text. All visual knowledge reaches the planner this way.
- **World** (`cell/`): one model of the workcell (robot, table, object primitive) that every other part queries.
- **Views** (`view/`): the camera positions the planner can address, as rays around the object at a safe distance.
- **Motion** (`motion/`): IK on the UR5e plus OMPL. Every move runs as a transaction.
- **Loop** (`run/machine.py`): a state machine that wires it together. The operator UI (`ui/`) watches and confirms; STOP is always live.
- **Record** (`record/`): each run is written to one schema for review and replay.

Dependency direction: run → view/motion → cell ← perception. `cell/` imports nothing from sibling modules.

Each module's `AGENTS.md` holds its main details.

## Does NOT belong here
- ROS, launch files, message definitions.
- Imports from `vla.*` or `humanoid.*`: the projects are decoupled.
