# e07 — the IWM update loop, designed with Anton playing the agent

**Aim.** Design the agentic capture loop that maintains the IWM (P, T, V, C → P′, T′, V′, C′):
per capture, what is new, what extends an existing part, what reorganises the tree. Method
(Anton, 2026-10-12): he plays the agent on real captures, we write down what he wants to see and
do, then derive the tools and the edit operations. No tool list up front.

**Loop being designed (Anton, 2026-10-12).** capture → depth fused into V (code, always) →
agentic segmentation (blind, with the IWM as prior, or both — both get tested) → the agent works
out new / existing / reorganise using code tools that bring V, C, P into the 2D frame → maintenance
round → (frame verdict: app loop, out of IWM scope).

**Settled so far.** T and V agree: no part without voxels; voxels without a label are normal (the
survey boots V, the first inspect inits T). T starts with an object root; the first inspect hangs
its frame tree under it; the root's voxels = union of its parts. The agent reconciles whole trees,
not node by node, and edits the in-memory IWM through tools (it never writes the tree out).

**Test when building.** A part's voxels = own geometry ∪ children's (plain union loses a face's own
surface). Edits applied immediately vs collected and committed at the end of the round. Blind vs
prior segmentation. Reorganisation (case c) rules — discussed once the whole system is visible.

## The loop, executed (loop.py) and its viewer (loop_viewer.py)

`loop.py` runs the loop for real on a recorded run, every captured step in order: depth fused into
one growing TSDF (step 0 = survey, V only) → e01 mask sets (UnSAMv2 ladder + SAM 3 text; made for
the steps e01 never saw, before the 1 mm grid takes ~16 GB of the GPU) → draft tree by containment
→ e02's segmentation agent, blind to the IWM → every mask the frame ever had lifted (L3) into this
step's V → reconcile: first inspect = frame tree under the object root (code), later = e05's keep
agent → commit, invariants checked. Every tool call is recorded with its full result, its image and
the state right after it. Record: `inspection/data/iwm/e07_update_loop/loops/<run>_<stamp>/`.

`loop_viewer.py` (port 8084) replays a record: capture / event / stage navigation; the frame with
the current frame tree's masks; the trace event with its image and the prompt; the frame tree and
the IWM tree as node-link diagrams (decisions next to frame nodes during reconcile, NEW / +view
next to parts at the commit); 3D: V of that step, IWM voxels per part (before / after the commit),
the frame masks lifted into 3D per node (deepest node or one level), a focus node in white.

    /home/anton/miniconda3/envs/robo/bin/python -u -m inspection.investigation.iwm.e07_update_loop.loop --run 2209-box-ray-demo1
    /home/anton/miniconda3/envs/robo/bin/python -m inspection.investigation.iwm.e07_update_loop.loop_viewer --run 2209-box-ray-demo1

Known limits of the loop as built: segmentation is blind only (the prior variant is to be designed);
the agents' per-turn thinking is not logged (only tool calls and the final reply); edits are
collected and applied at the commit (immediate edits: a parked test); e05's keep verbs only
(same / under / drop — no re-organisation yet).

## First viewer (superseded): replay of the e02 + e05 traces

`viewer.py` (viser, port 8084) replays a run as captures × inspect blocks: capture → V → segment
(e02 trace) → reconcile (e05 keep trace) → commit (P/T diff). Sidebar: navigation (capture, event,
stage), incoming frame, agent trace event (tool, args, result, the image the tool returned, the
final reply's evidence/reasoning/decisions, the prompt), IWM tree with change tags, part
highlighter. 3D: V mesh, IWM voxels coloured by deepest part (the state the event sees), capture
cameras, changed parts marked white at the commit.

    /home/anton/miniconda3/envs/robo/bin/python -m inspection.investigation.iwm.e07_update_loop.viewer --run 2209-compute-ray-demo5
    ... --run 2209-box-ray-demo1

**What the replay is, and is not.** It stitches two separate e02/e05 runs into the loop's order.
Known gaps against the designed loop:
- the agent's per-turn thinking is not in the stores — only tool calls and the final reply's
  evidence / reasoning;
- V in the 3D grows per captured step (`grow.py`, one TSDF integrated in order, step 0 = survey);
  but the e05 parts were LIFTED against the all-steps V, so their voxels come from that grid;
- every captured step is a block; only c5.2/c5.6/c5.7 and b1.1/b1.4 have agent traces;
- the first capture was initialised by e05's rule (frame root = object), not the object-root rule;
- the reconcile trace is e05's keep agent (verbs tree/existing/show/show_existing/same/under/drop).
