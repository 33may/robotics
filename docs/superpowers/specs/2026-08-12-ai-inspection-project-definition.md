# AI-Driven General Inspection — Project Definition

Date: 2026-08-12
Stakeholder: Albert (VBTI)
Owner: Anton

## Goal

Show how the world knowledge and spatial reasoning inside large vision-language models
can be converted into concrete robot inspection actions.

The robot must derive *how to inspect* from the question it is asked, instead of
replaying a path a human programmed. Deliverable is a proof of concept plus the research
and insights behind it — not a product feature.

## Context

VBTI ships **ADAM**, an inspection robot that executes pre-defined trajectories. A human
programs the inspection path per part. Albert's product vision is an inspector you can
simply ask: it understands what inspection means, decides how to inspect, and answers
questions about the object.

The gap between those two is the whole project. ADAM already moves and already inspects.
The only thing it cannot do is derive the path from the question. That derivation is the
research contribution; the arm, the camera and the control loop are plumbing.

## Demo

A cup is placed on the table. The robot is asked: *"Is there a logo on this cup?"* It
inspects the cup from whatever sides it needs and answers.

Generality is proven by live swap: at demo time an arbitrary object is placed on the
table and an arbitrary question is asked, with no configuration change.

The cup and the logo are an example, never a special case.

## Success and Failure Criteria

### Success

For any object and any question, the system returns the correct answer, having moved only
to viewpoints that served the question, without colliding with the object, the table, the
fixtures or itself.

Collision-free operation is a hard gate, not a quality metric. A run that produces the
right answer after touching something is a failed run.

### Failure modes

Four named failures. Each must be separately measurable — reporting a single success rate
hides which of these is actually happening.

**F1 — Incoherent viewpoints.** The robot moves to positions that do not make sense for
the question. Symptom of the model selecting without grounding. Measured by whether the
chosen view actually contained the region of interest, and by realised versus predicted
information gain.

**F2 — Non-termination.** The robot keeps moving and never commits to an answer. Measured
by view-count distribution and budget-exhaustion rate.

**F3 — Wrong answer.** Measured against ground truth. Note this can occur with perfect
viewpoint selection — a clean view of the logo that the model misreads is a perception
failure, not a planning failure, and the evaluation must separate the two.

**F4 — Premature stop.** The robot answers before gathering enough evidence. Hardest to
measure: it requires an oracle continuation — keep looking after the system stopped, and
check whether the answer would have changed.

### The central tension

**F2 and F4 are opposite ends of one dial.** Every tightening of the stopping rule trades
non-termination for premature stopping, and every loosening trades it back. The stopping
criterion is therefore not a threshold to be tuned quietly but the main object of study,
and both rates must be reported together — a system that never stops early because it
never stops is not a good system.

This is also where the research contribution sits. Published work almost never reports
premature-stop and over-exploration as distinct quantities; reporting them on a real arm
would itself be a result.

## Non-Goals

These are excluded by definition, not by schedule.

- **No hardcoded inspection path.** A fixed "orbit the object in N steps" sweep is ADAM's
  current behaviour. If the number and placement of views do not fall out of the model's
  understanding of the object and the question, the demo works and the research says
  nothing.
- **No object-specific or task-specific code.** No cup detector, no logo classifier, no
  strings naming either in the codebase.
- **No manipulation.** The camera moves around the object. The object is not touched,
  grasped or reoriented.
- **No embodiment-specific data collection or VLA finetuning** for the demo path. See
  Executor below.
- **No productisation.** Cycle time, throughput and robustness beyond demo conditions are
  out of scope.

## Hardware and Software Baseline

Verified 2026-08-12.

| Item | State |
|---|---|
| **UR5e** | 6-DOF, 5 kg payload, ~850 mm reach. Up and running in the office, previously driven from Python by a colleague. Camera already mounted on the gripper, plus a table camera. Target: gripper camera only. |
| **SO-ARM101** | **Not used in this project** (Anton, 2026-08-14). Its 5 DOF makes look-at exactly determined with no branch choice, so it would validate plumbing rather than kinematics — and the risk it would have retired, depth on glossy surfaces, needs only the camera on a bench. |
| **Cameras** | 4× RealSense D405 supported in `vbti/logic/cameras/cameras.py:33`, including aligned depth on the gripper camera. Additional RealSense units available if needed. |
| **Dev machine** | `vbti-MS-7E66`, Ubuntu 24.04, RTX 5090 32 GB (Blackwell, sm_120), 60 GB RAM. |
| **Remote compute** | `vbti@10.11.100.156` via `vbti/logic/train/remote.py` (rsync + ssh + tmux). GPU model not yet confirmed. Cloud rental available if required. |
| **ML environment** | **Not installed.** `~/projects/robotics/.venv` holds only `fire`, `PyYAML`, `tqdm`, `termcolor` and the editable `vbti` package. No torch, no LeRobot, no pyrealsense2. |
| **Existing perception** | Grounding DINO wired at `vbti/logic/detection/detect.py:18` — open-vocabulary model, currently hardcoded to a duck/cup prompt. |
| **Missing** | No IK, no FK, no cartesian control, no motion planning anywhere in `vbti/`. No hand-eye calibration. No task planner or state machine. No LLM/VLM API client. |

## Architecture Direction

**We own the geometry. The model owns the semantics.**

The system enumerates a reachable, IK-filtered set of candidate viewpoints around the
object. The vision-language model never emits a pose — it selects among enumerated
candidates and decides when it has seen enough.

This division is what makes the system general. The viewsphere does not know what a cup
is. The model does not need to know what a metre is.

### One loop, not two phases

Exploration and inspection are the same iterative step. There is no survey phase followed
by an inspection phase; instead the utility of a candidate viewpoint carries two terms —
geometric gain (reduce unknown extent) and semantic gain (evidence for the question).
Early on the object is unknown so the geometric term dominates; once extent is settled that
term decays and semantics takes over. The explore-then-inspect behaviour emerges rather
than being scheduled.

### The safety invariant that lets the loop start

To orbit safely you need collision geometry; to get collision geometry you must move.
The resolution is a **monotonic, pessimistic envelope**: assume anything standing on the
table is inside a conservative volume, plan only against that, and let each observation
shrink it. Safety never depends on knowing the object — only on never over-claiming
knowledge.

**The envelope is measured, not declared.** It comes from the depth cloud: table plane
removed, cluster fitted to an oriented bounding box, inflated by the calibration margin. Each
new view adds points and refines it. Nothing in the system states how tall or wide an object
may be — those are sensor readings, so a plate and a bottle get different envelopes.

Conservatism comes from how the unobserved region is filled, not from a declared bound. A
single view sees one surface, so the box is extended through the occluded volume behind it,
bounded by the table plane below. Looking down first makes that fill small: horizontal extent
is directly observed and only the volume beneath the visible top surface is hidden.

**The dependency this creates is depth quality**, which is why Fast-FoundationStereo is
load-bearing rather than an optimisation — the D405's own passive stereo fails on precisely
the glossy, textureless surfaces the demo uses. The silhouette-on-table-plane fallback remains
for when depth fails anyway, but it recovers footprint only, and an object of unknown height
cannot be approached closely.

### S0 — cell preparation (design deferred)

**How the starting geometry is produced is a separate design.** Not decided here.

What is fixed is the contract. Before the first motion command, the system must hold, in
the robot base frame:

| Output | Consumed by |
|---|---|
| Robot kinematic model — DH or URDF, ideally factory-calibrated | S4 IK, S5 path validation |
| `T_ee_cam` — hand-eye transform, camera relative to flange | Every projection: silhouette back-projection, viewpoint aiming, evidence pose |
| Camera intrinsics + distortion | S2 localisation, S6 pose tagging |
| Table plane — normal and offset | S2 plane removal, S3 half-space intersection, S5 collision |
| Workspace bounds — a box outside which no viewpoint is generated | S4 candidate filtering |
| Static obstacles — fixtures, mount, second camera, cabling | S5 collision |

Nothing here concerns the object, and nothing here changes when the object or the question
changes. That is the boundary: **per-cell setup is allowed, per-object setup is not.**
Anything that must be redone when a different object is placed on the table has leaked out
of S0 and belongs in the loop.

Open sub-questions for that design, listed so they are not lost:

- Whether the table plane is measured once or re-fitted from the first frames each run
- Whether static obstacles are hand-authored as primitives or captured as a point cloud
- Whether the factory DH parameters can be pulled off the controller without ROS 2 — see
  Risks; unresolved and it dominates the error budget
- What the accuracy acceptance test is, and what the system does when calibration drifts

### S1 — bootstrap sighting

The robot boots at a **declared bootstrap pose** — a cell property, not an object property.
It looks down at the table from above and sees the whole reachable working area, so "object
not in frame" would require a cell larger than one field of view.

**Depth builds the geometry. Detection only labels it.** That division is the point of the
stage. Nothing here declares how tall an object may be; the height is measured.

1. **Capture stationary.** Settle, dwell ~100 ms, then grab the raw IR stereo pair. Depth
   transformed while the arm is moving is worthless — up to 5 cm of displacement in the base
   frame, with 10 mm spikes tracking acceleration.
2. **Dense depth via Fast-FoundationStereo** on that IR pair. Zero-shot, no finetuning,
   nothing object-specific; metric scale comes from the D405's 18 mm baseline. ~20 ms and
   650 MB on the 5090. This replaces the camera's own passive stereo, which returns holes on
   glossy and textureless surfaces — the exact surfaces this project has to handle.
3. **Deproject** — pixel plus depth plus intrinsics gives a 3D point in millimetres; FK ×
   hand-eye moves the cloud into the robot base frame.
4. **Remove the table plane** — `segment_plane` at a 5 mm threshold.
5. **Everything left standing becomes the obstacle map** — voxelized, no semantics. Occupied
   space is a hazard whatever it belongs to, so this needs no clustering and no segmentation.
6. **Candidates come from 2D**, not from the cloud — segment the RGB frame, keep only masks
   whose points reach the table plane, dilate each one, and emit a crop per candidate.
7. **The chosen mask seeds the object**, and the object set grows from that seed by
   connectivity. Fit the box once, to the winner.

A plate measures 20 mm tall and a bottle measures 300 mm, because both numbers came off the
sensor.

**Why the view is from above, restated for the depth path.** One view sees only the surface
facing the camera, so a box fitted to it under-reports the extent away from the camera — an
*under*-approximation, which is the unsafe direction. Looking down makes that harmless: the
horizontal extent is directly observed, and the only unobserved region is between the visible
top surface and the table, which is bounded below by the table plane itself. Filling from the
observed top surface down to the table yields a solid that contains the object without any
declared bound.

**Fallback when depth still fails.** Back-project the 2D mask onto the known table plane. This
yields a footprint and a centroid but no height, which is enough to anchor the viewsphere and
not enough to approach closely. It is a degraded mode, flagged as such, and the next oblique
view is spent recovering the height.

**Search, if the first frame is empty.** Advance through a short list of poses verified once
against the worst-case table volume until something appears. Absence and detection failure are
indistinguishable and get identical treatment. The search carries its own budget, separate from
the inspection budget, so a failed search cannot consume the run. If the list exhausts, "I do
not see an object" is emitted as an answer, not raised as a fault.

**Deliverables**, into two different stores:

| Output | Store | Consumed by |
|---|---|---|
| First keyframe — image, camera pose, mask | evidence | Already usable evidence, retrievable at answer time |
| Object bounding box — centre, extents, axis, inflated | geometry | Viewpoint generation, IK collision filter |
| Object frame — box centre seated on the table plane | geometry | Anchors the azimuth–elevation viewpoint grid |

Which of several clusters is *the* object is decided next.

### Object selection — the model picks

**Decided (Anton, 2026-08-12).** The model chooses which object to inspect, given the
bootstrap image and the detection map. Not a largest-cluster rule, not a hardcoded noun.

The system cannot know what to inspect — nothing in the pixels marks one cluster as the
subject. The information lives in the question, and the question is the user's, so grounding
it is legitimate where hardcoding it would not be. This also covers the referring expressions
a detector cannot reach: *"the red one"*, *"the one on the left"*, *"the part in the fixture"*.

**The detection map** is the first instance of the geometry-as-text projection that recurs at
S7. Per cluster: an integer id, position on the table, measured extents, and a crop. The full
frame goes with it for context. Text and crops rather than drawn markers — models gain
substantially from images and text together over either alone, while marker overlays crowd and
occlude, and Set-of-Mark prompting is measurably fragile.

**Output is an integer id.** This is the same contract as viewpoint selection, and it holds
system-wide: *every model output in this system is a choice from an enumerated set, never a
continuous quantity.*

**Fixtures never become candidates**, because the static obstacles declared in S0 are
subtracted before clustering. This is what that part of cell prep is for.

**Degradation.** A question naming no object — *"is there any damage?"* — has nothing to
ground. One cluster: take it. Several: report the ambiguity as the answer rather than guessing.
An id that does not exist: re-ask once, then fall back to the same rule.

**The pick is stated in the final answer.** Object identity is fixed once, at S1, and the
viewsphere anchor, the coverage map and all retrieval are defined relative to it — so a wrong
pick invalidates the entire run with no recovery. Naming the chosen object in the answer turns
that from a silent failure into a visible one.

**Out of scope, and the natural extension:** letting the model ask the user when the reference
is ambiguous, instead of reporting ambiguity and stopping.

### Geometry module — initial estimation

All 3D work lives in one component. Nothing else in the system deprojects, clusters or fits
boxes. This section covers only the **initial estimate**, built from the bootstrap view.
In-loop refinement is deferred and designed separately.

**Input:** an RGB frame, an IR stereo pair, and `T_base_cam` for the pose they were taken
from, plus the static world from S0.

1. Learned stereo → dense metric depth
2. Deproject → cloud in camera frame → transform to base frame
3. Mask out the arm's own points, known exactly from FK — not noise, a known body
4. Crop to workspace, subtract the static obstacles from S0, remove the table plane
5. Everything left standing → **obstacle map** (voxelized, no semantics)
6. Candidates from **2D segmentation**, filtered to masks that reach the table plane
7. After selection: seed from the chosen mask, grow by connectivity, fit the box, build the
   safety envelope

**Two jobs, deliberately separated.** Motion planning needs to know which *space* is occupied,
not which object is which — so the obstacle map is built from raw occupancy with no semantics
at all. Only target identification needs candidates, and it needs them solely because the model
has to pick one. Conflating these was forcing 3D clustering to carry a job it is bad at.

**Candidates are enumerated in 2D.** The crop the model sees *is* the mask that selects the
points, so there is no matching problem between a cluster and its image region. A depth hole
leaves a candidate with fewer points rather than splitting one cup into two candidates, and a
cup touching a saucer separates semantically, which spatial clustering structurally cannot do.

Two guards make this safe. Masks are **dilated** before selecting points, so segmentation
error rounds outward. And **unassigned points remain obstacles** — a rim the mask missed is
still occupied space, so a segmentation failure costs tightness, never safety. Over-segmentation
into sub-parts is filtered geometrically: a candidate must **reach the table plane**, which a
logo or a handle-only mask does not.

**Object membership is a region of space, not an appearance.** Nothing is manipulated, so the
object is static for the whole run and identity does not have to be re-established per frame.
Detection runs once, to answer *which* thing; from then on membership is a geometric test.

**Growth rule — seeded connectivity (Anton, 2026-08-12).** The chosen mask's points are a
known-correct seed; a point joins the object if it lies within ε of a point already in the set.
This is spatial clustering, but seeded rather than blind, which changes its failure mode
entirely: it cannot latch onto the wrong object, it can only leak through physical contact.
A frozen gate would reject real points arriving from later views; an ungated one would absorb
every stereo flyer.

**Two geometries, not one.** A single view sees one surface, so these differ from the start
and conflating them is what makes the safety argument look self-contradictory:

| | What it is | Error direction |
|---|---|---|
| **Observed box** | OBB of the points actually seen | **Under**-reports extent away from the camera |
| **Safety envelope** | Observed box plus a conservative fill of the occluded volume behind it, floored by the table plane | **Over**-reports, which is the safe direction |

The envelope is what the IK filter tests against. The observed box is what the viewsphere is
sized from. Looking down keeps the gap between them small, because horizontal extent is
directly observed and only the volume beneath the visible top surface is hidden.

**The object frame is created here and frozen for the run** — origin at the box centre
projected onto the table, z along the table normal, x along the principal horizontal axis.
Freezing is not cosmetic: PCA's horizontal axis on a rotationally symmetric object is decided
by noise, so recomputing it later would rotate the azimuth origin underneath the coverage map
mid-run.

**Interface for this part:**

```
reset(static_world)              # from S0: table plane, workspace box, fixtures
estimate(view) -> candidates     # cluster; return the detection map
commit(object_id)                # freeze the object frame, build the envelope
state()                          # object frame, observed box, safety envelope
collision_world()                # primitives handed to the IK filter
```

**Deferred to the refinement design:** how points accumulate across views, how the envelope
tightens, and how observation directions are tracked for coverage. The association rule itself
is settled — seeded connectivity, above.

### S1 — process and subprocesses

```
┌─ S1 · BOOTSTRAP: IDENTIFY AND ESTIMATE ────────────────────────────────┐
│                                                                        │
│  P1  CAPTURE                                                           │
│      move → bootstrap pose (declared, top-down, whole table in view)   │
│      settle + dwell ~100 ms      ← never transform depth while moving  │
│      grab: RGB, IR stereo pair, joint angles q                         │
│            │                                                           │
│            ▼                                                           │
│  P2  DEPTH                                                             │
│      Fast-FoundationStereo(IR_L, IR_R) → dense metric depth            │
│      ✗ fails → footprint-only fallback, flagged, no close approach     │
│            │                                                           │
│            ▼                                                           │
│  P3  SCENE POINTS                                                      │
│      deproject(depth, K) → camera frame                                │
│      × T_base_cam = FK(q) · T_ee_cam → base frame                      │
│      − arm's own points (FK)     − static obstacles (S0)               │
│      ∩ workspace box             − table plane                         │
│      = FREE-STANDING POINTS                                            │
│            │                                                           │
│            ├───────────────────────────┐                               │
│            ▼                           ▼                               │
│  P4  OBSTACLE MAP            P5  CANDIDATES                            │
│      voxelize ALL                segment RGB → masks                   │
│      free-standing points        keep only masks reaching the table    │
│      no semantics                dilate  (errors round outward)        │
│      → occupancy                 per mask: crop, position, size        │
│            │                     → DETECTION MAP                       │
│            │                     ✗ empty → "I don't see an object"     │
│            │                           │                               │
│            │                           ▼                               │
│            │                 P6  SELECTION                             │
│            │                     question + frame + detection map      │
│            │                                    → model → integer id   │
│            │                     ✗ no referent, N>1 → report ambiguity │
│            │                           │                               │
│            │                           ▼                               │
│            │                 P7  OBJECT SEED                           │
│            │                     points inside the chosen mask         │
│            │                     = known-correct seed set              │
│            │                           │                               │
│            │                           ▼                               │
│            │                 P8  OBJECT GEOMETRY                       │
│            │                     seeded ε-connectivity growth          │
│            │                       over free-standing points           │
│            │                     → object point set                    │
│            │                     → observed OBB   (under-reports)      │
│            │                     → safety envelope = OBB + occluded    │
│            │                         fill, floored by the table plane  │
│            │                     → object frame, FROZEN                │
│            │                           │                               │
│            └─────────────┬─────────────┘                               │
│                          ▼                                             │
│  P9  COMMIT                                                            │
│      object points leave the hard-obstacle set — the target cannot     │
│        be a hard obstacle or it could never be approached; the         │
│        standoff floor replaces it                                      │
└────────────────────────────────────────────────────────────────────────┘
             │                    │                     │
             ▼                    ▼                     ▼
      GEOMETRY              EVIDENCE              RUN STATE
      object frame          keyframe 0:           chosen id
      observed OBB            RGB, T_base_cam     model's stated reason
      safety envelope         depth, mask         question
      obstacle occupancy
      table plane
```

**Three terminal exits**, none of them faults: no candidates → *"I do not see an object"*; no
referent with several candidates → ambiguity reported; depth unusable → degraded footprint-only
mode with close approach disabled.

### D1 — Viewsphere and cell structure

Settled 2026-08-12. This is the shared vocabulary: the action space picks a cell, coverage is
over cells, evidence is keyed by cell, IK filters cells.

**In v1 a viewpoint is `{h, v}`** — horizontal and vertical only. It becomes `{h, v, r, d}`
later, adding roll and depth as further coordinates. The addressing scheme does not change when
they arrive; the tuple just gets longer.

| | |
|---|---|
| **Centre** | Object centroid on the table plane; tracks the centroid as the box refines |
| **`h` — horizontal** | Azimuth, 12 bins × 30°, `h ∈ 0…11`. **`h = 0` is the side facing the robot base.** |
| **`v` — vertical** | Elevation, 3 bins at ~20°, 45°, 70°, `v ∈ 0…2`. Top-down cells are ordinary cells, not a special case. |
| **`r` — roll** | v1: fixed by convention — camera-up aligned with the table normal projected into the image plane, so every capture is upright. Not an action. |
| **`d` — depth** | v1: one shell. `R = D_max / (1.108 · f)`, where `D_max` is the object's largest horizontal extent and `f ≈ 0.45` the fraction of frame height it should fill; `1.108` comes from the D405's 58° vertical FOV. Clamped below by min-Z plus standoff. An 80 mm cup gives **R ≈ 160 mm**. |

**Why `h = 0` is robot-facing rather than object-derived.** The natural alternative — the
object's principal horizontal axis from PCA — is decided by noise on a rotationally symmetric
object, so `h = 0` would land somewhere different every run and the coverage map would rotate
underneath itself. The robot direction is always defined, identical across runs, independent of
object shape, and makes `h = 0` the most reachable side.

**Radius is derived, not fixed**, which is what keeps this general: a 30 mm bolt and a 300 mm
bottle each get a shell where the object fills the same fraction of the frame.

**Known and accepted for v1: cells are not equal-area.** Like lines of longitude, neighbouring
`h` cells are far apart at low elevation and nearly coincident near the top. Two consequences,
both accepted rather than fixed: a one-cell horizontal step is a large move low down and a small
one high up; and counting coverage as *cells seen ÷ total cells* over-credits the top, which is
chopped into many small cells. The fix, if it bites, is to weight each cell by how much object
surface it actually sees instead of counting cells. Near-duplicate cells around the top are
harmless as long as nothing forces the agent to visit them all.

**Rejected: equal-area sphere tilings** (Fibonacci, icosphere). They fix the skew but have no
clean left/right/up/down neighbour structure, and the egocentric action vocabulary — which the
frame research makes mandatory — depends on "move right" meaning `h + 1`.

### D2 — Candidate generation, IK, collision filter

No model anywhere in this stage. It turns the 36 cells into a menu of moves the robot is
guaranteed to be able to execute.

1. **Cell → camera pose.** Each `{h, v}` gives a camera position on the sphere and a look-at
   orientation, with roll fixed by convention.
2. **Camera pose → wrist pose**, via the hand-eye transform.
3. **Wrist pose → joint angles.** Inverse kinematics returns **8 postures** for the same pose —
   shoulder left/right, elbow up/down, wrist flipped or not. All eight put the camera in exactly
   the same place; they differ only in how the arm folds to get there.
4. **Discard bad postures.** Reject on joint limits, and on collision of any arm link with the
   table, the object's safety envelope, the fixtures, or another part of the arm.

**Each cell ends in one of two states:**

| State | Meaning | Shown as an action? | Shown in coverage? |
|---|---|---|---|
| **Reachable** | ≥1 posture survived; the surviving posture is recorded for execution | Yes | Yes |
| **Infeasible** | All 8 postures collide or exceed limits | **No** | **Yes** |

That asymmetry is the point. Infeasible cells are never offered as moves, so the model cannot
pick something that fails — which is what prevents the retry loop the agent research measured.
But they must still be *visible as facts*, or a coverage-gated negative answer waits forever for
a view that can never happen, which is F2 directly.

**The menu is rebuilt every step**, because the safety envelope changes as geometry refines. At
16 ms for 64 viewpoints this is free.

**Pose feasibility and path feasibility are separate checks.** IK proves a posture exists at the
cell; it says nothing about whether a safe path reaches it from where the arm is now. Path
validation happens at move time, with a radial retract waypoint inserted when a direct move
fails.

**Measured on this pipeline:** 64 viewpoints in **16 ms**, all 64 with at least one safe
posture, 31 postures rejected against the table and 3 on self-collision.

**Noted as possible, not designed for:** if an object is placed such that too few cells are
reachable — all on one side, say — the system cannot honestly support a negative answer about
the unseen side. Considered unlikely in the demo cell; revisit only if it occurs.

### D3 — Orchestration shape

Settled 2026-08-12. **Agentic within a step, orchestrated across steps.**

The split is by action cost, not by taste. Tools that only *read* state — `crop`, `recall`,
`coverage`, `describe` — are freely callable and chainable; they are reversible and cost
nothing. `look(h, v)` moves the arm, and **calling it ends the turn**. The orchestrator owns the
step counter, the budget, and the termination gate.

**Loop body: read → decide → (move + capture).** Starting at the read step makes S1 simply
iteration zero's capture, so every iteration is identical and bootstrap needs no special case.
It also means the answer path is live from iteration zero — a question answerable from the
bootstrap view alone does not require a move.

```
S1 bootstrap capture ──┐
                       ▼
   ┌──►  READ    agent looks at the newest image, writes a comment on it
   │       │
   │       ▼
   │     DECIDE  context assembled: question, coverage, feasible menu,
   │       │     prior comments, retrieved crops
   │       │     free tools: crop / recall / coverage / describe
   │       │
   │    ┌──┴────────────┐
   │    ▼               ▼
   │  ANSWER      LOOK(h, v)  ── ends the turn
   │    │               │
   │    │               ▼
   │    │      ORCHESTRATOR: validate path → move → settle → capture
   │    │               │
   │    │               ▼
   │    │      GEOMETRY + COVERAGE UPDATED
   │    │               │
   │    └───────────────┴──► back to READ
   ▼
 ANSWER + supporting keyframes
```

**Why not fully agentic.** Termination would become the model's decision, and termination is
this project's dependent variable — handing it over means we cannot measure the F2/F4 dial we
set out to study. The supporting evidence is about agent loops generally rather than robots:
31.3% of AgentBench episodes hit the step limit without finishing (GPT-4 still 23.9% on its
strongest environment); WebArena places its repeat-action detector in the orchestrator by
design; failed agentic runs cost roughly 2× successful ones because the failure mode is "keep
trying"; and out-of-policy tool calls ran 9–43% across models, every one caught by an external
validator — so the validator gets built either way.

**Why not fully orchestrated.** The user story — spot a logo edge, crop to check, recall a
neighbouring view, judge the offered step too coarse, then move — would have to be
pre-anticipated as schema fields instead of emerging.

**The read step earns its place twice.** It separates perceiving new evidence from deciding what
to do, so each call runs in its strong regime; and the comment it writes is the per-keyframe
text the retrieval layer needs, generated as a by-product rather than by a separate captioning
pass.

**Deferred to D8:** how the agent's context is assembled — single continuous conversation,
rebuilt-per-turn from the stores, or a reader/decider split with separate contexts. The
self-conditioning evidence argues against the first, but the decision waits until the stores it
would draw from are designed.

### D4 — Action space

Settled 2026-08-12. **Movement is absolute: `look(h, v)`.** Relative behaviour — "that was too
far right, come back two" — emerges across turns rather than being encoded in the action.

Absolute is idempotent, accumulates no drift when a move partially fails, and validates
trivially against the feasible set.

**The requirement absolute addressing creates.** With `h = 0` pinned to the robot base,
`look(7, 1)` is an *allocentric* address, and allocentric is the 2% regime. That measurement is
about *inferring* spatial relations from an image rather than picking from a labelled list, so
it does not sink the choice — but it does mean **the model must never have to compute the
address from the picture.** The menu therefore carries an egocentric gloss beside every absolute
ID:

```
you are at h=5, v=1
  h=6, v=1   one step right, same height
  h=7, v=1   two steps right, same height
  h=5, v=2   same side, higher
  h=3, v=1   two steps left, same height        [visited, step 2]
  h=9, v=0   opposite side, lower               [unreachable]
```

Absolute for the system, egocentric for the model. Marking visited cells in the menu also means
the agent *reads* where it has been instead of reconstructing it from the transcript, which is
the difference between the sturdy and the fragile version of overshoot-correct.

**Turn-ending actions — these are the whole set:**

| Action | Effect |
|---|---|
| `look(h, v)` | Move and capture. Ends the turn. Rejected if the cell is not in the feasible set. |
| `answer(text, supporting_cells)` | Terminates the run. Gated — coverage for negatives, patch re-observation for positives. |

**Injected by the orchestrator every turn, not exposed as tools:** the coverage state (seen,
unseen, permanently unreachable), the feasible menu with glosses, and the current cell. These
are state the agent always needs, so making it ask for them wastes a call and risks it not
asking.

**Deferred to the image-indexing design:** the read-only exploration toolset — recall, crop and
whatever navigation primitives the evidence structure supports. That family is defined by the
store it queries, so it is designed with the store rather than ahead of it.

**Oscillation guard.** Overshoot-correct settles only if corrections shrink. Two cheap
defences, to be validated in testing: cap a correction at the size of the previous move, and
keep visited cells visible so convergence is a map-reading task rather than a memory task.

**Not shipped, pending evidence: an `undetermined()` exit.** When a whole side is unreachable or
depth has failed, the agent's only options are to assert something it cannot support (F3) or to
look until the budget runs out (F2), and a third exit would turn that into a correct response.
Held back because escape hatches get over-used — GPT-4 wrongly declared **54.9%** of feasible
WebArena tasks impossible, so a cheap way out could collapse the answer rate. Add it only if
experiments show it is needed, and gate it like a negative answer: available only when coverage
actually reports unreachable cells.

### Stage chain

Nine stages. Six are ordinary engineering; the project lives or dies on S2, S7 and S8.

| | Stage | Mechanism | Risk |
|---|---|---|---|
| **S0** | Static world | See *S0 — cell preparation* above. Design deferred; contract fixed. | Routine, but hand-eye is the accuracy floor for everything downstream |
| **S1** | Bootstrap sighting | See *S1 — bootstrap sighting* above. Top-down pose → learned stereo depth → plane removal → cluster → OBB, plus the first keyframe. | **Risky** — rests on Fast-FoundationStereo working on glossy surfaces |
| **S2** | Envelope refinement | **Design deferred.** Later views tighten the envelope toward true extent | Unassessed until designed |
| **S3** | Collision envelope | Box as broad-phase reject; **sphere set or 5 mm voxel SDF as narrow phase** — a box cannot represent a handle | Routine given S2 |
| **S4** | Candidate viewpoints | Object-anchored azimuth–elevation grid, standoff from camera range and required pixels-per-mm; all 8 IK branches; collision-filtered before the model sees them | Routine |
| **S5** | Motion | `moveJ` to a chosen branch, path validated, settle, capture | Routine |
| **S6** | Evidence store | Keyframes `(image, pose, depth, readout)`; coverage as a bitmap over viewsphere cells | Routine |
| **S7** | **The decision** | Given question + retrieved evidence + geometric state: emit next view or answer | **This is the research** |
| **S8** | **Termination** | Coverage gate + agreement + budget | **Hardest — the F2/F4 dial** |
| **S9** | Answer with justification | Which views support the conclusion | Routine |

### Decided mechanisms

**Geometry ranks, the model re-weights, geometry vetoes.** Enumerate ~36 candidates
geometrically, filter by IK and collision, show the model a shortlist of **6**. If its pick
scores below half the geometric best, override and log it. Evidence: VLM-only 37.82, pure
geometry 36.93, combined 39.54 — neither alone is much good. More options make it worse,
not better: candidate markers crowd and occlude.

**Centroid-lock the aim (Anton, 2026-08-12).** Every viewpoint looks at the object centroid,
which reduces the viewpoint space to a clean viewsphere — azimuth, elevation, radius. The
centroid is allowed to shift as the box refines.

The published evidence cuts the other way and is recorded here so the trade is explicit:
forcing viewpoints to aim at the object centre made every planner in Zaenker et al.
(arXiv 2306.09801) worse — semantic NBV −9.8%, volumetric −11.3%, predefined wide −10.6%.
That result comes from fruit occluded in foliage, where free aim lets the camera peek past
occluders; a single isolated object on a table is a weaker case for it.

What centroid-locking buys is structural, and the rest of the loop design depends on it: the
action space stays small and enumerable, coverage becomes a clean partition of the sphere,
and *"a bit to the right"* maps to a step in azimuth instead of being ambiguous between moving
and re-aiming. Paying up to ~10% of planner efficiency for that is a reasonable PoC trade.

**Escape hatch if it binds:** keep aim locked for *navigation* — the sphere and the coverage
map stay defined by the centroid — and add ROI retargeting as a separate refinement action
rather than a free parameter on every pose.

**Nested shells, not one sphere (Anton, 2026-08-12).** The viewsphere has discrete zoom
levels — concentric shells at different radii. "Move closer" becomes "drop one shell", which
keeps radius quantized and every output a discrete choice. **Ship one shell first, add levels
after the loop works.**

The shells are not arbitrary: each corresponds to a resolution band, and the D405's usable
window bounds how many exist. Its field is roughly `1.9 × distance` wide, and min-Z is 7 cm,
so for an ~80 mm object:

| Shell | Radius | Object width in a 1280 px frame | Role |
|---|---|---|---|
| Survey | ~35 cm | ~170 px | Locate, gross shape, where to look |
| Inspect | ~18 cm | ~330 px | Normal reading distance |
| Detail | ~9 cm | ~660 px | Fine text and surface defects |

Three levels, not ten. Two consequences follow. **Coverage stops being one bitmap** — a cell
seen from the survey shell is not "covered" for a detail question, so coverage is per-level,
and "I have seen the whole cup" and "I have inspected the whole cup" become different claims.
That distinction is load-bearing for the stopping rule. And **inner shells are sparser after
filtering**, since closer poses collide and strain reach more often.

Convenient coincidence worth using: the camera's 7 cm min-Z means it cannot focus closer than
that anyway, so the optical limit and the safety standoff floor are the same number.

**Positive and negative answers need different evidence — but neither is cheap.** "No logo"
is a claim about the whole surface and is gated on coverage. "Yes, there is a logo" is gated
on **re-observation of a named patch**: the positive claim must say *which* cell and *which*
crop, and that specific patch must be re-observed from a different viewpoint and still read
positive.

The earlier version of this rule let a positive stop on one unambiguous sighting. That was
backwards. POPE (arXiv 2305.10355) measures VLM yes-bias on existence questions at
95.63–98.67% for mPLUG-Owl, 98.77–99.37% for LLaVA and **99.90–100.00% for MultiModal-GPT**;
even well-behaved InstructBLIP drifts 56.57% → 73.03% yes as negatives get harder while
accuracy collapses 88.57% → 72.10%. Attaching the cheap branch to the model's dominant failure
direction means a confabulated logo terminates the run in one step.

So the two branches get symmetric protection from different mechanisms: **coverage protects
negatives, patch re-observation protects positives.** Nothing else in the stack defends the
positive branch.

**Never stop on one view, positives included.** Agreement from a single read scores at chance.
Minimum two views, and the answer must survive a viewpoint change.

**Coverage must distinguish `UNSEEN` from `PERMANENTLY_OCCLUDED`.** A cup's underside and its
table-contact annulus can never be observed without manipulation, which is a non-goal. If those
cells stay merely "unseen", a coverage-gated negative answer can never be satisfied and the run
walks straight into F2. Because the candidate set is enumerated and IK-filtered up front, this
is exactly computable — a cell is permanently occluded when no feasible, collision-free
viewpoint in the set can see it. Voxel-gain systems infer unreachability indirectly from gain
collapse; the discrete enumeration gives it exactly, which is a genuine advantage of this
architecture and worth writing up.

**S7 emits an `expect:` field** — what the model believes the chosen view will show. Checked
afterwards against what was captured. One schema line; converts F1 into a measurable number.

Its value is as **scaffolding and metric, not as a prediction**. MultiView-Bench
(arXiv 2607.08970) measures exactly this geometry — asking a VLM what would be visible from a
shifted viewpoint — at **8–31% against 81–97% human**. Predicting-before-acting still pays as
scaffolding: ECoT gains **+28% absolute** over OpenVLA with no verification loop at all.

**Do not re-inject the verdict verbatim.** Contextual Drag (arXiv 2602.04288), 11 models across
8 tasks, finds failed attempts in context bias later generations toward structurally similar
errors, costing 10–20%, and states that neither external feedback nor successful
self-verification removes the effect. Distil the mismatch into a running rule or a scalar
surprise instead — WALL-E's shape, worth +15–30% success and 8–20 fewer replanning rounds.

**Never mosaic keyframes into a contact sheet.** Ten separate images cost GPT-4o almost nothing
(97.0%); the same content stitched into a 4×4 grid collapses to **26.9%**, Gemini to 6.09%,
Claude to 0.4%. Image *count* is survivable, sub-image *density* is not.

**Order the schema reasoning-first, and omit a confidence field.** Answer-before-reasoning
erases the entire chain-of-thought gain (14.3% → 6.1%), and JSON mode causes it silently — 100%
of GPT-3.5 responses emitted `answer` before `reason`. Dropping the reasoning field costs far
more in embodied tasks than in QA (ReAct on ALFWorld: 71 → 45). Verbalized confidence is
worthless at AUROC 51.2, chance; sampling at M=5 gives AUROC 92.7 instead.

**Failure feedback is categorical, not verbatim.** LLM³ (arXiv 2403.11552) returns motion-planner
failures as *"goal configuration in collision with object X"*, *"no feasible IK solution"*,
*"collision-free and reachable"* — raising success 40% → 60% while *cutting* retries from 15.1
to 11.4 LLM calls. Removing the explanation and keeping the raw signal drops correction success
79.1% → 41.9%. And the strongest anti-looping measure is pruning the action set, not instructing
against repetition: AgentOccam lifts WebArena 16.5% → 25.8% by removing distractor actions
alone, 43.1% with history filtering.

**Crop before reading.** A 40×40 px logo in a full frame survives downscaling as ~15×15 px —
under one visual token. Cropped to 600×600 it becomes ~460 tokens. On fine-detail
benchmarks every model scores under 20% one-shot on the full image; with crop-and-zoom the
best reaches 56%, and blacking the crop collapses it to 12%. The ROI crop is not an
optimisation, it is the difference between the demo working and not.

**Geometry reaches the model as text, never as pictures.** Models lose accuracy when handed
depth maps, and a bird's-eye view scored *below* a blind baseline. Images and text hints
together far exceed either alone (13.6% text-only, 45.7% images-only, 63.5% both).

**Retrieve, don't accumulate.** Going from one image to two costs ~6 points; a hundred
images costs thirty. Pull 1–2 relevant keyframes on demand from the indexed store.

### Model interface

The reasoning model sits behind **one swappable interface**, and candidates are compared
empirically rather than chosen up front.

- **Cloud** — `gemini-robotics-er-2-preview`, ~$0.16 per episode, native pointing, strongest
  reader. Exposes no token logprobs, so confidence comes from N-sample agreement.
- **Local** — Qwen3-VL-8B on vLLM. The only route to token-level answer entropy. Note
  **not** Cosmos-Reason: NVIDIA's own table places it below its own base model on the two
  spatial-grounding benchmarks that matter here (BlinkSpatial 84.62 vs 87.41, Where2Place
  50.0 vs 53.0).

### Executor

Inverse kinematics over the candidate viewsphere, not a learned policy.

A camera look-at is a rank-2 orientation constraint — position plus aim direction, with
roll about the optical axis irrelevant. This is exactly solvable, deterministic and
debuggable. A VLA would be a stochastic, data-hungry way to compute something IK solves
in a millisecond.

**Decided (Anton, 2026-08-12).** A measured 64-viewpoint pipeline — look-at → hand-eye →
8-branch IK → collision filter — runs in **16 ms** and found a safe branch for **64 of 64
viewpoints**, rejecting 31 branches on the table and 3 on self-collision. Evidence behind
choosing this over a VLA:

- No released VLA can drive a UR5e out of the box.

  Two separate things get conflated here, so stating them apart. **Seen during
  pretraining** is a claim about the base weights. **Released as a runnable policy** is a
  claim about what you can load and execute. They are not the same, and π0 has the first
  without the second.

  - **π0** — its pretraining explicitly names a UR5e with wrist and over-the-shoulder
    cameras, matching this rig, so the base model carries UR5e-shaped priors. But openpi
    ships post-trained experts only for DROID (Franka), ALOHA and LIBERO;
    `src/openpi/training/config.py` contains no `ur5` or `universal` entry, and
    `examples/ur5/` is documentation rather than a loadable config. Running it here needs
    an observation/action contract — camera mapping, state and action dimensions, and
    normalisation statistics, which are computed per-dataset and do not transfer. That
    contract comes from post-training on data from this setup. Pretraining makes that
    finetune cheaper and likelier to work; it does not remove it.
  - **openpi on this machine** — additionally pinned to jax 0.5.3 and failing on Blackwell
    (`ptxas does not support CC 12.0`, open issue #682).
  - **GR00T** — installs cleanly on the 5090, but has no UR data in any pretraining mix,
    making a UR5e strictly `NEW_EMBODIMENT`, and its finetune footprint (~35 GB) exceeds
    the 32 GB card.
  - **MolmoAct2** — genuinely zero-shot, but only on SO-100/101.

- Cloud compute removes the install barrier but not the embodiment barrier. The cost of
  the VLA path is data, not compute.
- **Untested, not ruled out:** running the pi0_droid (Franka, 7-DoF) checkpoint on the
  UR5e with a remapped action space. Both are 6/7-DOF arms with parallel grippers, so
  partial transfer is plausible. No published result either way. Cheap to try, not
  something to plan around.
- AP-VLM (UR5) does this task on UR hardware with classical motion, not a VLA. A second
  claimed example, VAP-TAMP on a UR5e, did not survive verification — see the retraction under
  *What the Research Settled*.

### Phase two — the IK system as a data engine

The VLA is not abandoned, it is sequenced. Once the IK loop works, every episode it runs
produces exactly the supervision a view-selection policy needs: scene, question, candidate
set, chosen view, and whether that view resolved the answer.

The evidence says this is where the performance actually is. Three independent 2025–26
results show a **distilled or fine-tuned 7B beating every frontier model at viewpoint
selection** — 83.72 vs GPT-5's 72.09 in one, 47.8% vs 18.5% in another, and EyeVLA reaching
96% task completion from **500 training samples**. Frontier models meanwhile sit barely
above a random-action baseline at the same task.

So phase one ships a working inspector; phase two distills it into the self-contained model
the product vision ultimately wants. 500 samples is a number of episodes, not a research
programme.

### Collision handling

**The inspected object must not be a hard obstacle** — if it is, it cannot be approached by
construction. This is what every published system does, and MoveIt has a sanctioned API for
it (`excludeWorldObjectsFromOctree`) that strips the target's depth points from the octomap
and substitutes a clean primitive. The problem decomposes:

- **Camera standoff** — guaranteed by the viewsphere radius plus a hard software floor.
  Geometry, not collision detection.
- **Arm links versus table, object and self** — the real risk, and what NBV papers skip.
  Pinocchio + coal, measured at **21 µs per check**, 0.43 ms for a 40-waypoint path.
- **Any voxel map** — information gain only, never a hard constraint.

**Never `moveL` between viewpoints.** A Cartesian straight line between two points on a
sphere is a chord — it passes inside the sphere and aims the camera through the object. Use
`moveJ` to an explicitly chosen IK branch; branch choice is what makes collision filtering
recoverable. When a direct move fails validation, insert a radial retract waypoint.

**Safety: the robot will not save the camera.** A cup is knocked over at forces far below
any protective-stop threshold, and the D405 plus bracket is the fragile element, not the
arm. Protection comes from the standoff floor, low speed, a compliant mount, and PolyScope
safety planes — which are enforced in the safety controller, below application code.

## Test Harness

Four tiers, cheapest first. The model half and the geometry half are validated separately and
integrated last.

**Tier 0 — text only, no images.** Synthetic run states: twelve object sides, some carrying
comments, some unknown, one unreachable. Ask the model to pick the next side. Isolates map
format and navigation reasoning from readout quality entirely, and settles the D5/D6/D8 format
question without a single image. Hours to build.

**Tier 1 — replay on ABO spins.** Amazon Berkeley Objects ships **72 real turntable photos per
product at 5° azimuth spacing for 8,200+ products**, CC BY 4.0 — subsampled to our 12 cells at
30°. `look(h, v)` returns a stored photo instead of moving an arm, so the whole model half runs
with no robot.

This tier matters most because of what it makes cheap. **F4 requires an oracle continuation** —
keep looking after the system committed and check whether the answer flips — which is expensive
on hardware and a loop here, since every view is already held. F1 is equally direct: check
whether the chosen view actually contained the region of interest.

**Tier 2 — rendered views from ABO meshes.** The same dataset provides **glTF models for 7,900+
products**, so the objects with real spin photos also have geometry. Turntables are a single
elevation, so `v` is untestable on Tier 1; rendering fills that gap with a full `h × v` grid and
ground truth by construction. A plain rasteriser — view generation, not physics.

**Tier 3 — real arm.** The geometry half, then integration.

**Fidelity ordering, stated so the numbers are not oversold.** ABO photos are studio product
shots — clean, evenly lit, white background — so Tier 1 reads optimistically against a D405 on a
table. Tier 2's synthetic texture is easier still. Both are development and ablation
instruments; **reported results come from Tier 3.**

**Ground truth** is the one piece needing construction: which sides actually show the feature
being asked about. Cheapest defensible route is pseudo-labelling — run a strong model over all
72 views at full resolution, then hand-check a sample to measure the labeller's own error rate
and report it.

**Simulation stays out.** Isaac was ruled out earlier on three counts: 3DGS loses fine text
legibility, splats degrade at exactly the close range this system operates in, and Isaac Lab
ships no UR5e configuration, so it is roughly four days before anything renders. Tier 2 delivers
what simulation would have delivered, in half a day.

## Open Questions — The Interactive Loop

Everything through bootstrap is settled. The loop is not. These are the questions to research,
in dependency order.

### The story the design has to support

The agent has four images. Part of a logo is visible at one edge. It moves right — say four
cells — captures, and now judges that it went too far. So it moves two cells back to the left,
and converges on the view it wanted.

Three things are latent in that story:

- The instruction is **relative and egocentric** — "move right", not "azimuth 47°"
- The step carries a **signed magnitude in whole cells**, not a metric angle
- **Overshooting is normal and recoverable.** Correction over several steps is the intended
  behaviour, not a planner failure

Note what this does *not* require: no sub-grid resolution, no request for an intermediate
viewpoint, no change to the 30° grid. It is ordinary movement plus feedback.

The real risk is not resolution, it is **convergence**. Overshoot-and-correct is proportional
control with a sloppy gain: it settles if the sign is right and each correction is smaller than
the error. Sign is what models are measurably good at — 78% in the camera frame. Magnitude is
what they are bad at, with only 37% of metric distance estimates landing within a factor of two.
Magnitude error merely slows convergence; **systematic overcorrection causes oscillation**
(+4, −3, +2, −3, …), which is an F2 non-termination path. Guarding that — monotonically
shrinking steps, and making visited cells legible on a map rather than recalled from the
transcript — is the substance of Q1.

### The questions

**Q0 — Orchestration shape.** Is the model an agent driving tools, or a decision function our
loop calls? *Everything else is shaped by this* — coverage becomes a tool response or a prompt
block, retrieval becomes a call or a pre-assembly. Settling it late means redoing the rest.
Blocks all.

**Q1 — Action space.** What the model is allowed to say. A viewpoint has three parts — where to
stand and how close — aim is centroid-locked, so the space is a viewsphere in azimuth,
elevation and radius. Standoff sets pixels-per-mm, so "move closer" is a different action from
"orbit right". Plus the non-motion actions: answer, ask for finer granularity, request a crop.
*Breaks if too coarse (the story becomes impossible) or continuous (the model cannot perform).*
Blocked by Q0.

**Q2 — Coverage representation.** What counts as seen, what counts as unseen, and how that
reaches the model. *This is the only signal in the stack that a confidently-confabulating model
cannot corrupt, and the primary defence against premature stopping on negative answers.*
Blocked by Q1 — coverage is defined over the action space.

**Q3 — Context assembly.** Exactly what is in the model's context at step k, including whether
it sees its own reasoning from step k−1 and how the `expect:` check re-enters. *Breaks if
unbounded: 1→2 images costs ~6 points, 100 images costs 30, degrading the readout everything
else depends on.* Blocked by Q2 and Q5.

**Q4 — Evidence store and index.** How keyframes are written and keyed. **Two keys, not one** —
by pose for spatial recall, by content for answering. Researchable in parallel.

**Q5 — Retrieval for deciding.** How *"what have I not seen, what is still ambiguous"* is
answered from the store.

**Q6 — Retrieval for answering.** How *"show me everything bearing on this claim"* is answered.
Same store, different query, probably a different mechanism.

**Q7 — Geometry refinement.** How the point set, box and envelope update per view and stay
plan-worthy. Already settled: seeded connectivity for membership, and the object frame freezes
its orientation while the origin tracks the centroid. Open: accumulation, envelope tightening,
observation-direction tracking, and whether to require two-viewpoint confirmation per voxel.

**Q8 — Stopping.** When the loop commits. *This is the F2/F4 dial and the main research object.*
Already settled: positive and negative answers stop on different criteria; never stop on one
view. Blocked by Q2 and Q3.

**Q9 — In-loop failure handling.** What happens when the pick is infeasible after IK, the move
fails, the capture is blurred, or the readout is unusable — and whether a failed attempt
consumes budget. *Unaccounted retries are a direct F2 path.*

## What the Research Settled

Two research passes on 2026-08-12. Full detail in agent memory under `perception/` and
`learning/`.

**Viewpoint selection must be discrete.** Every working system converts the model's
output into a symbol over a pre-enumerated viewpoint set — a grid vertex, a numbered
view, a direction word. No published system asks a VLM for a metric 6-DOF pose, because
they cannot produce one: GPT-5 scores 0.66 F1 on binary relative camera pose direction
(chance 0.50, humans 0.91) and 0.46 on roll, below chance. Only 37.2% of VLM metric
distance answers land within a factor of two. EyeVLA reports that replacing discretised
action tokens with continuous pose regression produced "near-zero task completion".

**Active view selection works, when the readout is clean.** AP-VLM lifts an occluded-object
scene from 0.0 passive to 1.0.

*Retracted 2026-08-12:* an earlier draft credited VAP-TAMP with 88% success on a UR5e and
1.61 views per answer versus 4.35 greedy. Checking the source (arXiv 2604.26988, *Robot
Planning and Situation Handling with Active Perception*) shows it is a **situation-handling**
system — recovering from jammed doors and fallen objects during service tasks, on a **mobile
manipulation platform**. Neither the UR5e, the success rate, nor the view counts could be
verified in the abstract or the PDF. Two separate research relays also returned *different*
values (1.61 and 2.65) for the same quantity. Treat all VAP-TAMP numbers as unsourced.

**Stopping is the weak link everywhere.** No published hierarchical system has reliable
open-vocabulary success detection. False positives are the specific killer: 10% detector
error costs nothing, beyond that the system degrades sharply, while false negatives are
absorbed by re-checks. Models also answer visual questions without looking — 24.56% false
positives in embodied QA, and at or below a random-action baseline on object-attribute
questions in E3VS-Bench.

**Where a contribution is available.** Classical next-best-view reduces *geometric*
uncertainty. Inspection needs *answer* uncertainty. The formalism exists without a
language front-end or a robot; nobody has connected VLM answer entropy to a per-voxel
information gain. Separately, no tabletop active-inspection QA benchmark exists at all.

## Risks

**The null result.** PInVerify ran nearly this experiment without an arm and found
single-view 0.844, multi-view 0.849, and LLM-chosen views 0.850 — all within the
confidence interval. Their conclusion: any gain from smarter view selection is swamped by
single-step verification noise. If the per-view readout is noisy, better viewpoints will
not show up in the numbers. The evaluation must measure per-view evidence quality
separately, or the project produces a null result by construction.

**Perception, not planning, is the bottleneck.** GPT-4o reaches 74.9% on industrial
anomaly benchmarks that usually supply a reference image, which the benchmark authors
call far short of industrial requirements. On robot-captured data under uncontrolled
lighting, VLMs perform poorly at both classification and localisation.

**Depth is now load-bearing, and it rests on one component.** The geometry path is
depth-first, so Fast-FoundationStereo moved from optimisation to dependency. Its repo pins
torch 2.6 / cu124, which ships **no sm_120 kernels** — it needs torch ≥2.7 + cu128 and a
flash-attn built for sm_120, roughly half a day, with the NGC PyTorch container as fallback.
If it does not work on this machine the system drops to the silhouette-on-table-plane
fallback, which recovers footprint but not height, and close approach becomes unsafe. Prove
this early; it gates the geometry stage.

**Environment build-out on Blackwell.** sm_120 needs CUDA 12.8+ wheels; the cu126 index
installs cleanly then fails at runtime with "no kernel image available". LeRobot is hours,
GR00T is a day or two, openpi may not resolve at all.

**Hand-eye calibration does not exist yet** and UR5e absolute accuracy (~1–4 mm, versus
±0.03 mm repeatability) dominates the error budget regardless of solver. Expect a 0.4–0.6 mm
residual from a ChArUco eye-in-hand calibration; above 1 mm the calibration has failed.

**The factory DH extraction is an open problem.** UR's own documentation says that without
the arm's factory kinematic calibration, end-effector positions are off "in the magnitude of
centimetres" — which dwarfs every algorithmic choice downstream. The tool that extracts it,
`ur_calibration`, is a ROS 2 Jazzy binary, and the route to pull those parameters off the
controller from a pure-Python stack is unresolved. Resolve early; ask whether the colleague
who previously drove the arm already has the file.

**Calibrated DH breaks the analytic IK.** The UR5e closed form depends on joints 2–4 being
exactly parallel; the real calibrated parameters perturb `alpha2` to 0.00139 rad and the
analytic solver refuses outright. The fix is two-stage — enumerate all 8 branches on nominal
DH, then Newton-refine each against the calibrated model, which preserves the branch.

## Timeline

Demo by 2026-08-31. Nominal three weeks, not strict.

The first work item is environment build-out and a UR5e connectivity check, because
everything downstream depends on both and neither is proven on this machine.

## References

Load-bearing claims above are sourced in agent memory:

- `project/north-star-ai-driven-inspection`
- `project/adam-inspection-robot-and-stakeholders`
- `project/open-research-slots-active-inspection`
- `perception/vlm-viewpoint-selection-discretize-first`
- `perception/vlm-success-detection-and-active-view-limits`
- `learning/vla-embodiment-fit-ur5e-and-blackwell`
- `tooling/vbti-ubuntu-rtx5090-blackwell-sm120`
- `project/executor-is-ik-vla-is-phase-two`
- `perception/scene-state-for-tabletop-inspection`
- `perception/vlm-model-choice-and-roi-cropping-for-inspection`
- `perception/nvidia-stack-for-active-inspection`
- `manipulation/ur5e-analytic-ik-and-calibration-trap`
- `manipulation/collision-free-viewpoint-motion-ur5e`
- `hardware/d405-passive-stereo-no-ir-projector`
- `hardware/ur5e-d405-hand-eye-calibration-recipe`
- `simulation/skip-sim-use-replay-benchmark-for-inspection`
