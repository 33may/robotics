# inspection/ui — the frontend and its contract with the loop

The operator-facing UI for the inspection loop. Four panels, built on **porthole**
(`~/projects/porthole`), talking to Python over the porthole bus.

**This module publishes. It does not decide.** v1 has no command path at all:
the approval gate and the look/answer decision stay on the terminal
(`run/decider.py`). The UI is a monitor. That keeps every question about
authority answered by `run/`, where it already is.

**Nothing in `run/` imports from here.** The existing contract ("nothing imports
from run/") is preserved in both directions: `publisher.py` takes plain data —
arrays, dicts, tuples — and knows nothing about `Loop`, `Ctx` or `RealRig`.

---

## 1. The four panels

| Panel | Topic(s) | Source |
|---|---|---|
| **cell** — robot + workcell + object + viewsphere markers | `scene/description`, `scene/poses`, `cloud/fused` | porthole `scene-view`, copied |
| **camera** — live wrist stream | `camera/wrist` | porthole `camera`, copied |
| **cloud** — fused cloud, clickable view markers, image of the picked view | `cloud/fused`, `views/state` | `src/panels/CloudInspectPanel.tsx`, app-owned |
| **log** — event stream | `log/events` | porthole `event-log`, copied |

Copied panels are yours: edit them freely, `porthole update` three-way-merges
upstream fixes as long as `src/.porthole/base/` survives.

---

## 2. Topics

| Topic | QoS | Rate | Payload |
|---|---|---|---|
| `scene/description` | stream, retained | on change | the whole 3D world (§3) |
| `scene/poses` | stream | 30 Hz while moving | robot link placements (§4) |
| `cloud/fused` | stream, retained | per capture | `{points: $nd float32 [N,3]}`, base frame |
| `camera/wrist` | stream | ~10 Hz | `$img` JPEG |
| `views/state` | stream, retained | per turn | viewsphere cells (§5) |
| `run/status` | stream, retained | per phase change | `{phase, step, question, ...}` (§6) |
| `log/events` | event | as they happen | `{level, msg}` |

Every retained topic means a UI opened mid-run sees the current world
immediately — no replay, no "restart the backend so the window has something to
show".

**Publish state, not deltas.** Retention keeps the newest message per stream
topic, so a payload that only makes sense applied on top of the previous one
leaves a late client permanently wrong.

---

## 3. `scene/description` — what exists

Built by `publisher.py`; you never hand-write it. Node paths, which are also the
group names the panel's visibility toggles use:

```
world/axes            base-frame triad
cell/<box>            every box in cell.yaml, coloured by the viewer's STYLE table
robot/<link>          7 UR5e visual meshes, served over HTTP (§7)
tool/<box>            RG2 envelope + cable loop, welded to the flange
object/obb            the reconstructed object primitive (world.set_object)
object/cloud          points node -> streams from `cloud/fused`
views/h<HH>v<V>       one marker sphere per viewsphere cell
```

The whole set is republished whenever anything changes — a marker turning green
resends the description. That is the protocol (`porthole/docs/scene-protocol.md`
§3), and it is cheap because the bulk data is not in here: the cloud is a
`points` node that names `cloud/fused` rather than carrying it.

### Marker colours

| State | Colour | Meaning |
|---|---|---|
| `visited` | `#7ee2a8` green | captured from here |
| `current` | `#8ab4f8` blue | where the camera is now |
| `available` | `#6d737d` grey | reachable, not yet visited |
| `blocked` | `#f2c76b` amber | planning refused it this round (`plan_failed`) |
| `unreachable` | `#5a3a3a` dim red | no free IK branch at any roll |

`blocked` and `unreachable` are deliberately different colours: one is
"try again after the arm moves", the other is "this cell is not addressable from
this object position". Collapsing them hides the fact that `plan_failed` is
cleared after every real move.

---

## 4. `scene/poses` — where things are

```python
pose_frame({"robot/base_link_inertia_0": T, ...}, overrides=..., background=...)
```

Absolute row-major 4×4 per node, in the base frame. Produced by
`pin.updateGeometryPlacements` → `geom_data.oMg[gid]`, which is exactly what
`MeshcatVisualizer.display()` already pushes today — the port is mechanical.

`publisher.replay_path(path)` is the port of `RobotCell.replay()`: discretize at
the same `DEFAULT_STEP`, publish a frame per config at 30 Hz, tint the tool red
and the background red at the first colliding config.

**Freeze-at-impact is preserved.** `replay()` today stops at the first invalid
config and leaves the viewer red. `replay_path` does the same: it stops
publishing new poses and leaves the retained frame showing the collision, so a
UI that connects afterwards still sees why the path was refused.

No kinematics run in the browser. Python does FK, the browser draws matrices.

---

## 5. `views/state` — the viewsphere as data

The 3D markers are scene nodes, but the cloud panel needs them as *data* it can
click, so the same state is published in a structured form. One producer, two
representations — `publisher.publish_views()` emits both from one call, so they
cannot disagree.

```python
{
  "center": [x, y, z],           # fused-cloud centroid, base frame
  "radius": 0.35,
  "current": [h, v] | None,
  "cells": [
    {"h": 3, "v": 1,
     "state": "visited",          # visited|current|available|blocked|unreachable
     "pos": [x, y, z],            # camera position of the cell, base frame
     "gloss": "one step right, same height",
     "step": 4,                   # turn that visited it, else absent
     "image": "/captures/004/rgb.png",   # present iff visited
     "comment": "handle faces away"      # the READ text, if any
    },
    ...
  ]
}
```

`image` is a URL on the UI's own server, not a filesystem path — see §7.
The panel shows the image when a `visited` marker is clicked, and the text
"not visited" when any other state is clicked. Clicking sends nothing.

---

## 6. `run/status` — the strip everything else reads

```python
{"phase": "idle|planning|preview|awaiting_approval|executing|capturing|fusing|done",
 "step": 4, "max_turns": 12,
 "question": "is there a logo on this cup?",
 "visited": 4, "reachable": 22, "total": 36,
 "detail": "tier 2, 42 waypoints"}
```

`phase` exists so the UI can say what the robot is doing without inferring it
from message timing. `awaiting_approval` is the one that matters: the terminal
owns the y/n, and the window should not look frozen while it waits.

---

## 7. Files that are not payloads

Captured images and robot meshes are served over HTTP by the same server that
serves the UI, never over the bus:

```python
serve_ui(dist, assets={
    "/captures": run_outdir,                  # NNN/rgb.png per capture
    "/meshes":   ur_description_visual_dir,   # 9 MB of DAE
})
```

An 848×480 PNG per view, twelve views, is not bus traffic — and the browser
already caches, streams and decodes them off the main thread. The bus carries
the URL.

---

## 8. Wiring it into the loop

`publisher.py` is a library with no loop dependency. The calls, in the order the
loop makes them:

| Where in `run/loop.py` | Call |
|---|---|
| after `RobotCell()` is built | `pub.publish_world(world)` |
| `_approve_and_move`, replacing `rig.preview(path)` | `pub.replay_path(world, path)` |
| while `console.readline("approve?")` blocks | `pub.status(phase="awaiting_approval")` |
| after `rig.move(path)` returns | `pub.publish_pose(world, q_now)` |
| after `rig.capture(...)` | `pub.publish_capture(cap)` |
| after `_recenter()` | `pub.publish_object(acc.points, dims, mid)` |
| after `reach` is computed in `turn()` | `pub.publish_views(sphere, reach, visited, plan_failed, current, captures)` |
| anywhere the loop prints | `pub.log(level, msg)` |

Keep `rig.preview` as the seam: pointing it at the publisher instead of
`world.replay` is a one-line change, and meshcat still works if the window is not
running.

**The publisher must never raise into the loop.** A UI that is not connected, a
closed socket, a browser that was killed — none of these are reasons to abort a
run with a robot in the middle of a move. Every public method swallows and logs.

---

## 9. Running it

```bash
cd ~/projects/porthole && npm run build -w @porthole/framework   # after any porthole change
cd ~/projects/robotics/inspection/ui && npm install && npm run build

p inspection/ui/app.py mock                      # fake run, no robot, no camera
p inspection/ui/app.py serve --run_dir=data/...  # window only; your loop owns the bus
npm run check                                    # headless render + PNG, 16 checks
```

`mock` drives every topic from a scripted run. What is real in it: the workcell,
the collision world, IK, the viewsphere, reachability, direct-move validation
and the replay. Only perception and the arm are faked. That makes it an
ordinary Python process on the ordinary socket — the frontend cannot tell, there
is no "demo mode" branch in the UI to rot, and awkward states (a cell that
refuses, a preview that freezes at impact) are reachable without hardware.

`serve` starts **no bus**: there can be one bus on a port and it belongs to
whoever owns the robot. Run the loop in its own process; this just serves the
bundle and opens the window.

### Pointing the UI at a different bus

`?bus=` on the URL overrides the default `ws://<page host>:8765`:

```
http://127.0.0.1:8767/?bus=8781                 same host, other port
http://127.0.0.1:8767/?bus=ws://192.168.2.10:8765   the robot's machine
```

That is how `npm run check` runs a mock beside a real run without the two
fighting over 8765, and how the window opens on a laptop during a demo.

---

## 10. Things that will bite

- **Two copies of React kill the app with an unrelated error.** `@porthole/framework`
  is a `file:` dependency, so it is a symlink, and Node resolves *its* imports
  from porthole's own `node_modules` — a second React, a second dockview. The
  symptom is `Cannot read properties of null (reading 'useMemo')` and a blank
  window, naming neither React nor the symlink. `vite.config.ts` has a
  `resolve.dedupe` list; anything stateful crossing the framework boundary
  belongs on it.
- **The UI dials `ws://<page host>:8765`** unless `?bus=` says otherwise. A bus
  on another port gives you a window that renders perfectly and shows nothing.
- **Do not read pixels back to prove a 3D panel drew.** `preserveDrawingBuffer`
  is false, so `readPixels` returns zeros exactly when the panel is visible.
  Screenshot the element twice and require the bytes to differ — that proves
  geometry *and* animation.
- **Anti-aliased marker edges are marker-coloured but not clickable.** A ray
  through an edge pixel misses the sphere, which looks exactly like a broken
  raycaster. `verify-ui.mjs` requires a matching neighbourhood before it clicks.
- **`porthole update` needs `src/.porthole/base/`.** It is the merge base that
  makes the copied panels updatable. Do not delete it.
- **Rebuild the framework after changing porthole**, or `npm run build` here
  silently links the old `dist/`.
- **Publish full descriptions.** A description missing a node deletes that node
  from the scene, including its GPU buffers.
- **Marker count is 36** (12 azimuth × 3 elevations). If `V_ELEVATIONS` changes,
  nothing breaks — the publisher reads `sphere.elevations`.
