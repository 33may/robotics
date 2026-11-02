# Scene protocol

How a Python backend describes a 3D scene to a Porthole UI, and how it animates
it. This is an **application-level payload schema** carried by the transport in
`bus-protocol.md` — it adds no new wire mechanics.

It is domain-agnostic. Boxes, meshes and transforms; nothing here knows what a
robot is. The design is modelled on meshcat, which has proven this shape works
for exactly this job.

## 1. Two topics

| Topic | QoS | Rate | Carries |
|---|---|---|---|
| `<prefix>/description` | `stream`, retained | rare — on change | what exists: every node, its geometry and material |
| `<prefix>/poses` | `stream` | animation rate | where things are right now |

Default prefix is `scene`, so `scene/description` and `scene/poses`.

The split exists because the two have completely different sizes and rates. A
description is kilobytes and changes when the world is re-modelled; a pose frame
is hundreds of bytes and changes 30 times a second.

Both are retained, so a UI that connects late gets the current world and the
current pose, not an empty canvas.

**A pose frame is atomic.** Everything that must agree with a pose — a tint
marking a colliding link, a background colour signalling a verdict — travels in
the same message as that pose. Splitting them across topics means a dropped
message can leave a green robot standing inside a table, and `stream` QoS drops
messages by design.

## 2. Coordinates

- **Z-up, right-handed, metres.** Coordinates are published exactly as the
  backend holds them; the UI does not convert. A number in the payload is the
  same number in the robot's base frame.
- **Transforms are absolute, not parent-relative.** Every node's transform maps
  its geometry into the scene frame. Paths look hierarchical but are names, not
  a transform tree.

The second rule is the load-bearing one. Forward kinematics already produces
absolute placements for every link — `pinocchio`'s `oMg`, meshcat's
`set_transform` after `display()` — so absolute transforms are what a backend
naturally has. A parent-relative tree would require the backend to invert
placements it just computed, and would let one wrong parent silently displace a
subtree.

`three.js` is Y-up by default. The panel sets the camera's up vector to
`(0, 0, 1)` rather than rotating the scene, so the world stays in the backend's
coordinates.

## 3. Description payload

```python
{
  "frame": "base",            # informational: what the coordinates mean
  "nodes": [
    {
      "path": "cell/table_slab",       # unique; "/" groups for visibility only
      "geometry": {...},               # §4
      "material": {...},               # §5, optional
      "transform": [16 floats],        # row-major 4x4, absolute
      "visible": true                  # optional, default true
    },
    ...
  ]
}
```

**Publish the whole scene every time.** Retention keeps the last message only,
so an incremental "add this node" message would leave a reconnecting UI with a
scene missing everything published before it. This is the general rule in
`bus-protocol.md` §5 applied here.

Nodes absent from a new description are removed from the scene, and their GPU
resources are freed.

## 4. Geometry

| `kind` | Fields | Notes |
|---|---|---|
| `box` | `size: [x, y, z]` | full extents, not half-extents; centred on the node origin |
| `sphere` | `radius` | |
| `cylinder` | `radius`, `length` | axis is **+Z**, matching the convention in every robotics geometry library |
| `axes` | `size` | an origin triad; R=X, G=Y, B=Z |
| `mesh` | `url`, `scale: [x, y, z]` | §7 |
| `points` | `topic`, `field`, `size`, `maxPoints` | a point cloud fed by another topic — §4.1 |

`box` takes full extents because that is what a human measures with a tape and
what URDF/`cell.yaml` files record. Coal and three.js both want full extents;
`pinocchio`'s `Box.halfSide` is the odd one out, so a backend reading it must
double.

### 4.1 Points reference a topic, they do not carry data

```python
{"kind": "points", "topic": "cloud/fused", "field": "points",
 "size": 0.003, "maxPoints": 200000}
```

The node says *a cloud lives here*; the cloud itself streams on `topic` as
`{<field>: $nd float32 [N, 3]}`. The panel subscribes for as long as the node
exists.

This indirection is the point. A description is republished in full whenever
anything in the world changes (§3), and a fused cloud is megabytes — inlining it
would mean resending the whole cloud every time a marker changes colour. Keeping
bulk data on its own stream topic lets the two move at their own rates, and the
same cloud can feed a dedicated cloud panel without being published twice.

Coordinates are in the scene frame, so the node's own transform is normally
identity. It is applied on top if you set it, which is how you show the same
cloud in two places.

## 5. Material

```python
{"color": "#ff8800", "opacity": 0.45, "wireframe": false, "side": "double"}
```

All fields optional. `opacity < 1` implies transparency. `side` is `"front"`,
`"back"` or `"double"`; default `"front"`.

A `mesh` node with no material keeps the materials baked into the mesh file. A
`mesh` node with a material overrides them — which is how a link gets tinted red
without shipping a second mesh.

## 6. Pose payload

```python
{
  "names": ["robot/wrist_3", "tool/rg2"],
  "transforms": <$nd float32, shape [N, 4, 4]>,   # row-major, absolute
  "overrides": {                                   # optional, per-frame
    "tool/rg2": {"color": "#dd2222", "opacity": 0.5, "visible": true}
  },
  "background": ["#8a3a3a", "#4a1c1c"]             # optional [top, bottom]
}
```

`names` travels in every frame rather than being fixed once by the description.
It costs ~20 bytes per node and removes a hidden index contract between two
topics — the kind that survives testing and breaks the day someone reorders a
list.

Names not present in the current description are ignored, not buffered. A pose
for a node that does not exist is a backend bug, and silently queueing it makes
that bug invisible.

`overrides` are per-frame and not sticky: a node not mentioned reverts to its
description material. This means a backend cannot forget to clear a red tint.

## 7. Meshes are URLs

Mesh geometry is fetched over HTTP from the same server that serves the UI, not
carried on the bus. The UR5e's seven visual meshes are 9 MB; base64 through
msgpack on every reconnect is wasteful when the browser already has a caching,
streaming, off-thread loader for exactly this.

Serve them by passing an asset mount to `serve_ui`:

```python
serve_ui(dist, assets={"/meshes/ur5e": "~/.cache/robot_descriptions/.../visual"})
```

Then publish `{"kind": "mesh", "url": "/meshes/ur5e/wrist3.dae"}`.

Supported extensions: `.dae`, `.gltf`, `.glb`, `.stl`, `.obj`.

**Collada files declaring `<up_axis>Z_UP</up_axis>` are rotated by three.js's
loader.** It sets `scene.rotation.x = -π/2` to convert to its own Y-up world and
warns on the console. Porthole's world is already Z-up, so the panel clears that
rotation after loading. Without that, every robot mesh lies on its side while
every box is correct — a symptom that reads like a backend bug and is not one.

## 8. What the UI does not do

The panel renders transforms. It does not compute them.

No URDF parsing, no forward kinematics, no IK, no joint interpolation in the
browser. Animation is the backend publishing pose frames in a loop — the same
loop that already drives a meshcat viewer. This follows from
`bus-protocol.md` §7: the frontend is IO only, and a second kinematic
implementation in TypeScript is a second thing that can disagree with the
planner about where the robot is.

Consequence worth stating: replaying a planned path at 30 Hz costs the backend
one FK call per frame and the bus ~800 bytes per frame. There is no reason to
push a joint-space trajectory to the browser and interpolate it there.

## 9. Conformance

A backend conforms if:

1. `<prefix>/description` and `<prefix>/poses` are declared `stream` and retained.
2. Every description carries the complete node set.
3. Every transform is row-major, absolute, in the declared frame, metres.
4. `transforms` is a float32 `$nd` of shape `[N, 4, 4]` with `len(names) == N`.
5. Mesh URLs resolve against the UI's own origin.
6. No pose frame references a node absent from the last description.

`node tools/verify-scene-protocol.mjs` checks 3, 4 and 6 against a live backend.
