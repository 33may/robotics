# porthole

UI toolkit for live monitor/manager dashboards. Python backend + React frontend
over a msgpack WebSocket bus. Domain-agnostic — nothing here knows about robots.

Read `docs/architecture.md` before changing structure, `docs/component-model.md`
before touching `packages/registry`, `docs/bus-protocol.md` before touching
anything on the wire, `docs/scene-protocol.md` before touching the 3D panel or
`porthole.scene`.

## Commands

Python lives in the conda env `robo`, invoked as `p <path>` (never `python`,
never `-m`). Node is 22 (Vite 8 and Tailwind 4 both require ≥20).

```
npm run check                                # everything below except :native
npm run check:unit                           # vitest, pure logic only
npm run check:python                         # protocol + scene payload conformance
npm run check:contract                       # Python encodes → TS decodes
npm run check:scene                          # scene-protocol conformance, live backend
npm run check:fork                           # fork survives upstream update
npm run check:render                         # headless render + PNG to inspect
npm run check:native                         # opens a real window ~10s, proves the ws connects
```

Run it as a native app: `p python/examples/demo_app.py` (add `--browser` to skip the shell).
Dev with HMR: `p python/examples/demo_publisher.py`, then `npm run dev -w @porthole/demo`.

## Non-obvious things that will bite you

- **The protocol document is the contract, not the code.** Change
  `docs/bus-protocol.md` first; the tests assert against it by section number.
- **Exported names must be ≥2 words, ≥1 domain-specific** (`PortholeBus`, not
  `Bus`). Agents grep this repo; one-word exports are 61% grep-unique vs 88% for
  two. Rationale in `docs/architecture.md`.
- **Canvas/WebGL panels must use `useTopicEffect`, never `useTopic`** — the
  latter re-renders React to produce zero DOM changes. This is the main
  performance trap.
- **`vite build --root apps/demo` silently uses the wrong config.** Always
  `cd apps/demo && npx vite build`, or the screenshot check tests a stale bundle.
- **Publish state, not deltas.** Last message per topic is retained and replayed
  to late clients; a delta-only topic leaves them permanently wrong.
- **`.porthole/base/` is load-bearing.** It is the merge base that makes forked
  components updatable. Never delete or regenerate it casually.
- The frontend is IO only. It has no authority; never add a permission model to
  it (`bus-protocol.md` §7).
- **Native shell = pywebview + Qt.** GTK cannot work under conda: the distro's
  PyGObject is built for the system Python's ABI. `qtpy` is required in addition
  to PySide6 — pywebview imports qtpy, and without it reports "QT cannot be
  loaded" while PySide6 sits there installed.
- The shell serves over HTTP, never `file://`. Opaque `file://` origins are the
  documented cause of localhost WebSocket failures inside embedded webviews;
  serving over http avoids the class, and is why the same build runs in a browser.
- The UI dials `ws://<page host>:8765` unless the app passes a `url`. A bus on a
  non-default port with an unconfigured UI produces a window that renders
  perfectly and connects to nothing.
- **The 3D world is Z-up and transforms are absolute.** `camera.up` must be set
  when the camera is created — `OrbitControls` snapshots `object.up` in its
  constructor and never re-reads it, so setting it in an effect silently gives
  you Y-up orbiting.
- **`ColladaLoader` rotates Z-up assets by -π/2 about X** and warns on the
  console. Every robot `.dae` is Z-up, so that rotation must be cleared or the
  meshes lie on their side while the primitives are correct.
- **Do not read pixels back from a WebGL canvas to prove it drew.**
  `preserveDrawingBuffer` is false by default, so `readPixels` returns zeros
  once the frame is composited — it appears to work only while the panel is
  hidden. Screenshot the element and compare two shots over time instead.
- Large static geometry is served over HTTP via `serve_ui(..., assets=...)`,
  never sent on the bus. The UR5e's visual meshes are 9 MB.
