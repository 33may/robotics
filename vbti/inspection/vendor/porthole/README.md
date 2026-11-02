# porthole

A window into a running Python process.

A UI toolkit for live monitor and manager dashboards: streaming cameras, 3D
scenes, plots, logs, file browsers, and controls. Python is the backend and the
only long-lived process at runtime. The frontend is static files — it runs in a
native window or any browser, including one on another machine.

Domain-agnostic by design. Nothing in the toolkit knows what a robot is.

```
┌── Python ──────────────┐         ┌── UI ────────────────────┐
│ PortholeBus            │ msgpack │ dockview shell           │
│  publish(topic, data)  │ ──────► │  panels subscribe        │
│  commands()            │ ◄────── │  buttons send commands   │
└────────────────────────┘  JSON   └──────────────────────────┘
```

## Try it

```bash
cd apps/demo && npx vite build && cd ../..
p python/examples/demo_app.py           # native window
p python/examples/demo_app.py --browser # same UI in a browser
```

For frontend work with hot reload, run the backend and the dev server separately:

```bash
p python/examples/demo_publisher.py
npm run dev -w @porthole/demo
```

## Two tiers

**Framework** (`@porthole/framework`, `porthole` on PyPI) — bus, panel host,
docking shell, theme. A pinned dependency you never edit, so a protocol fix
reaches every app by version bump.

**Components** (`packages/registry`) — panels, copied into your repo by
`porthole add`. You own and edit them. Because the installer records a pristine
base, `porthole update` is a real three-way merge: upstream fixes land in regions
you never touched, and only overlapping edits conflict.

That trade — continuous small drift instead of forced big-bang migrations — is the
point. See `docs/component-model.md`.

## Docs

| | |
|---|---|
| `docs/bus-protocol.md` | the wire contract; change this before changing code |
| `docs/scene-protocol.md` | how a backend describes and animates a 3D scene |
| `docs/component-model.md` | config vs fork, the update mechanism |
| `docs/architecture.md` | layering, naming, the subscription rule |
| `AGENTS.md` | commands and the things that will bite you |

## Panels

| | |
|---|---|
| `media-grid` | any number of live image/video cells, auto-shaped to the panel |
| `scene-view` | backend-driven 3D: boxes, meshes, animated poses, verdict tint |
| `camera` | one image topic, full panel |
| `point-cloud` | orbitable `[N,3]` cloud |
| `event-log` | streaming log with follow-tail |
| `command` | options, note field, submit |

## Verify

```bash
npm run check
```

Unit tests, protocol conformance (Python), the cross-language wire check, scene
protocol conformance against a live backend, the fork-merge proof, and a
headless render that leaves a PNG to look at.

## Status

v0.1. Bus, panel host, docking shell, registry, fork tooling, native shell and
verification loop all work end to end, verified on Ubuntu 24.04 / Wayland.

Not yet: plot panel, file-browser panel, bus recording and replay.
