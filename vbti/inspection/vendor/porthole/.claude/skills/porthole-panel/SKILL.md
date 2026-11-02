---
name: porthole-panel
description: Use when adding a panel to a Porthole dashboard, wiring a new topic between a Python backend and the UI, or scaffolding a new Porthole app. Covers the panel contract, the two subscription paths, config-vs-fork, and the verification loop.
---

# Building on Porthole

Three recipes. Follow the one that matches, then verify.

## Recipe 1 — add an existing panel

```bash
node tools/porthole.mjs list
node tools/porthole.mjs add camera --into apps/yourapp/src
```

Register it and place it:

```ts
import { definePanel } from '@porthole/framework';
import { CameraPanel, cameraPanelDefaultConfig } from './panels/CameraPanel.js';

export const panels = [
  definePanel({
    type: 'camera',                       // kebab-case; appears in saved layouts
    title: 'wrist',
    component: CameraPanel,
    defaultConfig: { ...cameraPanelDefaultConfig, topic: 'camera/wrist' },
  }),
];

const layout = [
  { id: 'cam', type: 'camera' },
  { id: 'log', type: 'event-log', position: { referencePanel: 'cam', direction: 'below' } },
];
```

Two instances on two topics is just two entries with different `config` — never a
second copy of the file.

## Recipe 2 — write a new panel

Decide the subscription path **first**; it determines the whole shape.

**Canvas / WebGL / imperative drawing → `useTopicEffect`, never re-render:**

```tsx
export interface ScopePanelConfig extends Record<string, unknown> {
  topic: string;
  windowSeconds: number;
}
export const scopePanelDefaultConfig: ScopePanelConfig = {
  topic: 'state/joints',
  windowSeconds: 30,
};

export function ScopePanel({ config }: PanelProps<ScopePanelConfig>) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useTopicEffect<{ values: number[] }>(config.topic, (envelope) => {
    drawInto(canvasRef.current, envelope.payload.values);   // no setState
  });

  return <canvas ref={canvasRef} className="porthole-scope-panel" />;
}
```

**DOM that genuinely changes → `useTopic` / `useTopicPayload`:**

```tsx
const state = useTopicPayload<{ step: number }>(config.topic);
return <div>{state?.step ?? '—'}</div>;
```

Non-negotiables for a new panel:

- Export `interface <Name>PanelConfig` and `const <name>PanelDefaultConfig`.
  Every config key the panel reads has a default there.
- Take `{ config }: PanelProps<YourConfig>` — never hardcode a topic.
- Style with `var(--porthole-*)` tokens, never literal colours.
- Handle `undefined`: a panel renders before its first message and must not throw.
- Tolerate `seq` gaps — stream topics drop frames by design.
- Set `keepMounted: true` only if a remount would rebuild something expensive
  (WebGL context, decoded mesh, accumulated history).

## Recipe 3 — publish a new topic from Python

```python
bus = PortholeBus(app="yourapp").start()
bus.declare("scene/cloud", qos="stream", kind="ndarray")
bus.declare("log/events",  qos="event")       # never dropped, keeps a backlog

bus.publish("scene/cloud", {"points": bus.array_payload(pts.astype("float32"))})
bus.publish("camera/wrist", bus.jpeg_payload(rgb))
```

- `qos="event"` for anything a human must not miss; `stream` for sampled data
  where message N makes N−1 worthless.
- **Publish state, not deltas** — the last message is replayed to late clients.
- `float32` for geometry. Never send raw pixel buffers; encode with
  `jpeg_payload`.
- Commands arriving from the UI are untrusted human input. Validate everything;
  the bus never acts on one.

Then add a mock so the panel is developable without hardware — a mock is an
ordinary publisher on the real socket, so there is no UI-side demo mode:

```python
rig = MockRig(bus)
rig.every(15, "camera/wrist", lambda t: bus.jpeg_payload(gradient_frame(t)))
rig.run()
```

## Verify — always, and look at the picture

```bash
cd apps/demo && npx vite build && cd ../..
node tools/screenshot-demo.mjs
```

Then **Read the PNG**. Every other check passes for a panel that mounts cleanly
and draws nothing; only looking catches that.

If you changed the wire: update `docs/bus-protocol.md` first, then
`cd python && p -m pytest tests -q` and `node tools/verify-contract.mjs`.

## When to fork instead of configure

Adding a config key for a *value* (topic, colour, limit, range) — configure.
Needing *different behaviour or new data* — fork the file and keep
`.porthole/base/` so `porthole update` can still merge upstream fixes. Set
`updateMechanism` honestly in `.porthole/manifest.json`; forking is expected, and
silently forking is the only mistake.
