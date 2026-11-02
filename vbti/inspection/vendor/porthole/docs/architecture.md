# Architecture

Porthole is a UI toolkit for live monitor and manager dashboards. Python is the
backend and the only long-lived process; the UI renders what Python publishes and
reports what a human clicked.

It is domain-agnostic. Nothing in the framework knows what a robot is.

## Layers and dependency direction

```
apps/*                    your dashboard: panel registration, layout, app glue
   │  depends on
packages/registry         copyable panel source — you own your copies
   │  depends on
packages/framework        bus client, panel host, docking shell, theme
   │  speaks
docs/bus-protocol.md      the wire contract
   │  implemented by
python/porthole           bus server, codec, static serving, native shell
```

Rules, in dependency order:

1. **Nothing imports upward.** The framework must not import from the registry;
   the registry must not import from an app.
2. **Registry components never import each other.** A shared helper belongs in
   the framework, or is duplicated on purpose. A copied file that imports another
   copied file turns one fork into two.
3. **`bus/` imports no React.** The client is framework-agnostic so replay tools
   and tests can use it headless. React bindings live in `react/`.
4. **The protocol document is the source of truth.** If behaviour and document
   disagree, the document changes first, then both implementations.

## The rule that decides dashboard performance

Two subscription paths exist, and picking wrong is the main way a Porthole
dashboard becomes slow:

| Hook | Re-renders? | Use for |
|---|---|---|
| `useTopic` / `useTopicPayload` | yes | DOM that genuinely changes: logs, readouts, menus, status |
| `useTopicEffect` | no | canvas, WebGL, any imperative drawing API |

**If the panel paints to a canvas, use `useTopicEffect`.** A 30 Hz point cloud
driven through `useTopic` asks React to reconcile a tree thirty times a second to
produce zero DOM changes, because the pixels are drawn by three.js. The
reconciliation is pure waste, and it is what makes streaming dashboards stutter.

Corollary: mutate GPU buffers in place. Allocating a `BufferAttribute` per frame
produces megabytes a second of garbage, and the GC pause lands exactly while the
human is dragging to orbit.

## Naming — the highest-leverage convention here

**Every exported identifier is at least two words, at least one of them
domain-specific.** `PortholeBus`, not `Bus`. `array_payload`, not `nd`.
`BusEnvelope`, not `Envelope`.

This is not style. Coding agents navigate this repo by grep, not by a semantic
index. Measured across 7,922 TypeScript exports, one-word names are 61%
grep-unique versus 88% for two words and 96% for three. The cost of a generic
name is not search time — ripgrep is equally fast either way — it is the context
burned ruling out false positives. `create` returns 1,585 hits across 459 files.

Consequences worth stating:

- Banned as exported names: `create`, `get`, `init`, `handle`, `run`, `Config`,
  `Result`, `Data`, `Item`.
- Banned as filenames: `utils.*`, `helpers.*`, `common.*`, `misc.*`. Grab-bag
  files become the worst file left standing after every refactor.
- Type names obey the same rule. A compiler error names a type, and grepping that
  name is the only next move available.
- One spelling per concept, repo-wide. Directory names are search terms.

## Never let a name, comment or docstring lie

Delete dead paths instead of leaving them; fix docstrings in the same commit that
invalidates them. Misleading natural-language cues cost measured LLM accuracy
(−23% on output prediction in controlled tests) and — this is the important part —
**instructing a model to ignore misleading comments does not recover the loss.**

A human builds immunity to a codebase's lies over months. An agent arrives fresh
every session and believes them.

## File and module structure

- **One concept per file, filename = concept.** Not one export per file.
- **No internal barrel files.** `export *` erases the names it forwards, so an
  agent resolving a symbol lands on a barrel that does not say where anything is.
  The only barrel is `packages/framework/src/index.ts`, the package's public edge,
  and it uses explicit named exports.
- **No `export *`, no `from x import *`, no re-export chains, no registration by
  side effect.** Each converts a text edge into a runtime edge, and a grep-first
  reader cannot follow runtime edges.
- **Kill god-files.** There is no blanket line limit — the evidence for "keep
  files under N lines" is nonexistent — but a 2,000-line file is a real problem
  and a grab-bag is worse than a large focused file.
- **Test filename = source filename + suffix.** `client.ts` → `client.test.ts`.

## Verification is the definition of done

Every claim about this repo working is backed by a command that prints evidence:

```
npx vitest run                         # pure logic, no DOM
python -m pytest python/tests          # protocol + scene payload conformance
node tools/verify-contract.mjs         # Python encodes → TypeScript decodes
node tools/verify-scene-protocol.mjs   # scene protocol against a live backend
node tools/verify-fork-merge.mjs       # fork survives upstream update
node tools/screenshot-demo.mjs         # renders headless + a PNG to look at
```

The cross-language check exists because unit tests on both sides can pass while
the two disagree about the wire. The screenshot check exists so that an agent
writing a panel has a real feedback signal instead of asserting success.

They have already caught real bugs — event topics silently retaining one message,
`useBusHello` never updating, and a headless check that passed only because the
panel it was testing was hidden. That is the standard: a check earns its place by
catching something.

A rule that came out of the last one: **prefer assertions that cannot pass for
the wrong reason.** Reading a WebGL buffer back to prove a panel drew is such an
assertion — it returns zeros exactly when the panel is visible. Screenshotting
the element twice and requiring the bytes to differ proves both that geometry
exists and that it is animating, and cannot be satisfied by a static scene.

## Security posture

There is none, deliberately, and it is only safe because of the deployment shape:
loopback by default, opt-in binding for demos, and a backend that treats every
command as untrusted input from a human. The UI has no authority and no
permission model, because giving it one would create the illusion that it is part
of the safety story. See `bus-protocol.md` §7–8.
