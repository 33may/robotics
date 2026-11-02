# The component model

> Copy-in exchanges continuous small drift for the absence of forced big-bang
> migrations. That trade is the reason this toolkit exists, and it tells you
> where the engineering budget goes: **drift is the expected cost, so the tooling
> detects and bounds it rather than trying to prevent forks.**

## Two tiers, and why

| Tier | What | How you get it | Do you edit it? |
|---|---|---|---|
| **Framework** | bus client, panel host, docking shell, theme tokens, Python bus | pinned dependency (`@porthole/framework`, `porthole` on PyPI) | **No.** |
| **Components** | panels and widgets | copied into your repo by `porthole add` | **Yes, freely.** |

The split is by *rate and blast radius of change*. A bug in the wire protocol is
wrong for every app that ever existed, and must propagate by version bump — not
by patching five forks. A camera panel's overlay colour is wrong only for you,
and needs no upstream involvement at all.

The failure mode this avoids is the one everyone has lived through with a
monolithic component library: you cannot upgrade one component, so you upgrade
all of them at once, so you never upgrade.

## The reusability law

**Anything that would otherwise make you copy a component belongs in its config.
Anything that changes how it works belongs in code, and that is a fork.**

Config is what makes a panel *instantiable* rather than *duplicable*. A camera
panel whose topic is config can appear twice in one dashboard on two topics. A
camera panel with `"camera/wrist"` baked in can only be duplicated by duplicating
its source — and then you own two copies of every future bug.

Decision rule, in order:

1. **Is it a value?** (topic, colour, limit, units, axis range, frame id) → config
   key with a default in `defaultConfig`. Not a fork.
2. **Is it a different arrangement of the same data?** → config key, usually an
   enum (`fit: 'contain' | 'cover'`).
3. **Is it different behaviour, or new data the panel never had?** → fork it.
   That is what forking is *for*; it is not a failure.
4. **Would every consumer want it?** → contribute it upstream instead, and let
   everyone's `porthole update` pick it up.

Resist the pull of rule 1. A config key added "just in case" is permanent API
surface on a file you also expect people to edit, and the two multiply.

## Forking, and why it is safe here

`porthole add` records a pristine copy of exactly what it installed under
`.porthole/base/`. Keep that directory. Commit it.

Because the base exists, `porthole update` is a genuine three-way merge:

```
git merge-file <yours> <base> <upstream>
```

Upstream changes to regions you never touched apply silently. Only edits that
overlap yours conflict, and those get ordinary git conflict markers.

This is the difference between owning source and being stranded on it. Tools
that record no base can only offer a two-way diff, in which your deliberate
edits and upstream's changes are indistinguishable — which is why nobody runs
them twice.

```
porthole list                    # what the registry offers
porthole add camera              # install + record base
porthole status                  # forked? upstream moved? which mechanism?
porthole update camera           # three-way merge
porthole update --all
```

### Declare your intent

Every installed item carries an `updateMechanism`, taken from Chromium's
vendoring taxonomy:

| Value | Meaning |
|---|---|
| `Auto` | merge upstream changes routinely (default) |
| `Manual` | a human reviews every update |
| `Static` | pinned deliberately; `update` skips it |
| `Static.HardFork` | diverged so far that merging is no longer meaningful |

Set it honestly, and write what you changed in `localModifications`. The point is
that "has anyone forked this?" should be a query, not an archaeology project.
Even a hard fork keeps its base — recording the point of divergence costs
nothing and is the only thing that makes a later reconciliation possible.

## Component structure

Keep the forkable surface small, so that when someone does fork, they fork a
little:

- **Logic in hooks, pixels in components.** A hook that accumulates a series and
  a component that draws it can be forked independently.
- **One concept per file, named after the concept.** `CameraPanel.tsx`, not
  `panel.tsx`. Filenames are search terms.
- **Config interface exported next to the component**, with a
  `<name>DefaultConfig` const. Every key a panel reads has a default there —
  a panel that reads an undeclared config key breaks silently on a fresh install.
- **No cross-imports between registry components.** If two panels need the same
  helper, it belongs in the framework or is duplicated deliberately. A copied
  file that imports another copied file turns one fork into two.
- **Style with the theme tokens** (`var(--porthole-*)`), never hardcoded colours,
  so a copied component still obeys its new home's theme.

## What belongs in the framework instead

Push something down into the framework when it is *invisible to the eye and
identical for everyone*: transport, decoding, subscription, layout persistence,
panel registration. If a fix to it would need to reach every app, it cannot live
in a tier that gets copied.

When you find yourself wanting to edit `@porthole/framework` to make a panel
work, that is a signal the framework is missing an escape hatch — raise it,
rather than forking the framework. Forking the tier whose entire job is to be
identical everywhere defeats the split.
