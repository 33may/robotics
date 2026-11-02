# deploybench/ — the interactive end-to-end localization deploy test

Answers **"given the baked map artifacts, spawn (known or kidnapped) → guide to a goal → does
it arrive, and what are the localization stats?"** for *any* localization algorithm. Sibling
of `logic/locbench/`; **never edits it** (locbench is the frozen accuracy oracle — different
instrument). Read `logic/locbench/AGENTS.md` for the shared spine it reuses.

## deploybench vs locbench (why a separate harness)

| | locbench | deploybench |
|---|---|---|
| course | frozen, seeded `EpisodeSet`, committed | interactive per-run `Scenario` (operator-picked) |
| question | localization **accuracy** (raw-frame error gates) | **deploy**: did the robot **arrive**, + loc stats |
| steering | GT-steered, candidate shadowed | steer on the **candidate's own estimate** (closed loop, D7) |
| start | known only (GT warm-start) | **known OR kidnapped** (no hint → self-localize) |
| scope | full frozen scene, every episode must pass | whatever partial map a demo baked |

Est-steering breaks locbench's "every candidate runs the same course" invariant (a worse
localizer drives a different path) — that is exactly why this is a separate harness, not an
edit to the oracle.

## Seam — algorithm-agnostic

The algorithm under test is any `reason/localization/realizations/<name>/` implementing
`LocalizationModule` (`start(LocalizationSetup)` → `step(LocalizationIn)` → `stop()`), run in
its own `bench-<name>` env, exactly as locbench hosts it. deploybench supplies the map via
`LocalizationSetup.map_dir` and the start hint via `initial_pose` (None for KIDNAPPED). You
build the algorithm in main; deploybench only tests it.

## Module map (built incrementally)

| file | role | status |
|---|---|---|
| `scenario.py` | interactive per-run course model + FR2 validate + save/load | DONE (tests green) |
| `maps.py` | resolve a baked map_dir → `OccupancyGrid` for display/validation | — |
| `pick.py` | interactive: render map, click start+goal, choose mode, save `Scenario` | — |
| `deploy_stats.py` | arrival verdict + loc stats + kidnapped time-to-first-fix (reuses locbench `pairs`/`stats`) | — |
| `report.py` / `plots.py` | `report.json` + GT-vs-est-vs-goal overlay, error timeline | — |
| `runner.py` / `evaluator.py` | boot the stack, drive the scenario, collect (est, GT) | reuse locbench `Evaluator` |
| `__main__.py` | CLI: `pick`, `run`, `score` | — |

## Rules

- **Reuse, don't fork.** Import locbench `pairs`/`stats`/`plots` and the `LocalizationModule`
  seam. Copy nothing you can import.
- **GT is a hidden oracle (FR6).** The algorithm sees only the map + (KNOWN) the operator hint.
  GT scores the run; it is never an algorithm input. Raw map-frame scoring, no alignment.
- **Pure core → `brain` env.** `scenario`/`deploy_stats`/`report` stay numpy/stdlib and
  `brain`-marked. matplotlib/subprocess/interactive glue (`pick`, `runner`) live OUTSIDE the
  brain boot, like locbench.
- **Deleting anything (envs, maps, runs) is always Anton's call** (realizations playbook).
