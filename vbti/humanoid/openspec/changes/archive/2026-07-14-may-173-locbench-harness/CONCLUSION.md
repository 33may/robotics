# Conclusion — locbench harness shipped; AI dev-loop continues in `may-173-locdev-flow`

**Status:** closed 2026-07-14. The oracle (localization eval harness) is built and green;
the AI **developer** that drives it forward lives in the sibling change `may-173-locdev-flow`,
continued on the same `33may/may-173-locbench` branch.

## What this change set out to do

Deliver the **oracle**: any `realizations/<name>/` localization candidate gets a reproducible
two-tier pass/fail verdict from frozen episodes, through the full in-brain hosting path
(shadow host → telemetry → evaluator → scorer), without ever breaking the world-invariance
boundary (brain imports neither `isaacsim` nor `limxsdk`).

## What shipped and is green (§2–§7, §8.1)

- **§2 service seam** — `logic/oli/service/`: W4 goal-in (33-byte struct) + W5 telemetry-out
  (JSON datagram) sockets; `brain_main --service`; brain-purity guard.
- **§3 episode sets** — `episodes.py`: seeded, clearance/separation/route-length-constrained,
  zone-biased (~70% between the rails), frozen `episodes/warehouse.json` v1 (Anton-approved
  render); deployment-true routes via `build_planner()`.
- **§4 scoring core** — `pairs.py` (nearest-stamp assoc, warmup exclusion), `stats.py` (raw
  map-frame error, NO alignment — constant bias stays visible), two-tier `verdict.py`,
  `report.py`, `plots.py`.
- **§5 in-brain localization host** — `reason/localization/host.py` side-thread (frame-paced,
  latest-wins, crash→`crashed`), realization `registry.py` (lazy import), `HostLocalizer`
  Stage-2 seam, `--shadow`/`--localizer` wiring.
- **§6 evaluator + runner** — `evaluator.py` (transit → warm-start → scored leg → teardown,
  sim-time timeouts on GT stamps), `runner.py` (launcher-subprocess boot, GT-from-telemetry,
  bench GT feed republish), `score`/`board` offline paths.
- **§6.2/§7 env tooling** — `envs.py` `locbench env create|remove`: disposable `bench-<name>`
  conda env per candidate, `lock.yml` frozen provenance, hard-guard on `brain|isaac|limx|hum`;
  `run` boots the brain in `bench-<candidate>` (hard-error if missing — Anton 2026-07-14).
- **§8.1 reference candidate** — `realizations/reference/`: GT-replay measuring stick with
  injectable bias/noise/dropout, its own minimal brain-compatible `bench-reference` recipe.

Test posture: full `brain`-marked suite green (497+), including the new §7 env tooling (18
cases, conda injected as a runner → no conda in CI).

## What was deferred, and where it goes

- **§1.2 spec delta** (`specs/localization-bench/spec.md`) — deliberately not formalized;
  capture-not-encode. Not folded into `openspec/specs/`.
- **§8.2 / §8.3 acceptance triplet + baseline row** — the harness proving itself (clean→PASS,
  0.2 m bias→fails max-pos, 20% dropout→fails coverage). This is a **live run Anton executes**;
  the tooling is ready (`locbench env create reference` → `run reference --smoke 3`). Until it
  is run, the harness is built-but-not-self-validated live.
- **§9 mapping pass** (`locbench map`) — future; lands with the first real map-building
  candidate (RTAB-Map), not needed for the GT-replay reference.
- **§10 Stage-2 closed loop** (`--localizer`, Nav on the candidate's pose) — future; gated on a
  Stage-1 PASS candidate. The `HostLocalizer` seam (§5.4) is already in place, dormant.
- **§11 docs / guard / memory** — §11.1 (`realizations/AGENTS.md` playbook) is **absorbed into
  `may-173-locdev-flow`** and built there. §11.2 (`logic/locbench/AGENTS.md`), §11.3
  (architecture-guard extensions: no brain-marked import of `realizations/`, `service/` purity,
  `logic/oli/` never imports `locbench`), §11.4 (memory/daily) **carry forward into
  locdev-flow** on the same branch.

## What continues

`openspec/changes/may-173-locdev-flow/` — the AI-driven development loop on top of this oracle
(playbook + `/loc-new` scaffolder shipped; `/loc-iterate` pair-mode skill, flow self-test on
reference, earned-autonomy workflow, and the carried-forward §11 guard/docs still to build).
Same branch; merges to `main` when the AI part lands.
