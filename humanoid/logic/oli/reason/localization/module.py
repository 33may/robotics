"""localization/module.py — the `LocalizationModule` protocol (design.md D5, D13, D14).

The host-agnostic surface a candidate localization algorithm implements. The SAME object is
hosted by two harnesses and cannot tell which is calling:

  - the **bench** (`logic/locbench/`, change b): replays a frozen bag as a `LocalizationIn`
    sequence and scores the raw map-frame output against ground truth;
  - the **live node** (MAY-173 endgame): feeds it from the frame/obs taps in real time and emits
    each `LocalizationOut` as a datagram to the thin in-brain `Localizer` client.

Lifecycle: `start(setup)` once (map artifacts + known-start hint + calibration), `step(loc_in)`
per camera tick, `stop()` once. No `reset` in v1 — known-start only; recovery is added when a
candidate actually supports it. Map BUILDING is deliberately not here: a candidate ships a
bench-only `build_map` entry alongside its module (D14) so the live contract stays clean.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .contracts import LocalizationIn, LocalizationOut, LocalizationSetup


@runtime_checkable
class LocalizationModule(Protocol):
    """`start(setup)` → `step(loc_in) -> LocalizationOut` per frame → `stop()`."""

    def start(self, setup: LocalizationSetup) -> None: ...

    def step(self, loc_in: LocalizationIn) -> LocalizationOut: ...

    def stop(self) -> None: ...
