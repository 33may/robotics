#!/usr/bin/env python3
"""Subprocess entry point for the headless e2e suite (task-11).

Runs exactly `ui/app.py mock` — same bus, same `Supervisor`/`FakeRig`, same
Flow A/B wiring, same `--collect` switch — with one substitution applied
first: the planner. This sandbox's freshly re-taught safety keep-outs
(`cell/cell.yaml`, 2026-08-31 freedrive-tap rebuild — see its own
`corner3: ... WRONG, reprobe pending` note) refuse the demo survey pose under
every analytic IK branch ("no collision-free IK branch", confirmed live
against `inspection.motion.plan.plan_viewpoint` while writing this suite),
so the REAL `Supervisor`/`SweepDriver` planners never reach "previewing" —
no confirm-button ever appears for a browser to click.

This is not new: `inspection/tests/test_flow_a.py:DirectPlanSupervisor` and
`inspection/tests/test_collect.py:_stub_planner` already carry this exact
workaround for every other mock-driven test in this checkout. This file
applies the SAME two substitutions to the real `ui/app.py mock` subprocess
the e2e fixtures launch, rather than inventing a third version.

Deliberately NOT inside `ui/mock.py` or `ui/app.py`: `ui/mock.py`'s own
docstring says "no scripted branch in the UI (or here) to rot" — this IS
that branch, kept in test-only code instead of the module it would rot.
`ui/app.py`'s `mock()` is called unmodified below; only the two module
globals it reads `Supervisor`/`_default_planner` from are swapped first.

    ~/miniconda3/envs/robo/bin/python -m inspection.tests.e2e._mock_launcher \\
        mock --port=<p> --bus_port=<p> --run_root=<dir> --no_window \\
        --open_browser=False [--collect=True]
"""
from __future__ import annotations

import sys


def main() -> int:
    import inspection.run.collect as collect_mod
    from inspection.tests.test_collect import _stub_planner
    from inspection.tests.test_flow_a import DirectPlanSupervisor
    from inspection.ui import app as ui_app
    from inspection.ui import mock as mock_mod

    # Same two swaps as the unit tests, applied before any Supervisor or
    # SweepDriver gets constructed: the dispatcher's own plan_viewpoint/
    # plan_to_cell calls (Supervisor._plan_worker) and the sweep's ranking
    # planner (run/collect.py's `_default_planner` factory, looked up by
    # bare name at SweepDriver.__init__ time — patching the module
    # attribute here is enough, no call-site changes needed).
    mock_mod.Supervisor = DirectPlanSupervisor
    collect_mod._default_planner = lambda sup, seed=0: _stub_planner

    import fire

    fire.core.Display = lambda lines, out: print(*lines, file=out)
    return fire.Fire({"mock": ui_app.mock}) or 0


if __name__ == "__main__":
    sys.exit(main())
