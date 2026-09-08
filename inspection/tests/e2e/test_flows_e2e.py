#!/usr/bin/env python3
"""Headless e2e over both flows (task-11, spec §8.3): a real Chromium page
drives the real mock app through the real DOM (`ask-input`/`ask-send`,
`confirm-button`, `trace-panel`) — the same selectors an operator's click
would hit — and the resulting run is read back exactly the way
`record show`/`record validate` would, through `Run.load`.

Excluded from the default `pytest inspection/tests` run (see the
`norecursedirs` entry in `pyproject.toml`); run explicitly:

    p -m pytest inspection/tests/e2e -m e2e -q
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from inspection.record.run import Run

pytestmark = pytest.mark.e2e

SCREEN_DIR = Path("/tmp/e2e-flows")
CONFIRM = "[data-testid=confirm-button]"


def _wait_for(predicate, timeout: float, interval: float = 0.5, msg: str = "") -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise TimeoutError(msg or "condition never became true")


def test_flow_a_ask_approve_writes_a_valid_live_run(mock_app_live, page):
    """Ask -> survey plans -> human confirms -> the trace grows -> a valid
    `source=="live"` run with a fused survey step, however the run ends.

    One approval is enough to prove the whole chain end to end; Brain keeps
    reasoning afterwards on its own, and a mock nobody keeps clicking closes
    "aborted" (no answer) — the honest status per `run/app.py:finish_run`,
    not a failure. The assertion is `validate().ok`, never `status`.
    """
    SCREEN_DIR.mkdir(parents=True, exist_ok=True)
    page.set_default_timeout(30_000)

    page.goto(mock_app_live.url)
    page.fill("[data-testid=ask-input]", "is there a logo?")
    page.click("[data-testid=ask-send]")

    # The survey plans (and the FakeRig moves) before any preview exists —
    # the slowest step in the flow, hence the wider timeout here only.
    page.wait_for_selector(CONFIRM, timeout=60_000)
    page.screenshot(path=str(SCREEN_DIR / "a-preview.png"))

    page.click(CONFIRM)
    # TracePanel.tsx's approval label for state=="approved" is "you
    # approved" — substring-matched by Playwright's `text=` engine. Proof
    # the click reached the bus and the beat landed in the trace, not just
    # proof the button existed.
    page.wait_for_selector("text=approved", timeout=30_000)
    page.screenshot(path=str(SCREEN_DIR / "a-trace.png"))

    # "approved" appears the instant the gate opens — well before the arm
    # finishes moving, capturing and fusing. Wait for the WRITE before
    # tearing the subprocess down, or "survey step fused" below races
    # RunWriter's own disk timing.
    def survey_fused() -> bool:
        try:
            survey = Run.load(mock_app_live.run_dir).survey
        except Exception:
            return False
        return survey is not None and survey.record.phase == "fused"

    _wait_for(survey_fused, timeout=60.0,
             msg="survey step never reached phase=fused on disk")

    mock_app_live.shutdown()

    run = Run.load(mock_app_live.run_dir)
    assert run.record.source == "live"
    survey = run.survey
    assert survey is not None and survey.record.phase == "fused"
    report = run.validate()
    assert report.ok, [(p.severity, p.where, p.what) for p in report.problems]


def test_flow_b_no_ai_panel_sweep_writes_a_valid_data_engine_run(mock_app_collect, page):
    """Flow B: no `run/meta.source=="data-engine"` -> no trace panel, no ask
    box (`InspectionApp.tsx`'s `collectMode` gate) -> the sweep proposes
    cells one at a time, cheapest first -> a valid data-engine run with at
    least one non-survey step the sweep (not a human or the AI) chose.
    """
    SCREEN_DIR.mkdir(parents=True, exist_ok=True)
    page.set_default_timeout(30_000)

    page.goto(mock_app_collect.url)
    assert page.locator("[data-testid=trace-panel]").count() == 0
    assert page.locator("[data-testid=ask-input]").count() == 0

    # Survey first, then two sweep-chosen cells — enough to prove the sweep
    # is really driving repeated approvals, not just the one-shot survey.
    # Survey -> cell switches `ActionsPanel` branches (the DOM node
    # unmounts), but cell -> cell reuses the SAME node in place (React keeps
    # the position, just updates its label/handler) — so a "wait until
    # detached" only works for the first transition. Comparing the button's
    # own label instead covers both: proof THIS approval was consumed
    # before the next iteration looks for one again, regardless of whether
    # the DOM node was replaced or reused.
    for _ in range(3):
        page.wait_for_selector(CONFIRM, timeout=60_000)
        label = page.locator(CONFIRM).inner_text()
        page.click(CONFIRM)
        page.wait_for_function(
            """(prevLabel) => {
                const el = document.querySelector('[data-testid=confirm-button]');
                return !el || el.innerText !== prevLabel;
            }""",
            arg=label, timeout=30_000)

    page.screenshot(path=str(SCREEN_DIR / "b-sweep.png"))

    def sweep_step_written() -> bool:
        try:
            steps = Run.load(mock_app_collect.run_dir).steps
        except Exception:
            return False
        return any(s.id > 0 and s.record.phase == "fused"
                   and s.view_state is not None
                   and s.view_state.decider == "sweep-shortest"
                   for s in steps)

    _wait_for(sweep_step_written, timeout=60.0,
             msg="no sweep-shortest step ever reached phase=fused on disk")

    mock_app_collect.shutdown()

    run = Run.load(mock_app_collect.run_dir)
    assert run.record.source == "data-engine"
    assert any(s.id > 0 and s.view_state is not None
              and s.view_state.decider == "sweep-shortest" for s in run.steps)
    report = run.validate()
    assert report.ok, [(p.severity, p.where, p.what) for p in report.problems]
