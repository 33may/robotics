#!/usr/bin/env python3
"""Which models a run sees with — chosen by name, never fallen into.

Two ways to see, and a caller picks one explicitly: `real_cognition()` or
`stub_cognition()`. Nothing here reads an environment variable to decide, and
no caller may write `vlm or StubVlm(...)` (Anton 2026-08-26). The old shape did
exactly that in three places, and the failure it produces is on record: in run
`2508-aiduck` the only view that exhausted its 16-turn budget spent it trying
to reconcile real pixels with a stub detector that answers every phrase with a
box at (10, 10, 100, 100) (`inspect_agent.py:46`). A demo that degrades into
that silently looks like a model failure and is not one.

So a live run that cannot see fails at boot, before the arm moves, and a mock
run is visibly a mock: `label` goes into the trace, so a screenshot of a
stubbed run cannot be mistaken for a real one.
"""
import os
from dataclasses import dataclass

from inspection.eyes.models import GeminiVlm, StubVlm
from inspection.eyes.verbs_local import LocalVerbs, SamOcrBackend, StubBackend

#: One entry is spent per inspection: the stub replies with an answer rather
#: than a tool call, so `inspect_view` returns on its first turn. A Brain run is
#: capped at 40 orchestrator turns (`brain/loop.py`, MAX_TURNS), and a caller
#: that builds the cognition once at boot keeps popping from the same list
#: across questions (`models.py:37`) — three runs' worth, so a mock session does
#: not trail off into "stub exhausted" while someone is watching it.
STUB_TURNS = 120

#: Says stub in the words the operator reads, not only in `label`.
STUB_REPLY = {"evidence": ["stub vision: no model looked at this frame"],
              "reasoning": "stub cognition — nothing here was actually seen",
              "answer": "unknown"}


@dataclass(frozen=True)
class Cognition:
    """The models one run thinks with, and what to call them in the trace."""

    vlm: object         # anything with .respond(parts, schema) — eyes/models.py
    verbs: object       # a LocalVerbs over some backend
    label: str          # "real" | "stub"; the trace and the UI show it


def real_cognition(sam3=None):
    """Gemini + SAM 3/OCR — the only cognition a live run may hold.

    The key is checked HERE rather than left to the first model call: a run
    that cannot see should die at boot, not half a minute in with the arm out
    over the table. Weights are deliberately NOT touched — both halves of
    `SamOcrBackend` stay lazy, so this costs boot nothing.

    Pass `sam3` when the caller already has one: `run/app.py` gives the same
    instance to `ObjectSegmenter` (`run/segmenter.py:77`), and loading the SAM 3
    checkpoint onto the GPU twice is the whole reason this argument exists.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY is not set — a live run has no vision without it. "
            "Export the key, or ask for stub_cognition() by name.")
    return Cognition(vlm=GeminiVlm(),
                     verbs=LocalVerbs(SamOcrBackend(sam3=sam3)),
                     label="real")


def stub_cognition(script=None):
    """No models at all: no GPU, no weights, no network, no API key.

    Reachable only by calling it by name — the mock and the tests, where what
    is being exercised is the wiring (approval gate, trace, store) and not the
    seeing.
    """
    if script is None:
        script = [dict(STUB_REPLY) for _ in range(STUB_TURNS)]
    return Cognition(vlm=StubVlm(script), verbs=LocalVerbs(StubBackend()),
                     label="stub")
