#!/usr/bin/env python3
"""The named cognition factories — no GPU, no weights, no network, no key.

Run: p inspection/tests/test_cognition.py
"""
import contextlib
import os
import sys

from inspection.eyes.cognition import (Cognition, real_cognition,
                                       stub_cognition)
from inspection.eyes.models import GeminiVlm, StubVlm
from inspection.eyes.verbs_local import (LocalVerbs, Sam3Backend,
                                         SamOcrBackend, StubBackend)


@contextlib.contextmanager
def _key(value):
    """Set or unset GEMINI_API_KEY for the block, restoring whatever was there.

    A developer machine has the key exported, so a test that only unset it
    would pass for the wrong reason on CI and fail here — and one that leaked
    the change would silently disarm every later test.
    """
    had = os.environ.get("GEMINI_API_KEY")
    try:
        if value is None:
            os.environ.pop("GEMINI_API_KEY", None)
        else:
            os.environ["GEMINI_API_KEY"] = value
        yield
    finally:
        if had is None:
            os.environ.pop("GEMINI_API_KEY", None)
        else:
            os.environ["GEMINI_API_KEY"] = had


def test_real_cognition_without_a_key_stops_the_run():
    """The whole decision in one assertion (Anton 2026-08-26): a live run with
    no vision must fail at boot, not quietly swap a stub in behind the arm."""
    with _key(None):
        try:
            real_cognition()
        except RuntimeError as e:
            assert "GEMINI_API_KEY" in str(e)
        else:
            raise AssertionError("a missing key returned a cognition")


def test_real_cognition_is_sam_plus_ocr_and_loads_nothing():
    """Eager on the key, lazy on the weights: boot must not slow down, and the
    injected SAM 3 must be the caller's own instance rather than a second
    checkpoint on the GPU."""
    sam3 = Sam3Backend()
    with _key("not-a-real-key-nothing-here-calls-out"):
        cog = real_cognition(sam3=sam3)
    assert isinstance(cog, Cognition) and cog.label == "real"
    assert isinstance(cog.vlm, GeminiVlm) and cog.vlm._client is None
    assert isinstance(cog.verbs, LocalVerbs)
    assert isinstance(cog.verbs._b, SamOcrBackend)
    assert cog.verbs._b.sam3 is sam3
    assert sam3._det is None and sam3._trk is None
    assert cog.verbs._b.ocr._det is None
    assert "torch" not in sys.modules and "paddlex" not in sys.modules


def test_stub_cognition_is_stub_backed_and_says_so():
    cog = stub_cognition()
    assert cog.label == "stub"                  # what the trace panel shows
    assert isinstance(cog.vlm, StubVlm)
    assert isinstance(cog.verbs._b, StubBackend)


def test_stub_script_outlasts_a_mock_run():
    """One entry per inspection against a 40-turn orchestrator budget, and the
    list is popped destructively across questions (models.py:37)."""
    cog = stub_cognition()
    assert len(cog.vlm.script) >= 80
    assert cog.vlm.respond(["anything"])["answer"] == "unknown"


def test_stub_cognition_takes_an_injected_script():
    """Explicit injection is the ONLY door to a stub — including the words it
    says, so a test can drive the loop through a chosen sequence."""
    cog = stub_cognition([{"answer": "yes"}])
    assert cog.vlm.respond(["anything"]) == {"answer": "yes"}


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("OK test_cognition")


if __name__ == "__main__":
    main()
