"""localization/testing.py — reusable contract-conformance checker (tasks.md 2.2).

`verify_module_contract` runs any `LocalizationModule` implementation through its lifecycle and
verifies the contract invariants hold. Imported by this repo's tests, the locbench harness, and
candidate test suites — one checker, every host. It raises `ContractViolation` (never bare
`assert`, so it works under `python -O`). Pure stdlib.

`stop()` is called in ALL exit paths, including a failing `start()` — implementations must
tolerate `stop()` after a failed/partial start (idempotent teardown).
"""

from __future__ import annotations

from typing import Iterable, List

from .contracts import LocalizationIn, LocalizationOut, LocalizationSetup, LocalizationStatus
from .module import LocalizationModule


class ContractViolation(AssertionError):
    """A `LocalizationModule` implementation (or the harness feeding it) broke the contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractViolation(message)


def verify_module_contract(
    module: LocalizationModule,
    setup: LocalizationSetup,
    loc_ins: Iterable[LocalizationIn],
) -> List[LocalizationOut]:
    """start → step each `loc_in` → stop, verifying invariants; returns all outputs.

    Checks (contract, not quality — the bench scorer judges accuracy):
      - inputs are stamp-monotonic (guards a broken HARNESS, not the module);
      - every `step` returns a real `LocalizationOut` answering the input's stamp
        (`LocalizationOut` itself enforces the pose⇔LOST invariant and status enum on
        construction — an implementation cannot emit a malformed verdict object);
      - `last_fix_stamp_ns`, when present, never exceeds the answered stamp and never
        goes backwards across steps.
    """
    outs: List[LocalizationOut] = []
    last_stamp = None
    last_fix_seen = None
    try:
        module.start(setup)
        for loc_in in loc_ins:
            _require(
                last_stamp is None or loc_in.stamp_ns > last_stamp,
                f"harness inputs must be stamp-monotonic ({loc_in.stamp_ns} after {last_stamp})",
            )
            last_stamp = loc_in.stamp_ns
            out = module.step(loc_in)
            _require(
                isinstance(out, LocalizationOut),
                f"step returned {type(out).__name__}, not LocalizationOut",
            )
            _require(
                isinstance(out.status, LocalizationStatus),
                f"status escaped the enum: {out.status!r}",
            )
            _require(
                out.stamp_ns == loc_in.stamp_ns,
                f"output stamp {out.stamp_ns} does not answer input stamp {loc_in.stamp_ns}",
            )
            if out.last_fix_stamp_ns is not None:
                _require(
                    out.last_fix_stamp_ns <= out.stamp_ns,
                    f"last_fix_stamp_ns {out.last_fix_stamp_ns} is in the future of {out.stamp_ns}",
                )
                _require(
                    last_fix_seen is None or out.last_fix_stamp_ns >= last_fix_seen,
                    f"last_fix_stamp_ns went backwards ({out.last_fix_stamp_ns} < {last_fix_seen})",
                )
                last_fix_seen = out.last_fix_stamp_ns
            outs.append(out)
    finally:
        module.stop()
    return outs
