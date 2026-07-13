"""localization/testing.py — reusable contract-conformance checker (tasks.md 2.2).

`verify_module_contract` runs any `LocalizationModule` implementation through its lifecycle and
asserts the contract invariants hold. Imported by this repo's tests, the locbench harness, and
candidate test suites — one checker, every host. Pure stdlib."""

from __future__ import annotations

from typing import Iterable, List

from .contracts import LocalizationIn, LocalizationOut, LocalizationSetup
from .module import LocalizationModule


def verify_module_contract(
    module: LocalizationModule,
    setup: LocalizationSetup,
    loc_ins: Iterable[LocalizationIn],
) -> List[LocalizationOut]:
    """start → step each `loc_in` → stop, asserting invariants; returns all outputs.

    Checks (contract, not quality — the bench scorer judges accuracy):
      - inputs are stamp-monotonic (guards a broken HARNESS, not the module);
      - every `step` returns a `LocalizationOut` answering the input's stamp;
      - the pose⇔status invariant holds by construction (`LocalizationOut` validates itself);
      - `last_fix_stamp_ns`, when present, never exceeds the answered stamp.
    """
    module.start(setup)
    outs: List[LocalizationOut] = []
    last_stamp = None
    try:
        for loc_in in loc_ins:
            assert last_stamp is None or loc_in.stamp_ns > last_stamp, (
                f"harness inputs must be stamp-monotonic ({loc_in.stamp_ns} after {last_stamp})"
            )
            last_stamp = loc_in.stamp_ns
            out = module.step(loc_in)
            assert isinstance(out, LocalizationOut), f"step returned {type(out).__name__}"
            assert out.stamp_ns == loc_in.stamp_ns, (
                f"output stamp {out.stamp_ns} does not answer input stamp {loc_in.stamp_ns}"
            )
            assert out.last_fix_stamp_ns is None or out.last_fix_stamp_ns <= out.stamp_ns, (
                f"last_fix_stamp_ns {out.last_fix_stamp_ns} is in the future of {out.stamp_ns}"
            )
            outs.append(out)
    finally:
        module.stop()
    return outs
