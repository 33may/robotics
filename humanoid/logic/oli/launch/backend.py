"""backend.py — the launcher's backend plugin contract.

A backend is a small module describing HOW to boot Oli against one world. Each backend
module (under `backends/`) exposes:

    NAME: str                              # the --sim value, e.g. "isaac"
    add_args(ap: ArgumentParser) -> None   # backend-specific CLI flags
    stages(a: Namespace) -> list[Stage]    # the ordered processes to boot
    reap(label: str) -> int                # OPTIONAL: kill leftover bus procs

The generic `Supervisor` consumes the returned Stages and stays backend-agnostic, so
adding a world (real, another sim) is one new module + one line in `backends/__init__`.
This Protocol just documents that contract for type-checkers; the registry is the plain
dict in `backends/__init__.py`.
"""

from __future__ import annotations

import argparse
from typing import Protocol, runtime_checkable

from .supervisor import Stage

__all__ = ["Backend", "Stage"]


@runtime_checkable
class Backend(Protocol):
    NAME: str

    def add_args(self, ap: argparse.ArgumentParser) -> None: ...

    def stages(self, a: argparse.Namespace) -> list[Stage]: ...
