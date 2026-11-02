#!/usr/bin/env python
"""Kill everything the toolkit started, using the tracked PID file.

    python cleanup.py            # kill tracked pids, clear session
    python cleanup.py --purge    # also delete screenshots/logs in $UITEST_DIR

Deliberately PID-file based.  Do NOT reach for `pkill -f rerun` here: the
pattern also matches the shell that is running the cleanup (and, in an agent
harness, the agent's own wrapper process), which kills the caller.  That has
bitten us; see README "Traps".
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cdp as C  # noqa: E402

KEEP = {"pids", "session.json"}


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--purge", action="store_true", help="also delete artifacts")
    args = ap.parse_args()

    report = {"killed": [], "already_dead": [], "failed": []}

    if C.PIDS_FILE.exists():
        entries = []
        for line in C.PIDS_FILE.read_text().splitlines():
            if not line.strip():
                continue
            pid_s, _, label = line.partition("\t")
            try:
                entries.append((int(pid_s), label or "?"))
            except ValueError:
                continue

        # newest first: kill the browser before the server it talks to
        for pid, label in reversed(entries):
            if not alive(pid):
                report["already_dead"].append({"pid": pid, "label": label})
                continue
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except Exception:
                try:
                    os.kill(pid, signal.SIGTERM)
                except Exception:
                    report["failed"].append({"pid": pid, "label": label})
                    continue
            report["killed"].append({"pid": pid, "label": label})

        time.sleep(1.0)
        for pid, label in entries:  # SIGKILL stragglers
            if alive(pid):
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except Exception:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except Exception:
                        pass

        C.PIDS_FILE.unlink()

    if C.SESSION_FILE.exists():
        C.SESSION_FILE.unlink()

    if args.purge and C.ART.exists():
        for p in C.ART.iterdir():
            if p.name in KEEP:
                continue
            shutil.rmtree(p) if p.is_dir() else p.unlink()
        report["purged"] = str(C.ART)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
