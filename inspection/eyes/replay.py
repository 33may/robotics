#!/usr/bin/env python3
"""Replay a captured data-engine run into a RunStore.

Join rule (run/machine.py: turn step vs capture pose_id are different
namespaces): dir 000 = survey; dir NNN's cell is the N-th non-survey turn
of run.json. Disk is truth — turns without a dir (failed captures) are
skipped. Debug CLI: p inspection/eyes/replay.py <run_dir>
"""
import json
import sys
from pathlib import Path

from inspection.eyes.store import FactWriter, RunStore


def load_run(run_dir, v_elevs=(10.0, 40.0, 70.0), h_bins=12):
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    cells = [t["target"] for t in run["turns"] if t["target"] != "survey"]

    if (run_dir / "eyes" / "store.json").exists():
        store = RunStore.open(run_dir)
        store._d["views"] = []                 # views: rebuilt from disk truth
    else:
        store = RunStore.create(run_dir, h_bins=h_bins, v_elevs=v_elevs,
                                r=run.get("r", 0.35))
    facts = FactWriter(store)
    for d in sorted(p for p in run_dir.iterdir()
                    if p.is_dir() and p.name.isdigit()):
        meta = json.loads((d / "meta.json").read_text())
        pid = meta["pose_id"]
        cell = None if pid == 0 else tuple(cells[pid - 1])
        facts.add_view(cell=cell, pose_id=pid, cap_dir=d.name,
                       T_base_cam=meta["T_base_cam"], t=meta["timestamp"])
    return store


if __name__ == "__main__":
    s = load_run(sys.argv[1])
    print(f"views {len(s.views())}  visited {sorted(s.visited())}")
    print(s.coverage().astype(int))
