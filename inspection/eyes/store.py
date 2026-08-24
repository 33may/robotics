#!/usr/bin/env python3
"""Run structure for the image machinery — the facts tier.

Three trust levels (design 2026-08-20): deterministic code writes geometry
facts via FactWriter; agents get their own writers (Task 2) that CANNOT
touch views/visited/coverage. Flushed to eyes/store.json on every write.
Never imports motion/run — the image tier does not move the robot.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ViewRecord:
    cell: tuple | None
    pose_id: int
    cap_dir: str
    t: float
    T_base_cam: np.ndarray


class RunStore:
    """Owns the dict + disk flush. Read API only — writes go through writers."""

    def __init__(self, run_dir: Path, data: dict):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "eyes"
        self._d = data

    @classmethod
    def create(cls, run_dir, h_bins, v_elevs, r):
        d = {"grid": {"h_bins": h_bins, "v_elevs": list(v_elevs), "r": r},
             "views": [], "notes": [], "hypothesis": None, "plan": None,
             "findings": []}
        store = cls(run_dir, d)
        store._flush()
        return store

    @classmethod
    def open(cls, run_dir):
        p = Path(run_dir) / "eyes" / "store.json"
        return cls(run_dir, json.loads(p.read_text()))

    def _flush(self):
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "store.json").write_text(
            json.dumps(self._d, indent=1) + "\n")

    def views(self, cell=None):
        out = []
        for v in self._d["views"]:
            c = tuple(v["cell"]) if v["cell"] is not None else None
            if cell is None or c == tuple(cell):
                out.append(ViewRecord(c, v["pose_id"], v["cap_dir"], v["t"],
                                      np.asarray(v["T_base_cam"], float)))
        return out

    def visited(self):
        return {v.cell for v in self.views() if v.cell is not None}

    def coverage(self):
        g = self._d["grid"]
        cov = np.zeros((len(g["v_elevs"]), g["h_bins"]), dtype=bool)
        for (h, v) in self.visited():
            cov[v, h] = True
        return cov


class FactWriter:
    """Deterministic-code tier: the ONLY writer of geometric ground truth."""

    def __init__(self, store: RunStore):
        self._s = store

    def add_view(self, cell, pose_id, cap_dir, T_base_cam, t):
        self._s._d["views"].append(
            {"cell": list(cell) if cell is not None else None,
             "pose_id": int(pose_id), "cap_dir": str(cap_dir),
             "t": float(t),
             "T_base_cam": np.asarray(T_base_cam, float).tolist()})
        self._s._flush()
