#!/usr/bin/env python3
"""The verb surface — what an agent may ask of a run.

Three things meet here: `view/grid.py` (how cells are named), `RunStore`
(what has actually been seen) and the capture dirs (the pixels). Every
return carries BOTH pixels and text, because which one a VLM reads better is
an empirical question the design leaves to measurement.

The text is deterministic provenance built from geometry — never a model's
words. Nothing in this module can move the robot: it reads captured views
and, at most, records that a view it wanted does not exist.
"""
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from inspection.eyes.store import RunStore
from inspection.view.grid import coverage_map, neighbors


@dataclass(frozen=True)
class ViewImage:
    """Pixels plus their provenance. `box` is set when this is a crop."""
    cell: tuple | None
    cap_dir: str
    rgb: np.ndarray                 # (H, W, 3) uint8, RGB order
    text: str
    box: tuple | None = None


class ViewTools:
    """Verbs over one run. `writer` (optional) receives miss notes."""

    def __init__(self, store: RunStore, writer=None):
        self._store = store
        self._writer = writer
        g = store._d["grid"]
        self._h_bins, self._v_elevs = g["h_bins"], tuple(g["v_elevs"])
        self._r = g["r"]

    # ----------------------------------------------------------- geometry
    def view_at(self, cell):
        """Newest view of a cell, or None. A revisit supersedes the earlier look."""
        vs = self._store.views(cell=cell)
        return max(vs, key=lambda v: v.t) if vs else None

    def views_near(self, cell):
        """[(neighbour, ViewRecord | None)] — 4-connected, h wraps, v clamps.

        An uncaptured neighbour is not an error: it is a next-move hint that
        cost no motion, so it is recorded as a note when a writer is bound.
        """
        out = []
        for c in neighbors(tuple(cell), self._h_bins, len(self._v_elevs)):
            rec = self.view_at(c)
            if rec is None and self._writer is not None:
                self._writer.note(
                    f"wanted neighbour {list(c)} of {list(cell)} "
                    f"— not captured", cell=c)
            out.append((c, rec))
        return out

    def coverage(self, cur=None):
        """ASCII map of what has been seen, plus the count and a legend."""
        cov = self._store.coverage()
        n_cells = cov.shape[0] * cov.shape[1]
        return (f"{int(cov.sum())}/{n_cells} cells seen  "
                f"(# seen · . unseen · @ current)\n"
                + coverage_map(cov, cur=cur, elevations=self._v_elevs))

    # ------------------------------------------------------------- pixels
    def get_view(self, target):
        """Load a view's RGB. `target` is a cell or a ViewRecord."""
        rec = self.view_at(tuple(target)) if isinstance(target, (tuple, list)) \
            else target
        if rec is None:
            raise KeyError(f"no captured view at cell {target}")
        path = self._store.run_dir / rec.cap_dir / "rgb.png"
        import cv2                       # local: keeps pixel-free callers cheap
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        # capture.py:44 writes through RGB2BGR — invert it, or every image the
        # VLM sees has its red and blue channels swapped.
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return ViewImage(rec.cell, rec.cap_dir, rgb, self._gloss(rec))

    def crop(self, img: ViewImage, box):
        """Sub-image; box (x0, y0, x1, y1) is clipped to the frame."""
        h, w = img.rgb.shape[:2]
        x0, y0, x1, y1 = box
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(w, int(x1)), min(h, int(y1))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"empty crop {box} of {w}x{h}")
        return replace(img, rgb=img.rgb[y0:y1, x0:x1], box=(x0, y0, x1, y1),
                       text=f"{img.text} · crop {x0},{y0}-{x1},{y1} "
                            f"({x1 - x0}x{y1 - y0} px)")

    # -------------------------------------------------------------- notes
    def note(self, text, cell=None):
        if self._writer is None:
            raise RuntimeError("no writer bound — this tier may not write")
        self._writer.note(text, cell=cell)

    # ------------------------------------------------------------ helpers
    def _gloss(self, rec):
        """Deterministic provenance for a view — geometry, never opinion."""
        if rec.cell is None:
            return f"survey view · dir {rec.cap_dir}"
        h, v = rec.cell
        return (f"cell [{h}, {v}] · azimuth {h * 360 / self._h_bins:.0f} deg "
                f"(h{h} of {self._h_bins}) · elevation {self._v_elevs[v]:.0f} deg "
                f"· r {self._r:.2f} m · dir {rec.cap_dir}")


if __name__ == "__main__":
    import sys

    from inspection.eyes.replay import load_run
    t = ViewTools(load_run(sys.argv[1]))
    print(t.coverage())
    for cell in sorted(t._store.visited())[:3]:
        print(" ", t._gloss(t.view_at(cell)))
