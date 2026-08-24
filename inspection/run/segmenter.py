#!/usr/bin/env python3
"""Object mask for one captured view — the identity half of the geometry.

Why this exists (run 2408-cup2, 2026-08-24): every guard the loop had reasons
about DISTANCE — seeded growth at 20 mm, the 8 cm jump gate, a percentile
extent. On view 003 a cable passed behind the cup within 20 mm, so it was
genuinely adjacent and no distance rule could refuse it; the object box grew
175.8 mm across and the cloud stopped being the cup. Appearance settles in one
call what distance cannot settle at all: masked, the same run measures
89.1 mm.

Where it sits. `cell/geometry.py` may not import `eyes` — `eyes/tools.py`
already imports `cell.geometry`, so it would be a cycle — and `perception/`
is barred from `eyes` too. `run/` is the one tier above both, so the model
lives here, geometry stays pure numpy, and the mask is passed DOWN as data.

Frames. Callers work in RAW capture orientation throughout (that is what
`T_base_cam`, `depth_raw` and `depth_aligned` in memory all agree with);
rotation happens only at disk-write time. Given the pose, this module
presents an upright frame to the model and rotates the mask back, so the
box in and the mask out are both in the caller's frame.
"""
import logging
from dataclasses import dataclass

import numpy as np

from inspection.cell.geometry import (is_half_turn, object_in_base, prompt_box,
                                      rotate180)

log = logging.getLogger(__name__)

#: Below this the model is guessing rather than finding. Measured on
#: 2408-cup2 the five real views scored 0.73-0.97, so this floor is well
#: clear of anything the object itself produced.
MIN_MASK_SCORE = 0.5
#: A mask this small is not our object at 848x480 — the survey mask, the
#: smallest of the run because it is the most distant view, was 7628 px.
MIN_MASK_PX = 200


@dataclass(frozen=True)
class Segmentation:
    mask: np.ndarray        # (H, W) bool, caller's frame
    box: tuple              # prompt box that produced it, caller's frame
    score: float


class _Frame:
    """`LocalVerbs` only ever reads `.rgb`; this avoids importing ViewImage
    (and with it the whole eyes store) into the run tier."""

    __slots__ = ("rgb",)

    def __init__(self, rgb):
        self.rgb = rgb


class ObjectSegmenter:
    """Box -> object mask, with the policy that decides when to believe it.

    `backend` is injected so tests run with `eyes.verbs_local.StubBackend` and
    never touch a GPU. The default backend is built lazily on first use, so
    importing this module stays free for callers that never segment.
    """

    def __init__(self, backend=None, min_score=MIN_MASK_SCORE,
                 min_px=MIN_MASK_PX):
        self._backend = backend
        self._verbs = None
        self.min_score, self.min_px = float(min_score), int(min_px)
        self.calls = self.misses = 0

    def _lazy(self):
        if self._verbs is None:
            from inspection.eyes.verbs_local import LocalVerbs, Sam3Backend
            self._verbs = LocalVerbs(self._backend or Sam3Backend())
        return self._verbs

    def mask_for(self, rgb, box, T_base_cam=None) -> Segmentation | None:
        """Segment the object inside `box`. None means "do not trust this".

        A miss is an ordinary outcome, not an error: the caller falls back to
        the depth-only path rather than fusing a mask it cannot vouch for. A
        backend that raises is treated the same way — a model failure must
        never take the run down with it.
        """
        if rgb is None or box is None:
            return None
        self.calls += 1
        h, w = rgb.shape[:2]
        flip = T_base_cam is not None and is_half_turn(T_base_cam)
        if flip:
            # Rotate INTO the model's preferred orientation, not out of it:
            # SAM 3 tolerates rotation but not aspect change, and a half turn
            # keeps the frame landscape (eyes/AGENTS.md, measured 2026-08-24).
            x0, y0, x1, y1 = box
            frame, prompt = rotate180(rgb), (w - x1, h - y1, w - x0, h - y0)
        else:
            frame, prompt = rgb, tuple(box)
        try:
            seg = self._lazy().segment(_Frame(frame), prompt)
        except Exception:
            log.exception("segmentation backend failed on box %s", prompt)
            self.misses += 1
            return None
        mask = rotate180(seg.mask) if flip else seg.mask
        n = int(mask.sum())
        if seg.score < self.min_score or n < self.min_px:
            log.warning("mask rejected: score %.2f (min %.2f), %d px (min %d)",
                        seg.score, self.min_score, n, self.min_px)
            self.misses += 1
            return None
        return Segmentation(np.asarray(mask, dtype=bool), tuple(box),
                            float(seg.score))


def object_view(cap, intr, intr_color, depth_scale, seed, segmenter,
                on_fallback=None):
    """One capture -> (view, Segmentation | None). The whole identity policy.

    Lives here rather than in the Supervisor so that the offline replay
    (`investigation/mask_probe.py`) exercises the SAME code the robot runs —
    a probe that reimplements the pipeline measures the probe, not the loop.

    `cap` is the rig contract dict: rgb / depth_raw / depth_aligned /
    T_base_cam, all in RAW capture orientation.

    Box source is the only thing that differs between the survey and later
    views: the accumulated cloud once there is one, this view's own depth
    cluster before that. Everything after the box is a single path.
    """
    depth_view = object_in_base(cap["depth_raw"], intr, depth_scale,
                                cap["T_base_cam"], seed=seed)
    if segmenter is None:
        return depth_view, None

    def _fell_back(why):
        if on_fallback is not None:
            on_fallback(why)
        return depth_view, None

    rgb = cap.get("rgb")
    if rgb is None:
        return _fell_back("capture carried no rgb — fell back to depth growth")
    source = seed if seed is not None and len(seed) else depth_view["points"]
    box = prompt_box(source, cap["T_base_cam"], intr_color, np.shape(rgb)[:2])
    # Recorded on the view so the chain overlay can draw the evidence the box
    # was built from. On the survey that is this frame's own depth cluster,
    # which nothing else keeps a handle on.
    depth_view["prompt_source"], depth_view["prompt_box"] = source, box
    if box is None:
        return _fell_back("object not visible in frame — fell back to depth growth")
    seg = segmenter.mask_for(rgb, box, cap["T_base_cam"])
    if seg is None:
        return _fell_back("no object mask — fell back to depth growth")
    # The mask is computed on rgb, so it must be applied to the depth sharing
    # the colour viewport: `depth_aligned` with the colour intrinsics.
    masked = object_in_base(cap["depth_aligned"], intr_color, depth_scale,
                            cap["T_base_cam"], mask=seg.mask)
    if masked["centroid"] is None:
        return _fell_back("object mask lifted no depth — fell back to depth growth")
    masked["prompt_source"], masked["prompt_box"] = source, box
    return masked, seg
