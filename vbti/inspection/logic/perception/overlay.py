#!/usr/bin/env python3
"""Diagnostic renderings of the identity decision, written beside a capture.

Four images per view — the chain the operator sees in the UI's `chain` tab and
the record left on disk for offline review:

    rgb.png            what the camera saw
    chain_prompt.png   the cloud so far, reprojected + the box built from it
    chain_mask.png     the mask the segmenter returned inside that box
    chain_kept.png     the points that actually entered the object cloud

Together they answer "why did this view contribute THAT?" without a rerun —
the question that cost two runs (2408-cup1, 2408-cup2) to answer by hand.

Drawing only. Takes plain arrays, decides nothing, and imports no model: the
identity policy is `run/segmenter.py`, and the geometry it renders comes from
`cell/geometry.py`. Everything is written in the SAME orientation as
`rgb.png` — callers hold raw-orientation arrays, and the half-turn correction
happens here exactly as `save_bundle` applies it.
"""
import logging
from pathlib import Path

import cv2
import numpy as np

from inspection.cell.geometry import is_half_turn, project_to_pixels, rotate180

log = logging.getLogger(__name__)

PROMPT_RGB = (0, 120, 255)      # reprojected cloud — the prompt's evidence
BOX_RGB = (255, 32, 32)         # the box handed to the segmenter
MASK_RGB = (0, 255, 0)          # what the model called object
KEPT_RGB = (255, 210, 0)        # points that survived into the cloud
MASK_ALPHA = 0.55
DOT = 1                         # half-width of a plotted point, px

#: File name per chain stage. `rgb` is the capture's own image, not rewritten.
CHAIN_FILES = {"prompt": "chain_prompt.png",
               "mask": "chain_mask.png",
               "kept": "chain_kept.png"}


def _dots(img, uv, colour, half=DOT):
    """Plot projected points as small squares. Clipped, no allocation per dot."""
    h, w = img.shape[:2]
    for u, v in np.asarray(uv, dtype=int):
        if 0 <= u < w and 0 <= v < h:
            img[max(0, v - half):v + half + 1,
                max(0, u - half):u + half + 1] = colour


def _box(img, box, colour=BOX_RGB, t=2):
    x0, y0, x1, y1 = (int(v) for v in box)
    img[y0:y0 + t, x0:x1] = img[max(0, y1 - t):y1, x0:x1] = colour
    img[y0:y1, x0:x0 + t] = img[y0:y1, max(0, x1 - t):x1] = colour


def _write(d: Path, name: str, img, flip: bool) -> None:
    img = rotate180(img) if flip else img
    cv2.imwrite(str(d / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def save_chain(capture_dir, rgb, T_base_cam, intr, *, source_points=None,
               box=None, mask=None, kept_points=None) -> dict:
    """Write the chain overlays for one view. Returns {stage: filename}.

    Every stage is optional: a view that fell back to depth growth has no box
    and no mask, and rendering what DID happen is more useful than rendering
    nothing. Returns only the stages actually written.

    `source_points` is the cloud that produced the prompt (base frame);
    `kept_points` is what the view contributed after masking and gating.
    """
    d = Path(capture_dir)
    if not d.is_dir() or rgb is None:
        return {}
    flip = T_base_cam is not None and is_half_turn(T_base_cam)
    written: dict[str, str] = {}
    try:
        if source_points is not None and len(source_points):
            img = np.ascontiguousarray(rgb).copy()
            _dots(img, project_to_pixels(source_points, T_base_cam, intr),
                  PROMPT_RGB)
            if box is not None:
                _box(img, box)
            _write(d, CHAIN_FILES["prompt"], img, flip)
            written["prompt"] = CHAIN_FILES["prompt"]

        if mask is not None:
            img = np.ascontiguousarray(rgb).copy()
            m = np.asarray(mask, dtype=bool)
            img[m] = ((1 - MASK_ALPHA) * img[m]
                      + MASK_ALPHA * np.array(MASK_RGB)).astype(np.uint8)
            if box is not None:
                _box(img, box)
            _write(d, CHAIN_FILES["mask"], img, flip)
            written["mask"] = CHAIN_FILES["mask"]

        if kept_points is not None and len(kept_points):
            img = np.ascontiguousarray(rgb).copy()
            _dots(img, project_to_pixels(kept_points, T_base_cam, intr),
                  KEPT_RGB)
            _write(d, CHAIN_FILES["kept"], img, flip)
            written["kept"] = CHAIN_FILES["kept"]
    except Exception:
        # Diagnostics must never cost a capture: whatever was written stays,
        # the rest is skipped, and the run carries on.
        log.exception("chain overlay rendering failed in %s", d)
    return written
