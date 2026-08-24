#!/usr/bin/env python3
"""Local verbs over a stub backend. Run: p inspection/tests/test_eyes_verbs.py"""
import numpy as np

from inspection.eyes.tools import ViewImage
from inspection.eyes.verbs_local import (Detection, LocalVerbs, Segment,
                                         StubBackend)

IMG = ViewImage(cell=(3, 1), cap_dir="003",
                rgb=np.zeros((480, 848, 3), np.uint8), text="cell [3, 1]")


def test_detect_sorts_and_clips():
    backend = StubBackend(boxes=[((10, 10, 100, 100), 0.4, "cup"),
                                 ((-20, 5, 900, 500), 0.9, "cup")])
    dets = LocalVerbs(backend).detect(IMG, "cup")
    assert [d.score for d in dets] == [0.9, 0.4]          # highest first
    assert dets[0].box == (0, 5, 848, 480)                # clipped to frame
    assert isinstance(dets[0], Detection) and dets[0].label == "cup"


def test_detect_drops_low_scores():
    backend = StubBackend(boxes=[((0, 0, 10, 10), 0.1, "cup")])
    assert LocalVerbs(backend).detect(IMG, "cup", min_score=0.3) == []


def test_segment_returns_mask_of_frame_size():
    seg = LocalVerbs(StubBackend()).segment(IMG, (10, 10, 100, 100))
    assert isinstance(seg, Segment)
    assert seg.mask.shape == (480, 848) and seg.mask.dtype == bool
    assert seg.mask[50, 50] and not seg.mask[400, 700]     # inside vs outside


def test_module_has_no_torch_at_import():
    import sys
    assert "torch" not in sys.modules      # backends load lazily; stub needs none


def main():
    test_detect_sorts_and_clips(); test_detect_drops_low_scores()
    test_segment_returns_mask_of_frame_size(); test_module_has_no_torch_at_import()
    print("OK test_eyes_verbs")


if __name__ == "__main__":
    main()
