#!/usr/bin/env python3
"""Local verbs over a stub backend. Run: p vbti/inspection/tests/test_eyes_verbs.py"""
import numpy as np

from inspection.eyes.tools import ViewImage
from inspection.eyes.verbs_local import (Detection, LocalVerbs,
                                         PaddleOcrBackend, Sam3Backend,
                                         SamOcrBackend, Segment, StubBackend,
                                         TextLine)

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


def test_read_text_filters_noise_and_reads_in_order():
    # The low-score entries are REAL measurements from run 2408-seeded, where
    # the cup carries no legible text — the detector proposes a region anyway.
    lines = [([[400, 300], [440, 300], [440, 330], [400, 330]], "三", 0.14),
             ([[10, 200], [90, 200], [90, 230], [10, 230]], "LOGO", 0.93),
             ([[10, 100], [80, 100], [80, 130], [10, 130]], "ACME", 0.88)]
    got = LocalVerbs(StubBackend(lines=lines)).read_text(IMG)
    assert [t.text for t in got] == ["ACME", "LOGO"]      # reading order, noise gone
    assert isinstance(got[0], TextLine)
    assert got[0].box == (10, 100, 80, 130) and len(got[0].quad) == 4


def test_read_text_drops_blank_recognitions():
    lines = [([[0, 0], [10, 0], [10, 10], [0, 10]], "   ", 0.99)]
    assert LocalVerbs(StubBackend(lines=lines)).read_text(IMG) == []


def test_module_has_no_torch_at_import():
    import sys
    assert "torch" not in sys.modules      # backends load lazily; stub needs none


# ── SamOcrBackend: the two model halves behind one verb surface ──────────────

class _FakeSam:
    """Detect/segment only. The OCR verb raises so a mis-route cannot pass."""

    def __init__(self):
        self.calls = []

    def detect(self, rgb, phrase):
        self.calls.append(("detect", phrase))
        return [((10, 10, 100, 100), 0.9, phrase)]

    def segment(self, rgb, box):
        self.calls.append(("segment", box))
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        mask[box[1]:box[3], box[0]:box[2]] = True
        return mask, 0.8

    def read_text(self, rgb):
        raise AssertionError("read_text belongs to the OCR half")


class _FakeOcr:
    def __init__(self):
        self.calls = []

    def read_text(self, rgb):
        self.calls.append("read_text")
        return [([[10, 100], [80, 100], [80, 130], [10, 130]], "ACME", 0.88)]

    def detect(self, rgb, phrase):
        raise AssertionError("detect belongs to the SAM 3 half")


def test_sam_ocr_backend_routes_each_verb_to_its_half():
    """The whole point of the class: one object that answers all three verbs
    `inspect_agent._dispatch` calls (inspect_agent.py:167,171,175)."""
    sam, ocr = _FakeSam(), _FakeOcr()
    verbs = LocalVerbs(SamOcrBackend(sam3=sam, ocr=ocr))
    assert [d.label for d in verbs.detect(IMG, "cup")] == ["cup"]
    assert verbs.segment(IMG, (10, 10, 100, 100)).score == 0.8
    assert [t.text for t in verbs.read_text(IMG)] == ["ACME"]
    assert [c[0] for c in sam.calls] == ["detect", "segment"]
    assert ocr.calls == ["read_text"]


def test_sam_ocr_backend_reuses_an_injected_sam3():
    """`run/app.py` already holds a Sam3Backend for `ObjectSegmenter`
    (run/segmenter.py:77); a second one would put the checkpoint on the GPU
    twice, so the instance is shared and readable back off the composite."""
    sam = Sam3Backend()
    b = SamOcrBackend(sam3=sam)
    assert b.sam3 is sam
    assert isinstance(b.ocr, PaddleOcrBackend)      # the half not injected


def test_sam_ocr_backend_builds_its_halves_without_loading_them():
    """Constructing the default must stay free — no GPU, no weights, no
    downloads — or importing the eyes tier stops being cheap."""
    b = SamOcrBackend()
    assert isinstance(b.sam3, Sam3Backend)
    assert b.sam3._det is None and b.sam3._trk is None and b.ocr._det is None


def test_warm_racing_first_use_loads_each_checkpoint_once():
    """`Sam3Backend.warm()` runs in a background thread at app start; a
    survey finishing mid-warm must not double-load (GPU memory x2) or see a
    half-built pair. Fake transformers module, real lock, real race."""
    import sys
    import threading
    import time
    import types

    class _Fake:
        calls = 0
        _gate = threading.Lock()

        @classmethod
        def from_pretrained(cls, repo, **kw):
            with cls._gate:
                cls.calls += 1
            time.sleep(0.05)          # wide window for the race
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

    fakes = {}
    mod = types.ModuleType("transformers")
    for name in ("Sam3Model", "Sam3Processor",
                 "Sam3TrackerModel", "Sam3TrackerProcessor"):
        fakes[name] = type(name, (_Fake,), {"calls": 0})
        setattr(mod, name, fakes[name])
    real = sys.modules.get("transformers")
    sys.modules["transformers"] = mod
    try:
        backend = Sam3Backend(device="cpu", dtype="float32")
        threads = [threading.Thread(target=backend.warm),
                   threading.Thread(target=backend._load_detector),
                   threading.Thread(target=backend._load_tracker)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for name, cls in fakes.items():
            assert cls.calls == 1, f"{name} loaded {cls.calls}x under race"
        det, det_proc = backend._load_detector()
        trk, trk_proc = backend._load_tracker()
        assert det is not None and det_proc is not None
        assert trk is not None and trk_proc is not None
        assert fakes["Sam3Model"].calls == 1    # the re-read loaded nothing
    finally:
        if real is not None:
            sys.modules["transformers"] = real
        else:
            del sys.modules["transformers"]


def main():
    test_detect_sorts_and_clips(); test_detect_drops_low_scores()
    test_segment_returns_mask_of_frame_size()
    test_read_text_filters_noise_and_reads_in_order()
    test_read_text_drops_blank_recognitions()
    test_module_has_no_torch_at_import()
    test_sam_ocr_backend_routes_each_verb_to_its_half()
    test_sam_ocr_backend_reuses_an_injected_sam3()
    test_sam_ocr_backend_builds_its_halves_without_loading_them()
    test_warm_racing_first_use_loads_each_checkpoint_once()
    print("OK test_eyes_verbs")


if __name__ == "__main__":
    main()
