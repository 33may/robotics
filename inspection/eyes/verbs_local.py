#!/usr/bin/env python3
"""The inspection subagent's local verbs: detect and segment.

Split on purpose: `LocalVerbs` owns POLICY (score threshold, clipping to the
frame, ordering) and needs no GPU, so it is fully testable on a stub; a
backend owns nothing but model I/O. Swapping SAM 3 for the Apache fallback
(LLMDet-L + SAM 2.1) is then one class, not a rewrite.

SAM 3 notes that cost real time to learn (2026-08-21):
- ONE checkpoint does detect AND segment, but through TWO model classes:
  `Sam3Model.input_boxes` are CONCEPT EXEMPLARS ("find more things like this
  box"), NOT "segment this box" — box->mask must go through
  `Sam3TrackerModel`. Using input_boxes for that returns plausible nonsense.
- Text prompts must be simple noun phrases. It finds "cup", not "the logo on
  the cup", which is why the pipeline is detect -> crop -> OCR.
- Inputs are resized to a square 1008, so our 848x480 is upscaled: good for
  small logos, and the reason latency is worth measuring rather than assuming.

Run (needs weights): p inspection/eyes/verbs_local.py <run_dir> <h> <v>
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Detection:
    box: tuple          # (x0, y0, x1, y1) clipped to the frame
    score: float
    label: str


@dataclass(frozen=True)
class Segment:
    mask: np.ndarray    # (H, W) bool, frame-sized
    box: tuple
    score: float


@dataclass(frozen=True)
class TextLine:
    text: str
    score: float
    box: tuple          # (x0, y0, x1, y1) enclosing the quad
    quad: tuple         # 4 (x, y) corners as returned by the detector


class LocalVerbs:
    """Verb policy over any backend. No torch here — see the backends."""

    def __init__(self, backend):
        self._b = backend

    def detect(self, img, phrase, min_score=0.3):
        """Boxes for a simple noun phrase, best first, clipped to the frame."""
        h, w = img.rgb.shape[:2]
        out = []
        for box, score, label in self._b.detect(img.rgb, phrase):
            if score < min_score:
                continue
            x0, y0, x1, y1 = box
            box = (max(0, int(x0)), max(0, int(y0)),
                   min(w, int(x1)), min(h, int(y1)))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue                       # entirely outside the frame
            out.append(Detection(box, float(score), str(label)))
        return sorted(out, key=lambda d: -d.score)

    def segment(self, img, box):
        """Mask for a given box. Frame-sized so masks compose across verbs."""
        mask, score = self._b.segment(img.rgb, tuple(int(v) for v in box))
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != img.rgb.shape[:2]:
            raise ValueError(f"backend returned {mask.shape}, "
                             f"expected {img.rgb.shape[:2]}")
        return Segment(mask, tuple(int(v) for v in box), float(score))

    def read_text(self, img, min_score=0.5):
        """Text lines in reading order (top-to-bottom, then left-to-right).

        `min_score` is not decoration. On a cup with no legible text the
        detector still proposes a region and the recogniser still returns a
        character — measured on run 2408-seeded: '三' 0.14, '2' 0.37, 'O' 0.49.
        Everything below the threshold is the model describing noise, and
        passing that to a VLM as "text found on the object" would invent
        evidence.
        """
        out = []
        for quad, text, score in self._b.read_text(img.rgb):
            if score < min_score or not str(text).strip():
                continue
            pts = np.asarray(quad, dtype=float).reshape(-1, 2)
            x0, y0 = pts.min(0)
            x1, y1 = pts.max(0)
            out.append(TextLine(str(text), float(score),
                                (int(x0), int(y0), int(x1), int(y1)),
                                tuple(map(tuple, pts.tolist()))))
        return sorted(out, key=lambda t: (t.box[1], t.box[0]))


class StubBackend:
    """Deterministic backend for tests: no models, no GPU, no downloads."""

    def __init__(self, boxes=None, lines=None):
        self._boxes = boxes if boxes is not None else [((10, 10, 100, 100),
                                                        0.9, "stub")]
        self._lines = lines or []

    def detect(self, rgb, phrase):
        return list(self._boxes)

    def segment(self, rgb, box):
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        x0, y0, x1, y1 = box
        mask[y0:y1, x0:x1] = True              # the box itself, filled
        return mask, 1.0

    def read_text(self, rgb):
        return list(self._lines)


class Sam3Backend:
    """facebook/sam3 — text->boxes via Sam3Model, box->mask via Sam3Tracker.

    Both checkpoints load lazily so importing this module stays free for
    callers that never touch pixels (and for the test suite).
    """

    def __init__(self, device="cuda", dtype="bfloat16", repo="facebook/sam3"):
        self.device, self.dtype, self.repo = device, dtype, repo
        self._det = self._det_proc = None
        self._trk = self._trk_proc = None

    # NOT AutoProcessor: for this repo it resolves to Sam3VideoProcessor, whose
    # image path rejects `text=` outright (Sam3ImageProcessorKwargs has no such
    # field). The still-image classes must be named explicitly.
    def _load_detector(self):
        if self._det is None:
            import torch
            from transformers import Sam3Model, Sam3Processor
            self._det_proc = Sam3Processor.from_pretrained(self.repo)
            self._det = Sam3Model.from_pretrained(
                self.repo, dtype=getattr(torch, self.dtype)).to(self.device).eval()
        return self._det, self._det_proc

    def _load_tracker(self):
        if self._trk is None:
            import torch
            from transformers import Sam3TrackerModel, Sam3TrackerProcessor
            self._trk_proc = Sam3TrackerProcessor.from_pretrained(self.repo)
            self._trk = Sam3TrackerModel.from_pretrained(
                self.repo, dtype=getattr(torch, self.dtype)).to(self.device).eval()
        return self._trk, self._trk_proc

    def detect(self, rgb, phrase):
        import torch
        model, proc = self._load_detector()
        inputs = proc(images=rgb, text=phrase, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = model(**inputs)
        res = proc.post_process_instance_segmentation(
            out, threshold=0.0, target_sizes=[rgb.shape[:2]])[0]
        return [(tuple(b.tolist()), float(s), phrase)
                for b, s in zip(res["boxes"], res["scores"])]

    def segment(self, rgb, box):
        import torch
        model, proc = self._load_tracker()
        # NOT Sam3Model.input_boxes — those are concept exemplars, not a
        # "segment this region" instruction.
        inputs = proc(images=rgb, input_boxes=[[list(box)]],
                      return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = model(**inputs)
        masks = proc.post_process_masks(out.pred_masks,
                                        inputs["original_sizes"])[0]
        scores = out.iou_scores.flatten()
        best = int(scores.argmax())
        return masks[0][best].cpu().numpy().astype(bool), float(scores[best])


class PaddleOcrBackend:
    """PP-OCRv6 through paddlex's TRANSFORMERS engine — never paddle's own.

    Two independent reasons the paddle engine is off-limits here: it needs the
    `paddlepaddle` runtime we deliberately do not install, and on Blackwell it
    dies with cudaErrorLaunchFailure after a handful of images (PaddleOCR
    #18235) — precisely the shape of failure a long inspection run would hit.

    Correction to the earlier research note (verified here 2026-08-24, paddlex
    3.7.2): the transformers engine has **no `box_type` parameter at all**, so
    POLYGON output is not available — it always returns 4-point quads. Polygon
    postprocessing lives only in the paddle DB path. Quads are looser than
    polygons on curved text, which for a cup means a crop carries some
    background; that is acceptable for crop-then-read and is the thing to
    revisit if curved logos read badly.

    Measured on the 5090 (PP-OCRv6_medium, 848x480, after warm-up): detect
    ~12 ms, recognise ~4 ms per crop. First call is ~200 ms — warm it once.
    """

    def __init__(self, device="gpu", size="medium"):
        self.device, self.size = device, size
        self._det = self._rec = None

    def _load(self):
        if self._det is None:
            from paddlex.inference.models import create_predictor
            self._det = create_predictor(model_name=f"PP-OCRv6_{self.size}_det",
                                         engine="transformers", device=self.device)
            self._rec = create_predictor(model_name=f"PP-OCRv6_{self.size}_rec",
                                         engine="transformers", device=self.device)
        return self._det, self._rec

    def read_text(self, rgb):
        det, rec = self._load()
        res = list(det([rgb]))[0]
        quads = res["dt_polys"]
        crops, keep = [], []
        for q in quads:
            pts = np.asarray(q, dtype=float).reshape(-1, 2)
            x0, y0 = np.floor(pts.min(0)).astype(int)
            x1, y1 = np.ceil(pts.max(0)).astype(int)
            crop = rgb[max(0, y0):y1, max(0, x0):x1]
            if crop.size:
                crops.append(crop)
                keep.append(pts)
        if not crops:
            return []
        out = list(rec(crops))
        return [(pts, o.get("rec_text", ""), float(o.get("rec_score", 0.0)))
                for pts, o in zip(keep, out)]


if __name__ == "__main__":
    import sys
    import time

    from inspection.eyes.replay import load_run
    from inspection.eyes.tools import ViewTools

    run_dir, h, v = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    tools = ViewTools(load_run(run_dir))
    img = tools.get_view((h, v))
    verbs = LocalVerbs(Sam3Backend())

    t0 = time.time(); dets = verbs.detect(img, "cup"); t_det = time.time() - t0
    print(f"{img.text}\ndetect 'cup': {len(dets)} in {t_det * 1000:.0f} ms")
    for d in dets[:5]:
        print(f"  {d.score:.3f}  {d.box}")
    if dets:
        t0 = time.time(); seg = verbs.segment(img, dets[0].box)
        print(f"segment: {int(seg.mask.sum())} px "
              f"({seg.mask.mean() * 100:.1f}% of frame) "
              f"in {(time.time() - t0) * 1000:.0f} ms")
