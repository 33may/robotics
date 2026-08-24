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


class StubBackend:
    """Deterministic backend for tests: no models, no GPU, no downloads."""

    def __init__(self, boxes=None):
        self._boxes = boxes if boxes is not None else [((10, 10, 100, 100),
                                                        0.9, "stub")]

    def detect(self, rgb, phrase):
        return list(self._boxes)

    def segment(self, rgb, box):
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        x0, y0, x1, y1 = box
        mask[y0:y1, x0:x1] = True              # the box itself, filled
        return mask, 1.0


class Sam3Backend:
    """facebook/sam3 — text->boxes via Sam3Model, box->mask via Sam3Tracker.

    Both checkpoints load lazily so importing this module stays free for
    callers that never touch pixels (and for the test suite).
    """

    def __init__(self, device="cuda", dtype="bfloat16", repo="facebook/sam3"):
        self.device, self.dtype, self.repo = device, dtype, repo
        self._det = self._det_proc = None
        self._trk = self._trk_proc = None

    def _load_detector(self):
        if self._det is None:
            import torch
            from transformers import AutoProcessor, Sam3Model
            self._det_proc = AutoProcessor.from_pretrained(self.repo)
            self._det = Sam3Model.from_pretrained(
                self.repo, dtype=getattr(torch, self.dtype)).to(self.device).eval()
        return self._det, self._det_proc

    def _load_tracker(self):
        if self._trk is None:
            import torch
            from transformers import AutoProcessor, Sam3TrackerModel
            self._trk_proc = AutoProcessor.from_pretrained(self.repo)
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
