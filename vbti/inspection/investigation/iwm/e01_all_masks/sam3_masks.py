"""e01 part 1, SAM 3: all masks from one frame, several prompt strategies.

Strategies
  grid32       point grid 32x32 over the full frame, pipeline default thresholds
  grid64       point grid 64x64 over the full frame
  grid64_loose grid 64x64 with loosened thresholds (iou 0.7, stability 0.85)
  crop32       grid 32x32 over the object crop (recorded box, else SAM 3 text box)
  crop64       grid 64x64 over the object crop
  text         SAM 3 text prompts with the planner's nouns

Output: vbti/inspection/data/iwm/e01_all_masks/masks/<frame>/sam3_<strategy>/
        masks.npz masks.json overlay.png run.json

Run (robo env, GPU):
  p vbti/inspection/investigation/iwm/e01_all_masks/sam3_masks.py --frames c5.6 --strategies grid32
"""
import argparse
import time

import numpy as np
import torch
from PIL import Image

from inspection.investigation.iwm.e01_all_masks import common as C

REPO = "facebook/sam3"
STRATEGIES = {
    "grid32": dict(points_per_crop=32, pred_iou_thresh=0.88, stability_score_thresh=0.95),
    "grid64": dict(points_per_crop=64, pred_iou_thresh=0.88, stability_score_thresh=0.95),
    "grid64_loose": dict(points_per_crop=64, pred_iou_thresh=0.70, stability_score_thresh=0.85),
    "crop32": dict(points_per_crop=32, pred_iou_thresh=0.88, stability_score_thresh=0.95, crop=True),
    "crop64": dict(points_per_crop=64, pred_iou_thresh=0.88, stability_score_thresh=0.95, crop=True),
    "text": dict(text=True, min_score=0.3),
    # points only inside the object, ALL THREE nested masks per point kept
    "obj3_32": dict(obj3=True, n=32, min_iou=0.5, dedupe_iou=0.85),
    "obj3_48": dict(obj3=True, n=48, min_iou=0.5, dedupe_iou=0.85),
}


class Sam3Points:
    """SAM 3 tracker head on explicit points; returns the 3 multimask outputs."""

    def __init__(self, device="cuda"):
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor
        self.proc = Sam3TrackerProcessor.from_pretrained(REPO)
        self.model = Sam3TrackerModel.from_pretrained(REPO, dtype=torch.bfloat16).to(device).eval()
        self.device = device

    def __call__(self, rgb, points, chunk=64):
        """points: list of (x, y). Returns masks (P,3,H,W) bool, scores (P,3)."""
        base = self.proc(images=rgb, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            emb = self.model.get_image_embeddings(base["pixel_values"].to(torch.bfloat16))
        if isinstance(emb, tuple):
            emb = emb[0]
        all_m, all_s = [], []
        for i in range(0, len(points), chunk):
            pts = points[i:i + chunk]
            inp = self.proc(images=rgb, input_points=[[[list(map(float, p))] for p in pts]],
                            input_labels=[[[1] for _ in pts]], return_tensors="pt").to(self.device)
            with torch.inference_mode():
                out = self.model(image_embeddings=emb, input_points=inp["input_points"],
                                 input_labels=inp["input_labels"], multimask_output=True)
            m = self.proc.post_process_masks(out.pred_masks, inp["original_sizes"])[0]
            all_m.append(m.cpu().numpy().astype(bool))
            all_s.append(out.iou_scores[0].float().cpu().numpy())
        return np.concatenate(all_m), np.concatenate(all_s)


def object_mask(f, rgb, sam3_text):
    m, box = C.recorded_object(f)
    if m is not None:
        return m, "recorded"
    phrase = C.OBJECT_PHRASE[f["id"].split(".")[0].rstrip("0123456789")]
    inst = sam3_text.instances(rgb, phrase)
    if not inst:
        raise RuntimeError(f"no object for {f['id']} with '{phrase}'")
    box, score, m = max(inst, key=lambda d: d[1])
    return m, f"sam3 text '{phrase}' score {score:.2f}"


def grid_in_mask(mask, n):
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pts = []
    for y in np.linspace(y0, y1, n):
        for x in np.linspace(x0, x1, n):
            if mask[int(y), int(x)]:
                pts.append((float(x), float(y)))
    return pts


def dedupe(masks, scores, thr, scale=4):
    """Greedy: best score first, drop masks with IoU >= thr to a kept one."""
    order = np.argsort(-scores)
    small = masks[:, ::scale, ::scale].reshape(len(masks), -1)
    kept = []
    kept_small = []
    for i in order:
        s = small[i]
        if kept_small:
            K = np.stack(kept_small)
            inter = (K & s).sum(1)
            union = (K | s).sum(1)
            if (inter / np.maximum(union, 1)).max() >= thr:
                continue
        kept.append(int(i))
        kept_small.append(s)
    return kept


class Sam3Grid:
    """transformers' mask-generation pipeline driven by the SAM 3 tracker head."""

    def __init__(self, device="cuda"):
        from transformers import Sam3TrackerModel, pipeline
        from transformers.models.sam3.image_processing_sam3 import Sam3ImageProcessor
        t = time.time()
        model = Sam3TrackerModel.from_pretrained(REPO, dtype=torch.bfloat16).to(device).eval()
        ip = Sam3ImageProcessor.from_pretrained(REPO)
        # transformers 5.15.1: the final NMS gets bf16 iou scores next to float32
        # boxes and torchvision refuses ("dets should have the same type as scores")
        orig = ip.post_process_for_mask_generation

        def _same_dtype(masks, scores, boxes, thr):
            return orig(masks, scores.float(), boxes.float(), thr)
        ip.post_process_for_mask_generation = _same_dtype
        self.pipe = pipeline("mask-generation", model=model, image_processor=ip,
                             device=device, dtype=torch.bfloat16)
        self.load_s = time.time() - t

    def __call__(self, rgb, points_per_crop, pred_iou_thresh, stability_score_thresh,
                 crops_nms_thresh=0.7, points_per_batch=256):
        out = self.pipe(Image.fromarray(rgb), points_per_batch=points_per_batch,
                        points_per_crop=points_per_crop, pred_iou_thresh=pred_iou_thresh,
                        stability_score_thresh=stability_score_thresh,
                        crops_nms_thresh=crops_nms_thresh)
        masks = [np.asarray(m).astype(bool) for m in out["masks"]]
        scores = [float(s) for s in np.asarray(out["scores"]).ravel()]
        return masks, scores


def object_box(f, rgb, sam3_text):
    m, box = C.recorded_object(f)
    if box is not None:
        return box, "recorded"
    phrase = C.OBJECT_PHRASE[f["id"].split(".")[0].rstrip("0123456789")]
    dets = sam3_text.detect(rgb, phrase)
    if not dets:
        raise RuntimeError(f"no object box for {f['id']} with '{phrase}'")
    box, score, _ = max(dets, key=lambda d: d[1])
    return tuple(int(v) for v in box), f"sam3 text '{phrase}' score {score:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="*")
    ap.add_argument("--strategies", nargs="*", default=list(STRATEGIES))
    args = ap.parse_args()

    grid = None
    text = None
    points = None
    for f in C.frames(args.frames):
        rgb = C.load_rgb(f)
        hw = rgb.shape[:2]
        for name in args.strategies:
            s = STRATEGIES[name]
            out = C.DATA / "masks" / f["id"] / f"sam3_{name}"
            t0 = time.time()
            if s.get("obj3"):
                if text is None:
                    from inspection.eyes.verbs_local import Sam3Backend
                    text = Sam3Backend()
                if points is None:
                    points = Sam3Points()
                obj, how = object_mask(f, rgb, text)
                pts = grid_in_mask(obj, s["n"])
                m3, s3 = points(rgb, pts)                     # (P,3,H,W), (P,3)
                P = len(pts)
                flat_m = m3.reshape(P * 3, *hw)
                flat_s = s3.reshape(P * 3)
                level = np.tile(np.arange(3), P)
                pt_idx = np.repeat(np.arange(P), 3)
                area = flat_m.reshape(P * 3, -1).sum(1)
                ok = (flat_s >= s["min_iou"]) & (area >= 30) & (area <= 1.3 * obj.sum())
                idx = np.nonzero(ok)[0]
                keep = [idx[k] for k in dedupe(flat_m[idx], flat_s[idx], s["dedupe_iou"])]
                masks = [flat_m[k] for k in keep]
                recs = [C.mask_record(flat_m[k], score=round(float(flat_s[k]), 3),
                                      level=int(level[k]), point=list(pts[pt_idx[k]]))
                        for k in keep]
                settings = {"strategy": name, "grid_n": s["n"], "points": P,
                            "min_iou": s["min_iou"], "dedupe_iou": s["dedupe_iou"],
                            "raw_masks": int(P * 3), "after_filter": int(ok.sum())}
                extra = {"object_from": how, "object_area": int(obj.sum())}
                scores = [float(flat_s[k]) for k in keep]
            elif s.get("text"):
                if text is None:
                    from inspection.eyes.verbs_local import Sam3Backend
                    text = Sam3Backend()
                masks, recs = [], []
                for noun in C.PLANNER_NOUNS:
                    for box, score, m in text.instances(rgb, noun):
                        if score >= s["min_score"] and m.sum() > 0:
                            masks.append(m)
                            recs.append(C.mask_record(m, score=round(score, 3), phrase=noun))
                settings = {"strategy": name, "nouns": C.PLANNER_NOUNS, "min_score": s["min_score"]}
                extra = {}
            else:
                if grid is None:
                    grid = Sam3Grid()
                if s.get("crop"):
                    if text is None:
                        from inspection.eyes.verbs_local import Sam3Backend
                        text = Sam3Backend()
                    box, how = object_box(f, rgb, text)
                    x0, y0, x1, y1 = C.expand_box(box, hw)
                    sub = rgb[y0:y1, x0:x1]
                    ms, scores = grid(sub, s["points_per_crop"], s["pred_iou_thresh"],
                                      s["stability_score_thresh"])
                    masks = []
                    for m in ms:
                        full = np.zeros(hw, bool)
                        full[y0:y1, x0:x1] = m
                        masks.append(full)
                    extra = {"crop_box": [x0, y0, x1, y1], "object_box_from": how}
                else:
                    masks, scores = grid(rgb, s["points_per_crop"], s["pred_iou_thresh"],
                                         s["stability_score_thresh"])
                    extra = {}
                recs = [C.mask_record(m, score=round(sc, 3)) for m, sc in zip(masks, scores)]
                settings = {"strategy": name, **{k: v for k, v in s.items() if k != "crop"},
                            "crops_nms_thresh": 0.7, "points_per_batch": 256}
            dt = time.time() - t0
            C.save_masks(out, masks, recs, settings)
            ov = C.overlay(rgb, masks) if masks else rgb.copy()
            if extra.get("crop_box"):
                from PIL import ImageDraw
                im = Image.fromarray(ov)
                ImageDraw.Draw(im).rectangle(extra["crop_box"], outline="red", width=3)
                ov = np.array(im)
            ov = C.label(ov, f"{f['id']}  sam3 {name}  {len(masks)} masks  {dt:.1f}s")
            Image.fromarray(ov).save(out / "overlay.png")
            C.run_record(out, {"id": f["id"], "run": f["run"], "step": f["step"]},
                         {"name": REPO, "dtype": "bfloat16"}, settings,
                         {"seconds": round(dt, 2), **extra})
            print(f"{f['id']:6s} sam3_{name:13s} {len(masks):4d} masks  {dt:5.1f}s")


if __name__ == "__main__":
    main()
