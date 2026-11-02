"""e01 shared bits: frames, mask storage, drawing, run records. Throwaway code."""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RUNS_DIR = ROOT / "inspection" / "data" / "runs"
DATA = ROOT / "inspection" / "data" / "iwm" / "e01_all_masks"
EVIDENCE = HERE / "evidence"

#: what the planner asked about across the 2209 runs (ai/000/trace.jsonl, inspect calls)
PLANNER_NOUNS = ["sticker", "label", "logo", "text", "barcode", "port",
                 "connector", "LED", "button", "hole", "screw"]
#: noun used to find the object when the recording has no mask for the step
OBJECT_PHRASE = {"c": "metal box", "b": "cardboard box", "bc": "box", "sc": "box"}


def frames(ids=None):
    sel = json.load(open(EVIDENCE / "frames_selected.json"))["frames"]
    if ids:
        sel = [f for f in sel if f["id"] in ids]
    return sel


def frame_paths(f):
    d = RUNS_DIR / f["run"] / "steps" / f"{f['step']:03d}"
    return {"rgb": d / "rgb.png", "mask": d / "mask.png", "step": d / "step.json",
            "depth": d / "depth_aligned.npy"}


def load_rgb(f):
    return np.array(Image.open(frame_paths(f)["rgb"]).convert("RGB"))


def recorded_object(f):
    """(mask bool HxW, box x0y0x1y1) from the recording, or (None, None)."""
    p = frame_paths(f)
    step = json.load(open(p["step"]))
    seg = step.get("segmentation")
    if not seg or not p["mask"].exists():
        return None, None
    m = np.array(Image.open(p["mask"])) > 0
    # NOT seg["box"]: on 180-degree views it is in raw-capture coordinates while
    # rgb.png / mask.png on disk are already turned upright. Take the mask's extent.
    ys, xs = np.nonzero(m)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return m, box


def expand_box(box, hw, margin=0.08):
    x0, y0, x1, y1 = box
    h, w = hw
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    return (max(0, int(x0 - mx)), max(0, int(y0 - my)),
            min(w, int(x1 + mx)), min(h, int(y1 + my)))


# ---------------------------------------------------------------- masks on disk

def save_masks(out, masks, records, settings):
    """masks: list of bool HxW; records: list of dicts (area, bbox, score...)."""
    out.mkdir(parents=True, exist_ok=True)
    if masks:
        arr = np.stack(masks).astype(bool)
    else:
        arr = np.zeros((0, 720, 1280), bool)
    np.savez_compressed(out / "masks.npz", packed=np.packbits(arr, axis=-1),
                        shape=np.array(arr.shape))
    json.dump({"n_masks": len(masks), "masks": records, "settings": settings},
              open(out / "masks.json", "w"), indent=1)


def save_packed(path, masks):
    arr = np.stack(masks).astype(bool)
    np.savez_compressed(path, packed=np.packbits(arr, axis=-1), shape=np.array(arr.shape))


def load_packed(path):
    z = np.load(path)
    n, h, w = [int(v) for v in z["shape"]]
    return np.unpackbits(z["packed"], axis=-1)[..., :w].astype(bool).reshape(n, h, w)


def load_masks(d):
    """Either our packed npz or UnSAMv2's raw masks.npy; records normalised to
    {area, bbox [x0,y0,x1,y1], score, ...}."""
    meta = json.load(open(d / "masks.json"))
    if (d / "masks.npz").exists():
        z = np.load(d / "masks.npz")
        n, h, w = [int(v) for v in z["shape"]]
        arr = np.unpackbits(z["packed"], axis=-1)[..., :w].astype(bool).reshape(n, h, w)
        recs = meta["masks"]
    else:
        arr = np.load(d / "masks.npy").astype(bool)
        recs = []
        for r in meta["masks"]:
            x, y, w_, h_ = r["bbox_xywh"]
            recs.append({"area": r["area"], "bbox": [int(x), int(y), int(x + w_), int(y + h_)],
                         "score": round(float(r["predicted_iou"]), 3),
                         "stability": round(float(r["stability_score"]), 3),
                         "point": r.get("point_xy")})
        meta = {"n_masks": len(recs), "masks": recs, "settings": meta["settings"]}
    return arr, meta


def mask_record(m, **extra):
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        box = [0, 0, 0, 0]
    else:
        box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    return {"area": int(m.sum()), "bbox": box, **extra}


def run_record(out, frame, model, settings, extra=None):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    rec = {"date": datetime.now().isoformat(timespec="seconds"), "commit": commit,
           "command": " ".join(sys.argv), "frame": frame, "model": model,
           "settings": settings}
    if extra:
        rec.update(extra)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(out / "run.json", "w"), indent=1)


# ---------------------------------------------------------------- drawing

def _palette(n, seed=0):
    rng = np.random.default_rng(seed)
    hsv = np.stack([rng.permutation(n) / max(n, 1), np.full(n, 0.65), np.full(n, 0.95)], 1)
    import colorsys
    return [tuple(int(255 * c) for c in colorsys.hsv_to_rgb(*row)) for row in hsv]


def _contour(mask):
    """Boolean edge of a mask (1 px, 4-neighbour)."""
    m = mask
    e = np.zeros_like(m)
    e[1:, :] |= m[1:, :] & ~m[:-1, :]
    e[:-1, :] |= m[:-1, :] & ~m[1:, :]
    e[:, 1:] |= m[:, 1:] & ~m[:, :-1]
    e[:, :-1] |= m[:, :-1] & ~m[:, 1:]
    return e


def overlay(rgb, masks, alpha=0.45, seed=0, colors=None, outline=True, thick=1):
    """Fill every mask with its own colour, small masks drawn last (on top)."""
    img = rgb.astype(np.float32).copy()
    order = np.argsort([-m.sum() for m in masks])
    cols = colors or _palette(len(masks), seed)
    for i in order:
        m = masks[i]
        c = np.array(cols[i], np.float32)
        img[m] = img[m] * (1 - alpha) + c * alpha
    if outline:
        for i in order:
            e = _contour(masks[i])
            if thick > 1:
                from scipy.ndimage import binary_dilation
                e = binary_dilation(e, iterations=thick - 1)
            img[e] = np.array(cols[i]) * 0.6
    return img.clip(0, 255).astype(np.uint8)


def outlines(rgb, masks, colors, thick=2, dim=0.35):
    """Only outlines, colour per mask (used for depth-coloured nesting)."""
    img = rgb.astype(np.float32) * (1 - dim) + 255 * dim
    from scipy.ndimage import binary_dilation
    order = np.argsort([-m.sum() for m in masks])
    for i in order:
        e = _contour(masks[i])
        if thick > 1:
            e = binary_dilation(e, iterations=thick - 1)
        img[e] = colors[i]
    return img.clip(0, 255).astype(np.uint8)


def label(img, text, pos=(6, 6), size=18, fill="white", bg="black"):
    im = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    d = ImageDraw.Draw(im)
    font = _font(size)
    box = d.textbbox(pos, text, font=font)
    d.rectangle([box[0] - 4, box[1] - 2, box[2] + 4, box[3] + 2], fill=bg)
    d.text(pos, text, fill=fill, font=font)
    return np.array(im)


def _font(size):
    for p in ["/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/TTF/DejaVuSans.ttf"]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def grid(panels, cols, pad=6, bg=(255, 255, 255)):
    """Tile equally sized uint8 panels into one image."""
    h, w = panels[0].shape[:2]
    rows = (len(panels) + cols - 1) // cols
    out = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad, 3), bg, np.uint8)
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        y, x = pad + r * (h + pad), pad + c * (w + pad)
        out[y:y + h, x:x + w] = p
    return out


def title_band(img, text, size=22, h=36):
    """A white band with the title above the image (no overlap with panels)."""
    band = np.full((h, img.shape[1], 3), 255, np.uint8)
    band = label(band, text, pos=(8, 6), size=size, fill="black", bg="white")
    return np.concatenate([band, img], 0)


def resize(img, w):
    im = Image.fromarray(img)
    h = int(round(im.height * w / im.width))
    return np.array(im.resize((w, h), Image.BILINEAR))
