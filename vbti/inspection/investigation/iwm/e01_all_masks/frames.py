"""e01 helper: catalogue the recorded frames we can choose from.

For each run, one contact sheet (evidence/frames_<run>.jpg) with every captured
step: thumbnail, step id, view address, camera-to-object distance, object size in
the frame. Plus evidence/frames_catalogue.json with the same numbers.

Run:  p inspection/investigation/iwm/e01_all_masks/frames.py
Reads recordings only; writes to evidence/.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[4]
RUNS_DIR = ROOT / "inspection" / "data" / "runs"
OUT = Path(__file__).resolve().parent / "evidence"

RUNS = [
    "2209-compute-ray-demo5",
    "2209-compute-ray-demo9",
    "2209-compute-ray-demo11",
    "2209-box-ray-demo1",
    "2209-box-ray-demo4",
    "2209-bigcam-ray-demo5",
    "2209-smallcam-ray-demo5",
]
THUMB_W = 320
COLS = 4
#: short run ids for the combined sheet: frame id = <run id>.<step>
RUN_ID = {
    "2209-compute-ray-demo5": "c5",
    "2209-compute-ray-demo9": "c9",
    "2209-compute-ray-demo11": "c11",
    "2209-box-ray-demo1": "b1",
    "2209-box-ray-demo4": "b4",
    "2209-bigcam-ray-demo5": "bc5",
    "2209-smallcam-ray-demo5": "sc5",
}


def object_center(run_dir, step_dir):
    """Object centre in base frame: this step's decomposition, else step 000's."""
    for d in (step_dir, run_dir / "steps" / "000"):
        f = d / "decomposition.json"
        if f.exists():
            prims = json.load(open(f)).get("primitives") or []
            if prims:
                return np.array(prims[0]["pose"])[:3, 3]
    return None


def catalogue(run):
    run_dir = RUNS_DIR / run
    rows = []
    for step_dir in sorted((run_dir / "steps").iterdir()):
        sj = step_dir / "step.json"
        if not sj.exists() or not (step_dir / "rgb.png").exists():
            continue
        s = json.load(open(sj))
        cam = np.array(s["T_base_cam"])[:3, 3]
        ctr = object_center(run_dir, step_dir)
        seg = s.get("segmentation") or {}
        box = seg.get("box")
        rows.append({
            "id": f"{RUN_ID[run]}.{s['step_id']}",
            "run": run,
            "step": s["step_id"],
            "address": s["view"]["address"],
            "dist_cm": None if ctr is None else round(float(np.linalg.norm(cam - ctr)) * 100, 1),
            "object_px_share": None if not seg.get("px") else round(seg["px"] / (1280 * 720), 3),
            "object_box_wh": None if not box else [box[2] - box[0], box[3] - box[1]],
            "seg_score": seg.get("score"),
            "rgb_rotation_deg": s.get("rgb_rotation_deg", 0),
            "rgb": str((step_dir / "rgb.png").relative_to(ROOT)),
        })
    return rows


def sheet(run, rows):
    font = ImageFont.load_default()
    th = THUMB_W * 720 // 1280
    label_h = 44
    n = len(rows)
    r = (n + COLS - 1) // COLS
    img = Image.new("RGB", (COLS * THUMB_W, r * (th + label_h) + 24), "white")
    d = ImageDraw.Draw(img)
    d.text((6, 6), run, fill="black", font=font)
    for i, row in enumerate(rows):
        x = (i % COLS) * THUMB_W
        y = 24 + (i // COLS) * (th + label_h)
        im = Image.open(ROOT / row["rgb"]).convert("RGB").resize((THUMB_W, th))
        img.paste(im, (x, y))
        a = row["address"]
        a = ":".join(map(str, a)) if isinstance(a, list) else str(a)
        d.text((x + 4, y + th + 2),
               f"step {row['step']}  view {a}", fill="black", font=font)
        d.text((x + 4, y + th + 16),
               f"dist {row['dist_cm']} cm  obj {row['object_px_share']} of frame",
               fill="black", font=font)
        d.text((x + 4, y + th + 30),
               f"box {row['object_box_wh']}  rot {row['rgb_rotation_deg']}",
               fill="black", font=font)
    img.save(OUT / f"frames_{run}.jpg", quality=88)


def sheet_all(rows, cols=6, thumb_w=256):
    """One sheet with every frame of every run, labelled by frame id."""
    font = ImageFont.load_default()
    th = thumb_w * 720 // 1280
    label_h = 30
    r = (len(rows) + cols - 1) // cols
    img = Image.new("RGB", (cols * thumb_w, r * (th + label_h)), "white")
    d = ImageDraw.Draw(img)
    for i, row in enumerate(rows):
        x = (i % cols) * thumb_w
        y = (i // cols) * (th + label_h)
        im = Image.open(ROOT / row["rgb"]).convert("RGB").resize((thumb_w, th))
        img.paste(im, (x, y))
        a = row["address"]
        a = ":".join(map(str, a)) if isinstance(a, list) else str(a)
        d.rectangle([x, y, x + 46, y + 14], fill="yellow")
        d.text((x + 3, y + 1), row["id"], fill="black", font=font)
        d.text((x + 4, y + th + 2), f"{row['id']}  {a}", fill="black", font=font)
        d.text((x + 4, y + th + 15),
               f"{row['dist_cm']} cm  obj {row['object_px_share']}",
               fill="black", font=font)
    img.save(OUT / "frames_all.jpg", quality=88)


def main():
    OUT.mkdir(exist_ok=True)
    allrows = []
    for run in RUNS:
        rows = catalogue(run)
        if not rows:
            print(run, "no frames")
            continue
        sheet(run, rows)
        allrows += rows
        print(f"{run}: {len(rows)} frames -> evidence/frames_{run}.jpg")
        for row in rows:
            print(f"   step {row['step']:2d} {str(row['address']):20s} "
                  f"{row['dist_cm']!s:>6} cm  obj {row['object_px_share']!s:>6}  "
                  f"box {row['object_box_wh']}")
    sheet_all(allrows)
    print(f"all {len(allrows)} frames -> evidence/frames_all.jpg")
    json.dump(allrows, open(OUT / "frames_catalogue.json", "w"), indent=1)


if __name__ == "__main__":
    main()
