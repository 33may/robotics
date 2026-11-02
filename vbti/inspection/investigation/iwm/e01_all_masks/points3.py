"""e01 explainer figure: what ONE prompt gives.

For a few hand-placed points on c5.6, SAM 3's three nested masks (its
whole / part / sub-part guesses) with their IoU scores. This is the raw
material the point-grid strategies filter; it shows why a plain metal
surface yields blobs and a port yields clean parts.

Output: evidence/c5.6_points3.png
Run:    p inspection/investigation/iwm/e01_all_masks/points3.py
"""
import numpy as np
from PIL import Image, ImageDraw

from inspection.investigation.iwm.e01_all_masks import common as C
from inspection.investigation.iwm.e01_all_masks.sam3_masks import Sam3Points

#: (name, x, y) in c5.6 frame coordinates
POINTS = [
    ("plain metal, top band", 700, 400),
    ("Chort sticker", 330, 368),
    ("letter o in Chort", 319, 370),
    ("RJ45 LAN 1G", 575, 580),
    ("DB9 COM/CAN", 940, 575),
    ("ANT hole", 390, 598),
    ("label text LAN 10G", 662, 530),
    ("audio jack", 455, 595),
]
CROP = (243, 310, 1073, 719)
PANEL_W = 420


def main():
    f = C.frames(["c5.6"])[0]
    rgb = C.load_rgb(f)
    pts = [(x, y) for _, x, y in POINTS]
    sam = Sam3Points()
    m3, s3 = sam(rgb, pts)                      # (P,3,H,W), (P,3)
    x0, y0, x1, y1 = CROP
    rows = []
    for i, (name, x, y) in enumerate(POINTS):
        row = []
        # the prompt
        im = Image.fromarray(rgb[y0:y1, x0:x1].copy())
        d = ImageDraw.Draw(im)
        d.ellipse([x - x0 - 7, y - y0 - 7, x - x0 + 7, y - y0 + 7], fill=(255, 0, 0), outline="white", width=2)
        row.append(C.label(C.resize(np.array(im), PANEL_W), f"prompt: {name}", size=15))
        order = np.argsort([m3[i, k].sum() for k in range(3)])[::-1]   # largest first
        for rank, k in enumerate(order):
            m = m3[i, k][y0:y1, x0:x1]
            ov = C.overlay(rgb[y0:y1, x0:x1], [m], alpha=0.55, colors=[(255, 80, 40)], thick=2)
            ov = C.resize(ov, PANEL_W)
            row.append(C.label(ov, f"mask {rank + 1} of 3  iou {s3[i, k]:.2f}  {int(m3[i, k].sum())} px", size=15))
        rows += row
    img = C.grid(rows, cols=4)
    img = C.title_band(img, "c5.6  SAM 3 (tracker head): one point prompt -> its three masks, largest to smallest, with the model's own IoU estimate")
    Image.fromarray(img).save(C.EVIDENCE / "c5.6_points3.png")
    C.run_record(C.DATA / "points3", {"id": "c5.6"}, {"name": "facebook/sam3 tracker"}, {"points": POINTS})
    print("evidence/c5.6_points3.png")


if __name__ == "__main__":
    main()
