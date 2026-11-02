"""e07 drawing: stable node colours, masks over the frame, trees as node-link diagrams."""
import colorsys

import numpy as np
from PIL import Image, ImageDraw

from inspection.investigation.iwm.e01_all_masks import common as C


def colour(i, salt=0):
    """Stable colour per id (golden-ratio hue), the same in the 2D overlay, the tree and the 3D."""
    h = ((int(i) + 7 * salt) * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


def overlay(rgb, masks_by_id, focus=None, alpha=0.45):
    """Masks over the frame, deepest drawn last (callers pass them parent-first); focus = white outline."""
    out = rgb.astype(np.float32) * 0.6
    for i, m in masks_by_id.items():
        c = np.array(colour(i), np.float32)
        out[m] = out[m] * (1 - alpha) + c * alpha
    out = out.clip(0, 255).astype(np.uint8)
    for i, m in masks_by_id.items():
        out[edge(m)] = colour(i)
    if focus is not None and focus in masks_by_id:
        out[edge(masks_by_id[focus], 3)] = (255, 255, 255)
    return out


def edge(m, thick=1):
    from scipy.ndimage import binary_erosion
    return m & ~binary_erosion(m, iterations=thick)


def tree_image(nodes, root, label, tag=None, focus=None, salt=0, col_w=170, row_h=22, max_chars=26, title=None):
    """Left-to-right node-link diagram: depth = column, one row per node in depth-first order,
    elbow links parent -> child. nodes: id -> {"children": [...]}; label(id) -> str; tag(id) -> str|None."""
    order, depth = [], {}

    def walk(i, d):
        order.append(i)
        depth[i] = d
        for c in nodes[i]["children"]:
            if c in nodes:
                walk(c, d + 1)

    walk(root, 0)
    font = C._font(13)
    top = 26 if title else 6
    W = col_w * (max(depth.values()) + 1) + 220
    H = top + row_h * len(order) + 6
    im = Image.new("RGB", (W, H), (22, 24, 30))
    d = ImageDraw.Draw(im)
    if title:
        d.text((8, 5), title, fill=(230, 230, 230), font=C._font(15))
    pos = {i: (8 + depth[i] * col_w, top + k * row_h) for k, i in enumerate(order)}
    for i in order:
        x, y = pos[i]
        for c in nodes[i]["children"]:
            if c in pos:
                cx, cy = pos[c]
                d.line([(x + 8, y + row_h // 2), (x + 8, cy + row_h // 2), (cx, cy + row_h // 2)], fill=(120, 124, 135), width=1)
    for i in order:
        x, y = pos[i]
        txt = label(i)
        txt = txt if len(txt) <= max_chars else txt[:max_chars - 1] + "…"
        t = tag(i) if tag else None
        tb = d.textbbox((x + 18, y + 3), txt, font=font)
        fill = colour(i, salt)
        d.rectangle([x, y + 3, x + 14, y + row_h - 4], fill=fill)
        if focus is not None and i == focus:
            d.rectangle([tb[0] - 3, tb[1] - 2, tb[2] + 3, tb[3] + 2], outline=(255, 255, 255), width=2)
        d.text((x + 18, y + 3), txt, fill=(235, 235, 235), font=font)
        if t:
            d.text((tb[2] + 8, y + 3), t, fill=(255, 210, 90), font=font)
    return np.array(im)
