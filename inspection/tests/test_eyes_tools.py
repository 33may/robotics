#!/usr/bin/env python3
"""Verb surface over the run structure. Run: p inspection/tests/test_eyes_tools.py"""
import tempfile
from pathlib import Path

import numpy as np

from inspection.eyes.store import FactWriter, FindingWriter, RunStore
from inspection.eyes.tools import ViewImage, ViewTools

T = np.eye(4)


def _rig(tmp):
    """Store with two views of (3,1) — a revisit — plus a neighbour at (4,1)."""
    root = Path(tmp)
    store = RunStore.create(root, h_bins=12, v_elevs=(10.0, 40.0, 70.0), r=0.35)
    facts = FactWriter(store)
    for pid, (cell, name) in enumerate([((3, 1), "001"), ((4, 1), "002"),
                                        ((3, 1), "003")], start=1):
        d = root / name; d.mkdir()
        img = np.zeros((480, 848, 3), np.uint8)
        img[:, :, 0] = pid * 10                      # RED channel marks the dir
        import cv2
        cv2.imwrite(str(d / "rgb.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        facts.add_view(cell=cell, pose_id=pid, cap_dir=name, T_base_cam=T,
                       t=float(pid))
    return store


def test_view_at_returns_newest():
    with tempfile.TemporaryDirectory() as tmp:
        tools = ViewTools(_rig(tmp))
        assert len(tools._store.views(cell=(3, 1))) == 2      # both kept
        assert tools.view_at((3, 1)).cap_dir == "003"         # newest wins
        assert tools.view_at((9, 0)) is None                  # never captured


def test_views_near_hits_and_logs_misses():
    with tempfile.TemporaryDirectory() as tmp:
        store = _rig(tmp)
        tools = ViewTools(store, writer=FindingWriter(store))
        near = dict(tools.views_near((3, 1)))
        assert set(near) == {(2, 1), (4, 1), (3, 0), (3, 2)}  # 4-connected
        assert near[(4, 1)].cap_dir == "002"                  # captured
        assert near[(2, 1)] is None                           # not captured
        misses = [n for n in store.notes() if "not captured" in n["text"]]
        assert len(misses) == 3 and misses[0]["who"] == "finding"


def test_get_view_pixels_are_rgb_not_bgr():
    with tempfile.TemporaryDirectory() as tmp:
        img = ViewTools(_rig(tmp)).get_view((4, 1))
        assert isinstance(img, ViewImage) and img.rgb.shape == (480, 848, 3)
        assert img.rgb[0, 0, 0] == 20 and img.rgb[0, 0, 2] == 0   # RED, not blue
        assert "cell [4, 1]" in img.text and "dir 002" in img.text


def test_crop_clips_and_records():
    with tempfile.TemporaryDirectory() as tmp:
        tools = ViewTools(_rig(tmp))
        c = tools.crop(tools.get_view((4, 1)), (800, 400, 900, 500))  # over edge
        assert c.rgb.shape == (80, 48, 3)          # clipped to 848x480
        assert c.box == (800, 400, 848, 480) and "crop" in c.text


def test_coverage_text():
    with tempfile.TemporaryDirectory() as tmp:
        txt = ViewTools(_rig(tmp)).coverage(cur=(3, 1))
        header, bitmap = txt.split("\n", 1)         # header carries the legend
        assert "2/36" in header
        assert bitmap.count("#") == 1               # (4,1) seen...
        assert bitmap.count("@") == 1               # ...(3,1) is where we are


def test_seeded_run_if_present():
    real = Path("inspection/data/runs/2408-seeded")
    if not (real / "run.json").exists():
        print("  (skip: 2408-seeded not present)"); return
    from inspection.eyes.replay import load_run
    tools = ViewTools(load_run(real))
    assert len(tools._store.views()) == 25 and len(tools._store.visited()) == 24
    cell = sorted(tools._store.visited())[0]
    assert tools.get_view(cell).rgb.shape == (480, 848, 3)
    print(tools.coverage(cur=cell))


def main():
    test_view_at_returns_newest(); test_views_near_hits_and_logs_misses()
    test_get_view_pixels_are_rgb_not_bgr(); test_crop_clips_and_records()
    test_coverage_text(); test_seeded_run_if_present()
    print("OK test_eyes_tools")


if __name__ == "__main__":
    main()
