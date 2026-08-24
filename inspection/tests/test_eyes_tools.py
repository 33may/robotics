#!/usr/bin/env python3
"""Verb surface over the run structure. Run: p inspection/tests/test_eyes_tools.py"""
import tempfile
from pathlib import Path

import numpy as np

from inspection.eyes.store import FactWriter, FindingWriter, RunStore
from inspection.eyes.tools import (ViewImage, ViewTools, image_tilt_deg,
                                   upright)

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


def _pose(tilt_deg):
    """T_base_cam whose image-up sits `tilt_deg` from world-up.

    Built the long way on purpose — from an UPRIGHT camera basis that is then
    rolled — so it checks `image_tilt_deg` against an independently
    constructed pose rather than against the same arithmetic rearranged.
    Boresight is horizontal (base +X); upright there means image-right = -Y
    and image-down = -Z.
    """
    a = np.radians(-tilt_deg)                        # roll sign is opposite
    x0, y0 = np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, -1.0])
    T = np.eye(4)
    T[:3, 0] = np.cos(a) * x0 + np.sin(a) * y0       # image-right
    T[:3, 1] = -np.sin(a) * x0 + np.cos(a) * y0      # image-down
    T[:3, 2] = [1.0, 0.0, 0.0]                       # boresight
    return T


def test_tilt_is_read_from_the_pose():
    for want in (0.0, 180.0, -90.0, 30.0):
        got = image_tilt_deg(_pose(want))
        assert abs((got - want + 180) % 360 - 180) < 1e-6, (want, got)


def test_half_turn_is_flipped_and_stays_landscape():
    img = np.zeros((480, 848, 3), np.uint8)
    img[0, 0] = [7, 8, 9]                            # a corner to follow
    out, note = upright(img, _pose(180.0))
    assert out.shape == img.shape                    # landscape preserved
    assert tuple(out[-1, -1]) == (7, 8, 9)           # corner went diagonal
    assert "flipped" in note


def test_upright_capture_is_untouched():
    img = np.random.default_rng(0).integers(0, 255, (480, 848, 3), dtype=np.uint8)
    out, note = upright(img, _pose(0.0))
    assert out is img and note == ""


def test_upright_on_disk_is_not_flipped_again():
    """The capture writer stores the colour frame upright since 2026-08-24.
    Flipping it again on read would put every new half-turn view back upside
    down — the failure this flag exists to prevent."""
    img = np.zeros((480, 848, 3), np.uint8)
    img[0, 0] = [7, 8, 9]
    out, note = upright(img, _pose(180.0), stored_rotation_deg=180)
    assert out is img and tuple(out[0, 0]) == (7, 8, 9)
    assert "upright on disk" in note
    # an OLD run has no such key -> None -> correct it now, from the pose
    out_old, note_old = upright(img, _pose(180.0), stored_rotation_deg=None)
    assert tuple(out_old[-1, -1]) == (7, 8, 9) and "flipped" in note_old


def test_capture_writes_the_colour_frame_upright():
    """save_bundle rotates rgb + depth_aligned together and records the flag,
    while depth_raw stays RAW so the reconstruction path is untouched."""
    from inspection.perception.capture import save_bundle
    import json as _json
    rgb = np.zeros((480, 848, 3), np.uint8); rgb[0, 0] = [1, 2, 3]
    daligned = np.zeros((480, 848), np.uint16); daligned[0, 0] = 77
    draw = np.zeros((480, 848), np.uint16); draw[0, 0] = 99
    bundle = {"rgb": rgb, "ir_left": np.zeros((480, 848), np.uint8),
              "ir_right": np.zeros((480, 848), np.uint8),
              "depth_raw": draw, "depth_aligned": daligned, "timestamp": 1.0}
    with tempfile.TemporaryDirectory() as tmp:
        d = save_bundle(Path(tmp), 1, bundle,
                        {"joints_rad": np.zeros(6), "T_base_flange": np.eye(4),
                         "T_base_cam": _pose(180.0)})
        meta = _json.loads((d / "meta.json").read_text())
        assert meta["rgb_rotation_deg"] == 180
        import cv2
        got = cv2.cvtColor(cv2.imread(str(d / "rgb.png")), cv2.COLOR_BGR2RGB)
        assert tuple(got[-1, -1]) == (1, 2, 3)              # rgb rotated
        assert np.load(d / "depth_aligned.npy")[-1, -1] == 77   # and with it
        assert np.load(d / "depth_raw.npy")[0, 0] == 99        # raw untouched


def test_odd_tilt_is_reported_not_rotated():
    """A quarter turn is lossless but would hand the VLM a portrait frame, and
    the taught survey pose is off by a few degrees. Both must be flagged rather
    than silently corrected — see `upright`."""
    img = np.zeros((480, 848, 3), np.uint8)
    for tilt in (-90.0, -16.0):
        out, note = upright(img, _pose(tilt))
        assert out.shape == img.shape and "not corrected" in note


def main():
    test_view_at_returns_newest(); test_views_near_hits_and_logs_misses()
    test_get_view_pixels_are_rgb_not_bgr(); test_crop_clips_and_records()
    test_coverage_text(); test_tilt_is_read_from_the_pose()
    test_half_turn_is_flipped_and_stays_landscape()
    test_upright_capture_is_untouched(); test_odd_tilt_is_reported_not_rotated()
    test_upright_on_disk_is_not_flipped_again()
    test_capture_writes_the_colour_frame_upright()
    test_seeded_run_if_present()
    print("OK test_eyes_tools")


if __name__ == "__main__":
    main()
