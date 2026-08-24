#!/usr/bin/env python3
"""ObjectSegmenter policy, on stubs — no GPU, no weights, no downloads.

Run: p inspection/tests/test_segmenter.py
"""
import numpy as np

from inspection.eyes.verbs_local import StubBackend
from inspection.run.segmenter import ObjectSegmenter

RGB = np.zeros((480, 848, 3), dtype=np.uint8)

# A side-looking camera, built the long way so the test checks the flip
# against an independently constructed pose rather than against two matrix
# entries chosen to satisfy it. cv frame: +X image-right, +Y image-down,
# +Z boresight. Boresight along world +X, image-up along world +Z.
_UPRIGHT = np.eye(4)
_UPRIGHT[:3, 0] = [0, -1, 0]        # image right
_UPRIGHT[:3, 1] = [0, 0, -1]        # image down  -> image-up is world-up
_UPRIGHT[:3, 2] = [1, 0, 0]         # boresight

_HALF_TURN = np.eye(4)              # the same camera rolled 180 deg
_HALF_TURN[:3, 0] = [0, 1, 0]
_HALF_TURN[:3, 1] = [0, 0, 1]
_HALF_TURN[:3, 2] = [1, 0, 0]


class _ScoreStub:
    def __init__(self, score, box=None):
        self.score, self.box = score, box

    def segment(self, rgb, box):
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        x0, y0, x1, y1 = self.box or box
        mask[y0:y1, x0:x1] = True
        return mask, self.score


class _AngryStub:
    def segment(self, rgb, box):
        raise RuntimeError("CUDA out of memory")


def _seg(backend=None, **kw):
    return ObjectSegmenter(backend=backend or StubBackend(), **kw)


def test_mask_returns_in_the_callers_frame():
    box = (300, 150, 500, 350)
    out = _seg().mask_for(RGB, box)
    assert out is not None and out.box == box and out.score == 1.0
    assert out.mask.shape == RGB.shape[:2] and out.mask.dtype == bool
    assert out.mask[150:350, 300:500].all()
    assert out.mask.sum() == 200 * 200          # nothing outside the box


def test_half_turn_pose_round_trips_the_mask():
    """The stub fills exactly the box it is handed, so if the flip were
    dropped — or applied once instead of twice — the mask would come back
    rotated about the frame centre and this would catch it."""
    box = (300, 150, 500, 350)
    out = _seg().mask_for(RGB, box, T_base_cam=_HALF_TURN)
    assert out is not None
    assert out.mask[150:350, 300:500].all()     # back in the caller's frame
    assert out.mask.sum() == 200 * 200
    assert out.box == box                       # reported box is the input one


def test_upright_pose_does_not_flip():
    box = (300, 150, 500, 350)
    a = _seg().mask_for(RGB, box, T_base_cam=_UPRIGHT)
    b = _seg().mask_for(RGB, box)
    assert np.array_equal(a.mask, b.mask)


def test_asymmetric_box_proves_the_flip_is_a_real_rotation():
    """A box off-centre in BOTH axes: a half turn maps it somewhere else
    entirely, so a no-op flip cannot pass this."""
    box = (100, 60, 240, 130)
    out = _seg().mask_for(RGB, box, T_base_cam=_HALF_TURN)
    ys, xs = np.nonzero(out.mask)
    assert (xs.min(), xs.max() + 1) == (100, 240)
    assert (ys.min(), ys.max() + 1) == (60, 130)


def test_low_score_is_a_miss():
    s = _seg(_ScoreStub(0.2))
    assert s.mask_for(RGB, (300, 150, 500, 350)) is None
    assert s.misses == 1


def test_tiny_mask_is_a_miss():
    """Score alone is not enough: a confident five-pixel mask is still not
    our object, and fusing it would shrink the collision box."""
    s = _seg(_ScoreStub(0.99, box=(10, 10, 13, 13)))
    assert s.mask_for(RGB, (10, 10, 13, 13)) is None
    assert s.misses == 1


def test_backend_exception_is_a_miss_not_a_crash():
    """A model failure must not take the run down — the caller falls back to
    the depth-only path and keeps orbiting."""
    s = _seg(_AngryStub())
    assert s.mask_for(RGB, (300, 150, 500, 350)) is None
    assert s.misses == 1 and s.calls == 1


def test_missing_inputs_return_none_without_calling_the_model():
    s = _seg(_AngryStub())                      # would raise if it were called
    assert s.mask_for(RGB, None) is None
    assert s.mask_for(None, (1, 2, 3, 4)) is None
    assert s.calls == 0


def test_backend_is_not_constructed_until_first_use():
    """Lazy on purpose: importing the run tier must not pull in torch."""
    s = ObjectSegmenter()
    assert s._verbs is None


# ── chain overlays ──────────────────────────────────────────────────────────

def test_chain_writes_only_the_stages_that_happened():
    """A view that fell back to depth growth has no mask; rendering what DID
    happen beats rendering nothing, and the caller must be able to tell which
    stages exist so the panel can say "not produced"."""
    import tempfile
    from pathlib import Path
    from inspection.perception.overlay import save_chain
    from inspection.tests.synth import INTR, T_BASE_CAM
    rgb = np.zeros((480, 848, 3), dtype=np.uint8)
    pts = np.random.default_rng(0).normal([0.30, 0.0, 0.05], 0.02, (300, 3))
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        only_prompt = save_chain(d, rgb, T_BASE_CAM, INTR, source_points=pts,
                                 box=(300, 150, 500, 350))
        assert set(only_prompt) == {"prompt"}
        assert (d / "chain_prompt.png").exists()
        assert not (d / "chain_mask.png").exists()

        full = save_chain(d, rgb, T_BASE_CAM, INTR, source_points=pts,
                          box=(300, 150, 500, 350),
                          mask=np.ones((480, 848), dtype=bool),
                          kept_points=pts)
        assert set(full) == {"prompt", "mask", "kept"}


def test_chain_draws_the_cloud_where_it_projects():
    """The prompt overlay is only useful if the dots land on the object. Uses
    the synthetic box, whose pixels are known."""
    import tempfile
    from pathlib import Path
    import cv2
    from inspection.cell.geometry import object_in_base
    from inspection.perception.overlay import PROMPT_RGB, save_chain
    from inspection.tests.synth import synth_capture
    depth, intr, scale, T_bc, _ = synth_capture()
    pts = object_in_base(depth, intr, scale, T_bc)["points"]
    rgb = np.zeros((480, 848, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        save_chain(d, rgb, T_bc, intr, source_points=pts)
        img = cv2.cvtColor(cv2.imread(str(d / "chain_prompt.png")),
                           cv2.COLOR_BGR2RGB)
        hit = np.all(img == np.array(PROMPT_RGB), axis=-1)
        ys, xs = np.nonzero(hit)
        assert hit.any(), "no prompt dots drawn"
        assert 370 < xs.mean() < 480 and 190 < ys.mean() < 290


def test_chain_never_raises_on_a_missing_dir_or_no_rgb():
    from inspection.perception.overlay import save_chain
    from inspection.tests.synth import INTR, T_BASE_CAM
    assert save_chain("/nope/not/here", np.zeros((4, 4, 3), np.uint8),
                      T_BASE_CAM, INTR) == {}
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        assert save_chain(tmp, None, T_BASE_CAM, INTR) == {}


# ── persistence ─────────────────────────────────────────────────────────────

def _capture_dir(tmp, pose):
    """A minimal already-written capture, as save_bundle would leave it."""
    import json
    from pathlib import Path
    d = Path(tmp) / "000"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps(
        {"pose_id": 0, "rgb_rotation_deg": 180 if pose is _HALF_TURN else 0}))
    return d


def test_save_mask_is_additive_and_records_provenance():
    import json
    import tempfile
    from inspection.perception.capture import save_mask
    with tempfile.TemporaryDirectory() as tmp:
        d = _capture_dir(tmp, _UPRIGHT)
        mask = np.zeros((480, 848), dtype=bool)
        mask[150:350, 300:500] = True
        save_mask(d, mask, 0.93, (300, 150, 500, 350), {"T_base_cam": _UPRIGHT})
        meta = json.loads((d / "meta.json").read_text())
        assert meta["pose_id"] == 0                     # existing keys survive
        assert meta["rgb_rotation_deg"] == 0
        assert meta["mask"]["score"] == 0.93
        assert meta["mask"]["px"] == 200 * 200
        assert meta["mask"]["box"] == [300, 150, 500, 350]
        assert (d / "mask.png").exists()


def test_saved_mask_matches_the_stored_rgb_orientation():
    """mask.png must overlay rgb.png directly. Callers hold masks in RAW
    orientation, and rgb.png is stored upright, so a half-turn capture has to
    be rotated on the way to disk exactly as save_bundle rotates the rgb."""
    import tempfile
    import cv2
    from inspection.perception.capture import save_mask
    with tempfile.TemporaryDirectory() as tmp:
        d = _capture_dir(tmp, _HALF_TURN)
        mask = np.zeros((480, 848), dtype=bool)
        mask[60:130, 100:240] = True                    # off-centre in BOTH axes
        save_mask(d, mask, 0.9, (100, 60, 240, 130), {"T_base_cam": _HALF_TURN})
        on_disk = cv2.imread(str(d / "mask.png"), cv2.IMREAD_GRAYSCALE) > 0
        ys, xs = np.nonzero(on_disk)
        assert (xs.min(), xs.max() + 1) == (848 - 240, 848 - 100)
        assert (ys.min(), ys.max() + 1) == (480 - 130, 480 - 60)


def test_save_mask_ignores_a_capture_dir_that_does_not_exist():
    """The fake rig reports a dir like "(fake 000)" — persisting must be a
    no-op there, not an exception in the settle leg."""
    from inspection.perception.capture import save_mask
    save_mask("(fake 000)", np.zeros((4, 4), bool), 1.0, (0, 0, 1, 1), None)


def main():
    # The miss paths log on purpose; keep the suite's output to its verdict so
    # a real failure is the only thing that stands out.
    import logging
    logging.getLogger("inspection.run.segmenter").setLevel(logging.CRITICAL)
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("OK test_segmenter")


if __name__ == "__main__":
    main()
