#!/usr/bin/env python3
"""object_in_base on a synthetic scene. Run: p inspection/tests/test_geometry_seed.py"""
import numpy as np

from inspection.cell.geometry import GROW_EPS_M, grow_from_seed, object_in_base
from inspection.tests.synth import synth_capture, synth_capture_two_objects


def test_seed_centroid_in_base_frame():
    depth, intr, scale, T_bc, expected = synth_capture()
    view = object_in_base(depth, intr, scale, T_bc)
    c = view["centroid"]
    assert c is not None and len(view["points"]) > 100
    assert abs(c[0] - expected[0]) < 0.02          # x ~ 0.30
    assert abs(c[1] - expected[1]) < 0.02          # y ~ 0.00
    assert 0.03 < c[2] < 0.07                      # box top face at 0.05
    # plane normal points up and the plane sits near z=0 in base frame
    assert view["plane"][2] > 0.9
    assert abs(view["plane"][3]) < 0.02


# ── grow_from_seed ──────────────────────────────────────────────────────────

def _slab(x0, n=6, step=0.005):
    """Small grid of points starting at x0, spaced `step` along x."""
    xs = x0 + np.arange(n) * step
    ys = np.arange(n) * step
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])


def test_grow_keeps_touching_points():
    seed = _slab(0.0)
    nearby = _slab(0.03 + GROW_EPS_M / 2)          # within eps of the seed edge
    kept = grow_from_seed(seed, nearby)
    assert len(kept) == len(nearby)


def test_grow_rejects_distant_blob():
    seed = _slab(0.0)
    blob = _slab(0.5)                              # half a metre away
    kept = grow_from_seed(seed, blob)
    assert len(kept) == 0


def test_grow_keeps_only_the_connected_part():
    seed = _slab(0.0)
    near, far = _slab(0.031), _slab(0.5)
    kept = grow_from_seed(seed, np.vstack([near, far]))
    assert len(kept) == len(near)
    assert kept[:, 0].max() < 0.1                  # nothing from the far blob


def test_grow_chains_beyond_one_eps():
    """Connectivity is transitive: a trail of close points extends the set
    further than eps from the original seed — that is how a later view adds
    genuinely new surface, and also the only way this can leak."""
    seed = _slab(0.0)
    trail = np.vstack([_slab(0.031 + i * 0.015) for i in range(5)])
    kept = grow_from_seed(seed, trail)
    assert len(kept) == len(trail)
    assert kept[:, 0].max() > 0.09                 # far past one eps step


def test_grow_with_empty_inputs():
    empty = np.empty((0, 3))
    assert len(grow_from_seed(_slab(0.0), empty)) == 0
    assert len(grow_from_seed(empty, _slab(0.0))) == 0


# ── object_in_base seeding ──────────────────────────────────────────────────

def test_unseeded_view_elects_the_biggest_cluster():
    """Bootstrap behaviour is unchanged: with no seed the largest cluster
    wins — which on this scene is the DECOY, the failure 2408-geomtest hit."""
    depth, intr, scale, T_bc, _, decoy = synth_capture_two_objects()
    view = object_in_base(depth, intr, scale, T_bc)
    assert np.abs(view["centroid"][:2] - decoy).max() < 0.03


def test_seeded_view_ignores_the_bigger_decoy():
    depth, intr, scale, T_bc, subject, decoy = synth_capture_two_objects()
    seed = object_in_base(*synth_capture()[:4])["points"]
    view = object_in_base(depth, intr, scale, T_bc, seed=seed)
    assert len(view["points"]) > 100
    assert np.abs(view["centroid"][:2] - subject).max() < 0.03
    assert np.abs(view["points"][:, 1]).max() < 0.10      # no decoy points


def test_seeded_view_with_nothing_near_returns_no_points():
    """The 'updated by 0 points' case: same call, same return shape, empty
    cloud — the caller adds nothing and carries on."""
    depth, intr, scale, T_bc, _, _ = synth_capture_two_objects()
    far_seed = np.column_stack([np.full(50, 0.55), np.full(50, 0.40),
                                np.full(50, 0.30)])
    view = object_in_base(depth, intr, scale, T_bc, seed=far_seed)
    assert len(view["points"]) == 0
    assert view["centroid"] is None
    assert view["plane"] is not None                # the view itself is valid


def test_accumulator_rejects_a_detached_blob():
    """Run 2408-cup1: the 10 deg ring caught a cable at the table edge. Seeded
    growth could not refuse it — at GROW_EPS_M the cable was genuinely
    connected to the cup inside that one view — so the cloud, the object box
    and finally the planner's start state were all corrupted."""
    from inspection.cell.geometry import CloudAccumulator, JUMP_GATE_M
    rng = np.random.default_rng(0)
    cup = rng.normal(scale=0.02, size=(400, 3)) + [0.30, 0.0, 0.05]
    acc = CloudAccumulator()
    assert acc.add(cup) == 0                       # bootstrap keeps everything

    near = cup[:50] + [0.01, 0.0, 0.0]             # more of the same object
    assert acc.add(near) == 0

    far = rng.normal(scale=0.01, size=(60, 3)) + [0.30 + 3 * JUMP_GATE_M, 0, 0.05]
    assert acc.add(far) == 60                      # detached -> all rejected
    assert acc.aabb()[1][0] < 0.30 + JUMP_GATE_M   # never entered the cloud


def test_box_percentile_survives_a_few_strays():
    """Raw min/max let 4.7% of points triple the box until it contained the
    arm. The trimmed extent must ignore them without moving the real object."""
    from inspection.cell.geometry import CloudAccumulator, BOX_PCT
    rng = np.random.default_rng(1)
    acc = CloudAccumulator()
    acc.add(rng.normal(scale=0.02, size=(2000, 3)) + [0.30, 0.0, 0.05])
    raw_lo, raw_hi = acc.aabb()
    trim_lo, trim_hi = acc.aabb(pct=BOX_PCT)
    assert np.all(trim_hi <= raw_hi) and np.all(trim_lo >= raw_lo)
    assert np.all((trim_hi - trim_lo) > 0.5 * (raw_hi - raw_lo))   # still the object


# ── mask-based identity ─────────────────────────────────────────────────────

def test_growth_absorbs_touching_clutter_but_a_mask_does_not():
    """The 2408-cup2 regression, in miniature. Same view, same depth: seeded
    growth swallows the adjacent bar because it IS connected, and the mask
    keeps only the subject. If these two ever agree, the scene stopped
    modelling the failure."""
    from inspection.tests.synth import synth_capture_touching_clutter
    depth, intr, scale, T_bc, mask = synth_capture_touching_clutter()
    seed = object_in_base(*synth_capture()[:4])["points"]

    grown = object_in_base(depth, intr, scale, T_bc, seed=seed)["points"]
    masked = object_in_base(depth, intr, scale, T_bc, seed=seed, mask=mask)["points"]

    assert grown[:, 0].max() > 0.45         # bar absorbed, reaches x ~ 0.50
    assert masked[:, 0].max() < 0.36        # subject only, ends at x ~ 0.345
    assert len(masked) > 100


def test_mask_overrides_the_seed_entirely():
    """A mask is the answer, not a hint: it must win even when the seed points
    somewhere else, so a caller can never get a silent blend of the two."""
    from inspection.tests.synth import synth_capture_touching_clutter
    depth, intr, scale, T_bc, mask = synth_capture_touching_clutter()
    far_seed = np.column_stack([np.full(50, 0.55), np.full(50, 0.40),
                                np.full(50, 0.30)])
    out = object_in_base(depth, intr, scale, T_bc, seed=far_seed, mask=mask)
    assert len(out["points"]) > 100         # seeded-only would return nothing
    assert out["points"][:, 0].max() < 0.36


def test_mask_points_are_not_clipped_by_the_workspace():
    """Run 2408-cup4: the mug sat against the WORKSPACE x face and 22% of every
    masked view was sliced off, leaving a flat-sided cloud 92.8 mm across
    instead of 126.9 — with the missing half on the side facing the robot.
    `WORKSPACE` filters the SCENE; the mask defines the OBJECT."""
    from inspection.cell.geometry import WORKSPACE, crop_workspace
    depth, intr, scale, T_bc, _ = synth_capture()
    mask = np.zeros(depth.shape, dtype=bool)
    mask[200:280, 384:464] = True

    out = object_in_base(depth, intr, scale, T_bc, mask=mask)["points"]
    assert len(out) > 100

    # Move the workspace so its x floor cuts through the middle of the object,
    # exactly as cup4's table position did.
    mid = float(np.median(out[:, 0]))
    import inspection.cell.geometry as geom
    real = geom.WORKSPACE
    geom.WORKSPACE = {**real, "x": (mid, real["x"][1])}
    try:
        clipped = crop_workspace(out)
        assert len(clipped) < 0.75 * len(out), "test scene does not straddle the face"
        still = object_in_base(depth, intr, scale, T_bc, mask=mask)["points"]
        assert len(still) == len(out), "masked points were workspace-cropped"
        assert still[:, 0].min() < mid - 0.005, "object was amputated at the face"
    finally:
        geom.WORKSPACE = real


def test_mask_points_are_still_range_bounded():
    """Not cropping by workspace must not mean unbounded: deproject's range
    clip is what keeps a stray mask from lifting the far wall."""
    from inspection.cell.geometry import MAX_RANGE_M
    depth, intr, scale, T_bc, _ = synth_capture()
    far = depth.copy()
    far[100:150, 100:200] = int((MAX_RANGE_M + 0.5) / scale)   # way past the gate
    mask = np.zeros(depth.shape, dtype=bool)
    mask[100:150, 100:200] = True
    out = object_in_base(far, intr, scale, T_bc, mask=mask)["points"]
    assert len(out) == 0, "points beyond MAX_RANGE_M entered the cloud"


def test_mask_shape_must_match_the_depth():
    depth, intr, scale, T_bc, _ = synth_capture()
    try:
        object_in_base(depth, intr, scale, T_bc, mask=np.zeros((4, 4), bool))
    except ValueError as e:
        assert "does not match" in str(e)
    else:
        raise AssertionError("mismatched mask shape must raise")


def test_empty_mask_yields_no_points_not_a_crash():
    depth, intr, scale, T_bc, _ = synth_capture()
    out = object_in_base(depth, intr, scale, T_bc,
                         mask=np.zeros(depth.shape, bool))
    assert len(out["points"]) == 0 and out["centroid"] is None
    assert out["plane"] is not None          # plane still fitted on the FULL view


# ── per-point colour ────────────────────────────────────────────────────────

def _rgb_two_tone(depth):
    """Frame painted blue, object pixels (the synth box) painted red on the
    left half and green on the right — so pixel pairing is checkable: a
    correctly lifted cloud contains ONLY reds and greens, in both tones."""
    rgb = np.zeros((*depth.shape, 3), dtype=np.uint8)
    rgb[:] = (0, 0, 255)
    rgb[200:280, 384:424] = (255, 0, 0)
    rgb[200:280, 424:464] = (0, 255, 0)
    return rgb


def test_mask_colors_are_pixel_paired():
    depth, intr, scale, T_bc, _ = synth_capture()
    mask = np.zeros(depth.shape, dtype=bool)
    mask[200:280, 384:464] = True
    out = object_in_base(depth, intr, scale, T_bc, mask=mask,
                         rgb=_rgb_two_tone(depth))
    C, P = out["colors"], out["points"]
    assert C is not None and C.shape == P.shape and C.dtype == np.uint8
    reds = (C == (255, 0, 0)).all(axis=1)
    greens = (C == (0, 255, 0)).all(axis=1)
    assert (reds | greens).all(), "a colour leaked in from outside the mask"
    assert reds.any() and greens.any(), "one half of the object lost its tone"


def test_bootstrap_and_seeded_colors_follow_the_points():
    depth, intr, scale, T_bc, _ = synth_capture()
    rgb = _rgb_two_tone(depth)
    boot = object_in_base(depth, intr, scale, T_bc, rgb=rgb)
    assert boot["colors"].shape == boot["points"].shape
    blue = (boot["colors"] == (0, 0, 255)).all(axis=1)
    assert not blue.any(), "bootstrap cluster picked up table pixels"

    seed = boot["points"]
    grown = object_in_base(depth, intr, scale, T_bc, seed=seed, rgb=rgb)
    assert grown["colors"].shape == grown["points"].shape
    assert not (grown["colors"] == (0, 0, 255)).all(axis=1).any()


def test_no_rgb_means_no_colors():
    depth, intr, scale, T_bc, _ = synth_capture()
    out = object_in_base(depth, intr, scale, T_bc)
    assert out.get("colors") is None


def test_accumulator_carries_colors_through_fusion():
    """Colours ride the same filters as the points: jump-gate rejects drop
    them, the voxel average blends them, and a colourless add is mid-grey —
    the row alignment with `points` must survive all three."""
    from inspection.cell.geometry import CloudAccumulator, JUMP_GATE_M
    rng = np.random.default_rng(0)
    cup = rng.normal(scale=0.02, size=(400, 3)) + [0.30, 0.0, 0.05]
    red = np.tile(np.array([[200, 10, 10]], dtype=np.uint8), (400, 1))
    acc = CloudAccumulator()
    acc.add(cup, red)
    assert acc.colors.shape == acc.points.shape
    assert acc.colors.dtype == np.uint8
    assert (acc.colors[:, 0] > 150).all()          # still red after voxelising

    far = rng.normal(scale=0.01, size=(60, 3)) + [0.30 + 3 * JUMP_GATE_M, 0, 0.05]
    green = np.tile(np.array([[10, 200, 10]], dtype=np.uint8), (60, 1))
    assert acc.add(far, green) == 60               # detached -> rejected
    assert not (acc.colors[:, 1] > 150).any(), "rejected points left colour behind"

    near = cup[:50] + [0.01, 0.0, 0.0]
    acc.add(near, None)                            # colourless add -> grey fill
    assert acc.colors.shape == acc.points.shape


# ── prompt_box ──────────────────────────────────────────────────────────────

def test_prompt_box_lands_on_the_object_it_came_from():
    """Round trip: lift the subject, project it back, and the box must sit on
    the pixels it was built from. This is the whole identity mechanism."""
    from inspection.cell.geometry import PROMPT_MARGIN_PX, prompt_box
    depth, intr, scale, T_bc, _ = synth_capture()
    pts = object_in_base(depth, intr, scale, T_bc)["points"]
    box = prompt_box(pts, T_bc, intr, depth.shape)
    x0, y0, x1, y1 = box
    m = PROMPT_MARGIN_PX + 4                        # margin + percentile slack
    assert 384 - m <= x0 <= 384 + m and 464 - m <= x1 <= 464 + m
    assert 200 - m <= y0 <= 200 + m and 280 - m <= y1 <= 280 + m


def test_prompt_box_is_none_when_the_cloud_is_not_visible():
    """Points behind the lens must not produce a box — the caller falls back
    instead of segmenting a guess."""
    from inspection.cell.geometry import prompt_box
    depth, intr, scale, T_bc, _ = synth_capture()
    behind = np.column_stack([np.full(200, 0.30), np.zeros(200),
                              np.full(200, 2.0)])   # above the camera
    assert prompt_box(behind, T_bc, intr, depth.shape) is None
    assert prompt_box(np.empty((0, 3)), T_bc, intr, depth.shape) is None


def test_prompt_box_floor_extrusion_reaches_the_table():
    """A top-only cloud seen side-on prompts a top-band box — the anchor that
    kept run 0809-0809box's sides out of the cloud for all 27 views. With
    `floor_z` the box must keep its footprint and top edge but grow DOWN to
    the table line, where the unseen sides live."""
    from inspection.cell.geometry import prompt_box
    rng = np.random.default_rng(0)
    top = np.column_stack([rng.uniform(-0.05, 0.05, 400),
                           rng.uniform(-0.05, 0.05, 400),
                           np.full(400, 0.07)])          # the SEEN top face
    T_bc = np.array([[0., 0., -1., 1.0],                 # side-on camera 1 m
                     [-1., 0., 0., 0.0],                 # out, level with the
                     [0., -1., 0., 0.05],                # object
                     [0., 0., 0., 1.]])
    intr = {"fx": 430.0, "fy": 430.0, "ppx": 424.0, "ppy": 240.0}
    plain = prompt_box(top, T_bc, intr, (480, 848))
    floored = prompt_box(top, T_bc, intr, (480, 848), floor_z=0.0)
    assert plain is not None and floored is not None
    assert abs(plain[0] - floored[0]) <= 2               # footprint unchanged
    assert abs(plain[2] - floored[2]) <= 2
    assert abs(plain[1] - floored[1]) <= 2               # top edge stays
    assert floored[3] >= plain[3] + 20                   # ~30 px of sides here


def test_prompt_box_clips_to_the_frame():
    from inspection.cell.geometry import prompt_box
    depth, intr, scale, T_bc, _ = synth_capture()
    edge = np.column_stack([np.linspace(0.06, 0.10, 300), np.zeros(300),
                            np.zeros(300)])
    box = prompt_box(edge, T_bc, intr, depth.shape)
    assert box is None or (box[0] >= 0 and box[1] >= 0
                           and box[2] <= 848 and box[3] <= 480)


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("OK test_geometry_seed")


if __name__ == "__main__":
    main()
