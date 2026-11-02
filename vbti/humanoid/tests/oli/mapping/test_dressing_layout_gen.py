"""Tests for dressing layout generation (MAY-173 locdev dressing).

From the nav occupancy grid to a placement list: perimeter walls only (racks
are interior occupied islands — plates on racks are forbidden), plates every
1.5–2 m alternating mount height, posters on the longest blank stretches.
Positions are the final quad centers: 2 cm off the wall, facing into the room.

Pure python (numpy + yaml + stdlib) — runs in the `hum` env.
"""

import numpy as np
import pytest

from humanoid.logic.simulation.mapping.dressing.layout_gen import (
    build_layout,
    extract_perimeter,
    load_layout,
    quad_corners,
    wall_segments,
    write_layout,
)

pytestmark = pytest.mark.brain

RES = 0.05
ORIGIN = (0.0, 0.0)


def _room(h=200, w=200, wall=2):
    """Synthetic map: hollow rectangle of occupied border cells (the perimeter
    wall), free inside. 200 cells @ 5cm = 10 m square."""
    occ = np.zeros((h, w), dtype=bool)
    occ[:wall, :] = occ[-wall:, :] = True
    occ[:, :wall] = occ[:, -wall:] = True
    return occ


def _room_with_rack():
    """Room + a free-standing interior island (a rack)."""
    occ = _room()
    occ[80:120, 90:110] = True
    return occ


# ── perimeter extraction ──────────────────────────────────────────────────


def test_perimeter_includes_border_walls_excludes_rack():
    occ = _room_with_rack()
    perim = extract_perimeter(occ)
    assert perim[0, 100]          # border wall cell
    assert not perim[100, 100]    # rack cell: occupied but NOT perimeter
    assert not perim[50, 50]      # free cell


def test_perimeter_is_subset_of_occupied():
    occ = _room_with_rack()
    perim = extract_perimeter(occ)
    assert not (perim & ~occ).any()


# ── wall segments ─────────────────────────────────────────────────────────


def test_square_room_yields_inward_facing_segments_on_all_four_sides():
    segs = wall_segments(_room(), resolution=RES, origin=ORIGIN)
    normals = {tuple(s.normal) for s in segs}
    assert normals == {(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)}


def test_segments_span_the_walls():
    segs = wall_segments(_room(), resolution=RES, origin=ORIGIN)
    longest = max(s.length for s in segs)
    # 10 m room minus corners: each wall's inward run is ~9.8 m
    assert 9.0 < longest <= 10.0


def test_rack_faces_are_not_segments():
    """Same room, with and without a rack: the rack's faces must not add
    segments — plates never land on racks."""
    base = wall_segments(_room(), resolution=RES, origin=ORIGIN)
    with_rack = wall_segments(_room_with_rack(), resolution=RES, origin=ORIGIN)
    assert len(with_rack) == len(base)


# ── placements ────────────────────────────────────────────────────────────


def _layout():
    return build_layout(_room_with_rack(), resolution=RES, origin=ORIGIN)


def test_plates_spaced_between_1_5_and_2_m_along_each_wall():
    lay = _layout()
    by_wall = {}
    for p in lay["plates"]:
        by_wall.setdefault(tuple(p["normal"]), []).append(p)
    for wall, plates in by_wall.items():
        pos = sorted((p["pos"][0], p["pos"][1]) for p in plates)
        # project onto the along-wall axis
        axis = 1 if wall[0] != 0 else 0
        coords = sorted({round(c[axis], 3) for c in pos})
        gaps = np.diff(coords)
        gaps = gaps[gaps > 0.1]  # ignore same-position (different height) pairs
        assert (gaps >= 1.5).all() and (gaps <= 2.0).all(), f"wall {wall}: {gaps}"


def test_plate_numbers_globally_unique():
    nums = [p["number"] for p in _layout()["plates"]]
    assert len(nums) == len(set(nums))


def test_plate_heights_alternate_between_low_and_high():
    lay = _layout()
    heights = {p["pos"][2] for p in lay["plates"]}
    assert heights == {1.2, 2.5}
    # alternation along a single wall: consecutive plates differ in height
    wall = [p for p in lay["plates"] if tuple(p["normal"]) == (1.0, 0.0)]
    wall.sort(key=lambda p: p["pos"][1])
    zs = [p["pos"][2] for p in wall]
    assert all(a != b for a, b in zip(zs, zs[1:]))


def test_plate_sizes_within_spec():
    sizes = {p["size_m"] for p in _layout()["plates"]}
    assert all(0.4 <= s <= 0.8 for s in sizes)
    assert len(sizes) >= 2  # varied, not uniform


def test_plates_sit_2cm_off_the_wall_facing_inward():
    occ = _room_with_rack()
    lay = build_layout(occ, resolution=RES, origin=ORIGIN)
    for p in lay["plates"]:
        x, y, _ = p["pos"]
        nx, ny = p["normal"]
        # wall face is 2 cm behind the quad center (against the normal)
        wx, wy = x - nx * 0.02, y - ny * 0.02
        ri, ci = int((wy - ORIGIN[1]) / RES - ny * 0.5), int((wx - ORIGIN[0]) / RES - nx * 0.5)
        assert occ[ri, ci], f"plate {p['number']} not against a wall"
        # quad center itself is in free space
        fi, fj = int((y - ORIGIN[1]) / RES + ny), int((x - ORIGIN[0]) / RES + nx)
        assert not occ[fi, fj]


def test_no_plate_on_the_rack():
    lay = _layout()
    for p in lay["plates"]:
        x, y, _ = p["pos"]
        # rack occupies [4.5,5.5]x[4.0,6.0] m in world coords (cells 90-110/80-120)
        assert not (4.3 < x < 5.7 and 3.8 < y < 6.2), f"plate {p['number']} on rack"


def test_posters_on_long_walls_sized_1_to_2m():
    lay = _layout()
    assert 2 <= len(lay["posters"]) <= 8
    for po in lay["posters"]:
        assert 1.0 <= po["size_m"] <= 2.0
        assert po["texture"].startswith("poster")


def test_layout_yaml_round_trips(tmp_path):
    lay = _layout()
    path = tmp_path / "layout.yaml"
    write_layout(lay, path)
    assert load_layout(path) == lay


# ── quad corner math (consumed by build_dressing_usd in the isaac env) ────


def test_quad_corners_face_normal_and_match_size():
    corners = quad_corners(pos=(1.0, 2.0, 1.5), normal=(1.0, 0.0), size=0.6)
    corners = np.array(corners)
    assert corners.shape == (4, 3)
    # all corners lie in the plane x=1.0 (quad is perpendicular to +x normal)
    np.testing.assert_allclose(corners[:, 0], 1.0)
    # extents match size in both quad axes
    assert np.ptp(corners[:, 1]) == pytest.approx(0.6)
    assert np.ptp(corners[:, 2]) == pytest.approx(0.6)
    # winding: computed face normal points along +x (into the room)
    v1, v2 = corners[1] - corners[0], corners[2] - corners[0]
    n = np.cross(v1, v2)
    assert n[0] > 0


def test_module_is_pure():
    import sys

    import humanoid.logic.simulation.mapping.dressing.layout_gen  # noqa: F401

    assert "isaacsim" not in sys.modules
    assert "limxsdk" not in sys.modules
