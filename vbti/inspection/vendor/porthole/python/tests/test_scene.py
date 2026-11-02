"""Scene protocol payload builders — docs/scene-protocol.md.

These tests guard the conversions where a mistake is invisible downstream. A
transposed transform still renders, still animates, and puts everything
somewhere plausible and wrong; nothing in the browser can detect it.
"""

from __future__ import annotations

import numpy as np
import pytest

from porthole.scene import (
    axes_node,
    box_node,
    cylinder_node,
    mesh_node,
    pose_frame,
    scene_description,
    sphere_node,
)


def _matrix(payload: dict, index: int = 0) -> np.ndarray:
    nd = payload["transforms"]["$nd"]
    assert nd["dtype"] == "float32"
    flat = np.frombuffer(nd["data"], dtype=np.float32).reshape(nd["shape"])
    return flat[index]


def test_transform_stays_row_major_through_the_encoder():
    T = np.eye(4)
    T[:3, 3] = [0.1, 0.2, 0.3]
    T[0, 1] = -1.0

    payload = pose_frame({"a": T})
    out = _matrix(payload)

    assert out[0, 3] == pytest.approx(0.1)
    assert out[1, 3] == pytest.approx(0.2)
    assert out[2, 3] == pytest.approx(0.3)
    # A transpose would move -1 to [1][0] and the translation to the last row.
    assert out[0, 1] == pytest.approx(-1.0)


def test_pose_frame_shape_matches_name_count():
    payload = pose_frame({"a": np.eye(4), "b": np.eye(4), "c": np.eye(4)})
    assert payload["names"] == ["a", "b", "c"]
    assert payload["transforms"]["$nd"]["shape"] == [3, 4, 4]


def test_pose_frame_accepts_flat_sixteen_and_nested_four_by_four():
    flat = pose_frame({"a": list(range(16))})
    nested = pose_frame({"a": [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]})
    assert np.allclose(_matrix(flat), _matrix(nested))


def test_pose_frame_accepts_an_se3_like_object():
    class FakeSE3:
        rotation = np.eye(3)
        translation = np.array([1.0, 2.0, 3.0])

    out = _matrix(pose_frame({"a": FakeSE3()}))
    assert out[:3, 3] == pytest.approx([1.0, 2.0, 3.0])


def test_wrong_sized_transform_is_rejected_rather_than_padded():
    with pytest.raises(ValueError):
        pose_frame({"a": [1, 2, 3, 4, 5, 6]})


def test_empty_pose_frame_is_valid():
    payload = pose_frame({})
    assert payload["names"] == []
    assert payload["transforms"]["$nd"]["shape"] == [0, 4, 4]


def test_overrides_and_background_are_omitted_when_absent():
    payload = pose_frame({"a": np.eye(4)})
    assert "overrides" not in payload
    assert "background" not in payload


def test_box_takes_full_extents_and_keeps_them():
    node = box_node("cell/slab", size=(1.0, 0.5, 0.03))
    assert node["geometry"] == {"kind": "box", "size": [1.0, 0.5, 0.03]}


def test_box_rejects_a_wrong_sized_extent():
    with pytest.raises(ValueError):
        box_node("cell/slab", size=(1.0, 0.5))


def test_identity_is_the_default_transform():
    node = sphere_node("marker", 0.03)
    assert node["transform"] == [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def test_material_is_omitted_when_nothing_was_specified():
    assert "material" not in axes_node("world/axes", 0.2)
    assert "material" not in mesh_node("robot/wrist", "/meshes/wrist3.dae")


def test_opacity_below_one_survives_as_a_material():
    node = cylinder_node("post", 0.02, 1.0, color="#ff8800", opacity=0.45)
    assert node["material"] == {"color": "#ff8800", "opacity": 0.45}


def test_duplicate_node_paths_are_rejected():
    # Two nodes on one path silently means the second wins and the first never
    # appears — a scene that is quietly missing geometry.
    with pytest.raises(ValueError, match="duplicate"):
        scene_description([box_node("a", (1, 1, 1)), box_node("a", (2, 2, 2))])


def test_description_carries_every_node_and_its_frame():
    description = scene_description(
        [box_node("a", (1, 1, 1)), mesh_node("b", "/meshes/b.dae")], frame="base"
    )
    assert description["frame"] == "base"
    assert [node["path"] for node in description["nodes"]] == ["a", "b"]
