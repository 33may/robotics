#!/usr/bin/env python3
"""publish_decomposition: boxguard fit -> orange fit/ box in the description;
abstain clears it; unknown method/kind skipped without raising.
Run: p inspection/tests/test_publisher_decomposition.py"""
import numpy as np
from inspection.ui.publisher import (
    FIT_COLOR,
    InspectionPublisher,
    TOPIC_DESCRIPTION,
)


class BusSpy:
    def __init__(self):
        self.published = []          # (topic, payload)
    def publish(self, topic, payload):
        self.published.append((topic, payload))
    def declare(self, *a, **kw):
        pass
    @staticmethod
    def array_payload(a):
        return a
    @staticmethod
    def jpeg_payload(a, quality=80):
        return a
    def last(self, topic):
        return [p for t, p in self.published if t == topic][-1]


def boxguard_rec(**over):
    """A DecompositionRecord dict the way steps/NNN/decomposition.json holds it."""
    pose = np.eye(4)
    pose[:3, 3] = [0.4, -0.1, 0.15]
    rec = {
        "step_id": 3,
        "source": "fused/cloud.npy",
        "n_points": 4200,
        "method": "boxguard",
        "method_version": "1",
        "abstain": False,
        "reason": "",
        "primitives": [{
            "kind": "box",
            "pose": pose.tolist(),
            "params": {"extents": [0.08, 0.12, 0.2]},
            "label": "body",
        }],
    }
    rec.update(over)
    return rec


def fit_nodes(bus):
    return [n for n in bus.last(TOPIC_DESCRIPTION)["nodes"]
            if n["path"].startswith("fit/")]


def test_boxguard_box_lands_in_description():
    bus = BusSpy()
    pub = InspectionPublisher(bus)
    pub.publish_decomposition(boxguard_rec())
    nodes = fit_nodes(bus)
    assert len(nodes) == 1
    node = nodes[0]
    assert node["path"] == "fit/body"
    assert node["geometry"] == {"kind": "box", "size": [0.08, 0.12, 0.2]}
    assert node["transform"] == [1, 0, 0, 0.4, 0, 1, 0, -0.1,
                                 0, 0, 1, 0.15, 0, 0, 0, 1]
    assert node["material"]["color"] == FIT_COLOR
    assert node["material"]["side"] == "double"


def test_abstain_clears_the_previous_fit():
    bus = BusSpy()
    pub = InspectionPublisher(bus)
    pub.publish_decomposition(boxguard_rec())
    assert fit_nodes(bus)
    pub.publish_decomposition(boxguard_rec(
        abstain=True, reason="cloud too thin", primitives=[]))
    assert fit_nodes(bus) == []                      # stale box is gone


def test_unlabelled_primitive_falls_back_to_index():
    bus = BusSpy()
    pub = InspectionPublisher(bus)
    pub.publish_decomposition(boxguard_rec(primitives=[{
        "kind": "box", "pose": np.eye(4).tolist(),
        "params": {"extents": [0.1, 0.1, 0.1]}, "label": "",
    }]))
    assert fit_nodes(bus)[0]["path"] == "fit/0"


def test_unknown_kind_is_skipped_not_raised():
    bus = BusSpy()
    pub = InspectionPublisher(bus)
    rec = boxguard_rec()
    rec["primitives"].append({
        "kind": "superellipsoid", "pose": np.eye(4).tolist(),
        "params": {}, "label": "blob",
    })
    pub.publish_decomposition(rec)                   # must not raise
    nodes = fit_nodes(bus)
    assert [n["path"] for n in nodes] == ["fit/body"]


def test_unknown_method_not_drawn_and_clears_stale():
    """New fitting methods reuse the record (single polymorphic slot) before a
    renderer exists for them; the old box must still disappear."""
    bus = BusSpy()
    pub = InspectionPublisher(bus)
    pub.publish_decomposition(boxguard_rec())
    assert fit_nodes(bus)
    pub.publish_decomposition(boxguard_rec(method="superquadric-v2"))
    assert fit_nodes(bus) == []


def main():
    test_boxguard_box_lands_in_description()
    test_abstain_clears_the_previous_fit()
    test_unlabelled_primitive_falls_back_to_index()
    test_unknown_kind_is_skipped_not_raised()
    test_unknown_method_not_drawn_and_clears_stale()
    print("OK test_publisher_decomposition")


if __name__ == "__main__":
    main()
