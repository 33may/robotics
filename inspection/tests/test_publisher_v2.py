#!/usr/bin/env python3
"""publisher v2: pending/previewing/survey in views/state. Run: p inspection/tests/test_publisher_v2.py"""
import numpy as np
from inspection.ui.publisher import InspectionPublisher, STATE_COLOR


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


class FakeSphere:
    center = np.array([0.4, 0.0, 0.1])
    r = 0.35
    elevations = [20.0, 45.0, 70.0]
    def cells(self):
        return [(h, v) for v in range(2) for h in range(3)]
    def cell_dir(self, h, v):
        d = np.array([np.cos(h), np.sin(h), 0.5 + v])
        return d / np.linalg.norm(d)


def test_states_and_survey():
    bus = BusSpy()
    pub = InspectionPublisher(bus)
    sphere = FakeSphere()
    reach = {c: 0.0 for c in sphere.cells()}
    reach[(2, 1)] = None                                   # unreachable
    pub.publish_views(sphere, reach, visited=[(0, 0)], blocked=[(1, 0)],
                      current=(0, 0), pending=(1, 1), previewing=None,
                      survey="visited")
    views = bus.last("views/state")
    st = {(c["h"], c["v"]): c["state"] for c in views["cells"]}
    assert st[(1, 1)] == "pending"
    assert st[(1, 0)] == "blocked"
    assert st[(2, 1)] == "unreachable"
    assert views["survey"] == {"state": "visited"}
    assert "pending" in STATE_COLOR and "previewing" in STATE_COLOR


def test_previewing_precedence_and_survey_only():
    bus = BusSpy()
    pub = InspectionPublisher(bus)
    sphere = FakeSphere()
    reach = {c: 0.0 for c in sphere.cells()}
    pub.publish_views(sphere, reach, visited=[(1, 1)], previewing=(1, 1),
                      survey="visited")
    views = bus.last("views/state")
    st = {(c["h"], c["v"]): c["state"] for c in views["cells"]}
    assert st[(1, 1)] == "previewing"                      # beats visited

    bus2 = BusSpy()
    pub2 = InspectionPublisher(bus2)
    pub2.publish_survey_only("pending")
    views2 = bus2.last("views/state")
    assert views2 == {"survey": {"state": "pending"}, "center": None,
                      "radius": None, "current": None, "cells": []}


def main():
    test_states_and_survey()
    test_previewing_precedence_and_survey_only()
    print("OK test_publisher_v2")


if __name__ == "__main__":
    main()
