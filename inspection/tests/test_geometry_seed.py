#!/usr/bin/env python3
"""object_in_base on a synthetic scene. Run: p inspection/tests/test_geometry_seed.py"""
import numpy as np

from inspection.cell.geometry import object_in_base
from inspection.tests.synth import synth_capture


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


def main():
    test_seed_centroid_in_base_frame()
    print("OK test_geometry_seed")


if __name__ == "__main__":
    main()
