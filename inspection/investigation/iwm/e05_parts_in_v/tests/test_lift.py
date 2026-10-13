import numpy as np
import pytest

from inspection.investigation.iwm.e05_parts_in_v import common as C5

pytestmark = pytest.mark.skipif(
    not (C5.DATA.parent / "e04_voxel_scene" / "2209-compute-ray-demo5" / "scene" / "2mm" / "vbg.npz").exists(),
    reason="needs e04 V")


def test_l2_roundtrip_c56_2mm():
    from inspection.investigation.iwm.e05_parts_in_v import lift as L
    from inspection.investigation.iwm.e05_parts_in_v.stv import STV, Surface
    view = C5.load_view("c5.6")
    vbg, size = C5.V(view["run"], 2)
    surf = Surface.from_vbg(vbg, size, C5.trunc_m(size))
    rows, stats = L.lift_L2(view, surf, vbg, size)
    assert all(s["hit_share"] > 0.5 for s in stats.values())           # holes exist but never dominate
    assert all((r >= 0).all() for r in rows.values())                   # never row -1 inside a node
    t = STV(surf)
    L.colour(t, view, rows)
    t.check()
    lab, _, _ = L.label_render(t, vbg, size, view["T"], view["K"])
    for nid, part in t.meta["c5.6"].items():
        if nid == 0:
            continue
        i, _, _ = L.iou(L.part_pixels(lab, t, part), view["masks"][nid])
        b = view["nodes"][nid]["bbox"]
        wide = min(b[2] - b[0], b[3] - b[1]) >= 40                     # a 2 mm grid quantises ~7 mm parts
        assert i > (0.5 if wide else 0.2), (nid, i)                    # catches a mirrored (180°) colouring
