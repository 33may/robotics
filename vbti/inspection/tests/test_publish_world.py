#!/usr/bin/env python3
"""publish_world must survive the taught keep-outs (Convex prisms, no
`halfSide`) and render them the SAME way cell/viewer.py does: the union
footprint extruded once — watertight blobs as mesh nodes, not per-shape
translucent boxes. Run: p vbti/inspection/tests/test_publish_world.py"""
import tempfile
from pathlib import Path

from inspection.cell.world import RobotCell
from inspection.ui.publisher import InspectionPublisher, TOPIC_DESCRIPTION


class BusSpy:
    def __init__(self):
        self.published = []          # (topic, payload)
    def publish(self, topic, payload):
        self.published.append((topic, payload))
    def declare(self, *a, **kw):
        pass
    def last(self, topic):
        return [p for t, p in self.published if t == topic][-1]


def test_publish_world_with_taught_keepouts():
    world = RobotCell()
    bus = BusSpy()
    with tempfile.TemporaryDirectory() as run_dir:
        pub = InspectionPublisher(bus, run_dir=run_dir)
        pub.publish_world(world)

        assert bus.published, "publish_world died — nothing published"
        desc = bus.last(TOPIC_DESCRIPTION)
        paths = [n["path"] for n in desc["nodes"]]

        # keep-out blobs present, as meshes served from the run dir
        blobs = [n for n in desc["nodes"] if "keepout" in n["path"]]
        assert blobs, f"no keep-out nodes in {paths}"
        for n in blobs:
            assert n["geometry"]["kind"] == "mesh", n
            obj = Path(run_dir) / Path(n["geometry"]["url"].split("?")[0]).name
            assert obj.exists(), f"{obj} not written"
            assert obj.read_text().startswith("v "), "not a wavefront OBJ"

        # the union mesh REPLACES the per-shape keep-outs — no taught panel
        # or corner may also render as a box (stacked alpha z-fights)
        taught = [p for p in paths
                  if p.startswith(("cell/plane_", "cell/corner_"))]
        assert not taught, f"taught keep-outs leaked as boxes: {taught}"

        # the plain yaml boxes still render
        assert any(p.startswith("cell/") for p in paths), paths
        # robot links unaffected by the new geometry
        assert any(p.startswith("robot/") for p in paths), paths


if __name__ == "__main__":
    test_publish_world_with_taught_keepouts()
    print("ok")
