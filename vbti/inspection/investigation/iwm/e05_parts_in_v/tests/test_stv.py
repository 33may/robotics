import numpy as np

from inspection.investigation.iwm.e05_parts_in_v.stv import STV, Surface


def toy(n=10):
    g = np.stack(np.meshgrid(np.arange(n), np.arange(n), [0], indexing="ij"), -1).reshape(-1, 3)
    return Surface(g.astype(np.int32), np.zeros(len(g), np.float32), np.ones(len(g), np.float32), 0.001)


def test_row_numbers_are_canonical_and_load_refuses_another_surface(tmp_path):
    g = np.stack(np.meshgrid(np.arange(10), np.arange(10), [0], indexing="ij"), -1).reshape(-1, 3).astype(np.int32)
    perm = np.random.default_rng(1).permutation(len(g))
    a = Surface(g, np.zeros(len(g), np.float32), np.ones(len(g), np.float32), 0.001)
    b = Surface(g[perm], np.zeros(len(g), np.float32), np.ones(len(g), np.float32), 0.001)   # same voxels, other order
    p = np.array([[0.003, 0.007, 0.0], [0.009, 0.001, 0.0]])
    assert (a.rows_at(p) == b.rows_at(p)).all()                                  # a row number means one voxel, always
    assert (a.centres(a.rows_at(p)) == b.centres(b.rows_at(p))).all()
    t = STV(a)
    t.add(t.root, "x", rows=a.rows_at(p))
    t.save(tmp_path)
    assert (STV.load(tmp_path, b).parts[1].rows == t.parts[1].rows).all()        # loads onto the other order too
    other = Surface(g[:90], np.zeros(90, np.float32), np.ones(90, np.float32), 0.001)
    import pytest
    with pytest.raises(AssertionError):
        STV.load(tmp_path, other)                                                 # a different surface is refused


def test_rows_at_and_near():
    s = toy()
    assert s.rows_at(np.array([[0.0031, 0.0049, 0.0]]))[0] == s.rows_at(np.array([[0.003, 0.005, 0.0]]))[0]
    assert s.rows_at(np.array([[0.02, 0.0, 0.0]]))[0] == -1
    assert s.rows_near(np.array([[0.0094, 0.0, 0.0]]))[0] == s.rows_at(np.array([[0.009, 0.0, 0.0]]))[0]
    assert s.rows_near(np.array([[0.012, 0.0, 0.0]]), max_voxels=1.5)[0] == -1
    assert s.rows_at(np.array([[-0.0004, 0.0, 0.0]]))[0] == s.rows_at(np.array([[0.0, 0.0, 0.0]]))[0]


def test_children_inside_parent_and_deepest_label():
    t = STV(toy())
    obj = t.root
    face = t.add(obj, "face", rows=np.arange(0, 50))
    port = t.add(face, "port", rows=np.arange(10, 14))
    assert set(t.parts[port].rows) <= set(t.parts[face].rows) <= set(t.parts[obj].rows)
    assert (t.label[10:14] == port).all() and (t.label[0:10] == face).all()
    t.assign(face, np.arange(10, 14))            # parent re-assigned over the child: child keeps the label
    assert (t.label[10:14] == port).all() and not t.conflicts
    r = t.assign(port, np.arange(14, 16))        # child extended into parent-only rows: relabelled, no conflict
    assert (t.label[14:16] == port).all() and not t.conflicts and r["relabelled"] == 2
    t.check()


def test_conflict_recorded_first_wins():
    t = STV(toy())
    a = t.add(t.root, "A", rows=np.arange(0, 5))
    b = t.add(t.root, "B", rows=np.arange(3, 8))
    assert (t.label[3:5] == a).all() and len(t.conflicts) == 2 and t.conflicts[0] == (3, a, b)
    assert set(np.arange(3, 5)) <= set(t.parts[b].rows)        # B still holds the rows
    t.check()


def test_aliases_from_later_views_are_kept_and_searchable(tmp_path):
    s = toy()
    t = STV(s)
    p = t.add(t.root, "Ethernet Port 1", rows=np.arange(0, 8), detection=("c5.2", 9))
    t.extend(p, np.arange(5, 12), detection=("c5.6", 5), alias="LAN 1G port")
    t.extend(p, np.arange(5, 12), detection=("c5.6", 5), alias="LAN 1G port")       # no duplicate
    assert t.parts[p].aliases == ["LAN 1G port"] and t.parts[p].name == "Ethernet Port 1"
    assert t.search("lan 1g") == [p] and t.search("ethernet") == [p]
    t.save(tmp_path)
    assert STV.load(tmp_path, s).parts[p].aliases == ["LAN 1G port"]


def test_part_of_path_search_roundtrip(tmp_path):
    s = toy()
    t = STV(s)
    face = t.add(t.root, "front face", rows=np.arange(0, 50))
    grp = t.add(face, "LAN ports", rows=np.arange(10, 20))
    p1 = t.add(grp, "LAN 1G port", rows=np.arange(10, 14), text="1G", detection=("c5.6", 5))
    assert t.part_of(12) == [p1, grp, face, t.root] and t.part_of(60) == []
    assert t.path(p1) == "object / front face / LAN ports / LAN 1G port"
    assert t.search("lan") == [grp, p1] and t.search("1g") == [p1]
    t.meta["c5.6"] = {0: t.root, 5: p1}
    t.save(tmp_path)
    u = STV.load(tmp_path, s)
    assert u.path(p1) == t.path(p1) and (u.label == t.label).all() and u.conflicts == t.conflicts
    assert u.parts[p1].detections == [("c5.6", 5)] and u.meta["c5.6"] == {0: t.root, 5: p1}
