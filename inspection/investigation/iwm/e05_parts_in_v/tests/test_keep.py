import numpy as np

from inspection.investigation.iwm.e05_parts_in_v.stv import STV, Surface


def toy(n=20):
    g = np.stack(np.meshgrid(np.arange(n), np.arange(n), [0], indexing="ij"), -1).reshape(-1, 3)
    return Surface(g.astype(np.int32), np.zeros(len(g), np.float32), np.ones(len(g), np.float32), 0.001)


def view_with(nodes):
    return {"id": "x", "tree": {"nodes": nodes}, "nodes": {n["id"]: n for n in nodes}}


def test_baseline_same_and_under_and_face_root():
    from inspection.investigation.iwm.e05_parts_in_v import keep as K
    t = STV(toy())
    face = t.add(t.root, "front face", rows=np.arange(0, 200))
    port = t.add(face, "port", rows=np.arange(40, 48))
    view = view_with([
        {"id": 0, "parent": None, "depth": 0, "name": "front face", "text": None, "children": [1, 2]},
        {"id": 1, "parent": 0, "depth": 1, "name": "a port", "text": None, "children": []},
        {"id": 2, "parent": 0, "depth": 1, "name": "a label", "text": None, "children": []}])
    rows = {0: np.arange(0, 200), 1: np.arange(41, 49), 2: np.arange(100, 110)}
    d = K.baseline_decisions(t, view, rows)
    assert d[0] == ("same", face)                 # the root is a face here, and the geometry says so
    assert d[1] == ("same", port)                 # 7 of 8 rows on the port
    assert d[2] == ("under", face, "a label")     # nothing holds it at the child level


def test_apply_extends_and_adds_and_keeps_invariants():
    from inspection.investigation.iwm.e05_parts_in_v import keep as K
    t = STV(toy())
    face = t.add(t.root, "front face", rows=np.arange(0, 200))
    port = t.add(face, "port", rows=np.arange(40, 48))
    view = view_with([
        {"id": 0, "parent": None, "depth": 0, "name": "front face", "text": None, "children": [1, 2]},
        {"id": 1, "parent": 0, "depth": 1, "name": "a port", "text": None, "children": []},
        {"id": 2, "parent": 0, "depth": 1, "name": "a label", "text": "AB-12", "children": []}])
    rows = {0: np.arange(0, 210), 1: np.arange(41, 49), 2: np.arange(100, 110)}
    res = K.apply_decisions(t, view, rows, {0: ("same", face), 1: ("same", port), 2: ("under", face, "a label")})
    t.check()
    assert len(t.parts[port].rows) == 9 and 48 in t.parts[port].rows            # extended by one row
    new = t.meta["x"][2]
    assert t.parts[new].name == "a label" and t.parts[new].parent == face and t.parts[new].text == "AB-12"
    assert (t.label[100:110] == new).all() and 209 in t.parts[face].rows and 209 in t.parts[t.root].rows
    assert res["same"] == 2 and res["under"] == 1 and res["conflicts"] == 0
    assert ("x", 1) in t.parts[port].detections


def test_normalise_decisions_accepts_strings_lists_and_dicts():
    from inspection.investigation.iwm.e05_parts_in_v import keep as K
    raw = {"0": "same:P0", "#5": ["same", 11], "6": ["same", "P12"], "2": ["under", 2, "lower front panel"],
           "21": "under:P2:audio ports group", "7": {"op": "under", "part": "P2", "name": "antenna port 1"},
           "8": ["drop"], "9": "drop", "x": "same:P1", "10": ["same"],
           "4": "under:#2:barcode", "16": ["under", "#2", "model number text"], "27": {"op": "under", "part": "#3", "name": "QR"}}
    d = K.normalise_decisions(raw)
    assert d[0] == ("same", 0) and d[5] == ("same", 11) and d[6] == ("same", 12)
    assert d[2] == ("under", 2, "lower front panel") and d[21] == ("under", 2, "audio ports group")
    assert d[7] == ("under", 2, "antenna port 1") and d[8] == ("drop",) and d[9] == ("drop",)
    assert "x" not in d and 10 not in d                       # junk is skipped, never raises
    # a new node of this view as the parent: "#2" -> ("node", 2); a bare number or "P2" -> existing part 2
    assert d[4] == ("under", ("node", 2), "barcode") and d[16] == ("under", ("node", 2), "model number text")
    assert d[27] == ("under", ("node", 3), "QR")


def test_normalise_is_idempotent_and_merge_keeps_node_targets():
    from inspection.investigation.iwm.e05_parts_in_v import keep as K
    t = STV(toy())
    face = t.add(t.root, "front face", rows=np.arange(0, 200))
    view = view_with([
        {"id": 0, "parent": None, "depth": 0, "name": "front face", "text": None, "children": [2]},
        {"id": 2, "parent": 0, "depth": 1, "name": "label", "text": None, "children": [4]},
        {"id": 4, "parent": 2, "depth": 2, "name": "barcode", "text": None, "children": []}])
    once = K.normalise_decisions({"0": "same:P1", "2": "under:P1:label", "4": "under:#2:barcode"})
    assert K.normalise_decisions(once) == once                       # VlmAgent normalises through extras= already
    merged = K.merge_final({}, once, view, t)
    assert merged[4] == ("under", ("node", 2), "barcode")           # a new-node parent survives the merge
    assert merged[0] == ("same", face)
    assert 9 not in K.merge_final({}, {9: ("same", 1)}, view, t)     # unknown node dropped
    assert 2 not in K.merge_final({}, {2: ("same", 77)}, view, t)    # unknown part dropped


def test_overlap_shares_count_row_membership_not_labels():
    from inspection.investigation.iwm.e05_parts_in_v import keep as K
    t = STV(toy())
    a = t.add(t.root, "A", rows=np.arange(0, 10))
    b = t.add(t.root, "B", rows=np.arange(5, 15))                     # rows 5-9 labelled A (first wins), held by B too
    sh = K.overlap_shares(t, np.arange(5, 10))
    assert sh[a] == 1.0 and sh[b] == 1.0 and sh[t.root] == 1.0        # B holds them even though the label says A


def test_undecided_root_defaults_to_object_and_nodes_under_parent():
    from inspection.investigation.iwm.e05_parts_in_v import keep as K
    t = STV(toy())
    face = t.add(t.root, "front face", rows=np.arange(0, 200))
    view = view_with([
        {"id": 0, "parent": None, "depth": 0, "name": "box", "text": None, "children": [5]},
        {"id": 5, "parent": 0, "depth": 1, "name": "top face", "text": None, "children": []}])
    d = K.resolve_defaults(t, view, {})
    assert d[0] == ("same", t.root) and d[5] == ("under", t.root, "top face")
