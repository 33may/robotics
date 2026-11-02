from types import SimpleNamespace as NS


def test_details_never_hang_on_the_object_and_the_agents_placement_stands():
    from inspection.investigation.iwm.e07_update_loop.loop import enforce_structure
    view = {"nodes": {0: {"parent": None}, 3: {"parent": 0}, 5: {"parent": 0}, 7: {"parent": 3}, 9: {"parent": 5},
                      55: {"parent": 0}}}
    stv = NS(root=0)
    dec = {0: ("same", 4),                       # the agent called the frame a face -> the object
           3: ("under", 2, "LAN face"),          # left under #0 by the segmenter, placed on P2: stands (2026-10-15)
           5: ("same", 6),                       # an existing face: kept
           7: ("under", 0, "LAN port"),          # a detail on the object -> on its new face #3
           9: ("under", 0, "label"),             # a detail on the object -> on the existing face P6
           55: ("under", 7, "antenna ports")}    # e07 step 6: left under #0, the agent put it on P7: stands
    out, fixes = enforce_structure(dec, view, stv)
    assert out[0] == ("same", 0) and out[3] == ("under", 2, "LAN face") and out[5] == ("same", 6)
    assert out[7] == ("under", ("node", 3), "LAN port") and out[9] == ("under", 6, "label")
    assert out[55] == ("under", 7, "antenna ports")
    assert len(fixes) == 3


def test_ids_with_names_appended_parse():
    from inspection.investigation.iwm.e07_update_loop.loop import _robust_id
    assert [_robust_id(x) for x in ("#2 connector panel", "P15", 4, " #7")] == [2, 15, 4, 7]
