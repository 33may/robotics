from types import SimpleNamespace as NS

from inspection.investigation.iwm.e05_parts_in_v import common as C5
from inspection.investigation.iwm.e07_update_loop import trace as Tr

KEEP_C56 = C5.DATA / "2209-compute-ray-demo5" / "1mm" / "keep_agent" / "after_c5.6" / "store"


def test_keep_trace_pairs_every_tool_call_with_its_result_and_ends_with_the_reply():
    ev, t = Tr.turn_events(KEEP_C56, "reconcile")
    calls = [x for x in t["turns"] if x["type"] == "reply" and "tool" in x]
    tools = [e for e in ev if e["kind"] == "tool"]
    assert len(tools) == len(calls) and [e["tool"] for e in tools] == [c["tool"] for c in calls]
    assert ev[-1]["kind"] == "final" and "decisions" in ev[-1]["fields"]


def test_artifacts_shown_only_for_image_tools_of_this_transcript():
    ev, t = Tr.turn_events(KEEP_C56, "reconcile")
    for e in ev:
        if e["kind"] == "tool" and e["image"] is not None:
            assert e["tool"] in ("show", "show_existing") and e["image"].stat().st_mtime >= t["t"] - 2


def _stv(parts):
    return NS(parts={p: NS(name=n, parent=par, detections=det, children=[], rows=[]) for p, (n, par, det) in parts.items()})


def test_diff_names_new_extended_moved_renamed_parts():
    prev = _stv({0: ("obj", None, [("a", 0)]), 1: ("face", 0, [("a", 1)]), 2: ("port", 1, [("a", 2)])})
    cur = _stv({0: ("obj", None, [("a", 0), ("b", 0)]), 1: ("Front", 0, [("a", 1)]), 2: ("port", 3, [("a", 2), ("b", 5)]),
                3: ("panel", 1, [("b", 2)])})
    ch = Tr.diff(prev, cur, "b")
    assert ch == {"new": [3], "extended": [0, 2], "moved": [2], "renamed": [1], "gone": []}
