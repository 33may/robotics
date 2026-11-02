#!/usr/bin/env python3
"""The state hook renderer. Run: p vbti/inspection/tests/test_render.py

These pin the 2026-08-26 hook format: stamped STATE + a complete, uncapped
MOVES ring with visited cells at other elevations blended in, and the
subagent's recommendation hanging verbatim under the row that produced it.
"""
from inspection.brain.render import (moves_block, opening, saw_note,
                                     state_block)


class FakeTools:
    """Just the surface moves_block touches — the renderer is pure text."""

    def __init__(self, captured=None, h_bins=12, v_elevs=(10.0, 40.0, 70.0)):
        self._h_bins = h_bins
        self._v_elevs = list(v_elevs)
        self._captured = None if captured is None else set(captured)

    def view_at(self, c):
        if self._captured is None:
            return "view"
        return "view" if tuple(c) in self._captured else None


def _nav(h_bins=12, n_v=3, visited=(), blocked=()):
    return {"reachable": {(h, v) for h in range(h_bins) for v in range(n_v)},
            "visited": set(visited), "blocked": set(blocked)}


def test_state_block_names_position_scale_and_counts():
    s = state_block(7, (7, 1), 12, [10.0, 40.0, 70.0], 5, 31, 6)
    assert "STATE · turn 7 · at [7, 1]" in s
    assert "azimuth 210°" in s and "elevation level 1" in s
    assert "0 lowest (10°), 1 middle (40°), 2 top (70°)" in s
    assert "visited 5 of 31 reachable · findings 6" in s


def test_moves_ring_is_complete_and_uncapped():
    # The old 8-row cap offered only 4 distinct azimuths in 6 of 7 menus on
    # `2508-aiduck2` and hid the answer cell in 5 of 7. All 11 render now.
    m = moves_block(3, FakeTools(), (7, 1), set(), nav=_nav())
    rows = [l for l in m.splitlines() if l.strip().startswith("move(")]
    assert len(rows) == 11                       # every azimuth but our own
    assert "MOVES · turn 3 · you are at [7, 1] — elevation level 1 of 0/1/2" in m
    assert rows[0].startswith("  move([8, 1])") and "30° right" in rows[0]
    assert rows[1].startswith("  move([6, 1])") and "30° left" in rows[1]
    assert "180° opposite" in m
    assert "more" not in m                       # no cap, so no "(+N more)"


def test_visited_rows_carry_saw_and_the_recommendation_hangs_under_origin():
    seen = {(3, 0): {"saw": "plain side, the mark turned away right",
                     "recommendation": "around right"}}
    m = moves_block(5, FakeTools(), (7, 1), {(3, 0)},
                    nav=_nav(visited=[(3, 0)]), seen=seen)
    lines = m.splitlines()
    i = next(n for n, l in enumerate(lines) if "move([3, 0])" in l)
    assert "· visited · plain side, the mark turned away right" in lines[i]
    assert lines[i + 1].strip() == "recommends: around right"
    # The recommendation names no cell and gets none here: the orchestrator
    # resolves the phrase itself (the contract is prompt text on both sides).
    assert "recommends: around right" in m and "recommends move" not in m


def test_unvisited_cells_at_other_elevations_do_not_render():
    # The ring is complete at the CURRENT elevation; other levels appear only
    # once visited. 36 rows of every-cell-every-level buried the azimuths.
    m = moves_block(2, FakeTools(), (7, 1), {(3, 0)},
                    nav=_nav(visited=[(3, 0)]))
    assert "move([3, 0])" in m
    assert "move([5, 0])" not in m and "move([3, 2])" not in m


def test_replay_offers_only_captured_cells():
    # No Supervisor: the sweep is the world. A cell without a capture is not
    # somewhere to go, it is somewhere with nothing to read.
    tools = FakeTools(captured={(6, 0), (8, 0), (9, 0)})
    m = moves_block(4, tools, (8, 0), {(8, 0)}, nav=None)
    assert "move([9, 0])" in m and "move([6, 0])" in m
    assert "move([2, 0])" not in m


def test_survey_menu_is_absolute_because_there_is_no_origin():
    m = moves_block(0, FakeTools(captured={(0, 0), (6, 1)}), None, set())
    assert "survey pose" in m
    assert "az 0° lvl 0" in m and "az 180° lvl 1" in m
    o = opening("is there a logo?", "a cup on a table",
                FakeTools(captured={(0, 0)}), set())
    assert "QUESTION · is there a logo?" in o and "MOVES · turn 0" in o


def test_survey_bearing_line_names_the_nearest_cell():
    from inspection.brain.render import survey_bearing_line
    line = survey_bearing_line(30.2, 30.2 / 30.0, 12)
    assert "azimuth 30°" in line and "[1]" in line and "(+0°)" in line
    line = survey_bearing_line(344.0, 344.0 / 30.0, 12)
    assert "[11]" in line and "(+14°)" in line


def test_saw_note_clips_but_never_mid_word_garbage():
    long = {"saw": "x" * 200}
    n = saw_note(long)
    assert len(n) <= 72 and n.endswith("…")
    assert saw_note({}) == "" and saw_note(None) == ""


def main():
    test_state_block_names_position_scale_and_counts()
    test_moves_ring_is_complete_and_uncapped()
    test_visited_rows_carry_saw_and_the_recommendation_hangs_under_origin()
    test_unvisited_cells_at_other_elevations_do_not_render()
    test_replay_offers_only_captured_cells()
    test_survey_menu_is_absolute_because_there_is_no_origin()
    test_survey_bearing_line_names_the_nearest_cell()
    test_saw_note_clips_but_never_mid_word_garbage()
    print("OK test_render")


if __name__ == "__main__":
    main()
