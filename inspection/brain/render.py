#!/usr/bin/env python3
"""Context rendering for the orchestrator — what the blind planner sees.

The orchestrator never receives pixels (Anton 2026-08-25). Everything it knows
arrives as text, and there are two kinds of text which must NOT travel by the
same mechanism:

- **Content accumulates.** View descriptions and the model's own reasoning live
  in the append-only transcript. The SDK carries them for free, and re-reading
  them costs nothing — the store is our filesystem.
- **State is recomputed.** Position, coverage and the move menu are derived
  from the RunStore and the Supervisor by deterministic code on every turn. No
  model ever authors them (the three-trust-levels rule: `eyes/AGENTS.md`).

State is re-rendered EVERY turn and stamped `· turn N ·`. An append-only
transcript keeps every block ever sent, so an old one cannot be retracted —
only dated. On `2508-aiduck2` menu 1 said "[9,0] — opposite side" and menu 6
said "[9,0] — 30 deg right"; both were true when sent, both stayed in context,
and nothing marked which was current. The stamp is that mark.

The menu is COMPLETE at the current elevation — no cap. The old 8-row cap was
measured doing harm on `2508-aiduck2`: 6 of its 7 menus offered only 4
distinct azimuths, spending half the budget on elevation variants of the cell
the arm stood on, and [9, 0] — the cell that held the answer — appeared in 2
of 7. A full ring is bounded by construction (h_bins - 1 rows plus visited
cells at other levels), so the AgentOccam distractor-pruning lever
(`2410.13825`, WebArena 16.5 -> 25.8%) is served by having only four VERBS,
not by hiding viewpoints.

Bearings are RELATIVE and egocentric ("30° right", "opposite"), the address
absolute. Absolute for the system, egocentric for the model (Anton 2026-08-12,
reaffirmed 2026-08-25 after the first live run): the model picks from a
labelled list and never has to COMPUTE an azimuth address itself.
"""

#: Which way `h` counts. The orchestrator is blind, so this is pure convention
#: — but it has to be stated once and used everywhere, including the prompt,
#: or "go 30 right" and "h+1" quietly disagree.
_RIGHT_IS_INCREASING_H = True

#: MEASURED, not assumed (2026-08-25, over every capture in `2508-aiduck`):
#: after `ViewTools.get_view` uprights a frame, the image's +x axis points
#: toward INCREASING azimuth on all eight poses — `dot(image_right, +az) =
#: +1.00`, including the `rgb_rotation_deg = 180` captures whose raw camera x
#: is flipped. Combined with `_RIGHT_IS_INCREASING_H`, the vision tier's
#: camera-centric "left"/"right" and this menu's "left"/"right" are THE SAME
#: WORD, and the conversion between them is the identity — which is what lets
#: a `recommendation` phrase cross from the subagent to this menu verbatim.
#:
#: That is a fact about our rig, not about cameras. It holds because the
#: upright step removes the roll; a rig that mounted the wrist camera mirrored,
#: or an upright convention that flipped handedness, would invert it and every
#: "move left" the planner makes would go the wrong way. Re-measure before
#: trusting it on new hardware — the check is nine lines: take `T_base_cam`,
#: rotate the camera x-axis by `rgb_rotation_deg`, dot it with the azimuth
#: tangent `(-y, x, 0)` about the target centre, and confirm the sign.
_IMAGE_RIGHT_IS_INCREASING_AZIMUTH = True

#: Visited menu rows carry the subagent's `saw` clause. Long enough to
#: distinguish "logo face-on" from "logo edge-on", short enough that a ring of
#: them does not bury the addresses they annotate.
NOTE_CAP = 72


def azimuth_deg(h, h_bins):
    return h * 360.0 / h_bins


def gloss(cell, h_bins, v_elevs):
    """Deterministic name for a cell — geometry, never opinion."""
    if cell is None or cell == "survey":
        return "survey pose (off-grid)"
    h, v = cell
    return (f"[{h}, {v}] azimuth {azimuth_deg(h, h_bins):.0f} deg, "
            f"elevation {v_elevs[v]:.0f} deg")


def _offset(cur_h, h, h_bins):
    """Azimuth offset as (degrees, word) — how a person would say the turn.

    The first live run showed why relative wording matters: the menu spoke in
    absolute azimuths and the model chose cells it could not reach, because
    nothing in the text told it where it was standing relative to them.
    """
    dh = (h - cur_h) % h_bins
    if dh > h_bins // 2:
        dh -= h_bins
    deg = abs(dh) * 360.0 / h_bins
    if dh == 0:
        return 0.0, "same azimuth"
    if abs(dh) * 2 == h_bins:
        return deg, "opposite"
    side = "right" if (dh > 0) == _RIGHT_IS_INCREASING_H else "left"
    return deg, side


def saw_note(view, cap=NOTE_CAP):
    """The visited-row clause: what that viewpoint showed, in one clipped line."""
    if not view or not view.get("saw"):
        return ""
    text = " ".join(str(view["saw"]).split())
    return text if len(text) <= cap else text[:cap - 1].rstrip(" ,;") + "…"


def survey_bearing_line(az_deg, h_float, h_bins):
    """Where the hand-taught survey pose sits on the ring — code-computed
    (`ViewTools.survey_bearing`), so the planner can anchor the survey
    declaration's frame-relative words to addresses. Nearest cell plus the
    signed offset, because "between [1] and [2]" for a pose 0.2° past cell
    [1] reads as halfway."""
    near = round(h_float) % h_bins
    delta = az_deg - (round(h_float) * 360.0 / h_bins)
    return (f"seen from azimuth {az_deg:.0f}° ≈ azimuth cell [{near}] "
            f"({delta:+.0f}°); image-right is increasing azimuth")


def _levels_line(v_elevs):
    """The elevation scale, stated once — never repeated per row."""
    names = {0: "lowest", len(v_elevs) - 1: "top"}
    if len(v_elevs) == 3:
        names[1] = "middle"
    bits = []
    for i, e in enumerate(v_elevs):
        name = names.get(i)
        bits.append(f"{i} {name} ({e:.0f}°)" if name else f"{i} ({e:.0f}°)")
    return "elevation levels: " + ", ".join(bits)


def state_block(turn, cell, h_bins, v_elevs, n_visited, n_reachable,
                n_findings):
    """Where the arm is, the elevation scale, and the two counts.

    Coverage is reported against the REACHABLE SPHERE, not against how many
    frames happen to exist. In a live run every move captures and then
    inspects, so "inspected N of N captured" counts to itself and says
    nothing; the fraction that supports a negative answer is the one over the
    sphere.
    """
    if cell is None or cell == "survey":
        at = "at the survey pose (off-grid)"
    else:
        h, v = cell
        at = (f"at [{h}, {v}] — azimuth {azimuth_deg(h, h_bins):.0f}°, "
              f"elevation level {v}")
    return (f"STATE · turn {turn} · {at}\n"
            f"        {_levels_line(v_elevs)}\n"
            f"        visited {n_visited} of {n_reachable} reachable · "
            f"findings {n_findings}")


def moves_block(turn, tools, cur_cell, agent_seen, nav=None, seen=None):
    """Where the arm may go next, and what the visited viewpoints showed.

    One blended list: the COMPLETE azimuth ring at the current elevation, plus
    every visited cell at the other elevations slotted beside its azimuth
    sibling — without those, the model loses all memory of level 0 the moment
    it moves to level 1. The address itself carries the level.

    `nav` is the Supervisor's live picture — reachable / visited / blocked
    (`SupervisorMover.nav`). Without it (replay) the only truth available is
    which cells a sweep already captured, and uncaptured cells are simply not
    options. Getting this distinction wrong is what made the first live run
    offer "[1, 0] — no capture here" for the only cells worth moving to.

    `seen` maps a visited cell to its `view` block. `saw` renders on the row;
    `recommendation` hangs UNDER the row as its own line, so the origin of the
    recommendation is structural — every one-slot representation of it has
    produced the same bug (a direction with no "from where"). The phrase is
    rendered verbatim: the orchestrator, not this code, turns it into an
    address (Anton 2026-08-26 — the whole contract is prompt text).
    """
    h_bins, v_elevs = tools._h_bins, tools._v_elevs
    n_v = len(v_elevs)
    seen = seen or {}

    def status(c):
        """(is this cell an option, how to mark it)."""
        if nav is None:
            # Replay: the sweep is the world. A cell without a capture is not
            # somewhere to go, it is somewhere with nothing to read.
            if tools.view_at(c) is None:
                return False, ""
            return True, "visited" if c in agent_seen else "not visited"
        if c not in nav["reachable"]:
            return False, ""                  # unreachable: not an option at all
        if c in nav["blocked"]:
            # Soft: the planner refused it from HERE. It may open up after
            # the next move, so it is named rather than hidden.
            return True, "no path from here right now"
        if c in nav["visited"] or c in agent_seen:
            return True, "visited"
        return True, "not visited"

    def fmt(c, note, mark):
        row = f"  move({list(c)})".ljust(17) + f"{note:<15}" + f"· {mark}"
        view = seen.get(c)
        got = saw_note(view)
        if got and mark.startswith("visited"):
            row += f" · {got}"
        rec = (view or {}).get("recommendation")
        if rec and mark.startswith("visited"):
            row += f"\n      recommends: {rec}"
        return row

    if cur_cell is None or cur_cell == "survey":
        # Off-grid: no origin to be relative to, so rows carry the absolute
        # geometry instead of a bearing.
        rows = []
        for c in [(h, v) for v in range(n_v) for h in range(h_bins)]:
            ok, mark = status(c)
            if ok:
                rows.append(fmt(c, f"az {azimuth_deg(c[0], h_bins):.0f}° "
                                   f"lvl {c[1]}", mark))
        header = f"MOVES · turn {turn} · you are at the survey pose (off-grid)"
    else:
        ch, cv = cur_cell
        ring = [(h, cv) for h in range(h_bins) if h != ch]
        # Visited cells at OTHER elevations still render. `agent_seen` is what
        # this agent has read; nav's visited adds cells the arm reached in a
        # live run even if a read there failed.
        elsewhere = {tuple(c) for c in agent_seen}
        if nav is not None:
            elsewhere |= {tuple(c) for c in nav["visited"]}
        elsewhere = sorted(c for c in elsewhere
                           if c[1] != cv and 0 <= c[0] < h_bins)

        def key(c):
            deg, word = _offset(ch, c[0], h_bins)
            # Right before left at the same offset, current elevation before
            # its visited siblings — matches how the ring reads outward.
            return (deg, 0 if word != "left" else 1, abs(c[1] - cv))

        rows = []
        for c in sorted(ring + elsewhere, key=key):
            ok, mark = status(c)
            if ok:
                deg, word = _offset(ch, c[0], h_bins)
                rows.append(fmt(c, f"{deg:>3.0f}° {word}", mark))
        levels = "/".join(str(i) for i in range(n_v))
        header = (f"MOVES · turn {turn} · you are at [{ch}, {cv}] — "
                  f"elevation level {cv} of {levels}")

    if not rows:
        return "MOVES · none available"
    return header + "\n" + "\n".join(rows)


def opening(question, survey_text, tools, agent_seen, nav=None):
    """The first turn: the question, what the survey showed, where we can go."""
    return (f"QUESTION · {question}\n\n"
            f"SURVEY VIEW · {survey_text}\n\n"
            f"{moves_block(0, tools, None, agent_seen, nav=nav)}\n\n"
            "Plan first: decompose the question into measurable criteria and "
            "derive the visit order from the survey geometry, then inspect "
            "and move until you can answer.")


#: Menu identity. THIS MODULE IS THE MENU: `state_block` + `moves_block` are
#: the pure function (ViewState, MenuInput) -> the text the planner reads, so
#: the menu's content hash is this file's source hash. Coarse but honest —
#: v1 snapshots the renderer wholesale rather than the template it does not
#: yet have; when the blocks become data, `template` fills in and the hash
#: narrows to it (schema.py:MenuDef).
MENU_ID = "brain-render"
MENU_VERSION = "1.0.0"

#: The tool surface offered alongside the menu — `brain/loop.py:_build_tools`.
#: Named here rather than imported: `loop` imports this module, and the menu
#: definition must not depend on the loop that renders with it.
MENU_VERBS = ("plan", "inspect", "move", "answer")


def menu_def():
    """This renderer as a `MenuDef` record — snapshot written per AI session.

    `content_hash` is the sha256 of this source file, `code_sha` the repo's
    HEAD: version alone binds nothing (a semver says what we intended, the
    hashes say what actually ran), which is the scores.jsonl provenance
    pattern applied to the menu.
    """
    import hashlib
    from pathlib import Path

    from inspection.record.schema import MenuDef
    from inspection.record.writer import git_sha

    src = Path(__file__).resolve().read_bytes()
    return MenuDef(menu_id=MENU_ID, version=MENU_VERSION,
                   content_hash=hashlib.sha256(src).hexdigest(),
                   code_sha=git_sha(), verbs=list(MENU_VERBS),
                   renders_kinds=["viewsphere"])
