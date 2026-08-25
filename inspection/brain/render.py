#!/usr/bin/env python3
"""Context rendering for the orchestrator — what the blind planner sees.

The orchestrator never receives pixels (Anton 2026-08-25). Everything it knows
arrives as text, and there are two kinds of text which must NOT travel by the
same mechanism:

- **Content accumulates.** View descriptions and the model's own reasoning live
  in the append-only transcript. The SDK carries them for free, and re-reading
  them costs nothing — the store is our filesystem.
- **State is recomputed.** Position, coverage and the action menu are derived
  from the RunStore and the Supervisor by deterministic code on every turn. No
  model ever authors them (the three-trust-levels rule: `eyes/AGENTS.md`).

State is emitted as a **delta, never a snapshot**. An append-only transcript
keeps everything you put in it, so a coverage table stops being true the moment
the next view lands — after 20 views the model carries 20 tables, 19 of them
false. "4/12 -> 5/12" stays true forever.

Two numbers that shaped this file:
- the action menu is capped (AgentOccam `2410.13825`: removing distractor
  actions ALONE lifted WebArena 16.5 -> 25.8%, the single biggest lever
  measured on agent loops);
- bearings are RELATIVE and egocentric ("30 deg right", "opposite"), while the
  address stays absolute. Absolute for the system, egocentric for the model
  (Anton 2026-08-12, reaffirmed 2026-08-25 after the first live run) — picking
  from a labelled list is not the 2%-accuracy allocentric regime, but the model
  must never have to COMPUTE the address itself.
"""

MENU_CAP = 8

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
#: WORD, and the conversion between them is the identity.
#:
#: That is a fact about our rig, not about cameras. It holds because the
#: upright step removes the roll; a rig that mounted the wrist camera mirrored,
#: or an upright convention that flipped handedness, would invert it and every
#: "move left" the planner makes would go the wrong way. Re-measure before
#: trusting it on new hardware — the check is nine lines: take `T_base_cam`,
#: rotate the camera x-axis by `rgb_rotation_deg`, dot it with the azimuth
#: tangent `(-y, x, 0)` about the target centre, and confirm the sign.
_IMAGE_RIGHT_IS_INCREASING_AZIMUTH = True

#: Menu rows carry a clause of what was seen from each visited cell. Long
#: enough to distinguish "logo face-on" from "logo edge-on", short enough that
#: eight of them do not bury the addresses they annotate.
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


def bearing(cur, nb, h_bins, v_elevs):
    """How to get from here to there, said the way a person would say it.

    "90 deg right", "opposite side", "one step up". The first live run showed
    why this matters: the menu spoke in absolute azimuths and the model chose
    cells it could not reach, because nothing in the text told it where it was
    standing relative to them.
    """
    if cur is None or cur == "survey":
        return gloss(nb, h_bins, v_elevs)
    dh = (nb[0] - cur[0]) % h_bins
    if dh > h_bins // 2:
        dh -= h_bins
    dv = nb[1] - cur[1]

    deg = abs(dh) * 360.0 / h_bins
    if dh == 0:
        turn = "same side"
    elif abs(dh) * 2 == h_bins:
        turn = "opposite side"
    else:
        side = "right" if (dh > 0) == _RIGHT_IS_INCREASING_H else "left"
        turn = f"{deg:.0f} deg {side}"

    if dv == 0:
        return turn
    updown = "up" if dv > 0 else "down"
    step = f"{abs(dv)} step{'s' if abs(dv) > 1 else ''} {updown}"
    step += f" (elevation {v_elevs[nb[1]]:.0f} deg)"
    return f"{turn}, {step}" if dh else step


def framing_line(framing):
    """The vision tier's viewing-geometry readout, as one line of prose.

    Travels with the finding, inside the tool result. This is the channel that
    was missing in `2508-aiduck`: at [9, 0] the subagent read the mark as "M>"
    and said nothing about the surface being oblique, so the planner had no
    signal for which way to orbit, guessed, and spent an approved move going
    the wrong way ([10, 0]) before correcting through [8, 0] to [7, 0].
    """
    if not framing:
        return ""
    bits = [f"{k}: {framing[k]}" for k in ("target", "facing", "better")
            if framing.get(k)]
    return "framing — " + "; ".join(bits) if bits else ""


def framing_note(framing, cap=NOTE_CAP):
    """The menu-row version: what this cell showed, in one clipped clause."""
    if not framing:
        return ""
    parts = [framing.get("target"), framing.get("facing")]
    text = ", ".join(p for p in parts if p)
    if not text:
        return ""
    text = " ".join(text.split())
    return text if len(text) <= cap else text[:cap - 1].rstrip(" ,;") + "…"


def snapshot(cur_cell, agent_seen, n_findings, n_reachable, n_visited):
    """The state a delta is computed against."""
    return {"cell": tuple(cur_cell) if isinstance(cur_cell, (tuple, list)) else cur_cell,
            "seen": len(agent_seen), "findings": n_findings,
            "reachable": n_reachable, "visited": n_visited}


def state_delta(prev, cur, h_bins, v_elevs):
    """What CHANGED. Empty string when nothing did — silence is cheaper.

    Coverage is reported against the REACHABLE SPHERE, not against how many
    frames happen to exist. In a live run every move captures and then
    inspects, so "inspected N of N captured" counts to itself and says
    nothing; the fraction that supports a negative answer is the one over the
    sphere.
    """
    if prev == cur:
        return ""
    lines = []
    if prev["cell"] != cur["cell"]:
        lines.append(f"now at {gloss(cur['cell'], h_bins, v_elevs)}")
    if prev["visited"] != cur["visited"] or prev["seen"] != cur["seen"]:
        lines.append(f"visited {cur['visited']}/{cur['reachable']} reachable "
                     f"viewpoints")
    if prev["findings"] != cur["findings"]:
        lines.append(f"findings on record: {cur['findings']}")
    return "STATE · " + " · ".join(lines) if lines else ""


def _ring(cur, h_bins, n_v):
    """Candidate cells, nearest-first, with the opposite side always offered.

    Nearest-first because refinement is the common move; the opposite side is
    forced in because for a question about a CUP the far face is the single
    most informative viewpoint and it would otherwise never make the cap.
    """
    if cur is None or cur == "survey":
        return [(h, v) for v in range(n_v) for h in range(h_bins)]
    ch, cv = cur
    cells = [(h, v) for v in range(n_v) for h in range(h_bins) if (h, v) != cur]

    def dist(c):
        dh = min((c[0] - ch) % h_bins, (ch - c[0]) % h_bins)
        return (dh, abs(c[1] - cv))

    cells.sort(key=dist)
    opposite = ((ch + h_bins // 2) % h_bins, cv)
    if opposite in cells:
        cells.remove(opposite)
        cells.insert(min(3, len(cells)), opposite)
    return cells


def action_menu(tools, cur_cell, agent_seen, nav=None, cap=MENU_CAP, seen=None):
    """Where the arm may go next, and what the visited ones showed.

    `nav` is the Supervisor's live picture — reachable / visited / blocked
    (`SupervisorMover.nav`). Without it (replay) the only truth available is
    which cells a sweep already captured, and the menu says so. Getting this
    distinction wrong is what made the first live run offer
    "[1, 0] — no capture here" for the only cells worth moving to.

    `seen` maps an already-read cell to its `framing_note`. This is the run's
    view ledger, and the menu is the right place for it precisely BECAUSE the
    menu is re-rendered from scratch on every move: a coverage table pasted
    into an append-only transcript rots on the next capture (see the module
    docstring), but a table rebuilt at the moment of each decision cannot. It
    puts "what did I get from over there" directly beside "here is how to go
    there", which is the comparison the planner is actually making.
    """
    h_bins, v_elevs = tools._h_bins, tools._v_elevs
    n_v = len(v_elevs)
    rows, dropped = [], 0
    seen = seen or {}

    for c in _ring(cur_cell, h_bins, n_v):
        if len(rows) >= cap:
            dropped += 1
            continue
        note = bearing(cur_cell, c, h_bins, v_elevs)
        if nav is None:
            # Replay: the sweep is the world. A cell without a capture is not
            # somewhere to go, it is somewhere with nothing to read.
            if tools.view_at(c) is None:
                continue
            mark = "already inspected" if c in agent_seen else "not yet inspected"
        else:
            if c not in nav["reachable"]:
                continue                      # unreachable: not an option at all
            if c in nav["blocked"]:
                # Soft: the planner refused it from HERE. It may open up after
                # the next move, so it is named rather than hidden.
                mark = "no path from here right now"
            elif c in nav["visited"] or c in agent_seen:
                mark = "already visited"
            else:
                mark = "not yet visited"
        row = f"  move({list(c)}) — {note}, {mark}"
        got = framing_note(seen.get(c))
        if got:
            row += f"\n      seen from there: {got}"
        rows.append(row)

    if not rows:
        return "MOVES · none available"
    out = "MOVES · absolute addresses, bearings relative to where you are now\n"
    out += "\n".join(rows)
    if dropped:
        # Never let a cap read as "that was everything".
        out += f"\n  (+{dropped} more reachable viewpoints — ask if you need one)"
    return out


def opening(question, survey_text, tools, agent_seen, nav=None):
    """The first turn: the question, what the survey showed, where we can go."""
    return (f"QUESTION · {question}\n\n"
            f"SURVEY VIEW · {survey_text}\n\n"
            f"{action_menu(tools, None, agent_seen, nav=nav)}\n\n"
            "Decompose the question into measurable criteria and record them "
            "with plan(), then inspect and move until you can answer.")
