# View hints — telling the planner how far, not just which way (design)

2026-08-26. **Not implemented.** This fixes the contract between the vision
tier and the orchestrator for viewpoint hints: what the subagent emits, how the
harness turns it into a cell, and how the state hook renders it. Companion to
`2026-08-20-image-machinery-design.md` (the eyes tier) and
`2026-08-19-loop-v1-design.md` (the loop).

## The problem, measured

The planner single-steps around the object. `2508-aiduck2` spent 8 views on an
answer that needed 2: from `[3, 0]` it stepped `[2,0] → [1,0] → [1,1]`, one 30°
cell at a time, before giving up and jumping to the far side.

The cause is in the data, not in the planner. Over the two runs that carry a
`framing` block (n = 9 findings — a weak sample, and every conclusion here
inherits that):

- The whole vocabulary is **three strings**: `face-on`, `turning away to the
  left`, `turning away to the right`. The prompt offered `edge-on` and
  `not visible`; the model used **neither, ever**. The realised magnitude scale
  is binary.
- `turning away to the right` was emitted at true offsets of **30°, 60°, 60°
  and 120°** — one phrase spanning 3 cells. Collapsed to magnitude alone the
  span is 30°–180°, 5 cells.
- No choice of object-front azimuth makes the language ordinal: sweeping θ_f
  across its entire 74° viable plateau, the most favourable value still leaves
  60° inside one phrase.
- **Direction is fine** — 5/6 correct (the sixth a genuine 180° tie), and
  `better` 3/3. Two independent fits, one from the magnitude channel and one
  from the direction words alone, agree on where the object's front is.

So the planner receives identical text whether it should step once or four
times, and stepping once is the only safe reading. Method and figures:
`inspection/investigation/phrase_calib.py`, `/tmp/phrase_calib/`.

## What was ruled out

Back-projecting the crop box through the view's own aligned depth to a local
surface normal, and naming the cell that opposes it. Tested over 5 runs and 18
scored cases (`/tmp/normal_probe/`, `fig1_money.png`).

It reaches 9/10 on homing cases — but a radial baseline that computes **no
normal at all** reaches 7/10 on the same cases and 6/8 on the "already at a good
view, stay put" control where the normal version gets 4/8. Union: **13/18 for
both**. A baseline using **no depth at all** gets 12/17.

Two further reasons not to build it now:

- **The flagship case is a coin toss.** `2508-aiduck2 [8,0] → [9,0]` — the case
  the idea came from — lands on `[9, 0]` **49%** of the time when the crop box
  is jittered by the *measured* VLM re-crop noise (±11 px, ±36% area, from 52
  real within-transcript re-crops).
- **The failure is systematic.** When a feature is clipped by the silhouette,
  the visible fragment is by construction the part nearest the limb, so its
  normal points back at the camera you are already standing at. Two cases on the
  same mug with anchors 1.7 mm apart undershoot identically. That is exactly the
  situation the hint exists for.

Effective independent n is ~5, not 10 — five distinct correct answers over 3
objects and 2 logo instances. Genuine ≥2-azimuth-cell homing is n = 2.

**Kept in the pocket:** the radial version (back-project the crop, walk outward,
no PCA) is ~20 lines and scores the same. If the ordinal below proves weak, that
is the fallback — not the normal.

Detail: `perception/crop-normal-cannot-name-best-cell` in the agent memory tree.

## The contract

### What the subagent emits

`framing` is replaced by `view`. `target` and `better` are deleted; `recommend`
never becomes a field. Three sub-fields, each with exactly one consumer:

```json
{
  "evidence":  [...],
  "reasoning": "...",
  "answer":    "...",
  "view": {
    "saw":    "plain blue, no marking",
    "facing": "hidden right",
    "why":    "the printed side is turned fully away"
  }
}
```

| field | consumer | lifetime |
|---|---|---|
| `evidence` / `reasoning` / `answer` | orchestrator, in the tool result | once, then ages in the transcript |
| `saw` | orchestrator, as the visited-cell note | re-rendered every turn |
| `facing` | the harness, as arithmetic | never rendered |
| `why` | orchestrator, as the recommendation line | re-rendered every turn |

**Why `target` dies.** For a question about the object it is always `center` —
the viewsphere centres the object by construction. For a question about a
feature its two real contents belong elsewhere: `not visible` is
`facing: "hidden <dir>"`, and `clipped at the edge` is a view-quality complaint,
which is `why`. Position-in-frame is the residue and nothing downstream can act
on it.

**Why `better` dies.** It answered "which way to shift" with direction only.
`facing` answers the same question with magnitude, and it crossed to the
orchestrator once inside the tool result and never appeared in the menu.

**Why there is no `where`.** `where` and `facing` are the same judgement. The
subagent never names a cell; the harness does the arithmetic.

**Open:** whether `saw` and `why` stay separate. They are identical when the
view is good and diverge only when it is not — which is the negative-answer
case that coverage arguments rest on.

### The prompt

```
`facing` — how much of it you can still see, and which way it turns away.

A surface turned away from the camera is compressed horizontally: its width
in the image is cos(angle) of its true width. Judge by WIDTH.

Reply with EXACTLY one of these phrases, nothing else:

  "face-on"                                 0°    full width, not compressed
  "slightly left"   / "slightly right"     30°    ~85% of its width
  "half left"       / "half right"         60°    ~50% of its width
  "edge-on left"    / "edge-on right"      90°    a sliver, barely a line
  "hidden left"     / "hidden right"      >90°    you cannot see it at all

The angles define what each phrase means — do not report an angle yourself,
pick the phrase. left/right are directions in THIS frame. If it is hidden,
say which way the object's front turned out of view.
```

Foreshortening is the anchor because it makes the judgement *checkable against
the pixels* — width is measurable, "clearly turned" is not. Five levels, because
cells are 30° and nothing finer is usable.

Two deliberate bets, both untested:

- **Degrees are shown.** VLMs score near-zero at emitting metric rotations, so
  numbers were kept out of earlier designs. Here the number *defines the label*
  rather than being the answer. The risk is anchoring toward the middle of the
  scale. That makes with-degrees vs without-degrees the obvious A/B.
- **`hidden` maps to the opposite side.** `cos φ` is not invertible past 90°, so
  this is a **prior**, not a measurement: a mark you cannot see at all is most
  likely on the far face. It is also the highest-leverage row — `hidden → [9,0]`
  answers `2508-aiduck2` in one move instead of eight.

Pass 1's real lesson was that offering levels is not enough. The model had
`edge-on` and `not visible` available and used neither, so the prompt must force
a choice with a testable criterion per level.

### What the harness does

`facing` → degrees → cells, added to the **origin** cell's azimuth. Direction is
frame-local and, on this rig, image-right is increasing azimuth on every pose, so
the conversion is the identity (see
`perception/image-right-is-increasing-azimuth`; re-measure on new hardware).

A recommendation is a relation between **two** cells — `(observed-from,
points-to)`. Every representation with one slot has produced the same bug:
`better: "right"` had no origin; a draft hook line said "from here" after the arm
had moved; a draft menu row implied a visit to a cell never occupied. Store and
render it keyed by both.

Supersession: a newer observation of the same feature replaces an older one
regardless of which cell each pointed at.

## The state hook

Two blocks, rendered on **every** turn including `inspect`. Today the menu
renders only when the arm moved, so a chain of inspects on held views sees
nothing but a counter — in `2608-aicam` four consecutive inspects each received
`STATE · findings on record: N` and nothing else.

```
STATE · turn 7 · at [7, 1] — azimuth 210°, elevation level 1
        elevation levels: 0 lowest (10°), 1 middle (40°), 2 top (70°)
        visited 5 of 31 reachable · findings 6

MOVES · turn 7 · you are at [7, 1] — elevation level 1 of 0/1/2
  move([8, 1])     30° right    · not visited
  move([6, 1])     30° left     · not visited
  move([9, 1])     60° right    · not visited
  move([5, 1])     60° left     · not visited
  move([10, 1])    90° right    · not visited
  move([4, 1])     90° left     · not visited
  move([11, 1])   120° right    · not visited
  move([3, 1])    120° left     · not visited
  move([3, 0])    120° left     · visited · center, turning away to the left
      recommends move([9, 0]) — the printed side faces directly away from here
  move([0, 1])    150° right    · not visited
  move([2, 1])    150° left     · not visited
  move([2, 0])    150° left     · visited · center, turning away to the left
  move([1, 1])    180° opposite · visited · center of the frame, turning away to the right
      recommends move([3, 1]) — only the edge of the mark shows, it curves away right
  move([1, 0])    180° opposite · visited
```

Rules the format encodes:

- **Azimuth ring at the current elevation, complete.** `MENU_CAP` and `_ring`'s
  forced-opposite special case are deleted: 11 rows is bounded by construction,
  so nothing is hidden and no `(+N more)` line implies otherwise. Measured on
  `2508-aiduck2`, the current 8-row cap offered only **4 distinct azimuths** in 6
  of 7 menus, spending half its budget on elevation variants of the cell the arm
  was standing on — and `[9, 0]`, which held the answer, appeared in 2 of 7.
- **Elevation is a level, stated once, with its scale.** One instruction line,
  not three rows per azimuth.
- **Visited cells at other elevations still render**, slotted next to their
  azimuth sibling. Without this the model loses all memory of level 0 the moment
  it moves to level 1. The address carries the level, so no separate column.
- **`· turn N ·` on both blocks.** The transcript accumulates menus; without a
  stamp the model must date them by position. On `2508-aiduck2`, menu 1 said
  `[9,0] — opposite side, not yet visited` and menu 6 said `[9,0] — 30 deg
  right, not yet visited`; both true when sent, both present, neither marked.
- **Recommendations hang off the row of the cell that produced them.** Origin is
  then structural and no line has to say "from here". The target keeps its own
  plain row and carries no marker, so nothing renders twice.
- **No diagnostics.** No fit statistics, no point counts, no confidence, no
  provenance disclaimers, no supersession rules. Address plus the subagent's own
  sentence. See `meta/no-explanatory-text-in-devapp-ui`.

Cost: ~900 chars per turn, ~2.9k tokens across a 13-turn run. The current format
is ~817 chars and shows a third of the ring.

## Parked

- **`target`** — Anton is collecting more data on whether the object is always
  centred. Nothing depends on it.
- **Absolute vs relative `where`** — relative for now, since the geometry route
  did not earn a direct cell. Revisit only if a recapture changes that.
- **The object prior** — asking "where on a rubber duck would a logo be?" once,
  before any view. Free, orthogonal, and the north-star claim in miniature:
  world knowledge choosing where to point the camera. Deferred to its own
  session.

## Next

**Pass 2 is the measurement this design rests on.** Re-ask the recorded frames
with the prompt above and score against true angles, which are exact cell
arithmetic given a reference cell. No motion, no GPU, API calls only. It answers
the one thing Pass 1 could not: the model does not *volunteer* a magnitude, but
can it *pick* one when forced?

Check `hidden` first. It is the least measurable level and the highest leverage.
