# Schema v1 — three-pass review, merged verdict

2026-09-03. Three independent passes over `inspection/record/schema.py`:
**completeness** (promised-vs-code diff), **advocate** (load-bearing properties),
**critic** (stress-test against real usage + on-disk data). Full reports in
`review/{completeness,advocate,critic}.md`. This file is the merged verdict and
fix plan.

Overall: the structure holds (all three agree the step join key, method/address
split, ViewState, config snapshot, transcript-once are right and load-bearing).
The gaps concentrate in: dropped entities (Session), missing failure/mock
representation, the versioning/manifest mechanics, and ViewState being narrower
than the menu's real input.

---

## Bucket A — mechanical fixes (no design decision; apply via TDD)

| fix | source |
|---|---|
| A1 `SessionRecord` model (serial, resolution, depth_scale, intrinsics ×4, IR extrinsics) + run relation — also closes depth↔scale positional join | comp#1, advocate invariant 2 |
| A2 token `usage` fields on AIRunRecord + TranscriptRecord | comp#2 |
| A3 `AIRunRecord.opening_prompt` verbatim | comp#3 |
| A4 `AnswerRecord.evidence_images` typed ({step_id, transcript_id, artifact, note}) | comp#4 |
| A5 `ViewState.extent` restored | comp#6 |
| A6 `MenuDef` on-disk location declared (`ai/<seq>/menu.json`) + `code_sha` (semver+code-hash provenance pattern) | comp#7, critic#20 |
| A7 `TranscriptRecord.t`; `answer` widened to `dict\|list\|str\|None` (archive has 2 list answers); `answer_schema` field | critic#17 |
| A8 `turns` entries require a `type` key (minimal discriminator; inner shapes stay loose) | critic#18 |
| A9 `GeometryStats.fused_points` (post-merge cloud size — replay's second anchor, today regexed from free text) | critic#2 |
| A10 Manifest entries `{sha256, bytes, mtime}` — cheap pre-pass health check + free size for catalog cards | critic#19 |
| A11 Per-artifact frame/orientation declared in-band: which artifacts `rgb_rotation_deg` applies to, which frame `box` lives in (raw), mask polarity | critic#1, comp#5 |
| A12 Layout docstring lists ALL contract files | comp cosmetic |

## Bucket B — design decisions (Anton)

**B1 Failed/rejected steps.** Today unrepresentable (pose fields required) — D5
impossible; archive shows the symptom twice. *Rec:* `StepRecord.outcome:
captured|rejected|failed` with capture fields optional off the captured path —
keeps the id space dense.

**B2 Step write protocol.** Step facts arrive in 3 phases across 2 threads;
one atomic write loses pose on crash, two writes recreate the ambiguous-null
bug. *Rec:* declare `phase: captured|fused` discriminator in step.json — null
geometry becomes unambiguous.

**B3 Segmentation trim revisit.** You trimmed to {score,box,px}; critic's
evidence: mask-path vs depth-growth clouds differ materially (89 vs 176 mm
extents on the same view) and the benchmark can't filter without provenance.
*Rec:* add `GeometryStats.source: mask|depth` + `fallback_reason` — it's
geometry provenance, not segmentation detail, so your trim stands.

**B4 Mock/debug identity.** `source`+`status` don't say whether depth was real;
"all cup runs" admits synthetic runs. *Rec:* `RunRecord.rig: real|fake` +
free `tags: list[str]`.

**B5 Object identity.** Free string means `LIKE '%cup%'` grouping, no
instance identity ("same physical cup across 5 runs"). Your call was "just the
name, AI queries it". *Rec:* keep free `object`, add optional
`object_instance` — cheap, preserves your intent.

**B6 Versioning mechanics.** Global SCHEMA_VERSION + extra="forbid" makes ANY
additive change breaking for all file kinds at once; per-file versioning today
is decorative. *Rec:* per-model version constants + read-tolerant
(`extra="ignore"`)/write-strict (`forbid`) split — forward tolerance is what
"additive" means.

**B7 Manifest vs migration gate contradiction.** A migration that adds a field
to step.json changes its bytes → recorded sha permanently wrong. *Rec:* content
manifest hashes immutable binaries only (images, npy, ply); schema-managed
JSON validated by re-validation, not byte hash.

**B8 Legacy adapter honesty.** 27 old runs lack t_arrived, 45 captures lack
rgb_rotation_deg; required fields force silent fabrication. *Rec:*
`provenance: native|legacy` + `synthesized: list[str]` on adapted records.

**B9 ViewState scope (biggest).** Critic verified the shipped menu also
consumes AI-tier state (agent-seen cells, findings, recommendations) — so menu
substitution/determinism can't run off ViewState alone. Also: no `unreachable`
status (hard vs soft constraint collapsed), roll discarded, no scores/chosen
for NBV research, temporal position undeclared, refused decisions leave no
record. *Rec:* ViewState stays PHYSICAL truth + add `unreachable` status,
`roll`, optional `scores/chosen/decider`, declared timing ("state in effect
when the next action was chosen"); AI-tier context becomes a per-turn
`menu_input` snapshot under the AIRun — menus are pure functions of
(ViewState, menu_input).

**B10 Operator identity.** Never decided in writing. *Rec:* skip (single
operator), record the decision.

## Bucket C — build-phase items (not schema; land in management system)

- `validate_run(dir)` cross-file pass: id references resolve, method ids exist,
  manifest coverage, `completed ⇒ answer` (critic#9, comp#8) — this is user
  story C2/F1's engine.
- Address normalization at write (`[3,0]` vs `[3.0,0.0]`) + `_KNOWN_ADDRESS`
  registry mirroring `_KNOWN_PARAMS` (critic#8).
- Atomic temp→rename writers, close ritual, migration runner.

## Invariants to protect (advocate, condensed)

1. `step_id` is the only join key; sparse; roster declared; no consumer derives
   step numbers from directory order.
2. Physical truth is method-invariant; no geometry path branches on
   `view.method`.
3. Nothing that is a function of run-time state is left to post-hoc
   recomputation (ViewState, plane, config) — moving a field to "recomputable"
   requires proving reconstruction exact.
4. The migration gate is never relaxed locally; untyped fields are the declared
   doors.
5. One canonical copy per fact, referenced by id.

## Acceptable trade-offs (critic, agreed — do not "fix")

ViewState size (~0.5 MB/run), inline transcript turns, manifest only at close,
no motion record (decision), segmentation trim itself, run-level q_survey,
version-less jsonl lines (run version covers the file), extra="forbid" on the
write path.
