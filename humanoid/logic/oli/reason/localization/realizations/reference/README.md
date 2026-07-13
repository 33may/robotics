# reference — GT replay with injectable degradation

**Approach.** Not a localizer — the harness's measuring stick (locbench design.md D13). Replays
the bench GT feed (`Setup.calibration["gt_feed_socket"]`, republished by the evaluator) through
the FULL hosting path: in-brain host → telemetry → evaluator → scorer. Degradations (constant
bias / Gaussian noise / dropout) are injected via `config.yaml` or `--shadow-config` overrides.

**Hypothesis.** Clean → **PASS** trivially (est ≡ GT up to wire latency); every configured
degradation must fail on the gate that names it. If any leg of that triplet disagrees, the
HARNESS is broken — never this module.

**The acceptance triplet** (tasks 8.2 — the gate for the whole bench):

| variant | override | must |
|---|---|---|
| clean | `{}` | PASS |
| bias | `{"inject": {"bias_x_m": 0.2}}` | FAIL on `max pos` (and `mean pos`) |
| dropout | `{"inject": {"dropout": 0.2, "seed": 1}}` | FAIL on `coverage` |

**Bench-only.** Requires the `gt_feed_socket` calibration key — the real robot has no GT, so
this candidate can never run outside the bench by construction. Pure stdlib: runs in the plain
`brain` env (proves the module contract needs nothing special — tasks 7.2).

**Status.** Contract-conformant (suite: `tests/oli/reason/localization/test_reference_realization.py`,
incl. `verify_module_contract`). Live triplet: see `JOURNAL.md`.
