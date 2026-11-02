# reference — GT replay with injectable degradation

**Approach.** Not a localizer — the harness's measuring stick (locbench design.md D13). Replays
the bench GT feed (`Setup.calibration["gt_feed_socket"]`, republished by the evaluator) through
the FULL hosting path: in-brain host → telemetry → evaluator → scorer. Degradations (constant
bias / Gaussian noise / dropout) are injected via `config.yaml` or `--shadow-config` overrides.

**Hypothesis.** Clean → **PASS** trivially (est ≡ GT up to wire latency); every configured
degradation must fail on the gate that names it. If any leg of that triplet disagrees, the
HARNESS is broken — never this module.

**Running it** (like every candidate — D8, one env per run):

```bash
locbench env create reference          # build bench-reference once (~2 min solve)
locbench run reference --smoke 3        # clean → PASS
```

**The acceptance triplet** (tasks 8.2 — the gate for the whole bench). Point `--shadow-config`
at a JSON file carrying each override:

| variant | override | must |
|---|---|---|
| clean | `{}` | PASS |
| bias | `{"inject": {"bias_x_m": 0.2}}` | FAIL on `max pos` (and `mean pos`) |
| dropout | `{"inject": {"dropout": 0.2, "seed": 1}}` | FAIL on `coverage` |

**Bench-only.** Requires the `gt_feed_socket` calibration key — the real robot has no GT, so
this candidate can never run outside the bench by construction. Its `bench-reference` env is a
minimal brain-compatible recipe (python+numpy+pyyaml+pytest, no `build.sh`) — proof that a
conforming candidate needs nothing special (tasks 7.2).

**Status.** Contract-conformant (suite: `tests/oli/reason/localization/test_reference_realization.py`,
incl. `verify_module_contract`). Live triplet: see `JOURNAL.md`.
