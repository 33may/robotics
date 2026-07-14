# cuvslam — iteration journal

Append-only; one entry per iteration; ONE change per iteration. Schema (playbook §JOURNAL.md):

```markdown
## it-<N> — <YYYY-MM-DD> — run <run-id|none>
hypothesis: <what we believed going in>
change:     <the ONE thing changed>
result:     <verdict + key numbers, vs previous iteration>
decision:   <improve|pivot|abandon|promote> (anton | agent:rule-<id>)
reasoning:  <why — the payload the promotion protocol mines>
next:       <the next hypothesis>
```

---

## it-0 — 2026-07-14 — run none
hypothesis: README's — bring-up PASS, gate FAIL from unbounded VO drift; deliverable = drift-vs-distance curve.
change:     scaffolded from _template @ 6e68866; vendored NVlabs/PyCuVSLAM @ 0558842 (full C++ source — py3.11 needs from-source build, wheels are 3.10/3.12 only).
result:     scaffold only, stub module conforming-green by construction.
decision:   n/a (pair mode, #locdev session with anton)
reasoning:  cuVSLAM is the L1 candidate from the 2026-07-10 research; measuring its raw drift sizes L3/L5 for the future RTAB-Map combo.
next:       step-0 dependency de-risk — `locbench env create cuvslam` (compiles vendor from source via build.sh), then import cuvslam + one 2-frame odometry step on GPU inside bench-cuvslam.
