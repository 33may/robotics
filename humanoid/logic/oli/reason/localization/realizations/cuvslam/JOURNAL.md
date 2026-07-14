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

## it-1 — 2026-07-14 — run none
hypothesis: PyCuVSLAM is buildable from source into a py3.11 brain-compatible env on Fedora/SM89/CUDA 12.9 (the blocking step-0 risk).
change:     env bring-up — environment.yml + build.sh (system gcc-14 + patched CUDA 12.9, SM 89 only) + 2 vendor patches: libjpeg.cmake CMAKE_INSTALL_LIBDIR=lib (Fedora lib64 vs hardcoded lib/), cuvslam2.cpp missing <algorithm> (GCC 14 copy_n).
result:     PASS — bench-cuvslam created, lock.yml frozen, import OK, GPU tracker init OK, 3 odometry steps on synthetic 1280x720 RGBD return ~1e-7 m motion (sane identity). Discovered: track() REQUIRES uint16 depth (float32 meters rejected) → adapter converts m→mm with depth_scale_factor=1000.
decision:   improve (anton — pair mode; de-risk cleared, proceed to the adapter)
reasoning:  both failure modes were Ubuntu→Fedora portability (lib64, stricter libstdc++ includes), not cuVSLAM-vs-py3.11 — the fallback OOP node is NOT needed.
next:       phase-3 adapter — CuvslamModule around Tracker (RGBD mode, head cam, known-start map-frame transform), every knob in config.yaml.

## it-2 — 2026-07-14 — run none
hypothesis: one RGBD camera (head) + known-start anchor is enough to emit map-frame SE(2) through the contract.
change:     real adapter in module.py — lazy rig/tracker from first frame's intrinsics (Setup has none), depth float32-m -> uint16 via odometry.rgbd_settings.depth_scale_factor, mounts from oli.camera_mounts (D10), T_map_vo anchored at first tracked frame, healthy VO -> DRIFTING (scorer counts any non-None pose; honest for pure VO), tracker exception -> dead -> LOST; config.yaml carries cuVSLAM's own odometry schema (vendored _apply_odometry_section).
result:     contract test green in bench-cuvslam (3 passed); end-to-end synthetic check: static 720p frames + warm start (2.0, -1.0, 0.7) reproduce exactly (DRIFTING, fix=anchor stamp) — transform chain verified.
decision:   improve (anton — pair mode)
reasoning:  adapter conforms and the math round-trips; remaining risk is live bench integration (frame cadence, sim-time stamps, Isaac boot), which only `locbench run` exercises.
next:       STEP 3 — `locbench run cuvslam --smoke 3`; expect bring-up PASS (coverage > 0, no crash) and drift visible in overlay/timeline.

## it-3 — 2026-07-14 — run 20260714-125728 (crashed)
hypothesis: the stack runs end-to-end with the candidate shadowed.
change:     add scipy to environment.yml (ONE change).
result:     run 20260714-125728 crashed BEFORE any localization step: brain died at first nav plan — costmap.clearance_cost lazily imports scipy.ndimage, absent from the "numpy+stdlib" bench env recipe. Candidate never stepped; not a cuVSLAM failure.
decision:   improve (anton — pair mode)
reasoning:  systemic finding to REPORT (not fix): the reference recipe's "glide needs numpy+stdlib" claim misses service-mode nav planning (scipy) — bench-reference would crash identically (consistent with no reference run ever finishing). ALSO for Anton: when the brain died mid-run, the evaluator itself crashed on episode 1 (FileNotFoundError on the goal socket) instead of marking remaining episodes crashed — locbench robustness gap, oracle-integrity says hands off.
next:       rerun smoke with scipy in the env; expect the scored leg to actually step the tracker.

## it-4 — 2026-07-14 — run 20260714-130101
hypothesis: with scipy in the env the run completes end-to-end and the drift curve becomes readable.
change:     none in the candidate (same code as it-3) — this is the rerun.
result:     BRING-UP PASS: all 3 episodes arrived, coverage 1.00, report + plots written, tier FAIL as hypothesized. BUT the failure mode is NOT drift: the estimate is FROZEN at the warm-start pose (byte-identical x/y/yaw across every tick, stamps advancing) — pos_mean 5.3–8.1 m is just the robot's distance from spawn. Drift-vs-distance NOT yet measured.
decision:   improve (anton — pair mode)
reasoning:  frozen-est signature = the module steps (stamps advance) but takes the carry branch every tick ⇒ the head frame never re-enters the bundle. Suspected HARNESS bug in reason/localization/host.py::_maybe_step: ONE `_last_frame_stamp` across ALL streams — once a poll catches chest@t before head@t lands, head's equal stamp is ≤ the processed max forever (world publishes chest first): permanent starvation ratchet. Fix would be per-stream last-stamps — host.py is brain-side harness infra, NOT the candidate; per the playbook this is journaled + PAUSED for Anton, not self-fixed.
next:       Anton's call on the host fix; then rerun — expect the red track to actually move and the REAL drift curve to appear. `--live-view` (built 2026-07-14 on Anton's ask) shows GT vs est in real time to watch it.

## it-5 — 2026-07-14 — run none (fix only; rerun follows)
hypothesis: it-4's frozen est is host frame starvation — ONE shared `_last_frame_stamp` across streams drops head@t forever once chest@t is consumed first.
change:     host.py::_maybe_step → PER-STREAM watermarks (`_last_stamps: Dict[str, int]`); Anton approved editing the brain-side harness this session (pair mode).
result:     hypothesis CONFIRMED deterministically — new test test_equal_stamp_multi_stream_frames_all_reach_the_module reproduces the starvation (red) and the per-stream fix turns it green; full brain suite green.
decision:   improve (anton — approved "fix it end 2 end")
reasoning:  aggregate-watermark vs per-stream-watermark: a shared newest-stamp filter asserts "all streams tick atomically", which is false — streams publish back-to-back with equal stamps, so the late-written stream starves permanently.
next:       rerun `locbench run cuvslam --smoke 3 --live-view` — est should now track; read the REAL drift-vs-distance numbers.
