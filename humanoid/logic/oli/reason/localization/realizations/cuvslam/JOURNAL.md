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

## it-6 — 2026-07-14 — runs 20260714-141927 (walk) + 20260714-144518 (teleport)
hypothesis: with the host starvation fixed the est tracks and the REAL drift curve appears.
change:     harness pair (Anton-commissioned, not candidate code): locbench teleport transit (World --teleport wire, evaluator snap+confirm+fallback, ~2x faster: 466->244 s wall) + nav.py localize-every-tick (last_pose was goal-gated -> teleport confirm starved on GT=None, run 143731 hung).
result:     drift MEASURED, and it is not a curve — it is a step function. In the racks VO is near-perfect (~0 err for 7–9 s of aisle driving; ep2 walk-run: 0.44 m mean over ~14 m ≈ 3%). Facing the blank warehouse walls / open floor the tracker LOSES itself and silently re-anchors 15–30 m off (discrete jumps in error_timeline; ep1 teleport-run jumped at t=0.2 s because the face-the-goal spawn heading stares at open space). Coverage stays 1.00 — cuVSLAM returns confident garbage after a loss, no LOST signal.
decision:   pivot (anton — moving to research the map-anchored candidate, RTAB-Map L2–L5)
reasoning:  the 0.15 m max-pos gate is architecturally out of reach for ANY unanchored L1: wall-loss is total, not gradual, so multicam/IMU polish raises the ceiling without touching the gate. cuvslam's job is done: harness proven with a real tracker, L1 envelope sized (map fixes needed every few meters; recovery must handle TOTAL loss), scene hazard named (feature-poor walls — every future candidate must survive them). Binding capabilities verified for a future combo frontend: Multicamera mode, Inertial mode + register_imu_measurement, Slam module (slam_pose).
next:       Anton researches the next move with the new information (RTAB-Map de-risks: ROS-free python path; prior-map building needs a bench build_map surface). cuvslam freezes as the measured L1 baseline on the board.

## it-7 — 2026-07-17 — map_relative mode (demo day, pair mode)
hypothesis: unfrozen from the it-6 L1 verdict — map-relative localize_in_map against a PyCuVSLAM-rebuilt map clears the wall-loss ceiling (the map re-anchors what VO alone cannot).
change:     mode switch in module.py (config `mode: map_relative`; rgbd_vo path preserved verbatim): stereo vehicle-frame rig (mount table = the map builder's rig), hint W→M via the map's registration_gt.json, localize_in_map on first pair, map-anchored track, poses emitted in W (prebuilt occupancy grid frame, Anton 17-07). Plus build_map.py (D14 bench build surface: edex → save_map + slam_poses.tum + registration_gt.json + audit).
result:     three findings en route: (1) container cuvslam_map LMDB does NOT load in vendored v16 — rebuilt from the same keyframes instead, audit 7/8 at mm (memory: cuvslam-container-map-not-pycuvslam-loadable); (2) edex/occupancy world is MIRRORED vs physical (R=diag(-1,1)) — dodged entirely by the prebuilt-grid decision, GT↔M registration is rigid, residual 0.037 m mean; (3) localize search step 0.25 REFUSES a 0.15 m-off hint (grid too coarse for PnP convergence) — step 0.0625 passes in 0.1 s (probe sweep, /tmp/probe_settings.py pattern). Also max_map_size MUST be 0 to match the builder (default 300 caps the loaded map). Replay smoke (60 fr): ATE 0.023 m mean, 100% TRACKING, 88 steps/s. Full-dump replay pending this entry.
decision:   improve (anton — pair mode, live throughout)
reasoning:  rig identity + config identity with the map builder is part of the map contract — every mismatch (rig frame, max_map_size, search grid) fails as the same opaque "Can't localize" refusal; bisect with a known-good probe harness instead of tweaking. Self-test caveat: replay = the mapping drive (fair-ish: dump jpgs ≠ edex jpgs, maxdiff 7); the live LOC MODE walk is the real exam.
next:       full replay verdict → wire into the brain (LocalizationHost + HostLocalizer, no sidecar — Anton 17-07) → live LOC MODE → nav on est pose.

## it-8 — 2026-07-17 — run none (replay_proof ×2, full dump)
hypothesis: periodic mid-tracking localize_in_map with the module's OWN slam_pose as prior (the corpus-designed pattern, nvidia-corpus://cuvslam-api/cpp/a00138#section-2) corrects the ~50 cm long-run drift the 188 s live trace showed (lc_events stayed 0 — LC never fires).
change:     config `localization.relocalize_period_s: 15.0` (0 = off) + module `_localize(stamp, imgs, guess_m)` refactor: start path keeps hint→m_from_w exactly; when localized and period elapsed (SIM time via stamp_ns), re-run localize with guess = own slam_pose (no registration roundtrip, no GT); success emits the anchored pose + `reloc_ok`, refusal swallows + `reloc_fail` (attempt-paced retry). Diagnostics carry reloc_ok/fail/last_reloc_stamp → drift-trace JSONL + LocPanel events ("reloc · fix"/"reloc · refused").
result:     machinery WORKS, pattern FAILS the gate in vendored v16. Full-dump replay (2268 fr, 75.6 s): period 0 → ATE mean 0.055 m, 0 jumps, PASS (first full-dump baseline, closes it-7's pending). period 15 → ATE mean 0.967 m, 2 jumps, FAIL; reloc_ok 2, reloc_fail 3. Instrumented trace: the reloc FIX itself is mm-accurate (err 0.041/0.028 at both OKs) and refusal is confirmed non-destructive (err continuous across all 3 FAILs) — but AFTER each successful reloc the swapped-in SLAM state drifts ~0.04–0.3 m/s (0.04→1.9 m in 30 s), which the start-localized state never does (0.055 m held 75 s). Root cause read from vendor source (async_slam_localize.cpp): success `std::swap`s the ENTIRE slam graph for a freshly-loaded map + 2 fake keyframes (one empty-images at the tail tip); mid-run — with a mature odometry/tail — the post-swap state tracks far worse; at start the identical path is fine. Also: every localize call re-opens the map LMDB from disk (upstream TODO), 0.8–12.6 s SYNC stall per attempt — a live 30 Hz demo would freeze visibly every period even if accuracy were fixed. The 2 refusals were the drifted prior exceeding horizontal_search_radius 1.0 m.
decision:   abandon periodic reloc for the demo — ship relocalize_period_s: 0 (anton, after report-back)
reasoning:  the ONE change is in and instrumented; the failure is vendor-internal (state-swap corruption), not adapter wiring — tuning period/radius would mask it. The period-0 baseline (0.055 m over the full 75.6 s dump) is demo-sufficient, so the feature ships OFF; machinery + counters + panel events stay behind the config switch for a future rebuild-Tracker variant.
next:       demo line continues on the it-7 baseline (period 0, gate PASS). If long-run drift bites live, the known-good reloc shape is destroy+rebuild Tracker→localize (start-path semantics, ~1-3 s stall budget) — a new iteration, not a tune of this one.
