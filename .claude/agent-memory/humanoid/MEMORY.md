# humanoid agent memory index

## Project direction
- [Humanoid summer 2026 plan](humanoid-summer-2026-plan.md) — End-of-summer reasoning demo target; acting research runs in parallel.
- [Isaac Sim over IsaacLab](isaac-sim-over-lab.md) — Sim is the primary 3D environment spine; Lab is the RL sub-flow.
- [Humanoid Oli docs before sim](humanoid-oli-docs-before-sim.md) — Build AI-ready docs, answer SDK map, then adapt simulation.
- [Invariant Oli interface](project_invariant_oli_interface.md) — Deployment-invariant Oli brain (Reason+Action) in a dedicated py3.11 env (NOT py3.8); World independent process; 3 canonical-PR contracts; brain=client/World=server (MAY-147).
- [MuJoCo World via LimX Comm edge](mujoco_world_via_limx_comm_edge.md) — MuJoCo sim runs unchanged; a py3.8 limxsdk Comm edge (LimxBody behind WorldComm) bridges bus↔our PR contracts; this edge IS the deferred RealComm (2026-06-25).
- [Dev app stack](project_dev_app_stack.md) — Oli "robot OS" = standalone imgui_bundle app booted in the brain process; camera/3D/state/reasoning are in-window plugin panels; sim+real (MAY-150).
- [Dev app build + validation](devapp_build_and_validation.md) — Built at logic/oli/devapp/ (Panel/registry/CameraSource/ProcessLauncher/BrainLink); one-command boot via launcher; xvfb screenshot loop; immvision gotchas (MAY-150).
- [Launcher single entrypoint](launcher_single_entrypoint.md) — ONE command `p logic/oli/launcher.py --sim {isaac,mujoco,real} --mode {stand,walk,forward,glide}`; generic Supervisor + per-world backend plugins; run_oli_sim/mujoco now shims (MAY-150, 2026-07-02).

## Architecture & agentic flow (2026-07-02 pivot)
- [Dataflow bus + PolicyRunner pivot](arch_dataflow_bus_and_policyrunner.md) — Path 2: Robot container = Comm+Reason+PolicyRunner+dev_app as nodes on a brokerless inner bus; PolicyRunner own process w/ buffer+2 clocks; supersedes June monolith.
- [Inner-comm middleware research](middleware_inner_comm_research.md) — MROS unmatchable, full ROS2 unjustified, observability orthogonal (MCAP/rerun); bus pick pending — dora-rs front-runner vs Zenoh+MCAP vs custom.
- [Agentic dev foundation initiative](agentic_dev_foundation_initiative.md) — build AGENTS.md + architecture.md + ADRs + path-scoped rules + CI invariance gate + OpenSpec archive→specs sync; Anton drafts, agent assembles → Linear task.

## User
- [Controls/robotics-theory level](user_controls_theory_level.md) — Anton is the architect + strong ML/systems eng, but not deep in controls theory; explain PD/AB-PR/drive-modes plainly and short.

## Workflow / task management
- [Humanoid session capture repo](humanoid-session-capture-repo.md) — Capture logic lives in its own repo; workflow commands should integrate, not duplicate.
- [Humanoid Linear task flow](humanoid-linear-task-flow.md) — Draft and clarify tasks before pushing clean issues to Linear.
- [Humanoid folder organization](humanoid-folder-organization.md) — Keep root clean; save files in semantic folders/assets dirs.
- [Humanoid daily note style](humanoid-daily-note-style.md) — Artifact-first dailies with detailed continuity and clear structure.
- [Humanoid daily note path](humanoid-daily-note-path.md) — Dailies live in `vbti/humanoid/daily/`, not `vbti/sessions/sprintN/`.

## Vendor / external repos
- [humanoid-mujoco-sim](vendor_humanoid_mujoco_sim.md) — LimX MuJoCo harness at `humanoid/vendor/`; canonical Oli assets, SDK wheel, reference simulator.
- [assets/oli symlinks into vendor](asset_usd_symlink_vendor.md) — `assets/oli`→vendor submodule; USDs are untracked regenerable build artifacts; treat vendor as our own repo (Anton 2026-07-01); build scripts are source of truth.
- [Oli corpus MCP](reference_oli-corpus-mcp.md) — Always-use queryable Oli/LimX docs with `oli-corpus://` citation URIs; FTS quirks.
- [Oli corpus structured tools](reference_oli-corpus-structured-tools.md) — 9 new tools (robots/joints/links/sdk_joint_order/pkg_info/nodes/topics/find_symbol/raw_file) added 2026-06-22; decision tree + compound patterns.
- [Oli Main Software tarball](oli_main_software_tarball.md) — v2.2.12 EDU colcon ROS2 install (148 pkgs); MROS clone, control headers, URDFs, configs; 0 .msg files.
- [Vendor submodule conversion](vendor-submodule-conversion.md) — Patch-extract→re-clone procedure; current submodule pin/ignore state; patches live in working trees only (MAY-146).
- [LimX SDK role gating](limx-sdk-role-gating.md) — MROS bus is role-gated bidirectionally; sim vs policy peers see different topics. Probe needs two passes.
- [kinematic_projection bus relay](limx_kinematic_projection_bus_relay.md) — kinematic_projection is a REQUIRED MROS relay for RobotState+RobotCmd (IMU goes direct); without it, state/cmd silently deliver zero between sim and policy peers.
- [LimX robot-joystick app](vendor_limx_robot_joystick.md) — Shipped joystick is a keyboard-driven pygame+limxsdk PyInstaller app publishing SensorJoy; axis/button/mode mapping.

## Reason layer (teleop)
- [Joystick teleop architecture](project_joystick_teleop_architecture.md) — Ported pygame app emits our own UDP JoyPacket into a pure brain; `reason/teleoperation/joystick/`; layout, port 9001, open button→mode (MAY-147).

## Isaac
- [Isaac Oli smoke loader](isaac_oli_smoke_loader.md) — `humanoid/logic/simulation/isaacsim/load_oli.py` already loads HU_D04_01.usd at /World/Oli, pinned root, prints DOF order. Baseline for MAY-147.
- [Isaac PD via implicit drive](isaac_pd_implicit_drive.md) — Realize deploy PD via PhysX drive (set_gains + targets), NOT explicit set_joint_efforts; the latter rings unstably for MuJoCo-tuned Kd.
- [Walk policy obs builder fidelity](walk_policy_obs_builder_fidelity.md) — Exact 102→510 obs layout/scales, projected-gravity identity, last_actions aliasing quirk, head-order swap; what the walk ONNX expects (MAY-147).
- [Isaac Oli stand spawn height](isaac_oli_stand_spawn_height.md) — `--spawn-height 1.1` settles the crouch stably; but walk policy still topples (see physics-fidelity memory).
- [Isaac walk physics fidelity](isaac_walk_physics_fidelity.md) — Two root causes: dead IMUSensor (standing, fixed) + serial ankle back-drives (forward walk); ankle-pitch kp×3 holds 2.0s but gain-tuning a serial ankle has a ceiling — parallel achilles is the faithful fix (MAY-147).
- [Walkmatch actuator-ID harness](walkmatch_actuator_id_harness.md) — Sim-to-sim system-ID toolkit proving Isaac legs match MuJoCo + isolating the ankle; reuse for any Isaac↔MuJoCo fidelity question.
- [LimX email awaiting reply](project_limx_email_awaiting_reply.md) — 2026-07-01 emailed LimX for HU-D04 Isaac asset + training actuator config; dynamic Isaac walk blocked on their reply; check before reopening.
- [Oli perception camera design](oli_perception_camera_design.md) — MAY-149 sim cameras: D435i chest/head baked into USD sensor layer (no chest link → waist_pitch_link), CameraFrame contract, dedicated SOCK_STREAM frame channel; BUILT + Isaac-verified 2026-07-02 (CameraStreamReader + CameraPublisher, per-name demux).
- [Isaac camera first-render not ready](isaac_camera_first_render_not_ready.md) — Isaac camera get_rgba/depth is empty on the first render tick; reading it in the World loop crashes the sim with no traceback. Fix = 8-tick warmup + a publisher guard.

## Locomotion / glide (MAY-172)
- [MAY-172 glide scope: defer the fit](may172_glide_scope_defer_fit.md) — Kinematic glide reliable-first; MuJoCo velocity fit deferred (SLAM gets exact GT pose free); camera bob/sway is the later realism lever.
- [MAY-172 glide wiring](may172_glide_wiring.md) — Glide = additive GLIDE_CMD msg + GlideAction/GlideModel in same Orchestrator loop; PolicyOut untouched; brain side green; World driver is block 4.

## Feedback
- [Single entrypoint, never multi-terminal](feedback_single_entrypoint_no_multiterminal.md) — Anton wants ONE command to boot the whole Oli stack; the launcher selects backend (mujoco/isaac/real) + mode. Never hand him a 2-3 terminal dance.
- [Branch naming: 33may not antonnedf](feedback-branch-naming-33may.md) — Use `33may/` prefix for all branches; ignore Linear's `gitBranchName` field default.
- [Check memory and corpus first](feedback_check_memory_and_corpus_first.md) — Query humanoid memories + oli-corpus + repo state before posing architectural questions.
- [Use oli-corpus aggressively](feedback_use_oli_corpus_aggressively.md) — Lean on the corpus as a routine second opinion via structured tools; cite with `oli-corpus://` URIs; the docs aren't infallible but the cross-check is nearly free.
- [General, not task-scoped docs](feedback_general_not_task_scoped_docs.md) — Document modules/vendors/configs as reusable references; keep task-context in memory.
- [AI-native documentation](feedback_ai_native_documentation.md) — Docs must expose entry points, I/O, side effects, failures, and stable structure for agents.
- [Combine adjacent tasks](feedback_combine-adjacent-tasks.md) — Bundle tasks that share substrate (docs/code/investigation) into one working pass.
- [Scope vs future reuse](feedback_scope_vs_future_reuse.md) — Separate current-task scope from future architectural possibility; avoid absolute reuse claims.
- [MD tables, not ASCII](feedback_md_tables_not_ascii.md) — In `.md` files always use Markdown pipe tables; box-drawing only in terminal/chat output.
- [Tests in repo, real TDD](feedback_tests_in_repo_tdd.md) — Committed TDD suite (red→green→refactor), never throwaway `/tmp` smoke scripts.
- [Comm is world edge, not hub](feedback_comm_is_world_edge_not_hub.md) — A Communication/IO layer sits at the robot↔world boundary only; internal modules talk directly, not through it.
- [Shell: `p` + humanoid cwd](feedback_shell_p_and_humanoid_cwd.md) — Use `p` (alias for python) and humanoid-relative paths in commands; Anton runs from `humanoid/`.
- [Reliable walking, not fall-time](feedback_reliable_walking_not_falltime.md) — Don't read fall-time noise (held 1s vs 2.5s) as progress; bar is a sustained gait; diff vs the reference that walks.
- [Project plan is source of truth](feedback_project_plan_source_of_truth.md) — Plan doc is strategic truth; Linear is just the working plan — reconcile the board against the plan when planning.
- [Autonomy: ask terminal, write prompt to file](feedback_autonomous_notebook_verdict.md) — On "do it autonomously", ask this-terminal-vs-separate; if separate write the NOTEBOOK-driven stop-with-verdict prompt to a file and point Anton to it.
- [Architecture doc = structure, not status](feedback_architecture_doc_structure_not_status.md) — Keep stable structure in architecture.md; strip transient status/blockers/roadmap (those go to dailies/Linear/memory).
