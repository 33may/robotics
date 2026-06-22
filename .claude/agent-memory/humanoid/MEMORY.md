# humanoid agent memory index

## Project direction
- [Humanoid summer 2026 plan](humanoid-summer-2026-plan.md) — End-of-summer reasoning demo target; acting research runs in parallel.
- [Isaac Sim over IsaacLab](isaac-sim-over-lab.md) — Sim is the primary 3D environment spine; Lab is the RL sub-flow.
- [Humanoid Oli docs before sim](humanoid-oli-docs-before-sim.md) — Build AI-ready docs, answer SDK map, then adapt simulation.

## Workflow / task management
- [Humanoid session capture repo](humanoid-session-capture-repo.md) — Capture logic lives in its own repo; workflow commands should integrate, not duplicate.
- [Humanoid Linear task flow](humanoid-linear-task-flow.md) — Draft and clarify tasks before pushing clean issues to Linear.
- [Humanoid folder organization](humanoid-folder-organization.md) — Keep root clean; save files in semantic folders/assets dirs.
- [Humanoid daily note style](humanoid-daily-note-style.md) — Artifact-first dailies with detailed continuity and clear structure.
- [Humanoid daily note path](humanoid-daily-note-path.md) — Dailies live in `vbti/humanoid/daily/`, not `vbti/sessions/sprintN/`.

## Vendor / external repos
- [humanoid-mujoco-sim](vendor_humanoid_mujoco_sim.md) — LimX MuJoCo harness at `humanoid/vendor/`; canonical Oli assets, SDK wheel, reference simulator.
- [Oli corpus MCP](reference_oli-corpus-mcp.md) — Always-use queryable Oli/LimX docs with `oli-corpus://` citation URIs; FTS quirks.
- [Oli Main Software tarball](oli_main_software_tarball.md) — v2.2.12 EDU colcon ROS2 install (148 pkgs); MROS clone, control headers, URDFs, configs; 0 .msg files.
- [LimX SDK role gating](limx-sdk-role-gating.md) — MROS bus is role-gated bidirectionally; sim vs policy peers see different topics. Probe needs two passes.

## Isaac
- [Isaac Oli smoke loader](isaac_oli_smoke_loader.md) — `humanoid/logic/simulation/isaacsim/load_oli.py` already loads HU_D04_01.usd at /World/Oli, pinned root, prints DOF order. Baseline for MAY-147.

## Feedback
- [Check memory and corpus first](feedback_check_memory_and_corpus_first.md) — Query humanoid memories + oli-corpus + repo state before posing architectural questions.
- [General, not task-scoped docs](feedback_general_not_task_scoped_docs.md) — Document modules/vendors/configs as reusable references; keep task-context in memory.
- [AI-native documentation](feedback_ai_native_documentation.md) — Docs must expose entry points, I/O, side effects, failures, and stable structure for agents.
- [Combine adjacent tasks](feedback_combine-adjacent-tasks.md) — Bundle tasks that share substrate (docs/code/investigation) into one working pass.
- [Scope vs future reuse](feedback_scope_vs_future_reuse.md) — Separate current-task scope from future architectural possibility; avoid absolute reuse claims.
- [MD tables, not ASCII](feedback_md_tables_not_ascii.md) — In `.md` files always use Markdown pipe tables; box-drawing only in terminal/chat output.
