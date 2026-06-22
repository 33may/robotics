---
name: isaac-sim-over-lab
description: Anton prefers Isaac Sim as the primary 3D environment; IsaacLab is treated as the RL sub-flow. Goal is to eventually merge the two flows.
metadata:
  type: project
---

Isaac Sim is the primary simulation environment for the humanoid project, not IsaacLab.

**Why:** Isaac Sim is a general 3D environment — fits the project's heavy use of 3D reconstruction and scene work. IsaacLab is more narrowly framed for RL. Anton wants the two flows merged over time, but Sim is the spine.

**How to apply:**
- Default new sim work to Isaac Sim standalone (USD/PhysX directly) unless the task is explicitly RL.
- When proposing imports, articulation setup, or scene scripting, write against the Isaac Sim API, not `omni.isaac.lab.assets.Articulation`.
- When existing IsaacLab work is touched, prefer porting toward Sim rather than deepening Lab.
- Related: [[humanoid-summer-2026-plan]] reasoning demo will run in Sim-shaped envs.
