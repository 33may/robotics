---
name: humanoid-repo-location
description: Where humanoid project code lives inside the monorepo (separate from vbti/)
metadata:
  type: project
---

Humanoid project code lives at `/home/may33/projects/ml_portfolio/robotics/humanoid/` — a sibling folder to `vbti/` inside the `33may/robotics` monorepo (single git remote `git@github.com:33may/robotics.git`).

- `humanoid/vendor/` holds upstream LimX clones: `humanoid-description`, `humanoid-mujoco-sim`, `humanoid-rl-deploy-cpp`
- Linear task notes for humanoid work live in Obsidian: `/home/may33/Documents/vbti/vbti/humanoid/tasks/may-NNN-*.md`
- OpenSpec for humanoid work will be initialized at `humanoid/openspec/` (not at monorepo root, not under vbti/)

**Why:** Anton wants humanoid and vbti tracked as separate project trees even though they share one git repo. Mixing OpenSpec across both would muddy spec ownership.

**How to apply:** When the user says "the humanoid repo" or "in this repo" during humanoid work, treat `humanoid/` as the project root. Don't write specs/changes/code into `vbti/` or the monorepo root.
