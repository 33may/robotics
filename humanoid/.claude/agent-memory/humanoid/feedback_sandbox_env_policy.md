---
name: feedback-sandbox-env-policy
description: No Docker for sandboxing experimental stacks — disposable per-candidate conda envs; must be cleanly removable AND immediately work in our env
metadata:
  type: feedback
---

**No Docker for sandboxing candidate/experimental stacks** (SLAM candidates, benchmark tools).
Use disposable per-candidate conda envs (e.g. `bench-<candidate>`) instead — "env copiers".

**Why:** Anton (2026-07-13, locbench design): "docker is too much". His two hard requirements for
any isolation choice: (a) if we DON'T keep the option, it must be removable as if it was never
there — no stale dependencies left on the machine; (b) if we DO pick it, it must immediately work
in our env. Per-candidate conda envs satisfy both (delete env dir = gone; native GPU/filesystem =
no container plumbing); Docker fails (b)'s "immediately" (nvidia-container-toolkit setup on Fedora)
and adds daemon/image residue against (a).

**How to apply:** whenever isolating an experimental toolchain, create a fresh/cloned conda env with
a disposable prefix; never install experiment deps into `brain`/`isaac`/`limx`/`hum`. Removal =
`conda env remove`. Links: [[architecture-locbench-harness]].
