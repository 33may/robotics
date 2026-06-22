---
name: gpu_upgrade
description: Two-machine GPU split — 4070 Ti SUPER local for inference, 5090 32GB remote for training (confirmed 2026-04-30)
type: project
originSessionId: 6b8a2b91-4695-46a5-967e-f7b16a9949f2
---
**Two-machine GPU setup, confirmed 2026-04-30:**
- **Local PC** (where inference + dataset prep runs): RTX 4070 Ti SUPER 16GB.
- **Remote training machine**: RTX 5090 32GB.

**Why:** training is offloaded to the bigger card; the local rig handles real-robot inference, recording, and dataset processing.

**How to apply:**
- For *training* sizing, plan for 5090's 32GB — comfortably fits SmolVLA at batch 64-128, GR00T N1.6, π0/π0.5, and dual-branch v017 architectures. SmolVLA training step ~150-200ms at batch 64.
- For *inference* and dataset processing on local, plan for 16GB — SmolVLA only, no GR00T or π0.
- For *PNG-sidecar / video-decode bottlenecks during training*: with 5090's faster step time, data-load becomes proportionally more visible. Still hidden behind GPU step at batch 64 + 16 workers, but margin tightens; verify on first long run.

**History:** 5090 upgrade was first claimed 2026-04-16, then retracted 2026-04-21 (memory said 4070 Ti SUPER only). On 2026-04-30 user clarified: 5090 is a separate **remote** training rig, 4070 Ti SUPER stays local. Both claims were partially right.
