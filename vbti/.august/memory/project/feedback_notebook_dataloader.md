---
name: feedback_notebook_dataloader
description: Jupyter notebooks cannot use DataLoader with num_workers>0 due to torchcodec fork issues
type: feedback
---

DataLoader with `num_workers>0` crashes in Jupyter notebooks (isaac env) because torchcodec's video decoder doesn't survive process forking. Works fine in standalone Python scripts.

**Why:** torchcodec holds decoder state that can't be pickled/forked into worker processes from within the notebook kernel context.

**How to apply:** For notebook workflows, extract frames via a standalone script (`extract_frames.py`) and cache to `.pt` file, then load cached tensors in notebook. Never use `DataLoader(num_workers>0)` in notebooks with video datasets.
