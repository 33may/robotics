---
name: duck_rotation_matters
description: Duck facing/rotation matters for grasping — fixed grip positions require specific orientation
type: feedback
---

Don't assume the gripper can grasp the duck from any angle. The robot has fixed grasp positions, so duck orientation matters.

**Why:** User corrected the assumption that grip angle doesn't matter. Future augmentation should include duck rotation prediction (via keypoints or custom head on detection model), trained from gripper camera.

**How to apply:** When planning grasp-related augmentations, duck rotation is a valid and motivated feature. Defer implementation until detection coords alone are validated, but keep it in the pipeline roadmap.
