# Joint name audit — Isaac DOF order vs PR-canonical (task 2.3 + 2.4)

Date: 2026-06-22
Source dumps: `_research/isaac_dof_dump.txt` (Isaac side), MAY-145 § 11 of `humanoid/docs/vendor/humanoid-rl-deploy-python.md` (PR canonical).

## Counts

| Source | DOF count | Joint name set |
|---|---|---|
| PR canonical (§11) | 31 | 31 unique |
| Isaac DOF (HU_D04_01.usd loaded at `/World/Oli`) | 31 | 31 unique |

**Set equality**: ✓ — the 31 PR names match the 31 Isaac DOF names exactly. No mimic / fixed / extra joints leak into Isaac's articulation. The 6 `contact_foot_*_joint_{L,R}` and 2 `*_hand_*_joint` prims under `/World/Oli/joints/` are typed `PhysicsFixedJoint` and are correctly excluded from the articulation's DOF list by Isaac.

## Orderings (side by side)

| idx | PR canonical (§11) | Isaac DOF (HU_D04_01.usd) |
|---:|---|---|
|  0 | left_hip_pitch_joint        | left_hip_pitch_joint        |
|  1 | left_hip_roll_joint         | right_hip_pitch_joint       |
|  2 | left_hip_yaw_joint          | waist_yaw_joint             |
|  3 | left_knee_joint             | left_hip_roll_joint         |
|  4 | left_ankle_pitch_joint      | right_hip_roll_joint        |
|  5 | left_ankle_roll_joint       | waist_roll_joint            |
|  6 | right_hip_pitch_joint       | left_hip_yaw_joint          |
|  7 | right_hip_roll_joint        | right_hip_yaw_joint         |
|  8 | right_hip_yaw_joint         | waist_pitch_joint           |
|  9 | right_knee_joint            | left_knee_joint             |
| 10 | right_ankle_pitch_joint     | right_knee_joint            |
| 11 | right_ankle_roll_joint      | head_yaw_joint              |
| 12 | waist_yaw_joint             | left_shoulder_pitch_joint   |
| 13 | waist_roll_joint            | right_shoulder_pitch_joint  |
| 14 | waist_pitch_joint           | left_ankle_pitch_joint      |
| 15 | head_yaw_joint              | right_ankle_pitch_joint     |
| 16 | head_pitch_joint            | head_pitch_joint            |
| 17 | left_shoulder_pitch_joint   | left_shoulder_roll_joint    |
| 18 | left_shoulder_roll_joint    | right_shoulder_roll_joint   |
| 19 | left_shoulder_yaw_joint     | left_ankle_roll_joint       |
| 20 | left_elbow_joint            | right_ankle_roll_joint      |
| 21 | left_wrist_yaw_joint        | left_shoulder_yaw_joint     |
| 22 | left_wrist_pitch_joint      | right_shoulder_yaw_joint    |
| 23 | left_wrist_roll_joint       | left_elbow_joint            |
| 24 | right_shoulder_pitch_joint  | right_elbow_joint           |
| 25 | right_shoulder_roll_joint   | left_wrist_yaw_joint        |
| 26 | right_shoulder_yaw_joint    | right_wrist_yaw_joint       |
| 27 | right_elbow_joint           | left_wrist_pitch_joint      |
| 28 | right_wrist_yaw_joint       | right_wrist_pitch_joint     |
| 29 | right_wrist_pitch_joint     | left_wrist_roll_joint       |
| 30 | right_wrist_roll_joint      | right_wrist_roll_joint      |

Observations:

- **PR groups by limb** (whole left leg → whole right leg → waist → head → left arm → right arm).
- **Isaac groups by depth** in the kinematic chain — paired left/right joints near the root come first, then progress outward. This is the standard "articulation DOF order = tree-traversal order" behavior in Isaac/USD.
- **Two coincidental fixed points**: idx 0 (`left_hip_pitch_joint`) and idx 16 (`head_pitch_joint`) happen to land at the same index in both schemes. No others.

## Permutation tables (computed + verified)

```python
# pr_to_isaac[pr_idx] = isaac_idx of the same joint name
pr_to_isaac = [0, 3, 6, 9, 14, 19, 1, 4, 7, 10,
               15, 20, 2, 5, 8, 11, 16, 12, 17, 21,
               23, 25, 27, 29, 13, 18, 22, 24, 26, 28, 30]

# isaac_to_pr[isaac_idx] = pr_idx of the same joint name (inverse)
isaac_to_pr = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3,
               9, 15, 17, 24, 4, 10, 16, 18, 25, 5,
               11, 19, 26, 20, 27, 21, 28, 22, 29, 23, 30]
```

Round-trip verified: `pr_order[isaac_to_pr[pr_to_isaac[i]]] == pr_order[i]` for every `i ∈ [0, 30]`.

These exact arrays go into `Oli.__init__` as the cached permutation indices. They are NOT hard-coded — the driver computes them at construction time from `oli.dof_names` against the PR canonical list (D7), so any USD re-import that reshuffles Isaac's DOF order is handled automatically. The arrays here are documented for sanity-checking the first run.

## Head joint order (resolves OQ8 deferral)

PR canonical (§11): `15 = head_yaw_joint, 16 = head_pitch_joint`.

Isaac dump: `11 = head_yaw_joint, 16 = head_pitch_joint`.

Both stacks agree: **yaw first, pitch second**. The on-robot `head_config.yaml` and the live wire probe both also agree. The corpus `sdk_joint_order` MCP tool is the only outlier (it reports pitch first) — confirmed as a corpus extractor bug per OQ8 of design.md.
