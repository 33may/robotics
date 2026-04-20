# Delta Action Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add step-wise delta action support — offline dataset conversion utility and inference delta mode flag.

**Architecture:** New `to_delta()` function in `convert_utils.py` reads an absolute LeRobot dataset, computes `action[t] - state[t]` for joints 0-4 (gripper stays absolute), writes a standalone copy with correct stats. New `--delta_actions` CLI flag in `run_real_inference.py` reconstructs absolute targets at inference time.

**Tech Stack:** Python, pandas, numpy, LeRobot v3.0 dataset format, fire CLI

**Spec:** `docs/superpowers/specs/2026-04-13-delta-action-prediction-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `vbti/logic/dataset/convert_utils.py` | Modify | Add `to_delta()` function + wire into fire CLI |
| `vbti/logic/inference/run_real_inference.py` | Modify | Add `--delta_actions` flag, conditional action application |

---

### Task 1: Implement `to_delta()` dataset conversion

**Files:**
- Modify: `vbti/logic/dataset/convert_utils.py:296+` (insert before `roundtrip_test`)

- [ ] **Step 1: Write the `to_delta()` function**

This function reads an existing absolute-action LeRobot dataset, transforms actions to step-wise deltas, and writes a full standalone copy (parquet + videos + meta).

Add this function after the `convert()` function (before `roundtrip_test`), at line 296:

```python
def to_delta(
    source: str,
    output: str,
    push_to_hub: bool = False,
):
    """Convert an absolute-action LeRobot dataset to step-wise delta actions.

    Transform: delta[t][0:5] = action[t][0:5] - state[t][0:5]
               delta[t][5]   = action[t][5]  (gripper stays absolute)

    Creates a full standalone copy — videos are copied, not symlinked.

    Args:
        source: path to source LeRobot dataset (absolute actions)
        output: path for the new delta dataset
        push_to_hub: push to HuggingFace Hub after conversion

    Usage:
        python vbti/logic/dataset/convert_utils.py to_delta \
            datasets/so101_v1_mix/sim/duck_cup_130eps \
            datasets/so101_v1_mix/sim/duck_cup_130eps_delta
    """
    import json
    import shutil

    import pandas as pd

    source = Path(source)
    output = Path(output)

    if not (source / "meta" / "info.json").exists():
        print(f"ERROR: {source} is not a valid LeRobot dataset")
        return

    if output.exists():
        print(f"ERROR: {output} already exists. Remove it first.")
        return

    # Copy entire dataset (videos, meta, parquet — everything)
    print(f"Copying {source} → {output} ...")
    shutil.copytree(source, output)

    # Transform parquet files: action = action - state (joints 0-4 only)
    data_dir = output / "data"
    all_actions = []  # collect for stats recomputation

    for pq_path in sorted(data_dir.rglob("*.parquet")):
        df = pd.read_parquet(pq_path)

        actions = np.stack(df["action"].values)
        states = np.stack(df["observation.state"].values)

        # Step-wise delta for body joints (0-4), gripper (5) stays absolute
        delta = actions.copy()
        delta[:, :GRIPPER_IDX] = actions[:, :GRIPPER_IDX] - states[:, :GRIPPER_IDX]

        all_actions.append(delta)

        # Write back
        df["action"] = list(delta.astype(np.float32))
        df.to_parquet(pq_path)

    # Recompute action stats
    all_actions = np.concatenate(all_actions)
    stats_path = output / "meta" / "stats.json"
    with open(stats_path) as f:
        stats = json.load(f)

    stats["action"]["mean"] = all_actions.mean(axis=0).tolist()
    stats["action"]["std"] = all_actions.std(axis=0).tolist()
    stats["action"]["min"] = all_actions.min(axis=0).tolist()
    stats["action"]["max"] = all_actions.max(axis=0).tolist()

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print(f"\nDelta conversion complete: {output}")
    print(f"  Frames: {len(all_actions)}")
    print(f"  Delta action stats (joints 0-4):")
    for i, name in enumerate(JOINT_NAMES[:GRIPPER_IDX]):
        print(f"    {name:20s}  mean={all_actions[:, i].mean():7.3f}  std={all_actions[:, i].std():7.3f}")
    print(f"  Gripper stats (absolute):")
    i = GRIPPER_IDX
    print(f"    {JOINT_NAMES[i]:20s}  mean={all_actions[:, i].mean():7.3f}  std={all_actions[:, i].std():7.3f}")

    if push_to_hub:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset(output)
        ds.push_to_hub()
        print(f"  Pushed to hub.")
```

- [ ] **Step 2: Wire `to_delta` into the fire CLI**

In the same file, find the `fire.Fire({...})` block at the bottom (line ~399) and add the entry:

```python
    fire.Fire({
        "convert":        convert,
        "to_delta":       to_delta,
        "discover":       discover_cameras,
        "verify":         verify,
        "roundtrip_test": roundtrip_test,
        "link":           link,
        "ls":             ls,
    })
```

- [ ] **Step 3: Test the conversion**

Run on the existing duck_cup dataset:

```bash
cd /home/may33/projects/ml_portfolio/robotics
conda activate lerobot
python vbti/logic/dataset/convert_utils.py to_delta \
    datasets/so101_v1_mix/sim/duck_cup_130eps \
    datasets/so101_v1_mix/sim/duck_cup_130eps_delta
```

Expected output:
- "Copying ... → ..." message
- "Delta conversion complete" with stats summary
- Joint 0-4 means should be near zero (small residual from action != next_state)
- Gripper stats should match the original (~31 mean, ~67 std)
- New dataset directory exists with same structure as source

- [ ] **Step 4: Verify the delta dataset with existing `verify` command**

```bash
python vbti/logic/dataset/convert_utils.py verify \
    datasets/so101_v1_mix/sim/duck_cup_130eps_delta
```

Note: The verify command checks for [-100, 100] body joint range and [0, 100] gripper range. Delta actions will likely exceed [-100, 100] in edge cases (large movements), so some joints may show "FAIL" — that's expected and correct for delta values.

- [ ] **Step 5: Commit**

```bash
git add vbti/logic/dataset/convert_utils.py
git commit -m "feat: add to_delta() conversion for step-wise delta action datasets"
```

---

### Task 2: Add `--delta_actions` flag to inference

**Files:**
- Modify: `vbti/logic/inference/run_real_inference.py:172` (function signature) and `~290` (action execution)

- [ ] **Step 1: Add `delta_actions` parameter to `run()` signature**

In `run_real_inference.py`, add `delta_actions: bool = False` to the `run()` function signature (after `print_actions_every`):

```python
def run(
    checkpoint: str,
    port: str = "/dev/ttyACM0",
    cameras: str = "realsense",
    camera_config: dict = None,
    camera_names: list = None,
    task: str = "pick up the object",
    robot_id: str = "frodeo-test",
    max_relative_target: float = 10.0,
    move_to_start: bool = True,
    action_horizon: int = 10,
    fps: int = 30,
    max_steps: int = 500,
    show_cameras: bool = True,
    record: str = "",
    device: str = "auto",
    print_actions_every: int = 0,
    delta_actions: bool = False,
):
```

- [ ] **Step 2: Add delta reconstruction in the action execution loop**

Find the action execution block (~line 288-295). Replace the action application logic with delta-aware version:

Current code (lines 290-295):
```python
                action = actions_deg[i]
                last_action = action

                action_dict = {f"{name}.pos": float(action[j])
                               for j, name in enumerate(JOINT_NAMES)}
                robot.send_action(action_dict)
```

Replace with:
```python
                action = actions_deg[i]

                if delta_actions:
                    # Reconstruct absolute target from delta prediction + current state
                    state_deg = np.array([robot.get_state()[f"{n}.pos"] for n in JOINT_NAMES])
                    target = state_deg + action
                    target[GRIPPER_IDX] = action[GRIPPER_IDX]  # gripper is absolute
                    # Safety clamp to joint limits
                    for j in range(len(REAL_LIMITS_DEG)):
                        lo, hi = REAL_LIMITS_DEG[j]
                        target[j] = np.clip(target[j], lo, hi)
                    action = target

                last_action = action

                action_dict = {f"{name}.pos": float(action[j])
                               for j, name in enumerate(JOINT_NAMES)}
                robot.send_action(action_dict)
```

- [ ] **Step 3: Add necessary imports**

At the top of `run_real_inference.py`, the file already imports `numpy as np`. You need to add the joint limits. Add after the `JOINT_NAMES` list (line ~39):

```python
GRIPPER_IDX = 5

REAL_LIMITS_DEG = [
    (-114.5, 125.5),   # shoulder_pan
    (-109.9, 101.0),   # shoulder_lift
    (-106.0,  89.5),   # elbow_flex
    (-103.7, 103.8),   # wrist_flex
    (-171.3, 165.2),   # wrist_roll
    (  -2.3, 110.4),   # gripper
]
```

Note: These are duplicated from `convert_utils.py`. This is intentional — inference runs standalone without importing the dataset module, and 8 lines of constants don't warrant a shared module.

- [ ] **Step 4: Add a log line when delta mode is active**

Inside `run()`, after the existing setup prints (find where checkpoint/task/fps are printed), add:

```python
    if delta_actions:
        print(f"  Delta actions: ENABLED (step-wise delta, joints reconstructed from state + delta)")
```

- [ ] **Step 5: Commit**

```bash
git add vbti/logic/inference/run_real_inference.py
git commit -m "feat: add --delta_actions flag to real inference for delta action mode"
```

---

### Task 3: End-to-end validation

- [ ] **Step 1: Verify delta dataset loads in LeRobot**

```bash
conda activate lerobot
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('datasets/so101_v1_mix/sim/duck_cup_130eps_delta')
print(f'Episodes: {ds.meta.total_episodes}')
print(f'Frames: {ds.meta.total_frames}')
sample = ds[0]
print(f'Action shape: {sample[\"action\"].shape}')
print(f'Action[0]: {sample[\"action\"]}')
print(f'State[0]: {sample[\"observation.state\"]}')
"
```

Expected: dataset loads without errors, action values are small (near-zero deltas), state values are normal absolute positions.

- [ ] **Step 2: Link delta dataset for training**

```bash
python vbti/logic/dataset/convert_utils.py link \
    datasets/so101_v1_mix/sim/duck_cup_130eps_delta \
    eternalmay33/so101_sim_pick_place_130eps_delta
```

- [ ] **Step 3: Commit all changes**

```bash
git add -A
git commit -m "feat: delta action prediction — conversion utility and inference flag"
```
