# LeRobot Dataset Format Reference (v2.1 / v3.0)

**Researched**: 2026-03-09

## Version Summary
- **v2.1**: Current stable (lerobot < 0.4.0). One parquet + one mp4 per episode.
- **v3.0**: Coming in lerobot >= 0.4.0. Many episodes per file. Streaming support.

---

## v2.1 Directory Structure
```
dataset_repo/
├── data/chunk-000/
│   ├── episode_000000.parquet
│   ├── episode_000001.parquet
│   └── ...
├── videos/chunk-000/
│   ├── observation.images.main/
│   │   ├── episode_000000.mp4
│   │   └── ...
│   └── observation.images.secondary_0/
│       └── ...
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── tasks.jsonl
│   └── episodes_stats.jsonl
└── README.md
```

## v3.0 Directory Structure
```
dataset_repo/
├── data/chunk-000/
│   ├── file-000.parquet    # multiple episodes per file
│   └── ...
├── videos/{camera_key}/chunk-000/
│   ├── file-000.mp4        # multiple episodes per file
│   └── ...
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── tasks.jsonl
│   └── episodes/           # chunked Parquet (not JSONL)
└── README.md
```

---

## Parquet Columns (always present)

| Column | dtype | Description |
|--------|-------|-------------|
| `index` | int64 | Global unique frame ID across dataset |
| `episode_index` | int64 | Episode identifier |
| `frame_index` | int64 | Frame position within episode (0-based) |
| `timestamp` | float32 | Seconds from episode start |
| `task_index` | int64 | Links to tasks.jsonl |
| `next.done` | bool | True on last frame of episode |
| `observation.state` | list[float32] | Joint angles / proprioception |
| `action` | list[float32] | Target joint angles / commands |

Camera columns stored as VideoFrame refs: `{'path': 'path/to/video.mp4', 'timestamp': float}`

**Action timing**: action at frame t causes observation at frame t+1.

---

## info.json Schema (v3.0 example from lerobot/pusht)

```json
{
  "codebase_version": "v3.0",
  "robot_type": "unknown",
  "total_episodes": 206,
  "total_frames": 25650,
  "total_tasks": 1,
  "chunks_size": 1000,
  "fps": 10,
  "splits": {"train": "0:206"},
  "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
  "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
  "features": {
    "observation.image": {
      "dtype": "video",
      "shape": [96, 96, 3],
      "names": ["height", "width", "channel"],
      "video_info": {
        "video.fps": 10.0,
        "video.codec": "av1",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": false,
        "has_audio": false
      }
    },
    "observation.state": {
      "dtype": "float32",
      "shape": [2],
      "names": {"motors": ["motor_0", "motor_1"]}
    },
    "action": {
      "dtype": "float32",
      "shape": [2],
      "names": {"motors": ["motor_0", "motor_1"]}
    }
  }
}
```

---

## Creating Custom Datasets Programmatically

### Features Dict (define before create)
```python
FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": {"axes": ["j0","j1","j2","j3","j4","j5"]},
    },
    "observation.images.front": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": {"axes": ["j0","j1","j2","j3","j4","j5"]},
    },
}
```

### Full Workflow
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

dataset = LeRobotDataset.create(
    repo_id="user/my_dataset",
    robot_type="so101",
    fps=30,
    features=FEATURES,
)

for ep_idx in range(num_episodes):
    for frame_idx in range(episode_length):
        frame = {
            "observation.state": state_array,           # np.float32
            "observation.images.front": image_array,    # np.uint8 (H,W,3)
            "action": action_array,                     # np.float32
            "task": "pick up the cube",                 # string, sets task_index
        }
        dataset.add_frame(frame)
    dataset.save_episode()

dataset.finalize()       # MUST call before push
dataset.push_to_hub()
```

### Key Rules
- `add_frame()` adds one timestep to episode buffer
- `save_episode()` persists buffered frames to parquet + encodes video
- `finalize()` closes parquet writers, writes metadata footers — **mandatory**
- `task` field in frame dict maps to tasks.jsonl automatically
- Cast fp64 → fp32 before add_frame
- Images: pass as numpy uint8 (H, W, 3)

---

## Loading / Streaming

```python
# Local/cached
dataset = LeRobotDataset("user/my_dataset")
sample = dataset[100]  # dict of tensors

# Temporal windows
dataset = LeRobotDataset("user/my_dataset",
    delta_timestamps={"observation.images.front": [-0.2, -0.1, 0.0]})

# Streaming (v3 only, no download)
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
dataset = StreamingLeRobotDataset("user/my_dataset")
```

---

## HDF5 Conversion

### Option 1: Direct Python (recommended)
Read HDF5 with h5py, iterate episodes, call add_frame() for each timestep.

### Option 2: Healthcare blog approach (Isaac Sim → HDF5 → LeRobot)
```bash
python -m training.hdf5_to_lerobot \
  --repo_id=my_dataset \
  --hdf5_path=/path/to/sim_dataset.hdf5 \
  --task_description="pick up cube"
```

### Option 3: Forge tool (multi-format converter)
```bash
pip install -e ".[all]"  # from github.com/arpitg1304/forge
forge convert /path/to/hdf5_dataset ./output --format lerobot-v3
```

### Option 4: v2.1 → v3.0 migration
```bash
python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --repo-id=user/dataset
```

---

## Sim-to-Real with LeRobot

### NVIDIA Healthcare Demo (93% synthetic data)
- Collect sim data via Isaac Lab teleoperation → HDF5
- Convert HDF5 → LeRobot format
- Mix ~70 sim episodes + 10-20 real episodes
- Train GR00T N1.5 on combined dataset
- Deploy to physical SO-101

### Isaac Lab Integration
- Isaac Lab 2.2+ has pre-built data collection environments
- GR00T N1.5 trainable directly via LeRobot (v0.4.0+)
- `Isaac-GR00T` repo has data_preparation.md for format details

---

## Dataset Tools CLI
```bash
# Delete episodes
lerobot-edit-dataset --repo_id user/ds --operation.type delete_episodes --operation.episode_indices "[0,2,5]"

# Split
lerobot-edit-dataset --repo_id user/ds --operation.type split --operation.splits '{"train":0.8,"test":0.2}'

# Merge
lerobot-edit-dataset --repo_id user/merged --operation.type merge --operation.repo_ids "['user/ds1','user/ds2']"

# Convert images to video
lerobot-edit-dataset --repo_id user/ds --operation.type convert_image_to_video --operation.vcodec libsvtav1

# Info
lerobot-edit-dataset --repo_id user/ds --operation.type info --operation.show_features true
```
