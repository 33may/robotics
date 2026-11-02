# Dataset CLI reference

All modules are `python -m vla.logic.dataset.<module> <command>`. Fire CLIs: `<command> --help` prints the signature. `<repo_id>` resolves to `~/.cache/huggingface/lerobot/<repo_id>` unless `--root` is given.

## check_utils: LeRobot datasets

| Command | Args | Does |
|---|---|---|
| `ls` | | All local LeRobot datasets: episodes, frames, fps, robot |
| `info` | `REPO_ID [--root=R]` | Features, stats, normalization |
| `cameras` | `REPO_ID [--root=R]` | Camera names, shapes, codecs, order |
| `report` | `PATH` | Meta, actions, states, action/state correspondence, videos |
| `plot_actions` | `REPO_ID [--root] [--save] [--alpha] [--xlim] [--joints]` | Per-episode action trajectories overlaid, mean in bold |
| `compare_actions` | `--datasets='{"a":"PATH","b":"PATH"}' [--bins=80] [--save=F]` | Overlaid action histograms |
| `delete` | `DATASET_PATH [--root] [--force]` | Delete a cached dataset. Destructive |

## trim_utils: rest frames

| Command | Args | Does |
|---|---|---|
| `preview` | `DATASET_PATH [--dist_thresh] [--n_eps] [--joints] [--save]` | Trajectories with rest shaded + before/after distributions |
| `trim` | `REPO_ID OUTPUT_REPO_ID [--root] [--output_root] [--dist_thresh] [--min_active_frames] [--push_to_hub]` | New dataset without rest frames |
| `trajectories` | `DATASET_PATH [--episodes] [--ep] [--n_episodes] [--save]` | Joint trajectories, rest shaded |
| `compare_trimmed` | `DATASETS [--dist_thresh] [--bins] [--save]` | Action distributions after trimming |
| `distribution` | `DATASET_PATH [--bins] [--save]` | Action distribution of one dataset |
| `drop_episodes` | `DATASET_PATH [--ep] [--min_std] [--output] [--inplace]` | Remove episodes (LeRobot `delete_episodes`). `--inplace` is destructive |

## hdf5_utils: raw sim HDF5

| Command | Args | Does |
|---|---|---|
| `info` | `PATH` | Episodes, cameras, sensors, shapes |
| `view` | `PATH [EPISODE] [CAMERA] [--sensor=rgb] [--fps=30] [--save=F]` | Play or save an episode video |
| `report` | `PATH` | Structure, actions, states, images |
| `tree` | `PATH [--episode=demo_0]` | Data tree with shapes and sizes |
| `episodes` | `PATH` | Episodes with frame counts |

## convert_utils: HDF5 to LeRobot v3.0

| Command | Args | Does |
|---|---|---|
| `discover` | `HDF5` | RGB camera keys from the first valid episode |
| `convert` | `HDF5 REPO_ID TASK --camera_map='{...}' [--root] [--fps=30] [--min_episode_frames=10] [--skip_first_n=0] [--push_to_hub]` | Convert; sim radians to normalized joints |
| `verify` | `DATASET_PATH` | Body joints in `[-100, 100]`, gripper in `[0, 100]` |
| `roundtrip_test` | | `sim_to_normalized` / `normalized_to_sim` roundtrip |
| `to_delta` | `SOURCE OUTPUT [--push_to_hub]` | Absolute to delta actions, new dataset |
| `recalibrate` | `SOURCE OLD_CALIB NEW_CALIB [--push_to_hub]` | Re-express joints under another calibration |
| `link` | `DATASET_PATH REPO_ID` | Symlink a local dataset into the HF cache |
| `ls` | | Linked/cached datasets |

## Transforms

| Module | Args | Does |
|---|---|---|
| `loading_utils aggregate` | `--datasets="a,b" --output=REPO [--output_root]` | Merge datasets with matching schemas |
| `subsample` | `SRC STRIDE DST [--output_dir] [--start]` | Every Nth episode, new dataset |
| `strip_feature` | `--src --dst --feature [--src-root] [--dst-root] [--overwrite] [--no-verify]` | Drop one feature, new dataset |
| `add_zero_depth_turbo` | `--src --dst --reference` | Zero depth placeholder so schemas match a depth dataset |
| `depth_viewer` | `raw` / `frame` / `replay` | Inspect packed or baked depth |

## replay_utils: replay on the real robot

| Command | Args | Does |
|---|---|---|
| `show` | `DATASET_PATH [--episode_idx=0]` | Print actions, no robot connection |
| `replay` | `DATASET_PATH [--episode_idx] [--port] [--robot_id] [--max_relative_target=10.0] [--speed] [--dry_run]` | Replay on the follower. Moves hardware |
| `goto` | `[--port] [--robot_id] [--duration] [--<joint>=deg]` | Move the follower to a pose. Moves hardware |
