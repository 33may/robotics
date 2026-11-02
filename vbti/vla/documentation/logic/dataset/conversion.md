# HDF5 To LeRobot Conversion

## Discover HDF5 Structure

```bash
python -m vbti.logic.dataset.convert_utils discover /path/to/data.hdf5
```

Use this before conversion to identify camera names, action/state paths, and episode structure.

## Convert HDF5 To LeRobot

```bash
python -m vbti.logic.dataset.convert_utils convert \
  /path/to/data.hdf5 \
  eternalmay33/my_dataset \
  "Pick up the duck and place it in the cup"
```

With explicit camera map:

```bash
python -m vbti.logic.dataset.convert_utils convert \
  /path/to/data.hdf5 \
  eternalmay33/my_dataset \
  "Pick up the duck and place it in the cup" \
  --camera_map='{"side_cam":"left","table_cam":"top","wrist":"gripper"}'
```

## Verify Output

```bash
python -m vbti.logic.dataset.convert_utils verify ~/.cache/huggingface/lerobot/eternalmay33/my_dataset
python -m vbti.logic.dataset.check_utils report eternalmay33/my_dataset
```

## Other Conversion Helpers

```bash
python -m vbti.logic.dataset.convert_utils to_delta <source> <output>
python -m vbti.logic.dataset.convert_utils recalibrate <source> <old_calib> <new_calib>
python -m vbti.logic.dataset.convert_utils link <dataset_path> <repo_id>
python -m vbti.logic.dataset.convert_utils ls
```

## Units

Simulation HDF5 often stores joint values in radians or simulation-native ranges. Real LeRobot SO-101 datasets use degree-like calibrated joint values. Conversion code maps simulation values into the real command/state range.

Do not assume unit conversion alone solves sim-real alignment. Calibration zero positions must also match.

## Output Expectations

Converted LeRobot dataset should have:

```text
meta/info.json
meta/stats.json
data/chunk-*/file-*.parquet
videos/observation.images.<camera>/chunk-*/file-*.mp4
```

Validate that:

- episode count matches expectation;
- no empty/too-short episodes were silently important;
- image streams are present;
- state/action dimensions are correct;
- task text is correct;
- camera keys match training config.

## Common Failures

| Symptom | Likely cause | Fix |
|---|---|---|
| Missing image feature | HDF5 camera path not mapped | Use `discover`, pass `--camera_map`. |
| Wrong action scale | Sim-real unit mismatch | Inspect action stats and conversion mapping. |
| Training ignores sim images | Feature key mismatch | Match `observation.images.<name>` to config rename map. |
| Mixed dataset fails | Different feature schemas | Strip/add zero features or regenerate consistent datasets. |
