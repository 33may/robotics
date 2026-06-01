# UVA Bake Chunked Parquet Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the SmolVLA-UVA dataset bake so the full `duck_cup_v020_all_uva` parquet rewrite completes without OOM while preserving the existing LeRobot feature schema.

**Architecture:** Keep the existing two-phase bake: first compute one `(N, S, S, D)` per-frame feature tensor, then rewrite dataset parquets. Change only the parquet rewrite boundary so each source parquet is written in row chunks, limiting the `tensor -> nested Python list -> Arrow array` expansion to a bounded number of rows.

**Tech Stack:** Python 3.10/3.12, PyTorch tensors, PyArrow parquet writer, LeRobot dataset format, `vbti.logic.dataset.add_video_features`.

---

## File Structure

- Modify `vbti/logic/dataset/add_video_features.py`
  - Add a small helper for the nested Arrow feature type.
  - Add a row-chunked parquet writer path inside `_build_dataset_copy`.
  - Add CLI option `--rewrite-chunk-rows` with a safe default.
- Create `tests/logic/dataset/test_add_video_features.py`
  - Unit-test the rewrite boundary using tiny synthetic parquets and baked tensors.
  - Avoid loading the SmolVLA teacher or real video data.

---

### Task 1: Add a failing unit test for chunked parquet rewrite

**Files:**
- Create: `tests/logic/dataset/test_add_video_features.py`
- Modify: none

- [ ] **Step 1: Create the test file**

```python
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from vbti.logic.dataset.add_video_features import _build_dataset_copy


def _write_source_dataset(root: Path) -> None:
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    videos = root / "videos"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    videos.mkdir(parents=True)

    (meta / "info.json").write_text(
        '{"features": {"index": {"dtype": "int64", "shape": [1], "names": null}, '
        '"episode_index": {"dtype": "int64", "shape": [1], "names": null}, '
        '"value": {"dtype": "int64", "shape": [1], "names": null}}}'
    )

    table = pa.table(
        {
            "index": pa.array([0, 1, 2, 3, 4], type=pa.int64()),
            "episode_index": pa.array([0, 0, 0, 1, 1], type=pa.int64()),
            "value": pa.array([10, 11, 12, 13, 14], type=pa.int64()),
        }
    )
    pq.write_table(table, data / "file-000.parquet")


def test_build_dataset_copy_writes_feature_column_in_row_chunks(tmp_path: Path) -> None:
    src = tmp_path / "src"
    out = tmp_path / "out"
    out.mkdir()
    _write_source_dataset(src)

    baked = torch.arange(5 * 1 * 1 * 2, dtype=torch.float16).reshape(5, 1, 1, 2)
    episode_index = [0, 0, 0, 1, 1]
    feature_key = "observation.video_features.siglip_output_1x1"

    _build_dataset_copy(
        src_root=src,
        output=out,
        feature_key=feature_key,
        baked=baked,
        episode_index=episode_index,
        dtype_str="fp16",
        spatial_size=1,
        t_future=2,
        rewrite_chunk_rows=2,
        log=__import__("logging").getLogger("test"),
    )

    written = out / "data" / "chunk-000" / "file-000.parquet"
    pf = pq.ParquetFile(written)
    assert pf.metadata.num_rows == 5
    assert pf.num_row_groups == 3

    table = pq.read_table(written)
    assert table.column("value").to_pylist() == [10, 11, 12, 13, 14]

    rows = table.column(feature_key).to_pylist()
    assert rows[0] == [[[[2.0, 3.0]]], [[[4.0, 5.0]]]]
    assert rows[2] == [[[[4.0, 5.0]]], [[[4.0, 5.0]]]]
    assert rows[3] == [[[[8.0, 9.0]]], [[[8.0, 9.0]]]]
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
pytest tests/logic/dataset/test_add_video_features.py -q
```

Expected before implementation:

```text
TypeError: _build_dataset_copy() got an unexpected keyword argument 'rewrite_chunk_rows'
```

---

### Task 2: Implement row-chunked parquet rewrite

**Files:**
- Modify: `vbti/logic/dataset/add_video_features.py:184-275`
- Test: `tests/logic/dataset/test_add_video_features.py`

- [ ] **Step 1: Add constants and helper functions**

In `vbti/logic/dataset/add_video_features.py`, add after imports:

```python
DEFAULT_REWRITE_CHUNK_ROWS = 256
```

Add near `_assemble_windows_for_indices`:

```python
def _feature_arrow_type(dtype_str: str) -> pa.DataType:
    pa_dtype = pa.float16() if dtype_str == "fp16" else pa.float32()
    innermost_type = pa.list_(pa.field("item", pa_dtype))
    inner_type = pa.list_(pa.field("item", innermost_type))
    mid_type = pa.list_(pa.field("item", inner_type))
    return pa.list_(pa.field("item", mid_type))
```

- [ ] **Step 2: Extend `_build_dataset_copy` signature**

Change the function signature to:

```python
def _build_dataset_copy(
    src_root: Path,
    output: Path,
    feature_key: str,
    baked: torch.Tensor,
    episode_index: list[int],
    dtype_str: str,
    spatial_size: int,
    t_future: int,
    log: logging.Logger,
    rewrite_chunk_rows: int = DEFAULT_REWRITE_CHUNK_ROWS,
) -> None:
```

- [ ] **Step 3: Replace whole-file `tolist()` rewrite with chunked writer**

Replace lines equivalent to the current whole-file block:

```python
        table = pq.read_table(str(src_pq))
        indices = table.column("index").to_pylist()
        window = _assemble_windows_for_indices(
            baked, episode_index, ep_last, indices, t_future,
        )
        rows = window.tolist()

        innermost_type = pa.list_(pa.field("item", pa_dtype))
        inner_type     = pa.list_(pa.field("item", innermost_type))
        mid_type       = pa.list_(pa.field("item", inner_type))
        outer_type     = pa.list_(pa.field("item", mid_type))
        new_col = pa.array(rows, type=outer_type)

        table = table.append_column(
            pa.field(feature_key, outer_type),
            new_col,
        )
        pq.write_table(table, str(dst_pq), compression="snappy")
```

with:

```python
        table = pq.read_table(str(src_pq))
        indices = table.column("index").to_pylist()
        feature_type = _feature_arrow_type(dtype_str)
        feature_field = pa.field(feature_key, feature_type)
        writer = None
        try:
            for row_start in range(0, table.num_rows, rewrite_chunk_rows):
                row_end = min(row_start + rewrite_chunk_rows, table.num_rows)
                chunk_indices = indices[row_start:row_end]
                chunk_table = table.slice(row_start, row_end - row_start)
                window = _assemble_windows_for_indices(
                    baked, episode_index, ep_last, chunk_indices, t_future,
                )
                new_col = pa.array(window.tolist(), type=feature_type)
                chunk_table = chunk_table.append_column(feature_field, new_col)

                if writer is None:
                    writer = pq.ParquetWriter(str(dst_pq), chunk_table.schema, compression="snappy")
                writer.write_table(chunk_table)
        finally:
            if writer is not None:
                writer.close()
```

- [ ] **Step 4: Run the unit test**

Run:

```bash
pytest tests/logic/dataset/test_add_video_features.py -q
```

Expected:

```text
1 passed
```

---

### Task 3: Add CLI validation and wire the option through

**Files:**
- Modify: `vbti/logic/dataset/add_video_features.py:307-427`
- Test: `tests/logic/dataset/test_add_video_features.py`

- [ ] **Step 1: Add CLI argument**

Add near the other argparse options:

```python
    p.add_argument("--rewrite-chunk-rows", type=int, default=DEFAULT_REWRITE_CHUNK_ROWS,
                   help="Rows per in-memory Arrow conversion while rewriting parquets. "
                        "Lower values use less RAM; default: 256")
```

- [ ] **Step 2: Add validation**

Add after the `t_future` validation:

```python
    if args.rewrite_chunk_rows < 1:
        raise SystemExit(f"ERROR: --rewrite-chunk-rows must be >= 1, got {args.rewrite_chunk_rows}")
```

- [ ] **Step 3: Pass the option into `_build_dataset_copy`**

Change the call to include:

```python
        rewrite_chunk_rows=args.rewrite_chunk_rows,
```

- [ ] **Step 4: Add a small validation test**

Append to `tests/logic/dataset/test_add_video_features.py`:

```python
from vbti.logic.dataset.add_video_features import DEFAULT_REWRITE_CHUNK_ROWS


def test_default_rewrite_chunk_rows_is_bounded() -> None:
    assert DEFAULT_REWRITE_CHUNK_ROWS == 256
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
pytest tests/logic/dataset/test_add_video_features.py -q
```

Expected:

```text
2 passed
```

---

### Task 4: Verify schema compatibility locally

**Files:**
- Modify: none
- Test: `tests/logic/dataset/test_add_video_features.py`

- [ ] **Step 1: Run import/syntax check**

Run:

```bash
python -m py_compile vbti/logic/dataset/add_video_features.py
```

Expected: no output and exit code 0.

- [ ] **Step 2: Run focused tests with verbose output**

Run:

```bash
pytest tests/logic/dataset/test_add_video_features.py -vv
```

Expected:

```text
tests/logic/dataset/test_add_video_features.py::test_build_dataset_copy_writes_feature_column_in_row_chunks PASSED
tests/logic/dataset/test_add_video_features.py::test_default_rewrite_chunk_rows_is_bounded PASSED
```

- [ ] **Step 3: Inspect the output parquet manually if test fails**

Run only if needed:

```bash
python - <<'PY'
import pyarrow.parquet as pq
p = 'tests/logic/dataset/test_add_video_features.py'
print('Open the pytest tmp path from the failure output and inspect with pq.read_table(path).schema')
PY
```

Expected: only used for debugging; no code change from this step.

---

### Task 5: Deploy and verify on remote before relaunching chain

**Files:**
- Modify remote copy of: `/home/vbti/anton/robotics/vbti/logic/dataset/add_video_features.py`
- Remote output target: `/home/vbti/anton/data/eternalmay33/duck_cup_v020_all_uva`

- [ ] **Step 1: Sync the patched file to remote**

Run:

```bash
rsync -av vbti/logic/dataset/add_video_features.py vbti@10.11.100.156:/home/vbti/anton/robotics/vbti/logic/dataset/add_video_features.py
```

Expected:

```text
add_video_features.py
sent ...
```

- [ ] **Step 2: Verify remote syntax**

Run:

```bash
ssh vbti@10.11.100.156 'source /home/vbti/anton/env/bin/activate && cd /home/vbti/anton/robotics && python -m py_compile vbti/logic/dataset/add_video_features.py'
```

Expected: no output and exit code 0.

- [ ] **Step 3: Confirm before deleting partial bake output**

Ask the user before running this destructive command:

```bash
ssh vbti@10.11.100.156 'rm -rf /home/vbti/anton/data/eternalmay33/duck_cup_v020_all_uva'
```

Expected after approval: partial 22GB output removed.

- [ ] **Step 4: Relaunch bake with bounded rewrite chunk size**

Run:

```bash
ssh vbti@10.11.100.156 'tmux new-session -d -s uva_bake "bash -lc '\''source /home/vbti/anton/env/bin/activate && cd /home/vbti/anton/robotics && python -m vbti.logic.dataset.add_video_features --dataset eternalmay33/duck_cup_v020_all --root /home/vbti/anton/data/eternalmay33/duck_cup_v020_all --teacher /home/vbti/anton/data/uva_teacher_v020_150k --layer siglip_output --spatial-size 4 --t-future 4 --target-camera observation.images.gripper --batch-size 32 --dtype fp16 --rewrite-chunk-rows 256 --output /home/vbti/anton/data/eternalmay33/duck_cup_v020_all_uva > /home/vbti/anton/data/uva_bake.log 2>&1; echo UVA_BAKE_EXIT:$? >> /home/vbti/anton/data/uva_bake.log'\''"'
```

Expected: tmux session `uva_bake` starts.

- [ ] **Step 5: Monitor bake progress and memory**

Run periodically:

```bash
ssh vbti@10.11.100.156 'tail -n 5 /home/vbti/anton/data/uva_bake.log; free -h; du -sh /home/vbti/anton/data/eternalmay33/duck_cup_v020_all_uva 2>/dev/null || true'
```

Expected during rewrite: progress passes `rewriting parquets: 87/103` without RSS explosion or process kill.

- [ ] **Step 6: Verify full dataset exists and is readable**

Run after tmux exits:

```bash
ssh vbti@10.11.100.156 'find /home/vbti/anton/data/eternalmay33/duck_cup_v020_all_uva/data -name "*.parquet" | wc -l; tail -n 20 /home/vbti/anton/data/uva_bake.log'
```

Expected:

```text
103
verification OK: observation.video_features.siglip_output_4x4 shape=...
bake complete
UVA_BAKE_EXIT:0
```

---

### Task 6: Relaunch the UVA training chain only after bake verification

**Files:**
- Read: `vbti/experiments/duck_cup_smolvla/v025/config.yaml`
- Read: `vbti/experiments/duck_cup_smolvla/v026/config.yaml`
- Read: `vbti/experiments/duck_cup_smolvla/v027/config.yaml`
- Read: `vbti/experiments/duck_cup_smolvla/v028/config.yaml`
- Read: `vbti/experiments/duck_cup_smolvla/v029/config.yaml`

- [ ] **Step 1: Confirm configs still point to the baked dataset**

Run:

```bash
grep -R "duck_cup_v020_all_uva\|smolvla_uva\|steps:" -n vbti/experiments/duck_cup_smolvla/v02{5,6,7,8,9}/config.yaml
```

Expected: all five configs use `repo_id: eternalmay33/duck_cup_v020_all_uva`, `type: smolvla_uva`, and explicit `steps`.

- [ ] **Step 2: Relaunch chain**

Run:

```bash
python -m vbti.logic.train.chain_remote --versions v025,v026,v027,v028,v029 --run_name lerobot_output_r2
```

Expected: chain dispatches to remote tmux and does not wait on a missing bake session.

- [ ] **Step 3: Verify W&B starts logging step 0+**

Run:

```bash
python - <<'PY'
import wandb
api = wandb.Api()
runs = api.runs('eternalmay33/vbti-training', filters={'display_name': {'$regex': 'duck_cup_smolvla_v02[5-9]_lerobot_output_r2'}}, per_page=20)
for r in runs:
    print(r.name, r.state, r.summary.get('_step'), r.url)
PY
```

Expected: first active run has `_step` not `None` after training starts.

---

## Self-Review

- Spec coverage: The plan fixes the identified OOM boundary (`window.tolist()` over whole parquet files), preserves the nested Arrow schema, validates with a unit test, and verifies remote bake before relaunching training.
- Placeholder scan: No TBD/TODO placeholders remain.
- Type consistency: `rewrite_chunk_rows`, `DEFAULT_REWRITE_CHUNK_ROWS`, and `_feature_arrow_type` are used consistently across implementation and tests.
- Safety check: The only destructive command is deleting the partial `_uva` output, and the plan explicitly requires user approval before running it.
