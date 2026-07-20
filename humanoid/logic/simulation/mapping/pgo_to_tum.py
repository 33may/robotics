"""pgo_to_tum — extract per-camera TUM poses from the cuSFM POSE-GRAPH meta.

The cuSFM recipe is PGO-ONLY (cusfm-30hz-pgo-works memory): mapper BA corrupts
poses (5.5 m mean, 300/320 flyaways on teleop_v1_demo), so `output_poses/` TUMs
must never be consumed. The good poses live in
`cusfm/pose_graph/frames_meta.json` — per-keyframe optical `camera_to_world`
blocks (axis-angle), the exact poses `fuse_reconstruction.py` fuses at.

Quaternion export uses scipy Rotation: naive quat-from-matrix silently breaks
near 180° yaw (dense-reconstruction-from-teleop memory).

Run (isaac env, humanoid package on PYTHONPATH):
    python logic/simulation/mapping/pgo_to_tum.py \
        --frames-meta <bake>/cusfm/pose_graph/frames_meta.json \
        --camera head_left --out <bake>/cusfm/output_poses/pgo_head_left.tum
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from humanoid.logic.simulation.mapping.nvblox_inject import T_from_meta


def extract(frames_meta: Path, camera: str) -> list[list[float]]:
    meta = json.loads(Path(frames_meta).read_text())
    rows = []
    for k in meta["keyframes_metadata"]:
        if not k["image_name"].startswith(f"{camera}/"):
            continue
        T = T_from_meta(k["camera_to_world"])
        q = Rotation.from_matrix(T[:3, :3]).as_quat()  # xyzw
        stamp = int(k["timestamp_microseconds"]) * 1e-6
        rows.append([stamp, *T[:3, 3], *q])
    rows.sort(key=lambda r: r[0])
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames-meta", required=True, type=Path,
                    help="cusfm/pose_graph/frames_meta.json (PGO poses)")
    ap.add_argument("--camera", default="head_left")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)

    rows = extract(a.frames_meta, a.camera)
    if not rows:
        raise SystemExit(f"no '{a.camera}/' keyframes in {a.frames_meta}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(" ".join(f"{v:.9f}" for v in r) + "\n")
    print(f"{len(rows)} {a.camera} PGO poses -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
